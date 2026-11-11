"""Joint regularised inversion (charge-neutral IMTS, full ODE): kinetic parameters theta AND a piecewise-constant temperature history T(t) (6 log-spaced segments, 10 Myr).
Phi = sum_k (N_k/2) log(S_k + 0.25 N_k)  [Jeffreys prior on the noise scale of each data set: no hand-chosen weights]
      + (T_present - 47)^2/(2*5^2)  + sum_j (T_j - T_present)^2/(2 sigdev^2) + sum_j (T_j - T_j+1)^2/(2 sigsm^2)   [geological prior: stable basin, geotherm 24 C/km]
      + sum (second difference of log N controls)^2/(2*1^2)                                                       [smooth N(E)]
S_TL = sum over the 162 curves of a robust (log) function of the reduced chi2 per curve; S_MAAD, S_lab, S_profile = chi2.
Block-coordinate minimisation: Nelder-Mead on theta (T fixed), then Nelder-Mead on T (theta fixed, 0.5 s per evaluation); repeated.
usage: python inv_joint.py start.json law_a law_b tag [n_jobs] [maxfev_theta] [cycles]"""
import sys, os, json, time, importlib, io, contextlib, numpy as np
os.environ["NEUTRAL"] = "1"; f0, la0, lb0, tag = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]; nj = int(sys.argv[5]) if len(sys.argv) > 5 else 10
maxfev = int(sys.argv[6]) if len(sys.argv) > 6 else 250; NCYC = int(sys.argv[7]) if len(sys.argv) > 7 else 2
TMP = os.path.abspath(f"tmp_inv_{tag}.json"); json.dump(json.load(open(f0)), open(TMP, "w")); os.environ["IMTS_PARAMS"] = TMP; os.environ.setdefault("MAAD", "1")
sys.argv = ["chi2_ode.py", f0, str(nj)]
import chi2_ode as C, jax.numpy as jnp
USEITL = bool(os.environ.get('USEITL'))
if USEITL: import itl_fit_mod as IT
src = open("cap_tests.py").read(); head = src[:src.index("# (ii) lab-only law fit")]
with contextlib.redirect_stdout(io.StringIO()): exec(compile(head, "cap_tests_head", "exec"))
from scipy.optimize import minimize
M = C.M
SIGDEV = float(os.environ.get("SIGDEV", 2.0)); SIGSM = float(os.environ.get("SIGSM", 2.0)); TPRES, SIGP = 47.0, 5.0; CROB = 30.0
SEG_KY = [(3000.0, 10000.0), (1000.0, 3000.0), (100.0, 1000.0), (30.0, 100.0), (10.0, 30.0), (0.0, 10.0)]; DUR_MYR = [(b - a) / 1000.0 for a, b in SEG_KY]
YR_ = SI.YR; kB_ = SI.kB
def lab_sat():
    W, LA = SI.W_K, SI.LA_K; nn = GI.irradiate(SI.E, SI.s, SI.N, SI.A, 19950.0, SI.D0); f = 0.1566 / SI.D0; lam_ = SI.s * np.exp(-SI.E / (kB_ * 293.15))
    ch = lambda LAx: f / (f + lam_[:, None] + LAx) * (1 - np.exp(-(f + lam_[:, None] + LAx) * 127000.0)); fac = (W * ch(LA)).sum(1) / ch(np.zeros((SI.NT, 1)))[:, 0]
    return HA.win(SI.readout(np.clip(nn * fac, 0, SI.N)))
def kars_multi(stages, ddot):
    W, LA = SI.W_K, SI.LA_K; f = ddot / (1000 * YR_) / SI.D0
    def chain(LAx):
        x = np.zeros((SI.NT, LAx.shape[1]))
        for T, t in stages:
            r = f + (SI.s * np.exp(-SI.E / (kB_ * (273.15 + T))))[:, None] + LAx; x = f / r + (x - f / r) * np.exp(-r * t * 1e6 * YR_)
        return x
    return ((W if W.ndim == 2 else W[None, :]) * chain(LA)).sum(1) / np.maximum(chain(np.zeros((SI.NT, 1)))[:, 0], 1e-300)
def hist_model(temps, sat, ddot=SI.DOT):
    stages = list(zip(temps, DUR_MYR)); n = np.zeros(SI.NT)
    for T, t in stages: n = SI.hold(n, 273.15 + T, t, ddot)
    return HA.win(SI.readout(np.clip(n * kars_multi(stages, ddot), 0, SI.N))) / sat
def prior_T(T):
    T = np.asarray(T, float); return (T[-1] - TPRES) ** 2 / (2 * SIGP ** 2) + np.sum((T[:-1] - T[-1]) ** 2) / (2 * SIGDEV ** 2) + np.sum((T[:-1] - T[1:]) ** 2) / (2 * SIGSM ** 2)
def S_tl(p):
    res = C.Parallel(n_jobs=nj)(C.delayed(C.ode_run)(p, ph, M.DOSE_VHR, range(6)) for ph in M.PREHEATS)
    norm = np.stack([[r[0] for r in rr] for rr in res]); mod = norm[None] * np.asarray(M.I_peak)[..., None]; v = np.asarray(M.var_of(jnp.array(mod)))
    red = (((mod - np.asarray(M.I_data)) ** 2 / v)[..., np.asarray(M.FIT_MASK)]).mean(-1)          # (cup, 9, 6) reduced chi2 per curve
    return float(np.sum(CROB * np.log1p(red / CROB) * np.asarray(M.RATE_W)[None, None, :])), red
NK = dict(tl=162.0, maad=66.0, lab=48.0, prof=15.0)
NOPROF = bool(os.environ.get('NOPROF'))
if USEITL: NK['itl'] = 42.0
if NOPROF: NK.pop('prof')
def phi_from(S, T, p):
    ph = sum(0.5 * NK[k] * np.log(S[k] + 0.25 * NK[k]) for k in NK) + prior_T(T)
    c = p[7:12]; ph += np.sum((c[:-2] - 2 * c[1:-1] + c[2:]) ** 2) / 2.0
    return ph
base = np.array(json.load(open(f0))["best"]); IDX = [0, 1] + list(range(7, 12)) + [12, 13, 14, 15, 16]; NI = len(IDX)
lo = np.array([-1.0, 8.0] + [6.5] * 5 + [2.0, 8.0] + [0.0] * 3 + [-6.5, -1.0]); hi = np.array([3.0, 16.0] + [11.5] * 5 + [4.4, 13.0] + [4.3] * 3 + [-4.5, 1.0])
th = np.clip(np.concatenate([base[IDX], [la0, lb0]]), lo, hi); T = np.array([TPRES] * 6, float); t0 = time.time(); nev = [0]; state = {}
def theta_eval(x, T, full=False):
    x = np.clip(x, lo, hi); p = base.copy(); p[IDX] = x[:NI]; a, b = x[NI], x[NI + 1]
    stl, red = S_tl(p); cm = C.chi2_maad_ode(p, nj)[0]; sitl = IT.S_itl(p, nj) if USEITL else 0.0
    json.dump(dict(best=p.tolist()), open(TMP, "w")); importlib.reload(SI)
    nodes = [a + b * (Tt - 250.0) / 100.0 for Tt in (175.0, 225.0, 275.0, 325.0)]; lab = lab_nodes(nodes); sat = lab_sat()
    state.update(p=p, sat=sat, law=(a, b), S=dict(tl=stl, maad=cm, lab=lab, itl=sitl)); m = hist_model(T, sat); prof = float(np.sum(((m - HA.NN_DATA) / HA.NN_SD) ** 2)); S = dict(tl=stl, maad=cm, lab=lab, itl=sitl, prof=prof)
    return phi_from(S, T, p), S, red
def fun_theta(x):
    try: ph, S, red = theta_eval(x, T)
    except Exception as e: print("eval failed", repr(e)[:100], flush=True); return 1e6
    nev[0] += 1
    if ph < state.get("best", 1e30): state["best"] = ph; print(f"[{time.time()-t0:6.0f}s th {nev[0]:4d}] Phi {ph:8.2f} | TLrob {S['tl']:.0f} MAAD {S['maad']:.0f} lab {S['lab']:.0f} itl {S['itl']:.0f} prof {S['prof']:.1f} | T {np.round(T,1).tolist()}", flush=True)
    return ph
def fun_T(Tv):
    Tv = np.clip(Tv, -10, 120); m = hist_model(Tv, state["sat"]); prof = float(np.sum(((m - HA.NN_DATA) / HA.NN_SD) ** 2)); S = dict(state["S"], prof=prof); return phi_from(S, Tv, state["p"])
for cyc in range(NCYC):
    state["best"] = 1e30; fun_theta(th)
    sim = th + np.vstack([np.zeros(len(th)), np.diag(np.array([0.3, 0.8] + [0.4] * 5 + [0.15, 0.4] + [0.4] * 3 + [0.05, 0.15]) * (0.6 ** cyc))])
    r = minimize(fun_theta, th, method="Nelder-Mead", options=dict(maxfev=maxfev, xatol=0.005, fatol=0.01, initial_simplex=sim)); th = np.clip(r.x, lo, hi)
    ph, S, red = theta_eval(th, T)
    if not NOPROF: rT = minimize(fun_T, T, method="Nelder-Mead", options=dict(maxfev=600, xatol=0.02, fatol=0.005, initial_simplex=T + np.vstack([np.zeros(6), 1.5 * np.eye(6)]))); T = np.clip(rT.x, -10, 120)
    ph, S, red = theta_eval(th, T)
    print(f"== cycle {cyc}: Phi {ph:.2f} S {({k: round(v, 1) for k, v in S.items()})} T {np.round(T, 1).tolist()}", flush=True)
    TSg = np.arange(10.0, 81.0, 1.0); cprof = np.array([float(np.sum(((hist_model([t_] * 6, state["sat"]) - HA.NN_DATA) / HA.NN_SD) ** 2)) for t_ in TSg]); ib = int(np.argmin(cprof))
    print(f"   best constant T for the natural profile: {TSg[ib]:.0f} C (chi2 {cprof[ib]:.1f}); chi2 at 47 C {cprof[37]:.1f}; escale {os.environ.get('ESCALE', '1')}", flush=True)
    p = state["p"]; json.dump(dict(best=p.tolist(), law=list(state["law"]), T=T.tolist(), Phi=ph, S=S, seg_ky=SEG_KY, sigdev=SIGDEV, sigsm=SIGSM, bestT=float(TSg[ib]), chi2_bestT=float(cprof[ib]), chi2_47=float(cprof[37]), escale=float(os.environ.get('ESCALE', 1.0))), open(f"inv_joint_{tag}.json", "w"))
print("done", flush=True)

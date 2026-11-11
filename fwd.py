"""Array-based forward model of the charge-neutral IMTS for the regularised inversion.  All data sets are returned as WHITENED residual vectors.
Parameter vector x (28): dE(5), eta(5), dN(5) [splines in T_m, see pgrid.py], logD0, logM0, logDnat(3), law a, law b, T_1..T_6 (history, oldest -> youngest)."""
import sys, os, json, io, contextlib, importlib, numpy as np
os.environ.setdefault("NEUTRAL", "1"); os.environ.setdefault("MAAD", "1")
sys.argv = ["chi2_ode.py", "x", "10"]
import chi2_ode as C
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import pgrid
G = C.G; M = C.M; NC = pgrid.NC
MODEL = os.environ.get('IMTS_MODEL', 'imts')                      # imts | a0 (independent first-order traps) | gok (general order, extra parameter b)
if MODEL != 'imts':
    import indep; indep.patch_G(G)           # a0 | gok | gok2 (growth order a AND decay order b)
IDX = dict(dE=slice(0, 5), eta=slice(5, 10), dN=slice(10, 15), logD0=15, logM0=16, logDnat=slice(17, 20), a=20, b=21, T=slice(22, 28), bexp=28, aexp=29); NX = {'gok': 29, 'gok2': 30}.get(MODEL, 28)
QUENCH = os.environ.get('QUENCH') == '1'                    # thermal-quenching efficiency eta(T) = 1/(1 + C exp(-W/kT)) applied to the TL ramps (2 extra parameters: log10 C, W in eV)
if QUENCH: IDX['qlogC'] = NX; IDX['qW'] = NX + 1; NX += 2
def model_arrays(ref, x):
    E, s, logN = ref.build(x[IDX["dE"]], x[IDX["eta"]], x[IDX["dN"]]); return dict(q=(float(x[IDX["qlogC"]]), float(x[IDX["qW"]])) if QUENCH else None, E=E, s=s, N=10.0 ** logN * (16.0 / G.N_FINE), D0=10.0 ** x[IDX["logD0"]], M0=10.0 ** x[IDX["logM0"]], Dnat=10.0 ** np.clip(np.asarray(x[IDX["logDnat"]]), 0.0, 4.3), b=float(x[IDX["bexp"]]) if MODEL in ('gok', 'gok2') else 1.0, a=float(x[IDX["aexp"]]) if MODEL == 'gok2' else 1.0)
def _stage(E, s, N, D0, M0, PH, D, ib, b=1.0, a=1.0):
    if MODEL != 'imts': indep.set_order(b, a)
    G.M0 = M0; A = G.A_val * np.ones_like(E); n = G.irradiate(E, s, N, A, D, D0)
    if n is None: return None
    out = [G.run_imts_irr(E, s, N, A, n, G.BETAS[b], PH + 273.15) for b in ib]
    return None if any(o is None for o in out) else [(o[0], o[1]) for o in out]
def tl_resid(m, nj):
    rs = [int(v) for v in os.environ["TL_RATES"].split(",")] if os.environ.get("TL_RATES") else list(range(6))              # heating-rate indices kept (0.1 0.2 0.5 1 2 5 C/s = 0..5)
    ps = [int(v) for v in os.environ["TL_PH"].split(",")] if os.environ.get("TL_PH") else list(range(len(M.PREHEATS)))        # preheat indices kept (60..300 C = 0..8)
    res = Parallel(n_jobs=nj)(delayed(_stage)(m["E"], m["s"], m["N"], m["D0"], m["M0"], M.PREHEATS[ip], M.DOSE_VHR, rs, m["b"], m["a"]) for ip in ps)
    full = np.tile(res[0][0][0][None, None, :], (len(M.PREHEATS), 6, 1))                                                   # filler for the curves that are not computed (not used below)
    if QUENCH:
        eta = 1.0 / (1.0 + 10.0 ** m["q"][0] * np.exp(-m["q"][1] / (8.617333262e-5 * np.asarray(M.T_surr, float))))
        def qn(c): c2 = np.asarray(c) * eta; return c2 / c2.max()
        full = np.stack([[qn(c) for c in row] for row in full])
    for ip, rr in zip(ps, res):
        for ir, r in zip(rs, rr): full[ip, ir] = qn(r[0]) if QUENCH else r[0]
    mod = full[None] * np.asarray(M.I_peak)[..., None]; v = np.asarray(M.var_of(jnp.array(mod)))
    r = ((mod - np.asarray(M.I_data)) / np.sqrt(v))[..., np.asarray(M.FIT_MASK)]; r = r * np.asarray(M.RATE_W, float)[None, None, :, None] ** 0.5
    return r[:, ps][:, :, rs]                                                                                               # (cup, PH, rate, channel)
def maad_resid(m, nj):
    keys = {}
    for S in M.MAAD_SERIES:
        for d in list(np.asarray(S["D"]) + m["Dnat"][S["wi"]]) + [M.D_REF]: keys[(S["PH"], round(float(d), 6))] = None
    ks = list(keys); res = Parallel(n_jobs=nj)(delayed(_stage)(m["E"], m["s"], m["N"], m["D0"], m["M0"], ph, d, [3], m["b"], m["a"]) for ph, d in ks); sig = {k: r[0] for k, r in zip(ks, res) if r is not None}; out = []
    for S in M.MAAD_SERIES:
        Dn = m["Dnat"][S["wi"]]; w = np.asarray(S["w"]); win = lambda k: 10.0 ** sig[k][1] * np.sum(sig[k][0] * w) * M.DT_SURR
        f = np.array([win((S["PH"], round(float(d), 6))) for d in np.asarray(S["D"]) + Dn]) / max(win((S["PH"], round(float(M.D_REF), 6))), 1e-300)
        y = np.asarray(S["R"]); keep = np.asarray(S["keep"]) > 0; sg = M.MAAD_SIG * y + 0.03; ww = keep / sg ** 2; a = np.sum(ww * f * y) / max(np.sum(ww * f * f), 1e-300); out.append(((a * f - y) / sg)[keep])
    return np.concatenate(out)
# ---- ITL
import pickle
ITL_WINS = [(lo, lo + 10) for lo in range(200, 350, 10)]; ITL_T = [180, 210, 240, 270, 300, 330, 360]; ITL_TH = [0, 10, 30, 100, 300, 1000]
_d = pickle.load(open("itl_data.pkl", "rb")); _W = lambda o, lo, hi: (o['res'] - o['bg'])[lo:hi].sum()
ITL_REF = np.array([np.median([_W(o, lo, hi) for o in _d if o['T'] == 150 and o['th'] == 0]) for lo, hi in ITL_WINS]); _D = {}
for o in _d:
    if o['th'] == 70 or o['T'] not in ITL_T: continue
    _D.setdefault((o['T'], o['th']), []).append(np.array([_W(o, lo, hi) for lo, hi in ITL_WINS]) / ITL_REF)
ITL_DATA = {k: np.mean(v, axis=0) for k, v in _D.items()}; ITL_SEL = {k: v > 0.004 for k, v in ITL_DATA.items()}
def _itl_cond(E, s, N, D0, M0, T, th, b=1.0, a=1.0, dose=300.05):
    if MODEL != 'imts':
        indep.set_order(b, a); return indep.itl_cond(G, E, s, N, D0, T, th, dose, windows=ITL_WINS)
    G.M0 = M0; A = G.A_val * np.ones_like(E); n0 = G.irradiate(E, s, N, A, dose, D0); y = np.concatenate([n0, [0.0]])
    if T > 22: y = solve_ivp(G.make_rhs(E, s, N, A, beta=1.0), [0, T + 273.15 - G.T0], y, method="LSODA", max_step=0.5, rtol=1e-6, atol=1e-8).y[:, -1]
    if th > 0: y = solve_ivp(G.make_rhs(E, s, N, A, T_const=T + 273.15), [0, th], y, method="LSODA", rtol=1e-6, atol=1e-8).y[:, -1]
    y = y.copy(); y[-1] = 0.0; y[:-1] = np.maximum(y[:-1], 0.0); t_max = G.T_max - G.T0
    sol = solve_ivp(G.make_rhs(E, s, N, A, beta=1.0), [0, t_max], y, method="LSODA", dense_output=True, max_step=0.5, rtol=1e-6, atol=1e-8)
    tt = np.arange(0, t_max, 1.0); Y = sol.sol(tt); I = G.B * Y[-1] * (G.M0 + Y[:-1].sum(0) + Y[-1]); Tc = tt + G.T0 - 273.15
    return np.array([float(I[(Tc >= lo) & (Tc < hi)].sum()) for lo, hi in ITL_WINS])
def itl_resid(m, nj):
    jobs = [(150, 0)] + [(T, th) for T in ITL_T for th in ITL_TH]; out = Parallel(n_jobs=nj)(delayed(_itl_cond)(m["E"], m["s"], m["N"], m["D0"], m["M0"], T, th, m["b"], m["a"]) for T, th in jobs); ref = out[0]; r = []
    for (T, th), mm in zip(jobs[1:], out[1:]):
        mm = mm / ref; y = ITL_DATA[(T, th)]; sg = np.sqrt((0.05 * y) ** 2 + 0.01 ** 2); r.append(((mm - y) / sg)[ITL_SEL[(T, th)]])
    return np.concatenate(r)
# ---- parts that use the (module-global) imts_step machinery: laboratory fading and the natural history
SEG_KY = [(3000.0, 10000.0), (1000.0, 3000.0), (100.0, 1000.0), (30.0, 100.0), (10.0, 30.0), (0.0, 10.0)]; DUR_MYR = [(b - a) / 1000.0 for a, b in SEG_KY]
def init_main():
    global SI, HA, GI, NS
    src = open("cap_tests.py").read(); head = src[:src.index("# (ii) lab-only law fit")]; NS = {}
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(head, "cap_tests_head", "exec"), NS)
    SI, HA, GI = NS["SI"], NS["HA"], NS["GI"]
    seg = open("inv_joint.py").read(); seg = seg[seg.index("def lab_sat"):seg.index("def prior_T")]
    NS.update(dict(np=np, kB_=SI.kB, YR_=SI.YR, DUR_MYR=DUR_MYR)); exec(compile(seg, "seg", "exec"), NS)
    if MODEL != 'imts': SI.hold = indep.hold_fn(SI); SI.readout = indep.readout_fn(SI, GI); NS['ramp_state'] = indep.ramp_state_fn(SI, GI)           # lab_sat, kars_multi, hist_model
def apply_SI(m, a, b):
    if MODEL != 'imts': indep.set_order(m['b'], m['a'])
    SI.E = m["E"].copy(); SI.s = m["s"].copy(); SI.N = m["N"].copy(); SI.D0 = m["D0"]; GI.M0 = m["M0"]; SI.A = GI.A_val * np.ones_like(SI.E); SI.RHS_RO = GI.make_rhs(SI.E, SI.s, SI.N, SI.A, beta=1.0)
    NS["SI"] = SI; HA.LOGRHO_T[:] = [175.0, 225.0, 275.0, 325.0]; HA.LOGRHO_V[:] = [a + b * (T - 250.0) / 100.0 for T in (175.0, 225.0, 275.0, 325.0)]; SI.W_K, SI.LA_K = SI.rate_classes(SI.E, SI.s)
def lab_resid():
    out = []
    for pos, ph in NS["DISCS"].items():
        P = NS["sim_IMTS"]("B", ph)
        for k, (lo, hi) in enumerate(NS["WIN"]): out.append((P[1:, k] - NS["DATA"][(pos, lo, hi)][1:]) / 0.01)
    return np.concatenate(out)
def prof_resid(T):
    sat = NS["lab_sat"](); r = (NS["hist_model"](np.asarray(T, float), sat) - HA.NN_DATA) / HA.NN_SD / float(os.environ.get("PROF_SDX", 1))       # PROF_SDX: inflate profile errors; PROF_STEP: use every k-th window only
    return r[::int(os.environ.get("PROF_STEP", 1))]
def resid_all(ref, x, nj=10):
    m = model_arrays(ref, x); apply_SI(m, x[IDX["a"]], x[IDX["b"]])
    DROP = os.environ.get('DROP', '').split(',')                                                          # data sets left out of the inversion (e.g. DROP=tl,maad,lab)
    z = np.zeros(0)
    return dict(tl=z if 'tl' in DROP else tl_resid(m, nj), maad=z if 'maad' in DROP else maad_resid(m, nj), lab=z if 'lab' in DROP else lab_resid(), itl=z if 'itl' in DROP else itl_resid(m, nj), prof=z if 'prof' in DROP else prof_resid(x[IDX["T"]]))

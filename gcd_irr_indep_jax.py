"""Independent-trap models (MODEL=a0 linear superposition, MODEL=gok general order) fitted to the VHR cup glow curves AND the MAAD growth series, same likelihood,
same per-window natural equivalent dose as gcd_irr_jax.py (IMTS). Filling by the lab dose: x0 = 1 - exp(-D/D0) for every trap (independent traps, as in the user's GOK code).
usage: MODEL=a0|gok python gcd_irr_indep_jax.py <cup17|cup19|cup21|joint>
Parameters: [slope, intercept, delta(5), logN(5), logD0, logDnat_w1, logDnat_w2, logDnat_w3, (b for gok)]    env: N_TRAPS (default 32), ERR, VHR_SCALE, MAAD_W, MAAD_SIGMA, NSTART, NRAND, TAG_SUFFIX"""
import json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import a0_model as A0M, light_models as LM
jax.config.update("jax_enable_x64", True)
MODEL = os.environ.get("MODEL", "a0"); IS_GOK = MODEL == "gok"
MAAD_MODE = os.environ.get("MAAD_MODE", "nN")
CUP = sys.argv[1] if len(sys.argv) > 1 else "cup17"; CUPS = ["cup17", "cup19", "cup21"] if CUP == "joint" else [CUP]
N_TRAPS = int(os.environ.get("N_TRAPS", 32)); ERR = os.environ.get("ERR", "0") == "1"
MAAD_SIG = float(os.environ.get("MAAD_SIGMA", 0.15)); MAAD_W = float(os.environ.get("MAAD_W", 1.0)); VHR_SCALE = float(os.environ.get("VHR_SCALE", 1.0))
ERR_F, ST0, ST1 = 0.03, 3.0, 0.3
TAG = f"irr{MODEL}{N_TRAPS}_{CUP}_maad" + ("_ERR" if ERR else "") + os.environ.get("TAG_SUFFIX", "")
BETAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]; PREHEATS = [60, 90, 120, 150, 180, 210, 240, 270, 300]; N_PH = 9
DOSE_VHR = 300.05; DOSE_RATE = 0.1566; D_TEST = 1629.0 * DOSE_RATE
INFL = jnp.array([5., 3., 1., 1., 1., 1.]); BETA_ARR = jnp.array(BETAS)
E_ctrl = np.load("E_ctrl.npy"); T_surr = np.load("surrogate_k16_spline_T_out.npy"); n_pts = len(T_surr); T_C = T_surr - 273.15
E_GRID = np.linspace(0.95, 2.05, N_TRAPS); E_J = jnp.array(E_GRID)
SPL = jnp.array(np.stack([CubicSpline(E_ctrl, np.eye(5)[j], bc_type="natural")(E_GRID) for j in range(5)], axis=1))
_noise = json.load(open("noise_model_ana05.json"))["recommended"]; NOISE_A, NOISE_B = _noise["a"], _noise["b"]
def load_data(cup, ph, beta):
    d = np.loadtxt(f"tl_data_ana05cups_{cup}_ph{ph}_b{beta:g}.txt", skiprows=1); return np.interp(T_surr, d[:, 0], d[:, 1])
I_data_np = np.stack([np.stack([[load_data(c_, ph, b) for b in BETAS] for ph in PREHEATS]) for c_ in CUPS]); I_data = jnp.array(I_data_np)
FIT_MASK = jnp.array(T_surr >= 70.0 + 273.15)
I_peak = jnp.array(np.max(savgol_filter(I_data_np, 15, 2, axis=-1)[..., np.asarray(FIT_MASK)], axis=-1)); PH_ARR = jnp.array(PREHEATS, dtype=float)
MAAD_SERIES = []
for o in json.load(open("maad_windows.json"))["obs"]:
    t = np.array(o["t_beta_s"]); D = t * DOSE_RATE; R = np.array(o["ratio"]); keep = ~np.array(o["clipped"])
    PHm = 20.0 if o["window"] == "w1" else float(o["PH_C"])
    wmask = np.asarray((T_C >= o["lo"]) & (T_C < o["hi"]), dtype=float)
    MAAD_SERIES.append(dict(wi=int(o["window"][1]) - 1, PH=PHm, D=jnp.array(D), R=jnp.array(R), keep=jnp.array(keep.astype(float)), w=jnp.array(wmask), name=f'{o["batch"]}/{o["window"]}'))
bounds = [(-1.0, 6.0), (3.0, 17.0)] + [(-1.0, 1.0)] * 5 + [(8.0, 11.0)] * 5 + [(2.0, 5.0)] + [(0.0, 4.3)] * 3 + ([(1.0, 2.0)] if IS_GOK else [])
ndim = len(bounds); LO = jnp.array([b[0] for b in bounds]); HI = jnp.array([b[1] for b in bounds])
def kin(p):
    ls = p[0] * E_J + p[1] + SPL @ p[2:7]; return 10.0 ** ls, 10.0 ** (SPL @ p[7:12])
def bvec(p): return jnp.full(N_TRAPS, p[16]) if IS_GOK else None
def x0_of(D, p): return 1.0 - jnp.exp(-D / 10.0 ** p[12])
def vhr_curves(p):
    s_, N_ = kin(p)
    if IS_GOK:
        b_ = bvec(p); x0 = x0_of(DOSE_VHR, p)
        f = lambda ph, beta: A0M.gok_curve(E_J, s_, N_, b_, beta, ph + 273.15, jnp.asarray(T_surr), nfine=4000, x0=x0)[0]
    else:
        f = lambda ph, beta: A0M.a0_curve(E_J, s_, N_, beta, ph + 273.15, jnp.asarray(T_surr), nfine=4000)[0]
    out = jax.vmap(lambda ph: jax.vmap(lambda b: f(ph, b))(BETA_ARR))(PH_ARR); mx = jnp.max(out, axis=-1, keepdims=True)
    return (out / jnp.maximum(mx, 1e-300))[None] * I_peak[..., None], jnp.any(mx <= 1e-250)
def var_of(c):
    v = jnp.maximum((NOISE_A + NOISE_B * c) * INFL[None, :, None], 1e-6)
    if ERR: v = v + (ERR_F * c) ** 2 + (ST0 ** 2 + (ST1 * BETA_ARR) ** 2)[None, :, None] * jnp.gradient(c, T_surr, axis=-1) ** 2
    return v
def chi2_vhr(p):
    c, coll = vhr_curves(p); v = var_of(c); t = (c - I_data) ** 2 / v + (jnp.log(v) if ERR else 0.0)
    return jnp.where(coll, 1e18, jnp.sum(jnp.where(FIT_MASK[None, None, :], t, 0.0)))
def signal(p, S, D):
    x0 = x0_of(D + 10.0 ** p[13 + S["wi"]], p)
    if not IS_GOK: return x0                                             # A=0: the ratio of window signals is just the ratio of fillings
    s_, N_ = kin(p)
    return jnp.sum(LM.gok_light(E_J, s_, N_, bvec(p), x0, 1.0, S["PH"] + 273.15, jnp.asarray(T_surr), nfine=4000) * S["w"])
def maad_pred(p, S):
    sig = jax.vmap(lambda D: signal(p, S, D))(S["D"])
    x0t = 1.0 - jnp.exp(-D_TEST / 10.0 ** p[12])                           # test dose: no natural contribution
    if IS_GOK:
        s_, N_ = kin(p); tst = jnp.sum(LM.gok_light(E_J, s_, N_, bvec(p), x0t, 1.0, S["PH"] + 273.15, jnp.asarray(T_surr), nfine=4000) * S["w"])
    else: tst = x0t
    return jnp.minimum(sig / jnp.maximum(tst, 1e-12 * jnp.max(sig) + 1e-300), 1e3)
def maad_frac(p, S, D=None):
    """fraction of saturation of the window signal: A=0 -> filling x0; GOK -> window light at x0 relative to the fully filled traps"""
    Dv = S["D"] if D is None else D
    if not IS_GOK: return jax.vmap(lambda d: x0_of(d + 10.0 ** p[13 + S["wi"]], p))(Dv)
    s_, N_ = kin(p); b_ = bvec(p); T_ = jnp.asarray(T_surr)
    sat = jnp.sum(LM.gok_light(E_J, s_, N_, b_, 1.0, 1.0, S["PH"] + 273.15, T_, nfine=4000) * S["w"])
    sig = jax.vmap(lambda d: jnp.sum(LM.gok_light(E_J, s_, N_, b_, x0_of(d + 10.0 ** p[13 + S["wi"]], p), 1.0, S["PH"] + 273.15, T_, nfine=4000) * S["w"]))(Dv)
    return sig / jnp.maximum(sat, 1e-300)
def amp_of(f, S):
    y = S["R"]; sg = MAAD_SIG * y + 0.03; w = S["keep"] / sg ** 2
    return jnp.sum(w * f * y) / jnp.maximum(jnp.sum(w * f * f), 1e-300)
def chi2_maad(p):
    tot = 0.0
    for S in MAAD_SERIES:
        if MAAD_MODE == "nN":
            f = maad_frac(p, S); y = S["R"]; r = (amp_of(f, S) * f - y) / (MAAD_SIG * y + 0.03)
        else: r = (maad_pred(p, S) - S["R"]) / (MAAD_SIG * S["R"] + 0.03)
        tot += jnp.sum(jnp.where(S["keep"] > 0, r ** 2, 0.0))
    return tot
def log_prior(p):
    ok = jnp.all((p >= LO) & (p <= HI)); d = p[2:7]; lN = p[7:12]
    lp = -0.5 * jnp.sum((d / 0.12) ** 2) - 0.5 * jnp.sum(((d[2:] - 2 * d[1:-1] + d[:-2]) / 0.15) ** 2) - 0.5 * jnp.sum((jnp.maximum(jnp.diff(lN), 0.0) / 0.1) ** 2)
    return jnp.where(ok, lp, -jnp.inf)
def objective(p): return 0.5 * jnp.nan_to_num(chi2_vhr(p) / VHR_SCALE + MAAD_W * chi2_maad(p), nan=1e18, posinf=1e18)
def neg_log_post(p):
    lp = log_prior(p); return jnp.where(jnp.isfinite(lp), objective(p) - lp, 1e30)
def _smooth(p):
    pc = jnp.clip(p, LO, HI); lp = log_prior(pc); return objective(p) - jnp.where(jnp.isfinite(lp), lp, 0.0)
nlp_batch = jax.jit(jax.vmap(neg_log_post)); nlp_vg = jax.jit(jax.value_and_grad(_smooth))
if __name__ == "__main__":
    print(f"{TAG}: backend={jax.default_backend()} ndim={ndim} cups={CUPS} VHR_SCALE={VHR_SCALE} ERR={ERR} N_TRAPS={N_TRAPS}", flush=True)
    rng = np.random.default_rng(11); lo, hi = np.array(LO), np.array(HI)
    nstart = int(os.environ.get("NSTART", 40)); nrand = int(os.environ.get("NRAND", 6000)); RB = 200
    cand = lo + (hi - lo) * rng.random((nrand, ndim)); cand[:, 2:7] *= 0.5
    f0 = np.nan_to_num(np.concatenate([np.asarray(nlp_batch(jnp.array(cand[i:i + RB]))) for i in range(0, nrand, RB)]), nan=1e30, posinf=1e30)
    starts = cand[np.argsort(f0)[:nstart]]; print(f"random search best {f0.min():.4e}", flush=True)
    res = []; t0 = time.time()
    for k, x0_ in enumerate(starts):
        def fg(x):
            v, g = nlp_vg(jnp.array(x)); g = np.asarray(g, dtype=float)
            return (float(v), g) if (np.isfinite(float(v)) and np.all(np.isfinite(g))) else (1e30, np.zeros_like(g))
        r = minimize(fg, x0_, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=3000, maxfun=6000)); res.append((r.fun if np.isfinite(r.fun) else 1e30, r.x))
        if k % 10 == 0: print(f"  start {k}/{nstart} done, best so far {min(t[0] for t in res):.4e}, {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda t: t[0]); p_map = res[0][1]; pj = jnp.array(p_map)
    cv = float(chi2_vhr(pj)); cm = float(chi2_maad(pj)); c_b, _ = vhr_curves(pj); v_b = var_of(c_b)
    red = np.asarray(jnp.sum(jnp.where(FIT_MASK[None, None, :], (c_b - I_data) ** 2 / v_b, 0.0), axis=(0, 1, 3)) / (len(CUPS) * N_PH * int(np.sum(np.asarray(FIT_MASK)))))
    print("best 6 objectives:", [f"{r[0]:.4e}" for r in res[:6]]); x = p_map
    print(f"best: slope={x[0]:.2f} int={x[1]:.2f} logN={np.round(x[7:12],2)} logD0={x[12]:.2f} logDnat_w={np.round(x[13:16],2)}" + (f" b={x[16]:.2f}" if IS_GOK else ""))
    print(f"chi2 VHR = {cv:.4e} (reduced per rate {np.round(red,1)}),  chi2 MAAD = {cm:.2f} for {int(sum(np.sum(np.asarray(S['keep'])) for S in MAAD_SERIES))} points", flush=True)
    json.dump(dict(tag=TAG, model=MODEL, n_traps=N_TRAPS, best=p_map.tolist(), best_objective=float(res[0][0]), chi2_vhr=cv, chi2_maad=cm, reduced_chi2_per_rate=red.tolist(),
                   top_objectives=[float(r[0]) for r in res[:20]], top_solutions=[r[1].tolist() for r in res[:20]], ERR=ERR, VHR_SCALE=VHR_SCALE, MAAD_SIGMA=MAAD_SIG), open(f"gcd_params_{TAG}.json", "w"), indent=1)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, S in zip(axes.flat, MAAD_SERIES):
        Dg = np.logspace(1.5, 4.3, 50); ax.semilogx(Dg, np.asarray(maad_frac(pj, S, jnp.array(Dg))), "b-"); a_ = float(amp_of(maad_frac(pj, S), S)); yn = np.asarray(S["R"]) / a_
        keep = np.asarray(S["keep"]) > 0; D = np.maximum(np.asarray(S["D"]), 5.0); ax.semilogx(D[keep], yn[keep], "ko"); ax.semilogx(D[~keep], yn[~keep], "rx")
        ax.set_title(S["name"] + f" ({MODEL})", fontsize=9); ax.set_xlabel("beta dose (Gy)"); ax.set_ylabel("fraction of saturation n/N"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"gcd_{TAG}_maad.png", dpi=100); print("saved", f"gcd_params_{TAG}.json")

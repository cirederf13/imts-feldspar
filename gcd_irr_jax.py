"""
IMTS inversion with the IRRADIATION-in-ODE surrogate (retrain_6beta_irr/): the filling is a prediction (dose, D0).
Fits (i) the VHR cup glow curves (9 preheats x 6 rates, dose 300.05 Gy, shapes scaled to the data peaks) and
     (ii) the MAAD growth series (windowed Lx/Tx against beta dose, 3 preheat batches x 3 windows, 7 doses + natural disk, test dose 255.1 Gy).
usage: python gcd_irr_jax.py <cup17|cup19|cup21|joint>
Parameters: [slope, intercept, delta(5), logN(5), logD0, [unused, fixed at 0], logDnat_w1, logDnat_w2, logDnat_w3]   (free 5-point shared N(E); logDnat_w = natural signal of each glow-curve window as an equivalent lab dose)
env: WEIGHTS, MAAD=1/0, MAAD_SIGMA (relative error on Lx/Tx, default 0.15), MAAD_W (weight of the MAAD chi2), VHR_SCALE (VHR chi2 is divided by this),
     ERR=1 (physical error model for the glow curves), RATES, INFLATE, NSTART, NRAND, MCMC, NSTEPS, TAG_SUFFIX, LOGN_HI (cap on log10 N per trap), S_CONST=1 (s independent of E), LOGD0_LO (lower bound of log10 D0), MAAD_WINDOWS (e.g. w2,w3)
"""
import json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

CUP = sys.argv[1] if len(sys.argv) > 1 else "cup17"
CUPS = ["cup17", "cup19", "cup21"] if CUP == "joint" else [CUP]
USE_MAAD = os.environ.get("MAAD", "1") == "1"
MAAD_MODE = os.environ.get("MAAD_MODE", "nN")      # nN: fraction of saturation per series (free amplitude per series, profiled); ratio: Lx/Tx against the test dose
MAAD_SIG = float(os.environ.get("MAAD_SIGMA", 0.15)); MAAD_W = float(os.environ.get("MAAD_W", 1.0)); VHR_SCALE = float(os.environ.get("VHR_SCALE", 1.0))
ERR = os.environ.get("ERR", "0") == "1"
ERR_F, ST0, ST1 = 0.03, 3.0, 0.3
RATE_IDX = [int(x) for x in os.environ.get("RATES", "0,1,2,3,4,5").split(",")]
TAG = f"irr_{CUP}" + ("_maad" if USE_MAAD else "") + ("_ERR" if ERR else "") + ("_r" + "".join(map(str, RATE_IDX)) if RATE_IDX != list(range(6)) else "") + os.environ.get("TAG_SUFFIX", "")
BETAS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
PREHEATS = [60, 90, 120, 150, 180, 210, 240, 270, 300]
N_CTRL, N_PH = 5, 9
DOSE_VHR = 300.05; DOSE_RATE = 0.1566; D_TEST = 1629.0 * DOSE_RATE
INFL = jnp.array([float(x) for x in os.environ.get("INFLATE", "5,3,1,1,1,1").split(",")])
RATE_W = jnp.array([1.0 if i in RATE_IDX else 0.0 for i in range(6)])

W = np.load(os.environ.get("WEIGHTS", "retrain_6beta_irr/surrogate_irr_final_weights.npz"))
Ws = [(jnp.array(W[f"W{i}"]), jnp.array(W[f"b{i}"])) for i in range(4)]
scaler_mean, scaler_scale = jnp.array(W["scaler_mean"]), jnp.array(W["scaler_scale"])
N_CURVE = int(W["n_curve"]); LP_MEAN, LP_STD, LP_W = jnp.array(W["lp_mean"]), jnp.array(W["lp_std"]), float(W["lp_w"])
E_ctrl = np.load("E_ctrl.npy"); T_surr = np.load("surrogate_k16_spline_T_out.npy"); n_pts = len(T_surr)
T_C = jnp.array(T_surr - 273.15); DT_SURR = float(T_surr[1] - T_surr[0])
_noise = json.load(open("noise_model_ana05.json"))["recommended"]; NOISE_A, NOISE_B = _noise["a"], _noise["b"]

def load_data(cup, ph, beta):
    d = np.loadtxt(f"tl_data_ana05cups_{cup}_ph{ph}_b{beta:g}.txt", skiprows=1); return np.interp(T_surr, d[:, 0], d[:, 1])
I_data_np = np.stack([np.stack([[load_data(c_, ph, b) for b in BETAS] for ph in PREHEATS]) for c_ in CUPS])   # (n_cup, 9, 6, n_pts)
I_data = jnp.array(I_data_np); FIT_MASK = jnp.array(T_surr >= 70.0 + 273.15)
I_peak = jnp.array(np.max(savgol_filter(I_data_np, 15, 2, axis=-1)[..., np.asarray(FIT_MASK)], axis=-1))
PH_ARR = jnp.array(PREHEATS, dtype=float); LOGD_VHR = float(np.log10(DOSE_VHR))

# ---- MAAD observables ----
MAAD = json.load(open("maad_windows.json"))["obs"] if USE_MAAD else []
MAAD_WINDOWS = os.environ.get("MAAD_WINDOWS", "w1,w2,w3").split(",")                 # MAAD_WINDOWS=w2,w3: leave the 90-150 C window out of the growth fit
MAAD = [o for o in MAAD if o["window"] in MAAD_WINDOWS]
MAAD_SERIES = []
for o in MAAD:
    t = np.array(o["t_beta_s"]); D = t * DOSE_RATE; R = np.array(o["ratio"]); keep = ~np.array(o["clipped"])
    PHm = 20.0 if o["window"] == "w1" else float(o["PH_C"])           # the 90-150 C window is always the first part of the read-out of a fresh dose (also inside the preheat TL)
    wmask = np.asarray((np.asarray(T_C) >= o["lo"]) & (np.asarray(T_C) < o["hi"]), dtype=float)
    MAAD_SERIES.append(dict(wi=int(o['window'][1]) - 1, PH=PHm, D=jnp.array(D), R=jnp.array(R), keep=jnp.array(keep.astype(float)), w=jnp.array(wmask), name=f'{o["batch"]}/{o["window"]}'))

# ---- parameters ----
slope_lo, slope_hi, intercept_lo, intercept_hi = -1.0, 6.0, 3.0, 17.0
logN_lo, logN_hi = float(os.environ.get('LOGN_LO', 8.0)), float(os.environ.get('LOGN_HI', 11.0))      # LOGN_HI: cap on log10 N per trap (limits the retrapping strength)
bounds = [(slope_lo, slope_hi), (intercept_lo, intercept_hi)] + [(-1.0, 1.0)] * N_CTRL + [(logN_lo, logN_hi)] * 5 + [(float(os.environ.get('LOGD0_LO', 2.3)), float(os.environ.get('LOGD0_HI', 5.0))), ((float(os.environ.get('LOGM0_LO', 9.7)), float(os.environ.get('LOGM0_HI', 12.7))) if os.environ.get('NEUTRAL', '0') == '1' else ((-1.0, 2.0) if os.environ.get('KAPPA_FREE', '0') == '1' else (0.0, 1e-9)))]      # logD0 (LOGD0_LO/LOGD0_HI); slot 13 = log10 M0 if NEUTRAL=1 (charge-neutral recombination centres), else trap-dependent capture exponent kappa (KAPPA_FREE=1) else fixed at 0
if USE_MAAD: bounds += [(1.5, 4.3)] * 3
if os.environ.get("S_CONST", "0") == "1":                                       # s independent of E: slope = 0, delta = 0, only the intercept is free
    bounds[0] = (0.0, 1e-9)
    for _i in range(2, 7): bounds[_i] = (0.0, 1e-9)
ndim = len(bounds); LO = jnp.array([b[0] for b in bounds]); HI = jnp.array([b[1] for b in bounds])
SIGMA_DELTA = 0.12; MONO = 0.1

def mlp(x):
    h = (x - scaler_mean) / scaler_scale
    for i in range(3): h = jax.nn.relu(h @ Ws[i][0] + Ws[i][1])
    return h @ Ws[3][0] + Ws[3][1]

def forward(p, PH, logD):
    PH = jnp.asarray(PH, dtype=float)
    x = jnp.concatenate([p[:12], PH[None], jnp.clip(logD, 1.5, 4.3)[None], p[12:14]])        # slope,int,delta5,logN5 | PH, logD, logD0, unused (0)
    o = mlp(x); curves = jnp.maximum(o[:N_CURVE].reshape(6, n_pts), 0.0)
    logpk = o[N_CURVE:] / LP_W * LP_STD + LP_MEAN
    return curves, logpk

def vhr_curves(p):
    out = jax.vmap(lambda ph: forward(p, ph, jnp.array(LOGD_VHR))[0])(PH_ARR)                  # (9, 6, n_pts)
    mx = jnp.max(out, axis=-1, keepdims=True)
    return (out / jnp.maximum(mx, 1e-30))[None] * I_peak[..., None], jnp.any(mx <= 0)

def var_of(c):
    v = jnp.maximum((NOISE_A + NOISE_B * c) * INFL[None, :, None], 1e-6)
    if ERR:
        dIdT = jnp.gradient(c, T_surr, axis=-1); sT2 = ST0 ** 2 + (ST1 * jnp.array(BETAS)) ** 2
        v = v + (ERR_F * c) ** 2 + sT2[None, :, None] * dIdT ** 2
    return v

def chi2_vhr(p):
    c, coll = vhr_curves(p); v = var_of(c)
    t = (c - I_data) ** 2 / v + (jnp.log(v) if ERR else 0.0)
    return jnp.where(coll, 1e18, jnp.sum(jnp.where(FIT_MASK[None, None, :], t, 0.0) * RATE_W[None, :, None]))

def window_signal(p, PH, D, wmask):
    c, lp = forward(p, PH, jnp.log10(D))
    return 10.0 ** lp[3] * jnp.sum(c[3] * wmask) * DT_SURR                                      # beta = 1 C/s curve

def maad_pred(p, S):
    Dnat = 10.0 ** p[14 + S['wi']]
    sig = jax.vmap(lambda D: window_signal(p, S["PH"], D + Dnat, S["w"]))(S["D"])
    tst = window_signal(p, S["PH"], jnp.array(D_TEST), S["w"])
    tst_safe = jnp.maximum(tst, 1e-12 * jnp.max(sig) + 1e-300)       # a model with no light at the test dose gives a ratio of 0, not inf/nan
    return jnp.minimum(sig / tst_safe, 1e3)

D_REF = 19950.0                                                       # reference 'saturation' dose (20 kGy, the upper edge of the surrogate)
def maad_frac(p, S, D=None):
    """modelled fraction of saturation of the window signal at the MAAD doses (natural added as an equivalent dose per window)"""
    Dnat = 10.0 ** p[14 + S['wi']]; Dv = S["D"] if D is None else D
    sig = jax.vmap(lambda d: window_signal(p, S["PH"], d + Dnat, S["w"]))(Dv)
    return sig / jnp.maximum(window_signal(p, S["PH"], jnp.array(D_REF), S["w"]), 1e-300)
def amp_of(f, S):
    y = S["R"]; sg = MAAD_SIG * y + 0.03; w = S["keep"] / sg ** 2
    return jnp.sum(w * f * y) / jnp.maximum(jnp.sum(w * f * f), 1e-300)
def chi2_maad(p):
    tot = 0.0
    for S in MAAD_SERIES:
        if MAAD_MODE == "nN":
            f = maad_frac(p, S); y = S["R"]; r = (amp_of(f, S) * f - y) / (MAAD_SIG * y + 0.03)
        else:
            r = (maad_pred(p, S) - S["R"]) / (MAAD_SIG * S["R"] + 0.03)
        tot += jnp.sum(jnp.where(S["keep"] > 0, r ** 2, 0.0))
    return tot

def log_prior(p):
    ok = jnp.all((p >= LO) & (p <= HI)); d = p[2:7]; lN = p[7:12]
    lp = -0.5 * jnp.sum((d / SIGMA_DELTA) ** 2) - 0.5 * jnp.sum(((d[2:] - 2 * d[1:-1] + d[:-2]) / 0.15) ** 2)
    lp += -0.5 * jnp.sum((jnp.maximum(jnp.diff(lN), 0.0) / MONO) ** 2)
    return jnp.where(ok, lp, -jnp.inf)

def objective(p):
    v = chi2_vhr(p) / VHR_SCALE
    if USE_MAAD: v = v + MAAD_W * chi2_maad(p)
    return 0.5 * jnp.nan_to_num(v, nan=1e18, posinf=1e18)

def neg_log_post(p):
    lp = log_prior(p); return jnp.where(jnp.isfinite(lp), objective(p) - lp, 1e30)

def _smooth(p):
    pc = jnp.clip(p, LO, HI); lp = log_prior(pc); return objective(p) - jnp.where(jnp.isfinite(lp), lp, 0.0)
nlp_batch = jax.jit(jax.vmap(neg_log_post)); nlp_vg = jax.jit(jax.value_and_grad(_smooth))

if __name__ == "__main__":
    print(f"{TAG}: backend={jax.default_backend()} ndim={ndim} cups={CUPS} MAAD={USE_MAAD} (sigma {MAAD_SIG}, weight {MAAD_W}) VHR_SCALE={VHR_SCALE} ERR={ERR}", flush=True)
    rng = np.random.default_rng(11); lo, hi = np.array(LO), np.array(HI)
    nstart = int(os.environ.get("NSTART", 40)); nrand = int(os.environ.get("NRAND", 10000)); RB = 500
    cand = lo + (hi - lo) * rng.random((nrand, ndim)); cand[:, 2:7] *= 0.5
    f0 = np.concatenate([np.asarray(nlp_batch(jnp.array(cand[i:i + RB]))) for i in range(0, nrand, RB)]); f0 = np.nan_to_num(f0, nan=1e30, posinf=1e30)
    starts = cand[np.argsort(f0)[:nstart]]; print(f"random search best {f0.min():.4e}", flush=True)
    res = []; t0 = time.time()
    for k, x0 in enumerate(starts):
        def fg(x):
            v, g = nlp_vg(jnp.array(x)); g = np.asarray(g, dtype=float)
            return (float(v), g) if (np.isfinite(float(v)) and np.all(np.isfinite(g))) else (1e30, np.zeros_like(g))
        r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=3000, maxfun=6000)); res.append((r.fun if np.isfinite(r.fun) else 1e30, r.x))
        if k % 10 == 0: print(f"  start {k}/{nstart} done, best so far {min(t[0] for t in res):.4e}, {time.time()-t0:.0f}s", flush=True)
    res.sort(key=lambda t: t[0]); p_map = res[0][1]
    print("best 8 objectives:", [f"{r[0]:.4e}" for r in res[:8]])
    for f, x in res[:5]:
        print(f"   f={f:.4e} slope={x[0]:.2f} int={x[1]:.2f} logN={np.round(x[7:12],2)} logD0={x[12]:.2f}" + (f" logDnat_w={np.round(x[14:17],2)}" if USE_MAAD else ""))
    pj = jnp.array(p_map); cv = float(chi2_vhr(pj)); cm = float(chi2_maad(pj)) if USE_MAAD else float("nan")
    c_b, _ = vhr_curves(pj); v_b = var_of(c_b)
    red = np.asarray(jnp.sum(jnp.where(FIT_MASK[None, None, :], (c_b - I_data) ** 2 / v_b, 0.0), axis=(0, 1, 3)) / (len(CUPS) * N_PH * int(np.sum(np.asarray(FIT_MASK)))))
    print(f"chi2 VHR = {cv:.4e} (reduced per rate {np.round(red,1)}),  chi2 MAAD = {cm:.2f} for {int(sum(np.sum(np.asarray(S['keep'])) for S in MAAD_SERIES)) if USE_MAAD else 0} points", flush=True)
    out = dict(tag=TAG, cup=CUP, best=p_map.tolist(), best_objective=float(res[0][0]), chi2_vhr=cv, chi2_maad=cm, reduced_chi2_per_rate=red.tolist(),
               top_objectives=[float(r[0]) for r in res[:20]], top_solutions=[r[1].tolist() for r in res[:20]], ERR=ERR, MAAD=USE_MAAD, MAAD_SIGMA=MAAD_SIG, VHR_SCALE=VHR_SCALE)
    json.dump(out, open(f"gcd_params_{TAG}.json", "w"), indent=1)
    if os.environ.get("MCMC", "0") == "1":
        import emcee
        nsteps = int(os.environ.get("NSTEPS", 20000)); nw = max(64, 2 * ndim + 2)
        lpv = jax.jit(jax.vmap(lambda p: -neg_log_post(p))); p0 = None
        for scale in [0.01, 0.003, 0.001, 3e-4, 1e-4, 0.0]:
            cand = np.clip(p_map + scale * (hi - lo) * rng.standard_normal((nw, ndim)), lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo))
            if np.all(np.isfinite(np.asarray(lpv(jnp.array(cand))))): p0 = cand; break
        if p0 is None: p0 = np.tile(np.clip(p_map, lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo)), (nw, 1))
        sampler = emcee.EnsembleSampler(nw, ndim, lambda P: np.asarray(lpv(jnp.array(P))), vectorize=True)
        st = sampler.run_mcmc(p0, nsteps, progress=False, skip_initial_state_check=True); sampler.reset()
        sampler.run_mcmc(st, nsteps, progress=False, skip_initial_state_check=True)
        flat = sampler.get_chain(flat=True, thin=5); np.save(f"{TAG}_flat_samples.npy", flat)
        out["median"] = np.median(flat, axis=0).tolist(); out["p16"] = np.percentile(flat, 16, axis=0).tolist(); out["p84"] = np.percentile(flat, 84, axis=0).tolist()
        json.dump(out, open(f"gcd_params_{TAG}.json", "w"), indent=1)
    # MAAD growth plot at the best fit
    if USE_MAAD:
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for ax, S, o in zip(axes.flat, MAAD_SERIES, MAAD):
            Dg = np.logspace(1.5, 4.3, 60); Dnat = 10.0 ** p_map[14 + S['wi']]
            fg = np.asarray(maad_frac(pj, S, jnp.array(Dg))); a_ = float(amp_of(maad_frac(pj, S), S))
            ax.semilogx(Dg, fg, "b-"); keep = np.asarray(S["keep"]) > 0; D = np.maximum(np.asarray(S["D"]), 5.0); yn = np.asarray(S["R"]) / a_
            ax.semilogx(D[keep], yn[keep], "ko"); ax.semilogx(D[~keep], yn[~keep], "rx"); ax.set_title(S["name"] + f"  (model PH={S['PH']:.0f} C)", fontsize=9)
            ax.set_xlabel("beta dose (Gy)"); ax.set_ylabel("fraction of saturation n/N (data scaled by fitted amplitude)"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"gcd_{TAG}_maad.png", dpi=100)
    print("saved", f"gcd_params_{TAG}.json")

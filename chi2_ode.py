"""chi2 (VHR glow curves + MAAD growth) of a fitted parameter set evaluated with the FULL ODE instead of the surrogate.
usage: JAX_PLATFORMS=cpu ERR=1 VHR_SCALE=100 python chi2_ode.py <params.json> [n_jobs]   (same env as the fit; uses gcd_irr_jax observables)
Prints surrogate chi2 (should reproduce the json) and ODE chi2."""
import sys, os, json, numpy as np
sys.argv, argv0 = ["x", "joint"], sys.argv
os.environ.setdefault("MAAD", "1")
sys.path.insert(0, "retrain_6beta_irr")
import gcd_irr_jax as M
import generate_training_data_irr_cpu as G
from joblib import Parallel, delayed
import jax.numpy as jnp
sys.argv = argv0

def ode_run(p, PH, D, ib):
    slope, ic = p[0], p[1]; delta, logN, logD0 = np.array(p[2:7]), np.array(p[7:12]), p[12]
    ES = float(os.environ.get("ESCALE", 1.0)); EG = G.E_GRID * ES                     # ESCALE: all trap energies scaled (s keeps its parametrisation on the original grid)
    s = 10.0 ** (slope * G.E_GRID + ic + G.spline_eval(delta)); N = 10.0 ** G.spline_eval(logN) * (16.0 / G.N_FINE); A = G.A_val * (np.ones_like(G.E_GRID) if G.NEUTRAL else 10.0 ** (float(p[13]) * (G.E_GRID - 1.5)))
    if G.NEUTRAL: G.M0 = 10.0 ** float(p[13])
    n_irr = G.irradiate(EG, s, N, A, D, 10.0 ** logD0)
    if n_irr is None: return None
    out = [G.run_imts_irr(EG, s, N, A, n_irr, G.BETAS[b], PH + 273.15) for b in ib]
    return None if any(o is None for o in out) else [(o[0], o[1]) for o in out]

def chi2_vhr_ode(p, nj):
    SEL = [i for i in range(len(M.PREHEATS)) if (os.environ.get("SUB_PH") is None or i in [int(v) for v in os.environ["SUB_PH"].split(",")])]
    res = Parallel(n_jobs=nj)(delayed(ode_run)(p, M.PREHEATS[i], M.DOSE_VHR, range(6)) for i in SEL)
    it = iter(res); res = [next(it) if i in SEL else None for i in range(len(M.PREHEATS))]; first = next(r for r in res if r is not None); res = [r if r is not None else first for r in res]           # unselected preheats: copy, weight 0 below
    c = np.stack([[r[0] for r in rr] for rr in res])[None]                       # (1,9,6,n_pts) peak-normalised
    c = c * np.asarray(M.I_peak)[..., None]; v = np.asarray(M.var_of(jnp.array(c)))
    t = (c - np.asarray(M.I_data)) ** 2 / v + (np.log(v) if M.ERR else 0.0)
    PHW = np.array([1.0 if i in SEL else 0.0 for i in range(len(M.PREHEATS))])[None, :, None, None]; RW = np.asarray(M.RATE_W, float).copy()
    if os.environ.get("SUB_RATES") is not None: RW = RW * np.array([1.0 if i in [int(v) for v in os.environ["SUB_RATES"].split(",")] else 0.0 for i in range(6)])
    return float(np.sum(np.where(np.asarray(M.FIT_MASK)[None, None, :], t, 0.0) * RW[None, :, None] * PHW))

def chi2_maad_ode(p, nj):
    keys = {}                                                                   # unique (PH, D) -> beta=1 C/s curve signal for each window mask
    SER = [S for S in M.MAAD_SERIES if os.environ.get("SUB_MAAD") is None or S["name"].split("/")[0] in os.environ["SUB_MAAD"].split(",")]
    for S in SER:
        Dnat = 10.0 ** p[14 + S["wi"]]
        for d in list(np.asarray(S["D"]) + Dnat) + [M.D_REF]: keys[(S["PH"], round(float(d), 6))] = None
    ks = list(keys); res = Parallel(n_jobs=nj)(delayed(ode_run)(p, ph, d, [3]) for ph, d in ks)
    sig = {k: r[0] for k, r in zip(ks, res) if r is not None}
    tot, n = 0.0, 0
    for S in SER:
        Dnat = 10.0 ** p[14 + S["wi"]]; w = np.asarray(S["w"]); win = lambda k: 10.0 ** sig[k][1] * np.sum(sig[k][0] * w) * M.DT_SURR
        f = np.array([win((S["PH"], round(float(d), 6))) for d in np.asarray(S["D"]) + Dnat]) / max(win((S["PH"], round(float(M.D_REF), 6))), 1e-300)
        y = np.asarray(S["R"]); keep = np.asarray(S["keep"]) > 0; sg = M.MAAD_SIG * y + 0.03; ww = keep / sg ** 2
        a = np.sum(ww * f * y) / max(np.sum(ww * f * f), 1e-300); r = (a * f - y) / sg
        tot += np.sum(r[keep] ** 2); n += keep.sum()
    return tot, n

if __name__ == "__main__":
    f = argv0[1]; nj = int(argv0[2]) if len(argv0) > 2 else 4; d = json.load(open(f)); p = np.array(d["best"])
    print(f"{f}\n  json (surrogate): chi2_vhr={d['chi2_vhr']:.4e} chi2_maad={d['chi2_maad']:.1f}")
    pj = jnp.array(p); print(f"  recomputed surrogate: chi2_vhr={float(M.chi2_vhr(pj)):.4e} chi2_maad={float(M.chi2_maad(pj)):.1f}")
    cv = chi2_vhr_ode(p, nj); cm, n = chi2_maad_ode(p, nj)
    print(f"  FULL ODE            : chi2_vhr={cv:.4e} chi2_maad={cm:.1f} ({n} pts)", flush=True)

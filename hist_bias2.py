"""Cold-to-warm step history: T0 before 100 ka, 0 C from 100 to 20 ka, Tw for the last 20 ka.  Constant-T fit of the natural profile (IMTS kinetics lmBc, noise-free).  usage: IMTS_MODEL=imts python hist_bias2.py"""
import json, numpy as np, os, sys
os.environ["IMTS_PARAMS"] = "start_joint1_final.json"
import fwd, pgrid
fwd.init_main(); ref = pgrid.Ref(os.environ["IMTS_PARAMS"]); x = np.array(json.load(open("lm_lmBc.json"))["x"]); m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[fwd.IDX["a"]], x[fwd.IDX["b"]])
sat = fwd.NS["lab_sat"](); sd = np.asarray(fwd.HA.NN_SD, float); NS = fwd.NS
def prof(temps, durs): NS["DUR_MYR"] = list(durs); return NS["hist_model"](np.asarray(temps, float), sat)
Ts = np.arange(-10, 61, 0.5); curves = np.array([prof([T], [10.0]) for T in Ts])
def fit(t, sl): c = (((curves[:, sl] - t[sl]) / sd[sl]) ** 2).sum(1); i = int(np.argmin(c)); return Ts[i], c[i]
kB = 8.617e-5
def tau(E, T): return 1.0 / (1e12 * np.exp(-E / (kB * (273.15 + T))))
print("history (>100 ka | 100-20 ka | last 20 ka)      low(205-295)  high(305-345)  all   chi2_all")
for T0, Tm_, Tw in [(0, 0, 10), (0, 0, 5), (0, 0, 15), (10, 0, 10), (10, 10, 10), (0, 0, 0)]:
    t = prof([T0, Tm_, Tw], [9.9, 0.08, 0.02]); lo, hi, al = fit(t, slice(0, 10)), fit(t, slice(10, 15)), fit(t, slice(0, 15))
    print(f"{T0:3d} | {Tm_:3d} | {Tw:3d} C                          {lo[0]:6.1f}      {hi[0]:6.1f}    {al[0]:6.1f}   {al[1]:6.1f}")
print("profile n/N (window 205..345) for 0|0|10 :", np.round(prof([0, 0, 10], [9.9, 0.08, 0.02]), 3).tolist()); print("profile n/N constant 10 C:", np.round(prof([10], [10.0]), 3).tolist()); print("profile n/N constant 0 C :", np.round(prof([0], [10.0]), 3).tolist()); print("noise sd:", np.round(sd, 3).tolist())

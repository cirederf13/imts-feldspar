import json, numpy as np, os, sys
TAG = sys.argv[1]
import fwd, pgrid
fwd.init_main(); REF = {"imts": "start_joint1_final.json", "a0": "gcd_params_irra024_joint_maad_ERR_nN_vs100.json", "gok": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json", "gok2": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json"}[fwd.MODEL]
ref = pgrid.Ref(REF); x = np.array(json.load(open(f"lm_{TAG}.json"))["x"]); m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[fwd.IDX["a"]], x[fwd.IDX["b"]])
Ts = np.arange(30, 61, 2.0); c = np.array([float(np.sum(fwd.prof_resid([T] * 6) ** 2)) for T in Ts]); i = int(np.argmin(c))
print(f"{TAG}: profile chi2 vs constant T (kinetics fixed):", " ".join(f"{T:.0f}:{v:.1f}" for T, v in zip(Ts, c)))
# approximate 1-sigma interval: Delta chi2 = lam_prof (noise factor of the profile) * 1 (Delta phi = 0.5 with kappa weight): report where chi2 rises by 3 (about 1 sigma at the estimated profile noise)
lo = Ts[np.where(c <= c[i] + 3.0)[0][0]]; hi = Ts[np.where(c <= c[i] + 3.0)[0][-1]]
print(f"{TAG}: best constant T = {Ts[i]:.0f} C (chi2 {c[i]:.1f}); chi2 within +3 of the minimum for T in [{lo:.0f}, {hi:.0f}] C")

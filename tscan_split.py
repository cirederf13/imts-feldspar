import json, numpy as np, os, sys
TAG = sys.argv[1]
import fwd, pgrid
fwd.init_main(); REF = {"imts": "start_joint1_final.json", "a0": "gcd_params_irra024_joint_maad_ERR_nN_vs100.json", "gok": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json", "gok2": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json"}[fwd.MODEL]
ref = pgrid.Ref(REF); x = np.array(json.load(open(f"lm_{TAG}.json"))["x"]); m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[fwd.IDX["a"]], x[fwd.IDX["b"]])
Ts=np.arange(30,61,1.0); R=np.array([np.asarray(fwd.prof_resid([T]*6)) for T in Ts])
n=R.shape[1]; k=n*10//15
for nm,sl in (("205-295",slice(0,k)),("305-345",slice(k,n)),("all",slice(0,n))):
    c=(R[:,sl]**2).sum(1); i=int(np.argmin(c)); print(f"{TAG} windows {nm}: best T {Ts[i]:.0f} C chi2 {c[i]:.1f}")

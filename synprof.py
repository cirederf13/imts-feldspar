import json, numpy as np, os, sys, pickle
M_ = os.environ["IMTS_MODEL"]; tag = {"imts": "I", "gok": "G", "a0": "A"}[M_]
import fwd, pgrid
fwd.init_main(); REF = {"imts": "start_joint1_final.json", "a0": "gcd_params_irra024_joint_maad_ERR_nN_vs100.json", "gok": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json"}[M_]
ref = pgrid.Ref(REF); HA = fwd.HA; out = {}
for Tt in (10, 30, 50):
    d = json.load(open(f"lm_syn{tag}_T{Tt}.json")); x = np.array(d["x"]); m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[fwd.IDX["a"]], x[fwd.IDX["b"]])
    T = x[fwd.IDX["T"]]; r = np.asarray(fwd.prof_resid(T)); mod = r * np.asarray(HA.NN_SD) + np.asarray(HA.NN_DATA)
    D = pickle.load(open(f"synth_T{Tt}.pkl", "rb"))
    out[Tt] = dict(T=float(T[-1]), sd=float(d["sd"][27]), model=mod, data=np.asarray(D["prof"]), psd=np.asarray(HA.NN_SD), cost=d["cost"])
pickle.dump(out, open(os.environ["OUTP"], "wb")); print("ok", M_, {k: (round(v["T"], 1), round(v["cost"], 1)) for k, v in out.items()})

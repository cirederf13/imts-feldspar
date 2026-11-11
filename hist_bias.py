"""Bias of a constant-temperature inversion of the natural profile when the true history is not constant (forward model, lmBc kinetics, noise-free).  usage: IMTS_MODEL=imts python hist_bias.py lmBc"""
import json, numpy as np, os, sys
TAG = sys.argv[1]
os.environ["IMTS_PARAMS"] = "start_joint1_final.json"
import fwd, pgrid
fwd.init_main(); ref = pgrid.Ref(os.environ["IMTS_PARAMS"]); x = np.array(json.load(open(f"lm_{TAG}.json"))["x"]); m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[fwd.IDX["a"]], x[fwd.IDX["b"]])
sat = fwd.NS["lab_sat"](); HA = fwd.HA; sd = np.asarray(HA.NN_SD, float); Ts = np.arange(10, 71, 0.5)
curves = np.array([fwd.NS["hist_model"](np.full(6, T), sat) for T in Ts]); xs = np.arange(205, 346, 10)
def fit(target, sl): c = (((curves[:, sl] - target[sl]) / sd[sl]) ** 2).sum(1); return Ts[int(np.argmin(c))]
TP = 47.0; k = 10
print("segments, oldest->youngest: 3-10 Ma | 1-3 Ma | 0.1-1 Ma | 30-100 ka | 10-30 ka | 0-10 ka     (present T = %g C)" % TP)
sc = {"constant": [TP] * 6, "warming 5 C over last 20 ka (cold 10-100 ka)": [TP] * 3 + [TP - 5] * 2 + [TP], "warming 10 C over last 20 ka (cold 10-100 ka)": [TP] * 3 + [TP - 10] * 2 + [TP],
      "warming 10 C, cold only 10-30 ka": [TP] * 4 + [TP - 10, TP], "cooling 10 C over last 20 ka": [TP] * 3 + [TP + 10] * 2 + [TP], "recent warming only 0-10 ka +5": [TP] * 5 + [TP + 5], "long-term cooler (1 Ma) by 5 C": [TP - 5] * 3 + [TP] * 3}
print(f"{'scenario':50s} low windows (205-295) | high windows (305-345) | all | mean-T over 0-100 ka")
for nm, H in sc.items():
    t = fwd.NS["hist_model"](np.array(H, float), sat); print(f"{nm:50s} {fit(t, slice(0, k)):6.1f} {fit(t, slice(k, 15)):6.1f} {fit(t, slice(0, 15)):6.1f}")

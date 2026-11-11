"""Evaluate finished LM runs on the FULL data set (all VHR rates and preheats): usage  python eval_subset.py tag[:model] ...   (model = imts | a0 | gok, default imts; the physics is chosen by env MODEL at import,
so tags of different models are evaluated by separate calls:  MODEL=a0 python eval_subset.py lmA0 )"""
import sys, os, json, numpy as np
TAGS = list(sys.argv[1:])
for v in ("TL_RATES", "TL_PH"): os.environ.pop(v, None)
import fwd, pgrid
REF = {"imts": "start_joint1_final.json", "a0": "gcd_params_irra024_joint_maad_ERR_nN_vs100.json", "gok": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json"}[fwd.MODEL]
os.environ["IMTS_PARAMS"] = REF; fwd.init_main(); ref = pgrid.Ref(REF); IDX = fwd.IDX; k = 8.617333e-5; Ts = [210, 240, 270, 300, 330, 360]; ths = [0, 10, 30, 100, 300, 1000]; wcen = [w[0] + 5 for w in fwd.ITL_WINS]
def est_E(R, iw):
    res = []
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        E = 1.5
        for _ in range(5):
            t1 = np.log(np.array(ths) + k * (T1 + 273.15) ** 2 / E); t2 = np.log(np.array(ths) + k * (T2 + 273.15) ** 2 / E); y1 = np.array([R[(T1, t)][iw] for t in ths]); y2 = np.array([R[(T2, t)][iw] for t in ths]); L1 = []; L2 = []
            for lev in np.exp(np.linspace(np.log(0.04), np.log(0.9), 25)):
                def inv(y, t):
                    yy = np.minimum.accumulate(y)
                    if yy.max() < lev or yy.min() > lev: return np.nan
                    return np.interp(np.log(lev), np.log(np.clip(yy, 1e-5, None))[::-1], t[::-1])
                L1.append(inv(y1, t1)); L2.append(inv(y2, t2))
            L1 = np.array(L1); L2 = np.array(L2); mk = ~np.isnan(L1) & ~np.isnan(L2)
            if mk.sum() < 3: E = np.nan; break
            E = np.mean(L1[mk] - L2[mk]) * k / (1 / (T1 + 273.15) - 1 / (T2 + 273.15))
        res.append(E)
    return np.nanmean(res) if np.isfinite(res).any() else np.nan
iws = [i for i, w in enumerate(wcen) if 215 <= w <= 345]; Ed = np.array([est_E(fwd.ITL_DATA, i) for i in iws]); rows = []
for tag in TAGS:
    d = json.load(open(f"lm_{tag}.json")); x = np.array(d["x"]); sd = np.array(d.get("sd", np.zeros(len(x)))); m = fwd.model_arrays(ref, x)
    R = fwd.resid_all(ref, x, nj=4); tl = (R["tl"] ** 2).mean(axis=(0, 1, 3)); mer = lambda k_: float(np.mean(R[k_] ** 2))
    jobs = [(150, 0)] + [(T, th) for T in fwd.ITL_T for th in fwd.ITL_TH]; out = fwd.Parallel(n_jobs=4)(fwd.delayed(fwd._itl_cond)(m["E"], m["s"], m["N"], m["D0"], m["M0"], T, th, m["b"]) for T, th in jobs); MOD = {j: o / out[0] for j, o in zip(jobs[1:], out[1:])}
    Em = np.array([est_E(MOD, i) for i in iws]); Tn = x[IDX["T"]]
    rows.append((tag, Tn[-1], sd[IDX["T"]][-1], x[IDX["dE"]], tl, mer("tl"), mer("itl"), mer("maad"), mer("lab"), mer("prof"), np.nanmean(Ed - Em), np.sqrt(np.nanmean((Ed - Em) ** 2))))
print("tag | present T +- sd | dE control values (eV) | FULL-VHR mean r2 per rate 0.1..5 | all | ITL r2 | MAAD | lab | prof | apparent-E gap to ITL data: mean, rms (eV)")
for r in rows: print(f"{r[0]:6s} | {r[1]:.1f} +- {r[2]:.1f} | {np.round(r[3], 3).tolist()} | {np.round(r[4], 1).tolist()} | {r[5]:.1f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} | {r[9]:.2f} | {r[10]:+.2f}, {r[11]:.2f}")

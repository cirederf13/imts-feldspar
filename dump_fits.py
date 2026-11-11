"""Dump model predictions and data for a finished LM run, for side-by-side plots.  usage: IMTS_MODEL=<model> python dump_fits.py tag Tprofile"""
import sys, json, pickle, numpy as np
TAG = sys.argv[1]; TPROF = float(sys.argv[2])
import fwd, pgrid
REF = {"imts": "start_joint1_final.json", "a0": "gcd_params_irra024_joint_maad_ERR_nN_vs100.json", "gok": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json", "gok2": "gcd_params_irrgok24_joint_maad_ERR_nN_vs100.json"}[fwd.MODEL]
import os; os.environ["IMTS_PARAMS"] = REF
fwd.init_main(); ref = pgrid.Ref(REF); x = np.array(json.load(open(f"lm_{TAG}.json"))["x"]); m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[fwd.IDX["a"]], x[fwd.IDX["b"]]); M = fwd.M; nj = 3; out = dict(tag=TAG, model=fwd.MODEL, Tprof=TPROF)
jobs = [(150, 0)] + [(T, th) for T in fwd.ITL_T for th in fwd.ITL_TH]
r = fwd.Parallel(n_jobs=nj)(fwd.delayed(fwd._itl_cond)(m["E"], m["s"], m["N"], m["D0"], m["M0"], T, th, m["b"], m["a"]) for T, th in jobs); out["itl_model"] = {j: o / r[0] for j, o in zip(jobs[1:], r[1:])}; out["itl_data"] = dict(fwd.ITL_DATA); out["itl_wins"] = [w[0] + 5 for w in fwd.ITL_WINS]
res = fwd.Parallel(n_jobs=nj)(fwd.delayed(fwd._stage)(m["E"], m["s"], m["N"], m["D0"], m["M0"], ph, M.DOSE_VHR, range(6), m["b"], m["a"]) for ph in M.PREHEATS)
norm = np.stack([[q[0] for q in rr] for rr in res]); out["tl_model"] = norm[None] * np.asarray(M.I_peak)[..., None]; out["tl_data"] = np.asarray(M.I_data); out["tl_T"] = np.asarray(M.T_surr) - 273.15; out["PH"] = list(M.PREHEATS); out["betas"] = list(M.BETAS)
Dg = np.logspace(1.3, 4.1, 14); keys = {}
for Sr in M.MAAD_SERIES:
    for dd in list(np.asarray(Sr["D"]) + m["Dnat"][Sr["wi"]]) + [M.D_REF] + list(Dg + m["Dnat"][Sr["wi"]]): keys[(Sr["PH"], round(float(dd), 6))] = None
ks = list(keys); rr = fwd.Parallel(n_jobs=nj)(fwd.delayed(fwd._stage)(m["E"], m["s"], m["N"], m["D0"], m["M0"], ph, dd, [3], m["b"], m["a"]) for ph, dd in ks); sig = {k_: q[0] for k_, q in zip(ks, rr) if q is not None}; ser = []
for Sr in M.MAAD_SERIES:
    wi = Sr["wi"]; Dn = m["Dnat"][wi]; w = np.asarray(Sr["w"]); win = lambda kk: 10.0 ** sig[kk][1] * np.sum(sig[kk][0] * w) * M.DT_SURR
    f = np.array([win((Sr["PH"], round(float(q), 6))) for q in np.asarray(Sr["D"]) + Dn]) / win((Sr["PH"], round(float(M.D_REF), 6))); fg = np.array([win((Sr["PH"], round(float(q + Dn), 6))) for q in Dg]) / win((Sr["PH"], round(float(M.D_REF), 6)))
    y = np.asarray(Sr["R"]); keep = np.asarray(Sr["keep"]) > 0; sg = M.MAAD_SIG * y + 0.03; ww = keep / sg ** 2; amp = np.sum(ww * f * y) / np.sum(ww * f * f)
    ser.append(dict(name=Sr["name"], wi=wi, D=np.asarray(Sr["D"]), y=y, keep=keep, Dg=Dg, curve=amp * fg))
out["maad"] = ser; P = {pos: fwd.NS["sim_IMTS"]("B", ph) for pos, ph in fwd.NS["DISCS"].items()}
out["lab"] = dict(P=P, DATA={k_: v for k_, v in fwd.NS["DATA"].items()}, WIN=list(fwd.NS["WIN"]), DELAYS=list(fwd.NS["DELAYS"]))
sat = fwd.NS["lab_sat"](); out["prof_model"] = fwd.NS["hist_model"](np.full(6, TPROF), sat); out["prof_data"] = fwd.HA.NN_DATA; out["prof_sd"] = fwd.HA.NN_SD
pickle.dump(out, open(f"fits_{TAG}.pkl", "wb")); print("dumped", TAG)

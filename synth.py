"""Synthetic-data benchmark for the inversion.  gen(): in an IMTS process, simulate every data set from a 'true' parameter vector (+ noise at the fit's own noise model) -> pickle.
apply(): in any fitting process (any model), overwrite the measured data in the fwd/chi2 modules with the synthetic ones (data selections/masks are kept).
usage: IMTS_MODEL=imts python synth.py gen truth.json T_true seed noise_scale out.pkl      (noise_scale 0 gives noise-free data: self-consistency test)"""
import sys, os, json, pickle, numpy as np
LAM = dict(tl=11.6, maad=1.0, lab=3.5, itl=3.5, prof=1.0)       # common noise factors of the model comparison
def gen(truth, T_true, seed, scale, out, ref_json):
    os.environ["IMTS_PARAMS"] = ref_json; import fwd, pgrid, jax.numpy as jnp
    from joblib import Parallel, delayed
    M = fwd.M; rng = np.random.default_rng(seed); ref = pgrid.Ref(ref_json); IDX = fwd.IDX; fwd.init_main()
    x = np.array(json.load(open(truth))["x"]); x[IDX["T"]] = T_true; m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[IDX["a"]], x[IDX["b"]]); nz = lambda sd, k: scale * np.sqrt(LAM[k]) * sd * rng.standard_normal(np.shape(sd)); D = {}
    # TL
    rs = list(range(6)); ps = list(range(len(M.PREHEATS)))
    res = Parallel(n_jobs=8)(delayed(fwd._stage)(m["E"], m["s"], m["N"], m["D0"], m["M0"], M.PREHEATS[ip], M.DOSE_VHR, rs, m["b"], m["a"]) for ip in ps)
    full = np.tile(res[0][0][0][None, None, :], (len(ps), 6, 1))
    for ip, rr in zip(ps, res):
        for ir, r in zip(rs, rr): full[ip, ir] = r[0]
    mod = full[None] * np.asarray(M.I_peak)[..., None]; v = np.asarray(M.var_of(jnp.array(mod))); sd = np.sqrt(v / np.asarray(M.RATE_W, float)[None, None, :, None])
    D["tl"] = mod + nz(sd, "tl")
    # MAAD
    keys = {}
    for S in M.MAAD_SERIES:
        for d in list(np.asarray(S["D"]) + m["Dnat"][S["wi"]]) + [M.D_REF]: keys[(S["PH"], round(float(d), 6))] = None
    ks = list(keys); rr = Parallel(n_jobs=8)(delayed(fwd._stage)(m["E"], m["s"], m["N"], m["D0"], m["M0"], ph, d, [3], m["b"], m["a"]) for ph, d in ks); sig = {k: r[0] for k, r in zip(ks, rr) if r is not None}; D["maad"] = []
    for S in M.MAAD_SERIES:
        Dn = m["Dnat"][S["wi"]]; w = np.asarray(S["w"]); win = lambda k: 10.0 ** sig[k][1] * np.sum(sig[k][0] * w) * M.DT_SURR
        f = np.array([win((S["PH"], round(float(d), 6))) for d in np.asarray(S["D"]) + Dn]) / max(win((S["PH"], round(float(M.D_REF), 6))), 1e-300)
        y = np.asarray(S["R"]); keep = np.asarray(S["keep"]) > 0; sg = M.MAAD_SIG * y + 0.03; ww = keep / sg ** 2; a = np.sum(ww * f * y) / max(np.sum(ww * f * f), 1e-300); D["maad"].append(a * f + nz(sg, "maad"))
    # ITL
    jobs = [(150, 0)] + [(T, th) for T in fwd.ITL_T for th in fwd.ITL_TH]; o = Parallel(n_jobs=8)(delayed(fwd._itl_cond)(m["E"], m["s"], m["N"], m["D0"], m["M0"], T, th, m["b"], m["a"]) for T, th in jobs); D["itl"] = {}
    for (T, th), mm in zip(jobs[1:], o[1:]):
        y = fwd.ITL_DATA[(T, th)]; sg = np.sqrt((0.05 * y) ** 2 + 0.01 ** 2); D["itl"][(T, th)] = mm / o[0] + nz(sg, "itl")
    # laboratory fading
    D["lab"] = {}
    for pos, ph in fwd.NS["DISCS"].items():
        P = fwd.NS["sim_IMTS"]("B", ph)
        for k, (lo, hi) in enumerate(fwd.NS["WIN"]):
            arr = np.array(fwd.NS["DATA"][(pos, lo, hi)], float); arr[1:] = P[1:, k] + nz(np.full(len(arr) - 1, 0.01), "lab"); D["lab"][(pos, lo, hi)] = arr
    # natural profile
    HA = fwd.HA; sat = fwd.NS["lab_sat"](); D["prof"] = fwd.NS["hist_model"](np.full(6, float(T_true)), sat) + nz(np.asarray(HA.NN_SD, float), "prof"); D["T_true"] = T_true; D["x_true"] = x
    pickle.dump(D, open(out, "wb")); print("synthetic data written", out, "T_true", T_true, "seed", seed, "scale", scale)
def apply(pkl):
    import fwd, jax.numpy as jnp
    D = pickle.load(open(pkl, "rb")); M = fwd.M
    M.I_data = jnp.array(D["tl"]) if not isinstance(M.I_data, np.ndarray) else D["tl"]
    for S, y in zip(M.MAAD_SERIES, D["maad"]): S["R"] = y
    for k, v in D["itl"].items(): fwd.ITL_DATA[k] = v
    for k, v in D["lab"].items(): fwd.NS["DATA"][k] = v
    fwd.HA.NN_DATA = D["prof"]; print("synthetic data applied from", pkl, "T_true", D["T_true"], flush=True)
if __name__ == "__main__" and sys.argv[1] == "gen":
    gen(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], os.environ["IMTS_PARAMS"])

"""Illustration: temperature-history inference with a scaled climate proxy (NGRIP d18O).  T(t) = T0 - dT * c(t), c = 0 at present, 1 at the LGM.
Synthetic truth: ANA-05 IMTS kinetics (fixed), T0 = 42.5 C, dT = 10 C.  Data: natural n/N profile (15 windows) + noise with 3 effective independent points.
Objective: same profile misfit as lm_inv (residual / sd * sqrt(3/15)) + weak Gaussian priors on (T0, dT); posterior on a grid."""
import sys, os, json, time, numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "start_joint1_final.json"); os.environ["IMTS_PARAMS"] = IM
NG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy", "d18o50.txt")
# ---- proxy -> coldness c(t) on a 100-yr grid, 123 ka -> 0 ; before that one coarse stage block held at the record mean ----
def load_proxy():
    rows = []
    for ln in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proxy", "edc3.txt")):
        p = ln.split()
        if len(p) == 5:
            try: rows.append((float(p[2]), float(p[3])))
            except ValueError: pass
    a = np.array(rows); a = a[(a[:, 0] >= 0)]; a = a[np.argsort(a[:, 0])]; age, d = a[:, 0], a[:, 1]
    d0 = d[age <= 1000].mean(); dlgm = d[(age >= 20000) & (age <= 25000)].mean()
    return age, (d0 - d) / (d0 - dlgm), d0, dlgm
def history_grid():
    age, c, d0, dl = load_proxy(); amax = 123000.0
    fine = np.arange(amax, 0.0, -100.0); mid = fine - 50.0; cf = np.interp(mid, age, c)
    coarse = np.arange(1_000_000.0, amax, -10_000.0); cc = np.full(coarse.size, c[age <= amax].mean())
    ages = np.r_[coarse, fine]; dts = np.r_[np.full(coarse.size, 10_000.0)[: coarse.size], np.full(fine.size, 100.0)]
    ages = np.r_[coarse, fine]; edges = np.concatenate([[0.0], np.cumsum(dts)]); ctot = np.r_[cc, cf]; return edges, ctot, ages - dts / 2.0, d0, dl
EDGES, CST, TMID, D0N, DLGM = history_grid()
def T_of(T0, dT): return T0 - dT * CST
ENG = None
def init_worker():
    global ENG
    import warnings; warnings.filterwarnings("ignore"); os.environ["JAX_PLATFORMS"] = "cpu"
    import fwd, pgrid; from history_jax import Engine
    fwd.init_main(); ref = pgrid.Ref(IM); IDX = fwd.IDX; x = np.array(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lm_lmBs.json")))["x"])
    m = fwd.model_arrays(ref, x); fwd.apply_SI(m, x[IDX["a"]], x[IDX["b"]]); sat = fwd.NS["lab_sat"](); ENG = Engine(fwd.SI, fwd.GI, fwd.HA, sat); ENG.sd = np.asarray(fwd.HA.NN_SD)
def model(args):
    T0, dT = args; return np.asarray(ENG.profile(EDGES, T_of(T0, dT)))
if __name__ == "__main__":
    out = sys.argv[1]; TT = float(os.environ.get("TRUE_T0", 42.5)); T0s = np.arange(TT - 16.5, TT + 17.6, 1.25); dTs = np.arange(-10.0, 30.1, 2.5)
    pts = [(a, b) for a in T0s for b in dTs] + [(TT, d) for d in (0.0, 3.0, 5.0, 10.0, 15.0)]
    t = time.time()
    with Pool(int(os.environ.get("NPROC", 8)), initializer=init_worker) as P: res = P.map(model, pts, chunksize=4)
    print("grid of %d histories: %.0fs" % (len(pts), time.time() - t), flush=True)
    init_worker(); sd = ENG.sd
    np.savez(out, T0s=T0s, dTs=dTs, prof=np.array(res[:-5]).reshape(T0s.size, dTs.size, -1), truths=np.array(res[-5:]), sd=sd, trueT0=TT, priorT0=float(os.environ.get('PRIOR_T0', 45.0)), edges=EDGES, c=CST, tmid=TMID, d0=D0N, dlgm=DLGM)

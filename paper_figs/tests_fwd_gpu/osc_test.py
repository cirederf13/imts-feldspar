import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import example_ngrip as E; E.init_worker(); from osc_jax import OscEngine
O = OscEngine(E.ENG)
def hist_edges():
    old = np.arange(1_000_000.0, 123000.0, -1000.0); fine = np.arange(123000.0, 0.0, -100.0); ages = np.r_[old, fine]; dts = np.r_[np.full(old.size, 1000.0), np.full(fine.size, 100.0)]
    return np.concatenate([[0.0], np.cumsum(dts)])
ED = hist_edges(); print("stages", ED.size - 1)
t = time.time(); r0 = O.lam_rows(ED, lambda a: np.full_like(a, 5.0)); p_osc = np.asarray(O.profile_lam(ED, r0)); print("const 5C via lam rows: %.1fs" % (time.time() - t)); print(np.round(p_osc[::2], 4))
# compare with the existing engine on the same stages (steps mode)
t = time.time(); p_ref = np.asarray(E.ENG.profile(ED, np.full(ED.size - 1, 5.0))); print("reference engine: %.1fs" % (time.time() - t)); print(np.round(p_ref[::2], 4)); print("max abs diff", np.abs(p_osc - p_ref).max())
for P, A in ((1.0, 10.0), (100.0, 10.0), (1000.0, 10.0), (41000.0, 10.0)):
    t = time.time(); rows = O.lam_rows(ED, lambda a: 5.0 + A * np.sin(2 * np.pi * a / P)); p = np.asarray(O.profile_lam(ED, rows)); print("P %g A %g: %.1fs  profile[::3] %s  dmax %.3f" % (P, A, time.time() - t, np.round(p[::3], 3), np.abs(p - p_ref).max()))

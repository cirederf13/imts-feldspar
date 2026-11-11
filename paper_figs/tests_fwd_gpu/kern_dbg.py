import sys, os, numpy as np
OUT=None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import osc_suite as S
S.init(); O = S.O; ED = S.ED
def prof(f): return np.asarray(O.profile_lam(ED, O.lam_rows(ED, f)))
p0 = prof(lambda a: np.full_like(a, 5.0)); print("base", np.round(p0[::3], 4))
for lo, hi in ((20000, 30000), (30000, 40000), (10000, 15000), (123000, 200000)):
    for d in (1.0, 5.0, 10.0):
        p = prof(lambda a: 5.0 + d * ((a >= lo) & (a < hi))); print("block %g-%g d=%g: dP*1e3 %s" % (lo, hi, d, np.round((p - p0)[::3] * 1e3, 2)))
p = prof(lambda a: np.full_like(a, 6.0)); print("const +1: dP*1e3", np.round((p - p0)[::3] * 1e3, 2))

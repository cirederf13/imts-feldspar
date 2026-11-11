import sys, os, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import example_ngrip as E
E.init_worker(); out = {}
for d in (0.0, 3.0, 5.0, 15.0):
    t = time.time(); out["d%g" % d] = E.model((5.0, d)); print("dT", d, "%.0fs" % (time.time() - t), flush=True)
np.savez(sys.argv[1], **out)

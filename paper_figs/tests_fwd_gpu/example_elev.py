"""Elevation transect: 6 samples, present T = Tb - 5 C/km * z (z = 0..2 km).  Grid of histories on (T0, dT), T0 integer steps, for later joint posterior."""
import sys, os, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import example_ngrip as E
if __name__ == "__main__":
    out = sys.argv[1]; T0s = np.arange(-6.0, 19.1, 1.0); dTs = np.arange(-6.0, 26.1, 2.0)
    tr = [(t, d) for t in (0., 2., 4., 6., 8., 10.) for d in (3.0, 10.0)]
    pts = [(a, b) for a in T0s for b in dTs] + tr; t = time.time()
    with Pool(int(os.environ.get("NPROC", 8)), initializer=E.init_worker) as P: res = P.map(E.model, pts, chunksize=4)
    print("grid of %d histories: %.0fs" % (len(pts), time.time() - t), flush=True)
    E.init_worker(); n = T0s.size * dTs.size
    np.savez(out, T0s=T0s, dTs=dTs, prof=np.array(res[:n]).reshape(T0s.size, dTs.size, -1), tr_pts=np.array(tr), tr_prof=np.array(res[n:]), sd=E.ENG.sd, edges=E.EDGES, c=E.CST, tmid=E.TMID)

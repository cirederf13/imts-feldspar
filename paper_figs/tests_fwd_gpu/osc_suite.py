import sys, os, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else "osc_suite.npz"
import example_ngrip as E
TM = 5.0
def hist_edges():
    old = np.arange(1_000_000.0, 123000.0, -1000.0); fine = np.arange(123000.0, 0.0, -100.0); dts = np.r_[np.full(old.size, 1000.0), np.full(fine.size, 100.0)]; return np.concatenate([[0.0], np.cumsum(dts)])
ED = hist_edges(); BL = np.array([0, 300, 1000, 2000, 4000, 7000, 10000, 15000, 20000, 30000, 40000, 60000, 80000, 123000, 200000, 400000, 1000000.0])
PER = np.logspace(np.log10(200.0), np.log10(1e5), 40)
def rand_comp(seed):
    r = np.random.default_rng(seed); a = PER ** 0.5; ph = r.uniform(0, 2 * np.pi, PER.size); a = a * 5.0 * np.sqrt(2.0) / np.sqrt((a ** 2).sum()); return a, ph        # rms 5 C
O = None
def init():
    global O
    E.init_worker(); from osc_jax import OscEngine; O = OscEngine(E.ENG)
def run(task):
    kind = task[0]
    if kind == "const": f = lambda a: np.full_like(a, task[1])
    elif kind == "harm": P, A = task[1], task[2]; f = lambda a: TM + A * np.sin(2 * np.pi * a / P)
    elif kind == "block": lo, hi, d = task[1], task[2], task[3]; f = lambda a: TM + d * ((a >= lo) & (a < hi))
    elif kind == "rand": amp, ph = rand_comp(task[1]); f = lambda a: TM + (amp[None, :] * np.sin(2 * np.pi * a[:, None] / PER[None, :] + ph[None, :])).sum(1)
    return np.asarray(O.profile_lam(ED, O.lam_rows(ED, f)))
if __name__ == "__main__":
    Ts = np.arange(-6.0, 30.1, 0.5); Ps = [1.0, 100.0, 1000.0, 5000.0, 10000.0, 41000.0, 100000.0]; As = [2.0, 5.0, 10.0]
    tasks = [("const", T) for T in Ts] + [("harm", P, A) for P in Ps for A in As] + [("block", BL[k], BL[k + 1], d) for k in range(BL.size - 1) for d in (1.0, -1.0)] + [("rand", s) for s in (1, 2, 3)]
    t = time.time()
    with Pool(8, initializer=init) as Pl: res = Pl.map(run, tasks, chunksize=2)
    print("%d histories %.0fs" % (len(tasks), time.time() - t), flush=True)
    E.init_worker(); SD = np.asarray(E.ENG.sd); np.save(OUT + ".partial.npy", np.array(res, dtype=object), allow_pickle=True)
    nT, nH = Ts.size, len(Ps) * len(As); nB = 2 * (BL.size - 1)
    np.savez(OUT, Ts=Ts, Ps=Ps, As=As, const=np.array(res[:nT]), harm=np.array(res[nT:nT + nH]).reshape(len(Ps), len(As), -1), blocks=np.array(res[nT + nH:nT + nH + nB]).reshape(BL.size - 1, 2, -1), rand=np.array(res[nT + nH + nB:]), BL=BL, sd=SD)

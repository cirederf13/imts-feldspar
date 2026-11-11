import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = np.load("elev_grid.npz"); T0s, dTs, P, sd, c, tmid = (Z[k] for k in ("T0s", "dTs", "prof", "sd", "c", "tmid"))
trp = {(float(a), float(b)): p for (a, b), p in zip(Z["tr_pts"], Z["tr_prof"])}
LAPSE, DZ = 5.0, 0.4; zs = np.arange(6) * DZ; offs = np.round(LAPSE * zs, 6)                      # 0, 2, ..., 10 C colder than the bottom sample
Tbs = np.arange(4.0, 18.1, 1.0); TB_TRUE = 10.0
def idx0(T): return int(round(T - T0s[0]))
def noise(r): return np.interp(np.arange(15), [0, 7, 14], r.normal(size=3))
def logpost(datas, use, kap, sdTb, Tb_c=10.0):
    """datas: list of 6 data vectors; use: sample indices.  Returns log posterior on (Tb, dT)."""
    lp = np.zeros((Tbs.size, dTs.size))
    for i in use:
        for a, Tb in enumerate(Tbs):
            lp[a] += -0.5 * kap * (((P[idx0(Tb - offs[i])] - datas[i]) / sd) ** 2).sum(-1)
    return lp - 0.5 * ((Tbs[:, None] - Tb_c) / sdTb) ** 2 - 0.5 * (dTs[None, :] / 15.0) ** 2
def stats(lp):
    w = np.exp(lp - lp.max()); w /= w.sum(); pt, pd = w.sum(1), w.sum(0); m0 = (pt * Tbs).sum(); m1 = np.interp(0.5, np.cumsum(pd), dTs); s0 = np.sqrt((pt * (Tbs - m0) ** 2).sum()); s1 = 0.5 * np.subtract(*np.interp([0.84, 0.16], np.cumsum(pd), dTs))
    lo, hi = np.interp([0.025, 0.975], np.cumsum(pd), dTs); return m0, s0, m1, s1, lo, hi, w
N = 300; res = {}
cfg = {"1 sample (bottom)": ([0], 2.0), "6 samples": (list(range(6)), 2.0)}
for dT in (3.0, 10.0):
    truth = [trp[(TB_TRUE - offs[i], dT)] for i in range(6)]
    for kap, kn in ((1.0, "15 windows"),):
        for name, (use, sdTb) in cfg.items():
            r = np.random.default_rng(7 + int(dT)); S = []
            for _ in range(N):
                datas = [np.clip(truth[i] + noise(r) * sd, 0, 1) for i in range(6)]; S.append(stats(logpost(datas, use, kap, sdTb))[:6])
            S = np.array(S); res[(dT, kn, name)] = dict(m1=S[:, 2].mean(), s1=S[:, 3].mean(), sc=S[:, 2].std(), ex=np.mean(S[:, 4] > 0), cov=np.mean((S[:, 4] < dT) & (S[:, 5] > dT)), m0=S[:, 0].mean(), s0=S[:, 1].mean())
            q = res[(dT, kn, name)]; print("dT %4.1f | %-14s | %-32s | median %5.1f ± %4.1f (half-width of 68%%) (scatter %.1f) excl0 %3.0f%% cov %3.0f%% | Tb %5.1f ± %.1f" % (dT, kn, name, q["m1"], q["s1"], q["sc"], 100 * q["ex"], 100 * q["cov"], q["m0"], q["s0"]))


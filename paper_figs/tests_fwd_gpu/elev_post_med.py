import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import style
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; style.apply()
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
            q = res[(dT, kn, name)]; print("dT %4.1f | %-14s | %-32s | dT %5.1f ± %4.1f (scatter %.1f) excl0 %3.0f%% cov %3.0f%% | Tb %5.1f ± %.1f" % (dT, kn, name, q["m1"], q["s1"], q["sc"], 100 * q["ex"], 100 * q["cov"], q["m0"], q["s0"]))

# ---- figure 1
fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.4), sharex=True, sharey=True)
for r_, dT in enumerate((3.0, 10.0)):
    truth = [trp[(TB_TRUE - offs[i], dT)] for i in range(6)]; rng = np.random.default_rng(5); datas = [np.clip(truth[i] + noise(rng) * sd, 0, 1) for i in range(6)]
    for k, name in enumerate(cfg):
        use, sdTb = cfg[name]; st = stats(logpost(datas, use, 1.0, sdTb)); w = st[6]; a = ax[r_, k]; srt = np.sort(w.ravel())[::-1]; cs = np.cumsum(srt); lev = [srt[np.searchsorted(cs, q)] for q in (0.95, 0.68)]
        Tg, Dg = np.meshgrid(Tbs, dTs, indexing="ij"); a.contourf(Tg, Dg, w, levels=[lev[0], lev[1], w.max() * 1.01], colors=["#fdd0a2", "#d95f02"], alpha=.85); a.plot(TB_TRUE, dT, "k*", ms=12); a.plot(st[0], st[2], "wo", mec="k"); a.axhline(0, color="0.5", lw=.6)
        a.set_title("%s\nΔT median %.1f ± %.1f, Tb %.1f ± %.1f" % (name, st[2], st[3], st[0], st[1]), fontsize=10); a.set_xlim(4, 18); a.set_ylim(-6, 26)
        if k == 0: a.set_ylabel("true cooling %g °C\n\nLGM cooling (°C)" % dT)
        if r_ == 1: a.set_xlabel("temperature of the lowest sample today (°C)")
plt.tight_layout(); plt.savefig("../out/t2_transect_posteriors.png", dpi=150)
# ---- figure 2
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2)); names = list(cfg); cols = ["#6baed6", "#a50f15"]; kn = "15 windows"
for dT, x0 in ((3.0, 0), (10.0, 3)):
    for j, nm in enumerate(names):
        q = res[(dT, kn, nm)]; ax[0].errorbar(x0 + j, q["m1"], yerr=q["s1"], fmt="o", color=cols[j], capsize=3, label=nm if dT == 3.0 else None)
    ax[0].hlines(dT, x0 - 0.4, x0 + 1.4, color="k", ls="--", lw=.8)
ax[0].set_xticks([0.5, 3.5]); ax[0].set_xticklabels(["true 3 °C", "true 10 °C"]); ax[0].set_ylabel("recovered cooling (°C)"); ax[0].axhline(0, color="0.7", lw=.6); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a) median ± 68% half-width")
for j, nm in enumerate(names): ax[1].bar(np.arange(2) + 0.2 * (2 * j - 1) + 0.0, [100 * res[(d, kn, nm)]["ex"] for d in (3.0, 10.0)], width=0.38, color=cols[j])
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["3 °C", "10 °C"]); ax[1].set_ylim(0, 100); ax[1].set_ylabel("% of noise draws"); ax[1].set_title("(b) 95% interval excludes zero")
for j, nm in enumerate(names): ax[2].bar(np.arange(2) + 0.2 * (2 * j - 1), [res[(d, kn, nm)]["s1"] for d in (3.0, 10.0)], width=0.38, color=cols[j])
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["3 °C", "10 °C"]); ax[2].set_ylabel("68% half-width of cooling (°C)"); ax[2].set_title("(c) uncertainty (68% half-width)")
plt.tight_layout(); plt.savefig("../out/t2_transect_summary.png", dpi=150)

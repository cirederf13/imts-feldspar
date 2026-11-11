import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import style
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; style.apply()
Z = np.load("ngrip_grid_T5.npz"); TR = np.load("truths_T5.npz")
T0s, dTs, P, sd, c, tmid = (Z[k] for k in ("T0s", "dTs", "prof", "sd", "c", "tmid")); import os; KAP = float(os.environ.get('KAP', 1.0)); SDT0 = float(os.environ.get('SDT0', 2.0)); TAG = os.environ.get('TAG', 'c'); T0T = 5.0; PRI = float(Z['priorT0']) if SDT0 > 5 else 5.0; age = tmid / 1e3; sel = age <= 123
truths = {0.0: TR["d0"], 3.0: TR["d3"], 5.0: TR["d5"], 10.0: Z["truth"], 15.0: TR["d15"]}
def noise(rng): return np.interp(np.arange(15), [0, 7, 14], rng.normal(size=3))
def post(data):
    chi = (((P - data) / sd) ** 2).sum(-1); lp = -0.5 * KAP * chi - 0.5 * ((T0s[:, None] - PRI) / SDT0) ** 2 - 0.5 * (dTs[None, :] / 15.0) ** 2
    w = np.exp(lp - lp.max()); return w / w.sum()
def stats(w):
    pt, pd = w.sum(1), w.sum(0); m0 = (pt * T0s).sum(); m1 = np.interp(0.5, np.cumsum(pd), dTs); s0 = np.sqrt((pt * (T0s - m0) ** 2).sum()); s1 = 0.5 * np.subtract(*np.interp([0.84, 0.16], np.cumsum(pd), dTs))
    cd = np.cumsum(pd); lo, hi = np.interp([0.025, 0.975], cd, dTs); return m0, s0, m1, s1, lo, hi
N = 500; summ = {}
for d, tr in truths.items():
    r = np.random.default_rng(100 + int(d)); S = np.array([stats(post(np.clip(tr + noise(r) * sd, 0, 1))) for _ in range(N)])
    summ[d] = dict(m1=S[:, 2], s1=S[:, 3], m0=S[:, 0], excl0=np.mean(S[:, 4] > 0), cov=np.mean((S[:, 4] < d) & (S[:, 5] > d)), t0err=np.mean(S[:, 0] - T0T), t0sd=np.mean(S[:, 1]))
    print("dT true %4.1f: recovered dT mean %5.1f (draw scatter %.1f, posterior sd %.1f); 95%% interval excludes 0 in %3.0f%% of draws; 95%% interval covers truth %3.0f%%; T0 bias %+.1f, T0 sd %.1f" % (d, S[:, 2].mean(), S[:, 2].std(), S[:, 3].mean(), 100 * summ[d]["excl0"], 100 * summ[d]["cov"], summ[d]["t0err"], summ[d]["t0sd"]))
# ---- figure 1: history density + posterior for one noise draw per case
cases = [3.0, 5.0, 10.0, 15.0]; fig, ax = plt.subplots(2, 4, figsize=(17, 7.2))
for j, d in enumerate(cases):
    rng = np.random.default_rng(3); data = np.clip(truths[d] + noise(rng) * sd, 0, 1); w = post(data); st = stats(w); pw = w.ravel()
    dr = np.random.default_rng(2).choice(w.size, 2500, p=pw); jT, jD = np.unravel_index(dr, w.shape); cur = np.array([T0s[a] - dTs[b] * c for a, b in zip(jT, jD)])
    ag = age[sel]; Tb = np.linspace(-45, 40, 170); ab = np.linspace(0, 123, 247); H = np.zeros((Tb.size - 1, ab.size - 1))
    for cv in cur: H += np.histogram2d(ag, cv[sel], bins=[ab, Tb])[0].T
    H /= H.max(); a0 = ax[0, j]; a0.pcolormesh(ab, Tb, np.ma.masked_less(H, 0.01), cmap="YlOrBr", shading="auto", vmin=0, vmax=1)
    a0.plot(ag, (T0T - d * c)[sel], "k", lw=1.0, label="true"); a0.plot(ag, np.median(cur, 0)[sel], color="#a50f15", ls="--", lw=1.1, label="posterior median"); a0.set_xlim(123, 0); a0.set_ylim(-30, 25)
    a0.set_title("LGM cooling %g °C" % d); a0.set_xlabel("age (ka)")
    if j == 0: a0.set_ylabel("temperature (°C)"); a0.legend(frameon=False, fontsize=8, loc="lower left")
    a2 = ax[1, j]; Tg, Dg = np.meshgrid(T0s, dTs, indexing="ij"); srt = np.sort(w.ravel())[::-1]; cs = np.cumsum(srt); lev = [srt[np.searchsorted(cs, q)] for q in (0.95, 0.68)]
    a2.contourf(Tg, Dg, w, levels=[lev[0], lev[1], w.max() * 1.01], colors=["#fdd0a2", "#d95f02"], alpha=.85); a2.plot(T0T, d, "k*", ms=12); a2.plot(st[0], st[2], "wo", mec="k", ms=6); a2.axhline(0, color="0.5", lw=0.6)
    a2.set_xlim(-12, 22); a2.set_ylim(-10, 30); a2.set_xlabel("present temperature (°C)"); a2.set_title("T0 %.1f ± %.1f, ΔT median %.1f ± %.1f °C" % (st[0], st[1], st[2], st[3]), fontsize=10)
    if j == 0: a2.set_ylabel("LGM cooling (°C)")
style.panel_labels(ax.ravel())
plt.tight_layout(); plt.savefig("../out/t1_single_histories.png", dpi=150)
# ---- figure 2: summary
ds = sorted(summ); fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
for d in ds: ax[0].plot([d] * 60, summ[d]["m1"][:60], ".", color="#d95f02", alpha=.25, ms=4)
ax[0].errorbar(ds, [summ[d]["m1"].mean() for d in ds], yerr=[summ[d]["s1"].mean() for d in ds], fmt="o", color="#a50f15", capsize=3, label="median ± 68% half-width"); ax[0].plot([-1, 16], [-1, 16], "k--", lw=0.8, label="1:1"); ax[0].set_xlabel("true LGM cooling (°C)"); ax[0].set_ylabel("recovered cooling (°C)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("(a) recovered vs true (dots: noise draws)")
ax[1].bar([str(int(d)) for d in ds], [100 * summ[d]["excl0"] for d in ds], color="#d95f02"); ax[1].set_ylim(0, 100); ax[1].set_xlabel("true LGM cooling (°C)"); ax[1].set_ylabel("% of noise draws"); ax[1].set_title("(b) 95% interval excludes zero cooling")
ax[2].plot(ds, [summ[d]["s1"].mean() for d in ds], "o-", color="#a50f15", label="cooling (68% half-width)"); ax[2].plot(ds, [summ[d]["t0sd"] for d in ds], "s-", color="#08519c", label="present temperature"); ax[2].set_xlabel("true LGM cooling (°C)"); ax[2].set_ylabel("68% half-width (°C)"); ax[2].set_ylim(0, 12); ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("(c) uncertainty")
plt.tight_layout(); plt.savefig("../out/t1_single_summary.png", dpi=150)

np.savez("sens_%sm.npz" % TAG, **{("m1_%g" % d): summ[d]["m1"] for d in summ}, **{("s1_%g" % d): summ[d]["s1"] for d in summ}, **{("ex_%g" % d): summ[d]["excl0"] for d in summ})

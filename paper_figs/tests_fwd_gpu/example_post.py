import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Z = np.load(sys.argv[1]); out = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 3
T0s, dTs, P, truth, sd, edges, c, tmid = (Z[k] for k in ("T0s", "dTs", "prof", "truth", "sd", "edges", "c", "tmid")); KAP = 3.0 / 15.0; TRUE = (float(Z["trueT0"]) if "trueT0" in Z.files else 42.5, 10.0)
def noise(rng):
    v = rng.normal(size=3); return np.interp(np.arange(15), [0, 7, 14], v)                        # 3 effective independent points over the 15 windows
def post(data):
    chi = (((P - data) / sd) ** 2).sum(-1); lp = -0.5 * KAP * chi - 0.5 * ((T0s[:, None] - (float(Z["priorT0"]) if "priorT0" in Z.files else 45.0)) / 15.0) ** 2 - 0.5 * (dTs[None, :] / 15.0) ** 2
    w = np.exp(lp - lp.max()); return w / w.sum(), chi
def summ(w):
    m0 = (w.sum(1) * T0s).sum(); s0 = np.sqrt((w.sum(1) * (T0s - m0) ** 2).sum()); m1 = (w.sum(0) * dTs).sum(); s1 = np.sqrt((w.sum(0) * (dTs - m1) ** 2).sum()); return m0, s0, m1, s1
rng = np.random.default_rng(seed); eps = noise(rng); data = np.clip(truth + eps * sd, 0.0, 1.0); w, chi = post(data); m0, s0, m1, s1 = summ(w)
print("seed %d: T0 = %.1f +- %.1f (true %.1f), dT = %.1f +- %.1f (true 10)" % (seed, m0, s0, TRUE[0], m1, s1))
cov0 = cov1 = 0; N = 500; r2 = np.random.default_rng(0)
for _ in range(N):
    ww = post(np.clip(truth + noise(r2) * sd, 0.0, 1.0))[0]; a, b, c_, d = summ(ww); cov0 += abs(a - TRUE[0]) < b; cov1 += abs(c_ - TRUE[1]) < d if False else abs(c_ - TRUE[1]) < d
print("coverage of the 1-sigma interval over %d noise draws: T0 %.0f%%, dT %.0f%%" % (N, 100 * cov0 / N, 100 * cov1 / N))
# ------------------------------------------------------------------ figure
age_ka = tmid / 1e3; sel = age_ka <= 123; idx = np.flatnonzero(w.ravel() > 0); pw = w.ravel() / w.sum()
draw = np.random.default_rng(1).choice(w.size, 300, p=pw); iT, iD = np.unravel_index(draw, w.shape)
curves = np.array([T0s[a] - dTs[b] * c for a, b in zip(iT, iD)]); lo, med, hi = np.percentile(curves, [2.5, 50, 97.5], axis=0)
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
a0 = ax[0]; draw2 = np.random.default_rng(2).choice(w.size, 3000, p=pw); jT, jD = np.unravel_index(draw2, w.shape); cur = np.array([T0s[a_] - dTs[b_] * c for a_, b_ in zip(jT, jD)])
ag = age_ka[sel]; Tb = np.linspace(cur[:, sel].min() - 1, cur[:, sel].max() + 1, 160); ab = np.linspace(0, 123, 247)
H = np.zeros((Tb.size - 1, ab.size - 1))
for cv in cur: H += np.histogram2d(cv[sel], ag, bins=[Tb, ab])[0] if False else np.histogram2d(ag, cv[sel], bins=[ab, Tb])[0].T
H = H / H.max(); im = a0.pcolormesh(ab, Tb, np.ma.masked_less(H, 0.01), cmap="YlOrBr", shading="auto", vmin=0, vmax=1)
a0.plot(ag, (TRUE[0] - TRUE[1] * c)[sel], color="k", lw=1.0, label="true history"); a0.plot(ag, np.median(cur, 0)[sel], color="#a50f15", lw=1.2, ls="--", label="posterior median")
a0.set_xlim(123, 0); a0.set_xlabel("age (ka)"); a0.set_ylabel("temperature (°C)"); a0.legend(frameon=False, fontsize=8, loc="lower left"); a0.set_title("(a) recovered history (posterior density)"); plt.colorbar(im, ax=a0, pad=0.01, label="relative posterior density")
a1 = ax[1]; x = np.arange(15) + 1; a1.axhline(0, color="0.6", lw=0.6); a1.errorbar(x, data, yerr=sd, fmt="o", color="k", ms=3.5, capsize=2, label="synthetic data"); a1.plot(x, truth, color="k", lw=0.8, ls="--", label="noise-free truth")
for a, b in zip(iT[:40], iD[:40]): a1.plot(x, P[a, b], color="#d95f02", alpha=.15, lw=1)
a1.plot([], [], color="#d95f02", label="posterior draws"); a1.set_xlabel("TL window (10 °C steps)"); a1.set_ylabel("n/N"); a1.legend(frameon=False, fontsize=8); a1.set_title("(b) natural profile")
a2 = ax[2]; Tg, Dg = np.meshgrid(T0s, dTs, indexing="ij"); srt = np.sort(w.ravel())[::-1]; cs = np.cumsum(srt); lev = [srt[np.searchsorted(cs, q)] for q in (0.95, 0.68)]
a2.contourf(Tg, Dg, w, levels=[lev[0], lev[1], w.max() * 1.01], colors=["#fdd0a2", "#d95f02"], alpha=.8); a2.plot(*TRUE, "k*", ms=12, label="truth"); a2.plot(m0, m1, "wo", mec="k", ms=6, label="posterior mean")
a2.set_xlabel("present temperature (°C)"); a2.set_ylabel("LGM cooling (°C)"); a2.legend(frameon=False, fontsize=8, loc="upper left"); a2.set_title("(c) posterior, 68%% / 95%%\nT0 %.1f ± %.1f, ΔT %.1f ± %.1f °C" % (m0, s0, m1, s1))
plt.tight_layout(); plt.savefig(out, dpi=150); print("saved", out)

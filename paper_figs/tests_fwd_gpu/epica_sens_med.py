import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import style
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; style.apply()
ZE = np.load("epica_grid_T5.npz"); ZN = np.load("ngrip_grid_T5.npz"); TR = np.load("truths_T5.npz"); KAP, SDT0, T0T = 1.0, 2.0, 5.0; DS = [0.0, 3.0, 5.0, 10.0, 15.0]
tN = {0.: TR["d0"], 3.: TR["d3"], 5.: TR["d5"], 10.: ZN["truth"], 15.: TR["d15"]}; tE = dict(zip(DS, ZE["truths"]))
def noise(r): return np.interp(np.arange(15), [0, 7, 14], r.normal(size=3))
def post(Z, data):
    T0s, dTs, P, sd = Z["T0s"], Z["dTs"], Z["prof"], Z["sd"]; chi = (((P - data) / sd) ** 2).sum(-1)
    lp = -0.5 * KAP * chi - 0.5 * ((T0s[:, None] - T0T) / SDT0) ** 2 - 0.5 * (dTs[None, :] / 15.0) ** 2; w = np.exp(lp - lp.max()); return w / w.sum()
def stats(Z, w):
    T0s, dTs = Z["T0s"], Z["dTs"]; pt, pd = w.sum(1), w.sum(0); m0 = (pt * T0s).sum(); m1 = np.interp(0.5, np.cumsum(pd), dTs); s0 = np.sqrt((pt * (T0s - m0) ** 2).sum()); s1 = 0.5 * np.subtract(*np.interp([0.84, 0.16], np.cumsum(pd), dTs))
    lo, hi = np.interp([0.025, 0.975], np.cumsum(pd), dTs); return m0, s0, m1, s1, lo, hi
def run(Z, truths, N=500):
    out = {}
    for d in DS:
        r = np.random.default_rng(100 + int(d)); sd = Z["sd"]; S = np.array([stats(Z, post(Z, np.clip(truths[d] + noise(r) * sd, 0, 1))) for _ in range(N)])
        out[d] = dict(m1=S[:, 2].mean(), s1=S[:, 3].mean(), sc=S[:, 2].std(), ex=np.mean(S[:, 4] > 0), cov=np.mean((S[:, 4] < d) & (S[:, 5] > d)), t0=S[:, 0].mean())
    return out
cases = {"NGRIP truth, NGRIP shape": run(ZN, tN), "EPICA truth, EPICA shape": run(ZE, tE), "NGRIP truth, EPICA shape (wrong shape)": run(ZE, tN)}
for k, o in cases.items():
    print(k)
    for d in DS: q = o[d]; print("  true %4.1f: recovered %5.1f ± %4.1f (draw scatter %.1f)  excl0 %3.0f%%  cov95 %3.0f%%  T0 %.1f" % (d, q["m1"], q["s1"], q["sc"], 100 * q["ex"], 100 * q["cov"], q["t0"]))
# profile response (distance from dT=0 truth)
for nm, t in (("NGRIP", tN), ("EPICA", tE)): print(nm, "response sqrt(KAP chi2) vs 0:", {d: round(float(np.sqrt(KAP * (((t[d] - t[0.]) / ZE["sd"]) ** 2).sum())), 2) for d in DS[1:]})
# figure: histories posterior (EPICA/EPICA) + proxies
Z = ZE; T0s, dTs, c, tmid = Z["T0s"], Z["dTs"], Z["c"], Z["tmid"]; age = tmid / 1e3; sel = age <= 123; cN = ZN["c"]
fig, ax = plt.subplots(2, 4, figsize=(17, 7.2))
for j, d in enumerate([3.0, 5.0, 10.0, 15.0]):
    rng = np.random.default_rng(3); data = np.clip(tE[d] + noise(rng) * Z["sd"], 0, 1); w = post(Z, data); st = stats(Z, w)
    dr = np.random.default_rng(2).choice(w.size, 2500, p=w.ravel()); jT, jD = np.unravel_index(dr, w.shape); cur = np.array([T0s[a] - dTs[b] * c for a, b in zip(jT, jD)])
    ag = age[sel]; Tb = np.linspace(-45, 40, 170); ab = np.linspace(0, 123, 247); H = np.zeros((Tb.size - 1, ab.size - 1))
    for cv in cur: H += np.histogram2d(ag, cv[sel], bins=[ab, Tb])[0].T
    H /= H.max(); a0 = ax[0, j]; a0.pcolormesh(ab, Tb, np.ma.masked_less(H, 0.01), cmap="YlOrBr", shading="auto", vmin=0, vmax=1)
    a0.plot(ag, (T0T - d * c)[sel], "k", lw=1.0, label="true"); a0.plot(ag, np.median(cur, 0)[sel], color="#a50f15", ls="--", lw=1.1, label="posterior median"); a0.set_xlim(123, 0); a0.set_ylim(-30, 25); a0.set_title("EPICA shape, LGM cooling %g °C" % d); a0.set_xlabel("age (ka)")
    if j == 0: a0.set_ylabel("temperature (°C)"); a0.legend(frameon=False, fontsize=8, loc="lower left")
    a2 = ax[1, j]; Tg, Dg = np.meshgrid(T0s, dTs, indexing="ij"); srt = np.sort(w.ravel())[::-1]; cs = np.cumsum(srt); lev = [srt[np.searchsorted(cs, q)] for q in (0.95, 0.68)]
    a2.contourf(Tg, Dg, w, levels=[lev[0], lev[1], w.max() * 1.01], colors=["#fdd0a2", "#d95f02"], alpha=.85); a2.plot(T0T, d, "k*", ms=12); a2.plot(st[0], st[2], "wo", mec="k", ms=6); a2.axhline(0, color="0.5", lw=.6)
    a2.set_xlim(-12, 22); a2.set_ylim(-10, 30); a2.set_xlabel("present temperature (°C)"); a2.set_title("T0 %.1f ± %.1f, ΔT median %.1f ± %.1f °C" % (st[0], st[1], st[2], st[3]), fontsize=10)
    if j == 0: a2.set_ylabel("LGM cooling (°C)")
plt.tight_layout(); plt.savefig("../out/t1_epica_histories.png", dpi=150)
# summary comparing the three cases + proxy shapes
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2)); cols = ["#6baed6", "#d95f02", "#756bb1"]
ax[0].plot(ZN["tmid"] / 1e3, ZN["c"], color=cols[0], lw=.8, label="NGRIP"); ax[0].plot(ZE["tmid"] / 1e3, ZE["c"], color=cols[1], lw=.8, label="EPICA"); ax[0].set_xlim(123, 0); ax[0].set_xlabel("age (ka)"); ax[0].set_ylabel("scaled coldness c(t)"); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title("proxy shapes")
for j, (k, o) in enumerate(cases.items()):
    ax[1].errorbar(np.array(DS) + 0.5 * (j - 1), [o[d]["m1"] for d in DS], yerr=[o[d]["s1"] for d in DS], fmt="o", color=cols[j], capsize=3, label=k)
    ax[2].plot(DS, [100 * o[d]["ex"] for d in DS], "o-", color=cols[j]); ax[3].plot(DS, [o[d]["s1"] for d in DS], "o-", color=cols[j])
ax[1].plot([0, 15], [0, 15], "k--", lw=.8); ax[1].set_xlabel("true cooling (°C)"); ax[1].set_ylabel("recovered (°C)"); ax[1].legend(frameon=False, fontsize=7, loc="upper left"); ax[1].set_title("median ± 68% half-width")
ax[2].set_xlabel("true cooling (°C)"); ax[2].set_ylabel("% of noise draws"); ax[2].set_ylim(0, 100); ax[2].set_title("95% interval excludes zero"); ax[3].set_xlabel("true cooling (°C)"); ax[3].set_ylabel("68% half-width (°C)"); ax[3].set_ylim(0, 10); ax[3].set_title("uncertainty")
style.panel_labels(np.ravel(ax)); plt.tight_layout(); plt.savefig("../out/t1_epica_summary.png", dpi=150)

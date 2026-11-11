import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickle, string, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import style; style.apply(size=20, tick=18, legend=15)
import os; HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
A = pickle.load(open(ROOT + "/fits_lmBs.pkl", "rb")); B = pickle.load(open(ROOT + "/fits_lmGOKs.pkl", "rb")); C3 = pickle.load(open(ROOT + "/fits_lmA0s.pkl", "rb")); names = ("IMTS", "GOK", "no interaction (A=0)")
Ts = [180, 210, 240, 270, 300, 330, 360]; ths = np.array([0, 10, 30, 100, 300, 1000]); cols = plt.cm.viridis(np.linspace(0, 0.9, len(Ts)))
fig, ax = plt.subplots(6, 3, figsize=(22, 30))
for c, (F, nm) in enumerate(zip((A, B, C3), names)):
    ax[0, c].set_title(nm, fontsize=22)
    for r, w in enumerate((245, 325)):
        a = ax[r, c]; iw = F["itl_wins"].index(w)
        for T, col in zip(Ts, cols):
            y = [F["itl_data"][(T, t)][iw] for t in ths]; a.plot(ths + 17, y, "o", color=col, ms=4); a.plot(ths + 17, [F["itl_model"][(T, t)][iw] for t in ths], "-", color=col, lw=1.5, label=f"{T} C" if (r == 0 and c == 0) else None)
        a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(2e-3, 1.6); a.set_ylabel(f"ITL, window {w-5}-{w+5} C"); a.grid(alpha=.3, which="both")
        if r == 1: a.set_xlabel("hold time + ramp (s)")
    a = ax[2, c]; T = F["tl_T"]
    for ib, col in ((0, "C0"), (3, "C2"), (5, "C3")): a.plot(T, F["tl_data"][2, 4, ib], "k", lw=.8); a.plot(T, F["tl_model"][2, 4, ib], col, lw=1.3, label=f"{F['betas'][ib]:g} C/s" if c == 0 else None)
    a.set_xlim(70, 472); a.set_ylabel("TL signal"); a.set_xlabel("T (C)"); a.grid(alpha=.3)
    a = ax[3, c]
    for s_ in F["maad"]:
        if not s_["name"].startswith("none"): continue
        col = {0: "C0", 1: "C1", 2: "C3"}[s_["wi"]]; a.semilogx(np.maximum(s_["D"], 5)[s_["keep"]], s_["y"][s_["keep"]], "o", color=col, ms=4); a.semilogx(s_["Dg"], s_["curve"], color=col, label=f"w{s_['wi']+1}" if c == 0 else None)
    a.set_ylabel("MAAD, Lx/Tx"); a.set_xlabel("beta dose (Gy)"); a.grid(alpha=.3)
    a = ax[4, c]; L = F["lab"]; DL = np.array(L["DELAYS"]) + 920.0
    for pos, ls in ((1, "-"), (3, "--"), (5, ":")):
        for k, col in ((0, "C0"), (3, "C3")):
            lo, hi = L["WIN"][k]; a.plot(DL[1:], L["DATA"][(pos, lo, hi)][1:], "o", color=col, ms=3); a.plot(DL[1:], L["P"][pos][1:, k], ls, color=col, lw=1.2)
    a.set_xscale("log"); a.set_ylabel("lab. fading"); a.set_xlabel("storage time (s)"); a.grid(alpha=.3)
    a = ax[5, c]; xs = np.arange(205, 346, 10); a.errorbar(xs, F["prof_data"], F["prof_sd"], fmt="ko", ms=4); a.plot(xs, F["prof_model"], "C3", lw=2); chi = float(np.sum(((F["prof_model"] - F["prof_data"]) / F["prof_sd"]) ** 2))
    a.set_ylabel("natural n/N"); a.set_xlabel("TL window (C)"); a.set_title(f"constant {F['Tprof']:.1f} C, profile chi2 = {chi:.1f}", fontsize=16); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=15, ncol=2); ax[2, 0].legend(fontsize=16); ax[3, 0].legend(fontsize=16)
for r in range(6):
    lo = min(ax[r, c].get_ylim()[0] for c in range(3)); hi = max(ax[r, c].get_ylim()[1] for c in range(3))
    for c in range(3): ax[r, c].set_ylim(lo, hi)
style.panel_labels(ax.ravel(), size=24)
plt.tight_layout(); plt.savefig(HERE + "/out/compare3_slow.png", dpi=90); print("ok")

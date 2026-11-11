import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import style; import pickle, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; style.apply()
import os; HERE = os.path.dirname(os.path.abspath(__file__)); SP = HERE + "/inputs"
S = {m: pickle.load(open(f"{SP}/sp_{m}.pkl", "rb")) for m in ("imts", "gok", "a0")}
names = {"imts": "IMTS", "gok": "GOK", "a0": "$A=0$"}; col = {"imts": "C3", "gok": "C0", "a0": "C2"}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
a = ax[0]; a.plot([20, 60], [20, 60], "k--", lw=1)
for i, m in enumerate(S):
    for Tt in (30, 50): a.errorbar(Tt + (i - 1) * 1.6, S[m][Tt]["T"], S[m][Tt]["sd"], fmt="o", color=col[m], capsize=3, label=names[m] if Tt == 30 else None)
a.set_xlim(25, 55); a.set_ylim(15, 60); a.set_xticks([30, 50]); a.set_xlabel("true temperature ($^\\circ$C)"); a.set_ylabel("recovered temperature ($^\\circ$C)"); a.legend(); a.grid(alpha=.3); a.set_title("(a)", loc="left")
b = ax[1]; xs = np.arange(205, 346, 10); d = S["imts"][50]
b.errorbar(xs, d["data"], d["psd"], fmt="ko", ms=4, label="synthetic data (IMTS, 50$^\\circ$C)")
for m in S: b.plot(xs, S[m][50]["model"], color=col[m], lw=2, label=f"{names[m]}, {S[m][50]['T']:.1f}$^\\circ$C")
b.set_xlabel("TL window ($^\\circ$C)"); b.set_ylabel("natural $n/N$"); b.legend(fontsize=8); b.grid(alpha=.3); b.set_title("(b)", loc="left")
plt.tight_layout(); plt.savefig("" + HERE + "/out/synthetic_tests.png", dpi=200)

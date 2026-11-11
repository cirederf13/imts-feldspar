"""Growth of the MAAD signal with dose in the three TL windows: one saturating exponential per preheat batch and window,
Lx/Tx = a (1 - exp(-(D + Dnat)/D0)), fitted with sigma = 0.15 y + 0.03 to the points that are not detector-saturated (clipped).
Run from the repository root or from paper_figs; writes out/maad_window_growth.png and prints D0 and Dnat of every series."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import style
style.apply(size=13, tick=11.7, legend=10.4)       # fonts 30 % larger than the matplotlib default

d = json.load(open(os.path.join(HERE, "..", "maad_windows.json"))); dr = d["dose_rate_Gy_per_s"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3)); rows = []
col = {"none": "C0", "160": "C1", "220": "C3"}
for o in d["obs"]:
    D = np.array(o["t_beta_s"]) * dr; y = np.array(o["ratio"]); keep = ~np.array(o["clipped"]); sg = 0.15 * y + 0.03
    best = None
    for lD0 in np.linspace(1.5, 4.5, 13):
        def res(p): a, lD, ln = p; return ((a * (1 - np.exp(-(D + 10 ** ln) / 10 ** lD)) - y) / sg)[keep]
        r = least_squares(res, [y.max(), lD0, 1.5], bounds=([0, 1.0, 0.0], [1e3, 5.0, 4.3]))
        if best is None or r.cost < best.cost: best = r
    a, lD, ln = best.x; rows.append((o["batch"], o["window"], 10 ** lD, 10 ** ln, 2 * best.cost, int(keep.sum())))
    k = int(o["window"][1]) - 1; Dg = np.logspace(1.2, 4.2, 100)
    ax[k].semilogx(np.maximum(D, 5), y, "o", c=col[o["batch"]], label=f'preheat {o["batch"]}')
    ax[k].semilogx(Dg, a * (1 - np.exp(-(Dg + 10 ** ln) / 10 ** lD)), "-", c=col[o["batch"]], lw=1)
for a_ in ax: a_.set_xlabel("beta dose (Gy)"); a_.set_ylabel("Lx/Tx"); a_.grid(alpha=.3)
ax[0].legend(loc="upper left")
style.panel_labels(ax, size=16)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "out", "maad_window_growth.png"), dpi=110)
print("batch window   D0(Gy)   Dnat(Gy)  chi2/npts")
for r in rows: print(f"{r[0]:>5s} {r[1]}   {r[2]:8.0f}  {r[3]:8.0f}   {r[4]:.1f}/{r[5]}")

import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag = sys.argv[1]; import fwd, pgrid
fwd.init_main(); ref = pgrid.Ref("start_joint1_final.json"); d = json.load(open(f"lm_{tag}.json")); x = np.array(d["x"]); IDX = fwd.IDX
x0 = np.zeros(fwd.NX); x0[15:] = x[15:]; x0[IDX["T"]] = 45.0
def itl_model(xx):
    m = fwd.model_arrays(ref, xx); jobs = [(150, 0)] + [(T, th) for T in fwd.ITL_T for th in fwd.ITL_TH]
    out = fwd.Parallel(n_jobs=4)(fwd.delayed(fwd._itl_cond)(m["E"], m["s"], m["N"], m["D0"], m["M0"], T, th) for T, th in jobs); return {j: o / out[0] for j, o in zip(jobs[1:], out[1:])}
M1 = itl_model(x); M0 = itl_model(x0); wc = [w[0] + 5 for w in fwd.ITL_WINS]; cols = plt.cm.viridis(np.linspace(0, 0.9, len(fwd.ITL_T))); th = np.array(fwd.ITL_TH); te = th + 17.0
wins = (225, 245, 265, 285, 305, 325)
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for a, w in zip(ax.flat, wins):
    iw = wc.index(w)
    for T, c in zip(fwd.ITL_T, cols):
        y = np.array([fwd.ITL_DATA[(T, t)][iw] for t in th]); a.plot(te, y, "o", color=c, ms=4); a.plot(te, [M1[(T, t)][iw] for t in th], "-", color=c, lw=1.6, label=f"{T} C" if w == wins[0] else None); a.plot(te, [M0[(T, t)][iw] for t in th], "--", color=c, lw=1, alpha=.6)
    a.set_xscale("log"); a.set_yscale("log"); a.set_ylim(2e-3, 1.6); a.set_xlabel("hold time + ramp (s)"); a.set_ylabel("signal left in the window / unheated"); a.set_title(f"read-out window {w-5}-{w+5} C", fontsize=10); a.grid(alpha=.3, which="both")
ax[0, 0].legend(fontsize=7, ncol=2, title="hold temperature (dots: data, solid: fit with ITL, dashed: before ITL)", title_fontsize=7)
plt.tight_layout(); S = "/tmp/claude-10054/-work3-fherman-IMTS/7d1176a2-2242-4e0a-97c3-8fbdb1497bab/scratchpad/"; plt.savefig(S + f"lm_{tag}_itl_fit.png", dpi=105)
r1 = np.concatenate([(np.array([M1[k][i] for i in range(len(wc))]) - fwd.ITL_DATA[k])[fwd.ITL_SEL[k]] / np.sqrt((0.05 * fwd.ITL_DATA[k]) ** 2 + 0.01 ** 2)[fwd.ITL_SEL[k]] for k in M1]); r0 = np.concatenate([(np.array([M0[k][i] for i in range(len(wc))]) - fwd.ITL_DATA[k])[fwd.ITL_SEL[k]] / np.sqrt((0.05 * fwd.ITL_DATA[k]) ** 2 + 0.01 ** 2)[fwd.ITL_SEL[k]] for k in M0])
print("ITL mean r^2: with ITL %.2f, before ITL %.2f" % (np.mean(r1 ** 2), np.mean(r0 ** 2)))

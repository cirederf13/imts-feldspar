import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); import style
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; style.apply()
from scipy.interpolate import CubicSpline
kB = 8.617333262e-5; TM = 5.0
Z = np.load("osc_suite.npz"); Ts, Ps, As, C, H, B, R, BL, sd = (Z[k] for k in ("Ts", "Ps", "As", "const", "harm", "blocks", "rand", "BL", "sd"))
KAP = 1.0; W = np.sqrt(KAP) / sd
cs = CubicSpline(Ts, C, axis=0)
def fit(p):
    Tg = np.linspace(Ts[0], Ts[-1], 3601); chi = (((cs(Tg) - p) * W) ** 2).sum(1); i = chi.argmin(); J = cs(Tg[i], 1); return Tg[i], chi[i], 1 / np.sqrt(((J * W) ** 2).sum())
# analytic effective temperature of a sinusoid at trap depth E (exact phase average)
def Teff(E, A, Tm=TM):
    ph = np.linspace(0, 2 * np.pi, 4001)[:-1]; T = Tm + 273.15 + A * np.sin(ph); m = np.exp(-E / (kB * T)).mean(); return -E / (kB * np.log(m)) - 273.15 - Tm
print("analytic T_eff - Tm (C) at Tm = 5 C"); print("E(eV)  " + "  ".join("A=%g" % a for a in (2, 5, 10)))
for E in (1.0, 1.2, 1.5, 1.8, 2.0): print("%.1f   " % E + "  ".join("%5.2f" % Teff(E, a) for a in (2, 5, 10)))
print("\nbest constant temperature of the oscillating history (noise-free):  T_fit - Tm, chi (sigma, 15 windows), sd(T)")
Eeq = np.linspace(0.9, 2.2, 261)
for i, P in enumerate(Ps):
    row = []
    for j, A in enumerate(As):
        Tf, chi, s = fit(H[i, j]); ee = Eeq[np.argmin([abs(Teff(e, A) - (Tf - TM)) for e in Eeq])]; row.append("A=%g: dT %+5.2f chi %4.2f (E_eq %.2f)" % (A, Tf - TM, np.sqrt(chi), ee))
    print("P=%-7g " % P + " | ".join(row))
print("posterior sd of constant T (15 windows, flat prior): %.2f C" % fit(C[np.argmin(abs(Ts - 5))])[2])
print("\nrandom 1/f histories (rms 5 C):", [("dT %+.2f chi %.2f" % (fit(r)[0] - TM, np.sqrt(fit(r)[1]))) for r in R])
# kernel
K = (B[:, 0] - B[:, 1]) / 2.0                     # (nblocks, nwin) response to +1 C in the block
G = (K * W).T                                      # (nwin, nblocks) whitened
print("\nkernel: sum over blocks of K vs d profile / d T0 at 5 C: ", np.round(K.sum(0)[::3], 4), np.round(cs(TM, 1)[::3], 4))
sdb = 5.0; Cp = np.linalg.inv(G.T @ G + np.eye(G.shape[1]) / sdb ** 2); Rm = Cp @ (G.T @ G)
w = np.diff(BL)
lab = ["%g-%g ka" % (BL[k] / 1e3, BL[k + 1] / 1e3) for k in range(len(w))]
print("block      |K|/sd (sum over windows, per C)   post.sd (prior 5 C)   resolution diag")
for k in range(len(w)): print("%-14s %7.3f   %6.2f   %5.2f" % (lab[k], np.sqrt((G[:, k] ** 2).sum()), np.sqrt(Cp[k, k]), Rm[k, k]))
gt = G.sum(1); frac = np.array([gt @ G[:, k] for k in range(G.shape[1])]) / (gt @ gt)
print("fraction of the sensitivity to a uniform temperature change, by block:", {lab[k]: round(float(frac[k]), 3) for k in range(len(w))}); print("cumulative last 20 ka %.2f, last 123 ka %.2f" % (frac[:8].sum(), frac[:14].sum()))
np.savez("osc_post.npz", K=K, G=G, Cp=Cp, Rm=Rm)
# figure
fig, ax = plt.subplots(2, 2, figsize=(12.5, 9)); a = np.linspace(0, 12, 49)
for E, c in ((1.2, "#9ecae1"), (1.5, "#3182bd"), (1.8, "#08519c")): ax[0, 0].plot(a, [Teff(E, x) for x in a], color=c, label="E = %.1f eV" % E)
ax[0, 0].set_xlabel("amplitude A of the sinusoid (°C)"); ax[0, 0].set_ylabel("T_eff − Tm (°C)"); ax[0, 0].legend(frameon=False); ax[0, 0].set_title("Arrhenius-weighted effective temperature, Tm = 5 °C")
cols = plt.cm.viridis(np.linspace(0, 0.9, len(Ps)))
for i, P in enumerate(Ps): ax[0, 1].plot(As, [fit(H[i, j])[0] - TM for j in range(len(As))], "o-", color=cols[i], label="P = %g yr" % P)
ax[0, 1].set_xlabel("amplitude A (°C)"); ax[0, 1].set_ylabel("best constant T − Tm (°C)"); ax[0, 1].legend(frameon=False, fontsize=8); ax[0, 1].set_title("temperature recovered by the constant-T fit")
for i, P in enumerate(Ps): ax[1, 0].plot(As, [np.sqrt(fit(H[i, j])[1]) for j in range(len(As))], "o-", color=cols[i])
ax[1, 0].set_xlabel("amplitude A (°C)"); ax[1, 0].set_ylabel("misfit of the best constant-T fit (σ, 15 windows)"); ax[1, 0].set_title("can the profile tell the oscillation from a constant T?")
x = np.arange(len(w)); ax[1, 1].bar(x - 0.2, frac, 0.4, color="#d95f02", label="share of the sensitivity to a uniform ΔT"); ax[1, 1].bar(x + 0.2, [Rm[k, k] for k in range(len(w))], 0.4, color="#08519c", label="resolution (diagonal)")
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(lab, rotation=70, fontsize=7); ax[1, 1].legend(frameon=False, fontsize=8); ax[1, 1].set_title("which ages the profile is sensitive to")
style.panel_labels(np.ravel(ax)); plt.tight_layout(); plt.savefig("../out/t3_oscillations.png", dpi=150)

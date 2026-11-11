"""For a given IMTS parameter set (env IMTS_PARAMS): (i) where the deep-window TL signal comes from, (ii) lab fading of coupling B with a 2-parameter rho' law fitted to the lab fading only -> misfit and ladder chi2 at 47 C, (iii) joint fit of 4 rho' nodes to lab fading + ladder."""
import os, sys, json, time, numpy as np
src = open("lab_fading_refit2.py").read(); src = src[:src.index("chi2 = lambda")]; exec(compile(src, "lf2", "exec"))
from scipy.optimize import minimize
chi2 = lambda m: float(np.sum(((m - HA.NN_DATA) / HA.NN_SD) ** 2)); RATE_ = 0.1566; T_LONG = 127000.0; kB_ = SI.kB
print("params:", os.environ.get("IMTS_PARAMS"), " D0 = %.0f Gy" % SI.D0, " logN:", np.round(np.log10(SI.N), 2)[[0, 4, 8, 12, 15]])
# (i)
tm = np.array([SI._tm1(SI.E[i], SI.s[i]) for i in range(SI.NT)]); n = GI.irradiate(SI.E, SI.s, SI.N, SI.A, 250.0, SI.D0)
w = lambda n_: np.array([(lambda I: I[(TC >= lo) & (TC < hi)].sum())(SI.readout(np.clip(n_, 0, SI.N))) for lo, hi in WIN]); full = w(n)
print("(i) 250 Gy readout, windows 150-200/200-250/250-300/300-350: signal change if DEEP traps' charge -6%:", np.round((w(np.where(tm >= 250, n * 0.94, n)) / full - 1) * 100, 2), "%;  with deep traps EMPTY the signal is", np.round(w(np.where(tm < 250, n, 0)) / full, 2), "of full")
def set_nodes(v): HA.LOGRHO_T[:] = [175.0, 225.0, 275.0, 325.0]; HA.LOGRHO_V[:] = list(v); SI.W_K, SI.LA_K = SI.rate_classes(SI.E, SI.s)
def lab_nodes(v):
    set_nodes(v); tot = 0.0
    for pos, ph in DISCS.items():
        P = sim_IMTS("B", ph)
        for k, (lo, hi) in enumerate(WIN): tot += float(np.sum(((P[1:, k] - DATA[(pos, lo, hi)][1:]) / 0.01) ** 2))
    return tot
def field47():
    W, LA = SI.W_K, SI.LA_K; nn = GI.irradiate(SI.E, SI.s, SI.N, SI.A, 19950.0, SI.D0); f = RATE_ / SI.D0; lam_ = SI.s * np.exp(-SI.E / (kB_ * 293.15))
    ch = lambda LAx: f / (f + lam_[:, None] + LAx) * (1 - np.exp(-(f + lam_[:, None] + LAx) * T_LONG)); fac = (W * ch(LA)).sum(1) / ch(np.zeros((SI.NT, 1)))[:, 0]
    sat = HA.win(SI.readout(np.clip(nn * fac, 0, SI.N))); m = SI.model_nN_fade(47, 5, 47, 5, method="B") * SI.SAT / sat; return chi2(m), m
# (ii) lab-only law fit
t0 = time.time(); grid = [(a, b) for a in (-5.8, -5.4, -5.0) for b in (-0.8, -0.3, 0.2)]; gv = [misfit(g, "B") for g in grid]; x0 = np.array(grid[int(np.argmin(gv))])
r = minimize(misfit, x0, args=("B",), method="Nelder-Mead", options=dict(xatol=0.01, fatol=0.5, maxiter=40, initial_simplex=x0 + np.array([[0, 0], [0.2, 0], [0, 0.15]])))
a, b = r.x; ns_ = [a + b * (T - 250.0) / 100.0 for T in (175.0, 225.0, 275.0, 325.0)]; set_nodes(ns_); fc, _ = field47()
print(f"(ii) coupling B, rho' law fitted to the LAB fading only: lab misfit {r.fun:.0f} (1% errors, 48 pts); ladder chi2 at 47 C = {fc:.1f}   [a={a:.2f}, b={b:.2f}]  ({time.time()-t0:.0f}s)", flush=True)
# (iii) joint
def J(v):
    v = np.asarray(v)
    if np.any(v < -7.5) or np.any(v > -4.0): return 1e30
    return 0.25 * lab_nodes(v) + field47()[0]
best = None
for s0 in (np.array([-5.34, -5.52, -5.58, -5.81]), np.array([-5.7, -5.6, -5.5, -5.4])):
    rr = minimize(J, s0, method="Nelder-Mead", options=dict(xatol=0.01, fatol=0.05, maxiter=120, initial_simplex=s0 + np.vstack([np.zeros(4), 0.3 * np.eye(4)])))
    if best is None or rr.fun < best.fun: best = rr
set_nodes(best.x); print(f"(iii) joint (lab/4 + ladder): nodes {np.round(best.x, 2)}; lab misfit {lab_nodes(best.x):.0f}; ladder chi2 at 47 C = {field47()[0]:.1f}", flush=True)

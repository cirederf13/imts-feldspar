"""Within-coupling calibration of the tunnelling parameter on the LAB fading experiment (3 discs: preheat none/160/220 C), then field chi2.
log10 rho'(Tm) = a + b*(Tm-250)/100 (monotonic law, 2 parameters). Experiment per cycle: beta 1597 s at 0.1566 Gy/s (fading acts during it) -> [preheat TL to 160/220 C at 1 C/s] -> storage (delay + 120 s) at 20 C -> TL read-out.
Couplings for IMTS: A = tunnelling loss from every trap/r' class; B = thermal IMTS state x decoupled Kars factor; C = fast-tunnelling classes excluded. Reference: A=0 (independent traps).
usage: python lab_fading_refit2.py  (writes lab_fading_refit2.json, log)"""
import os, sys, json, time, numpy as np
os.environ.update(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", ERR="1", VHR_SCALE="100", K=os.environ.get("NTRAP", "16"))
import a0_history as HA, imts_step as SI, gok_history as GK, step_history as SA, generate_training_data_irr_cpu as GI
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
YR = SI.YR; kB = SI.kB; WIN = [(150, 200), (200, 250), (250, 300), (300, 350)]; TC = HA.TC; RATE = 0.1566; T_IRR = 1597.0; HANDLE = 120.0
cycles = json.load(open("fading_cycles.json")); DELAYS = [0.0, 1000.0, 10000.0, 100000.0, 300000.0]
DISCS = {1: 0, 3: 160, 5: 220}
DATA = {}
for pos in DISCS:
    cs = [c for c in cycles if c["pos"] == pos]
    for lo, hi in WIN:
        k = f"{lo}-{hi}"; y = {d: np.mean([c["lx"][k] / c["tx"][k] for c in cs if c["delay"] == d]) for d in DELAYS}; DATA[(pos, lo, hi)] = np.array([y[d] / y[0.0] for d in DELAYS])
def setlaw(a, b):
    HA.LOGRHO_T[:] = [0.0, 200.0, 400.0, 600.0]; HA.LOGRHO_V[:] = [a + b * (T - 250.0) / 100.0 for T in (0.0, 200.0, 400.0, 600.0)]
def wins_I(I): return np.array([I[(TC >= lo) & (TC < hi)].sum() for lo, hi in WIN])
def ramp_state(n_i, T_ph):
    """IMTS TL ramp (1 C/s) from T0 to T_ph: occupancies at the end (conduction-band state dropped)"""
    sol = solve_ivp(SI.RHS_RO, [0, T_ph + 273.15 - GI.T0], np.concatenate([n_i, [0.0]]), method="LSODA", max_step=1.0, rtol=1e-6, atol=1e-8)
    return np.clip(sol.y[:SI.NT, -1], 0, SI.N)
def ddot_irr(): return RATE * 1000.0 * YR                                  # Gy/ka equivalent of 0.1566 Gy/s
def sim_IMTS(meth, ph):
    W, LA = SI.W_K, SI.LA_K; out = []; t_prev = 0.0
    if meth == "B":
        n0 = GI.irradiate(SI.E, SI.s, SI.N, SI.A, RATE * T_IRR, SI.D0)
        if ph: n0 = ramp_state(n0, ph)
        f = RATE / SI.D0; lam = SI.s * np.exp(-SI.E / (kB * 293.15))
        def chain(LAx, t_st):
            r = f + lam[:, None] + LAx; x = f / r * (1 - np.exp(-r * T_IRR)); return x * np.exp(-(lam[:, None] + LAx) * t_st)
        for d in DELAYS:
            t_st = d + HANDLE; n = SI.hold(n0, 293.15, t_st / (1e6 * YR), 0.0)
            xk = (W * chain(LA, t_st)).sum(1); xt = chain(np.zeros((SI.NT, 1)), t_st)[:, 0]; out.append(wins_I(SI.readout(np.clip(n * xk / np.maximum(xt, 1e-300), 0, SI.N))))
    else:
        n = SI.hold_kars(np.zeros((SI.NT, LA.shape[1])), 293.15, T_IRR / (1e6 * YR), ddot_irr(), W, LA, meth)
        if ph:
            ni = n.sum(1); nn = ramp_state(ni, ph); n = n * (nn / np.maximum(ni, 1e-300))[:, None]
        for d in DELAYS:
            t_st = d + HANDLE; n = SI.hold_kars(n, 293.15, (t_st - t_prev) / (1e6 * YR), 0.0, W, LA, meth); t_prev = t_st      # storage is autonomous: continue from the previous delay
            out.append(wins_I(SI.readout(np.clip(n.sum(1), 0, SI.N))))
    S = np.array(out); return S / S[0]
def sim_A0(ph):
    E, s, N, D0 = HA.E, HA.s, HA.N, HA.D0; LA = HA.LAM_BIN; W = HA.WBIN; f = RATE / D0; lam = s * np.exp(-E / (kB * 293.15)); out = []
    for d in DELAYS:
        r = f + lam[:, None] + LA; x = f / r * (1 - np.exp(-r * T_IRR))
        if ph:
            Tg = np.linspace(293.15, ph + 273.15, 400); ex = np.array([np.trapezoid(s[i] * np.exp(-E[i] / (kB * Tg)), Tg) for i in range(SI.NT)]); x = x * np.exp(-ex)[:, None]
        r2 = lam[:, None] + LA; x = x * np.exp(-r2 * (d + HANDLE)); xi = (W[None, :] * x).sum(1)
        out.append(wins_I(HA.light(xi)))
    S = np.array(out); return S / S[0]
def misfit(ab, model):
    a, b = ab
    if not (-7.0 < a < -4.0 and -2.0 < b < 1.0): return 1e30
    setlaw(a, b)
    if model == "A0": HA.LAM_BIN = SI.lam_bins(HA.E, HA.s)
    else: SI.W_K, SI.LA_K = SI.rate_classes(SI.E, SI.s)
    tot = 0.0
    for pos, ph in DISCS.items():
        P = sim_A0(ph) if model == "A0" else sim_IMTS(model, ph)
        for k, (lo, hi) in enumerate(WIN): tot += float(np.sum(((P[1:, k] - DATA[(pos, lo, hi)][1:]) / 0.01) ** 2))
    return tot
chi2 = lambda m: float(np.sum(((m - HA.NN_DATA) / HA.NN_SD) ** 2))
TG = [30, 35, 40, 45, 47.5, 50, 55, 60]; res = {}
for model in ("B", "C", "A0", "A"):
    t0 = time.time(); grid = [(a, b) for a in (-5.8, -5.4, -5.0) for b in (-0.8, -0.3, 0.2)]; gv = [misfit(g, model) for g in grid]; x0 = np.array(grid[int(np.argmin(gv))])
    r = minimize(misfit, x0, args=(model,), method="Nelder-Mead", options=dict(xatol=0.01, fatol=0.5, maxiter=40, initial_simplex=x0 + np.array([[0, 0], [0.2, 0], [0, 0.15]])))
    a, b = r.x; setlaw(a, b); print(f"{model}: lab misfit (1% errors; 3 discs x 4 windows x 4 delays = 48 points) best {r.fun:.0f}  a = {a:.2f}, b = {b:.2f}  [log rho' at Tm 175/325 C = {a+b*(175-250)/100:.2f} / {a+b*(325-250)/100:.2f}]  ({time.time()-t0:.0f}s)", flush=True)
    if model == "A0":
        HA.LAM_BIN = SI.lam_bins(HA.E, HA.s); GK.LAM_BIN = SI.lam_bins(GK.E, GK.s)
        import importlib; src = open("step_history.py").read().replace('importlib.import_module(sys.argv[1] if __name__ == "__main__" else "a0_history")', 'importlib.import_module("gok_history")'); SG = type(sys)("gokstep"); exec(compile(src, "gokstep", "exec"), SG.__dict__)
        c = [chi2(SA.model_nN(Tq, 5, Tq, 5, kars=True)) for Tq in TG]; cg = [chi2(SG.model_nN(Tq, 5, Tq, 5, kars=True)) for Tq in TG]
        print("   field A=0 :", " ".join(f"{q}:{v:.1f}" for q, v in zip(TG, c)), f"  best {min(c):.1f} at {TG[int(np.argmin(c))]}; at 47: {chi2(SA.model_nN(47, 5, 47, 5, kars=True)):.1f}", flush=True)
        print("   field GOK (same law):", " ".join(f"{q}:{v:.1f}" for q, v in zip(TG, cg)), f"  best {min(cg):.1f} at {TG[int(np.argmin(cg))]}", flush=True)
    else:
        SI.W_K, SI.LA_K = SI.rate_classes(SI.E, SI.s); c = [chi2(SI.model_nN_fade(Tq, 5, Tq, 5, method=model)) for Tq in TG]
        print(f"   field IMTS {model}:", " ".join(f"{q}:{v:.1f}" for q, v in zip(TG, c)), f"  best {min(c):.1f} at {TG[int(np.argmin(c))]}; at 47: {chi2(SI.model_nN_fade(47, 5, 47, 5, method=model)):.1f}", flush=True)
    P1 = sim_A0(0) if model == "A0" else sim_IMTS(model, 0)
    print("   predicted I(3e5)/I(t0), disc 1, windows:", np.round(P1[-1], 3), " measured:", np.round([DATA[(1, lo, hi)][-1] for lo, hi in WIN], 3), flush=True)
    res[model] = dict(a=float(a), b=float(b), lab=float(r.fun), field=dict(zip(map(str, TG), map(float, c))))
    json.dump(res, open("lab_fading_refit2.json", "w"))
open("LABREFIT2_DONE.marker", "w").write("done")

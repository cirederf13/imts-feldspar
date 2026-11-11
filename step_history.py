"""Two-stage (step) thermal history, no fading: traps empty -> T_warm for t_warm -> T_final for t_final (monotonic cooling: T_warm >= T_final).
Model n/N per 10 C TL window (as in a0_history / gok_history), compared to the natural ladder. Scan over T_final: for each T_final fit (T_warm, t_warm, t_final).
usage: python step_history.py a0_history|gok_history   (dose rate fixed 1.455 Gy/ka)"""
import sys, os, json, time, numpy as np, importlib
H = importlib.import_module(sys.argv[1] if __name__ == "__main__" else "a0_history")
YR = 3.15576e7; kB = 8.617333262e-5; NT = H.NT; DOT = 1.455
IS_GOK = hasattr(H, "B")
KARS = os.environ.get("KARS", "0") == "1"
def stage(x, T, dt_myr, ddot, nsub=200, kars=None):
    kars = KARS if kars is None else kars
    f = ddot / (1000 * YR) / H.D0; lam = (H.s * np.exp(-H.E / (kB * T)))[:, None]; dt_s = dt_myr * 1e6 * YR
    LA = H.LAM_BIN if kars else np.zeros((NT, 1))                      # Kars tunnelling loss per r' bin (independent first-order loss, traps x bins)
    if x.shape[1] != LA.shape[1]: x = np.zeros(LA.shape)
    if not IS_GOK:
        r = f + lam + LA; return f / r + (x - f / r) * np.exp(-r * dt_s)  # exact for the linear A=0 equation
    t = np.concatenate([[0.0], dt_s * np.logspace(-4, 0, nsub)]); B = H.B
    for k in range(nsub):
        dt = t[k + 1] - t[k]
        def step(le): r = f + le + LA; return f / r + (x - f / r) * np.exp(-r * dt)
        xp = step(lam * np.maximum(x, 1e-12) ** (B - 1)); x = step(lam * 0.5 * (np.maximum(x, 1e-12) ** (B - 1) + np.maximum(xp, 1e-12) ** (B - 1)))
    return x
def model_nN(Tw, tw_myr, Tf, tf_myr, ddot=DOT, kars=None):
    kars = KARS if kars is None else kars
    x = np.zeros((NT, 1)); x = stage(x, 273.15 + Tw, tw_myr, ddot, kars=kars); x = stage(x, 273.15 + Tf, tf_myr, ddot, kars=kars)
    w = H.WBIN if kars else np.ones(1)
    return H.win(H.light((w[None, :] * x).sum(1))) / H.SAT
if __name__ == "__main__":
    from scipy.optimize import minimize
    def obj(q, Tf):
        dT, lw, lf = q
        if dT < 0 or dT > 150 or lw < -2 or lw > 1.5 or lf < -2 or lf > 1.5: return 1e30
        m = model_nN(Tf + dT, 10 ** lw, Tf, 10 ** lf); return 0.5 * np.sum(((m - H.NN_DATA) / H.NN_SD) ** 2)
    rows = []; t0 = time.time()
    for Tf in sorted(set(list(range(0, 101, 5)) + [47])):
        cands = sorted(((obj([d, lw, lf], Tf), d, lw, lf) for d in (0, 20, 50, 100) for lw in (-1, 0, 1) for lf in (-1, 0, 1)))[:3]
        best = min((minimize(obj, [d, lw, lf], args=(Tf,), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-5, maxiter=500)) for _, d, lw, lf in cands), key=lambda r: r.fun)
        rows.append((Tf, best.fun, Tf + best.x[0], 10 ** best.x[1], 10 ** best.x[2]))
        print(f"T_final={Tf:3d} C  obj={best.fun:7.2f}  T_warm={Tf+best.x[0]:6.1f} C  t_warm={10**best.x[1]:8.3g} Myr  t_final={10**best.x[2]:8.3g} Myr  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(f"step_profile_{sys.argv[1].split('_')[0]}{'_fade' if KARS else ''}.json", "w"))

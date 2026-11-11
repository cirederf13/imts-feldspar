"""Independent-trap physics (A = 0: first order; GOK: general order b) with the same interface as the IMTS routines of generate_training_data_irr_cpu (G) and imts_step (SI).
Trap occupancy x_i = n_i/N_i:  irradiation dx/dt = f (1 - x) - lam x^b ;  heating / storage  dx/dt = -lam x^b (+ f (1-x) for the natural history);  light I = sum_i N_i lam_i x_i^b.  b = 1 -> exact exponentials."""
import numpy as np
from scipy.integrate import solve_ivp
kB = 8.617333262e-5; YR = 3.15576e7; STATE = dict(b=1.0, a=1.0)
def set_order(b, a=1.0): STATE["b"] = float(b); STATE["a"] = float(a)
def _lam(E, s, T): return s * np.exp(-E / (kB * T))
def make_irradiate(G):
    def irradiate(E, s, N, A, D, D0):
        b = STATE["b"]; a = STATE["a"]; f = G.DOT / D0; t = D / G.DOT; lam = _lam(E, s, G.T_IRR)
        if abs(b - 1.0) < 1e-9 and abs(a - 1.0) < 1e-9: x = f / (f + lam) * (1.0 - np.exp(-(f + lam) * t))
        else:
            r = solve_ivp(lambda tt, x: f * np.maximum(1 - x, 0) ** a - lam * np.maximum(x, 0) ** b, [0, t], np.zeros(len(E)), method="LSODA", rtol=1e-7, atol=1e-10); x = r.y[:, -1]
        return np.clip(x, 0.0, 1.0) * N
    return irradiate
def ramp(x0, E, s, beta, T_start, T_end, T_grid=None):
    """heating from T_start to T_end (K) at beta; returns x(T_end) and, if T_grid is given, the light per second on that grid (N applied by the caller)"""
    b = STATE["b"]; t_end = (T_end - T_start) / beta
    rhs = lambda t, x: -_lam(E, s, T_start + beta * t) * np.maximum(x, 0) ** b
    r = solve_ivp(rhs, [0, t_end], x0, method="LSODA", max_step=0.5 / beta, rtol=1e-6, atol=1e-9, dense_output=bool(T_grid is not None))
    if T_grid is None: return np.maximum(r.y[:, -1], 0.0), None
    X = np.maximum(r.sol((np.asarray(T_grid) - T_start) / beta), 0.0); return np.maximum(r.y[:, -1], 0.0), _lam(E[:, None], s[:, None], np.asarray(T_grid)[None, :]) * X ** b
def make_run_ramps(G):
    def run_imts_irr(E, s, N, A, n_irr, beta, T_PH):
        x0 = n_irr / N
        if T_PH > G.T0 + 1e-6: x0, _ = ramp(x0, E, s, beta, G.T0, T_PH)
        q_rem = float(np.sum(x0 * N) / np.sum(n_irr)); _, L = ramp(x0, E, s, beta, G.T0, G.T_max, G.T_out); I = (N[:, None] * L).sum(0); mx = float(np.max(I))
        if mx <= 0: return None
        return I / mx, np.log10(mx), q_rem
    return run_imts_irr
def hold_fn(SI):
    def hold(n, T, t_myr, ddot):
        b = STATE["b"]; a = STATE["a"]; N = SI.N; x0 = np.asarray(n) / N; f = ddot / (1000 * YR) / SI.D0; lam = _lam(SI.E, SI.s, T); t = t_myr * 1e6 * YR
        if abs(b - 1.0) < 1e-9 and abs(a - 1.0) < 1e-9: xe = f / (f + lam); x = xe + (x0 - xe) * np.exp(-(f + lam) * t)
        else: x = solve_ivp(lambda tt, y: f * np.maximum(1 - y, 0) ** a - lam * np.maximum(y, 0) ** b, [0, t], x0, method="Radau", rtol=1e-6, atol=1e-10).y[:, -1]
        return np.clip(x, 0.0, 1.0) * N
    return hold
def readout_fn(SI, G):
    def readout(n0):
        _, L = ramp(np.asarray(n0) / SI.N, SI.E, SI.s, 1.0, G.T0, G.T_max, G.T_out); return (SI.N[:, None] * L).sum(0)
    return readout
def ramp_state_fn(SI, G):
    def ramp_state(n_i, T_ph):
        x, _ = ramp(np.asarray(n_i) / SI.N, SI.E, SI.s, 1.0, G.T0, T_ph + 273.15); return np.clip(x * SI.N, 0, SI.N)
    return ramp_state
def itl_cond(G, E, s, N, D0, T, th, dose=300.05, windows=None):
    x0 = make_irradiate(G)(E, s, N, None, dose, D0) / N
    if T > 22: x0, _ = ramp(x0, E, s, 1.0, G.T0, T + 273.15)
    if th > 0:
        b = STATE["b"]; lam = _lam(E, s, T + 273.15)
        x0 = (x0 * np.exp(-lam * th)) if abs(b - 1.0) < 1e-9 else solve_ivp(lambda tt, y: -lam * np.maximum(y, 0) ** b, [0, th], x0, method="LSODA", rtol=1e-6, atol=1e-10).y[:, -1]
    tt = np.arange(0, G.T_max - G.T0, 1.0); Tg = G.T0 + tt; _, L = ramp(x0, E, s, 1.0, G.T0, G.T_max, Tg); I = (N[:, None] * L).sum(0); Tc = Tg - 273.15
    return np.array([float(I[(Tc >= lo) & (Tc < hi)].sum()) for lo, hi in windows])
def patch_G(G): G.irradiate = make_irradiate(G); G.run_imts_irr = make_run_ramps(G)

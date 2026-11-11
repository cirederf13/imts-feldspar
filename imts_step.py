"""IMTS (interacting traps through the conduction band, equal capture coefficient, kappa=0, 16 traps) with the same two-step history as step_history.py (no fading):
traps empty -> T_warm for t_warm -> T_final for t_final; dose rate 1.455 Gy/ka (default). Model n/N per 10 C TL window = window signal of the 1 C/s read-out / saturated-state window signal."""
import os, json, time, numpy as np
import a0_history as HA                              # sets env, provides THERMO windows, data n/N, win(), TC, constants
import generate_training_data_irr_cpu as GI
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
kB = 8.617333262e-5; YR = 3.15576e7; DOT = 1.455
Ec = HA.Ec; NT = HA.NT
p = np.array(json.load(open(os.environ.get("IMTS_PARAMS", "gcd_params_irr_joint_maad_ERR_nN16_vs100.json")))["best"])
E = np.linspace(0.95, 2.05, NT); s = 10 ** (p[0] * E + p[1] + CubicSpline(Ec, p[2:7], bc_type="natural")(E)); E = E * float(os.environ.get("ESCALE", 1.0)); N = 10 ** CubicSpline(Ec, p[7:12], bc_type="natural")(E) * (16.0 / NT); D0 = 10 ** p[12]
if GI.NEUTRAL: GI.M0 = 10.0 ** float(p[13])
KAPPA = (float(p[13]) if len(p) > 13 else 0.0) if not GI.NEUTRAL else 0.0; A = GI.A_val * 10.0 ** (KAPPA * (E - 1.5)); RHS_RO = GI.make_rhs(E, s, N, A, beta=1.0)
RTOL_RO, MAXSTEP = 1e-6, 0.5
def hold(n, T, t_myr, ddot):
    lam = s * np.exp(-E / (kB * T)); G = ddot / (1000 * YR) / D0 * N.sum()
    def rhs(_, y):
        nc = (np.sum(lam * y) + G) / (np.sum(A * (N - y)) + GI.B * ((GI.M0 + np.sum(y)) if GI.NEUTRAL else GI.N_r)); return -lam * y + A * nc * (N - y)
    return np.clip(solve_ivp(rhs, [0, t_myr * 1e6 * YR], n, method="LSODA", rtol=1e-8, atol=1e-6 * N.min(), first_step=1.0).y[:, -1], 0, N)
def readout(n0):
    sol = solve_ivp(RHS_RO, [0, GI.T_max - GI.T0], np.concatenate([n0, [0.0]]), method="LSODA", dense_output=True, max_step=MAXSTEP, rtol=RTOL_RO, atol=1e-8)
    Y = sol.sol((GI.T_out - GI.T0)); return GI.B * Y[-1] * ((GI.M0 + Y[:-1].sum(0) + Y[-1]) if GI.NEUTRAL else GI.N_r)
SAT = HA.win(readout(GI.irradiate(E, s, N, A, 19950.0, D0)))
def model_nN(Tw, tw_myr, Tf, tf_myr, ddot=DOT):
    n = hold(np.zeros(NT), 273.15 + Tw, tw_myr, ddot); n = hold(n, 273.15 + Tf, tf_myr, ddot)
    return HA.win(readout(n)) / SAT
if __name__ == "__main__":  # (module-level globals are reassigned below)
    t0 = time.time(); a = model_nN(169.0, 0.078, 47.0, 3.76); print(f"one history: {time.time()-t0:.2f}s (rtol 1e-6, max_step 0.5)")
    RTOL_RO, MAXSTEP = 1e-5, 2.0
    t0 = time.time(); b = model_nN(169.0, 0.078, 47.0, 3.76); print(f"one history: {time.time()-t0:.2f}s (rtol 1e-5, max_step 2); max |d n/N| = {np.abs(a-b).max():.5f}; data sd min {HA.NN_SD.min():.3f}")
    print("n/N model:", np.round(a, 3)); print("n/N data :", np.round(HA.NN_DATA, 3))

# ---------------- fading (Kars/Huntley) coupled as in coupling 1: tunnelling is a first-order LOSS from every trap/r'-class, so the trap's contribution to the conduction-band pool shrinks ----------------
W_FULL = HA.WBIN
from scipy.optimize import brentq
def _tm1(Ei, si): return brentq(lambda T: np.log(Ei / (kB * T ** 2)) - (np.log(si) - Ei / (kB * T)), 150, 1500) - 273.15
def lam_bins(E_, s_, dlog=0.0):
    """tunnelling rate per trap and r' bin; rho' of each trap from ITS OWN glow-peak temperature (log rho' shifted by dlog for sensitivity tests)"""
    rho = 10 ** (np.interp([_tm1(E_[i], s_[i]) for i in range(len(E_))], HA.LOGRHO_T, HA.LOGRHO_V) + dlog)
    la = HA.STILDE * np.exp(-(rho[:, None] ** (-1 / 3.0)) * HA.RB_[None, :])
    return np.stack([np.array([(HA.PR_[HA.BIN_ == k] * la[i][HA.BIN_ == k]).sum() / max(HA.WBIN[k], 1e-300) for k in range(HA.KB_)]) for i in range(len(E_))])
LA_FULL_A0 = HA.LAM_BIN                                    # (previous, wrong for IMTS: rates of the A=0 traps)
LA_FULL = lam_bins(E, s)
def coarse(group=3):
    """merge the intermediate r' bins (1..35) in groups of `group` (weight-averaged log rate); bins 0 (instantly emptied) and 36 (no fading) kept as they are"""
    idx = [[0]] + [list(range(i, min(i + group, 36))) for i in range(1, 36, group)] + [[36]]
    W = np.array([W_FULL[g].sum() for g in idx])
    L = np.stack([np.array([np.exp((W_FULL[g] * np.log(np.maximum(LA_FULL[i, g], 1e-300))).sum() / W_FULL[g].sum()) if len(g) > 1 else LA_FULL[i, g[0]] for g in idx]) for i in range(NT)])
    return W, L
W_K, LA_K = coarse(3)
def hold_kars(n, T, t_myr, ddot, W=None, LA=None, method="A"):
    """method A: tunnelling = loss from every trap/r' class (coupling 1); C: classes with rate > 1e-9 1/s take no part (no capture, no loss). (B is handled in model_nN_fade)"""
    W = W_K if W is None else W; LA = LA_K if LA is None else LA; K = LA.shape[1]
    if n.shape[1] != K: n = np.zeros((NT, K))
    lam = s * np.exp(-E / (kB * T)); G = ddot / (1000 * YR) / D0 * N.sum()
    slow = (LA < 1e-9).astype(float) if method == "C" else np.ones_like(LA)
    Nik = N[:, None] * (W if W.ndim == 2 else W[None, :]) * slow; LAe = LA * slow
    def rhs(_, y):
        x = y.reshape(NT, K); nc = (np.sum(lam[:, None] * x) + G) / (A[0] * np.sum(Nik - x) + GI.B * GI.N_r)
        return (-(lam[:, None] + LAe) * x + A[0] * nc * (Nik - x)).ravel()
    sol = solve_ivp(rhs, [0, t_myr * 1e6 * YR], n.ravel(), method="Radau", rtol=1e-5, atol=1e-5 * N.min() * 1e-3, first_step=1.0)
    return np.clip(sol.y[:, -1].reshape(NT, K), 0, Nik)
def _kars_factor(Tw, tw, Tf, tf, ddot, W, LA):
    """B: decoupled multiplicative Kars factor per trap = (independent-trap occupancy with fading) / (without), same two-stage history"""
    f = ddot / (1000 * YR) / D0
    def chain(LAx):
        x = np.zeros((NT, LAx.shape[1]))
        for T, t in ((273.15 + Tw, tw), (273.15 + Tf, tf)):
            r = f + (s * np.exp(-E / (kB * T)))[:, None] + LAx; x = f / r + (x - f / r) * np.exp(-r * t * 1e6 * YR)
        return x
    xk = ((W if W.ndim == 2 else W[None, :]) * chain(LA)).sum(1); xt = chain(np.zeros((NT, 1)))[:, 0]
    return xk / np.maximum(xt, 1e-300)
def model_nN_fade(Tw, tw_myr, Tf, tf_myr, ddot=DOT, W=None, LA=None, method="A"):
    W_ = W_K if W is None else W; LA_ = LA_K if LA is None else LA
    if method == "B":
        n = hold(np.zeros(NT), 273.15 + Tw, tw_myr, ddot); n = hold(n, 273.15 + Tf, tf_myr, ddot)
        return HA.win(readout(np.clip(n * _kars_factor(Tw, tw_myr, Tf, tf_myr, ddot, W_, LA_), 0, N))) / SAT
    n = hold_kars(np.zeros((NT, 1)), 273.15 + Tw, tw_myr, ddot, W, LA, method); n = hold_kars(n, 273.15 + Tf, tf_myr, ddot, W, LA, method)
    return HA.win(readout(np.clip(n.sum(1), 0, N))) / SAT

def rate_classes(E_, s_, dlog=0.0, lo=-17.0, hi=1.0, step=0.5):
    """r' classes defined by tunnelling RATE (step dex wide, own weights per trap, geometric-mean rate): returns W (NT,K), LA (NT,K); accurate at lab AND field times"""
    rho = 10 ** (np.interp([_tm1(E_[i], s_[i]) for i in range(len(E_))], HA.LOGRHO_T, HA.LOGRHO_V) + dlog)
    la = HA.STILDE * np.exp(-(rho[:, None] ** (-1 / 3.0)) * HA.RB_[None, :]); ll = np.log10(np.maximum(la, 1e-300)); edges = np.arange(lo, hi + step / 2, step); K = len(edges) + 1
    W = np.zeros((len(E_), K)); LA = np.ones((len(E_), K))
    for i in range(len(E_)):
        c = np.digitize(ll[i], edges)
        for k in range(K):
            m = c == k; w = HA.PR_[m].sum()
            if w > 0: W[i, k] = w; LA[i, k] = 10 ** ((HA.PR_[m] * ll[i][m]).sum() / w)
            else: W[i, k] = 1e-30; LA[i, k] = 10 ** (edges[min(k, len(edges) - 1)])
    return W, LA

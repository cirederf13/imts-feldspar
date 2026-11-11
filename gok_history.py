"""GOK (independent traps, general-order retrapping x^b, b fitted) forward model with a monotonic cooling history and Kars fading.
History: traps empty at t=0 (start of an AGE-long history), T(t) = T_f + dT*exp(-t/tau) (dT>=0, monotonic cooling to the final temperature T_f), dose rate Ddot.
Model n/N per 10 C TL window = window signal of the read-out (1 C/s, no preheat) of the final occupancy / same window of the saturated state.
Reuses kinetics and fading of infer_T_kars3.py (A=0, 16 traps, rho' of each trap from its glow-peak temperature)."""
import os, json, numpy as np
NT_ = int(os.environ.get("NTRAP", 16)); os.environ["MODEL"] = "gok"; os.environ["N_TRAPS"] = str(NT_); os.environ.setdefault("ERR", "1")
import sys; sys.path.insert(0, 'retrain_6beta_irr'); sys.path.insert(0, '.')
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import light_models as LM
YR = 3.15576e7; kB = 8.617333262e-5; NT = NT_
STILDE = 3e15; LOGRHO_T = [175, 225, 275, 325]; LOGRHO_V = [-5.35, -5.52, -5.58, -5.83]
RB_ = np.linspace(0.00125, 2.5, 2000); PR_ = 3 * RB_ ** 2 * np.exp(-RB_ ** 3); PR_ = PR_ / PR_.sum()
EDGES = np.concatenate([[0.0, 0.6], np.arange(0.62, 1.3, 0.02), [1.3, 2.51]]); BIN_ = np.digitize(RB_, EDGES) - 1; KB_ = len(EDGES) - 1
WBIN = np.array([PR_[BIN_ == k].sum() for k in range(KB_)])
Ec = np.load("E_ctrl.npy"); Tsurr = np.load("surrogate_k16_spline_T_out.npy"); TC = Tsurr - 273.15
THERMO = {f"{r['lo']}-{r['hi']}": r for r in json.load(open("maad_thermometers.json")) if "nN" in r and r["lo"] >= 200}
W = {k: (v["lo"], v["hi"]) for k, v in THERMO.items()}
p = np.array(json.load(open(f"gcd_params_irrgok{NT}_joint_maad_ERR_nN_vs100.json"))["best"])
E = np.linspace(0.95, 2.05, NT); s = 10 ** (p[0] * E + p[1] + CubicSpline(Ec, p[2:7], bc_type="natural")(E)); N = 10 ** CubicSpline(Ec, p[7:12], bc_type="natural")(E); D0 = 10 ** p[12]; B = float(p[16])
Ej = jnp.array(E); sj = jnp.array(s); Tj = jnp.asarray(Tsurr)
def tm1(Ei, si): return brentq(lambda T: np.log(Ei / (kB * T ** 2)) - (np.log(si) - Ei / (kB * T)), 150, 1500) - 273.15
RHO = 10 ** np.interp([tm1(E[i], s[i]) for i in range(NT)], LOGRHO_T, LOGRHO_V)
LAM_A = STILDE * np.exp(-(RHO[:, None] ** (-1 / 3.0)) * RB_[None, :])
LAM_BIN = np.stack([np.array([(PR_[BIN_ == k] * LAM_A[i][BIN_ == k]).sum() / max(WBIN[k], 1e-300) for k in range(KB_)]) for i in range(NT)])
def light(x): return np.asarray(LM.gok_light(Ej, sj, jnp.array(N), jnp.full(NT, B), jnp.array(x), 1.0, 293.15, Tj))
def win(I): return np.array([I[(TC >= lo) & (TC < hi)].sum() for lo, hi in W.values()])
SAT = win(light(np.ones(NT)))
NN_DATA = np.array([THERMO[k]["nN"] for k in W]); NN_SD = np.array([THERMO[k]["nN_sd"] for k in W])
def occupancy(Tfun, age_myr, ddot_kagy, kars=True, nstep=400, fine=False):
    """final mean occupancy x_i (traps) after history Tfun(t_myr) [K]; exact exponential integrator per step (piecewise-constant T)."""
    A = age_myr * 1e6 * YR; t = np.concatenate([[0.0], A * np.logspace(-4, 0, nstep)]); f = ddot_kagy / (1000 * YR) / D0
    LA = (LAM_A if fine else LAM_BIN) if kars else np.zeros((NT, 1)); w = (PR_ if fine else WBIN) if kars else np.ones(1)
    x = np.zeros((NT, LA.shape[1]))
    for k in range(nstep):
        T = Tfun(0.5 * (t[k] + t[k + 1]) / (1e6 * YR)); lam = (s * np.exp(-E / (kB * T)))[:, None]; dt = t[k + 1] - t[k]
        def step(le):
            r = f + le + LA; return f / r + (x - f / r) * np.exp(-r * dt)
        le0 = lam * np.maximum(x, 1e-12) ** (B - 1.0); xp = step(le0)                       # predictor with the rate frozen at the step start
        x = step(lam * (0.5 * (np.maximum(x, 1e-12) ** (B - 1.0) + np.maximum(xp, 1e-12) ** (B - 1.0))))   # corrector (Heun-type)
    return (w[None, :] * x).sum(1)
def model_nN(Tf, dT, tau_myr, ddot, age=20.0, kars=True, **kw):
    return win(light(occupancy(lambda tm: 273.15 + Tf + dT * np.exp(-tm / tau_myr), age, ddot, kars, **kw))) / SAT
if __name__ == "__main__":
    import time
    from scipy.integrate import solve_ivp
    print("b =", B, "D0 =", D0)
    def ref(Tfun, age, ddot):                                  # reference: stiff ODE solve, same binned fading
        A = age * 1e6 * YR; f = ddot / (1000 * YR) / D0; K = KB_
        def rhs(t, y):
            x = np.clip(y.reshape(NT, K), 0, 1); lam = s * np.exp(-E / (kB * Tfun(t / (1e6 * YR))))
            return (-lam[:, None] * x ** B - LAM_BIN * x + f * (1 - x)).ravel()
        sol = solve_ivp(rhs, [0, A], np.zeros(NT * K), method="Radau", rtol=1e-6, atol=1e-9, first_step=1.0)
        return (WBIN[None, :] * np.clip(sol.y[:, -1].reshape(NT, K), 0, 1)).sum(1)
    for name, Tf_ in (("const 30 C", lambda tm: 303.15), ("const 47 C", lambda tm: 320.15), ("cool 100->47, tau 1 Myr", lambda tm: 320.15 + 53 * np.exp(-tm / 1.0))):
        t0 = time.time(); xr = ref(Tf_, 20.0, 1.455); t1 = time.time() - t0; t0 = time.time(); xm = occupancy(Tf_, 20.0, 1.455); t2 = time.time() - t0
        wr, wm = win(light(xr)) / SAT, win(light(xm)) / SAT
        print(f"{name}: max |dx| {np.abs(xr - xm).max():.4f};  window n/N max abs diff {np.abs(wr - wm).max():.4f} (data sd min {NN_SD.min():.3f})  [Radau {t1:.1f}s, fast {t2:.3f}s]")
    print("data n/N:", np.round(NN_DATA, 3))

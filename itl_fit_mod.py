"""ITL term for the inversion: window signals after ramp+hold, model (full ODE) vs data (cups averaged), importable by joblib workers."""
import sys, os, pickle, numpy as np
sys.argv = ["chi2_ode.py", "x", "10"]
import chi2_ode as C
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
G = C.G
WINS = [(lo, lo + 10) for lo in range(200, 350, 10)]; TS = [180, 210, 240, 270, 300, 330, 360]; THS = [0, 10, 30, 100, 300, 1000]
_d = pickle.load(open("itl_data.pkl", "rb"))
def _W(o, lo, hi): return (o['res'] - o['bg'])[lo:hi].sum()
REF = np.array([np.median([_W(o, lo, hi) for o in _d if o['T'] == 150 and o['th'] == 0]) for lo, hi in WINS]); _D = {}
for o in _d:
    if o['th'] == 70 or o['T'] not in TS: continue
    _D.setdefault((o['T'], o['th']), []).append(np.array([_W(o, lo, hi) for lo, hi in WINS]) / REF)
DATA = {k: np.mean(v, axis=0) for k, v in _D.items()}
def cond(p, T, th, dose=300.05):
    ES = float(os.environ.get("ESCALE", 1.0)); EG = G.E_GRID * ES; s = 10.0 ** (p[0] * G.E_GRID + p[1] + G.spline_eval(np.array(p[2:7]))); N = 10.0 ** G.spline_eval(np.array(p[7:12])) * (16.0 / G.N_FINE); A = G.A_val * np.ones_like(EG); G.M0 = 10.0 ** float(p[13])
    n0 = G.irradiate(EG, s, N, A, dose, 10.0 ** p[12]); y = np.concatenate([n0, [0.0]])
    if T > 22:
        r = solve_ivp(G.make_rhs(EG, s, N, A, beta=1.0), [0, T + 273.15 - G.T0], y, method="LSODA", max_step=0.5, rtol=1e-6, atol=1e-8); y = r.y[:, -1]
    if th > 0:
        r = solve_ivp(G.make_rhs(EG, s, N, A, T_const=T + 273.15), [0, th], y, method="LSODA", rtol=1e-6, atol=1e-8); y = r.y[:, -1]
    y = y.copy(); y[-1] = 0.0; y[:-1] = np.maximum(y[:-1], 0.0); t_max = G.T_max - G.T0
    sol = solve_ivp(G.make_rhs(EG, s, N, A, beta=1.0), [0, t_max], y, method="LSODA", dense_output=True, max_step=0.5, rtol=1e-6, atol=1e-8)
    tt = np.arange(0, t_max, 1.0); Y = sol.sol(tt); I = G.B * Y[-1] * (G.M0 + Y[:-1].sum(0) + Y[-1]); Tc = tt + G.T0 - 273.15
    return np.array([float(I[(Tc >= lo) & (Tc < hi)].sum()) for lo, hi in WINS])
def S_itl(p, nj=10):
    jobs = [(150, 0)] + [(T, th) for T in TS for th in THS]; out = Parallel(n_jobs=nj)(delayed(cond)(p, T, th) for T, th in jobs); ref = out[0]; tot = 0.0
    for (T, th), m in zip(jobs[1:], out[1:]):
        m = m / ref; y = DATA[(T, th)]; sg = np.sqrt((0.05 * y) ** 2 + 0.01 ** 2); sel = (y > 0.004) | (m > 0.004); tot += float(np.mean(((m - y) / sg)[sel] ** 2)) if sel.any() else 0.0
    return tot                      # sum over the 42 (T, hold) conditions of the mean reduced chi2 over windows

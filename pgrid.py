"""Peak-grid parametrisation of the traps (grid in the observable T_m, free functions of T_m):
   E_i = E_ref_i + dE(Tm_i);  log10 s_i = log10 s_FO(E_i, Tm_i) + eta(Tm_i);  log10 N_i = log10 N_ref_i + dN(Tm_i)
with Tm_i the first-order glow-peak temperature (1 C/s) of the reference trap i, s_FO(E, Tm) = (E/(k Tm^2)) exp(E/(k Tm))  (beta = 1 K/s).
dE, eta, dN: natural cubic splines through NC control values at fixed Tm control points.  At dE = eta = dN = 0 the reference model is reproduced exactly."""
import json, numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
kB = 8.617333262e-5; NC = 5
def tm_first_order(E, s):
    return brentq(lambda T: np.log(E / (kB * T ** 2)) - (np.log(s) - E / (kB * T)), 150.0, 1500.0)          # K
def log_s_fo(E, Tm): return np.log10(E / (kB * Tm ** 2)) + E / (kB * Tm) / np.log(10.0)
class Ref:
    def __init__(self, ref_json, NT=24):
        p = np.array(json.load(open(ref_json))["best"]); Ec = np.linspace(0.95, 2.05, NC); self.E = np.linspace(0.95, 2.05, NT)
        self.s = 10.0 ** (p[0] * self.E + p[1] + CubicSpline(Ec, p[2:7], bc_type="natural")(self.E)); self.logN = CubicSpline(Ec, p[7:12], bc_type="natural")(self.E)   # N before the 16/NT capacity scaling
        self.Tm = np.array([tm_first_order(E, s) for E, s in zip(self.E, self.s)]); self.NT = NT
        self.Tc = np.linspace(self.Tm.min(), self.Tm.max(), NC)                                              # control points in T_m (K)
        self.p_ref = p
    def build(self, dE, eta, dN):
        sp = lambda v: CubicSpline(self.Tc, np.asarray(v, float), bc_type="natural")(self.Tm)
        E = self.E + sp(dE); s = 10.0 ** (log_s_fo(E, self.Tm) + sp(eta)); logN = self.logN + sp(dN)
        return E, s, logN
if __name__ == "__main__":
    R = Ref("start_joint1_final.json"); z = np.zeros(NC); E, s, logN = R.build(z, z, z)
    print("max |E-E_ref|", np.abs(E - R.E).max(), " max |log10 s - log10 s_ref|", np.abs(np.log10(s) - np.log10(R.s)).max(), " max |logN-ref|", np.abs(logN - R.logN).max())
    print("Tm (C) of the 24 traps:", np.round(R.Tm - 273.15).astype(int).tolist()); print("control Tm (C):", np.round(R.Tc - 273.15).astype(int).tolist())
    E2, s2, _ = R.build(0.1 * np.ones(NC), z, z); print("dE=+0.1 eV: peak shift of the first-order peaks (K), should be 0:", np.abs(np.array([tm_first_order(a, b) for a, b in zip(E2, s2)]) - R.Tm).max())

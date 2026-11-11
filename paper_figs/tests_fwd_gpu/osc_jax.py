"""Oscillating / arbitrary thermal histories through STAGE-AVERAGED trap rates.  For each stage the thermal rate of every trap is averaged over the stage
(and over the oscillation phase when the period is shorter than the stage): lam_i = < s exp(-E_i/(kB T(t))) >.  This is exact in the fast-oscillation limit
because the conduction-band charge is linear in lam; Kars chain uses the same rows.  Rates are interpolated linearly between stage mid-points (no jumps)."""
import numpy as np, jax, jax.numpy as jnp, diffrax
from history_jax import Engine, kB, YR
class OscEngine(Engine):
    def __init__(self, base): self.__dict__.update(base.__dict__)
    def lam_rows(self, edges, Tfun, nq=48):
        """Tfun(age_yr_array) -> degC (age measured from present: age = t_end - t).  Returns (nst, ntrap) stage-averaged rates (1/yr)."""
        edges = np.asarray(edges, float); t_end = edges[-1]; E = np.asarray(self.E); s = np.asarray(self.s); rows = np.empty((edges.size - 1, E.size))
        u = (np.arange(nq) + 0.5) / nq
        for k in range(edges.size - 1):
            t = edges[k] + u * (edges[k + 1] - edges[k]); T = np.asarray(Tfun(t_end - t)) + 273.15; rows[k] = (s[None, :] * np.exp(-E[None, :] / (kB * T[:, None]))).mean(0) * YR
        return rows
    def fill_lam(self, edges, rows):
        edges = jnp.asarray(edges, float); t_end = float(edges[-1]); mid = 0.5 * (edges[1:] + edges[:-1]); R = jnp.asarray(rows); A, B, M0, N, G = self.A * YR, self.B * YR, self.M0, self.N, self.G
        def lam_t(t):
            i = jnp.clip(jnp.searchsorted(mid, t) - 1, 0, mid.size - 2); w = jnp.clip((t - mid[i]) / (mid[i + 1] - mid[i]), 0.0, 1.0); return R[i] * (1 - w) + R[i + 1] * w
        def rhs(t, n, _):
            lam = lam_t(t); nc = (jnp.sum(lam * n) + G) / (A * jnp.sum(N - n) + B * (M0 + jnp.sum(n))); return -lam * n + A * nc * (N - n)
        import os
        ctrl = diffrax.PIDController(rtol=self.rtol, atol=self.atol_hold, pcoeff=0.3, icoeff=0.4, dtmax=float(os.environ.get('OSC_DTMAX', 'inf')) if os.environ.get('OSC_DTMAX') else None)
        sol = diffrax.diffeqsolve(diffrax.ODETerm(rhs), diffrax.Kvaerno5(), 0.0, t_end, 1.0, jnp.zeros_like(N), stepsize_controller=ctrl, max_steps=4_000_000, saveat=diffrax.SaveAt(t1=True))
        return jnp.clip(sol.ys[-1], 0.0, N)
    def kars_lam(self, edges, rows):
        edges = jnp.asarray(edges, float); dt = edges[1:] - edges[:-1]; R = jnp.asarray(rows); f = self.f
        def chain(LAx):
            def step(x, a):
                lam, dtk = a; r = f + lam[:, None] + LAx; xe = f / r; return xe + (x - xe) * jnp.exp(-r * dtk), None
            x, _ = jax.lax.scan(step, jnp.zeros_like(LAx), (R, dt)); return x
        LA = self.LA * YR; return (self.W * chain(LA)).sum(1) / jnp.maximum(chain(jnp.zeros((self.E.size, 1)))[:, 0], 1e-300)
    def profile_lam(self, edges, rows):
        n = self.fill_lam(edges, rows); n = jnp.clip(n * self.kars_lam(edges, rows), 0.0, self.N); return (self.win @ self.readout(n)) / self.sat

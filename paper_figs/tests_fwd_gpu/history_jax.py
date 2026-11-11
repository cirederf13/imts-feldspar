"""JAX forward model of the natural n/N profile for a prescribed T(t): geological filling (one ODE), Kars/Huntley fading chain (scan), lab read-out ramp (one ODE), window sums.
Mirrors inv_joint.hist_model (IMTS, charge neutral).  Units: time in years for the geological part, seconds for the read-out."""
import numpy as np, jax, jax.numpy as jnp, diffrax
jax.config.update("jax_enable_x64", True)
kB = 8.617333262e-5; YR = 3.15576e7

class Engine:
    def __init__(self, SI, GI, HA, lab_sat_value, ddot_kagy=None, rtol=1e-8):
        self.E, self.s, self.N = (jnp.asarray(v) for v in (SI.E, SI.s, SI.N)); self.Nnp = np.asarray(SI.N)
        self.D0 = float(SI.D0); self.A = float(GI.A_val); self.B = float(GI.B); self.M0 = float(GI.M0)
        self.ddot = SI.DOT if ddot_kagy is None else ddot_kagy                                # Gy/ka
        self.W = jnp.asarray(SI.W_K if SI.W_K.ndim == 2 else np.tile(SI.W_K[None, :], (SI.NT, 1))); self.LA = jnp.asarray(SI.LA_K)
        self.sat = np.asarray(lab_sat_value); self.rtol = rtol
        self.T0, self.Tmax = float(GI.T0), float(GI.T_max); self.Tout = jnp.asarray(GI.T_out - GI.T0)
        tc = np.asarray(HA.TC); self.win = jnp.asarray(np.array([((tc >= lo) & (tc < hi)).astype(float) for lo, hi in HA.W.values()]))   # (nwin, nT_out)
        self.G = self.ddot / 1000.0 / self.D0 * float(self.N.sum()); self.f = self.ddot / 1000.0 / self.D0                           # per year
        self.atol_hold = 1e-6 * float(self.Nnp.min())
    # --- geological filling: ONE implicit ODE over the whole history ---------------------------------------------------------------------
    # mode "steps": T piecewise constant on `edges` (needs small jumps between neighbouring stages, as for a proxy record: ~<1 C per step);
    # mode "linear": T linear between the nodes `edges` (continuous; use for coarse histories with large changes).  Known limitation: restarts after
    # a >1 C jump at long stages make the stepper collapse (diffrax max_steps), so coarse piecewise-constant histories must use "linear".
    def fill(self, edges, temps_C, n0=None, mode="steps"):
        edges = jnp.asarray(edges, float); T = jnp.asarray(temps_C, float) + 273.15; t_end = float(edges[-1]); A, B, M0, N, E, s, G = self.A * YR, self.B * YR, self.M0, self.N, self.E, self.s, self.G
        if mode == "steps": Tf = lambda t: T[jnp.clip(jnp.searchsorted(edges, t, side="right") - 1, 0, T.size - 1)]; jt = edges[1:-1] if edges.size > 2 else None
        else: Tf = lambda t: jnp.interp(t, edges, T); jt = None
        def rhs(t, n, _):
            lam = s * jnp.exp(-E / (kB * Tf(t))) * YR; nc = (jnp.sum(lam * n) + G) / (A * jnp.sum(N - n) + B * (M0 + jnp.sum(n))); return -lam * n + A * nc * (N - n)
        ctrl = diffrax.PIDController(rtol=self.rtol, atol=self.atol_hold, pcoeff=0.3, icoeff=0.4, jump_ts=jt)
        n_init = jnp.zeros_like(N) if n0 is None else jnp.asarray(n0)
        sol = diffrax.diffeqsolve(diffrax.ODETerm(rhs), diffrax.Kvaerno5(), 0.0, t_end, 1.0, n_init, stepsize_controller=ctrl, max_steps=4_000_000, saveat=diffrax.SaveAt(t1=True))
        return jnp.clip(sol.ys[-1], 0.0, N)
    # --- Kars fading factor for the same history (exact exponential steps) ------------------------------------------------------------
    def kars(self, edges, temps_C):
        edges = jnp.asarray(edges, float); dt = (edges[1:] - edges[:-1]); T = jnp.asarray(temps_C, float) + 273.15; f = self.f
        def chain(LAx):                                                                                           # LAx (NT,K) 1/yr
            def step(x, a):
                Tk, dtk = a; lam = (self.s * jnp.exp(-self.E / (kB * Tk)) * YR)[:, None]; r = f + lam + LAx; xe = f / r
                return xe + (x - xe) * jnp.exp(-r * dtk), None
            x, _ = jax.lax.scan(step, jnp.zeros_like(LAx), (T, dt)); return x
        LA = self.LA * YR
        return (self.W * chain(LA)).sum(1) / jnp.maximum(chain(jnp.zeros((self.E.size, 1)))[:, 0], 1e-300)
    # --- lab read-out ramp of the natural state (1 K/s) -------------------------------------------------------------------------------
    def readout(self, n):
        A, B, M0, N, E, s = self.A, self.B, self.M0, self.N, self.E, self.s
        def rhs(t, y, _):
            nn, nc = y[:-1], y[-1]; T = self.T0 + t; detr = s * nn * jnp.exp(-E / (kB * T)); retr = A * nc * (N - nn); loss = nc * (jnp.sum(A * (N - nn)) + B * (M0 + jnp.sum(nn) + nc))
            return jnp.concatenate([-detr + retr, jnp.array([jnp.sum(detr) - loss])])
        ctrl = diffrax.PIDController(rtol=1e-6, atol=1e-8, pcoeff=0.3, icoeff=0.4)
        sol = diffrax.diffeqsolve(diffrax.ODETerm(rhs), diffrax.Kvaerno5(), 0.0, self.Tmax - self.T0, 1e-3, jnp.concatenate([n, jnp.zeros(1)]), stepsize_controller=ctrl, max_steps=200000, saveat=diffrax.SaveAt(ts=self.Tout))
        Y = sol.ys; return B * Y[:, -1] * (M0 + Y[:, :-1].sum(1) + Y[:, -1])
    def profile(self, edges, temps_C):
        n = self.fill(edges, temps_C); n = jnp.clip(n * self.kars(edges, temps_C), 0.0, self.N); return (self.win @ self.readout(n)) / self.sat

"""JAX/diffrax geological filling of the charge-neutral IMTS under a time-varying temperature T(t) (prototype).
State n (traps), n_c quasi-stationary (same closed form as imts_step.hold). Time in years."""
import jax, jax.numpy as jnp, numpy as np, diffrax
jax.config.update("jax_enable_x64", True)
kB = 8.617333262e-5; YR = 3.15576e7

def make_hold(E, s, N, D0, M0, A=1e-8, B=2e-8, ddot_kagy=1.455, rtol=1e-8, atol_rel=1e-6):
    E, s, N = map(jnp.asarray, (E, s, N)); sumN = N.sum(); G = ddot_kagy / 1000.0 / D0 * sumN      # charge per YEAR... converted below
    G = G                                                                                          # (ddot/1000 Gy/yr)/D0 * sumN  [charges / yr]
    def rhs(t, n, args):
        Tfun = args
        lam = s * jnp.exp(-E / (kB * Tfun(t))) * YR                                                 # 1/yr
        nc = (jnp.sum(lam * n) + G) / (A * YR * jnp.sum(N - n) + B * YR * (M0 + jnp.sum(n)))
        return -lam * n + A * YR * nc * (N - n)
    term = diffrax.ODETerm(rhs); solver = diffrax.Kvaerno5()
    ctrl = diffrax.PIDController(rtol=rtol, atol=atol_rel * float(N.min()), pcoeff=0.3, icoeff=0.4)
    def hold(n0, t0, t1, Tfun, max_steps=200000):
        sol = diffrax.diffeqsolve(term, solver, t0, t1, 1.0, jnp.asarray(n0), args=Tfun, stepsize_controller=ctrl, max_steps=max_steps, saveat=diffrax.SaveAt(t1=True))
        return jnp.clip(sol.ys[-1], 0.0, N), sol.stats["num_steps"]
    return hold

def piecewise_T(edges_yr_old2young, temps_C):
    """T(t) in K for t = years since the OLDEST edge; piecewise linear in time through stage midpoints would smear steps, so use right-continuous steps"""
    e = jnp.asarray(edges_yr_old2young); T = jnp.asarray(temps_C) + 273.15
    def f(t): return T[jnp.clip(jnp.searchsorted(e, t, side="right") - 1, 0, T.size - 1)]
    return f

"""Exact non-interacting (A=0) model of the IMTS protocol. Independent first-order traps:
light(T) ∝ sum_i 0.8 N_i exp(-I_i(PH)) k_i(T) exp(-I_i(T)),  I_i(T)=∫_{T0}^{T} s_i exp(-E_i/kT') dT'/beta.
Preheat and read-out both start at T0 with the same beta -> one cumulative integral per trap serves both stages."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
kB, T0 = 8.617333262e-5, 295.15
def cum_integral(E, s, beta, Tfine):
    """I_i on the fine grid, shape (n_traps, len(Tfine)); trapezoid of s exp(-E/kT)/beta in T."""
    k = s[:, None] * jnp.exp(-E[:, None] / (kB * Tfine[None, :]))
    inc = 0.5 * (k[:, 1:] + k[:, :-1]) * jnp.diff(Tfine)[None, :] / beta
    return jnp.concatenate([jnp.zeros((len(E), 1)), jnp.cumsum(inc, axis=1)], axis=1), k
def a0_curve(E, s, N, beta, PH_K, T_out, nfine=6000, Tmax=745.0):
    Tfine = jnp.linspace(T0, Tmax, nfine)
    I, k = cum_integral(E, s, beta, Tfine)
    I_ph = jax.vmap(lambda row: jnp.interp(PH_K, Tfine, row))(I)                    # (n_traps,)
    w = 0.8 * N * jnp.exp(-I_ph)
    light_fine = jnp.sum(w[:, None] * k * jnp.exp(-I), axis=0)
    light = jnp.interp(T_out, Tfine, light_fine)
    mx = jnp.max(light)
    return light / jnp.maximum(mx, 1e-300), jnp.exp(-jnp.sum(0.0 * w))

def gok_curve(E, s, N, b, beta, PH_K, T_out, nfine=4000, Tmax=745.0, x0=0.8):
    """Independent general-order traps, normalised occupancy x=n/N:  dx/dt = -s x^b exp(-E/kT), x(0)=x0 (b>=1, per-trap vector).
    x(T) = x0 * [1 + (b-1) x0^(b-1) J]^(-1/(b-1)),  J = I_i(PH) + I_i(T)  (preheat and read-out both start at T0 at the same beta).
    light(T) ∝ sum_i N_i s_i exp(-E_i/kT) x_i^b."""
    Tfine = jnp.linspace(T0, Tmax, nfine)
    I, k = cum_integral(E, s, beta, Tfine)
    I_ph = jax.vmap(lambda row: jnp.interp(PH_K, Tfine, row))(I)
    bm = jnp.maximum(b - 1.0, 1e-6)
    J = I_ph[:, None] + I
    x = x0 * jnp.exp(-jnp.log1p(bm[:, None] * x0 ** bm[:, None] * J) / bm[:, None])
    light_fine = jnp.sum((N[:, None] * k) * x ** b[:, None], axis=0)
    light = jnp.interp(T_out, Tfine, light_fine)
    return light / jnp.maximum(jnp.max(light), 1e-300), None

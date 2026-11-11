"""Raw (not peak-normalised) TL light of the independent-trap models, for the MAAD growth fits.
A=0 :  L(T) = sum_i x0 N_i exp(-I_i(PH)) k_i(T) exp(-I_i(T))          (first order; initial filling x0 only scales the light)
GOK :  L(T) = sum_i N_i k_i(T) x_i(T)^b,  x = x0 [1 + (b-1) x0^(b-1) J]^(-1/(b-1)),  J = I_i(PH) + I_i(T)      (normalised occupancy x = n/N)
a0_model.cum_integral gives I_i(T) = (1/beta) int s exp(-E/kT') dT' on a fine temperature grid."""
import jax, jax.numpy as jnp
import a0_model as M
def _grids(E, s, beta, PH_K, nfine, Tmax):
    Tfine = jnp.linspace(M.T0, Tmax, nfine)
    I, k = M.cum_integral(E, s, beta, Tfine)
    I_ph = jax.vmap(lambda row: jnp.interp(PH_K, Tfine, row))(I)
    return Tfine, I, k, I_ph
def a0_light(E, s, N, x0, beta, PH_K, T_out, nfine=4000, Tmax=745.0):
    Tfine, I, k, I_ph = _grids(E, s, beta, PH_K, nfine, Tmax)
    fine = jnp.sum((x0 * N * jnp.exp(-I_ph))[:, None] * k * jnp.exp(-I), axis=0)
    return jnp.interp(T_out, Tfine, fine)
def gok_light(E, s, N, b, x0, beta, PH_K, T_out, nfine=4000, Tmax=745.0):
    Tfine, I, k, I_ph = _grids(E, s, beta, PH_K, nfine, Tmax)
    bm = jnp.maximum(b - 1.0, 1e-6)
    J = I_ph[:, None] + I
    x0c = jnp.broadcast_to(jnp.asarray(x0), E.shape)[:, None]                # x0 may be a scalar or a per-trap vector
    x = x0c * jnp.exp(-jnp.log1p(bm[:, None] * x0c ** bm[:, None] * J) / bm[:, None])
    fine = jnp.sum((N[:, None] * k) * x ** b[:, None], axis=0)
    return jnp.interp(T_out, Tfine, fine)

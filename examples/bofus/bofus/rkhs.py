from typing import NamedTuple, Self

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import erf

from typax import Float, Key, fn

from . import utils

X_MAX = 1.0  # domain is [-X_MAX, X_MAX]^d, periodic dimensions wrap

Periodic = tuple[bool, ...] | None


class RBFMixture[*B, M, D](NamedTuple):
    """RBF mixture f = sum_i a_i prod_j phi(x_j - x_ij; l_ij)."""

    l: Float[*B, M, D]
    x: Float[*B, M, D]
    a: Float[*B, M]

    @fn(jit=True)
    def __call__(self, x: Float[D], periodic: Periodic = None) -> Float[*B]:
        """Evaluate f at a single point x, lift for more. Periodic dimensions wrap."""
        return jnp.sum(self.a * jnp.exp(log_rbf(x - self.x, self.l, periodic)), axis=-1)

    @classmethod
    @fn(jit=True)
    def sample(
        cls,
        key: Key,
        shape: tuple[int, ...],
        l_range: tuple[Float, Float],
        a_range: tuple[Float, Float],
        sigma0: float = 1.0,
        coverage: float = 1.0,
    ) -> Self:
        """Draw iid mixtures of shape (*b, m, d) from the prior, then clip l and a to their ranges.
        log l ~ N(median, sigma0^2 / d), the median solved so the atoms fill coverage of the box (m + 1) E[phi^2] = coverage.
        Amplitudes make the box mean of Var f(x) 1 before clipping. Periodic dims use the line energy.
        """
        *b, m, d = shape
        key_l, key_x, key_a = jr.split(key, 3)
        sigma = sigma0 / np.sqrt(d)

        def log_energy(log_median):
            """log E[phi^2] over the box and the log-normal l."""
            z = jnp.linspace(-6.0, 6.0, 512)
            l = jnp.exp(log_median[:, None] + sigma * z)
            energy = X_MAX * jnp.sqrt(jnp.pi * l) * erf(2 * X_MAX / jnp.sqrt(l))
            energy = (energy + 0.5 * l * jnp.expm1(-4 * X_MAX**2 / l)) / (2 * X_MAX**2)
            return d * jnp.log(energy @ jax.nn.softmax(-0.5 * z**2))

        # coverage median interpolated over a wide range of candidate medians
        candidates = jnp.linspace(-20.0, 10.0, 512)
        log_median = jnp.interp(jnp.log(coverage / (m + 1)), log_energy(candidates), candidates)

        # at the median m E[phi^2] = m coverage / (m + 1), the amplitude variance is its inverse
        l = jnp.exp(log_median + sigma * jr.normal(key_l, shape))
        x = jr.uniform(key_x, shape, minval=-X_MAX, maxval=X_MAX)
        a = jr.normal(key_a, (*b, m)) * jnp.sqrt((m + 1) / (m * coverage))
        return cls(l.clip(*l_range), x, a.clip(*a_range))

    @fn(K="m", where="m >= M")
    def pad_to[K](self, m: int) -> "RBFMixture[*B, K, D]":
        """Pad with inert atoms: zero amplitude, period wide lengthscale, at the origin."""
        *b, m0, d = self.l.shape
        pad = (*b, m - m0, d)
        l = jnp.concat([self.l, jnp.full(pad, (2 * X_MAX) ** 2)], axis=-2)
        x = jnp.concat([self.x, jnp.zeros(pad)], axis=-2)
        a = jnp.concat([self.a, jnp.zeros((*b, m - m0))], axis=-1)
        return self._replace(l=l, x=x, a=a)


@fn(lift=True)
def _log_circle(theta: Float, s: Float, terms: int = 4) -> Float:
    """log of a unit-mass gaussian of angular variance s wrapped on the circle, at angle theta."""
    n = jnp.arange(-terms, terms + 1, dtype=theta.dtype)
    if s < 2 * jnp.pi:
        # narrow atoms: nearest images, error ~exp(-2 pi^2 (terms + 1/2)^2 / s)
        return jax.nn.logsumexp(-0.5 * (theta + 2 * jnp.pi * n) ** 2 / s)
    # wide atoms: Poisson dual Fourier series, error ~exp(-(terms + 1)^2 s / 2)
    harmonics = jnp.exp(-0.5 * n**2 * s) * jnp.cos(n * theta)
    return 0.5 * jnp.log(s / (2 * jnp.pi)) + jnp.log(harmonics.sum())


def log_rbf(x, l, periodic: Periodic = None, expansion_terms: int = 4):
    """log prod_j exp(-0.5 * x_j^2 / l_j), repeated with period 2 X_MAX on periodic dims.
    Choses the periodic evaluation path such that the error is below ~exp(-pi (terms + 1/2)^2).
    """
    if periodic is None or not any(periodic):
        return -0.5 * jnp.sum(x**2 / l, axis=-1)

    # only the dimensions that wrap pay for the periodic series
    wraps = np.asarray(periodic)
    line = -0.5 * jnp.sum(x[..., ~wraps] ** 2 / l[..., ~wraps], axis=-1)
    l, x = l[..., wraps], x[..., wraps]

    # angle in [-pi, pi] and angular variance on the circle
    theta = jnp.pi * x / X_MAX
    theta = theta - 2 * jnp.pi * jnp.round(theta / (2 * jnp.pi))
    s = (jnp.pi / X_MAX) ** 2 * l
    return line + jnp.sum(_log_circle(theta, s, expansion_terms), axis=-1)


@fn(lift=True)
def inner[M1, M2, D](
    f1: RBFMixture[M1, D],
    f2: RBFMixture[M2, D],
    l0: Float[D] | None = None,
    periodic: Periodic = None,
) -> Float:
    """Inner product in the rkhs of phi(.; l0) on [-X_MAX, X_MAX]^d, in L2 if l0 is None.
    RBF are analytic, so the rkhs inner product on the box is an isometry of the one on the line.
    See Steinwart, Hush, Scovel 2006, https://doi.org/10.1109/TIT.2006.881713
    The L2 inner product is scaled by 1 / sqrt(2 pi) per dimension.
    """
    # atom pairs on (m1, m2, d)
    l1, x1 = f1.l[:, None, :], f1.x[:, None, :]
    l2, x2 = f2.l[None, :, :], f2.x[None, :, :]
    wraps = np.zeros(f1.x.shape[-1], bool) if periodic is None else np.asarray(periodic)

    if l0 is None:  # L2 inner product, scaled by 1 / sqrt(2 pi)^d
        # box dimensions keep only the mass of each atom product N(mu, s) inside [-X_MAX, X_MAX]
        ls = l1 + l2
        mu = (l2 * x1 + l1 * x2) / ls
        root_2s = jnp.sqrt(2.0 * l1 * l2 / ls)
        mass = 0.5 * (erf((X_MAX - mu) / root_2s) - erf((-X_MAX - mu) / root_2s))
        mass = jnp.maximum(mass, jnp.finfo(mass.dtype).tiny)  # finite log for atoms off the box
        log_scale = 0.5 * jnp.log(l1 * l2 / ls) + jnp.where(wraps, 0.0, jnp.log(mass))
    else:
        # deconvolved atom variances: infinite if any atom does not fit the ambient rkhs
        v1 = l1 - l0 / 2
        v2 = l2 - l0 / 2
        if not jnp.all((v1 > 0.0) & (v2 > 0.0)):
            return jnp.inf
        ls = v1 + v2
        log_scale = 0.5 * jnp.log(l1 * l2 / (l0 * ls))

    K = jnp.exp(jnp.sum(log_scale, axis=-1) + log_rbf(x1 - x2, ls, periodic))
    return jnp.einsum("ij,i,j->", K, f1.a, f2.a)


@fn(lift=True)
def distance[M1, M2, D](
    f1: RBFMixture[M1, D],
    f2: RBFMixture[M2, D],
    l0: Float[D] | None = None,
    periodic: Periodic = None,
) -> Float:
    """Distance between RBF mixtures in the rkhs of phi(.; l0), in L2 on the box if l0 is None."""
    norms = inner(f1, f1, l0, periodic) + inner(f2, f2, l0, periodic)
    d2 = norms - 2.0 * inner(f1, f2, l0, periodic)
    if jnp.isinf(d2) or jnp.isnan(d2):  # inf - inf gives nan
        return jnp.inf
    return utils.safe_sqrt(d2)

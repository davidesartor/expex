from typing import Any, Callable

import jax.numpy as jnp
from jax.scipy.special import gammaln

from typax import Float, fn

from . import utils

Profile = Callable[..., Any]


@fn(lift=True)
def euclidean_distance[D](l: Float[D], x1: Float[D], x2: Float[D]) -> Float:
    """Euclidean distance scaled by squared lengthscales l."""
    return utils.safe_sqrt(jnp.sum(jnp.square(x1 - x2) / l))


@fn(lift=True)
def rbf(d: Float) -> Float:
    """Radial basis function kernel."""
    if jnp.isinf(d):
        return 0.0
    return jnp.exp(-0.5 * d**2)


@fn(lift=True)
def matern12(d: Float) -> Float:
    """Matern 1/2 kernel."""
    if jnp.isinf(d):
        return 0.0
    return jnp.exp(-d)


@fn(lift=True)
def matern32(d: Float) -> Float:
    """Matern 3/2 kernel."""
    if jnp.isinf(d):
        return 0.0
    return (1 + jnp.sqrt(3.0) * d) * jnp.exp(-jnp.sqrt(3.0) * d)


@fn(lift=True)
def matern52(d: Float) -> Float:
    """Matern 5/2 kernel."""
    if jnp.isinf(d):
        return 0.0
    return (1 + jnp.sqrt(5.0) * d + 5.0 / 3.0 * d**2) * jnp.exp(-jnp.sqrt(5.0) * d)


def _log_coef(p, i):
    return gammaln(p + 1) - gammaln(2 * p + 1) + gammaln(p + i + 1) - gammaln(i + 1) - gammaln(p - i + 1)


def matern(p: int):
    @fn(lift=True)
    def profile(d: Float) -> Float:
        """Matern kernel with half-integer smoothness p + 1/2."""
        if jnp.isinf(d):
            return 0.0
        r = jnp.sqrt(2.0 * p + 1.0) * d
        if r == 0.0:
            return 1.0
        # sum terms fully in log space, coefficients and powers overflow separately for large p
        log_2r = jnp.log(2.0 * r)
        k = 0.0
        for i in range(p + 1):
            k = k + jnp.exp(_log_coef(p, i) + (p - i) * log_2r - r)
        return k

    return profile

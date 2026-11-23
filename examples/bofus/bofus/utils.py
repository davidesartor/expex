import functools
import math
import time

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx
from typax import Bool, Float, Key, fn


@jax.jit
@fn
def mask_covariance[*B, N](cov: Float[*B, N, N], mask: Bool[*B, N]) -> Float[*B, N, N]:
    """Zero the masked-out rows/columns, an identity block keeps the factorization valid."""
    I = jnp.eye(cov.shape[-1], dtype=cov.dtype)
    observed = mask[..., None, :] & mask[..., :, None]
    return jnp.where(observed, cov, I)


@tx.lift
@fn
def safe_sqrt(x: Float) -> Float:
    """Square root with a zero subgradient wherever x <= 0."""
    if x > 0.0:
        return jnp.sqrt(x)
    return 0.0


@functools.partial(jax.jit, static_argnames="shape")
@fn
def latin_hypercube_sample(key: Key, shape: tuple[int, ...]):
    """Sample shape[0] unit-cube points, stratified over the flattened trailing dims."""
    n, *rest = shape
    dim = math.prod(rest)
    perm_key, u_key = jr.split(key)
    perms = []
    for k in jr.split(perm_key, dim):
        perms.append(jr.permutation(k, n))
    u = jr.uniform(u_key, (n, dim))
    x = (jnp.asarray(perms).T + u) / n
    return x.reshape(shape)


@jax.jit
@fn
def rescale(x, low, high):
    """Map unit-cube points to the box [low, high]."""
    return low + x * (high - low)


def timed(fn):
    """Wrap a function to also return its wall time in seconds."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        timer = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, time.perf_counter() - timer

    return wrapper

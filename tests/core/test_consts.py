"""``expex.eps``, ``expex.tiny``, ``expex.huge`` take the dtype of the array they meet."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import expex
from expex import f32, fn


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
def test_the_value_follows_the_dtype(dtype):
    x = jnp.ones(3, dtype)
    assert (x + expex.eps).dtype == dtype
    assert jnp.all((x + expex.eps) - x == jnp.finfo(dtype).eps)
    assert (jnp.zeros(3, dtype) + expex.tiny)[0] == jnp.finfo(dtype).smallest_normal
    assert (x * expex.huge)[0] == jnp.finfo(dtype).max
    assert expex.eps(x) == jnp.finfo(dtype).eps and expex.tiny(dtype) == jnp.finfo(dtype).smallest_normal


def test_scaling_stays_lazy_and_works_everywhere():
    scaled = 1e3 * expex.eps
    assert repr(scaled) == "1000 * expex.eps"
    x = jnp.ones(2, jnp.bfloat16)
    assert (x + scaled).dtype == jnp.bfloat16
    assert (x + expex.eps / 2)[0] == 1  # half an epsilon rounds away
    assert jnp.all(jnp.array([0.0, 1.0]) > expex.tiny) == jnp.array(False)
    assert (np.ones(2, np.float64) + expex.eps)[0] - 1 == np.finfo(np.float64).eps
    assert (-expex.eps + jnp.ones(1))[0] < 1

    @fn
    def safe_log[N](p: f32[N]) -> f32[N]:
        return jnp.log(p + expex.tiny)

    assert jnp.isfinite(jax.jit(safe_log)(jnp.zeros(3))).all()
    with pytest.raises(TypeError, match="needs a floating-point array"):
        jnp.arange(3) + expex.eps

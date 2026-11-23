"""`/` and `**` on sizes in an annotation are refused, not silently left unchecked."""

import jax.numpy as jnp
import pytest

import expex
from expex import f32
from expex.annotations.expr import ExprError


@pytest.mark.parametrize("op, fix", [("**", "T \\* T"), ("/", "//")])
def test_unsupported_size_arithmetic_is_refused(op, fix):
    ns: dict = {}
    src = f"""
import jax.numpy as jnp, expex
from expex import f32
@expex.fn
def f[T](x: f32[T]) -> f32[T {op} 2]:
    return jnp.zeros(5)
"""
    with pytest.raises(ExprError, match=fix) as e:
        exec(src, ns)
        ns["f"](jnp.ones(4))
    assert e.value.code == "EX403"


def test_supported_size_arithmetic_still_checks():
    @expex.fn
    def f[T](x: f32[T]) -> f32[T // 2]:
        return jnp.zeros(5)

    with pytest.raises(expex.ShapeError):
        f(jnp.ones(4))

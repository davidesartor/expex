"""Gradients through a traced `if` are NaN-free where a branch is not taken."""

from typing import NamedTuple

import pytest

import jax
import jax.numpy as jnp

import expex
from expex import Float, f32, fn


def where_sqrt(x):
    return jnp.where(x > 0, jnp.sqrt(x), 0.0)


@fn
def safe_sqrt(x: f32) -> f32:
    if x > 0:
        return jnp.sqrt(x)
    return 0.0


def test_the_problem_exists_with_plain_where():
    g = jax.vmap(jax.grad(where_sqrt))(jnp.array([4.0, 0.0, -1.0]))
    assert jnp.isnan(g[1]) and jnp.isnan(g[2])


def test_if_is_gradient_safe_under_vmap_and_vectorize():
    xs = jnp.array([4.0, 0.0, -1.0])
    for mapped in (jax.vmap(jax.grad(safe_sqrt)), jax.jit(jax.vmap(jax.grad(safe_sqrt)))):
        assert mapped(xs).tolist() == [0.25, 0.0, 0.0]
    g = jax.grad(lambda v: safe_sqrt.vectorize(v).sum())(xs)
    assert g.tolist() == [0.25, 0.0, 0.0]
    assert safe_sqrt.vectorize(xs).tolist() == [2.0, 0.0, 0.0]
    # forward mode and unbatched reverse mode agree
    assert jax.vmap(jax.jacfwd(safe_sqrt))(xs).tolist() == [0.25, 0.0, 0.0]
    assert jax.grad(safe_sqrt)(-1.0) == 0.0


def test_three_way_branch_with_closure_and_struct_reads():
    class P(NamedTuple):
        scale: f32
        shift: f32

    @fn
    def f(z: f32, p: P) -> f32:
        s = p.scale
        if z > 1:
            out = jnp.log(z - 1) * s
        elif z < -1:
            out = jnp.sqrt(-z - 1) + p.shift
        else:
            out = z * z * s
        return out

    zs = jnp.array([3.0, -5.0, 0.5, 1.0, -1.0])
    p = P(jnp.float32(2.0), jnp.float32(1.0))
    gz, gp = jax.vmap(jax.grad(f, argnums=(0, 1)), in_axes=(0, None))(zs, p)
    assert jnp.isfinite(gz).all() and jnp.isfinite(gp.scale).all() and jnp.isfinite(gp.shift).all()
    assert jnp.allclose(gz, jnp.array([1.0, -0.25, 2.0, 4.0, -4.0]))
    assert jnp.allclose(gp.scale, jnp.array([jnp.log(2.0), 0.0, 0.25, 1.0, 1.0]))
    assert jnp.allclose(gp.shift, jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]))
    # values are unchanged
    assert jnp.allclose(jax.vmap(f, in_axes=(0, None))(zs, p), jnp.array([jnp.log(2.0) * 2, 3.0, 0.5, 2.0, 2.0]))


def test_ternary_and_boolean_operands_are_safe_too():
    @fn
    def f(x: f32) -> f32:
        y = jnp.log(x) if x > 0 else -1.0
        return y

    assert jax.vmap(jax.grad(f))(jnp.array([2.0, 0.0, -3.0])).tolist() == [0.5, 0.0, 0.0]


def test_enclosing_function_variables_are_guarded():
    def make(scale):
        @fn
        def f(x: f32) -> f32:
            if x > 0:
                return jnp.sqrt(scale * x)
            return x * 0.0

        return f

    def loss(scale, xs):
        return jax.vmap(make(scale))(xs).sum()

    g = jax.grad(loss)(2.0, jnp.array([2.0, 0.0, -1.0]))
    assert jnp.isfinite(g) and jnp.allclose(g, 0.5)


def test_second_derivatives_and_loops():
    assert jax.vmap(jax.grad(jax.grad(safe_sqrt)))(jnp.array([4.0, -1.0])).tolist() == [-0.03125, 0.0]

    @fn
    def total[N](xs: Float[N]) -> Float:
        acc = 0.0
        for x in xs:
            if x > 0:
                acc = acc + jnp.log(x)
        return acc

    g = jax.jit(jax.grad(total))(jnp.array([2.0, -1.0, 0.0, 4.0]))
    assert g.tolist() == [0.5, 0.0, 0.0, 0.25]


def test_variadic_generic_structs():
    class Mix[*B, M, D](NamedTuple):
        l: Float[*B, M, D]
        a: Float[*B, M]

    @fn
    def weights[N, O, M, D](f: Mix[N, O, M, D]) -> Float[N, O]:
        return f.a.sum(-1)

    @fn
    def any_batch[*B, M, D](f: Mix[*B, M, D]) -> Float[*B]:
        return f.a.sum(-1)

    @fn
    def single[M, D](f: Mix[M, D]) -> Float:
        return f.a.sum()

    m = Mix(jnp.ones((2, 3, 4, 5)), jnp.ones((2, 3, 4)))
    assert weights(m).shape == (2, 3) and any_batch(m).shape == (2, 3)
    assert single(Mix(jnp.ones((4, 5)), jnp.ones(4))) == 4.0
    assert "f: Mix[N, O, M, D]" in repr(weights) and "f: Mix[*B, M, D]" in repr(any_batch)
    with pytest.raises(expex.ShapeError, match=r"declared  f.a: Float\[N, O, M\]"):
        weights(Mix(jnp.ones((2, 3, 4, 5)), jnp.ones((2, 3, 9))))
    with pytest.raises(expex.ShapeError, match="expected 2 axes, got 4"):
        single(m)
    for f in (weights, any_batch, single):
        assert expex.prove(f).status == "proved"

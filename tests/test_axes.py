"""Axes by name: ``x.sum(T)`` and ``axis=T`` inside an ``@fn``."""

import jax
import jax.numpy as jnp
import pytest

import typax as tx
from typax import Dim, ShapeError, f32, fn


class Vocab(Dim): ...


@fn
def pool[B, T, D](x: f32[B, T, D]) -> f32[B, D]:
    return x.mean(T)


def test_annotated_parameters_resolve_exactly():
    assert pool(jnp.ones((2, 3, 4))).shape == (2, 4)
    assert pool(jnp.ones((3, 3, 3))).shape == (3, 3)  # sizes coincide: the annotation decides
    x = jnp.arange(24.0).reshape(2, 3, 4)
    assert jnp.allclose(pool(x), x.mean(1))
    assert tx.prove(pool).ok
    assert jax.jit(pool)(x).shape == (2, 4) and pool.lift(jnp.ones((5, 2, 3, 4))).shape == (5, 2, 4)


def test_intermediates_resolve_by_size_and_locals_by_annotation():
    @fn
    def norm[B, T, D](x: f32[B, T, D]) -> f32[B, D]:
        y = jnp.tanh(x)
        return y.sum(T)

    assert jnp.allclose(norm(jnp.ones((2, 3, 4))), jnp.tanh(1.0) * 3)
    assert tx.prove(norm).ok

    @fn
    def square[B, T](x: f32[B, T]) -> f32[B]:
        y: f32[B, T] = x * 2
        return y.sum(T)

    assert jnp.allclose(square(jnp.ones((3, 3))), 6.0)  # B == T: the local annotation decides


def test_keywords_tuples_transpose_and_dim_classes():
    @fn
    def logits[T](x: f32[T, Vocab]) -> f32[T]:
        return jax.nn.logsumexp(x, axis=Vocab)

    @fn
    def perm[B, T, D](x: f32[B, T, D]) -> f32[B, D, T]:
        return x.transpose(B, D, T)

    @fn
    def total[B, T, D](x: f32[B, T, D]) -> f32[D]:
        return x.sum((B, T))

    x = jnp.arange(24.0).reshape(2, 3, 4)
    assert logits(jnp.ones((5, 7))).shape == (5,)
    assert jnp.array_equal(perm(x), x.transpose(0, 2, 1)) and tx.prove(perm).ok
    assert jnp.allclose(total(x), x.sum((0, 1)))


def test_struct_dimensions_and_variadics():
    @tx.struct
    class Pool[D]:
        w: f32[D]

        def __call__[T](self, x: f32[T, D]) -> f32[T]:
            return (x * self.w).sum(D)

    @fn
    def flat[*B, D](x: f32[*B, D]) -> f32[D]:
        return x.sum(B)

    assert Pool(jnp.ones(4))(jnp.ones((3, 4))).shape == (3,) and tx.prove(Pool.__call__).ok
    assert flat(jnp.ones((2, 3, 4))).shape == (4,)


def test_ambiguity_is_an_error_that_says_what_to_do():
    @fn
    def attn[T, D](q: f32[T, D], k: f32[T, D]) -> f32[T, T]:
        scores = q @ k.T
        return jax.nn.softmax(scores, axis=T)

    with pytest.raises(ShapeError, match="several axes of size T"):
        attn(jnp.ones((3, 4)), jnp.ones((3, 4)))

    @fn
    def twice[T](x: f32[T, T]) -> f32[T]:
        return x.sum(T)

    with pytest.raises(ShapeError, match="more than one axis"):
        twice(jnp.ones((3, 3)))


def test_names_that_are_not_axes_are_left_alone():
    @fn
    def f[N](x: f32[N], axis: int) -> f32:
        k = 0
        return x.sum(axis) + k

    assert f(jnp.ones(3), 0) == 3.0


def test_size_gives_the_size_of_a_name():
    @fn
    def init[*B, T](x: f32[*B, T]) -> f32[*B, T]:
        return jnp.zeros((*tx.size(B), tx.size(T)))

    @fn(U="T + 1")
    def pad[T, U](x: f32[T]) -> f32[U]:
        return jnp.zeros(tx.size(U)).at[: tx.size(T)].set(x)

    assert init(jnp.ones((2, 5, 3))).shape == (2, 5, 3) and tx.prove(init).ok
    assert jax.jit(init)(jnp.ones((4, 3))).shape == (4, 3)
    assert pad(jnp.ones(3)).tolist() == [1, 1, 1, 0] and tx.prove(pad).ok
    with pytest.raises(ShapeError, match="only available inside an @fn"):
        tx.size(Vocab)

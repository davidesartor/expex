"""Axes by name: ``x.sum(T)`` and ``axis=T`` inside an ``@fn``."""

from __future__ import annotations

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


def test_size_names_are_sizes_as_values():
    @fn
    def init[*B, T](x: f32[*B, T]) -> f32[*B, T]:
        return jnp.zeros((*B, T))

    @fn
    def pad[T](x: f32[T]) -> f32[T + 1]:
        return jnp.zeros(T + 1).at[:T].set(x)

    @fn
    def split[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:
        assert D % heads == 0
        h: f32[T, heads, D // heads] = x.reshape(T, heads, D // heads)  # sizes in the reshape and annotation
        return h.transpose(1, 0, 2)

    @fn
    def mean[N, D](x: f32[N, D]) -> f32[D]:
        acc = jnp.zeros(D)
        for i in range(N):
            acc = acc + x[i]
        return acc / N

    assert init(jnp.ones((2, 5, 3))).shape == (2, 5, 3) and tx.prove(init).ok
    assert jax.jit(init)(jnp.ones((4, 3))).shape == (4, 3)
    assert pad(jnp.ones(3)).tolist() == [1, 1, 1, 0] and tx.prove(pad).ok
    x = jnp.arange(40.0).reshape(5, 8)
    assert jnp.array_equal(split(x, 2), x.reshape(5, 2, 4).transpose(1, 0, 2)) and tx.prove(split).ok
    assert jnp.allclose(mean(x), x.mean(0)) and tx.prove(mean).ok
    with tx.config(checked=False):  # the sizes are still known
        assert pad(jnp.ones(2)).tolist() == [1, 1, 0]


def test_unknown_sizes_are_explained():
    @fn
    def onehot[T](tokens: tx.Int[T]) -> f32[T, Vocab]:
        return jax.nn.one_hot(tokens, Vocab)

    with pytest.raises(ShapeError, match="onehot.*the size Vocab is not known"):
        onehot(jnp.arange(3))


def test_sizes_defined_in_the_body():
    @fn
    def split[T, D](x: f32[T, D], heads: int) -> f32[H, T, K]:
        H, K = heads, D // heads
        assert D % heads == 0
        h: f32[T, H, K] = x.reshape(T, H, K)
        return h.transpose(H, T, K)

    x = jnp.arange(40.0).reshape(5, 8)
    assert jnp.array_equal(split(x, 2), x.reshape(5, 2, 4).transpose(1, 0, 2))
    assert tx.prove(split).ok
    with pytest.raises(ShapeError, match="D % heads == 0"):
        split(x, 3)

    @fn
    def wrong[T, D](x: f32[T, D], heads: int) -> f32[H, T, K]:
        H, K = heads, D // heads
        return x.reshape(T, H, K)

    with pytest.raises(ShapeError):
        wrong(x, 2)

    @tx.struct
    class Att[D]:
        w: f32[D, D]
        heads: int = 2

        def split[T](self, x: f32[T, D]) -> f32[H, T, K]:
            H, K = self.heads, D // self.heads
            assert D % self.heads == 0
            return x.reshape(T, H, K).transpose(H, T, K)

    assert Att(jnp.eye(8)).split(jnp.ones((3, 8))).shape == (2, 3, 4) and tx.prove(Att.split).ok

    @fn
    def ordinary[N](x: f32[N]) -> f32[N]:
        m = N * 2  # not in the signature: an ordinary local variable
        return x * m

    assert ordinary(jnp.ones(3)).tolist() == [6.0, 6.0, 6.0]


def test_einsum_by_names():
    @fn
    def attention[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]:
        return tx.einsum(q, k, to=(B, T, S))

    q, k = jnp.ones((2, 3, 4)), jnp.arange(40.0).reshape(2, 5, 4)
    assert jnp.allclose(attention(q, k), jnp.einsum("btd,bsd->bts", q, k)) and tx.prove(attention).ok

    @fn
    def self_attention[B, T, D, S](x: f32[B, T, D]) -> f32[B, T, T]:
        keys: f32[B, S, D] = x  # a second name for the same axis
        return tx.einsum(x, keys, to=(B, T, S))

    assert jnp.allclose(self_attention(k), jnp.einsum("btd,bsd->bts", k, k)) and tx.prove(self_attention).ok

    @fn
    def outer[T, E, N](dt: f32[T, E], b: f32[T, N], u: f32[T, E]) -> f32[T, E, N]:
        return tx.einsum(dt, b, u, to=(T, E, N))

    dt, b, u = jnp.arange(21.0).reshape(7, 3), jnp.ones((7, 5)), jnp.ones((7, 3))
    assert jnp.allclose(outer(dt, b, u), jnp.einsum("te,tn,te->ten", dt, b, u)) and tx.prove(outer).ok

    @fn
    def summed[T, D](x: f32[T, D]) -> f32[D]:
        h = jnp.tanh(x)  # unannotated: its names come from its sizes
        return tx.einsum(h, to=(D,))

    assert jnp.allclose(summed(jnp.ones((3, 4))), jnp.tanh(1.0) * 3)

    @fn
    def gram[T, D](x: f32[T, D]) -> f32[D, D]:
        return tx.einsum(x, x, to=(D, D))

    with pytest.raises(ShapeError, match="appears more than once in `to`"):
        gram(jnp.ones((3, 4)))


def test_rearrange_by_names():
    @fn
    def split[T, D](x: f32[T, D], heads: int) -> f32[H, T, K]:
        H, K = heads, D // heads
        assert D % heads == 0
        return tx.rearrange(x, to=(H, T, K))

    @fn
    def merge[H, T, K](h: f32[H, T, K]) -> f32[T, D]:
        D = H * K
        return tx.rearrange(h, to=(T, D))

    @fn
    def swap[B, T](x: f32[B, T]) -> f32[T, B]:
        return tx.rearrange(x, to=(T, B))

    x = jnp.arange(40.0).reshape(5, 8)
    assert jnp.array_equal(split(x, 2), x.reshape(5, 2, 4).transpose(1, 0, 2)) and tx.prove(split).ok
    assert jnp.array_equal(merge(split(x, 2)), x) and tx.prove(merge).ok
    assert jnp.array_equal(swap(x), x.T)

"""@tx.struct: typed, immutable modules that JAX can trace."""

from typing import Callable

import pytest

import jax
import jax.numpy as jnp

import typax as tx
from typax import ShapeError, f32, fn


@tx.struct
class Linear[I, O]:
    w: f32[O, I]
    b: f32[O]
    act: Callable = jnp.tanh
    name: str = "linear"

    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        return self.act(x @ self.w.T + self.b)

    def wrong[B](self, x: f32[B, I]) -> f32[B, I]:
        return x @ self.w.T

    @staticmethod
    def init(key, i: int, o: int) -> "Linear":
        return Linear(jax.random.normal(key, (o, i)), jnp.zeros(o))


@tx.struct
class MLP[I, H, O]:
    layers: tuple[Linear[I, H], Linear[H, O]]
    depth: int = 2

    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        for layer in self.layers:
            x = layer(x)
        return x


def test_struct_is_a_checked_pytree():
    lin = Linear.init(jax.random.key(0), 3, 5)
    assert lin(jnp.ones((2, 3))).shape == (2, 5)
    leaves, tree = jax.tree.flatten(lin)
    assert len(leaves) == 2  # w and b; act and name are static
    assert jax.tree.unflatten(tree, leaves).name == "linear"
    with pytest.raises(ShapeError, match="I is 3 \\(from `self.w: f32\\[5, 3\\]`\\)"):
        lin(jnp.ones((2, 4)))
    with pytest.raises(ShapeError, match="declared  self.b: f32\\[O\\]"):
        Linear(jnp.ones((5, 3)), jnp.zeros(4))(jnp.ones((2, 3)))


def test_struct_is_immutable_with_replace():
    lin = Linear.init(jax.random.key(0), 3, 5)
    with pytest.raises(Exception):
        lin.w = jnp.zeros((5, 3))  # type: ignore[misc]
    lin2 = lin.replace(b=jnp.ones(5))
    assert float(lin2.b[0]) == 1.0 and float(lin.b[0]) == 0.0


def test_struct_through_jit_grad_and_vmap():
    lin = Linear.init(jax.random.key(0), 3, 5)
    x = jnp.ones((2, 3))
    assert jnp.allclose(jax.jit(lambda m, x: m(x))(lin, x), lin(x))
    g = jax.grad(lambda m: m(x).sum())(lin)
    assert isinstance(g, Linear) and g.w.shape == (5, 3) and g.name == "linear"
    stacked = jax.vmap(lambda k: Linear.init(k, 3, 5))(jax.random.split(jax.random.key(1), 4))
    assert stacked.w.shape == (4, 5, 3)
    assert jax.vmap(lambda m: m(x))(stacked).shape == (4, 2, 5)


def test_nested_structs_and_proofs():
    k1, k2 = jax.random.split(jax.random.key(0))
    mlp = MLP((Linear.init(k1, 3, 8), Linear.init(k2, 8, 2)))
    assert mlp(jnp.ones((4, 3))).shape == (4, 2)
    with pytest.raises(ShapeError, match="H is 8"):
        MLP((Linear.init(k1, 3, 8), Linear.init(k2, 7, 2)))(jnp.ones((4, 3)))
    assert tx.prove(Linear.__call__).status == "proved"
    assert tx.prove(Linear.wrong).status == "failed"
    assert tx.prove(MLP.__call__).status == "proved"


def test_struct_options_apply_to_methods():
    @tx.struct(jit=True)
    class Scaled:
        k: float
        v: f32[3]  # type: ignore[valid-type]

        def apply(self, x: f32[3]) -> f32[3]:  # type: ignore[valid-type]
            return x * self.k + self.v

        @fn
        def eager(self, x: f32[3]) -> f32[3]:  # type: ignore[valid-type]
            return x

    s = Scaled(2.0, jnp.ones(3))
    assert s.apply(jnp.ones(3)).tolist() == [3.0, 3.0, 3.0]
    assert s.replace(v=jnp.zeros(3)).apply(jnp.ones(3)).tolist() == [2.0, 2.0, 2.0]
    assert len(Scaled.apply.traces) == 1  # new array field values: no retrace
    assert s.replace(k=3.0).apply(jnp.ones(3)).tolist() == [4.0, 4.0, 4.0]
    assert len(Scaled.apply.traces) == 2  # k is static
    assert not Scaled.eager.jit

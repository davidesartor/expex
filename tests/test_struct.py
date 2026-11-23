"""@tx.struct: typed, immutable modules that JAX can trace."""

from typing import Callable

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

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
        return Linear(jr.normal(key, (o, i)), jnp.zeros(o))


@tx.struct
class MLP[I, H, O]:
    layers: tuple[Linear[I, H], Linear[H, O]]
    depth: int = 2

    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        for layer in self.layers:
            x = layer(x)
        return x


def test_struct_is_a_checked_pytree():
    lin = Linear.init(jr.key(0), 3, 5)
    assert lin(jnp.ones((2, 3))).shape == (2, 5)
    leaves, tree = jax.tree.flatten(lin)
    assert len(leaves) == 2  # w and b; act and name are static
    assert jax.tree.unflatten(tree, leaves).name == "linear"
    with pytest.raises(ShapeError, match="I is 3 \\(from `self.w: f32\\[5, 3\\]`\\)"):
        lin(jnp.ones((2, 4)))
    with pytest.raises(ShapeError, match="declared  self.b: f32\\[O\\]"):
        Linear(jnp.ones((5, 3)), jnp.zeros(4))(jnp.ones((2, 3)))


def test_struct_is_immutable_with_replace():
    lin = Linear.init(jr.key(0), 3, 5)
    with pytest.raises(Exception):
        lin.w = jnp.zeros((5, 3))  # type: ignore[misc]
    lin2 = lin.replace(b=jnp.ones(5))
    assert float(lin2.b[0]) == 1.0 and float(lin.b[0]) == 0.0


def test_struct_through_jit_grad_and_vmap():
    lin = Linear.init(jr.key(0), 3, 5)
    x = jnp.ones((2, 3))
    assert jnp.allclose(jax.jit(lambda m, x: m(x))(lin, x), lin(x))
    g = jax.grad(lambda m: m(x).sum())(lin)
    assert isinstance(g, Linear) and g.w.shape == (5, 3) and g.name == "linear"
    stacked = jax.vmap(lambda k: Linear.init(k, 3, 5))(jr.split(jr.key(1), 4))
    assert stacked.w.shape == (4, 5, 3)
    assert jax.vmap(lambda m: m(x))(stacked).shape == (4, 2, 5)


def test_nested_structs_and_proofs():
    k1, k2 = jr.split(jr.key(0))
    mlp = MLP((Linear.init(k1, 3, 8), Linear.init(k2, 8, 2)))
    assert mlp(jnp.ones((4, 3))).shape == (4, 2)
    with pytest.raises(ShapeError, match="H is 8"):
        MLP((Linear.init(k1, 3, 8), Linear.init(k2, 7, 2)))(jnp.ones((4, 3)))
    assert tx.prove(Linear.__call__).status == "proved"
    assert tx.prove(Linear.wrong).status == "failed"
    assert tx.prove(MLP.__call__).status == "proved"


def test_methods_keep_their_own_decorators():
    @tx.struct
    class Scaled:
        k: float
        v: f32[3]  # type: ignore[valid-type]

        @jax.jit
        @fn
        def apply(self, x: f32[3]) -> f32[3]:  # type: ignore[valid-type]
            return x * self.k + self.v

        @tx.lift
        def norm(self, x: f32[3]) -> f32:  # type: ignore[valid-type]
            return jnp.linalg.norm(x * self.k)

    s = Scaled(2.0, jnp.ones(3))
    assert s.apply(jnp.ones(3)).tolist() == [3.0, 3.0, 3.0]
    assert s.replace(v=jnp.zeros(3)).apply(jnp.ones(3)).tolist() == [2.0, 2.0, 2.0]
    assert s.replace(k=3.0).apply(jnp.ones(3)).tolist() == [4.0, 4.0, 4.0]  # k is static
    with pytest.raises(tx.ShapeError):
        s.apply(jnp.ones(4))
    assert s.norm(jnp.ones((5, 3))).shape == (5,)

    with pytest.raises(TypeError, match="Put @tx.fn under @jax.jit"):

        @tx.struct
        class Bare:
            @jax.jit
            def f(self, x):
                return x


@tx.struct
class Block[D]:
    w: f32[D, D]
    b: f32[D]

    def __call__(self, x: f32[D]) -> f32[D]:
        return x + jnp.tanh(self.w @ x + self.b)


@tx.struct
class Deep[D]:
    blocks: list[Block[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for block in self.blocks:
            x = block(x)
        return x


def test_loops_over_identical_layers_are_scanned():
    keys = jr.split(jr.key(0), 24)
    deep = Deep([Block(0.3 * jr.normal(k, (8, 8)), jnp.zeros(8)) for k in keys])
    x = jnp.ones(8)

    def unrolled(m, x):
        for block in m.blocks:
            x = block(x)
        return x

    assert jnp.allclose(jax.jit(lambda m, x: m(x))(deep, x), unrolled(deep, x), atol=1e-5)
    assert any("lax.scan" in how for how in Deep.__call__.flow.values())
    staged = tx.debug.ops(lambda m, x: m(x), deep, x).count
    plain = tx.debug.ops(unrolled, deep, x).count
    assert staged * 5 < plain
    g = jax.grad(lambda m: m(x).sum())(deep)
    g_ref = jax.grad(lambda m: unrolled(m, x).sum())(deep)
    for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(g_ref)):
        assert jnp.allclose(a, b, atol=1e-5)
    assert isinstance(g.blocks, list) and len(g.blocks) == 24


def test_lists_of_different_layers_still_unroll():
    k1, k2 = jr.split(jr.key(0))
    mlp = MLP((Linear.init(k1, 3, 8), Linear.init(k2, 8, 2)))
    assert jax.jit(lambda m, x: m(x))(mlp, jnp.ones((4, 3))).shape == (4, 2)


@tx.struct
class Heads[D]:
    w: f32[D, D]
    heads: int = 2

    @fn
    def split[T](self, x: f32[T, D]) -> f32[self.heads, T, D // self.heads]:
        assert D % self.heads == 0
        t, d = x.shape
        return x.reshape(t, self.heads, d // self.heads).transpose(1, 0, 2)

    def __call__(self, x: f32[D]) -> f32[D]:
        return self.w @ x


@tx.struct
class UsesHeads[D]:
    inner: Heads[D]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        h = self.inner.split(x)
        return h.transpose(1, 0, 2).reshape(x.shape)


def test_invariants_are_checked_and_assumed():
    good = Heads(jnp.eye(4), heads=2)
    assert good.split(jnp.ones((3, 4))).shape == (2, 3, 2)
    with pytest.raises(ShapeError, match="Heads requires `D % heads == 0`, which does not hold \\(D=5, heads=2\\)"):
        Heads(jnp.eye(5), heads=2).split(jnp.ones((3, 5)))
    with pytest.raises(ShapeError, match="requires `D % heads == 0`"):
        UsesHeads(Heads(jnp.eye(5), heads=2))(jnp.ones((3, 5)))
    # the caller never states the constraint: it is assumed from the struct
    assert tx.prove(UsesHeads.__call__).status == "proved"
    assert tx.prove(Heads.split).status == "proved"
    # field values can be used by name in a method's derived sizes
    assert "-> f32[heads, T, D // heads]" in repr(Heads.split)


def test_struct_where_is_rejected():
    with pytest.raises(TypeError, match="@tx.struct got where=.*write invariants as an `assert`"):
        tx.struct(where="D % heads == 0")


def test_module_lift():
    m = Heads(jnp.eye(4) * 2, heads=2)
    assert m.lift(jnp.ones((5, 4))).shape == (5, 4)
    assert jnp.allclose(m.lift(jnp.ones((5, 4))), 2.0)


@tx.struct
class Attn[D]:
    wq: f32[D, D]
    wo: f32[D, D]
    heads: int = 2

    def __init__(self, key, d: int, heads: int = 2):
        kq, ko = jr.split(key)
        self.wq = jr.normal(kq, (d, d))
        self.wo = jr.normal(ko, (d, d))
        self.heads = heads

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        assert D % self.heads == 0
        return x @ self.wq @ self.wo


def test_custom_init():
    a = Attn(jr.key(0), 4)
    assert a.heads == 2 and a(jnp.ones((3, 4))).shape == (3, 4)
    with pytest.raises(Exception):
        a.wq = jnp.zeros((4, 4))  # frozen after __init__
    # rebuilt by JAX without calling __init__
    doubled = jax.tree.map(lambda w: 2 * w, a)
    assert isinstance(doubled, Attn) and jnp.allclose(doubled.wq, 2 * a.wq)
    g = jax.grad(lambda m: m(jnp.ones((3, 4))).sum())(a)
    assert isinstance(g, Attn) and g.heads == 2
    assert jnp.allclose(a.replace(heads=4).wq, a.wq)
    stacked = jax.vmap(lambda k: Attn(k, 4))(jr.split(jr.key(1), 3))
    assert stacked.wq.shape == (3, 4, 4)
    # the finished object is checked against its annotations and invariants
    with pytest.raises(ShapeError, match="Attn requires `D % heads == 0`, which does not hold \\(D=5, heads=2\\)"):
        Attn(jr.key(0), 5)
    assert jax.tree_util.keystr(jax.tree_util.tree_flatten_with_path(a)[0][0][0]) == ".wq"

    @tx.struct
    class Incomplete:
        w: f32[2]  # type: ignore[valid-type]

        def __init__(self):
            pass

    with pytest.raises(TypeError, match="did not set w"):
        Incomplete()

"""The default constructor: ``Cls(key, <sizes>)`` builds, ``Cls(<fields>)`` takes values."""

import dataclasses
import inspect
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import expex
from expex import Key, ShapeError, f32, i32


@expex.struct
class Block[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0
    act: Callable = jax.nn.gelu

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        return x + self.act(x @ self.w + self.b)


@expex.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for block in self.blocks:
            x = block(x)
        return x


@expex.struct
class Attention[D]:
    wq: f32[D, D] = jax.nn.initializers.normal(0.1)
    heads: int = 4

    def split[T](self, x: f32[T, D]) -> f32[self.heads, T, D // self.heads]:
        assert D % self.heads == 0
        return (x @ self.wq).reshape(T, self.heads, D // self.heads).transpose(1, 0, 2)


@expex.struct
class Model[D]:
    attn: Attention[D]
    blocks: list[Block[D]]
    scale: f32[D] = 1.0


@expex.struct
class State:  # a record of arrays: nothing to build
    x: f32
    t: i32


@expex.struct
class Seeded[D]:  # the first field is a key: a key first is its value
    key: Key
    w: f32[D] = 0.0


@expex.struct
class Custom[D]:
    w: f32[D, D] = jax.nn.initializers.normal(1.0)

    def __init__(self, n: int):
        self.w = jnp.eye(n)


def test_a_key_first_builds():
    model = Stack(jr.key(0), D=8, blocks=3)
    assert type(model) is Stack and len(model.blocks) == 3
    assert model.blocks[0].w.shape == (8, 8) and model.blocks[2].b.tolist() == [0.0] * 8
    assert not jnp.allclose(model.blocks[0].w, model.blocks[1].w)
    assert jnp.array_equal(Stack(jr.key(0), D=8, blocks=3).blocks[1].w, model.blocks[1].w)
    raw = Stack(jr.PRNGKey(0), D=8, blocks=2)  # a raw uint32[2] key
    assert raw.blocks[1].w.shape == (8, 8)
    assert model(jnp.ones((2, 8))).shape == (2, 8)


def test_same_as_expex_build():
    a = Stack(jr.key(3), D=4, blocks=2)
    b = expex.build(Stack, jr.key(3), D=4, blocks=2)
    assert jax.tree.all(jax.tree.map(jnp.array_equal, a, b))


def test_fields_and_settings_by_name_when_building():
    m = Model(jr.key(0), D=8, blocks=2, heads=2, scale=jnp.full(8, 3.0))
    assert m.attn.heads == 2 and m.scale.tolist() == [3.0] * 8  # a nested field, an own field
    a = Attention(jr.key(0), D=8, heads=8)  # a static field of the class itself
    assert a.heads == 8 and a.split(jnp.ones((3, 8))).shape == (8, 3, 1)
    w = jnp.ones((4, 4))
    blk = Block(jr.key(0), w=w)  # the first field by name, D read from it
    assert blk.w is w and blk.b.shape == (4,)
    s = Stack(jr.key(0), blocks=[blk, blk])  # a list of modules given, not built
    assert s.blocks[1] is blk


def test_field_values_as_before():
    w, b = jnp.ones((3, 3)), jnp.zeros(3)
    for blk in (Block(w, b), Block(w=w, b=b), Block(w, b=b), Block(w, b, jnp.tanh)):
        assert blk.w is w and blk.b is b
    assert Block(w, b).act is jax.nn.gelu and Block(w, b, jnp.tanh).act is jnp.tanh
    assert Stack([Block(w, b)]).blocks[0].w is w and Stack(blocks=[]).blocks == []
    st = State(jnp.float32(1), jnp.int32(2))
    assert st.t == 2 and State(x=jnp.float32(0), t=jnp.int32(0)).x == 0
    assert dataclasses.replace(Block(w, b), b=w[0]).b is not b  # the dataclass protocol too


def test_a_key_as_the_value_of_a_key_field():
    k = jr.key(0)
    s = Seeded(k, jnp.ones(3))
    assert s.key is k and s.w.shape == (3,)
    built = expex.build(Seeded, jr.key(1), D=3, key=k)  # building still works, explicitly
    assert built.w.tolist() == [0.0] * 3


def test_errors_name_what_is_missing():
    with pytest.raises(TypeError, match=r"^Stack\(key, \.\.\.\) needs D: the size of `blocks\[\*\]\.w`"):
        Stack(jr.key(0), blocks=2)
    with pytest.raises(TypeError, match=r"Stack\(key, \.\.\.\) needs blocks: the length of the list `blocks`"):
        Stack(jr.key(0), D=4)
    with pytest.raises(TypeError, match="unexpected keyword argument 'wat'"):
        Stack(jr.key(0), D=4, blocks=1, wat=1)
    with pytest.raises(TypeError, match=r"Block\(\) got D=\.\.\.: sizes .* only when building from a key"):
        Block(jnp.ones((2, 2)), jnp.zeros(2), D=2)
    with pytest.raises(TypeError, match=r"Block\(\): b not given\. .* Block\(key, <sizes>\)"):
        Block(jnp.ones((2, 2)))
    with pytest.raises(TypeError, match="two values for field `w`"):
        Block(jnp.ones((2, 2)), w=jnp.ones((2, 2)))
    with pytest.raises(ShapeError, match=r"Stack\(key, \.\.\.\): `blocks\[0\]\.w` is f32\[3, 3\]"):
        Stack(jr.key(0), D=4, blocks=1, w=jnp.ones((3, 3)))
    with pytest.raises(TypeError, match=r"expex\.build\(State, key, \.\.\.\) cannot build `x`: f32\[\(\)\] has no initializer"):
        expex.build(State, jr.key(0))


def test_a_custom_init_is_kept():
    c = Custom(3)
    assert c.w.tolist() == jnp.eye(3).tolist()
    built = expex.build(Custom, jr.key(0), D=4)
    assert built.w.shape == (4, 4)
    with pytest.raises(TypeError, match=r"expex\.build\(Custom, key, \.\.\.\) needs D"):
        expex.build(Custom, jr.key(0))
    assert not hasattr(Custom, "init") and not hasattr(Stack, "init")


def test_under_jit_and_vmap():
    keys = jr.split(jr.key(0), 3)
    stacked = jax.vmap(lambda k: Block(k, D=4))(keys)  # typed keys, traced
    assert stacked.w.shape == (3, 4, 4)
    raw = jax.vmap(lambda k: Block(k, D=4))(jr.split(jr.PRNGKey(0), 3))  # raw keys, traced
    assert raw.w.shape == (3, 4, 4)
    m = jax.jit(lambda k: Stack(k, D=4, blocks=2))(jr.key(1))
    assert isinstance(m, Stack) and m.blocks[1].w.shape == (4, 4)
    w = jax.jit(lambda w: Block(w, jnp.zeros(4)).w * 2)(jnp.ones((4, 4)))
    assert w.sum() == 32


def test_pytree_rebuilds_do_not_call_init():
    m = Stack(jr.key(0), D=4, blocks=2)
    calls = []
    original = Block.__init__
    Block.__init__ = lambda self, *a, **k: calls.append(1) or original(self, *a, **k)  # type: ignore[method-assign]
    try:
        m2 = jax.tree.map(lambda x: x + 1, m)
        m3 = m.replace(blocks=m.blocks[:1])
        jax.jit(lambda m: m)(m)
    finally:
        Block.__init__ = original  # type: ignore[method-assign]
    assert calls == [] and type(m2.blocks) is list and len(m3.blocks) == 1


def test_summary_diff_and_io_still_build(tmp_path):
    s = expex.summary(Stack, D=4, blocks=3, B=2)
    assert "× 3" in str(s)
    m = Stack(jr.key(0), D=4, blocks=2)
    expex.save(m, tmp_path / "m.npz")
    back = expex.load(Stack, tmp_path / "m.npz")
    assert str(expex.diff(m, back)) == "4 unchanged"
    assert np.array_equal(back.blocks[1].w, m.blocks[1].w)


def test_signature_shows_the_fields():
    params = list(inspect.signature(Block).parameters)
    assert params[:3] == ["w", "b", "act"] and params[-1] == "sizes"
    assert list(inspect.signature(State).parameters) == ["x", "t"]


def test_sizes_for_a_class_with_nothing_to_build():
    """One array field without a default: the sizes are still taken, and the error names the field."""

    @expex.struct
    class Embed[V, D]:
        table: f32[V, D]

    with pytest.raises(TypeError, match=r"cannot build `table`: f32\[V, D\] has no initializer"):
        Embed(jr.key(0), V=10, D=4)
    assert Embed(jnp.ones((10, 4))).table.shape == (10, 4)
    assert Embed(table=jnp.ones((10, 4))).table.shape == (10, 4)
    with pytest.raises(TypeError, match="positional"):
        Embed(jnp.ones((10, 4)), 10)
    with pytest.raises(TypeError, match="only when building from a key"):
        Embed(jnp.ones((10, 4)), V=10)
    assert expex.build(Embed, jr.key(0), table=jnp.ones((10, 4))).table.shape == (10, 4)

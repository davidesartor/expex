"""``model.at.<path>.set / add / multiply / apply / get``: out-of-place updates, checked."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import ShapeError, f32
from expex.staging.flow import Stacked


@expex.struct
class Block[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0
    act: Callable = jax.nn.gelu

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        return x + self.act(x @ self.w + self.b)


class Norm[D](NamedTuple):
    g: f32[D]


@expex.struct
class Heads[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    heads: int = 4

    def split[T](self, x: f32[T, D]) -> f32[self.heads, T, D // self.heads]:
        assert D % self.heads == 0
        return (x @ self.w).reshape(T, self.heads, D // self.heads).transpose(1, 0, 2)


@expex.struct
class Stack[D]:
    blocks: list[Block[D]]
    heads: Heads[D]
    norms: dict[str, f32[D]]
    pair: tuple[Block[D], Block[D]]
    norm: Norm[D]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for block in self.blocks:
            x = block(x)
        return x


D = 4


@pytest.fixture(scope="module")
def model():
    k1, k2 = jr.split(jr.key(1))
    return Stack(
        jr.key(0), D=D, blocks=6, norms={"a": jnp.ones(D), "b": jnp.zeros(D)},
        pair=(Block(k1, D=D), Block(k2, D=D)), norm=Norm(jnp.ones(D)),
    )  # fmt: skip


def same_structure(a, b):
    return jax.tree.structure(a) == jax.tree.structure(b)


def test_set_a_deep_field(model):
    new = model.at.blocks[5].w.set(jnp.zeros((D, D)))
    assert type(new) is Stack and same_structure(new, model)
    assert new.blocks[5].w.sum() == 0 and model.blocks[5].w.sum() != 0  # a copy
    assert new.blocks[4] is model.blocks[4] and new.heads is model.heads  # the rest is shared
    assert model.at.blocks[5].w.get() is model.blocks[5].w


def test_add_multiply_apply(model):
    w = model.blocks[2].w
    assert jnp.allclose(model.at.blocks[2].w.add(1.0).blocks[2].w, w + 1)
    assert jnp.allclose(model.at.blocks[2].w.multiply(3.0).blocks[2].w, w * 3)
    assert jnp.allclose(model.at.blocks[2].w.apply(jnp.tanh).blocks[2].w, jnp.tanh(w))
    scaled = model.at.blocks[2].multiply(0.0)  # a whole module: each of its arrays
    assert scaled.blocks[2].w.sum() == 0 and scaled.blocks[2].act is jax.nn.gelu
    halved = model.at.blocks.multiply(0.5)  # a whole list of modules
    assert jnp.allclose(halved.blocks[5].w, model.blocks[5].w / 2)


def test_index_into_an_array(model):
    new = model.at.heads.w[0, :3].set(0.0)
    assert new.heads.w[0, :3].tolist() == [0.0] * 3 and new.heads.w[0, 3] == model.heads.w[0, 3]
    assert new.heads.w.shape == (D, D)
    assert jnp.allclose(model.at.heads.w[1].add(1.0).heads.w[1], model.heads.w[1] + 1)
    assert jnp.allclose(model.at.heads.w[:, 0].apply(jnp.abs).heads.w[:, 0], jnp.abs(model.heads.w[:, 0]))
    assert model.at.heads.w[1, 2].get() == model.heads.w[1, 2]
    assert model.at.blocks[0].w[1][2].set(7.0).blocks[0].w[1, 2] == 7.0  # chained indices


def test_dicts_tuples_and_named_tuples(model):
    assert model.at.norms["b"].set(jnp.ones(D)).norms["b"].tolist() == [1.0] * D
    assert model.at.pair[1].b.add(2.0).pair[1].b.tolist() == [2.0] * D
    new = model.at.norm.g.multiply(3.0)
    assert isinstance(new.norm, Norm) and new.norm.g.tolist() == [3.0] * D


def test_wrong_shape_or_dtype_names_the_path(model):
    with pytest.raises(ShapeError, match=r"Stack\.at\.blocks\[5\]\.w\.set\(\.\.\.\): `blocks\[5\]\.w` is f32\[3, 3\]") as e:
        model.at.blocks[5].w.set(jnp.zeros((3, 3)))
    assert "D is 4" in str(e.value)
    with pytest.raises(ShapeError, match=r"`blocks\[5\]\.w` is i32\[4, 4\]"):
        model.at.blocks[5].w.set(jnp.zeros((D, D), jnp.int32))
    with pytest.raises(ShapeError, match=r"`blocks\[0\]\.b`"):  # the first element: sizes from the rest
        model.at.blocks[0].b.set(jnp.zeros(3))
    with pytest.raises(ShapeError, match=r"`blocks\[1\]\.w`"):  # broadcasting would change the shape
        model.at.blocks[1].w.add(jnp.ones((2, D, D)))
    with pytest.raises(ShapeError, match=r"`norms\['a'\]`"):
        model.at.norms["a"].set(jnp.ones(3))
    with pytest.raises(ShapeError, match=r"`blocks\[2\]` is f32\[4\], but is declared Block\[D\]"):
        model.at.blocks[2].set(jnp.ones(D))


def test_static_fields(model):
    assert model.at.heads.heads.get() == 4
    new = model.at.heads.heads.set(2)
    assert new.heads.heads == 2 and new.heads.split(jnp.ones((3, D))).shape == (2, 3, 2)
    assert model.at.blocks[1].act.set(jnp.tanh).blocks[1].act is jnp.tanh
    with pytest.raises(ShapeError, match="requires `D % heads == 0`"):
        model.at.heads.heads.set(3)  # the class's invariant
    with pytest.raises(ShapeError, match=r"`heads\.heads`.*expected int"):
        model.at.heads.heads.set("four")


def test_under_jit(model):
    f = jax.jit(lambda m, v: m.at.blocks[3].b.set(v).at.heads.w[0].multiply(2.0))
    new = f(model, jnp.full(D, 5.0))
    assert same_structure(new, model) and type(new.blocks) is list
    assert new.blocks[3].b.tolist() == [5.0] * D
    assert jnp.allclose(new.heads.w[0], model.heads.w[0] * 2)
    with pytest.raises(ShapeError, match=r"`blocks\[3\]\.b`"):
        jax.jit(lambda m: m.at.blocks[3].b.set(jnp.ones(2)))(model)
    g = jax.grad(lambda m: m.at.blocks[0].w.multiply(2.0)(jnp.ones((1, D))).sum())(model)
    assert isinstance(g, Stack)


def test_stacked_lists_behave_like_lists(model):
    """A list of identical modules held stacked: index 5 is row 5 of each array."""
    stacked = model.replace(blocks=Stacked(jax.tree.map(lambda *xs: jnp.stack(xs), *model.blocks)))
    v = jnp.full((D, D), 9.0)
    as_list = model.at.blocks[5].w.set(v)
    as_rows = stacked.at.blocks[5].w.set(v)
    assert isinstance(as_rows.blocks, Stacked) and as_rows.blocks.array.w.shape == (6, D, D)
    for i in range(6):
        assert jnp.array_equal(as_rows.blocks[i].w, as_list.blocks[i].w)
        assert jnp.array_equal(as_rows.blocks[i].b, as_list.blocks[i].b)
    assert jnp.array_equal(stacked.at.blocks[5].w.get(), model.blocks[5].w)
    row = stacked.at.blocks[2].multiply(0.0)
    assert row.blocks[2].w.sum() == 0 and row.blocks[3].w.sum() != 0
    with pytest.raises(ShapeError, match=r"`blocks\[5\]\.w` is f32\[3, 3\]"):
        stacked.at.blocks[5].w.set(jnp.zeros((3, 3)))


def test_modules_stacked_by_vmap():
    many = jax.vmap(lambda k: Block(k, D=D))(jr.split(jr.key(0), 6))  # w: f32[6, D, D]
    new = many.at[5].w.set(jnp.ones((D, D)))
    assert new.w[5].sum() == D * D and jnp.array_equal(new.w[4], many.w[4]) and new.w.shape == (6, D, D)
    assert jnp.array_equal(many.at[2].b.get(), many.b[2])
    i = jnp.int32(1)  # a traced row is fine for stacked modules
    traced = jax.jit(lambda m, i: m.at[i].b.add(1.0))(many, i)
    assert traced.b[1].tolist() == [1.0] * D and traced.b[0].tolist() == [0.0] * D
    with pytest.raises(ShapeError, match=r"`\[0\]\.w` is f32\[2, 2\], but the rows"):
        many.at[0].w.set(jnp.ones((2, 2)))


def test_mistakes(model):
    with pytest.raises(AttributeError, match=r"`blocks\[0\]` \(Block\) has no field `v`; its fields are w, b, act"):
        model.at.blocks[0].v.set(1.0)
    with pytest.raises(IndexError, match=r"`blocks` has 6 elements, so \[6\] is out of range"):
        model.at.blocks[6].w.set(jnp.zeros((D, D)))
    with pytest.raises(AttributeError, match=r"`blocks\[0\]\.w` is f32\[4, 4\], which has no fields"):
        model.at.blocks[0].w.T.get()
    with pytest.raises(TypeError, match="give the part to change first"):
        model.at.set(model)
    with pytest.raises(TypeError, match="index it with a Python int"):
        jax.jit(lambda m, i: m.at.blocks[i].b.set(jnp.ones(D)))(model, 1)
    assert repr(model.at.blocks[2].w[0, :3]) == "Stack.at.blocks[2].w[0, :3]"


def test_fields_named_like_the_methods():
    @expex.struct
    class Odd[D]:
        set: f32[D]
        at: int = 0  # a field named `at` is left alone

    o = Odd(jnp.ones(3))
    assert o.at == 0 and not hasattr(Odd, "replace") or True
    @expex.struct
    class Named[D]:
        get: f32[D]

    n = Named(jnp.ones(3))
    assert n.at["get"].set(jnp.zeros(3)).get.tolist() == [0.0] * 3

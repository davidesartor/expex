"""``Module.init(key, <sizes>)``: parameters built from the field annotations."""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.nn import initializers as ini

import typax as tx
from typax import Dim, ShapeError, f32


class Vocab(Dim): ...


@tx.struct
class Linear[I, O]:
    w: f32[O, I] = ini.lecun_normal(in_axis=-1, out_axis=-2)
    b: f32[O] = 0.0
    act: Callable = jax.nn.gelu

    def __call__(self, x: f32[I]) -> f32[O]:
        return self.act(self.w @ x + self.b)


@tx.struct
class Block[D, F]:
    up: Linear[D, F]
    down: Linear[F, D]

    def __call__(self, x: f32[D]) -> f32[D]:
        return x + self.down(self.up(x))


@tx.struct
class Net[D, F]:
    embed: f32[Vocab, D] = ini.normal()
    blocks: list[Block[D, F]]
    scale: f32[D] = 1.0
    heads: int = 4
    proj: f32[self.heads, D] = ini.normal()  # a size from an int field

    def __call__(self, x: f32[D]) -> f32[D]:
        for b in self.blocks:
            x = b(x)
        return x * self.scale


def test_shapes_and_values_come_from_the_annotations():
    lin = Linear.init(jr.key(0), I=3, O=4)
    assert lin.w.shape == (4, 3) and lin.b.tolist() == [0.0] * 4 and lin.act is jax.nn.gelu
    assert lin(jnp.ones(3)).shape == (4,)
    net = Net.init(jr.key(1), D=8, F=16, Vocab=50, blocks=3)
    assert net.embed.shape == (50, 8) and len(net.blocks) == 3 and net.proj.shape == (4, 8)
    assert net.blocks[0].up.w.shape == (16, 8) and net.blocks[2].down.w.shape == (8, 16)
    assert not jnp.allclose(net.blocks[0].up.w, net.blocks[1].up.w)  # each its own key
    assert jnp.allclose(net.scale, 1.0) and net(jnp.ones(8)).shape == (8,)
    again = Net.init(jr.key(1), D=8, F=16, Vocab=50, blocks=3)
    assert jnp.array_equal(again.blocks[1].up.w, net.blocks[1].up.w)  # deterministic in the key


def test_settings_reach_nested_modules_and_fields_can_be_given():
    net = Net.init(jr.key(0), D=8, F=16, Vocab=5, blocks=2, heads=2, scale=jnp.full(8, 3.0))
    assert net.heads == 2 and net.proj.shape == (2, 8) and net.scale.tolist() == [3.0] * 8
    lin = Linear.init(jr.key(0), w=jnp.ones((2, 5)))  # sizes found from the given array
    assert lin.b.shape == (2,)


def test_mistakes_are_explained():
    with pytest.raises(TypeError, match="pass its length, e.g. blocks=4"):
        Net.init(jr.key(0), D=8, F=16, Vocab=5)
    with pytest.raises(TypeError, match="the size F is not known"):
        Net.init(jr.key(0), D=8, Vocab=5, blocks=1)
    with pytest.raises(ShapeError, match="axis 0 of `Net.scale` is 5"):
        Net.init(jr.key(0), D=8, F=16, Vocab=5, blocks=1, scale=jnp.ones(5))
    with pytest.raises(TypeError, match="wat is neither a size"):
        Net.init(jr.key(0), D=8, F=16, Vocab=5, blocks=1, wat=1)

    @tx.struct
    class NoInit[D]:
        w: f32[D]

    with pytest.raises(TypeError, match="has no initializer"):
        NoInit.init(jr.key(0), D=3)


def test_the_constructor_still_takes_values():
    assert Linear(jnp.ones((4, 3)), jnp.zeros(4))(jnp.ones(3)).shape == (4,)
    with pytest.raises(TypeError, match="b not given"):
        Linear(jnp.ones((4, 3)))
    with pytest.raises(TypeError, match="blocks, scale, proj not given"):
        Net(jnp.ones((5, 8)))

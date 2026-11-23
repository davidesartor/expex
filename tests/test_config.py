"""Sizes from a config: ``Model(key, cfg)``, ``expex.cost(f, cfg)``, ``model.sizes``."""

import argparse
import dataclasses
import timeit
import warnings
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import f32


@expex.struct
class Attention[D]:
    wq: f32[D, D] = jax.nn.initializers.normal(0.1)
    heads: int = 4

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        assert D % self.heads == 0
        h = (x @ self.wq).reshape(T, self.heads, D // self.heads)
        return jax.nn.softmax(h, axis=-1).reshape(T, D)


@expex.struct
class Block[D, F]:
    attn: Attention[D]
    w1: f32[D, F] = jax.nn.initializers.normal(0.1)
    w2: f32[F, D] = jax.nn.initializers.normal(0.1)

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        x = x + self.attn(x)
        return x + jax.nn.gelu(x @ self.w1) @ self.w2


@expex.struct
class Model[D, F]:
    embed: f32[D] = 1.0
    blocks: list[Block[D, F]]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        x = x * self.embed
        for block in self.blocks:
            x = block(x)
        return x


@expex.struct
class Single[D]:  # one field: the config binds to it in the constructor
    w: f32[D] = 0.0


@expex.struct
class Activated[D]:  # the field after the key takes any callable
    w: f32[D] = 0.0
    act: Callable = jax.nn.relu


@dataclasses.dataclass
class TrainConfig:
    D: int = 16
    F: int = 32
    blocks: int = 3
    heads: int = 2
    lr: float = 3e-4  # not a size: ignored
    steps: int = 100


class Small:  # a typed config class, used as it is
    D: int = 8
    F: int = 16
    blocks: int = 2
    heads: int = 2
    batch_size: int = 64


class Named(NamedTuple):
    D: int
    F: int
    blocks: int


def same(a, b):
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        x.shape == y.shape and bool((x == y).all()) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def test_every_kind_of_config_builds_the_same_model():
    cfg = TrainConfig()
    want = Model(jr.key(0), D=16, F=32, blocks=3, heads=2)
    for c in [
        cfg,
        dataclasses.asdict(cfg),
        argparse.Namespace(**dataclasses.asdict(cfg)),
        type("Cfg", (), dict(D=16, F=32, blocks=3, heads=2, lr=0.1)),
    ]:
        got = Model(jr.key(0), c)
        assert same(got, want)
        assert got.blocks[0].attn.heads == 2
        assert same(expex.build(Model, jr.key(0), c), want)


def test_a_typed_config_class_and_a_namedtuple():
    m = Model(jr.key(0), Small)
    assert m.sizes == {"D": 8, "F": 16, "blocks": 2, "heads": 2}
    m = Model(jr.key(0), Named(8, 16, 2))
    assert m.sizes == {"D": 8, "F": 16, "blocks": 2, "heads": 4}  # heads: the default


def test_keywords_override_the_config():
    m = Model(jr.key(0), TrainConfig(), D=8, heads=4)
    assert m.sizes == {"D": 8, "F": 32, "blocks": 3, "heads": 4}
    m = expex.build(Model, jr.key(0), TrainConfig(), blocks=1)
    assert len(m.blocks) == 1


def test_a_config_for_a_one_field_class():
    assert Single(jr.key(0), {"D": 5, "lr": 1.0}).w.shape == (5,)


def test_a_callable_field_after_the_key():
    m = Activated(jr.key(0), {"D": 3})
    assert m.w.shape == (3,) and m.act is jax.nn.relu

    class Swish:  # a class by name, for a Callable field: its value
        beta: float = 1.0

    m = Activated(jr.key(0), D=3, act=Swish)
    assert m.act is Swish
    m = Activated(jr.key(0), Small)  # a class holding the sizes: a config
    assert m.w.shape == (8,) and m.act is jax.nn.relu


def test_keyword_dicts_stay_strict():
    with pytest.raises(TypeError, match="lr"):
        Model(jr.key(0), **dataclasses.asdict(TrainConfig()))


def test_a_misspelt_size_warns():
    with pytest.warns(expex.ConfigWarning, match=r"`haeds`.*did you mean `heads`"):
        m = Model(jr.key(0), dict(D=8, F=8, blocks=1, haeds=2))
    assert m.blocks[0].attn.heads == 4
    with pytest.warns(expex.ConfigWarning, match=r"`d`.*`D`"):
        with pytest.raises(TypeError, match="needs D"):
            Model(jr.key(0), dict(d=8, F=8, blocks=1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model(jr.key(0), dict(D=8, F=8, blocks=1, lr=0.1, dt=0.01, seed=0, block_lr=0.1))  # not close


def test_the_warning_points_at_the_caller():
    with pytest.warns(expex.ConfigWarning) as record:
        Model(jr.key(0), dict(D=8, F=8, blocks=1, head=2))
    assert record[0].filename == __file__
    with warnings.catch_warnings():  # `block` is close to `blocks`, which the config has too
        warnings.simplefilter("error")
        Model(jr.key(0), dict(D=8, F=8, blocks=1, block=2))


def test_model_sizes():
    m = Model(jr.key(0), TrainConfig())
    assert m.sizes == {"D": 16, "F": 32, "blocks": 3, "heads": 2}
    assert same(Model(jr.key(0), m.sizes), m)  # the same shape again
    abstract = jax.eval_shape(lambda: Model(jr.key(0), Small))
    assert abstract.sizes == {"D": 8, "F": 16, "blocks": 2, "heads": 2}
    assert Attention(jnp.eye(4)).sizes == {"D": 4, "heads": 4}


def test_cost_takes_a_config():
    cfg = dataclasses.replace(TrainConfig(), heads=4)  # the class default: formulas in the sizes
    want = expex.cost(Model.__call__, D=16, F=32, blocks=3, T=8)
    got = expex.cost(Model.__call__, cfg, T=8)
    assert got.flops == want.flops and got.parameters == want.parameters
    assert str(got) == str(want)
    m = Model(jr.key(0), cfg)
    bound = expex.cost(m.__call__, m.sizes, T=8)
    assert bound.flops == got.flops and bound.parameters == got.parameters
    symbolic = expex.cost(Model.__call__, {"F": 32, "lr": 0.1})  # D, L and T stay symbolic
    assert symbolic.flops == expex.cost(Model.__call__, F=32).flops


def test_cost_applies_a_field_the_class_default_does_not_have():
    cfg = dataclasses.replace(TrainConfig(), heads=4)  # the class default
    four = expex.cost(Model.__call__, cfg, T=8)
    cfg = dataclasses.replace(TrainConfig(), heads=8)  # built abstractly, so it counts
    eight = expex.cost(Model.__call__, cfg, T=8)
    m = Model(jr.key(0), cfg)
    assert eight.flops == expex.cost(m.__call__, T=8).flops != four.flops


def test_summary_and_diagram_take_a_config():
    cfg = TrainConfig()
    want = expex.summary(Model, D=16, F=32, blocks=3, heads=2, T=8)
    got = expex.summary(Model, cfg, T=8)
    assert str(got) == str(want)
    m = Model(jr.key(0), cfg)
    assert str(expex.summary(m, m.sizes, T=8)) == str(expex.summary(m, T=8))
    d = expex.diagram(Model.__call__, cfg)
    assert "Block" in d.text


def test_a_tool_given_something_else():
    with pytest.raises(TypeError, match="expected a config"):
        expex.cost(Model.__call__, 3)


def test_field_wise_construction_costs_the_same():
    """The config goes through the constructor's slow path only: values are as fast as before."""

    @expex.struct
    class Plain[D]:
        w: f32[D, D] = jax.nn.initializers.normal(0.1)
        b: f32[D] = 0.0

    w, b = jnp.ones((4, 4)), jnp.zeros(4)
    code = Plain.__init__.__code__
    assert "config" not in code.co_varnames and code.co_argcount == 5  # self, the first field, b, w by name, D
    t = min(timeit.repeat(lambda: Plain(w, b), number=2000, repeat=5)) / 2000
    assert t < 20e-6

"""Methods: ``self``'s annotated fields are checked and its dims are shared."""

import os
from dataclasses import dataclass, field

import pytest

import jax
import jax.numpy as jnp

import typax as tx
from typax import ShapeError, Static, f32, fn


@dataclass
class Linear[I, O]:
    w: f32[O, I]
    b: f32[O]
    name: str = "linear"

    @fn
    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        return x @ self.w.T + self.b

    @fn
    def wrong[B](self, x: f32[B, I]) -> f32[B, I]:
        return x @ self.w.T

    @fn(where="I >= 2")
    def first_two(self, x: f32[I]) -> f32[2]:  # type: ignore[valid-type]
        return x[:2]


def test_self_fields_bind_the_class_dimensions():
    lin = Linear(jnp.ones((5, 3)), jnp.zeros(5))
    assert lin(jnp.ones((2, 3))).shape == (2, 5)
    with pytest.raises(ShapeError) as e:
        lin(jnp.ones((2, 4)))
    assert "I is 3 (from `self.w: f32[5, 3]`), but axis 1 of `x` is 4." in str(e.value)
    assert "(self: Linear, x: f32[B, I]) -> f32[B, O]" in repr(Linear.__call__)


def test_inconsistent_instance_is_reported():
    bad = Linear(jnp.ones((5, 3)), jnp.zeros(4))
    with pytest.raises(ShapeError, match="declared  self.b: f32\\[O\\]"):
        bad(jnp.ones((2, 3)))


def test_methods_are_proved_without_an_instance():
    assert tx.prove(Linear.__call__).status == "proved"
    r = tx.prove(Linear.wrong)
    assert r.status == "failed" and "returned     f32[B, O]" in r.detail
    assert tx.prove(Linear.first_two).status == "proved"
    # and with one
    lin = Linear(jnp.ones((5, 3)), jnp.zeros(5))
    assert tx.prove(lin.__call__).status == "proved"


def test_method_lift_jit_and_grad():
    lin = Linear(jnp.ones((5, 3)), jnp.zeros(5))
    assert lin.__call__.lift(jnp.ones((7, 2, 3))).shape == (7, 2, 5)
    assert jax.jit(lin.__call__)(jnp.ones((2, 3))).shape == (2, 5)
    g = jax.grad(lambda x: lin(x).sum())(jnp.ones((2, 3)))
    assert g.shape == (2, 3)


def test_plain_classes_are_unaffected():
    class Plain:
        def __init__(self):
            self.k = 2.0

        @fn
        def call[D](self, x: f32[D]) -> f32[D]:
            return x * self.k

    assert Plain().call(jnp.ones(3)).shape == (3,)
    assert tx.prove(Plain.call).status == "unchecked"


def test_stacked_model():
    @dataclass
    class MLP[I, H, O]:
        l1: Linear[I, H]
        l2: Linear[H, O]
        depth: int = field(default=2)

        @fn
        def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
            return self.l2(jax.nn.relu(self.l1(x)))

    mlp = MLP(Linear(jnp.ones((8, 3)), jnp.zeros(8)), Linear(jnp.ones((2, 8)), jnp.zeros(2)))
    assert mlp(jnp.ones((4, 3))).shape == (4, 2)
    assert tx.prove(MLP.__call__).status == "proved"
    broken = MLP(Linear(jnp.ones((8, 3)), jnp.zeros(8)), Linear(jnp.ones((2, 7)), jnp.zeros(2)))
    with pytest.raises(ShapeError, match="H is 8"):
        broken(jnp.ones((4, 3)))


def test_compile_cache(tmp_path):
    old = jax.config.jax_compilation_cache_dir
    try:
        path = tx.compile_cache(str(tmp_path / "cache"), min_compile_seconds=0.0)
        assert os.path.isdir(path) and jax.config.jax_compilation_cache_dir == path
        assert jax.jit(lambda x: x * 2 + 1)(jnp.ones(3)).shape == (3,)
    finally:
        jax.config.update("jax_compilation_cache_dir", old)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

"""Type-checked by pyright and mypy in tests/test_typing.py; never executed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, assert_type

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx
from typax import Dim, Float, Int, Key, Static, f32, fn, i32


class RGB(Dim, size=3): ...


class Hidden(Dim): ...


class Params[D](NamedTuple):
    w: f32[D, D]
    b: f32[D]


@dataclass
class Layer[I, O]:
    w: f32[O, I]


@fn
def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    return jnp.einsum("btd,bsd->bts", q, k)


@fn
def pad[B, T](x: f32[B, T]) -> f32[B, T + 1]:  # type: ignore[valid-type]  # size arithmetic
    assert T >= 2  # type: ignore[misc]
    return jnp.pad(x, ((0, 0), (0, 1)))


@tx.random
@fn(examples={"n": [1, 2]})
def step[B](x: f32[B], lr: f32, n: int, key: Key, mode: str = "a") -> f32[B]:
    return x * lr * n + jr.normal(tx.key(), x.shape)


@fn
def misc[*Batch, H, W](
    img: f32[*Batch, H, W, RGB],
    idx: Int[*Batch],
    fixed: f32[Literal[2], Any],
    scalar: f32,
    p: Params[Hidden],
    layer: Layer[Hidden, H],
    opt: f32[H] | None = None,
) -> tuple[f32[*Batch, H, W], i32]:
    return img.mean(-1), jnp.int32(0)


class Model:
    @fn
    def call[D](self, x: f32[D], scale: float = 1.0) -> f32[D]:
        return x * scale


x = jnp.ones((2, 3, 4))
assert_type(attend(x, x), jax.Array)
assert_type(attend.lift(x, x), jax.Array)
assert_type(pad(x[0]), jax.Array)
assert_type(step(x[0, 0], jnp.float32(0.1), 2, jr.key(0)), jax.Array)
assert_type(Model().call(x[0, 0]), jax.Array)
assert_type(Model().call.lift(x[0], scale=2.0), jax.Array)

report = tx.prove(attend)
assert_type(report.ok, bool)
attend.check().raise_if_failed()
jax.jit(step, static_argnames=("n", "mode"))(x[0, 0], jnp.float32(0.1), 2, jr.key(0))
lifted = tx.lift(attend)
assert_type(lifted(x, x), jax.Array)
attend.flow
with tx.rng(0):
    tx.key()

# Errors a type checker should catch:
attend(x)  # E: missing argument
step(x[0, 0], jnp.float32(0.1), "two", jr.key(0))  # E: n must be an int
Model().call(x[0, 0], scale="big")  # E: scale must be a float

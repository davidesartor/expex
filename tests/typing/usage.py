"""Type-checked by pyright and mypy in tests/test_typing.py; never executed."""

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, assert_type

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Dim, Float, Int, Key, Static, f32, fn, i32


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


@expex.random
@fn(examples={"n": [1, 2]})
def step[B](x: f32[B], lr: f32, n: int, key: Key, mode: str = "a") -> f32[B]:
    return x * lr * n + jr.normal(expex.key(), x.shape)


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


# Fixed sizes, in the three spellings expex reads. `R = 6` is flagged on its line
# only ("default must be a type"); the `Literal` forms pass both checkers as is.
@fn
def rows[B, R = 6, C = 7](x: f32[B, R, C]) -> f32[B, R]:  # type: ignore[misc]  # R = 6: a fixed size
    return x.sum(-1)


@fn
def rows_literal[B, R = Literal[6], C = Literal[7]](x: f32[B, R, C]) -> f32[B, R]:
    return x.sum(-1)


@fn
def rows_bound[B, R: Literal[6], C: Literal[7]](x: f32[B, R, C]) -> f32[B, R]:
    return x.sum(-1)


@dataclass
class Board[Rows = 6, Cols = 7]:  # type: ignore[misc]  # Rows = 6: a fixed size
    cells: i32[Rows, Cols]


@dataclass
class BoardLiteral[Rows = Literal[6], Cols = Literal[7]]:
    cells: i32[Rows, Cols]


@dataclass
class BoardBound[Rows: Literal[6], Cols: Literal[7]]:
    cells: i32[Rows, Cols]


@expex.struct
class Dense[I, O]:
    w: f32[I, O]
    b: f32[O]
    act: bool = True

    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        return x @ self.w + self.b


class Model:
    @fn
    def call[D](self, x: f32[D], scale: float = 1.0) -> f32[D]:
        return x * scale


x = jnp.ones((2, 3, 4))
assert_type(attend(x, x), jax.Array)
assert_type(attend.vectorize(x, x), jax.Array)
assert_type(pad(x[0]), jax.Array)
assert_type(step(x[0, 0], jnp.float32(0.1), 2, jr.key(0)), jax.Array)
assert_type(Model().call(x[0, 0]), jax.Array)
assert_type(Model().call.vectorize(x[0], scale=2.0), jax.Array)
assert_type(rows(x[:, :2]), jax.Array)
assert_type(rows_literal(x[:, :2]), jax.Array)
assert_type(rows_bound(x[:, :2]), jax.Array)
assert_type(Board(jnp.zeros((6, 7), jnp.int32)).cells, jax.Array)
assert_type(BoardLiteral(jnp.zeros((6, 7), jnp.int32)).cells, jax.Array)
assert_type(BoardBound(jnp.zeros((6, 7), jnp.int32)).cells, jax.Array)

dense: Dense[Hidden, Hidden] = Dense(jnp.ones((4, 3)), jnp.zeros(3))  # a frozen dataclass to checkers
assert_type(dense.w, jax.Array)
assert_type(dense.act, bool)
assert_type(dense(x[0]), jax.Array)
assert_type(Dense(w=jnp.ones((4, 3)), b=jnp.zeros(3), act=False).b, jax.Array)

report = expex.prove(attend)
assert_type(report.ok, bool)
attend.check().raise_if_failed()
jax.jit(step, static_argnames=("n", "mode"))(x[0, 0], jnp.float32(0.1), 2, jr.key(0))
lifted = expex.vectorize(attend)
assert_type(lifted(x, x), jax.Array)
attend.flow
with expex.rng(0):
    expex.key()

# Errors a type checker should catch:
attend(x)  # E: missing argument
step(x[0, 0], jnp.float32(0.1), "two", jr.key(0))  # E: n must be an int
Model().call(x[0, 0], scale="big")  # E: scale must be a float
rows("x")  # E: still checked where sizes are fixed
rows_literal("x")  # E: still checked where sizes are fixed
rows_bound("x")  # E: still checked where sizes are fixed
Dense(jnp.ones((4, 3)))  # E: the field b is missing
Dense(jnp.ones((4, 3)), jnp.zeros(3), act="yes")  # E: act must be a bool
dense.w = jnp.ones((4, 3))  # E: a struct is frozen

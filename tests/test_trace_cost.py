"""What tracing costs and what it emits, pinned without timing anything.

Compile time is trace time (Python, and JAX's passes over what is traced) plus
XLA time (which grows with the program). These tests count what both depend
on: how many times a body is traced, and which equations the program holds,
for the patterns the ports use (benchmarks/run.py --compile-only measures the
times themselves).
"""

import collections

import jax
import jax.extend.core
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import expex
from expex import f32, fn, i32


def primitives(f, *args) -> collections.Counter:
    """How many times each primitive appears in the program, inside loops and branches too."""
    counts: collections.Counter = collections.Counter()

    def walk(j):
        for e in j.eqns:
            counts[e.primitive.name] += 1
            for p in e.params.values():
                for sub in p if isinstance(p, (tuple, list)) else (p,):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        walk(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        walk(sub)

    walk(jax.make_jaxpr(f)(*args).jaxpr)
    return counts


# ------------------------------------------------- `a and b` on traced values


@fn
def run_length(board: i32[6, 7], row: i32, col: i32) -> i32:
    """Connect Four's walk: a `while` whose test is a chain of `and`s on traced values."""
    n, r = 1, row + 1
    while 0 <= r < 6 and 0 <= col < 7 and board[r, col] == board[row, col]:
        n += 1
        r += 1
    return n


BOARD = jnp.array([[0] * 7] * 3 + [[1] * 7] * 3, jnp.int32)


def test_and_of_small_traced_conditions_is_a_select_not_a_cond():
    counts = primitives(run_length, BOARD, jnp.int32(3), jnp.int32(2))
    assert counts["cond"] == 0  # was one lax.cond per `and`, per evaluation of the test
    assert counts["while"] == 1
    assert counts["not"] == 0  # the untaken path's condition, built only if an assert reads it
    assert sum(counts.values()) <= 60
    assert int(jax.jit(run_length)(BOARD, 3, 2)) == 3
    assert int(jax.jit(run_length)(BOARD, 2, 2)) == 1
    assert int(jax.jit(run_length)(BOARD, 5, 2)) == 1  # r = 6: `board[r, col]` is not reached in Python
    assert jax.vmap(run_length, (None, 0, None))(BOARD, jnp.arange(6), 0).tolist() == [1, 1, 1, 3, 2, 1]


def test_and_or_values_are_python_s():
    @fn
    def pick(a: i32, b: i32) -> tuple[i32, i32, bool]:
        return a and b, a or b, (a > 0) and (b > 0)

    for a, b in [(0, 5), (3, 5), (3, 0), (0, 0), (-2, 7)]:
        got = jax.jit(pick)(jnp.int32(a), jnp.int32(b))
        assert [int(v) for v in got] == [a and b, a or b, (a > 0) and (b > 0)]
    assert primitives(pick, jnp.int32(1), jnp.int32(2))["cond"] == 0


def test_a_costly_or_float_or_effectful_side_stays_a_cond():
    @fn
    def costly(x: f32[64, 64]) -> bool:
        return (x[0, 0] > 0) and ((x @ x @ x).sum() > 0)  # a matmul is worth skipping

    @fn
    def floating(x: f32) -> f32:
        return (x > 0) and jnp.sqrt(x)  # a float: gradient safety needs the cond

    @fn
    def printing(x: i32) -> i32:
        return (x > 0) and _shout(x)

    def _shout(x):
        jax.debug.print("{}", x)
        return x

    assert primitives(costly, jnp.ones((64, 64)))["cond"] == 1
    assert primitives(floating, 1.0)["cond"] == 1
    assert primitives(printing, jnp.int32(1))["cond"] == 1
    assert float(jax.jit(jax.grad(floating))(0.0)) == 0.0  # no NaN from the untaken sqrt


# ---------------------------------------------- `if` on a mapped condition


@fn
def safe_sqrt(x: f32) -> f32:
    if x > 0:
        return jnp.sqrt(x)
    return 0.0


def test_if_on_a_mapped_condition_is_one_select_with_masks_on_float_inputs():
    xs = jnp.array([4.0, 0.0, -1.0])
    counts = primitives(jax.vmap(safe_sqrt), xs)
    assert counts["cond"] == 0  # JAX's batching rule would turn it into the same select
    assert counts["select_n"] == 3  # the result, and x masked in each branch
    assert counts["stop_gradient"] == 2
    assert counts["convert_element_type"] == counts["eq"] == 0  # JAX's rule compares an index per branch
    # the gradient is still NaN-free, in every order of the transformations
    for g in (jax.vmap(jax.grad(safe_sqrt)), jax.jit(jax.vmap(jax.grad(safe_sqrt))),
              lambda v: jax.grad(lambda v: jax.vmap(safe_sqrt)(v).sum())(v)):  # fmt: skip
        assert g(xs).tolist() == [0.25, 0.0, 0.0]
    assert jax.vmap(safe_sqrt)(xs).tolist() == [2.0, 0.0, 0.0]


def test_integer_and_boolean_inputs_of_a_mapped_if_are_not_masked():
    @fn
    def step(board: i32[6, 7], done: bool, score: f32) -> tuple[i32[6, 7], f32]:
        if done:
            return board, score
        return board.at[0, 0].set(1), score + 1.0

    boards = jnp.zeros((4, 6, 7), jnp.int32)
    counts = primitives(jax.vmap(step), boards, jnp.array([True, False, True, False]), jnp.zeros(4))
    assert counts["cond"] == 0
    assert counts["stop_gradient"] == 2  # `score` in each branch; never the board or the flag
    b, s = jax.vmap(step)(boards, jnp.array([True, False, True, False]), jnp.zeros(4))
    assert s.tolist() == [0.0, 1.0, 0.0, 1.0] and b[:, 0, 0].tolist() == [0, 1, 0, 1]


def test_unmapped_if_on_floats_stays_a_cond():
    assert primitives(safe_sqrt, 1.0)["cond"] == 1
    assert float(jax.grad(safe_sqrt)(-1.0)) == 0.0


def test_mapped_if_with_a_print_is_left_to_jax(monkeypatch):
    from expex.staging import flow

    built = []
    monkeypatch.setattr(flow, "_select", lambda *a, **k: built.append(1))

    @fn
    def loud(x: f32) -> f32:
        if x > 0:
            print(x)  # traced: a callback, an effect
            return x
        return -x

    jax.make_jaxpr(jax.vmap(loud))(jnp.ones(3))
    assert not built  # lax.cond, which JAX's batching rule handles as before


# --------------------------------------------------- bodies traced once


def counting(body):
    """``body``, counting the times it runs (once per trace when staged)."""
    calls = []

    def wrapped(*a, **k):
        calls.append(1)
        return body(*a, **k)

    return wrapped, calls


def test_staged_loop_bodies_are_traced_once():
    step, calls = counting(lambda h, x: jnp.tanh(h + x))

    @fn
    def scan[T](xs: f32[T, 4]) -> f32[4]:
        h = jnp.zeros(4)
        for x in xs:
            h = step(h, x)
        return h

    jax.jit(scan).lower(jnp.ones((100, 4)))
    assert len(calls) == 1

    test, tests = counting(lambda n, x: n < x)
    grow, grows = counting(lambda n: n * 2)

    @fn
    def doubling(x: i32) -> i32:
        n = 1
        while test(n, x):
            n = grow(n)
        return n

    jax.jit(doubling).lower(jnp.int32(9))
    assert len(grows) == 1
    assert len(tests) == 2  # once to find the condition traced, once as the loop's condition
    assert int(jax.jit(doubling)(jnp.int32(9))) == 16

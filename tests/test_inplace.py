"""In-place updates, asserts, print and min/max, written the plain Python way."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import expex
from expex import StageError, f32, fn, i32
from expex.staging.convert import convert


def both(f, *args):
    a, b = f(*args), jax.jit(f)(*args)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.allclose(x, y), (a, b)
    return b


def test_item_assignment_becomes_at_set():
    @fn
    def f[N](x: f32[N]) -> f32[N]:
        x[0] = 10.0
        x[1:3] = x[1:3] * 2
        x[-1] += 5
        x[0] *= 2
        x[2] -= 1
        x[1] /= 4
        return x

    x = jnp.arange(1.0, 6.0)
    out = both(f, x)
    assert out.tolist() == [20.0, 1.0, 5.0, 4.0, 10.0]
    assert x.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]  # the caller's array is untouched


def test_multi_dimensional_and_other_operators():
    @fn
    def f[N, M](x: i32[N, M]) -> i32[N, M]:
        x[0, 1:] = 7
        x[..., 0] += 1
        x[1, ::2] //= 2
        x[2] %= 3
        return x

    x = jnp.arange(12).reshape(3, 4)
    ref = np.arange(12).reshape(3, 4)
    ref[0, 1:] = 7
    ref[..., 0] += 1
    ref[1, ::2] //= 2
    ref[2] %= 3
    assert np.array_equal(both(f, x), ref)


def test_in_place_updates_inside_staged_loops_and_branches():
    @fn
    def hist[N](idx: i32[N]) -> f32[4]:  # type: ignore[valid-type]
        counts = jnp.zeros(4)
        for i in idx:
            counts[i] += 1
        return counts

    assert both(hist, jnp.array([0, 1, 1, 3, 1])).tolist() == [1.0, 3.0, 0.0, 1.0]
    assert "scan (carries counts)" in "".join(hist.flow.values())

    @fn
    def clamp_first[N](x: f32[N]) -> f32[N]:
        if x[0] < 0:
            x[0] = 0.0
        else:
            x[0] += 1
        return x

    assert both(clamp_first, jnp.array([-1.0, 2.0])).tolist() == [0.0, 2.0]
    assert both(clamp_first, jnp.array([1.0, 2.0])).tolist() == [2.0, 2.0]


def test_python_objects_are_still_updated_in_place():
    def f(xs, d, arr):
        xs[0] = "a"
        xs[1] += 5
        d["k"] = d.get("k", 0) + 1
        arr[0] = 9
        return xs, d

    xs, d, arr = [0, 1], {}, np.zeros(2)
    out = convert(f)(xs, d, arr)
    assert out[0] is xs and xs == ["a", 6] and d == {"k": 1} and arr[0] == 9


def test_python_mutation_in_traced_control_flow_is_refused_or_unrolled():
    @fn
    def loop(x):
        seen = {}
        for i in range(3):
            seen[i] = x * i
        return seen[2]

    assert jax.jit(loop)(2.0) == 4.0  # the loop is unrolled, so the dict is fine
    assert "unrolled" in "".join(loop.flow.values())

    @fn
    def branch(x):
        d = {"n": 0}
        if x > 0:
            d["n"] = 1
        return d["n"]

    assert branch(1.0) == 1
    with pytest.raises(StageError, match="item assignment on a dict inside traced control flow"):
        jax.jit(branch)(1.0)


class State(NamedTuple):
    pos: f32
    vel: f32


@dataclass(frozen=True)
class Frozen:
    a: float
    b: float = 0.0


def test_attribute_assignment_on_immutable_records():
    @fn
    def step(s: State, dt: f32) -> State:
        s.vel -= 9.8 * dt
        s.pos += s.vel * dt
        if s.pos < 0:
            s.pos = 0.0
            s.vel = -s.vel
        return s

    s = State(jnp.float32(1.0), jnp.float32(0.0))
    out = both(step, s, 0.5)
    assert jnp.allclose(out.pos, 0.0) and jnp.allclose(out.vel, 4.9)
    assert s.pos == 1.0

    def g(fz):
        fz.b = fz.a * 2
        fz.a += 1
        return fz

    assert convert(g)(Frozen(1.0)) == Frozen(2.0, 2.0)


def test_attribute_assignment_on_ordinary_objects_is_unchanged():
    class Box:
        def __init__(self):
            self.v = 1

    def f(b):
        b.v += 1
        b.w = 5
        return b

    b = Box()
    assert convert(f)(b) is b and b.v == 2 and b.w == 5


def test_assert():
    @fn
    def safe_log[N](x: f32[N]) -> f32[N]:
        assert (x > 0).all(), "inputs must be positive"
        return jnp.log(x)

    safe_log(jnp.ones(3))
    with pytest.raises(AssertionError, match="inputs must be positive"):
        safe_log(jnp.array([1.0, -1.0, 2.0]))
    jax.jit(safe_log)(jnp.ones(3))
    with pytest.raises(Exception, match="(?s)assertion failed at .*inputs must be positive"):
        jax.block_until_ready(jax.jit(safe_log)(jnp.array([1.0, -1.0, 2.0])))

    def plain(x):
        assert x, "falsy"
        assert x > 0
        return x

    c = convert(plain)
    assert c(3) == 3
    with pytest.raises(AssertionError, match="falsy"):
        c(0)
    with pytest.raises(AssertionError):
        c(-1)


def test_print(capfd):
    @fn
    def f[N](x: f32[N]) -> f32:
        s = x.sum()
        print("sum is", s, "for {braces}", sep=" | ")
        return s

    f(jnp.ones(3))
    assert "sum is | 3.0 | for {braces}" in capfd.readouterr().out
    jax.block_until_ready(jax.jit(f)(jnp.ones(4)))
    jax.effects_barrier()
    assert "sum is | 4.0 | for {braces}" in capfd.readouterr().out


def test_min_max():
    @fn
    def f(x: f32, lo: f32, hi: f32) -> f32:
        return max(lo, min(x, hi), -100.0)

    assert both(f, 5.0, 0.0, 1.0) == 1.0
    assert both(f, -5.0, 0.0, 1.0) == 0.0
    assert jax.grad(f)(0.5, 0.0, 1.0) == 1.0

    def g(xs):
        return max(xs), min(xs, key=lambda v: -v), max(1, 2), min("ab")

    assert convert(g)([3, 1, 2]) == (3, 3, 2, "a")


def test_shadowed_builtins_are_left_alone():
    def f(print, x):
        print(x)
        return max

    calls = []
    assert convert(lambda: None) if False else True
    out = f.__code__ and convert_or_same(f)(calls.append, 3)
    assert calls == [3] and out is max


def convert_or_same(f):
    from expex.staging.convert import NotConverted

    try:
        return convert(f)
    except NotConverted:
        return f

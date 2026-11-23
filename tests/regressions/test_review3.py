"""Regression tests for the third adversarial review (jumps inside loops)."""

import itertools
import time

import pytest

import jax
import jax.numpy as jnp

from expex import f32, fn, i32
from expex._convert import NotConverted, convert


def same(f, make_args):
    def run(g):
        try:
            return ("ok", g(*make_args()))
        except Exception as e:  # noqa: BLE001
            return ("raised", type(e).__name__)

    try:
        c = convert(f)
    except NotConverted:
        return
    assert run(c) == run(f), f.__name__


def test_loops_inside_try_or_with_keep_their_real_jumps():
    def first_even(it):
        try:
            for x in it:
                if x % 2 == 0:
                    break
        finally:
            pass
        return x, next(it, "exhausted")

    def in_loop(rows):
        out = []
        for row in rows:
            try:
                for x in row:
                    if x < 0:
                        break
                    out.append(x)
            finally:
                out.append("|")
        return out

    def endless():
        try:
            for i in itertools.count():
                if i == 3:
                    return i
        finally:
            pass

    same(first_even, lambda: (iter([1, 2, 3, 4]),))
    same(in_loop, lambda: ([[1, -1, 2], [3]],))
    same(endless, lambda: ())


def test_break_loops_see_variables_changed_by_closures():
    def count_to_two(n):
        x = 0

        def bump():
            nonlocal x
            x += 1

        for i in range(n):
            bump()
            if x == 2:
                break
        return x, i

    def in_if(n):
        x, r = 0, []

        def bump():
            nonlocal x
            x += 1

        if n > 0:
            bump()
            r.append(x)
        return r

    same(count_to_two, lambda: (4,))
    same(in_if, lambda: (1,))
    assert convert(count_to_two)(4) == (2, 1)


def test_return_in_a_loop_without_a_return_after_it():
    @fn
    def halve(x: f32) -> f32:
        while True:
            x = x / 2
            if x < 1:
                return x

    assert halve(37.0) == jax.jit(halve)(37.0) == 0.578125

    @fn
    def first_big[N](x: f32[N]) -> f32:
        for v in x:
            if v > 3:
                return v
        raise ValueError("nothing above 3")

    assert jax.jit(first_big)(jnp.array([1.0, 5.0])) == 5.0
    with pytest.raises(ValueError):
        first_big(jnp.array([1.0]))
    with pytest.raises(Exception, match="nothing above 3"):  # raised when the code runs
        jax.block_until_ready(jax.jit(first_big)(jnp.array([1.0])))
        jax.effects_barrier()


def test_return_behind_a_python_counter_in_a_traced_loop():
    @fn
    def capped(x: f32) -> f32:
        n = 0
        while x > 1:
            x = x / 2
            n += 1
            if n > 3:
                return -1.0
        return x

    assert capped(37.0) == jax.jit(capped)(37.0) == -1.0
    assert capped(6.0) == jax.jit(capped)(6.0) == 0.75

    @fn
    def assigned_on_break(x: f32) -> f32:
        n, y = 0, x
        while x > 1:
            x = x / 2
            n += 1
            if n > 3:
                y = x * 100
                break
        return y

    assert jnp.allclose(jax.jit(assigned_on_break)(64.0), assigned_on_break(64.0))


def test_return_in_a_loop_with_traced_trip_count():
    @fn
    def grow(x: f32, n: i32) -> f32:
        for i in range(n):
            x = x * 2
            if x > 100:
                return x
        return -x

    assert grow(1.0, 20) == jax.jit(grow)(1.0, 20) == 128.0
    assert grow(1.0, 3) == jax.jit(grow)(1.0, 3) == -8.0


def test_non_array_values_on_one_side_of_a_traced_jump():
    @fn
    def check[N](x: f32[N]):
        for v in x:
            if v > 3:
                return
        return

    @fn
    def label[N](x: f32[N]):
        for v in x:
            if v > 3:
                break
            name = "ok"
        return v, len(name)

    x = jnp.array([1.0, 2.0])
    assert check(x) is None and jax.jit(check)(x) is None
    v, n = jax.jit(label)(x)
    assert float(v) == 2.0 and n == 2


def test_many_conditional_jumps_convert_in_linear_time():
    body = "\n".join(f"        if a > {i}:\n            if b > {i}:\n                continue\n        s += {i}" for i in range(30))
    src = f"def f(xs, a, b):\n    s = 0\n    for x in xs:\n{body}\n    return s\n"
    import linecache

    name = "<blowup-test>"
    linecache.cache[name] = (len(src), None, src.splitlines(True), name)
    ns: dict = {}
    exec(compile(src, name, "exec"), ns)
    t = time.perf_counter()
    c = convert(ns["f"])
    assert time.perf_counter() - t < 5
    for args in ([1, 2], 50, 50), ([1, 2], 10, 3), ([1], 0, 0):
        assert c(*args) == ns["f"](*args)

    def returns(a, b):
        r = 0
        if a > 0:
            if b > 0:
                return "ab"
            r += 1
        if a > 1:
            if b > 1:
                return "AB"
            r += 10
        if a > 2:
            if b > 2:
                return "xy"
            r += 100
        return r

    for a in range(4):
        for b in range(4):
            same(returns, lambda a=a, b=b: (a, b))


def test_flow_keeps_the_entry_of_a_loop_containing_return():
    @fn
    def first_above[N](x: f32[N], limit: f32) -> f32:
        for v in x:
            if v > limit:
                return v
        return -1.0

    assert jax.jit(first_above)(jnp.arange(4.0), 2.0) == 3.0
    assert any(how.startswith("for -> lax.scan") for how in first_above.flow.values())

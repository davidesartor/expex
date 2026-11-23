"""break / continue / return inside loops, on Python values and on traced ones."""

import itertools

import pytest

import jax
import jax.numpy as jnp

import expex
from expex import f32, fn, i32
from expex._convert import convert


def same(f, *args):
    def run(g):
        try:
            return ("ok", g(*args))
        except Exception as e:  # noqa: BLE001
            return ("raised", type(e).__name__)

    assert run(convert(f)) == run(f), (f.__name__, args)


def agree(f, *args):
    a, b = f(*args), jax.jit(f)(*args)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.allclose(x, y), (a, b)
    return b


# ---- plain Python behaviour is unchanged -----------------------------------


def find_first(xs, target):
    for i, x in enumerate(xs):
        if x == target:
            return i
    return -1


def sum_until_negative(xs):
    total = 0
    for x in xs:
        if x < 0:
            break
        if x % 2:
            continue
        total += x
    return total, x


def nested(grid, target):
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == target:
                return r, c
            if v < 0:
                break
        else:
            continue
    return None


def while_true(n):
    steps = 0
    while True:
        if n == 1:
            break
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


def continue_in_while(n):
    i, out = 0, []
    while i < n:
        i += 1
        if i % 3 == 0:
            continue
        out.append(i)
    return out


def consumes_generator(limit):
    gen = itertools.count()
    for v in gen:
        if v >= limit:
            break
    return v, next(gen)  # the generator must not have been advanced past the break


def return_in_nested_ifs(xs):
    seen = 0
    for x in xs:
        seen += 1
        if x > 10:
            if x > 100:
                return "huge", seen
            return "big", seen
        elif x < 0:
            continue
        seen += 100
    return "none", seen


def break_then_code_after(xs):
    last = None
    for x in xs:
        if x is None:
            break
        last = x
    else_ran = False
    return last, x, else_ran


CASES = [
    (find_first, [([1, 2, 3], 3), ([1, 2, 3], 9), ([], 1)]),
    (sum_until_negative, [([2, 3, 4, -1, 8],), ([2, 4],), ([-1],), ([],)]),
    (nested, [([[1, 2], [3, 4]], 4), ([[1, -2, 4], [4]], 4), ([[1]], 9)]),
    (while_true, [(27,), (1,)]),
    (continue_in_while, [(10,), (0,)]),
    (consumes_generator, [(3,), (0,)]),
    (return_in_nested_ifs, [([1, -1, 50],), ([1, 500],), ([1, 2],), ([],)]),
    (break_then_code_after, [([1, 2, None, 3],), ([1],)]),
]


@pytest.mark.parametrize("func,arglists", CASES, ids=[c[0].__name__ for c in CASES])
def test_python_equivalence(func, arglists):
    for args in arglists:
        same(func, *args)


# ---- traced ---------------------------------------------------------------


def test_break_on_a_traced_condition():
    @fn
    def first_negative[N](x: f32[N]) -> tuple[i32, f32]:
        idx, total = -1, 0.0
        for i, v in enumerate(x):
            if v < 0:
                idx = i
                break
            total = total + v
        return idx, total

    idx, total = agree(first_negative, jnp.array([1.0, 2.0, -3.0, 4.0, -5.0]))
    assert int(idx) == 2 and float(total) == 3.0
    idx, total = agree(first_negative, jnp.array([1.0, 2.0]))
    assert int(idx) == -1 and float(total) == 3.0
    assert any("lax.scan" in h for h in first_negative.flow.values())
    out = jax.vmap(first_negative)(jnp.array([[1.0, -1.0, 5.0], [-2.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))
    assert out[0].tolist() == [1, 0, -1] and out[1].tolist() == [1.0, 0.0, 3.0]


def test_continue_on_a_traced_condition():
    @fn
    def sum_positive[N](x: f32[N]) -> f32:
        total = 0.0
        for v in x:
            if v <= 0:
                continue
            total = total + jnp.sqrt(v)
        return total

    x = jnp.array([4.0, -1.0, 9.0, 0.0])
    assert agree(sum_positive, x) == 5.0
    g = jax.jit(jax.grad(sum_positive))(x)
    assert jnp.allclose(g, jnp.array([0.25, 0.0, 1 / 6, 0.0]))  # and no NaN from the skipped sqrt


def test_return_from_a_traced_loop():
    @fn
    def first_above[N](x: f32[N], limit: f32) -> f32:
        for v in x:
            if v > limit:
                return v * 10
        return -1.0

    x = jnp.array([1.0, 5.0, 2.0, 7.0])
    assert agree(first_above, x, 3.0) == 50.0
    assert agree(first_above, x, 6.0) == 70.0
    assert agree(first_above, x, 9.0) == -1.0
    assert jax.vmap(first_above, (None, 0))(x, jnp.array([0.0, 3.0, 9.0])).tolist() == [10.0, 50.0, -1.0]


def test_while_true_with_traced_break():
    @fn
    def collatz(n: i32) -> i32:
        steps = 0
        while True:
            if n == 1:
                break
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        return steps

    assert agree(collatz, 27) == 111
    assert any("lax.while_loop" in h for h in collatz.flow.values())
    assert jax.vmap(collatz)(jnp.array([27, 6, 1])).tolist() == [111, 8, 0]


def test_break_in_while_and_nested_loops():
    @fn
    def bisect(lo: f32, hi: f32) -> f32:
        it = 0
        while hi - lo > 1e-4:
            mid = (lo + hi) / 2
            if mid * mid == 2.0:
                lo = mid
                break
            if mid * mid < 2.0:
                lo = mid
            else:
                hi = mid
            it += 1
            if it > 100:
                break
        return lo

    assert jnp.allclose(agree(bisect, 0.0, 2.0), 2**0.5, atol=1e-3)

    @fn
    def search[R, C](grid: f32[R, C], target: f32) -> tuple[i32, i32]:
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v == target:
                    return r, c
        return -1, -1

    grid = jnp.arange(12.0).reshape(3, 4)
    r, c = agree(search, grid, 6.0)
    assert (int(r), int(c)) == (1, 2)
    r, c = agree(search, grid, 99.0)
    assert (int(r), int(c)) == (-1, -1)


def test_unsupported_jumps_keep_plain_python_semantics():
    @fn
    def in_try(xs):
        out = 0
        for x in xs:
            try:
                if x < 0:
                    break
            finally:
                out += 1
        return out

    assert in_try([1, -1, 5]) == 2

    @fn
    def with_else(xs):
        for x in xs:
            if x < 0:
                break
        else:
            return "clean"
        return "broke"

    assert with_else([1, 2]) == "clean" and with_else([1, -2]) == "broke"


def test_prover_handles_loops_with_jumps():
    @fn
    def clipped_sum[N](x: f32[N]) -> f32:
        total = 0.0
        for v in x:
            if total > 10:
                break
            total = total + v
        return total

    assert expex.prove(clipped_sum).status == "proved"

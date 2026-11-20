"""Regression tests for the second adversarial review (newer features)."""

import io
import sys
from dataclasses import dataclass

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import typax as tx
from typax import StageError, Static, f32, fn
from typax._convert import convert


def scoped(f, key, *args):
    with tx.rng(key):
        return f(*args)


def test_keys_agree_in_ternaries_and_boolean_operators():
    @fn(random=True)
    def f(x):
        a = jax.random.uniform(tx.key()) if x > 0 else -jax.random.uniform(tx.key())
        b = (x > 100) or (jax.random.uniform(tx.key()) > 2)
        return a, jax.random.uniform(tx.key()), b

    k = jax.random.key(7)
    eager = scoped(f, k, jnp.float32(1.0))
    staged = jax.jit(lambda k, x: scoped(f, k, x))(k, jnp.float32(1.0))
    assert jnp.allclose(eager[0], staged[0]) and jnp.allclose(eager[1], staged[1])


def test_while_test_and_body_draw_different_numbers():
    seen = []

    def draw():
        v = float(jax.random.uniform(tx.key()))
        seen.append(v)
        return v

    @fn(random=True)
    def f():
        n = 0
        while draw() < 2.0 and n < 3:
            draw()
            n += 1
        return n

    tx.seed(0)
    f()
    assert len(seen) == 7 and len(set(seen)) == 7


@fn(random=True)
def noise(x: f32) -> f32:
    return x + jax.random.uniform(tx.key())


def test_vmap_with_a_shared_key_is_refused():
    with tx.rng(0):
        assert len(set(noise.lift(jnp.zeros(4)).tolist())) == 4
        with pytest.raises(StageError, match="shared by the whole batch"):
            jax.vmap(noise)(jnp.zeros(4))


def test_augmented_assignment_reads_the_target_first():
    def bump(z):
        z[0] = 10
        return 1

    def f(a):
        a[0] += bump(a)
        return a

    class C:
        x = 1

    def g(c):
        c.x += [setattr(c, "x", 10), 1][1]
        return c.x

    log = []

    def tick(v):
        log.append(v)
        return 0 if v == "i" else 5

    def h(a):
        a[tick("i")] += tick("v")
        return a

    assert convert(f)([1]) == f([1]) == [2]
    assert convert(g)(C()) == g(C()) == 2
    assert convert(h)([1]) == [6] and log == ["i", "v"]


def test_static_values_that_compare_equal_get_their_own_compilation():
    @fn(jit=True)
    def pick(x: f32[2], idx: Static[tuple]):  # type: ignore[valid-type]
        return x[np.asarray(idx)]

    x = jnp.array([10.0, 20.0])
    assert pick(x, (1, 0)).tolist() == [20.0, 10.0]
    assert pick(x, (True, False)).tolist() == [10.0]
    assert len(pick.traces) == 2

    @fn(jit=True)
    def signed(x: f32, s: Static[float]) -> f32:
        return x + np.copysign(1.0, s)

    assert signed(0.0, 0.0) == 1.0 and signed(0.0, -0.0) == -1.0


def test_max_min_agree_between_eager_and_jit():
    @fn
    def f(n, x, u):
        return max(n, x) // 2, max(x, 0), max(u, 100) + 100, max(x, float("nan"))

    args = (jnp.int32(7), jnp.float32(2.5), jnp.uint8(200))
    for a, b in zip(f(*args), jax.jit(f)(*args)):
        assert jnp.asarray(a).dtype == jnp.asarray(b).dtype
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def test_traced_print_keeps_braces_and_keyword_arguments(capfd):
    @fn
    def p(x):
        print("set {x}:", x)
        print("cfg", {"a": 1}, x, sep="|")
        print("err", x, file=sys.stderr)
        print("no newline", x, end="!")
        return x

    jax.block_until_ready(jax.jit(p)(jnp.float32(1.5)))
    jax.effects_barrier()
    out, err = capfd.readouterr()
    assert "set {x}: 1.5" in out and "cfg|{'a': 1}|1.5" in out
    assert "no newline 1.5!" in out and "err 1.5" in err and "err" not in out


def test_assert_only_fires_where_the_code_is_reached():
    @fn
    def f(x):
        if x > 0:
            assert x > 0, "must be positive"
            return jnp.sqrt(x)
        return 0.0 * x

    assert jnp.allclose(jax.vmap(f)(jnp.array([4.0, -1.0])), jnp.array([2.0, 0.0]))
    jax.effects_barrier()

    @fn
    def g(x):
        while x < 10:
            assert x < 10
            x = x * 2
        return x

    assert jax.vmap(g)(jnp.array([1.0, 9.0])).tolist() == [16.0, 18.0]
    jax.effects_barrier()

    @fn
    def strict(x):
        if x > 0:
            assert x < 3, "too big"
        return x

    with pytest.raises(Exception, match="too big"):
        jax.block_until_ready(jax.vmap(strict)(jnp.array([1.0, 5.0])))
        jax.effects_barrier()


def test_local_scratch_objects_in_a_callee_are_fine_under_traced_control_flow():
    @fn
    def norm2(x):
        buf = [0.0, 0.0]
        buf[0] = x * x
        return buf[0] + buf[1]

    @fn
    def f(x):
        if x > 0:
            y = norm2(x)
        else:
            y = x
        return y

    assert jax.jit(f)(jnp.float32(2.0)) == 4.0


@dataclass
class Layer:
    w: f32[2]  # type: ignore[valid-type]
    k: Static[int]


def test_jit_takes_dataclass_arguments():
    @fn(jit=True)
    def apply(d: Layer, x: f32) -> f32[2]:  # type: ignore[valid-type]
        return x * d.w * d.k

    assert apply(Layer(jnp.ones(2), 2), 1.0).tolist() == [2.0, 2.0]
    assert apply(Layer(jnp.ones(2) * 3, 2), 1.0).tolist() == [6.0, 6.0]
    assert len(apply.traces) == 1  # new array values do not retrace
    assert apply(Layer(jnp.ones(2), 5), 1.0).tolist() == [5.0, 5.0]
    assert len(apply.traces) == 2


def test_reversed_range_with_traced_length():
    @fn
    def f(xs, n):
        acc = 0.0
        for i in reversed(range(n)):
            acc = acc * 2 + xs[i]
        return acc

    xs = jnp.arange(5.0)
    assert f(xs, 3) == jax.jit(f)(xs, 3) == 10.0


def test_a_variable_named_slice():
    def f(x, slice):
        x[1:3] = [slice]
        return x

    assert convert(f)([0, 1, 2, 3], 7) == [0, 7, 3]


def test_lift_maps_optional_arguments():
    @fn(lift=True)
    def f(x: f32, opt: f32 | None = None) -> f32:
        return x + (0 if opt is None else opt)

    assert f(jnp.arange(3.0)).tolist() == [0.0, 1.0, 2.0]
    assert f(jnp.arange(3.0), 1.0).tolist() == [1.0, 2.0, 3.0]
    assert f(jnp.arange(3.0), jnp.ones(3)).tolist() == [1.0, 2.0, 3.0]


def test_and_or_operands_are_evaluated_inside_their_branch():
    @fn
    def f(x):
        return (x > 0) and jnp.sqrt(x)

    g = jax.grad(lambda v: jax.vmap(f)(v).sum())(jnp.array([4.0, 0.0, -1.0]))
    assert g.tolist() == [0.25, 0.0, 0.0]


def test_minor_findings():
    @fn(jit=True)
    def f(x: f32, n: Static[int] | None = None) -> f32:
        return x if n is None else x * n

    assert f(2.0) == 2.0 and f(2.0, 3) == 6.0

    @fn
    def masked(x):
        x[x > 2] = 0.0
        x[x < 1] += 10
        return x

    x = jnp.array([0.0, 1.5, 3.0])
    assert masked(x).tolist() == jax.jit(masked)(x).tolist() == [10.0, 1.5, 10.0]

    @fn
    def dup(x, idx):
        x[idx] += 1
        return x

    ref = np.zeros(3)
    ref[np.array([0, 0, 1])] += 1  # NumPy: each duplicate is assigned once
    assert dup(jnp.zeros(3), jnp.array([0, 0, 1])).tolist() == ref.tolist()

    @fn
    def chunked(xs):
        out = []
        for i, x in enumerate(reversed(tx.rows(xs, chunk=2))):
            out.append(x * i)
        return jnp.asarray(out)

    assert jax.jit(chunked)(jnp.arange(4.0)).tolist() == [0.0, 2.0, 2.0, 0.0]
    assert "lax.map in chunks of 2" in "".join(chunked.flow.values())

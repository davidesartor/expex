"""Regression tests for bugs found by an adversarial review of the conversion."""

import functools

import pytest

import jax
import jax.numpy as jnp

from typax import fn
from typax._convert import NotConverted, convert


def same(f, *args):
    """Converted and original agree on plain Python values (result or exception type)."""

    def run(g):
        try:
            return ("ok", g(*args))
        except Exception as e:  # noqa: BLE001
            return ("raised", type(e).__name__)

    try:
        c = convert(f)
    except NotConverted:
        return
    assert run(c) == run(f)


# ---- A: Python equivalence -------------------------------------------------


def test_assignments_before_a_caught_exception_survive():
    def loop(xs):
        total = 0
        try:
            for x in xs:
                total += x
        except TypeError:
            pass
        return total

    def branch(flag):
        a = 0
        try:
            if flag:
                a = 1
                raise ValueError
        except ValueError:
            pass
        return a

    same(loop, [1, 2, "a"])
    same(branch, True)


def a2_shared(n):
    x = 0
    get = lambda: x  # noqa: E731
    out = []
    for i in range(n):
        x = i
        out.append(get())
    return out


def a2_late(n):
    fs = []
    for i in range(n):
        fs.append(lambda: i)
    return [f() for f in fs]


def a2_nonlocal(flag):
    n = 0
    if flag:
        n = 1

        def inc():
            nonlocal n
            n += 1

    inc()
    return n


def test_closures_keep_sharing_variables():
    same(a2_shared, 3)
    same(a2_late, 3)
    same(a2_nonlocal, True)
    assert convert(a2_shared)(3) == [0, 1, 2]


def test_walrus_in_boolean_and_ternary():
    def f(v):
        if (y := v) and y > 3:
            return y
        return -y

    def g(v):
        r = 1 if v else (z := 7)
        return r

    same(f, 5)
    same(f, 1)
    same(g, 0)


def test_match_captures():
    def f(vs):
        for v in vs:
            match v:
                case [k, *rest]:
                    pass
        return k, rest

    same(f, [[1, 2, 3]])


def test_unbound_variables_raise_as_in_python():
    def f(flag):
        if flag:
            seen = [x]  # noqa: F821 - x is not bound yet
            x = 1
        return seen

    def g(vs):
        for v in vs:
            try:
                acc = acc + v
            except NameError:
                acc = v
        return acc

    def h(flag):
        if flag:
            y = 1
        try:
            return y
        except UnboundLocalError:
            return "unbound"

    same(f, True)
    same(g, [1, 2, 3])
    same(h, False)
    same(h, True)


def test_wrapped_functions_are_not_replaced_by_their_inner_function():
    def doubled(f):
        @functools.wraps(f)
        def w(*a):
            return 2 * f(*a)

        return w

    @fn(checked=False)
    @doubled
    def g(x):
        if x > 0:
            return x
        return -x

    assert g(jnp.float32(3.0)) == 6.0
    assert not g.converted and "wraps another function" in g.convert_note


def test_shadowed_builtins_from_an_enclosing_scope():
    def outer():
        range = lambda n: [7]  # noqa: A001, E731

        def f(n):
            out = []
            for i in range(n):
                out.append(i)
            return out

        return f

    same(outer(), 3)


def test_enumerate_keyword():
    def f(xs):
        out = []
        for i, x in enumerate(iterable=xs, start=1):
            out.append(i * x)
        return out

    same(f, [1, 2, 3])


def test_chained_comparison_evaluates_operands_once():
    calls = []

    def mid():
        calls.append(1)
        return 5

    def f(a, b):
        if a < mid() < b:
            return "in"
        return "out"

    c = convert(f)
    assert c(0, 10) == "in" and c(7, 10) == "out" and c(0, 3) == "out"
    assert len(calls) == 3


# ---- B: trace equivalence --------------------------------------------------


def agree(f, *args):
    a, b = f(*args), jax.jit(f)(*args)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.allclose(x, y), (a, b)
        assert jnp.asarray(x).dtype == jnp.asarray(y).dtype, (x, y)
    return b


def test_or_and_return_operands_not_booleans():
    @fn(checked=False)
    def f(x, d):
        y = x or d
        z = x and d
        return y * 2.0 + z

    assert agree(f, jnp.float32(3.0), jnp.float32(7.0)) == 13.0
    assert agree(f, jnp.float32(0.0), jnp.float32(7.0)) == 14.0
    assert jax.grad(f)(1.5, 2.0) == 2.0
    assert jax.vmap(f)(jnp.array([3.0, 0.0]), jnp.array([7.0, 7.0])).tolist() == [13.0, 14.0]


def test_variable_read_in_loop_else():
    @fn(checked=False)
    def f(x):
        t = jnp.float32(0.0)
        for a in (1, 2):
            for v in x:
                t = v * a
        else:
            r = t
        return r

    assert agree(f, jnp.arange(1.0, 6.0)) == 10.0


def test_in_place_container_updates_in_loops():
    @fn(checked=False)
    def mean(x):
        stats = {"n": 0}
        tot = jnp.float32(0.0)
        for v in x:
            stats["n"] = stats["n"] + 1
            tot = tot + v
        return tot / stats["n"]

    @fn(checked=False)
    def ema(x):
        st = {"h": jnp.float32(0.0)}
        for v in x:
            st["h"] = st["h"] * 0.5 + v
        return st["h"]

    X = jnp.arange(1.0, 6.0)
    assert agree(mean, X) == 3.0
    agree(ema, X)

    @fn(checked=False)
    def cond_mutation(x):
        d = {"n": 0}
        if x > 0:
            d["n"] += 1
        return d["n"]

    assert cond_mutation(1.0) == 1
    with pytest.raises(jax.errors.TracerBoolConversionError):  # refused, not silently wrong
        jax.jit(cond_mutation)(1.0)


def test_reusing_names_across_traced_loops():
    @fn(checked=False)
    def f(n, x):
        acc = 0.0
        for i in range(n):
            acc = acc + x[i]
        for i in range(n):
            acc = acc * x[i]
        return acc

    assert agree(f, jnp.int32(3), jnp.arange(1.0, 6.0)) == 36.0

    @fn(checked=False)
    def g(x):
        while x > 10:
            t = x / 2
            x = t
        while x < 1:
            t = x * 2
            x = t
        return x

    assert agree(g, jnp.float32(100.0)) == 6.25
    assert agree(g, jnp.float32(0.1)) == 1.6


def test_collected_lists_can_be_returned_and_combined():
    @fn(checked=False)
    def f(x):
        ys = []
        for v in x:
            ys.append(v * 2)
        return ys

    out = jax.jit(f)(jnp.arange(3.0))
    assert isinstance(out, list) and [float(v) for v in out] == [0.0, 2.0, 4.0]

    @fn(checked=False)
    def g(x):
        ys = []
        for v in x:
            ys.append(v * 2)
        both = ys + [x[0]]
        return {"all": ys, "n": len(both)}, jnp.stack(both)

    (d, stacked) = jax.jit(g)(jnp.arange(3.0))
    assert isinstance(d["all"], list) and int(d["n"]) == 4 and stacked.shape == (4,)


def test_dtypes_match_between_eager_and_jit():
    @fn(checked=False)
    def idx(x):
        ys = []
        for i in range(3):
            ys.append(x * i)
        return jnp.stack(ys)

    @fn(checked=False)
    def enum(x):
        t = x[0] * 0
        for i, v in enumerate(x):
            t = t + v * i
        return t

    @fn(checked=False)
    def branch(c, x):
        if c:
            k = 1
        else:
            k = 2.5
        return x * k

    for dt in (jnp.int8, jnp.float16, jnp.float32):
        agree(idx, jnp.ones(2, dt))
        agree(enum, jnp.ones(3, dt))
    agree(branch, jnp.bool_(True), jnp.ones(2, jnp.float16))


def test_jit_only_crashes():
    @fn(checked=False)
    def chained(x):
        if 0.0 < x < 10.0:
            return x
        return -x

    assert agree(chained, jnp.float32(5.0)) == 5.0
    assert agree(chained, jnp.float32(50.0)) == -50.0

    @fn(checked=False)
    def countdown(n):
        s = 0
        while n:
            s = s + n
            n = n - 1
        return s

    assert agree(countdown, jnp.int32(4)) == 10

    @fn(checked=False)
    def pair(x):
        if x > 0:
            return x, 1
        return x, 2.5

    a, b = jax.jit(pair)(jnp.float32(-1.0))
    assert a == -1.0 and b == 2.5

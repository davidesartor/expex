"""Converted control flow: plain if / while / for that also work when traced."""

import pytest

import jax
import jax.numpy as jnp

import typax as tx
from typax import StageError, Static, f32, fn, i32


def both(f, *args, **kw):
    """Run eagerly and under jit; check they agree; return the jitted result."""
    a = f(*args, **kw)
    b = jax.jit(f)(*args, **kw)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert jnp.allclose(x, y), (a, b)
    return b


def n_eqns(f, *args):
    return len(jax.make_jaxpr(f)(*args).jaxpr.eqns)


# ------------------------------------------------------------------------ if


def test_if_else_assignment():
    @fn
    def clip(x: f32, lo: f32, hi: f32) -> f32:
        if x < lo:
            y = lo
        elif x > hi:
            y = hi
        else:
            y = x
        return y

    assert both(clip, 5.0, 0.0, 1.0) == 1.0
    assert both(clip, -5.0, 0.0, 1.0) == 0.0
    assert both(clip, 0.5, 0.0, 1.0) == 0.5
    assert clip.converted and "lax.cond" in "".join(clip.flow.values())


def test_early_return():
    @fn
    def f[B](x: f32[B]) -> f32[B]:
        if x.sum() > 0:
            return x
        if x.sum() < -10:
            return x * 0
        return -x

    assert both(f, jnp.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert both(f, jnp.array([-1.0, -2.0])).tolist() == [1.0, 2.0]
    assert both(f, jnp.array([-10.0, -2.0])).tolist() == [0.0, 0.0]


def test_if_without_else_and_augmented_assignment():
    @fn
    def f(x: f32) -> f32:
        y = x
        if x > 0:
            y += 10
            t = y * 2  # a temporary that is dead after the if
            y = t / 2
        return y

    assert both(f, 1.0) == 11.0
    assert both(f, -1.0) == -1.0


def test_python_conditions_are_untouched():
    calls = []

    @fn
    def f[B](x: f32[B], flag: Static[bool] = True, mode: Static[str] = "a") -> f32[B]:
        if flag and mode == "a":
            calls.append("a")
            return x * 2
        return x

    assert both(f, jnp.ones(2)).tolist() == [2.0, 2.0]
    assert jax.jit(lambda x: f(x, False))(jnp.ones(2)).tolist() == [1.0, 1.0]
    assert f.flow == {}  # nothing was staged: the conditions were Python values


def test_shape_dependent_branch_is_python():
    @fn
    def f[B](x: f32[B]) -> f32:
        if x.shape[0] > 2:
            return x[:2].sum()
        return x.sum()

    assert both(f, jnp.ones(5)) == 2.0
    assert f.flow == {}


def test_boolean_operators_and_ternary():
    @fn
    def f(x: f32, y: f32) -> f32:
        z = x if x > 0 and not y > 0 else y
        if x > 100 or y > 100:
            z = z + 1000
        return z

    assert both(f, 1.0, -2.0) == 1.0
    assert both(f, 1.0, 2.0) == 2.0
    assert both(f, 1.0, 200.0) == 1200.0


def test_weak_python_scalars_promote_between_branches():
    @fn
    def f(x: f32) -> f32:
        if x > 0:
            y = 1
        else:
            y = x * 2.5
        return y + 0.0

    assert both(f, 3.0) == 1.0
    assert both(f, -2.0) == -5.0


def test_branch_shape_mismatch_names_the_variable():
    @fn
    def f[B](x: f32[B]):
        if x.sum() > 0:
            y = x
        else:
            y = x[:-1]
        return y

    with pytest.raises(StageError, match="`y` has shape .* in one branch"):
        jax.jit(f)(jnp.ones(3))


def test_non_array_values_in_branches():
    @fn
    def same(x: f32) -> f32:
        if x > 0:
            mode = "a"
            y = x
        else:
            mode = "a"
            y = -x
        return y if mode == "a" else x

    assert both(same, -2.0) == 2.0

    @fn
    def differ(x: f32):
        if x > 0:
            mode = "a"
        else:
            mode = "b"
        return mode

    assert differ(1.0) == "a"
    with pytest.raises(StageError, match="`mode` is set to different non-array values"):
        jax.jit(differ)(1.0)


def test_non_scalar_condition():
    @fn
    def f[B](x: f32[B]) -> f32[B]:
        if x > 0:
            return x
        return -x

    with pytest.raises(StageError, match=r"(?s)traced array of shape \(3,\).*jnp.where"):
        jax.jit(f)(jnp.ones(3))


def test_append_in_traced_if_is_refused():
    @fn
    def f(x: f32):
        out = []
        if x > 0:
            out.append(x)
        return out

    assert len(f(1.0)) == 1
    with pytest.raises(StageError, match=r"out.append\(\.\.\.\)` runs inside the traced `if`"):
        jax.jit(f)(1.0)


def test_grad_and_vmap_through_converted_if():
    @fn
    def f(x: f32) -> f32:
        if x > 0:
            return x * x
        return -x

    assert jax.grad(f)(3.0) == 6.0
    assert jax.grad(f)(-3.0) == -1.0
    assert jax.vmap(f)(jnp.array([3.0, -3.0])).tolist() == [9.0, 3.0]


# --------------------------------------------------------------------- while


def test_while():
    @fn
    def collatz(n: i32) -> i32:
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        return steps

    assert both(collatz, 27) == 111
    assert "lax.while_loop" in "".join(collatz.flow.values())
    assert jax.vmap(collatz)(jnp.array([27, 6, 1])).tolist() == [111, 8, 0]


def test_while_with_array_state_and_temporaries():
    @fn
    def newton(a: f32) -> f32:
        x = a
        err = 1.0
        while err > 1e-6:
            nxt = 0.5 * (x + a / x)
            err = abs(nxt - x)
            x = nxt
        return x

    assert jnp.allclose(both(newton, 9.0), 3.0)


def test_while_carry_type_error_names_variables():
    @fn
    def f(n: i32):
        x = jnp.zeros(1)
        while n > 0:
            x = jnp.concatenate([x, x])
            n -= 1
        return x

    assert f(2).shape == (4,)
    with pytest.raises(TypeError) as e:
        jax.jit(f)(2)
    assert any("carried variables, in order: n, x" in n for n in e.value.__notes__)


def test_loops_with_break_are_staged():
    @fn
    def f[T](x: f32[T]) -> f32:
        total = 0.0
        for i in range(x.shape[0]):
            if i >= 2:
                break
            total = total + x[i]
        return total

    assert both(f, jnp.arange(5.0)) == 1.0
    assert any("for -> lax.scan" in how for how in f.flow.values())


# ----------------------------------------------------------------------- for


def test_for_over_array_becomes_scan():
    @fn
    def cumsum[T](x: f32[T]) -> f32[T]:
        total = 0.0
        out = []
        for v in x:
            total = total + v
            out.append(total)
        return jnp.asarray(out)

    x = jnp.arange(50.0)
    assert jnp.allclose(both(cumsum, x), jnp.cumsum(x))
    assert "lax.scan (carries total)" in "".join(cumsum.flow.values())
    # the graph does not grow with the loop length
    assert n_eqns(cumsum, x) == n_eqns(cumsum, jnp.arange(5.0)) < 10


def test_independent_iterations_become_vmap():
    @fn
    def squares[T](x: f32[T]) -> f32[T]:
        out = []
        for v in x:
            t = v * v
            out.append(t + 1)
        return jnp.stack(out)

    assert both(squares, jnp.arange(4.0)).tolist() == [1.0, 2.0, 5.0, 10.0]
    assert "vmap" in "".join(squares.flow.values())


def test_integer_accumulator_promotes():
    @fn
    def total[T](x: f32[T]) -> f32:
        s = 0
        for v in x:
            s = s + v
        return s

    assert both(total, jnp.arange(5.0)) == 10.0
    assert "scan" in "".join(total.flow.values())


def test_range_enumerate_zip():
    @fn
    def f[T](x: f32[T], y: f32[T]) -> f32:
        s = jnp.float32(0.0)
        for i in range(x.shape[0]):
            s = s + x[i] * i
        for i, v in enumerate(y):
            s = s + v * i
        for a, b in zip(x, y):
            s = s + a * b
        for j, (a, b) in enumerate(zip(x, y), 1):
            s = s + j * (a - b)
        return s

    x, y = jnp.arange(4.0), jnp.arange(4.0) + 1
    ref = fn(convert=False)(f.raw)(x, y)
    assert jnp.allclose(both(f, x, y), ref)
    assert sum("scan" in v for v in f.flow.values()) == 4


def test_traced_trip_count():
    @fn
    def power(x: f32, n: i32) -> f32:
        r = 1.0
        for _ in range(n):
            r = r * x
        return r

    assert both(power, 2.0, 5) == 32.0
    assert both(power, 2.0, 0) == 1.0
    assert "traced number of iterations" in "".join(power.flow.values())

    @fn
    def tri(n: i32) -> i32:
        s = 0
        for i in range(1, n + 1, 2):
            s += i
        return s

    assert both(tri, 8) == 16


def test_loop_variable_and_last_value_after_loop():
    @fn
    def f[T](x: f32[T]) -> f32:
        for v in x:
            last = v * 2
        return last + v

    assert both(f, jnp.arange(4.0)) == 9.0


def test_conditional_update_is_carried():
    @fn
    def argmax_val[T](x: f32[T]) -> f32:
        best = x[0]
        for v in x:
            if v > best:
                best = v
        return best

    assert both(argmax_val, jnp.array([1.0, 5.0, 3.0])) == 5.0
    assert "scan (carries best)" in "".join(argmax_val.flow.values())


def test_nested_loops():
    @fn
    def f[N, M](x: f32[N, M]) -> f32:
        total = 0.0
        for row in x:
            acc = 0.0
            for v in row:
                acc = acc * 0.5 + v
            total = total + acc
        return total

    x = jnp.arange(12.0).reshape(3, 4)
    assert jnp.allclose(both(f, x), fn(convert=False)(f.raw)(x))
    assert n_eqns(f, x) < 12


def test_python_iterables_unroll():
    @fn
    def f[D](x: f32[D], ws: list[f32[D]]) -> f32[D]:
        for w in ws:
            x = x * w
        return x

    assert both(f, jnp.ones(2), [jnp.ones(2) * 2, jnp.ones(2) * 3]).tolist() == [6.0, 6.0]
    assert "unrolled (iterating over a list)" in "".join(f.flow.values())


def test_falls_back_when_index_must_be_a_python_int():
    layers = [lambda x: x + 1, lambda x: x * 2, lambda x: x - 3]

    @fn
    def f(x: f32) -> f32:
        for i in range(3):
            x = layers[i](x)
        return x

    assert both(f, 1.0) == 1.0
    assert "unrolled" in "".join(f.flow.values())


def test_falls_back_when_shape_changes():
    @fn
    def f(x: f32):
        acc = jnp.zeros(1)
        for i in range(3):
            acc = jnp.concatenate([acc, acc])
        return acc

    assert both(f, 1.0).shape == (8,)
    assert "unrolled" in "".join(f.flow.values())


def test_multiple_appends_and_tuple_items():
    @fn
    def f[T](x: f32[T]):
        pairs, flat = [], []
        for v in x:
            pairs.append((v, v * 2))
            flat.append(v)
            flat.append(-v)
        a, b = jnp.asarray(pairs[0]), jnp.asarray(flat)
        return a, b, len(flat)

    a, b, n = jax.jit(f)(jnp.arange(3.0))
    assert n == 6
    assert b.tolist() == [0.0, -0.0, 1.0, -1.0, 2.0, -2.0]
    ea, eb, en = f(jnp.arange(3.0))
    assert en == 6 and jnp.allclose(eb, b)


def test_append_after_lowered_loop():
    @fn
    def f[T](x: f32[T]):
        out = []
        for v in x:
            out.append(v)
        out.append(jnp.float32(99.0))
        return jnp.stack(out)

    assert both(f, jnp.arange(3.0)).tolist() == [0.0, 1.0, 2.0, 99.0]


def test_rnn_matches_unconverted_and_stays_small():
    @fn
    def rnn[T, D](xs: f32[T, D], w: f32[D, D]) -> tuple[f32[D], f32[T, D]]:
        h = jnp.zeros(xs.shape[1])
        hs = []
        for i, x in enumerate(xs):
            h = jnp.tanh(w @ h + x)
            if i % 2 == 0:
                h = h * 0.5
            hs.append(h)
        return h, jnp.asarray(hs)

    xs, w = jnp.ones((20, 3)), jnp.eye(3) * 0.5
    plain = fn(convert=False)(rnn.raw)
    ref = plain(xs, w)
    out = both(rnn, xs, w)
    assert jnp.allclose(out[0], ref[0]) and jnp.allclose(out[1], ref[1])
    assert n_eqns(rnn, xs, w) * 10 < n_eqns(plain, xs, w)
    g = jax.grad(lambda w: rnn(xs, w)[0].sum())(w)
    g_ref = jax.grad(lambda w: plain(xs, w)[0].sum())(w)
    assert jnp.allclose(g, g_ref, atol=1e-6)
    assert tx.prove(rnn).status == "proved"


# ---------------------------------------------------------------- conversion


def test_closures_recursion_defaults_and_methods():
    scale = 3.0

    @fn
    def fact(n: Static[int]) -> int:
        if n <= 1:
            return 1
        return n * fact(n - 1)

    assert fact(5) == 120

    @fn
    def f(x: f32, k: Static[float] = 2.0, *, bias: Static[float] = 1.0) -> f32:
        if x > 0:
            return x * scale * k + bias
        return x

    assert both(f, 1.0) == 7.0
    assert jax.jit(lambda x: f(x, 1.0, bias=0.0))(1.0) == 3.0

    class M:
        def __init__(self):
            self.t = 0.5

        @fn
        def call(self, x: f32) -> f32:
            if x > self.t:
                return x
            return jnp.float32(self.t)

    m = M()
    assert m.call(1.0) == 1.0 and jax.jit(m.call)(0.0) == 0.5


def test_late_binding_closure_is_preserved():
    def make():
        @fn
        def f(x: f32) -> f32:
            if x > 0:
                return x * k
            return x

        k = 5.0  # bound after the function is defined
        return f

    assert make()(2.0) == 10.0


def test_conversion_can_be_disabled_or_skipped():
    @fn(convert=False)
    def f(x: f32) -> f32:
        if x > 0:
            return x
        return -x

    assert not f.converted and f.convert_note == "convert=False"
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(f)(1.0)

    @fn
    def g(x: f32) -> f32:
        return x * 2

    assert not g.converted and g.convert_note == "no control flow to convert"
    assert not fn(lambda x: x if x else 0).converted


def test_tracebacks_point_at_original_lines():
    @fn
    def f(x: f32) -> f32:
        if x > 0:
            y = undefined_name + x  # noqa: F821
        else:
            y = x
        return y

    with pytest.raises(NameError) as e:
        jax.jit(f)(1.0)
    tb = e.traceback[-1]
    assert "undefined_name" in str(tb.statement)


def test_converted_source_is_inspectable():
    @fn
    def f(x: f32) -> f32:
        if x > 0:
            return x
        return -x

    assert "__typax_flow__.if_(" in f.converted_source


def test_loop_lowering_can_be_switched_off(monkeypatch):
    @fn
    def total[T](x: f32[T]) -> f32:
        s = 0.0
        for v in x:
            s = s + v
        return s

    monkeypatch.setattr(tx.config, "lower_loops", False)
    assert jax.jit(total)(jnp.arange(4.0)) == 6.0
    assert "unrolled (config.lower_loops is off)" in "".join(total.flow.values())
    assert n_eqns(total, jnp.arange(4.0)) > 4


def test_rows_stack_and_reversed():
    @fn
    def returns[T](reward: f32[T], done: f32[T], gamma: Static[float] = 0.9) -> f32[T]:
        g = 0.0
        out = []
        for step in reversed(tx.rows({"r": reward, "d": done})):
            g = step["r"] + gamma * g * (1 - step["d"])
            out.append({"g": g})
        return tx.stack(out)["g"][::-1]

    r = jnp.array([1.0, 2.0, 3.0, 4.0])
    d = jnp.array([0.0, 1.0, 0.0, 0.0])
    ref = fn(convert=False)(returns.raw)(r, d)
    assert jnp.allclose(both(returns, r, d), ref)
    assert "scan (carries g)" in "".join(returns.flow.values())

    @fn
    def rev[T](x: f32[T]) -> f32[T]:
        out = []
        for i in reversed(range(x.shape[0])):
            out.append(x[i] * i)
        return jnp.asarray(out)

    assert both(rev, jnp.arange(4.0)).tolist() == [9.0, 4.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="leading axes differ"):
        tx.rows((jnp.ones(2), jnp.ones(3)))

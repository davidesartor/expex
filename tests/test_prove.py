"""The prover: shape proofs from annotations alone."""

from typing import NamedTuple

import pytest

import jax
import jax.numpy as jnp

from barnacle import Dim, Etc, Key, Static, f32, f64, fn, format_reports, i32, prove


def status(f, **ex):
    return prove(f, ex or None).status


def test_proved():
    @fn
    def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
        return jnp.einsum("btd,bsd->bts", q, k)

    r = prove(attend)
    assert r.status == "proved" and r.ok
    assert "proved for all sizes" in str(r)
    assert attend.check().status == "proved"


def test_off_by_one_is_reported_symbolically():
    @fn
    def bad[B, T](x: f32[B, T]) -> f32[B, T]:
        return x[:, 1:]

    r = prove(bad)
    assert r.status == "failed" and not r.ok
    assert "returned     f32[B, T - 1]" in r.detail
    assert "axis 1 of `return` should be T, but is T - 1." in r.detail
    with pytest.raises(AssertionError):
        r.raise_if_failed()


def test_catches_bugs_invisible_at_square_sizes():
    @fn
    def transposed[B, T](x: f32[B, T]) -> f32[B, T]:
        return x.T

    transposed(jnp.ones((3, 3)))  # a concrete test with B == T passes
    assert status(transposed) == "failed"


def test_dtype_mismatch():
    @fn
    def f[D](x: f32[D]) -> i32[D]:
        return x * 2

    r = prove(f)
    assert r.status == "failed" and "expected dtype i32, got f32" in r.detail


def test_derived_dims():
    @fn(U="T + 1")
    def pad[B, T, U](x: f32[B, T]) -> f32[B, U]:
        return jnp.pad(x, ((0, 0), (0, 1)))

    @fn(U="2 * T")
    def half[T, U](x: f32[U]) -> f32[T]:
        return x[::2]

    @fn(U="T + 2")
    def wrong[B, T, U](x: f32[B, T]) -> f32[B, U]:
        return jnp.pad(x, ((0, 0), (0, 1)))

    assert status(pad) == "proved"
    assert status(half) == "proved"
    r = prove(wrong)
    assert r.status == "failed" and "T + 1" in r.detail


def test_where_is_assumed():
    @fn(where="T >= 2")
    def pairs[T, P](x: f32[T]) -> f32[P]:
        return x[1:] - x[:-1]

    assert status(pairs) == "proved"


def test_call_sites_are_proved_too():
    @fn
    def inner[N](a: f32[N], b: f32[N]) -> f32[N]:
        return a + b

    @fn
    def outer[B](x: f32[B]) -> f32[B]:
        return inner(x, x[:-1])

    r = prove(outer)
    assert r.status == "failed"
    assert "inner(): argument `b`" in r.detail
    assert "received     f32[B - 1]" in r.detail


def test_static_needs_default_or_examples():
    @fn
    def a[B, D](x: f32[B, D], n: Static[int]) -> f32[B, D]:
        return x * n

    @fn
    def b[B, D](x: f32[B, D], n: Static[int] = 2) -> f32[B, D]:
        return x * n

    @fn(examples={"n": [1, 3]})
    def c[B, D](x: f32[B, D], n: Static[int]) -> f32[B, D]:
        return x * n

    r = prove(a)
    assert r.status == "unchecked" and 'examples={"n": [...]}' in r.detail
    assert status(b) == "proved"
    assert prove(c).cases == 2 and status(c) == "proved"
    assert status(a, n=[2]) == "proved"
    assert a.check(n=2).status == "proved"


def test_failure_names_the_static_case():
    @fn(examples={"n": [1, 2]}, U="2 * T")
    def rep[T, U](x: f32[T], n: Static[int]) -> f32[U]:
        return jnp.tile(x, n)

    r = prove(rep)
    assert r.status == "failed" and "with n=1:" in r.detail


def test_reshape_needs_its_constraint():
    def heads[B, D](x: f32[B, D], n: Static[int] = 2) -> f32[B, D]:
        b, d = x.shape
        return x.reshape(b, n, d // n).reshape(b, d)

    # Without the constraint the function is wrong for odd D, and the prover says so.
    r = prove(fn(heads))
    assert r.status == "error" and "where=" in r.detail
    # With it, the reshape is provably size-preserving.
    assert status(fn(where="D % 2 == 0")(heads)) == "proved"


def test_reshape_is_validated_symbolically():
    # JAX itself accepts this for symbolic sizes; the prover must not.
    @fn
    def grow[B, D, E](x: f32[B, D]) -> f32[B, E]:
        b, d = x.shape
        return x.reshape(b, d + 1)

    assert status(grow) in ("error", "failed")


def test_spot_check_when_symbolic_trace_is_inconclusive():
    @fn
    def branchy[T](x: f32[T]) -> f32[T]:
        if x.shape[0] > 4:  # shape-dependent Python control flow
            return x * 2
        return x

    r = prove(branchy)
    assert r.status == "spot-checked" and r.ok
    assert "could not be proved for all sizes" in r.detail


def test_spot_check_still_finds_failures():
    @fn
    def branchy[T](x: f32[T]) -> f32[T]:
        if x.shape[0] > 4:
            return x[1:]
        return x

    assert status(branchy) == "failed"


def test_size_one_is_tried():
    @fn
    def squeezed[B, T](x: f32[B, T]) -> f32[B, T]:
        return x.squeeze() + 0  # drops axes of size 1

    r = prove(squeezed)
    assert r.status == "failed" and "B=1, T=1" in "".join(r.notes)

    @fn
    def diff_max[T](x: f32[T]) -> f32:
        return jnp.max(x[1:] - x[:-1])  # empty for T == 1

    assert status(diff_max) == "error"
    assert status(fn(where="T >= 2")(diff_max.raw)) == "proved"


def test_untraceable_function_is_an_error_with_hint():
    @fn
    def relu[B](x: f32[B]) -> f32[B]:
        if x.sum() > 0:
            return x
        return -x

    r = prove(relu)
    assert r.status == "error"
    assert "TracerBoolConversionError" in r.detail and "jnp.where" in r.detail


def test_variadic_ranks():
    @fn
    def ok[*Batch, T](x: f32[*Batch, T], y: f32[*Batch]) -> f32[*Batch]:
        return x.sum(-1) + y

    @fn
    def bad[*Batch, T](x: f32[*Batch, T]) -> f32[*Batch]:
        return x.sum(0)

    assert prove(ok).cases == 3 and status(ok) == "proved"
    r = prove(bad)
    assert r.status == "failed" and "*Batch of rank 1" in r.detail


class Params[D](NamedTuple):
    w: f32[D, D]
    b: f32[D]


class RGB(Dim, size=3): ...


def test_structs_keys_containers_and_64_bit():
    @fn
    def dense[B, D](p: Params[D], x: f32[B, D]) -> tuple[f32[B, D], f32]:
        return x @ p.w + p.b, jnp.sum(x)

    @fn
    def noise[D](x: f32[D], key: Key) -> f32[D]:
        return x + jax.random.normal(key, x.shape)

    @fn
    def legacy[D](x: f32[D], key: Key) -> f32[D]:
        assert key.shape == (2,)
        return x

    @fn
    def dbl[D](x: f64[D]) -> f64[D]:
        return x * 2

    @fn
    def gray[H, W](img: f32[H, W, RGB]) -> f32[H, W]:
        return img.mean(-1)

    @fn
    def opt[D](x: f32[D], m: f32[D] | None = None) -> f32[D]:
        return x if m is None else x * m

    @fn
    def lst[D](xs: list[f32[D]], d: dict[str, f32[D]], e: f32[Etc, D]) -> f32[D]:
        return sum(xs)  # type: ignore[return-value]

    for f in (dense, noise, legacy, dbl, gray, opt, lst):
        assert status(f) == "proved", prove(f)


def test_output_only_dim_is_free():
    @fn
    def f[B, K](x: f32[B]) -> tuple[f32[B, K], f32[K]]:
        return jnp.stack([x, x], -1), jnp.ones(2)

    @fn
    def g[B, K](x: f32[B]) -> tuple[f32[B, K], f32[K]]:
        return jnp.stack([x, x], -1), jnp.ones(3)

    assert status(f) == "proved"
    assert status(g) == "failed"


def test_methods():
    class M:
        @fn
        def call[D](self, x: f32[D]) -> f32[D]:
            return x

        @fn
        def bad[D](self, x: f32[D]) -> f32[D]:
            return x[1:]

    r = prove(M.call)
    assert r.status == "unchecked" and "prove(instance.method)" in r.detail
    assert prove(M().call).status == "proved"
    assert prove(M().bad).status == "failed"


def test_unannotated_parameter():
    @fn
    def f[D](x: f32[D], scale) -> f32[D]:
        return x * scale

    assert status(f) == "unchecked"
    assert status(f, scale=[2.0]) == "proved"
    with pytest.raises(TypeError, match="unknown parameter"):
        prove(f, {"nope": [1]})


def test_checked_false_is_still_proved():
    @fn(checked=False)
    def bad[T](x: f32[T]) -> f32[T]:
        return x[1:]

    assert status(bad) == "failed"


def test_jit_functions_are_proved_without_compiling():
    @fn(jit=True)
    def f[T](x: f32[T], n: Static[int] = 2) -> f32[T]:
        return x * n

    assert status(f) == "proved"
    assert f.traces == []


def test_format_reports():
    @fn
    def good[T](x: f32[T]) -> f32[T]:
        return x

    @fn
    def bad[T](x: f32[T]) -> f32[T]:
        return x[1:]

    text = format_reports([prove(good), prove(bad)])
    assert "ok  " in text and "FAIL" in text
    assert "2 functions: 1 proved, 1 failed" in text

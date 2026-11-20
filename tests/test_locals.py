"""Annotated local variables are checked against the call's dimensions."""

import pytest

import jax
import jax.numpy as jnp

import barnacle as bn
from barnacle import ShapeError, f32, fn, i32


@fn
def encode[B, T, D](x: f32[B, T], table: f32[T, D]) -> f32[B, D]:
    h: f32[B, T, D] = x[:, :, None] * table
    pooled: f32[B, D] = h.mean(1)
    count: int = 3  # ordinary annotations are ignored
    return pooled * count


def test_locals_pass():
    assert encode(jnp.ones((2, 3)), jnp.ones((3, 4))).shape == (2, 4)
    assert jax.jit(encode)(jnp.ones((2, 3)), jnp.ones((3, 4))).shape == (2, 4)
    assert encode.converted


def test_local_mismatch_is_reported_at_its_line():
    @fn
    def bad[B, T, D](x: f32[B, T], table: f32[T, D]) -> f32[B, D]:
        h: f32[B, T, D] = x[:, :, None] * table
        pooled: f32[B, D] = h.mean(0)  # wrong axis
        return pooled

    with pytest.raises(ShapeError) as e:
        bad(jnp.ones((2, 3)), jnp.ones((3, 4)))
    msg = str(e.value)
    assert "local variable `pooled` does not match its annotation" in msg
    assert "declared  pooled: f32[B, D]" in msg and "assigned          f32[3, 4]" in msg
    assert "B is 2 (from `x: f32[2, 3]`), but axis 0 of `pooled` is 3." in msg
    assert "> pooled: f32[B, D] = h.mean(0)  # wrong axis" in msg


def test_locals_are_proved_for_all_sizes():
    assert bn.prove(encode).status == "proved"

    @fn
    def bad[B, T](x: f32[B, T]) -> f32[B]:
        d: f32[B, T] = x[:, 1:] - x[:, :-1]
        return d.sum(1)

    r = bn.prove(bad)
    assert r.status == "failed"
    assert "local variable `d`" in r.detail and "assigned     f32[B, T - 1]" in r.detail


def test_local_can_introduce_a_dimension_and_check_dtype():
    @fn
    def f[T, K](x: f32[T]) -> f32[K]:
        idx: i32[K] = jnp.arange(2)
        y: f32[K] = x[idx]
        return y

    assert f(jnp.arange(5.0)).shape == (2,)

    @fn
    def g[T](x: f32[T]) -> f32[T]:
        n: i32[T] = x * 2  # float, not int
        return x

    with pytest.raises(ShapeError, match="expected dtype i32, got f32"):
        g(jnp.ones(3))


def test_locals_inside_staged_control_flow():
    @fn
    def f[T, D](xs: f32[T, D]) -> f32[D]:
        acc: f32[D] = jnp.zeros(xs.shape[1])
        for x in xs:
            step: f32[D] = x * 2
            acc = acc + step
        if acc.sum() > 0:
            out: f32[D] = acc
        else:
            out = -acc
        return out

    assert jnp.allclose(jax.jit(f)(jnp.ones((4, 3))), 8.0)
    assert bn.prove(f).status == "proved"


def test_locals_are_skipped_when_checks_are_off(monkeypatch):
    @fn
    def bad[T](x: f32[T]) -> f32[T]:
        y: f32[T, T] = x
        return y

    with pytest.raises(ShapeError):
        bad(jnp.ones(3))
    monkeypatch.setattr(bn.config, "checked", False)
    assert bad(jnp.ones(3)).shape == (3,)

"""``.lift`` (vectorising from the signature) and ``jit=True`` (retrace reports)."""

import warnings
from typing import NamedTuple

import pytest

import jax
import jax.numpy as jnp

from typax import Key, RetraceWarning, ShapeError, Static, config, f32, fn


@fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return x @ w


def test_lift_matches_vmap():
    xs = jax.random.normal(jax.random.key(0), (32, 5, 3))
    w = jnp.arange(3.0)
    out = score.lift(xs, w)
    assert out.shape == (32, 5)
    assert jnp.allclose(out, jax.vmap(score, in_axes=(0, None))(xs, w))


def test_lift_without_batch_is_a_plain_call():
    assert score.lift(jnp.ones((5, 3)), jnp.ones(3)).shape == (5,)


def test_lift_broadcasts_batch_axes():
    xs = jax.random.normal(jax.random.key(0), (8, 1, 5, 3))
    ws = jax.random.normal(jax.random.key(1), (7, 3))
    out = score.lift(xs, ws)
    assert out.shape == (8, 7, 5)
    assert jnp.allclose(out, jnp.einsum("bitd,jd->bjt", xs, ws), atol=1e-5)
    assert score.lift(jnp.ones((1, 5, 3)), jnp.ones(3)).shape == (1, 5)
    assert score.lift(jnp.ones((1, 5, 3)), jnp.ones((1, 3))).shape == (1, 5)


def test_lift_keyword_arguments():
    assert score.lift(w=jnp.ones((4, 3)), x=jnp.ones((5, 3))).shape == (4, 5)


def test_lift_errors():
    with pytest.raises(ShapeError, match="batch axes of the arguments do not broadcast"):
        score.lift(jnp.ones((4, 5, 3)), jnp.ones((6, 3)))
    # the per-example check still runs
    with pytest.raises(ShapeError, match="D is 3"):
        score.lift(jnp.ones((4, 5, 3)), jnp.ones((4, 2)))

    @fn
    def v[*Batch](x: f32[*Batch]) -> f32[*Batch]:
        return x

    with pytest.raises(TypeError, match="variadic shape"):
        v.lift(jnp.ones(3))


def test_lift_splits_a_single_key():
    @fn
    def noise[D](x: f32[D], key: Key) -> f32[D]:
        return x + jax.random.normal(key, x.shape)

    out = noise.lift(jnp.zeros((4, 3)), jax.random.key(0))
    assert out.shape == (4, 3)
    assert len({tuple(r.tolist()) for r in out}) == 4  # no shared randomness
    # legacy keys work too
    assert noise.lift(jnp.zeros((4, 3)), jax.random.PRNGKey(0)).shape == (4, 3)
    # a batch of keys is mapped over, not split
    keys = jax.random.split(jax.random.key(1), 4)
    out2 = noise.lift(jnp.zeros((4, 3)), keys)
    assert jnp.allclose(out2[2], noise(jnp.zeros(3), keys[2]))
    # a batch of keys alone defines the batch
    assert noise.lift(jnp.zeros(3), keys).shape == (4, 3)


def test_lift_static_and_structs():
    class P[D](NamedTuple):
        w: f32[D]
        b: f32

    @fn
    def f[D](p: P[D], x: f32[D], n: Static[int]) -> tuple[f32, f32[D]]:
        return (p.w * x).sum() * n + p.b, x

    a, b = f.lift(P(jnp.ones((6, 3)), jnp.float32(1.0)), jnp.ones((6, 3)), 2)
    assert a.shape == (6,) and b.shape == (6, 3)
    assert jnp.allclose(a, 7.0)


def test_lift_composes_with_jit_and_grad():
    xs, w = jnp.ones((4, 5, 3)), jnp.ones(3)
    assert jax.jit(score.lift)(xs, w).shape == (4, 5)
    g = jax.grad(lambda w: score.lift(xs, w).sum())(w)
    assert jnp.allclose(g, 20.0)


def test_jit_runs_and_checks_once_per_trace():
    calls = []

    @fn(jit=True)
    def step[B](x: f32[B], lr: f32, n: Static[int]) -> f32[B]:
        calls.append(1)
        return x * lr * n

    for _ in range(3):
        assert jnp.allclose(step(jnp.ones(3), 0.5, 2), 1.0)
    assert len(calls) == 1 and len(step.traces) == 1
    with pytest.raises(ShapeError):
        step(jnp.ones((3, 3)), 0.5, 2)


def test_static_by_position_and_keyword():
    @fn(jit=True)
    def f[B](x: f32[B], n: Static[int] = 3) -> f32[B]:
        return x * n

    assert jnp.allclose(f(jnp.ones(2)), 3)
    assert jnp.allclose(f(jnp.ones(2), 4), 4)
    assert jnp.allclose(f(jnp.ones(2), n=5), 5)


def test_retrace_explanation(monkeypatch):
    @fn(jit=True)
    def step[B](x: f32[B], lr: f32, n: Static[int]) -> f32[B]:
        return x * lr * n

    monkeypatch.setattr(config, "retrace_warn", 1)
    step(jnp.ones(3), 0.1, 2)
    with pytest.warns(RetraceWarning) as w:
        step(jnp.ones(4), 0.1, 2)
    assert "x: f32[3]  ->  f32[4]" in str(w[0].message)
    with pytest.warns(RetraceWarning) as w:
        step(jnp.ones(4), jnp.float32(0.1), 3)
    msg = str(w[0].message)
    assert "lr: weak f32[()] (a Python scalar)  ->  f32[()]" in msg
    assert "n: static 2  ->  static 3" in msg
    assert len(step.traces) == 3
    assert "trace #3 differs from trace #2" in step.explain_retrace()
    assert "first call" in step.explain_retrace(0)


def test_no_warning_below_threshold():
    @fn(jit=True)
    def f[B](x: f32[B]) -> f32[B]:
        return x

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n in range(1, 5):
            f(jnp.ones(n))
    with pytest.warns(RetraceWarning, match="traced 5 times"):
        f(jnp.ones(9))


def test_unhashable_static_gets_a_note():
    @fn(jit=True)
    def f[B](x: f32[B], n: Static[list]) -> f32[B]:
        return x

    with pytest.raises(ValueError) as e:
        f(jnp.ones(2), [1])
    assert any("must be hashable" in n for n in e.value.__notes__)

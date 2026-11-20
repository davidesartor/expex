"""``.lift`` (vectorising from the signature) and ``jit=True`` (retrace reports)."""

import warnings
from typing import NamedTuple

import pytest

import jax
import jax.numpy as jnp

import typax as tx
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


def test_no_warning_below_threshold(monkeypatch):
    monkeypatch.setattr(config, "retrace_warn", 4)

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

    with pytest.raises(TypeError, match="must be hashable"):
        f(jnp.ones(2), [1])


def test_nested_lift_traces_each_function_once_per_example_shape():
    calls = {"inner": 0, "outer": 0}

    @fn
    def inner[D](x: f32[D]) -> f32:
        calls["inner"] += 1
        return x.sum()

    @fn
    def outer[N, D](xs: f32[N, D]) -> f32:
        calls["outer"] += 1
        return inner.lift(xs).max()

    out = jax.jit(outer.lift)(jnp.ones((32, 16, 3)))
    assert out.shape == (32,)
    # tracing and checking see one example at each level, never the batch
    assert calls == {"inner": 1, "outer": 1}


def test_jit_treats_non_array_values_as_static():
    from typing import Callable, NamedTuple

    class Model(NamedTuple):
        act: Static[Callable]
        mode: Static[str]
        w: f32[3]  # type: ignore[valid-type]
        scale: f32 | None

    @fn(jit=True)
    def apply(m: Model, x: f32[3], post=None, opts: dict | None = None) -> f32[3]:  # type: ignore[valid-type]
        y = m.act(x * m.w)
        if m.mode == "neg":
            y = -y
        if m.scale is not None:
            y = y * m.scale
        if post is not None:
            y = post(y)
        return y + (opts or {}).get("bias", 0.0)

    x = jnp.arange(3.0)
    m = Model(jnp.tanh, "pos", jnp.ones(3), None)
    assert jnp.allclose(apply(m, x), jnp.tanh(x))
    assert jnp.allclose(apply(m._replace(w=jnp.ones(3) * 2), x), jnp.tanh(2 * x))
    assert len(apply.traces) == 1  # new array values: no retrace
    assert jnp.allclose(apply(m._replace(mode="neg"), x), -jnp.tanh(x))
    assert jnp.allclose(apply(m._replace(scale=jnp.float32(3.0)), x), 3 * jnp.tanh(x))
    assert jnp.allclose(apply(m, x, jnp.abs, {"bias": 1.0}), jnp.abs(jnp.tanh(x)) + 1)
    assert jnp.allclose(apply(m, x, jnp.abs, {"bias": 2.0}), jnp.abs(jnp.tanh(x)) + 2)
    assert len(apply.traces) == 4  # mode, scale present, post given; bias is a traced number


def test_rows_chunk_runs_like_lax_map():
    @fn
    def f[N, D](xs: f32[N, D], chunk: Static[int]) -> f32[N]:
        out = []
        for x in tx.rows(xs, chunk=chunk):
            out.append(jnp.tanh(x).sum())
        return jnp.asarray(out)

    xs = jax.random.normal(jax.random.key(0), (7, 3))
    ref = jnp.tanh(xs).sum(-1)
    assert jnp.allclose(f(xs, 1), ref)
    assert jnp.allclose(jax.jit(lambda v: f(v, 1))(xs), ref)
    assert "lax.map in chunks of 1" in "".join(f.flow.values())
    assert jnp.allclose(jax.jit(lambda v: f(v, 3))(xs), ref, atol=1e-6)
    assert "lax.map in chunks of 3" in "".join(f.flow.values())


def test_lift_option_and_forward_references():
    from typing import NamedTuple

    @fn(lift=True)
    def safe_sqrt(x: f32) -> f32:
        if x > 0:
            return jnp.sqrt(x)
        return 0.0

    assert safe_sqrt(4.0) == 2.0
    assert safe_sqrt(jnp.array([[4.0, -1.0], [0.0, 9.0]])).tolist() == [[2.0, 0.0], [0.0, 3.0]]
    g = jax.grad(lambda v: safe_sqrt(v).sum())(jnp.array([4.0, 0.0, -1.0]))
    assert g.tolist() == [0.25, 0.0, 0.0]

    @fn(lift=True)
    def dot[D](a: f32[D], b: f32[D]) -> f32:
        return a @ b

    assert dot(jnp.ones((5, 1, 3)), jnp.ones((4, 3))).shape == (5, 4)
    assert tx.prove(dot).status == "proved"


class Poly[*B, K](__import__("typing").NamedTuple):
    c: "f32[*B, K]"

    @fn(M="m")
    def pad_to[M](self, m: Static[int]) -> "Poly[*B, M]":
        extra = jnp.zeros(self.c.shape[:-1] + (m - self.c.shape[-1],))
        return Poly(jnp.concatenate([self.c, extra], -1))

    @staticmethod
    @fn
    def zeros[N](like: f32[N]) -> "Poly[N]":
        return Poly(jnp.zeros_like(like))

    @fn
    def bad(self) -> "Poly[*B, K]":
        return Poly(self.c[..., 1:])


def test_methods_can_name_their_own_class():
    p = Poly(jnp.ones((2, 3)))
    assert p.pad_to(5).c.shape == (2, 5)
    assert "-> Poly[*B, M]" in repr(Poly.pad_to)
    assert str(Poly.pad_to.ret.fields["c"]) == "f32[*B, m]"  # M is tied to the argument
    assert Poly.zeros(jnp.ones(4)).c.shape == (4,)
    with pytest.raises(ShapeError, match=r"declared  return.c: f32\[\*B, K\]"):
        p.bad()
    assert tx.prove(Poly.zeros).status == "proved"

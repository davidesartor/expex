"""``.lift`` / ``@tx.lift`` (vectorising from the signature) and ``jax.jit`` on typed functions."""

import warnings
from typing import NamedTuple

import pytest

import functools

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx
from typax import Key, ShapeError, Static, f32, fn


@fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return x @ w


def test_lift_matches_vmap():
    xs = jr.normal(jr.key(0), (32, 5, 3))
    w = jnp.arange(3.0)
    out = score.lift(xs, w)
    assert out.shape == (32, 5)
    assert jnp.allclose(out, jax.vmap(score, in_axes=(0, None))(xs, w))


def test_lift_without_batch_is_a_plain_call():
    assert score.lift(jnp.ones((5, 3)), jnp.ones(3)).shape == (5,)


def test_lift_broadcasts_batch_axes():
    xs = jr.normal(jr.key(0), (8, 1, 5, 3))
    ws = jr.normal(jr.key(1), (7, 3))
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
        return x + jr.normal(key, x.shape)

    out = noise.lift(jnp.zeros((4, 3)), jr.key(0))
    assert out.shape == (4, 3)
    assert len({tuple(r.tolist()) for r in out}) == 4  # no shared randomness
    # legacy keys work too
    assert noise.lift(jnp.zeros((4, 3)), jr.PRNGKey(0)).shape == (4, 3)
    # a batch of keys is mapped over, not split
    keys = jr.split(jr.key(1), 4)
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


def test_checks_run_once_per_trace_under_jax_jit():
    calls = []

    @functools.partial(jax.jit, static_argnames="n")
    @fn
    def step[B](x: f32[B], lr: f32, n: Static[int]) -> f32[B]:
        calls.append(1)
        return x * lr * n

    for _ in range(3):
        assert jnp.allclose(step(jnp.ones(3), 0.5, 2), 1.0)
    assert len(calls) == 1
    with pytest.raises(ShapeError):
        step(jnp.ones((3, 3)), 0.5, 2)


def test_fn_options():
    calls = []

    @fn(jit=True)
    def tile[N](x: f32[N], n: int, lr: float) -> f32[N * n]:
        calls.append(1)
        return jnp.tile(x, n) * lr

    tile(jnp.ones(3), 2, 0.1)
    tile(jnp.ones(3), 2, 0.5)
    assert len(calls) == 1 and tx.prove(tile).ok

    @fn(jit=True, random=True, lift=True)
    def noise[D](x: f32[D]) -> f32[D]:
        return x + jr.normal(tx.key(), x.shape)

    a = noise(jnp.zeros((2, 3)))
    assert a.shape == (2, 3) and not jnp.allclose(a[0], a[1])
    assert not jnp.allclose(a, noise(jnp.zeros((2, 3))))

    @fn(remat="dots", jit=True)
    def layer[B, D](x: f32[B, D], w: f32[D, D], depth: int) -> f32[B, D]:
        for _ in range(depth):
            x = jnp.tanh(x @ w)
        return x

    def loss(w):
        return layer(jnp.ones((4, 8)), w, 3).sum()

    g = jax.grad(loss)(jnp.eye(8) * 0.5)
    plain = jax.grad(lambda w: jnp.tanh(jnp.tanh(jnp.tanh(jnp.ones((4, 8)) @ w) @ w) @ w).sum())(jnp.eye(8) * 0.5)
    assert jnp.allclose(g, plain, atol=1e-6) and tx.prove(layer).ok

    @fn(jit=True, donate="state")
    def step[N](state: f32[N], grad: f32[N]) -> f32[N]:
        return state - 0.1 * grad

    assert step(jnp.ones(3), jnp.ones(3)).tolist() == pytest.approx([0.9] * 3)
    with pytest.raises(TypeError, match="only applies with jit"):
        fn(donate="x")(lambda x: x)
    with pytest.raises(TypeError, match="called lift=True"):
        fn(vectorize=True)
    with pytest.raises(ValueError, match="remat must be"):
        fn(remat="sometimes")(lambda x: x)


def test_options_on_struct_methods():
    @tx.struct
    class Lin[I, O]:
        w: f32[O, I]

        @fn(jit=True, lift=True)
        def __call__(self, x: f32[I]) -> f32[O]:
            return self.w @ x

    assert Lin(jnp.ones((2, 3)))(jnp.ones((5, 3))).shape == (5, 2)
    assert tx.prove(Lin.__call__).ok


def test_print_works_under_jit(capfd):
    @fn(jit=True)
    def show[N](x: f32[N]) -> f32[N]:
        print("x =", x)
        return x * 2

    show(jnp.arange(3.0))
    jax.effects_barrier()
    assert "x = [0. 1. 2.]" in capfd.readouterr().out


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


def test_rows_chunk_runs_like_lax_map():
    @fn
    def f[N, D](xs: f32[N, D], chunk: Static[int]) -> f32[N]:
        out = []
        for x in tx.rows(xs, chunk=chunk):
            out.append(jnp.tanh(x).sum())
        return jnp.asarray(out)

    xs = jr.normal(jr.key(0), (7, 3))
    ref = jnp.tanh(xs).sum(-1)
    assert jnp.allclose(f(xs, 1), ref)
    assert jnp.allclose(jax.jit(lambda v: f(v, 1))(xs), ref)
    assert "lax.map in chunks of 1" in "".join(f.flow.values())
    assert jnp.allclose(jax.jit(lambda v: f(v, 3))(xs), ref, atol=1e-6)
    assert "lax.map in chunks of 3" in "".join(f.flow.values())


def test_lift_option_and_forward_references():
    from typing import NamedTuple

    @tx.lift
    @fn
    def safe_sqrt(x: f32) -> f32:
        if x > 0:
            return jnp.sqrt(x)
        return 0.0

    assert safe_sqrt(4.0) == 2.0
    assert safe_sqrt(jnp.array([[4.0, -1.0], [0.0, 9.0]])).tolist() == [[2.0, 0.0], [0.0, 3.0]]
    g = jax.grad(lambda v: safe_sqrt(v).sum())(jnp.array([4.0, 0.0, -1.0]))
    assert g.tolist() == [0.25, 0.0, 0.0]

    @tx.lift
    @fn
    def dot[D](a: f32[D], b: f32[D]) -> f32:
        return a @ b

    assert dot(jnp.ones((5, 1, 3)), jnp.ones((4, 3))).shape == (5, 4)
    assert tx.prove(dot).status == "proved"


class Poly[*B, K](__import__("typing").NamedTuple):
    c: "f32[*B, K]"

    @fn
    def pad_to(self, m: Static[int]) -> "Poly[*B, m]":
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
    assert "-> Poly[*B, m]" in repr(Poly.pad_to)
    assert str(Poly.pad_to.ret.fields["c"]) == "f32[*B, m]"  # the size is the argument
    assert Poly.zeros(jnp.ones(4)).c.shape == (4,)
    with pytest.raises(ShapeError, match=r"declared  return.c: f32\[\*B, K\]"):
        p.bad()
    assert tx.prove(Poly.zeros).status == "proved"


def test_tx_jit_reads_static_arguments_from_the_signature():
    calls = []

    @tx.jit
    @fn
    def tile[N](x: f32[N], n: int, lr: float, mode: str = "a") -> "f32[N * n]":  # quoted forward reference
        calls.append(1)
        return jnp.tile(x, n) * lr

    assert tile(jnp.ones(3), 2, 0.1).shape == (6,)
    tile(jnp.ones(3), 2, 0.5)  # a float is traced: no retrace
    assert len(calls) == 1
    tile(jnp.ones(3), 3, 0.5)  # an int is static
    assert len(calls) == 2
    assert tx.prove(tile).ok

    @tx.jit
    def scaled[N](x: f32[N], k: Static[float]) -> f32[N]:  # Static[...] forces static
        calls.append(2)
        return x * k

    scaled(jnp.ones(2), 1.0)
    scaled(jnp.ones(2), 2.0)
    assert calls.count(2) == 2


def test_tx_jit_gives_random_functions_a_fresh_key():
    @tx.jit
    @tx.random
    @fn
    def noise[D](x: f32[D]) -> f32[D]:
        return x + jr.normal(tx.key(), x.shape)

    assert not jnp.allclose(noise(jnp.zeros(2)), noise(jnp.zeros(2)))
    assert jnp.allclose(noise(jnp.zeros(2), key=jr.key(0)), noise(jnp.zeros(2), key=jr.key(0)))


def test_tx_jit_on_struct_methods():
    @tx.struct
    class Lin[I, O]:
        w: f32[O, I]

        @tx.jit
        def __call__(self, x: f32[I], scale: int = 1) -> f32[O]:
            return self.w @ x * scale

    m = Lin(jnp.ones((2, 3)))
    assert m(jnp.ones(3), 2).tolist() == [6.0, 6.0]
    with pytest.raises(ShapeError):
        m(jnp.ones(4))
    assert tx.prove(Lin.__call__).ok

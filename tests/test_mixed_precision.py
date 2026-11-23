"""Mixed precision: ``@expex.fn(compute=jnp.bfloat16)``, ``@expex.struct(compute=...)``, ``expex.config.compute``."""

import re
import timeit

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import Float, Key, ShapeError, f32, f64, i32

bf16, F32 = jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)
seen: dict[str, object] = {}


@expex.fn(compute=jnp.bfloat16)
def mlp[D, F](x: f32[D], w1: f32[D, F], w2: f32[F, D], idx: i32[D], mask: bool) -> f32[D]:
    seen.update(x=x.dtype, w1=w1.dtype, idx=idx.dtype)
    return jax.nn.gelu(x @ w1) @ w2 + idx * 0 + mask


def args(B=()):
    k1, k2, k3 = jr.split(jr.key(0), 3)
    return (
        jr.normal(k1, (*B, 8)), jr.normal(k2, (8, 16)) * 0.3, jr.normal(k3, (16, 8)) * 0.3,
        jnp.arange(8), True,
    )  # fmt: skip


def test_inputs_cast_in_and_outputs_back():
    y = mlp(*args())
    assert seen == {"x": bf16, "w1": bf16, "idx": jnp.dtype(jnp.int32)}
    assert y.dtype == F32
    ref = expex.fn(mlp.raw)(*args())  # the same body, f32 throughout
    assert jnp.allclose(y, ref, atol=5e-2, rtol=5e-2) and not jnp.array_equal(y, ref)


def test_the_boundary_checks_the_storage_dtype():
    a = list(args())
    a[0] = a[0].astype(jnp.bfloat16)  # f32 is declared: bf16 from outside is a mistake
    with pytest.raises(ShapeError, match="expected dtype f32, got bfloat16"):
        mlp(*a)
    a = list(args())
    a[1] = a[1][:3]
    with pytest.raises(ShapeError, match="w1"):
        mlp(*a)


def test_composes_with_jit_vectorize_and_grad():
    assert jax.jit(mlp)(*args()).dtype == F32
    assert mlp(*args((5,))).shape == (5, 8)  # mapped over the extra axis
    assert mlp.vectorize(*args((2, 3))).dtype == F32
    g = jax.grad(lambda w: mlp(args()[0], w, *args()[2:]).sum())(args()[1])
    assert g.dtype == F32 and bool(jnp.isfinite(g).all())

    @expex.fn(jit=True, remat=True, compute=jnp.bfloat16)
    def layer[D](x: f32[D], w: f32[D, D]) -> f32[D]:
        return jnp.tanh(x @ w)

    g = jax.grad(lambda w: layer(jnp.ones(4), w).sum())(jnp.eye(4))
    assert g.dtype == F32


def test_eps_follows_the_compute_dtype():
    @expex.fn(compute=jnp.bfloat16)
    def norm[N](x: f32[N]) -> f32[N]:
        seen["eps"] = (x + expex.eps).dtype
        return x / (jnp.sqrt((x * x).mean()) + expex.tiny)

    assert norm(jnp.ones(3)).dtype == F32 and seen["eps"] == bf16


def test_keys_ints_and_python_numbers_are_left_alone():
    @expex.random
    @expex.fn(compute=jnp.bfloat16)
    def noisy[N](x: f32[N], scale: float, k: Key, n: i32[N]) -> f32[N]:
        seen.update(k=k.dtype, n=n.dtype, x=x.dtype)
        return x + scale * jr.normal(expex.key(), x.shape, x.dtype) + jr.normal(k, x.shape, x.dtype)

    assert noisy(jnp.ones(3), 0.5, jr.key(1), jnp.arange(3)).dtype == F32
    assert seen["x"] == bf16 and seen["n"] == jnp.dtype(jnp.int32)
    assert jnp.issubdtype(seen["k"], jax.dtypes.prng_key)


def test_results_without_a_declared_dtype():
    @expex.fn(compute=jnp.bfloat16)
    def any_float[N](x: f32[N]) -> Float[N]:
        return x * 2

    @expex.fn(compute=jnp.bfloat16)
    def wider[N](x: f32[N]) -> f64[N]:
        return x * 2

    assert any_float(jnp.ones(3)).dtype == bf16  # Float: no storage dtype to go back to
    with jax.enable_x64(True):
        assert wider(jnp.ones(3, jnp.float32)).dtype == jnp.float64


# ------------------------------------------------------------------ structs


@expex.struct
class Norm[D]:
    scale: f32[D] = 1.0

    def __call__(self, x: f32[D]) -> f32[D]:
        seen["norm"] = x.dtype
        return x / jnp.sqrt((x * x).mean() + expex.eps) * self.scale


@expex.struct
class Dense[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.3)
    norm: Norm[D]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        seen["dense"] = (x.dtype, self.w.dtype)
        return self.norm(x @ self.w)  # Norm per row: mapped over T


@expex.struct(compute=jnp.bfloat16)
class Net[D]:
    layers: list[Dense[D]]
    head: f32[D] = jax.nn.initializers.normal(1.0)

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        for layer in self.layers:
            x = x + layer(x)
        return x

    def score[T](self, x: f32[T, D]) -> f32[T]:
        return self(x) @ self.head


@expex.struct(compute=jnp.float32)
class Norm32[D]:  # a struct of its own setting, inside a bf16 model
    scale: f32[D] = 1.0

    def __call__(self, x: f32[D]) -> f32[D]:
        seen["norm32"] = x.dtype
        return x / jnp.sqrt((x * x).mean() + expex.eps) * self.scale


@expex.struct(compute=jnp.bfloat16)
class Mixed[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.3)
    norm: Norm32[D]

    def __call__(self, x: f32[D]) -> f32[D]:
        h = self.norm(x @ self.w)
        seen["back"] = h.dtype
        return h


def test_a_struct_computes_in_its_dtype_and_nested_modules_follow():
    net = Net(jr.key(0), D=8, layers=2)
    x = jr.normal(jr.key(1), (5, 8))
    y = net(x)
    assert y.dtype == F32 and net.layers[0].w.dtype == F32
    assert seen["dense"] == (bf16, bf16) and seen["norm"] == bf16
    assert net.score(x).dtype == F32
    with expex.config(checked=False):  # unchecked calls cast too
        assert net(x).dtype == F32


def test_inner_settings_win():
    m = Mixed(jr.key(0), D=4)
    assert m(jnp.ones(4)).dtype == F32
    assert seen["norm32"] == F32 and seen["back"] == bf16  # back in the caller's dtype

    @expex.fn(compute=jnp.float32)
    def own[D](x: f32[D]) -> f32[D]:
        seen["own"] = x.dtype
        return x

    with expex.config(compute=jnp.bfloat16):
        own(jnp.ones(2))
    assert seen["own"] == F32


def test_a_subclass_with_a_setting():
    @expex.struct(compute=jnp.bfloat16)
    class Dense16(Dense):
        pass

    d = Dense16(jr.key(0), D=4)
    assert d(jnp.ones((3, 4))).dtype == F32 and seen["dense"] == (bf16, bf16)
    Dense(jr.key(0), D=4)(jnp.ones((3, 4)))
    assert seen["dense"] == (F32, F32)  # the parent class is unchanged


def test_gradients_come_back_in_f32():
    net = Net(jr.key(0), D=8, layers=3)
    x = jr.normal(jr.key(1), (5, 8))
    g = jax.jit(jax.grad(lambda m: (m(x) ** 2).mean()))(net)
    assert {a.dtype for a in jax.tree.leaves(g)} == {F32}
    assert all(bool(jnp.isfinite(a).all()) for a in jax.tree.leaves(g))


# ------------------------------------------------------------------ globally


@expex.struct
class Plain[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.3)

    def __call__(self, x: f32[D]) -> f32[D]:
        seen["plain"] = (x.dtype, self.w.dtype)
        return jnp.tanh(x @ self.w)


def test_the_global_setting():
    p = Plain(jr.key(0), D=4)
    with expex.config(compute="bfloat16"):
        assert p(jnp.ones(4)).dtype == F32 and seen["plain"] == (bf16, bf16)
        assert jax.jit(lambda p, x: p(x))(p, jnp.ones(4)).dtype == F32
    p(jnp.ones(4))
    assert seen["plain"] == (F32, F32)


def test_bad_settings():
    with pytest.raises(TypeError, match="float dtype"):
        expex.fn(compute=jnp.int32)(lambda x: x)
    with pytest.raises(TypeError, match="float dtype"):

        @expex.struct(compute="nope")
        class Bad:
            w: f32[3]


# ------------------------------------------------------------------ the transformer


def test_the_transformer_in_bf16(monkeypatch):
    import pathlib
    import sys

    monkeypatch.syspath_prepend(str(pathlib.Path(__file__).parent.parent / "examples"))
    import transformer as T

    @expex.struct(compute=jnp.bfloat16)
    class Transformer16(T.Transformer):
        pass

    cfg = dict(Vocab=50, D=32, F=64, blocks=2, heads=4)
    m32, m16 = T.Transformer(jr.key(0), cfg), Transformer16(jr.key(0), cfg)
    tokens = jr.randint(jr.key(1), (4, 24), 0, 50)
    loss = jax.jit(lambda m, t: m.loss(t).mean())
    l32, l16 = loss(m32, tokens), loss(m16, tokens)
    assert l16.dtype == F32 and abs(float(l16) - float(l32)) < 2e-2 * float(l32)

    text = loss.lower(m16, tokens).as_text()
    dots = [line for line in text.splitlines() if "dot_general" in line]
    copies = 2 if expex.config.unroll_layers >= 2 else 1  # the blocks' scan, unrolled when compiled
    assert len(dots) == 8 * copies + 1  # 4 projections, 2 attention products, 2 MLP per block; the unembedding
    assert all(re.search(r": \(tensor<[0-9x]*xbf16>, tensor<[0-9x]*xbf16>\) -> tensor<[0-9x]*xbf16>", d) for d in dots)
    assert "xbf16>" not in loss.lower(m32, tokens).as_text()

    grads = jax.jit(jax.grad(lambda m, t: m.loss(t).mean()))(m16, tokens)
    assert {g.dtype for g in jax.tree.leaves(grads)} == {F32}
    sys.modules.pop("transformer", None)


def test_zero_cost_when_unset():
    """A function without a setting runs its body as before: no wrapper, nothing per call."""

    @expex.fn
    def f[N](x: f32[N]) -> f32[N]:
        return x

    assert f.compute is None and f._impl is not None and not hasattr(f._impl, "__expex_body__")
    assert hasattr(mlp._impl, "__expex_body__")
    x = jnp.ones(3)
    t = min(timeit.repeat(lambda: f(x), number=500, repeat=5)) / 500
    assert t < 200e-6

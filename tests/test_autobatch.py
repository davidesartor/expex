"""Automatic batching: a call whose arguments have extra leading axes maps over them by itself.

And eager mapped calls (``f.vectorize``, or mapped by themselves) run compiled
from the second call with the same shapes on.
"""

import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import Key, ShapeError, f32, fn
from expex import batching
from expex import function as _fn

ROOT = pathlib.Path(__file__).resolve().parent.parent


@expex.struct
class Norm[D]:
    scale: f32[D] = 1.0

    def __call__(self, x: f32[D]) -> f32[D]:
        return (x - x.mean()) * self.scale


@expex.struct
class Block[D]:
    norm: Norm[D]
    w: f32[D, D] = jax.nn.initializers.normal(0.1)

    def __call__[T](self, h: f32[T, D]) -> f32[T, D]:
        return self.norm(h) @ self.w  # Norm is per token: mapped over T


@fn
def add[D](x: f32[D], y: f32[D]) -> f32[D]:
    return x + y


def model(D=4):
    return Block(norm=Norm(scale=jnp.linspace(0.5, 1.5, D)), w=jr.normal(jr.key(0), (D, D)) * 0.1)


# ------------------------------------------------------------------ mapping


def test_function_maps_like_vectorize():
    x, y = jr.normal(jr.key(1), (5, 3)), jr.normal(jr.key(2), (3,))
    assert jnp.array_equal(add(x, y), add.vectorize(x, y))
    assert jnp.array_equal(add(x, y), x + y)
    assert add(jnp.ones((2, 1, 3)), jnp.ones((4, 3))).shape == (2, 4, 3)  # batch axes broadcast
    assert add(jnp.ones((2, 5, 3)), jnp.ones((2, 5, 3))).shape == (2, 5, 3)


def test_method_on_a_sequence():
    m = model()
    h = jr.normal(jr.key(1), (6, 4))
    assert m(h).shape == (6, 4)
    assert jnp.allclose(m(h), m.norm.vectorize(h) @ m.w)
    assert m.norm(jnp.ones((2, 6, 4))).shape == (2, 6, 4)


def test_under_jit_grad_and_vmap():
    m, h = model(), jr.normal(jr.key(1), (6, 4))
    eager = m(h)
    assert jnp.allclose(jax.jit(lambda m, h: m(h))(m, h), eager, atol=1e-6)
    assert jnp.allclose(jax.jit(lambda h: m.norm(h))(h), m.norm.vectorize(h))
    g = jax.grad(lambda h: m(h).sum())(h)
    assert g.shape == (6, 4) and jnp.isfinite(g).all()
    assert jax.vmap(m)(jnp.stack([h, 2 * h])).shape == (2, 6, 4)
    assert jax.vmap(lambda h: m.norm(h))(jnp.ones((3, 6, 4))).shape == (3, 6, 4)  # mapped inside a vmap


def test_prover_proves_a_call_on_a_batch():
    assert expex.prove(Block.__call__).status == "proved"


def test_inconsistent_extra_axes_are_named():
    with pytest.raises(ShapeError) as e:
        add(jnp.ones((5, 3)), jnp.ones((4, 3)))
    assert str(e.value) == (
        "add(): the arguments have extra leading axes that do not broadcast against each other, "
        "so they cannot be mapped over.\n\n"
        "    x: f32[5, 3]  = batch [5] + f32[D]\n"
        "    y: f32[4, 3]  = batch [4] + f32[D]\n\n"
        "    add[D](x: f32[D], y: f32[D]) -> f32[D]"
    )


def test_core_mismatch_is_still_an_error():
    m = model()
    with pytest.raises(ShapeError) as e:
        m.norm(jnp.ones((3, 5)))
    text = str(e.value)
    assert text.startswith(
        "Norm.__call__(): argument `x` does not match its annotation, even with the extra leading axes mapped over.\n\n"
        "    declared  x: f32[D]\n"
        "    received     f32[5]  (one element)\n\n"
        "    D is 4 (from `self.scale: f32[4]`), but axis 0 of `x` is 5.\n\n"
        "    Extra leading axes are mapped over automatically: `x: f32[3, 5]` is a batch [3] of f32[D].\n"
    ), text
    with pytest.raises(ShapeError, match="even with the extra leading axes mapped over"):
        jax.jit(lambda h: m.norm(h))(jnp.ones((3, 5)))


def test_dtype_mismatch_is_not_mapped():
    with pytest.raises(ShapeError, match="expected dtype f32, got i32"):
        add(jnp.ones((5, 3), jnp.int32), jnp.ones(3))


def test_variadic_annotations_are_not_mapped():
    @fn
    def shift[*B, D](x: f32[*B, D], b: f32[D]) -> f32[*B, D]:
        return x + b

    assert shift(jnp.ones((2, 3)), jnp.ones(3)).shape == (2, 3)  # the variadic takes any axes
    out = shift(jnp.ones((7, 3)), jnp.ones((2, 3)))  # `b` mapped, `x` passed whole to each element
    assert out.shape == (2, 7, 3)


def test_self_is_not_mapped():
    stacked = Norm(jnp.ones((5, 4)))  # five models: mapping over them must be asked for
    with pytest.raises(ShapeError, match=r"(?s)declared  self.scale: f32\[D\].*Hint: `self.scale` has 1 extra leading axis"):
        stacked(jnp.ones(4))
    assert stacked.vectorize(jnp.ones((5, 4))).shape == (5, 4)  # .vectorize maps over self too


def test_struct_arguments_are_mapped():
    @fn
    def apply[D](n: Norm[D], x: f32[D]) -> f32[D]:
        return n(x)

    assert apply(Norm(jnp.ones((5, 4))), jnp.ones(4)).shape == (5, 4)


# ------------------------------------------------------------------ options


def test_opt_out_per_function_and_globally():
    @fn(vectorize=False)
    def strict[D](x: f32[D]) -> f32[D]:
        return x

    with pytest.raises(ShapeError, match=r"(?s)expected 1 axis, got 2.*Hint: `x` has 1 extra leading axis. To map over it, call .*strict.vectorize"):
        strict(jnp.ones((2, 3)))
    assert strict.vectorize(jnp.ones((2, 3))).shape == (2, 3)
    assert "vectorize=False" in strict.info()

    with expex.config(vectorize=False):
        with pytest.raises(ShapeError, match="expected 1 axis, got 2"):
            add(jnp.ones((2, 3)), jnp.ones(3))
        assert add.vectorize(jnp.ones((2, 3)), jnp.ones(3)).shape == (2, 3)

        @fn(vectorize=True)  # on for this function, whatever the global setting
        def kept[D](x: f32[D]) -> f32[D]:
            return x

        @expex.vectorize
        def kept2[D](x: f32[D]) -> f32[D]:
            return x

        assert kept(jnp.ones((2, 3))).shape == kept2(jnp.ones((2, 3))).shape == (2, 3)
    assert add(jnp.ones((2, 3)), jnp.ones(3)).shape == (2, 3)


def test_unchecked_calls_still_map():
    with expex.config(checked=False):
        assert add(jnp.ones((2, 3)), jnp.ones(3)).shape == (2, 3)
        assert jnp.allclose(model()(jnp.ones((6, 4))), model()(jnp.ones((6, 4))))


def test_happy_path_does_not_batch(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("a correctly shaped call went through batching")

    monkeypatch.setattr(batching, "vectorize", fail)
    monkeypatch.setattr(batching, "extra_axes", fail)
    assert add(jnp.ones(3), jnp.ones(3)).shape == (3,)
    assert model().norm(jnp.ones(4)).shape == (4,)


# ------------------------------------------------------------------ random


def test_random_elements_and_calls_get_their_own_keys():
    @expex.random
    @fn
    def noise[D](x: f32[D]) -> f32[D]:
        return x + jr.normal(expex.key(), x.shape)

    for _ in range(3):  # also once compiled
        a, b = noise(jnp.zeros((4, 2))), noise(jnp.zeros((4, 2)))
        assert len({float(v) for v in a[:, 0]}) == 4  # one key per element
        assert not jnp.array_equal(a, b)  # and per call
    k = jr.key(3)
    assert jnp.array_equal(noise(jnp.zeros((4, 2)), key=k), noise.vectorize(jnp.zeros((4, 2)), key=k))

    @fn
    def draw[D](x: f32[D], key: Key) -> f32[D]:
        return x + jr.normal(key, x.shape)

    out = draw(jnp.zeros((4, 2)), jr.key(0))  # a single key: split per element, as .vectorize does
    assert len({float(v) for v in out[:, 0]}) == 4


# ------------------------------------------------------------------ tools


def _load(name, src, tmp_path):
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_diagram_labels_mapped_calls_per_element(tmp_path):
    src = (ROOT / "examples" / "transformer.py").read_text()
    plain = src.replace("self.mlp.vectorize(self.norm2.vectorize(x))", "self.mlp(self.norm2(x))")
    plain = plain.replace("self.attn(self.norm1.vectorize(x))", "self.attn(self.norm1(x))")
    plain = plain.replace("self.norm.vectorize(x) @ self.unembed", "self.norm(x) @ self.unembed")
    assert plain.count(".vectorize(") == src.count(".vectorize(") - 4
    a = _load("autobatch_transformer_a", src, tmp_path)
    b = _load("autobatch_transformer_b", plain, tmp_path)
    for depth in (1, 2):
        da, db = expex.diagram(a.Transformer.__call__, depth=depth), expex.diagram(b.Transformer.__call__, depth=depth)
        assert [n[2] for n in da.nodes] == [n[2] for n in db.nodes]
        assert da.edges == db.edges
    calls = [n[2].split("\n")[0] for n in expex.diagram(b.Transformer.__call__, depth=2).nodes if n[1] == "call"]
    assert calls == ["LayerNorm.__call__ · vectorized", "Attention.__call__", "LayerNorm.__call__ · vectorized", "MLP.__call__ · vectorized"]
    assert _fn.CALL_HOOK.get() is None


def test_watch_sees_mapped_calls():
    m, h = model(), jr.normal(jr.key(1), (6, 4))
    for _ in range(3):  # compiled after the first call, but not while watched
        m(h)
    auto = expex.debug.watch(lambda m: m(h).sum(), m)
    explicit = expex.debug.watch(lambda m: (m.norm.vectorize(h) @ m.w).sum(), m)
    assert [r.name for r in auto.rows if "norm" in r.name] == [r.name for r in explicit.rows if "norm" in r.name] != []


# ------------------------------------------------------- compiled eager calls


def test_eager_mapped_calls_compile_after_the_first():
    traced = []

    @fn
    def f[D](x: f32[D]) -> f32[D]:
        traced.append(1)  # runs whenever the body is traced
        return x * 2

    x = jnp.ones((8, 3))
    for _ in range(4):
        assert jnp.array_equal(f(x), 2 * x)
    assert len(traced) == 2  # the first call (under vmap), then once for jit
    f.vectorize(x)
    f.vectorize(x)
    f.vectorize(x)
    assert len(traced) == 4  # .vectorize: its own cache, same rule
    f(jnp.ones((5, 3)))
    assert len(traced) == 5  # new shapes: the first call runs as before
    with expex.config(compile_vectorized=False):
        f(x)
        f(x)
    assert len(traced) == 7


def test_tools_bypass_the_compiled_calls():
    seen = []

    @fn
    def f[D](x: f32[D]) -> f32[D]:
        seen.append(1)
        return x * 2

    x = jnp.ones((8, 3))
    f(x), f(x), f(x)
    n = len(seen)
    hooked = []

    def hook(fn_, args, kwargs, run):
        hooked.append(fn_.name)
        return run(args, kwargs)

    token = _fn.CALL_HOOK.set(hook)
    try:
        f(x)
    finally:
        _fn.CALL_HOOK.reset(token)
    assert len(seen) == n + 1 and hooked == [f.name]  # the hook saw the element call
    jax.make_jaxpr(lambda x: f(x))(x)
    assert len(seen) == n + 2  # traced: never cached
    report = expex.debug.find_nan(lambda x: jnp.sqrt(-f(x)), x)
    assert report is not None and "sqrt" in str(report)


def test_compiled_calls_fall_back_when_jit_cannot_trace():
    @fn
    def scaled[D](x: f32[D], s: f32) -> f32[D]:
        return x * float(s)  # reads a value in Python: fine eagerly, not under jit

    x = jnp.ones((4, 3))
    for _ in range(3):
        assert jnp.array_equal(scaled(x, jnp.float32(2.0)), 2 * x)

    assert batching._Compiled.of(scaled, auto=True).off  # never tried again


def test_compiled_struct_method():
    m = model()
    h = jr.normal(jr.key(1), (16, 6, 4))
    first = m.vectorize(h)
    for _ in range(3):
        assert jnp.allclose(m.vectorize(h), first, atol=1e-6)
    m2 = m.replace(w=2 * m.w)  # new weights, same shapes: the compiled call takes them
    assert jnp.allclose(m2.vectorize(h), 2 * first, atol=1e-5)

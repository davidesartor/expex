"""find_nan, summary and timing."""

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

import typax as tx
from typax import debug, f32, fn


def test_find_nan_points_at_the_line():
    def loss(p):
        q = p / p.sum()
        logq = jnp.log(q)  # -inf where q == 0
        return -(q * logq).sum()  # 0 * -inf = nan

    assert debug.find_nan(loss, jnp.array([0.5, 0.5])) is None
    r = debug.find_nan(loss, jnp.array([1.0, 0.0]))
    assert r is not None and r.op == "mul"
    assert "(q * logq)" in r.line and "test_debug.py" in r.source
    text = str(r)
    assert "NaN first appears at" in text and "-inf" in text
    assert any("jnp.log(q)" in s for s in r.infs)  # the inf's origin is reported too


def test_find_nan_in_inputs():
    r = debug.find_nan(lambda x: x * 2, jnp.array([1.0, jnp.nan]))
    assert r is not None and r.kind == "input"
    assert "already present in the inputs" in str(r)


def test_find_nan_inside_scan_iteration():
    def f(xs):
        def step(c, x):
            c = c - x
            return c, jnp.sqrt(c)  # nan once c < 0

        return lax.scan(step, 10.0, xs)[1]

    r = debug.find_nan(f, jnp.full(50, 3.0))
    assert r is not None and r.op == "sqrt"
    assert r.path == ["scan iteration 3"]
    assert "jnp.sqrt(c)" in r.line


def test_find_nan_inside_reverse_scan_and_while_and_cond():
    def rev(xs):
        return lax.scan(lambda c, x: (c - x, jnp.log(c - x)), 4.0, xs, reverse=True)[1]

    r = debug.find_nan(rev, jnp.array([9.0, 1.0, 1.0]))  # runs right-to-left
    assert r is not None and r.path == ["scan iteration 0"]

    def loop(x):
        return lax.while_loop(lambda s: s[0] < 5, lambda s: (s[0] + 1, jnp.log(s[1] - 1.0)), (0, x))[1]

    r = debug.find_nan(loop, 3.0)
    assert r is not None and r.op == "log" and r.path[0].startswith("while iteration")

    def branch(x):
        return lax.cond(x > 0, lambda v: jnp.sqrt(-v), lambda v: v, x)

    r = debug.find_nan(branch, 2.0)
    assert r is not None and r.op == "sqrt" and r.path == ["cond branch 1"]
    assert debug.find_nan(branch, -2.0) is None


def test_find_nan_through_fn_jit_grad_and_converted_loops():
    @jax.jit
    @fn
    def norm[D](x: f32[D]) -> f32[D]:
        return x / jnp.linalg.norm(x)

    r = debug.find_nan(norm, jnp.zeros(3))
    assert r is not None and r.op == "div"

    g = jax.grad(lambda x: jnp.sqrt(x).sum())
    assert debug.find_nan(g, jnp.array([1.0, 4.0])) is None
    assert debug.find_nan(lambda x: jnp.sqrt(x - 5).sum(), jnp.array([1.0])) is not None

    @fn
    def decay[T](xs: f32[T]) -> f32[T]:
        level = 5.0
        out = []
        for x in xs:
            level = level - x
            out.append(jnp.log(level))
        return jnp.asarray(out)

    r = debug.find_nan(decay, jnp.full(20, 2.0))
    assert r is not None and "scan iteration 2" in r.path and "jnp.log(level)" in r.line


def test_summary_counts_and_lints():
    def unrolled(x):
        for _ in range(100):
            x = jnp.sin(x) + 1
        return x

    s = debug.summary(unrolled, jnp.ones(3))
    assert s.ops == 200 and dict(s.by_op)["sin"] == 100
    assert any("unrolled" in w for w in s.warnings)
    assert "by source line" in str(s)

    @fn
    def staged[D](x: f32[D]) -> f32[D]:
        for _ in range(100):
            x = jnp.sin(x) + 1
        return x

    s2 = debug.summary(staged, jnp.ones(3))
    assert s2.top_level_ops < 10 and not s2.warnings


def test_summary_flags_large_constants_and_reports_intermediates():
    table = jnp.ones((600, 600))

    def f(x):
        return (table * x).sum()

    s = debug.summary(f, 2.0)
    assert any("captured as constants" in w for w in s.warnings)
    assert s.largest and s.largest[0][0] == 600 * 600 * 4


def test_timing():
    @fn
    def f[D](x: f32[D]) -> f32[D]:
        return jnp.tanh(x) * 2

    t = debug.timing(f, jnp.ones(8), repeat=3)
    assert t.trace > 0 and t.compile > 0 and t.run > 0 and t.ops >= 2
    assert "trace" in str(t) and "compile" in str(t)

    @jax.jit
    @tx.random
    @fn
    def r[D](x: f32[D]) -> f32[D]:
        return x + jr.normal(tx.key(), x.shape)

    assert debug.timing(r, jnp.ones(8), repeat=1).ops >= 1


def test_debug_tools_keep_static_arguments_static():
    from typax import Static

    @fn
    def rep[T](x: f32[T], n: Static[int], mode: Static[str] = "log") -> f32:
        for _ in range(n):  # a Python loop over a Static count
            x = jnp.log(x) if mode == "log" else x
        return x.sum()

    assert debug.summary(rep, jnp.ones(3), 2).ops >= 2
    assert debug.timing(rep, jnp.ones(3), n=2, repeat=1).ops >= 2
    r = debug.find_nan(rep, jnp.full(3, 0.5), 2)
    assert r is not None and r.op == "log"

    class M:
        @fn
        def call[T](self, x: f32[T], n: Static[int]) -> f32[T]:
            return jnp.sqrt(x - n)

    assert debug.find_nan(M().call, jnp.ones(3), 5) is not None
    assert debug.find_nan(M().call, jnp.ones(3), 0) is None


def test_shapes_concrete_and_symbolic():
    @fn
    def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
        scores = q @ k.swapaxes(-1, -2)
        diffs = scores[:, 1:] - scores[:, :-1]
        return jax.nn.softmax(scores, axis=-1)

    text = str(debug.shapes(attend))
    assert "# scores: f32[B, T, T]" in text and "# diffs: f32[B, T - 1, T]" in text
    assert "-> f32[B, T, T]" in text and "[symbolic B, D, T]" in text

    concrete = str(debug.shapes(attend, jnp.ones((2, 3, 4)), jnp.ones((2, 3, 4))))
    assert "# scores: f32[2, 3, 3]" in concrete

    @fn
    def rnn[T, D](xs: f32[T, D], w: f32[D, D]) -> f32[D]:
        h = jnp.zeros(xs.shape[1])
        for x in xs:
            pre = w @ h + x
            h = jnp.tanh(pre)
        return h

    text = str(debug.shapes(rnn))
    assert "# pre: f32[D]" in text and "# h: f32[D]" in text  # one iteration's shapes

    def plain(x):  # works on any Python function, given arguments
        y = x.sum(0)
        return y

    assert "# y: f32[5]" in str(debug.shapes(plain, jnp.ones((4, 5))))


def test_jax_checkpoint_on_methods_saves_memory_and_keeps_gradients():
    import typax as tx

    def make(policy):
        @tx.struct
        class Layer[D]:
            w: f32[D, D]

            def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
                h = jnp.tanh(x @ self.w)
                h = jnp.tanh(h @ self.w) * h
                return jnp.tanh(h @ self.w) + x

        if policy is not None:
            Layer.__call__ = jax.checkpoint(Layer.__call__, policy=policy)
        return Layer

    x = jnp.ones((128, 64))

    def loss(layers, x):
        for layer in layers:
            x = layer(x)
        return (x**2).mean()

    results = {}
    for name, policy in [
        ("plain", None),
        ("remat", jax.checkpoint_policies.nothing_saveable),
        ("dots", jax.checkpoint_policies.dots_saveable),
    ]:
        Layer = make(policy)
        layers = [Layer(jnp.eye(64) * 0.5 + 0.01 * i) for i in range(8)]
        grads = jax.jit(jax.grad(loss))(layers, x)
        results[name] = (debug.memory(jax.grad(loss), layers, x), grads)
    plain, low = results["plain"], results["remat"]
    assert low[0].live_peak < plain[0].live_peak / 2
    for a, b in zip(jax.tree.leaves(plain[1]), jax.tree.leaves(low[1])):
        assert jnp.allclose(a, b, atol=1e-6)
    assert "compiled:" in str(low[0]) and "alive at once" in str(low[0])

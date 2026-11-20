"""find_nan, summary and timing."""

import jax
import jax.numpy as jnp
from jax import lax

import barnacle as bn
from barnacle import debug, f32, fn


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
    @fn(jit=True)
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

    @fn(jit=True, random=True)
    def r[D](x: f32[D]) -> f32[D]:
        return x + jax.random.normal(bn.key(), x.shape)

    assert debug.timing(r, jnp.ones(8), repeat=1).ops >= 1


def test_debug_tools_keep_static_arguments_static():
    from barnacle import Static

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

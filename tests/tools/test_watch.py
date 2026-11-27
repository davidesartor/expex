"""expex.debug.watch: activation and gradient statistics of every typed call."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import expex
from expex import f32, fn
from expex.tools.watch import watch

sys.path.insert(0, str(Path(__file__).parents[2] / "examples"))
from transformer import Transformer  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    m = Transformer(jr.key(0), Vocab=20, D=16, F=32, blocks=3, heads=2)
    return m, jr.randint(jr.key(1), (4, 9), 0, 20)


def by_name(w):
    return {r.name: r for r in w.rows}


def check(stats, x):
    x = np.asarray(x, np.float64)
    assert stats["mean"] == pytest.approx(x.mean(), abs=1e-5 * np.abs(x).max())
    assert stats["std"] == pytest.approx(x.std(), rel=1e-3)
    assert stats["rms"] == pytest.approx(np.sqrt((x**2).mean()), rel=1e-3)
    assert stats["absmax"] == pytest.approx(np.abs(x).max(), rel=1e-5)


def test_rows_per_layer_with_exact_stats(setup):
    m, batch = setup
    w = watch(lambda m: m.loss.vectorize(batch).mean(), m)
    rows = by_name(w)
    assert [r.name for r in w.rows][:4] == [
        "Transformer.loss", "Transformer.__call__", "blocks[0].__call__", "blocks[0].norm1.__call__",
    ]  # fmt: skip
    assert {"blocks[2].attn.split#2", "blocks[2].mlp.__call__", "Attention.causal#2", "norm.__call__"} <= set(rows)
    assert [r.depth for r in w.rows[:5]] == [0, 1, 2, 3, 3]
    assert rows["blocks[1].norm2.__call__"].count == 4 * 8  # lifted twice: over the batch and T
    assert rows["Attention.causal#0"].shape == "f32[8, 8]" and rows["Attention.causal#0"].count == 4 * 2
    assert w.loss == pytest.approx(float(jax.vmap(m.loss)(batch).mean()), rel=1e-6)

    # the logits and d loss / d logits, computed without watch
    logits = jax.vmap(lambda t: m(t[:-1]))(batch)

    def loss_of_logits(z):
        logp = jax.nn.log_softmax(z)
        return -jnp.take_along_axis(logp, batch[:, 1:, None], axis=2).mean()

    check(rows["Transformer.__call__"].out, logits)
    check(rows["Transformer.__call__"].grad, jax.grad(loss_of_logits)(logits))
    assert rows["Transformer.loss"].grad["mean"] == pytest.approx(1 / 4)
    assert not w.flagged
    text = str(w)
    assert "blocks[1].attn.__call__" in text and "d loss / d output" in text and "\n!" not in text


def test_planted_scale_is_flagged(setup):
    m, batch = setup
    blocks = list(m.blocks)
    blocks[1] = blocks[1].replace(mlp=blocks[1].mlp.replace(w2=blocks[1].mlp.w2 * 1e4))
    w = watch(lambda m: m.loss.vectorize(batch).mean(), m.replace(blocks=blocks), growth=20.0)
    rows = by_name(w)
    assert any("activations growing" in f and "blocks[0].mlp" in f for f in rows["blocks[1].mlp.__call__"].flags)
    assert any("activations growing" in f for f in rows["blocks[1].__call__"].flags)
    assert any("gradient vanishing" in f for f in rows["blocks[2].attn.__call__"].flags)
    assert not any("activations" in f for f in rows["blocks[0].mlp.__call__"].flags)
    assert any("gradient exploding" in f for f in rows["blocks[0].mlp.__call__"].flags)  # 2 of 3 layers vanish
    assert w.options == {"grad_ratio": 100.0, "growth": 20.0}
    flagged = [line for line in str(w).splitlines() if line.startswith("!")]
    assert len(flagged) == len(w.flagged) and "blocks[1].mlp.__call__" in "".join(flagged)


@expex.struct
class Logs[D]:
    w: f32[D, D]

    def __call__(self, x: f32[D]) -> f32[D]:
        return jnp.log(x @ self.w)


@expex.struct
class Net[D]:
    layers: list[Logs[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for layer in self.layers:
            x = layer(jnp.abs(x) + 1)
        return x


def test_log_of_negative_is_flagged():
    net = Net([Logs(jnp.eye(3)), Logs(-jnp.eye(3)), Logs(jnp.eye(3))])
    w = watch(lambda net, x: net.vectorize(x).sum(), net, jnp.ones((5, 3)))
    rows = by_name(w)
    assert rows["layers[1].__call__"].out["nan"] == 15
    assert rows["layers[1].__call__"].flags[0] == "output not finite: 15 NaN, 0 inf"
    assert rows["layers[2].__call__"].out["nan"] == 15  # NaN propagates
    assert rows["layers[0].__call__"].out["nan"] == 0
    assert rows["layers[0].__call__"].flags == ["gradient not finite: 15 NaN, 0 inf"]  # back through layers[1]


def test_staged_loop_is_one_row(setup):
    m, batch = setup
    w = watch(lambda m: m.loss.vectorize(batch).mean(), m, unroll=False)
    rows = by_name(w)
    assert "Block.__call__" in rows and "blocks[0].__call__" not in rows
    assert rows["Block.__call__"].count == 3 * 4  # every layer and every batch element
    unrolled = by_name(watch(lambda m: m.loss.vectorize(batch).mean(), m))
    assert rows["Transformer.__call__"].grad["rms"] == pytest.approx(unrolled["Transformer.__call__"].grad["rms"], rel=1e-4)  # fmt: skip


@fn
def scale[N](x: f32[N], s: float) -> f32[N]:
    return x * s


@fn(jit=True)
def square[N](x: f32[N]) -> f32[N]:
    return scale(x, 1.0) ** 2


def test_functions_aux_and_inner_jit():
    x = jnp.arange(4.0)

    def loss(x):
        y = scale(scale(x, 2.0), 3.0)
        return square(y).sum(), {"y": y}

    square(x)  # compiled before: traced again while watching
    w = watch(loss, x)
    rows = by_name(w)
    assert [r.name for r in w.rows] == ["scale#0", "scale#1", "square", "scale#2"]
    np.testing.assert_allclose(jax.grad(lambda x: square(x).sum())(x), 2 * x)  # no hooked trace reused
    assert rows["scale#2"].depth == 1 and rows["square"].depth == 0
    check(rows["scale#1"].out, 6 * x)
    check(rows["scale#1"].grad, 2 * 6 * x)  # d sum(y²) / dy
    check(rows["scale#0"].grad, 3 * 2 * 6 * x)
    assert w.loss == pytest.approx(float((36 * x**2).sum()))
    np.testing.assert_allclose(w.aux["y"], 6 * x)


def test_not_a_scalar():
    with pytest.raises(TypeError, match="float scalar loss"):
        watch(lambda x: scale(x, 2.0), jnp.ones(3))


def test_html(setup, tmp_path):
    m, batch = setup
    blocks = list(m.blocks)
    blocks[2] = blocks[2].replace(mlp=blocks[2].mlp.replace(w2=blocks[2].mlp.w2 * 1e4))
    w = watch(lambda m: m.loss.vectorize(batch).mean(), m.replace(blocks=blocks))
    page = w.save(tmp_path / "w.html").read_text()
    assert page.startswith("<!doctype html>") and "http" not in page  # nothing from a CDN
    assert page.count('class="flag"') == len(w.flagged) > 0
    assert page.count("<svg") > 20 and "prefers-color-scheme:dark" in page
    assert "blocks[2].mlp.__call__" in w._repr_html_()


def test_repeated_modules_without_gradients(setup):
    # the loss does not reach the blocks (only `aux` does): their group of layers has
    # no gradient statistics, and statistics.median([]) raised StatisticsError
    m, batch = setup

    def loss(m):
        out = m.vectorize(batch[:, :-1])  # every block runs
        return (m.embed**2).mean(), out

    w = watch(loss, m)
    blocks = [r for r in w.rows if r.name.startswith("blocks[") and r.name.endswith("].__call__")]
    assert len(blocks) == 3 and all(r.grad is None for r in blocks)
    assert not any("gradient" in f for r in blocks for f in r.flags)
    assert "blocks[1].__call__" in w._repr_html_() and "blocks[1].__call__" in str(w)

"""`expex.inspect`: one page with a tab per tool over one module tree, and selection wired between them."""

import base64
import gzip
import importlib.util
import json
import pathlib
import re
import sys

import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import f32

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tf():
    spec = importlib.util.spec_from_file_location("inspect_transformer", ROOT / "examples" / "transformer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def model(tf):
    return tf.Transformer(jr.key(0), Vocab=50, D=32, F=64, blocks=2, heads=4)


@pytest.fixture(scope="module")
def tokens():
    return jr.randint(jr.key(1), (12,), 0, 50)


@pytest.fixture(scope="module")
def page(model, tokens):
    return expex.inspect(model, tokens, loss=lambda m, x: m.loss(x))


def data(html: str) -> dict:
    blob = re.search(r'<script type="application/json" id="expex-data">(.*?)</script>', html, re.S)
    assert blob is not None
    return json.loads(blob.group(1))


def diagram_page(html: str) -> str:
    return gzip.decompress(base64.b64decode(data(html)["diagram"])).decode("utf-8")


def test_every_tab_is_built(page):
    assert page.errors == {}
    assert page.summary is not None and page.diagram is not None and page.profile is not None
    assert page.watch is not None and page.watch.loss is not None and page.watch.rows[0].grad is not None
    assert page.sizes["T"] == 12  # from the inputs
    assert "summary, diagram, cost, activations, profile" in repr(page)


def test_page_has_tabs_and_panels(page):
    html = page.html()
    assert html.startswith("<!doctype html>")
    for tab in ("summary", "diagram", "cost", "activations", "profile"):
        assert f'role="tab" id="t-{tab}"' in html and f'id="tab-{tab}"' in html
    assert "http://" not in html.split("expex-data")[0] and "https://" not in html.split("expex-data")[0]  # no CDN


def test_tools_render_their_own_fragments(page):
    html = page.html()
    assert 'class="txs"' in html  # the summary's tree table
    assert re.search(r'id="txw\d+"', html)  # watch's
    assert 'class="txp"' in html  # profile's fragment
    assert html.count('class="txc"') == len(page.costs)  # one cost card per estimate


def test_module_tree_is_shared(page):
    mods = {m["path"]: m["cls"] for m in data(page.html())["modules"]}
    assert mods == {"": "Transformer", "blocks[*]": "Block", "blocks[*].norm1": "LayerNorm", "blocks[*].attn": "Attention",
                    "blocks[*].norm2": "LayerNorm", "blocks[*].mlp": "MLP", "norm": "LayerNorm"}  # fmt: skip


def test_rows_carry_their_module_paths(page):
    html = page.html()
    assert 'data-path="blocks[*].attn"' in html  # summary rows
    assert 'data-path="blocks[1].attn"' in html  # watch rows, per layer
    cards = re.findall(r'class="card" data-paths="([^"]*)"', html)
    paths = [json.loads(c.replace("&quot;", '"')) for c in cards]
    assert ["blocks[*].norm1", "blocks[*].norm2", "norm"] in paths  # equal estimates share a card


def test_cost_cards_have_formulas_and_numbers(page):
    by = {c.name: (c, sym) for _, c, sym, _ in page.costs}
    c, sym = by["Attention.__call__"]
    assert c.flops.value is not None and sym is not None and "T" in sym.flops.names()


def test_diagram_is_embedded_with_paths_and_hooks(page):
    inner = diagram_page(page.html())
    boxes = json.loads(re.search(r"const M = (.*?);\n", inner, re.S).group(1))["boxes"]
    assert {b["path"] for b in boxes if b["k"] == "call"} >= {"blocks[*]", "blocks[*].attn", "blocks[*].mlp", "norm"}
    assert 'expex === "select"' in inner and 'expex === "theme"' in inner  # selection and theme from the page
    assert 'expex: "select", path, from: "diagram"' in inner  # and clicks sent back


def test_selection_is_wired(page):
    html = page.html()
    js = html.split("<script>")[-1]
    assert "function select(" in js
    for target in ('#tab-summary tr[data-path]', '#tab-cost .card', '#tab-profile details[data-name]', 'postMessage'):
        assert target in js
    assert 'addEventListener("message"' in js and "DecompressionStream" in js
    assert "prefers-color-scheme:dark" in html and ":root[data-theme=dark]" in html and 'id="theme"' in html


def test_forward_only_and_no_inputs(model, tokens, tf):
    fwd = expex.inspect(model, tokens)
    assert fwd.watch is not None and fwd.watch.loss is None and all(r.grad is None for r in fwd.watch.rows)
    assert "forward only" in fwd.html() and fwd.sizes["T"] == 12
    bare = expex.inspect(model, T=16)
    assert bare.watch is None and "Give example inputs" in bare.html() and bare.sizes["T"] == 16
    built = expex.inspect(tf.Transformer, Vocab=50, D=32, F=64, blocks=2, T=8)
    assert built.errors == {} and built.summary.total.params == model_params(model)


def model_params(m) -> int:
    import jax

    return sum(x.size for x in jax.tree.leaves(m))


def test_a_failing_tool_leaves_its_tab_with_the_error(model):
    bad = expex.inspect(model, jnp.ones((12, 3)))  # tokens must be Int[T]: the call fails, the rest is drawn
    assert "activations" in bad.errors and isinstance(bad.errors["activations"], expex.ShapeError)
    assert bad.summary is not None and 'class="txe"' in bad.html()


def test_save_and_notebook(page, tmp_path):
    path = page.save(tmp_path / "model.html")
    assert path.read_text(encoding="utf-8") == page.html()
    assert page._repr_html_().startswith("<iframe srcdoc=")


def test_not_a_struct():
    @expex.fn
    def f[N](x: f32[N]) -> f32[N]:
        return x

    with pytest.raises(TypeError, match="not an @expex.struct"):
        expex.inspect(f)

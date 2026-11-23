"""Architecture diagrams: calls as boxes, loops as panels, shapes under the boxes, drawn as SVG."""

import importlib.util
import json
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import f32
from expex import _diagram
from expex import function as _fn
from expex._diagram import Diagram, diagram

ROOT = pathlib.Path(__file__).resolve().parent.parent
NS = "{http://www.w3.org/2000/svg}"


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def transformer():
    return load("diagram_transformer", ROOT / "examples" / "transformer.py")


@pytest.fixture(scope="module")
def mamba():
    return load("diagram_mamba", ROOT / "examples" / "mamba" / "mamba.py")


def labels(d: Diagram, kind: str | None = None) -> list[str]:
    return [label for _, k, label, _ in d.nodes if kind in (None, k)]


def node(d: Diagram, prefix: str) -> tuple[str, str, str, int]:
    (n,) = [n for n in d.nodes if n[2].startswith(prefix)]
    return n


def edge(d: Diagram, a: str, b: str) -> str:
    """The type on the edge from the node labelled ``a...`` to ``b...``."""
    src, dst = node(d, a)[0], node(d, b)[0]
    (t,) = [t for u, v, t in d.edges if (u, v) == (src, dst)]
    return t


def box_of(d: Diagram, prefix: str) -> str:
    return d.boxes[f"b{node(d, prefix)[3]}"]


def text(el: ET.Element) -> list[str]:
    return ["".join(t.itertext()) for t in el.iter(NS + "text")]


def check_svg(d: Diagram) -> ET.Element:
    """The SVG parses; every node is a group with a box and its label; every edge has a path and an arrow head."""
    root = ET.fromstring(d.svg)
    assert root.tag == NS + "svg" and root.get("viewBox").startswith("0 0 ")
    groups = {g.get("data-id"): g for g in root.iter(NS + "g") if g.get("class", "").startswith("nd")}
    assert set(groups) == {n[0] for n in d.nodes}
    for n in d.nodes:
        g = groups[n[0]]
        assert g.find(NS + "rect") is not None and text(g)
    top = [el.tag for el in root]  # the edges are drawn at the top level, the internals' arrows in their stacks
    assert top.count(NS + "polygon") == top.count(NS + "path") == len(d.edges)
    assert "cdn" not in d.svg and "mermaid" not in d.svg.lower()
    return root


def drawn(d: Diagram, prefix: str) -> tuple[list[str], tuple[float, float], float]:
    """A node as drawn: its texts, its box's (top, bottom), and the y of the last text (the shape under it)."""
    root = ET.fromstring(d.svg)
    (g,) = [g for g in root.iter(NS + "g") if g.get("data-id") == node(d, prefix)[0]]
    rect = g.findall(NS + "rect")[-1]
    y, h = float(rect.get("y")), float(rect.get("height"))
    return text(g), (y, y + h), float(g.findall(NS + "text")[-1].get("y"))


def titles(d: Diagram) -> list[str]:
    root = ET.fromstring(d.svg)
    return [text(g)[0] for g in root.iter(NS + "g") if g.get("class") == "pt"]


def test_transformer_depth_1(transformer):
    d = diagram(transformer.Transformer.__call__)
    assert labels(d, "in") == ["tokens: i32[T]"]
    assert labels(d, "out") == ["return f32[T, Vocab]"]
    assert sorted(labels(d, "op")) == ["index", "matmul"]
    assert node(d, "Block.__call__")[2] == "Block.__call__\nf32[T, D] → f32[T, D]"
    assert node(d, "LayerNorm")[2] == "LayerNorm.__call__ · vectorized\nf32[D] → f32[D]"
    assert box_of(d, "Block.__call__") == "× L"  # the loop over self.blocks: one scan
    assert list(d.boxes.values()) == ["× L"]
    assert edge(d, "tokens", "index") == "i32[T]"
    assert edge(d, "index", "Block") == "f32[T, D]"
    assert edge(d, "Block", "LayerNorm") == "f32[T, D]"
    assert edge(d, "matmul", "return") == "f32[T, Vocab]"
    assert len(d.edges) == 5
    assert str(d) == d.text and not hasattr(d, "mermaid")
    check_svg(d)
    texts, (top, bottom), shape_y = drawn(d, "matmul")
    assert texts == ["matmul", "f32[T, Vocab]"] and bottom < shape_y < bottom + 16  # the shape just under its box
    assert drawn(d, "tokens")[0] == ["tokens", "i32[T]"] and drawn(d, "return")[0] == ["return", "f32[T, Vocab]"]
    assert drawn(d, "Block")[0] == ["Block  × L"]  # one block, drawn as a stack: same shape in and out
    assert drawn(d, "LayerNorm")[0] == ["LayerNorm", "vectorized"]
    assert "Transformer" in text(ET.fromstring(d.svg))[0] and "L = len(blocks)" in d.svg
    assert "Component internals" in d.svg and {"Block", "LayerNorm"} <= set(text(ET.fromstring(d.svg)))
    assert _fn.CALL_HOOK.get() is None


def test_transformer_depth_2(transformer):
    d = diagram(transformer.Transformer.__call__, depth=2)
    assert {"Block.__call__", "× L", "LayerNorm.__call__"} <= set(d.boxes.values())
    assert box_of(d, "Attention.__call__") == "Block.__call__"
    assert box_of(d, "MLP.__call__") == "Block.__call__"
    assert [label.split("\n")[0] for label in labels(d, "call")] == [
        "LayerNorm.__call__ · vectorized", "Attention.__call__", "LayerNorm.__call__ · vectorized", "MLP.__call__ · vectorized",
    ]  # fmt: skip
    attn, mlp = node(d, "Attention")[0], node(d, "MLP")[0]
    assert any(u == attn for u, _, _ in d.edges) and any(v == mlp for _, v, _ in d.edges)
    assert labels(d, "op").count("+") == 2  # the residual connections
    plus = [n[0] for n in d.nodes if n[2] == "+"]
    assert {(u, v) for u, v, _ in d.edges if v == plus[0]} == {(node(d, "index")[0], plus[0]), (attn, plus[0])}
    assert titles(d) == ["Block  × L", "LayerNorm  vectorized"]  # the loop and the call in it: one panel
    root = check_svg(d)
    panels = [g for g in root.iter(NS + "g") if g.get("class", "").startswith("pn")]
    assert len(panels) == 2 and all(g.find(NS + "rect").get("class") == "f-panel k-frame" for g in panels)


def test_attention(transformer):
    d = diagram(transformer.Attention.__call__)
    calls = labels(d, "call")
    assert sum(c.startswith("Attention.split\nf32[T, D] → f32[4, T, D // 4]") for c in calls) == 3
    assert any(c.startswith("Attention.causal · vectorized") for c in calls)
    assert labels(d, "op") == ["matmul"] * 4  # q, k, v and the output projection
    causal = [n[0] for n in d.nodes if n[2].startswith("Attention.causal")][0]
    into = [t for _, v, t in d.edges if v == causal]
    assert into == ["f32[4, T, D // 4]"] * 3
    small = transformer.Attention.init(jr.key(0), D=8, heads=2)  # sizes from the instance
    assert labels(diagram(small.__call__), "call")[0] == "Attention.split\nf32[T, 8] → f32[2, T, 4]"


def test_sizes_and_bound_methods(transformer):
    d = diagram(transformer.Transformer.__call__, T=128, blocks=6)
    assert labels(d, "in") == ["tokens: i32[128]"] and "× 6" in d.boxes.values()
    model = transformer.Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=4, heads=4)
    d = diagram(model.__call__)
    assert labels(d, "out") == ["return f32[T, 50]"] and "× 4" in d.boxes.values()
    with pytest.raises(TypeError, match="unknown size or parameter `Q`"):
        diagram(transformer.Transformer.__call__, Q=3)


def test_mamba(mamba):
    d = diagram(mamba.Mamba.__call__)
    assert box_of(d, "Block.__call__") == "× L"
    assert labels(d, "out") == ["return f32[T, V]"]
    assert node(d, "RMSNorm")[2] == "RMSNorm.__call__ · vectorized\nf32[D] → f32[D]"
    root = check_svg(d)
    (g,) = [g for g in root.iter(NS + "g") if g.get("data-id") == node(d, "RMSNorm")[0]]
    assert g.find(NS + "rect").get("class") == "f-norm k-edge"  # norms are yellow


def test_mixer_time_loop(mamba):
    d = diagram(mamba.Mixer.__call__)
    scan = node(d, "selective_scan")
    assert scan[2].endswith("→ f32[T, E]") and node(d, "Mixer.conv")[2] == "Mixer.conv\nf32[T, E] → f32[T, E]"
    conv = node(d, "Mixer.conv")[0]
    assert [t for _, v, t in d.edges if v == conv] == ["f32[T, E]"]  # the slice xz[:, :E] is passed through
    assert edge(d, "Mixer.conv", "selective_scan") == "f32[T, E]"
    d2 = diagram(mamba.Mixer.__call__, depth=2)
    loop = [b for b, label in d2.boxes.items() if label == "× T"]
    assert loop and "selective_scan" in d2.boxes.values()
    (inside,) = [n for n in d2.nodes if f"b{n[3]}" == loop[0]]
    assert inside[2] == "matmul"  # h @ C[t]
    assert titles(d2) == ["Mixer.conv", "selective_scan", "loop  × T"]
    check_svg(d2)
    texts = text(ET.fromstring(d.svg))
    assert "f32[T, E]" in texts  # the slice xz[:, :E] enters Mixer.conv: written by the arrow
    assert texts.count("+") == 1  # selective_scan(...) + u * skip
    check_svg(d)


@expex.fn
def unit[T, D](x: f32[T, D]) -> f32[T, D]:
    return x / jnp.linalg.norm(x)


@expex.fn
def gate[T, D](x: f32[T, D], w: f32[D, D]) -> f32[T, D]:
    if x.sum() > 0:
        y = unit(x @ w)
    else:
        y = x * 2.0
    while y.max() > 1.0:
        y = y / 2
    return y


def test_branches_and_while():
    d = diagram(gate)
    assert labels(d, "in") == ["x: f32[T, D]", "w: f32[D, D]"]
    assert box_of(d, "unit") == "then" and {"if", "then", "while"} <= set(d.boxes.values())
    assert edge(d, "w", "matmul") == "f32[D, D]"
    assert edge(d, "unit", "return") == "f32[T, D]"
    assert edge(d, "x", "return") == "f32[T, D]"  # the else branch passes x through
    check_svg(d)
    assert {"if", "then", "while"} <= {t.split("  ")[0] for t in titles(d)}  # the else branch has nothing to draw


def test_page(transformer, tmp_path):
    d = diagram(transformer.Transformer.__call__)
    path = d.save(tmp_path / "t.html")
    page = path.read_text()
    assert "<script src" not in page and "cdn" not in page.lower()  # everything is inline
    blob = re.search(r"const M = (.*);\n", page).group(1)
    data = json.loads(blob)
    assert "</script" not in blob and data["name"] == "Transformer.__call__"
    calls = [b for b in data["boxes"] if b["k"] == "call"]
    assert {b["name"] for b in calls} >= {"Block.__call__", "Attention.__call__", "Attention.causal", "MLP.__call__"}
    attn = next(i for i, b in enumerate(data["boxes"]) if b["name"] == "Attention.__call__")
    assert data["boxes"][attn]["o"] and data["boxes"][attn]["loc"].endswith("transformer.py:52")
    assert "def __call__" in data["sources"][data["boxes"][attn]["src"]]
    assert data["start"] == data["depths"][0] and len(data["depths"]) >= 3  # a view per depth
    assert data["svgs"][data["start"]] == d.svg
    block = next(i for i, b in enumerate(data["boxes"]) if b["name"] == "Block.__call__")
    opened = data["clicks"][data["start"]][str(block)]  # a click on Block opens it in place
    assert f'data-box="{attn}"' in data["svgs"][opened] and f'data-box="{attn}"' not in d.svg.split("Component internals")[0]
    assert data["clicks"][opened][str(block)] == data["start"]  # and a click on its title folds it
    assert set(data["colors"]) >= {"bg", "s0", "norm", "h4", "err"}  # the dark colours
    assert d._repr_html_().startswith("<iframe srcdoc=")


def test_save(transformer, tmp_path):
    d = diagram(transformer.Transformer.__call__)
    assert d.save(tmp_path / "t.svg").read_text() == d.svg
    png = d.save(tmp_path / "t.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="cannot save a diagram as '.pdf'"):
        d.save(tmp_path / "t.pdf")


def test_png_fallbacks(transformer, tmp_path, monkeypatch):
    d = diagram(transformer.Transformer.__call__)
    monkeypatch.setattr(_diagram, "_chrome", lambda: None)
    monkeypatch.setitem(sys.modules, "playwright", None)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    if importlib.util.find_spec("matplotlib") is not None:  # drawn again by matplotlib
        assert d.save(tmp_path / "m.png").read_bytes()[:4] == b"\x89PNG"
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(RuntimeError, match="needs a browser"):
        d.save(tmp_path / "none.png")


# ---------------------------------------------------------------- measured time


@pytest.fixture(scope="module")
def timed(transformer):
    model = transformer.Transformer.init(jr.key(0), Vocab=64, D=32, F=64, blocks=2, heads=4)
    return expex.diagram.timed(model.__call__, jr.randint(jr.key(1), (16,), 0, 64))


def test_timed(timed):
    d, t = timed, timed.model["timing"]
    assert t["how"] == "profiler"  # the CPU profiler's events are matched to scopes through the compiled HLO
    assert labels(d, "in") == ["tokens: i32[16]"] and "× 2" in d.boxes.values()
    assert node(d, "Block.__call__")[2].startswith("Block.__call__\nf32[16, 32] → f32[16, 32]\n")
    loop = next(i for i, b in enumerate(d.model["boxes"]) if b["l"] == "× 2")
    block = next(i for i, b in enumerate(d.model["boxes"]) if b["l"] == "Block.__call__")
    secs, share = d.timings[f"b{block}"]
    assert 0 < secs and 0 < share <= d.timings[f"b{loop}"][1] <= 1
    top = [b["t"] for b in d.model["boxes"] if b["p"] < 0]
    assert sum(top) + t["outside"] == pytest.approx(t["total"])
    assert re.search(r"\n[\d.]+ (ms|µs|s) · [\d.]+%$", node(d, "Block.__call__")[2])
    texts, _, _ = drawn(d, "Block.__call__")
    assert re.fullmatch(r"[\d.]+ (ms|µs|s) · [\d.]+%", texts[-1])  # the time in the box
    root = check_svg(d)
    (g,) = [g for g in root.iter(NS + "g") if g.get("data-id") == node(d, "Block.__call__")[0]]
    assert g.find(NS + "rect").get("class") == f"f-h{_diagram.heat(d.model['boxes'][block]['s'])} k-edge"
    assert "measured with jax.profiler" in d.svg and "≥ 40%" in d.svg  # the header and the heat legend
    d2 = d.at(2)
    check_svg(d2)
    assert re.fullmatch(r"Block  × 2  [\d.]+ (ms|µs|s) · [\d.]+%", titles(d2)[0])
    assert "measured with jax.profiler" in d.text and "Block.__call__" in d.text and "(outside the calls)" in d.text
    assert _fn.CALL_HOOK.get() is None


def test_timed_loops_add_up_iterations(transformer):
    """A staged loop's body is one box, timed over all its iterations."""
    runs = {}
    for n in (2, 4):
        model = transformer.Transformer.init(jr.key(0), Vocab=64, D=32, F=64, blocks=n, heads=4)
        d = expex.diagram.timed(model.__call__, jr.randint(jr.key(1), (16,), 0, 64))
        (block,) = [b for b in d.model["boxes"] if b["l"] == "Block.__call__"]
        runs[n] = block["n"]  # XLA instructions run per call, in the block
    assert runs[4] == pytest.approx(2 * runs[2], abs=2)


def test_timed_estimate(transformer, monkeypatch):
    def no_profiler(*a, **k):
        raise RuntimeError("no profiler here")

    monkeypatch.setattr(_diagram, "_profile", no_profiler)
    model = transformer.Transformer.init(jr.key(0), Vocab=64, D=32, F=64, blocks=2, heads=4)
    d = expex.diagram.timed(model.__call__, jr.randint(jr.key(1), (16,), 0, 64))
    t = d.model["timing"]
    assert t["how"] == "estimate" and "no profiler here" in t["why"]
    assert "ESTIMATED from FLOPs" in d.text and "FLOP" in node(d, "Block.__call__")[2]
    block = node(d, "Block.__call__")[0]
    assert d.timings[block][1] > 0.5  # 2 blocks of 32×32 and 32×64 matmuls dominate
    check_svg(d)
    assert "ESTIMATED from FLOPs" in d.svg


def test_timed_page(timed, tmp_path):
    page = timed.save(tmp_path / "t.html").read_text()
    data = json.loads(re.search(r"const M = (.*);\n", page).group(1))
    assert data["timing"]["how"] == "profiler" and all("f-h" in svg for svg in data["svgs"].values())


# ---------------------------------------------------------------- shape errors


@expex.fn
def proj[T, D, E](x: f32[T, D], w: f32[D, E]) -> f32[T, E]:
    return x @ w


@expex.fn
def broken[T, D](x: f32[T, D], w: f32[D, D]) -> f32[T, D]:
    h = unit(proj(x, w))
    return proj(h, w[:-1])  # w loses a row: proj's D does not match


@pytest.fixture(scope="module")
def buggy(tmp_path_factory):
    """The example transformer with a bug: the MLP called without .vectorize."""
    src = (ROOT / "examples" / "transformer.py").read_text()
    src = src.replace("x + self.mlp.vectorize(self.norm2.vectorize(x))", "x + self.mlp(self.norm2.vectorize(x))")
    path = tmp_path_factory.mktemp("buggy") / "buggy_transformer.py"
    path.write_text(src)
    return load("diagram_buggy", path)


def test_error_diagram(buggy):
    model = buggy.Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=3, heads=4)
    with pytest.raises(expex.ShapeError, match=r"MLP.__call__\(\): argument `x`") as e:
        model(jr.randint(jr.key(1), (24,), 0, 50))
    assert "e.diagram" not in str(e.value)  # the message is unchanged; the hint is a note
    assert e.value.__notes__[-1] == '(see e.diagram.save("error.html"))'
    d = e.value.diagram
    assert isinstance(d, Diagram) and "_todo" in vars(d)  # built only when used
    assert d.failed == node(d, "✗ MLP.__call__")[0]
    assert node(d, "✗ MLP.__call__")[2] == "✗ MLP.__call__\nreceived f32[24, 32]\ndeclared x: f32[D]"
    assert box_of(d, "✗ MLP.__call__") == "Block.__call__"  # opened to show the failing call
    names = {n[0]: n[2].split("\n")[0] for n in d.nodes}
    assert sorted(names[n] for n in d.feeds) == [
        "+", "Attention.__call__", "LayerNorm.__call__ · vectorized", "LayerNorm.__call__ · vectorized", "index", "tokens: i32[24]"
    ]  # fmt: skip
    assert labels(d, "out") == []  # the trace stopped: nothing returned
    root = check_svg(d)
    texts, _, _ = drawn(d, "✗ MLP.__call__")
    assert texts == ["✗ MLP", "received f32[24, 32]", "declared x: f32[D]"]
    (g,) = [g for g in root.iter(NS + "g") if g.get("data-id") == d.failed]
    assert g.find(NS + "rect").get("class") == "f-err k-errline"
    red = [el for el in root if el.tag == NS + "path" and el.get("class") == "k-errline"]
    assert len(red) == sum(1 for u, v, _ in d.edges if u in d.feeds and (v in d.feeds or v == d.failed))
    assert titles(d) == ["Block  ✗ fails inside"]
    assert "✗ received f32[24, 32], declared x: f32[D]" in d.text
    assert "path:  tokens: i32[24] → index → LayerNorm.__call__ · vectorized → Attention.__call__ → + → LayerNorm.__call__ · vectorized → ✗ MLP.__call__" in d.text
    assert expex.diagram.explain(e.value) is d
    assert _fn.CALL_HOOK.get() is None


def test_explain_function():
    x, w = jnp.ones((5, 3)), jnp.ones((3, 3))
    d = expex.diagram.explain(broken, x, w)
    fail = node(d, "✗ proj")
    assert fail[2] == "✗ proj\nreceived f32[2, 3]\ndeclared w: f32[D, E]"
    assert d.failed == fail[0] and "argument `w` does not match" in d.error
    assert edge(d, "unit", "✗ proj") == "f32[5, 3]"  # the arguments it received
    assert edge(d, "w", "✗ proj") == "f32[2, 3]"  # w[:-1] is passed through: the type it enters with
    assert {n[2].split("\n")[0] for n in d.nodes if n[0] in d.feeds} == {"x: f32[5, 3]", "w: f32[3, 3]", "proj", "unit"}
    root = check_svg(d)
    assert "f32[2, 3]" in text(root)  # w[:-1] enters the failing call: written by its arrow
    with pytest.raises(expex.ShapeError) as e:
        broken(x, w)
    assert e.value.diagram.nodes == d.nodes
    ok = expex.diagram.explain(gate, x, w)  # nothing fails: the plain diagram on these shapes
    assert ok.error is None and ok.failed is None and labels(ok, "out") == ["return f32[5, 3]"]
    with pytest.raises(ValueError, match="no diagram"):
        expex.diagram.explain(ValueError("x"))


def test_explain_root_fails():
    d = expex.diagram.explain(proj, jnp.ones((5, 3)), jnp.ones((4, 3)))
    assert labels(d, "op") == ["✗ ShapeError"] and d.failed == node(d, "✗ ShapeError")[0]
    assert "proj(): argument `w`" in d.error
    check_svg(d)


def test_no_cost_on_success(monkeypatch):
    calls = []
    monkeypatch.setattr(_diagram, "attach", lambda *a: calls.append(a))
    gate(jnp.ones((5, 3)), jnp.ones((3, 3)))
    jax.jit(gate)(jnp.ones((5, 3)), jnp.ones((3, 3)))
    assert calls == []
    with pytest.raises(expex.ShapeError):
        broken(jnp.ones((5, 3)), jnp.ones((3, 3)))
    assert len(calls) == 1  # once, by the outermost call


def test_error_page(tmp_path):
    with pytest.raises(expex.ShapeError) as e:
        jax.jit(broken)(jnp.ones((5, 3)), jnp.ones((3, 3)))
    page = e.value.diagram.save(tmp_path / "e.html").read_text()
    data = json.loads(re.search(r"const M = (.*);\n", page).group(1))
    assert data["error"]["declared"] == "w: f32[D, E]" and data["error"]["got"] == "f32[2, 3]"
    assert [b["err"] for b in data["boxes"] if b["err"]] == [2]
    assert data["error"]["message"].startswith("proj(): argument `w`")

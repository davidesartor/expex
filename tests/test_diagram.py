"""Architecture diagrams: calls as nodes, loops as boxes, shapes on the edges."""

import importlib.util
import json
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import typax as tx
from typax import _fn, f32
from typax import _diagram
from typax._diagram import Diagram, diagram

ROOT = pathlib.Path(__file__).resolve().parent.parent


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
    """The label of the edge from the node labelled ``a...`` to ``b...``."""
    src, dst = node(d, a)[0], node(d, b)[0]
    (t,) = [t for u, v, t in d.edges if (u, v) == (src, dst)]
    return t


def box_of(d: Diagram, prefix: str) -> str:
    return d.boxes[f"b{node(d, prefix)[3]}"]


_LINE = re.compile(
    r'(flowchart TD|end|subgraph b\d+ \["[^"\[\]]*"\]|[bn]\d+(\["|\(\["|\[/"|\[\\")[^"\[\]]*("\]|"\]\)|"/\]|"\\\])'
    r'|[bn]\d+ -->\|"[^"\[\]|]*"\| [bn]\d+|classDef \w+ [\w:#,.-]+|class [bn\d,]+ \w+|style b\d+ [\w:#,. -]+'
    r"|linkStyle [\d,]+ [\w:#,.-]+)",
    re.S,
)


def check_mermaid(text: str) -> None:
    """Every statement is one Mermaid knows, and labels hold no raw brackets or quotes."""
    lines = re.split(r"\n(?=\s*(?:flowchart|end|subgraph|[bn]\d+[\[(\s]|classDef|class|style|linkStyle))", text)
    for line in lines:
        assert _LINE.fullmatch(line.strip()), line
    ids = set(re.findall(r"^\s*([bn]\d+)[\[(]", text, re.M)) | set(re.findall(r"subgraph (b\d+)", text))
    for u, v in re.findall(r"([bn]\d+) -->\|[^|]*\| ([bn]\d+)", text):
        assert u in ids and v in ids


def test_transformer_depth_1(transformer):
    d = diagram(transformer.Transformer.__call__)
    assert labels(d, "in") == ["tokens: i32[T]"]
    assert labels(d, "out") == ["return f32[T, Vocab]"]
    assert sorted(labels(d, "op")) == ["index", "matmul"]
    assert node(d, "Block.__call__")[2] == "Block.__call__\nf32[T, D] → f32[T, D]"
    assert node(d, "LayerNorm")[2] == "LayerNorm.__call__ · lift\nf32[D] → f32[D]"
    assert box_of(d, "Block.__call__") == "× L"  # the loop over self.blocks: one scan
    assert list(d.boxes.values()) == ["× L"]
    assert edge(d, "tokens", "index") == "i32[T]"
    assert edge(d, "index", "Block") == "f32[T, D]"
    assert edge(d, "Block", "LayerNorm") == "f32[T, D]"
    assert edge(d, "matmul", "return") == "f32[T, Vocab]"
    assert len(d.edges) == 5
    assert d.mermaid.startswith("flowchart TD\n") and str(d) == d.mermaid
    assert "Block.#95;#95;call#95;#95;\nf32#91;T, D#93; → f32#91;T, D#93;" in d.mermaid
    check_mermaid(d.mermaid)
    assert _fn.CALL_HOOK.get() is None


def test_transformer_depth_2(transformer):
    d = diagram(transformer.Transformer.__call__, depth=2)
    assert {"Block.__call__", "× L", "LayerNorm.__call__"} <= set(d.boxes.values())
    assert box_of(d, "Attention.__call__") == "Block.__call__"
    assert box_of(d, "MLP.__call__") == "Block.__call__"
    assert [label.split("\n")[0] for label in labels(d, "call")] == [
        "LayerNorm.__call__ · lift", "Attention.__call__", "LayerNorm.__call__ · lift", "MLP.__call__ · lift",
    ]  # fmt: skip
    attn, mlp = node(d, "Attention")[0], node(d, "MLP")[0]
    assert any(u == attn for u, _, _ in d.edges) and any(v == mlp for _, v, _ in d.edges)
    check_mermaid(d.mermaid)


def test_attention(transformer):
    d = diagram(transformer.Attention.__call__)
    calls = labels(d, "call")
    assert sum(c.startswith("Attention.split\nf32[T, D] → f32[4, T, D // 4]") for c in calls) == 3
    assert any(c.startswith("Attention.causal · lift") for c in calls)
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
    assert node(d, "RMSNorm")[2] == "RMSNorm.__call__ · lift\nf32[D] → f32[D]"
    check_mermaid(d.mermaid)


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
    check_mermaid(d2.mermaid)


@tx.fn
def unit[T, D](x: f32[T, D]) -> f32[T, D]:
    return x / jnp.linalg.norm(x)


@tx.fn
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
    check_mermaid(d.mermaid)


def test_page(transformer, tmp_path):
    d = diagram(transformer.Transformer.__call__)
    path = d.save(tmp_path / "t.html")
    page = path.read_text()
    assert "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js" in page
    data = json.loads(re.search(r"const M = (.*);\n", page).group(1))
    assert data["depth"] == 1 and data["name"] == "Transformer.__call__"
    calls = [b for b in data["boxes"] if b["k"] == "call"]
    assert {b["name"] for b in calls} >= {"Block.__call__", "Attention.__call__", "Attention.causal", "MLP.__call__"}
    attn = next(b for b in calls if b["name"] == "Attention.__call__")
    assert attn["o"] and attn["d"] == 2 and attn["loc"].endswith("transformer.py:52")
    assert "def __call__" in data["sources"][attn["src"]]
    assert "</script" not in re.search(r"const M = (.*);\n", page).group(1)
    assert d._repr_html_().startswith("<iframe srcdoc=")


# ---------------------------------------------------------------- measured time


@pytest.fixture(scope="module")
def timed(transformer):
    model = transformer.Transformer.init(jr.key(0), Vocab=64, D=32, F=64, blocks=2, heads=4)
    return tx.diagram.timed(model.__call__, jr.randint(jr.key(1), (16,), 0, 64))


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
    assert "classDef heat" in d.mermaid and f"style b{loop} fill:#" in d.mermaid
    check_mermaid(d.mermaid)
    check_mermaid(d.at(2).mermaid)
    assert "measured with jax.profiler" in d.text and "Block.__call__" in d.text and "(outside the calls)" in d.text
    assert _fn.CALL_HOOK.get() is None


def test_timed_loops_add_up_iterations(transformer):
    """A staged loop's body is one box, timed over all its iterations."""
    runs = {}
    for n in (2, 4):
        model = transformer.Transformer.init(jr.key(0), Vocab=64, D=32, F=64, blocks=n, heads=4)
        d = tx.diagram.timed(model.__call__, jr.randint(jr.key(1), (16,), 0, 64))
        (block,) = [b for b in d.model["boxes"] if b["l"] == "Block.__call__"]
        runs[n] = block["n"]  # XLA instructions run per call, in the block
    assert runs[4] == pytest.approx(2 * runs[2], abs=2)


def test_timed_estimate(transformer, monkeypatch):
    def no_profiler(*a, **k):
        raise RuntimeError("no profiler here")

    monkeypatch.setattr(_diagram, "_profile", no_profiler)
    model = transformer.Transformer.init(jr.key(0), Vocab=64, D=32, F=64, blocks=2, heads=4)
    d = tx.diagram.timed(model.__call__, jr.randint(jr.key(1), (16,), 0, 64))
    t = d.model["timing"]
    assert t["how"] == "estimate" and "no profiler here" in t["why"]
    assert "ESTIMATED from FLOPs" in d.text and "FLOP" in node(d, "Block.__call__")[2]
    block = node(d, "Block.__call__")[0]
    assert d.timings[block][1] > 0.5  # 2 blocks of 32×32 and 32×64 matmuls dominate
    check_mermaid(d.mermaid)


def test_timed_page(timed, tmp_path):
    page = timed.save(tmp_path / "t.html").read_text()
    data = json.loads(re.search(r"const M = (.*);\n", page).group(1))
    assert data["timing"]["how"] == "profiler" and set(data["styles"]) >= {"heat0", "heat4", "error"}
    assert 'id="legend"' in page and "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js" in page


# ---------------------------------------------------------------- shape errors


@tx.fn
def proj[T, D, E](x: f32[T, D], w: f32[D, E]) -> f32[T, E]:
    return x @ w


@tx.fn
def broken[T, D](x: f32[T, D], w: f32[D, D]) -> f32[T, D]:
    h = unit(proj(x, w))
    return proj(h, w[:-1])  # w loses a row: proj's D does not match


@pytest.fixture(scope="module")
def buggy(tmp_path_factory):
    """The example transformer with a bug: the MLP called without .lift."""
    src = (ROOT / "examples" / "transformer.py").read_text()
    src = src.replace("x + self.mlp.lift(self.norm2.lift(x))", "x + self.mlp(self.norm2.lift(x))")
    path = tmp_path_factory.mktemp("buggy") / "buggy_transformer.py"
    path.write_text(src)
    return load("diagram_buggy", path)


def test_error_diagram(buggy):
    model = buggy.Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=3, heads=4)
    with pytest.raises(tx.ShapeError, match=r"MLP.__call__\(\): argument `x`") as e:
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
        "Attention.__call__", "LayerNorm.__call__ · lift", "LayerNorm.__call__ · lift", "index", "tokens: i32[24]"
    ]  # fmt: skip
    assert labels(d, "out") == []  # the trace stopped: nothing returned
    assert "classDef error" in d.mermaid and "linkStyle" in d.mermaid
    check_mermaid(d.mermaid)
    assert "✗ received f32[24, 32], declared x: f32[D]" in d.text
    assert "path:  tokens: i32[24] → index → LayerNorm.__call__ · lift → Attention.__call__ → LayerNorm.__call__ · lift → ✗ MLP.__call__" in d.text
    assert tx.diagram.explain(e.value) is d
    assert _fn.CALL_HOOK.get() is None


def test_explain_function():
    x, w = jnp.ones((5, 3)), jnp.ones((3, 3))
    d = tx.diagram.explain(broken, x, w)
    fail = node(d, "✗ proj")
    assert fail[2] == "✗ proj\nreceived f32[2, 3]\ndeclared w: f32[D, E]"
    assert d.failed == fail[0] and "argument `w` does not match" in d.error
    assert edge(d, "unit", "✗ proj") == "f32[5, 3]"  # the arguments it received
    assert edge(d, "w", "✗ proj") == "f32[2, 3]"  # w[:-1] is passed through: the type it enters with
    assert {n[2].split("\n")[0] for n in d.nodes if n[0] in d.feeds} == {"x: f32[5, 3]", "w: f32[3, 3]", "proj", "unit"}
    check_mermaid(d.mermaid)
    with pytest.raises(tx.ShapeError) as e:
        broken(x, w)
    assert e.value.diagram.nodes == d.nodes
    ok = tx.diagram.explain(gate, x, w)  # nothing fails: the plain diagram on these shapes
    assert ok.error is None and ok.failed is None and labels(ok, "out") == ["return f32[5, 3]"]
    with pytest.raises(ValueError, match="no diagram"):
        tx.diagram.explain(ValueError("x"))


def test_explain_root_fails():
    d = tx.diagram.explain(proj, jnp.ones((5, 3)), jnp.ones((4, 3)))
    assert labels(d, "op") == ["✗ ShapeError"] and d.failed == node(d, "✗ ShapeError")[0]
    assert "proj(): argument `w`" in d.error
    check_mermaid(d.mermaid)


def test_no_cost_on_success(monkeypatch):
    calls = []
    monkeypatch.setattr(_diagram, "attach", lambda *a: calls.append(a))
    gate(jnp.ones((5, 3)), jnp.ones((3, 3)))
    jax.jit(gate)(jnp.ones((5, 3)), jnp.ones((3, 3)))
    assert calls == []
    with pytest.raises(tx.ShapeError):
        broken(jnp.ones((5, 3)), jnp.ones((3, 3)))
    assert len(calls) == 1  # once, by the outermost call


def test_error_page(tmp_path):
    with pytest.raises(tx.ShapeError) as e:
        jax.jit(broken)(jnp.ones((5, 3)), jnp.ones((3, 3)))
    page = e.value.diagram.save(tmp_path / "e.html").read_text()
    data = json.loads(re.search(r"const M = (.*);\n", page).group(1))
    assert data["error"]["declared"] == "w: f32[D, E]" and data["error"]["got"] == "f32[2, 3]"
    assert [b["err"] for b in data["boxes"] if "err" in b] == [2]

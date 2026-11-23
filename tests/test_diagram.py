"""Architecture diagrams: calls as nodes, loops as boxes, shapes on the edges."""

import importlib.util
import json
import pathlib
import re
import sys

import jax.numpy as jnp
import jax.random as jr
import pytest

import typax as tx
from typax import _fn, f32
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
    r'|[bn]\d+ -->\|"[^"\[\]|]*"\| [bn]\d+|classDef \w+ [\w:#,]+|class [bn\d,]+ \w+|style b\d+ [\w:#,. -]+)',
    re.S,
)


def check_mermaid(text: str) -> None:
    """Every statement is one Mermaid knows, and labels hold no raw brackets or quotes."""
    lines = re.split(r"\n(?=\s*(?:flowchart|end|subgraph|[bn]\d+[\[(\s]|classDef|class|style))", text)
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

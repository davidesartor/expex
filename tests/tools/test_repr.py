"""Rich display in notebooks: structs, shape errors, proofs and cost estimates, computed only when displayed."""

import sys
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import f32


@expex.struct
class Block[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0
    act: Callable = jax.nn.gelu

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        return x + self.act(x @ self.w + self.b)


@expex.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for block in self.blocks:
            x = block(x)
        return x


@expex.struct
class Probe[D]:
    stack: Stack[D]
    w: f32[D] = 1.0

    def __call__[B](self, x: f32[B, D]) -> f32[B]:
        return self.stack(x[:, 1:]) @ self.w  # loses a feature


class Pretty:
    """IPython's printer, as far as ``_repr_pretty_`` uses it."""

    def __init__(self) -> None:
        self.out: list[str] = []

    def text(self, s: str) -> None:
        self.out.append(s)


def pretty(x) -> str:
    p = Pretty()
    x._repr_pretty_(p, False)
    return "".join(p.out)


@pytest.fixture(scope="module")
def stack():
    return Stack(jr.key(0), D=16, blocks=24)


def test_struct_html_is_a_tree_of_fields(stack):
    html = stack._repr_html_()
    assert html.startswith('<div class="txs" id="txt')
    assert "Stack[D=16]" in html and "6,528 parameters" in html and "26.1 kB" in html
    assert "f32[16, 16]" in html and "f32[D, D]" in html  # the shape and the annotation
    assert "list[24]" in html and "⋯ 20 more" in html  # long lists: first three and the last
    assert html.count("Block[D=16]") == 4
    assert 'class="shut"' in html and " hidden>" in html  # the list starts folded
    assert "&lt;function gelu&gt;" in html and "0x" not in html


def test_big_arrays_are_not_printed_small_ones_are():
    big = Block(jnp.ones((16, 16)), jnp.arange(16.0))
    small = Block(jnp.eye(2), jnp.array([1.5, -2.0]))
    assert "15." not in big._repr_html_()  # no values of b (16 elements)
    assert "[ 1.5, -2. ]" in small._repr_html_()


def test_struct_pretty(stack):
    text = pretty(stack)
    assert text.splitlines()[0] == "Stack[D=16]   6,528 parameters · 26.1 kB"
    assert "w        f32[16, 16]" in text and "⋯ 20 more" in text and "[23]" in text
    assert len(text.splitlines()) < 30  # 24 blocks, not 24 × 4 lines


def test_struct_display_under_eval_shape():
    shapes = jax.eval_shape(lambda: Stack(jr.key(0), D=8, blocks=2))
    html = shapes._repr_html_()
    assert "f32[8, 8]" in html and "Stack[D=8]" in html


def test_user_repr_html_is_kept():
    @expex.struct
    class Own:
        x: f32[3]

        def _repr_html_(self) -> str:
            return "mine"

    assert Own(jnp.ones(3))._repr_html_() == "mine"


def test_shape_error_display_draws_its_diagram_when_shown():
    probe = Probe(jr.key(1), D=16, blocks=2)
    with pytest.raises(expex.ShapeError) as info:
        probe(jnp.ones((4, 16)))
    e = info.value
    assert "_todo" in vars(e.diagram)  # not traced when raised
    html = e._repr_html_()
    assert "_todo" not in vars(e.diagram)  # traced now
    assert html.startswith('<div class="txe"') and "<svg" in html and "does not match its annotation" in html
    assert "e.diagram" not in html  # the note that points at it is left out: it is drawn
    text = pretty(e)
    assert text.startswith("ShapeError: ") and "e.diagram" not in text


def test_prove_reports(tmp_path):
    (tmp_path / "repr_shapes.py").write_text(
        "import expex\nfrom expex import f32\n\n"
        "@expex.fn\ndef ok[N](x: f32[N]) -> f32[N]:\n    return 2 * x\n\n"
        "@expex.fn\ndef short[T](x: f32[T]) -> f32[T]:\n    return x[1:]\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import repr_shapes  # type: ignore[import-not-found]

        reports = expex.prove(repr_shapes)
    finally:
        sys.path.remove(str(tmp_path))
    html = reports._repr_html_()
    assert "2 functions: 1 proved, 1 failed" in html
    assert '<span class="b ok">ok</span>' in html and '<span class="b bad">fail</span>' in html
    assert "<details><summary>" in html and "should be T, but is T - 1" in html  # the failure opens
    assert pretty(reports) == str(reports)
    one = expex.prove(Stack.__call__)
    assert "proved for all sizes" in one._repr_html_() and pretty(one) == str(one)


def test_cost_and_summary_html(stack):
    c = expex.cost(Stack.__call__, B=64, D=1024, blocks=24)
    html = c._repr_html_()
    assert html.startswith('<div class="txc"') and "3.24 GFLOP" in html and "25.2 M" in html
    sym = expex.cost(Stack.__call__)._repr_html_()
    assert "2·B·D²·L + 10·B·D·L" in sym
    assert 'data-path="blocks[*]"' in expex.summary(stack, B=4)._repr_html_()


def test_fragments_follow_the_theme_switch(stack):
    html = stack._repr_html_()
    assert "prefers-color-scheme:dark" in html and ":root[data-theme=dark] #txt" in html


def test_nothing_is_built_on_normal_calls():
    """Building, calling and erring never load the display code: it is imported when something is shown."""
    import subprocess

    code = (
        "import sys, jax, jax.numpy as jnp, jax.random as jr, expex\n"
        "from expex import f32\n"
        "@expex.struct\nclass L[D]:\n    w: f32[D, D] = 0.0\n"
        "    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:\n        return x @ self.w\n"
        "m = L(jr.key(0), D=4)\nm(jnp.ones((2, 4)))\nrepr(m)\n"
        "try:\n    m(jnp.ones((2, 3)))\nexcept expex.ShapeError as e:\n    str(e)\n"
        "print('expex.tools.display' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr

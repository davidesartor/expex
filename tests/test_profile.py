"""Line-by-line analytical profiles: FLOPs, bytes and shapes per source line."""

import importlib.util
import pathlib
import re
import sys

import jax.numpy as jnp

import expex
from expex import f32, fn
from expex.tools.cost import Poly, cost
from expex.tools.profile import profile

ROOT = pathlib.Path(__file__).resolve().parent.parent


def P(s: str) -> Poly:
    """``P("2*T*D^2 + T")``: a polynomial from a small formula."""
    out = Poly()
    for term in s.replace(" ", "").split("+"):
        p = Poly.of(1)
        for f in term.split("*"):
            name, _, k = f.partition("^")
            p = p * (Poly.of(int(name)) if name.isdigit() else Poly.of(name) ** int(k or 1))
        out = out + p
    return out


@fn
def mlp[B, T, D, F](x: f32[B, T, D], w: f32[D, F]) -> f32[B, T]:
    h = x @ w
    return jnp.tanh(h).sum(-1)


@fn
def recur[T, D](xs: f32[T, D], w: f32[D, D]) -> f32[D]:
    h = jnp.zeros(D)
    for t in range(T):
        h = jnp.tanh(h @ w + xs[t])
    return h


@expex.struct
class Linear[D]:
    w: f32[D, D]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        return jnp.tanh(x @ self.w)


@expex.struct
class Stack[D]:
    layers: list[Linear[D]]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for layer in self.layers:
            x = layer(x)
        return x


def line(p, section: str, text: str):
    sec = next(s for s in p.sections if s.name == section)
    return next(ln for ln in sec.lines if text in ln.text)


def total(p) -> Poly:
    return sum((ln.flops for s in p.sections for ln in s.lines), p.other)


def test_matmul_line_and_shapes():
    p = profile(mlp)
    h = line(p, "mlp", "h = x @ w")
    assert h.flops == P("2*B*T*D*F")
    assert h.bytes == P("4*B*T*F")
    assert h.shapes[-1] == "f32[B,T,F]"
    out = str(p)
    assert "2·B·T·D·F" in out and "f32[B,T,F]" in out
    assert p.sections[0].name == "mlp"


def test_lines_add_up_to_cost():
    for target in (mlp, recur, Stack.__call__):
        p = profile(target)
        assert total(p) == p.flops == cost(target).flops


def test_loop_body_is_multiplied_and_says_so():
    p = profile(recur)
    body = line(p, "recur", "h = jnp.tanh(h @ w + xs[t])")
    assert body.loops == ["T"]
    rest = body.flops - P("2*T*D^2 + 2*T*D")
    assert all(sum(k for _, k in m) <= 1 for m in rest.terms)  # scalar index arithmetic
    assert "×T 2·T·D²" in str(p)


def test_list_of_modules_is_L_times_and_callees_get_sections():
    p = profile(Stack.__call__)
    assert [s.name for s in p.sections] == ["Stack.__call__", "Linear.__call__"]
    inner = line(p, "Linear.__call__", "jnp.tanh(x @ self.w)")
    assert inner.flops == P("2*L*B*D^2 + L*B*D") and inner.loops == ["L"]
    call = line(p, "Stack.__call__", "x = layer(x)")
    assert call.calls == inner.flops and "L" in call.loops
    assert "×L" in str(p)


def test_sizes_give_numbers_and_shares():
    p = profile(Stack.__call__, B=8, D=64, layers=3)
    assert total(p).subs(p.given).value == cost(Stack.__call__, B=8, D=64, layers=3).flops.value
    text = str(p)
    assert "L=3" in text and "2·B·D²·L" in text
    n = 2 * 3 * 8 * 64 * 64 + 3 * 8 * 64
    assert f"{n / 1e3:.3g} kFLOP" in text
    inner = line(p, "Linear.__call__", "jnp.tanh")
    assert p.share(inner.flops, p.flops) > 0.99 and "100.0% ██████████" in text


def test_html_is_self_contained(tmp_path):
    p = profile(recur, T=16, D=8)
    page = p.html()
    assert page.startswith("<!doctype html>") and "prefers-color-scheme:dark" in page
    fragment = p._repr_html_()  # for notebooks: styles scoped to its own id, not the page's
    assert fragment.startswith('<div class="txp" id="txp') and ":root{" not in fragment.split("</style>")[0].split("@media")[0]
    assert "http://" not in page and "https://" not in page  # no CDN
    assert "2·T·D²" in page and "\"loops\": [\"T\"]" in page
    path = p.save(str(tmp_path / "p.html"))
    same = lambda t: re.sub(r"txp\d+", "txp", t)  # noqa: E731  (each rendering has its own id)
    assert same(pathlib.Path(path).read_text(encoding="utf-8")) == same(page)


def test_transformer_example():
    spec = importlib.util.spec_from_file_location("transformer_prof", ROOT / "examples" / "transformer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["transformer_prof"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    sizes = dict(T=1024, D=512, F=2048, Vocab=32000, blocks=12)
    p = profile(mod.Transformer.__call__, **sizes)
    assert total(p).subs(p.given).value == cost(mod.Transformer.__call__, **sizes).flops.value
    names = [s.name for s in p.sections]
    assert names[0] == "Transformer.__call__" and {"MLP.__call__", "Attention.causal"} <= set(names)
    assert line(p, "MLP.__call__", "gelu").flops == P("4*T*D*F*L + 8*T*F*L")
    assert line(p, "Attention.__call__", "self.split(x @ self.wq)").loops == ["L"]
    assert line(p, "Transformer.__call__", "@ self.unembed").flops == P("2*T*D*Vocab")

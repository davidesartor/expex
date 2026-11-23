"""Symbolic cost estimates: FLOPs and parameters as formulas in the sizes."""

import importlib.util
import pathlib
import sys
from fractions import Fraction

import jax

import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import f32, fn
from expex.tools.cost import Poly, cost

ROOT = pathlib.Path(__file__).resolve().parent.parent


def P(s: str) -> Poly:
    """``P("2*T*D^2 + T")``: a polynomial from a small formula (+ of * of name^k / ints)."""
    out = Poly()
    for term in s.replace(" ", "").split("+"):
        p = Poly.of(1)
        for f in term.split("*"):
            name, _, k = f.partition("^")
            p = p * (Poly.of(int(name)) if name.isdigit() else Poly.of(name) ** int(k or 1))
        out = out + p
    return out


@fn
def project[B, T, D, F](x: f32[B, T, D], w: f32[D, F]) -> f32[B, T, F]:
    return x @ w


@fn
def recur[T, D](xs: f32[T, D], w: f32[D, D]) -> f32[D]:
    h = jnp.zeros(D)
    for t in range(T):
        h = jnp.tanh(h @ w + xs[t])
    return h


@expex.struct
class Linear[I, O]:
    w: f32[I, O]
    b: f32[O]

    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        return x @ self.w + self.b


@expex.struct
class Stack[D]:
    layers: list[Linear[D, D]]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for layer in self.layers:
            x = layer(x)
        return x


def test_matmul_is_exact():
    c = cost(project)
    assert c.flops == P("2*B*T*D*F")
    assert str(c).splitlines()[1].split() == ["flops", "2·B·T·D·F"]
    assert c.parameters is None


def test_loop_multiplies_by_its_length():
    c = cost(recur)
    # each step: a D×D matrix-vector product (2·D²), an add and a tanh (D each)
    rest = c.flops - P("2*T*D^2 + 2*T*D")
    assert all(sum(k for _, k in m) <= 1 for m in rest.terms)  # scalar index arithmetic
    assert c.flops.format(c.sizes).startswith("2·T·D² + 2·T·D + ")


def test_concrete_sizes_give_a_number():
    c = cost(project, B=2, T=3, D=4, F=5)
    assert c.flops.value == 2 * 2 * 3 * 4 * 5
    assert "= 240 FLOP" in str(c)
    big = cost(project, B=8, T=1024, D=1024, F=4096)
    assert "= 68.7 GFLOP" in str(big)
    partial = cost(project, D=4)
    assert partial.flops == P("8*B*T*F") and partial.sizes == ["B", "T", "F"]


def test_unknown_keyword_is_an_error():
    with pytest.raises(TypeError, match="unknown size or parameter `Q`"):
        cost(project, Q=3)


def test_method_parameters_are_a_formula_in_the_class_sizes():
    c = cost(Linear.__call__)
    assert c.parameters == P("I*O + O")
    assert c.flops == P("2*B*I*O + B*O")
    bound = cost(Linear(jnp.ones((3, 4)), jnp.ones(4)).__call__)
    assert bound.parameters.value == 16 and bound.given == {"I": 3, "O": 4}
    assert bound.flops == P("24*B + 4*B")


def test_list_of_modules_has_length_L():
    c = cost(Stack.__call__)
    assert c.parameters == P("L*D^2 + L*D")
    assert c.flops == P("2*L*B*D^2 + L*B*D")
    assert "L = len(self.layers)" in str(c)
    assert cost(Stack.__call__, L=3).parameters == P("3*D^2 + 3*D")


def test_while_loop_is_counted_once_and_said_so():
    @fn
    def halve(x: f32) -> f32:
        while x > 1.0:
            x = x / 2
        return x

    assert "trip count unknown, one iteration counted" in str(cost(halve))


def test_static_int_from_keyword_or_noted():
    @fn
    def split[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:
        assert D % heads == 0
        return jnp.tanh(x.reshape(T, heads, D // heads).transpose(1, 0, 2))

    assert "heads=1 (choose with heads=...)" in str(cost(split))
    c = cost(split, heads=4)
    assert c.flops == P("T*D") and "heads" not in str(c)


def test_formatting():
    T, D = Poly.of("T"), Poly.of("D")
    assert (2 * T * D**2 + T * D + 3).format(["T", "D"]) == "2·T·D² + T·D + 3"
    assert (T * T - 2 * D).format(["T", "D"]) == "T² - 2·D"
    assert (T - 2 * D).format() == "-2·D + T"  # same degree: larger coefficient first
    assert (T * D * Fraction(1, 4)).format(["T", "D"]) == "T·D/4"


@pytest.fixture(scope="module")
def transformer():
    spec = importlib.util.spec_from_file_location("cost_transformer", ROOT / "examples" / "transformer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_transformer(transformer):
    c = cost(transformer.Transformer.__call__)
    assert c.sizes == ["T", "D", "F", "Vocab", "L"]
    assert c.parameters == P("2*D*Vocab + 4*L*D^2 + 2*L*D*F + 4*L*D + 2*D")
    # q, k, v, o projections, the MLP, attention scores and weighted sum, unembedding
    leading = P("8*T*D^2*L + 4*T*D*F*L + 4*T^2*D*L + 2*T*D*Vocab")
    rest = c.flops - leading
    assert all(sum(k for _, k in m) <= 3 for m in rest.terms)  # only lower-order terms remain
    assert c.flops.format(c.sizes).startswith("8·T·D²·L + 4·T·D·F·L + 4·T²·D·L")
    assert "2·T·D·Vocab" in str(c)

    sized = cost(transformer.Transformer.__call__, T=1024, D=512, F=2048, Vocab=32000, L=12)
    assert sized.flops.value == c.flops.subs(dict(T=1024, D=512, F=2048, Vocab=32000, L=12)).value
    assert "GFLOP" in str(sized) and "= 70.5 M" in str(sized)

    assert c.parameters is not None
    by_name = cost(transformer.Transformer.__call__, blocks=12)
    assert by_name.parameters == c.parameters.subs({"L": 12}) and "L" not in by_name.sizes

    model = transformer.Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=4, heads=4)
    bound = cost(model.__call__, T=16)
    assert bound.parameters is not None
    assert bound.parameters.value == sum(x.size for x in jax.tree.leaves(model))
    assert bound.flops.value == c.flops.subs(dict(T=16, D=32, F=64, Vocab=50, L=4)).value

"""Fixed sizes declared with the type parameters: ``def f[R = 6]`` reads "R, which is 6"."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import pytest

import expex
from expex import Dim, ShapeError, f32, fn, i32

R = 3  # a global of the same name: the type parameter's R is its own


@fn
def rows_sum[R = 6](x: f32[R, 7]) -> f32[R]:
    return x.sum(1)


@fn
def board[B, Rows = 6, Cols = 7](x: f32[B, Rows, Cols]) -> f32[B, Cols]:
    empty = jnp.zeros((Rows, Cols))  # sizes
    return (x + empty).sum(Rows)  # an axis


def test_a_fixed_size_is_matched_and_reused():
    assert rows_sum(jnp.ones((6, 7))).shape == (6,)
    assert rows_sum.signature == "rows_sum[R = 6](x: f32[R, 7]) -> f32[R]"
    with pytest.raises(ShapeError) as e:
        rows_sum(jnp.ones((5, 7)))
    assert "axis 0 should be R, which is fixed at 6, but is 5." in str(e.value)
    assert "declared  x: f32[R, 7]" in str(e.value)


def test_the_three_spellings():
    @fn
    def a[B, R = 6](x: f32[B, R]) -> f32[B]:
        return x.sum(R)

    @fn
    def b[B, R = Literal[6]](x: f32[B, R]) -> f32[B]:
        return x.sum(R)

    @fn
    def c[R: Literal[6], B](x: f32[B, R]) -> f32[B]:  # a bound: any position
        return x.sum(R)

    for f in (a, b, c):
        assert f(jnp.ones((2, 6))).tolist() == [6.0, 6.0]
        with pytest.raises(ShapeError, match="axis 1 should be R, which is fixed at 6, but is 5"):
            f(jnp.ones((2, 5)))
        assert expex.prove(f).ok
    assert "R = 6" in a.signature and "R = 6" in b.signature and "R = 6" in c.signature
    assert a.__type_params__[1].__default__ == 6 and a.__type_params__[1].has_default()


def test_the_body_uses_it_as_a_size_and_as_an_axis():
    x = jnp.arange(2 * 6 * 7.0).reshape(2, 6, 7)
    assert jnp.allclose(board(x), x.sum(1))
    assert jax.jit(board)(x).shape == (2, 7)
    assert board.vectorize(jnp.ones((3, 2, 6, 7))).shape == (3, 2, 7)
    with pytest.raises(ShapeError, match="axis 2 should be Cols, which is fixed at 7, but is 8"):
        board(jnp.ones((2, 6, 8)))


def test_nested_functions_and_local_annotations_see_it():
    @fn
    def f[R = 6, C = 7](x: f32[R, C]) -> f32[R]:
        h: f32[R, C] = x * 2

        def row_total(r):
            return r.sum(C) + jnp.zeros(()) * R

        return jax.vmap(row_total)(h)

    assert f(jnp.ones((6, 7))).tolist() == [14.0] * 6
    assert expex.prove(f).ok


def test_arithmetic_on_a_fixed_size():
    @fn
    def pad[R = 6](x: f32[R]) -> f32[R + 1]:
        return jnp.pad(x, (0, 1))

    @fn
    def mix[T, R = 6](x: f32[T, R]) -> f32[T + R]:
        return jnp.concatenate([x[:, 0], jnp.zeros(R)])

    assert pad.signature == "pad[R = 6](x: f32[R]) -> f32[R + 1]"
    assert pad(jnp.ones(6)).shape == (7,) and mix(jnp.ones((3, 6))).shape == (9,)
    assert expex.prove(pad).ok and expex.prove(mix).ok


def test_the_prover_treats_it_as_a_constant():
    assert expex.prove(board).ok
    assert expex.prove(rows_sum).ok

    @fn
    def wrong[R = 6](x: f32[R, 7]) -> f32[R]:
        return x[1:].sum(1)

    report = expex.prove(wrong)
    assert not report.ok and "axis 0 should be R, which is fixed at 6, but is 5" in str(report)


def test_a_global_of_the_same_name_is_not_used():
    assert R == 3 and rows_sum(jnp.ones((6, 7))).shape == (6,)


@pytest.mark.parametrize(
    "source, message",
    [
        ('def f[R = "x"](x: f32[R]): return x', "the type parameter R has the default 'x', but a default fixes a size"),
        ("def f[R = 6.0](x: f32[R]): return x", "the type parameter R has the default 6.0"),
        ("def f[R = -1](x: f32[R]): return x", "the type parameter R has the default -1"),
        ("def f[R = True](x: f32[R]): return x", "the type parameter R has the default True"),
        ("def f[R = int](x: f32[R]): return x", "the type parameter R has the default int"),
        ('def f[R = Literal["a"]](x: f32[R]): return x', "the default Literal['a']"),
        ("def f[R = Literal[6, 7]](x: f32[R]): return x", "the default Literal[6, 7]"),
        ("def f[R: int](x: f32[R]): return x", "the type parameter R has the bound int, but the only bound a size takes is Literal[<int>]"),
        ("def f[R: Literal[6.0]](x: f32[R]): return x", "the bound Literal[6.0]"),
        ("def f[R: (int, str)](x: f32[R]): return x", "R is constrained to (int, str), but a size takes no constraints"),
        ("def f[R: Literal[6] = 7](x: f32[R]): return x", "R is fixed twice, with different values: at 6 by its bound, and at 7 by its default"),
        ("def f[*B = 6](x: f32[*B]): return x", "the variadic type parameter *B has a default"),
        ("def f[R = undefined_name](x: f32[R]): return x", "the default of the type parameter R cannot be evaluated"),
    ],
)  # fmt: skip
def test_other_defaults_and_bounds_are_errors(source, message):
    ns = {"f32": f32, "Literal": Literal}
    exec(source, ns)
    with pytest.raises(TypeError) as e:
        fn(ns["f"])
    assert e.value.code == "EX404"  # type: ignore[attr-defined]
    assert str(e.value).startswith("f: ") and message in str(e.value)


def test_the_same_value_as_bound_and_default_is_fine():
    @fn
    def f[R: Literal[6] = 6](x: f32[R]) -> f32[R]:
        return x

    assert f(jnp.ones(6)).shape == (6,)


def test_the_old_slice_spelling_is_refused():
    with pytest.raises(TypeError) as e:

        @fn
        def f(x: f32[R:6, 7]) -> f32[R]:  # type: ignore[misc]
            return x.sum(1)

    assert e.value.code == "EX404"  # type: ignore[attr-defined]
    assert "`R:6` in `x: f32[R:6, 7]` declares a size in the annotation, which expex no longer reads" in str(e.value)
    assert "def f[R = 6](...) (or R = Literal[6] for type checkers)" in str(e.value)

    with pytest.raises(TypeError, match=r"class Old\[N = 3\]"):

        @expex.struct
        class Old:
            w: f32[N:3, 4]  # type: ignore[misc]  # noqa: F821


def test_dim_classes_keep_working_alongside():
    class RGB(Dim, size=3): ...

    @fn
    def f[H, W = 4](img: f32[H, W, RGB]) -> f32[H, W]:
        return img.mean(RGB) + jnp.zeros(W)

    assert f(jnp.ones((2, 4, 3))).shape == (2, 4)
    with pytest.raises(ShapeError, match="axis 2 should be RGB, which is fixed at 3, but is 4"):
        f(jnp.ones((2, 4, 4)))
    assert expex.prove(f).ok


# --------------------------------------------------------------------- structs


@dataclass(frozen=True)
class Board[Rows = 6, Cols = 7]:
    cells: i32[Rows, Cols]

    @fn
    def heights(self) -> i32[Cols]:
        return (self.cells != 0).sum(Rows)

    @fn
    def empty(self) -> i32[Rows, Cols]:
        return jnp.zeros((Rows, Cols), jnp.int32)


class Pair[N = 3](NamedTuple):
    w: f32[N, 4]


def test_a_class_s_fixed_sizes():
    b = Board(jnp.zeros((6, 7), jnp.int32).at[5, 2].set(1))
    assert b.heights().tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert b.empty().shape == (6, 7)
    assert expex.prove(Board.heights).ok and expex.prove(Board.empty).ok
    with pytest.raises(ShapeError) as e:
        Board(jnp.zeros((5, 7), jnp.int32)).heights()
    assert "declared  self.cells: i32[Rows, Cols]" in str(e.value)
    assert "axis 0 should be Rows, which is fixed at 6, but is 5" in str(e.value)

    @fn
    def total(p: Pair) -> f32[4]:
        return p.w.sum(0)

    assert total(Pair(jnp.ones((3, 4)))).shape == (4,)
    with pytest.raises(ShapeError, match="axis 0 should be N, which is fixed at 3, but is 2"):
        total(Pair(jnp.ones((2, 4))))


def test_expex_struct():
    @expex.struct
    class Grid[GR = 2, GC = 3]:
        cells: f32[GR, GC]

        def __call__(self, x: f32[GC]) -> f32[GR]:
            return self.cells @ x + jnp.zeros(GR)

    g = Grid(cells=jnp.ones((2, 3)))
    assert g(jnp.ones(3)).shape == (2,)
    assert expex.prove(Grid.__call__).ok
    assert "GR (= 2), GC (= 3)" in str(expex.summary(g))


def test_a_struct_built_from_a_key_with_fixed_and_free_sizes():
    @expex.struct
    class Layer[N, R = 6]:
        w: f32[N, R] = jax.nn.initializers.normal(0.1)
        b: f32[R] = jax.nn.initializers.zeros

    layer = Layer(jax.random.key(0), N=3)
    assert layer.w.shape == (3, 6) and layer.b.shape == (6,)
    assert layer.sizes == {"N": 3}
    with pytest.raises(TypeError, match="unexpected keyword argument 'R'"):
        Layer(jax.random.key(0), N=3, R=5)  # fixed: not a size to give

    @fn
    def first[M](layer: Layer[M]) -> f32[M]:  # Layer[M]: Python fills in R = 6
        return layer.w[:, 0]

    assert first.signature == "first[M](layer: Layer[M]) -> f32[M]"
    assert first(layer).shape == (3,)
    with pytest.raises(ShapeError, match="axis 1 should be R, which is fixed at 6, but is 5"):
        first(Layer(jnp.ones((3, 5)), jnp.ones(5)))
    with pytest.raises(TypeError, match="Layer's size R is fixed at 6, but is given as 5"):

        @fn
        def other[M](layer: Layer[M, 5]) -> f32[M]:  # type: ignore[type-arg]
            return layer.w[:, 0]


def test_a_struct_with_a_bad_default():
    with pytest.raises(TypeError) as e:

        @expex.struct
        class Bad[K = "two"]:
            a: f32[K]

    assert e.value.code == "EX404"  # type: ignore[attr-defined]
    assert "Bad: the type parameter K has the default 'two'" in str(e.value)


# --------------------------------------------------------------------- display


def test_tools_list_it_as_fixed():
    @fn
    def g[B, R = 6, C = 7, H = 5](x: f32[B, R, C], w: f32[C, H]) -> f32[B, R, H]:
        return x @ w

    assert "(sizes: B, R (= 6), C (= 7), H (= 5))" in str(expex.cost(g))
    d = expex.diagram(g)
    assert d.text.splitlines()[0].endswith(".g   sizes B · R (= 6) · C (= 7) · H (= 5)")
    assert "R (= 6)" in d.svg


def test_inspect_lists_it_as_fixed():
    @expex.struct
    class Grid[GR = 2, GC = 3]:
        cells: f32[GR, GC]

    page = expex.inspect(Grid(cells=jnp.ones((2, 3))))
    assert "GR (= 2)" in page.html()

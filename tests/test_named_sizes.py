"""Sizes declared by name in an annotation: ``f32[R:6, 7]`` reads "R, which is 6"."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pytest

import expex
from expex import Dim, ShapeError, f32, fn, i32

R = 3  # a global of the same name: the annotation's R is its own


@fn
def rows_sum(x: f32[R:6, 7]) -> f32[R]:
    return x.sum(1)


@fn
def board[B](x: f32[B, Rows:6, Cols:7]) -> f32[B, Cols]:
    empty = jnp.zeros((Rows, Cols))  # sizes
    return (x + empty).sum(Rows)  # an axis


def test_a_declared_size_is_matched_and_reused():
    assert rows_sum(jnp.ones((6, 7))).shape == (6,)
    assert rows_sum.signature == "rows_sum(x: f32[R, 7]) -> f32[R]"
    with pytest.raises(ShapeError) as e:
        rows_sum(jnp.ones((5, 7)))
    assert "axis 0 should be R, which is fixed at 6, but is 5." in str(e.value)
    assert "declared  x: f32[R, 7]" in str(e.value)


def test_the_body_uses_it_as_a_size_and_as_an_axis():
    x = jnp.arange(2 * 6 * 7.0).reshape(2, 6, 7)
    assert jnp.allclose(board(x), x.sum(1))
    assert jax.jit(board)(x).shape == (2, 7)
    assert board.vectorize(jnp.ones((3, 2, 6, 7))).shape == (3, 2, 7)
    with pytest.raises(ShapeError, match="axis 2 should be Cols, which is fixed at 7, but is 8"):
        board(jnp.ones((2, 6, 8)))


def test_the_prover_treats_it_as_a_constant():
    assert expex.prove(board).ok
    assert expex.prove(rows_sum).ok

    @fn
    def wrong(x: f32[R:6, 7]) -> f32[R]:
        return x[1:].sum(1)

    report = expex.prove(wrong)
    assert not report.ok and "axis 0 should be R, which is fixed at 6, but is 5" in str(report)


def test_a_global_of_the_same_name_is_not_used():
    assert R == 3 and rows_sum(jnp.ones((6, 7))).shape == (6,)


def test_declared_twice_with_different_sizes():
    with pytest.raises(TypeError) as e:

        @fn
        def f(x: f32[R:6], y: f32[R:5]) -> f32[R]:
            return x

    msg = str(e.value)
    assert "the size R is declared twice, with different values" in msg
    assert "R:6 in `x: f32[R:6]`" in msg and "R:5 in `y: f32[R:5]`" in msg

    @fn
    def same(x: f32[R:6], y: f32[R:6]) -> f32[R]:  # the same value twice is fine
        return x + y

    assert same(jnp.ones(6), jnp.ones(6)).shape == (6,)


def test_a_type_parameter_or_a_parameter_of_the_same_name():
    with pytest.raises(TypeError, match="R is also a type parameter"):

        @fn
        def f[R](x: f32[R:6]) -> f32[R]:
            return x

    with pytest.raises(TypeError, match="R is also a parameter"):

        @fn
        def g(R: int, x: f32[R:6]) -> f32[R]:
            return x


def test_the_size_must_be_an_int_known_where_it_is_declared():
    with pytest.raises(TypeError, match="its size is not an int known where it is declared"):

        @fn
        def f(x: f32[R:n], n: int) -> f32[R]:  # noqa: F821
            return x

    @fn
    def g(x: f32[R:SIX]) -> f32[R]:  # a global int
        return x

    assert g(jnp.ones(6)).shape == (6,)


SIX = 6


def test_quoted_and_in_a_union():
    @fn
    def f(x: "f32[R:2, 3]", y: f32[R, 3] | None = None) -> f32[R]:
        return x.sum(1)

    assert f(jnp.ones((2, 3))).shape == (2,)
    with pytest.raises(ShapeError, match="R, which is fixed at 2"):
        f(jnp.ones((2, 3)), jnp.ones((4, 3)))


def test_dim_classes_keep_working_alongside():
    class RGB(Dim, size=3): ...

    @fn
    def f[H](img: f32[H, W:4, RGB]) -> f32[H, W]:
        return img.mean(RGB) + jnp.zeros(W)

    assert f(jnp.ones((2, 4, 3))).shape == (2, 4)
    with pytest.raises(ShapeError, match="axis 2 should be RGB, which is fixed at 3, but is 4"):
        f(jnp.ones((2, 4, 4)))
    assert expex.prove(f).ok


# --------------------------------------------------------------------- structs


@dataclass(frozen=True)
class Board:
    cells: i32[Rows:6, Cols:7]

    @fn
    def heights(self) -> i32[Cols]:
        return (self.cells != 0).sum(Rows)

    @fn
    def empty(self) -> i32[Rows, Cols]:
        return jnp.zeros((Rows, Cols), jnp.int32)


class Pair(NamedTuple):
    w: f32[N:3, 4]


def test_struct_fields_declare_class_sizes():
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


def test_struct_conflicts():
    @dataclass(frozen=True)
    class Bad:
        a: f32[K:2]
        b: f32[K:3]

    with pytest.raises(TypeError, match="K is declared twice, with different values: K:2 in `a: f32.K:2.`, and K:3 in `b"):

        @fn
        def use(x: Bad) -> f32[2]:
            return x.a

    with pytest.raises(TypeError, match="the size K is declared twice"):

        @expex.struct
        class Worse:
            a: f32[K:2]
            b: f32[K:3]


def test_expex_struct():
    @expex.struct
    class Grid:
        cells: f32[GR:2, GC:3]

        def __call__(self, x: f32[GC]) -> f32[GR]:
            return self.cells @ x + jnp.zeros(GR)

    g = Grid(cells=jnp.ones((2, 3)))
    assert g(jnp.ones(3)).shape == (2,)
    assert expex.prove(Grid.__call__).ok
    assert "GR (= 2), GC (= 3)" in str(expex.summary(g))


# --------------------------------------------------------------------- display


def test_tools_list_it_as_fixed():
    @fn
    def g[B](x: f32[B, R:6, C:7], w: f32[C, H:5]) -> f32[B, R, H]:
        return x @ w

    assert "(sizes: B, R (= 6), C (= 7), H (= 5))" in str(expex.cost(g))
    d = expex.diagram(g)
    assert d.text.splitlines()[0].endswith(".g   sizes B · R (= 6) · C (= 7) · H (= 5)")
    assert "R (= 6)" in d.svg

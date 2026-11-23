"""Bugs found while porting Pgx's Connect Four and Othello (examples/pgx/)."""

import jax
import jax.numpy as jnp

import expex
from expex import Bool, Dim, i32


class Rows(Dim, size=6): ...
class Cols(Dim, size=7): ...


@expex.fn
def empty_board() -> i32[Rows, Cols]:
    # a fixed-size Dim is a known size in any body, even when no argument has it
    return jnp.full((Rows, Cols), -1, jnp.int32)


@expex.fn
def filled(board: i32[Rows, Cols]) -> i32[Cols]:
    # ... and an axis, by the name in the annotation
    return (board >= 0).sum(Rows)


def test_fixed_size_dims_are_sizes_and_axes_in_the_body():
    board = empty_board()
    assert board.shape == (6, 7)
    assert filled(board.at[5, 2].set(0)).tolist() == [0, 0, 1, 0, 0, 0, 0]
    local = jnp.zeros((6, 7)).at[:, 0].set(1)

    @expex.fn
    def by_size(x: i32[Rows, Cols]) -> i32[Cols]:
        y = x * 2  # not annotated: found by its size
        return y.sum(Rows)

    assert by_size(local.astype(jnp.int32)).tolist() == [12, 0, 0, 0, 0, 0, 0]


@expex.fn
def inside[N](x: i32[N]) -> Bool[N]:
    return 0 <= x < 3


@expex.fn
def on_board(r: i32) -> Bool:
    return 0 <= r < 8


def test_a_chained_comparison_alone_is_converted():
    # with nothing else to convert, the function ran as plain Python: under
    # jit / vmap `0 <= r < 8` asked for the truth value of a tracer
    assert jax.jit(on_board)(3) and not jax.jit(on_board)(8)
    assert jax.vmap(on_board)(jnp.array([-1, 0, 7, 8])).tolist() == [False, True, True, False]


def test_chained_comparison_of_concrete_arrays_is_elementwise():
    # traced, the chain was combined elementwise; eager calls raised
    # "The truth value of an array with more than one element is ambiguous"
    x = jnp.array([-1, 0, 2, 3])
    assert inside(x).tolist() == [False, True, True, False]
    assert jax.jit(inside)(x).tolist() == [False, True, True, False]

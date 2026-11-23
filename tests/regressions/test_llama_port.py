"""Bugs found while porting Llama 2 (examples/llama/)."""

import jax
import jax.numpy as jnp
import pytest

import expex
from expex import Int, f32, fn


@expex.struct
class Node[D]:
    w: f32[D]

    def grow[T](self, x: f32[T, D]) -> Node[D]:  # the class itself, in its own method
        return Node(self.w + x.sum(0))


def test_a_method_naming_its_own_class_keeps_its_annotations():
    # the annotations could not be evaluated while the class was being defined;
    # once it was, all of them were dropped, so nothing was checked or proved
    node = Node(jnp.ones(3))
    with pytest.raises(expex.ShapeError, match="`x`"):
        node.grow(jnp.ones((2, 4)))
    assert expex.prove(Node.grow).status == "proved"


@expex.struct
class Heads[D, H, K]:
    wq: f32[D, H, K]

    def __call__[T](self, x: f32[T, D]) -> f32[H, T, K]:
        return expex.einsum(x, self.wq, to=(H, T, K))


def test_einsum_reads_a_fields_annotation():
    # the operand's axes came from its sizes, and H == K was ambiguous
    out = Heads(jnp.ones((5, 4, 4)))(jnp.ones((3, 5)))
    assert out.shape == (4, 3, 4)


@fn
def zeros(n: int) -> f32[n]:
    return jnp.zeros(n)


@fn
def at(x: f32[...], start: Int) -> f32:
    return jax.lax.dynamic_index_in_dim(x, start, keepdims=False)


@fn
def padded[T](x: f32[T]) -> f32[T]:
    return x + zeros(T) + at(x, T - 1)


def test_sizes_passed_on_as_values_are_proved():
    # under the prover T is symbolic: it failed the `int` and `Int` checks
    assert expex.prove(padded).status == "proved", str(expex.prove(padded))

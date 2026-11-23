"""Bugs found while porting the Vision Transformer (examples/vit/)."""

import jax.numpy as jnp

import expex
from expex import f32
from expex.annotations.expr import Expr, divisibility
from expex.tools import trace


@expex.fn
def patches[H, W, P](image: f32[H, W], patch: f32[P]) -> f32[G, P, W]:
    G = H // P
    assert H % P == 0
    return expex.rearrange(image, to=(G, P, W))


def test_a_size_divisible_by_another_size_is_proved():
    # `H % P == 0` was only used when P was a number, so splitting H into
    # (H // P, P) was spot-checked instead of proved
    assert patches(jnp.ones((6, 5)), jnp.ones(3)).shape == (2, 3, 5)
    assert expex.prove(patches).status == "proved", str(expex.prove(patches))


@expex.fn
def both[H, P, Q](x: f32[H], p: f32[P], q: f32[Q]) -> f32[H]:
    assert H % P == 0 and H % Q == 0
    return x


@expex.fn
def cycle[H, P](x: f32[H], p: f32[P]) -> f32[H]:
    assert H % P == 0 and P % H == 0
    return x


def test_only_one_divisor_size_per_name_and_no_cycles():
    # tracing H as P·Q·n, or H as P·n with P as H·m, would prove the function
    # for fewer sizes than the constraints allow
    assert list(trace.multiples(both, {}).items()) == [("H", "P")]
    assert len(trace.multiples(cycle, {})) == 1
    assert trace.multiples(patches, {}) == {"H": "P"}
    assert expex.prove(both).status == "proved" and expex.prove(cycle).status == "proved"


def test_divisibility_by_a_number_or_a_size():
    assert divisibility(Expr("K % n == 0", constraint=True), {"n": 4}, {"K", "n"}) == ("K", 4)
    assert divisibility(Expr("K % P == 0", constraint=True), {}, {"K", "P"}) == ("K", "P")
    assert divisibility(Expr("K % P == 0", constraint=True), {}, {"K"}) is None


def test_a_divisor_size_not_built_in_stays_a_constraint():
    # the second divisor of `both` is not built into the traced sizes, so it is
    # passed on to JAX as a constraint, as before
    assert trace.constraints(both, {}) == (["mod(H, Q) == 0"], [])

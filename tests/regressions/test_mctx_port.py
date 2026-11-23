"""Bugs found while porting mctx (examples/mctx/)."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Key, f32, i32


@expex.struct
class Out[A]:
    logits: f32[A]
    value: f32


@expex.fn
def evaluate[A, E](model: Callable[..., tuple[Out[A], f32[E]]], params: Any, x: f32[E], prior: f32[A]) -> f32[A]:
    out, y = model(params, x)
    return out.logits + prior * out.value + y.sum()


type Pick = Callable[..., i32]


@expex.fn
def pick_twice[N](choose: Pick, x: f32[N]) -> f32:
    return x[choose(x)] + x[choose(x)]


@expex.fn
def draw[N](make_key: Callable[..., Key], x: f32[N]) -> f32[N]:
    return x + jr.uniform(make_key(), (N,))


def test_a_callable_with_a_declared_result_is_proved_for_any_such_function():
    # a Static callable without a default was skipped ("the prover cannot invent
    # a value"); now the prover stands in a function returning the declared result
    report = expex.prove(evaluate, examples={"params": None})
    assert report.status == "proved", report
    assert expex.prove(pick_twice).status == "proved"  # through a type alias
    assert expex.prove(draw).status == "proved"  # a function returning a key


@expex.fn
def wrong[A, E](model: Callable[..., f32[E]], x: f32[E], prior: f32[A]) -> f32[A]:
    return model(x) + prior  # E + A: not the same size


def test_the_stand_in_keeps_the_declared_sizes():
    report = expex.prove(wrong)
    assert report.status != "proved" and "incompatible shapes" in str(report)


@expex.fn
def running[T](x: f32[T]) -> f32[T]:
    total, out = 0.0, []
    for t in range(T):
        if T > 1:  # a Python comparison of a size: inconclusive while proving
            total = total + x[t]
        out.append(total)
    return jnp.asarray(out)


def test_a_loop_of_symbolic_length_reports_the_error_inside_it():
    # staging the loop failed on `T > 1`; the fallback to unrolling then failed
    # on its own warning, `'T' > '16'`, and the report named that instead
    report = str(expex.prove(running))
    assert "'T' > '1'" in report and "16" not in report

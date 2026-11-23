"""The cost of expex itself, per call: each case against the same thing in plain JAX.

From scratchpad scripts used while restructuring the library (bench_call.py,
restructure/bench.py): a typed eager call, a jitted typed call, a jitted method
of a struct, a `donate="auto"` training-style step, and an eager `.vectorize`.
"Plain JAX" is the undecorated function, `jax.jit`, `jax.jit(..., donate_argnums=0)`
and `jax.vmap`. The difference (expex - JAX, in microseconds) is the overhead
the README's zero-overhead claim is about.
"""

import jax
import jax.numpy as jnp

import expex
from expex import f32, fn
from harness import Fn, Measure

NAME = "overhead"
ORIGINALS: list[str] = []
CLAIM: dict[str, str] = {}
CONFIGS = {"quick": dict(steps=100), "default": dict(steps=100)}
DEFAULTS = {"quick": ["quick"], "cpu": ["default"], "gpu": ["default"]}


def plain_identity(x, w):
    return x


@fn
def typed[B, D](x: f32[B, D], w: f32[D, D]) -> f32[B, D]:
    return x


def plain_matmul(x, w):
    return x @ w


@fn(jit=True)
def jitted[B, D](x: f32[B, D], w: f32[D, D]) -> f32[B, D]:
    return x @ w


def plain_step(state, x):
    return state + x


@fn(jit=True, donate="auto")
def donated_step[D](state: f32[D], x: f32[D]) -> f32[D]:
    return state + x


@expex.struct
class Scale[D]:
    w: f32[D]

    @fn(jit=True)
    def apply(self, x: f32[D]) -> f32[D]:
        return x * self.w


def plain_affine(x):
    return 2.0 * x + 1.0


@fn
def affine[D](x: f32[D]) -> f32[D]:
    return 2.0 * x + 1.0


def setup(cfg: dict) -> list[Measure]:
    n = cfg["steps"]
    x, w, v = jnp.ones((4, 8)), jnp.eye(8), jnp.ones(8)
    batch = jnp.ones((16, 8))
    scale = Scale(jnp.ones(8))
    jit_matmul = jax.jit(plain_matmul)
    jit_method = jax.jit(lambda m, x: x * m.w)
    jit_donated = jax.jit(plain_step, donate_argnums=0)
    vmapped = jax.vmap(plain_affine)

    def loop(step):
        def run(x):
            state = jnp.zeros(64)
            for _ in range(n):
                state = step(state, x)
            return state

        return run

    ones = jnp.ones(64)
    labels = ("jax", "expex")
    return [
        Measure("typed eager call", Fn(plain_identity, (x, w)), Fn(typed, (x, w)), labels=labels, compile=False,
                note="f(x: f32[B, D], w: f32[D, D]) returning x"),
        Measure("jitted call", Fn(jit_matmul, (x, w)), Fn(jitted, (x, w)), labels=labels, compile=False,
                note="x @ w, 4 x 8"),
        Measure("jitted method", Fn(jit_method, (scale, v)), Fn(scale.apply, (v,)), labels=labels, compile=False,
                note="x * self.w on a struct"),
        Measure("donate step", Fn(loop(jit_donated), (ones,)), Fn(loop(donated_step), (ones,)), labels=labels,
                compile=False, items=n, unit="steps", note='donate="auto" vs donate_argnums=0, per step'),
        Measure("eager vectorize", Fn(vmapped, (batch,)), Fn(affine.vectorize, (batch,)), labels=labels,
                compile=False, note="2x + 1 over 16 rows, not jitted"),
    ]  # fmt: skip

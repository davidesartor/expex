"""Mamba (Gu & Dao, 2023) written with typax.

A port of github.com/vvvm23/mamba-jax (Equinox), checked against it in
tests/ports/test_mamba.py. The model is the same; what changes is how it is
written:

- every module states its sizes (`Mixer[D, E, N, R, K]`), so shapes are
  checked on every call and proved for all sizes: `python -m typax examples/mamba/mamba.py`
- the selective scan is a plain `for` loop over time (staged as `lax.scan`),
  and so is the loop over layers (identical blocks: one `lax.scan`)
- sizes are used by name in the body: `xz[:, :E]`, `jnp.pad(u, ((K - 1, 0), ...))`
- no einops strings, no `jax.vmap(layer)` for per-token layers: `.lift`
"""

from __future__ import annotations  # size arithmetic in annotations (built in on Python 3.14)

import math

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx
from typax import Int, Key, f32, fn


@tx.struct
class RMSNorm[D]:
    weight: f32[D]
    bias: f32[D]
    eps: float = 1e-5

    def __init__(self, d: int, eps: float = 1e-5):
        self.weight, self.bias, self.eps = jnp.ones(d), jnp.zeros(d), eps

    def __call__(self, x: f32[D]) -> f32[D]:
        return self.weight * (jax.lax.rsqrt(jnp.mean(x**2) + self.eps) * x) + self.bias


@fn
def selective_scan[T, E, N](
    u: f32[T, E], dt: f32[T, E], A: f32[E, N], B: f32[T, N], C: f32[T, N]
) -> f32[T, E]:
    """h_t = exp(dt_t A) h_{t-1} + dt_t u_t B_t,  y_t = h_t C_t."""
    h = jnp.zeros((E, N))
    ys = []
    for t in range(T):  # a lax.scan carrying h
        h = jnp.exp(dt[t][:, None] * A) * h + (dt[t] * u[t])[:, None] * B[t]
        ys.append(h @ C[t])
    return jnp.asarray(ys)


@tx.struct
class State[E, N, K]:
    """What the mixer remembers between tokens when generating."""

    conv: f32[E, K]
    ssm: f32[E, N]


@tx.struct
class Mixer[D, E, N, R, K]:
    """The Mamba block: model width D, inner width E, state N, dt rank R, conv width K."""

    in_proj: f32[2 * E, D]
    conv_weight: f32[E, K]
    conv_bias: f32[E]
    x_proj: f32[R + 2 * N, E]
    dt_weight: f32[E, R]
    dt_bias: f32[E]
    A_log: f32[E, N]
    skip: f32[E]
    out_proj: f32[D, E]

    def __init__(self, key: Key, d: int, state: int = 16, conv: int = 4, expand: int = 2):
        e, r = expand * d, math.ceil(d / 16)
        k = jr.split(key, 6)
        self.in_proj = jr.uniform(k[0], (2 * e, d), minval=-1, maxval=1) / math.sqrt(d)
        self.conv_weight = jr.uniform(k[1], (e, conv), minval=-1, maxval=1) / math.sqrt(conv)
        self.conv_bias = jnp.zeros(e)
        self.x_proj = jr.uniform(k[2], (r + 2 * state, e), minval=-1, maxval=1) / math.sqrt(e)
        self.dt_weight = jr.uniform(k[3], (e, r), minval=-1, maxval=1) / math.sqrt(r)
        dt = jnp.exp(jr.uniform(k[4], (e,)) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = dt + jnp.log(-jnp.expm1(-dt))  # softplus(dt_bias) = dt
        self.A_log = jnp.log(jnp.tile(jnp.arange(1.0, state + 1), (e, 1)))  # S4D real
        self.skip = jnp.ones(e)
        self.out_proj = jr.uniform(k[5], (d, e), minval=-1, maxval=1) / math.sqrt(e)

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        xz = x @ self.in_proj.T
        u, z = xz[:, :E], xz[:, E:]
        u = jax.nn.silu(self.conv(u))
        dbc = u @ self.x_proj.T
        dt, B, C = dbc[:, :R], dbc[:, R : R + N], dbc[:, R + N :]
        dt = jax.nn.softplus(dt @ self.dt_weight.T + self.dt_bias)
        y = selective_scan(u, dt, -jnp.exp(self.A_log), B, C) + u * self.skip
        return (y * jax.nn.silu(z)) @ self.out_proj.T

    def conv[T](self, u: f32[T, E]) -> f32[T, E]:
        """Causal depthwise convolution over time."""
        padded = jnp.pad(u, ((K - 1, 0), (0, 0)))
        windows = padded[jnp.arange(T)[:, None] + jnp.arange(K)]  # the K inputs each output sees
        return (windows * self.conv_weight.T).sum(1) + self.conv_bias

    def step(self, x: f32[D], state: State[E, N, K]) -> tuple[f32[D], State[E, N, K]]:
        """One token, with the state the full sequence would have built."""
        xz = self.in_proj @ x
        u, z = xz[:E], xz[E:]
        conv = jnp.roll(state.conv, -1, axis=1).at[:, -1].set(u)
        u = jax.nn.silu((conv * self.conv_weight).sum(1) + self.conv_bias)
        dbc = self.x_proj @ u
        dt, B, C = dbc[:R], dbc[R : R + N], dbc[R + N :]
        dt = jax.nn.softplus(self.dt_weight @ dt + self.dt_bias)
        ssm = state.ssm * jnp.exp(dt[:, None] * -jnp.exp(self.A_log)) + u[:, None] * (dt[:, None] * B)
        y = ssm @ C + u * self.skip
        return self.out_proj @ (y * jax.nn.silu(z)), State(conv, ssm)


@tx.struct
class Block[D, E, N, R, K]:
    norm: RMSNorm[D]
    mixer: Mixer[D, E, N, R, K]

    def __call__[T](self, h: f32[T, D]) -> f32[T, D]:
        return h + self.mixer(self.norm.lift(h))

    def step(self, h: f32[D], state: State[E, N, K]) -> tuple[f32[D], State[E, N, K]]:
        y, state = self.mixer.step(self.norm(h), state)
        return h + y, state


@tx.struct
class Mamba[V, P, D, E, N, R, K]:
    """A language model: vocabulary V (embedding table padded to P rows), width D."""

    embedding: f32[P, D]
    layers: list[Block[D, E, N, R, K]]
    norm_f: RMSNorm[D]
    lm_head: f32[V, D]

    def __init__(self, key: Key, vocab: int, d: int, layers: int, state: int = 16, conv: int = 4, expand: int = 2):
        ke, kh, *kl = jr.split(key, layers + 2)
        padded = -(-vocab // 8) * 8
        self.embedding = jr.normal(ke, (padded, d))
        self.layers = [Block(RMSNorm(d), Mixer(k, d, state, conv, expand)) for k in kl]
        self.norm_f = RMSNorm(d)
        self.lm_head = jr.uniform(kh, (vocab, d), minval=-1, maxval=1) / math.sqrt(d)

    def __call__[T](self, tokens: Int[T]) -> f32[T, V]:
        h = self.embedding[tokens]
        for layer in self.layers:  # identical blocks: one lax.scan
            h = layer(h)
        return self.norm_f.lift(h) @ self.lm_head.T

    def loss[T](self, tokens: Int[T]) -> f32:
        assert T >= 2
        logp = jax.nn.log_softmax(self(tokens[:-1]))
        return -jnp.take_along_axis(logp, tokens[1:, None], axis=1).mean()

    def step(self, token: Int, states: list[State[E, N, K]]) -> tuple[f32[V], list[State[E, N, K]]]:
        h = self.embedding[token]
        new = []
        for layer, state in zip(self.layers, states):
            h, state = layer.step(h, state)
            new.append(state)
        return self.lm_head @ self.norm_f(h), new

    def generate[T](self, prompt: Int[T], n: int, key: Key, temperature: float = 1.0) -> Int[T + n]:
        """Feed the prompt one token at a time, then sample n more."""
        states = [State(jnp.zeros((E, K)), jnp.zeros((E, N))) for _ in self.layers]
        for t in range(T - 1):
            _, states = self.step(prompt[t], states)
        token, out = prompt[-1], []
        for _ in range(n):
            key, sub = jr.split(key)
            logits, states = self.step(token, states)
            token = jr.categorical(sub, logits / temperature)
            out.append(token)
        return jnp.concatenate([prompt, jnp.asarray(out)])


if __name__ == "__main__":
    model = Mamba(jr.key(0), vocab=100, d=64, layers=3)
    tokens = jr.randint(jr.key(1), (24,), 0, 100)
    print("logits", model(tokens).shape, " loss", float(model.loss(tokens)))
    print("sampled", model.generate(tokens[:8], 8, jr.key(2)))
    print()
    print(tx.prove(__import__("__main__")))

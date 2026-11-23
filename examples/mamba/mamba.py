"""Mamba (Gu & Dao, 2023) written with expex.

A port of github.com/vvvm23/mamba-jax (Equinox), checked against it in
tests/ports/test_mamba.py. The model is the same; what changes is how it is
written:

- every module states its sizes (`Mixer[D, E, N, R, K]`), so shapes are
  checked on every call and proved for all sizes: `python -m expex examples/mamba/mamba.py`
- the selective scan is a plain `for` loop over time (staged as `lax.scan`),
  and so is the loop over layers (identical blocks: one `lax.scan`)
- sizes are used by name in the body: `xz[:, :E]`, `jnp.pad(u, ((K - 1, 0), ...))`
- no einops strings, no `jax.vmap(layer)` for per-token layers: `.vectorize`
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Int, Key, f32, fn


@expex.struct
class RMSNorm[D]:
    weight: f32[D] = 1.0
    bias: f32[D] = 0.0
    eps: float = 1e-5

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


@expex.struct
class State[E, N, K]:
    """What the mixer remembers between tokens when generating."""

    conv: f32[E, K]
    ssm: f32[E, N]


# weights are used as w @ x: uniform with variance 1 / (3 input size), as Equinox's Linear
linear = jax.nn.initializers.variance_scaling(1 / 3, "fan_in", "uniform", in_axis=-1, out_axis=-2)


def dt_bias(key: Key, shape: tuple[int, ...], dtype=jnp.float32):
    """Biases whose softplus is a step size between 0.001 and 0.1, spread log-uniformly."""
    dt = jnp.exp(jr.uniform(key, shape, dtype) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
    return dt + jnp.log(-jnp.expm1(-dt))


def s4d_real(key: Key, shape: tuple[int, ...], dtype=jnp.float32):
    """log A for A = 1, 2, ..., N along the state axis (S4D real)."""
    return jnp.log(jnp.broadcast_to(jnp.arange(1, shape[-1] + 1, dtype=dtype), shape))


@expex.struct
class Mixer[D, E, N, R, K]:
    """The Mamba block: model width D, inner width E, state N, dt rank R, conv width K."""

    in_proj: f32[2 * E, D] = linear
    conv_weight: f32[E, K] = linear
    conv_bias: f32[E] = 0.0
    x_proj: f32[R + 2 * N, E] = linear
    dt_weight: f32[E, R] = linear
    dt_bias: f32[E] = dt_bias
    A_log: f32[E, N] = s4d_real
    skip: f32[E] = 1.0
    out_proj: f32[D, E] = linear

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


@expex.struct
class Block[D, E, N, R, K]:
    norm: RMSNorm[D]
    mixer: Mixer[D, E, N, R, K]

    def __call__[T](self, h: f32[T, D]) -> f32[T, D]:
        return h + self.mixer(self.norm.vectorize(h))

    def step(self, h: f32[D], state: State[E, N, K]) -> tuple[f32[D], State[E, N, K]]:
        y, state = self.mixer.step(self.norm(h), state)
        return h + y, state


@expex.struct
class Mamba[V, P, D, E, N, R, K]:
    """A language model: vocabulary V (embedding table padded to P rows), width D."""

    embedding: f32[P, D] = jax.nn.initializers.normal(1.0)
    layers: list[Block[D, E, N, R, K]]
    norm_f: RMSNorm[D]
    lm_head: f32[V, D] = linear

    @classmethod
    def create(cls, key: Key, vocab: int, d: int, layers: int, state: int = 16, conv: int = 4, expand: int = 2):
        """The original's sizes from its settings: inner width, dt rank, padded vocabulary."""
        padded = -(-vocab // 8) * 8
        return cls.init(
            key, V=vocab, P=padded, D=d, E=expand * d, N=state, R=math.ceil(d / 16), K=conv, layers=layers
        )

    def __call__[T](self, tokens: Int[T]) -> f32[T, V]:
        h = self.embedding[tokens]
        for layer in self.layers:  # identical blocks: one lax.scan
            h = layer(h)
        return self.norm_f.vectorize(h) @ self.lm_head.T

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
    model = Mamba.create(jr.key(0), vocab=100, d=64, layers=3)
    tokens = jr.randint(jr.key(1), (24,), 0, 100)
    print("logits", model(tokens).shape, " loss", float(model.loss(tokens)))
    print("sampled", model.generate(tokens[:8], 8, jr.key(2)))
    print()
    print(expex.prove(__import__("__main__")))

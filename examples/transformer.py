"""A small decoder-only transformer written with typax modules.

    python -m typax examples/transformer.py     # prove every shape
    python examples/transformer.py              # train a few steps
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx
from typax import Dim, Int, f32


class Vocab(Dim): ...


# weights are used as x @ w: variance 1 / (input size)
dense = jax.nn.initializers.variance_scaling(1.0, "fan_in", "normal")


@tx.struct
class LayerNorm[D]:
    scale: f32[D] = 1.0
    bias: f32[D] = 0.0

    def __call__(self, x: f32[D]) -> f32[D]:
        return (x - x.mean()) / (x.std() + 1e-5) * self.scale + self.bias


@tx.struct
class Attention[D]:
    wq: f32[D, D] = dense
    wk: f32[D, D] = dense
    wv: f32[D, D] = dense
    wo: f32[D, D] = dense
    heads: int = 4

    def split[T](self, x: f32[T, D]) -> f32[H, T, K]:
        H, K = self.heads, D // self.heads
        assert D % self.heads == 0
        return tx.rearrange(x, to=(H, T, K))  # D is split into H, K

    @staticmethod
    @tx.fn
    def causal[T, K](q: f32[T, K], k: f32[T, K], v: f32[T, K]) -> f32[T, K]:
        scores = q @ k.T / jnp.sqrt(q.shape[-1])
        mask = jnp.tril(jnp.ones((q.shape[0], q.shape[0]), bool))
        weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
        return weights @ v

    def __call__[T, H, K](self, x: f32[T, D]) -> f32[T, D]:
        q, k, v = self.split(x @ self.wq), self.split(x @ self.wk), self.split(x @ self.wv)
        out: f32[H, T, K] = Attention.causal.lift(q, k, v)  # over heads
        return tx.rearrange(out, to=(T, D)) @ self.wo  # H, K are merged back into D


@tx.struct
class MLP[D, F]:
    w1: f32[D, F] = dense
    w2: f32[F, D] = dense

    def __call__(self, x: f32[D]) -> f32[D]:
        return jax.nn.gelu(x @ self.w1) @ self.w2


@tx.struct
class Block[D, F]:
    norm1: LayerNorm[D]
    attn: Attention[D]
    norm2: LayerNorm[D]
    mlp: MLP[D, F]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        x = x + self.attn(self.norm1.lift(x))
        return x + self.mlp.lift(self.norm2.lift(x))


@tx.struct
class Transformer[D, F]:
    embed: f32[Vocab, D] = dense
    blocks: list[Block[D, F]]
    norm: LayerNorm[D]
    unembed: f32[D, Vocab] = dense

    def __call__[T](self, tokens: Int[T]) -> f32[T, Vocab]:
        x = self.embed[tokens]
        for block in self.blocks:  # identical blocks: one lax.scan
            x = block(x)
        return self.norm.lift(x) @ self.unembed

    def loss[T](self, tokens: Int[T]) -> f32:
        assert T >= 2
        logp = jax.nn.log_softmax(self(tokens[:-1]))
        return -jnp.take_along_axis(logp, tokens[1:, None], axis=1).mean()


if __name__ == "__main__":
    model = Transformer.init(jr.key(0), Vocab=50, D=32, F=64, blocks=4, heads=4)
    data = jr.randint(jr.key(1), (16, 24), 0, 50)

    @jax.jit
    def step(m: Transformer, batch):
        value, grads = jax.value_and_grad(lambda m: m.loss.lift(batch).mean())(m)
        return value, jax.tree.map(lambda p, g: p - 0.1 * g, m, grads)

    for i in range(5):
        value, model = step(model, data)
        print(f"step {i}: loss {value:.4f}")
    print()
    print(tx.prove(__import__("__main__")))
    print()
    print(tx.debug.shapes(Attention.__call__))

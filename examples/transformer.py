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


def dense(key, i: int, o: int):
    return jr.normal(key, (i, o)) / jnp.sqrt(i)


@tx.struct
class LayerNorm[D]:
    scale: f32[D]
    bias: f32[D]

    def __call__(self, x: f32[D]) -> f32[D]:
        return (x - x.mean()) / (x.std() + 1e-5) * self.scale + self.bias

    @staticmethod
    def init(d: int) -> "LayerNorm":
        return LayerNorm(jnp.ones(d), jnp.zeros(d))


@tx.struct(where="D % heads == 0")
class Attention[D]:
    wq: f32[D, D]
    wk: f32[D, D]
    wv: f32[D, D]
    wo: f32[D, D]
    heads: int = 4

    @tx.fn(H="heads", K="D // heads")
    def split[T, H, K](self, x: f32[T, D]) -> f32[H, T, K]:
        t, d = x.shape
        return x.reshape(t, self.heads, d // self.heads).transpose(1, 0, 2)

    @staticmethod
    @tx.fn
    def causal[T, K](q: f32[T, K], k: f32[T, K], v: f32[T, K]) -> f32[T, K]:
        scores = q @ k.T / jnp.sqrt(q.shape[-1])
        mask = jnp.tril(jnp.ones((q.shape[0], q.shape[0]), bool))
        weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
        return weights @ v

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        q, k, v = self.split(x @ self.wq), self.split(x @ self.wk), self.split(x @ self.wv)
        out = Attention.causal.lift(q, k, v)  # over heads
        return out.transpose(1, 0, 2).reshape(x.shape) @ self.wo

    @staticmethod
    def init(key, d: int, heads: int) -> "Attention":
        kq, kk, kv, ko = jr.split(key, 4)
        return Attention(dense(kq, d, d), dense(kk, d, d), dense(kv, d, d), dense(ko, d, d), heads)


@tx.struct
class MLP[D, F]:
    w1: f32[D, F]
    w2: f32[F, D]

    def __call__(self, x: f32[D]) -> f32[D]:
        return jax.nn.gelu(x @ self.w1) @ self.w2

    @staticmethod
    def init(key, d: int, f: int) -> "MLP":
        k1, k2 = jr.split(key)
        return MLP(dense(k1, d, f), dense(k2, f, d))


@tx.struct
class Block[D, F]:
    norm1: LayerNorm[D]
    attn: Attention[D]
    norm2: LayerNorm[D]
    mlp: MLP[D, F]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        x = x + self.attn(self.norm1.lift(x))
        return x + self.mlp.lift(self.norm2.lift(x))

    @staticmethod
    def init(key, d: int, f: int, heads: int) -> "Block":
        ka, km = jr.split(key)
        return Block(LayerNorm.init(d), Attention.init(ka, d, heads), LayerNorm.init(d), MLP.init(km, d, f))


@tx.struct
class Transformer[D, F]:
    embed: f32[Vocab, D]
    blocks: list[Block[D, F]]
    norm: LayerNorm[D]
    unembed: f32[D, Vocab]

    def __call__[T](self, tokens: Int[T]) -> f32[T, Vocab]:
        x = self.embed[tokens]
        for block in self.blocks:  # identical blocks: one lax.scan
            x = block(x)
        return self.norm.lift(x) @ self.unembed

    @tx.fn(where="T >= 2")
    def loss[T](self, tokens: Int[T]) -> f32:
        logp = jax.nn.log_softmax(self(tokens[:-1]))
        return -jnp.take_along_axis(logp, tokens[1:, None], axis=1).mean()

    @staticmethod
    def init(key, vocab: int, d: int, f: int, layers: int, heads: int) -> "Transformer":
        ke, ku, *kb = jr.split(key, 2 + layers)
        blocks = [Block.init(k, d, f, heads) for k in kb]
        return Transformer(dense(ke, vocab, d), blocks, LayerNorm.init(d), dense(ku, d, vocab))


if __name__ == "__main__":
    model = Transformer.init(jr.key(0), vocab=50, d=32, f=64, layers=4, heads=4)
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

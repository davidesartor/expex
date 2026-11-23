"""A small decoder-only transformer written with typax modules.

    python -m typax examples/transformer.py     # prove every shape
    python examples/transformer.py              # train a few steps
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx
from typax import Dim, Int, Key, f32


class Vocab(Dim): ...


def dense(key, i: int, o: int):
    return jr.normal(key, (i, o)) / jnp.sqrt(i)


@tx.struct
class LayerNorm[D]:
    scale: f32[D]
    bias: f32[D]

    def __init__(self, d: int):
        self.scale, self.bias = jnp.ones(d), jnp.zeros(d)

    def __call__(self, x: f32[D]) -> f32[D]:
        return (x - x.mean()) / (x.std() + 1e-5) * self.scale + self.bias


@tx.struct(where="D % heads == 0")
class Attention[D]:
    wq: f32[D, D]
    wk: f32[D, D]
    wv: f32[D, D]
    wo: f32[D, D]
    heads: int = 4

    def __init__(self, key: Key, d: int, heads: int = 4):
        kq, kk, kv, ko = jr.split(key, 4)
        self.wq, self.wk, self.wv, self.wo = (dense(k, d, d) for k in (kq, kk, kv, ko))
        self.heads = heads

    @tx.fn(H="heads", K="D // heads")
    def split[T, H, K](self, x: f32[T, D]) -> f32[H, T, K]:
        h: f32[T, H, K] = x.reshape(tx.size(T), tx.size(H), tx.size(K))
        return h.transpose(H, T, K)

    @staticmethod
    @tx.fn
    def causal[T, K](q: f32[T, K], k: f32[T, K], v: f32[T, K]) -> f32[T, K]:
        scores = q @ k.T / jnp.sqrt(q.shape[-1])
        mask = jnp.tril(jnp.ones((q.shape[0], q.shape[0]), bool))
        weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
        return weights @ v

    @tx.fn(H="heads", K="D // heads")
    def __call__[T, H, K](self, x: f32[T, D]) -> f32[T, D]:
        q, k, v = self.split(x @ self.wq), self.split(x @ self.wk), self.split(x @ self.wv)
        out: f32[H, T, K] = Attention.causal.lift(q, k, v)  # over heads
        return out.transpose(T, H, K).reshape(x.shape) @ self.wo


@tx.struct
class MLP[D, F]:
    w1: f32[D, F]
    w2: f32[F, D]

    def __init__(self, key: Key, d: int, f: int):
        k1, k2 = jr.split(key)
        self.w1, self.w2 = dense(k1, d, f), dense(k2, f, d)

    def __call__(self, x: f32[D]) -> f32[D]:
        return jax.nn.gelu(x @ self.w1) @ self.w2


@tx.struct
class Block[D, F]:
    norm1: LayerNorm[D]
    attn: Attention[D]
    norm2: LayerNorm[D]
    mlp: MLP[D, F]

    def __init__(self, key: Key, d: int, f: int, heads: int):
        ka, km = jr.split(key)
        self.norm1, self.attn = LayerNorm(d), Attention(ka, d, heads)
        self.norm2, self.mlp = LayerNorm(d), MLP(km, d, f)

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        x = x + self.attn(self.norm1.lift(x))
        return x + self.mlp.lift(self.norm2.lift(x))


@tx.struct
class Transformer[D, F]:
    embed: f32[Vocab, D]
    blocks: list[Block[D, F]]
    norm: LayerNorm[D]
    unembed: f32[D, Vocab]

    def __init__(self, key: Key, vocab: int, d: int, f: int, layers: int, heads: int):
        ke, ku, *kb = jr.split(key, 2 + layers)
        self.embed, self.unembed = dense(ke, vocab, d), dense(ku, d, vocab)
        self.blocks = [Block(k, d, f, heads) for k in kb]
        self.norm = LayerNorm(d)

    def __call__[T](self, tokens: Int[T]) -> f32[T, Vocab]:
        x = self.embed[tokens]
        for block in self.blocks:  # identical blocks: one lax.scan
            x = block(x)
        return self.norm.lift(x) @ self.unembed

    @tx.fn(where="T >= 2")
    def loss[T](self, tokens: Int[T]) -> f32:
        logp = jax.nn.log_softmax(self(tokens[:-1]))
        return -jnp.take_along_axis(logp, tokens[1:, None], axis=1).mean()


if __name__ == "__main__":
    model = Transformer(jr.key(0), vocab=50, d=32, f=64, layers=4, heads=4)
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

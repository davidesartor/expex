"""A small transformer written with typax.

    python -m typax examples/transformer.py     # prove every shape
    python examples/transformer.py                 # train a few steps
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import typax as tx
from typax import Dim, Int, Static, f32, fn


class Vocab(Dim): ...


class Block[D, F](NamedTuple):
    wq: f32[D, D]
    wk: f32[D, D]
    wv: f32[D, D]
    wo: f32[D, D]
    w1: f32[D, F]
    w2: f32[F, D]


class Model[D, F](NamedTuple):
    embed: f32[Vocab, D]
    blocks: list[Block[D, F]]
    unembed: f32[D, Vocab]


@fn
def layer_norm[D](x: f32[D]) -> f32[D]:
    return (x - x.mean()) / (x.std() + 1e-5)


@fn(H="heads", K="D // heads", where="D % heads == 0", examples={"heads": [1, 2]})
def split_heads[T, D, H, K](x: f32[T, D], heads: Static[int]) -> f32[H, T, K]:
    t, d = x.shape
    return x.reshape(t, heads, d // heads).transpose(1, 0, 2)


@fn
def causal_attention[T, K](q: f32[T, K], k: f32[T, K], v: f32[T, K]) -> f32[T, K]:
    scores = q @ k.T / jnp.sqrt(q.shape[-1])
    mask = jnp.tril(jnp.ones((q.shape[0], q.shape[0]), bool))
    weights = jax.nn.softmax(jnp.where(mask, scores, -jnp.inf), axis=-1)
    return weights @ v


@fn(examples={"heads": [1, 2]}, where="D % heads == 0")
def block[T, D, F](p: Block[D, F], x: f32[T, D], heads: Static[int]) -> f32[T, D]:
    h = layer_norm.lift(x)
    q, k, v = (split_heads(h @ w, heads) for w in (p.wq, p.wk, p.wv))
    a = causal_attention.lift(q, k, v)                      # over heads
    a = a.transpose(1, 0, 2).reshape(x.shape)
    x = x + a @ p.wo
    return x + jax.nn.gelu(layer_norm.lift(x) @ p.w1) @ p.w2


@fn(examples={"heads": [2]}, where="D % heads == 0")
def forward[T, D, F](m: Model[D, F], tokens: Int[T], heads: Static[int]) -> f32[T, Vocab]:
    x = m.embed[tokens]
    for b in m.blocks:
        x = block(b, x, heads)
    return layer_norm.lift(x) @ m.unembed


@fn(examples={"heads": [2]}, where=["D % heads == 0", "T >= 2"])
def loss[T, D, F](m: Model[D, F], tokens: Int[T], heads: Static[int]) -> f32:
    logits = forward(m, tokens[:-1], heads)
    logp = jax.nn.log_softmax(logits)
    return -jnp.take_along_axis(logp, tokens[1:, None], axis=1).mean()


def init(key, vocab: int, d: int, f: int, layers: int) -> Model:
    keys = iter(jax.random.split(key, 2 + 6 * layers))
    w = lambda *shape: jax.random.normal(next(keys), shape) / jnp.sqrt(shape[0])  # noqa: E731
    blocks = [Block(w(d, d), w(d, d), w(d, d), w(d, d), w(d, f), w(f, d)) for _ in range(layers)]
    return Model(w(vocab, d), blocks, w(d, vocab))


if __name__ == "__main__":
    model = init(jax.random.key(0), vocab=50, d=32, f=64, layers=2)
    data = jax.random.randint(jax.random.key(1), (16, 24), 0, 50)

    @jax.jit
    def step(m: Model, batch):
        def batch_loss(m):
            return loss.lift(m, batch, 4).mean()

        value, grads = jax.value_and_grad(batch_loss)(m)
        return value, jax.tree.map(lambda p, g: p - 0.1 * g, m, grads)

    for i in range(5):
        value, model = step(model, data)
        print(f"step {i}: loss {value:.4f}")
    print()
    print(tx.format_reports(tx.prove(__import__("__main__"))))
    print()
    print(tx.debug.timing(loss, model, data[0], 4))

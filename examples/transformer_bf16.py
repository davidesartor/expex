"""The transformer of `transformer.py`, computing in bfloat16 with its parameters kept in float32.

    python examples/transformer_bf16.py

`@expex.struct(compute=jnp.bfloat16)` on a subclass makes every method of the
model compute in bf16: the parameters (f32, as annotated) and the other float
inputs are cast to bf16 when a call starts, the modules it calls (blocks,
attention, MLP) compute in bf16 too, and the loss comes back as the f32 its
annotation declares, as do the gradients of the f32 parameters. `m.loss(batch)`
on a batch of sequences maps over it by itself, as for the f32 model.
"""

import dataclasses
import re

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from transformer import Transformer


@expex.struct(compute=jnp.bfloat16)
class Transformer16(Transformer):
    """The same model and annotations; only the dtype of the arithmetic changes."""


@dataclasses.dataclass
class Config:
    Vocab: int = 50
    D: int = 64
    F: int = 128
    blocks: int = 4
    heads: int = 4
    lr: float = 0.1  # not a size: the constructor ignores it


def train(model, data, cfg: Config, steps: int = 5) -> list[float]:
    @jax.jit
    def step(m, batch):
        value, grads = jax.value_and_grad(lambda m: m.loss(batch).mean())(m)
        return value, jax.tree.map(lambda p, g: p - cfg.lr * g, m, grads)

    losses = []
    for _ in range(steps):
        value, model = step(model, data)
        losses.append(float(value))
    return losses


def matmul_dtypes(model, data) -> list[str]:
    """The operand dtypes of every matrix product in the program JAX hands to XLA."""
    text = jax.jit(lambda m, t: m.loss(t).mean()).lower(model, data).as_text()
    return [m.group(1) for m in re.finditer(r"dot_general .*: \(tensor<[0-9x]*x(\w+)>", text)]


if __name__ == "__main__":
    cfg = Config()
    data = jr.randint(jr.key(1), (16, 24), 0, cfg.Vocab)
    f32_model = Transformer(jr.key(0), cfg)
    bf16_model = Transformer16(jr.key(0), cfg)  # the same parameters, stored in f32
    print("sizes:", bf16_model.sizes)

    a, b = train(f32_model, data, cfg), train(bf16_model, data, cfg)
    for i, (x, y) in enumerate(zip(a, b)):
        print(f"step {i}: loss f32 {x:.4f}   bf16 compute {y:.4f}   ({abs(x - y) / x:.2%} apart)")
    assert all(abs(x - y) < 0.02 * x for x, y in zip(a, b))

    print("matmuls, f32 model: ", matmul_dtypes(f32_model, data))
    print("matmuls, bf16 model:", matmul_dtypes(bf16_model, data))
    grads = jax.grad(lambda m: m.loss(data).mean())(bf16_model)
    print("gradient dtypes:", sorted({str(g.dtype) for g in jax.tree.leaves(grads)}))
    print(expex.cost(Transformer16.__call__, cfg, T=24))

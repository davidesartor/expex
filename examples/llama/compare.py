"""Compare the port with the original llama-2-jax, from its saved outputs.

    python examples/llama/compare.py tests/data/llama_reference.npz

The reference file is written by run_original.py: parameters of a small
model, its logits, the gradients of a loss, and the tokens its generation
loop picks greedily with the KV cache.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from llama import MLP, Attention, Block, Llama, RMSNorm

PROMPT = 6  # as in run_original.py


def from_reference(ref: dict) -> Llama:
    """The port's model with the original's parameters.

    The original lays query heads out as [R, G] (R queries per key/value head);
    the port as [G, R], each group next to its key/value head.
    """
    p = {k[len("param."):]: jnp.asarray(v) for k, v in ref.items() if k.startswith("param.")}
    dec = "model.decoder."
    layers = []
    for i in range(p[dec + "input_norm"].shape[0]):
        attn = Attention(
            wq=p[dec + "attention.q_proj"][i].transpose(0, 2, 1, 3),  # [D, R, G, K] -> [D, G, R, K]
            wk=p[dec + "attention.k_proj"][i],
            wv=p[dec + "attention.v_proj"][i],
            wo=p[dec + "attention.out_proj"][i].transpose(1, 0, 2, 3),  # [R, G, K, D] -> [G, R, K, D]
        )
        mlp = MLP(p[dec + "gate_proj"][i], p[dec + "up_proj"][i], p[dec + "down_proj"][i])
        layers.append(Block(RMSNorm(p[dec + "input_norm"][i]), attn, RMSNorm(p[dec + "post_attn_norm"][i]), mlp))
    return Llama(embed=p["model.embedding"], layers=layers, norm=RMSNorm(p["model.norm"]), unembed=p["lm_head"])


def gradients_by_reference_name(grads: Llama) -> dict[str, np.ndarray]:
    """The port's gradients, named and laid out like the original's parameters."""

    def stacked(f):
        return np.stack([np.asarray(f(b)) for b in grads.layers])

    dec = "model.decoder."
    return {
        "model.embedding": np.asarray(grads.embed), "lm_head": np.asarray(grads.unembed),
        "model.norm": np.asarray(grads.norm.weight),
        dec + "input_norm": stacked(lambda b: b.attn_norm.weight),
        dec + "post_attn_norm": stacked(lambda b: b.mlp_norm.weight),
        dec + "attention.q_proj": stacked(lambda b: b.attn.wq.transpose(0, 2, 1, 3)),
        dec + "attention.k_proj": stacked(lambda b: b.attn.wk),
        dec + "attention.v_proj": stacked(lambda b: b.attn.wv),
        dec + "attention.out_proj": stacked(lambda b: b.attn.wo.transpose(1, 0, 2, 3)),
        dec + "gate_proj": stacked(lambda b: b.mlp.gate),
        dec + "up_proj": stacked(lambda b: b.mlp.up),
        dec + "down_proj": stacked(lambda b: b.mlp.down),
    }  # fmt: skip


def compare(ref: dict) -> dict[str, float]:
    """Largest differences between the port and the original."""
    model = from_reference(ref)
    tokens = jnp.asarray(ref["tokens"])
    logits = jax.jit(lambda m, t: m.vectorize(t))(model, tokens)
    loss, grads = jax.jit(jax.value_and_grad(lambda m, t: m.loss.vectorize(t).mean()))(model, tokens)
    n = ref["generated"].shape[1]
    generated = np.stack([np.asarray(model.generate(row[:PROMPT], n))[PROMPT:] for row in tokens])
    g = gradients_by_reference_name(grads)
    scale = lambda a: max(float(np.abs(a).max()), 1e-12)  # noqa: E731
    return {
        "logits": float(np.abs(np.asarray(logits) - ref["logits"]).max()),
        "loss": abs(float(loss) - float(ref["loss"])),
        "gradients (relative)": max(
            float(np.abs(g[k] - ref[f"grad.{k}"]).max()) / scale(ref[f"grad.{k}"]) for k in g
        ),
        "generated tokens differing": float((generated != ref["generated"]).sum()),
    }


if __name__ == "__main__":
    ref = dict(np.load(sys.argv[1]))
    for name, diff in compare(ref).items():
        print(f"{name:28s} {diff:.3g}")

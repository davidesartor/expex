"""Compare the port with the original Vision Transformer, from its saved outputs.

    python examples/vit/compare.py tests/data/vit_reference.npz

The reference file is written by run_original.py: parameters of a small
model, its logits on two images, and a cross-entropy loss with its gradients.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from vit import MLP, Attention, Block, Encoder, LayerNorm, PatchEmbed, ViT


def from_reference(ref: dict) -> ViT:
    """The port's model with the original's parameters.

    The original's position embeddings are one [1, 1 + GH * GW, D] array, the
    class token's first; the port keeps the class token's apart and the
    patches' on their [GH, GW] grid. The grid's shape is the image's divided
    by the patch's.
    """
    p = {k[len("param."):]: jnp.asarray(v) for k, v in ref.items() if k.startswith("param.")}
    P, Q = p["embedding.kernel"].shape[:2]
    GH, GW = ref["images"].shape[1] // P, ref["images"].shape[2] // Q
    pos = p["Transformer.posembed_input.pos_embedding"][0]

    def norm(name: str) -> LayerNorm:
        return LayerNorm(p[name + ".scale"], p[name + ".bias"])

    def block(i: int) -> Block:
        b = f"Transformer.encoderblock_{i}."
        a = b + "MultiHeadDotProductAttention_0."
        attn = Attention(*(p[a + n + "." + w] for n in ("query", "key", "value", "out") for w in ("kernel", "bias")))
        m = b + "MlpBlock_0."
        mlp = MLP(p[m + "Dense_0.kernel"], p[m + "Dense_0.bias"], p[m + "Dense_1.kernel"], p[m + "Dense_1.bias"])
        return Block(norm(b + "LayerNorm_0"), attn, norm(b + "LayerNorm_1"), mlp)

    n = sum(k.startswith("Transformer.encoderblock_") and k.endswith("LayerNorm_0.scale") for k in p)
    return ViT(
        embed=PatchEmbed(p["embedding.kernel"], p["embedding.bias"]),
        cls=p["cls"][0, 0], cls_pos=pos[0], pos=pos[1:].reshape(GH, GW, -1),
        encoder=Encoder([block(i) for i in range(n)], norm("Transformer.encoder_norm")),
        head=p["head.kernel"], head_bias=p["head.bias"],
    )  # fmt: skip


def gradients_by_reference_name(grads: ViT) -> dict[str, np.ndarray]:
    """The port's gradients, named and laid out like the original's parameters."""
    out = {
        "embedding.kernel": grads.embed.kernel, "embedding.bias": grads.embed.bias,
        "cls": grads.cls[None, None],
        "Transformer.posembed_input.pos_embedding": jnp.concatenate([grads.cls_pos[None], grads.pos.reshape(-1, grads.pos.shape[-1])])[None],
        "Transformer.encoder_norm.scale": grads.encoder.norm.scale,
        "Transformer.encoder_norm.bias": grads.encoder.norm.bias,
        "head.kernel": grads.head, "head.bias": grads.head_bias,
    }  # fmt: skip
    for i, b in enumerate(grads.encoder.layers):
        pre = f"Transformer.encoderblock_{i}."
        a = pre + "MultiHeadDotProductAttention_0."
        attn = b.attn
        for n, w, bias in (("query", attn.wq, attn.bq), ("key", attn.wk, attn.bk), ("value", attn.wv, attn.bv), ("out", attn.wo, attn.bo)):  # fmt: skip
            out[a + n + ".kernel"], out[a + n + ".bias"] = w, bias
        for j, norm in enumerate((b.attn_norm, b.mlp_norm)):
            out[pre + f"LayerNorm_{j}.scale"], out[pre + f"LayerNorm_{j}.bias"] = norm.scale, norm.bias
        m = pre + "MlpBlock_0."
        out[m + "Dense_0.kernel"], out[m + "Dense_0.bias"] = b.mlp.w1, b.mlp.b1
        out[m + "Dense_1.kernel"], out[m + "Dense_1.bias"] = b.mlp.w2, b.mlp.b2
    return {k: np.asarray(v) for k, v in out.items()}


def compare(ref: dict) -> dict[str, float]:
    """Largest differences between the port and the original."""
    model = from_reference(ref)
    images, labels = jnp.asarray(ref["images"]), jnp.asarray(ref["labels"])
    logits = jax.jit(lambda m, x: m(x))(model, images)
    loss, grads = jax.jit(jax.value_and_grad(lambda m: m.loss(images, labels).mean()))(model)
    g = gradients_by_reference_name(grads)
    names = {k[len("grad."):] for k in ref if k.startswith("grad.")}
    assert names == g.keys(), names ^ g.keys()  # every parameter compared
    # a key bias shifts every score of a query alike, which the softmax ignores:
    # its gradient is zero, and both sides hold rounding noise (about 1e-8)
    zero = {k for k in g if k.endswith("key.bias")}
    diff = {k: float(np.abs(g[k] - ref[f"grad.{k}"]).max()) for k in g}
    return {
        "logits": float(np.abs(np.asarray(logits) - ref["logits"]).max()),
        "loss": abs(float(loss) - float(ref["loss"])),
        "gradients (relative)": max(diff[k] / float(np.abs(ref[f"grad.{k}"]).max()) for k in g.keys() - zero),
        "key-bias gradients (zero)": max(max(diff[k], float(np.abs(g[k]).max())) for k in zero),
    }


if __name__ == "__main__":
    ref = dict(np.load(sys.argv[1]))
    for name, diff in compare(ref).items():
        print(f"{name:28s} {diff:.3g}")

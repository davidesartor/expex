"""Compare the port with the original mamba-jax, from its saved outputs.

    python examples/mamba/compare.py tests/data/mamba_reference.npz

The reference file is written by run_original.py: parameters of a small
model, its logits (with both of its scan kernels), the gradients of a loss,
and tokens sampled with the recurrent step.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from mamba import Block, Mamba, Mixer, RMSNorm


def from_reference(ref: dict) -> Mamba:
    """The port's model with the original's parameters."""
    p = {k[len("param."):]: jnp.asarray(v) for k, v in ref.items() if k.startswith("param.")}
    d = p["model.norm_f.weight"].shape[0]

    def norm(prefix: str) -> RMSNorm:
        return RMSNorm(d).replace(weight=p[f"{prefix}.weight"], bias=p[f"{prefix}.bias"])

    blocks = []
    i = 0
    while f"model.layers[{i}].norm.weight" in p:
        m = f"model.layers[{i}].mixer"
        mixer = Mixer(jax.random.key(0), d).replace(
            in_proj=p[f"{m}.in_proj.weight"],
            conv_weight=p[f"{m}.conv1d.weight"][:, 0, :],
            conv_bias=p[f"{m}.conv1d.bias"][:, 0],
            x_proj=p[f"{m}.x_proj.weight"],
            dt_weight=p[f"{m}.dt_proj.weight"],
            dt_bias=p[f"{m}.dt_proj.bias"],
            A_log=p[f"{m}.A_log"],
            skip=p[f"{m}.D"],
            out_proj=p[f"{m}.out_proj.weight"],
        )
        blocks.append(Block(norm(f"model.layers[{i}].norm"), mixer))
        i += 1
    model = Mamba(jax.random.key(0), vocab=p["lm_head.weight"].shape[0], d=d, layers=len(blocks))
    return model.replace(
        embedding=p["model.embedding.weight"], layers=blocks, norm_f=norm("model.norm_f"),
        lm_head=p["lm_head.weight"],
    )  # fmt: skip


def gradients_by_reference_name(grads: Mamba) -> dict[str, np.ndarray]:
    """The port's gradients, named like the original's parameters."""
    out = {
        "model.embedding.weight": grads.embedding, "lm_head.weight": grads.lm_head,
        "model.norm_f.weight": grads.norm_f.weight, "model.norm_f.bias": grads.norm_f.bias,
    }  # fmt: skip
    for i, b in enumerate(grads.layers):
        m, x = f"model.layers[{i}].mixer", b.mixer
        out.update({
            f"model.layers[{i}].norm.weight": b.norm.weight, f"model.layers[{i}].norm.bias": b.norm.bias,
            f"{m}.in_proj.weight": x.in_proj, f"{m}.conv1d.weight": x.conv_weight[:, None, :],
            f"{m}.conv1d.bias": x.conv_bias[:, None], f"{m}.x_proj.weight": x.x_proj,
            f"{m}.dt_proj.weight": x.dt_weight, f"{m}.dt_proj.bias": x.dt_bias,
            f"{m}.A_log": x.A_log, f"{m}.D": x.skip, f"{m}.out_proj.weight": x.out_proj,
        })  # fmt: skip
    return {k: np.asarray(v) for k, v in out.items()}


def compare(ref: dict) -> dict[str, float]:
    """Largest differences between the port and the original."""
    model = from_reference(ref)
    tokens = jnp.asarray(ref["tokens"])
    logits = jax.jit(lambda m, t: m(t))(model, tokens)
    loss, grads = jax.jit(jax.value_and_grad(lambda m, t: m.loss(t)))(model, tokens)
    sampled = model.generate(tokens[:8], 8, jax.random.PRNGKey(2))
    g = gradients_by_reference_name(grads)
    scale = lambda a: max(float(np.abs(a).max()), 1e-12)  # noqa: E731
    return {
        "logits": float(np.abs(np.asarray(logits) - ref["logits"]).max()),
        "logits (associative scan)": float(np.abs(np.asarray(logits) - ref["logits_assoc"]).max()),
        "loss": abs(float(loss) - float(ref["loss"])),
        "gradients (relative)": max(
            float(np.abs(g[k] - ref[f"grad.{k}"]).max()) / scale(ref[f"grad.{k}"]) for k in g
        ),
        "sampled tokens differing": float((np.asarray(sampled) != ref["sampled"]).sum()),
    }


if __name__ == "__main__":
    ref = dict(np.load(sys.argv[1]))
    for name, diff in compare(ref).items():
        print(f"{name:28s} {diff:.3g}")

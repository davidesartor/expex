"""Run the original mamba-jax (github.com/vvvm23/mamba-jax) and save reference outputs.

    git clone https://github.com/vvvm23/mamba-jax /tmp/mamba-jax
    PYTHONPATH=/tmp/mamba-jax python examples/mamba/run_original.py tests/data/mamba_reference.npz

Saves the parameters of a small randomly initialised model, its logits on a
token sequence, the gradients of a loss, and tokens sampled with the
recurrent step, for tests/ports/test_mamba.py.
"""

import sys
import types

# The package imports its pretrained-weights loader (which needs torch and
# huggingface) at import time; the model itself does not need it.
_loader = types.ModuleType("mamba_jax.modelling.equinox.loader")
_loader.load_pretrained = None  # type: ignore[attr-defined]
sys.modules[_loader.__name__] = _loader

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

_clip = jnp.clip  # the original (January 2024) uses the old keyword names
jnp.clip = lambda x, min=None, max=None, a_min=None, a_max=None, **k: _clip(  # type: ignore[assignment]
    x, min if a_min is None else a_min, max if a_max is None else a_max, **k
)
from mamba_jax.kernels.interface import KernelType
from mamba_jax.modelling.equinox.model import MambaLLM

CONFIG = dict(dim=64, num_layers=3, vocab_size=100, state_dim=16, kernel_size=4, expand=2)


def main(out: str) -> None:
    model = MambaLLM(**CONFIG, kernel_mode=KernelType.XLA, key=jax.random.PRNGKey(0))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (24,), 0, CONFIG["vocab_size"])

    logits = model(tokens)
    assoc = jax.tree_util.tree_map(lambda x: x, model)
    for layer in assoc.model.layers:  # the associative-scan kernel, for comparison
        object.__setattr__(layer.mixer, "kernel_mode", KernelType.XLA_ASSOCIATIVE)
    logits_assoc = assoc(tokens)

    def loss(m):
        lp = jax.nn.log_softmax(m(tokens[:-1]))
        return -jnp.take_along_axis(lp, tokens[1:, None], axis=1).mean()

    value, grads = eqx.filter_value_and_grad(loss)(model)
    sampled = model.generate(tokens[:8], gen_len=8, temperature=1.0, key=jax.random.PRNGKey(2))

    arrays = {}
    for prefix, tree in (("param", model), ("grad", grads)):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            if isinstance(leaf, jax.Array) and leaf.dtype != jnp.dtype("O"):
                arrays[f"{prefix}{jax.tree_util.keystr(path)}"] = np.asarray(leaf)
    np.savez(
        out, tokens=np.asarray(tokens), logits=np.asarray(logits), logits_assoc=np.asarray(logits_assoc),
        loss=np.asarray(value), sampled=np.asarray(sampled), **arrays,
    )  # fmt: skip
    print(f"saved {len(arrays)} arrays to {out}; loss {float(value):.6f}")


if __name__ == "__main__":
    main(sys.argv[1])

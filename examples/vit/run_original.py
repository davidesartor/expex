"""Run the original Vision Transformer (github.com/google-research/vision_transformer) and save reference outputs.

    git clone https://github.com/google-research/vision_transformer /tmp/vit    # 64801f1, August 2026
    pip install flax
    PYTHONPATH=/tmp/vit python examples/vit/run_original.py tests/data/vit_reference.npz

Saves the parameters of a small randomly initialised `vit_jax.models_vit.VisionTransformer`
(width 32, 4 heads, MLP width 48, 2 layers, 10 classes), its logits on two images, a
cross-entropy loss and its gradients, for tests/ports/test_vit.py. The images
are 32 x 24 and the patches 4 x 2, a grid of 8 x 12, so that a port mixing up
height and width, or rows and columns of the grid, does not match.
"""

import sys
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from vit_jax.models_vit import VisionTransformer


class Patches(NamedTuple):  # the original's configs are ml_collections.ConfigDicts with `.size`
    size: tuple[int, int]


MODEL = VisionTransformer(
    num_classes=10,
    patches=Patches(size=(4, 2)),
    hidden_size=32,
    transformer=dict(mlp_dim=48, num_heads=4, num_layers=2, attention_dropout_rate=0.0, dropout_rate=0.0),
    classifier="token",
    representation_size=None,
)
IMAGES = (2, 32, 24, 3)  # batch, height, width, channels


def main(out: str) -> None:
    images = jax.random.normal(jax.random.key(1), IMAGES)
    labels = jax.random.randint(jax.random.key(2), (IMAGES[0],), 0, MODEL.num_classes)
    params = MODEL.init(jax.random.key(0), images, train=False)["params"]
    # the class token and the head start at zero, which would leave every other
    # gradient zero: give them random values, as after some training
    k1, k2 = jax.random.split(jax.random.key(3))
    params["cls"] = 0.02 * jax.random.normal(k1, params["cls"].shape)
    params["head"]["kernel"] = 0.2 * jax.random.normal(k2, params["head"]["kernel"].shape)

    logits = MODEL.apply({"params": params}, images, train=False)

    def loss(p):
        logp = jax.nn.log_softmax(MODEL.apply({"params": p}, images, train=False))
        return -jnp.take_along_axis(logp, labels[:, None], axis=1).mean()

    value, grads = jax.jit(jax.value_and_grad(loss))(params)

    arrays = {}
    for prefix, tree in (("param", params), ("grad", grads)):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            arrays[f"{prefix}.{jax.tree_util.keystr(path, simple=True, separator='.')}"] = np.asarray(leaf)
    np.savez_compressed(
        out, images=np.asarray(images), labels=np.asarray(labels), logits=np.asarray(logits),
        loss=np.asarray(value), **arrays,
    )  # fmt: skip
    print(f"saved {len(arrays)} arrays to {out}; loss {float(value):.6f}; logits {logits.shape}")


if __name__ == "__main__":
    main(sys.argv[1])

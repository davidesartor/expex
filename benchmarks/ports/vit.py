"""google-research/vision_transformer (Flax) vs examples/vit: forward and gradient, same parameters.

The original model class and its `Patches` stand-in come from
examples/vit/run_original.py; the port gets the original's parameters through
examples/vit/compare.py's `from_reference`. No speed claim exists yet.
"""

import jax
import jax.numpy as jnp
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_example, use_original

NAME = "vit"
ORIGINALS = ["vision_transformer"]
CLAIM: dict[str, str] = {}

CONFIGS = {
    "quick": dict(hidden=64, heads=4, mlp=128, layers=2, image=32, patch=4, batch=8, classes=10),
    # ViT-Ti/16 at 224 x 224
    "tiny": dict(hidden=192, heads=3, mlp=768, layers=12, image=224, patch=16, batch=8, classes=1000),
    # ViT-B/16 at 224 x 224, a training-sized batch
    "realistic": dict(hidden=768, heads=12, mlp=3072, layers=12, image=224, patch=16, batch=64, classes=1000),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["tiny"], "gpu": ["tiny", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    use_original("vision_transformer")
    use_example("vit")
    ro = load_module("vit_run_original", EXAMPLES / "vit" / "run_original.py")
    compare = load_module("vit_compare", EXAMPLES / "vit" / "compare.py")

    model = ro.VisionTransformer(
        num_classes=cfg["classes"], patches=ro.Patches(size=(cfg["patch"], cfg["patch"])), hidden_size=cfg["hidden"],
        transformer=dict(mlp_dim=cfg["mlp"], num_heads=cfg["heads"], num_layers=cfg["layers"],
                         attention_dropout_rate=0.0, dropout_rate=0.0),
        classifier="token", representation_size=None,
    )  # fmt: skip
    B, S = cfg["batch"], cfg["image"]
    images = jax.random.normal(jax.random.key(1), (B, S, S, 3))
    labels = jax.random.randint(jax.random.key(2), (B,), 0, cfg["classes"])
    params = model.init(jax.random.key(0), images[:1], train=False)["params"]
    k1, k2 = jax.random.split(jax.random.key(3))  # as run_original.py: non-zero class token and head
    params["cls"] = 0.02 * jax.random.normal(k1, params["cls"].shape)
    params["head"]["kernel"] = 0.2 * jax.random.normal(k2, params["head"]["kernel"].shape)
    ref = {f"param.{jax.tree_util.keystr(p, simple=True, separator='.')}": np.asarray(x)
           for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}  # fmt: skip
    ref["images"] = np.asarray(images)
    port = compare.from_reference(ref)

    def loss_original(p, x, y):  # as in run_original.py
        logp = jax.nn.log_softmax(model.apply({"params": p}, x, train=False))
        return -jnp.take_along_axis(logp, y[:, None], axis=1).mean()

    def max_diff(a, b):
        return {"max abs diff": float(jnp.abs(a - b).max())}

    def loss_diff(a, b):
        return {"loss abs diff": abs(float(a[0]) - float(b[0]))}

    return [
        Measure(
            "forward",
            Fn(jax.jit(lambda p, x: model.apply({"params": p}, x, train=False)), (params, images)),
            Fn(jax.jit(lambda m, x: m.vectorize(x)), (port, images)),
            items=B, unit="images", check=max_diff, note=f"batch {B} of {S} x {S}",
        ),
        Measure(
            "gradient",
            Fn(jax.jit(jax.value_and_grad(loss_original)), (params, images, labels)),
            Fn(jax.jit(jax.value_and_grad(lambda m, x, y: m.loss.vectorize(x, y).mean())), (port, images, labels)),
            items=B, unit="images", check=loss_diff,
        ),
    ]

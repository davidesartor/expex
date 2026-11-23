"""Run the original JaxNeRF (github.com/google-research/google-research, jaxnerf/) and save reference outputs.

    git clone --filter=blob:none --sparse https://github.com/google-research/google-research /tmp/gr
    git -C /tmp/gr sparse-checkout set jaxnerf        # 87e25a5, November 2026
    pip install flax                                  # 0.12; jaxnerf uses only flax.linen
    PYTHONPATH=/tmp/gr python examples/nerf/run_original.py tests/data/nerf_reference.npz

Builds a small model with the original's own `construct_nerf`, makes 64 rays
with its own `_generate_rays` (an 8 x 8 camera), and saves the parameters,
coarse and fine rgb / disp / acc with randomized=True and False, a training
loss (as in train.py: coarse plus fine MSE) and its gradients, for
tests/ports/test_nerf.py.
"""

import sys
import types

import jax
import jax.numpy as jnp
import numpy as np

# jaxnerf.nerf.utils imports flax.optim (removed from flax since) for the type
# of its TrainState; datasets imports cv2 to load images. Neither is used here.
_optim = types.ModuleType("flax.optim")
_optim.Optimizer = object  # type: ignore[attr-defined]
sys.modules["flax.optim"] = _optim
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from jaxnerf.nerf import datasets, models, utils

# the original's flag defaults (utils.define_flags), smaller where it matters for the file size
CONFIG = dict(
    model="nerf", near=2.0, far=6.0, net_depth=4, net_width=32, net_depth_condition=1, net_width_condition=16,
    skip_layer=2, num_rgb_channels=3, num_sigma_channels=1, randomized=True, min_deg_point=0, max_deg_point=10,
    deg_view=4, num_coarse_samples=64, num_fine_samples=128, use_viewdirs=True, noise_std=None, lindisp=False,
    net_activation="relu", rgb_activation="sigmoid", sigma_activation="relu", legacy_posenc_order=False,
    white_bkgd=True,
)  # fmt: skip
SIDE, FOCAL = 8, 10.0  # an 8 x 8 camera: 64 rays


def look_at(eye):
    """Camera-to-world matrix [3, 4] of a camera at `eye` looking at the origin (OpenGL: it looks down -z)."""
    back = eye / np.linalg.norm(eye)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back, eye], axis=1).astype(np.float32)


def camera_rays(pose):
    cam = types.SimpleNamespace(w=SIDE, h=SIDE, focal=FOCAL, camtoworlds=pose[None], use_pixel_centers=False)
    datasets.Dataset._generate_rays(cam)  # the original's ray generation, on a stand-in dataset
    return utils.namedtuple_map(lambda r: jnp.asarray(r.reshape(-1, 3)), cam.rays)


def main(out: str) -> None:
    pose = look_at(np.array([3.0, -2.0, 1.5]))
    rays = camera_rays(pose)
    args = types.SimpleNamespace(**CONFIG)
    example = {"rays": utils.namedtuple_map(lambda r: r[None], rays)}  # with the leading device axis of train.py
    model, variables = models.construct_nerf(jax.random.PRNGKey(0), example, args)
    rng_0, rng_1 = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    pixels = jax.random.uniform(jax.random.PRNGKey(3), (SIDE * SIDE, 3))

    render = jax.jit(model.apply, static_argnums=4)
    outputs = {}
    for randomized in (True, False):
        (rgb_c, disp_c, acc_c), (rgb, disp, acc) = render(variables, rng_0, rng_1, rays, randomized)
        for name, x in dict(rgb_c=rgb_c, disp_c=disp_c, acc_c=acc_c, rgb=rgb, disp=disp, acc=acc).items():
            outputs[f"{name}.{'random' if randomized else 'fixed'}"] = np.asarray(x)

    def loss(variables):  # train.py's loss_fn, without weight decay (weight_decay_mult = 0)
        (rgb_c, _, _), (rgb, _, _) = model.apply(variables, rng_0, rng_1, rays, True)
        return ((rgb - pixels) ** 2).mean() + ((rgb_c - pixels) ** 2).mean()

    value, grads = jax.jit(jax.value_and_grad(loss))(variables)

    arrays = {}
    for prefix, tree in (("param", variables["params"]), ("grad", grads["params"])):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            arrays[f"{prefix}.{jax.tree_util.keystr(path, simple=True, separator='.')}"] = np.asarray(leaf)
    np.savez_compressed(
        out, pose=pose, focal=np.float32(FOCAL), origins=np.asarray(rays.origins),
        directions=np.asarray(rays.directions), viewdirs=np.asarray(rays.viewdirs),
        rng_0=np.asarray(rng_0), rng_1=np.asarray(rng_1), pixels=np.asarray(pixels), loss=np.asarray(value),
        **outputs, **arrays,
    )  # fmt: skip
    print(f"saved {len(arrays)} parameter and gradient arrays to {out}; loss {float(value):.6f}")


if __name__ == "__main__":
    main(sys.argv[1])

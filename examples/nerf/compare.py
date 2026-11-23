"""Compare the port with the original JaxNeRF, from its saved outputs.

    python examples/nerf/compare.py tests/data/nerf_reference.npz

The reference file is written by run_original.py: parameters of a small
model, the coarse and fine rgb / disp / acc of 64 rays with randomized=True
and False, the training loss and its gradients.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from nerf import MLP, Dense, NeRF, Ray, camera_ray

SIDE = 8  # the camera of run_original.py: 8 x 8 pixels


def mlp_from(p: dict, name: str) -> MLP:
    """The port's MLP from Flax's `name.Dense_i.{kernel,bias}`: hidden layers, then sigma, bottleneck, condition, rgb."""
    dense = [Dense(p[f"{name}.Dense_{i}.kernel"], p[f"{name}.Dense_{i}.bias"]) for i in range(len(p) // 2)]
    *hidden, sigma, bottleneck, condition, rgb = dense
    width = hidden[0].kernel.shape[1]
    skip = next(i for i, d in enumerate(hidden) if i > 0 and d.kernel.shape[0] != width)  # the layer taking W + P
    return MLP(hidden[0], hidden[1:skip], hidden[skip], hidden[skip + 1 :], sigma, bottleneck, condition, rgb)


def from_reference(ref: dict) -> NeRF:
    """The port's model with the original's parameters (MLP_0 is the coarse MLP, MLP_1 the fine one)."""
    p = {k[len("param.") :]: jnp.asarray(v) for k, v in ref.items() if k.startswith("param.")}
    coarse = mlp_from({k: v for k, v in p.items() if k.startswith("MLP_0.")}, "MLP_0")
    fine = mlp_from({k: v for k, v in p.items() if k.startswith("MLP_1.")}, "MLP_1")
    return NeRF(coarse, fine)  # the other settings: the original's defaults, as run_original.py uses


def gradients_by_reference_name(grads: NeRF) -> dict[str, np.ndarray]:
    """The port's gradients, named like the original's parameters."""
    out = {}
    for name, mlp in (("MLP_0", grads.coarse), ("MLP_1", grads.fine)):
        layers = [mlp.first, *mlp.before_skip, mlp.skip, *mlp.after_skip, mlp.sigma, mlp.bottleneck]
        for i, d in enumerate([*layers, mlp.condition, mlp.rgb]):
            out[f"{name}.Dense_{i}.kernel"], out[f"{name}.Dense_{i}.bias"] = np.asarray(d.kernel), np.asarray(d.bias)
    return out


def rays_from(ref: dict) -> Ray:
    return Ray(jnp.asarray(ref["origins"]), jnp.asarray(ref["directions"]), jnp.asarray(ref["viewdirs"]))


def compare(ref: dict) -> dict[str, float]:
    """Largest differences between the port and the original."""
    model, rays = from_reference(ref), rays_from(ref)
    keys = jnp.asarray(ref["rng_0"]), jnp.asarray(ref["rng_1"])
    render = jax.jit(NeRF.__call__, static_argnums=4)
    diffs: dict[str, float] = {}
    for mode, randomized in (("random", True), ("fixed", False)):
        coarse, fine = render(model, rays, *keys, randomized)
        for name, x in dict(rgb_c=coarse.rgb, disp_c=coarse.disp, acc_c=coarse.acc).items():
            diffs[f"{name} ({mode})"] = float(np.abs(np.asarray(x) - ref[f"{name}.{mode}"]).max())
        for name, x in dict(rgb=fine.rgb, disp=fine.disp, acc=fine.acc).items():
            diffs[f"{name} ({mode})"] = float(np.abs(np.asarray(x) - ref[f"{name}.{mode}"]).max())

    pixels = jnp.asarray(ref["pixels"])
    loss, grads = jax.jit(jax.value_and_grad(NeRF.loss))(model, rays, pixels, *keys)
    g = gradients_by_reference_name(grads)
    scale = lambda a: max(float(np.abs(a).max()), 1e-12)  # noqa: E731
    diffs["loss"] = abs(float(loss) - float(ref["loss"]))
    diffs["gradients (relative)"] = max(float(np.abs(g[k] - ref[f"grad.{k}"]).max()) / scale(ref[f"grad.{k}"]) for k in g)

    pose, focal = jnp.asarray(ref["pose"]), float(ref["focal"])
    ys, xs = jnp.meshgrid(jnp.arange(SIDE, dtype=jnp.float32), jnp.arange(SIDE, dtype=jnp.float32), indexing="ij")
    made = camera_ray.vectorize(pose, focal, xs.ravel(), ys.ravel(), SIDE, SIDE)
    diffs["camera rays"] = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(made, rays))
    return diffs


if __name__ == "__main__":
    ref = dict(np.load(sys.argv[1]))
    for name, diff in compare(ref).items():
        print(f"{name:24s} {diff:.3g}")

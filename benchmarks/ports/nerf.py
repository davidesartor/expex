"""JaxNeRF (google-research/jaxnerf) vs examples/nerf: rendering a batch of rays and the training gradient.

The original model is built by its own `construct_nerf` with
examples/nerf/run_original.py's settings and import stubs, its rays by its own
ray generation (run_original.camera_rays); the port gets the original's
parameters through examples/nerf/compare.py's `from_reference`. Both sample
64 coarse and 128 fine points per ray (the original's defaults, which the
port's `from_reference` assumes). No speed claim exists yet.
"""

import types

import jax
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_example, use_original

NAME = "nerf"
ORIGINALS = ["google-research"]
CLAIM: dict[str, str] = {}

CONFIGS = {
    "quick": dict(side=8, net_depth=4, net_width=32, net_width_condition=16, skip_layer=2),
    # JaxNeRF's defaults: 8 layers of 256, a 1,024-ray batch (32 x 32 camera)
    "default": dict(side=32, net_depth=8, net_width=256, net_width_condition=128, skip_layer=4),
    "realistic": dict(side=64, net_depth=8, net_width=256, net_width_condition=128, skip_layer=4),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["default"], "gpu": ["default", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    use_original("google-research")
    use_example("nerf")
    ro = load_module("nerf_run_original", EXAMPLES / "nerf" / "run_original.py")  # stubs, CONFIG, camera_rays
    compare = load_module("nerf_compare", EXAMPLES / "nerf" / "compare.py")

    ro.SIDE = cfg["side"]  # read by camera_rays
    rays = ro.camera_rays(ro.look_at(np.array([3.0, -2.0, 1.5])))
    args = types.SimpleNamespace(**{**ro.CONFIG, **{k: v for k, v in cfg.items() if k != "side"}})
    example = {"rays": ro.utils.namedtuple_map(lambda r: r[None], rays)}
    model, variables = ro.models.construct_nerf(jax.random.PRNGKey(0), example, args)
    rng_0, rng_1 = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    n = cfg["side"] ** 2
    pixels = jax.random.uniform(jax.random.PRNGKey(3), (n, 3))

    ref = {f"param.{jax.tree_util.keystr(p, simple=True, separator='.')}": np.asarray(x)
           for p, x in jax.tree_util.tree_flatten_with_path(variables["params"])[0]}  # fmt: skip
    ref.update(origins=np.asarray(rays.origins), directions=np.asarray(rays.directions), viewdirs=np.asarray(rays.viewdirs))
    port, port_rays = compare.from_reference(ref), compare.rays_from(ref)

    def loss_original(v, rays, pixels, r0, r1):  # run_original.py's (train.py's) loss
        (rgb_c, _, _), (rgb, _, _) = model.apply(v, r0, r1, rays, True)
        return ((rgb - pixels) ** 2).mean() + ((rgb_c - pixels) ** 2).mean()

    def rgb_diff(a, b):
        return {"fine rgb max abs diff": float(np.abs(np.asarray(a[1][0]) - np.asarray(b[1].rgb)).max())}

    def loss_diff(a, b):
        return {"loss abs diff": abs(float(a[0]) - float(b[0]))}

    return [
        Measure(
            "render",
            Fn(jax.jit(lambda v, rays, r0, r1: model.apply(v, r0, r1, rays, True)), (variables, rays, rng_0, rng_1)),
            Fn(jax.jit(lambda m, rays, r0, r1: m(rays, r0, r1, True)), (port, port_rays, rng_0, rng_1)),
            items=n, unit="rays", check=rgb_diff, note=f"{n} rays, randomized, 64 + 128 samples",
        ),
        Measure(
            "gradient",
            Fn(jax.jit(jax.value_and_grad(loss_original)), (variables, rays, pixels, rng_0, rng_1)),
            Fn(jax.jit(jax.value_and_grad(lambda m, *a: m.loss(*a))), (port, port_rays, pixels, rng_0, rng_1)),
            items=n, unit="rays", check=loss_diff,
        ),
    ]

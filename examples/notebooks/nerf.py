"""Source of nerf.ipynb (see build.py)."""

# %% [header] NeRF in expex

# %% [markdown]
# A port of [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf)
# (Flax) to expex: [`examples/nerf/nerf.py`](../nerf/nerf.py), 138 lines of code against 334.
# Everything is written for one ray (`S` samples along it, `F` more drawn from its coarse
# weights); called on a batch of rays, it maps over them, so there is no batch axis to carry.
# The MLP is written for one point and states its sizes (`MLP[P, V, W, C]`).
#
# This notebook renders a few rays, shows what expex reads from the annotations, checks the port
# against the original's saved outputs, and times it.

# %% [setup] nerf

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import nerf  # examples/nerf/nerf.py
from nerf import NeRF, Ray

print(inspect.getsource(nerf.NeRF.render))
print(inspect.getsource(nerf.composite))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method in the module, as expex reads it:

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(nerf):
    print(signature(report.fn))

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line, for all sizes at once. Here the
# hierarchical sampling: `F` depths drawn from the coarse weights on `N` bins.

# %%
print(expex.debug.shapes(nerf.sample_pdf))

# %% [markdown]
# ## Run it
#
# A small NeRF (8 layers of width 64) renders an 8 x 8 camera looking at the origin: 64 rays,
# each with 64 coarse and 128 fine samples. Untrained, so the picture is noise; the point is the
# shapes.

# %%
model = NeRF.create(jr.key(0), depth=8, width=64, width_condition=32)
pose = jnp.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4.0]])  # camera-to-world: at z = 4, looking down -z
ys, xs = jnp.meshgrid(jnp.arange(8.0), jnp.arange(8.0), indexing="ij")
rays = nerf.camera_ray(pose, 10.0, xs.ravel(), ys.ravel(), 8, 8)  # mapped over the 64 pixels
print("rays:", expex.show(rays))

render = jax.jit(NeRF.__call__, static_argnums=4)
coarse, fine = render(model, rays, jr.key(1), jr.key(2), True)
print("coarse:", expex.show(coarse))
print("fine opacity:\n", np.asarray(fine.acc.reshape(8, 8)).round(2))

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(nerf)

# %% [markdown]
# **A summary** of the model. The MLPs' view-direction width `V` only appears inside `W + V`
# (the colour branch takes the features and the encoded direction), so it cannot be read back
# from the parameters alone: the summary shows `V=?`, and estimates for `NeRF`'s own methods need
# the call's arguments. The MLP and the compositing are costed as formulas:

# %%
expex.summary(model)

# %%
print(expex.cost(nerf.MLP.__call__))
print(expex.cost(nerf.composite))

# %% [markdown]
# **A diagram** of the MLP, drawn from the code (interactive when you run the notebook): the
# layers before and after the skip connection are each one `lax.scan`, drawn as `× L`.

# %%
expex.diagram(nerf.MLP.__call__, depth=2)

# %% [markdown]
# **`expex.inspect`**: every tool in one page, for the coarse MLP on the 64 encoded sample
# points of the first ray (it is written for one point, so it maps over them):

# %%
points = rays.origin[0] + jnp.outer(jnp.linspace(2.0, 6.0, 64), rays.direction[0])
encoded = nerf.posenc(points, 0, 10)  # mapped over the 64 points
view = nerf.posenc(rays.viewdir[0], 0, 4)
expex.inspect(model.coarse, encoded, jnp.broadcast_to(view, (64, view.shape[0])))

# %% [markdown]
# ## A shape error
#
# Rays are `Ray(origin: f32[XYZ], ...)`, with `XYZ` a size fixed at 3. Rays in 2D are refused,
# with the field that disagrees:

# %%
flat = Ray(rays.origin[:, :2], rays.direction[:, :2], rays.viewdir[:, :2])
try:
    model(flat, jr.key(1), jr.key(2), True)
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Activations and gradients, call by call
#
# `expex.debug.watch` of the training loss on 16 of the rays, with the untrained model's colours as
# targets plus noise:

# %%
pixels = jnp.clip(fine.rgb + 0.1 * jr.normal(jr.key(3), fine.rgb.shape), 0, 1)
few = jax.tree.map(lambda x: x[:16], rays)  # 16 of the rays
watch = expex.debug.watch(lambda m: m.loss(few, pixels[:16], jr.key(1), jr.key(2)), model)
print(f"{len(watch.rows)} calls, {len(watch.flagged)} flagged")
watch

# %% [markdown]
# ## Against the original
#
# `tests/data/nerf_reference.npz` holds the parameters of a small JaxNeRF model built by the
# unmodified original, the coarse and fine pixels it renders for 64 rays (random and fixed
# sampling), and its training loss with gradients. `compare.py` loads those parameters into the
# port and measures the largest differences:

# %%
import compare  # examples/nerf/compare.py

reference = dict(np.load(REPO / "tests" / "data" / "nerf_reference.npz"))
for name, diff in compare.compare(reference).items():
    print(f"{name:24} {diff:.3g}")

# %% [markdown]
# ## Speed
#
# Compiled rendering and gradients of the 64 rays. Timings depend on the machine, so compare
# them only with each other. The repository's [benchmarks/](../../benchmarks/) compare the port
# with the original: about the same on CPU.

# %% [bench]
def bench(f, *args, n=10):
    """Median seconds per call, after a first call that compiles."""
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(n):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


grads = jax.jit(jax.grad(NeRF.loss))
print(f"render 64 rays    {bench(render, model, rays, jr.key(1), jr.key(2), True) * 1e3:7.2f} ms")
print(f"gradients 64 rays {bench(grads, model, rays, pixels, jr.key(1), jr.key(2)) * 1e3:7.2f} ms")

"""Source of vit.ipynb (see build.py)."""

# %% [header] The Vision Transformer in expex

# %% [markdown]
# A port of `vit_jax/models_vit.py` from
# [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
# (Flax) to expex: [`examples/vit/vit.py`](../vit/vit.py), 85 lines of code against 183. An image
# is cut into patches by naming the axes it is split into, heads are contracted by axis name,
# and the image size is in the model's signature (`f32[GH * P, GW * Q, C]`).
#
# This notebook runs it on a small model, shows what expex reads from the annotations, checks it
# against the original's saved outputs, and times it.

# %% [setup] vit

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import vit  # examples/vit/vit.py
from vit import ViT

print(inspect.getsource(vit.PatchEmbed))
print(inspect.getsource(vit.ViT.__call__))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method in the module, as expex reads it:

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(vit):
    print(signature(report.fn))

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line of the attention, for all sizes at once:

# %%
print(expex.debug.shapes(vit.Attention.__call__))

# %% [markdown]
# ## Run it
#
# ViT-Ti's width (192, 3 heads of 64, MLP width 768) with 2 of its 12 layers, on 32 x 32 images
# in 4 x 4 patches: an 8 x 8 grid. The model is written for one image; a batch maps over it.

# %%
model = ViT(jr.key(0), GH=8, GW=8, P=4, Q=4, C=3, D=192, H=3, K=64, F=768, Classes=10, layers=2)
images = jr.normal(jr.key(1), (16, 32, 32, 3))
labels = jr.randint(jr.key(2), (16,), 0, 10)


@jax.jit  # the model's call checks the images against its sizes
def step(model: ViT, images: jax.Array, labels: jax.Array) -> tuple[ViT, jax.Array]:
    loss, grads = jax.value_and_grad(lambda m: m.loss(images, labels).mean())(model)
    return jax.tree.map(lambda p, g: p - 0.05 * g, model, grads), loss


print("logits", model(images[0]).shape, "for one image;", model(images).shape, "for the batch")
for i in range(5):
    model, loss = step(model, images, labels)
    print(f"step {i}: loss {float(loss):.4f}")

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(vit)

# %% [markdown]
# **A summary** of the model, and its **cost** as formulas in the sizes, then for ViT-B/16 on
# 224 x 224 images:

# %%
expex.summary(model)

# %%
print(expex.cost(ViT.__call__))
print(expex.cost(ViT.__call__, GH=14, GW=14, P=16, Q=16, C=3, D=768, H=12, K=64, F=3072, Classes=1000, layers=12))

# %% [markdown]
# **A diagram** drawn from the code. When you run the notebook it is an interactive page: click a
# call to open it.

# %%
expex.diagram(ViT.__call__, depth=3)

# %% [markdown]
# **`expex.inspect`**: every tool in one page, with activations and gradients of the loss.

# %%
expex.inspect(model, images[0], loss=lambda m, x: m.loss(x, labels[0]))

# %% [markdown]
# ## A shape error
#
# The model's signature fixes the image size. A 32 x 36 image says which size it disagrees with,
# and the patch embedding alone, which takes any grid of whole patches, refuses a width that is
# not a multiple of the patch:

# %%
try:
    model(jnp.ones((32, 36, 3)))
except expex.ShapeError as e:
    error = e
error

# %%
try:
    model.embed(jnp.ones((32, 30, 3)))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Activations and gradients, layer by layer

# %%
watch = expex.debug.watch(lambda m: m.loss(images[0], labels[0]), model)
print(f"{len(watch.rows)} calls, {len(watch.flagged)} flagged")
watch

# %% [markdown]
# ## Against the original
#
# `tests/data/vit_reference.npz` holds the parameters of a small `VisionTransformer` built by the
# unmodified original, its logits on two 32 x 24 images in 4 x 2 patches, and a loss with its
# gradients. `compare.py` loads those parameters into the port and measures the largest
# differences:

# %%
import compare  # examples/vit/compare.py

reference = dict(np.load(REPO / "tests" / "data" / "vit_reference.npz"))
for name, diff in compare.compare(reference).items():
    print(f"{name:28} {diff:.3g}")

# %% [markdown]
# ## Speed
#
# Compiled forward pass and gradients at the size above. Timings depend on the machine, so
# compare them only with each other. The repository's [benchmarks/](../../benchmarks/) compare
# the port with the original: about the same on CPU.

# %%
def bench(f, *args, n=20):
    """Median seconds per call, after a first call that compiles."""
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(n):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


forward = jax.jit(lambda m, x: m(x))
grads = jax.jit(jax.grad(lambda m, x, y: m.loss(x, y).mean()))
print(f"forward, 16 images   {bench(forward, model, images) * 1e3:7.2f} ms")
print(f"gradients, 16 images {bench(grads, model, images, labels) * 1e3:7.2f} ms")

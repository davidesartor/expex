"""Source of mamba.ipynb (see build.py)."""

# %% [header] Mamba in expex

# %% [markdown]
# A port of [mamba-jax](https://github.com/vvvm23/mamba-jax) (Equinox) to expex:
# [`examples/mamba/mamba.py`](../mamba/mamba.py). Every module states its sizes
# (`Mixer[D, E, N, R, K]`: width `D`, inner width `E`, state `N`, dt rank `R`, conv width `K`),
# the selective scan is a plain `for` loop over time (staged as a `lax.scan`), and per-token
# layers are written for one token.
#
# This notebook runs it on a small model, shows what expex reads from the annotations, checks it
# against the original's saved outputs, and times it.

# %% [setup] mamba

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import mamba  # examples/mamba/mamba.py
from mamba import Mamba

print(inspect.getsource(mamba.selective_scan))
print(inspect.getsource(mamba.Mixer.__call__))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method in the module, as expex reads it:

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(mamba):
    print(signature(report.fn))

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line of the mixer, for all sizes at once:

# %%
print(expex.debug.shapes(mamba.Mixer.__call__))

# %% [markdown]
# ## Run it
#
# A small model: vocabulary 100, width 64, 3 layers. `Mamba.create` derives the inner width, the
# dt rank and the padded vocabulary from the original's settings. The model is written for one
# sequence; a batch maps over its rows.

# %%
model = Mamba.create(jr.key(0), vocab=100, d=64, layers=3)
tokens = jr.randint(jr.key(1), (8, 32), 0, 100)  # 8 sequences of 32 tokens

print("logits", model(tokens[0]).shape, "for one sequence;", model(tokens).shape, "for the batch")
print("loss", float(model.loss(tokens).mean()))
print("sampled", model.generate(tokens[0, :8], 8, jr.key(2)))
print("how the loop in selective_scan ran:", mamba.selective_scan.flow)

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(mamba)

# %% [markdown]
# **A summary** of the model, and its **cost** as formulas in the sizes:

# %%
expex.summary(model, T=32)

# %%
print(expex.cost(Mamba.__call__))

# %% [markdown]
# **A diagram** drawn from the code: the selective scan is a container `loop × T`. When you run
# the notebook it is an interactive page: click a call to open it.

# %%
expex.diagram(mamba.Mixer.__call__, depth=2)

# %% [markdown]
# **`expex.inspect`**: every tool in one page, with activations and gradients of the loss.

# %%
expex.inspect(model, tokens[0], loss=lambda m, t: m.loss(t))

# %% [markdown]
# ## A shape error
#
# Changing part of a model with `.at` is checked against its annotation. `x_proj` is
# `f32[R + 2 * N, E]`; a replacement with one row too few says which size disagrees:

# %%
mixer = model.layers[0].mixer
try:
    model.at.layers[0].mixer.x_proj.set(jnp.zeros((mixer.x_proj.shape[0] - 1, mixer.x_proj.shape[1])))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Activations and gradients, layer by layer
#
# `expex.debug.watch` runs one forward and backward pass and reports every typed call:

# %%
watch = expex.debug.watch(lambda m: m.loss(tokens[0]), model)
print(f"{len(watch.rows)} calls, {len(watch.flagged)} flagged")
watch

# %% [markdown]
# ## Against the original
#
# `tests/data/mamba_reference.npz` holds the parameters of a small model built by the unmodified
# mamba-jax, its logits (with its sequential and its associative scan), the gradients of a loss
# and the tokens it samples. `compare.py` loads those parameters into the port and measures the
# largest differences:

# %%
import compare  # examples/mamba/compare.py

reference = dict(np.load(REPO / "tests" / "data" / "mamba_reference.npz"))
for name, diff in compare.compare(reference).items():
    print(f"{name:28} {diff:.3g}")

# %% [markdown]
# ## Speed
#
# Compiled forward pass, gradients and sampling at the small size above. Timings depend on the
# machine, so compare them only with each other. The repository's
# [benchmarks/](../../benchmarks/) compare the port with the original: 10-14x faster forward and
# 5-6x faster gradients on CPU, generation the same.

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


forward = jax.jit(lambda m, t: m(t))
grads = jax.jit(jax.grad(lambda m, t: m.loss(t).mean()))
sample = jax.jit(lambda m, p, k: m.generate(p, 16, k))
print(f"forward, 8 x 32 tokens   {bench(forward, model, tokens) * 1e3:7.2f} ms")
print(f"gradients, 8 x 32 tokens {bench(grads, model, tokens) * 1e3:7.2f} ms")
print(f"sample 16 tokens         {bench(sample, model, tokens[0, :8], jr.key(3)) * 1e3:7.2f} ms")

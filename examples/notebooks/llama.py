"""Source of llama.ipynb (see build.py)."""

# %% [header] Llama 2 in expex

# %% [markdown]
# A port of [llama-2-jax](https://github.com/ayaka14732/llama-2-jax) (pure JAX) to expex:
# [`examples/llama/llama.py`](../llama/llama.py), 101 lines of code in one file against the
# original's 478 across 15. Every module states its sizes (`Attention[D, G, R, K]`: width `D`,
# `G` groups of `R` query heads sharing a key/value head of size `K`), heads are contracted by
# axis name, and one code path serves training and generation: a forward pass is a pass over an
# empty KV cache.
#
# This notebook runs it on a small model, shows what expex reads from the annotations, checks it
# against the original's saved outputs, and times it.

# %% [setup] llama

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import llama  # examples/llama/llama.py
from llama import Llama

print(inspect.getsource(llama.Attention))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method in the module, as expex reads it. `expex.prove` lists them,
# and proves each one for all sizes (below).

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(llama):
    print(signature(report.fn))

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line, for all sizes at once:

# %%
print(expex.debug.shapes(llama.Attention.__call__))

# %% [markdown]
# ## Run it
#
# A small model: vocabulary 100, width 64, 8 query heads in 2 groups sharing 2 key/value heads,
# 3 layers. The model is built from its annotations: a key, then the sizes. It is written for one
# sequence, so a batch of sequences maps over the rows.

# %%
model = Llama(jr.key(0), V=100, D=64, F=172, G=2, R=4, K=8, layers=3)
tokens = jr.randint(jr.key(1), (16, 24), 0, 100)  # 16 sequences of 24 tokens

forward = jax.jit(lambda m, t: m(t))  # checked while tracing, then plain XLA
print("logits", forward(model, tokens[0]).shape, "for one sequence;", forward(model, tokens).shape, "for the batch")


@expex.fn(jit=True, donate="auto")
def step[B, T](model: Llama, batch: expex.Int[B, T]) -> tuple[Llama, expex.f32]:
    loss, grads = jax.value_and_grad(lambda m: m.loss(batch).mean())(model)
    return jax.tree.map(lambda p, g: p - 0.5 * g, model, grads), loss


for i in range(5):
    model, loss = step(model, tokens)
    print(f"step {i}: loss {float(loss):.4f}")
print("generated", model.generate(tokens[0, :8], 8))

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(llama)

# %% [markdown]
# **A summary** of the model: parameters, bytes and FLOPs per module, with their formulas.

# %%
expex.summary(model, T=24)

# %% [markdown]
# **Cost** as formulas in the sizes, or as numbers for given sizes (here a 7B-like width):

# %%
print(expex.cost(Llama.__call__))
print(expex.cost(Llama.__call__, T=4096, V=32000, D=4096, F=11008, G=32, R=1, K=128, layers=32))

# %% [markdown]
# **A diagram** drawn from the code. When you run the notebook it is an interactive page: click a
# call to open it, hover for its signature.

# %%
expex.diagram(Llama.__call__, depth=3)

# %% [markdown]
# **`expex.inspect`** puts every tool in one page, over the module tree: summary, diagram, cost,
# activations and gradients of a loss, and the FLOPs per source line.

# %%
expex.inspect(model, tokens[0, :12], loss=lambda m, t: m.loss(t))

# %% [markdown]
# ## A shape error
#
# A cache shorter than the tokens written into it breaks `T <= S`, stated in `extend`'s
# signature. The error says which constraint, with the sizes it saw:

# %%
try:
    model.extend(tokens[0, :12], 0, model.empty(8))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# And a layer given activations one feature short says where its width came from:

# %%
try:
    model.layers[0](jnp.ones((4, 63)), jnp.arange(4), model.empty(8)[0])
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Activations and gradients, layer by layer
#
# `expex.debug.watch` runs one forward and backward pass and reports every typed call. Here one
# MLP's weights are scaled by 1000 to plant a problem, and the rows it affects are flagged.

# %%
broken = model.at.layers[1].mlp.down.multiply(1e3)
watch = expex.debug.watch(lambda m: m.loss(tokens[0, :12]), broken)
for row in watch.flagged:
    print(f"{row.name:32} {'; '.join(row.flags)}")

# %% [markdown]
# ## Against the original
#
# `tests/data/llama_reference.npz` holds the parameters of a small model built by the unmodified
# llama-2-jax, its logits, the gradients of a loss and the tokens its own generation loop picks
# greedily with the KV cache (written by [`run_original.py`](../llama/run_original.py)).
# `compare.py` loads those parameters into the port and measures the largest differences:

# %%
import compare  # examples/llama/compare.py

reference = dict(np.load(REPO / "tests" / "data" / "llama_reference.npz"))
for name, diff in compare.compare(reference).items():
    print(f"{name:28} {diff:.3g}")

# %% [markdown]
# ## Speed
#
# Compiled forward pass, gradients and generation at the small size above. Timings depend on the
# machine (a Colab CPU is slower than a laptop), so compare them only with each other. The
# repository's [benchmarks/](../../benchmarks/) compare the port with the original: 1.3x forward
# and gradients on CPU.

# %% [bench]
def bench(f, *args, n=20):
    """Median seconds per call, after a first call that compiles."""
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(n):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


grads = jax.jit(jax.grad(lambda m, t: m.loss(t).mean()))
print(f"forward, 16 x 24 tokens   {bench(forward, model, tokens) * 1e3:7.2f} ms")
print(f"gradients, 16 x 24 tokens {bench(grads, model, tokens) * 1e3:7.2f} ms")
print(f"generate 8 tokens         {bench(model.generate, tokens[0, :8], 8) * 1e3:7.2f} ms")

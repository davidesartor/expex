"""Source of quickstart.ipynb (see build.py)."""

# %% [header] expex quickstart

# %% [markdown]
# **Write the shapes in the signature; get checks, proofs, loops and model tools for free.**
#
# expex is a thin layer over JAX. Annotate arrays with their sizes and the library reads those
# annotations: it checks every call, proves the shapes for every size, turns plain Python loops
# into `lax.scan`, and draws, costs and debugs your model. It is still JAX underneath: `jax.jit`
# and `jax.grad` work as usual, and a compiled call costs the same as plain `jax.jit`.
#
# This notebook runs the examples of the [README](../../README.md). The ported projects have a
# notebook each: see [the list](README.md).

# %% [setup] .

# %%
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
from expex import f32

key = jr.key(0)

# %% [markdown]
# ## Shapes are checked, not commented
#
# Sizes are type parameters (`[B, T, S, D]`), and arrays are annotated with them. A wrong call
# says which axis is wrong and where its size came from:

# %%
@expex.fn
def attend[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]:
    return expex.einsum(q, k, to=(B, T, S))


print(attend(jnp.ones((2, 3, 4)), jnp.ones((2, 5, 4))).shape)
try:
    attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Loops are loops, branches are `if` / `else`
#
# A `for` over a size becomes a `lax.scan` carrying what the body updates; an `if` on a traced
# value becomes a select (or a `lax.cond`). A function written for one number is called on arrays:

# %%
@expex.fn
def rnn[T, D](w: f32[D, D], xs: f32[T, D]) -> f32[T, D]:
    h, hs = jnp.zeros(D), []
    for t in range(T):  # a lax.scan carrying h
        h = jnp.tanh(w @ h + xs[t])
        hs.append(h)
    return jnp.asarray(hs)


@expex.fn(jit=True)
def safe_log(x: f32) -> f32:  # written for one number, called on arrays
    return jnp.log(x) if x > 0 else -jnp.inf  # a select per element


print(jax.jit(rnn)(0.1 * jr.normal(key, (8, 8)), jnp.ones((20, 8))).shape)
print("how the loop ran:", rnn.flow)
print(safe_log(jnp.array([2.0, 0.0, -1.0])))

# %% [markdown]
# ## Models are classes; identical layers scan themselves; batches map themselves
#
# A struct is built from its annotations (a key first, then the sizes). The loop over identical
# blocks is one `lax.scan`, and a model written for one example maps over a batch:

# %%
@expex.struct
class Block[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0

    def __call__(self, x: f32[D]) -> f32[D]:
        y = self.w @ x + self.b
        return x + jax.nn.gelu(y)


@expex.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for block in self.blocks:  # identical layers: one lax.scan
            x = block(x)
        return x


model = Stack(key, D=64, blocks=12)
ys = model(jnp.ones((8, 64)))  # extra leading axes are mapped over
print(ys.shape)
model

# %% [markdown]
# ## What else the annotations give you
#
# **Proofs for every size**, without data:

# %%
expex.prove(Stack.__call__)

# %% [markdown]
# **Cost as formulas in the sizes**, or as numbers for given sizes:

# %%
print(expex.cost(Stack.__call__))
print(expex.cost(Stack.__call__, D=1024, blocks=24))

# %% [markdown]
# **Changing part of a model**, checked against its annotation, and **model summaries**:

# %%
edited = model.at.blocks[5].w.multiply(0.5)  # a copy with one block's weights scaled
edited = edited.at.blocks[0].b.set(jnp.ones(64))  # also .add, .apply(f), .get()
try:
    model.at.blocks[0].b.set(jnp.ones(63))
except expex.ShapeError as e:
    print(e)
expex.summary(model)

# %% [markdown]
# **Architecture diagrams**, drawn from the code. Here the small transformer of
# [`examples/transformer.py`](../transformer.py). When you run the notebook it is an interactive
# page: click a call to open it, hover for its signature.

# %%
from transformer import Transformer  # examples/transformer.py

expex.diagram(Transformer.__call__, depth=2)

# %% [markdown]
# The same diagram timed on real inputs, each call coloured by its share of the measured time:

# %%
lm = Transformer(key, Vocab=50, D=32, F=64, blocks=4, heads=4)
tokens = jr.randint(jr.key(1), (24,), 0, 50)
timed = expex.diagram.timed(lm.__call__, tokens)
print(timed.text)
timed

# %% [markdown]
# A `ShapeError` raised inside a model carries a diagram of the trace up to the call that
# failed, with that call and the path into it in red:

# %%
@expex.struct
class Probe[D]:
    stack: Stack[D]
    w: f32[D] = 1.0

    def __call__[B](self, x: f32[B, D]) -> f32[B]:
        return self.stack(x[:, 1:]) @ self.w  # a bug: x loses a feature


probe = Probe(jr.key(1), D=16, blocks=2)
try:
    probe(jnp.ones((4, 16)))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# **`expex.inspect`** puts every tool in one page, over the module tree: summary, diagram, cost,
# activations and gradients of a loss, and the FLOPs per source line.

# %%
expex.inspect(lm, tokens, loss=lambda m, t: m.loss(t))

# %% [markdown]
# ## Debugging
#
# Shapes line by line, and where a NaN first appeared:

# %%
print(expex.debug.shapes(Block.__call__))


@expex.fn
def entropy[N](logits: f32[N]) -> f32:
    p = jax.nn.softmax(logits)
    return -(p * jnp.log(p)).sum()


print(expex.debug.find_nan(entropy, jnp.array([0.0, 200.0, -200.0])))

# %% [markdown]
# Activations and gradients of every layer, flagging exploding or dead ones. Here one block's
# weights are scaled by 1000:

# %%
x = jr.normal(jr.key(1), (8, 64))
broken = model.at.blocks[5].w.multiply(1e3)
watch = expex.debug.watch(lambda m: (m(x) ** 2).mean(), broken, growth=10.0)
for row in watch.flagged:
    print(f"{row.name:20} {'; '.join(row.flags)}")
watch

# %% [markdown]
# ## Porting existing JAX code
#
# `expex.annotate` runs a plain function on examples and proposes its annotated signature, with
# sizes grouped by the dataflow:

# %%
def attend_jax(q, k, v):
    scores = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(q.shape[-1])
    return jax.nn.softmax(scores, axis=-1) @ v


print(expex.annotate(attend_jax, jnp.ones((2, 3, 8)), jnp.ones((2, 5, 8)), jnp.ones((2, 5, 4))))

# %% [markdown]
# ## Speed
#
# A compiled call costs the same as plain `jax.jit`: the checks run once, while tracing. Here the
# 12-block stack on a batch of 256, compiled by expex (`@expex.fn(jit=True)`) and by `jax.jit` on
# the same function, and the plain JAX version of the README (stacked weights, `lax.scan`, `vmap`).
# Timings depend on the machine.

# %% [bench]
def bench(f, *args, n=50):
    """Median seconds per call, after a first call that compiles."""
    jax.block_until_ready(f(*args))
    times = []
    for _ in range(n):
        start = time.perf_counter()
        jax.block_until_ready(f(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


@expex.fn(jit=True)
def run_stack[B, D](model: Stack[D], x: f32[B, D]) -> f32[B, D]:
    return model(x)


def plain(params, x):  # the README's JAX version
    def layer(x, p):
        return x + jax.nn.gelu(p["w"] @ x + p["b"]), None

    return jax.lax.scan(layer, x, params)[0]


params = {"w": jnp.stack([b.w for b in model.blocks]), "b": jnp.stack([b.b for b in model.blocks])}
batch = jr.normal(jr.key(2), (256, 64))
print(f"@expex.fn(jit=True)   {bench(run_stack, model, batch) * 1e6:8.1f} µs")
print(f"jax.jit(model call)   {bench(jax.jit(lambda m, x: m(x)), model, batch) * 1e6:8.1f} µs")
print(f"plain JAX (scan+vmap) {bench(jax.jit(jax.vmap(plain, in_axes=(None, 0))), params, batch) * 1e6:8.1f} µs")

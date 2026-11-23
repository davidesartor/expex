"""Source of multidevice.ipynb (see build.py)."""

# %% [header] Training on a device mesh with expex

# %% [markdown]
# The transformer of [`examples/transformer.py`](../transformer.py), trained on a 4 x 2 mesh of
# devices (data x model), as in [`examples/multidevice/train.py`](../multidevice/train.py).
# Every sharding comes from two axis names, `B="data"` and `D="model"`: the batch is split four
# ways, the model width two ways, and the optimizer state (an optax AdamW state whose moments
# are models) follows the parameters. No `PartitionSpec` is written anywhere.
#
# On a CPU, JAX can simulate 8 devices. The device count is fixed when JAX starts, so the first
# cell sets it, before anything imports JAX. (On a fresh Colab runtime, run it first.)

# %%
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"  # 8 simulated CPU devices

# %% [setup] multidevice optax

# %%
import inspect
import json
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import expex

sys.argv = ["train.py", "--devices", "8"]  # train.py reads its options when it is imported
import train  # examples/multidevice/train.py
from transformer import Transformer  # examples/transformer.py

print(jax.device_count(), "devices:", jax.devices()[0].platform)
print(inspect.getsource(train.train_step))
print(inspect.getsource(Transformer))

# %% [markdown]
# ## The annotated signatures
#
# The model's methods and the training step, as expex reads them. The names `B` and `D` in
# these annotations are what `expex.shard` maps to mesh axes.

# %%
import transformer


def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(transformer):
    print(signature(report.fn))
for f in (train.train_step, train.hidden, train.logits):
    print(signature(f))

# %% [markdown]
# ## Shardings from the annotations
#
# `expex.device_put(state, mesh, B="data", D="model")` places every array by the names in its
# annotation: a `f32[Vocab, D]` embedding is split by columns over `"model"`, a batch `Int[B, T]`
# by rows over `"data"`, and everything else is replicated.

# %%
mesh = train.make_mesh(8)
print(f"mesh: {dict(mesh.shape)}, names: {train.NAMES}")
state = expex.device_put(train.init(), mesh, **train.NAMES)
model = state["model"]
tokens = expex.device_put(train.batch(0), mesh, annotation=train.TOKENS, **train.NAMES)
hidden = expex.shard(train.hidden, mesh, **train.NAMES)(model, tokens)
adam = state["opt"][1][0]  # chain(clip, adamw): adamw's state

print(f"\n{'array':<28} {'shape':<14} {'per device':<14} spec")
for name, x in [("embed  f32[Vocab, D]", model.embed), ("blocks[0].mlp.w1  f32[D, F]", model.blocks[0].mlp.w1),
                ("norm.scale  f32[D]", model.norm.scale), ("adam mu.blocks[0].attn.wq", adam.mu.blocks[0].attn.wq),
                ("tokens  Int[B, T]", tokens), ("hidden  f32[B, T, D]", hidden)]:  # fmt: skip
    print(f"{name:<28} {str(x.shape):<14} {str(train.per_device(x)):<14} {x.sharding.spec}")

# %% [markdown]
# ## Run it
#
# 10 steps of character-level training on the repository's LICENSE, with a jitted, sharded step
# (`expex.shard(train_step, mesh, donate="auto", B="data", D="model")`) that reuses the model's
# and optimizer's buffers:

# %%
losses, times, donated, state = train.train(state, mesh, 0, 10)
for i, loss in enumerate(losses):
    print(f"step {i}: loss {loss:.4f}")
print(f"{donated}/10 steps reused their inputs' buffers; {train.check_shardings(state, mesh)} arrays are where expex.shardings says")
model = state["model"]

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, a **summary**, and the **cost** of the loss: they read the
# annotations, not the shardings, so they count the whole model.

# %%
expex.prove(transformer)

# %%
expex.summary(model, T=train.SEQ)

# %%
print(expex.cost(Transformer.loss, train.CONFIG, T=train.SEQ + 1))

# %% [markdown]
# **A diagram** of the model, drawn from the code (interactive when you run the notebook):

# %%
expex.diagram(Transformer.__call__, depth=2)

# %% [markdown]
# **`expex.inspect`**: every tool in one page, for the trained (sharded) model on one sequence:

# %%
sequence = jnp.asarray(train.batch(1)[0])
expex.inspect(model, sequence[:-1], loss=lambda m, t: m.loss(sequence))

# %% [markdown]
# ## A shape error
#
# The sharded functions check their arguments like any typed call. Tokens as floats are refused:

# %%
try:
    expex.shard(train.logits, mesh, **train.NAMES)(model, tokens.astype(jnp.float32))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# And placing a batch says when the mesh cannot split it: 30 sequences over the 4 devices of
# `"data"`:

# %%
try:
    expex.device_put(train.batch(0)[:30], mesh, annotation=train.TOKENS, **train.NAMES)
except ValueError as e:
    print(e)

# %% [markdown]
# ## Activations and gradients, block by block
#
# `expex.debug.watch` of the loss on one sequence, with the sharded model:

# %%
watch = expex.debug.watch(lambda m: m.loss(sequence), model)
print(f"{len(watch.rows)} calls, {len(watch.flagged)} flagged")
watch

# %% [markdown]
# ## Against one device
#
# There is no original to compare with here; the check is that sharding changes nothing. The same
# 10 steps on 1 device, in a new process (the device count is fixed per process):

# %%
script = REPO / "examples" / "multidevice" / "train.py"
env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
out = subprocess.run([sys.executable, str(script), "--devices", "1", "--steps", "10", "--json"],
                     capture_output=True, text=True, check=True, env=env)  # fmt: skip
single = json.loads(out.stdout.strip().splitlines()[-1])
gap = max(abs(a - b) / abs(b) for a, b in zip(losses, single["losses"]))
print(f"{'step':>4}  {'8 devices':>10}  {'1 device':>10}")
for i, (a, b) in enumerate(zip(losses, single["losses"])):
    print(f"{i:>4}  {a:>10.6f}  {b:>10.6f}")
print(f"largest relative difference: {gap:.1e}")

# %% [markdown]
# ## Speed
#
# Median step time, compile excluded. The 8 simulated devices share this machine's cores, so they
# are no speedup: the point is that it runs. Timings depend on the machine.

# %%
print(f"1 device:            {np.median(single['times'][1:]) * 1e3:6.1f} ms per step")
print(f"8 simulated devices: {np.median(times[1:]) * 1e3:6.1f} ms per step  ({os.cpu_count()} CPU cores)")

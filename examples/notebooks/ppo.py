"""Source of ppo.ipynb (see build.py)."""

# %% [header] PPO (PureJaxRL) in expex

# %% [markdown]
# A port of [PureJaxRL](https://github.com/luchris429/purejaxrl)'s `ppo.py` (PPO on CartPole) to
# expex: [`examples/ppo/ppo.py`](../ppo/ppo.py). The original is nested `jax.lax.scan`s over
# closures and `jax.vmap` with hand-written `in_axes`. The port keeps the algorithm, the
# random-number stream and the arithmetic identical, but writes it as modules written for one
# environment or one example (called on batches, which they map over) and plain Python `for`
# loops and `if` statements, staged by `@expex.fn`.
#
# This notebook trains it briefly, shows what expex reads from the annotations, checks it against
# the original's saved run, and times it.

# %% [setup] ppo optax

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import ppo  # examples/ppo/ppo.py
from ppo import Config, Net

print(inspect.getsource(ppo.gae))
print(inspect.getsource(ppo.Env.step))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method in the module, as expex reads it:

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(ppo):
    print(signature(report.fn))

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line of the loss, for all sizes at once:

# %%
print(expex.debug.shapes(Net.loss))

# %% [markdown]
# ## Run it
#
# 100 updates of 4 environments x 128 steps (the original runs 976): enough for the returns to
# climb. The whole training run is one jitted call.

# %%
cfg = Config(num_updates=100)
start = time.perf_counter()
net, info = jax.jit(lambda k: ppo.train(k, cfg))(jr.key(30))
jax.block_until_ready(net)
print(f"trained in {time.perf_counter() - start:.1f} s (trace, compile and run)")

returns = jnp.where(info.returned_episode, info.returned_return, jnp.nan)
for lo in range(0, cfg.num_updates, 20):
    print(f"updates {lo:3d}-{lo + 19:3d}: mean episode return {jnp.nanmean(returns[lo : lo + 20]):6.1f}")

# %% [markdown]
# Each Python loop in `train` was staged as a `lax.scan`, and the `if` in `Env.step` as a select:

# %%
for line, how in ppo.train.flow.items():
    print(f"line {line}: {how}")
print(ppo.Env.step.flow)

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(ppo)

# %% [markdown]
# **A summary** of the network, and its **cost** per observation:

# %%
expex.summary(net)

# %%
print(expex.cost(Net.__call__))

# %% [markdown]
# **A diagram** of the network, drawn from the code (interactive when you run the notebook):

# %%
expex.diagram(Net.__call__, depth=2)

# %% [markdown]
# **`expex.inspect`**: every tool in one page, for the network on a batch of observations.

# %%
obs = jr.normal(jr.key(1), (16, 4))
expex.inspect(net, obs)

# %% [markdown]
# ## A shape error
#
# The network takes CartPole's 4 observations (`Obs = Literal[4]`); 5 are refused:

# %%
try:
    net(jnp.ones(5))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Activations and gradients
#
# `expex.debug.watch` of a stand-in loss on the observations (the PPO loss needs a batch of
# transitions): each layer's outputs and the gradients reaching them.

# %%
def surrogate(net):
    logp, value = net(obs)
    return (value**2).mean() - logp.mean()


watch = expex.debug.watch(surrogate, net)
print(f"{len(watch.rows)} calls, {len(watch.flagged)} flagged")
watch

# %% [markdown]
# ## Against the original
#
# `tests/data/purejaxrl_ppo_3_updates.npz` holds the initial and final parameters and the episode
# statistics of the unmodified PureJaxRL for 3 updates. From the same initial network and key,
# the port gives the same numbers:

# %%
reference = np.load(REPO / "tests" / "data" / "purejaxrl_ppo_3_updates.npz")


def net_from(prefix):
    """A network holding the reference parameters (actor layers, then critic layers)."""
    leaves = []
    for i in range(6):
        leaves += [reference[f"{prefix}:params/Dense_{i}/kernel"], reference[f"{prefix}:params/Dense_{i}/bias"]]
    return jax.tree.unflatten(jax.tree.structure(Net(jr.key(0))), [jnp.asarray(x) for x in leaves])


cfg3 = Config(num_updates=3)
net3, info3 = jax.jit(lambda k, n: ppo.train(k, cfg3, n))(jr.PRNGKey(30), net_from("init"))
worst = max(float(jnp.abs(a - b).max()) for a, b in zip(jax.tree.leaves(net3), jax.tree.leaves(net_from("param"))))
print(f"final parameters: largest difference {worst:.2g}")
for field, key in [("returned_episode", "episode"), ("returned_length", "lengths"),
                   ("returned_return", "returns"), ("timestep", "timestep")]:  # fmt: skip
    print(f"{field:17} identical: {np.array_equal(np.asarray(getattr(info3, field)), reference[key])}")

# %% [markdown]
# ## Speed
#
# Environment steps per second of the jitted training run (after compiling), at 4 environments.
# Timings depend on the machine. The repository's [benchmarks/](../../benchmarks/) compare the
# port with the original: about the same speed on CPU.

# %% [bench]
cfg = Config(num_updates=20)
run = jax.jit(lambda k: ppo.train(k, cfg))
jax.block_until_ready(run(jr.key(0)))  # compile
times = []
for i in range(3):
    start = time.perf_counter()
    jax.block_until_ready(run(jr.key(i)))
    times.append(time.perf_counter() - start)
steps = cfg.num_updates * cfg.num_steps * cfg.num_envs
print(f"{steps} environment steps (with the updates) in {np.median(times) * 1e3:.0f} ms: "
      f"{steps / np.median(times):,.0f} steps/s")  # fmt: skip

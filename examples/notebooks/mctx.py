"""Source of mctx.ipynb (see build.py)."""

# %% [header] mctx's MuZero search in expex

# %% [markdown]
# A port of [mctx](https://github.com/google-deepmind/mctx)'s MuZero and Gumbel MuZero tree
# search to expex: [`examples/mctx/search.py`](../mctx/search.py), 340 lines of code against 691
# across 7 files. mctx searches a batch of trees at once (tree arrays with a batch axis, a
# `lax.fori_loop` around `vmap`ped `lax.while_loop`s). The port writes the search for **one**
# tree, with plain Python (`while node != ROOT:`, `visits[parent] += 1`), and a call on a batch
# of roots maps over them.
#
# This notebook searches a small random MDP, shows what expex reads from the annotations, checks
# every tree against mctx's saved searches, and times it.

# %% [setup] mctx

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import search  # examples/mctx/search.py
from search import ROOT, Tree, gumbel_muzero_policy, muzero_policy, random_mdp, recurrent_fn

print(inspect.getsource(search.search))
print(inspect.getsource(Tree.backward))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method in the module, as expex reads it:

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for report in expex.prove(search):
    print(signature(report.fn))

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line, for all sizes at once:

# %%
print(expex.debug.shapes(search.muzero_action_selection))

# %% [markdown]
# ## Run it
#
# A random deterministic MDP with 12 states and 4 actions (its tables stand in for the policy and
# value networks), and a batch of 8 roots, searched with 32 simulations each:

# %%
mdp = random_mdp(jr.key(0), 12, 4)
roots = mdp.root(jnp.arange(8))  # 8 roots: mapped over the states
print("roots:", expex.show(roots))

for policy in (muzero_policy, gumbel_muzero_policy):
    out = jax.jit(policy, static_argnums=(3, 4))(mdp, jr.key(1), roots, recurrent_fn, 32)
    print(f"{policy.__name__}: actions {out.action}, root visits of tree 0 {out.search_tree.children_visits[0, ROOT]}")
print("trees:", expex.show(out.search_tree))

# %% [markdown]
# The loops and branches were staged as written: `while` as `lax.while_loop`, `for` as
# `lax.scan`, `if` as `lax.cond`:

# %%
print("Tree.simulate:", Tree.simulate.flow)
print("Tree.backward:", Tree.backward.flow)
print("search:       ", search.search.flow)

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(search)

# %% [markdown]
# **Cost** of a selection step and of one search, as formulas in the sizes (a `while` loop counts
# one iteration, and says so):

# %%
print(expex.cost(search.muzero_action_selection))
print(expex.cost(Tree.backward))

# %% [markdown]
# **A summary** of one search tree's arrays (tree 0 of the batch):

# %%
expex.summary(jax.tree.map(lambda x: x[0], out.search_tree))

# %% [markdown]
# **A diagram** of the search for one tree, drawn from the code (interactive when you run the
# notebook: click a call to open it). The loop over simulations is a dashed container `loop × S`.

# %%
expex.diagram(search.search)

# %% [markdown]
# `expex.inspect` and `expex.debug.watch` are for models with a `__call__` and a loss; a search
# has neither, so they are not used here.
#
# ## A shape error
#
# The invalid-action mask must have one entry per action of the roots. A mask for 5 actions,
# where the roots have 4, is refused with the size it disagrees with:

# %%
try:
    muzero_policy(mdp, jr.key(1), roots, recurrent_fn, 8, jnp.zeros((8, 5), bool))
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Against the original
#
# `tests/data/mctx_reference.npz` holds searches by the unmodified mctx on a fixed random MDP: a
# batch of 8 trees, with MuZero and Gumbel MuZero, with and without an invalid-action mask and a
# `max_depth`. For each case, the chosen actions, the action weights and every array of the final
# trees. `compare.py` runs the port on the same problem and keys; the largest difference per case
# (0 is bit-identical):

# %%
import compare  # examples/mctx/compare.py

reference = dict(np.load(REPO / "tests" / "data" / "mctx_reference.npz"))
for case, diffs in compare.compare(reference).items():
    print(f"{case:13} largest difference {max(diffs.values()):.3g} over {len(diffs)} outputs")

# %% [markdown]
# ## Speed
#
# Searches per second, a batch of 64 roots with 32 simulations, jitted. Timings depend on the
# machine. With mctx checked out, `python examples/mctx/compare.py` times both; the repository's
# [benchmarks/](../../benchmarks/) report the same speed on CPU (best of 25 rounds, median in
# brackets): MuZero 1.02x (0.98x), Gumbel MuZero 1.08x (1.11x).

# %% [bench]
roots64 = mdp.root(jnp.arange(64) % 12)
for policy in (muzero_policy, gumbel_muzero_policy):
    run = jax.jit(policy, static_argnums=(3, 4))
    jax.block_until_ready(run(mdp, jr.key(0), roots64, recurrent_fn, 32))  # compile
    times = []
    for i in range(10):
        start = time.perf_counter()
        jax.block_until_ready(run(mdp, jr.key(i), roots64, recurrent_fn, 32))
        times.append(time.perf_counter() - start)
    print(f"{policy.__name__:20} {np.median(times) * 1e3:6.2f} ms per batch of 64 searches")

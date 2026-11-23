"""Source of pgx.ipynb (see build.py)."""

# %% [header] Pgx's Connect Four and Othello in expex

# %% [markdown]
# Ports of two games of [Pgx](https://github.com/sotetsuk/pgx) to expex:
# [`examples/pgx/connect_four.py`](../pgx/connect_four.py) and
# [`examples/pgx/othello.py`](../pgx/othello.py). Pgx writes the rules as array programs (a flat
# board, a table of all 69 four-in-a-row index tuples, `lax.cond`s over lambdas). The ports write
# one game as one `@expex.struct` with plain Python: `board[row, col] = self.color`, a `while`
# loop that walks from the new stone, `if self.terminated:`. A batch of games maps over the keys.
#
# This notebook plays random games, shows what expex reads from the annotations, checks every
# state against Pgx's saved games, and times it.

# %% [setup] pgx

# %%
import inspect
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
import connect_four  # examples/pgx/connect_four.py
import othello  # examples/pgx/othello.py
from connect_four import ConnectFour
from othello import Othello

print(inspect.getsource(ConnectFour.drop))
print(inspect.getsource(connect_four.line_length))

# %% [markdown]
# ## The annotated signatures
#
# Every typed function and method of both games, as expex reads it:

# %%
def signature(f):
    owner = f.name.rpartition(".")[0]
    return f"{owner}.{f.signature}" if owner else f.signature


for module in (connect_four, othello):
    for report in expex.prove(module):
        print(f"{module.__name__:13} {signature(report.fn)}")

# %% [markdown]
# `expex.debug.shapes` shows the shape of every line of a move, for all sizes at once:

# %%
print(expex.debug.shapes(ConnectFour.drop))

# %% [markdown]
# ## Run it
#
# 1000 games of each, with uniformly random legal moves, in one jitted call per game:

# %%
for game, module, moves in ((ConnectFour, connect_four, 42), (Othello, othello, 64)):
    state, returns = module.random_games(jr.key(0), 1000, moves)
    first = returns[:, 0]
    print(f"{game.__name__}: player 0 won {(first > 0).sum()}, lost {(first < 0).sum()}, drew {(first == 0).sum()}")

state, _ = connect_four.random_games(jr.key(0), 1000, 42)
print("\nConnect Four, game 0 at the end:")
for row in state.board[0]:
    print("  " + " ".join(".xo"[int(v) + 1] for v in row))

# %% [markdown]
# The `if`s became `lax.cond`s, the `while` a `lax.while_loop`, and Othello's loop over the
# eight directions one `lax.scan`:

# %%
print("ConnectFour.step: ", ConnectFour.step.flow)
print("line_length:      ", connect_four.line_length.flow)
print("Othello.play:     ", Othello.play.flow)

# %% [markdown]
# ## What the annotations give
#
# **Proofs** for every size, without data:

# %%
expex.prove(connect_four)

# %%
expex.prove(othello)

# %% [markdown]
# **Cost** of one move, and **a summary** of one game's state (game 0 of the batch):

# %%
print(expex.cost(ConnectFour.step))
print(expex.cost(Othello.step))

# %%
expex.summary(jax.tree.map(lambda x: x[0], state))

# %% [markdown]
# **A diagram** of an Othello move, drawn from the code (interactive when you run the notebook):
# the loop over the eight directions is a dashed container `loop × 8`, and the `if` another.

# %%
expex.diagram(Othello.play)

# %% [markdown]
# `expex.inspect` and `expex.debug.watch` are for models with a `__call__` and a loss; a game
# state has neither, so they are not used here.
#
# ## A shape error
#
# `Rows` is fixed at 6 (`class Rows(Dim, size=6)`); a board with 5 rows is refused:

# %%
try:
    connect_four.line_length(state.board[0, :5], 0, 0, 1, 0)
except expex.ShapeError as e:
    error = e
error

# %% [markdown]
# ## Against the original
#
# `tests/data/pgx_reference.npz` holds 64 random games of each, played by the unmodified Pgx from
# fixed keys: the actions, and every field of the state after each step. The ports replay the
# same actions from the same keys; `compare.py` counts the (step, game) pairs where a field
# differs:

# %%
import compare  # examples/pgx/compare.py

reference = dict(np.load(REPO / "tests" / "data" / "pgx_reference.npz"))
for game, mismatches in compare.compare(reference).items():
    print(f"{game:13} {sum(mismatches.values())} mismatches over {', '.join(mismatches)}")

# %% [markdown]
# ## Speed
#
# Random games stepped per second, 1024 games x 100 moves, jitted. Timings depend on the
# machine. With Pgx installed, `python examples/pgx/compare.py --pgx <pgx checkout>` times both;
# the repository's [benchmarks/](../../benchmarks/) report Connect Four 1.4x faster than Pgx on
# CPU, Othello about the same.

# %% [bench]
for game, module in ((ConnectFour, connect_four), (Othello, othello)):
    jax.block_until_ready(module.random_games(jr.key(0), 1024, 100))  # compile
    times = []
    for i in range(3):
        start = time.perf_counter()
        jax.block_until_ready(module.random_games(jr.key(i), 1024, 100))
        times.append(time.perf_counter() - start)
    print(f"{game.__name__:12} {1024 * 100 / np.median(times) / 1e6:5.2f} M steps/s")

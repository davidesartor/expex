"""Pgx vs examples/pgx: steps per second of a batch of random games, Connect Four and Othello.

The same rollout as examples/pgx/compare.py's `benchmark`: one jit-compiled
`fori_loop` of vmapped (Pgx) / `.vectorize`d (port) steps, each game playing a
random legal move drawn from its legal-action mask, from the same keys; both
must end on the same boards. Pgx's svgwrite stub comes from run_original.py.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_example, use_original

NAME = "pgx"
ORIGINALS = ["pgx"]
# README: 1,024 parallel games, CPU
CLAIM = {"connect_four": "1.50x (1.61x)", "othello": "1.07x (1.10x)"}  # README, from results/cpu-2026-11-25-v0.1.md: best (median)

CONFIGS = {
    "quick": dict(games=128, steps=20),
    "readme": dict(games=1024, steps=100),
    "realistic": dict(games=8192, steps=100),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["readme"], "gpu": ["readme", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    use_example("pgx")
    load_module("pgx_run_original", EXAMPLES / "pgx" / "run_original.py")  # stubs svgwrite, which Pgx imports
    compare = load_module("pgx_compare", EXAMPLES / "pgx" / "compare.py")
    use_original("pgx")
    import pgx

    batch, steps = cfg["games"], cfg["steps"]
    key = jr.PRNGKey(0)

    def pick(key, mask):
        return jr.categorical(key, jnp.log(mask), axis=-1).astype(jnp.int32)

    measures = []
    for name, game in compare.GAMES.items():
        env = pgx.make(name)

        def original(key, env=env):
            state = jax.vmap(env.init)(jr.split(key, batch))

            def body(t, state):
                return jax.vmap(env.step)(state, pick(jr.fold_in(key, t), state.legal_action_mask))

            return jax.lax.fori_loop(0, steps, body, state)

        def port(key, game=game):
            state = game.new.vectorize(jr.split(key, batch))

            def body(t, state):
                return state.step.vectorize(pick(jr.fold_in(key, t), state.legal_action_mask))

            return jax.lax.fori_loop(0, steps, body, state)

        def same_boards(a, b):
            same = np.asarray(a._x.board).reshape(b.board.shape) == np.asarray(b.board)
            return {"games with a different final board": int((~same.reshape(batch, -1).all(-1)).sum())}

        measures.append(
            Measure(
                name, Fn(jax.jit(original), (key,)), Fn(jax.jit(port), (key,)),
                items=batch * steps, unit="steps", check=same_boards,
                labels=("pgx", "port"), note=f"{batch} games x {steps} moves",
            )  # fmt: skip
        )
    return measures

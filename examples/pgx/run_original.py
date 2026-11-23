"""Play random games with the unmodified Pgx and save every state, for the port's tests.

    git clone https://github.com/sotetsuk/pgx /tmp/pgx
    python examples/pgx/run_original.py /tmp/pgx tests/data/pgx_reference.npz [games]

For Connect Four and Othello, `games` games (batched with `jax.vmap`, as Pgx is
meant to be used) start from fixed keys. Every step picks, per game, a random
legal action; in every fourth game, with probability 5% it picks a full
column (Connect Four, once there is one) or any empty square or pass
(Othello) instead, so that Pgx's illegal-action penalty is exercised too. Games that have ended keep
receiving actions, which Pgx must ignore. Saved: the keys, the actions and,
after `init` and every step, all of Pgx's public state plus the board.
"""

import sys
import types

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Pgx imports its SVG visualizer (and svgwrite) at import time; the games do not need it.
sys.modules["svgwrite"] = types.ModuleType("svgwrite")

STEPS = {"connect_four": 48, "othello": 80}  # long enough for the games to end, and steps after the end
FIELDS = ("current_player", "observation", "rewards", "terminated", "truncated", "legal_action_mask", "step_count")
EXPLORE = 0.05


def playable(name: str, state) -> jax.Array:
    """Actions to try now and then: a full column (or any, if none is), or an empty square or pass."""
    if name == "connect_four":
        full = ~state.legal_action_mask
        return full | ~full.any(-1, keepdims=True)
    return jnp.concatenate([state._x.board == 0, jnp.ones_like(state._x.board[..., :1], bool)], -1)


def pick(key: jax.Array, legal: jax.Array, allowed: jax.Array) -> jax.Array:
    """Per game: a random legal action or, in every fourth game with probability 5%, a random allowed one."""
    k1, k2, k3 = jr.split(key, 3)
    games = legal.shape[0]
    explore = jr.bernoulli(k1, EXPLORE, (games,)) & (jnp.arange(games) % 4 == 0)
    return jnp.where(
        explore, jr.categorical(k3, jnp.log(allowed), axis=-1), jr.categorical(k2, jnp.log(legal), axis=-1)
    ).astype(jnp.int32)


def board(name: str, state) -> jax.Array:
    """Pgx's flat board as a 2D array (Connect Four: -1/0/1 by colour; Othello: +-1 for the player to move)."""
    shape = (6, 7) if name == "connect_four" else (8, 8)
    return state._x.board.reshape(state._x.board.shape[:-1] + shape)


def record(name: str, state) -> dict[str, np.ndarray]:
    out = {f: np.asarray(getattr(state, f if f != "step_count" else "_step_count")) for f in FIELDS}
    out["board"] = np.asarray(board(name, state)).astype(np.int8)
    return out


def play(pgx, name: str, games: int, seed: int = 0, steps: int | None = None):
    """Random games with Pgx: (keys, actions[T, B], {field: [T + 1, B, ...]})."""
    env = pgx.make(name)
    init, step = jax.jit(jax.vmap(env.init)), jax.jit(jax.vmap(env.step))
    keys = jr.split(jr.PRNGKey(seed), games)
    state = init(keys)
    history = [record(name, state)]
    actions = []
    for t in range(steps or STEPS[name]):
        action = pick(jr.fold_in(jr.PRNGKey(seed + 1), t), state.legal_action_mask, playable(name, state))
        state = step(state, action)
        actions.append(np.asarray(action))
        history.append(record(name, state))
    stacked = {f: np.stack([h[f] for h in history]) for f in history[0]}
    return np.asarray(keys), np.stack(actions), stacked


def main(pgx_dir: str, out: str, games: int = 32) -> None:
    sys.path.insert(0, pgx_dir)
    import pgx

    arrays = {}
    for name in STEPS:
        keys, actions, states = play(pgx, name, games)
        arrays[f"{name}/keys"] = keys
        arrays[f"{name}/actions"] = actions.astype(np.int8)
        arrays.update({f"{name}/{f}": v for f, v in states.items()})
        print(f"{name}: {games} games x {len(actions)} steps, {states['terminated'][-1].mean():.0%} finished")
    np.savez_compressed(out, **arrays)
    print(f"saved {len(arrays)} arrays to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:]))

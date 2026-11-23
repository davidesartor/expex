"""Check the Connect Four and Othello ports against Pgx, and time both.

    # against the saved reference (no Pgx needed):
    python examples/pgx/compare.py tests/data/pgx_reference.npz
    # against Pgx itself, live, on many games, plus speed and compile time:
    python examples/pgx/compare.py --pgx /tmp/pgx [games]

The ports replay Pgx's action sequences from the same keys, and every field of
every state (current player, observation, rewards, termination, truncation,
legal-action mask, step count, board) must be identical after every step.
"""

import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from connect_four import ConnectFour  # noqa: E402
from othello import Othello  # noqa: E402

GAMES = {"connect_four": ConnectFour, "othello": Othello}
FIELDS = ("current_player", "observation", "rewards", "terminated", "truncated", "legal_action_mask", "step_count", "board")


@jax.jit(static_argnums=0)
def replay(game, keys, actions):
    """Every state of a batch of games playing `actions[t, b]`: fields of shape [T + 1, B, ...]."""

    def step(state, action):
        state = state.step.vectorize(action)
        return state, state

    first = game.new(keys)
    _, rest = jax.lax.scan(step, first, actions)
    return jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), first, rest)


def mismatches(name: str, keys, actions, want: dict) -> dict[str, int]:
    """For each field, how many (step, game) pairs differ from `want`."""
    got = replay(GAMES[name], jnp.asarray(keys), jnp.asarray(actions, jnp.int32))
    out = {}
    for f in FIELDS:
        g, w = np.asarray(getattr(got, f)), np.asarray(want[f])
        g = g.reshape(w.shape)  # Othello's board is flat in both; the reference stores it as 8 x 8
        out[f] = int((g != w).reshape(g.shape[0], g.shape[1], -1).any(-1).sum())
    return out


def compare(ref: dict) -> dict[str, dict[str, int]]:
    """Mismatch counts per game and field against a reference saved by run_original.py."""
    out = {}
    for name in GAMES:
        want = {f: ref[f"{name}/{f}"] for f in FIELDS}
        out[name] = mismatches(name, ref[f"{name}/keys"], ref[f"{name}/actions"], want)
    return out


def describe(name: str, actions, states) -> str:
    """How the games went: lengths, outcomes, illegal moves."""
    legal = states["legal_action_mask"][:-1]
    chosen = np.take_along_axis(legal, actions[..., None].astype(np.int64), -1)[..., 0]
    live = ~states["terminated"][:-1]
    illegal = (~chosen & live).sum()
    ended = states["terminated"][-1].mean()
    length = (~states["terminated"]).sum(0).mean()
    final = states["rewards"].sum(0)
    draws = (np.abs(final).sum(-1) == 0).sum()
    return (
        f"{name}: {actions.shape[1]} games, mean length {length:.1f} moves, {ended:.0%} finished, "
        f"{illegal} lost by an illegal move, {draws} draws"
    )


# ------------------------------------------------------------------- speed


def benchmark(pgx, name: str, batch: int = 1024, steps: int = 100, rounds: int = 7) -> dict[str, float]:
    """Steps per second for `batch` parallel games playing random legal moves, Pgx vs port.

    Both run the same jit-compiled rollout (`fori_loop` over `vmap`ped steps,
    actions drawn from the legal-action mask); the runs alternate and the best
    of `rounds` counts. Compile time is tracing plus XLA compilation.
    """
    env, game = pgx.make(name), GAMES[name]

    def pick(key, mask):
        return jr.categorical(key, jnp.log(mask), axis=-1).astype(jnp.int32)

    def original(key):
        state = jax.vmap(env.init)(jr.split(key, batch))

        def body(t, state):
            return jax.vmap(env.step)(state, pick(jr.fold_in(key, t), state.legal_action_mask))

        return jax.lax.fori_loop(0, steps, body, state)

    def port(key):
        state = game.new(jr.split(key, batch))

        def body(t, state):
            return state.step.vectorize(pick(jr.fold_in(key, t), state.legal_action_mask))

        return jax.lax.fori_loop(0, steps, body, state)

    key = jr.PRNGKey(0)
    out, compiled = {}, {}
    for label, f in (("pgx", original), ("port", port)):
        t = time.perf_counter()
        compiled[label] = jax.jit(f).lower(key).compile()
        out[f"{label} compile s"] = time.perf_counter() - t
        jax.block_until_ready(compiled[label](key))
    best = {label: float("inf") for label in compiled}
    for _ in range(rounds):
        for label, c in compiled.items():
            t = time.perf_counter()
            jax.block_until_ready(c(key))
            best[label] = min(best[label], time.perf_counter() - t)
    for label in compiled:
        out[f"{label} steps/s"] = batch * steps / best[label]
    a, b = compiled["pgx"](key), compiled["port"](key)  # the same games: the same final boards
    out["same final boards"] = bool((np.asarray(a._x.board).reshape(b.board.shape) == np.asarray(b.board)).all())
    return out


def main() -> None:
    if sys.argv[1] != "--pgx":
        ref = dict(np.load(sys.argv[1]))
        result = compare(ref)
        for name, diffs in result.items():
            print(f"{name}: mismatching (step, game) pairs per field: {diffs}")
        ok = all(v == 0 for d in result.values() for v in d.values())
        print("MATCH" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)

    sys.path.insert(0, str(HERE))
    from run_original import play  # also stubs svgwrite, which Pgx imports

    sys.path.insert(0, sys.argv[2])
    import pgx

    games = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    ok = True
    for name in GAMES:
        keys, actions, states = play(pgx, name, games, seed=7)
        print(describe(name, actions, states))
        diffs = mismatches(name, keys, actions, states)
        steps = actions.shape[0] * actions.shape[1]
        print(f"  {steps} (step, game) pairs compared; mismatches per field: {diffs}")
        ok &= all(v == 0 for v in diffs.values())
    print("MATCH" if ok else "MISMATCH")

    print(f"\nspeed on {jax.devices()[0].platform.upper()}, 1024 games x 100 random legal moves, jit-compiled:")
    for name in GAMES:
        r = benchmark(pgx, name)
        print(
            f"  {name:13s} pgx {r['pgx steps/s'] / 1e6:6.2f} M steps/s (compile {r['pgx compile s']:.2f} s)   "
            f"port {r['port steps/s'] / 1e6:6.2f} M steps/s (compile {r['port compile s']:.2f} s)   "
            f"same final boards: {r['same final boards']}"
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

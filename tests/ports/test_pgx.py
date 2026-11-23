"""The Connect Four and Othello ports in examples/pgx/ reproduce Pgx.

``tests/data/pgx_reference.npz`` holds 64 random games of each, played by the
unmodified Pgx from fixed keys: the actions, and every field of the state
after ``init`` and after each step (see examples/pgx/run_original.py).
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "pgx"))
import compare  # noqa: E402
import connect_four  # noqa: E402
import othello  # noqa: E402

REF = dict(np.load(ROOT / "tests" / "data" / "pgx_reference.npz"))


def test_every_state_matches_the_original():
    for game, diffs in compare.compare(REF).items():
        assert all(v == 0 for v in diffs.values()), (game, diffs)


def test_the_reference_covers_the_rules():
    for name in ("connect_four", "othello"):
        actions = REF[f"{name}/actions"]
        legal, live = REF[f"{name}/legal_action_mask"][:-1], ~REF[f"{name}/terminated"][:-1]
        chosen = np.take_along_axis(legal, actions[..., None].astype(np.int64), -1)[..., 0]
        assert (live & ~chosen).sum() > 0, f"{name}: no illegal move"
        assert REF[f"{name}/terminated"][-1].all(), f"{name}: a game did not finish"
        assert (REF[f"{name}/rewards"] == 1).any() and (REF[f"{name}/rewards"] == -1).any()
    passes = (REF["othello/actions"] == othello.PASS) & REF["othello/legal_action_mask"][:-1, :, -1]
    assert (passes & ~REF["othello/terminated"][:-1]).sum() > 0, "othello: no pass"


def test_the_rules_are_staged_as_written():
    # build the batched step functions once, then read how each statement ran
    for game in (connect_four.ConnectFour, othello.Othello):
        state = game.new.vectorize(jr.split(jr.PRNGKey(0), 4))
        jax.jit(lambda s, a: s.step.vectorize(a))(state, jnp.zeros(4, jnp.int32))
    assert set(connect_four.ConnectFour.step.flow.values()) == {"if -> lax.cond"}
    assert "while -> lax.while_loop" in connect_four.line_length.flow.values()
    assert "if -> lax.cond" in connect_four.ConnectFour.drop.flow.values()
    play = set(othello.Othello.play.flow.values())
    assert "for -> lax.scan (carries flips)" in play and "if -> lax.cond" in play
    assert set(othello.ray.flow.values()) == {"for -> lax.scan (carries line)"}


def test_eager_and_compiled_agree():
    # one game, not jitted: the loops and branches run as plain Python on concrete values
    for name in ("connect_four", "othello"):
        game, keys = compare.GAMES[name], REF[f"{name}/keys"][:1]
        actions = REF[f"{name}/actions"][:10, :1].astype(np.int32)
        state = game.new(jnp.asarray(keys[0]))
        for a in actions[:, 0]:
            state = state.step(jnp.int32(a))
        compiled = jax.tree.map(lambda x: x[-1, 0], compare.replay(game, jnp.asarray(keys), jnp.asarray(actions)))
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(compiled)):
            assert np.array_equal(a, b)


def test_random_games_play_to_the_end():
    _, returns = connect_four.random_games(jr.key(1), 16, 42)
    assert (jnp.abs(returns).sum(-1) <= 2).all() and (returns.sum(-1) == 0).all()
    state, returns = othello.random_games(jr.key(1), 16, 80)
    assert state.terminated.all() and (returns.sum(-1) == 0).all()


def test_all_shapes_are_proved():
    for module, n in ((connect_four, 6), (othello, 8)):
        reports = expex.prove(module)
        assert len(reports) == n and all(r.status == "proved" for r in reports), reports

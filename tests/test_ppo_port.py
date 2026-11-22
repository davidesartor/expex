"""The PPO port in examples/ppo.py reproduces PureJaxRL's ppo.py.

``tests/data/purejaxrl_ppo_3_updates.npz`` holds the initial parameters, final
parameters and episode statistics produced by the unmodified original for 3
updates (see examples/run_purejaxrl_original.py).
"""

import pathlib
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

import typax as tx

pytest.importorskip("optax")
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "examples"))
import ppo  # noqa: E402

REF = np.load(ROOT / "tests" / "data" / "purejaxrl_ppo_3_updates.npz")


def net_from(prefix):
    def layer(i):
        return ppo.Layer(
            jnp.asarray(REF[f"{prefix}:params/Dense_{i}/kernel"]),
            jnp.asarray(REF[f"{prefix}:params/Dense_{i}/bias"]),
        )

    return ppo.Net((layer(0), layer(1), layer(2)), (layer(3), layer(4), layer(5)))


def test_matches_the_original():
    cfg = ppo.Config(num_updates=3)
    net, info = jax.jit(lambda k, n: ppo.train(k, cfg, n))(jr.PRNGKey(30), net_from("init"))
    for got, want in zip(jax.tree.leaves(net), jax.tree.leaves(net_from("param"))):
        assert jnp.allclose(got, want, atol=1e-6)
    assert np.array_equal(np.asarray(info.returned_episode), REF["episode"])
    assert np.array_equal(np.asarray(info.returned_length), REF["lengths"])
    assert np.array_equal(np.asarray(info.returned_return), REF["returns"])
    assert np.array_equal(np.asarray(info.timestep), REF["timestep"])


def test_every_loop_was_staged_and_the_program_is_small():
    cfg = ppo.Config(num_updates=3)
    jax.jit(lambda k: ppo.train(k, cfg))(jr.key(0))
    flow = list(ppo.train.flow.values())
    assert len(flow) == 4 and all("lax.scan" in how for how in flow), flow
    small = tx.debug.summary(ppo.train, jr.key(0), cfg)
    assert small.top_level_ops < 200


def test_eager_and_compiled_agree():
    cfg = ppo.Config(num_envs=2, num_steps=4, num_updates=1, num_minibatches=2, update_epochs=1)
    eager_net, eager_info = ppo.train(jr.key(1), cfg)
    jit_net, jit_info = jax.jit(lambda k: ppo.train(k, cfg))(jr.key(1))
    for a, b in zip(jax.tree.leaves((eager_net, eager_info)), jax.tree.leaves((jit_net, jit_info))):
        assert jnp.allclose(a, b, atol=1e-5)


def test_environment_step_matches_a_reference_rollout():
    # plain `if done:` inside step(), lifted over environments
    keys = jr.split(jr.key(0), 5)
    obs, env = ppo.reset.lift(keys)
    for t in range(60):
        step_keys = jr.split(jr.key(t), 5)
        obs, env, reward, done, info = ppo.step.lift(step_keys, env, jnp.ones(5, jnp.int32))
        assert obs.shape == (5, 4) and reward.shape == (5,)
    assert int(info.timestep[0]) == 60
    assert bool((env.cart.time < 60).all())  # always pushing right ends episodes


def test_all_shapes_are_proved():
    reports = tx.prove(ppo)
    assert len(reports) == 11
    assert all(r.status == "proved" for r in reports), reports

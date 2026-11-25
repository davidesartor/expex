"""PureJaxRL's ppo.py vs examples/ppo: a whole jit-compiled training run on CartPole.

The original is PureJaxRL's unmodified `make_train` (with gymnax 0.0.8 and the
module stubs of examples/ppo/run_original.py); the port's `train` starts from
the original's initial network, so both run the same computation (identical
for the first updates, then drifting as float32 programs do).
"""

import sys
from unittest import mock

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_original

NAME = "ppo"
ORIGINALS = ["purejaxrl", "gymnax"]
CLAIM = {"train": "same, 1.01x (1.02x)"}  # README, from results/cpu-2026-11-25-v0.1.md: best (median)

CONFIGS = {
    "quick": dict(num_envs=4, num_steps=32, num_updates=8, num_minibatches=4, update_epochs=2),
    "readme": dict(num_envs=4, num_steps=128, num_updates=976, num_minibatches=4, update_epochs=4),
    # a GPU-sized batch: 1,024 envs, 32 minibatches of 4,096
    "realistic": dict(num_envs=1024, num_steps=128, num_updates=50, num_minibatches=32, update_epochs=4),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["readme"], "gpu": ["readme", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    use_original("gymnax")
    repo = use_original("purejaxrl")
    # as examples/ppo/run_original.py: PureJaxRL's wrappers.py imports brax and navix, unused for CartPole
    for name in ["brax", "brax.envs", "brax.envs.wrappers", "brax.envs.wrappers.training", "navix"]:
        sys.modules[name] = mock.MagicMock()
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree.map  # removed alias used by gymnax 0.0.8
    original = load_module("purejaxrl_ppo", repo / "ppo.py")
    port = load_module("expex_ppo", EXAMPLES / "ppo" / "ppo.py")

    config = {"LR": 2.5e-4, "NUM_ENVS": cfg["num_envs"], "NUM_STEPS": cfg["num_steps"],
              "UPDATE_EPOCHS": cfg["update_epochs"], "NUM_MINIBATCHES": cfg["num_minibatches"],
              "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5,
              "MAX_GRAD_NORM": 0.5, "ACTIVATION": "tanh", "ENV_NAME": "CartPole-v1", "ANNEAL_LR": True,
              "DEBUG": False,
              "TOTAL_TIMESTEPS": cfg["num_updates"] * cfg["num_steps"] * cfg["num_envs"]}  # fmt: skip
    pcfg = port.Config(**cfg)
    key = jr.PRNGKey(30)

    # the original's initial network (as make_train builds it), in the port's layout (as compare.py's net_from)
    _, k0 = jr.split(key)
    init = original.ActorCritic(2, activation="tanh").init(k0, jnp.zeros(4))["params"]
    leaves = [jnp.asarray(init[f"Dense_{i}"][w]) for i in range(6) for w in ("kernel", "bias")]
    net = jax.tree.unflatten(jax.tree.structure(port.Net(jr.key(0))), leaves)

    def same_returns(a, b):
        want = np.asarray(a["metrics"]["returned_episode_returns"])
        got = np.asarray(b[1].returned_return)
        same = (want == got).reshape(want.shape[0], -1).all(-1)
        first = int(np.argmin(same)) if not same.all() else len(same)
        return {"updates identical": first, "of": len(same)}

    steps = cfg["num_updates"] * cfg["num_steps"] * cfg["num_envs"]
    return [
        Measure(
            "train",
            Fn(jax.jit(original.make_train(config)), (key,)),
            Fn(jax.jit(lambda k, n: port.train(k, pcfg, n)), (key, net)),
            items=steps, unit="env steps", check=same_returns,
            note=f"{cfg['num_updates']} updates x {cfg['num_envs']} envs x {cfg['num_steps']} steps",
        )
    ]

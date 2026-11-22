"""Run the unmodified PureJaxRL ppo.py. Usage: run_original.py <purejaxrl dir> <old gymnax dir> <out.npz> <updates>"""
import sys, types
from unittest import mock
repo, gym_dir, out, n_updates = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, gym_dir)
for name in ["brax", "brax.envs", "brax.envs.wrappers", "brax.envs.wrappers.training", "navix"]:
    sys.modules[name] = mock.MagicMock()
sys.path.insert(0, repo + "/purejaxrl")
import jax, numpy as np
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map  # removed alias used by old gymnax
import ppo
config = {"LR": 2.5e-4, "NUM_ENVS": 4, "NUM_STEPS": 128, "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4,
          "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5,
          "MAX_GRAD_NORM": 0.5, "ACTIVATION": "tanh", "ENV_NAME": "CartPole-v1", "ANNEAL_LR": True,
          "DEBUG": False}
config["TOTAL_TIMESTEPS"] = n_updates * config["NUM_STEPS"] * config["NUM_ENVS"]
import time
t = time.time()
res = jax.jit(ppo.make_train(config))(jr.PRNGKey(30))
jax.block_until_ready(res)
print("original: %.1fs" % (time.time() - t))
ts = res["runner_state"][0]
flat = {"/".join(map(str, [getattr(k, "key", getattr(k, "name", k)) for k in path])): np.asarray(v)
        for path, v in jax.tree_util.tree_flatten_with_path(ts.params)[0]}
import jax.numpy as jnp
import jax.random as jr
_, k0 = jr.split(jr.PRNGKey(30))
init = ppo.ActorCritic(2, activation="tanh").init(k0, jnp.zeros(4))
init_flat = {"/".join(map(str, [getattr(k, "key", getattr(k, "name", k)) for k in path])): np.asarray(v)
        for path, v in jax.tree_util.tree_flatten_with_path(init)[0]}
m = res["metrics"]
np.savez(out, **{"param:" + k: v for k, v in flat.items()}, **{"init:" + k: v for k, v in init_flat.items()},
         returns=np.asarray(m["returned_episode_returns"]), episode=np.asarray(m["returned_episode"]),
         timestep=np.asarray(m["timestep"]), lengths=np.asarray(m["returned_episode_lengths"]))
print({k: v.shape for k, v in flat.items()})

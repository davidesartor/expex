"""Check that examples/ppo.py reproduces PureJaxRL's ppo.py.

    # 1. run the unmodified original (needs a PureJaxRL checkout and gymnax==0.0.8):
    python examples/run_purejaxrl_original.py <purejaxrl> <gymnax-0.0.8 dir> orig.npz <updates>
    # 2. compare:
    python examples/compare_ppo.py orig.npz
"""

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ppo import Config, Layer, Net, train  # noqa: E402

orig = np.load(sys.argv[1])
updates = orig["returns"].shape[0]


def net_from(prefix: str) -> Net:
    def layer(i):
        return Layer(jnp.asarray(orig[f"{prefix}:params/Dense_{i}/kernel"]), jnp.asarray(orig[f"{prefix}:params/Dense_{i}/bias"]))

    return Net((layer(0), layer(1), layer(2)), (layer(3), layer(4), layer(5)))


cfg = Config(num_updates=updates)
t = time.time()
net, info = jax.jit(lambda k, n: train(k, cfg, n))(jax.random.PRNGKey(30), net_from("init"))
jax.block_until_ready(net)
print(f"typax port: {time.time() - t:.1f}s for {updates} updates (trace + compile + run)")

worst = 0.0
for got, want in zip(jax.tree.leaves(net), jax.tree.leaves(net_from("param"))):
    worst = max(worst, float(jnp.max(jnp.abs(got - want))))
print(f"final parameters: max abs difference {worst:.3e}")

same_done = np.array_equal(np.asarray(info.returned_episode), orig["episode"])
same_len = np.array_equal(np.asarray(info.returned_length), orig["lengths"])
ret_diff = float(np.max(np.abs(np.asarray(info.returned_return) - orig["returns"])))
print(f"episode boundaries identical: {same_done};  episode lengths identical: {same_len}")
print(f"episode returns: max abs difference {ret_diff:.3e}")
print(f"timesteps identical: {np.array_equal(np.asarray(info.timestep), orig['timestep'])}")
ok = same_done and same_len and ret_diff == 0 and worst < 1e-4
print("MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

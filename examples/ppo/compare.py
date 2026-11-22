"""Check that examples/ppo/ppo.py reproduces PureJaxRL's ppo.py.

    # 1. run the unmodified original (needs a PureJaxRL checkout and gymnax==0.0.8):
    python examples/ppo/run_original.py <purejaxrl> <gymnax-0.0.8 dir> orig.npz <updates>
    # 2. compare:
    python examples/ppo/compare.py orig.npz
"""

import sys
import time

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ppo import Config, Net, train  # noqa: E402

orig = np.load(sys.argv[1])
updates = orig["returns"].shape[0]


def net_from(prefix: str) -> Net:
    """A network holding the original's parameters (actor layers, then critic layers)."""
    leaves = []
    for i in range(6):
        leaves += [orig[f"{prefix}:params/Dense_{i}/kernel"], orig[f"{prefix}:params/Dense_{i}/bias"]]
    return jax.tree.unflatten(jax.tree.structure(Net(jax.random.key(0))), [jnp.asarray(x) for x in leaves])


cfg = Config(num_updates=updates)
t = time.time()
net, info = jax.jit(lambda k, n: train(k, cfg, n))(jr.PRNGKey(30), net_from("init"))
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

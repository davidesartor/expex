"""mctx vs examples/mctx: batched MuZero and Gumbel MuZero searches on a small tabular MDP.

The same programs as examples/mctx/compare.py's `benchmark`: the original runs
with run_original.py's batched model functions (imported under the name `mctx`
through compare.py's `original_mctx`, since the port's module is `mctx` too),
the port with its own `recurrent_fn`, from the same roots and keys. No speed
claim exists yet.
"""

import jax
import jax.numpy as jnp
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_example, use_original

NAME = "mctx"
ORIGINALS = ["mctx"]
CLAIM: dict[str, str] = {}

CONFIGS = {
    "quick": dict(batch=8, simulations=16),
    "default": dict(batch=64, simulations=64),  # compare.py --mctx's default
    "realistic": dict(batch=1024, simulations=128),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["default"], "gpu": ["default", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    path = str(use_original("mctx"))
    import sys

    sys.path.remove(path)  # imported only inside compare.original_mctx: the port's module is `mctx` as well
    use_example("mctx")
    compare = load_module("mctx_compare", EXAMPLES / "mctx" / "compare.py")
    ro = sys.modules["run_original"]  # examples/mctx/run_original.py, imported by compare.py

    tables = ro.problem()
    params = {k: jnp.asarray(v) for k, v in tables.items()}
    mdp = compare.MDP(params["transitions"], params["rewards"], params["discounts"], params["logits"], params["values"])
    batch, sims = cfg["batch"], cfg["simulations"]
    starts = jax.random.randint(jax.random.key(0), (batch,), 0, ro.STATES)
    invalid = (jax.random.uniform(jax.random.key(1), (batch, ro.ACTIONS)) < 0.3).at[:, 0].set(False)
    key = jax.random.key(3)

    def same_trees(a, b):
        fields = ("children_visits", "parents")
        same = all(np.array_equal(getattr(a.search_tree, f), getattr(b.search_tree, f)) for f in fields)
        return {"same visit counts and parents": same}

    measures = []
    for policy in ("muzero", "gumbel"):

        def original(params, key, policy=policy):
            with compare.original_mctx(path) as mctx:
                f = mctx.muzero_policy if policy == "muzero" else mctx.gumbel_muzero_policy
                return f(params, key, ro.root_fn(params, starts), ro.recurrent_fn, sims, invalid_actions=invalid)

        def port(mdp, key, policy=policy):
            return compare.POLICIES[policy](mdp, key, mdp.root.vectorize(starts), compare.recurrent_fn, sims, invalid)

        measures.append(
            Measure(
                policy, Fn(jax.jit(original), (params, key)), Fn(jax.jit(port), (mdp, key)),
                items=batch, unit="searches", check=same_trees, labels=("mctx", "port"),
                note=f"batch {batch} x {sims} simulations",
            )  # fmt: skip
        )
    return measures

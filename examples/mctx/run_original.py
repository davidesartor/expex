"""Run the original mctx (github.com/google-deepmind/mctx) and save reference searches.

    git clone https://github.com/google-deepmind/mctx /tmp/mctx    # 428bfb7, 2026
    pip install chex
    PYTHONPATH=/tmp/mctx python examples/mctx/run_original.py tests/data/mctx_reference.npz

The problem is a small random tabular MDP with fixed tables: states are
one-hot vectors, and the model's reward, discount, prior and value are looked
up in the tables. A batch of 8 searches from different start states, with
fixed keys, runs through `mctx.muzero_policy` and `mctx.gumbel_muzero_policy`
in four settings (with and without an invalid-action mask, with and without
`max_depth`). For each, the file holds the chosen actions, the action weights,
the root values and every array of the final search tree, for
tests/ports/test_mctx.py.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

STATES, ACTIONS, BATCH = 12, 6, 8

# name: (policy, simulations, keyword arguments, use the invalid-action mask)
CASES = {
    "muzero": ("muzero", 32, {}, True),
    "muzero_depth": ("muzero", 32, {"max_depth": 2}, False),
    "gumbel": ("gumbel", 32, {"max_num_considered_actions": 4}, True),
    "gumbel_depth": ("gumbel", 32, {"max_depth": 3}, False),
}
TREE_FIELDS = (
    "node_visits", "raw_values", "node_values", "parents", "action_from_parent", "children_index",
    "children_prior_logits", "children_visits", "children_rewards", "children_discounts",
    "children_values", "embeddings", "root_invalid_actions",
)  # fmt: skip


def problem(seed: int = 0) -> dict[str, np.ndarray]:
    """The MDP's tables, the start states of the batch and an invalid-action mask."""
    rng = np.random.RandomState(seed)
    invalid = rng.rand(BATCH, ACTIONS) < 0.35
    invalid[np.arange(BATCH), rng.randint(ACTIONS, size=BATCH)] = False  # at least one valid action
    return {
        "transitions": rng.randint(STATES, size=(STATES, ACTIONS)).astype(np.int32),
        "rewards": rng.randn(STATES, ACTIONS).astype(np.float32),
        "discounts": np.where(rng.rand(STATES, ACTIONS) < 0.15, 0.0, 0.95).astype(np.float32),  # some terminal
        "logits": rng.randn(STATES, ACTIONS).astype(np.float32),
        "values": rng.randn(STATES).astype(np.float32),
        "starts": rng.randint(STATES, size=BATCH).astype(np.int32),
        "invalid": invalid,
    }


def root_fn(params, starts):
    import mctx

    return mctx.RootFnOutput(
        prior_logits=params["logits"][starts],
        value=params["values"][starts],
        embedding=jax.nn.one_hot(starts, STATES),
    )


def recurrent_fn(params, rng_key, action, embedding):
    """mctx's batched model: the state after `action` in each of the B states."""
    import mctx

    del rng_key
    state = jnp.argmax(embedding, axis=-1)
    following = params["transitions"][state, action]
    out = mctx.RecurrentFnOutput(
        reward=params["rewards"][state, action],
        discount=params["discounts"][state, action],
        prior_logits=params["logits"][following],
        value=params["values"][following],
    )
    return out, jax.nn.one_hot(following, STATES)


def run(name: str, params, seed: int):
    """One case: mctx's PolicyOutput for the batch, jit-compiled."""
    import mctx

    policy, sims, kwargs, masked = CASES[name]
    f = mctx.muzero_policy if policy == "muzero" else mctx.gumbel_muzero_policy
    invalid = jnp.asarray(params["invalid"]) if masked else None

    @jax.jit
    def go(params, key):
        root = root_fn(params, params["starts"])
        return f(params, key, root, recurrent_fn, sims, invalid_actions=invalid, **kwargs)

    return go(params, jax.random.key(seed))


def main(out: str) -> None:
    tables = problem()
    params = {k: jnp.asarray(v) for k, v in tables.items()}
    arrays = {f"problem/{k}": v for k, v in tables.items()}
    for seed, name in enumerate(CASES):
        result = run(name, params, seed)
        tree = result.search_tree
        arrays[f"{name}/seed"] = np.int32(seed)
        arrays[f"{name}/action"] = np.asarray(result.action)
        arrays[f"{name}/action_weights"] = np.asarray(result.action_weights)
        arrays[f"{name}/root_value"] = np.asarray(tree.summary().value)
        for field in TREE_FIELDS:
            arrays[f"{name}/tree/{field}"] = np.asarray(getattr(tree, field))
        if tree.extra_data is not None:
            arrays[f"{name}/tree/root_gumbel"] = np.asarray(tree.extra_data.root_gumbel)
        depth = (np.asarray(tree.parents) >= 0).sum(-1)
        print(f"{name:13s} actions {np.asarray(result.action)}  nodes used per tree {depth + 1}")
    np.savez_compressed(out, **arrays)
    print(f"saved {len(arrays)} arrays to {out}")


if __name__ == "__main__":
    main(sys.argv[1])

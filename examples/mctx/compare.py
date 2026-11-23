"""Check the mctx port against the original, and time both.

    # against the saved reference (no mctx needed):
    python examples/mctx/compare.py tests/data/mctx_reference.npz
    # against mctx itself, live, plus speed and compile time:
    python examples/mctx/compare.py tests/data/mctx_reference.npz --mctx /tmp/mctx [batch] [simulations]

The reference file is written by run_original.py: four searches of a batch of
8 trees on a small tabular MDP, with both policies. The port runs the same
searches from the same keys; every array of every tree, the action weights,
the actions and the root values are compared.
"""

import contextlib
import importlib
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from mctx import MDP, gumbel_muzero_policy, muzero_policy, recurrent_fn  # noqa: E402
from run_original import CASES, TREE_FIELDS  # noqa: E402

POLICIES = {"muzero": muzero_policy, "gumbel": gumbel_muzero_policy}


def problem(ref: dict) -> tuple[MDP, jax.Array, jax.Array]:
    """The MDP, the batch's start states and the invalid-action mask of the reference."""
    p = {k[len("problem/") :]: jnp.asarray(v) for k, v in ref.items() if k.startswith("problem/")}
    mdp = MDP(p["transitions"], p["rewards"], p["discounts"], p["logits"], p["values"])
    return mdp, p["starts"], p["invalid"]


def run(name: str, mdp: MDP, starts, invalid, seed: int, jit: bool = True):
    """The port's `Policy` for one case of run_original.CASES."""
    policy, sims, kwargs, masked = CASES[name]
    f = POLICIES[policy]

    def go(mdp, key):
        roots = mdp.root.vectorize(starts)
        return f(mdp, key, roots, recurrent_fn, sims, invalid if masked else None, **kwargs)

    return (jax.jit(go) if jit else go)(mdp, jax.random.key(seed))


def outputs(result) -> dict[str, np.ndarray]:
    """The port's outputs, named as in the reference file."""
    tree = result.search_tree
    out = {
        "action": result.action, "action_weights": result.action_weights,
        "root_value": tree.summary.vectorize().value, "tree/root_gumbel": tree.root_gumbel,
    }  # fmt: skip
    out.update({f"tree/{f}": getattr(tree, f) for f in TREE_FIELDS})
    return {k: np.asarray(v) for k, v in out.items()}


def compare(ref: dict, jit: bool = True) -> dict[str, dict[str, float]]:
    """For each case and output, the largest difference from the reference (0 for identical)."""
    mdp, starts, invalid = problem(ref)
    result = {}
    for name in CASES:
        got = outputs(run(name, mdp, starts, invalid, int(ref[f"{name}/seed"]), jit))
        diffs = {}
        for k, g in got.items():
            key = f"{name}/{k}"
            if key not in ref:  # MuZero has no Gumbel noise; the port keeps zeros there
                assert not g.any(), key
                continue
            w = ref[key]
            diffs[k] = float(np.abs(g.astype(np.float64) - w.astype(np.float64)).max())
        result[name] = diffs
    return result


# ------------------------------------------------------------------- speed


@contextlib.contextmanager
def original_mctx(path: str):
    """The original package as `mctx`, for run_original's functions; the port's module is `mctx` too."""
    port = sys.modules.pop("mctx")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("mctx")
    finally:
        sys.path.remove(path)
        sys.modules["mctx"] = port


def benchmark(path: str, ref: dict, batch: int, sims: int, rounds: int = 15) -> dict[str, dict[str, float]]:
    """Searches per second and compile time, mctx vs port, both jit-compiled.

    The batch's roots are random states of the reference MDP; mctx runs with
    its batched recurrent function (run_original.recurrent_fn), the port with
    its own. The runs alternate and the best of `rounds` counts.
    """
    import run_original

    mdp, _, _ = problem(ref)
    params = {k[len("problem/") :]: jnp.asarray(v) for k, v in ref.items() if k.startswith("problem/")}
    starts = jax.random.randint(jax.random.key(0), (batch,), 0, run_original.STATES)
    invalid = jax.random.uniform(jax.random.key(1), (batch, run_original.ACTIONS)) < 0.3
    invalid = invalid.at[:, 0].set(False)
    out: dict[str, dict[str, float]] = {}
    for policy in ("muzero", "gumbel"):

        def go_original(params, key):
            with original_mctx(path) as mctx:
                f = mctx.muzero_policy if policy == "muzero" else mctx.gumbel_muzero_policy
                root = run_original.root_fn(params, starts)
                return f(params, key, root, run_original.recurrent_fn, sims, invalid_actions=invalid)

        def go_port(mdp, key):
            roots = mdp.root.vectorize(starts)
            return POLICIES[policy](mdp, key, roots, recurrent_fn, sims, invalid)

        key = jax.random.key(3)
        programs = {"mctx": (go_original, params), "port": (go_port, mdp)}
        compiled, row = {}, {label: float("inf") for label in ("mctx compile s", "port compile s")}
        for _ in range(3):  # tracing + XLA; the first compile in a process also pays a warm-up
            for label, (f, arg) in programs.items():
                jax.clear_caches()
                t = time.perf_counter()
                compiled[label] = jax.jit(f).lower(arg, key).compile()
                row[f"{label} compile s"] = min(row[f"{label} compile s"], time.perf_counter() - t)
        best = {label: float("inf") for label in compiled}
        for _ in range(rounds):
            for label, c in compiled.items():
                arg = programs[label][1]
                t = time.perf_counter()
                jax.block_until_ready(c(arg, key))
                best[label] = min(best[label], time.perf_counter() - t)
        for label in compiled:
            row[f"{label} searches/s"] = batch / best[label]
        a, b = compiled["mctx"](params, key), compiled["port"](mdp, key)
        row["same trees"] = float(
            all(np.array_equal(getattr(a.search_tree, f), getattr(b.search_tree, f)) for f in ("children_visits", "parents"))
        )
        out[policy] = row
    return out


def main() -> None:
    ref = dict(np.load(sys.argv[1]))
    result = compare(ref)
    for name, diffs in result.items():
        worst = max(diffs.values())
        exact = [k for k, v in diffs.items() if v == 0]
        print(f"{name:13s} largest difference {worst:.3g}; identical: {len(exact)} of {len(diffs)} arrays")
        for k, v in diffs.items():
            if v:
                print(f"    {k}: {v:.3g}")
    ok = all(v < 1e-5 for d in result.values() for v in d.values())
    print("MATCH" if ok else "MISMATCH")
    if "--mctx" in sys.argv:
        i = sys.argv.index("--mctx")
        sizes = [int(a) for a in sys.argv[i + 2 :]] + [64, 64]
        batch, sims = sizes[0], sizes[1]
        print(f"\nspeed on {jax.devices()[0].platform.upper()}, batch {batch} x {sims} simulations, jit-compiled:")
        for policy, r in benchmark(sys.argv[i + 1], ref, batch, sims).items():
            print(
                f"  {policy:7s} mctx {r['mctx searches/s']:8.0f} searches/s (compile {r['mctx compile s']:.2f} s)   "
                f"port {r['port searches/s']:8.0f} searches/s (compile {r['port compile s']:.2f} s)   "
                f"same trees: {bool(r['same trees'])}"
            )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

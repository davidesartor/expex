"""The MuZero and Gumbel MuZero port in examples/mctx/ reproduces mctx.

``tests/data/mctx_reference.npz`` holds four searches of a batch of 8 trees on
a small tabular MDP, run by the unmodified mctx from fixed keys: both
policies, with and without an invalid-action mask and ``max_depth``. For each,
the actions, action weights, root values and every array of the final trees
(see examples/mctx/run_original.py).
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

import expex

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "mctx"))
import mctx  # noqa: E402  (the port)

# several examples have a compare.py: load this one under its own name
_spec = importlib.util.spec_from_file_location("mctx_compare", ROOT / "examples" / "mctx" / "compare.py")
assert _spec is not None and _spec.loader is not None
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

REF = dict(np.load(ROOT / "tests" / "data" / "mctx_reference.npz"))
INTEGER = {"action", "tree/node_visits", "tree/parents", "tree/action_from_parent", "tree/children_index",
           "tree/children_visits", "tree/root_invalid_actions"}  # fmt: skip


def check(diffs: dict[str, float], tol: float = 1e-6) -> None:
    for k, d in diffs.items():
        assert d == 0 if k in INTEGER else d <= tol, (k, d)


@pytest.fixture(scope="module")
def compiled():
    """The port's differences from the reference, jit-compiled (compiled once for the module)."""
    return compare.compare(REF)


def test_trees_and_outputs_match_the_original(compiled):
    assert set(compiled) == {"muzero", "muzero_depth", "gumbel", "gumbel_depth"}
    for name, diffs in compiled.items():
        assert len(diffs) >= 16, name  # outputs and every array of the tree
        check(diffs, tol=0.0)  # identical, floats included


def test_the_reference_covers_the_search():
    for name in ("muzero_depth", "gumbel_depth"):  # max_depth cut some simulations short
        visits = REF[f"{name}/tree/node_visits"]
        assert (visits[:, 1:] > 1).any() and (REF[f"{name}/tree/parents"][:, -1] == -1).any(), name
    for name in ("muzero", "gumbel"):  # masked root actions are never visited
        tree = f"{name}/tree/"
        invalid = REF[tree + "root_invalid_actions"].astype(bool)
        assert invalid.any() and (REF[tree + "children_visits"][:, 0][invalid] == 0).all()
    discounts = REF["muzero/tree/children_discounts"]
    assert ((discounts == 0) & (REF["muzero/tree/children_visits"] > 0)).any()  # a terminal transition


def test_the_search_is_staged_as_written(compiled):
    assert set(mctx.Tree.simulate.flow.values()) == {"while -> lax.while_loop"}
    assert set(mctx.Tree.backward.flow.values()) == {"while -> lax.while_loop"}
    assert set(mctx.search.flow.values()) == {"for -> lax.scan (carries tree)", "if -> lax.cond"}


def test_eager_and_compiled_agree():
    # the batched policies, not jitted
    for diffs in compare.compare(REF, jit=False, cases=("muzero_depth", "gumbel")).values():
        check(diffs)
    # one tree alone in plain Python (no vmap, no jit): every while / if runs as Python
    for name, b in (("muzero", 0), ("gumbel_depth", 3)):
        tree = compare.one_tree(REF, name, b)
        for f in compare.TREE_FIELDS:
            got, want = np.asarray(getattr(tree, f)).astype(np.float64), REF[f"{name}/tree/{f}"][b].astype(np.float64)
            assert np.abs(got - want).max() <= (0 if f"tree/{f}" in INTEGER else 1e-5), (name, f)


def test_every_shape_is_proved():
    reports = expex.prove(mctx)
    assert all(r.status == "proved" for r in reports), str(reports)
    assert len(reports) == 26

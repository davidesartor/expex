"""The Mamba port in examples/mamba/mamba.py reproduces vvvm23/mamba-jax.

``tests/data/mamba_reference.npz`` holds the parameters of a small model built
by the unmodified original, its logits (with both of its scan kernels), the
gradients of a loss and tokens sampled with its recurrent step (see
examples/mamba/run_original.py).
"""

import pathlib
import sys

import numpy as np

import typax as tx

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "mamba"))
import compare  # noqa: E402
import mamba  # noqa: E402

REF = dict(np.load(ROOT / "tests" / "data" / "mamba_reference.npz"))


def test_matches_the_original():
    diffs = compare.compare(REF)
    assert diffs["logits"] < 1e-5 and diffs["logits (associative scan)"] < 1e-5
    assert diffs["loss"] < 1e-5
    assert diffs["gradients (relative)"] < 1e-4
    assert diffs["sampled tokens differing"] == 0


def test_every_shape_is_proved():
    reports = tx.prove(mamba)
    assert all(r.status == "proved" for r in reports), str(reports)
    assert len(reports) == 12

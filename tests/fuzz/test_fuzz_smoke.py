"""A short, seeded differential-fuzzing run: generated programs mean the same under expex
(eagerly, under jax.jit, batched, under jax.grad) as in plain Python, or raise a coded error.

The long run is ``python -m tests.fuzz.run --n 20000`` (see ``run.py``).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fuzzgen  # noqa: E402
import fuzzoracle  # noqa: E402

SEEDS = list(range(900_000, 900_012))


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_program_agrees_with_plain_python(seed):
    prog = fuzzgen.generate(seed)
    modes = ("eager", "jit", "batch", "grad") if seed % 2 else ("eager", "jit", "vmap")
    rep = fuzzoracle.check(prog.source(), prog.source("truth"), seed, modes=modes)
    if rep.truth_failed:
        pytest.skip(f"the plain-Python program raised: {rep.truth_failed}")
    bad = [f"{o.mode}: {o.kind}: {o.detail[:600]}" for o in rep.failures]
    assert not bad, "\n".join(bad) + "\n\n" + prog.source()

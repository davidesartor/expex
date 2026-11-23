"""The Llama 2 port in examples/llama/llama.py reproduces ayaka14732/llama-2-jax.

``tests/data/llama_reference.npz`` holds the parameters of a small model built
by the unmodified original, its logits, the gradients of a loss and the tokens
its own generation loop picks greedily with the KV cache (see
examples/llama/run_original.py).
"""

import importlib.util
import pathlib
import sys

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "llama"))
import llama  # noqa: E402

# examples/mamba has a compare.py too: load this one under its own name
_spec = importlib.util.spec_from_file_location("llama_compare", ROOT / "examples" / "llama" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

REF = dict(np.load(ROOT / "tests" / "data" / "llama_reference.npz"))


def test_matches_the_original():
    diffs = compare.compare(REF)
    assert diffs["logits"] < 1e-5
    assert diffs["loss"] < 1e-5
    assert diffs["gradients (relative)"] < 1e-4
    assert diffs["generated tokens differing"] == 0


def test_cached_steps_match_the_full_pass():
    model = llama.Llama(jr.key(0), V=50, D=32, F=48, G=2, R=3, K=8, layers=2)
    tokens = jr.randint(jr.key(1), (10,), 0, 50)
    full = model(tokens)
    h, caches = model.extend(tokens[:4], 0, model.empty(10))  # a prompt, then one token at a time
    steps = [h]
    for t in range(4, 10):
        h, caches = model.extend(tokens[t : t + 1], t, caches)
        steps.append(h)
    assert jnp.abs(jnp.concatenate(steps) @ model.unembed - full).max() < 1e-5


def test_every_shape_is_proved():
    reports = expex.prove(llama)
    assert all(r.status == "proved" for r in reports), str(reports)
    assert len(reports) == 11

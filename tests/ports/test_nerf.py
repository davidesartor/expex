"""The NeRF port in examples/nerf/nerf.py reproduces google-research's JaxNeRF.

``tests/data/nerf_reference.npz`` holds the parameters of a small model built
by the unmodified original, its coarse and fine renders of 64 camera rays
(randomized and not), and the gradients of its training loss (see
examples/nerf/run_original.py).
"""

import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "nerf"))
import nerf  # noqa: E402

# other examples have a compare.py too: load this one under its own name
_spec = importlib.util.spec_from_file_location("nerf_compare", ROOT / "examples" / "nerf" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

REF = dict(np.load(ROOT / "tests" / "data" / "nerf_reference.npz"))


def test_matches_the_original():
    diffs = compare.compare(REF)
    for mode in ("random", "fixed"):
        for name in ("rgb_c", "disp_c", "acc_c", "rgb", "disp", "acc"):
            assert diffs[f"{name} ({mode})"] < 1e-5, (name, mode, diffs)
    assert diffs["loss"] < 1e-6
    assert diffs["gradients (relative)"] < 1e-4
    assert diffs["camera rays"] < 1e-6


def test_the_default_model_has_the_originals_layers():
    model = jax.eval_shape(lambda: nerf.NeRF.create(jr.key(0)))  # depth 8, width 256, skip after layer 4
    mlp = model.coarse
    assert mlp.first.kernel.shape == (63, 256) and mlp.skip.kernel.shape == (256 + 63, 256)
    assert len(mlp.before_skip) == 4 and len(mlp.after_skip) == 2
    assert mlp.condition.kernel.shape == (256 + 27, 128) and mlp.rgb.kernel.shape == (128, 3)


def test_sample_pdf_puts_samples_where_the_weight_is():
    bins = jnp.linspace(0.0, 4.0, 5)
    weights = jnp.array([0.0, 1.0, 0.0, 0.0])  # all weight in [1, 2]
    z = nerf.sample_pdf(bins, weights, jnp.linspace(0.0, 0.999, 16))
    assert bool(((z >= 1.0) & (z <= 2.0)).all())


def test_every_shape_is_proved():
    reports = expex.prove(nerf)
    assert all(r.status == "proved" for r in reports), str(reports)
    assert len(reports) == 12

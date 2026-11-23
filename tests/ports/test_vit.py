"""The ViT port in examples/vit/vit.py reproduces google-research/vision_transformer.

``tests/data/vit_reference.npz`` holds the parameters of a small
`vit_jax.models_vit.VisionTransformer` built by the unmodified original, its
logits on two 32 x 24 images in 4 x 2 patches, and a loss with its gradients
(see examples/vit/run_original.py).
"""

import importlib.util
import pathlib
import sys

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import expex

ROOT = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "examples" / "vit"))
import vit  # noqa: E402

# other examples have a compare.py too: load this one under its own name
_spec = importlib.util.spec_from_file_location("vit_compare", ROOT / "examples" / "vit" / "compare.py")
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)

REF = dict(np.load(ROOT / "tests" / "data" / "vit_reference.npz"))


def test_matches_the_original():
    diffs = compare.compare(REF)
    assert diffs["logits"] < 1e-5
    assert diffs["loss"] < 1e-5
    assert diffs["gradients (relative)"] < 1e-4
    assert diffs["key-bias gradients (zero)"] < 1e-6


def test_images_of_another_size_are_refused():
    model = vit.ViT(jr.key(0), GH=2, GW=3, P=4, Q=2, C=3, D=16, H=2, K=8, F=32, Classes=5, layers=2)
    assert model(jnp.ones((8, 6, 3))).shape == (5,)
    with pytest.raises(expex.ShapeError, match="GW \\* Q"):
        model(jnp.ones((8, 8, 3)))
    with pytest.raises(expex.ShapeError, match="Wpx % Q == 0"):
        model.embed(jnp.ones((8, 7, 3)))  # the patch embedding alone takes any grid, of whole patches


def test_every_shape_is_proved():
    reports = expex.prove(vit)
    assert all(r.status == "proved" for r in reports), str(reports)
    assert len(reports) == 8

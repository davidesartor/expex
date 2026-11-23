"""expex.save / expex.load: checked saving and loading of typed modules."""

import pathlib
import sys
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import expex
from expex import f32
from expex.io import load, save

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "examples" / "mamba"))
import mamba  # noqa: E402


@expex.struct
class Layer[D, H]:
    w: f32[H, D]
    b: f32[H]
    scale: float = 1.0
    act: Callable = jnp.tanh

    def __init__(self, key, d: int, h: int, scale: float = 1.0):
        self.w = jr.normal(key, (h, d))
        self.b = jnp.zeros(h)
        self.scale = scale


class Stats(NamedTuple):
    mean: f32[4]
    note: str = "none"


@expex.struct
class Net[D, H]:
    proj: f32[H + 1, D]
    layers: list[Layer[D, H]]
    stats: Stats
    name: str
    dims: tuple[int, ...] = ()
    bias: f32[H] | None = None
    act: Callable = jax.nn.relu

    def __init__(self, key, d: int, h: int, depth: int, name: str = "net"):
        k0, *ks = jr.split(key, depth + 1)
        self.proj = jr.normal(k0, (h + 1, d))
        self.layers = [Layer(k, d, h, scale=float(i + 1)) for i, k in enumerate(ks)]
        self.stats = Stats(jnp.arange(4.0), note="hello")
        self.name = name
        self.dims = (d, h)


@expex.struct
class NeedsAct[D]:
    w: f32[D]
    act: Callable


def _keys(path):
    with np.load(path) as f:
        return set(f.files)


def _rewrite(path, **changes):
    """The file at ``path`` with some arrays replaced (None removes one)."""
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    for k, v in changes.items():
        if v is None:
            del data[k]
        else:
            data[k] = v
    np.savez(path, **data)


@pytest.fixture
def net():
    return Net(jr.key(0), d=3, h=5, depth=3, name="tiny")


def test_round_trip(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    assert {"proj", "layers[2].w", "layers[0].b", "stats.mean"} <= _keys(p)
    back = load(Net, p)
    assert type(back) is Net and len(back.layers) == 3
    for a, b in zip(jax.tree.leaves(net), jax.tree.leaves(back)):
        np.testing.assert_array_equal(a, b)
    assert isinstance(back.proj, jax.Array)
    assert back.stats == back.stats._replace(note="hello") and isinstance(back.stats, Stats)


def test_static_fields_restored(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    back = load(Net, p)
    assert back.name == "tiny" and back.dims == (3, 5)
    assert [layer.scale for layer in back.layers] == [1.0, 2.0, 3.0]
    assert back.bias is None
    assert back.act is jax.nn.relu and back.layers[0].act is jnp.tanh  # class defaults
    y = back.layers[1].act(back.layers[1].w)  # usable as a module
    assert y.shape == (5, 3)


def test_optional_array_and_template(net, tmp_path):
    p = tmp_path / "net.npz"
    net = net.replace(bias=jnp.ones(5), act=jax.nn.gelu)
    save(net, p)
    back = load(net, p)
    assert back.act is jax.nn.gelu  # from the template
    np.testing.assert_array_equal(back.bias, jnp.ones(5))
    np.testing.assert_array_equal(load(Net, p).bias, jnp.ones(5))


def test_wrong_shape_names_the_field(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, **{"layers[2].w": np.zeros((5, 4), np.float32)})
    with pytest.raises(expex.ShapeError) as e:
        load(Net, p)
    msg = str(e.value)
    assert "layers[2].w: declared f32[H, D], file has f32[5, 4]" in msg
    assert "D is 3" in msg


def test_wrong_computed_size(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, proj=np.zeros((5, 3), np.float32))
    with pytest.raises(expex.ShapeError, match=r"proj: declared f32\[H \+ 1, D\], file has f32\[5, 3\]"):
        load(Net, p)


def test_wrong_dtype(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, **{"stats.mean": np.zeros(4, np.int32)})
    with pytest.raises(expex.ShapeError, match=r"stats.mean: declared f32\[4\], file has i32\[4\]"):
        load(Net, p)


def test_missing_and_extra_keys(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, **{"layers[1].b": None, "junk": np.zeros(2)})
    with pytest.raises(expex.ShapeError) as e:
        load(Net, p)
    assert "missing from the file: layers[1].b" in str(e.value)
    assert "junk" in str(e.value)

    save(net, p)
    _rewrite(p, junk=np.zeros(2))
    with pytest.raises(expex.ShapeError, match="in the file but not in Net: junk"):
        load(Net, p)
    assert len(load(Net, p, strict=False).layers) == 3


def test_template_with_different_length_reports_keys(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    short = net.replace(layers=net.layers[:2])
    with pytest.raises(expex.ShapeError, match=r"in the file but not in Net: layers\[2\]\.b, layers\[2\]\.w"):
        load(short, p)


def test_size_keyword(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    assert load(Net, p, D=3, H=5).proj.shape == (6, 3)
    with pytest.raises(expex.ShapeError, match=r"D is 4 \(from D=4 passed to expex.load\)"):
        load(Net, p, D=4)
    with pytest.raises(TypeError, match="no size `Q`"):
        load(Net, p, Q=1)


def test_unsaved_static_without_default(tmp_path):
    p = tmp_path / "a.npz"
    m = NeedsAct(jnp.ones(3), jnp.sin)
    save(m, p)
    with pytest.raises(TypeError, match=r"static field `act` \(NeedsAct.act\)"):
        load(NeedsAct, p)
    assert load(m, p).act is jnp.sin


def test_mamba(tmp_path):
    model = mamba.Mamba.create(jr.key(0), vocab=20, d=16, layers=2, state=4)
    p = tmp_path / "mamba.npz"
    save(model, p)
    assert "layers[1].mixer.x_proj" in _keys(p)
    back = load(mamba.Mamba, p, D=16)
    tokens = jnp.arange(6) % 20
    np.testing.assert_allclose(back(tokens), model(tokens), rtol=1e-6)
    assert back.layers[0].norm.eps == model.layers[0].norm.eps

    _rewrite(p, **{"layers[1].mixer.x_proj": np.zeros((40, 32), np.float32)})
    with pytest.raises(expex.ShapeError) as e:
        load(mamba.Mamba, p)
    assert "layers[1].mixer.x_proj: declared f32[R + 2 * N, E], file has f32[40, 32]" in str(e.value)


def test_bfloat16(tmp_path):
    p = tmp_path / "h.npz"
    @expex.struct
    class Half[D]:
        w: expex.Float[D]

    save(Half(jnp.arange(3, dtype=jnp.bfloat16)), p)
    back = load(Half, p)
    assert back.w.dtype == jnp.bfloat16
    np.testing.assert_array_equal(back.w, jnp.arange(3, dtype=jnp.bfloat16))


class Moments(NamedTuple):  # as an optax state: no annotations, models inside
    count: Any
    mu: Any


def test_pytree_of_structs(net, tmp_path):
    state = {"model": net, "opt": (Moments(jnp.int32(7), net), jnp.arange(3.0)), "step": jnp.int32(7)}
    save(state, tmp_path / "s.npz")
    keys = set(np.load(tmp_path / "s.npz").files)
    assert {"['model'].proj", "['opt'][0].mu.layers[1].w", "['opt'][0].count", "['opt'][1]", "['step']"} <= keys

    loaded = load(state, tmp_path / "s.npz")
    assert jax.tree.structure(loaded) == jax.tree.structure(state)
    for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(state)):
        assert a == b if isinstance(b, str) else isinstance(a, jax.Array) and jnp.array_equal(a, b) and a.dtype == b.dtype
    assert loaded["opt"][0].mu.layers[0].scale == net.layers[0].scale  # static fields from the template

    other = dict(state, step=jnp.zeros(2))
    with pytest.raises(expex.ShapeError, match=r"\['step'\]: the template has"):
        load(other, tmp_path / "s.npz")
    with pytest.raises(expex.ShapeError, match="missing from the file: \\['extra'\\]"):
        load(dict(state, extra=jnp.zeros(2)), tmp_path / "s.npz")
    with pytest.raises(TypeError, match="no struct in dict has a size `Q`"):
        load(state, tmp_path / "s.npz", Q=3)

    arrays = dict(np.load(tmp_path / "s.npz"))
    arrays["['opt'][0].mu.proj"] = np.zeros((7, 3), np.float32)  # declared f32[H + 1, D]: H + 1 is 6
    np.savez(tmp_path / "b.npz", **arrays)
    with pytest.raises(expex.ShapeError, match=r"\['opt'\]\[0\]\.mu\.proj"):
        load(state, tmp_path / "b.npz")


def test_pytree_into_an_abstract_template(tmp_path):
    model = mamba.Mamba.create(jr.key(0), vocab=20, d=16, layers=2, state=4)
    state = (model, Moments(jnp.int32(3), jax.tree.map(jnp.zeros_like, model)))
    save(state, tmp_path / "m.npz")
    loaded = load(jax.eval_shape(lambda: state), tmp_path / "m.npz")  # no memory used for the template
    assert all(jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(loaded), jax.tree.leaves(state)))


def test_names_need_a_mesh(net, tmp_path):
    save(net, tmp_path / "n.npz")
    with pytest.raises(TypeError, match="needs mesh="):
        load(Net, tmp_path / "n.npz", names={"D": "model"})

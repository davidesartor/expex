"""tx.save / tx.load: checked saving and loading of typed modules."""

import pathlib
import sys
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import typax as tx
from typax import f32
from typax._io import load, save

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "examples" / "mamba"))
import mamba  # noqa: E402


@tx.struct
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


@tx.struct
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


@tx.struct
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
    with pytest.raises(tx.ShapeError) as e:
        load(Net, p)
    msg = str(e.value)
    assert "layers[2].w: declared f32[H, D], file has f32[5, 4]" in msg
    assert "D is 3" in msg


def test_wrong_computed_size(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, proj=np.zeros((5, 3), np.float32))
    with pytest.raises(tx.ShapeError, match=r"proj: declared f32\[H \+ 1, D\], file has f32\[5, 3\]"):
        load(Net, p)


def test_wrong_dtype(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, **{"stats.mean": np.zeros(4, np.int32)})
    with pytest.raises(tx.ShapeError, match=r"stats.mean: declared f32\[4\], file has i32\[4\]"):
        load(Net, p)


def test_missing_and_extra_keys(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    _rewrite(p, **{"layers[1].b": None, "junk": np.zeros(2)})
    with pytest.raises(tx.ShapeError) as e:
        load(Net, p)
    assert "missing from the file: layers[1].b" in str(e.value)
    assert "junk" in str(e.value)

    save(net, p)
    _rewrite(p, junk=np.zeros(2))
    with pytest.raises(tx.ShapeError, match="in the file but not in Net: junk"):
        load(Net, p)
    assert len(load(Net, p, strict=False).layers) == 3


def test_template_with_different_length_reports_keys(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    short = net.replace(layers=net.layers[:2])
    with pytest.raises(tx.ShapeError, match=r"in the file but not in Net: layers\[2\]\.b, layers\[2\]\.w"):
        load(short, p)


def test_size_keyword(net, tmp_path):
    p = tmp_path / "net.npz"
    save(net, p)
    assert load(Net, p, D=3, H=5).proj.shape == (6, 3)
    with pytest.raises(tx.ShapeError, match=r"D is 4 \(from D=4 passed to tx.load\)"):
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
    with pytest.raises(tx.ShapeError) as e:
        load(mamba.Mamba, p)
    assert "layers[1].mixer.x_proj: declared f32[R + 2 * N, E], file has f32[40, 32]" in str(e.value)


def test_bfloat16(tmp_path):
    p = tmp_path / "h.npz"
    @tx.struct
    class Half[D]:
        w: tx.Float[D]

    save(Half(jnp.arange(3, dtype=jnp.bfloat16)), p)
    back = load(Half, p)
    assert back.w.dtype == jnp.bfloat16
    np.testing.assert_array_equal(back.w, jnp.arange(3, dtype=jnp.bfloat16))

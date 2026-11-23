"""Sharding by axis names (``expex.sharding``) on a simulated 8-device CPU mesh.

The multi-device tests need ``--xla_force_host_platform_device_count=8`` before
JAX creates its CPU backend, and setting it here would give every other test
module 8 devices too. So unless this process already has 8 devices,
``test_on_eight_devices`` reruns this file in a fresh process with the flag,
and the in-process copies skip.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

import expex
from expex import f32
from expex.sharding import device_put, shard, shardings

EIGHT = jax.device_count() >= 8
needs8 = pytest.mark.skipif(not EIGHT, reason="run in a subprocess by test_on_eight_devices")


def test_on_eight_devices():
    if EIGHT:
        pytest.skip("this process already has 8 devices; the tests ran here")
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8")
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    env["PYTHONPATH"] = os.pathsep.join([src, env.get("PYTHONPATH", "")])
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__],
        env=env, capture_output=True, text=True, timeout=600, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" not in r.stdout.splitlines()[-1].replace("1 skipped", ""), r.stdout


@expex.struct
class Block[D, F]:
    w: f32[D, F]
    b: f32[F]
    scale: float = 1.0

    def __call__[B](self, x: f32[B, D]) -> f32[B, F]:
        return jnp.tanh(x @ self.w + self.b) * self.scale


@expex.struct
class Model[D, F]:
    up: Block[D, F]
    blocks: list[Block[F, F]]
    head: f32[F]

    def __call__[B](self, x: f32[B, D]) -> f32[B]:
        x = self.up(x)
        for block in self.blocks:
            x = block(x)
        return x @ self.head


def make(d=8, f=16, depth=3):
    keys = jr.split(jr.key(0), depth + 2)
    blk = lambda k, i, o: Block(jr.normal(k, (i, o)) / i**0.5, jnp.zeros(o))
    return Model(blk(keys[0], d, f), [blk(k, f, f) for k in keys[2:]], jr.normal(keys[1], (f,)))


@pytest.fixture
def mesh():
    return jax.make_mesh((2, 4), ("data", "model"), axis_types=(AxisType.Auto,) * 2)


@needs8
def test_struct_with_list_of_layers(mesh):
    model = device_put(make(), mesh, B="data", F="model")
    assert model.up.w.sharding.spec == P(None, "model")
    assert model.up.b.sharding.spec == P("model")
    assert [b.w.sharding.spec for b in model.blocks] == [P("model")] * 3  # f32[F, F]: once
    assert model.head.sharding.spec == P("model")
    assert model.blocks[0].scale == 1.0  # static fields untouched

    s = shardings(make(), mesh, D="data", F=("model",))
    assert isinstance(s, Model) and len(s.blocks) == 3
    assert s.up.w == NamedSharding(mesh, P("data", "model"))


@needs8
def test_abstract_values_and_unmapped_names(mesh):
    abstract = jax.eval_shape(make)
    s = shardings(abstract, mesh, D="model")
    assert s.up.w.spec == P("model") and s.blocks[1].w.spec == P() and s.head.spec == P()


@needs8
def test_plain_trees_annotations_and_variadics(mesh):
    tree = {"m": make(), "x": jnp.ones((4, 8)), "n": 3}
    s = shardings(tree, mesh, F="model")
    assert s["m"].head.spec == P("model") and s["x"].spec == P()  # unannotated: replicated

    def stacked[*Lead, D](x: f32[*Lead, D]) -> None: ...

    spec = stacked.__annotations__["x"]
    s = shardings(jnp.ones((4, 6, 8)), mesh, annotation=spec, D="model", Lead="data")
    assert s.spec == P(None, None, "model")  # axes of a variadic are replicated


@needs8
def test_bare_annotation(mesh):
    def f[B, D](x: f32[B, D]) -> None: ...

    s = shardings(f.__annotations__["x"], mesh, B="data", D=("model",))
    assert s == NamedSharding(mesh, P("data", "model"))


@needs8
def test_sharded_function_matches_unsharded(mesh):
    model = make()
    x = jr.normal(jr.key(1), (16, 8))
    names = {"B": "data", "F": "model"}
    call = shard(Model.__call__, mesh, **names)
    y = call(model, x)
    assert y.sharding.spec == P("data")  # follows `-> f32[B]`
    assert jnp.allclose(y, model(x), atol=1e-5)

    @expex.fn
    def layer[B, D, F](x: f32[B, D], w: f32[D, F], n: int, lr: float) -> f32[B, F]:
        for _ in range(n):
            x = x * lr
        return x @ w

    g = shard(layer, mesh, **names)
    w = jr.normal(jr.key(2), (8, 16))
    out = g(x, w, 2, lr=0.5)  # n static, lr traced
    assert out.sharding.spec == P("data", "model")
    assert jnp.allclose(out, layer(x, w, 2, 0.5), atol=1e-5)
    assert g(x, w=w, n=3, lr=0.1).shape == (16, 16)


@needs8
def test_bound_methods_and_random_functions(mesh):
    model, x = make(), jr.normal(jr.key(1), (16, 8))
    y = shard(model.__call__, mesh, B="data", F="model")(x)
    assert y.sharding.spec == P("data") and jnp.allclose(y, model(x), atol=1e-5)

    @expex.random
    @expex.fn
    def dropout[B, D](x: f32[B, D]) -> f32[B, D]:
        return jnp.where(jr.bernoulli(expex.key(), 0.5, x.shape), x, 0.0)

    g = shard(dropout, mesh, B="data", D="model")
    assert g(x).sharding.spec == P("data", "model")
    assert not jnp.array_equal(g(x), g(x))  # a fresh key per call, as expex.jit
    assert jnp.array_equal(g(x, key=jr.key(0)), dropout(x, key=jr.key(0)))


@needs8
def test_divisibility_error_names_array_and_axis(mesh):
    with pytest.raises(ValueError) as e:
        shardings(make(f=6), mesh, F="model")
    msg = str(e.value)
    assert "Model.up.w" in msg and "F = 6 on axis 1" in msg and "'model'" in msg, msg

    g = shard(Block.__call__, mesh, B="data")
    with pytest.raises(ValueError, match=r"x: f32\[B, D\] has B = 3 on axis 0"):
        g(make().up, jnp.ones((3, 8)))


@needs8
def test_unknown_mesh_axis(mesh):
    with pytest.raises(ValueError, match="the mesh has no axis 'batch'"):
        shardings(make(), mesh, B="batch")
    with pytest.raises(TypeError, match="jax.eval_shape"):
        shardings(Model, mesh)

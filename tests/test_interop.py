"""expex with Equinox, Flax (NNX and linen), Optax and Orbax (docs/interop.md).

The first tests need none of those libraries: they pin the general changes that
make them work (classes that are not dataclasses, hand-written ``__init__``,
bound methods as pytrees, ``expex.mask``). The others skip without their library.
"""

import dataclasses
from collections.abc import MutableSequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import Batch, f32


# --------------------------------------------------------------------------- without the libraries


@dataclasses.dataclass(frozen=True)
class Affine[I, O]:
    """A dataclass whose ``__init__`` does not take its fields, as Equinox modules have."""

    w: f32[I, O]
    b: f32[O]

    def __init__(self, key: Any, i: int, o: int):
        object.__setattr__(self, "w", jr.normal(key, (i, o)))
        object.__setattr__(self, "b", jnp.zeros(o))

    @expex.fn
    def __call__(self, x: f32[I]) -> f32[O]:
        return x @ self.w + self.b


def test_handwritten_init_is_not_called_by_the_tools():
    m = Affine(jr.key(0), 3, 5)
    assert m(jnp.ones((2, 3))).shape == (2, 5)
    assert expex.prove(Affine.__call__).ok  # abstract instances, made without __init__
    assert str(expex.cost(Affine.__call__).flops) == "2·I·O + O"
    assert expex.summary(m).total.params == 20
    with pytest.raises(expex.ShapeError, match=r"I is 3 \(from `self.w: f32\[3, 5\]`\)"):
        m(jnp.ones(4))


class Plain[D]:
    """Not a dataclass (as a Flax NNX module is not): its annotated attributes are its fields."""

    w: f32[D, D]
    heads: int = 2  # the prover's value, for Plain.split: `self` is made without __init__

    def __init__(self, d: int, heads: int = 2):
        self.w, self.heads, self.note = jnp.eye(d), heads, "not annotated: not checked"

    @expex.fn
    def __call__(self, x: f32[D]) -> f32[D]:
        return self.w @ x

    @expex.fn
    def split[T](self, x: f32[T, D]) -> f32[self.heads, T, D // self.heads]:
        assert D % self.heads == 0
        return x.reshape(T, self.heads, D // self.heads).transpose(1, 0, 2)


class Layers(MutableSequence):
    """A list-like container that is not a list, as Flax's ``nnx.List`` is."""

    def __init__(self, items: Any):
        self.items = list(items)

    def __getitem__(self, i: Any) -> Any:
        return self.items[i]

    def __setitem__(self, i: Any, v: Any) -> None:
        self.items[i] = v

    def __delitem__(self, i: Any) -> None:
        del self.items[i]

    def __len__(self) -> int:
        return len(self.items)

    def insert(self, i: int, v: Any) -> None:
        self.items.insert(i, v)


class PlainStack[D]:
    blocks: list[Plain[D]]

    def __init__(self, d: int, n: int):
        self.blocks = Layers(Plain(d) for _ in range(n))

    @expex.fn
    def __call__(self, x: f32[D]) -> f32[D]:
        for b in self.blocks:
            x = b(x)
        return x


def test_plain_class_attributes_are_checked():
    m = Plain(4)
    assert m(jnp.ones((3, 4))).shape == (3, 4)  # mapped over; self passed whole
    with pytest.raises(expex.ShapeError, match=r"D is 4 \(from `self.w: f32\[4, 4\]`\)"):
        m(jnp.ones(5))
    assert m.split(jnp.ones((5, 4))).shape == (2, 5, 2)
    with pytest.raises(expex.ShapeError, match=r"`D % heads == 0` is violated \(D=4, heads=3\)"):
        Plain(4, heads=3).split(jnp.ones((5, 4)))
    assert expex.prove(Plain.__call__).ok and expex.prove(Plain.split).ok and expex.prove(m.split).ok
    s = PlainStack(4, 3)
    assert s(jnp.ones(4)).shape == (4,)
    with pytest.raises(expex.ShapeError, match=r"self.blocks\[1\].w"):
        s.blocks[1] = Plain(5)
        s(jnp.ones(4))


def test_plain_class_without_array_annotations_is_unchanged():
    class Config:
        lr: float = 0.1

        @expex.fn
        def scale[D](self, x: f32[D]) -> f32[D]:
            return x * self.lr

    assert Config().scale(jnp.ones(3)).shape == (3,)


def test_bound_method_is_a_pytree():
    m = Layer(jr.key(0), D=3)
    assert [a.shape for a in jax.tree.leaves(m.__call__)] == [(3, 3), (3,), (3,)]
    traces = []

    @jax.jit
    def run(f, x):
        traces.append(1)
        return f(x)

    run(m.__call__, jnp.ones(3))
    run(Layer(jr.key(1), D=3).__call__, jnp.ones(3))  # another model: the same program
    assert len(traces) == 1
    g = jax.grad(lambda f: f(jnp.ones(3)).sum())(m.__call__)
    assert type(g.obj) is Layer and g.obj.w.shape == (3, 3)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WithTable[D]:
    w: f32[D, D]
    table: jax.Array  # an array no expex annotation describes (a layer from another library, say)

    @expex.fn
    def __call__(self, x: f32[D]) -> f32[D]:
        return self.w @ x + self.table


def test_unannotated_arrays_in_tools(tmp_path):
    m = WithTable(jnp.eye(4), jnp.ones(4))
    c = expex.cost(m.__call__)  # counted at the instance's sizes: the table's are not symbolic
    assert c.flops.value is not None and c.parameters.value == 20
    s = expex.summary(m)
    assert s.total.params == 16 and "4 parameters in parts without array annotations are not counted" in s.notes
    with pytest.raises(TypeError, match=r"`table` \(.*\) holds arrays that no annotation describes") as e:
        expex.save(m, tmp_path / "m.npz")
    assert e.value.code == "EX501"


@expex.struct
class Layer[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0
    scale: f32[D] = 1.0

    def __call__(self, x: f32[D]) -> f32[D]:
        return x + jax.nn.gelu(x @ self.w + self.b) * self.scale


@expex.struct
class Net[D]:
    layers: list[Layer[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for layer in self.layers:
            x = layer(x)
        return x


@expex.struct
class Deep[D, L]:
    layers: Batch[Layer[D], L]


def test_mask():
    net = Net(jr.key(0), D=4, layers=3)
    seen = []
    m = expex.mask(net, lambda path, spec: seen.append((path, str(spec), spec.ndim)) or spec.ndim >= 2)
    assert type(m) is Net and m.layers[1] == Layer(True, False, False)
    assert seen[:3] == [("layers[0].w", "f32[D, D]", 2), ("layers[0].b", "f32[D]", 1), ("layers[0].scale", "f32[D]", 1)]
    ensemble = jax.vmap(lambda k: Net(k, D=4, layers=2))(jr.split(jr.key(1), 3))  # axes no annotation declares
    assert expex.mask(ensemble, lambda p, s: s.ndim >= 2).layers[0] == Layer(True, False, False)
    deep = Deep(jr.key(0), D=4, L=2)  # Batch[...] declares the layer axis
    assert expex.mask(deep, lambda p, s: s.ndim >= 3).layers == Layer(True, False, False)
    tree = (net, {"count": jnp.zeros(()), "table": WithTable(jnp.eye(4), jnp.ones(4))})
    out = expex.mask(tree, lambda p, s: True)
    assert jax.tree.structure(out) == jax.tree.structure(tree)
    assert out[1]["count"] is False and out[1]["table"] == WithTable(True, False)


def test_ndim():
    from expex.annotations.spec import parse, parse_owner

    assert [parse(a).ndim for a in (f32, f32[3], f32[3, 4])] == [0, 1, 2]
    assert [s.ndim for s in parse_owner(Layer).fields.values()] == [2, 1, 1]


# --------------------------------------------------------------------------- Equinox


def test_equinox_methods():
    eqx = pytest.importorskip("equinox")

    class Linear[I, O](eqx.Module):
        w: f32[I, O]
        b: f32[O]

        def __init__(self, key: Any, i: int, o: int):
            self.w, self.b = jr.normal(key, (i, o)), jnp.zeros(o)

        @expex.fn
        def __call__(self, x: f32[I]) -> f32[O]:
            return x @ self.w + self.b

    class MLP[D, F](eqx.Module):
        up: Linear[D, F]
        down: Linear[F, D]
        heads: int = eqx.field(static=True, default=2)

        def __init__(self, key: Any, d: int, f: int):
            k1, k2 = jr.split(key)
            self.up, self.down = Linear(k1, d, f), Linear(k2, f, d)

        @expex.fn
        def __call__(self, x: f32[D]) -> f32[D]:
            return x + self.down(jax.nn.gelu(self.up(x)))

        @expex.fn
        def split(self, x: f32[D]) -> f32[self.heads, D // self.heads]:
            assert D % self.heads == 0
            return x.reshape(self.heads, D // self.heads)

    m = MLP(jr.key(0), 4, 8)
    assert m(jnp.ones((3, 4))).shape == (3, 4)
    with pytest.raises(expex.ShapeError, match=r"D is 4 \(from `self.up.w: f32\[4, 8\]`\)"):
        m(jnp.ones(5))
    assert m.split(jnp.ones(4)).shape == (2, 2)
    traces = []

    @eqx.filter_jit
    def call(f, x):
        traces.append(1)
        return f(x)

    call(m.__call__, jnp.ones(4))
    call(MLP(jr.key(1), 4, 8).__call__, jnp.ones(4))
    assert len(traces) == 1  # the weights are traced, not frozen into the program
    g = eqx.filter_grad(lambda f: f(jnp.ones(4)).sum())(m.__call__)
    assert g.obj.up.w.shape == (4, 8) and bool((g.obj.up.w != 0).any())
    ms = eqx.filter_vmap(lambda k: MLP(k, 4, 8))(jr.split(jr.key(0), 3))
    assert eqx.filter_vmap(lambda m, x: m(x))(ms, jnp.ones((3, 4))).shape == (3, 4)
    assert expex.prove(MLP.__call__).ok and expex.prove(MLP.split).ok and expex.prove(Linear.__call__).ok
    assert str(expex.cost(MLP.__call__).parameters) == "2·D·F + D + F"
    assert expex.summary(m).total.params == 76
    assert "Linear.__call__" in str(expex.diagram(MLP.__call__, D=4, F=8))


def test_equinox_and_structs_nested(tmp_path):
    eqx = pytest.importorskip("equinox")

    @expex.struct
    class Norm[D]:
        scale: f32[D] = 1.0

        def __call__(self, x: f32[D]) -> f32[D]:
            return (x - x.mean()) / (x.std() + 1e-5) * self.scale

    class Outer[D](eqx.Module):  # a struct in an Equinox module
        norm: Norm[D]
        w: f32[D, D]

        def __init__(self, key: Any, d: int):
            self.norm, self.w = Norm(key, D=d), jnp.eye(d)

        @expex.fn
        def __call__(self, x: f32[D]) -> f32[D]:
            return self.norm(x) @ self.w

    @expex.struct
    class Head[D, V]:  # an Equinox layer in a struct
        norm: Norm[D]
        out: eqx.nn.Linear

        def __call__(self, x: f32[D]) -> f32[V]:
            return self.out(self.norm(x))

    o = Outer(jr.key(0), 4)
    assert o(jnp.ones((2, 4))).shape == (2, 4) and expex.prove(Outer.__call__).ok
    expex.save(o, tmp_path / "o.npz")
    back = expex.load(Outer, tmp_path / "o.npz")  # built from the annotations, without __init__
    assert type(back) is Outer and jnp.allclose(back(jnp.arange(4.0)), o(jnp.arange(4.0)))
    h = Head(jr.key(0), D=4, out=eqx.nn.Linear(4, 6, key=jr.key(1)))
    assert h(jnp.ones((2, 4))).shape == (2, 6)
    assert eqx.filter_jit(lambda h, x: h(x))(h, jnp.ones(4)).shape == (6,)
    assert eqx.filter_grad(lambda h: h(jnp.ones(4)).sum())(h).out.weight.shape == (6, 4)
    assert expex.cost(h.__call__).parameters.value == 4 + 30
    assert "30 parameters in parts without array annotations are not counted" in expex.summary(h).notes
    with pytest.raises(TypeError, match="EX501|holds arrays that no annotation describes"):
        expex.save(h, tmp_path / "h.npz")


# --------------------------------------------------------------------------- Flax


def test_nnx_modules(tmp_path):
    nnx = pytest.importorskip("flax.nnx")
    optax = pytest.importorskip("optax")

    class Dense[I, O](nnx.Module):
        w: f32[I, O]
        b: f32[O]

        def __init__(self, i: int, o: int, rngs: Any):
            self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (i, o)))
            self.b = nnx.Param(jnp.zeros(o))

        @expex.fn
        def __call__(self, x: f32[I]) -> f32[O]:
            return x @ self.w + self.b

    class Block[D, F](nnx.Module):
        up: Dense[D, F]
        down: Dense[F, D]

        def __init__(self, d: int, f: int, rngs: Any):
            self.up, self.down = Dense(d, f, rngs), Dense(f, d, rngs)

        @expex.fn
        def __call__(self, x: f32[D]) -> f32[D]:
            return x + self.down(jax.nn.gelu(self.up(x)))

    class Model[D, F](nnx.Module):
        blocks: list[Block[D, F]]
        heads: int = 2  # a class default: the prover makes `self` without __init__

        def __init__(self, d: int, f: int, n: int, rngs: Any):
            self.blocks = nnx.List([Block(d, f, rngs) for _ in range(n)])
            self.heads = 2

        @expex.fn
        def __call__[T](self, x: f32[T, D]) -> f32[self.heads, T, D // self.heads]:
            assert D % self.heads == 0
            for b in self.blocks:
                x = b(x)
            return x.reshape(T, self.heads, D // self.heads).transpose(1, 0, 2)

    m = Model(8, 16, 2, nnx.Rngs(0))
    assert m(jnp.ones((5, 8))).shape == (2, 5, 4)
    with pytest.raises(expex.ShapeError, match=r"D is 8 \(from `self.blocks\[0\].up.w: f32\[8, 16\]`\)"):
        m(jnp.ones((5, 6)))
    for f in (Dense.__call__, Block.__call__, Model.__call__, m.__call__):
        assert expex.prove(f).ok, f
    opt = nnx.Optimizer(m, optax.adam(1e-2), wrt=nnx.Param)

    @nnx.jit
    def step(m: Any, opt: Any, x: Any) -> Any:
        loss, g = nnx.value_and_grad(lambda m: (m(x) ** 2).mean())(m)
        opt.update(m, g)
        return loss

    losses = [float(step(m, opt, jnp.ones((5, 8)))) for _ in range(3)]
    assert losses[2] < losses[0]
    expex.save(m, tmp_path / "m.npz")  # an NNX module: every array, by its path
    template = nnx.eval_shape(lambda: Model(8, 16, 2, nnx.Rngs(1)))
    back = expex.load(template, tmp_path / "m.npz")
    assert type(back.blocks[0].up.w) is nnx.Param
    assert jnp.allclose(back(jnp.ones((5, 8))), m(jnp.ones((5, 8))))


def test_nnx_state_in_a_call():
    nnx = pytest.importorskip("flax.nnx")
    from flax.errors import TraceContextError

    class Count(nnx.Variable):
        pass

    class Noisy[D](nnx.Module):
        w: f32[D, D]

        def __init__(self, d: int, rngs: Any):
            self.w = nnx.Param(jnp.eye(d))
            self.drop = nnx.Dropout(0.5, rngs=rngs)
            self.seen = Count(jnp.array(0))

        @expex.fn(vectorize=False)
        def batch[B](self, x: f32[B, D]) -> f32[B, D]:
            self.seen[...] += B
            return self.drop(x @ self.w)

        @expex.fn
        def one(self, x: f32[D]) -> f32[D]:
            self.seen[...] += 1
            return x @ self.w

    m = Noisy(4, nnx.Rngs(0))
    assert m.batch(jnp.ones((3, 4))).shape == (3, 4) and int(m.seen[...]) == 3
    assert m.one(jnp.ones(4)).shape == (4,) and int(m.seen[...]) == 4
    with pytest.raises(TraceContextError):  # mapped over the batch: under vmap, NNX refuses the update
        m.one(jnp.ones((3, 4)))


def test_linen_functions():
    nn = pytest.importorskip("flax.linen")

    @expex.fn
    def mix[T, S, D](q: f32[T, D], k: f32[S, D], v: f32[S, D]) -> f32[T, D]:
        return jax.nn.softmax(q @ k.T / jnp.sqrt(D), axis=-1) @ v

    class SelfAttention(nn.Module):
        features: int
        bug: bool = False

        @nn.compact
        def __call__(self, x: Any) -> Any:
            q, k, v = (nn.Dense(self.features)(x) for _ in range(3))
            return mix(q, k, v[..., :-1, :] if self.bug else v)

    x = jnp.ones((2, 5, 8))
    params = SelfAttention(8).init(jr.key(0), x)
    assert jax.jit(SelfAttention(8).apply)(params, x).shape == (2, 5, 8)
    assert jax.grad(lambda p: SelfAttention(8).apply(p, x).sum())(params)["params"]["Dense_0"]["kernel"].shape == (8, 8)
    with pytest.raises(expex.ShapeError, match=r"S is 5 \(from `k: f32\[5, 8\]`\), but axis 0 of `v` is 4"):
        SelfAttention(8, bug=True).init(jr.key(0), x)
    assert expex.prove(mix).ok


# --------------------------------------------------------------------------- Optax and Orbax


def test_optax_masks_and_freezing():
    optax = pytest.importorskip("optax")
    net = Net(jr.key(0), D=4, layers=3)
    x = jnp.ones((2, 4))

    def step(tx: Any, net: Any) -> Any:
        g = jax.grad(lambda m: (m(x) ** 2).mean())(net)
        u, _ = tx.update(g, tx.init(net), net)
        return optax.apply_updates(net, u)

    tx = optax.adamw(1e-2, weight_decay=1.0, mask=expex.mask(net, lambda p, s: s.ndim >= 2))
    assert type(step(tx, net)) is Net

    def labels(params: Any) -> Any:
        last = expex.mask(params, lambda path, spec: path.startswith("layers[2]"))
        return jax.tree.map(lambda t: "train" if t else "freeze", last)

    tx = optax.multi_transform({"train": optax.adam(1e-2), "freeze": optax.set_to_zero()}, labels)
    new = step(tx, net)
    changed = [bool((a != b).any()) for a, b in zip(jax.tree.leaves(net), jax.tree.leaves(new))]
    assert changed == [False] * 6 + [True] * 3


def test_orbax_checkpoints(tmp_path):
    ocp = pytest.importorskip("orbax.checkpoint")
    optax = pytest.importorskip("optax")
    from jax.sharding import AxisType

    tx = optax.adamw(1e-3)

    def fresh() -> Any:
        net = Net(jr.key(0), D=4, layers=2)
        return {"net": net, "opt": tx.init(net)}

    state = fresh()
    state["net"] = state["net"].at.layers[1].w.multiply(3.0)
    ckpt = ocp.StandardCheckpointer()
    ckpt.save(tmp_path / "1", state)
    ckpt.wait_until_finished()
    abstract = jax.eval_shape(fresh)
    back = ckpt.restore(tmp_path / "1", abstract)
    assert type(back["net"]) is Net and type(back["opt"][0].mu) is Net
    assert jnp.array_equal(back["net"].layers[1].w, state["net"].layers[1].w)
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
    placed = expex.shardings(abstract, mesh, D="model")
    target = jax.tree.map(lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), abstract, placed)
    back = ckpt.restore(tmp_path / "1", target)
    assert back["opt"][0].mu.layers[0].w.sharding == placed["net"].layers[0].w
    wrong = jax.eval_shape(lambda: Net(jr.key(0), D=8, layers=2))
    with pytest.raises(ValueError, match="not compatible with the stored shape"):
        ckpt.restore(tmp_path / "1", {"net": wrong, "opt": abstract["opt"]})

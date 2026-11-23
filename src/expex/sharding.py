"""Sharding by axis names: map sizes to mesh axes once, get every sharding.

::

    mesh = jax.make_mesh((2, 4), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
    names = dict(B="data", D="model")

    model = expex.device_put(model, mesh, **names)     # w: f32[D, F] -> P("model")
    step = expex.shard(train_step, mesh, **names)      # inputs placed, result constrained

Every array's axes are named by its annotation, so a mapping from size names
(``B``, ``D``) to mesh axes (``"data"``, ``"model"``) decides each array's
``PartitionSpec``: an axis whose size is mapped is split over that mesh axis,
every other axis is replicated. Values in the mapping are a mesh axis name, a
tuple of names (split over their product), or ``None``. Intermediates inside a
function need nothing: XLA propagates from the inputs and the result, on mesh
axes of type ``Auto`` (``jax.make_mesh`` makes ``Explicit`` ones by default,
where JAX asks for an ``out_sharding`` at contractions over a split axis).
"""

import functools
import math
from collections.abc import Callable
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .annotations.dims import DConst, DName
from .function import BoundFn, Fn, fn, unwrap
from .annotations.spec import ArraySpec, KeySpec, Spec, StaticSpec, _struct_fields, parse, parse_owner
from .annotations.types import ArrayAnn, DType


def shardings(tree: Any, mesh: Mesh, /, *, annotation: Any = None, **names: Any) -> Any:
    """The ``NamedSharding`` of every array in ``tree``, read from its annotations.

    ``tree`` is a value: a ``@expex.struct`` / annotated dataclass, or any pytree
    of them, with arrays or ``jax.ShapeDtypeStruct`` leaves (so
    ``jax.eval_shape(Model, ...)`` gives shardings before any memory is used).
    The result has the structure of ``tree``. ``annotation=`` gives the spec of
    a tree that carries none (``dict[str, f32[B, D]]``). A bare array
    annotation (``f32[B, D]``) gives its one sharding, unchecked. Arrays not
    covered by an annotation are replicated.

    Raises ``ValueError`` when a mapped size is not divisible by its devices.
    """
    names = _check_names(mesh, names)
    if isinstance(tree, (ArrayAnn, DType)):
        spec: Any = parse(tree)
        if spec.var_index is not None:
            raise TypeError(f"{spec}: a variadic shape needs a value to know its rank")
        return NamedSharding(mesh, _partition(spec, None, names, mesh, "annotation"))
    if isinstance(tree, type):
        raise TypeError(
            f"expex.shardings needs a value, not the class {tree.__name__} (lists of layers have "
            f"no length yet). Pass jax.eval_shape(lambda: {tree.__name__}(...)) instead."
        )
    return _tree(tree, mesh, names, annotation, None)


def device_put(tree: Any, mesh: Mesh, /, *, annotation: Any = None, **names: Any) -> Any:
    """``jax.device_put(tree, shardings(tree, mesh, **names))``."""
    return jax.device_put(tree, shardings(tree, mesh, annotation=annotation, **names))


def shard(f: Callable[..., Any], mesh: Mesh, /, **names: Any) -> Callable[..., Any]:
    """``f`` compiled for ``mesh``, with arguments and result sharded by their annotations.

    Each call places the array arguments as their annotations say (a no-op if
    they already are) and runs ``jax.jit(f)``, which constrains the result to
    the return annotation's shardings. Static parameters are those of
    :func:`expex.jit <expex.jit>`; parameters with no array annotation keep their
    placement. A bound method (``model.__call__``) is compiled with the
    instance as an argument; a plain function is wrapped with ``@expex.fn`` first.
    """
    names = _check_names(mesh, names)
    g = unwrap(f)
    if isinstance(g, BoundFn):
        return functools.partial(shard(g.fn, mesh, **names), g.obj)
    g = g if isinstance(g, Fn) else fn(f)
    static = tuple(
        n for n, p in g.params.items()
        if isinstance(p.spec, StaticSpec) and (p.spec.explicit or p.spec.inner not in (float, complex))
    )  # fmt: skip

    def body(*args: Any, **kwargs: Any) -> Any:
        out = g(*args, **kwargs)
        put = lambda s, x, path: jax.lax.with_sharding_constraint(x, _leaf(s, x, path, mesh, names))
        return g.ret.walk(out, "return", put)

    compiled = jax.jit(functools.update_wrapper(body, g), static_argnames=static)

    @functools.wraps(g)
    def call(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        extra = {}
        if g.random and "key" not in g.params:  # as expex.jit: a fresh key per call
            from .keys import fresh

            extra["key"] = kwargs.pop("key") if "key" in kwargs else fresh(f"{g.name}()")
        bound = g._sig.bind(*args, **kwargs)
        dynamic = {k: v for k, v in bound.arguments.items() if k not in static}
        place = lambda s, x, path: jax.device_put(x, _leaf(s, x, path, mesh, names))
        bound.arguments.update(g._walk(dynamic, place))
        return compiled(*bound.args, **bound.kwargs, **extra)

    return call


def _check_names(mesh: Mesh, names: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Normalise ``B="data"`` / ``B=("data", "fsdp")`` / ``B=None`` to tuples of mesh axes."""
    out: dict[str, tuple[str, ...]] = {}
    for size, axes in names.items():
        axes = () if axes is None else (axes,) if isinstance(axes, str) else tuple(axes)
        for a in axes:
            if a not in mesh.axis_names:
                raise ValueError(
                    f"{size}={a!r}: the mesh has no axis {a!r} (its axes are "
                    f"{', '.join(map(repr, mesh.axis_names))})."
                )
        out[size] = axes
    return out


def _tree(tree: Any, mesh: Mesh, names: dict[str, Any], annotation: Any, path: str | None) -> Any:
    """Shardings for ``tree``: by its annotation, its own struct annotations, or replicated."""
    spec: Spec | None = None
    if annotation is not None:
        spec = annotation if isinstance(annotation, Spec) else parse(annotation)
    elif _struct_fields(type(tree)) is not None:
        spec = parse_owner(type(tree))
        if not spec.leafy:
            spec = None
    if spec is not None:
        visit = lambda s, x, p: _leaf(s, x, p, mesh, names)
        tree = spec.walk(tree, path or type(tree).__name__, visit)
    else:
        is_struct = lambda x: x is not tree and _struct_fields(type(x)) is not None
        sub = lambda kp, x: _tree(x, mesh, names, None, (path or "tree") + jax.tree_util.keystr(kp))
        tree = jax.tree_util.tree_map_with_path(
            lambda kp, x: sub(kp, x) if is_struct(x) else x, tree, is_leaf=is_struct
        )
    # Leaves no annotation reached (unannotated fields, plain containers): replicated.
    repl = NamedSharding(mesh, PartitionSpec())
    return jax.tree.map(lambda x: x if isinstance(x, NamedSharding) else repl, tree)


def _leaf(spec: Spec, x: Any, path: str, mesh: Mesh, names: dict[str, Any]) -> NamedSharding:
    if not isinstance(spec, ArraySpec) or isinstance(spec, KeySpec):
        return NamedSharding(mesh, PartitionSpec())
    return NamedSharding(mesh, _partition(spec, getattr(x, "shape", ()), names, mesh, path))


def _partition(
    spec: ArraySpec, shape: tuple[int, ...] | None, names: dict[str, Any], mesh: Mesh, path: str
) -> PartitionSpec:
    """One ``PartitionSpec``: per axis, the mesh axes its size name maps to.

    A mesh axis is used at most once per array: in ``f32[D, D]`` the first
    ``D`` is split and the second replicated. Axes of a variadic (``*Batch``),
    computed sizes (``T + 1``) and literals are replicated.
    """
    dims = list(spec.dims)
    if spec.var_index is not None and shape is not None:  # align to the value's rank
        i = spec.var_index
        dims = dims[:i] + [None] * (len(shape) - len(dims) + 1) + dims[i + 1 :]
    used: set[str] = set()
    out: list[Any] = []
    for axis, d in enumerate(dims):
        name = d.name if isinstance(d, DName) else d.label if isinstance(d, DConst) else None
        if name is not None and name not in names:
            name = name.rpartition(".")[2]  # `Block.D` of an unparametrised `Block` field
        axes = names.get(name, ()) if name is not None else ()
        if not axes or used & set(axes):
            out.append(None)
            continue
        used |= set(axes)
        if shape is not None:
            n = math.prod(mesh.shape[a] for a in axes)
            if shape[axis] % n:
                on = " x ".join(map(repr, axes))
                raise ValueError(
                    f"{path}: {spec} has {name} = {shape[axis]} on axis {axis}, which mesh "
                    f"axis {on} cannot split evenly ({n} devices do not divide {shape[axis]})."
                )
        out.append(axes[0] if len(axes) == 1 else axes)
    while out and out[-1] is None:
        out.pop()
    return PartitionSpec(*out)

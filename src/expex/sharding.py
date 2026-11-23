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

Values with no annotation of their own, such as an optax state, are searched
for structs: an Adam state's moments are models, sharded like the model.
"""

import functools
import inspect
import math
from collections.abc import Callable
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .annotations.dims import DConst, DName
from .annotations.errors import coded
from .annotations.spec import (
    ArraySpec, KeySpec, OpaqueSpec, Spec, TupleSpec, _struct_fields, parse, parse_owner,
)  # fmt: skip
from .annotations.types import ArrayAnn, DType
from .decorators import _AutoDonate, fn, static_argnames
from .function import BoundFn, Fn, unwrap


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
            raise coded("EX603", TypeError(f"{spec}: a variadic shape needs a value to know its rank"))
        return NamedSharding(mesh, _partition(spec, None, names, mesh, "annotation"))
    if isinstance(tree, type):
        raise coded("EX603", TypeError(
            f"expex.shardings needs a value, not the class {tree.__name__} (lists of layers have "
            f"no length yet). Pass jax.eval_shape(lambda: {tree.__name__}(...)) instead."
        ))
    return _tree(tree, mesh, names, annotation, None)


def device_put(tree: Any, mesh: Mesh, /, *, annotation: Any = None, **names: Any) -> Any:
    """``jax.device_put(tree, shardings(tree, mesh, **names))``."""
    return jax.device_put(tree, shardings(tree, mesh, annotation=annotation, **names))


def shard(
    f: Callable[..., Any], mesh: Mesh, /, *, donate: str | tuple[str, ...] = (), **names: Any
) -> Callable[..., Any]:
    """``f`` compiled for ``mesh``, with arguments and result sharded by their annotations.

    Each call places the array arguments as their annotations say (a no-op if
    they already are) and runs ``jax.jit(f)``, which constrains the result to
    the return annotation's shardings. Static parameters are those of
    :func:`expex.jit <expex.jit>`. Unannotated parameters and results (``Any``,
    an optax state) are searched for structs, which are sharded by their own
    annotations; their other arrays keep their placement. ``donate`` is as in
    ``@expex.fn(jit=True, donate=...)``: parameter names, or ``"auto"`` to donate
    what each call site reassigns (``state = step(state, batch)``). A bound
    method (``model.__call__``) is compiled with the instance as an argument; a
    plain function is wrapped with ``@expex.fn`` first.
    """
    names = _check_names(mesh, names)
    g = unwrap(f)
    if isinstance(g, BoundFn):
        if donate == "auto":  # the call site's positions would be off by the instance
            raise coded("EX801", TypeError(
                f"expex.shard(..., donate='auto') needs the unbound method: "
                f"shard({type(g.obj).__name__}.{g.fn.name}, ...), called with the instance first."
            ))
        return functools.partial(shard(g.fn, mesh, donate=donate, **names), g.obj)
    g = g if isinstance(g, Fn) else fn(f)
    static = tuple(static_argnames(g))

    def body(*args: Any, **kwargs: Any) -> Any:
        out = g(*args, **kwargs)
        put = lambda s, x, path: jax.lax.with_sharding_constraint(x, _leaf(s, x, path, mesh, names))
        return _walk(g.ret, out, "return", put)

    body = functools.update_wrapper(body, g)

    def variant(how: tuple[str, ...]) -> Callable[..., Any]:
        compiled = jax.jit(body, static_argnames=static, donate_argnames=how)

        @functools.wraps(g)
        def call(*args: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            extra = {}
            if g.random and "key" not in g.params:  # as expex.jit: a fresh key per call
                from .keys import fresh

                extra["key"] = kwargs.pop("key") if "key" in kwargs else fresh(f"{g.name}()")
            bound = g._sig.bind(*args, **kwargs)
            place = lambda s, x, path: jax.device_put(x, _leaf(s, x, path, mesh, names))
            for k, v in bound.arguments.items():
                if k not in static:
                    bound.arguments[k] = _walk_param(g, k, v, place)
            return compiled(*bound.args, **bound.kwargs, **extra)

        return call

    if donate == "auto":
        return _AutoShard(variant, g, static)
    return variant((donate,) if isinstance(donate, str) else tuple(donate))


class _AutoShard(_AutoDonate):
    """``shard(..., donate="auto")``: the call site decides what is donated, as ``expex.jit``'s."""

    def __init__(self, variant: Callable[[tuple[str, ...]], Any], g: Fn[Any, Any], static: tuple[str, ...]):
        self.variant = variant
        super().__init__(variant(()), g, static, {})

    def _variant(self, donate: tuple[str, ...]) -> Any:
        if donate not in self.compiled:
            self.compiled[donate] = self.variant(donate)
        return self.compiled[donate]


def _walk_param(g: Fn[Any, Any], name: str, value: Any, visit: Callable[..., Any]) -> Any:
    """:func:`_walk` for parameter ``name`` of ``g`` (``*args`` and ``**kwargs`` element by element)."""
    spec = g.params[name].spec
    kind = g._var_kinds.get(name)
    if kind is None:
        return _walk(spec, value, name, visit)
    if kind is inspect.Parameter.VAR_POSITIONAL:
        return tuple(_walk(spec, x, f"{name}[{i}]", visit) for i, x in enumerate(value))
    return {k: _walk(spec, x, k, visit) for k, x in value.items()}


def _walk(spec: Spec, value: Any, path: str, visit: Callable[..., Any]) -> Any:
    """``spec.walk(value, path, visit)``, also into unannotated parts (see :func:`_structs`)."""
    if isinstance(spec, OpaqueSpec):
        return _structs(value, path, visit)
    if isinstance(spec, TupleSpec) and type(value) is tuple and len(value) == len(spec.elems):
        return tuple(_walk(s, v, f"{path}[{i}]", visit) for i, (s, v) in enumerate(zip(spec.elems, value)))
    return spec.walk(value, path, visit)


def _structs(tree: Any, path: str | None, visit: Callable[..., Any]) -> Any:
    """``tree`` with each struct in it walked by its own annotations; other leaves as they are.

    The paths of a struct at the root start with its class name (``Model.up.w``),
    others with ``path`` and the pytree path (``opt_state[0].mu.up.w``).
    """
    is_struct = lambda x: _owner(type(x)) is not None

    def one(kp: Any, x: Any) -> Any:
        if not is_struct(x):
            return x
        where = (path or type(x).__name__) if not kp else (path or "tree") + jax.tree_util.keystr(kp)
        return _owner(type(x)).walk(x, where, visit)  # type: ignore[union-attr]

    return jax.tree_util.tree_map_with_path(one, tree, is_leaf=is_struct)


@functools.cache
def _owner(cls: type) -> Spec | None:
    """The spec of a struct class that annotates arrays (a model; not an optax NamedTuple), else None."""
    if _struct_fields(cls) is None:
        return None
    spec = parse_owner(cls)
    return spec if spec.leafy else None


def _check_names(mesh: Mesh, names: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Normalise ``B="data"`` / ``B=("data", "fsdp")`` / ``B=None`` to tuples of mesh axes."""
    out: dict[str, tuple[str, ...]] = {}
    for size, axes in names.items():
        axes = () if axes is None else (axes,) if isinstance(axes, str) else tuple(axes)
        for a in axes:
            if a not in mesh.axis_names:
                raise coded("EX601", ValueError(
                    f"{size}={a!r}: the mesh has no axis {a!r} (its axes are "
                    f"{', '.join(map(repr, mesh.axis_names))})."
                ))
        out[size] = axes
    return out


def _tree(tree: Any, mesh: Mesh, names: dict[str, Any], annotation: Any, path: str | None) -> Any:
    """Shardings for ``tree``: by its annotation, its own struct annotations, or replicated."""
    visit = lambda s, x, p: _leaf(s, x, p, mesh, names)
    if annotation is not None:
        spec = annotation if isinstance(annotation, Spec) else parse(annotation)
        tree = _walk(spec, tree, path or type(tree).__name__, visit)
    else:
        tree = _structs(tree, path, visit)
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
    computed sizes (``T + 1``) and literals are replicated, and so are extra
    leading axes that automatic batching maps over (``f32[D]`` given a
    ``[B, D]`` value is aligned to its last axis).
    """
    dims: list[Any] = list(spec.dims)
    if spec.var_index is not None and shape is not None:  # align to the value's rank
        i = spec.var_index
        dims = dims[:i] + [None] * (len(shape) - len(dims) + 1) + dims[i + 1 :]
    elif shape is not None and len(shape) > len(dims):
        dims = [None] * (len(shape) - len(dims)) + dims
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
                raise coded("EX602", ValueError(
                    f"{path}: {spec} has {name} = {shape[axis]} on axis {axis}, which mesh "
                    f"axis {on} cannot split evenly ({n} devices do not divide {shape[axis]})."
                ))
        out.append(axes[0] if len(axes) == 1 else axes)
    while out and out[-1] is None:
        out.pop()
    return PartitionSpec(*out)

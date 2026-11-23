"""``Module.init(key, D=64)``: build a struct's parameters from its annotations.

::

    @expex.struct
    class Linear[I, O]:
        w: f32[O, I] = jax.nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
        b: f32[O] = 0.0

    Linear.init(key, I=3, O=4)

An array field's default says how to make it: a JAX initializer (any function
of ``(key, shape, dtype)``) or a number to fill it with. Its shape comes from
the annotation, with the sizes given to ``init`` (and the int fields of the
struct). Nested structs are built the same way; a list of them needs its
length (``blocks=4``). Any field can be passed instead of built.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ._dims import DConst, DExpr, DName, Env, describe, resolve
from ._errors import Mismatch, ShapeError
from ._spec import ArraySpec, KeySpec, NoneSpec, SeqSpec, Spec, StructSpec, UnionSpec, parse


class Initializer:
    """How to make an array field: the default it was given in the class."""

    __slots__ = ("make",)

    def __init__(self, make: Any):
        self.make = make

    def __call__(self, key: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        if callable(self.make):
            return self.make(key, shape, dtype)
        return jnp.full(shape, self.make, dtype)

    def __repr__(self) -> str:
        return f"<initializer {getattr(self.make, '__name__', self.make)!r}>"


class _Required:
    """Default of a field without one that comes after a field with one."""

    def __repr__(self) -> str:
        return "<required>"


REQUIRED = _Required()


def mark(cls: Any, anns: dict[str, Any]) -> None:
    """Turn the defaults of array fields into :class:`Initializer`s (before ``dataclass``).

    Fields keep their order, and a dataclass field without a default may not
    follow one with a default, so such fields get :data:`REQUIRED`, which the
    constructor refuses to leave in place, like an unbuilt initializer.
    """
    from ._sym import texts

    own = list(texts(cls))  # names in order; the values may not evaluate (f32[2 * E, D])
    seen_default = False
    for name in own:
        default = vars(cls).get(name, dataclasses.MISSING)
        if default is dataclasses.MISSING:
            if seen_default:
                setattr(cls, name, REQUIRED)
            continue
        seen_default = True
        if default is dataclasses.MISSING or isinstance(default, (Initializer, jax.Array, np.ndarray)):
            continue
        if not (callable(default) or (isinstance(default, (int, float)) and not isinstance(default, bool))):
            continue
        try:
            spec = parse(anns.get(name))
        except Exception:
            continue
        if isinstance(spec, (ArraySpec, KeySpec)):
            setattr(cls, name, Initializer(default))


def unbuilt(cls: Any, obj: Any) -> None:
    """Refuse an object some of whose arrays were never made (constructed without them)."""
    left = [
        f.name for f in dataclasses.fields(cls)
        if isinstance(getattr(obj, f.name, None), (Initializer, _Required))
    ]  # fmt: skip
    if left:
        raise TypeError(
            f"{cls.__name__}(): {', '.join(left)} not given. Pass them, or build every "
            f"array from its annotation with {cls.__name__}.init(key, <sizes>)."
        )


def init(cls: Any, key: Any, **given: Any) -> Any:
    """Build ``cls`` from its annotations. See the module docstring."""
    from ._spec import parse_owner

    spec = parse_owner(cls)
    named: set[str] = set()
    spec.names(named)  # sizes in the annotations, Dim classes included
    sizes = {p.__name__ for p in getattr(cls, "__type_params__", ())}
    sizes |= {n for n in named if not n.startswith("*") and "." not in n}
    fields = _all_fields(spec)
    sizes -= fields  # `heads` in f32[self.heads, D] is a field: its value is the size
    unknown = set(given) - sizes - fields
    if unknown:
        raise TypeError(
            f"{cls.__name__}.init(): {', '.join(sorted(unknown))} is neither a size "
            f"({', '.join(sorted(sizes)) or 'none'}) nor a field."
        )
    env = Env()
    for n in sizes & set(given):
        env.dims[n] = int(given[n])
        env.origin[n] = f"{n}={given[n]} passed to {cls.__name__}.init"
    obj = _build(cls, spec, key, env, {k: v for k, v in given.items() if k not in sizes}, cls.__name__)
    try:  # the whole object against its annotations, sizes shared
        spec.match(obj, cls.__name__, env)
        if env.deferred:
            resolve(env)
    except Mismatch as m:
        raise ShapeError(
            f"{cls.__name__}.init(): {m.path} is {describe(m.value)}, but is declared "
            f"{m.spec}.\n\n    {m.reason}"
        ) from None
    return obj


def _build(cls: Any, spec: Any, key: Any, env: Env, given: dict[str, Any], path: str) -> Any:
    fields = dataclasses.fields(cls)
    fields_here = {f.name for f in fields}
    specs: dict[str, Spec] = getattr(spec, "fields", {})
    obj = object.__new__(cls)
    values: dict[str, Any] = {}
    # plain fields first: their int values may be sizes of the arrays (f32[heads, D])
    for f in fields:
        s = specs.get(f.name)
        if f.name in given and not (isinstance(s, SeqSpec) and isinstance(given[f.name], int)):
            values[f.name] = given[f.name]
            if s is not None and s.leafy:
                try:
                    s.match(given[f.name], f"{path}.{f.name}", env)  # binds sizes from it
                except Mismatch as m:
                    raise ShapeError(f"{path}.init(): {m.path}: {m.reason}") from None
        elif s is None or not s.leafy:
            values[f.name] = _plain(f, path)
        if f.name in values and isinstance(values[f.name], (int, np.integer)) and not isinstance(values[f.name], bool):
            env.dims.setdefault(f.name, int(values[f.name]))
    keys = jr.split(key, max(len(fields), 1))
    for k, f in zip(keys, fields):
        if f.name not in values:
            deeper = {n: v for n, v in given.items() if n not in fields_here}
            values[f.name] = _make(specs[f.name], f, k, env, given.get(f.name), f"{path}.{f.name}", deeper)
    for name, v in values.items():
        object.__setattr__(obj, name, v)
    return obj


def _all_fields(spec: Any) -> set[str]:
    """Field names of a struct and of every struct inside it."""
    out: set[str] = set()
    todo = [spec]
    while todo:
        s = todo.pop()
        if isinstance(s, SeqSpec):
            todo.append(s.elem)
        elif isinstance(s, StructSpec):
            out |= set(s.fields)
            todo += list(s.fields.values())
    return out


def _plain(f: Any, path: str) -> Any:
    if f.default is not dataclasses.MISSING and not isinstance(f.default, (Initializer, _Required)):
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    raise TypeError(f"{path}.{f.name} has no default: pass {f.name}=... to init.")


def _make(spec: Spec, f: Any, key: Any, env: Env, count: Any, path: str, deeper: dict[str, Any]) -> Any:
    """Build one field. ``deeper``: the settings given to init, for nested structs."""
    if isinstance(spec, UnionSpec):  # `X | None`: None unless there is a way to build X
        if f.default is None:
            return None
        spec = next(s for s in spec.options if not isinstance(s, NoneSpec))
    if isinstance(spec, (ArraySpec, KeySpec)):
        how = f.default if isinstance(f.default, Initializer) else None
        if how is None:
            raise TypeError(
                f"{path}: {spec} has no initializer. Give the field a default (a JAX "
                f"initializer such as jax.nn.initializers.normal(), or a number), or "
                f"pass it to init."
            )
        if isinstance(spec, KeySpec):
            return jr.split(key, _shape(spec.dims, env, path)) if spec.dims else key
        dtype = spec.dtype.dtype if spec.dtype.dtype is not None else spec.dtype.example
        return how(key, _shape(spec.dims, env, path), dtype)
    if isinstance(spec, StructSpec):
        return _build(spec.cls, spec, key, env, deeper, path)
    if isinstance(spec, SeqSpec) and isinstance(spec.elem, StructSpec):
        if not isinstance(count, int):
            raise TypeError(f"{path} is a list of {spec.elem}: pass its length, e.g. {f.name}=4, to init.")
        items = [
            _build(spec.elem.cls, spec.elem, k, env, deeper, f"{path}[{i}]")
            for i, k in enumerate(jr.split(key, count))
        ]  # fmt: skip
        return items if spec.kind is list else tuple(items)
    raise TypeError(f"{path}: cannot build a {spec} from its annotation; pass it to init.")


def _shape(dims: tuple[Any, ...], env: Env, path: str) -> tuple[int, ...]:
    out = []
    for d in dims:
        if isinstance(d, DConst):
            out.append(d.value)
        elif isinstance(d, DName) and d.name in env.dims:
            out.append(int(env.dims[d.name]))
        elif isinstance(d, DExpr) and d.expr.names <= env.dims.keys():
            out.append(int(d.expr(env.dims)))
        else:
            names = sorted({d.name} if isinstance(d, DName) else set(getattr(getattr(d, "expr", None), "names", ())) - env.dims.keys())
            raise TypeError(f"{path}: the size {', '.join(names) or d} is not known: pass it to init, e.g. {names[0] if names else 'D'}=64.")
    return tuple(out)

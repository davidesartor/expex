"""Specs: the parsed form of annotations, and how values are matched to them.

``parse`` turns an annotation (``f32[B, T]``, ``tuple[f32[B], Key]``, a
dataclass with annotated fields, ...) into a tree of ``Spec`` objects. A spec
can then

* ``match`` a value, binding dimension names in an :class:`Env`;
* ``walk`` a value, visiting (and optionally replacing) its array leaves;
* ``abstract`` itself into example inputs for the prover.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
import typing
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.core import Tracer

from ._errors import Mismatch
from ._expr import Expr
from ._types import SHORT_NAMES, ArrayAnn, BatchAnn, Dim, DType, Etc, KeyAnn, StaticAnn

# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------


class DName:
    """A named axis whose size is bound on first use: ``B``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name


class DConst:
    """A fixed size: ``Literal[3]`` or ``class RGB(Dim, size=3)``."""

    __slots__ = ("value", "label")

    def __init__(self, value: int, label: str | None = None):
        self.value = value
        self.label = label

    def __str__(self) -> str:
        return self.label or str(self.value)


class DAny:
    """One axis of any size (``typing.Any``)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "Any"


class DVar:
    """Zero or more named axes: ``*Batch``."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return "*" + self.name


class DEtc:
    """Zero or more unchecked axes (``Etc``)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "Etc"


class DExpr:
    """A size computed from other dimensions: ``T + 1``."""

    __slots__ = ("expr", "label")

    def __init__(self, expr: Expr, label: str | None = None):
        self.expr = expr
        self.label = label

    def __str__(self) -> str:
        return self.expr.src


_ANY = DAny()
_ETC = DEtc()


def parse_dim(item: Any, subst: dict[Any, Any]) -> Any:
    if item in subst if _hashable(item) else False:
        return subst[item]
    if isinstance(item, typing.TypeVar):
        return DName(item.__name__)
    if isinstance(item, typing.TypeVarTuple):
        return DVar(item.__name__)
    origin = typing.get_origin(item)
    if origin is typing.Unpack:
        (inner,) = typing.get_args(item)
        if inner in subst:
            return subst[inner]
        if isinstance(inner, typing.TypeVarTuple):
            return DVar(inner.__name__)
        raise TypeError(f"cannot unpack {inner!r} in a shape")
    if type(item) is int:
        return DConst(item)
    if origin is typing.Literal:
        args = typing.get_args(item)
        if len(args) == 1 and type(args[0]) is int:
            return DConst(args[0])
        raise TypeError(f"{item!r} is not a valid dimension; use Literal[<int>]")
    if isinstance(item, typing.TypeAliasType):
        d = parse_dim(item.__value__, subst)
        if isinstance(d, DConst) and d.label is None:
            d = DConst(d.value)
        return d
    if item is typing.Any:
        return _ANY
    if item is Etc or item is Ellipsis:
        return _ETC
    if isinstance(item, type) and issubclass(item, Dim):
        if item.size is not None:
            return DConst(item.size, item.__name__)
        return DName(item.__name__)
    if isinstance(item, str):
        e = Expr(item)
        if item.strip().isidentifier():
            return DName(item.strip())
        return DExpr(e)
    raise TypeError(
        f"{item!r} is not a valid dimension. Use a type parameter, Literal[n], "
        f"a Dim subclass, Any or Etc."
    )


def _hashable(x: Any) -> bool:
    try:
        hash(x)
    except TypeError:
        return False
    return True


def _parse_dims(items: tuple[Any, ...], subst: dict[Any, Any]) -> tuple[tuple[Any, ...], int | None]:
    if items == ((),):
        items = ()
    dims: list[Any] = []
    for it in items:
        d = parse_dim(it, subst)
        if isinstance(d, tuple):  # a substituted variadic
            dims.extend(d)
        else:
            dims.append(d)
    var = [i for i, d in enumerate(dims) if isinstance(d, (DVar, DEtc))]
    if len(var) > 1:
        raise TypeError("a shape can contain at most one variadic (*Name or Etc)")
    return tuple(dims), (var[0] if var else None)


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


class Env:
    """Dimension bindings for one call."""

    __slots__ = ("dims", "origin", "vars", "deferred")

    def __init__(self) -> None:
        self.dims: dict[str, Any] = {}
        self.origin: dict[str, str] = {}
        self.vars: dict[str, tuple[Any, ...]] = {}
        self.deferred: list[tuple[DExpr, Any, str, Any, Any, int]] = []

    def snapshot(self) -> tuple[Any, ...]:
        return dict(self.dims), dict(self.origin), dict(self.vars), len(self.deferred)

    def restore(self, snap: tuple[Any, ...]) -> None:
        self.dims, self.origin, self.vars, n = snap[0], snap[1], snap[2], snap[3]
        del self.deferred[n:]

    def summary(self) -> str:
        parts = [
            f"{k}={v}" for k, v in self.dims.items() if not k.startswith("_") and str(v) != k
        ]
        parts += [f"*{k}={_tup(v)}" for k, v in self.vars.items()]
        return ", ".join(parts)


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # inconclusive symbolic comparison
        return False


def _tup(t: Any) -> str:
    return "(" + ", ".join(map(str, t)) + ("," if len(t) == 1 else "") + ")"


def is_symbolic(x: Any) -> bool:
    return not isinstance(x, (int, np.integer))


def resolve(env: Env) -> None:
    """Check deferred ``DExpr`` dimensions, solving for one unknown if needed."""
    pending = env.deferred
    while pending:
        rest = []
        progress = False
        for item in pending:
            d, actual, path, spec, value, axis = item
            unbound = [n for n in d.expr.names if n not in env.dims]
            if not unbound:
                try:
                    expected = d.expr(env.dims)
                except Exception as e:
                    raise Mismatch(path, spec, value, f"cannot evaluate `{d}`: {e}") from None
                if not _same(expected, actual):
                    binds = ", ".join(f"{n}={env.dims[n]}" for n in sorted(d.expr.names))
                    raise Mismatch(
                        path, spec, value,
                        f"axis {axis} should be {d} = {expected} (with {binds}), but is {actual}.",
                    )  # fmt: skip
                progress = True
            elif len(unbound) == 1 and not is_symbolic(actual):
                name = unbound[0]
                solved = _solve(d.expr, name, int(actual), env.dims)
                if solved is None:
                    raise Mismatch(
                        path, spec, value,
                        f"axis {axis} is {actual}, but no size of {name} makes `{d}` equal {actual}.",
                    )  # fmt: skip
                env.dims[name] = solved
                env.origin[name] = f"solving `{d}` = {actual} for `{path}`"
                progress = True
            else:
                rest.append(item)
        if not progress:
            d, actual, path, spec, value, axis = rest[0]
            names = ", ".join(n for n in sorted(d.expr.names) if n not in env.dims)
            raise Mismatch(
                path, spec, value,
                f"cannot determine {names} from `{d}`; it must also appear on its own "
                f"in some argument's shape.",
            )  # fmt: skip
        pending = rest
    env.deferred = []


def _solve(expr: Expr, name: str, target: int, dims: dict[str, Any]) -> int | None:
    """Solve ``expr(name) == target`` assuming ``expr`` is linear in ``name``."""
    try:
        f0 = expr({**dims, name: 0})
        f1 = expr({**dims, name: 1})
        a = f1 - f0
        if a == 0 or (target - f0) % a:
            return None
        cand = (target - f0) // a
        if cand < 0 or expr({**dims, name: cand}) != target:
            return None
        return int(cand)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------

_SCALARS = (bool, int, float, complex)


def _is_key_dtype(dtype: Any) -> bool:
    try:
        return bool(jnp.issubdtype(dtype, jax.dtypes.prng_key))
    except Exception:
        return False


def short_dtype(dtype: Any) -> str:
    name = SHORT_NAMES.get(dtype)
    if name:
        return name
    if _is_key_dtype(dtype):
        return "Key"
    return str(dtype)


def fmt_shape(shape: Any) -> str:
    return "[" + ", ".join(map(str, shape)) + "]" if len(shape) else "[()]"


def describe(value: Any) -> str:
    """Short description of a received value, e.g. ``f32[32, 10]``."""
    t = type(value)
    if t in _SCALARS:
        return f"Python {t.__name__}"
    if value is None:
        return "None"
    try:
        return short_dtype(value.dtype) + fmt_shape(tuple(value.shape))
    except Exception:
        return t.__name__


def _match_shape(
    dims: tuple[Any, ...],
    var_index: int | None,
    shape: tuple[Any, ...],
    path: str,
    env: Env,
    spec: Any,
    value: Any,
) -> None:
    n = len(dims)
    if var_index is None:
        if len(shape) != n:
            raise Mismatch(
                path, spec, value,
                f"expected {n} ax{'is' if n == 1 else 'es'}, got {len(shape)}.",
            )  # fmt: skip
        k = 0
    else:
        k = len(shape) - (n - 1)
        if k < 0:
            raise Mismatch(path, spec, value, f"expected at least {n - 1} axes, got {len(shape)}.")
    axis = 0
    for i, d in enumerate(dims):
        t = type(d)
        if i == var_index:
            if t is DVar:
                got = shape[axis : axis + k]
                have = env.vars.get(d.name)
                if have is None:
                    env.vars[d.name] = got
                    env.origin["*" + d.name] = f"`{path}: {describe(value)}`"
                elif len(have) != len(got) or not all(map(_same, have, got)):
                    raise Mismatch(
                        path, spec, value,
                        f"*{d.name} is {_tup(have)} (from "
                        f"{env.origin.get('*' + d.name, '?')}), but here it is {_tup(got)}.",
                    )  # fmt: skip
            axis += k
            continue
        size = shape[axis]
        if t is DName:
            have = env.dims.get(d.name)
            if have is None:
                env.dims[d.name] = size
                env.origin[d.name] = f"`{path}: {describe(value)}`"
            elif have is not size and not _same(have, size):
                if str(have) == d.name:  # symbolic: the name stands for itself
                    reason = f"axis {axis} of `{path}` should be {d.name}, but is {size}."
                else:
                    reason = (
                        f"{d.name} is {have} (from {env.origin.get(d.name, '?')}), "
                        f"but axis {axis} of `{path}` is {size}."
                    )
                raise Mismatch(path, spec, value, reason)
        elif t is DConst:
            if not _same(d.value, size):
                want = f"{d.label} = {d.value}" if d.label else str(d.value)
                raise Mismatch(path, spec, value, f"axis {axis} should be {want}, but is {size}.")
        elif t is DExpr:
            env.deferred.append((d, size, path, spec, value, axis))
        axis += 1


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------

Visit = Callable[["Spec", Any, str], Any]


class Unsynthesizable(Exception):
    """The prover cannot invent an example value for this spec."""


class Spec:
    """Base class. ``leafy`` is True if the spec constrains anything."""

    leafy = True

    def match(self, value: Any, path: str, env: Env) -> None:
        raise NotImplementedError

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        """Rebuild ``value`` with ``visit(spec, leaf, path)`` applied to array leaves."""
        return value

    def abstract(self, ctx: Any, path: str) -> Any:
        raise Unsynthesizable(path)

    def names(self, into: set[str]) -> None:
        """Collect free dimension names (``*Name`` for variadics)."""

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return self


def _dims_names(dims: tuple[Any, ...], into: set[str]) -> None:
    for d in dims:
        if isinstance(d, DName):
            into.add(d.name)
        elif isinstance(d, DVar):
            into.add("*" + d.name)
        elif isinstance(d, DExpr):
            into.update(d.expr.names)


class ArraySpec(Spec):
    def __init__(self, dtype: DType, dims: tuple[Any, ...], var_index: int | None):
        self.dtype = dtype
        self.dims = dims
        self.var_index = var_index

    def __str__(self) -> str:
        if not self.dims:
            return f"{self.dtype.name}[()]"
        return f"{self.dtype.name}[{', '.join(map(str, self.dims))}]"

    @property
    def core_rank(self) -> int | None:
        return None if self.var_index is not None else len(self.dims)

    def match(self, value: Any, path: str, env: Env) -> None:
        t = type(value)
        if t in _SCALARS:
            if t not in self.dtype.weak:
                raise Mismatch(
                    path, self, value, f"expected dtype {self.dtype.name}, got a Python {t.__name__}."
                )
            shape: tuple[Any, ...] = ()
        else:
            try:
                shape = tuple(value.shape)
                dtype = value.dtype
            except AttributeError:
                raise Mismatch(
                    path, self, value, f"expected an array, got {t.__name__}."
                ) from None
            if not self.dtype.accepts(dtype):
                raise Mismatch(
                    path, self, value,
                    f"expected dtype {self.dtype.name}, got {short_dtype(dtype)}.",
                )  # fmt: skip
        _match_shape(self.dims, self.var_index, shape, path, env, self, value)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        return visit(self, value, path)

    def abstract(self, ctx: Any, path: str) -> Any:
        return jax.ShapeDtypeStruct(ctx.shape(self.dims), ctx.dtype(self.dtype))

    def names(self, into: set[str]) -> None:
        _dims_names(self.dims, into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return ArraySpec(self.dtype, tuple(f(d) for d in self.dims), self.var_index)


class KeySpec(Spec):
    """A PRNG key (new-style typed key, or legacy ``uint32[2]``)."""

    def __init__(self, dims: tuple[Any, ...], var_index: int | None):
        self.dims = dims
        self.var_index = var_index

    def __str__(self) -> str:
        return "Key" + (f"[{', '.join(map(str, self.dims))}]" if self.dims else "")

    @property
    def core_rank(self) -> int | None:
        return None if self.var_index is not None else len(self.dims)

    @staticmethod
    def batch_shape(value: Any) -> tuple[Any, ...] | None:
        """Shape of the batch of keys, or None if ``value`` is not a key."""
        try:
            shape, dtype = tuple(value.shape), value.dtype
        except AttributeError:
            return None
        if _is_key_dtype(dtype):
            return shape
        if dtype == np.uint32 and shape and _same(shape[-1], 2):
            return shape[:-1]
        return None

    def match(self, value: Any, path: str, env: Env) -> None:
        shape = self.batch_shape(value)
        if shape is None:
            raise Mismatch(
                path, self, value,
                f"expected a PRNG key (jax.random.key(seed)), got {describe(value)}.",
            )  # fmt: skip
        _match_shape(self.dims, self.var_index, shape, path, env, self, value)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        return visit(self, value, path)

    def abstract(self, ctx: Any, path: str) -> Any:
        return ctx.key(ctx.shape(self.dims))

    def names(self, into: set[str]) -> None:
        _dims_names(self.dims, into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return KeySpec(tuple(f(d) for d in self.dims), self.var_index)


_LENIENT = {int: (int, np.integer), float: (int, float, np.integer, np.floating)}


class StaticSpec(Spec):
    """A value that must be concrete at trace time."""

    def __init__(self, inner: Any):
        self.inner = inner

    def __str__(self) -> str:
        return f"Static[{getattr(self.inner, '__name__', None) or repr(self.inner)}]"

    def match(self, value: Any, path: str, env: Env) -> None:
        if isinstance(value, Tracer):
            m = Mismatch(
                path, self, value,
                "it is a traced value, but Static arguments must be concrete Python "
                "values at trace time. Pass a Python value, or annotate the parameter "
                "as an array if it is meant to be traced.",
            )  # fmt: skip
            m.stage = True  # type: ignore[attr-defined]
            raise m
        inner = self.inner
        if type(inner) is type:
            ok = _LENIENT.get(inner, inner)
            if not isinstance(value, ok) or (inner is not bool and type(value) is bool and inner in _LENIENT):
                raise Mismatch(
                    path, self, value,
                    f"expected {inner.__name__}, got {type(value).__name__}.",
                )  # fmt: skip


class NoneSpec(Spec):
    def __str__(self) -> str:
        return "None"

    def match(self, value: Any, path: str, env: Env) -> None:
        if value is not None:
            raise Mismatch(path, self, value, f"expected None, got {describe(value)}.")

    def abstract(self, ctx: Any, path: str) -> Any:
        return None


class OpaqueSpec(Spec):
    """Anything typax does not look into."""

    leafy = False

    def __init__(self, ann: Any = None):
        self.ann = ann

    def __str__(self) -> str:
        a = self.ann
        if a is None or a is inspect.Parameter.empty:
            return "<unannotated>"
        return getattr(a, "__name__", None) or str(a)

    def match(self, value: Any, path: str, env: Env) -> None:
        pass


class TupleSpec(Spec):
    def __init__(self, elems: tuple[Spec, ...]):
        self.elems = elems

    def __str__(self) -> str:
        return f"tuple[{', '.join(map(str, self.elems))}]"

    def match(self, value: Any, path: str, env: Env) -> None:
        if not isinstance(value, tuple):
            raise Mismatch(path, self, value, f"expected a tuple, got {describe(value)}.")
        if len(value) != len(self.elems):
            raise Mismatch(
                path, self, value,
                f"expected a tuple of {len(self.elems)} elements, got {len(value)}.",
            )  # fmt: skip
        for i, (s, v) in enumerate(zip(self.elems, value)):
            s.match(v, f"{path}[{i}]", env)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        return tuple(s.walk(v, f"{path}[{i}]", visit) for i, (s, v) in enumerate(zip(self.elems, value)))

    def abstract(self, ctx: Any, path: str) -> Any:
        return tuple(s.abstract(ctx, f"{path}[{i}]") for i, s in enumerate(self.elems))

    def names(self, into: set[str]) -> None:
        for s in self.elems:
            s.names(into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return TupleSpec(tuple(s.replace_dims(f) for s in self.elems))


class SeqSpec(Spec):
    """Homogeneous ``list[X]`` or ``tuple[X, ...]``."""

    def __init__(self, elem: Spec, kind: type):
        self.elem = elem
        self.kind = kind

    def __str__(self) -> str:
        return f"list[{self.elem}]" if self.kind is list else f"tuple[{self.elem}, ...]"

    def match(self, value: Any, path: str, env: Env) -> None:
        if not isinstance(value, self.kind):
            raise Mismatch(
                path, self, value, f"expected a {self.kind.__name__}, got {describe(value)}."
            )
        seq: Any = value
        for i, v in enumerate(seq):
            self.elem.match(v, f"{path}[{i}]", env)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        return type(value)(self.elem.walk(v, f"{path}[{i}]", visit) for i, v in enumerate(value))

    def abstract(self, ctx: Any, path: str) -> Any:
        return self.kind(self.elem.abstract(ctx, f"{path}[{i}]") for i in range(2))

    def names(self, into: set[str]) -> None:
        self.elem.names(into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return SeqSpec(self.elem.replace_dims(f), self.kind)


class DictSpec(Spec):
    def __init__(self, value: Spec):
        self.value = value

    def __str__(self) -> str:
        return f"dict[str, {self.value}]"

    def match(self, value: Any, path: str, env: Env) -> None:
        if not isinstance(value, dict):
            raise Mismatch(path, self, value, f"expected a dict, got {describe(value)}.")
        for k, v in value.items():
            self.value.match(v, f"{path}[{k!r}]", env)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        return {k: self.value.walk(v, f"{path}[{k!r}]", visit) for k, v in value.items()}

    def abstract(self, ctx: Any, path: str) -> Any:
        return {k: self.value.abstract(ctx, f"{path}[{k!r}]") for k in ("a", "b")}

    def names(self, into: set[str]) -> None:
        self.value.names(into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return DictSpec(self.value.replace_dims(f))


class UnionSpec(Spec):
    def __init__(self, options: tuple[Spec, ...]):
        self.options = options

    def __str__(self) -> str:
        return " | ".join(map(str, self.options))

    def _select(self, value: Any, path: str, env: Env) -> Spec:
        errors: list[Mismatch] = []
        for opt in self.options:
            snap = env.snapshot()
            try:
                opt.match(value, path, env)
                return opt
            except Mismatch as e:
                env.restore(snap)
                errors.append(e)
        real = [e for e, o in zip(errors, self.options) if not isinstance(o, NoneSpec)]
        if len(real) == 1 and value is not None:
            raise real[0]
        raise Mismatch(
            path, self, value,
            "it matches none of the alternatives:\n"
            + "\n".join(f"      {o}: {e.reason}" for o, e in zip(self.options, errors)),
        )  # fmt: skip

    def match(self, value: Any, path: str, env: Env) -> None:
        self._select(value, path, env)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        try:
            opt = self._select(value, path, Env())
        except Mismatch:
            return value
        return opt.walk(value, path, visit)

    def abstract(self, ctx: Any, path: str) -> Any:
        for opt in self.options:
            if not isinstance(opt, NoneSpec):
                return opt.abstract(ctx, path)
        return None

    def names(self, into: set[str]) -> None:
        for o in self.options:
            o.names(into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return UnionSpec(tuple(o.replace_dims(f) for o in self.options))


class StructSpec(Spec):
    """A NamedTuple or dataclass whose fields carry annotations."""

    def __init__(self, cls: type, fields: dict[str, Spec], label: str):
        self.cls = cls
        self.fields = fields
        self.label = label

    def __str__(self) -> str:
        return self.label

    def match(self, value: Any, path: str, env: Env) -> None:
        if not isinstance(value, self.cls):
            raise Mismatch(
                path, self, value, f"expected {self.cls.__name__}, got {describe(value)}."
            )
        for name, s in self.fields.items():
            if s.leafy:
                s.match(getattr(value, name), f"{path}.{name}", env)

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        if not isinstance(value, self.cls):
            return value
        new = {
            name: s.walk(getattr(value, name), f"{path}.{name}", visit)
            for name, s in self.fields.items()
            if s.leafy
        }
        if all(new[n] is getattr(value, n) for n in new):
            return value
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(value, **new)  # type: ignore[type-var]
        return value._replace(**new)  # type: ignore[attr-defined]

    def abstract(self, ctx: Any, path: str) -> Any:
        kwargs = {}
        defaults = _field_defaults(self.cls)
        for name, s in self.fields.items():
            if not s.leafy or isinstance(s, StaticSpec):
                if name in defaults:
                    continue
                raise Unsynthesizable(f"{path}.{name}")
            kwargs[name] = s.abstract(ctx, f"{path}.{name}")
        try:
            return self.cls(**kwargs)
        except Exception as e:
            raise Unsynthesizable(f"{path} ({self.cls.__name__}: {e})") from None

    def names(self, into: set[str]) -> None:
        for s in self.fields.values():
            s.names(into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return StructSpec(
            self.cls, {n: s.replace_dims(f) for n, s in self.fields.items()}, self.label
        )


def _field_defaults(cls: type) -> set[str]:
    if dataclasses.is_dataclass(cls):
        return {
            f.name
            for f in dataclasses.fields(cls)
            if f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
            or not f.init
        }
    return set(getattr(cls, "_field_defaults", {}))


# --------------------------------------------------------------------------
# Parsing annotations
# --------------------------------------------------------------------------


def _struct_fields(cls: type) -> list[str] | None:
    if dataclasses.is_dataclass(cls):
        return [f.name for f in dataclasses.fields(cls)]
    if isinstance(cls, type) and issubclass(cls, tuple) and hasattr(cls, "_fields"):
        return list(cls._fields)  # type: ignore[attr-defined]
    return None


def _class_annotations(cls: type) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in reversed(cls.__mro__):
        if c is object:
            continue
        try:
            out.update(inspect.get_annotations(c, eval_str=True))
        except Exception:
            try:
                out.update(
                    {k: v for k, v in inspect.get_annotations(c).items() if not isinstance(v, str)}
                )
            except Exception:
                pass
    return out


def _prepend(spec: Spec, lead: tuple[Any, ...]) -> Spec:
    """``spec`` with the axes ``lead`` in front of every array (for ``Batch[...]``)."""
    if isinstance(spec, (ArraySpec, KeySpec)):
        dims = lead + spec.dims
        var = [i for i, d in enumerate(dims) if isinstance(d, (DVar, DEtc))]
        if len(var) > 1:
            raise TypeError("Batch[...] would give a shape with more than one variadic")
        v = var[0] if var else None
        return ArraySpec(spec.dtype, dims, v) if isinstance(spec, ArraySpec) else KeySpec(dims, v)
    if isinstance(spec, TupleSpec):
        return TupleSpec(tuple(_prepend(s, lead) for s in spec.elems))
    if isinstance(spec, SeqSpec):
        return SeqSpec(_prepend(spec.elem, lead), spec.kind)
    if isinstance(spec, DictSpec):
        return DictSpec(_prepend(spec.value, lead))
    if isinstance(spec, UnionSpec):
        return UnionSpec(tuple(_prepend(s, lead) for s in spec.options))
    if isinstance(spec, StructSpec):
        label = f"Batch[{spec.label}, {', '.join(map(str, lead))}]"
        return StructSpec(spec.cls, {n: _prepend(s, lead) for n, s in spec.fields.items()}, label)
    return spec


def parse_owner(cls: type) -> Spec:
    """Spec for ``self`` inside a method of ``cls``.

    The class's own type parameters keep their plain names, so that a method
    can use them: ``class Linear[I, O]`` with ``def __call__(self, x: f32[I])``.
    """
    names = _struct_fields(cls)
    if names is None:
        return OpaqueSpec()
    params = getattr(cls, "__type_params__", ()) or getattr(cls, "__parameters__", ())
    inner = {p: DName(p.__name__) for p in params if isinstance(p, typing.TypeVar)}
    anns = _class_annotations(cls)
    fields: dict[str, Spec] = {}
    for n in names:
        try:
            fields[n] = parse(anns.get(n, inspect.Parameter.empty), inner, (cls,))
        except TypeError:
            fields[n] = OpaqueSpec(anns.get(n))
    if any(s.leafy for s in fields.values()):
        return StructSpec(cls, fields, cls.__name__)
    return OpaqueSpec()


def parse(ann: Any, subst: dict[Any, Any] | None = None, _stack: tuple[type, ...] = ()) -> Spec:
    """Turn an annotation into a :class:`Spec`."""
    subst = subst or {}
    if ann is inspect.Parameter.empty:
        return OpaqueSpec(ann)
    if ann is None or ann is type(None):
        return NoneSpec()
    if isinstance(ann, DType):
        return ArraySpec(ann, (), None)
    if isinstance(ann, ArrayAnn):
        dims, var = _parse_dims(ann.items, subst)
        return ArraySpec(ann.dtype, dims, var)
    if isinstance(ann, KeyAnn):
        dims, var = _parse_dims(ann.items, subst)
        return KeySpec(dims, var)
    if isinstance(ann, StaticAnn):
        return StaticSpec(ann.inner)
    if isinstance(ann, BatchAnn):
        lead, _ = _parse_dims(ann.items, subst)
        return _prepend(parse(ann.inner, subst, _stack), lead)
    if isinstance(ann, typing.TypeAliasType):
        return parse(ann.__value__, subst, _stack)

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Annotated:
        return parse(args[0], subst, _stack)
    if origin is typing.Union or origin is types.UnionType:
        opts = tuple(parse(a, subst, _stack) for a in args)
        if not any(o.leafy for o in opts if not isinstance(o, NoneSpec)):
            return OpaqueSpec(ann)
        return UnionSpec(opts)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            elem = parse(args[0], subst, _stack)
            return SeqSpec(elem, tuple) if elem.leafy else OpaqueSpec(ann)
        if args == ((),):
            args = ()
        elems = tuple(parse(a, subst, _stack) for a in args)
        return TupleSpec(elems) if any(e.leafy for e in elems) else OpaqueSpec(ann)
    if origin is list:
        elem = parse(args[0], subst, _stack)
        return SeqSpec(elem, list) if elem.leafy else OpaqueSpec(ann)
    if origin is dict:
        val = parse(args[1], subst, _stack)
        return DictSpec(val) if val.leafy else OpaqueSpec(ann)

    cls = origin if isinstance(origin, type) else ann
    if isinstance(cls, type) and cls not in _stack:
        names = _struct_fields(cls)
        if names is not None:
            params = getattr(cls, "__type_params__", ()) or getattr(cls, "__parameters__", ())
            inner: dict[Any, Any] = {}
            if origin is not None and args:
                shown = []
                for p, a in zip(params, args):
                    d = parse_dim(a, subst) if not isinstance(p, typing.TypeVarTuple) else None
                    if d is not None:
                        inner[p] = d
                        shown.append(str(d))
                label = f"{cls.__name__}[{', '.join(shown)}]"
            else:
                # Unparametrised generic: its dims are private to this value.
                for p in params:
                    if isinstance(p, typing.TypeVar):
                        inner[p] = DName(f"{cls.__name__}.{p.__name__}")
                label = cls.__name__
            anns = _class_annotations(cls)
            fields: dict[str, Spec] = {}
            for n in names:
                try:
                    fields[n] = parse(anns.get(n, inspect.Parameter.empty), inner, _stack + (cls,))
                except TypeError:
                    fields[n] = OpaqueSpec(anns.get(n))
            if any(s.leafy for s in fields.values()):
                return StructSpec(cls, fields, label)
    return OpaqueSpec(ann)

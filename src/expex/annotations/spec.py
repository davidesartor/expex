"""Specs: the parsed form of annotations, and how values are matched to them.

``parse`` turns an annotation (``f32[B, T]``, ``tuple[f32[B], Key]``, a
dataclass with annotated fields, ...) into a tree of ``Spec`` objects. A spec
can then

* ``match`` a value, binding dimension names in an :class:`Env`;
* ``walk`` a value, visiting (and optionally replacing) its array leaves;
* ``abstract`` itself into example inputs for the prover.
"""

import dataclasses
import inspect
import types
import typing
from typing import Any, Callable, Iterator

import jax
import numpy as np
from jax.core import Tracer

from .dims import (
    DimError,
    DEllipsis,
    _SCALARS,
    DConst,
    DExpr,
    DName,
    DVar,
    Env,
    _is_key_dtype,
    _match_shape,
    _parse_dims,
    _same,
    describe,
    is_symbolic,
    short_dtype,
)
from .errors import Mismatch, code_of, coded, strip_see
from .expr import Expr
from .sym import evaluate, named_sizes, texts
from .types import ArrayAnn, BatchAnn, DType, KeyAnn, StaticAnn

# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------

Visit = Callable[["Spec", Any, str], Any]


def join(path: str, name: str) -> str:
    """The path of field ``name`` under ``path``: ``blocks[2].mlp`` + ``w1``, or ``w1`` at the root."""
    return f"{path}.{name}" if path else name


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


def _computing(dtype: Any, declared: DType) -> bool:
    """After a failed dtype check: inside a mixed-precision body, its compute dtype passes for a float one."""
    from ..precision import accepts

    return accepts(dtype, declared)


def _dims_names(dims: tuple[Any, ...], into: set[str]) -> None:
    for d in dims:
        if isinstance(d, DName):
            into.add(d.name)
        elif isinstance(d, DVar):
            into.add("*" + d.name)
        elif isinstance(d, DExpr):
            into.update(d.expr.names)
        elif isinstance(d, DEllipsis):
            into.add("*...")  # lets the prover try it at several ranks


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
        if t not in _SCALARS and not hasattr(value, "shape") and bool(jax.export.is_symbolic_dim(value)):
            t = int  # a size, symbolic under the prover: an int at run time (checked last: it is slow)
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
            if not self.dtype.accepts(dtype) and not (
                # a NumPy array is judged by the dtype JAX will give it
                isinstance(value, (np.ndarray, np.generic))
                and self.dtype.accepts(jax.dtypes.canonicalize_dtype(dtype))
            ) and not _computing(dtype, self.dtype):
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


_LENIENT = {
    int: (int, np.integer),
    float: (int, float, np.integer, np.floating),
    complex: (int, float, complex, np.integer, np.floating, np.complexfloating),
}
_KIND_NAMES = {bool: "a boolean", int: "an integer", float: "a real", complex: "a numeric"}
_SCALAR_KINDS = {bool: "b", int: "iu", float: "iuf", complex: "iufc"}


def _ann_name(ann: Any) -> str:
    return getattr(ann, "__name__", None) or repr(ann).replace("typing.", "")


class StaticSpec(Spec):
    """A plain Python value: a number, a string, a function, a config object, ...

    Numbers (``int``, ``float``, ``bool``, ``complex``) also accept scalar arrays,
    so that ``jax.jit`` may trace them. Anything else must be concrete.
    """

    def __init__(self, inner: Any, explicit: bool = False):
        self.inner = inner
        #: Written as ``Static[...]``: static even where a plain number would be traced.
        self.explicit = explicit

    def __str__(self) -> str:
        return _ann_name(self.inner)

    def match(self, value: Any, path: str, env: Env) -> None:
        inner = self.inner
        kinds = _SCALAR_KINDS.get(inner) if type(inner) is type else None
        if kinds is not None and hasattr(value, "dtype") and hasattr(value, "shape"):
            if value.shape == () and np.dtype(value.dtype).kind in kinds:
                return
            what = "a scalar" if value.shape != () else f"{_KIND_NAMES[inner]} scalar"
            raise Mismatch(path, self, value, f"expected {what} for `{inner.__name__}`, got {describe(value)}.")
        if inner is int and jax.export.is_symbolic_dim(value):
            return  # a size, symbolic under the prover: `self.empty(T)` for `def empty(self, S: int)`
        if any(isinstance(x, Tracer) for x in jax.tree.leaves(value)):
            root = path.split(".")[0].split("[")[0]
            m = Mismatch(
                path, self, value,
                f"it is traced, but a `{_ann_name(inner)}` must be known when the function is "
                f"traced. Under jax.jit, mark it static: jax.jit(f, static_argnames=\"{root}\"). "
                f"If it is meant to be an array, annotate it with a dtype (f32, i32, ...).",
            )  # fmt: skip
            m.stage = True  # type: ignore[attr-defined]
            raise m
        if typing.get_origin(inner) is typing.Literal:
            if not any(value == a and type(value) is type(a) for a in typing.get_args(inner)):
                raise Mismatch(path, self, value, f"expected one of {list(typing.get_args(inner))}, got {value!r}.")
            return
        if isinstance(inner, type) and type(inner) is not type:  # enums, ABCs, ...
            try:
                ok_type = isinstance(value, inner)
            except TypeError:  # e.g. a Protocol that is not runtime-checkable
                return
            if not ok_type:
                raise Mismatch(path, self, value, f"expected {inner.__name__}, got {type(value).__name__}.")
            return
        if type(inner) is type:
            ok = _LENIENT.get(inner, inner)
            if not isinstance(value, ok) or (inner is not bool and type(value) is bool and inner in _LENIENT):
                hint = ""
                raise Mismatch(
                    path, self, value,
                    f"expected {inner.__name__}, got {describe(value) if hasattr(value, 'shape') else type(value).__name__}.{hint}",
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
    """Anything expex does not look into."""

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
        ctx.lists.append(path)
        return self.kind(self.elem.abstract(ctx, f"{path}[{i}]") for i in range(ctx.list_length))

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

    def option(self, value: Any) -> Spec:
        """The member that describes ``value``: ``None``, a struct of its class, else the first array-like one."""
        if value is None:
            return next((o for o in self.options if isinstance(o, NoneSpec)), NoneSpec())
        structs: list[Spec] = [o for o in self.options if isinstance(o, StructSpec) and isinstance(value, o.cls)]
        others = [o for o in self.options if not isinstance(o, NoneSpec)]
        return (structs or [o for o in others if o.leafy] or others)[0]

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        try:
            opt = self._select(value, path, Env())
        except Mismatch:
            # No exact match, e.g. an array with extra batch axes (as `vectorize`
            # sees it): fall back to the first member of the right kind.
            if value is None:
                return value
            arrayish = hasattr(value, "shape") and hasattr(value, "dtype")
            for o in self.options:
                if isinstance(o, NoneSpec):
                    continue
                cls = getattr(o, "cls", None)
                if (cls is not None and isinstance(value, cls)) or (
                    arrayish and isinstance(o, (ArraySpec, KeySpec))
                ):
                    return o.walk(value, path, visit)
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


def rebuild(obj_or_cls: Any, **values: Any) -> Any:
    """A NamedTuple / dataclass built from field values, without calling ``__init__``.

    ``rebuild(obj, f=v)`` copies ``obj`` with some fields changed;
    ``rebuild(cls, **fields)`` makes a new one, filling in field defaults.
    Classes with a custom ``__init__`` (``@expex.struct`` modules) need this.
    """
    cls: Any
    base: dict[str, Any] | None
    if isinstance(obj_or_cls, type):
        cls, base = obj_or_cls, {}
    else:
        cls, base = type(obj_or_cls), None
    if issubclass(cls, tuple):
        if base is None:
            record: Any = obj_or_cls
            return record._replace(**values)
        return cls(**values)
    if not getattr(cls, "__expex_struct__", False):
        if base is None:
            return dataclasses.replace(obj_or_cls, **values)  # type: ignore[type-var]
        return cls(**values)
    new = object.__new__(cls)
    for f in dataclasses.fields(cls):
        if f.name in values:
            v = values[f.name]
        elif base is None:
            v = getattr(obj_or_cls, f.name)
        elif f.default is not dataclasses.MISSING:
            v = f.default
        elif f.default_factory is not dataclasses.MISSING:
            v = f.default_factory()
        else:
            raise coded("EX302", TypeError(f"{cls.__name__}: no value for field `{f.name}`"))
        object.__setattr__(new, f.name, v)
    return new


def _ident(name: str) -> str:
    return name.replace(".", "__")


def _rename(expr: Expr, mapping: dict[str, str]) -> Expr:
    """``expr`` with the names in ``mapping`` replaced by other source text."""
    import ast

    tree = ast.parse(expr.src, mode="eval")

    class Sub(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> Any:
            if node.id in mapping:
                return ast.parse(f"({mapping[node.id]})", mode="eval").body
            return node

    return Expr(ast.unparse(Sub().visit(tree)), constraint=expr.is_constraint)


class StructSpec(Spec):
    """A NamedTuple or dataclass whose fields carry annotations."""

    def __init__(
        self,
        cls: type,
        fields: dict[str, Spec],
        label: str,
        subst: dict[Any, Any] | None = None,
        invariants: list[tuple[Expr, dict[str, str]]] | None = None,
    ):
        self.cls = cls
        self.fields = fields
        self.label = label
        #: The class's `where` invariants, over the dimensions of this use of it,
        #: each with a map from the safe names used in it to environment keys.
        self.invariants = invariants if invariants is not None else self._translate(subst or {})

    def _translate(self, subst: dict[Any, Any]) -> list[tuple[Expr, dict[str, str]]]:
        out = []
        params = getattr(self.cls, "__type_params__", ()) or getattr(self.cls, "__parameters__", ())
        for inv in getattr(self.cls, "__expex_where__", ()):
            mapping: dict[str, str] = {}
            back: dict[str, str] = {}
            ok = True
            for p in params:
                if p.__name__ not in inv.names:
                    continue
                d = subst.get(p)
                if isinstance(d, DName):
                    mapping[p.__name__] = _ident(d.name)
                    back[_ident(d.name)] = d.name
                elif isinstance(d, DConst):
                    mapping[p.__name__] = str(d.value)
                else:
                    ok = False  # a variadic or computed size: not expressible
            if ok:
                out.append((_rename(inv, mapping), back))
        return out

    def assumptions(self) -> list[Expr]:
        """The invariants with field names replaced by the class's defaults (for the prover)."""
        out = []
        defaults: dict[str, Any] = {}
        if dataclasses.is_dataclass(self.cls):
            defaults = {f.name: f.default for f in dataclasses.fields(self.cls)}
        else:
            defaults = dict(getattr(self.cls, "_field_defaults", {}))
        for inv, back in self.invariants:
            mapping = {}
            for n in inv.names:
                if n in back:
                    mapping[n] = back[n]
                elif isinstance(defaults.get(n), int) and not isinstance(defaults.get(n), bool):
                    mapping[n] = str(defaults[n])
            if len(mapping) == len(inv.names) and all(v.isidentifier() or v.isdigit() for v in mapping.values()):
                out.append(_rename(inv, mapping))
        for s in self.fields.values():
            out += _assumptions_of(s)
        return out

    def __str__(self) -> str:
        return self.label

    def match(self, value: Any, path: str, env: Env) -> None:
        if not isinstance(value, self.cls):
            raise Mismatch(
                path, self, value, f"expected {self.cls.__name__}, got {describe(value)}."
            )
        for name, s in self.fields.items():
            if s.leafy:
                s.match(getattr(value, name), join(path, name), env)
        for inv, back in self.invariants:
            vals: dict[str, Any] = {}
            for n in inv.names:
                if n in back and back[n] in env.dims:
                    vals[n] = env.dims[back[n]]
                elif n not in back and isinstance(getattr(value, n, None), (int, np.integer)):
                    vals[n] = int(getattr(value, n))
            if len(vals) != len(inv.names) or any(is_symbolic(v) for v in vals.values()):
                continue
            try:
                ok = bool(inv(vals))
            except Exception:
                continue
            if not ok:
                shown = ", ".join(f"{back.get(k, k)}={v}" for k, v in sorted(vals.items()))
                raise Mismatch(
                    path, self, value,
                    f"{self.cls.__name__} requires `{inv}`, which does not hold ({shown}).",
                )  # fmt: skip

    def walk(self, value: Any, path: str, visit: Visit) -> Any:
        if not isinstance(value, self.cls):
            return value
        new = {
            name: s.walk(getattr(value, name), join(path, name), visit)
            for name, s in self.fields.items()
            if s.leafy
        }
        if all(new[n] is getattr(value, n) for n in new):
            return value
        if dataclasses.is_dataclass(value):
            return rebuild(value, **new)
        return value._replace(**new)  # type: ignore[attr-defined]

    def abstract(self, ctx: Any, path: str) -> Any:
        kwargs = {}
        defaults = _field_defaults(self.cls)
        for name, s in self.fields.items():
            if not s.leafy or isinstance(s, StaticSpec):
                if name in defaults:
                    continue
                raise Unsynthesizable(join(path, name))
            kwargs[name] = s.abstract(ctx, join(path, name))
        try:
            return rebuild(self.cls, **kwargs)
        except Exception as e:
            raise Unsynthesizable(f"{path} ({self.cls.__name__}: {strip_see(str(e))})") from None

    def names(self, into: set[str]) -> None:
        for s in self.fields.values():
            s.names(into)

    def replace_dims(self, f: Callable[[Any], Any]) -> "Spec":
        return StructSpec(
            self.cls,
            {n: s.replace_dims(f) for n, s in self.fields.items()},
            self.label,
            invariants=self.invariants,
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
    """Field annotations of a class and its bases, with strings evaluated."""
    import sys

    out: dict[str, Any] = {}
    for c in reversed(cls.__mro__):
        if c is object:
            continue
        anns = texts(c)
        glb = getattr(sys.modules.get(c.__module__), "__dict__", {})
        scope = dict(getattr(c, "__expex_localns__", {}))
        scope.update({p.__name__: p for p in getattr(c, "__type_params__", ())})
        scope.update(named_sizes(c))  # cells: i32[Rows:6, Cols:7]
        scope[c.__name__] = c
        for k, v in anns.items():
            text = v if isinstance(v, str) else getattr(v, "__forward_arg__", None)
            if isinstance(text, str):  # a string, or the ForwardRef NamedTuple makes of one
                try:
                    v = evaluate(text, glb, scope)
                except Exception:
                    continue
            out[k] = v
    return out


def _bind_params(cls: type, params: Any, dims: tuple[Any, ...]) -> dict[Any, Any]:
    """Match a generic class's type parameters to the dimensions it was given.

    ``class Mix[*B, M, D]`` used as ``Mix[N, O, M, D]`` binds ``B`` to ``(N, O)``.
    """
    var = [i for i, p in enumerate(params) if isinstance(p, typing.TypeVarTuple)]
    if len(var) > 1:
        raise coded("EX401", TypeError(f"{cls.__name__} has more than one variadic type parameter"))
    fixed = len(params) - len(var)
    if (not var and len(dims) != fixed) or len(dims) < fixed:
        raise coded("EX401", TypeError(
            f"{cls.__name__} takes {'at least ' if var else ''}{fixed} dimension(s), got {len(dims)}"
        ))
    if not var:
        return dict(zip(params, dims))
    t = var[0]
    after = len(params) - t - 1
    inner = dict(zip(params[:t], dims[:t]))
    inner[params[t]] = tuple(dims[t : len(dims) - after])
    inner.update(zip(params[t + 1 :], dims[len(dims) - after :]))
    return inner


def _prepend(spec: Spec, lead: tuple[Any, ...]) -> Spec:
    """``spec`` with the axes ``lead`` in front of every array (for ``Batch[...]``)."""
    if isinstance(spec, (ArraySpec, KeySpec)):
        dims = lead + spec.dims
        var = [i for i, d in enumerate(dims) if isinstance(d, (DVar, DEllipsis))]
        if len(var) > 1:
            raise coded("EX401", TypeError("Batch[...] would give a shape with more than one variadic"))
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
        fields = {n: _prepend(s, lead) for n, s in spec.fields.items()}
        return StructSpec(spec.cls, fields, label, invariants=spec.invariants)
    return spec


_PLAIN = (int, float, bool, str, bytes, complex)


def static_type(ann: Any, _seen: tuple[type, ...] = ()) -> bool:
    """Is ``ann`` a plain Python type, whose values are configuration rather than data?

    ``int``, ``float``, ``bool``, ``str``, ``Literal[...]``, enums, callables,
    tuples of these, ``X | None``, and NamedTuples / dataclasses made only of
    these. Function parameters with such annotations are static.
    """
    import enum

    if ann in _PLAIN or ann is type(None):
        return True
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return True
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Literal:
        return True
    if origin is typing.Annotated:
        return static_type(args[0], _seen)
    abc_callable = __import__("collections.abc").abc.Callable
    if ann is typing.Callable or ann is abc_callable or origin in (typing.Callable, abc_callable):
        return True
    if origin is type or ann is type:
        return True
    if origin is typing.Union or origin is types.UnionType:
        return all(static_type(a, _seen) for a in args)
    if origin in (tuple, frozenset):
        return all(a is Ellipsis or static_type(a, _seen) for a in args) if args else True
    if isinstance(ann, typing.TypeAliasType):
        return static_type(ann.__value__, _seen)
    cls = origin if isinstance(origin, type) else ann
    if isinstance(cls, type) and cls not in _seen and _struct_fields(cls) is not None:
        if dataclasses.is_dataclass(cls) and not cls.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            return False  # mutable: cannot be a compile-time constant
        anns = _class_annotations(cls)
        names = _struct_fields(cls) or []
        return bool(names) and all(
            n in anns and static_type(anns[n], _seen + (cls,)) for n in names
        )
    return False


def parse_owner(cls: type) -> Spec:
    """Spec for ``self`` inside a method of ``cls``.

    The class's own type parameters keep their plain names, so that a method
    can use them: ``class Linear[I, O]`` with ``def __call__(self, x: f32[I])``.
    """
    names = _struct_fields(cls)
    if names is None:
        return OpaqueSpec()
    params = getattr(cls, "__type_params__", ()) or getattr(cls, "__parameters__", ())
    inner: dict[Any, Any] = {p: DName(p.__name__) for p in params if isinstance(p, typing.TypeVar)}
    inner.update({p: (DVar(p.__name__),) for p in params if isinstance(p, typing.TypeVarTuple)})
    anns = _class_annotations(cls)
    fields: dict[str, Spec] = {}
    for n in names:
        try:
            fields[n] = parse(anns.get(n, inspect.Parameter.empty), inner, (cls,))
        except DimError as e:
            raise coded(code_of(e) or "EX401", DimError(f"{cls.__name__}.{n}: {e}")) from None
        except TypeError:
            fields[n] = OpaqueSpec(anns.get(n))
    if any(s.leafy for s in fields.values()):
        return StructSpec(cls, fields, cls.__name__, inner)
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
        return StaticSpec(ann.inner, explicit=True)
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
        if any(isinstance(a, StaticAnn) for a in args):
            # `Static[int] | None`: the whole value is static
            plain = tuple(a.inner if isinstance(a, StaticAnn) else a for a in args)
            return StaticSpec(typing.Union[plain], explicit=True)  # type: ignore[arg-type]
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
                flat, _ = _parse_dims(tuple(args), subst)
                inner = _bind_params(cls, params, flat)
                label = f"{cls.__name__}[{', '.join(map(str, flat))}]"
            else:
                # Unparametrised generic: its dims are private to this value.
                for p in params:
                    if isinstance(p, typing.TypeVar):
                        inner[p] = DName(f"{cls.__name__}.{p.__name__}")
                    elif isinstance(p, typing.TypeVarTuple):
                        inner[p] = (DVar(f"{cls.__name__}.{p.__name__}"),)
                label = cls.__name__
            anns = _class_annotations(cls)
            fields: dict[str, Spec] = {}
            for n in names:
                try:
                    fields[n] = parse(anns.get(n, inspect.Parameter.empty), inner, _stack + (cls,))
                except DimError as e:
                    raise coded(code_of(e) or "EX401", DimError(f"{cls.__name__}.{n}: {e}")) from None
                except TypeError:
                    fields[n] = OpaqueSpec(anns.get(n))
            if any(s.leafy for s in fields.values()):
                return StructSpec(cls, fields, label, inner)
    return OpaqueSpec(ann)


def _assumptions_of(spec: Spec) -> list[Expr]:
    """Struct invariants anywhere inside ``spec``, over its dimensions."""
    if isinstance(spec, StructSpec):
        return spec.assumptions()
    out: list[Expr] = []
    for attr in ("elems", "options"):
        for s in getattr(spec, attr, ()):
            out += _assumptions_of(s)
    for attr in ("elem", "value"):
        s = getattr(spec, attr, None)
        if isinstance(s, Spec):
            out += _assumptions_of(s)
    return out


def fields(spec: Spec, value: Any, path: str = "") -> Iterator[tuple[str, Spec, Any]]:
    """Every part of ``value`` that ``spec`` describes, parents first, as ``(path, spec, value)``.

    Paths are those of error messages and checkpoints: ``blocks[2].mlp.w1``,
    ``heads['q']``. Structs, tuples, lists and dicts are entered; a union is
    resolved to the member that describes the value (:meth:`UnionSpec.option`);
    static and unannotated fields are yielded but not entered.
    """
    if isinstance(spec, UnionSpec):
        spec = spec.option(value)
    yield path, spec, value
    if isinstance(spec, StructSpec):
        for name, s in spec.fields.items():
            yield from fields(s, getattr(value, name), join(path, name))
    elif isinstance(spec, (TupleSpec, SeqSpec)):
        elems = spec.elems if isinstance(spec, TupleSpec) else [spec.elem] * len(value)
        for i, (s, v) in enumerate(zip(elems, value)):
            yield from fields(s, v, f"{path}[{i}]")
    elif isinstance(spec, DictSpec):
        for k, v in value.items():
            yield from fields(spec.value, v, f"{path}[{k!r}]")


def fixed_sizes(*specs: Spec) -> dict[str, int]:
    """The named fixed sizes in ``specs`` (``R:6``, ``class RGB(Dim, size=3)``), for listing as ``R (= 6)``."""
    out: dict[str, int] = {}

    def visit(d: Any) -> Any:
        if isinstance(d, DConst) and d.label:
            out.setdefault(d.label, d.value)
        return d

    for s in specs:
        s.replace_dims(visit)
    return out


def fn_fixed_sizes(f: Any) -> dict[str, int]:
    """:func:`fixed_sizes` of an ``@fn``'s parameters (``self`` included) and result."""
    return fixed_sizes(*(p.spec for p in f.params.values()), f.ret)

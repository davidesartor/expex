"""How ``@fn(jit=True)`` splits arguments into traced arrays and static values.

Arrays and Python numbers are traced; everything else (functions, strings,
``Static[...]`` values, configuration objects) becomes part of the compilation
key. ``_Frozen`` is that key: the argument structure with the arrays taken out.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from ._spec import (
    Spec,
    StaticSpec,
    rebuild,
)

if TYPE_CHECKING:
    from ._fn import Fn


class _StaticValue:
    """Wraps a value that must stay a compile-time constant under jit."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StaticValue) and _static_key(self.value) == _static_key(other.value)

    def __hash__(self) -> int:
        return hash(_static_key(self.value))

    def __repr__(self) -> str:
        return f"static {self.value!r}"


def _static_key(v: Any) -> Any:
    """A hashable stand-in under which two static values count as the same.

    Stricter than ``==``: ``1``, ``True`` and ``1.0`` are equal in Python (and so
    are ``0.0`` and ``-0.0``) but can make a function behave differently, so
    each must get its own compiled version.
    """
    t = type(v)
    if t in (tuple, list):
        return (t, tuple(_static_key(x) for x in v))
    if t is dict:
        return (t, tuple((_static_key(k), _static_key(x)) for k, x in v.items()))
    if t in (frozenset, set):
        return (t, frozenset(_static_key(x) for x in v))
    if t is float:
        return (t, v.hex())
    if t is complex:
        return (t, v.real.hex(), v.imag.hex())
    if isinstance(v, tuple) and hasattr(v, "_fields"):
        return (t, tuple(_static_key(x) for x in v))
    hash(v)  # raises TypeError for unhashable values
    return (t, v)


def _is_traced_leaf(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic)) or type(x) in (bool, int, float, complex)


def _has_static(spec: Spec) -> bool:
    cached = getattr(spec, "_has_static", None)
    if cached is None:
        if isinstance(spec, StaticSpec):
            cached = True
        else:
            subs: list[Spec] = []
            for attr in ("elems", "options"):
                subs += list(getattr(spec, attr, ()))
            for attr in ("elem", "value"):
                if isinstance(getattr(spec, attr, None), Spec):
                    subs.append(getattr(spec, attr))
            subs += list(getattr(spec, "fields", {}).values())
            cached = any(_has_static(s) for s in subs)
        try:
            spec._has_static = cached  # type: ignore[attr-defined]
        except AttributeError:
            pass
    return bool(cached)


def _opaque_dataclass(v: Any) -> bool:
    """A dataclass instance that is not a registered pytree (JAX sees it as one leaf)."""
    if not dataclasses.is_dataclass(v) or isinstance(v, type):
        return False
    leaves = jax.tree.leaves(v)
    return len(leaves) == 1 and leaves[0] is v


class _Record:
    """The fields of a dataclass that is not a pytree, so that jit can trace its arrays."""

    def __init__(self, cls: type, names: tuple[str, ...], values: tuple[Any, ...]):
        self.cls = cls
        self.names = names
        self.values = values

    def restore(self, values: dict[str, Any]) -> Any:
        obj: Any = object.__new__(self.cls)
        for n, x in values.items():
            object.__setattr__(obj, n, x)
        return obj


jax.tree_util.register_pytree_node(
    _Record,
    lambda r: (r.values, (r.cls, r.names)),
    lambda aux, values: _Record(aux[0], aux[1], tuple(values)),
)


def _mark(spec: Spec, v: Any, wrap: bool) -> Any:
    """Wrap (or unwrap) the values that ``spec`` declares Static, inside ``v``."""
    if isinstance(v, _Record) or (wrap and getattr(spec, "fields", None) is not None and _opaque_dataclass(v)):
        pass  # handled below even when the class has no Static field
    elif not _has_static(spec):
        return v
    if isinstance(spec, StaticSpec):
        if wrap:
            return _StaticValue(v)
        return v.value if isinstance(v, _StaticValue) else v
    fields = getattr(spec, "fields", None)
    if isinstance(v, _Record) and fields is not None:  # unwrap
        restored = {
            n: _mark(fields[n], x, wrap) if n in fields else x for n, x in zip(v.names, v.values)
        }
        return v.restore(restored)
    if fields is not None and isinstance(v, getattr(spec, "cls", ())):
        if wrap and _opaque_dataclass(v):
            # a dataclass JAX cannot look into: carry its fields as a record
            dc: Any = v
            names = tuple(f.name for f in dataclasses.fields(dc))
            packed = tuple(
                _mark(fields[n], getattr(v, n), True) if n in fields else getattr(v, n) for n in names
            )
            return _Record(type(v), names, packed)
        new = {n: _mark(s, getattr(v, n), wrap) for n, s in fields.items() if _has_static(s)}
        if isinstance(v, tuple):
            record: Any = v
            return record._replace(**new)
        return rebuild(v, **new)
    elems = getattr(spec, "elems", None)
    if elems is not None and isinstance(v, tuple) and len(v) == len(elems):
        return tuple(_mark(s, x, wrap) for s, x in zip(elems, v))
    elem = getattr(spec, "elem", None)
    if isinstance(elem, Spec) and isinstance(v, (list, tuple)):
        return type(v)(_mark(elem, x, wrap) for x in v)
    value = getattr(spec, "value", None)
    if isinstance(value, Spec) and isinstance(v, dict):
        return {k: _mark(value, x, wrap) for k, x in v.items()}
    for opt in getattr(spec, "options", ()):
        cls = getattr(opt, "cls", None)
        if cls is not None and isinstance(v, cls):
            return _mark(opt, v, wrap)
    return v


def _avals(dyn: tuple[Any, ...]) -> tuple[Any, ...]:
    """Shape, dtype and weak-typedness of each traced leaf: what jit specialises on."""
    return tuple(jax.typeof(x) for x in dyn)


class _Frozen:
    """The static part of a jit call: the argument structure and its non-array leaves."""

    __slots__ = ("names", "tree", "static", "_hash")

    def __init__(self, names: tuple[str, ...], tree: Any, static: tuple[Any, ...]):
        self.names = names
        self.tree = tree
        self.static = static
        self._hash: int | None = None

    @staticmethod
    def split(f: "Fn[Any, Any]", arguments: Any) -> tuple["_Frozen", tuple[Any, ...]]:
        marked = {n: _mark(f.params[n].spec, v, True) for n, v in arguments.items()}
        leaves, tree = jax.tree.flatten(tuple(marked.values()))
        dyn = tuple(x for x in leaves if _is_traced_leaf(x))
        hole = _Frozen  # any unique marker
        static = tuple(hole if _is_traced_leaf(x) else x for x in leaves)
        return _Frozen(tuple(marked), tree, static), dyn

    def rebuild(self, f: "Fn[Any, Any]", dyn: tuple[Any, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        it = iter(dyn)
        leaves = [next(it) if x is _Frozen else x for x in self.static]
        values = self.tree.unflatten(leaves)
        arguments = {n: _mark(f.params[n].spec, v, False) for n, v in zip(self.names, values)}
        ba = inspect.BoundArguments(f._sig, arguments)  # type: ignore[arg-type]
        return ba.args, ba.kwargs

    def _key(self) -> Any:
        return (
            self.names,
            self.tree,
            tuple(x if x is _Frozen or isinstance(x, _StaticValue) else _static_key(x) for x in self.static),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Frozen) and self._key() == other._key()

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self._key())
        return self._hash

"""Axes by name: ``x.sum(T)``, ``jax.nn.softmax(x, axis=T)`` inside an ``@fn``.

A type parameter (``T``), a variadic one (``*B``) or a ``Dim`` class names an
axis. The converter routes calls that use one as an axis through
:func:`named_`, which turns it into a position:

1. If the array is an annotated parameter or local variable, from its annotation.
2. Otherwise from the sizes: the one axis whose size is the name's size in this
   call. Under the prover every size is a distinct symbol, so this always
   works there; the position found is remembered for the call site.
3. If several axes have that size, the remembered position, else an error
   asking for an annotation.
"""

from __future__ import annotations

import typing
from typing import Any

import jax.numpy as jnp

from ._dims import DName, DVar, _same, describe
from ._errors import ShapeError
from ._types import Dim

#: Keyword arguments that take axes.
AXIS_KEYWORDS = frozenset({"axis", "axes", "source", "destination", "axis1", "axis2"})
#: Array methods whose positional arguments are axes.
AXIS_METHODS = frozenset({
    "sum", "mean", "max", "min", "prod", "std", "var", "any", "all", "argmax", "argmin",
    "cumsum", "cumprod", "squeeze", "swapaxes", "transpose", "ptp", "nansum", "nanmean",
})  # fmt: skip

_SEEN: dict[tuple[str, str, int], Any] = {}


def axis_name(v: Any) -> str | None:
    """The axis a value names, or None if it is not an axis name."""
    if isinstance(v, typing.TypeVar):
        return v.__name__
    if isinstance(v, typing.TypeVarTuple):
        return "*" + v.__name__
    if isinstance(v, type) and issubclass(v, Dim) and v is not Dim:
        return v.__name__
    return None


def lifted_(i: int) -> bool:
    """Is the body's ``i``-th leading assert checked as a constraint (so the body skips it)?"""
    from ._ops import CALLS

    return bool(CALLS.stack) and i in getattr(CALLS.stack[-1][1], "_lifted", ())


def size_(name: Any) -> Any:
    """``T`` used as a value inside an ``@fn``: its size in the current call.

    The converter rewrites ``jnp.zeros((B, T))`` to use this. For a variadic
    ``*B`` it is a tuple of sizes, for a derived size (``U = T + 1``) its value.
    Under the prover the sizes are symbolic, so the code is proved for all sizes.
    """
    from ._ops import CALLS

    n = axis_name(name)
    if n is None:
        return name  # not a size name after all (e.g. rebound in an enclosing scope)
    if not CALLS.stack:
        raise ShapeError(
            f"the size {n} is only known inside an @fn call."
        )
    env, fn = CALLS.stack[-1]
    if n.startswith("*"):
        if n[1:] in env.vars:
            return tuple(env.vars[n[1:]])
    elif n in env.dims:
        return env.dims[n]
    elif n in getattr(fn, "defs", {}) and fn.defs[n].names <= env.dims.keys():
        return fn.defs[n](env.dims)  # a derived size, e.g. U = T + 1
    raise ShapeError(
        f"{getattr(fn, 'name', '?')}(): the size {n.lstrip('*')} is not known in this call: only sizes that appear in "
        f"the arguments, or are computed from them in an annotation, are known inside "
        f"the body."
    )


def named_(target: Any, attr: str | None, args: tuple[Any, ...], kwargs: dict[str, Any], site: str, var: str | None) -> Any:
    """Call ``target.attr(*args, **kwargs)`` (or ``target(...)``) with axis names resolved."""
    __tracebackhide__ = True
    if attr is not None:
        f, arr = getattr(target, attr), target
        args = tuple(_axes(a, arr, site, var) for a in args)
    else:
        f, arr = target, args[0] if args else None
        if isinstance(arr, (list, tuple)) and arr:
            arr = arr[0]
    kwargs = {k: _axes(v, arr, site, var) if k in AXIS_KEYWORDS else v for k, v in kwargs.items()}
    return f(*args, **kwargs)


def _axes(v: Any, arr: Any, site: str, var: str | None) -> Any:
    if isinstance(v, (tuple, list)) and any(axis_name(x) for x in v):
        out: list[int] = []
        for x in v:
            r = _axes(x, arr, site, var)
            out.extend(r if isinstance(r, tuple) else (r,))
        return tuple(out)
    name = axis_name(v)
    return v if name is None else resolve(name, arr, site, var)


def resolve(name: str, arr: Any, site: str, var: str | None) -> Any:
    """The position of axis ``name`` in ``arr`` (a tuple for a variadic name)."""
    from ._ops import CALLS

    shape = tuple(jnp.shape(arr))
    env, fn = CALLS.stack[-1] if CALLS.stack else (None, None)

    # 1. the variable's annotation
    spec = None
    if var is not None and env is not None:
        spec = env.locals.get(var)
        if spec is None and fn is not None and var in fn.params:
            spec = fn.params[var].spec
    dims = getattr(spec, "dims", None)
    if dims is not None:
        pos = _from_dims(name, dims, getattr(spec, "var_index", None), len(shape), env)
        if pos is not None:
            return pos

    # 2. the sizes
    where = f"{var or 'the array'}: {describe(arr)}"
    if name.startswith("*"):
        raise ShapeError(
            f"{site}: cannot tell where {name} is in `{where}`. Annotate it "
            f"(`{var or 'y'}: f32[{name}, ...] = ...`)."
        )
    size = env.dims.get(name) if env is not None else None
    if size is None:
        raise ShapeError(
            f"{site}: axis {name} is not bound in this call, so `{where}` cannot be "
            f"searched for it. Is {name} a dimension of the signature?"
        )
    found = [i for i, s in enumerate(shape) if s is size or _same(s, size)]
    key = (site, name, len(shape))
    if len(found) == 1:
        _SEEN[key] = found[0]
        return found[0]
    seen = _SEEN.get(key)
    if seen is not None and len(found) > 1 and seen in found:
        return seen
    if not found:
        raise ShapeError(f"{site}: `{where}` has no axis of size {name} = {size}.")
    raise ShapeError(
        f"{site}: `{where}` has several axes of size {name} = {size} "
        f"(axes {', '.join(map(str, found))}), so the name does not say which one. "
        f"Annotate the value (`{var or 'y'}: f32[...] = ...`), with a different name "
        f"for each axis."
    )


def fmt(dims: tuple[Any, ...]) -> str:
    return "[" + ", ".join(map(str, dims)) + "]"


def _from_dims(name: str, dims: tuple[Any, ...], var_index: int | None, rank: int, env: Any) -> Any:
    star = name.startswith("*")
    base = name[1:] if star else name
    if sum(isinstance(d, DName) and d.name == base for d in dims) > 1:
        raise ShapeError(
            f"{name} names more than one axis of `{fmt(dims)}`, so it cannot pick one. "
            f"Give the axes different names, e.g. f32[T, S] with S for the second."
        )
    for i, d in enumerate(dims):
        if star and isinstance(d, DVar) and d.name == base:
            n = len(env.vars.get(base, ())) if env is not None else rank - (len(dims) - 1)
            return tuple(range(i, i + n))
        if not star and isinstance(d, DName) and d.name == base:
            if var_index is not None and i > var_index:
                return i - len(dims)  # counted from the end, past the variadic
            return i
    return None

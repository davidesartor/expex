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

import typing
from typing import Any

import jax.numpy as jnp

from ..annotations.dims import DConst, DName, DVar, _same, describe
from ..annotations.errors import ShapeError
from ..annotations.types import Dim

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


def _size_of(name: str, env: Any, fn: Any) -> Any:
    """The size ``name`` has in this call: bound by the arguments, or defined (H = heads)."""
    if env is None:
        return None
    if name in env.dims:
        return env.dims[name]
    expr = getattr(fn, "defs", {}).get(name)
    if expr is not None and expr.names <= env.dims.keys():
        return expr(env.dims)
    return None


def dim_(name: str) -> Any:
    """A size defined in the body (``H = heads``): a name for it, like a type parameter."""
    return typing.TypeVar(name)  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]


def lifted_(i: int) -> bool:
    """Is the body's ``i``-th leading assert checked as a constraint (so the body skips it)?"""
    from .ops import CALLS

    return bool(CALLS.stack) and i in getattr(CALLS.stack[-1][1], "_lifted", ())


def size_(name: Any) -> Any:
    """``T`` used as a value inside an ``@fn``: its size in the current call.

    The converter rewrites ``jnp.zeros((B, T))`` to use this. For a variadic
    ``*B`` it is a tuple of sizes, for a derived size (``U = T + 1``) its value.
    Under the prover the sizes are symbolic, so the code is proved for all sizes.
    """
    from .ops import CALLS

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
    else:
        v = _size_of(n, env, fn)
        if v is not None:
            return v
        if isinstance(name, type) and issubclass(name, Dim) and name.size is not None:
            return name.size  # a fixed size (class Rows(Dim, size=6)) is known everywhere
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
    return v if name is None else resolve(name, arr, site, var, getattr(v, "size", None))


def resolve(name: str, arr: Any, site: str, var: str | None, fixed: int | None = None) -> Any:
    """The position of axis ``name`` in ``arr`` (a tuple for a variadic name)."""
    from .ops import CALLS

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
    size = _size_of(name, env, fn)
    if size is None:
        size = fixed  # class Rows(Dim, size=6)
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
        if not star and (isinstance(d, DName) and d.name == base or isinstance(d, DConst) and d.label == base):
            if var_index is not None and i > var_index:
                return i - len(dims)  # counted from the end, past the variadic
            return i
    return None


def einsum(*operands: Any, to: Any) -> Any:
    """Contract arrays by their axis names: ``expex.einsum(q, k, to=(B, T, S))``.

    Axes with the same name are matched, and names missing from ``to`` are
    summed over, as in ``jnp.einsum``. Each operand's names come from its
    annotation (a parameter, or an annotated local such as
    ``k_: f32[B, S, D] = k``), or else from its sizes in this call. Only inside
    an ``@fn``.
    """
    return einsum_(operands, (None,) * len(operands), to, "expex.einsum")


def einsum_(operands: tuple[Any, ...], variables: tuple[str | None, ...], to: Any, site: str) -> Any:
    """``expex.einsum`` with the operands' variable names (passed by the converter)."""
    __tracebackhide__ = True
    from .ops import CALLS

    if not CALLS.stack:
        raise ShapeError("expex.einsum() finds axes by name, so it only works inside an @fn.")
    env, fn = CALLS.stack[-1]
    outs = _names(to if isinstance(to, (tuple, list)) else (to,), env)
    twice = sorted({n for n in outs if outs.count(n) > 1})
    if twice:
        raise ShapeError(
            f"{site}: {', '.join(twice)} appears more than once in `to`. Two axes need two "
            f"names: annotate a copy of the operand with another one, e.g. "
            f"`h2: f32[T, E] = h` for `to=(D, E)`."
        )
    letters: dict[str, str] = {}
    sizes: dict[str, Any] = {}
    specs = []
    for x, var in zip(operands, variables):
        names = _operand_names(x, var, env, fn, site)
        for n, s in zip(names, jnp.shape(x)):
            if n in sizes and not (sizes[n] is s or _same(sizes[n], s)):
                raise ShapeError(
                    f"{site}: axis {n} has size {sizes[n]} in one operand and {s} in "
                    f"`{var or 'another'}: {describe(x)}`."
                )
            sizes.setdefault(n, s)
            letters.setdefault(n, chr(ord("a") + len(letters)))
        specs.append("".join(letters[n] for n in names))
    missing = [n for n in outs if n not in letters]
    if missing:
        raise ShapeError(f"{site}: `to` names {', '.join(missing)}, which no operand has.")
    return jnp.einsum(",".join(specs) + "->" + "".join(letters[n] for n in outs), *operands)


def _names(items: Any, env: Any) -> list[str]:
    out: list[str] = []
    for v in items:
        n = axis_name(v)
        if n is None:
            raise TypeError(f"expex.einsum(to=...) takes size names (B, T, ...), not {v!r}")
        if n.startswith("*"):
            out += [f"{n}{i}" for i in range(len(env.vars.get(n[1:], ())))]
        else:
            out.append(n)
    return out


def _operand_names(x: Any, var: str | None, env: Any, fn: Any, site: str) -> list[str]:
    """The axis names of one operand: from its annotation, else from its sizes."""
    shape = tuple(jnp.shape(x))
    spec = None
    if var is not None:
        head, _, field = var.partition(".")  # `self.w`: the field's annotation
        spec = env.locals.get(head)
        if spec is None and fn is not None and head in getattr(fn, "params", {}):
            spec = fn.params[head].spec
        if field:
            spec = getattr(spec, "fields", {}).get(field)
    dims = getattr(spec, "dims", None)
    if dims is not None:
        out: list[str] = []
        for d in dims:
            if isinstance(d, DVar):
                out += [f"*{d.name}{i}" for i in range(len(env.vars.get(d.name, ())))]
            elif isinstance(d, DName):
                out.append(d.name)
            else:
                break  # a literal or computed size: fall back to the sizes
        else:
            if len(out) == len(shape):
                return out
    # by size: each axis must match exactly one size of the call
    known = {n: s for n, s in env.dims.items() if not n.startswith("_")}
    for n, e in getattr(fn, "defs", {}).items():
        v = _size_of(n, env, fn)
        if v is not None:
            known[n] = v
    out = []
    for i, s in enumerate(shape):
        hits = [n for n, v in known.items() if v is s or _same(v, s)]
        if len(hits) != 1:
            what = "no size of this call" if not hits else f"{' and '.join(hits)} alike"
            raise ShapeError(
                f"{site}: axis {i} of `{var or 'an operand'}: {describe(x)}` has size {s}, "
                f"which matches {what}. Annotate the operand to name its axes: "
                f"`{var or 'y'}: f32[...] = ...`."
            )
        out.append(hits[0])
    return out


def rearrange(x: Any, *, to: Any) -> Any:
    """Move, split and merge axes by name: ``expex.rearrange(x, to=(H, T, K))``.

    For ``x: f32[T, D]`` and sizes ``H, K = heads, D // heads``, ``D`` is not in
    ``to`` but ``H`` and ``K`` are new, so ``D`` is split into ``(H, K)`` (in the
    order they appear in ``to``); the reverse merges them. Names in both are
    moved. Only inside an ``@fn``.
    """
    return rearrange_(x, None, to, "expex.rearrange")


def rearrange_(x: Any, var: str | None, to: Any, site: str) -> Any:
    """``expex.rearrange`` with the operand's variable name (passed by the converter)."""
    __tracebackhide__ = True
    from .ops import CALLS

    if not CALLS.stack:
        raise ShapeError("expex.rearrange() finds axes by name, so it only works inside an @fn.")
    env, fn = CALLS.stack[-1]
    src = _operand_names(x, var, env, fn, site)
    tgt = _names(to if isinstance(to, (tuple, list)) else (to,), env)
    size = dict(zip(src, jnp.shape(x)))
    for n in tgt:
        if n not in size:
            v = _size_of(n, env, fn)
            if v is None:
                raise ShapeError(f"{site}: the size {n} in `to` is not known in this call.")
            size[n] = v
    new, gone = [n for n in tgt if n not in src], [n for n in src if n not in tgt]
    # the finest axes: source axes, with the one(s) being split replaced by their parts
    if len(gone) == 1 and len(new) > 1 and all(n not in src for n in new):
        fine = [m for n in src for m in (new if n == gone[0] else [n])]
        target = tgt  # a split
    elif len(new) == 1 and len(gone) > 1:
        fine = src
        target = [m for n in tgt for m in (gone if n == new[0] else [n])]  # a merge
    elif not new and not gone:
        fine, target = src, tgt  # a transpose
    else:
        raise ShapeError(
            f"{site}: cannot rearrange [{', '.join(src)}] to [{', '.join(tgt)}]: one axis may be "
            f"split into several new ones, or several merged into one new one, per call."
        )
    y = jnp.reshape(x, tuple(size[n] for n in fine))
    y = jnp.transpose(y, tuple(fine.index(n) for n in target))
    return jnp.reshape(y, tuple(size[n] for n in tgt))

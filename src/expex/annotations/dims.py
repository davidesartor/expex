"""Dimensions: what the items of ``f32[B, T + 1]`` become, and how sizes are bound.

An ``Env`` holds the sizes bound during one call; ``_match_shape`` binds or
checks them against an actual shape, and ``resolve`` settles derived sizes.
``describe`` / ``show`` print values as ``f32[3, 4]``.
"""

import typing
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .errors import Mismatch, coded
from .expr import Expr
from .types import SHORT_NAMES, Dim


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


class DEllipsis:
    """Zero or more unchecked axes: the ``...`` in ``f32[..., D]``."""

    __slots__ = ()

    def __str__(self) -> str:
        return "..."


_ELLIPSIS = DEllipsis()


class DExpr:
    """A size computed from other dimensions: ``T + 1``."""

    __slots__ = ("expr", "label")

    def __init__(self, expr: Expr, label: str | None = None):
        self.expr = expr
        self.label = label

    def __str__(self) -> str:
        return self.expr.src


_ANY = DAny()


# jaxtyping's broadcastable sizes (``#b``, ``#4``, ``*#batch``): an axis of size 1 also
# matches. Subclasses, so that everything but matching treats them as the plain kind.


class DBroad(DName):
    """A named size that an axis of size 1 also matches (jaxtyping's ``#b``)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "#" + self.name


class DBroadConst(DConst):
    """A fixed size that an axis of size 1 also matches (jaxtyping's ``#4``)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "#" + super().__str__()


class DVarBroad(DVar):
    """Named axes whose occurrences need only broadcast together (jaxtyping's ``*#batch``)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "*#" + self.name


def _broadcast(a: tuple[Any, ...], b: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """The shape ``a`` and ``b`` broadcast to, or None."""
    n = max(len(a), len(b))
    a, b = (1,) * (n - len(a)) + tuple(a), (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a, b):
        if _same(x, y) or _same(y, 1):
            out.append(x)
        elif _same(x, 1):
            out.append(y)
        else:
            return None
    return tuple(out)


class DimError(TypeError):
    """An annotation's size is malformed (as opposed to: not an array annotation)."""


def param_dim(tp: Any, name: str | None = None) -> Any:
    """A type parameter as a dimension: fixed (``R = 6``, as ``R``), or a size named ``name``."""
    from .sym import fixed_param

    dim = fixed_param(tp)
    if dim is not None:
        return DConst(getattr(dim, "size"), tp.__name__)
    return DName(name or tp.__name__)


def parse_dim(item: Any, subst: dict[Any, Any]) -> Any:
    from .sym import Sym

    if isinstance(item, Sym):  # written in the annotation: T, D // heads, self.heads
        if item.size is not None:  # def f[R = 6]: R, or R + 1, a fixed size
            return DConst(item.size, item.src)
        return DName(item.src) if item.is_dim else DExpr(Expr(item.src))
    if item in subst if _hashable(item) else False:
        return subst[item]
    if isinstance(item, typing.TypeVar):
        return param_dim(item)  # def f[R = 6]: R is fixed at 6
    if isinstance(item, typing.TypeVarTuple):
        return DVar(item.__name__)
    origin = typing.get_origin(item)
    if origin is typing.Unpack:
        (inner,) = typing.get_args(item)
        if inner in subst:
            return subst[inner]
        if isinstance(inner, typing.TypeVarTuple):
            return DVar(inner.__name__)
        raise coded("EX401", TypeError(f"cannot unpack {inner!r} in a shape"))
    if type(item) is int:
        return DConst(item)
    if origin is typing.Literal:
        args = typing.get_args(item)
        if len(args) == 1 and type(args[0]) is int:
            return DConst(args[0])
        raise coded("EX401", TypeError(f"{item!r} is not a valid dimension; use Literal[<int>]"))
    if isinstance(item, typing.TypeAliasType):
        d = parse_dim(item.__value__, subst)
        if isinstance(d, DConst) and d.label is None:
            d = DConst(d.value)
        return d
    if item is typing.Any:
        return _ANY
    if item is Ellipsis:
        return _ELLIPSIS
    if isinstance(item, type) and issubclass(item, Dim):
        if item.size is not None:
            return DConst(item.size, item.__name__)
        return DName(item.__name__)
    if type(item) is slice:  # f32[R:6], the old spelling of def f[R = 6]
        raise coded("EX404", DimError(
            f"{_slice_src(item)} is not a valid dimension. A fixed size is declared with the "
            f"type parameters, def f[R = 6](...) (or R = Literal[6] for type checkers), "
            f"and written as plain R in the annotations."
        ))
    if isinstance(item, str):
        raise coded("EX401", DimError(
            f"{item!r}: sizes are type parameters, not strings. Declare them in the "
            f"signature (def f[T](...)) and compute with them directly: f32[T + 1] "
            f"(annotations are evaluated lazily, so this works as written)."
        ))
    raise coded("EX401", DimError(
        f"{item!r} is not a valid dimension. Use a type parameter, Literal[n], "
        f"a Dim subclass, Any or `...`."
    ))


def _slice_src(s: slice) -> str:
    def part(x: Any) -> str:
        return "" if x is None else getattr(x, "__name__", None) or repr(x)

    return f"`{part(s.start)}:{part(s.stop)}" + (f":{part(s.step)}`" if s.step is not None else "`")


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
    var = [i for i, d in enumerate(dims) if isinstance(d, (DVar, DEllipsis))]
    if len(var) > 1:
        raise coded("EX401", TypeError("a shape can contain at most one variadic (*Name or ...)"))
    return tuple(dims), (var[0] if var else None)


class Env:
    """Dimension bindings for one call."""

    __slots__ = ("dims", "origin", "vars", "deferred", "locals")

    def __init__(self) -> None:
        #: Annotations of local variables seen so far, for axes by name.
        self.locals: dict[str, Any] = {}
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
            elif len(unbound) == 1:
                name = unbound[0]
                solved = _solve(d.expr, name, actual, env.dims)
                if solved is None and is_symbolic(actual):  # `W + V` = `W + 27`: V is 27; else wait
                    rest.append(item)
                    continue
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


def _solve(expr: Expr, name: str, target: Any, dims: dict[str, Any]) -> int | None:
    """Solve ``expr(name) == target`` assuming ``expr`` is linear in ``name``.

    ``target`` and the other sizes may be symbolic, as long as the solution is a number.
    """
    try:
        f0 = expr({**dims, name: 0})
        f1 = expr({**dims, name: 1})
        a = f1 - f0
        if a == 0 or (target - f0) % a:
            return None
        cand = (target - f0) // a
        if not isinstance(cand, (int, np.integer)) or cand < 0 or expr({**dims, name: cand}) != target:
            return None
        return int(cand)
    except Exception:
        return None


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


_FUNC = __import__("re").compile(r"\b(floordiv|mod)\(([^(),]+), ([^(),]+)\)")


def fmt_dim(d: Any) -> str:
    """A size as text: ``floordiv(D, 4)`` from JAX's symbolic sizes reads ``D // 4``."""
    text = str(d)
    while True:
        new = _FUNC.sub(lambda m: f"{m[2]} {'//' if m[1] == 'floordiv' else '%'} {m[3]}", text)
        if new == text:
            return text
        text = new


def fmt_shape(shape: Any) -> str:
    return "[" + ", ".join(map(fmt_dim, shape)) + "]" if len(shape) else "[()]"


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


def show(tree: Any) -> str:
    """A pytree with every array replaced by its dtype and shape.

    ``show(params)`` -> ``Params(w=f32[3, 3], b=f32[3])``.
    """
    return repr(jax.tree.map(_Shown, tree)).replace("'", "")


class _Shown:
    def __init__(self, value: Any):
        self.text = describe(value)

    def __repr__(self) -> str:
        return self.text


class _From:
    """Where a size came from, formatted only when an error message needs it."""

    __slots__ = ("path", "value")

    def __init__(self, path: str, value: Any):
        self.path, self.value = path, value

    def __str__(self) -> str:
        return f"`{self.path}: {describe(self.value)}`"


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
                    env.origin["*" + d.name] = _From(path, value)  # type: ignore[assignment]
                elif len(have) != len(got) or not all(map(_same, have, got)):
                    raise Mismatch(
                        path, spec, value,
                        f"*{d.name} is {_tup(have)} (from "
                        f"{env.origin.get('*' + d.name, '?')}), but here it is {_tup(got)}.",
                    )  # fmt: skip
            elif t is DVarBroad:
                _match_broadcast_var(d, shape[axis : axis + k], path, env, spec, value)
            axis += k
            continue
        size = shape[axis]
        if t is DName:
            have = env.dims.get(d.name)
            if have is None:
                env.dims[d.name] = size
                env.origin[d.name] = _From(path, value)  # type: ignore[assignment]
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
                want = f"{d.label}, which is fixed at {d.value}" if d.label else str(d.value)
                raise Mismatch(path, spec, value, f"axis {axis} should be {want}, but is {size}.")
        elif t is DExpr:
            env.deferred.append((d, size, path, spec, value, axis))
        elif t is DBroad:
            if not _same(size, 1):
                have = env.dims.get(d.name)
                if have is None:
                    env.dims[d.name] = size
                    env.origin[d.name] = _From(path, value)  # type: ignore[assignment]
                elif not _same(have, size):
                    raise Mismatch(
                        path, spec, value,
                        f"{d.name} is {have} (from {env.origin.get(d.name, '?')}), but axis {axis} "
                        f"of `{path}` is {size}, which is neither {have} nor 1 (`#{d.name}` broadcasts).",
                    )  # fmt: skip
        elif t is DBroadConst:
            if not (_same(size, 1) or _same(d.value, size)):
                raise Mismatch(path, spec, value, f"axis {axis} should be {d.value} or 1 (`{d}`), but is {size}.")
        axis += 1


def _match_broadcast_var(d: DVarBroad, got: tuple[Any, ...], path: str, env: Env, spec: Any, value: Any) -> None:
    """``*#name``: every occurrence's axes broadcast with the others'; the name holds the result."""
    have = env.vars.get(d.name)
    if have is None:
        env.vars[d.name] = got
        env.origin["*" + d.name] = _From(path, value)  # type: ignore[assignment]
        return
    both = _broadcast(have, got)
    if both is None:
        raise Mismatch(
            path, spec, value,
            f"*{d.name} is {_tup(have)} (from {env.origin.get('*' + d.name, '?')}), and here it is "
            f"{_tup(got)}, which does not broadcast with it (`*#{d.name}` broadcasts).",
        )  # fmt: skip
    env.vars[d.name] = both

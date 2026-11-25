"""Converted operators and statements other than control flow.

``and`` / ``or`` / ``not`` and chained comparisons on traced values; in-place
updates (``x[i] += v`` becomes ``x.at[i].add(v)``); ``assert``, ``raise``,
``print``, ``min``, ``max``; and checks of annotated local variables.
"""

import operator
import threading
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from .. import keys
from ..annotations.errors import coded
from .flow import (
    _NUM,
    _PATH,
    _STAGE,
    _CannotLower,
    _is_arr,
    _select_expr,
)


class _Calls(threading.local):
    def __init__(self) -> None:
        self.stack: list[tuple[Any, Any]] = []  # (Env, Fn) of the running @fn calls


CALLS = _Calls()


_LOCAL_SPECS: dict[str, Any] = {}


def local_(value: Any, ann: Callable[[], Any], name: str, site: str, text: str | None = None) -> Any:
    """Check ``name: <annotation> = value`` against the enclosing call's dimensions."""
    if not CALLS.stack:
        return value
    env, fn = CALLS.stack[-1]
    spec = _LOCAL_SPECS.get(site, _LOCAL_SPECS)
    if spec is _LOCAL_SPECS:
        from ..annotations.spec import parse

        try:
            spec = parse(ann())
            if not spec.leafy:
                spec = None
        except Exception:  # size arithmetic, a name not defined: read from the text below
            spec = None
        if spec is None and text is not None and hasattr(fn, "_local_spec"):
            # `h: f32[B, T + 1]`, `h: "f32[T]"`: evaluated from the text as the signature's
            # annotations are; one that cannot be checked is an error, not skipped
            spec = fn._local_spec(name, text)
        _LOCAL_SPECS[site] = spec
    if spec is None:
        return value
    env.locals[name] = spec
    from ..function import config

    if config.checked:
        fn._check_local(spec, value, name, site, env)
    return value


def and_(a: Callable[[], Any], b: Callable[[], Any]) -> Any:
    x = a()
    if isinstance(x, Tracer):
        if x.shape != ():  # an extension: elementwise on boolean arrays
            return jnp.logical_and(x, b())
        # Python's `x and y` is `y if x else x`; y is only evaluated inside its branch
        return _select_expr(x, b, lambda: x, "an `and` expression")
    k = keys.sub()
    if not x:
        return x
    with keys.child(k):
        return b()


def or_(a: Callable[[], Any], b: Callable[[], Any]) -> Any:
    x = a()
    if isinstance(x, Tracer):
        if x.shape != ():
            return jnp.logical_or(x, b())
        return _select_expr(x, lambda: x, b, "an `or` expression")
    k = keys.sub()
    if x:
        return x
    with keys.child(k):
        return b()


_CMP = {
    "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge,
    "Eq": operator.eq, "NotEq": operator.ne, "Is": operator.is_, "IsNot": operator.is_not,
    "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b,
}  # fmt: skip


def _sym_eq(a: Any, b: Any, op: str) -> bool:
    """``a == b`` (``op`` "Eq") or ``a != b`` on symbolic sizes, decided soundly.

    JAX answers ``T == 6`` with False because the expressions differ, which holds
    for some sizes only: the prover would then take one branch of ``if T == 6`` as
    the only one. Decided here from the bounds JAX can prove; otherwise it is
    inconclusive, like ``T > 1``, and the prover splits on it.
    """
    from jax._src.export.shape_poly import InconclusiveDimensionOperation

    def decide(f: Callable[[], Any]) -> bool | None:
        try:
            return bool(f())
        except InconclusiveDimensionOperation:
            return None

    ge, le = decide(lambda: a >= b), decide(lambda: a <= b)
    if ge is False or le is False:
        eq = False
    elif ge and le:
        eq = True
    else:
        sign = "==" if op == "Eq" else "!="
        raise InconclusiveDimensionOperation(f"Symbolic dimension comparison '{a}' {sign} '{b}' is inconclusive.")
    return eq if op == "Eq" else not eq


def _symbolic(v: Any) -> bool:
    """A symbolic size, or a shape (a tuple or list) with one."""
    t = type(v)
    if t is int or t is bool or t is float or t is str or v is None:
        return False
    if t is tuple or t is list:  # a shape, or a few sizes (`n in (4, 5)`)
        return len(v) <= 16 and any(type(x) not in (int, bool) and _symbolic(x) for x in v)
    return not isinstance(v, (Tracer, jax.Array, np.ndarray)) and jax.export.is_symbolic_dim(v)


def _sym_cmp(a: Any, b: Any, op: str) -> bool:
    """``==`` / ``!=`` / ``in`` / ``not in`` with a symbolic size on either side, decided soundly."""
    if op == "NotEq":
        return not _sym_cmp(a, b, "Eq")
    if op == "NotIn":
        return not _sym_cmp(a, b, "In")
    if op == "In":
        if not isinstance(b, (tuple, list)):
            return a in b
        from jax._src.export.shape_poly import InconclusiveDimensionOperation

        unsure: BaseException | None = None
        for x in b:
            try:
                if _sym_cmp(a, x, "Eq"):
                    return True
            except InconclusiveDimensionOperation as e:
                unsure = e
        if unsure is not None:
            raise unsure
        return False
    seq = (tuple, list)
    if isinstance(a, seq) or isinstance(b, seq):  # shapes
        if not (isinstance(a, seq) and isinstance(b, seq)) or type(a) is not type(b) or len(a) != len(b):
            return a == b
        from jax._src.export.shape_poly import InconclusiveDimensionOperation

        unsure = None
        for x, y in zip(a, b):
            try:
                if not _sym_cmp(x, y, "Eq"):
                    return False  # one axis differs for sure: decided, even if another is unsure
            except InconclusiveDimensionOperation as e:
                unsure = e
        if unsure is not None:
            raise unsure
        return True
    if not (_symbolic(a) or _symbolic(b)):
        return a == b
    return _sym_eq(a, b, "Eq")


def cmp_(left: Any, ops: tuple[str, ...], *rights: Callable[[], Any]) -> Any:
    """A comparison ``a < b`` or ``a < b < c``: each operand is evaluated once.

    Converted where a comparison may meet a symbolic size (the prover's ``T``):
    ``T == 6`` is then decided soundly, or inconclusive (see :func:`_sym_eq`).
    """
    if len(ops) == 1:  # exactly Python's `a op b` (no truth test of the result)
        op, right = ops[0], rights[0]()
        if op in ("Eq", "NotEq", "In", "NotIn") and (_symbolic(left) or _symbolic(right)):
            return _sym_cmp(left, right, op)
        return _CMP[op](left, right)
    acc: Any = None
    r: Any = True
    for op, thunk in zip(ops, rights):
        right = thunk()
        if op in ("Eq", "NotEq", "In", "NotIn") and (_symbolic(left) or _symbolic(right)):
            r = _sym_cmp(left, right, op)
        else:
            r = _CMP[op](left, right)
        if acc is not None:
            r = jnp.logical_and(acc, r)
        if isinstance(r, Tracer) or jnp.ndim(r) > 0:
            acc = r  # cannot short-circuit on a traced value, or elementwise on arrays
        elif not r:
            return r
        left = right
    return r if acc is None else acc


def not_(x: Any) -> Any:
    if isinstance(x, Tracer):
        return jnp.logical_not(x)
    return not x


def _leaf(v: Any) -> bool:
    return _is_arr(v) or type(v) in _NUM


def _no_mutation(obj: Any, what: str) -> None:
    if _STAGE.depth:
        raise coded("EX209", _CannotLower(
            f"{what} on a {type(obj).__name__} inside traced control flow: the update "
            f"would happen once while tracing, not once per iteration or branch. "
            f"Use a JAX array (updated functionally) or move it out of the traced code"
        ))


_IOP = {
    "Add": operator.iadd, "Sub": operator.isub, "Mult": operator.imul, "Div": operator.itruediv,
    "FloorDiv": operator.ifloordiv, "Mod": operator.imod, "Pow": operator.ipow,
    "MatMult": operator.imatmul, "BitAnd": operator.iand, "BitOr": operator.ior,
    "BitXor": operator.ixor, "LShift": operator.ilshift, "RShift": operator.irshift,
}  # fmt: skip


slice_ = slice  # the builtin, reachable even where the name `slice` is shadowed


def _mask(obj: Any, index: Any) -> Any:
    """``index`` as a mask broadcastable against ``obj`` if it is a boolean array, else None."""
    if isinstance(index, (jax.Array, np.ndarray)) and index.dtype == np.bool_ and index.ndim:
        return index.reshape(index.shape + (1,) * (obj.ndim - index.ndim))
    return None


def setitem_(value: Any, obj: Any, index: Any) -> Any:
    """``obj[index] = value``; for a JAX array, ``obj.at[index].set(value)``."""
    if isinstance(obj, jax.Array):
        mask = _mask(obj, index)
        if mask is not None:  # a boolean mask keeps the shape fixed, so it can be traced
            return jnp.where(mask, value, obj).astype(obj.dtype)
        return obj.at[index].set(value)
    _no_mutation(obj, "item assignment")
    obj[index] = value
    return obj


def augitem_(obj: Any, index: Any, value: Callable[[], Any], op: str) -> Any:
    """``obj[index] op= value``; for a JAX array a functional update with NumPy's semantics.

    As in Python, the old element is read before the right-hand side is evaluated.
    """
    if isinstance(obj, jax.Array):
        mask = _mask(obj, index)
        if mask is not None:
            return jnp.where(mask, _IOP[op](obj, value()), obj).astype(obj.dtype)
        return obj.at[index].set(_IOP[op](obj[index], value()))
    _no_mutation(obj, "item assignment")
    old = obj[index]
    obj[index] = _IOP[op](old, value())
    return obj


def _replaceable(obj: Any) -> bool:
    if isinstance(obj, tuple) and hasattr(obj, "_replace"):
        return True
    params = getattr(type(obj), "__dataclass_params__", None)
    return bool(params is not None and params.frozen)


def _replace(obj: Any, name: str, value: Any) -> Any:
    from ..annotations.spec import rebuild

    return rebuild(obj, **{name: value})


def setattr_(value: Any, obj: Any, name: str) -> Any:
    """``obj.name = value``; NamedTuples and frozen dataclasses are replaced instead."""
    if _replaceable(obj):
        return _replace(obj, name, value)
    _no_mutation(obj, "attribute assignment")
    setattr(obj, name, value)
    return obj


def augattr_(obj: Any, name: str, value: Callable[[], Any], op: str) -> Any:
    new = _IOP[op](getattr(obj, name), value())
    if _replaceable(obj):
        return _replace(obj, name, new)
    _no_mutation(obj, "attribute assignment")
    setattr(obj, name, new)
    return obj


def print_(*args: Any, **kwargs: Any) -> None:
    """``print`` that also works on traced values (prints when the code runs)."""
    idx = [i for i, a in enumerate(args) if isinstance(a, Tracer)]
    if not idx:
        print(*args, **kwargs)
        return
    template = [None if i in idx else a for i, a in enumerate(args)]

    def show(*values: Any) -> None:
        out = list(template)
        for i, v in zip(idx, values):
            out[i] = v
        print(*out, **kwargs)

    jax.debug.callback(show, *[args[i] for i in idx])


def raise_(exc: Callable[[], Any], cause: Callable[[], Any] | None, site: str) -> None:
    """``raise exc() from cause()``.

    In a traced branch the exception cannot be raised while tracing (both
    branches are traced); it is raised when the compiled code reaches it.
    """
    if not _PATH.preds:
        e = exc()
        if cause is None:
            raise e
        raise e from cause()
    e = exc()
    if isinstance(e, type):
        e = e()
    reached = jnp.asarray(True)
    for pred in _PATH.preds:
        reached = jnp.logical_and(reached, pred())

    def check(hit: Any) -> None:
        if np.any(hit):
            raise e

    jax.debug.callback(check, reached)


def _extreme(args: tuple[Any, ...], kwargs: dict[str, Any], builtin: Any, pair: Any) -> Any:
    """``max`` / ``min`` as plain Python means them, wherever Python can answer.

    With no traced argument, the builtin: ``max(x[0], 1)`` on a concrete ``x``
    is ``1`` (an int, usable as an index) when Python says so, not a float
    array. Only traced arguments whose values are unknown while tracing go
    through ``jnp.maximum`` / ``jnp.minimum``; the concrete ones are first
    reduced by the builtin.
    """
    if kwargs or len(args) < 2:
        return builtin(*args, **kwargs)
    traced = [a for a in args if isinstance(a, Tracer)]
    if not traced:
        return builtin(*args)
    try:  # traced, but with values Python can compare (under jax.grad, outside jit)
        return builtin(*args)
    except (jax.errors.ConcretizationTypeError, ValueError):  # ValueError: several elements
        pass
    out = traced[0]
    for a in traced[1:]:
        out = pair(out, a)
    rest = [a for a in args if not isinstance(a, Tracer)]
    if rest:
        try:
            concrete = builtin(*rest) if len(rest) > 1 else rest[0]
        except (TypeError, ValueError):  # arrays with several elements: compared element-wise
            concrete = rest[0]
            for a in rest[1:]:
                concrete = pair(concrete, a)
        out = pair(out, concrete)
    return out


def max_(*args: Any, **kwargs: Any) -> Any:
    return _extreme(args, kwargs, max, jnp.maximum)


def min_(*args: Any, **kwargs: Any) -> Any:
    return _extreme(args, kwargs, min, jnp.minimum)


def assert_(test: Any, msg: Callable[[], Any] | None, site: str) -> None:
    """``assert``; a traced condition is checked when the compiled code runs."""
    if not isinstance(test, Tracer):
        if not test:
            raise AssertionError(*(() if msg is None else (msg(),)))
        return
    import linecache

    fname, _, line = site.rpartition(":")
    text = linecache.getline(fname, int(line)).strip() if line.isdigit() else ""
    detail = "" if msg is None else f": {msg()}"
    message = f"assertion failed at {site}{detail}" + (f"\n    > {text}" if text else "")

    def check(ok: Any) -> None:
        if not np.all(ok):
            raise AssertionError(message)

    # Under vmap both branches of a traced `if` run for every element, so the
    # assertion only counts where the code around it was really reached.
    ok = jnp.all(test)
    for reached in _PATH.preds:
        ok = jnp.logical_or(ok, jnp.logical_not(reached()))
    jax.debug.callback(check, ok)

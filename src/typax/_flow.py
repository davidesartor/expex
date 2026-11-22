"""Runtime for converted control flow.

``@fn`` rewrites ``if`` / ``while`` / ``for`` statements into calls to the
helpers in this module (see ``_convert.py``). Each helper decides *when it
runs* what to do:

* the condition / iterable is an ordinary Python value: behave exactly like
  the original statement;
* it is traced: use ``lax.cond`` / ``lax.while_loop`` / ``lax.scan`` (or
  ``vmap`` when iterations are independent).

Every helper receives the variables the statement assigns as a tuple
(``state``) together with their ``names``, and returns the new tuple.
"""

from __future__ import annotations

import operator
import threading
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax.core import Tracer

from . import _rng
from ._errors import StageError

__all__ = [
    "if_", "ifexp", "while_", "for_", "and_", "or_", "not_", "cmp_", "ld", "ret_",
    "range_", "enumerate_", "zip_", "reversed_", "rows", "stack", "append_",
    "setitem_", "augitem_", "setattr_", "augattr_", "print_", "max_", "min_", "assert_", "raise_", "slice_", "local_", "Stacked", "UNDEF", "LOG",
]  # fmt: skip

#: site ("file:line") -> how the statement there was last executed.
LOG: dict[str, str] = {}


class _Undefined:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<undefined>"

    def __bool__(self) -> bool:
        raise NameError("this variable is not defined on every path that reaches here")


UNDEF = _Undefined()


def ret_(thunk: Callable[[], Any]) -> Any:
    """The function's return value; ``None`` if no ``return`` statement set it."""
    try:
        value = thunk()
    except NameError:
        return None
    return None if value is UNDEF else value


def ld(thunk: Callable[[], Any]) -> Any:
    """Read a variable that may not be defined yet."""
    try:
        return thunk()
    except NameError:
        return UNDEF


# ------------------------------------------------------------ checked locals


class _Calls(threading.local):
    def __init__(self) -> None:
        self.stack: list[tuple[Any, Any]] = []  # (Env, Fn) of the running @fn calls


CALLS = _Calls()
_LOCAL_SPECS: dict[str, Any] = {}


def local_(value: Any, ann: Callable[[], Any], name: str, site: str) -> Any:
    """Check ``name: <annotation> = value`` against the enclosing call's dimensions."""
    if not CALLS.stack:
        return value
    spec = _LOCAL_SPECS.get(site, _LOCAL_SPECS)
    if spec is _LOCAL_SPECS:
        from ._spec import parse

        try:
            spec = parse(ann())
            if not spec.leafy:
                spec = None
        except Exception:  # not a typax annotation, or names not available here
            spec = None
        _LOCAL_SPECS[site] = spec
    if spec is None:
        return value
    env, fn = CALLS.stack[-1]
    fn._check_local(spec, value, name, site, env)
    return value


# ------------------------------------------------------------------ values


def _is_arr(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic))


_NUM = (bool, int, float, complex)


def _function(v: Any) -> bool:
    """A function or other callable that is not data (a module with ``__call__`` is data)."""
    return callable(v) and not getattr(type(v), "__typax_struct__", False)


def _dyn(v: Any) -> bool:
    """Can ``v`` be carried through lax control flow (a pytree of arrays/numbers)?"""
    if v is UNDEF or v is None:
        return False
    if _is_arr(v) or type(v) in _NUM:
        return True
    if isinstance(v, (str, bytes, _Collector, Stacked)) or _function(v):
        return False
    try:
        leaves = jax.tree.leaves(v)
    except Exception:
        return False
    return bool(leaves) and all(_is_arr(x) or type(x) in _NUM for x in leaves)


def _same_static(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


def _desc(v: Any) -> str:
    from ._spec import describe

    if v is UNDEF:
        return "undefined"
    if _is_arr(v) or type(v) in _NUM:
        return describe(v)
    if _dyn(v):
        return f"{type(v).__name__} of arrays"
    return f"{type(v).__name__} {v!r}"[:60]


# --------------------------------------------------------------------- bool


def _boolean(v: Any) -> bool:
    return type(v) is bool or getattr(v, "dtype", None) == np.bool_


def and_(a: Callable[[], Any], b: Callable[[], Any]) -> Any:
    x = a()
    if isinstance(x, Tracer):
        if x.shape != ():  # an extension: elementwise on boolean arrays
            return jnp.logical_and(x, b())
        # Python's `x and y` is `y if x else x`; y is only evaluated inside its branch
        return _select_expr(x, b, lambda: x, "an `and` expression")
    k = _rng.sub()
    if not x:
        return x
    with _rng.child(k):
        return b()


def or_(a: Callable[[], Any], b: Callable[[], Any]) -> Any:
    x = a()
    if isinstance(x, Tracer):
        if x.shape != ():
            return jnp.logical_or(x, b())
        return _select_expr(x, lambda: x, b, "an `or` expression")
    k = _rng.sub()
    if x:
        return x
    with _rng.child(k):
        return b()


_CMP = {
    "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge,
    "Eq": operator.eq, "NotEq": operator.ne, "Is": operator.is_, "IsNot": operator.is_not,
    "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b,
}  # fmt: skip


def cmp_(left: Any, ops: tuple[str, ...], *rights: Callable[[], Any]) -> Any:
    """A chained comparison ``a < b < c``: each operand is evaluated once."""
    acc: Any = None
    for op, thunk in zip(ops, rights):
        right = thunk()
        r = _CMP[op](left, right)
        if acc is not None:
            r = jnp.logical_and(acc, r)
        if isinstance(r, Tracer):
            acc = r  # cannot short-circuit on a traced value
        elif not r:
            return r
        left = right
    return True if acc is None else acc


def not_(x: Any) -> Any:
    if isinstance(x, Tracer):
        return jnp.logical_not(x)
    return not x


# ----------------------------------------------------------------------- if


def _check_pred(pred: Any, what: str, site: str) -> None:
    if pred.shape != ():
        raise StageError(
            f"the condition of the `{what}` at {site} is a traced array of shape "
            f"{tuple(pred.shape)}, not a single True/False.\n"
            f"    - to test all/any elements:   cond.all()  /  cond.any()\n"
            f"    - to choose per element:      jnp.where(cond, a, b)"
        )


def _cond(
    pred: Any,
    fns: tuple[Callable[[], tuple[Any, ...]], Callable[[], tuple[Any, ...]]],
    names: tuple[str, ...],
    site: str,
) -> tuple[Any, ...]:
    """``lax.cond`` over branches returning tuples that may mix arrays and other values."""
    plan: dict[int, Any] = {}
    recs: list[list[tuple[str, Any]]] = []
    res: Any = ()
    for attempt in range(3):
        recs = []

        def wrap(fn: Callable[[], tuple[Any, ...]]) -> Callable[[], tuple[Any, ...]]:
            def w() -> tuple[Any, ...]:
                with _staged():
                    out = fn()
                metas: list[tuple[str, Any]] = []
                dyn: list[Any] = []
                for i, v in enumerate(out):
                    p = plan.get(i)
                    if isinstance(p, dict):
                        # assigned in the other branch only: a placeholder here, so
                        # that the variable exists whichever branch ran
                        if _dyn(v):  # same (strongly typed) dtypes as the placeholder
                            v = jax.tree.map(lambda a, s: _cast(a, s.dtype, False), v, p["fill"])
                        else:
                            v = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), p["fill"])
                        metas.append(("dyn", v))
                        dyn.append(v)
                    elif _dyn(v):
                        if p is not None:
                            leaves, tree = jax.tree.flatten(v)
                            tree_: Any = tree
                            v = tree_.unflatten([_cast(x, d, w) for x, (d, w) in zip(leaves, p)])
                        metas.append(("dyn", v))
                        dyn.append(v)
                    else:
                        metas.append(("static", v))
                recs.append(metas)
                return tuple(dyn)

            return w

        err: TypeError | None = None
        try:
            res = lax.cond(pred, wrap(fns[0]), wrap(fns[1]))
        except TypeError as e:
            if len(recs) < 2:
                raise
            err = e
        new_plan = _plan(recs, names, site)
        new_plan = {i: p for i, p in new_plan.items() if not isinstance(plan.get(i), dict)}
        if new_plan and attempt < 2:
            plan = {**plan, **new_plan}
            continue
        if err is not None:
            err.add_note(f"typax: in the `if` at {site}; variables it assigns: {', '.join(names)}")
            raise err
        break

    it = iter(res)
    out: list[Any] = []
    for i, (m0, m1) in enumerate(zip(recs[0], recs[1])):
        if m0[0] == "dyn":
            out.append(next(it))
        elif _same_static(m0[1], m1[1]):
            out.append(m0[1])
        elif m0[1] is UNDEF or m1[1] is UNDEF:
            # set on one path only: the value from that path (like the zero
            # placeholder arrays get), since the other path must not read it
            out.append(m1[1] if m0[1] is UNDEF else m0[1])
        else:
            raise StageError(
                f"`{names[i]}` is set to different non-array values in the two branches of "
                f"the traced `if` at {site}: {_desc(m0[1])} vs {_desc(m1[1])}.\n"
                f"    A traced condition can only select between arrays of the same shape "
                f"and dtype."
            )
    return tuple(out)


def _plan(recs: list[list[tuple[str, Any]]], names: tuple[str, ...], site: str) -> dict[int, Any]:
    """Work out how to reconcile the two branches' outputs, or raise if impossible."""
    plan: dict[int, Any] = {}
    for i, (m0, m1) in enumerate(zip(recs[0], recs[1])):
        if m0[0] != m1[0]:
            other = m1[1] if m0[0] == "dyn" else m0[1]
            if other is UNDEF:
                given = m0[1] if m0[0] == "dyn" else m1[1]
                plan[i] = {"fill": _shapes(given)}
                continue
            a, b = _desc(m0[1]), _desc(m1[1])
            raise StageError(
                f"`{names[i]}` is {a} in one branch of the traced `if` at {site} and {b} "
                f"in the other.\n    Both branches must give it an array of the same shape "
                f"and dtype."
            )
        if m0[0] == "dyn":
            t0: Any
            t1: Any
            l0, t0 = jax.tree.flatten(m0[1])
            l1, t1 = jax.tree.flatten(m1[1])
            if t0 != t1:
                raise StageError(
                    f"`{names[i]}` has a different structure in the two branches of the "
                    f"traced `if` at {site}: {_desc(m0[1])} vs {_desc(m1[1])}."
                )
            casts = []
            change = False
            for v0, v1 in zip(l0, l1):
                if jnp.shape(v0) != jnp.shape(v1):
                    raise StageError(
                        f"`{names[i]}` has shape {_desc(v0)} in one branch of the traced `if` "
                        f"at {site} and {_desc(v1)} in the other.\n    Both branches must "
                        f"give it the same shape and dtype."
                    )
                d0, d1 = jnp.result_type(v0), jnp.result_type(v1)
                change = change or d0 != d1 or _weak(v0) != _weak(v1)
                casts.append((jnp.result_type(v0, v1), _weak(v0) and _weak(v1)))
            if change:
                plan[i] = casts
    return plan


def _shapes(tree: Any) -> Any:
    return jax.tree.map(lambda a: jax.ShapeDtypeStruct(jnp.shape(a), jnp.result_type(a)), tree)


def _probe(
    body: Callable[..., tuple[Any, ...]], state: tuple[Any, ...], idxs: list[int]
) -> dict[int, Any]:
    """Zero placeholders for the variables ``idxs`` that a loop body assigns.

    A variable first assigned inside a traced loop needs a value of the right
    shape before the loop starts. The body is traced once, abstractly, to learn
    it, with every array or number in ``state`` traced as it will be in the loop
    (a Python counter must not make a branch look untaken).
    """
    dyn = [i for i, v in enumerate(state) if _dyn(v)]

    def f(*values: Any) -> dict[int, Any]:
        full = list(state)
        for i, v in zip(dyn, values):
            full[i] = v
        with _staged():
            out = body(*full)
        return {i: out[i] for i in idxs if _dyn(out[i])}

    try:
        shapes = jax.eval_shape(f, *[state[i] for i in dyn])
    except Exception:
        return {}
    return {i: jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), sh) for i, sh in shapes.items()}


def _cast(x: Any, dtype: Any, weak: bool) -> Any:
    """Convert to ``dtype``, keeping the value weakly typed if asked (as Python scalars are)."""
    try:
        from jax._src.lax.lax import _convert_element_type

        return _convert_element_type(x, dtype, weak_type=weak)
    except Exception:
        return jnp.asarray(x, dtype)


def _weaken(i: Any) -> Any:
    """A traced loop index that promotes like the Python int it replaces."""
    try:
        from jax._src.lax.lax import _convert_element_type

        return _convert_element_type(i, weak_type=True)
    except Exception:
        return i


def _leaf(v: Any) -> bool:
    return _is_arr(v) or type(v) in _NUM


def _weak(v: Any) -> bool:
    if type(v) in _NUM:
        return True
    return bool(getattr(getattr(v, "aval", None), "weak_type", False) or getattr(v, "weak_type", False))


def _guard_lists(state: tuple[Any, ...], names: tuple[str, ...], what: str, site: str) -> Callable[[tuple[Any, ...]], None]:
    """Detect in-place growth of Python lists inside traced control flow."""
    lens = {i: len(v) for i, v in enumerate(state) if isinstance(v, list)}
    colls = [i for i, v in enumerate(state) if isinstance(v, _Collector)]
    if colls:
        raise _CannotLower(f"`{names[colls[0]]}.append(...)` inside a traced `{what}`")

    def check(out: tuple[Any, ...]) -> None:
        for i, n in lens.items():
            if out[i] is state[i] and len(out[i]) != n:
                raise StageError(
                    f"`{names[i]}.append(...)` runs inside the traced `{what}` at {site}, so "
                    f"the list's length would depend on data.\n"
                    f"    Append unconditionally and select the value instead: "
                    f"{names[i]}.append(jnp.where(cond, a, b))."
                )

    return check


class _Path(threading.local):
    def __init__(self) -> None:
        self.preds: list[Callable[[], Any]] = []  # conditions under which the code being traced runs


_PATH = _Path()


class _on_path:
    """While tracing a branch: remember the condition under which it really runs."""

    def __init__(self, pred: Any):
        self.pred = pred if callable(pred) else (lambda: pred)

    def __enter__(self) -> None:
        _PATH.preds.append(self.pred)

    def __exit__(self, *exc: Any) -> None:
        _PATH.preds.pop()


def _truth(pred: Any) -> Any:
    return pred if _boolean(pred) else pred != 0


def if_(
    pred: Any,
    true_fn: Callable[..., tuple[Any, ...]],
    false_fn: Callable[..., tuple[Any, ...]],
    state: tuple[Any, ...],
    names: tuple[str, ...],
    site: str,
) -> tuple[Any, ...]:
    k = _rng.sub()
    if not isinstance(pred, Tracer):
        with _rng.child(k):
            return (true_fn if pred else false_fn)(*state)
    _check_pred(pred, "if", site)
    check = _guard_lists(state, names, "if", site)
    taken = _truth(pred)

    def run(fn: Callable[..., tuple[Any, ...]], on: Any) -> Callable[[], tuple[Any, ...]]:
        def r() -> tuple[Any, ...]:
            with _rng.child(k), _on_path(on):
                out = fn(*state)
            check(out)
            return out

        return r

    LOG[site] = "if -> lax.cond"
    return _cond(pred, (run(true_fn, taken), run(false_fn, ~taken)), names, site)


def ifexp(pred: Any, a: Callable[[], Any], b: Callable[[], Any], site: str) -> Any:
    if not isinstance(pred, Tracer):
        k = _rng.sub()
        with _rng.child(k):
            return a() if pred else b()
    return _select_expr(pred, a, b, site)


def _select_expr(pred: Any, a: Callable[[], Any], b: Callable[[], Any], site: str) -> Any:
    """``a() if pred else b()`` for a traced scalar ``pred``; each side runs in its branch."""
    _check_pred(pred, "... if ... else ...", site)
    k = _rng.sub()
    taken = _truth(pred)

    def side(fn: Callable[[], Any], on: Any) -> Callable[[], tuple[Any, ...]]:
        def r() -> tuple[Any, ...]:
            with _rng.child(k), _on_path(on):
                return (fn(),)

        return r

    return _cond(pred, (side(a, taken), side(b, ~taken)), ("<the expression>",), site)[0]


# -------------------------------------------------------------------- while


class _CannotLower(StageError):
    """A loop cannot be turned into lax control flow; the reason is the message."""


class _Stage(threading.local):
    depth = 0  # > 0 while the body of a staged if / while / for is being traced


_STAGE = _Stage()


class _staged:
    def __enter__(self) -> None:
        _STAGE.depth += 1

    def __exit__(self, *exc: Any) -> None:
        _STAGE.depth -= 1


# ------------------------------------------------------- in-place updates


def _no_mutation(obj: Any, what: str) -> None:
    if _STAGE.depth:
        raise _CannotLower(
            f"{what} on a {type(obj).__name__} inside traced control flow: the update "
            f"would happen once while tracing, not once per iteration or branch. "
            f"Use a JAX array (updated functionally) or move it out of the traced code"
        )


_AT = {"Add": "add", "Sub": "subtract", "Mult": "multiply", "Div": "divide", "Pow": "power"}
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
    from ._spec import rebuild

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


# --------------------------------------------------------- staged builtins


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
    # With a JAX array among several arguments, use the array function in every
    # mode, so that eager and compiled runs agree on dtype and on NaN handling.
    if kwargs or len(args) < 2 or not any(isinstance(a, jax.Array) for a in args):
        return builtin(*args, **kwargs)
    out = args[0]
    for a in args[1:]:
        out = pair(out, a)
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


def while_(
    test_fn: Callable[..., Any],
    body_fn: Callable[..., tuple[Any, ...]],
    state: tuple[Any, ...],
    names: tuple[str, ...],
    live: tuple[str, ...],
    site: str,
) -> tuple[Any, ...]:
    k = _rng.sub()
    it = 0
    while True:
        with _rng.child(k, 2 * it):
            c = test_fn(*state)
        if isinstance(c, Tracer):
            break
        if not c:
            return state
        with _rng.child(k, 2 * it + 1):
            state = body_fn(*state)
        it += 1

    _check_pred(c, "while", site)
    truth = (lambda v: v) if _boolean(c) else (lambda v: v != 0)
    try:
        check = _guard_lists(state, names, "while", site)
    except _CannotLower as e:
        raise StageError(f"{e} at {site}: the number of iterations depends on data.") from None
    need = [i for i, v in enumerate(state) if v is UNDEF and names[i] in live]
    if need:
        zeros = _probe(body_fn, state, need)
        state = tuple(zeros.get(i, v) for i, v in enumerate(state))
    dyn = [i for i, v in enumerate(state) if _dyn(v)]
    init = tuple(state[i] for i in dyn)
    count0 = jnp.asarray(it, jnp.int32)

    def merge(carry: tuple[Any, ...]) -> list[Any]:
        full = list(state)
        for i, v in zip(dyn, carry):
            full[i] = v
        return full

    def cond_f(c2: Any) -> Any:
        n, carry = c2
        with _rng.child(k, 2 * n):
            return truth(test_fn(*merge(carry)))

    def body_f(c2: Any) -> Any:
        n, carry = c2
        still = lambda: truth(test_fn(*merge(carry)))  # noqa: E731 - only evaluated by an assert
        with _rng.child(k, 2 * n + 1), _staged(), _on_path(still):
            out = body_fn(*merge(carry))
        check(out)
        for i, v in enumerate(out):
            if i in dyn:
                continue
            if state[i] is UNDEF:
                if names[i] in live and v is not UNDEF:
                    raise StageError(
                        f"`{names[i]}` is first assigned inside the traced `while` at {site} "
                        f"and used after it. Give it a value before the loop."
                    )
            elif not _same_static(v, state[i]):
                raise StageError(
                    f"`{names[i]}` ({_desc(state[i])}) changes inside the traced `while` at "
                    f"{site}; only arrays can change in a traced loop."
                )
        return n + 1, tuple(out[i] for i in dyn)

    LOG[site] = "while -> lax.while_loop"
    try:
        _, final = lax.while_loop(cond_f, body_f, (count0, init))
    except TypeError as e:
        e.add_note(
            f"typax: in the `while` at {site}; carried variables, in order: "
            f"{', '.join(names[i] for i in dyn)}.\n"
            f"    Each must keep the same shape and dtype on every iteration."
        )
        raise
    return tuple(merge(final))


# ---------------------------------------------------------------------- for


class _Collector:
    """Stands in for a list inside a lowered loop body; records what is appended."""

    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: list[Any] = []


class Stacked(Sequence):  # type: ignore[type-arg]
    """The result of ``ys.append(...)`` in a lowered loop: a list stored as one array.

    Behaves like a read-only list of per-iteration values. ``jnp.asarray(ys)``
    returns the stacked array at no cost.
    """

    __slots__ = ("array",)

    def __init__(self, array: Any):
        self.array = array

    def __len__(self) -> int:
        return int(jax.tree.leaves(self.array)[0].shape[0])

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return Stacked(jax.tree.map(lambda a: a[i], self.array))
        return jax.tree.map(lambda a: a[i], self.array)

    def __iter__(self) -> Any:
        return (self[i] for i in range(len(self)))

    def __jax_array__(self) -> Any:
        return self.array

    def __add__(self, other: Any) -> list[Any]:
        return list(self) + list(other)

    def __radd__(self, other: Any) -> list[Any]:
        return list(other) + list(self)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (list, Stacked)) and list(self) == list(other)

    __hash__ = None  # type: ignore[assignment]

    def __array__(self, *a: Any, **k: Any) -> Any:
        return np.asarray(self.array, *a, **k)

    def __repr__(self) -> str:
        return f"Stacked({self.array!r})"


def unstack(out: Any) -> Any:
    """Turn ``Stacked`` values in a function's result back into real lists."""
    t = type(out)
    if t is Stacked:
        return list(out)
    if t is tuple:
        return tuple(unstack(v) for v in out) if any(_has_stacked(v) for v in out) else out
    if t is list:
        return [unstack(v) for v in out] if any(_has_stacked(v) for v in out) else out
    if t is dict:
        return {k: unstack(v) for k, v in out.items()} if any(map(_has_stacked, out.values())) else out
    return out


def _has_stacked(v: Any) -> bool:
    t = type(v)
    if t is Stacked:
        return True
    if t in (tuple, list):
        return any(_has_stacked(x) for x in v)
    if t is dict:
        return any(_has_stacked(x) for x in v.values())
    return False


def append_(lst: Any, item: Any) -> Any:
    """``lst.append(item)``, returning the list (so that loops can collect)."""
    if type(lst) is _Collector:
        lst.items.append(item)
        return lst
    if type(lst) is Stacked:
        lst = list(lst)
    lst.append(item)
    return lst


class _Iter:
    """An iterable the loop lowering understands: ``n`` items built by ``make(x, i)``."""

    def __init__(
        self,
        n: Any,
        xs: Any,
        make: Callable[[Any, Any], Any],
        py: Callable[[], Any],
        chunk: int | None = None,
    ):
        self.n = n
        self.xs = xs
        self.make = make
        self.py = py
        self.chunk = chunk  # run independent iterations `chunk` at a time, in sequence

    def __iter__(self) -> Any:
        return iter(self.py())


def _as_iter(it: Any) -> _Iter | None:
    if isinstance(it, _Iter):
        return it
    if isinstance(it, range):
        start, step = it.start, it.step
        return _Iter(len(it), (), lambda x, i: start + _weaken(i) * step, lambda: it)
    if _is_arr(it) and getattr(it, "ndim", 0) >= 1:
        return _Iter(it.shape[0], it, lambda x, i: x, lambda: it)
    if isinstance(it, Stacked):
        return _Iter(len(it), it.array, lambda x, i: x, lambda: it)
    if isinstance(it, rows):
        return _Iter(len(it), it.tree, lambda x, i: x, lambda: it, it.chunk)
    if isinstance(it, (list, tuple)) and len(it) >= 2:
        stacked = _stack_alike(it)
        if stacked is not None:  # e.g. a list of identical layers: one scan, not n copies
            return _Iter(len(it), stacked, lambda x, i: x, lambda: it)
    return None


def _stack_alike(items: Any) -> Any:
    """Stack a list of pytrees with identical structure, shapes and dtypes, or None."""
    first_tree: Any
    first_leaves, first_tree = jax.tree.flatten(items[0])
    if not first_leaves or not all(isinstance(x, jax.Array) for x in first_leaves):
        return None
    sig = [(x.shape, x.dtype) for x in first_leaves]
    for item in items[1:]:
        leaves, tree = jax.tree.flatten(item)
        if tree != first_tree or len(leaves) != len(sig):
            return None
        if any(not isinstance(x, jax.Array) or (x.shape, x.dtype) != s for x, s in zip(leaves, sig)):
            return None
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


class rows:
    """Iterate over the leading axis of every array in a pytree at once.

    ``for batch in rows(data)`` yields ``data`` with each leaf indexed by
    ``[0]``, ``[1]``, ... Inside ``@fn`` such a loop is staged like a loop over
    a single array.
    """

    def __init__(self, tree: Any, chunk: int | None = None):
        # Independent iterations normally run all at once (vmap). With chunk=k
        # they run k at a time, one chunk after another, like
        # jax.lax.map(..., batch_size=k): bounded memory, and one slow
        # iteration does not hold up the rest. chunk=1 is fully sequential.
        self.chunk = chunk
        leaves = jax.tree.leaves(tree)
        if not leaves:
            raise ValueError("rows() needs a pytree with at least one array")
        sizes = {int(jnp.shape(x)[0]) for x in leaves}
        if len(sizes) != 1:
            raise ValueError(f"rows(): leading axes differ: {sorted(sizes)}")
        self.tree = tree
        self.n = sizes.pop()

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Any:
        if not -self.n <= i < self.n:
            raise IndexError(i)
        return jax.tree.map(lambda a: a[i], self.tree)

    def __iter__(self) -> Any:
        return (self[i] for i in range(self.n))


def stack(seq: Any) -> Any:
    """Stack a list of pytrees collected in a loop into one pytree of arrays.

    Free when the list came from a staged loop; otherwise ``jnp.stack`` per leaf.
    """
    if isinstance(seq, Stacked):
        return seq.array
    seq = list(seq)
    if not seq:
        raise ValueError("stack() of an empty list")
    return jax.tree.map(lambda *xs: jnp.stack(xs), *seq)


def reversed_(it: Any) -> Any:
    inner = _as_iter(it) if _rng.tracing() else None
    if inner is None:
        return reversed(it)
    n = inner.n
    if isinstance(n, Tracer) and jax.tree.leaves(inner.xs):
        return reversed(it)
    return _Iter(
        n,
        jax.tree.map(lambda a: a[::-1], inner.xs),
        lambda x, i: inner.make(x, n - 1 - i),
        lambda: reversed(list(inner.py())),
        inner.chunk,
    )


def range_(*args: Any) -> Any:
    lo, hi, step = (0, args[0], 1) if len(args) == 1 else (*args, 1) if len(args) == 2 else args
    if not any(isinstance(a, Tracer) for a in args):
        try:
            return range(*args)
        except TypeError:
            if not _rng.tracing() or type(step) is not int:
                raise
        # a symbolic size (the prover's `T`): a loop of statically unknown but fixed length
        n = (hi - lo) if step == 1 else -(-(hi - lo) // step)
        return _Iter(n, (), lambda x, i: lo + _weaken(i) * step, lambda: range(lo, hi, step))
    if isinstance(step, Tracer):
        raise StageError("range() with a traced step is not supported; make the step a Python int.")
    n = jnp.maximum(0, -(-(jnp.asarray(hi) - lo) // step))
    return _Iter(n, (), lambda x, i: lo + _weaken(i) * step, lambda: range(lo, hi, step))


def enumerate_(iterable: Any, start: int = 0) -> Any:
    it = iterable
    inner = _as_iter(it) if _rng.tracing() else None
    if inner is None:
        return enumerate(it, start)
    return _Iter(
        inner.n,
        inner.xs,
        lambda x, i: (_weaken(i) + start, inner.make(x, i)),
        lambda: enumerate(inner.py(), start),
        inner.chunk,
    )


def zip_(*its: Any, **kw: Any) -> Any:
    inners = [_as_iter(it) for it in its] if _rng.tracing() and not kw else [None]
    if not inners or any(i is None or isinstance(i.n, Tracer) for i in inners):
        return zip(*its, **kw)
    n = min(i.n for i in inners)  # type: ignore[union-attr]
    xs = tuple(jax.tree.map(lambda a: a[:n], i.xs) for i in inners)  # type: ignore[union-attr]
    return _Iter(
        n, xs,
        lambda x, i: tuple(inn.make(xi, i) for inn, xi in zip(inners, x)),  # type: ignore[union-attr]
        lambda: zip(*[inn.py() for inn in inners]),  # type: ignore[union-attr]
    )  # fmt: skip


def for_(
    iterable: Any,
    body_fn: Callable[..., tuple[Any, ...]],
    state: tuple[Any, ...],
    names: tuple[str, ...],
    carried: tuple[str, ...],
    live: tuple[str, ...],
    site: str,
    brk: str | None = None,
) -> tuple[Any, ...]:
    """Run a ``for`` loop: in Python, or as ``vmap`` / ``scan`` when tracing.

    ``carried`` are the variables whose value flows from one iteration to the
    next; ``live`` are those used after the loop. ``brk`` names the flag a
    ``break`` sets: the Python loop stops when it is set to a concrete True.
    """
    k = _rng.sub()
    brk_i = names.index(brk) if brk is not None else None
    it = _as_iter(iterable) if _rng.tracing() else None
    if it is not None and not isinstance(it.n, Tracer) and it.n == 0:
        return state
    if it is not None:
        if isinstance(it.n, Tracer):
            return _fori(it, body_fn, state, names, live, site, k)
        from ._fn import config

        if not config.lower_loops:
            it = None
            LOG[site] = "for -> unrolled (config.lower_loops is off)"
    if it is not None:
        try:
            return _lower_for(it, body_fn, state, names, carried, live, site, k)
        except _CannotLower as e:
            LOG[site] = f"for -> unrolled ({e})"
        except StageError:
            raise
        except Exception as e:  # e.g. the loop index is used as a Python int
            first = (str(e).strip().splitlines() or [""])[0]
            LOG[site] = f"for -> unrolled ({type(e).__name__}: {first[:120]})"
    elif _rng.tracing() and site not in LOG:
        LOG[site] = f"for -> unrolled (iterating over a {type(iterable).__name__})"
    for i, item in enumerate(iterable):
        if k is None:
            state = body_fn(item, *state)
        else:
            with _rng.child(k, i):
                state = body_fn(item, *state)
        if brk_i is not None and state[brk_i] is True:
            break  # a real `break`; a traced flag masks the remaining iterations instead
    return state


def _lower_for(
    it: _Iter,
    body_fn: Callable[..., tuple[Any, ...]],
    state: tuple[Any, ...],
    names: tuple[str, ...],
    carried: tuple[str, ...],
    live: tuple[str, ...],
    site: str,
    k: Any,
) -> tuple[Any, ...]:
    n = it.n
    coll = [i for i, v in enumerate(state) if isinstance(v, (list, Stacked))]
    for i in coll:
        if len(state[i]) != 0:
            raise _CannotLower(f"the list `{names[i]}` is not empty when the loop starts")
    last_idx = [
        i for i in range(len(state)) if i not in coll and names[i] not in carried and names[i] in live
    ]
    need = [
        i for i, v in enumerate(state)
        if i not in coll and names[i] in carried and v is UNDEF and names[i] in live
    ]  # fmt: skip
    if need:

        def first_iteration(*args: Any) -> tuple[Any, ...]:
            full = [_Collector() if j in coll else v for j, v in enumerate(args)]
            x0 = jax.tree.map(lambda a: a[0], it.xs)
            return body_fn(it.make(x0, jnp.zeros((), jnp.int32)), *full)

        zeros = _probe(first_iteration, state, need)
        if set(zeros) != set(need):
            missing = next(i for i in need if i not in zeros)
            raise _CannotLower(f"`{names[missing]}` may be undefined after the loop")
        state = tuple(zeros.get(i, v) for i, v in enumerate(state))
    carry_idx = [
        i for i, v in enumerate(state) if i not in coll and names[i] in carried and _dyn(v)
    ]

    info: dict[str, Any] = {}

    def step(carry: tuple[Any, ...], xi: Any) -> Any:
        x, i = xi
        args = list(state)
        for j, v in zip(carry_idx, carry):
            args[j] = v
        cs = {}
        for j in coll:
            cs[j] = args[j] = _Collector()
        for j in last_idx:
            args[j] = UNDEF
        with _rng.child(k, i), _staged():
            out = body_fn(it.make(x, i), *args)
        for j, v in enumerate(out):
            if j in coll:
                if v is not cs[j]:
                    raise _CannotLower(f"the list `{names[j]}` is replaced inside the loop")
            elif j in carry_idx:
                if not _dyn(v):
                    raise _CannotLower(f"`{names[j]}` stops being an array inside the loop")
            elif j in last_idx:
                if not _dyn(v):
                    raise _CannotLower(f"`{names[j]}` is {_desc(v)} after an iteration")
            elif names[j] in carried and state[j] is not UNDEF and not _same_static(v, state[j]):
                raise _CannotLower(f"`{names[j]}` ({_desc(state[j])}) changes inside the loop")
        info["counts"] = {j: len(cs[j].items) for j in coll}
        ys = tuple(tuple(cs[j].items) for j in coll)
        return tuple(out[j] for j in carry_idx), (ys, tuple(out[j] for j in last_idx))

    idx = jnp.arange(n)
    init = tuple(state[j] for j in carry_idx)
    final: tuple[Any, ...] = ()
    if not carry_idx and it.chunk:
        LOG[site] = f"for -> lax.map in chunks of {it.chunk} (iterations are independent)"
        ys, lasts = lax.map(
            lambda xi: step((), xi)[1], (it.xs, idx), batch_size=None if it.chunk == 1 else it.chunk
        )
    elif not carry_idx:
        LOG[site] = "for -> vmap (iterations are independent)"
        _, (ys, lasts) = jax.vmap(lambda x, i: step((), (x, i)))(it.xs, idx)
    else:
        LOG[site] = f"for -> lax.scan (carries {', '.join(names[j] for j in carry_idx)})"
        final, (ys, lasts) = lax.scan(step, init, (it.xs, idx))

    out = list(state)
    for j, v in zip(carry_idx, final):
        out[j] = v
    for j, v in zip(last_idx, lasts):
        out[j] = jax.tree.map(lambda a: a[-1], v)
    for j, items in zip(coll, ys):
        if len(items) == 1:
            out[j] = Stacked(items[0])
        elif len(items) > 1:
            try:
                both = jax.tree.map(lambda *a: jnp.stack(a, axis=1), *items)
                out[j] = Stacked(jax.tree.map(lambda a: a.reshape((-1,) + a.shape[2:]), both))
            except Exception:
                raise _CannotLower(
                    f"`{names[j]}` receives values of different shapes in one iteration"
                ) from None
    return tuple(out)


def _fori(
    it: _Iter,
    body_fn: Callable[..., tuple[Any, ...]],
    state: tuple[Any, ...],
    names: tuple[str, ...],
    live: tuple[str, ...],
    site: str,
    k: Any,
) -> tuple[Any, ...]:
    """A loop whose trip count is traced: ``lax.while_loop`` with a counter."""
    for i, v in enumerate(state):
        if isinstance(v, (list, Stacked)):
            raise StageError(
                f"`{names[i]}.append(...)` runs in the `for` at {site}, whose number of "
                f"iterations is traced, so the list's length would depend on data.\n"
                f"    Preallocate an array and fill it with {names[i]}.at[i].set(...), or "
                f"make the trip count Static."
            )
    need = [i for i, v in enumerate(state) if v is UNDEF and names[i] in live]
    if need:
        zeros = _probe(
            lambda *args: body_fn(it.make((), jnp.zeros((), jnp.int32)), *args), state, need
        )
        state = tuple(zeros.get(i, v) for i, v in enumerate(state))
    dyn = [i for i, v in enumerate(state) if _dyn(v)]

    def merge(carry: tuple[Any, ...]) -> list[Any]:
        full = list(state)
        for i, v in zip(dyn, carry):
            full[i] = v
        return full

    def body_f(c: Any) -> Any:
        i, carry = c
        with _rng.child(k, i), _staged():
            out = body_fn(it.make((), i), *merge(carry))
        for j, v in enumerate(out):
            if j in dyn:
                continue
            if state[j] is UNDEF:
                if names[j] in live and v is not UNDEF:
                    raise StageError(
                        f"`{names[j]}` is first assigned inside the `for` at {site}, whose "
                        f"number of iterations is traced, and used after it. Give it a value "
                        f"before the loop."
                    )
            elif not _same_static(v, state[j]):
                raise StageError(
                    f"`{names[j]}` ({_desc(state[j])}) changes inside the traced-length `for` "
                    f"at {site}; only arrays can change in a traced loop."
                )
        return i + 1, tuple(out[j] for j in dyn)

    LOG[site] = "for -> lax.while_loop (traced number of iterations)"
    try:
        _, final = lax.while_loop(
            lambda c: c[0] < it.n, body_f, (jnp.zeros((), jnp.int32), tuple(state[i] for i in dyn))
        )
    except TypeError as e:
        e.add_note(
            f"typax: in the `for` at {site}; carried variables, in order: "
            f"{', '.join(names[i] for i in dyn)}.\n"
            f"    Each must keep the same shape and dtype on every iteration."
        )
        raise
    return tuple(merge(final))

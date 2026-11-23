"""Runtime for converted branches and ``while`` loops, and what they share.

``@fn`` rewrites a function's statements into calls to helpers (see
``_convert.py``). Each helper decides *when it runs* what to do: for ordinary
Python values it behaves exactly like the original statement; for traced
values it uses ``lax.cond`` / ``lax.while_loop`` / ``lax.scan`` / ``vmap``.
Every helper receives the variables the statement assigns as a tuple
(``state``) together with their ``names``, and returns the new tuple.

This module holds ``if`` / ``while`` and the machinery the other runtime
modules share; ``for`` lives in ``_loops.py``, operators and builtins in
``_ops.py``. Generated code reaches all of them through ``_runtime.py``.
"""

import math
import threading
from collections.abc import Sequence
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import Tracer, eval_jaxpr

from .. import keys
from ..annotations.errors import StageError, coded, strip_see

#: site ("file:line") -> how the statement there was last executed.
LOG: dict[str, str] = {}


class _Undefined:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<undefined>"

    def __bool__(self) -> bool:
        raise NameError("this variable is not defined on every path that reaches here")


UNDEF = _Undefined()


def ret_(thunk: Callable[[], Any], returned: Any = True) -> Any:
    """The function's return value; ``None`` if no ``return`` statement set it.

    ``returned`` says whether one did, for a function that may fall off its end: traced,
    some data would return a value and other data None, which no traced function can.
    """
    if isinstance(returned, Tracer) and ld(thunk) is not UNDEF:  # (no path set a value: None)
        from .ops import CALLS

        name = getattr(CALLS.stack[-1][1], "name", "the function") if CALLS.stack else "the function"
        raise coded("EX213", StageError(
            f"{name}(): whether it returns a value or falls off its end (returning None) "
            f"depends on a traced value. A traced function returns the same kind of value "
            f"on every path: end it with a `return`, or give the `if` an `else` that returns."
        ))
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


# ------------------------------------------------------------------ values


def _is_arr(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic))


_NUM = (bool, int, float, complex)


def _function(v: Any) -> bool:
    """A function or other callable that is not data (a module with ``__call__`` is data)."""
    return callable(v) and not getattr(type(v), "__expex_struct__", False)


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
    from ..annotations.dims import describe

    if v is UNDEF:
        return "undefined"
    if _is_arr(v) or type(v) in _NUM or isinstance(v, jax.ShapeDtypeStruct):
        return describe(v)
    if _dyn(v) or (jax.tree.leaves(v) and all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(v))):
        return f"{type(v).__name__} of arrays"
    return f"{type(v).__name__} {v!r}"[:60]


# --------------------------------------------------------------------- bool


def _boolean(v: Any) -> bool:
    return type(v) is bool or getattr(v, "dtype", None) == np.bool_


# ----------------------------------------------------------------------- if


def _check_pred(pred: Any, what: str, site: str) -> None:
    """A condition must be one True/False; an array of them has no single answer."""
    if isinstance(pred, Tracer) or _is_arr(pred):
        shape = tuple(pred.shape)
        if shape != () and (isinstance(pred, Tracer) or pred.size != 1):
            raise coded("EX203", StageError(
                f"the condition of the `{what}` at {site} is an array of shape {shape}, "
                f"not a single True/False.\n"
                f"    - to choose per element:      write the function for one element (x: f32): a call\n"
                f"                                  on an array maps over it. Or jnp.where(cond, a, b)\n"
                f"    - to test all/any elements:   cond.all()  /  cond.any()"
            ))


def _cond(
    pred: Any,
    fns: tuple[Callable[[], tuple[Any, ...]], Callable[[], tuple[Any, ...]]],
    names: tuple[str, ...],
    site: str,
) -> tuple[Any, ...]:
    """``lax.cond`` over branches returning tuples that may mix arrays and other values.

    Each branch is traced once, to a jaxpr closed over the values it uses. The
    two are then reconciled (a dtype both can take, a zero placeholder for a
    variable only one branch assigns) and run by one ``lax.cond``, which only
    replays the jaxprs.
    """
    recs: list[list[tuple[str, Any]]] = []
    jaxprs: list[Any] = []
    for fn in fns:
        metas: list[tuple[str, Any]] = []

        def traced(fn: Callable[[], tuple[Any, ...]] = fn, metas: list[tuple[str, Any]] = metas) -> list[Any]:
            with _staged():
                out = fn()
            metas.extend(("dyn", None) if _dyn(v) else ("static", v) for v in out)
            return [v for v in out if _dyn(v)]

        closed, shapes = jax.make_jaxpr(traced, return_shape=True)()
        it = iter(shapes)
        recs.append([("dyn", next(it)) if kind == "dyn" else (kind, v) for kind, v in metas])
        jaxprs.append(closed)
    plan = _plan(recs, names, site)

    def replay(k: int, consts: list[Any] | None = None) -> Callable[[], tuple[Any, ...]]:
        def run() -> tuple[Any, ...]:
            values = iter(eval_jaxpr(jaxprs[k].jaxpr, jaxprs[k].consts if consts is None else consts))
            out: list[Any] = []
            for i, (kind, v) in enumerate(recs[k]):
                p = plan.get(i)
                if kind != "dyn" and not isinstance(p, dict):
                    continue
                if kind != "dyn":  # assigned in the other branch only: a placeholder here
                    out.append(jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), p["fill"]))  # type: ignore[index]
                    continue
                leaves, tree = jax.tree.flatten(v)
                xs = [next(values) for _ in leaves]
                if isinstance(p, dict):
                    xs = [_cast(x, s.dtype, False) for x, s in zip(xs, jax.tree.leaves(p["fill"]))]
                elif p is not None:
                    xs = [_cast(x, d, w) for x, (d, w) in zip(xs, p)]
                out.append(jax.tree.unflatten(tree, xs))
            return tuple(out)

        return run

    how = _selectable(pred, jaxprs)
    res = _select(pred, jaxprs, replay, masked=how == "masked") if how else None
    try:
        if res is None:
            res = lax.cond(pred, replay(0), replay(1))
    except TypeError as e:
        e.add_note(f"expex: in the `if` at {site}; variables it assigns: {', '.join(names)}")
        raise
    it = iter(res)
    out: list[Any] = []
    for i, (m0, m1) in enumerate(zip(recs[0], recs[1])):
        if m0[0] == "dyn" or m1[0] == "dyn":
            out.append(next(it))
        elif _same_static(m0[1], m1[1]):
            out.append(m0[1])
        elif m0[1] is UNDEF or m1[1] is UNDEF:
            # set on one path only: the value from that path (like the zero
            # placeholder arrays get), since the other path must not read it
            out.append(m1[1] if m0[1] is UNDEF else m0[1])
        else:
            raise coded("EX204", StageError(
                f"`{names[i]}` is set to different non-array values in the two branches of "
                f"the traced `if` at {site}: {_desc(m0[1])} vs {_desc(m1[1])}.\n"
                f"    A traced condition can only select between arrays of the same shape "
                f"and dtype."
            ))
    return tuple(out)


#: A branch this small is computed outright rather than put in a ``lax.cond`` (see ``_selectable``).
_SELECT_EQNS = 32
_SELECT_ELEMENTS = 4096
#: Primitives that make a branch worth skipping at run time (or that ``_cheap`` cannot see into).
_COSTLY = {"while", "scan", "cond", "dot_general", "conv_general_dilated", "custom_jvp_call", "custom_vjp_call"}

try:  # private, but there is no public way to ask "is this value being vmapped?"
    from jax._src.interpreters.batching import BatchTracer as _BatchTracer
except Exception:  # pragma: no cover
    _BatchTracer = ()  # type: ignore[assignment,misc]


def _selectable(pred: Any, jaxprs: list[Any]) -> str | None:
    """Whether a traced two-way choice is emitted as a ``select`` of both branches rather than a ``lax.cond``.

    Only where the ``cond`` would give the same values, gradients and effects:

    * the condition is one ``lax.cond`` takes (a boolean or an integer), and
      neither branch has effects (``print``, ``assert``, ``raise``, callbacks);
    * ``"masked"``: the condition is mapped over (``vmap``, a mapped call). JAX
      then turns the ``cond`` into this same ``select`` of both branches, each
      branch's inputs masked with ``stop_gradient`` where it is not taken (the
      NaN-safe gradient); building it here saves tracing and batching the
      ``cond``, and masks only the float inputs (the others carry no gradient);
    * ``"plain"``: every value the branches give is a boolean or an integer, so
      no gradient flows through the choice, and both branches are small, so
      computing both is no slower than branching. The common case is ``a and
      b`` on traced conditions, as in ``while 0 <= r < R and board[r, c] == x:``.
    """
    kind = getattr(getattr(pred, "dtype", None), "kind", None)
    if kind not in ("b", "i", "u") or any(closed.effects for closed in jaxprs):
        return None
    if isinstance(pred, _BatchTracer) and getattr(pred, "batch_dim", None) is not None:
        return "masked"
    for closed in jaxprs:
        if any(getattr(a, "dtype", None) is None or a.dtype.kind not in "biu" for a in closed.out_avals):
            return None
    budget = [_SELECT_EQNS, _SELECT_ELEMENTS]
    return "plain" if all(_cheap(closed.jaxpr, budget) for closed in jaxprs) else None


def _cheap(jaxpr: Any, budget: list[int]) -> bool:
    """Whether ``jaxpr`` fits in ``budget`` (equations, output elements), which it uses up."""
    for e in jaxpr.eqns:
        if e.primitive.name in _COSTLY:
            return False
        budget[0] -= 1
        budget[1] -= sum(math.prod(getattr(v.aval, "shape", ())) for v in e.outvars)
        if budget[0] < 0 or budget[1] < 0:
            return False
        for p in e.params.values():
            sub = getattr(p, "jaxpr", p)
            if hasattr(sub, "eqns") and not _cheap(sub, budget):
                return False
    return True


def _select(
    pred: Any, jaxprs: list[Any], replay: Callable[..., Callable[[], tuple[Any, ...]]], masked: bool
) -> tuple[Any, ...] | None:
    """Both branches' values and a ``select`` between them (see ``_selectable``); None if their types differ."""
    taken = _truth(pred)
    if masked:  # as JAX's batching rule of `cond`: a branch's inputs carry no gradient where it is not taken
        branches = []
        for k, on in ((0, lambda: taken), (1, lambda: ~taken)):
            consts = list(jaxprs[k].consts)
            masks = [i for i, c in enumerate(consts) if isinstance(c, Tracer) and jnp.issubdtype(c.dtype, jnp.inexact)]
            if masks:
                where = on()
                for i in masks:
                    consts[i] = lax.select(where, consts[i], lax.stop_gradient(consts[i]))
            branches.append(replay(k, consts)())
    else:
        branches = [replay(0)(), replay(1)()]
    out: list[Any] = []
    ta: Any
    tb: Any
    for a, b in zip(*branches):
        la, ta = jax.tree.flatten(a)
        lb, tb = jax.tree.flatten(b)
        if ta != tb:
            return None
        xs = []
        for x, y in zip(la, lb):
            x, y = _array(x), _array(y)
            if jnp.shape(x) != jnp.shape(y) or x.dtype != y.dtype or _weak(x) != _weak(y):
                return None
            xs.append(lax.select(taken, x, y))
        out.append(jax.tree.unflatten(ta, xs))
    return tuple(out)


def _array(x: Any) -> Any:
    """A value replayed from a jaxpr as an array: a literal comes back as a Python or NumPy scalar."""
    return x if isinstance(x, (jax.Array, Tracer)) else jnp.asarray(x)


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
            raise coded("EX204", StageError(
                f"`{names[i]}` is {a} in one branch of the traced `if` at {site} and {b} "
                f"in the other.\n    Both branches must give it an array of the same shape "
                f"and dtype."
            ))
        if m0[0] == "dyn":
            t0: Any
            t1: Any
            l0, t0 = jax.tree.flatten(m0[1])
            l1, t1 = jax.tree.flatten(m1[1])
            if t0 != t1:
                raise coded("EX204", StageError(
                    f"`{names[i]}` has a different structure in the two branches of the "
                    f"traced `if` at {site}: {_desc(m0[1])} vs {_desc(m1[1])}."
                ))
            casts = []
            change = False
            for v0, v1 in zip(l0, l1):
                if jnp.shape(v0) != jnp.shape(v1):
                    raise coded("EX204", StageError(
                        f"`{names[i]}` has shape {_desc(v0)} in one branch of the traced `if` "
                        f"at {site} and {_desc(v1)} in the other.\n    Both branches must "
                        f"give it the same shape and dtype."
                    ))
                d0, d1 = jnp.result_type(v0), jnp.result_type(v1)
                change = change or d0 != d1 or _weak(v0) != _weak(v1)
                casts.append((jnp.result_type(_promotes_as(v0), _promotes_as(v1)), _weak(v0) and _weak(v1)))
            if change:
                plan[i] = casts
    return plan


def _promotes_as(v: Any) -> Any:
    """What ``v`` promotes like: its dtype, or the Python type of its kind if weakly typed."""
    d = jnp.result_type(v)
    return {"b": bool, "i": int, "u": int, "f": float, "c": complex}.get(d.kind, d) if _weak(v) else d


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


def _weak(v: Any) -> bool:
    if type(v) in _NUM:
        return True
    return bool(getattr(getattr(v, "aval", None), "weak_type", False) or getattr(v, "weak_type", False))


def _guard_lists(state: tuple[Any, ...], names: tuple[str, ...], what: str, site: str) -> Callable[[tuple[Any, ...]], None]:
    """Detect in-place growth of Python lists inside traced control flow."""
    lens = {i: len(v) for i, v in enumerate(state) if isinstance(v, list)}
    colls = [i for i, v in enumerate(state) if isinstance(v, _Collector)]
    if colls:
        raise coded("EX205", _CannotLower(f"`{names[colls[0]]}.append(...)` inside a traced `{what}`"))

    def check(out: tuple[Any, ...]) -> None:
        for i, n in lens.items():
            if out[i] is state[i] and len(out[i]) != n:
                raise coded("EX205", StageError(
                    f"`{names[i]}.append(...)` runs inside the traced `{what}` at {site}, so "
                    f"the list's length would depend on data.\n"
                    f"    Append unconditionally and select the value instead: "
                    f"{names[i]}.append(jnp.where(cond, a, b))."
                ))

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
    k = keys.sub()
    _check_pred(pred, "if", site)
    if not isinstance(pred, Tracer):
        with keys.child(k):
            return (true_fn if pred else false_fn)(*state)
    check = _guard_lists(state, names, "if", site)
    taken = _truth(pred)

    def run(fn: Callable[..., tuple[Any, ...]], on: Any) -> Callable[[], tuple[Any, ...]]:
        def r() -> tuple[Any, ...]:
            with keys.child(k), _on_path(on):
                out = fn(*state)
            check(out)
            return out

        return r

    LOG[site] = "if -> lax.cond"
    return _cond(pred, (run(true_fn, taken), run(false_fn, lambda: ~taken)), names, site)  # `not` only if read


def ifexp(pred: Any, a: Callable[[], Any], b: Callable[[], Any], site: str) -> Any:
    _check_pred(pred, "... if ... else ...", site)
    if not isinstance(pred, Tracer):
        k = keys.sub()
        with keys.child(k):
            return a() if pred else b()
    return _select_expr(pred, a, b, site)


def _select_expr(pred: Any, a: Callable[[], Any], b: Callable[[], Any], site: str) -> Any:
    """``a() if pred else b()`` for a traced scalar ``pred``; each side runs in its branch."""
    _check_pred(pred, "... if ... else ...", site)
    k = keys.sub()
    taken = _truth(pred)

    def side(fn: Callable[[], Any], on: Any) -> Callable[[], tuple[Any, ...]]:
        def r() -> tuple[Any, ...]:
            with keys.child(k), _on_path(on):
                return (fn(),)

        return r

    return _cond(pred, (side(a, taken), side(b, lambda: ~taken)), ("<the expression>",), site)[0]


# ------------------------------------------------------------- traced depth


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


# -------------------------------------------------------------------- while


def while_(
    test_fn: Callable[..., Any],
    body_fn: Callable[..., tuple[Any, ...]],
    state: tuple[Any, ...],
    names: tuple[str, ...],
    live: tuple[str, ...],
    site: str,
) -> tuple[Any, ...]:
    k = keys.sub()
    it = 0
    while True:
        with keys.child(k, 2 * it):
            c = test_fn(*state)
        if isinstance(c, Tracer):
            break
        _check_pred(c, "while", site)
        if not c:
            return state
        with keys.child(k, 2 * it + 1):
            state = body_fn(*state)
        it += 1

    _check_pred(c, "while", site)
    truth = (lambda v: v) if _boolean(c) else (lambda v: v != 0)
    try:
        check = _guard_lists(state, names, "while", site)
    except _CannotLower as e:
        raise coded("EX205", StageError(f"{strip_see(str(e))} at {site}: the number of iterations depends on data.")) from None
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
        with keys.child(k, 2 * n):
            return truth(test_fn(*merge(carry)))

    def body_f(c2: Any) -> Any:
        n, carry = c2
        still = lambda: truth(test_fn(*merge(carry)))  # noqa: E731 - only evaluated by an assert
        with keys.child(k, 2 * n + 1), _staged(), _on_path(still):
            out = body_fn(*merge(carry))
        check(out)
        for i, v in enumerate(out):
            if i in dyn:
                continue
            if state[i] is UNDEF:
                if names[i] in live and v is not UNDEF:
                    raise coded("EX206", StageError(
                        f"`{names[i]}` is first assigned inside the traced `while` at {site} "
                        f"and used after it. Give it a value before the loop."
                    ))
            elif not _same_static(v, state[i]):
                raise coded("EX207", StageError(
                    f"`{names[i]}` ({_desc(state[i])}) changes inside the traced `while` at "
                    f"{site}; only arrays can change in a traced loop."
                ))
        new = tuple(out[i] for i in dyn)
        _check_carry([names[i] for i in dyn], carry, new, "while", site)
        return n + 1, new

    LOG[site] = "while -> lax.while_loop"
    try:
        _, final = lax.while_loop(cond_f, body_f, (count0, init))
    except TypeError as e:
        e.add_note(
            f"expex: in the `while` at {site}; carried variables, in order: "
            f"{', '.join(names[i] for i in dyn)}.\n"
            f"    Each must keep the same shape and dtype on every iteration."
        )
        raise
    return tuple(merge(final))


def _check_carry(names: list[str], before: tuple[Any, ...], after: tuple[Any, ...], what: str, site: str) -> None:
    """A traced loop's variables keep their shape and dtype (a Python int may become a float)."""
    ta: Any
    tb: Any
    for name, a, b in zip(names, before, after):
        la, ta = jax.tree.flatten(a)
        lb, tb = jax.tree.flatten(b)
        if ta != tb:
            raise coded("EX212", StageError(
                f"`{name}` changes structure inside the traced `{what}` at {site}: "
                f"{_desc(a)} before an iteration, {_desc(b)} after."
            ))
        for x, y in zip(la, lb):
            dx, dy = jnp.result_type(x), jnp.result_type(y)
            if jnp.shape(x) != jnp.shape(y) or (dx != dy and not _weak(x)):
                raise coded("EX212", StageError(
                    f"`{name}` is {_desc(x)} before an iteration of the traced `{what}` at {site} "
                    f"and {_desc(y)} after it: a traced loop keeps the shape and dtype of what "
                    f"it carries.\n    Give it its final shape and dtype before the loop"
                    + (f" ({name} = jnp.asarray({name}, jnp.{dy.name}))." if jnp.shape(x) == jnp.shape(y) else ".")
                ))


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
        n = jax.tree.leaves(self.array)[0].shape[0]
        try:
            return int(n)
        except Exception:
            raise coded("EX210", TypeError(
                f"the values collected by this loop number {n}, a size not known while "
                f"proving: use jnp.asarray(ys) (or expex.stack(ys) for pytrees), which takes "
                f"the stacked array as it is, rather than jnp.stack(ys)."
            )) from None

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



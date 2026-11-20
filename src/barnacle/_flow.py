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
    "if_", "ifexp", "while_", "for_", "and_", "or_", "not_", "ld",
    "range_", "enumerate_", "zip_", "append_", "Stacked", "UNDEF", "LOG",
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


def _dyn(v: Any) -> bool:
    """Can ``v`` be carried through lax control flow (a pytree of arrays/numbers)?"""
    if v is UNDEF or v is None:
        return False
    if _is_arr(v) or type(v) in _NUM:
        return True
    if isinstance(v, (str, bytes, _Collector, Stacked)) or callable(v):
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


def and_(a: Callable[[], Any], b: Callable[[], Any]) -> Any:
    x = a()
    if isinstance(x, Tracer):
        return jnp.logical_and(x, b())
    if not x:
        return x
    return b()


def or_(a: Callable[[], Any], b: Callable[[], Any]) -> Any:
    x = a()
    if isinstance(x, Tracer):
        return jnp.logical_or(x, b())
    if x:
        return x
    return b()


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
    for attempt in range(3):
        recs: list[list[tuple[str, Any]]] = []

        def wrap(fn: Callable[[], tuple[Any, ...]]) -> Callable[[], tuple[Any, ...]]:
            def w() -> tuple[Any, ...]:
                out = fn()
                metas: list[tuple[str, Any]] = []
                dyn: list[Any] = []
                for i, v in enumerate(out):
                    p = plan.get(i)
                    if p == "drop":
                        metas.append(("static", UNDEF))
                    elif _dyn(v):
                        if p is not None:
                            v = jnp.asarray(v, p)
                        metas.append(("dyn", v))
                        dyn.append(v)
                    else:
                        metas.append(("static", v))
                recs.append(metas)
                return tuple(dyn)

            return w

        err: TypeError | None = None
        res: Any = ()
        try:
            res = lax.cond(pred, wrap(fns[0]), wrap(fns[1]))
        except TypeError as e:
            if len(recs) < 2:
                raise
            err = e
        new_plan = _plan(recs, names, site)
        if new_plan and new_plan != plan and attempt < 2:
            plan = {**plan, **new_plan}
            continue
        if err is not None:
            err.add_note(f"barnacle: in the `if` at {site}; variables it assigns: {', '.join(names)}")
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
            out.append(UNDEF)
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
                plan[i] = "drop"  # defined in one branch only: undefined afterwards
                continue
            a, b = _desc(m0[1]), _desc(m1[1])
            raise StageError(
                f"`{names[i]}` is {a} in one branch of the traced `if` at {site} and {b} "
                f"in the other.\n    Both branches must give it an array of the same shape "
                f"and dtype."
            )
        if m0[0] == "dyn":
            v0, v1 = m0[1], m1[1]
            if not (_leaf(v0) and _leaf(v1)):
                continue
            s0, s1 = jnp.shape(v0), jnp.shape(v1)
            if s0 != s1:
                raise StageError(
                    f"`{names[i]}` has shape {_desc(v0)} in one branch of the traced `if` at "
                    f"{site} and {_desc(v1)} in the other.\n    Both branches must give it "
                    f"the same shape and dtype."
                )
            d0, d1 = jnp.result_type(v0), jnp.result_type(v1)
            if d0 != d1 or _weak(v0) != _weak(v1):
                plan[i] = jnp.result_type(v0, v1)
    return plan


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

    def run(fn: Callable[..., tuple[Any, ...]]) -> Callable[[], tuple[Any, ...]]:
        def r() -> tuple[Any, ...]:
            with _rng.child(k):
                out = fn(*state)
            check(out)
            return out

        return r

    LOG[site] = "if -> lax.cond"
    return _cond(pred, (run(true_fn), run(false_fn)), names, site)


def ifexp(pred: Any, a: Callable[[], Any], b: Callable[[], Any], site: str) -> Any:
    if not isinstance(pred, Tracer):
        return a() if pred else b()
    _check_pred(pred, "... if ... else ...", site)
    return _cond(pred, (lambda: (a(),), lambda: (b(),)), ("<the expression>",), site)[0]


# -------------------------------------------------------------------- while


class _CannotLower(Exception):
    """A loop cannot be turned into lax control flow; the reason is the message."""


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
        with _rng.child(k, it):
            c = test_fn(*state)
        if isinstance(c, Tracer):
            break
        if not c:
            return state
        with _rng.child(k, it):
            state = body_fn(*state)
        it += 1

    _check_pred(c, "while", site)
    try:
        check = _guard_lists(state, names, "while", site)
    except _CannotLower as e:
        raise StageError(f"{e} at {site}: the number of iterations depends on data.") from None
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
        with _rng.child(k, n):
            return test_fn(*merge(carry))

    def body_f(c2: Any) -> Any:
        n, carry = c2
        with _rng.child(k, n):
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
            f"barnacle: in the `while` at {site}; carried variables, in order: "
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

    def __array__(self, *a: Any, **k: Any) -> Any:
        return np.asarray(self.array, *a, **k)

    def __repr__(self) -> str:
        return f"Stacked({self.array!r})"


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

    def __init__(self, n: Any, xs: Any, make: Callable[[Any, Any], Any], py: Callable[[], Any]):
        self.n = n
        self.xs = xs
        self.make = make
        self.py = py

    def __iter__(self) -> Any:
        return iter(self.py())


def _as_iter(it: Any) -> _Iter | None:
    if isinstance(it, _Iter):
        return it
    if isinstance(it, range):
        start, step = it.start, it.step
        return _Iter(len(it), (), lambda x, i: start + i * step, lambda: it)
    if _is_arr(it) and getattr(it, "ndim", 0) >= 1:
        return _Iter(it.shape[0], it, lambda x, i: x, lambda: it)
    if isinstance(it, Stacked):
        return _Iter(len(it), it.array, lambda x, i: x, lambda: it)
    return None


def range_(*args: Any) -> Any:
    if not any(isinstance(a, Tracer) for a in args):
        return range(*args)
    lo, hi, step = (0, args[0], 1) if len(args) == 1 else (*args, 1) if len(args) == 2 else args
    if isinstance(step, Tracer):
        raise StageError("range() with a traced step is not supported; make the step a Python int.")
    n = jnp.maximum(0, -(-(jnp.asarray(hi) - lo) // step))
    return _Iter(n, (), lambda x, i: lo + i * step, lambda: range(lo, hi, step))


def enumerate_(it: Any, start: int = 0) -> Any:
    inner = _as_iter(it) if _rng.tracing() else None
    if inner is None:
        return enumerate(it, start)
    return _Iter(
        inner.n, inner.xs, lambda x, i: (i + start, inner.make(x, i)), lambda: enumerate(inner.py(), start)
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
) -> tuple[Any, ...]:
    """Run a ``for`` loop: in Python, or as ``vmap`` / ``scan`` when tracing.

    ``carried`` are the variables whose value flows from one iteration to the
    next; ``live`` are those used after the loop.
    """
    k = _rng.sub()
    it = _as_iter(iterable) if _rng.tracing() else None
    if it is not None and not isinstance(it.n, Tracer) and it.n == 0:
        return state
    if it is not None:
        if isinstance(it.n, Tracer):
            return _fori(it, body_fn, state, names, live, site, k)
        try:
            return _lower_for(it, body_fn, state, names, carried, live, site, k)
        except _CannotLower as e:
            LOG[site] = f"for -> unrolled ({e})"
        except StageError:
            raise
        except Exception as e:  # e.g. the loop index is used as a Python int
            first = (str(e).strip().splitlines() or [""])[0]
            LOG[site] = f"for -> unrolled ({type(e).__name__}: {first[:120]})"
    else:
        if _rng.tracing():
            LOG[site] = f"for -> unrolled (iterating over a {type(iterable).__name__})"
    for i, item in enumerate(iterable):
        if k is None:
            state = body_fn(item, *state)
        else:
            with _rng.child(k, i):
                state = body_fn(item, *state)
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
    carry_idx = [
        i for i, v in enumerate(state) if i not in coll and names[i] in carried and _dyn(v)
    ]
    last_idx = [
        i for i in range(len(state)) if i not in coll and names[i] not in carried and names[i] in live
    ]
    for i, v in enumerate(state):
        if i not in coll and names[i] in carried and v is UNDEF and names[i] in live:
            raise _CannotLower(f"`{names[i]}` may be undefined after the loop")

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
        with _rng.child(k, i):
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
    if not carry_idx:
        LOG[site] = "for -> vmap (iterations are independent)"
        _, (ys, lasts) = jax.vmap(lambda x, i: step((), (x, i)))(it.xs, idx)
        final: tuple[Any, ...] = ()
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
    dyn = [i for i, v in enumerate(state) if _dyn(v)]

    def merge(carry: tuple[Any, ...]) -> list[Any]:
        full = list(state)
        for i, v in zip(dyn, carry):
            full[i] = v
        return full

    def body_f(c: Any) -> Any:
        i, carry = c
        with _rng.child(k, i):
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
            f"barnacle: in the `for` at {site}; carried variables, in order: "
            f"{', '.join(names[i] for i in dyn)}.\n"
            f"    Each must keep the same shape and dtype on every iteration."
        )
        raise
    return tuple(merge(final))

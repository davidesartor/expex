"""Staged loops: `for` over arrays, ranges and pytrees, as `vmap`, `scan` or `while_loop`.

``for_`` runs a converted ``for`` statement; ``range_``, ``enumerate_``, ``zip_``
and ``reversed_`` stand in for the builtins so that a loop over them can be
staged; ``rows`` and ``stack`` are the public helpers for pytrees.
"""

import warnings
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.core import Tracer

from .. import keys
from ..annotations.errors import StageError
from .flow import (
    LOG,
    UNDEF,
    Stacked,
    _CannotLower,
    _Collector,
    _desc,
    _dyn,
    _is_arr,
    _probe,
    _same_static,
    _staged,
    _weaken,
)


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
    inner = _as_iter(it) if keys.tracing() else None
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
            if not keys.tracing() or type(step) is not int:
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
    inner = _as_iter(it) if keys.tracing() else None
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
    inners = [_as_iter(it) for it in its] if keys.tracing() and not kw else [None]
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
    k = keys.sub()
    brk_i = names.index(brk) if brk is not None else None
    it = _as_iter(iterable) if keys.tracing() else None
    if it is not None and not isinstance(it.n, Tracer) and it.n == 0:
        return state
    if it is not None:
        if isinstance(it.n, Tracer):
            return _fori(it, body_fn, state, names, live, site, k)
        from ..function import config

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
        _warn_unrolled(site, it.n)
    elif keys.tracing() and site not in LOG:
        LOG[site] = f"for -> unrolled (iterating over a {type(iterable).__name__})"
    for i, item in enumerate(iterable):
        if k is None:
            state = body_fn(item, *state)
        else:
            with keys.child(k, i):
                state = body_fn(item, *state)
        if brk_i is not None and state[brk_i] is True:
            break  # a real `break`; a traced flag masks the remaining iterations instead
    return state


#: Loops longer than this that cannot be staged are unrolled with a warning (once per site).
UNROLL_QUIETLY = 16
_WARNED: set[str] = set()


def _warn_unrolled(site: str, n: Any) -> None:
    if n > UNROLL_QUIETLY and site not in _WARNED:
        _WARNED.add(site)
        why = LOG[site].removeprefix("for -> unrolled (").removesuffix(")")
        warnings.warn(
            f"the `for` loop at {site} is unrolled into {n} copies of its body: {why}. "
            f"Tracing and compiling grow with the length (f.flow lists how each loop ran).",
            stacklevel=2,
        )


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
    # An empty list collects `append`ed values; a list of arrays that the loop
    # replaces (states = step(states)) is ordinary state, carried like an array.
    coll = [
        i for i, v in enumerate(state)
        if isinstance(v, Stacked) or (isinstance(v, list) and not (v and _dyn(v)))
    ]  # fmt: skip
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
        with keys.child(k, i), _staged():
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
        with keys.child(k, i), _staged():
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
            f"expex: in the `for` at {site}; carried variables, in order: "
            f"{', '.join(names[i] for i in dyn)}.\n"
            f"    Each must keep the same shape and dtype on every iteration."
        )
        raise
    return tuple(merge(final))

"""Mapping an ``@fn`` over extra leading axes: ``f.vectorize(...)``, and plain calls that need it.

A call whose array arguments have more leading axes than their annotations
declare is mapped over them (``vectorize``): explicitly with ``f.vectorize(...)``,
or automatically, when the argument check of a plain call fails on exactly that
(``Fn._run`` catches the check's ``Rebatch`` and comes here with ``auto=True``).

Outside any JAX transformation the mapping would be traced and run op by op
on every call, so from the second call with the same abstract signature it
runs as one compiled ``jax.jit(vmap(...))`` per function (``_Compiled``).
"""

import contextvars
import inspect
import math
import re
import weakref
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.core import Tracer
from jax.interpreters import batching as batching_rules

from . import keys
from .annotations.dims import describe
from .annotations.errors import ShapeError, StageError, coded
from .annotations.spec import KeySpec, Spec

if TYPE_CHECKING:
    from .function import Fn


class Rebatch(Exception):
    """Internal: a call's argument check failed on extra leading axes it may map over.

    Raised by ``Fn._checked`` (or before a call hook sees the call) and caught
    by ``Fn._run``, which maps the call with ``vectorize(..., auto=True)``.
    """


_RANK = re.compile(r"expected \d+ ax(is|es), got \d+\.")


def extra_axes(f: Fn[Any, Any], m: Any) -> bool:
    """Whether a failed argument match (a ``Mismatch``) is extra leading axes to map over.

    Only on an annotation with a fixed number of axes (not ``*B``), only for the
    number of axes (the dtype matched), and never on ``self``: a model's own
    fields are not mapped over automatically (``f.vectorize`` does map them).
    """
    rank = getattr(m.spec, "core_rank", None)
    if rank is None or not _RANK.fullmatch(m.reason):
        return False
    if isinstance(m.spec, KeySpec):
        shape = KeySpec.batch_shape(m.value)
    else:
        shape = getattr(m.value, "shape", None)
    if shape is None or len(shape) <= rank:
        return False
    return not _on_self(f, m.path)


def _on_self(f: Fn[Any, Any], path: str) -> bool:
    return path.split(".")[0].split("[")[0] == "self" and next(iter(f.params), None) == "self"


def vectorize(f: Fn[Any, Any], args: tuple[Any, ...], kwargs: dict[str, Any], auto: bool = False) -> Any:
    """Call the function with extra leading batch axes on any array argument.

    Each argument's batch axes are whatever precedes its declared shape.
    Arguments without extra axes are passed to every call unchanged; batch
    axes of different arguments broadcast against each other. The result
    gains the batch axes in front.

    A single PRNG key is split into one key per batch element, so examples
    do not share randomness.

    ``auto``: a plain call whose check found extra leading axes. Arguments with
    a variadic annotation (``*B``) and ``self`` are then passed whole to every
    element, and a mismatch per element says the axes were mapped.
    """
    __tracebackhide__ = True
    key = None
    original = kwargs
    if f.random and "key" in kwargs and "key" not in f.params:
        kwargs = dict(kwargs)
        key = kwargs.pop("key")
    try:
        ba = f._sig.bind(*args, **kwargs)
    except TypeError:
        return f._direct(args, original)  # let Python raise its own error
    what = "" if auto else ".vectorize"
    leaves, batches, bshape = batch_plan(f, ba.arguments, what, auto)
    if not bshape:
        return f._direct(args, original)

    if not keys.tracing() and _config().compile_vectorized:
        out = _Compiled.of(f, auto).call(args, original)
        if out is not _UNCACHED:
            return out

    n = len(bshape)
    prepared: list[Any] = []
    levels: list[set[int]] = []
    if f.random and "key" not in f.params:  # one key per example
        if key is None:
            key = keys.fresh(f"{f.name}.vectorize()")
        if KeySpec.batch_shape(key):  # already one key per example
            prepared.append(jnp.broadcast_to(key, bshape))
        else:
            prepared.append(jr.split(key, bshape))
        levels.append(set(range(n)))
    for (spec, v, path), b in zip(leaves, batches):
        if isinstance(spec, KeySpec) and not b and spec.core_rank is not None and not (auto and _on_self(f, path)):
            split: Callable[[Any], Any] = lambda k: jr.split(k, bshape)  # noqa: E731
            for _ in range(spec.core_rank or 0):
                split = jax.vmap(split, out_axes=n)
            prepared.append(split(v))
            levels.append(set(range(n)))
            continue
        off = n - len(b)
        ones = tuple(i for i, s in enumerate(b) if _is_one(s))
        if ones:
            v = jnp.squeeze(jnp.asarray(v), axis=ones)
        prepared.append(v)
        levels.append({off + i for i in range(len(b)) if i not in ones})

    def core_fn(*lv: Any) -> Any:
        it = iter(lv)
        k = next(it) if f.random and "key" not in f.params else None
        new = f._walk(ba.arguments, lambda s, x, p: next(it))
        b2 = inspect.BoundArguments(f._sig, new)  # type: ignore[arg-type]
        return f._run(b2.args, b2.kwargs, key=k)

    g: Callable[..., Any] = core_fn
    if n > 1 and all(not lv or len(lv) == n for lv in levels):
        # every argument has all the batch axes or none: map over them as one
        # axis, so that each op inside sees one batch axis (see _flat_dot)
        g = _collapsed(core_fn, bshape, [bool(lv) for lv in levels])
    else:
        for level in reversed(range(n)):
            in_axes = tuple(0 if level in lv else None for lv in levels)
            if any(a is not None for a in in_axes):
                g = jax.vmap(g, in_axes=in_axes)
            else:
                g = jax.vmap(g, in_axes=in_axes, axis_size=bshape[level])
    token = _MAPPING.set(True)
    try:
        return g(*prepared)
    except ShapeError as e:
        if not auto:
            raise
        failed = getattr(e, "_expex_arguments", None)
        if failed is None or failed[0] is not f:
            raise
        raise _per_element(f, e, failed[1], leaves, batches) from None
    finally:
        _MAPPING.reset(token)


def _collapsed(core_fn: Callable[..., Any], bshape: tuple[Any, ...], batched: list[bool]) -> Callable[..., Any]:
    """``core_fn`` mapped over the leading axes ``bshape`` of the batched arguments, as one flattened axis."""
    n, size = len(bshape), math.prod(bshape)
    mapped = jax.vmap(core_fn, in_axes=tuple(0 if b else None for b in batched))

    def g(*lv: Any) -> Any:
        flat = [jnp.reshape(v, (size, *jnp.shape(v)[n:])) if b else v for v, b in zip(lv, batched)]
        return jax.tree.map(lambda y: jnp.reshape(y, (*bshape, *jnp.shape(y)[1:])), mapped(*flat))

    return g


# ---------------------------------------------- dots under nested mapped calls

#: Whether an expex mapped call is being traced (see ``_flat_dot``).
_MAPPING: contextvars.ContextVar[bool] = contextvars.ContextVar("expex_mapping", default=False)


def _flat_dot(axis_data: Any, args: Any, dims: Any, **params: Any) -> Any:
    """JAX's batching rule of ``dot_general``, which merges the mapped axis into the free axes it is next to.

    A layer written for one element and mapped over several axes (a per-point
    MLP in a per-ray function, each mapped over its batch) gets ``x @ w`` with
    ``x: [rays, points, I]``: JAX's rule makes the mapped axis one more free
    axis of the dot. Its gradient for ``w`` then contracts over two axes,
    which XLA:CPU computes by transposing the whole activation first, at about
    twice the cost of contracting a flattened ``[rays * points, I]``. Here,
    while an expex mapped call is traced, an operand mapped over an axis next
    to its free axes is reshaped so that they are one axis, and the result
    reshaped back: same values, one free axis per operand. Anything else, and
    every ``vmap`` outside expex's mapped calls, goes to JAX's rule.
    """
    if _MAPPING.get():
        out = _flat_dot_general(args, dims, **params)
        if out is not None:
            return out
    return _jax_dot_rule(axis_data, args, dims, **params)


def _flat_dot_general(args: Any, dims: Any, *, dimension_numbers: Any, **params: Any) -> Any:
    (lc, rc), (lb, rb) = dimension_numbers
    if lb or rb or (dims[0] is None) == (dims[1] is None):
        return None  # dot batch axes, or both operands mapped: JAX's rule makes the axis a batch axis
    if params.get("out_sharding") is not None or params.get("reduction_order") is not None:
        return None
    side = 0 if dims[0] is not None else 1
    x, bd = args[side], dims[side]
    if type(bd) is not int:
        return None
    contract = lc if side == 0 else rc
    rank = jnp.ndim(x) - 1  # without the mapped axis
    free = [d for d in range(rank) if d not in contract]
    if not free:
        return None  # one free axis after mapping already
    # the mapped axis and the free axes, in x's own axis order: they must be adjacent
    group = sorted([bd] + [d + (d >= bd) for d in free])
    start = group[0]
    if group != list(range(start, start + len(group))):
        return None
    shape = jnp.shape(x)
    sizes = shape[start : start + len(group)]
    merged = jnp.reshape(x, (*shape[:start], math.prod(sizes), *shape[start + len(group) :]))
    k = len(group) - 1
    new_contract = tuple(c + (c >= bd) for c in contract)
    new_contract = tuple(c if c < start else c - k for c in new_contract)
    if side == 0:
        numbers = ((new_contract, rc), ((), ()))
        out = jax.lax.dot_general(merged, args[1], numbers, precision=params.get("precision"),
                                  preferred_element_type=params.get("preferred_element_type"))  # fmt: skip
        at = 0  # the lhs's one free axis comes first
    else:
        numbers = ((lc, new_contract), ((), ()))
        out = jax.lax.dot_general(args[0], merged, numbers, precision=params.get("precision"),
                                  preferred_element_type=params.get("preferred_element_type"))  # fmt: skip
        at = jnp.ndim(out) - 1  # after the lhs's free axes
    oshape = jnp.shape(out)
    out = jnp.reshape(out, (*oshape[:at], *sizes, *oshape[at + 1 :]))
    return out, at + group.index(bd)


_jax_dot_rule = batching_rules.fancy_primitive_batchers[jax.lax.dot_general_p]
batching_rules.fancy_primitive_batchers[jax.lax.dot_general_p] = _flat_dot


def batch_plan(
    f: Fn[Any, Any], arguments: Any, what: str, auto: bool = False
) -> tuple[list[tuple[Spec, Any, str]], list[tuple[Any, ...]], tuple[Any, ...]]:
    """Split each array argument's shape into batch axes + declared axes.

    ``what`` names the call in messages (``".vectorize"``, or ``""`` for a plain
    call). With ``auto``, an argument with a variadic annotation, or on ``self``,
    has no batch axes; otherwise a variadic annotation is a ``TypeError``.
    """
    if f._lazy:
        f._resolve_owner()
    leaves: list[tuple[Spec, Any, str]] = []

    def collect(spec: Spec, v: Any, path: str) -> Any:
        leaves.append((spec, v, path))
        return v

    f._walk(arguments, collect)
    batches: list[tuple[Any, ...]] = []
    for spec, v, path in leaves:
        core = spec.core_rank  # type: ignore[attr-defined]
        if core is None or (auto and _on_self(f, path)):
            if auto:
                batches.append(())
                continue
            raise coded("EX106", TypeError(
                f"{f.name}{what}(): `{path}: {spec}` has a variadic shape, so its "
                f"batch axes cannot be told apart from its own axes."
            ))
        if isinstance(spec, KeySpec):
            shape = KeySpec.batch_shape(v) or ()
        else:
            shape = tuple(jnp.shape(v))
        nb = len(shape) - core
        batches.append(shape[:nb] if nb > 0 else ())
    try:
        bshape = tuple(jnp.broadcast_shapes(*batches)) if batches else ()
    except ValueError:
        listing = "\n".join(
            f"    {path}: {describe(v)}  = batch {_axes(b)} + {spec}"
            for (spec, v, path), b in zip(leaves, batches)
            if b
        )
        head = (
            f"{f.name}(): the arguments have extra leading axes that do not broadcast "
            f"against each other, so they cannot be mapped over."
            if auto
            else f"{f.name}{what}(): the batch axes of the arguments do not broadcast."
        )
        raise coded("EX105", ShapeError(f"{head}\n\n{listing}\n\n    {f.signature}")) from None
    return leaves, batches, bshape


def _axes(b: tuple[Any, ...]) -> str:
    return "[" + ", ".join(map(str, b)) + "]"


def _per_element(
    f: Fn[Any, Any], e: ShapeError, path: str, leaves: list[tuple[Spec, Any, str]], batches: list[tuple[Any, ...]]
) -> ShapeError:
    """The error of an automatically mapped call whose elements do not match: says what was mapped.

    ``e`` is the error of one element's call, about the argument at ``path``.
    """
    lines = str(e).split("\n")
    lines[0] = lines[0].removesuffix(".") + ", even with the extra leading axes mapped over."
    if len(lines) > 3 and lines[3].startswith("    received") and any(p == path and b for (_, _, p), b in zip(leaves, batches)):
        lines[3] += "  (one element)"
    mapped = [
        f"`{path}: {describe(v)}` is a batch {_axes(b)} of {spec}"
        for (spec, v, path), b in zip(leaves, batches)
        if b
    ]
    note = "    Extra leading axes are mapped over automatically: " + "; ".join(mapped) + "."
    at = 5  # after the reason (see messages.error): head, "", declared, received, "", reason
    while at < len(lines) and lines[at] != "":
        at += 1
    lines[at:at] = ["", note]
    out = coded(e.code or "EX101", ShapeError("\n".join(lines)))
    out.__notes__ = list(getattr(e, "__notes__", ()))
    return out


def _config() -> Any:
    from .function import config

    return config


def _is_one(s: Any) -> bool:
    try:
        return bool(s == 1)
    except Exception:
        return False


# ------------------------------------------------------- compiled eager calls

_UNCACHED = object()
#: How many abstract signatures a function is compiled for before it stops caching.
_MAX_SIGNATURES = 32


class _Compiled:
    """The compiled eager calls of one function: ``jax.jit`` of its mapped call.

    The first call with an abstract signature runs as before (traced op by op
    under ``vmap``); later ones run the compiled function. Nothing is cached
    while a call hook (a diagram, ``watch``) is installed, inside a JAX
    transformation, or when an argument cannot be part of a cache key. If the
    function turns out not to trace under ``jit`` (it reads a mapped-over
    argument's value, or draws a key from a scope opened outside), it is never
    compiled again.
    """

    def __init__(self, f: Fn[Any, Any], auto: bool):
        self.owner = weakref.ref(f)
        self.auto = auto
        self.seen: set[Any] = set()
        self.jits: dict[Any, Any] = {}
        self.off = False

    @staticmethod
    def of(f: Fn[Any, Any], auto: bool) -> "_Compiled":
        slot = "_compiled_auto" if auto else "_compiled"
        c = vars(f).get(slot)
        if c is None or c.owner() is not f:  # absent, or copied along with the Fn
            c = _Compiled(f, auto)
            vars(f)[slot] = c
        return c

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        __tracebackhide__ = True
        from .function import CALL_HOOK

        config = _config()
        if self.off or CALL_HOOK.get() is not None:
            return _UNCACHED
        f = self.owner()
        assert f is not None
        settings = (config.checked, config.lower_loops, config.notes, config.vectorize)
        sig = _signature(f, args, kwargs, settings)
        if sig is None:
            return _UNCACHED
        if sig not in self.seen:
            if len(self.seen) >= _MAX_SIGNATURES:
                self.off = True
            self.seen.add(sig)
            return _UNCACHED
        if f.random and "key" not in f.params and "key" not in kwargs:
            kwargs = {**kwargs, "key": keys.fresh(f"{f.name}()")}  # drawn outside, as expex.jit does
        compiled = self.jits.get(settings) or self._build(f, settings)
        try:
            return compiled(*args, **kwargs)
        except (StageError, ShapeError, TypeError):
            # raised while tracing for jit (an argument jit cannot take, a value read
            # in Python, a key from outside): nothing ran, so run it as before, always
            self.off = True
            self.jits.clear()
            return _UNCACHED

    def _build(self, f: Fn[Any, Any], settings: Any) -> Any:
        from .decorators import static_argnames

        auto = self.auto

        def mapped(*args: Any, **kwargs: Any) -> Any:
            __tracebackhide__ = True
            return vectorize(f, args, kwargs, auto=auto)

        mapped.__signature__ = f._sig  # type: ignore[attr-defined]
        mapped.__qualname__ = f"{f.name}.vectorize"
        compiled = jax.jit(mapped, static_argnames=tuple(static_argnames(f)))
        self.jits[settings] = compiled
        return compiled


def _signature(f: Fn[Any, Any], args: tuple[Any, ...], kwargs: dict[str, Any], settings: Any) -> Any:
    """A hashable abstract signature of a call (shapes and dtypes, static values), or None."""
    try:
        leaves, tree = jax.tree.flatten((args, kwargs))
        out: list[Any] = [tree, settings]
        for x in leaves:
            if isinstance(x, Tracer):
                return None
            if isinstance(x, (jax.Array, np.ndarray, np.generic)):
                out.append((x.shape, x.dtype, getattr(x, "weak_type", False)))
            elif isinstance(x, (float, complex)):
                out.append(type(x))
            else:
                hash(x)
                out.append((type(x), x))
        sig = tuple(out)
        hash(sig)
        return sig
    except Exception:
        return None

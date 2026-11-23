"""``f.lift(...)``: call an ``@fn`` on arguments with extra leading batch axes."""

import inspect
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from . import _rng
from ._dims import describe
from ._errors import ShapeError
from ._spec import KeySpec, Spec

if TYPE_CHECKING:
    from ._fn import Fn


def lift(f: Fn[Any, Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Call the function with extra leading batch axes on any array argument.

    Each argument's batch axes are whatever precedes its declared shape.
    Arguments without extra axes are passed to every call unchanged; batch
    axes of different arguments broadcast against each other. The result
    gains the batch axes in front.

    A single PRNG key is split into one key per batch element, so examples
    do not share randomness.
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
    try:
        leaves, batches, bshape = batch_plan(f, ba.arguments, "lift")
    except TypeError:
        if not f.auto_lift:
            raise
        return f._direct(args, original)  # variadic shapes: nothing to infer
    if not bshape:
        return f._direct(args, original)

    n = len(bshape)
    prepared: list[Any] = []
    levels: list[set[int]] = []
    if f.random and "key" not in f.params:  # one key per example
        if key is None:
            key = _rng.fresh(f"{f.name}.lift()")
        if KeySpec.batch_shape(key):  # already one key per example
            prepared.append(jnp.broadcast_to(key, bshape))
        else:
            prepared.append(jr.split(key, bshape))
        levels.append(set(range(n)))
    for (spec, v, path), b in zip(leaves, batches):
        if isinstance(spec, KeySpec) and not b:
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
    for level in reversed(range(n)):
        in_axes = tuple(0 if level in lv else None for lv in levels)
        if any(a is not None for a in in_axes):
            g = jax.vmap(g, in_axes=in_axes)
        else:
            g = jax.vmap(g, in_axes=in_axes, axis_size=bshape[level])
    return g(*prepared)


def batch_plan(
    f: Fn[Any, Any], arguments: Any, what: str
) -> tuple[list[tuple[Spec, Any, str]], list[tuple[Any, ...]], tuple[Any, ...]]:
    """Split each array argument's shape into batch axes + declared axes."""
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
        if core is None:
            raise TypeError(
                f"{f.name}.{what}(): `{path}: {spec}` has a variadic shape, so its "
                f"batch axes cannot be told apart from its own axes."
            )
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
            f"    {path}: {describe(v)}  = batch {b} + {spec}"
            for (spec, v, path), b in zip(leaves, batches)
            if b
        )
        raise ShapeError(
            f"{f.name}.{what}(): the batch axes of the arguments do not broadcast.\n\n"
            f"{listing}\n\n    {f.signature}"
        ) from None
    return leaves, batches, bshape


def _is_one(s: Any) -> bool:
    try:
        return bool(s == 1)
    except Exception:
        return False

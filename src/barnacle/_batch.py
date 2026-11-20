"""``f.batched(...)``: vectorise and compile once for any batch size."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp

from . import _rng
from ._fn import Fn
from ._spec import KeySpec


class Batched:
    """A lifted, jit-compiled function with a fixed set of compiled batch sizes.

    Call it like ``f.lift``. ``traces`` counts how many times it was traced
    (and so compiled); it grows with the number of distinct ``sizes`` used and
    argument layouts seen, never with the batch size.
    """

    def __init__(self, fn: Fn[Any, Any], sizes: tuple[int, ...]):
        if not sizes or any((not isinstance(s, int)) or s < 1 for s in sizes):
            raise ValueError(f"batch sizes must be positive ints, got {sizes!r}")
        self.fn = fn
        self.sizes = tuple(sorted(set(sizes)))
        self.traces = 0
        self._cache: dict[Any, Callable[..., Any]] = {}

    def __repr__(self) -> str:
        return f"<batched {self.fn.signature} sizes={self.sizes}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        f = self.fn
        ba = f._sig.bind(*args, **kwargs)
        leaves, batches, bshape = f._batch_plan(ba.arguments, "batched")
        if not bshape:
            return f(*args, **kwargs)
        n = math.prod(bshape)

        mapped: list[Any] = []
        consts: list[Any] = []
        pattern: list[bool] = []
        stand_ins: list[Any] = []
        for (spec, v, path), b in zip(leaves, batches):
            is_key = isinstance(spec, KeySpec)
            if is_key and not b:  # one key: give every example its own
                split: Callable[[Any], Any] = lambda k: jax.random.split(k, n)  # noqa: E731
                for _ in range(spec.core_rank or 0):
                    split = jax.vmap(split, out_axes=1)
                v = split(v)
                b = (n,)
            if b:
                v = jnp.asarray(v) if not is_key else v
                core = tuple(v.shape[len(b):])
                if b != (n,):
                    v = jnp.broadcast_to(v, bshape + core).reshape((n,) + core)
                mapped.append(v)
                pattern.append(True)
            else:
                core = tuple(jnp.shape(v))
                consts.append(v)
                pattern.append(False)
            dtype = v.dtype if hasattr(v, "dtype") else jnp.result_type(v)
            stand_ins.append(jax.ShapeDtypeStruct(core, dtype))
        keys = jax.random.split(_rng.fresh(f"{f.name}.batched()"), n) if f.random else None

        # The arguments with every array replaced by a shape/dtype stand-in: this
        # is what the compiled function is specialised on.
        it = iter(stand_ins)
        template = f._walk(ba.arguments, lambda s, x, p: next(it))
        try:
            cache_key = (tuple(pattern), _freeze(template))
            hash(cache_key)
        except TypeError:
            raise TypeError(
                f"{f.name}.batched(): the non-array arguments must be hashable, because the "
                f"compiled function is specialised on them. Annotate array arguments "
                f"(e.g. f32[...]) so that they are treated as data."
            ) from None
        run = self._cache.get(cache_key)
        if run is None:
            run = self._cache[cache_key] = self._compile(template, tuple(pattern))

        outs: list[Any] = []
        pos = 0
        while pos < n:
            rem = n - pos
            size = max((s for s in self.sizes if s <= rem), default=self.sizes[0])
            take = min(size, rem)
            part = tuple(_pad(m[pos : pos + take], size) for m in mapped)
            kpart = None if keys is None else _pad(keys[pos : pos + take], size)
            out = run(part, tuple(consts), kpart)
            outs.append(out if take == size else jax.tree.map(lambda a: a[:take], out))
            pos += take
        res = outs[0] if len(outs) == 1 else jax.tree.map(lambda *a: jnp.concatenate(a), *outs)
        if bshape != (n,):
            res = jax.tree.map(lambda a: a.reshape(bshape + a.shape[1:]), res)
        return res

    def _compile(self, template: dict[str, Any], pattern: tuple[bool, ...]) -> Callable[..., Any]:
        f = self.fn

        def one(m: tuple[Any, ...], c: tuple[Any, ...], k: Any) -> Any:
            im, ic, ip = iter(m), iter(c), iter(pattern)
            new = f._walk(template, lambda s, x, p: next(im) if next(ip) else next(ic))
            b2 = inspect.BoundArguments(f._sig, new)  # type: ignore[arg-type]
            return f._run(b2.args, b2.kwargs, key=k)

        def many(m: tuple[Any, ...], c: tuple[Any, ...], k: Any) -> Any:
            self.traces += 1
            return jax.vmap(one, in_axes=(0, None, None if k is None else 0))(m, c, k)

        return jax.jit(many)


def _pad(a: Any, size: int) -> Any:
    """Pad the leading axis to ``size`` by repeating the last row."""
    short = size - a.shape[0]
    if short <= 0:
        return a
    return jnp.concatenate([a, jnp.broadcast_to(a[-1:], (short,) + a.shape[1:])])


def _freeze(x: Any) -> Any:
    if isinstance(x, jax.ShapeDtypeStruct):
        return ("array", tuple(x.shape), str(x.dtype))
    if isinstance(x, dict):
        return ("dict", tuple((k, _freeze(v)) for k, v in x.items()))
    if isinstance(x, (list, tuple)) and type(x) in (list, tuple):
        return (type(x).__name__, tuple(_freeze(v) for v in x))
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return (type(x), tuple(_freeze(v) for v in x))
    import dataclasses

    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x), tuple((fl.name, _freeze(getattr(x, fl.name))) for fl in dataclasses.fields(x)))
    return x

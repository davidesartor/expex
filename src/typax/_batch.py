"""``f.batched(...)``: vectorise and compile once for any batch size."""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable

import numpy as np

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

        # Everything that depends on the batch size happens in NumPy on the host:
        # a JAX operation would itself be compiled once per new shape, which is
        # exactly the cost this method exists to avoid.
        mapped: list[Any] = []
        consts: list[Any] = []
        modes: list[Any] = []
        stand_ins: list[Any] = []
        for (spec, v, path), b in zip(leaves, batches):
            is_key = isinstance(spec, KeySpec)
            typed = is_key and _is_typed_key(v)
            if b:
                host = np.asarray(jax.random.key_data(v) if typed else v)
                core = tuple(host.shape[len(b):])
                if b != (n,):
                    host = np.broadcast_to(host, bshape + core).reshape((n,) + core)
                mapped.append(host)
                modes.append(("mapkey", str(jax.random.key_impl(v))) if typed else ("map", None))
                core = core[:-1] if typed else core
            else:
                core = tuple(jnp.shape(v))
                consts.append(v)
                # one key for the whole batch: every example derives its own from it
                modes.append(("foldkey", spec.core_rank or 0) if is_key else ("const", None))  # type: ignore[attr-defined]
            dtype = v.dtype if hasattr(v, "dtype") else jnp.result_type(v)
            stand_ins.append(jax.ShapeDtypeStruct(core, dtype))
        base_key = _rng.fresh(f"{f.name}.batched()") if f.random else None
        index = np.arange(n, dtype=np.uint32)

        # The arguments with every array replaced by a shape/dtype stand-in: this
        # is what the compiled function is specialised on.
        it = iter(stand_ins)
        template = f._walk(ba.arguments, lambda s, x, p: next(it))
        # Arrays inside arguments that have no shape annotation (a params pytree,
        # `self`, ...) are passed in as data too, unmapped, rather than being
        # frozen into the compiled function.
        for name, value in list(template.items()):
            if not f.params[name].spec.leafy and name not in f.static_names:
                template[name] = jax.tree.map(lambda x: _slot(x, consts), value)
        try:
            cache_key = (tuple(modes), _freeze(template))
            hash(cache_key)
        except TypeError:
            raise TypeError(
                f"{f.name}.batched(): the non-array arguments must be hashable, because the "
                f"compiled function is specialised on them."
            ) from None
        run = self._cache.get(cache_key)
        if run is None:
            run = self._cache[cache_key] = self._compile(template, tuple(modes))

        outs: list[Any] = []
        pos = 0
        while pos < n:
            rem = n - pos
            size = max((s for s in self.sizes if s <= rem), default=self.sizes[0])
            take = min(size, rem)
            part = tuple(_pad(m[pos : pos + take], size) for m in mapped)
            out = run(part, tuple(consts), _pad(index[pos : pos + take], size), base_key)
            outs.append(jax.tree.map(lambda a: _to_host(a)[:take], out))
            pos += take
        res = outs[0] if len(outs) == 1 else jax.tree.map(lambda *a: np.concatenate(a), *outs)
        if bshape != (n,):
            res = jax.tree.map(lambda a: a.reshape(bshape + a.shape[1:]), res)
        return jax.tree.map(jnp.asarray, res)

    def _compile(self, template: dict[str, Any], modes: tuple[Any, ...]) -> Callable[..., Any]:
        f = self.fn

        def one(m: tuple[Any, ...], c: tuple[Any, ...], i: Any, base: Any) -> Any:
            im, ic, imode = iter(m), iter(c), iter(modes)

            def leaf(s: Any, x: Any, p: str) -> Any:
                mode, arg = next(imode)
                if mode == "map":
                    return next(im)
                if mode == "mapkey":
                    return jax.random.wrap_key_data(next(im), impl=arg)
                if mode == "const":
                    return next(ic)
                fold: Callable[[Any], Any] = lambda k: jax.random.fold_in(k, i)  # noqa: E731
                for _ in range(arg):
                    fold = jax.vmap(fold)
                return fold(next(ic))

            new = f._walk(template, leaf)
            for name, value in new.items():
                if not f.params[name].spec.leafy and name not in f.static_names:
                    new[name] = jax.tree.map(
                        lambda x: next(ic) if isinstance(x, _Slot) else x, value
                    )
            b2 = inspect.BoundArguments(f._sig, new)  # type: ignore[arg-type]
            key = None if base is None else jax.random.fold_in(base, i)
            return f._run(b2.args, b2.kwargs, key=key)

        def many(m: tuple[Any, ...], c: tuple[Any, ...], i: Any, base: Any) -> Any:
            self.traces += 1
            return jax.vmap(one, in_axes=(0, None, 0, None))(m, c, i, base)

        return jax.jit(many)


def _is_typed_key(v: Any) -> bool:
    try:
        return bool(jnp.issubdtype(v.dtype, jax.dtypes.prng_key))
    except Exception:
        return False


def _to_host(a: Any) -> Any:
    if _is_typed_key(a):
        raise TypeError("batched() cannot return PRNG keys; return jax.random.key_data(key)")
    return np.asarray(a)


class _Slot:
    """Marks where an array was taken out of an unannotated argument."""

    def __init__(self, shape: tuple[Any, ...], dtype: Any):
        self.sig = (tuple(shape), str(dtype))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Slot) and other.sig == self.sig

    def __hash__(self) -> int:
        return hash(self.sig)


def _slot(x: Any, consts: list[Any]) -> Any:
    if isinstance(x, (jax.Array, np.ndarray)):
        consts.append(x)
        return _Slot(x.shape, x.dtype)
    return x


def _pad(a: Any, size: int) -> Any:
    """Pad the leading axis of a host array to ``size`` by repeating the last row."""
    short = size - a.shape[0]
    if short <= 0:
        return a
    return np.concatenate([a, np.broadcast_to(a[-1:], (short,) + a.shape[1:])])


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

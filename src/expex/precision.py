"""Mixed precision: compute in one dtype, store in another.

::

    @expex.fn(compute=jnp.bfloat16)
    def mlp[D, F](x: f32[D], w1: f32[D, F], w2: f32[F, D]) -> f32[D]:
        return jax.nn.gelu(x @ w1) @ w2         # bf16 matmuls; the caller gets f32

Annotations keep meaning the storage dtype: arguments are checked as they are
passed (``f32``), then the float arrays among them, struct fields included,
are cast to the compute dtype for the body. Float results are cast back to the
dtype their annotation declares. Integer and bool arrays, keys and Python
numbers are left alone.

The setting comes from, innermost first: the function (``@expex.fn(compute=)``),
its struct (``@expex.struct(compute=)``, for the methods defined in the class),
the body of a mixed-precision function it is called from, and
``expex.config.compute``. Inside a mixed-precision body:

- a function without a setting of its own gets values already in the compute
  dtype and computes in it; its annotations accept the compute dtype wherever
  they declare a float dtype (:func:`accepts`);
- a function with its own setting (``compute=jnp.float32`` for a softmax, say)
  casts to its dtype, and its float results come back in the caller's compute
  dtype, so the caller keeps computing in one dtype.

A function with its own (or its struct's) setting has its body wrapped once,
when it is decorated (:func:`install`); every other function runs as before,
so the feature costs nothing where it is not used. ``expex.config.compute``
is looked at by each call while it is set (one test while it is not).
"""

import contextvars
import copy
import functools
import weakref
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .annotations.errors import coded
from .annotations.spec import ArraySpec

#: The compute dtype of the mixed-precision body being run, if any.
REGION: contextvars.ContextVar[np.dtype | None] = contextvars.ContextVar("expex_compute", default=None)

_FLOATING: dict[Any, bool] = {}


def _floating(dtype: Any) -> bool:
    try:
        return _FLOATING[dtype]
    except (KeyError, TypeError):
        pass
    try:
        ok = bool(jnp.issubdtype(dtype, jnp.floating))
    except TypeError:
        ok = False
    try:
        _FLOATING[dtype] = ok
    except TypeError:
        pass
    return ok


def dtype_of(compute: Any, who: str = "compute") -> np.dtype | None:
    """``compute=`` as a NumPy dtype (``jnp.bfloat16``, ``"bfloat16"``, ...); None stays None."""
    if compute is None:
        return None
    try:
        dtype = np.dtype(jnp.dtype(compute))
    except TypeError:
        raise coded("EX801", TypeError(f"{who}={compute!r}: expected a float dtype such as jnp.bfloat16.")) from None
    if not _floating(dtype):
        raise coded("EX801", TypeError(f"{who}={compute!r}: the compute dtype must be a float dtype, such as jnp.bfloat16."))
    return dtype


def accepts(dtype: Any, declared: Any) -> bool:
    """Whether a value of ``dtype`` passes for an annotation of dtype ``declared`` after all.

    Asked only when the plain dtype check failed: inside a mixed-precision body,
    the compute dtype stands for any declared float dtype.
    """
    region = REGION.get()
    if region is None or dtype != region:
        return False
    return _floating(declared.dtype if declared.dtype is not None else declared.example)


def _cast(x: Any, dtype: Any) -> Any:
    dt = getattr(x, "dtype", None)
    if dt is None or dt == dtype or not _floating(dt):
        return x
    return jax.lax.convert_element_type(x, dtype)


def install(f: Any, compute: Any) -> None:
    """Make the ``Fn`` ``f`` compute in ``compute``: its body is wrapped to cast in and out.

    Installing again replaces the setting; ``None`` removes it.
    """
    dtype = dtype_of(compute)
    body = getattr(f._impl, "__expex_body__", f._impl)
    f.compute = dtype
    f._impl = body if dtype is None else _wrap(f, body, dtype)


def _wrap(f: Any, body: Callable[..., Any], dtype: np.dtype) -> Callable[..., Any]:
    def mixed(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        outer = REGION.get()
        args, kwargs = jax.tree.map(lambda x: _cast(x, dtype), (args, kwargs))
        token = REGION.set(dtype)
        try:
            out = body(*args, **kwargs)
        finally:
            REGION.reset(token)
        return _restore(f, out, outer)

    functools.update_wrapper(mixed, body)  # keeps what the converter attached (__expex_sizes__, ...)
    mixed.__expex_body__ = body  # type: ignore[attr-defined]
    return mixed


def _restore(f: Any, out: Any, outer: np.dtype | None) -> Any:
    """Float results in the caller's compute dtype, or else in the dtype their annotation declares."""
    if f.converted:
        from .staging.flow import unstack

        out = unstack(out)
    if f._lazy:
        f._resolve_owner()
    if f.ret.leafy:

        def visit(spec: Any, x: Any, path: str) -> Any:
            if not isinstance(spec, ArraySpec):
                return x
            target = outer if outer is not None else spec.dtype.dtype
            return x if target is None or not _floating(target) else _cast(x, target)

        try:
            return f.ret.walk(out, "return", visit)
        except Exception:  # the wrong structure: the result check says so
            return out
    if outer is not None:
        return jax.tree.map(lambda x: _cast(x, outer), out)
    return out


_GLOBAL: "weakref.WeakKeyDictionary[Any, dict[np.dtype, Any]]" = weakref.WeakKeyDictionary()
_NAMES: dict[Any, np.dtype | None] = {}


def under_config(f: Any) -> Any:
    """The ``Fn`` to run while ``expex.config.compute`` is set: ``f`` itself inside a
    mixed-precision body (it inherits the caller's dtype), else a copy of ``f`` that
    computes in the configured dtype."""
    if REGION.get() is not None:
        return f
    from .function import config

    raw = config.compute
    try:
        dtype = _NAMES[raw]
    except (KeyError, TypeError):
        dtype = dtype_of(raw, "expex.config.compute")
        try:
            _NAMES[raw] = dtype
        except TypeError:
            pass
    if dtype is None:
        return f
    variants = _GLOBAL.setdefault(f, {})
    g = variants.get(dtype)
    if g is None:
        if f._lazy:
            f._resolve_owner()
        g = variants[dtype] = copy.copy(f)
        install(g, dtype)
    return g

"""``@tx.struct``: a typed, immutable module class that JAX can trace.

::

    @tx.struct
    class Linear[I, O]:
        w: f32[O, I]
        b: f32[O]
        act: Callable = jnp.tanh          # a plain Python type: static

        def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
            return self.act(x @ self.w.T + self.b)

The class becomes a frozen dataclass registered as a JAX pytree: fields with
array annotations are traced leaves, fields with plain Python types (numbers,
strings, callables, tuples of those, ...) are static metadata. Every method
is an ``@fn``, so its arguments, its result and ``self``'s fields are
checked against the annotations, with the class's own dimensions shared.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, TypeVar, overload

import jax

from ._fn import Fn, fn
from ._spec import _class_annotations, static_type

T = TypeVar("T", bound=type)

_SKIP = {"__init__", "__post_init__", "__repr__", "__eq__", "__hash__", "__init_subclass__"}


@overload
def struct(cls: T, /) -> T: ...
@overload
def struct(*, jit: bool = False, **options: Any) -> Callable[[T], T]: ...
def struct(cls: Any = None, /, *, jit: bool = False, **options: Any) -> Any:
    """Turn a class into a typed module. See the module docstring.

    Options (``jit=True``, ``random=True``, ...) are passed to every method's
    ``@fn``. A method already decorated with ``@fn(...)`` keeps its own.
    """

    def build(original: Any) -> Any:
        cls: Any = original
        params = getattr(cls, "__dataclass_params__", None)
        if not (isinstance(cls, type) and params is not None and params.frozen):
            cls = dataclasses.dataclass(frozen=True)(cls)
        anns = _class_annotations(cls)
        data: list[str] = []
        meta: list[str] = []
        for f in dataclasses.fields(cls):
            (meta if static_type(anns.get(f.name, Any)) else data).append(f.name)
        jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)

        for name, value in list(vars(cls).items()):
            if name in _SKIP or (name.startswith("__") and name != "__call__"):
                continue
            wrapper = None
            if isinstance(value, (staticmethod, classmethod)):
                wrapper, value = type(value), value.__func__
            if not inspect.isfunction(value) and not isinstance(value, Fn):
                continue
            method = value if isinstance(value, Fn) else fn(jit=jit, **options)(value)
            method.__set_name__(cls, name)
            setattr(cls, name, wrapper(method) if wrapper else method)

        if "replace" not in vars(cls):

            def replace(self: Any, **changes: Any) -> Any:
                """A copy with some fields changed."""
                return dataclasses.replace(self, **changes)

            cls.replace = replace
        cls.__typax_struct__ = True
        return cls

    return build(cls) if cls is not None else build

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
``module.lift(x)`` maps ``__call__`` over extra leading axes of ``x``, and
``module.replace(field=...)`` returns a modified copy.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from typing import Any, Callable, TypeVar, overload

import jax
from jax.core import Tracer

from ._dims import Env, resolve
from ._errors import Mismatch, ShapeError, StageError
from . import _sym
from ._fn import Fn, fn, unwrap
from ._spec import _class_annotations, parse_owner, rebuild, static_type

T = TypeVar("T", bound=type)

_SKIP = {"__init__", "__post_init__", "__repr__", "__eq__", "__hash__", "__init_subclass__"}


@overload
def struct(cls: T, /) -> T: ...
@overload
def struct(**moved: Any) -> Callable[[T], T]: ...
def struct(cls: Any = None, /, **moved: Any) -> Any:
    """Turn a class into a typed module. See the module docstring.

    Methods are wrapped with ``@tx.fn``; one already decorated (``@tx.fn(...)``,
    ``@tx.lift``, ``@tx.random``, ``@tx.jit``) keeps its own options.

    A method's leading ``assert`` that only involves the class's sizes and
    fields, such as ``assert D % self.heads == 0``, is an invariant of the
    class: checked whenever an instance is passed to an ``@fn`` (or built by a hand-written ``__init__``), and
    assumed by every function that receives one (the prover does).
    """
    for k in moved:
        raise TypeError(
            f"@tx.struct got {k}=...: write invariants as an `assert` at the top of a "
            f"method instead, e.g. assert D % self.heads == 0."
        )

    localns = _sym.caller_locals()

    def build(original: Any) -> Any:
        cls: Any = original
        custom_init = "__init__" in vars(cls)  # dataclass() keeps a hand-written __init__
        if localns:
            cls.__typax_localns__ = localns
        params = vars(cls).get("__dataclass_params__")  # not a parent's
        if not (isinstance(cls, type) and params is not None and params.frozen):
            cls = dataclasses.dataclass(frozen=True)(cls)
        anns = _class_annotations(cls)
        data: list[str] = []
        meta: list[str] = []
        for f in dataclasses.fields(cls):
            (meta if static_type(anns.get(f.name, Any)) else data).append(f.name)
        _register(cls, tuple(data), tuple(meta))
        if custom_init:
            _wrap_init(cls)

        fields = {f.name for f in dataclasses.fields(cls)}
        for name, value in list(vars(cls).items()):
            if name in _SKIP or name in fields or (name.startswith("__") and name != "__call__"):
                continue
            wrapper = None
            if isinstance(value, (staticmethod, classmethod)):
                wrapper, value = type(value), value.__func__
            inner = unwrap(value)
            if inner is not value and isinstance(inner, Fn):  # e.g. @jax.jit over @tx.fn
                inner.__set_name__(cls, name)
                continue
            if inspect.isfunction(getattr(value, "__wrapped__", None)) and hasattr(value, "lower"):
                raise TypeError(
                    f"{cls.__name__}.{name} is jitted but not checked. Put @tx.fn under "
                    f"@jax.jit:\n\n    @jax.jit\n    @tx.fn\n    def {name}(self, ...): ..."
                )
            if not inspect.isfunction(value) and not isinstance(value, Fn):
                continue
            method = value if isinstance(value, Fn) else fn(value)
            method.__set_name__(cls, name)
            setattr(cls, name, wrapper(method) if wrapper else method)

        if isinstance(unwrap(getattr(cls, "__call__", None)), Fn) and "lift" not in vars(cls):

            def lift(self: Any, *args: Any, **kwargs: Any) -> Any:
                """Call the module on inputs with extra leading batch axes."""
                method: Any = unwrap(type(self).__call__)
                return method.__get__(self).lift(*args, **kwargs)

            cls.lift = lift
        if "replace" not in vars(cls):

            def replace(self: Any, **changes: Any) -> Any:
                """A copy with some fields changed (``__init__`` is not called)."""
                return rebuild(self, **changes)

            cls.replace = replace
        cls.__typax_struct__ = True
        cls.__typax_where__ = _invariants(cls)
        return cls

    return build(cls) if cls is not None else build


def _invariants(cls: Any) -> tuple[Any, ...]:
    """Leading asserts of the methods that only involve the class's sizes and fields."""
    names = {p.__name__ for p in getattr(cls, "__type_params__", ())}
    names |= {f.name for f in dataclasses.fields(cls)}
    out: list[Any] = []
    for base in reversed(cls.__mro__[1:]):
        out += [e for e in getattr(base, "__typax_where__", ()) if e not in out]
    for value in vars(cls).values():
        method = unwrap(getattr(value, "__func__", value))
        for e in getattr(method, "_asserted", ()):
            if e.names <= names and e not in out:
                out.append(e)
    return tuple(out)


def _register(cls: Any, data: tuple[str, ...], meta: tuple[str, ...]) -> None:
    """Register ``cls`` as a pytree that is rebuilt without calling ``__init__``."""
    from jax.tree_util import GetAttrKey

    def static(obj: Any) -> tuple[Any, ...]:
        out = tuple(getattr(obj, n) for n in meta)
        for n, v in zip(meta, out):
            if any(isinstance(x, Tracer) for x in jax.tree.leaves(v)):
                raise StageError(
                    f"{cls.__name__}.{n} is a static field (its annotation is a plain Python "
                    f"type), but it holds a traced value. Pass a Python value, or annotate "
                    f"the field with a dtype (f32, i32, ...) to make it an array."
                )
        return out

    def flatten_with_keys(obj: Any) -> Any:
        children = [(GetAttrKey(n), getattr(obj, n)) for n in data]
        return children, static(obj)

    def flatten(obj: Any) -> Any:
        return [getattr(obj, n) for n in data], static(obj)

    def unflatten(aux: tuple[Any, ...], children: Any) -> Any:
        obj = object.__new__(cls)
        for n, v in zip(data, children):
            object.__setattr__(obj, n, v)
        for n, v in zip(meta, aux):
            object.__setattr__(obj, n, v)
        return obj

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)


def _wrap_init(cls: Any) -> None:
    """Let a hand-written ``__init__`` assign fields, then freeze and check the object."""
    user_init = cls.__init__
    frozen_setattr = cls.__setattr__

    @functools.wraps(user_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "__typax_building__", True)
        try:
            user_init(self, *args, **kwargs)
        finally:
            object.__delattr__(self, "__typax_building__")
        missing = []
        for f in dataclasses.fields(cls):
            if f.name in self.__dict__:
                continue
            if f.default is not dataclasses.MISSING:
                object.__setattr__(self, f.name, f.default)
            elif f.default_factory is not dataclasses.MISSING:
                object.__setattr__(self, f.name, f.default_factory())
            else:
                missing.append(f.name)
        if missing:
            raise TypeError(f"{cls.__name__}.__init__ did not set {', '.join(missing)}")
        _check_new(cls, self)

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if self.__dict__.get("__typax_building__"):
            object.__setattr__(self, name, value)
        else:
            frozen_setattr(self, name, value)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__


def _check_new(cls: Any, obj: Any) -> None:
    """Check a freshly built object against its own annotations and invariants."""
    from ._fn import config

    if not config.checked:
        return
    spec = parse_owner(cls)
    if not spec.leafy:
        return
    env = Env()
    try:
        spec.match(obj, "self", env)
        if env.deferred:
            resolve(env)
    except Mismatch as m:
        from ._dims import describe

        raise ShapeError(
            f"{cls.__name__}(): the constructed object does not match its annotations.\n\n"
            f"    declared  {m.path}: {m.spec}\n"
            f"    got       {describe(m.value)}\n\n"
            f"    {m.reason}"
        ) from None

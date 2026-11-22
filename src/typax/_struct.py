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

from ._fn import Fn, fn
from ._errors import Mismatch, ShapeError
from ._spec import Env, _class_annotations, parse_owner, rebuild, resolve, static_type

T = TypeVar("T", bound=type)

_SKIP = {"__init__", "__post_init__", "__repr__", "__eq__", "__hash__", "__init_subclass__"}


@overload
def struct(cls: T, /) -> T: ...
@overload
def struct(
    *, where: str | list[str] | tuple[str, ...] = (), jit: bool = False, **options: Any
) -> Callable[[T], T]: ...
def struct(
    cls: Any = None,
    /,
    *,
    where: str | list[str] | tuple[str, ...] = (),
    jit: bool = False,
    **options: Any,
) -> Any:
    """Turn a class into a typed module. See the module docstring.

    ``where`` states invariants over the class's dimensions and int fields,
    e.g. ``where="D % heads == 0"``. They are checked whenever an instance is
    passed to an ``@fn``, and every function that receives an instance, however
    deeply nested, may assume them (the prover does).

    Other options (``jit=True``, ``random=True``, ...) are passed to every
    method's ``@fn``. A method already decorated with ``@fn(...)`` keeps its own.
    """
    from ._expr import Expr

    invariants = tuple(Expr(w, constraint=True) for w in ([where] if isinstance(where, str) else where))

    def build(original: Any) -> Any:
        cls: Any = original
        custom_init = "__init__" in vars(cls)  # dataclass() keeps a hand-written __init__
        params = getattr(cls, "__dataclass_params__", None)
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

        if isinstance(vars(cls).get("__call__"), Fn) and "lift" not in vars(cls):

            def lift(self: Any, *args: Any, **kwargs: Any) -> Any:
                """Call the module on inputs with extra leading batch axes."""
                method: Any = vars(type(self))["__call__"]
                return method.__get__(self).lift(*args, **kwargs)

            cls.lift = lift
        if "replace" not in vars(cls):

            def replace(self: Any, **changes: Any) -> Any:
                """A copy with some fields changed (``__init__`` is not called)."""
                return rebuild(self, **changes)

            cls.replace = replace
        cls.__typax_struct__ = True
        cls.__typax_where__ = invariants
        return cls

    return build(cls) if cls is not None else build


def _register(cls: Any, data: tuple[str, ...], meta: tuple[str, ...]) -> None:
    """Register ``cls`` as a pytree that is rebuilt without calling ``__init__``."""
    from jax.tree_util import GetAttrKey

    def flatten_with_keys(obj: Any) -> Any:
        children = [(GetAttrKey(n), getattr(obj, n)) for n in data]
        return children, tuple(getattr(obj, n) for n in meta)

    def flatten(obj: Any) -> Any:
        return [getattr(obj, n) for n in data], tuple(getattr(obj, n) for n in meta)

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
        from ._spec import describe

        raise ShapeError(
            f"{cls.__name__}(): the constructed object does not match its annotations.\n\n"
            f"    declared  {m.path}: {m.spec}\n"
            f"    got       {describe(m.value)}\n\n"
            f"    {m.reason}"
        ) from None

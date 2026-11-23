"""``@expex.struct``: a typed, immutable module class that JAX can trace.

::

    @expex.struct
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
``module.vectorize(x)`` maps ``__call__`` over extra leading axes of ``x``, and
``module.replace(field=...)`` returns a modified copy.

``Module.init(key, D=64)`` builds a struct's parameters from its annotations:

::

    @expex.struct
    class Linear[I, O]:
        w: f32[O, I] = jax.nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
        b: f32[O] = 0.0

    Linear.init(key, I=3, O=4)

An array field's default says how to make it: a JAX initializer (any function
of ``(key, shape, dtype)``) or a number to fill it with. Its shape comes from
the annotation, with the sizes given to ``init`` (and the int fields of the
struct). Nested structs are built the same way; a list of them needs its
length (``blocks=4``). Any field can be passed instead of built.
"""

import dataclasses
import functools
import inspect
from typing import Any, Callable, TypeVar, overload

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.core import Tracer

from .annotations import sym
from .annotations.dims import DConst, DExpr, DName, Env, describe, resolve
from .annotations.errors import Mismatch, ShapeError, StageError
from .annotations.spec import (
    ArraySpec, KeySpec, NoneSpec, SeqSpec, Spec, StructSpec, UnionSpec,
    _class_annotations, join, parse, parse_owner, rebuild, static_type,
)  # fmt: skip
from .decorators import CompiledMethod, fn
from .function import Fn, unwrap

T = TypeVar("T", bound=type)

_SKIP = {"__init__", "__post_init__", "__repr__", "__eq__", "__hash__", "__init_subclass__"}


@overload
def struct(cls: T, /) -> T: ...
@overload
def struct(**moved: Any) -> Callable[[T], T]: ...
def struct(cls: Any = None, /, **moved: Any) -> Any:
    """Turn a class into a typed module. See the module docstring.

    Methods are wrapped with ``@expex.fn``; one already decorated (``@expex.fn(...)``,
    ``@expex.vectorize``, ``@expex.random``, ``@expex.jit``) keeps its own options.

    A method's leading ``assert`` that only involves the class's sizes and
    fields, such as ``assert D % self.heads == 0``, is an invariant of the
    class: checked whenever an instance is passed to an ``@fn`` (or built by a hand-written ``__init__``), and
    assumed by every function that receives one (the prover does).
    """
    for k in moved:
        raise TypeError(
            f"@expex.struct got {k}=...: write invariants as an `assert` at the top of a "
            f"method instead, e.g. assert D % self.heads == 0."
        )

    localns = sym.caller_locals()

    def build(original: Any) -> Any:
        cls: Any = original
        custom_init = "__init__" in vars(cls)  # dataclass() keeps a hand-written __init__
        if localns:
            cls.__expex_localns__ = localns
        params = vars(cls).get("__dataclass_params__")  # not a parent's
        if not (isinstance(cls, type) and params is not None and params.frozen):
            mark(cls, _class_annotations(cls))  # array defaults are initializers
            cls = dataclasses.dataclass(frozen=True)(cls)
        anns = _class_annotations(cls)
        data: list[str] = []
        meta: list[str] = []
        for f in dataclasses.fields(cls):
            (meta if static_type(anns.get(f.name, Any)) else data).append(f.name)
        _register(cls, tuple(data), tuple(meta))
        if custom_init:
            _wrap_init(cls)
        else:
            _check_built(cls)
        if "init" not in vars(cls):
            cls.init = classmethod(init)

        fields = {f.name for f in dataclasses.fields(cls)}
        for name, value in list(vars(cls).items()):
            if name in _SKIP or name in fields or (name.startswith("__") and name != "__call__"):
                continue
            wrapper = None
            if isinstance(value, (staticmethod, classmethod)):
                wrapper, value = type(value), value.__func__
            inner = unwrap(value)
            if inner is not value and isinstance(inner, Fn):  # e.g. @jax.jit over @expex.fn
                inner.__set_name__(cls, name)
                if hasattr(value, "vectorize") and wrapper is None:  # @expex.jit: bind .vectorize too
                    setattr(cls, name, CompiledMethod(value))
                continue
            if inspect.isfunction(getattr(value, "__wrapped__", None)) and hasattr(value, "lower"):
                raise TypeError(
                    f"{cls.__name__}.{name} is jitted but not checked. Put @expex.fn under "
                    f"@jax.jit:\n\n    @jax.jit\n    @expex.fn\n    def {name}(self, ...): ..."
                )
            if not inspect.isfunction(value) and not isinstance(value, Fn):
                continue
            method = value if isinstance(value, Fn) else fn(value)
            method.__set_name__(cls, name)
            setattr(cls, name, wrapper(method) if wrapper else method)

        if isinstance(unwrap(getattr(cls, "__call__", None)), Fn) and "vectorize" not in vars(cls):

            def vectorize(self: Any, *args: Any, **kwargs: Any) -> Any:
                """Call the module on inputs with extra leading batch axes."""
                method: Any = unwrap(type(self).__call__)
                return method.__get__(self).vectorize(*args, **kwargs)

            cls.vectorize = vectorize
        if "replace" not in vars(cls):

            def replace(self: Any, **changes: Any) -> Any:
                """A copy with some fields changed (``__init__`` is not called)."""
                return rebuild(self, **changes)

            cls.replace = replace
        cls.__expex_struct__ = True
        cls.__expex_where__ = _invariants(cls)
        return cls

    return build(cls) if cls is not None else build


def _invariants(cls: Any) -> tuple[Any, ...]:
    """Leading asserts of the methods that only involve the class's sizes and fields."""
    names = {p.__name__ for p in getattr(cls, "__type_params__", ())}
    names |= {f.name for f in dataclasses.fields(cls)}
    out: list[Any] = []
    for base in reversed(cls.__mro__[1:]):
        out += [e for e in getattr(base, "__expex_where__", ()) if e not in out]
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
        object.__setattr__(self, "__expex_building__", True)
        try:
            user_init(self, *args, **kwargs)
        finally:
            object.__delattr__(self, "__expex_building__")
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
        unbuilt(cls, self)
        _check_new(cls, self)

    def __setattr__(self: Any, name: str, value: Any) -> None:
        if self.__dict__.get("__expex_building__"):
            object.__setattr__(self, name, value)
        else:
            frozen_setattr(self, name, value)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__


def _check_built(cls: Any) -> None:
    """The dataclass constructor refuses to leave an initializer in place of an array."""
    if not any(isinstance(f.default, (Initializer, _Required)) for f in dataclasses.fields(cls)):
        return
    generated = cls.__init__

    @functools.wraps(generated)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        generated(self, *args, **kwargs)
        unbuilt(cls, self)

    cls.__init__ = __init__


def _check_new(cls: Any, obj: Any) -> None:
    """Check a freshly built object against its own annotations and invariants."""
    from .function import config

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
        from .annotations.dims import describe

        raise ShapeError(
            f"{cls.__name__}(): the constructed object does not match its annotations.\n\n"
            f"    declared  {m.path}: {m.spec}\n"
            f"    got       {describe(m.value)}\n\n"
            f"    {m.reason}"
        ) from None


# ------------------------------------------------------------------ init


class Initializer:
    """How to make an array field: the default it was given in the class."""

    __slots__ = ("make",)

    def __init__(self, make: Any):
        self.make = make

    def __call__(self, key: Any, shape: tuple[int, ...], dtype: Any) -> Any:
        if callable(self.make):
            return self.make(key, shape, dtype)
        return jnp.full(shape, self.make, dtype)

    def __repr__(self) -> str:
        return f"<initializer {getattr(self.make, '__name__', self.make)!r}>"


class _Required:
    """Default of a field without one that comes after a field with one."""

    def __repr__(self) -> str:
        return "<required>"


REQUIRED = _Required()


def mark(cls: Any, anns: dict[str, Any]) -> None:
    """Turn the defaults of array fields into :class:`Initializer`s (before ``dataclass``).

    Fields keep their order, and a dataclass field without a default may not
    follow one with a default, so such fields get :data:`REQUIRED`, which the
    constructor refuses to leave in place, like an unbuilt initializer.
    """
    own = list(sym.texts(cls))  # names in order; the values may not evaluate (f32[2 * E, D])
    seen_default = False
    for name in own:
        default = vars(cls).get(name, dataclasses.MISSING)
        if default is dataclasses.MISSING:
            if seen_default:
                setattr(cls, name, REQUIRED)
            continue
        seen_default = True
        if default is dataclasses.MISSING or isinstance(default, (Initializer, jax.Array, np.ndarray)):
            continue
        if not (callable(default) or (isinstance(default, (int, float)) and not isinstance(default, bool))):
            continue
        try:
            spec = parse(anns.get(name))
        except Exception:
            continue
        if isinstance(spec, (ArraySpec, KeySpec)):
            setattr(cls, name, Initializer(default))


def unbuilt(cls: Any, obj: Any) -> None:
    """Refuse an object some of whose arrays were never made (constructed without them)."""
    left = [
        f.name for f in dataclasses.fields(cls)
        if isinstance(getattr(obj, f.name, None), (Initializer, _Required))
    ]  # fmt: skip
    if left:
        raise TypeError(
            f"{cls.__name__}(): {', '.join(left)} not given. Pass them, or build every "
            f"array from its annotation with {cls.__name__}.init(key, <sizes>)."
        )


def init(cls: Any, key: Any, **given: Any) -> Any:
    """Build ``cls`` from its annotations. See the module docstring."""
    spec = parse_owner(cls)
    named: set[str] = set()
    spec.names(named)  # sizes in the annotations, Dim classes included
    sizes = {p.__name__ for p in getattr(cls, "__type_params__", ())}
    sizes |= {n for n in named if not n.startswith("*") and "." not in n}
    fields = _all_fields(spec)
    sizes -= fields  # `heads` in f32[self.heads, D] is a field: its value is the size
    unknown = set(given) - sizes - fields
    if unknown:
        raise TypeError(
            f"{cls.__name__}.init(): {', '.join(sorted(unknown))} is neither a size "
            f"({', '.join(sorted(sizes)) or 'none'}) nor a field."
        )
    env = Env()
    for n in sizes & set(given):
        env.dims[n] = int(given[n])
        env.origin[n] = f"{n}={given[n]} passed to {cls.__name__}.init"
    obj = _build(cls, spec, key, env, {k: v for k, v in given.items() if k not in sizes}, cls.__name__)
    try:  # the whole object against its annotations, sizes shared
        spec.match(obj, cls.__name__, env)
        if env.deferred:
            resolve(env)
    except Mismatch as m:
        raise ShapeError(
            f"{cls.__name__}.init(): {m.path} is {describe(m.value)}, but is declared "
            f"{m.spec}.\n\n    {m.reason}"
        ) from None
    return obj


def _build(cls: Any, spec: Any, key: Any, env: Env, given: dict[str, Any], path: str) -> Any:
    fields = dataclasses.fields(cls)
    fields_here = {f.name for f in fields}
    specs: dict[str, Spec] = getattr(spec, "fields", {})
    obj = object.__new__(cls)
    values: dict[str, Any] = {}
    # plain fields first: their int values may be sizes of the arrays (f32[heads, D])
    for f in fields:
        s = specs.get(f.name)
        if f.name in given and not (isinstance(s, SeqSpec) and isinstance(given[f.name], int)):
            values[f.name] = given[f.name]
            if s is not None and s.leafy:
                try:
                    s.match(given[f.name], join(path, f.name), env)  # binds sizes from it
                except Mismatch as m:
                    raise ShapeError(f"{path}.init(): {m.path}: {m.reason}") from None
        elif s is None or not s.leafy:
            values[f.name] = _plain(f, path)
        if f.name in values and isinstance(values[f.name], (int, np.integer)) and not isinstance(values[f.name], bool):
            env.dims.setdefault(f.name, int(values[f.name]))
    keys = jr.split(key, max(len(fields), 1))
    for k, f in zip(keys, fields):
        if f.name not in values:
            deeper = {n: v for n, v in given.items() if n not in fields_here}
            values[f.name] = _make(specs[f.name], f, k, env, given.get(f.name), join(path, f.name), deeper)
    for name, v in values.items():
        object.__setattr__(obj, name, v)
    return obj


def _all_fields(spec: Any) -> set[str]:
    """Field names of a struct and of every struct inside it."""
    out: set[str] = set()
    todo = [spec]
    while todo:
        s = todo.pop()
        if isinstance(s, SeqSpec):
            todo.append(s.elem)
        elif isinstance(s, StructSpec):
            out |= set(s.fields)
            todo += list(s.fields.values())
    return out


def _plain(f: Any, path: str) -> Any:
    if f.default is not dataclasses.MISSING and not isinstance(f.default, (Initializer, _Required)):
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    raise TypeError(f"{path}.{f.name} has no default: pass {f.name}=... to init.")


def _make(spec: Spec, f: Any, key: Any, env: Env, count: Any, path: str, deeper: dict[str, Any]) -> Any:
    """Build one field. ``deeper``: the settings given to init, for nested structs."""
    if isinstance(spec, UnionSpec):  # `X | None`: None unless there is a way to build X
        if f.default is None:
            return None
        spec = next(s for s in spec.options if not isinstance(s, NoneSpec))
    if isinstance(spec, (ArraySpec, KeySpec)):
        how = f.default if isinstance(f.default, Initializer) else None
        if how is None:
            raise TypeError(
                f"{path}: {spec} has no initializer. Give the field a default (a JAX "
                f"initializer such as jax.nn.initializers.normal(), or a number), or "
                f"pass it to init."
            )
        if isinstance(spec, KeySpec):
            return jr.split(key, _shape(spec.dims, env, path)) if spec.dims else key
        dtype = spec.dtype.dtype if spec.dtype.dtype is not None else spec.dtype.example
        return how(key, _shape(spec.dims, env, path), dtype)
    if isinstance(spec, StructSpec):
        return _build(spec.cls, spec, key, env, deeper, path)
    if isinstance(spec, SeqSpec) and isinstance(spec.elem, StructSpec):
        if not isinstance(count, int):
            raise TypeError(f"{path} is a list of {spec.elem}: pass its length, e.g. {f.name}=4, to init.")
        items = [
            _build(spec.elem.cls, spec.elem, k, env, deeper, f"{path}[{i}]")
            for i, k in enumerate(jr.split(key, count))
        ]  # fmt: skip
        return items if spec.kind is list else tuple(items)
    raise TypeError(f"{path}: cannot build a {spec} from its annotation; pass it to init.")


def _shape(dims: tuple[Any, ...], env: Env, path: str) -> tuple[int, ...]:
    out = []
    for d in dims:
        if isinstance(d, DConst):
            out.append(d.value)
        elif isinstance(d, DName) and d.name in env.dims:
            out.append(int(env.dims[d.name]))
        elif isinstance(d, DExpr) and d.expr.names <= env.dims.keys():
            out.append(int(d.expr(env.dims)))
        else:
            names = sorted({d.name} if isinstance(d, DName) else set(getattr(getattr(d, "expr", None), "names", ())) - env.dims.keys())
            raise TypeError(f"{path}: the size {', '.join(names) or d} is not known: pass it to init, e.g. {names[0] if names else 'D'}=64.")
    return tuple(out)

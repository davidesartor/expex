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

``Module(key, D=64)`` builds a struct's parameters from its annotations:

::

    @expex.struct
    class Linear[I, O]:
        w: f32[O, I] = jax.nn.initializers.lecun_normal(in_axis=-1, out_axis=-2)
        b: f32[O] = 0.0

    Linear(key, I=3, O=4)                     # built: a PRNG key first, sizes by name
    Linear(jnp.ones((4, 3)), jnp.zeros(4))    # field by field, as for any dataclass

An array field's default says how to make it: a JAX initializer (any function
of ``(key, shape, dtype)``) or a number to fill it with. Its shape comes from
the annotation, with the sizes given by name (and the int fields of the
struct). Nested structs are built the same way; a list of them needs its
length (``blocks=4``). Any field can be passed instead of built. A class with
its own ``__init__`` keeps it, and is built with ``expex.build(Cls, key, D=64)``.

``model.at.blocks[5].w.set(v)`` returns a copy of ``model`` with one part
changed (also ``.add``, ``.multiply``, ``.apply`` and ``.get``), checked
against the annotations.
"""

import dataclasses
import functools
import inspect
import operator
import re
from typing import Any, Callable, Generic, TypeVar, overload

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.core import Tracer

from .annotations import sym
from .annotations.dims import DConst, DExpr, DName, Env, describe, resolve
from .annotations.errors import Mismatch, ShapeError, StageError
from .annotations.spec import (
    ArraySpec, DictSpec, KeySpec, NoneSpec, SeqSpec, Spec, StaticSpec, StructSpec, TupleSpec,
    UnionSpec, _class_annotations, join, parse, parse_owner, rebuild, static_type,
)  # fmt: skip
from .decorators import CompiledMethod, fn
from .function import Fn, unwrap

T = TypeVar("T", bound=type)
M = TypeVar("M")

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
        own = vars(cls).get("__init__")  # dataclass() keeps a hand-written __init__
        custom_init = own is not None and not getattr(own, "__expex_generated__", False)
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
        elif not getattr(vars(cls).get("__init__"), "__expex_generated__", False):
            _constructor(cls)

        fields = {f.name for f in dataclasses.fields(cls)}
        if "at" not in vars(cls) and "at" not in fields:
            cls.at = _AT
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


class _NotGiven:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<not given>"


_NOT_GIVEN = _NotGiven()
_U32 = np.dtype("uint32")
try:
    from jax._src.array import ArrayImpl as _ArrayImpl
except ImportError:  # pragma: no cover - every array then takes the longer look
    _ArrayImpl = type("_NoArrayImpl", (), {})
#: Types whose values are never keys: the first argument of a constructor is not looked at closer.
_PLAIN: set[type] = {list, tuple, dict, int, float, bool, str, type(None)}


def is_key(x: Any) -> bool:
    """A single PRNG key: a typed key (``jr.key``) or a raw ``uint32[2]`` (``jr.PRNGKey``)."""
    dtype = getattr(x, "dtype", None)
    if dtype is None:
        return False
    shape = getattr(x, "shape", None)
    if isinstance(dtype, np.dtype):
        return dtype == _U32 and tuple(shape or ()) == (2,)
    try:
        return bool(jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)) and tuple(shape or ()) == ()
    except TypeError:
        return False


def _accepts(cls: Any, name: str, value: Any) -> bool:
    """Does the annotation of field ``name`` accept ``value`` (a key given first)?"""
    spec = parse_owner(cls)
    s = spec.fields.get(name) if isinstance(spec, StructSpec) else None
    if s is None:
        return False
    try:
        s.match(value, name, Env())
    except Exception:
        return False
    return True


def _unbuilt_default(f: Any) -> bool:
    """A field the constructor cannot fill in by itself."""
    if f.default_factory is not dataclasses.MISSING:
        return False
    return f.default is dataclasses.MISSING or isinstance(f.default, (Initializer, _Required))


def _constructor(cls: Any) -> None:
    """``Cls(key, D=16, blocks=4)`` builds from the annotations; ``Cls(w, b)`` takes values.

    The constructor is generated for each class, like the dataclass one it
    replaces, so construction from values costs what a dataclass's does: an
    identity test per field, and a look at the first value's dtype (is it a
    PRNG key?). Anything unusual (keywords that are not fields, a key, missing
    fields) goes to ``slow``, which also writes the error messages.
    """
    fields = dataclasses.fields(cls)
    names = [f.name for f in fields]
    accepts: dict[bool, bool] = {}  # does the first field take a (typed / raw) key as its value?

    def keyed(x: Any) -> bool:
        """Is ``x``, given first, a key that means "build" rather than the first field?"""
        if not is_key(x):
            if getattr(x, "dtype", None) is None and not isinstance(x, (jax.Array, np.ndarray)):
                _PLAIN.add(type(x))  # e.g. a struct: never a key, not looked at again
            return False
        typed = not isinstance(x.dtype, np.dtype)
        if typed not in accepts:
            accepts[typed] = bool(names) and _accepts(cls, names[0], x)
        return not accepts[typed]

    def slow(self: Any, values: tuple[Any, ...], kw: dict[str, Any]) -> None:
        first = values[0] if values else _NOT_GIVEN
        if first is not _NOT_GIVEN and keyed(first):
            if any(v is not _NOT_GIVEN for v in values[1:]):
                raise TypeError(
                    f"{cls.__name__}(key, ...): after the key, give sizes and fields by name, "
                    f"e.g. {cls.__name__}(key, D=64)."
                )
            _build_root(cls, first, kw, f"{cls.__name__}(key, ...)", self)
            return
        given = {n: v for n, v in zip(names, values) if v is not _NOT_GIVEN}
        unknown = []
        for k, v in kw.items():
            if k in given:
                raise TypeError(f"{cls.__name__}() got two values for field `{k}`.")
            if k in names:
                given[k] = v
            else:
                unknown.append(k)
        if unknown:
            raise TypeError(_unexpected(cls, unknown))
        missing = []
        for f in fields:
            if f.name in given:
                continue
            if f.default_factory is not dataclasses.MISSING:
                given[f.name] = f.default_factory()
            elif _unbuilt_default(f):
                missing.append(f.name)
            else:
                given[f.name] = f.default
        if missing:
            raise TypeError(_not_given(cls, missing))
        for n in names:
            object.__setattr__(self, n, given[n])

    simple = (
        bool(fields)
        and all(f.init and not f.kw_only for f in fields)
        and not hasattr(cls, "__post_init__")
        and all(n.isidentifier() and n != "self" and not n.startswith("_expex") for n in names)
    )
    if simple:
        __init__ = _generate(cls, fields, slow, keyed, _buildable(cls, fields))
    else:  # init=False / kw_only fields, __post_init__: around the dataclass constructor
        generated = cls.__init__
        check = any(isinstance(f.default, (Initializer, _Required)) for f in fields)

        @functools.wraps(generated)
        def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
            if args and keyed(args[0]):
                return slow(self, args, kwargs)
            generated(self, *args, **kwargs)
            if check:
                unbuilt(cls, self)

    __init__.__expex_generated__ = True  # type: ignore[attr-defined]
    cls.__init__ = __init__


def _buildable(cls: Any, fields: Any) -> bool:
    """Can a key given first build anything: an array with an initializer, or a nested module?

    If not (a record of arrays such as a game state), building would need every
    field by name, so the constructor does not look for a key in front of the
    field values: building then goes through keywords, which ``slow`` handles.
    """
    anns = _class_annotations(cls)
    for f in fields:
        if isinstance(f.default, Initializer):
            return True
        if not _unbuilt_default(f):
            continue  # a plain default
        try:
            spec = parse(anns.get(f.name))
        except Exception:
            return True  # not known yet (a forward reference): look for keys
        if not isinstance(spec, (ArraySpec, KeySpec, StaticSpec, NoneSpec)):
            return True  # a nested module, a list of them, ...
    return False


def _generate(cls: Any, fields: Any, slow: Any, keyed: Any, buildable: bool) -> Any:
    """The source of a constructor taking field values, with a way out to ``slow``."""
    names = [f.name for f in fields]
    ns: dict[str, Any] = {
        "_expex_N": _NOT_GIVEN, "_expex_slow": slow, "_expex_keyed": keyed,
        "_expex_set": object.__setattr__, "_expex_U32": _U32, "_expex_Array": _ArrayImpl,
        "_expex_plain": _PLAIN,
    }  # fmt: skip
    every = "(" + "".join(f"{n}, " for n in names) + ")"
    out = f"return _expex_slow(self, {every}, _expex_kw)"

    def fill(f: Any) -> str:
        if f.default_factory is not dataclasses.MISSING:
            ns[f"_expex_F_{f.name}"] = f.default_factory
            return f"{f.name} = _expex_F_{f.name}()"
        if _unbuilt_default(f):
            return out
        ns[f"_expex_D_{f.name}"] = f.default
        return f"{f.name} = _expex_D_{f.name}"

    f0 = names[0]
    rest = "".join(f", {n}=_expex_N" for n in names[1:])
    lines = [
        f"def __init__(self, {f0}=_expex_N, /{rest}, **_expex_kw):",
        "    if _expex_kw:",  # the first field by name, or anything unusual
        f"        if {f0} is not _expex_N or len(_expex_kw) != 1 or {f0!r} not in _expex_kw:",
        f"            {out}",
        f"        {f0} = _expex_kw[{f0!r}]",
        f"    elif {f0} is _expex_N:",
        f"        {fill(fields[0])}",
    ]
    if buildable:  # a PRNG key first: build. A jax array is looked at closer only if uint32.
        lines += [
            f"    elif (_expex_c := {f0}.__class__) is _expex_Array:",
            f"        if {f0}.aval.dtype is _expex_U32 and _expex_keyed({f0}):",
            f"            {out}",
            f"    elif _expex_c not in _expex_plain and _expex_keyed({f0}):",
            f"        {out}",
        ]
    lines += [f"    if {f.name} is _expex_N: {fill(f)}" for f in fields[1:]]
    lines += [f"    _expex_set(self, {n!r}, {n})" for n in names]
    exec("\n".join(lines), ns)
    init = ns["__init__"]
    init.__qualname__ = f"{cls.__qualname__}.__init__"
    init.__module__ = cls.__module__
    init.__doc__ = (
        f"{cls.__name__}(<field values>), or {cls.__name__}(key, <sizes and fields by name>) "
        f"to build the arrays from their annotations."
    )
    return init


def _not_given(cls: Any, missing: list[str]) -> str:
    how = (
        f"{cls.__name__}(key, <sizes>)"
        if getattr(cls.__init__, "__expex_generated__", False)
        else f"expex.build({cls.__name__}, key, <sizes>)"
    )
    return (
        f"{cls.__name__}(): {', '.join(missing)} not given. Pass them, or build every "
        f"array from its annotation with {how}."
    )


def _sizes_and_fields(cls: Any) -> tuple[set[str], set[str], Spec]:
    """The sizes a class (and the modules in it) uses, and the field names, nested ones too."""
    spec = parse_owner(cls)
    named: set[str] = set()
    spec.names(named)  # sizes in the annotations, Dim classes included
    sizes = {p.__name__ for p in getattr(cls, "__type_params__", ())}
    sizes |= {n for n in named if not n.startswith("*") and "." not in n}
    fields = _all_fields(spec) | {f.name for f in dataclasses.fields(cls)}
    return sizes - fields, fields, spec  # `heads` in f32[self.heads, D] is a field


def _unexpected(cls: Any, unknown: list[str]) -> str:
    sizes, _, _ = _sizes_and_fields(cls)
    names = ", ".join(f"{k}=..." for k in unknown)
    if all(k in sizes or k in _all_fields(parse_owner(cls)) for k in unknown):
        return (
            f"{cls.__name__}() got {names}: sizes and the fields of nested modules are taken "
            f"only when building from a key, {cls.__name__}(key, {names}). Given values, "
            f"pass the fields of {cls.__name__} ({', '.join(f.name for f in dataclasses.fields(cls))})."
        )
    return (
        f"{cls.__name__}() got an unexpected keyword argument {unknown[0]!r}. Its fields are "
        f"{', '.join(f.name for f in dataclasses.fields(cls))}."
    )


def unbuilt(cls: Any, obj: Any) -> None:
    """Refuse an object some of whose arrays were never made (constructed without them)."""
    left = [
        f.name for f in dataclasses.fields(cls)
        if isinstance(getattr(obj, f.name, None), (Initializer, _Required))
    ]  # fmt: skip
    if left:
        raise TypeError(_not_given(cls, left))


def build[C](cls: type[C], key: Any, /, **given: Any) -> C:
    """Build ``cls`` from its annotations: ``expex.build(Stack, key, D=64, blocks=12)``.

    The same as ``Stack(key, D=64, blocks=12)``; for a class with its own
    ``__init__``, the way to build it from its annotations. ``given``: sizes
    (``D=64``, any size the class or a module in it uses), lengths of lists of
    modules by field name (``blocks=12``), and values of fields at any depth
    (``heads=4``, ``w=...``), which are taken instead of built.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"expex.build: {getattr(cls, '__name__', cls)!r} is not an @expex.struct class.")
    return _build_root(cls, key, given, f"expex.build({cls.__name__}, key, ...)", None)


class _Ctx:
    """What error messages from building need: who was called, and how to call it."""

    __slots__ = ("who", "call")

    def __init__(self, who: str, call: str):
        self.who = who
        self.call = call  # e.g. "Stack(key, {})", filled with a keyword

    def example(self, kw: str) -> str:
        return self.call.format(kw)


def _star(path: str) -> str:
    """``blocks[3].w`` -> ``blocks[*].w``: the field of every element of a list of modules."""
    return re.sub(r"\[\d+\]", "[*]", path)


def _build_root(cls: Any, key: Any, given: dict[str, Any], who: str, into: Any) -> Any:
    if who.startswith("expex.build"):
        ctx = _Ctx(who, f"expex.build({cls.__name__}, key, {{}}, ...)")
    else:
        ctx = _Ctx(who, f"{cls.__name__}(key, {{}}, ...)")
    if not is_key(key):
        raise TypeError(f"{who}: the key must be a PRNG key such as jax.random.key(0), got {describe(key)}.")
    sizes, fields, spec = _sizes_and_fields(cls)
    unknown = sorted(set(given) - sizes - fields)
    if unknown:
        raise TypeError(
            f"{who} got {', '.join(f'{k}=...' for k in unknown)}: not a size "
            f"({', '.join(sorted(sizes)) or 'none'}) or a field ({', '.join(sorted(fields))}) of "
            f"{cls.__name__} or of the modules in it."
        )
    env = Env()
    for n in sizes & set(given):
        env.dims[n] = int(given[n])
        env.origin[n] = f"{n}={given[n]} passed to {who}"
    rest = {k: v for k, v in given.items() if k not in sizes}
    obj = _build(cls, spec, key, env, rest, "", ctx, into)
    try:  # the whole object against its annotations, sizes shared
        spec.match(obj, "", env)
        if env.deferred:
            resolve(env)
    except Mismatch as m:
        raise ShapeError(
            f"{who}: `{m.path or cls.__name__}` is {describe(m.value)}, but is declared "
            f"{m.spec}.\n\n    {m.reason}"
        ) from None
    return obj


def _build(
    cls: Any, spec: Any, key: Any, env: Env, given: dict[str, Any], path: str, ctx: _Ctx, into: Any = None
) -> Any:
    fields = dataclasses.fields(cls)
    fields_here = {f.name for f in fields}
    specs: dict[str, Spec] = getattr(spec, "fields", {})
    obj = object.__new__(cls) if into is None else into
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
                    raise ShapeError(
                        f"{ctx.who}: `{m.path}` is {describe(m.value)}, but is declared "
                        f"{m.spec}.\n\n    {m.reason}"
                    ) from None
        elif s is None or not s.leafy or isinstance(s, StaticSpec):
            values[f.name] = _plain(f, join(path, f.name), ctx)
        if f.name in values and isinstance(values[f.name], (int, np.integer)) and not isinstance(values[f.name], bool):
            env.dims.setdefault(f.name, int(values[f.name]))
    keys = jr.split(key, max(len(fields), 1))
    for k, f in zip(keys, fields):
        if f.name not in values:
            deeper = {n: v for n, v in given.items() if n not in fields_here}
            values[f.name] = _make(specs[f.name], f, k, env, given.get(f.name), join(path, f.name), deeper, ctx)
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
        elif isinstance(s, TupleSpec):
            todo += list(s.elems)
        elif isinstance(s, UnionSpec):
            todo += list(s.options)
        elif isinstance(s, StructSpec):
            out |= set(s.fields)
            todo += list(s.fields.values())
    return out


def _plain(f: Any, path: str, ctx: _Ctx) -> Any:
    if f.default is not dataclasses.MISSING and not isinstance(f.default, (Initializer, _Required)):
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    raise TypeError(
        f"{ctx.who} needs {f.name}: `{_star(path)}` has no default. Pass it by name, "
        f"e.g. {ctx.example(f'{f.name}=...')}."
    )


def _make(
    spec: Spec, f: Any, key: Any, env: Env, count: Any, path: str, deeper: dict[str, Any], ctx: _Ctx
) -> Any:
    """Build one field. ``deeper``: the settings given, for nested structs."""
    if isinstance(spec, UnionSpec):  # `X | None`: None unless there is a way to build X
        if f.default is None:
            return None
        spec = next(s for s in spec.options if not isinstance(s, NoneSpec))
    if isinstance(spec, (ArraySpec, KeySpec)):
        how = f.default if isinstance(f.default, Initializer) else None
        if how is None:
            raise TypeError(
                f"{ctx.who} cannot build `{_star(path)}`: {spec} has no initializer. Give the "
                f"field a default (a JAX initializer such as jax.nn.initializers.normal(), or a "
                f"number), or pass it by name, e.g. {ctx.example(f'{f.name}=...')}."
            )
        if isinstance(spec, KeySpec):
            return jr.split(key, _shape(spec.dims, env, path, spec, ctx)) if spec.dims else key
        dtype = spec.dtype.dtype if spec.dtype.dtype is not None else spec.dtype.example
        return how(key, _shape(spec.dims, env, path, spec, ctx), dtype)
    if isinstance(spec, StructSpec):
        return _build(spec.cls, spec, key, env, deeper, path, ctx)
    if isinstance(spec, SeqSpec) and isinstance(spec.elem, StructSpec):
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(
                f"{ctx.who} needs {f.name}: the length of the list `{_star(path)}` of "
                f"{spec.elem}. Pass it by name, e.g. {ctx.example(f'{f.name}=4')}."
            )
        items = [
            _build(spec.elem.cls, spec.elem, k, env, deeper, f"{path}[{i}]", ctx)
            for i, k in enumerate(jr.split(key, count))
        ]  # fmt: skip
        return items if spec.kind is list else tuple(items)
    if isinstance(spec, TupleSpec) and spec.elems and all(isinstance(s, StructSpec) for s in spec.elems):
        ks = jr.split(key, len(spec.elems))
        return tuple(
            _build(s.cls, s, k, env, deeper, f"{path}[{i}]", ctx)  # type: ignore[attr-defined]
            for i, (s, k) in enumerate(zip(spec.elems, ks))
        )
    raise TypeError(
        f"{ctx.who} cannot build `{_star(path)}` ({spec}) from its annotation. Pass it by name, "
        f"e.g. {ctx.example(f'{f.name}=...')}."
    )


def _shape(dims: tuple[Any, ...], env: Env, path: str, spec: Any, ctx: _Ctx) -> tuple[int, ...]:
    out = []
    for d in dims:
        if isinstance(d, DConst):
            out.append(d.value)
        elif isinstance(d, DName) and d.name in env.dims:
            out.append(int(env.dims[d.name]))
        elif isinstance(d, DExpr) and d.expr.names <= env.dims.keys():
            out.append(int(d.expr(env.dims)))
        else:
            if isinstance(d, DName):
                names = [d.name]
            else:
                names = sorted(set(getattr(getattr(d, "expr", None), "names", ())) - env.dims.keys())
            what = ", ".join(names) or str(d)
            raise TypeError(
                f"{ctx.who} needs {what}: the size of `{_star(path)}` ({spec}). Pass it by "
                f"name, e.g. {ctx.example(f'{names[0] if names else what}=64')}."
            )
    return tuple(out)


# ------------------------------------------------------------------ at

_ATTR, _ITEM = "attr", "item"
Steps = tuple[tuple[str, Any], ...]


class At(Generic[M]):
    """A path into a struct, for out-of-place updates: ``model.at.blocks[5].w.set(v)``.

    ``.set(v)``, ``.add(v)``, ``.multiply(v)`` and ``.apply(f)`` return a copy of
    the model with that part changed, ``.get()`` returns the part. Fields are
    reached with ``.name`` (or ``["name"]``, for a field called ``set``, ``get``,
    ...), elements of lists, tuples and dicts with ``[i]``, and arrays are indexed
    as with ``x.at[...]``: ``model.at.w[0, :3].set(0.0)``. A list of modules
    stored stacked (one array per field with a leading layer axis, as a loop over
    layers makes them) is indexed by row: ``[5]`` is row 5 of every array.

    The new value is checked against the annotations, with the sizes that the
    rest of the model fixes; a mismatch is a :class:`ShapeError` naming the path.
    ``add`` and ``multiply`` on a module (or a list of them) apply to each of its
    arrays.
    """

    __slots__ = ("_root", "_steps")

    def __init__(self, root: M, steps: Steps = ()) -> None:
        self._root = root
        self._steps = steps

    def __getattr__(self, name: str) -> "At[M]":
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return At(self._root, self._steps + ((_ATTR, name),))

    def __getitem__(self, index: Any) -> "At[M]":
        return At(self._root, self._steps + ((_ITEM, index),))

    def __repr__(self) -> str:
        return f"{type(self._root).__name__}.at{_fmt_steps(self._steps)}"

    def get(self) -> Any:
        """The part of the model at this path."""
        node: Any = self._root
        path = ""
        for kind, key in self._steps:
            node, path = _child(node, kind, key, path, repr(self) + ".get()")
        return node

    def set(self, value: Any) -> M:
        """A copy of the model with this part replaced by ``value`` (checked)."""
        return _update(self, "set", lambda old: value, lambda a, i: a.at[i].set(value))

    def add(self, value: Any) -> M:
        """A copy of the model with ``value`` added to this part."""
        return _update(
            self, "add", lambda old: _combine(old, value, operator.add), lambda a, i: a.at[i].add(value)
        )

    def multiply(self, value: Any) -> M:
        """A copy of the model with this part multiplied by ``value``."""
        return _update(
            self, "multiply", lambda old: _combine(old, value, operator.mul), lambda a, i: a.at[i].multiply(value)
        )

    def apply(self, f: Callable[[Any], Any]) -> M:
        """A copy of the model with this part replaced by ``f(part)`` (checked)."""
        return _update(self, "apply", f, lambda a, i: a.at[i].set(f(a[i])))


class _AtProperty:
    """``model.at``: see :class:`At`."""

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        if obj is None:
            return self
        return At(obj)


_AT = _AtProperty()


def _fmt_index(key: Any) -> str:
    if isinstance(key, tuple):
        return ", ".join(_fmt_index(k) for k in key)
    if isinstance(key, slice):
        parts = ["" if key.start is None else str(key.start), "" if key.stop is None else str(key.stop)]
        if key.step is not None:
            parts.append(str(key.step))
        return ":".join(parts)
    if key is Ellipsis:
        return "..."
    if isinstance(key, (int, np.integer)) or key is None:
        return str(key)
    if isinstance(key, str):
        return repr(key)
    return describe(key) if hasattr(key, "shape") else type(key).__name__


def _fmt_steps(steps: Steps) -> str:
    return "".join(f".{k}" if kind is _ATTR else f"[{_fmt_index(k)}]" for kind, k in steps)


def _is_array(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic))


def _record_fields(x: Any) -> tuple[str, ...] | None:
    """Field names of a dataclass or NamedTuple instance, else None."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return tuple(f.name for f in dataclasses.fields(x))
    if isinstance(x, tuple) and hasattr(x, "_fields"):
        return tuple(x._fields)
    return None


def _no_field(node: Any, name: str, path: str, label: str) -> AttributeError:
    names = _record_fields(node)
    where = f"`{path}`" if path else "the model"
    if names is None:
        return AttributeError(f"{label}: {where} is {describe(node)}, which has no fields (no `.{name}`).")
    return AttributeError(
        f"{label}: {where} ({type(node).__name__}) has no field `{name}`; its fields are {', '.join(names)}."
    )


def _list_index(node: Any, key: Any, path: str, label: str) -> int:
    try:
        i = operator.index(key)
    except TypeError:
        raise TypeError(
            f"{label}: `{path}` is a Python {type(node).__name__}: index it with a Python int "
            f"or a slice, not {describe(key) if hasattr(key, 'shape') else type(key).__name__}."
        ) from None
    if not -len(node) <= i < len(node):
        raise IndexError(f"{label}: `{path}` has {len(node)} elements, so [{i}] is out of range.")
    return i % len(node)


def _rows(node: Any) -> Any:
    """The arrays of a stacked list of modules: ``Stacked`` (from a loop) or a stacked struct."""
    from .staging.flow import Stacked

    return node.array if isinstance(node, Stacked) else node


def _child(node: Any, kind: str, key: Any, path: str, label: str) -> tuple[Any, str]:
    """One step of ``.get()``."""
    from .staging.flow import Stacked

    if _is_array(node):
        if kind is _ATTR:
            raise _no_field(node, key, path, label)
        return node[key], f"{path}[{_fmt_index(key)}]"
    fields = _record_fields(node)
    if kind is _ATTR or (isinstance(key, str) and fields is not None):
        if fields is None or key not in fields:
            raise _no_field(node, key, path, label)
        return getattr(node, key), join(path, key)
    p = f"{path}[{_fmt_index(key)}]"
    if isinstance(node, dict):
        return node[key], p
    if isinstance(node, (list, tuple)) and fields is None:
        if isinstance(key, slice):
            return node[key], p
        return node[_list_index(node, key, path, label)], p
    if isinstance(node, Stacked) or fields is not None:  # stacked rows
        return jax.tree.map(lambda a: a[key], _rows(node)), p
    raise TypeError(f"{label}: `{path}` is {describe(node)}; it cannot be indexed with [{_fmt_index(key)}].")


def _combine(old: Any, value: Any, op: Callable[[Any, Any], Any]) -> Any:
    """``old + value`` for an array or number; for a module, on each of its arrays."""
    if _is_array(old) or isinstance(old, (int, float, complex)):
        return op(old, value)
    if isinstance(value, (int, float, complex)) or (_is_array(value) and np.ndim(value) == 0):
        return jax.tree.map(lambda a: op(a, value), old)
    return jax.tree.map(op, old, value)


def _update(at: At[Any], name: str, whole: Callable[[Any], Any], indexed: Callable[[Any, Any], Any]) -> Any:
    from .function import config

    root, steps = at._root, at._steps
    label = f"{at!r}.{name}(...)"
    if not steps:
        raise TypeError(f"{label}: give the part to change first, e.g. model.at.w.{name}(...).")
    spec = parse_owner(type(root)) if config.checked else None
    env = Env()
    try:
        new = _put(root, spec, steps, "", env, whole, indexed, label)
        if env.deferred:
            resolve(env)
    except Mismatch as m:
        raise ShapeError(
            f"{label}: `{m.path or type(root).__name__}` is {describe(m.value)}, but is declared "
            f"{m.spec}.\n\n    {m.reason}"
        ) from None
    return new


def _put(
    node: Any, spec: Any, steps: Steps, path: str, env: Env,
    whole: Callable[[Any], Any], indexed: Callable[[Any, Any], Any], label: str,
) -> Any:  # fmt: skip
    """``node`` with the part at ``steps`` changed.

    On the way down, the parts left alone are matched against their specs
    first, so that ``env`` holds the sizes the rest of the model fixes; the
    changed part is then matched with those sizes.
    """
    from .staging.flow import Stacked

    kind, key = steps[0]
    rest = steps[1:]
    if isinstance(spec, UnionSpec):
        spec = spec.option(node)

    def down(child: Any, cspec: Any, p: str) -> Any:
        new = _put(child, cspec, rest, p, env, whole, indexed, label) if rest else whole(child)
        if cspec is not None and cspec.leafy:
            cspec.match(new, p, env)
        return new

    if _is_array(node):
        if kind is _ATTR:
            raise _no_field(node, key, path, label)
        arr = jnp.asarray(node) if not isinstance(node, jax.Array) else node
        if rest:
            return arr.at[key].set(_put(arr[key], None, rest, f"{path}[{_fmt_index(key)}]", env, whole, indexed, label))
        return indexed(arr, key)

    fields = _record_fields(node)
    if kind is _ATTR or (isinstance(key, str) and fields is not None):
        if fields is None or key not in fields:
            raise _no_field(node, key, path, label)
        specs = spec.fields if isinstance(spec, StructSpec) and isinstance(node, spec.cls) else {}
        for n, s in specs.items():  # the sizes the rest of the struct fixes
            if n != key and s.leafy:
                s.match(getattr(node, n), join(path, n), env)
        new = down(getattr(node, key), specs.get(key), join(path, key))
        out = rebuild(node, **{key: new})
        if specs and spec.invariants:  # e.g. `assert D % self.heads == 0` after heads changed
            spec.match(out, path, Env())
        return out

    p = f"{path}[{_fmt_index(key)}]"
    if isinstance(node, dict):
        if key not in node:
            raise KeyError(f"{label}: `{path}` has no key {key!r}; its keys are {', '.join(map(repr, node))}.")
        vspec = spec.value if isinstance(spec, DictSpec) else None
        if vspec is not None:
            for k, v in node.items():
                if k != key:
                    vspec.match(v, f"{path}[{k!r}]", env)
        out = dict(node)
        out[key] = down(node[key], vspec, p)
        return type(node)(out) if type(node) is not dict else out

    if isinstance(node, (list, tuple)) and fields is None:
        n = len(node)
        if isinstance(spec, SeqSpec):
            elems: list[Any] = [spec.elem] * n
        elif isinstance(spec, TupleSpec) and len(spec.elems) == n:
            elems = list(spec.elems)
        else:
            elems = [None] * n
        chosen = list(range(n))[key] if isinstance(key, slice) else [_list_index(node, key, path, label)]
        for j in range(n):  # the sizes the other elements fix
            if j not in chosen and elems[j] is not None and elems[j].leafy:
                elems[j].match(node[j], f"{path}[{j}]", env)
        out = list(node)
        if isinstance(key, slice):
            new = _put(node[key], None, rest, p, env, whole, indexed, label) if rest else whole(node[key])
            if not isinstance(new, (list, tuple)) or len(new) != len(chosen):
                raise ShapeError(
                    f"{label}: `{p}` is {len(chosen)} elements, so the new value must be a list "
                    f"of {len(chosen)}, not {describe(new) if not isinstance(new, (list, tuple)) else len(new)}."
                )
            for j, v in zip(chosen, new):
                if elems[j] is not None and elems[j].leafy:
                    elems[j].match(v, f"{path}[{j}]", env)
                out[j] = v
        else:
            i = chosen[0]
            out[i] = down(node[i], elems[i], f"{path}[{i}]")
        return type(node)(out)

    if isinstance(node, Stacked) or fields is not None:  # a stacked list of modules: by row
        arrays = _rows(node)
        old = jax.tree.map(lambda a: a[key], arrays)
        espec = spec.elem if isinstance(spec, SeqSpec) else None
        new = down(old, espec, p)
        old_leaves, old_tree = jax.tree_util.tree_flatten_with_path(old)
        new_leaves, new_tree = jax.tree_util.tree_flatten_with_path(new)
        if jax.tree.structure(old) != jax.tree.structure(new):
            raise ShapeError(f"{label}: `{p}` is a row of a stacked list, and the new value has another structure.")
        for (kp, a), (_, b) in zip(old_leaves, new_leaves):
            if np.shape(a) != np.shape(b) or jnp.result_type(a) != jnp.result_type(b):
                raise ShapeError(
                    f"{label}: `{p}{jax.tree_util.keystr(kp)}` is {describe(b)}, but the rows of "
                    f"this stacked list are {describe(a)}."
                )
        merged = jax.tree.map(lambda a, b: a.at[key].set(b), arrays, new)
        return Stacked(merged) if isinstance(node, Stacked) else merged

    raise TypeError(f"{label}: `{path}` is {describe(node)}; it cannot be indexed with [{_fmt_index(key)}].")

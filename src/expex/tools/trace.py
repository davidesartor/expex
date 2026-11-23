"""Tracing for the tools: an ``@fn`` traced on symbolic sizes, or a function on its arguments.

The prover, ``expex.cost``, ``debug.profile``, ``expex.diagram`` and
``debug.shapes`` all trace a function without data. :func:`resolve` turns
what they are given (an ``@fn``, an unbound or bound method, examples, size
keywords) into a :class:`Target`, and ``Target.trace(n)`` traces it once:
every annotated argument becomes an abstract value whose sizes are JAX
symbolic dimensions (``B``, ``T``, ...), lists of modules get ``n`` elements,
and the jaxpr comes back with the paths of those lists. The prover drives the
same pieces itself (:func:`choices`, :func:`statics`, :func:`constraints`,
:func:`attempt`) to try several cases and concrete sizes.

The debugging tools trace on real arguments instead: :func:`concrete` gives
the callable and arguments to trace (``Static`` arguments closed over), and
:func:`make_jaxpr` traces it. The rest of the module walks jaxprs.
"""

import dataclasses
import enum
import inspect
import os
import types
import typing
from typing import Any, Callable, Iterator

import jax
import jax.random as jr
import numpy as np
from jax import export

from .. import keys
from ..annotations.dims import DAny, DConst, DEllipsis, DExpr, DName, DVar, Env
from ..annotations.errors import ShapeError, StageError
from ..annotations.expr import to_jax_constraint
from ..annotations.spec import KeySpec, Spec, StaticSpec, Unsynthesizable
from ..function import BoundFn, Fn, unwrap

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the expex package
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
#: The choice for a parameter that is traced as an abstract value.
ABSTRACT = object()


class Unchecked(Exception):
    """The function cannot be traced without data: a value is missing (the message says which)."""


class BadSizes(Exception):
    """The concrete sizes tried make a size negative."""


class Template:
    """An example object to keep, except for its annotated arrays."""

    def __init__(self, obj: Any):
        self.obj = obj

    def __repr__(self) -> str:
        return f"<{type(self.obj).__name__} instance>"


class Sizes:
    """Supplies sizes for abstract inputs: symbolic (``scope``) or concrete (``sizes``).

    Lists of modules get ``list_length`` elements; their paths are collected in ``lists``.
    """

    def __init__(
        self,
        scope: Any | None,
        sizes: dict[str, int] | None,
        var_rank: dict[str, int],
        preset: dict[str, int] | None = None,
        list_length: int = 2,
    ):
        self.scope = scope
        self.sizes = sizes
        self.var_rank = var_rank
        self.dims: dict[str, Any] = dict(preset or {})
        self.vars: dict[str, tuple[Any, ...]] = {}
        self._anon = 0
        self.need_x64 = False
        self.legacy_keys = False
        self.list_length = list_length
        self.lists: list[str] = []
        #: The arguments of the last trace (examples as given, as :class:`Template` if so).
        self.arguments: dict[str, Any] = {}

    def _sym(self, name: str) -> Any:
        if self.scope is not None:
            return export.symbolic_shape(name, scope=self.scope)[0]
        assert self.sizes is not None
        if name not in self.sizes:
            self.sizes[name] = PRIMES[len(self.sizes) % len(PRIMES)]
        return self.sizes[name]

    def dim(self, name: str) -> Any:
        if name not in self.dims:
            self.dims[name] = self._sym(name)
        return self.dims[name]

    def fresh(self) -> Any:
        self._anon += 1
        return self._sym(f"_a{self._anon}")

    def var(self, name: str) -> tuple[Any, ...]:
        if name not in self.vars:
            self.vars[name] = tuple(self._sym(f"{name}_{i}") for i in range(self.var_rank.get(name, 1)))
        return self.vars[name]

    def shape(self, dims: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for d in dims:
            if isinstance(d, DName):
                out.append(self.dim(d.name))
            elif isinstance(d, DConst):
                out.append(d.value)
            elif isinstance(d, DAny):
                out.append(self.fresh())
            elif isinstance(d, DVar):
                out.extend(self.var(d.name))
            elif isinstance(d, DEllipsis):
                out.extend(self.fresh() for _ in range(self.var_rank.get("...", 1)))
            elif isinstance(d, DExpr):
                v = d.expr({n: self.dim(n) for n in d.expr.names})
                if isinstance(v, (int, np.integer)) and v < 0:
                    raise BadSizes(f"`{d}` is negative")
                out.append(v)
        return tuple(out)

    def dtype(self, dt: Any) -> Any:
        d = dt.example
        if d.itemsize * (1 if d.kind != "c" else 0.5) >= 8 and not jax.config.jax_enable_x64:
            self.need_x64 = True
        return d

    def key(self, shape: tuple[Any, ...]) -> Any:
        if self.legacy_keys:
            return jax.ShapeDtypeStruct(shape + (2,), np.uint32)
        return jax.ShapeDtypeStruct(shape, jr.key(0).dtype)


# ------------------------------------------------------------------ what to trace


def auto_examples(ann: Any) -> list[Any]:
    """Values tried for a static parameter that has no default."""
    if ann is bool:
        return [False, True]
    if ann is int:
        return [1, 2]
    if ann in (float, complex):
        return [ann(0.5)]  # a number sets no size, so one value is enough
    if isinstance(ann, type) and issubclass(ann, enum.Enum):
        return list(ann)
    origin = typing.get_origin(ann)
    if origin is typing.Literal:
        return list(typing.get_args(ann))
    if origin in (typing.Union, types.UnionType):
        out: list[Any] = []
        for a in typing.get_args(ann):
            out += [None] if a is type(None) else auto_examples(a)
        return out
    if isinstance(ann, type) and (dataclasses.is_dataclass(ann) or hasattr(ann, "_fields")):
        try:
            return [ann()]  # a configuration class whose fields all have defaults
        except TypeError:
            return []
    return []


def choices(f: Fn[Any, Any], examples: dict[str, Any]) -> dict[str, list[Any]]:
    """For each parameter: ``[ABSTRACT]`` or the list of concrete values to try."""
    for k in examples:
        if k not in f.params:
            raise TypeError(f"{f.name}: examples given for unknown parameter `{k}`")
    out: dict[str, list[Any]] = {}
    for name, p in f.params.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in examples:
            ex = examples[name]
            out[name] = list(ex) if isinstance(ex, (list, tuple)) else [ex]
            if not out[name]:
                raise Unchecked(f"`examples` for `{name}` is empty.")
        elif isinstance(p.spec, StaticSpec) and p.default is inspect.Parameter.empty and (
            auto := auto_examples(p.spec.inner)
        ):
            out[name] = auto
        elif isinstance(p.spec, StaticSpec) or not p.spec.leafy:
            if p.default is inspect.Parameter.empty:
                owner = getattr(f, "_owner_cls", None)
                if name == "cls" and name == next(iter(f.params)) and owner is not None:
                    out[name] = [owner]  # a classmethod: the class itself
                    continue
                if name in ("self", "cls") and name == next(iter(f.params)):
                    raise Unchecked(
                        "it is a method, so it needs an instance: "
                        "prove it with expex.prove(instance.method)."
                    )
                what = "Static parameter" if isinstance(p.spec, StaticSpec) else "parameter"
                why = "" if isinstance(p.spec, StaticSpec) else " and no shape annotation"
                raise Unchecked(
                    f"{what} `{name}` has no default{why}, so the prover cannot invent "
                    f'a value for it. Add @fn(examples={{"{name}": [...]}}).'
                )
            # otherwise: leave it out and let the default apply
        else:
            out[name] = [ABSTRACT]
    return out


def statics(f: Fn[Any, Any], fixed: dict[str, Any]) -> dict[str, int]:
    """Values of the Static int parameters that dimension expressions refer to."""
    out = {}
    for n in f.static_refs:
        v = fixed[n] if n in fixed else f.params[n].default
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out[n] = int(v)
    if f.field_refs:  # int fields of `self`: from the example instance, or the class default
        first = next(iter(f.params))
        obj = fixed.get(first)
        obj = getattr(obj, "obj", obj)
        owner: Any = getattr(f, "_owner_cls", None)
        defaults = (
            {fl.name: fl.default for fl in dataclasses.fields(owner)}
            if dataclasses.is_dataclass(owner)
            else {}
        )
        for n in f.field_refs:
            v = getattr(obj, n, defaults.get(n))
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                out[n] = int(v)
    return out


def constraints(f: Fn[Any, Any], statics: dict[str, int]) -> tuple[list[str], list[str]]:
    """``where`` constraints in jax.export syntax, and those that cannot be expressed."""
    good, dropped = [], []
    inputs = {n for n in f.input_names if not n.startswith("*")} | statics.keys()
    for w in [*f.where, *f.assumed]:
        if not w.names <= inputs:
            continue
        if w.names <= statics.keys():
            continue  # no free sizes: checked directly when the function is called
        j = to_jax_constraint(w, statics)
        try:
            if j is None:
                raise ValueError
            export.SymbolicScope([j])
            good.append(j)
        except Exception:
            dropped.append(w.src)
    return good, dropped


# ------------------------------------------------------------------ tracing


def trace(f: Fn[Any, Any], fixed: dict[str, Any], abstract: list[str], ctx: Sizes) -> Any:
    """Trace ``f`` once on abstract inputs and return the jaxpr.

    Raises whatever the function raises, including ``ShapeError`` from its own
    (and nested) annotation checks.
    """
    arguments: dict[str, Any] = dict(fixed)
    abstract = list(abstract)
    for name in abstract:
        arguments[name] = f.params[name].spec.abstract(ctx, name)
    ctx.arguments = dict(arguments)
    for name, v in fixed.items():
        if isinstance(v, Template):
            # a real object whose annotated arrays are replaced by abstract ones
            arguments[name] = f.params[name].spec.walk(v.obj, name, lambda s, x, p: s.abstract(ctx, p))
            abstract.append(name)

    leaves: list[Any] = []

    def collect(spec: Spec, v: Any, path: str) -> Any:
        leaves.append(v)
        return v

    abstract_args = {n: arguments[n] for n in abstract}
    f._walk(abstract_args, collect)
    leaves.append(ctx.key(()))  # a key, in case anything inside draws random numbers

    def g(*lv: Any) -> Any:
        it = iter(lv[:-1])
        new = dict(arguments)
        new.update(f._walk(abstract_args, lambda s, x, p: next(it)))
        ordered = {n: new[n] for n in f.params if n in new}
        ba = inspect.BoundArguments(f._sig, ordered)  # type: ignore[arg-type]
        env = Env()
        for n, v in ctx.dims.items():
            env.dims[n] = v
            env.origin[n] = "the declared inputs"
        with keys.rng(lv[-1]):
            f._run(ba.args, ba.kwargs, env, force=True)
        return ()

    if ctx.need_x64:
        with jax.enable_x64(True):
            return jax.make_jaxpr(g)(*leaves)
    return jax.make_jaxpr(g)(*leaves)


def attempt(
    f: Fn[Any, Any], fixed: dict[str, Any], abstract: list[str], make_ctx: Callable[..., Sizes]
) -> tuple[Any, Sizes, BaseException | None]:
    """Trace with new-style keys, retrying with legacy keys if that fails: (jaxpr, sizes, error)."""
    first: tuple[Sizes, BaseException] | None = None
    for legacy in (False, True):
        ctx = make_ctx()
        ctx.legacy_keys = legacy
        try:
            return trace(f, fixed, abstract, ctx), ctx, None
        except (Unsynthesizable, BadSizes):
            raise
        except Exception as e:
            if first is None:
                first = (ctx, e)
            if isinstance(e, (ShapeError, StageError)) or not _mentions_key(f):
                break
    assert first is not None
    return None, first[0], first[1]


def _mentions_key(f: Fn[Any, Any]) -> bool:
    found = False

    def visit(spec: Spec, v: Any, path: str) -> Any:
        nonlocal found
        found = found or isinstance(spec, KeySpec)
        return v

    ctx = Sizes(None, {}, {})
    for p in f.params.values():
        if p.spec.leafy and not isinstance(p.spec, StaticSpec):
            try:
                p.spec.walk(p.spec.abstract(ctx, p.name), p.name, visit)
            except Exception:
                pass
    return found


@dataclasses.dataclass
class Target:
    """A function resolved for tracing on symbolic sizes. See :func:`resolve`."""

    f: Any  # the Fn (typed Any: Fn is a descriptor, so a dataclass field of type Fn reads as bound)
    fixed: dict[str, Any]  # parameters with a value: static ones, a bound method's instance
    abstract: list[str]  # parameters traced as abstract values
    scope: list[str]  # size constraints, in jax.export syntax
    statics: dict[str, int]  # sizes set by static parameters
    static_kw: dict[str, Any]  # keywords that chose static parameters
    sizes: dict[str, Any]  # the keywords, plus a bound method's own sizes
    notes: list[str]  # the static parameters that took a default value

    def trace(self, n: int = 2, preset: dict[str, int] | None = None) -> tuple[Any, dict[str, Any], list[str]]:
        """Trace once with lists of length ``n``: (jaxpr, arguments, paths of the lists).

        ``preset`` fixes some sizes to numbers; the others are symbolic.
        """
        var_rank = {v[1:]: 1 for v in self.f.input_names if v.startswith("*")}
        new = lambda: Sizes(export.SymbolicScope(self.scope), None, var_rank, {**(preset or {}), **self.statics}, n)  # noqa: E731
        jaxpr, ctx, err = attempt(self.f, self.fixed, self.abstract, new)
        if err is not None:
            raise err
        return jaxpr, ctx.arguments, list(dict.fromkeys(ctx.lists))


def resolve(target: Any, sizes: dict[str, Any] | None = None, what: str = "trace") -> Target:
    """An ``@fn``, unbound or bound method (and keywords choosing static parameters) as a :class:`Target`.

    A bound method's instance is kept, its arrays traced abstractly, and its
    sizes count as given. Static parameters not chosen take their first
    example value (noted). ``what`` completes the error message, ``cannot
    {what} f: ...``, when a value is missing.
    """
    sizes = dict(sizes or {})
    target = unwrap(target)
    f: Fn[Any, Any] = target.fn if isinstance(target, BoundFn) else target
    if not isinstance(f, Fn):
        raise TypeError(f"cannot {what} {target!r}: expected an @fn function or a struct method")
    fixed: dict[str, Any] = {}
    if f._lazy:
        f._resolve_owner()
    first = next(iter(f.params), None)
    if isinstance(target, BoundFn) and first is not None:  # the instance's own sizes count as given
        env, spec = Env(), f.params[first].spec
        fixed[first] = Template(target.obj) if spec.leafy else target.obj
        spec.match(target.obj, first, env)
        sizes = {**{k: int(v) for k, v in env.dims.items() if isinstance(v, int)}, **sizes}
    static_kw = {k: v for k, v in sizes.items() if k in f.params}
    try:
        options = choices(f, {**f.examples, **static_kw, **fixed})
    except Unchecked as u:
        raise TypeError(f"cannot {what} {f.name}: {u}") from None
    abstract = [a for a, c in options.items() if c == [ABSTRACT]]
    fixed = {a: c[0] for a, c in options.items() if a not in abstract}
    notes = [f"{a}={v!r} (choose with {a}=...)" for a, v in fixed.items()
             if a not in static_kw and a not in f.examples and a != first]  # fmt: skip
    found = statics(f, fixed)
    scope, _ = constraints(f, found)
    return Target(f, fixed, abstract, scope, found, static_kw, sizes, notes)


# ------------------------------------------------------------------ on real arguments


def concrete(
    f: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
    """The un-jitted callable behind ``f`` and the arguments to trace it with.

    ``Static`` arguments of an ``@fn`` are closed over, so that tracing the
    result does not turn them into traced values.
    """
    f = unwrap(f)
    if isinstance(f, BoundFn):
        fn, args = f.fn, (f.obj, *args)
    elif isinstance(f, Fn):
        fn = f
    else:
        return f, args, kwargs
    try:
        bound = fn._sig.bind(*args, **kwargs)
    except TypeError:
        return (lambda *a, **k: fn._run(a, k)), args, kwargs
    static = {n: bound.arguments[n] for n in fn.static_names if n in bound.arguments}
    if isinstance(f, BoundFn):
        static[next(iter(fn.params))] = f.obj
    dyn = {n: v for n, v in bound.arguments.items() if n not in static}

    def run(dyn_args: dict[str, Any]) -> Any:
        ba = inspect.BoundArguments(fn._sig, {n: (static[n] if n in static else dyn_args[n]) for n in bound.arguments})  # type: ignore[arg-type]
        return fn._run(ba.args, ba.kwargs)

    return run, (dyn,), {}


def keyed(
    f: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
    """As :func:`concrete`, with the callable taking the random key first: ``run(key, *args, **kwargs)``."""
    g, args, kwargs = concrete(f, args, kwargs)

    def run(key: Any, *a: Any, **k: Any) -> Any:
        with keys.rng(key):
            return g(*a, **k)

    return run, args, kwargs


def make_jaxpr(f: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, list[Any]]:
    """Trace ``f`` on its arguments: the closed jaxpr and the flat list of traced inputs."""
    run, args, kwargs = keyed(f, args, kwargs)
    key = keys.current().next() if keys.current() else jr.key(0)  # type: ignore[union-attr]
    return jax.make_jaxpr(lambda *a, **k: run(key, *a, **k))(*args, **kwargs), jax.tree.leaves((args, kwargs))


# ------------------------------------------------------------------ jaxprs


def sub_jaxprs(params: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield ``(param_name, jaxpr)`` for every jaxpr nested in an equation's params."""
    for k, v in params.items():
        for i, j in enumerate(_as_jaxprs(v)):
            yield (k if i == 0 else f"{k}[{i}]"), j


def _as_jaxprs(v: Any) -> list[Any]:
    if hasattr(v, "eqns"):
        return [v]
    if hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
        return [v.jaxpr]
    if isinstance(v, (tuple, list)):
        out: list[Any] = []
        for x in v:
            out.extend(_as_jaxprs(x))
        return out
    return []


def iter_eqns(jaxpr: Any) -> Iterator[Any]:
    """All equations of a jaxpr, including those in nested jaxprs."""
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    for eqn in jaxpr.eqns:
        yield eqn
        for _, sub in sub_jaxprs(eqn.params):
            yield from iter_eqns(sub)


def _user_frame(eqn: Any) -> Any:
    """The innermost frame of user code (not expex's) that created an equation, or None."""
    from jax._src import source_info_util

    tb = eqn.source_info.traceback
    for frame in source_info_util.user_frames(tb) if tb is not None else ():
        if not os.path.abspath(frame.file_name).startswith(_HERE):
            return frame
    return None


def source_of(eqn: Any) -> str:
    """``file:line (function)`` of the user code that created an equation."""
    try:
        from jax._src import source_info_util

        if (frame := _user_frame(eqn)) is not None:
            return f"{frame.file_name}:{frame.start_line} ({frame.function_name})"
        return str(source_info_util.summarize(eqn.source_info))
    except Exception:
        return "<unknown>"


def source_key(eqn: Any) -> tuple[str, int, str]:
    """``(file, line, function)`` for grouping equations by source line."""
    try:
        if (frame := _user_frame(eqn)) is not None:
            return frame.file_name, frame.start_line, frame.function_name
    except Exception:
        pass
    return "<unknown>", 0, ""

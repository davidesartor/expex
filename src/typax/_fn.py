"""``@fn``: the decorator that reads a signature and puts it to work."""

import contextvars
import copy
import dataclasses
import functools
import inspect
import types
import typing
import os
import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

import jax
import numpy as np
from jax.core import Tracer

from . import _flow, _lift, _messages, _ops, _rng, _sym
from ._convert import NotConverted
from ._convert import convert as _convert
from ._dims import DExpr, DName, Env, describe, is_symbolic, resolve
from ._errors import Mismatch, ShapeError, StageError
from ._expr import Expr
from ._spec import (
    OpaqueSpec,
    Spec,
    StaticSpec,
    _assumptions_of,
    parse,
    parse_owner,
    static_type,
)

if TYPE_CHECKING:
    from ._check import Report

_BatchTracer: Any = ()
try:  # private, but there is no public way to ask "is this value being vmapped?"
    from jax._src.interpreters import batching as _batching

    _BatchTracer = _batching.BatchTracer
except Exception:  # pragma: no cover
    pass


#: Set by ``tx.diagram`` while it traces: ``hook(fn, args, kwargs, run)`` wraps every
#: checked call (``run(args, kwargs)`` runs it), to put it in a ``jax.named_scope``.
CALL_HOOK: contextvars.ContextVar[Callable[..., Any] | None] = contextvars.ContextVar(
    "typax_call_hook", default=None
)

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
S = TypeVar("S")


class Config:
    """Global switches. Also settable through ``TYPAX_*`` environment variables."""

    def __init__(self) -> None:
        #: Check arguments and return values on every (traced or eager) call.
        self.checked: bool = os.environ.get("TYPAX_CHECKED", "1") not in ("0", "false", "")
        #: Attach "in f(x: f32[3, 4])" notes to exceptions passing through ``@fn``.
        self.notes: bool = os.environ.get("TYPAX_NOTES", "1") not in ("0", "false", "")
        #: Turn ``for`` loops into ``lax.scan`` / ``vmap`` when tracing. With False,
        #: loops over arrays and ranges are unrolled, as in plain JAX.
        self.lower_loops: bool = os.environ.get("TYPAX_LOWER_LOOPS", "1") not in ("0", "false", "")


    def __call__(self, **changes: Any) -> "_Scoped":
        """``with tx.config(checked=False): ...`` changes settings inside the block."""
        for k in changes:
            if k not in vars(self):
                raise AttributeError(f"typax.config has no setting {k!r}")
        return _Scoped(self, changes)

    def __repr__(self) -> str:
        return "typax.config(" + ", ".join(f"{k}={v!r}" for k, v in vars(self).items()) + ")"


class _Scoped:
    def __init__(self, cfg: Config, changes: dict[str, Any]):
        self.cfg, self.changes = cfg, changes
        self.saved: dict[str, Any] = {}

    def __enter__(self) -> Config:
        self.saved = {k: getattr(self.cfg, k) for k in self.changes}
        vars(self.cfg).update(self.changes)
        return self.cfg

    def __exit__(self, *exc: Any) -> None:
        vars(self.cfg).update(self.saved)


config = Config()

def compile_cache(path: str = "~/.cache/typax", min_compile_seconds: float = 1.0) -> str:
    """Keep compiled functions on disk, so later runs of a program skip compilation.

    Turns on JAX's persistent compilation cache in ``path``. Only functions that
    took at least ``min_compile_seconds`` to compile are stored. Returns the
    directory used.
    """
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_seconds))
    return path

#: Every live ``Fn``; used by the prover to find functions to check.
REGISTRY: "weakref.WeakSet[Fn[Any, Any]]" = weakref.WeakSet()


class Param:
    __slots__ = ("name", "spec", "kind", "default")

    def __init__(self, name: str, spec: Spec, kind: Any, default: Any):
        self.name = name
        self.spec = spec
        self.kind = kind
        self.default = default


def _raw_annotations(raw: Callable[..., Any]) -> dict[str, Any]:
    return _sym.texts(raw)


def _enclosing_class(raw: Callable[..., Any]) -> type | None:
    """The class a function was defined in, found through its qualified name."""
    parts = getattr(raw, "__qualname__", "").split(".")[:-1]
    if not parts or "<locals>" in parts:
        return None
    obj: Any = getattr(raw, "__globals__", {}).get(parts[0])
    for name in parts[1:]:
        obj = getattr(obj, name, None)
    return obj if isinstance(obj, type) else None


def _annotations(raw: Callable[..., Any], localns: dict[str, Any]) -> dict[str, Any]:
    try:
        anns = inspect.get_annotations(raw, eval_str=True)
        if not any(isinstance(v, str) for v in anns.values()):  # e.g. quoted twice
            return anns
    except Exception:
        pass
    # Some annotation could not be evaluated (e.g. a TYPE_CHECKING-only import).
    # Evaluate them one by one and ignore the ones that fail.
    # Or it does arithmetic on sizes, f32[T + 1], which is evaluated symbolically.
    out: dict[str, Any] = {}
    glb = getattr(raw, "__globals__", {})
    loc = {**localns, **{p.__name__: p for p in getattr(raw, "__type_params__", ())}}
    params = list(_sym.signature(raw).parameters)
    for k, v in _sym.texts(raw).items():
        if isinstance(v, str):
            try:
                v = _sym.evaluate(v, glb, loc, params)
            except Exception:
                v = inspect.Parameter.empty
        out[k] = v
    return out


class Fn(Generic[P, R]):
    """A function whose shape annotations are checked, provable and usable.

    Created by :func:`fn`. Calling it calls the original function, after
    matching the arguments (and afterwards the result) against the annotations.
    """

    def __init__(
        self,
        raw: Callable[P, R],
        *,
        prove: bool = True,
        convert: bool = True,
        examples: dict[str, Any] | None = None,
        where: str | list[str] | tuple[str, ...] = (),
        defs: dict[str, str] | None = None,
    ):
        functools.update_wrapper(self, raw)
        self.raw = raw
        #: Set by ``@tx.random``: the function gets an implicit PRNG key.
        self.random = False
        #: Set by ``@tx.lift``: a plain call maps over extra leading axes by itself.
        self.auto_lift = False
        #: The function that actually runs: ``raw`` with its control flow converted.
        self._impl: Callable[..., Any] = raw
        #: Why the control flow was not converted ("" if it was).
        self.convert_note = "" if convert else "convert=False"
        if convert:
            try:
                self._impl = _convert(raw)
            except NotConverted as e:
                self.convert_note = str(e)
            except Exception as e:  # never let conversion break a working function
                self.convert_note = f"conversion failed: {type(e).__name__}: {e}"
        self.converted = self._impl is not raw
        #: The body uses sizes by name (or annotated locals): it needs this call's sizes.
        self._needs_sizes = bool(getattr(self._impl, "__typax_sizes__", 0))
        self.name: str = getattr(raw, "__qualname__", getattr(raw, "__name__", repr(raw)))
        self.prove = prove
        self.examples = dict(examples or {})
        self._sig = _sym.signature(raw)
        # what inspect.signature (and so jax.jit) sees: evaluating the raw
        # annotations would fail on size arithmetic such as f32[n] or f32[T + 1]
        self.__signature__ = self._sig

        defs = {**getattr(self._impl, "__typax_defs__", {}), **(defs or {})}  # H = heads in the body
        self.defs = {k: Expr(v) for k, v in defs.items()}
        if isinstance(where, str):
            where = [where]
        self.where = [Expr(w, constraint=True) for w in where]
        #: Leading `assert`s on sizes in the body: constraints like `where`.
        self._groups: list[list[Expr]] = []
        for group in getattr(self._impl, "__typax_where__", ()):
            try:
                self._groups.append([Expr(w, constraint=True) for w in group])
            except Exception:
                self._groups.append([])
        #: Which leading asserts are constraints (the body skips those).
        self._lifted = {i for i, g in enumerate(self._groups) if g}
        self._asserted = [e for g in self._groups for e in g]
        self.where += self._asserted

        derived = self.defs

        def derive(d: Any) -> Any:
            if isinstance(d, DName) and d.name in derived:
                return DExpr(derived[d.name], d.name)
            return d

        #: Where the function was defined (for annotations written as text).
        self._localns = _sym.caller_locals()
        # sizes defined in the body (H = heads) are names in the annotations
        defined = getattr(self._impl, "__typax_defs__", {})
        self._localns.update({n: typing.TypeVar(n) for n in defined})  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
        anns = _annotations(raw, self._localns)
        # Annotations that could not be evaluated yet (a forward reference to the
        # class being defined, say) are retried at the first call.
        self._pending = {
            k for k, v in _raw_annotations(raw).items()
            if isinstance(v, str) and anns.get(k, inspect.Parameter.empty) is inspect.Parameter.empty
        }  # fmt: skip
        self._lazy = bool(self._pending)
        names = list(self._sig.parameters)
        if self._pending and not (names and names[0] in ("self", "cls")):
            # not a method, so no class sizes to wait for: report a misspelt size now
            glb = getattr(raw, "__globals__", {})
            scope = {**self._localns, **{p.__name__: p for p in getattr(raw, "__type_params__", ())}}
            raw_texts = _raw_annotations(raw)
            for k in self._pending:
                self._unknown_in(k, raw_texts.get(k), glb, scope, names)
        self._derive = derive
        self.params: dict[str, Param] = {}
        for name, p in self._sig.parameters.items():
            ann = anns.get(name, inspect.Parameter.empty)
            try:
                spec = parse(ann)
            except TypeError as e:
                raise TypeError(f"{self.name}: bad annotation for `{name}`: {e}") from None
            if not spec.leafy and ann is not inspect.Parameter.empty and static_type(ann):
                # `n: int`, `mode: str`, `cfg: Config`: configuration, not data
                spec = StaticSpec(ann)
            if self.defs:
                spec = spec.replace_dims(derive)
            self.params[name] = Param(name, spec, p.kind, p.default)
        try:
            ret = parse(anns.get("return", inspect.Parameter.empty))
        except TypeError as e:
            raise TypeError(f"{self.name}: bad return annotation: {e}") from None
        self.ret: Spec = ret.replace_dims(derive) if self.defs else ret

        # Names in the raw signature (before derivation), for validation.
        declared: set[str] = set()
        for name in self._sig.parameters:
            parse(anns.get(name, inspect.Parameter.empty)).names(declared)
        parse(anns.get("return", inspect.Parameter.empty)).names(declared)
        type_params = {t.__name__ for t in getattr(raw, "__type_params__", ())}
        for k in self.defs:
            # a size may also be declared as a type parameter and used only in the body
            if k not in declared and k not in type_params and not self._pending:
                raise TypeError(
                    f"{self.name}: `{k}=...` was given to @fn, but `{k}` does not appear "
                    f"in the signature."
                )

        self._owner: type | None = None  # set by __set_name__ when defined in a class
        self.static_names = [n for n, p in self.params.items() if isinstance(p.spec, StaticSpec)]
        self._declared = declared
        self._index()

        REGISTRY.add(self)

    def _index(self) -> None:
        """Derive lookup tables from ``self.params`` (re-run if a spec changes)."""
        declared = self._declared
        self.input_names: set[str] = set()
        for prm in self.params.values():
            prm.spec.names(self.input_names)
        # Int parameters may be used by name in sizes and constraints:
        # -> f32[heads, D // heads], assert D % heads == 0
        used: set[str] = set()
        for expr in [*self.defs.values(), *self.where]:
            used |= expr.names
        # ... and so may sizes written in the annotations, f32[D // heads]
        dims: set[str] = set()

        def visit(d: Any) -> Any:
            if isinstance(d, DExpr) and d.label is None:
                used.update(d.expr.names)
            elif isinstance(d, DName):
                dims.add(d.name)
            return d

        for spec in [*(p.spec for p in self.params.values()), self.ret]:
            spec.replace_dims(visit)
        self.static_refs = [n for n in self.static_names if n in used]
        # ... and so may a method's int fields of `self`: @fn(H="heads") for self.heads
        fields = _field_names(getattr(self, "_owner_cls", None))
        self.field_refs = sorted((used - dims - set(self.static_names)) & fields)
        known = declared | set(self.static_names) | set(self.field_refs)
        first = next(iter(self.params), None)
        pending_owner = first in ("self", "cls") and not getattr(self, "_resolved", False)
        if self._asserted and not pending_owner:
            # an assert that is not about sizes after all stays an ordinary assert
            # (e.g. `assert flag == 1` on a non-int parameter): the body checks it
            bad = {i for i, g in enumerate(self._groups) if any(not e.names <= known for e in g)}
            self._lifted -= bad
            drop = [e for i in bad for e in self._groups[i]]
            self.where = [w for w in self.where if w not in drop]
            self._asserted = [e for e in self._asserted if e not in drop]
            used = set().union(*(e.names for e in [*self.defs.values(), *self.where]))
            if used:
                self.static_refs = [n for n in self.static_names if n in used or n in self.static_refs]
        unknown = used - declared - set(self.static_names) - set(self.field_refs)
        maybe_method = self._owner is None and first in ("self", "cls")
        if unknown and not (maybe_method and not getattr(self, "_resolved", False)):
            raise TypeError(
                f"{self.name}: {', '.join(sorted(unknown))} used in a size or a constraint is "
                f"neither a size of the signature, nor an int parameter, nor a field of self."
            )
        self.input_names -= set(self.static_refs) | set(self.field_refs)
        self._avail = self.input_names | set(self.static_refs) | set(self.field_refs)
        #: Invariants of the structs among the inputs: the prover may assume them.
        self.assumed: list[Any] = []
        for prm in self.params.values():
            for e in _assumptions_of(prm.spec):
                if e.names <= self.input_names and e not in self.assumed:
                    self.assumed.append(e)
        self._check_inputs = [
            (n, p.spec) for n, p in self.params.items() if p.spec.leafy
        ]
        self._check_ret = self.ret.leafy
        self._var_kinds = {
            n: p.kind
            for n, p in self.params.items()
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

    def __set_name__(self, owner: type, name: str) -> None:
        self._owner = owner
        self._lazy = True

    def _resolve_owner(self) -> None:
        """For a method: check ``self``'s annotated fields, binding the class's dims.

        Done lazily because the class is not finished (not yet a dataclass)
        when its methods are decorated.
        """
        owner, self._owner = self._owner, None
        self._lazy = False
        if owner is None:
            owner = _enclosing_class(self.raw)
        self._owner_cls = owner
        if self._pending:
            scope = {**self._localns, **{p.__name__: p for p in getattr(self.raw, "__type_params__", ())}}
            if owner is not None:
                scope.update({p.__name__: p for p in getattr(owner, "__type_params__", ())})
                scope[owner.__name__] = owner
            glb = getattr(self.raw, "__globals__", {})
            for k, text in _raw_annotations(self.raw).items():
                if k not in self._pending:
                    continue
                try:
                    spec = parse(_sym.evaluate(text, glb, scope, self.params))
                except Exception:
                    self._unknown_in(k, text, glb, scope, self.params)
                    continue
                spec.names(self._declared)
                if self.defs:
                    spec = spec.replace_dims(self._derive)
                if k == "return":
                    self.ret = spec
                elif k in self.params:
                    self.params[k].spec = spec
            self._pending = set()
        first = next(iter(self.params.values()), None)
        if (
            owner is not None
            and first is not None
            and first.name == "self"
            and isinstance(first.spec, OpaqueSpec)
            and first.spec.ann is inspect.Parameter.empty
        ):
            spec = parse_owner(owner)
            if spec.leafy:
                first.spec = spec
                spec.names(self._declared)
        self._resolved = True
        self._index()

    # ------------------------------------------------------------------ call

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        if self.auto_lift:
            return self.lift(*args, **kwargs)
        return self._direct(args, kwargs)

    def _direct(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        __tracebackhide__ = True
        key = None
        if self.random and "key" in kwargs and "key" not in self.params:
            kwargs = dict(kwargs)
            key = kwargs.pop("key")
        return self._run(args, kwargs, key=key)

    def _run(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        env: Env | None = None,
        force: bool = False,
        key: Any = None,
    ) -> Any:
        __tracebackhide__ = True
        if self.random:
            if key is None and "key" in self.params:
                key = self._own_key(args, kwargs)
            if not isinstance(key, _BatchTracer) and any(
                isinstance(x, _BatchTracer) for x in jax.tree.leaves((args, kwargs))
            ):
                raise StageError(
                    f"{self.name}() is being mapped with jax.vmap, but its random key is "
                    f"shared by the whole batch: every element would draw the same numbers.\n"
                    f"    - use {self.name}.lift(...), which gives each element its own key\n"
                    f"    - or split a key yourself and pass the keys: {self.name}(..., key=keys)"
                )
            if key is None:
                key = _rng.fresh(f"{self.name}()")
            with _rng.child(key):
                return self._run_checked(args, kwargs, env, force)
        return self._run_checked(args, kwargs, env, force)

    def _unknown_in(
        self, where: str, text: Any, glb: dict[str, Any], scope: dict[str, Any], params: Any
    ) -> None:
        """Raise if an array annotation names something undefined (a misspelt size, say).

        Other unresolvable annotations (a ``TYPE_CHECKING``-only class) are ignored.
        """
        if not isinstance(text, str):
            return
        try:
            value, missing = _sym.unknown_names(text, glb, scope, params)
            leafy = parse(value).leafy
        except Exception:
            return
        if missing and leafy:
            what = "the result" if where == "return" else f"`{where}`"
            raise TypeError(
                f"{self.name}: the annotation of {what} ({text}) uses "
                f"{', '.join(sorted(missing))}, which is not defined. Sizes are type "
                f"parameters (def f[T, D](...)), int parameters, or fields of self."
            )

    def _own_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """For a random function with its own ``key`` parameter: derive the implicit keys from it."""
        try:
            k = self._sig.bind(*args, **kwargs).arguments.get("key")
        except TypeError:
            return None
        if k is None:
            return None
        if getattr(getattr(k, "dtype", None), "name", "").startswith("key<"):
            return jax.random.fold_in(k, 0x7478)
        return jax.random.fold_in(jax.random.wrap_key_data(k), 0x7478)

    def _run_checked(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], env: Env | None, force: bool
    ) -> Any:
        __tracebackhide__ = True
        if (hook := CALL_HOOK.get()) is not None:
            return hook(self, args, kwargs, lambda a, k: self._checked(a, k, env, force))
        return self._checked(args, kwargs, env, force)

    def _checked(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], env: Env | None, force: bool
    ) -> Any:
        __tracebackhide__ = True
        if not force and not config.checked and not self._needs_sizes:
            out = self._impl(*args, **kwargs)
            return _flow.unstack(out) if self.converted else out
        if self._lazy:
            self._resolve_owner()
        if env is None:
            env = Env()
        try:
            bound = self._sig.bind(*args, **kwargs).arguments
        except TypeError:
            return self._impl(*args, **kwargs)  # let Python raise its own error
        for name in self.static_refs:
            v = bound[name] if name in bound else self.params[name].default
            v = _concrete_int(v)
            if isinstance(v, Tracer):
                raise StageError(
                    f"{self.name}(): `{name}` sets a size, so it must be known when the "
                    f"function is traced, but it is traced. Under jax.jit, mark it static: "
                    f"jax.jit(f, static_argnames=\"{name}\")."
                )
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                env.dims[name] = int(v)
                env.origin[name] = f"the argument `{name}`"
        if self.field_refs:
            obj = bound.get(next(iter(self.params)))
            for name in self.field_refs:
                v = _concrete_int(getattr(obj, name, None))
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    env.dims[name] = int(v)
                    env.origin[name] = f"`self.{name}`"
        try:
            for name, spec in self._check_inputs:
                if name in bound:
                    kind = self._var_kinds.get(name)
                    if kind is None:
                        spec.match(bound[name], name, env)
                    elif kind is inspect.Parameter.VAR_POSITIONAL:
                        for i, v in enumerate(bound[name]):
                            spec.match(v, f"{name}[{i}]", env)
                    else:
                        for k, v in bound[name].items():
                            spec.match(v, k, env)
            if env.deferred:
                resolve(env)
        except Mismatch as m:
            raise _messages.error(self, m, "argument", env) from None
        if self.where:
            self._check_where(env, final=False)
        # Objects this call creates are its own: updating them in place is fine
        # even when the call itself sits inside a traced branch of the caller.
        depth, _flow._STAGE.depth = _flow._STAGE.depth, 0
        try:
            _ops.CALLS.stack.append((env, self))  # for checked locals, axis names, tx.size
            try:
                out = self._impl(*args, **kwargs)
            finally:
                _ops.CALLS.stack.pop()
        except Exception as e:
            _flow._STAGE.depth = depth
            if config.notes and not getattr(e, "_typax_local", None) is self:
                _messages.annotate(self, e, bound, env)
            raise
        _flow._STAGE.depth = depth
        if self.converted:
            out = _flow.unstack(out)  # lists collected by a staged loop
        if self._check_ret:
            try:
                self.ret.match(out, "return", env)
                if env.deferred:
                    resolve(env)
            except Mismatch as m:
                raise _messages.error(self, m, "return", env) from None
            if self.where:
                self._check_where(env, final=True)
        return out

    def _check_where(self, env: Env, final: bool) -> None:
        for w in self.where:
            if not w.names <= env.dims.keys():
                continue
            if final == (w.names <= self._avail):
                continue  # input-only constraints are checked before the call
            try:
                ok = bool(w(env.dims))
            except Exception:
                continue  # inconclusive for symbolic sizes
            if not ok and any(is_symbolic(env.dims[n]) for n in w.names):
                continue  # cannot be refuted for symbolic sizes
            if not ok:
                binds = ", ".join(f"{n}={env.dims[n]}" for n in sorted(w.names))
                raise ShapeError(
                    f"{self.name}(): constraint `{w}` is violated ({binds}).\n\n"
                    f"    {self.signature}\n    defined at {self.location}"
                )

    # ---------------------------------------------------------------- errors

    @property
    def signature(self) -> str:
        """The signature as typax understands it."""
        if self._lazy:
            self._resolve_owner()
        tps = getattr(self.raw, "__type_params__", ())
        head = getattr(self.raw, "__name__", self.name)
        if tps:
            head += "[" + ", ".join(("*" if "Tuple" in type(t).__name__ else "") + t.__name__ for t in tps) + "]"
        parts = []
        for n, p in self.params.items():
            star = {inspect.Parameter.VAR_POSITIONAL: "*", inspect.Parameter.VAR_KEYWORD: "**"}.get(p.kind, "")
            parts.append(f"{star}{n}: {p.spec}" if p.spec.leafy or not isinstance(p.spec, OpaqueSpec) else f"{star}{n}")
        s = f"{head}({', '.join(parts)})"
        if self.ret.leafy:
            s += f" -> {self.ret}"
        return s

    @property
    def location(self) -> str:
        try:
            code = self.raw.__code__  # type: ignore[attr-defined]
            return f"{code.co_filename}:{code.co_firstlineno}"
        except AttributeError:
            return "<unknown>"

    def _check_local(self, spec: Spec, value: Any, name: str, site: str, env: Env) -> None:
        """Match an annotated local variable against this call's dimensions."""
        __tracebackhide__ = True
        try:
            spec.match(value, name, env)
            if env.deferred:
                resolve(env)
        except Mismatch as m:
            e = _messages.error(self, m, "local", env, site)
            setattr(e, "_typax_local", self)
            raise e from None


    # ------------------------------------------------------------------ lift

    def lift(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function with extra leading batch axes on any array argument.

        Each argument's batch axes are whatever precedes its declared shape.
        Arguments without extra axes are passed to every call unchanged; batch
        axes of different arguments broadcast against each other. The result
        gains the batch axes in front. A single PRNG key is split into one key
        per batch element, so examples do not share randomness.
        """
        __tracebackhide__ = True
        return _lift.lift(self, args, kwargs)


    def _walk(self, arguments: Any, visit: Callable[[Spec, Any, str], Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, v in arguments.items():
            spec = self.params[name].spec
            kind = self._var_kinds.get(name)
            if kind is None:
                out[name] = spec.walk(v, name, visit)
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                out[name] = tuple(spec.walk(x, f"{name}[{i}]", visit) for i, x in enumerate(v))
            else:
                out[name] = {k: spec.walk(x, k, visit) for k, x in v.items()}
        return out

    # ------------------------------------------------------------------ flow

    @property
    def flow(self) -> dict[int, str]:
        """How each converted ``if``/``while``/``for`` last ran, by line number.

        For example ``{12: "for -> lax.scan (carries h)", 15: "if -> lax.cond"}``.
        Statements that have only run as plain Python are not listed.
        """
        from ._flow import LOG

        try:
            code = self.raw.__code__  # type: ignore[attr-defined]
            lines, start = inspect.getsourcelines(self.raw)
        except (OSError, TypeError, AttributeError):
            return {}
        out = {}
        prefix = code.co_filename + ":"
        for site, how in LOG.items():
            if site.startswith(prefix):
                try:
                    line = int(site[len(prefix):])
                except ValueError:
                    continue
                if start <= line < start + len(lines):
                    out[line] = how
        return dict(sorted(out.items()))

    @property
    def converted_source(self) -> str:
        """The control-flow-converted source, for inspection."""
        from ._convert import source

        return source(self.raw)

    # ----------------------------------------------------------------- prove

    def check(self, **examples: Any) -> "Report":
        """Prove the declared shapes for all sizes. See :func:`typax.prove`."""
        from ._check import prove

        return prove(self, examples=examples or None)

    # ------------------------------------------------------------------ misc

    @overload
    def __get__(self, obj: None, cls: Any = None) -> "Fn[P, R]": ...
    @overload
    def __get__(self: "Fn[Concatenate[S, Q], R]", obj: S, cls: Any = None) -> "BoundFn[Q, R]": ...
    def __get__(self, obj: Any, cls: Any = None) -> Any:
        if obj is None:
            return self
        return BoundFn(self, obj)

    def info(self) -> str:
        """Everything typax knows about this function, as text."""
        lines = [self.signature, f"  defined at {self.location}"]
        opts = [k for k, v in (("lift", self.auto_lift), ("random", self.random)) if v]
        if opts:
            lines.append("  options: " + ", ".join(opts))
        for k, e in self.defs.items():
            lines.append(f"  {k} = {e}")
        for w in self.where:
            lines.append(f"  where {w}")
        if self.converted:
            lines.append("  control flow: converted")
            for line, how in self.flow.items():
                lines.append(f"    line {line}: {how}")
        else:
            lines.append(f"  control flow: not converted ({self.convert_note})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<fn {self.signature}>"


class BoundFn(Generic[P, R]):
    """An ``@fn`` method bound to an instance."""

    def __init__(self, fn: Fn[Any, R], obj: Any):
        self.fn = fn
        self.obj = obj
        self.__wrapped__ = fn.raw.__get__(obj)  # type: ignore[attr-defined]
        self.__signature__ = fn._sig.replace(parameters=list(fn._sig.parameters.values())[1:])

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return self.fn(self.obj, *args, **kwargs)

    def lift(self, *args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return self.fn.lift(self.obj, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fn, name)

    def __repr__(self) -> str:
        return f"<bound {self.fn!r} of {type(self.obj).__name__}>"


def unwrap(f: Any) -> Any:
    """The ``Fn`` (or ``BoundFn``) under wrappers such as ``jax.jit``, else ``f`` itself."""
    if isinstance(f, types.MethodType):  # e.g. a jitted method, bound to an instance
        inner = unwrap(f.__func__)
        return inner.__get__(f.__self__) if isinstance(inner, Fn) else f
    g = f
    for _ in range(16):
        if isinstance(g, (Fn, BoundFn)):
            return g
        g = getattr(g, "__wrapped__", None)
        if g is None:
            break
    return f


def _concrete_int(v: Any) -> Any:
    """A concrete 0-d integer array as a Python int; anything else unchanged."""
    if not isinstance(v, Tracer) and getattr(v, "shape", None) == () and hasattr(v, "dtype"):
        if np.dtype(v.dtype).kind in "iu":
            return int(v)
    return v


def _field_names(cls: Any) -> set[str]:
    if cls is None:
        return set()
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(getattr(cls, "_fields", ()))


# ---------------------------------------------------------------- stage hints


# ------------------------------------------------------------------ decorator


@overload
def fn(f: Callable[P, R], /) -> Fn[P, R]: ...
@overload
def fn(
    *,
    jit: bool = False,
    lift: bool = False,
    random: bool = False,
    remat: Any = False,
    donate: str | tuple[str, ...] = (),
    examples: dict[str, Any] | None = None,
    prove: bool = True,
    convert: bool = True,
    **moved: Any,
) -> Callable[[Callable[P, R]], Fn[P, R]]: ...
def fn(
    f: Any = None,
    /,
    *,
    jit: bool = False,
    lift: bool = False,
    random: bool = False,
    remat: Any = False,
    donate: str | tuple[str, ...] = (),
    examples: dict[str, Any] | None = None,
    prove: bool = True,
    convert: bool = True,
    **moved: Any,
) -> Any:
    """Make a function's shape annotations real.

    ::

        @tx.fn
        def split_heads[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:
            assert D % heads == 0
            return x.reshape(T, heads, D // heads).transpose(1, 0, 2)

    Arguments and the result are checked on every call (once per trace under
    ``jax.jit``), the prover checks the shapes for all sizes, and plain
    ``if`` / ``while`` / ``for`` work on traced values. Sizes can be computed in
    the annotations from other sizes, int parameters and fields of ``self``.
    ``assert``s on sizes at the top of the body are constraints: checked before
    the body runs, and assumed by the prover.

    ``jit``
        Compile, as :func:`typax.jit`: static arguments read from the signature.
    ``lift``
        Plain calls take extra leading batch axes, as :func:`typax.lift`.
    ``random``
        An implicit PRNG key, ``tx.key()``, as :func:`typax.random`.
    ``remat``
        Gradient checkpointing, as :func:`typax.remat`: ``True``, or a policy
        (``"dots"``, ``"dots_no_batch"``, ``"everything"`` or a JAX policy).
    ``donate``
        With ``jit``: parameters whose buffers the call may reuse, e.g. the
        state of a training step.
    ``examples``
        Values the prover should try for parameters it cannot guess,
        as ``{"name": [value, ...]}``.
    ``prove``
        ``False`` leaves the function out of ``tx.prove`` and ``pytest --typax``.
    ``convert``
        ``False`` keeps Python control flow as it is (no rewriting).

    Combine with :func:`lift`, :func:`random` and :func:`jit`, and with JAX's
    own transformations.
    """
    hints = {
        "checked": "use `with tx.config(checked=False):`",
        "vectorize": "the option is called lift=True",
        "where": "write the constraint as an `assert` at the top of the body: assert T >= 2",
    }
    for k, v in moved.items():
        hint = hints.get(
            k, f"write the size in the annotation instead, e.g. -> f32[{v}] in place of {k}"
        )
        raise TypeError(f"@tx.fn got {k}={v!r}: {hint}.")

    def wrap(raw: Callable[..., Any]) -> Fn[Any, Any]:
        flags: dict[str, bool] = {}
        if isinstance(raw, Fn):
            flags = {"random": raw.random, "auto_lift": raw.auto_lift}
            REGISTRY.discard(raw)
            raw = raw.raw
        out = Fn(raw, prove=prove, convert=convert, examples=examples)
        vars(out).update(flags)
        if random:
            out.random = True
        if lift:
            out.auto_lift = True
        result: Any = out
        if remat:
            result = _remat(out, remat)
        if jit:
            result = _jit(result, donate_argnames=(donate,) if isinstance(donate, str) else tuple(donate))
        elif donate:
            raise TypeError("@tx.fn(donate=...) only applies with jit=True")
        return result

    if f is not None:
        return wrap(f)
    return wrap


def _flagged(f: Any, flag: str) -> Any:
    if isinstance(f, (staticmethod, classmethod)):
        return type(f)(_flagged(f.__func__, flag))
    if not isinstance(f, Fn) and hasattr(f, "__wrapped__"):
        deco = "@tx.random" if flag == "random" else "@tx.lift"
        raise TypeError(
            f"{deco} must go under @jax.jit / @jax.checkpoint (inside it would be "
            f"compiled once and repeat): put it directly on the @tx.fn function, or use "
            f"@tx.jit, which handles random functions."
        )
    if isinstance(f, Fn):
        g = copy.copy(f)
        g.params = {k: Param(p.name, p.spec, p.kind, p.default) for k, p in f.params.items()}
        g._declared = set(f._declared)
        g.examples = dict(f.examples)
        REGISTRY.discard(f)
    else:
        g = fn(f)
    setattr(g, flag, True)
    REGISTRY.add(g)
    return g


def lift(f: Callable[P, R], /) -> Fn[P, R]:
    """Let plain calls take extra leading batch axes, as ``f.lift(...)`` does.

    ::

        @tx.lift
        @tx.fn
        def norm[D](x: f32[D]) -> f32: ...

        norm(jnp.ones((8, 3, 5)))   # f32[8, 3]

    Arguments without extra axes are shared by every element. A plain
    function is wrapped with :func:`fn` first.
    """
    return _flagged(f, "auto_lift")


def jit(f: Callable[P, R] | None = None, /, **options: Any) -> Any:
    return _jit(f, **options)


def _jit(f: Any = None, /, **options: Any) -> Any:
    """``jax.jit``, with the static arguments read from the signature.

    ::

        @tx.jit
        def tile[N, K](x: f32[N], n: int) -> f32[K]: ...    # n is static

    Parameters annotated with plain Python types (``int``, ``bool``, ``str``,
    tuples, callables, enums, ...) and ``Static[...]`` are static; arrays and
    ``float`` / ``complex`` numbers are traced, so a new learning rate does not
    recompile. A ``@tx.random`` function draws a fresh key outside the compiled
    code on every call, unless ``key=`` is given. Other options go to ``jax.jit``.
    A plain function is wrapped with :func:`fn` first.
    """
    if f is None:
        return lambda g: _jit(g, **options)
    if isinstance(f, (staticmethod, classmethod)):
        return type(f)(_jit(f.__func__, **options))
    if not isinstance(unwrap(f), Fn):
        f = fn(f)
    g: Any = unwrap(f)
    static = _static_names(g)
    given = options.pop("static_argnames", ())
    static += [given] if isinstance(given, str) else list(given)
    compiled = jax.jit(f, static_argnames=tuple(dict.fromkeys(static)), **options)
    if not g.random:
        return compiled

    @functools.wraps(compiled)
    def call(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        if "key" not in kwargs and "key" not in g.params:
            kwargs["key"] = _rng.fresh(f"{g.name}()")
        return compiled(*args, **kwargs)

    return call


def _static_names(g: Any) -> list[str]:
    """Parameters that must be static: plain Python types and Static[...], not float/complex."""
    return [
        n for n, p in g.params.items()
        if isinstance(p.spec, StaticSpec) and (p.spec.explicit or p.spec.inner not in (float, complex))
    ]  # fmt: skip


_POLICIES = {
    "nothing": jax.checkpoint_policies.nothing_saveable,
    "everything": jax.checkpoint_policies.everything_saveable,
    "dots": jax.checkpoint_policies.dots_saveable,
    "dots_no_batch": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
}


def remat(f: Any = None, /, *, policy: Any = None) -> Any:
    """Gradient checkpointing (``jax.checkpoint``) that knows the static arguments.

    In the backward pass, recompute this function's intermediates instead of
    storing them. ``policy`` chooses what is kept anyway: ``"dots"`` (matrix
    products), ``"dots_no_batch"``, ``"everything"``, or a
    ``jax.checkpoint_policies`` policy. Arguments with plain Python types stay
    Python values, as under :func:`jit`.
    """
    if f is None:
        return lambda g: remat(g, policy=policy)
    if isinstance(f, (staticmethod, classmethod)):
        return type(f)(remat(f.__func__, policy=policy))
    return _remat(f if isinstance(unwrap(f), Fn) else fn(f), True if policy is None else policy)


def _remat(f: Any, how: Any) -> Any:
    g: Any = unwrap(f)
    if isinstance(how, str) and how not in _POLICIES:
        raise ValueError(f"remat must be True, a JAX policy or one of {sorted(_POLICIES)}, not {how!r}")
    policy = None if how is True else _POLICIES.get(how, how) if isinstance(how, str) else how
    static = set(_static_names(g))
    sig = g._sig

    @functools.wraps(f)
    def call(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        ba = sig.bind(*args, **kwargs)
        fixed = {n: v for n, v in ba.arguments.items() if n in static}
        dyn = {n: v for n, v in ba.arguments.items() if n not in static}

        def body(dyn: dict[str, Any]) -> Any:
            b = inspect.BoundArguments(sig, {n: fixed[n] if n in fixed else dyn[n] for n in ba.arguments})  # type: ignore[arg-type]
            return f(*b.args, **b.kwargs)

        return jax.checkpoint(body, policy=policy)(dyn)

    call.__signature__ = sig  # type: ignore[attr-defined]
    return call


jit.__doc__ = _jit.__doc__


def random(f: Callable[P, R], /) -> Fn[P, R]:
    """Give a function an implicit PRNG key: ``tx.key()`` inside it is fresh on every call.

    Each call draws a new key from the global key (``tx.seed``) or from the
    enclosing ``with tx.rng(key):`` block. Pass ``key=`` to choose it, which is
    required under ``jax.jit`` (the global key is Python state and would be
    frozen into the compiled function)::

        @tx.random
        @tx.fn
        def noisy[N](x: f32[N]) -> f32[N]:
            return x + jr.normal(tx.key(), x.shape)

        noisy(x)                                    # a new draw every time
        jax.jit(noisy)(x, key=jr.key(0))            # compiled: pass the key

    Under ``f.lift(...)`` each element gets its own key.
    """
    return _flagged(f, "random")

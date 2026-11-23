"""``Fn``: a function whose signature is read, checked on every call and put to work.

An ``Fn`` (made by the decorators in ``decorators.py``) parses its annotations
into specs once. A call goes ``__call__`` → ``_direct`` → ``_run`` (opens a
key scope if random) → ``_run_checked`` (through ``CALL_HOOK`` when a tool
installed one) → ``_checked``, which matches the arguments, runs the converted
body and matches the result. When the arguments fail to match only because
they have extra leading axes, ``_checked`` raises ``batching.Rebatch`` and
``_run`` maps the call over those axes (``batching.vectorize``). ``config``
holds the global switches.
"""

import contextvars
import dataclasses
import functools
import inspect
import os
import types
import typing
import weakref
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Generic, ParamSpec, TypeVar, overload

import jax
import numpy as np
from jax.core import Tracer

from . import batching, keys
from .annotations import messages, sym
from .annotations.dims import DExpr, DName, Env, is_symbolic, resolve
from .annotations.errors import Mismatch, ShapeError, StageError, code_of, coded
from .annotations.expr import Expr
from .annotations.spec import ArraySpec, KeySpec, OpaqueSpec, Spec, StaticSpec, _assumptions_of, _struct_fields, parse, parse_owner, static_type
from .staging import flow, ops
from .staging.convert import NotConverted
from .staging.convert import convert as _convert

if TYPE_CHECKING:
    from .tools.prove import Report

_BatchTracer: Any = ()
try:  # private, but there is no public way to ask "is this value being vmapped?"
    from jax._src.interpreters import batching as _batching

    _BatchTracer = _batching.BatchTracer
except Exception:  # pragma: no cover
    pass


#: Set by ``expex.diagram`` while it traces: ``hook(fn, args, kwargs, run)`` wraps every
#: checked call (``run(args, kwargs)`` runs it), to put it in a ``jax.named_scope``.
CALL_HOOK: contextvars.ContextVar[Callable[..., Any] | None] = contextvars.ContextVar(
    "expex_call_hook", default=None
)

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
S = TypeVar("S")


class Config:
    """Global switches. Also settable through ``EXPEX_*`` environment variables."""

    def __init__(self) -> None:
        #: Check arguments and return values on every (traced or eager) call.
        self.checked: bool = os.environ.get("EXPEX_CHECKED", "1") not in ("0", "false", "")
        #: Attach "in f(x: f32[3, 4])" notes to exceptions passing through ``@fn``.
        self.notes: bool = os.environ.get("EXPEX_NOTES", "1") not in ("0", "false", "")
        #: Turn ``for`` loops into ``lax.scan`` / ``vmap`` when tracing. With False,
        #: loops over arrays and ranges are unrolled, as in plain JAX.
        self.lower_loops: bool = os.environ.get("EXPEX_LOWER_LOOPS", "1") not in ("0", "false", "")
        #: Map a call over extra leading axes of its arguments, when that is all that
        #: keeps them from matching (``@expex.fn(vectorize=False)`` opts one function out).
        self.vectorize: bool = os.environ.get("EXPEX_VECTORIZE", "1") not in ("0", "false", "")
        #: Outside ``jax.jit``, run a mapped call (``f.vectorize``, or mapped
        #: automatically) compiled from its second call with the same shapes on.
        self.compile_vectorized: bool = os.environ.get("EXPEX_COMPILE_VECTORIZED", "1") not in ("0", "false", "")
        #: Mixed precision for every ``@fn`` without a setting of its own (``jnp.bfloat16``,
        #: or a name): float inputs cast to it, float results back to their declared dtype.
        self.compute: Any = os.environ.get("EXPEX_COMPUTE") or None

    def __call__(self, **changes: Any) -> "_Scoped":
        """``with expex.config(checked=False): ...`` changes settings inside the block."""
        for k in changes:
            if k not in vars(self):
                raise coded("EX803", AttributeError(f"expex.config has no setting {k!r}"))
        return _Scoped(self, changes)

    def __repr__(self) -> str:
        return "expex.config(" + ", ".join(f"{k}={v!r}" for k, v in vars(self).items()) + ")"


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

def compile_cache(path: str = "~/.cache/expex", min_compile_seconds: float = 1.0) -> str:
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


def _enclosing_class(raw: Callable[..., Any]) -> type | None:
    """The class a function was defined in, found through its qualified name."""
    parts = getattr(raw, "__qualname__", "").split(".")[:-1]
    if not parts or "<locals>" in parts:
        return None
    obj: Any = getattr(raw, "__globals__", {}).get(parts[0])
    for name in parts[1:]:
        obj = getattr(obj, name, None)
    return obj if isinstance(obj, type) else None


def _scope(raw: Callable[..., Any], localns: dict[str, Any], owner: type | None = None) -> dict[str, Any]:
    """The names annotations are evaluated with: where ``raw`` was defined, its (and its class's) type parameters."""
    out = {**localns, **{p.__name__: p for p in getattr(raw, "__type_params__", ())}}
    if owner is not None:
        out.update({p.__name__: p for p in getattr(owner, "__type_params__", ())})
        out.update(sym.named_sizes(owner))  # cells: i32[Rows:6, Cols:7]
        out[owner.__name__] = owner
    return out


def _annotations(raw: Callable[..., Any], localns: dict[str, Any]) -> dict[str, Any]:
    try:
        if sym.named_sizes(raw):  # f32[R:6]: R is evaluated from the text, below
            raise NameError
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
    loc = _scope(raw, localns)
    params = list(sym.signature(raw).parameters)
    for k, v in sym.texts(raw).items():
        if isinstance(v, str):
            try:
                v = sym.evaluate(v, glb, loc, params)
            except Exception:
                v = inspect.Parameter.empty
        out[k] = v
    return out


class Fn(Generic[P, R]):
    """A function whose shape annotations are checked, provable and usable.

    Created by :func:`fn`. Calling it calls the original function, after
    matching the arguments (and afterwards the result) against the annotations.
    """

    #: ``@expex.fn(compute=...)``: the dtype the body computes in (see ``precision.py``).
    compute: Any = None

    def __init__(
        self,
        raw: Callable[P, R],
        *,
        prove: bool = True,
        convert: bool = True,
        examples: dict[str, Any] | None = None,
    ):
        functools.update_wrapper(self, raw)
        self.raw = raw
        #: Set by ``@expex.random``: the function gets an implicit PRNG key.
        self.random = False
        #: ``@expex.fn(vectorize=...)``: whether a call maps over extra leading axes of its
        #: arguments by itself. None follows ``expex.config.vectorize`` (on by default).
        self.auto_vectorize: bool | None = None
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
        self._needs_sizes = bool(getattr(self._impl, "__expex_sizes__", 0))
        self.name: str = getattr(raw, "__qualname__", getattr(raw, "__name__", repr(raw)))
        self.prove = prove
        self.examples = dict(examples or {})
        self._sig = sym.signature(raw)
        # what inspect.signature (and so jax.jit) sees: evaluating the raw
        # annotations would fail on size arithmetic such as f32[n] or f32[T + 1]
        self.__signature__ = self._sig

        #: Sizes defined in the body: ``H = heads``.
        self.defs = {k: Expr(v) for k, v in getattr(self._impl, "__expex_defs__", {}).items()}
        #: Leading `assert`s on sizes in the body: constraints.
        self._groups: list[list[Expr]] = []
        for group in getattr(self._impl, "__expex_where__", ()):
            try:
                self._groups.append([Expr(w, constraint=True) for w in group])
            except Exception:
                self._groups.append([])
        #: Which leading asserts are constraints (the body skips those).
        self._lifted = {i for i, g in enumerate(self._groups) if g}
        self._asserted = [e for g in self._groups for e in g]
        self.where = list(self._asserted)

        derived = self.defs

        def derive(d: Any) -> Any:
            if isinstance(d, DName) and d.name in derived:
                return DExpr(derived[d.name], d.name)
            return d

        #: Where the function was defined (for annotations written as text).
        self._localns = sym.caller_locals()
        # sizes defined in the body (H = heads) are names in the annotations
        defined = getattr(self._impl, "__expex_defs__", {})
        self._localns.update({n: typing.TypeVar(n) for n in defined})  # type: ignore[misc]  # pyright: ignore[reportGeneralTypeIssues]
        self._localns.update(sym.named_sizes(raw))  # f32[R:6, 7]: R is a fixed size (raises for conflicts)
        anns = _annotations(raw, self._localns)
        # Annotations that could not be evaluated yet (a forward reference to the
        # class being defined, say) are retried at the first call.
        self._pending = {
            k for k, v in sym.texts(raw).items()
            if isinstance(v, str) and anns.get(k, inspect.Parameter.empty) is inspect.Parameter.empty
        }  # fmt: skip
        self._lazy = bool(self._pending)
        names = list(self._sig.parameters)
        if self._pending and not (names and names[0] in ("self", "cls")):
            # not a method, so no class sizes to wait for: report a misspelt size now
            glb = getattr(raw, "__globals__", {})
            scope = _scope(raw, self._localns)
            raw_texts = sym.texts(raw)
            for k in self._pending:
                self._unknown_in(k, raw_texts.get(k), glb, scope, names)
        self._derive = derive
        self.params: dict[str, Param] = {}
        for name, p in self._sig.parameters.items():
            ann = anns.get(name, inspect.Parameter.empty)
            try:
                spec = parse(ann)
            except TypeError as e:
                raise coded(code_of(e) or "EX401", TypeError(f"{self.name}: bad annotation for `{name}`: {e}")) from None
            if not spec.leafy and ann is not inspect.Parameter.empty and static_type(ann):
                # `n: int`, `mode: str`, `cfg: Config`: configuration, not data
                spec = StaticSpec(ann)
            if self.defs:
                spec = spec.replace_dims(derive)
            self.params[name] = Param(name, spec, p.kind, p.default)
        try:
            ret = parse(anns.get("return", inspect.Parameter.empty))
        except TypeError as e:
            raise coded(code_of(e) or "EX401", TypeError(f"{self.name}: bad return annotation: {e}")) from None
        self.ret: Spec = ret.replace_dims(derive) if self.defs else ret

        # Names in the raw signature (before derivation), for validation.
        declared: set[str] = set()
        for name in self._sig.parameters:
            parse(anns.get(name, inspect.Parameter.empty)).names(declared)
        parse(anns.get("return", inspect.Parameter.empty)).names(declared)

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
        owner = getattr(self, "_owner_cls", None)
        fields = set(_struct_fields(owner) or ()) if owner is not None else set()
        self.field_refs = sorted((used - dims - set(self.static_names)) & fields)
        known = declared | set(self.defs) | set(self.static_names) | set(self.field_refs)
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
        unknown = used - declared - set(self.defs) - set(self.static_names) - set(self.field_refs)
        maybe_method = self._owner is None and first in ("self", "cls")
        if unknown and not (maybe_method and not getattr(self, "_resolved", False)):
            raise coded("EX402", TypeError(
                f"{self.name}: {', '.join(sorted(unknown))} used in a size or a constraint is "
                f"neither a size of the signature, nor an int parameter, nor a field of self."
            ))
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
        #: Plain array parameters: (position or None, name, rank, is a key), for the
        #: unchecked fast path to see extra leading axes (see ``_extra_rank``).
        kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self._ranked = [
            (i if p.kind in kinds else None, n, p.spec.core_rank, isinstance(p.spec, KeySpec))
            for i, (n, p) in enumerate(self.params.items())
            if isinstance(p.spec, (ArraySpec, KeySpec)) and p.spec.core_rank is not None and n != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]  # fmt: skip
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
            scope = _scope(self.raw, self._localns, owner)
            glb = getattr(self.raw, "__globals__", {})
            for k, text in sym.texts(self.raw).items():
                if k not in self._pending:
                    continue
                try:
                    spec = parse(sym.evaluate(text, glb, scope, self.params))
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
        if config.compute is not None and self.compute is None:  # mixed precision, globally
            from .precision import under_config

            self = under_config(self)
        try:
            if self.random:
                if key is None and "key" in self.params:
                    key = self._own_key(args, kwargs)
                if not isinstance(key, _BatchTracer) and any(
                    isinstance(x, _BatchTracer) for x in jax.tree.leaves((args, kwargs))
                ):
                    raise coded("EX223", StageError(
                        f"{self.name}() is being mapped with jax.vmap, but its random key is "
                        f"shared by the whole batch: every element would draw the same numbers.\n"
                        f"    - use {self.name}.vectorize(...), which gives each element its own key\n"
                        f"    - or split a key yourself and pass the keys: {self.name}(..., key=keys)"
                    ))
                if key is None:
                    key = keys.fresh(f"{self.name}()")
                with keys.child(key):
                    return self._run_checked(args, kwargs, env, force)
            return self._run_checked(args, kwargs, env, force)
        except batching.Rebatch:
            # the arguments have extra leading axes: map over them (a key is split per element)
            if key is not None and "key" not in self.params:
                kwargs = {**kwargs, "key": key}
            return batching.vectorize(self, args, kwargs, auto=True)

    def _auto(self) -> bool:
        """Whether calls map over extra leading axes by themselves."""
        return config.vectorize if self.auto_vectorize is None else self.auto_vectorize

    def _extra_rank(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        """Unchecked calls: whether a plain array parameter has more axes than declared."""
        for i, name, rank, is_key in self._ranked:
            v = args[i] if i is not None and i < len(args) else kwargs.get(name)
            shape = KeySpec.batch_shape(v) if is_key else getattr(v, "shape", None)
            if shape is not None and len(shape) > rank:
                return True
        return False

    def _wants_batch(self, args: tuple[Any, ...], kwargs: dict[str, Any], force: bool) -> bool:
        """Whether the argument check would ask to map over extra leading axes.

        Asked before a call hook sees the call, so that tools see only the
        mapped calls, as with ``f.vectorize``; without a hook ``_checked`` finds out.
        """
        if not self._auto():
            return False
        if not force and not config.checked and not self._needs_sizes:
            return bool(self._ranked) and self._extra_rank(args, kwargs)
        if self._lazy:
            self._resolve_owner()
        try:
            self._match_args(self._sig.bind(*args, **kwargs).arguments, Env())
        except Mismatch as m:
            return batching.extra_axes(self, m)
        except Exception:
            return False
        return False

    def _unknown_in(
        self, where: str, text: Any, glb: dict[str, Any], scope: dict[str, Any], params: Any
    ) -> None:
        """Raise if an array annotation names something undefined (a misspelt size, say).

        Other unresolvable annotations (a ``TYPE_CHECKING``-only class) are ignored.
        """
        if not isinstance(text, str):
            return
        try:
            value, missing = sym.unknown_names(text, glb, scope, params)
            leafy = parse(value).leafy
        except Exception:
            return
        if missing and leafy:
            what = "the result" if where == "return" else f"`{where}`"
            raise coded("EX402", TypeError(
                f"{self.name}: the annotation of {what} ({text}) uses "
                f"{', '.join(sorted(missing))}, which is not defined. Sizes are type "
                f"parameters (def f[T, D](...)), int parameters, or fields of self."
            ))

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
            if self._wants_batch(args, kwargs, force):
                raise batching.Rebatch
            return hook(self, args, kwargs, lambda a, k: self._checked(a, k, env, force))
        return self._checked(args, kwargs, env, force)

    def _checked(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], env: Env | None, force: bool
    ) -> Any:
        __tracebackhide__ = True
        if not force and not config.checked and not self._needs_sizes:
            if self._ranked and self._auto() and self._extra_rank(args, kwargs):
                raise batching.Rebatch
            out = self._impl(*args, **kwargs)
            return flow.unstack(out) if self.converted else out
        if self._lazy:
            self._resolve_owner()
        if env is None:
            env = Env()
        try:
            bound = self._sig.bind(*args, **kwargs).arguments
        except TypeError:
            return self._impl(*args, **kwargs)  # let Python raise its own error
        try:  # as _match_args, inlined: this is every checked call
            for name in self.static_refs:
                v = bound[name] if name in bound else self.params[name].default
                v = _concrete_int(v)
                if isinstance(v, Tracer):
                    raise coded("EX202", StageError(
                        f"{self.name}(): `{name}` sets a size, so it must be known when the "
                        f"function is traced, but it is traced. Under jax.jit, mark it static: "
                        f"jax.jit(f, static_argnames=\"{name}\")."
                    ))
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    env.dims[name] = int(v)
                    env.origin[name] = f"the argument `{name}`"
                elif jax.export.is_symbolic_dim(v):  # a size passed on under the prover: `self.empty(T)`
                    env.dims[name] = v
                    env.origin[name] = f"the argument `{name}`"
            if self.field_refs:
                obj = bound.get(next(iter(self.params)))
                for name in self.field_refs:
                    v = _concrete_int(getattr(obj, name, None))
                    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                        env.dims[name] = int(v)
                        env.origin[name] = f"`self.{name}`"
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
            if self._auto() and batching.extra_axes(self, m):
                raise batching.Rebatch from None
            err = messages.error(self, m, "argument", env)
            setattr(err, "_expex_arguments", (self, m.path))  # for a mapped call to say it was mapped
            raise err from None
        if self.where:
            self._check_where(env, final=False)
        # Objects this call creates are its own: updating them in place is fine
        # even when the call itself sits inside a traced branch of the caller.
        depth, flow._STAGE.depth = flow._STAGE.depth, 0
        try:
            ops.CALLS.stack.append((env, self))  # for checked locals, axis names, expex.size
            try:
                out = self._impl(*args, **kwargs)
            finally:
                ops.CALLS.stack.pop()
        except Exception as e:
            flow._STAGE.depth = depth
            if config.notes and not getattr(e, "_expex_local", None) is self:
                messages.annotate(self, e, bound, env)
            if isinstance(e, ShapeError) and not ops.CALLS.stack and CALL_HOOK.get() is None:
                from ._diagram import attach  # the outermost call: e.diagram, built only if used

                attach(self, e, args, kwargs)
            raise
        flow._STAGE.depth = depth
        if self.converted:
            out = flow.unstack(out)  # lists collected by a staged loop
        if self._check_ret:
            try:
                self.ret.match(out, "return", env)
                if env.deferred:
                    resolve(env)
            except Mismatch as m:
                raise messages.error(self, m, "return", env) from None
            if self.where:
                self._check_where(env, final=True)
        return out

    def _match_args(self, bound: dict[str, Any], env: Env) -> None:
        """Bind this call's sizes in ``env`` and match the arguments (raises ``Mismatch``).

        For ``_wants_batch``; ``_checked`` does the same inline (keep them in step).
        """
        for name in self.static_refs:
            v = bound[name] if name in bound else self.params[name].default
            v = _concrete_int(v)
            if isinstance(v, Tracer):
                raise coded("EX202", StageError(
                    f"{self.name}(): `{name}` sets a size, so it must be known when the "
                    f"function is traced, but it is traced. Under jax.jit, mark it static: "
                    f"jax.jit(f, static_argnames=\"{name}\")."
                ))
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                env.dims[name] = int(v)
                env.origin[name] = f"the argument `{name}`"
            elif jax.export.is_symbolic_dim(v):  # a size passed on under the prover: `self.empty(T)`
                env.dims[name] = v
                env.origin[name] = f"the argument `{name}`"
        if self.field_refs:
            obj = bound.get(next(iter(self.params)))
            for name in self.field_refs:
                v = _concrete_int(getattr(obj, name, None))
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    env.dims[name] = int(v)
                    env.origin[name] = f"`self.{name}`"
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
                raise coded("EX104", ShapeError(
                    f"{self.name}(): constraint `{w}` is violated ({binds}).\n\n"
                    f"    {self.signature}\n    defined at {self.location}"
                ))

    # ---------------------------------------------------------------- errors

    @property
    def signature(self) -> str:
        """The signature as expex understands it."""
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
            e = messages.error(self, m, "local", env, site)
            setattr(e, "_expex_local", self)
            raise e from None


    # ------------------------------------------------------------------ vectorize

    def vectorize(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function with extra leading batch axes on any array argument.

        Each argument's batch axes are whatever precedes its declared shape.
        Arguments without extra axes are passed to every call unchanged; batch
        axes of different arguments broadcast against each other. The result
        gains the batch axes in front. A single PRNG key is split into one key
        per batch element, so examples do not share randomness.

        A plain call does the same by itself when extra leading axes are all
        that keep the arguments from matching, except on ``self``: this maps
        over a batch of models too, and works with ``vectorize=False``.
        """
        __tracebackhide__ = True
        return batching.vectorize(self, args, kwargs)


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
        from .staging.flow import LOG

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
        from .staging.convert import source

        return source(self.raw)

    # ----------------------------------------------------------------- prove

    def check(self, **examples: Any) -> "Report":
        """Prove the declared shapes for all sizes. See :func:`expex.prove`."""
        from .tools.prove import prove

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
        """Everything expex knows about this function, as text."""
        lines = [self.signature, f"  defined at {self.location}"]
        opts = [k for k, v in (("vectorize", self.auto_vectorize), ("random", self.random)) if v]
        if self.auto_vectorize is False:
            opts.append("vectorize=False")
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

    __slots__ = ("fn", "obj", "__weakref__")

    def __init__(self, fn: Fn[Any, R], obj: Any):
        self.fn = fn
        self.obj = obj

    @property
    def __wrapped__(self) -> Any:
        return self.fn.raw.__get__(self.obj)  # type: ignore[attr-defined]

    @property
    def __signature__(self) -> inspect.Signature:
        return self.fn._sig.replace(parameters=list(self.fn._sig.parameters.values())[1:])

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return self.fn(self.obj, *args, **kwargs)

    def vectorize(self, *args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        return self.fn.vectorize(self.obj, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fn, name)

    def __repr__(self) -> str:
        return f"<bound {self.fn!r} of {type(self.obj).__name__}>"


class BoundCompiled(functools.partial):  # type: ignore[type-arg]
    """A compiled method bound to an instance: called as directly as ``jax.jit``'s own, ``.vectorize`` bound too."""

    @property
    def __func__(self) -> Any:
        return self.func

    @property
    def __self__(self) -> Any:
        return self.args[0]

    def vectorize(self, *args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        return getattr(self.func, "vectorize")(self.args[0], *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.func, name)


def unwrap(f: Any) -> Any:
    """The ``Fn`` (or ``BoundFn``) under wrappers such as ``jax.jit``, else ``f`` itself."""
    if isinstance(f, (types.MethodType, BoundCompiled)):  # e.g. a jitted method, bound to an instance
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

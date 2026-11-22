"""``@fn``: the decorator that reads a signature and puts it to work."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import os
import warnings
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

from . import _flow, _lift, _messages, _ops, _rng
from ._convert import NotConverted
from ._convert import convert as _convert
from ._dims import DExpr, DName, Env, describe, is_symbolic, resolve
from ._errors import Mismatch, RetraceWarning, ShapeError, StageError
from ._expr import Expr
from ._jit import _avals, _Frozen, _is_traced_leaf, _mark
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
        #: Warn when a ``jit=True`` function has been traced more than this many times.
        self.retrace_warn: int = int(os.environ.get("TYPAX_RETRACE_WARN", "16"))


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

_REMAT_POLICIES = {
    "nothing": jax.checkpoint_policies.nothing_saveable,
    "dots": jax.checkpoint_policies.dots_saveable,
    "dots_no_batch": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
}


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
    try:
        return inspect.get_annotations(raw)
    except Exception:
        return {}


def _enclosing_class(raw: Callable[..., Any]) -> type | None:
    """The class a function was defined in, found through its qualified name."""
    parts = getattr(raw, "__qualname__", "").split(".")[:-1]
    if not parts or "<locals>" in parts:
        return None
    obj: Any = getattr(raw, "__globals__", {}).get(parts[0])
    for name in parts[1:]:
        obj = getattr(obj, name, None)
    return obj if isinstance(obj, type) else None


def _annotations(raw: Callable[..., Any]) -> dict[str, Any]:
    try:
        return inspect.get_annotations(raw, eval_str=True)
    except Exception:
        pass
    # Some annotation could not be evaluated (e.g. a TYPE_CHECKING-only import).
    # Evaluate them one by one and ignore the ones that fail.
    out: dict[str, Any] = {}
    glb = getattr(raw, "__globals__", {})
    loc = {p.__name__: p for p in getattr(raw, "__type_params__", ())}
    for k, v in inspect.get_annotations(raw).items():
        if isinstance(v, str):
            try:
                v = eval(v, glb, loc)  # noqa: S307 - same as typing.get_type_hints
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
        jit: bool = False,
        checked: bool = True,
        prove: bool = True,
        convert: bool = True,
        random: bool = False,
        lift: bool = False,
        remat: Any = False,
        examples: dict[str, Any] | None = None,
        where: str | list[str] | tuple[str, ...] = (),
        defs: dict[str, str] | None = None,
    ):
        functools.update_wrapper(self, raw)
        self.raw = raw
        self.random = random
        if isinstance(remat, str) and remat not in _REMAT_POLICIES:
            raise ValueError(f"remat must be True, a policy or one of {sorted(_REMAT_POLICIES)}")
        #: Recompute this function's intermediates in the backward pass (``jax.checkpoint``)
        self.remat = remat
        #: With ``lift=True``, a plain call maps over extra leading axes by itself.
        self.auto_lift = lift
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
        self._has_locals = bool(getattr(self._impl, "__typax_locals__", 0))
        self.name: str = getattr(raw, "__qualname__", getattr(raw, "__name__", repr(raw)))
        self.checked = checked
        self.prove = prove
        self.examples = dict(examples or {})
        self.jit = jit
        self._sig = inspect.signature(raw)

        defs = defs or {}
        self.defs = {k: Expr(v) for k, v in defs.items()}
        if isinstance(where, str):
            where = [where]
        self.where = [Expr(w, constraint=True) for w in where]

        def derive(d: Any) -> Any:
            if isinstance(d, DName) and d.name in self.defs:
                return DExpr(self.defs[d.name], d.name)
            return d

        anns = _annotations(raw)
        # Annotations that could not be evaluated yet (a forward reference to the
        # class being defined, say) are retried at the first call.
        self._pending = {
            k for k, v in _raw_annotations(raw).items()
            if isinstance(v, str) and anns.get(k, inspect.Parameter.empty) is inspect.Parameter.empty
        }  # fmt: skip
        self._lazy = bool(self._pending)
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
        for k in self.defs:
            if k not in declared and not self._pending:
                raise TypeError(
                    f"{self.name}: `{k}=...` was given to @fn, but `{k}` does not appear "
                    f"in the signature."
                )

        self._owner: type | None = None  # set by __set_name__ when defined in a class
        self.static_names = [n for n, p in self.params.items() if isinstance(p.spec, StaticSpec)]
        self._declared = declared
        self._index()

        #: One entry per trace of the jitted function (``jit=True`` only).
        self.traces: list[dict[str, str]] = []
        self._jitted: Callable[..., Any] | None = None
        self._out_struct: dict[Any, tuple[Any, tuple[Any, ...]]] = {}
        if jit:

            def core(frozen: "_Frozen", dyn: tuple[Any, ...], key: Any) -> Any:
                args, kwargs = frozen.rebuild(self, dyn)
                self._record_trace(args, kwargs)
                out = _mark(self.ret, self._run(args, kwargs, key=key), True)
                # Only arrays can leave a compiled function. Anything else in the
                # result (a function, a string, a Static field) depends only on
                # the static inputs, so it is remembered here and put back after.
                leaves, tree = jax.tree.flatten(out)
                static = tuple(_Frozen if _is_traced_leaf(x) else x for x in leaves)
                self._out_struct[(frozen, _avals(dyn), key is None)] = (tree, static)
                return tuple(x for x in leaves if _is_traced_leaf(x))

            core.__name__ = getattr(raw, "__name__", "fn")
            core.__qualname__ = self.name
            self._jitted = jax.jit(core, static_argnums=(0,))

        REGISTRY.add(self)

    def _index(self) -> None:
        """Derive lookup tables from ``self.params`` (re-run if a spec changes)."""
        declared = self._declared
        self.input_names: set[str] = set()
        for prm in self.params.values():
            prm.spec.names(self.input_names)
        # Static int parameters may be used by name in derived sizes and constraints,
        # e.g. @fn(H="heads", where="D % heads == 0").
        used: set[str] = set()
        for expr in [*self.defs.values(), *self.where]:
            used |= expr.names
        self.static_refs = [n for n in self.static_names if n in used]
        # ... and so may a method's int fields of `self`: @fn(H="heads") for self.heads
        fields = _field_names(getattr(self, "_owner_cls", None))
        self.field_refs = sorted((used - declared - set(self.static_names)) & fields)
        unknown = used - declared - set(self.static_names) - set(self.field_refs)
        first = next(iter(self.params), None)
        maybe_method = self._owner is None and first in ("self", "cls")
        if unknown and not (maybe_method and not getattr(self, "_resolved", False)):
            raise TypeError(
                f"{self.name}: {', '.join(sorted(unknown))} used in a derived size or `where` "
                f"constraint is neither a dimension in the signature nor a Static parameter."
            )
        self.input_names -= set(self.static_refs)
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
            scope = {p.__name__: p for p in getattr(self.raw, "__type_params__", ())}
            if owner is not None:
                scope.update({p.__name__: p for p in getattr(owner, "__type_params__", ())})
                scope[owner.__name__] = owner
            glb = getattr(self.raw, "__globals__", {})
            for k, text in _raw_annotations(self.raw).items():
                if k not in self._pending:
                    continue
                try:
                    spec = parse(eval(text, glb, scope))  # noqa: S307 - as typing.get_type_hints
                except Exception:
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
        if self._jitted is not None:
            # The key is drawn out here and passed in as a real argument, so the
            # compiled function gets fresh randomness on every call.
            k = _rng.fresh(f"{self.name}()") if self.random else None
            return self._call_jit(args, kwargs, k)
        return self._run(args, kwargs)

    def _call_jit(self, args: tuple[Any, ...], kwargs: dict[str, Any], key: Any) -> Any:
        """Call the compiled function: arrays are traced, everything else is static.

        ``Static[...]`` parameters and fields, and any non-array value inside an
        argument (a function, a string, an arbitrary object), become part of the
        compilation key, so no manual ``static_argnames`` is needed.
        """
        __tracebackhide__ = True
        assert self._jitted is not None
        if self._lazy:
            self._resolve_owner()
        try:
            bound = self._sig.bind(*args, **kwargs)
        except TypeError:
            return self.raw(*args, **kwargs)  # let Python raise its own error
        frozen, dyn = _Frozen.split(self, bound.arguments)
        try:
            hash(frozen)
        except TypeError as e:
            raise TypeError(
                f"{self.name}() is jit=True, so its non-array arguments are part of the "
                f"compilation key and must be hashable ({e}). Use tuples instead of lists "
                f"for Static values."
            ) from None
        dyn_out = self._jitted(frozen, dyn, key)
        tree, static = self._out_struct[(frozen, _avals(dyn), key is None)]
        it = iter(dyn_out)
        out = tree.unflatten([next(it) if x is _Frozen else x for x in static])
        return _mark(self.ret, out, False)

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
            if key is None:
                key = _rng.fresh(f"{self.name}()")
            if not isinstance(key, _BatchTracer) and any(
                isinstance(x, _BatchTracer) for x in jax.tree.leaves((args, kwargs))
            ):
                raise StageError(
                    f"{self.name}() is being mapped with jax.vmap, but its random key is "
                    f"shared by the whole batch: every element would draw the same numbers.\n"
                    f"    - use {self.name}.lift(...), which gives each element its own key\n"
                    f"    - or split a key yourself and pass the keys as a mapped argument"
                )
            with _rng.child(key):
                return self._run_checked(args, kwargs, env, force)
        return self._run_checked(args, kwargs, env, force)

    def _run_checked(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], env: Env | None, force: bool
    ) -> Any:
        __tracebackhide__ = True
        if not force and not (self.checked and config.checked):
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
            if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                env.dims[name] = int(v)
                env.origin[name] = f"the argument `{name}`"
        if self.field_refs:
            obj = bound.get(next(iter(self.params)))
            for name in self.field_refs:
                v = getattr(obj, name, None)
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
            if self._has_locals:
                _ops.CALLS.stack.append((env, self))
                try:
                    out = self._call_impl(args, kwargs)
                finally:
                    _ops.CALLS.stack.pop()
            else:
                out = self._call_impl(args, kwargs)
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

    def _call_impl(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run the function body, under ``jax.checkpoint`` if ``remat`` was asked for."""
        if not self.remat:
            return self._impl(*args, **kwargs)
        policy = _REMAT_POLICIES.get(self.remat) if isinstance(self.remat, str) else (
            None if self.remat is True else self.remat
        )
        bound = self._sig.bind(*args, **kwargs)
        frozen, dyn = _Frozen.split(self, bound.arguments)
        box: dict[str, Any] = {}

        def body(dyn: tuple[Any, ...]) -> tuple[Any, ...]:
            a, k = frozen.rebuild(self, dyn)
            leaves, tree = jax.tree.flatten(self._impl(*a, **k))
            box["tree"] = tree
            box["static"] = tuple(_Frozen if _is_traced_leaf(x) else x for x in leaves)
            return tuple(x for x in leaves if _is_traced_leaf(x))

        out = jax.checkpoint(body, policy=policy)(dyn)
        it = iter(out)
        return box["tree"].unflatten([next(it) if x is _Frozen else x for x in box["static"]])

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


    # ------------------------------------------------------------------- jit

    def _record_trace(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        try:
            bound = self._sig.bind(*args, **kwargs).arguments
        except TypeError:
            return
        rec: dict[str, str] = {}
        for name, v in bound.items():
            if name in self.static_names:
                rec[name] = f"static {v!r}"
            else:
                leaves = jax.tree.leaves(v)
                rec[name] = ", ".join(map(_trace_desc, leaves)) if leaves else repr(v)
        self.traces.append(rec)
        n = len(self.traces)
        # warn at the threshold, then only each time the count doubles
        if n > max(config.retrace_warn, 1) and (n - 1) & (n - 2) == 0 or n == config.retrace_warn + 1 > 1:
            warnings.warn(
                f"{self.name}() has been traced {n} times; each trace recompiles.\n"
                + self.explain_retrace()
                + f"\n(inspect `{self.name}.traces`, or raise typax.config.retrace_warn)",
                RetraceWarning,
                stacklevel=4,
            )

    def explain_retrace(self, index: int = -1) -> str:
        """Describe what changed between trace ``index`` and the closest earlier trace."""
        if not self.traces:
            return f"{self.name}() has not been traced."
        index = index % len(self.traces)
        cur = self.traces[index]
        if index == 0:
            return f"{self.name}(): trace #1 (first call)."
        best, best_i = None, 0
        for i, prev in enumerate(self.traces[:index]):
            diff = [k for k in cur.keys() | prev.keys() if cur.get(k) != prev.get(k)]
            if best is None or len(diff) < len(best):
                best, best_i = diff, i
        assert best is not None
        prev = self.traces[best_i]
        lines = [f"{self.name}(): trace #{index + 1} differs from trace #{best_i + 1} in:"]
        if not best:
            lines.append(
                "    nothing visible: same shapes, dtypes and static values. A static "
                "argument may hash differently, or the jit cache was cleared."
            )
        for k in sorted(best):
            lines.append(f"    {k}: {prev.get(k, '<not passed>')}  ->  {cur.get(k, '<not passed>')}")
        return "\n".join(lines)

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
        opts = [
            k for k, v in (("jit", self.jit), ("random", self.random), ("unchecked", not self.checked)) if v
        ]
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
        if self.jit:
            lines.append(f"  traced {len(self.traces)} time(s)")
            for i in range(1, len(self.traces)):
                lines.append("    " + self.explain_retrace(i).replace("\n", "\n    "))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<fn {self.signature}>"


class BoundFn(Generic[P, R]):
    """An ``@fn`` method bound to an instance."""

    def __init__(self, fn: Fn[Any, R], obj: Any):
        self.fn = fn
        self.obj = obj
        self.__wrapped__ = fn.raw.__get__(obj)  # type: ignore[attr-defined]

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


def _field_names(cls: Any) -> set[str]:
    if cls is None:
        return set()
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set(getattr(cls, "_fields", ()))


def _trace_desc(leaf: Any) -> str:
    d = describe(leaf)
    aval = getattr(leaf, "aval", None)
    if getattr(aval, "weak_type", False):
        d = f"weak {d}" + (" (a Python scalar)" if d.endswith("[()]") else "")
    return d


# ---------------------------------------------------------------- stage hints


# ------------------------------------------------------------------ decorator


@overload
def fn(f: Callable[P, R], /) -> Fn[P, R]: ...
@overload
def fn(
    *,
    jit: bool = False,
    checked: bool = True,
    prove: bool = True,
    convert: bool = True,
    random: bool = False,
    lift: bool = False,
    remat: Any = False,
    examples: dict[str, Any] | None = None,
    where: str | list[str] | tuple[str, ...] = (),
    **dims: str,
) -> Callable[[Callable[P, R]], Fn[P, R]]: ...
def fn(
    f: Any = None,
    /,
    *,
    jit: bool = False,
    checked: bool = True,
    prove: bool = True,
    convert: bool = True,
    random: bool = False,
    lift: bool = False,
    remat: Any = False,
    examples: dict[str, Any] | None = None,
    where: str | list[str] | tuple[str, ...] = (),
    **dims: str,
) -> Any:
    """Make a function's shape annotations real.

    ::

        @fn
        def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]: ...

    Options (all keyword-only):

    ``jit``
        Also ``jax.jit`` the function. ``Static[...]`` parameters become static
        arguments, and retraces are recorded in ``.traces``.
    ``checked``
        Check arguments and results on each call (default). Under ``jit`` this
        runs once per trace and costs nothing in the compiled function.
    ``prove``
        Include the function when the prover / pytest plugin runs (default).
    ``convert``
        Rewrite plain ``if`` / ``while`` / ``for`` so that they also work on traced
        values (default). See ``.flow`` for how each statement ran.
    ``random``
        Give the function an implicit PRNG key: ``typax.key()`` inside it
        returns a fresh key on every call.
    ``lift``
        Map over extra leading axes on a plain call, as ``f.lift(...)`` does. For
        functions written for one element that should also accept batches.
    ``remat``
        Gradient checkpointing: in the backward pass, recompute this function's
        intermediates instead of storing them. ``True`` saves nothing,
        ``"dots"`` keeps matrix-multiply results, ``"dots_no_batch"`` keeps
        those without batch dimensions; or pass a ``jax.checkpoint_policies``
        policy. Trades compute for memory, usually per layer.
    ``examples``
        Values the prover should use for ``Static`` or unannotated parameters,
        as ``{"name": [value, ...]}``.
    ``where``
        Constraints on dimensions, e.g. ``where="T >= 2"``.
    ``**dims``
        Derived dimensions, e.g. ``U="T + 1"``. Expressions (here and in ``where``)
        may also use ``Static`` int parameters by name: ``H="heads"``.
    """

    def wrap(raw: Callable[..., Any]) -> Fn[Any, Any]:
        if isinstance(raw, Fn):
            raw = raw.raw
        return Fn(
            raw, jit=jit, checked=checked, prove=prove, convert=convert, random=random,
            lift=lift, remat=remat, examples=examples, where=where, defs=dims,
        )  # fmt: skip

    if f is not None:
        return wrap(f)
    return wrap

"""``@fn``: the decorator that reads a signature and puts it to work."""

from __future__ import annotations

import functools
import inspect
import linecache
import os
import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Concatenate, Generic, ParamSpec, TypeVar, overload

import jax
import jax.numpy as jnp

from . import _rng
from ._convert import NotConverted, convert as _convert
from ._errors import Mismatch, RetraceWarning, ShapeError, StageError
from ._expr import Expr
from ._spec import (
    ArraySpec,
    DExpr,
    DName,
    Env,
    KeySpec,
    OpaqueSpec,
    Spec,
    StaticSpec,
    describe,
    is_symbolic,
    parse,
    resolve,
)

if TYPE_CHECKING:
    from ._batch import Batched
    from ._check import Report

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
S = TypeVar("S")


class Config:
    """Global switches. Also settable through ``BARNACLE_*`` environment variables."""

    def __init__(self) -> None:
        #: Check arguments and return values on every (traced or eager) call.
        self.checked: bool = os.environ.get("BARNACLE_CHECKED", "1") not in ("0", "false", "")
        #: Attach "in f(x: f32[3, 4])" notes to exceptions passing through ``@fn``.
        self.notes: bool = os.environ.get("BARNACLE_NOTES", "1") not in ("0", "false", "")
        #: Warn when a ``jit=True`` function has been traced more than this many times.
        self.retrace_warn: int = int(os.environ.get("BARNACLE_RETRACE_WARN", "4"))


config = Config()

#: Every live ``Fn``; used by the prover to find functions to check.
REGISTRY: "weakref.WeakSet[Fn[Any, Any]]" = weakref.WeakSet()


class Param:
    __slots__ = ("name", "spec", "kind", "default")

    def __init__(self, name: str, spec: Spec, kind: Any, default: Any):
        self.name = name
        self.spec = spec
        self.kind = kind
        self.default = default


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
        examples: dict[str, Any] | None = None,
        where: str | list[str] | tuple[str, ...] = (),
        defs: dict[str, str] | None = None,
    ):
        functools.update_wrapper(self, raw)
        self.raw = raw
        self.random = random
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
        self.params: dict[str, Param] = {}
        for name, p in self._sig.parameters.items():
            try:
                spec = parse(anns.get(name, inspect.Parameter.empty))
            except TypeError as e:
                raise TypeError(f"{self.name}: bad annotation for `{name}`: {e}") from None
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
            if k not in declared:
                raise TypeError(
                    f"{self.name}: `{k}=...` was given to @fn, but `{k}` does not appear "
                    f"in the signature."
                )

        self.input_names: set[str] = set()
        for prm in self.params.values():
            prm.spec.names(self.input_names)
        self.static_names = [n for n, p in self.params.items() if isinstance(p.spec, StaticSpec)]
        self._check_inputs = [
            (n, p.spec) for n, p in self.params.items() if p.spec.leafy
        ]
        self._check_ret = self.ret.leafy
        self._var_kinds = {
            n: p.kind
            for n, p in self.params.items()
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }

        #: One entry per trace of the jitted function (``jit=True`` only).
        self.traces: list[dict[str, str]] = []
        self._jitted: Callable[..., Any] | None = None
        if jit:

            @functools.wraps(raw)
            def traced(*args: Any, __bn_key__: Any = None, **kwargs: Any) -> Any:
                self._record_trace(args, kwargs)
                return self._run(args, kwargs, key=__bn_key__)

            self._jitted = jax.jit(traced, static_argnames=tuple(self.static_names) or None)

        REGISTRY.add(self)

    # ------------------------------------------------------------------ call

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        __tracebackhide__ = True
        if self._jitted is not None:
            try:
                if self.random:
                    # The key is drawn out here and passed in as a real argument, so
                    # the compiled function gets fresh randomness on every call.
                    k = _rng.fresh(f"{self.name}()")
                    return self._jitted(*args, __bn_key__=k, **kwargs)
                return self._jitted(*args, **kwargs)
            except ValueError as e:
                if "Non-hashable static arguments" in str(e):
                    e.add_note(
                        f"barnacle: {self.name}() is jit=True and its Static parameters "
                        f"({', '.join(self.static_names)}) are passed to jax.jit as static "
                        f"arguments, so their values must be hashable (use tuples, not lists "
                        f"or arrays)."
                    )
                raise
        return self._run(args, kwargs)

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
            with _rng.child(key):
                return self._run_checked(args, kwargs, env, force)
        return self._run_checked(args, kwargs, env, force)

    def _run_checked(
        self, args: tuple[Any, ...], kwargs: dict[str, Any], env: Env | None, force: bool
    ) -> Any:
        __tracebackhide__ = True
        if not force and not (self.checked and config.checked):
            return self._impl(*args, **kwargs)
        if env is None:
            env = Env()
        try:
            bound = self._sig.bind(*args, **kwargs).arguments
        except TypeError:
            return self._impl(*args, **kwargs)  # let Python raise its own error
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
            raise self._error(m, "argument", env) from None
        if self.where:
            self._check_where(env, final=False)
        try:
            out = self._impl(*args, **kwargs)
        except Exception as e:
            if config.notes:
                self._annotate(e, bound, env)
            raise
        if self._check_ret:
            try:
                self.ret.match(out, "return", env)
                if env.deferred:
                    resolve(env)
            except Mismatch as m:
                raise self._error(m, "return", env) from None
            if self.where:
                self._check_where(env, final=True)
        return out

    def _check_where(self, env: Env, final: bool) -> None:
        for w in self.where:
            if not w.names <= env.dims.keys():
                continue
            if final == (w.names <= self.input_names):
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
        """The signature as barnacle understands it."""
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

    def _error(self, m: Mismatch, role: str, env: Env) -> Exception:
        if role == "argument":
            top = m.path.split(".")[0].split("[")[0]
            head = f"{self.name}(): argument `{top}` does not match its annotation."
            declared = f"{m.path}: {m.spec}"
            got = "received"
        else:
            head = f"{self.name}(): the return value does not match its annotation."
            sub = m.path[len("return"):]
            declared = f"-> {m.spec}" if not sub else f"return{sub}: {m.spec}"
            got = "returned"
        pad = " " * max(0, len(declared) - len(str(m.spec)))
        lines = [
            head,
            "",
            f"    declared  {declared}",
            f"    {got}  {pad}{describe(m.value)}",
            "",
            f"    {m.reason}",
        ]
        binds = env.summary()
        if binds and role == "return":
            lines.append(f"    Bound so far: {binds}.")
        lines += ["", f"    {self.signature}", f"    defined at {self.location}"]
        cls = StageError if getattr(m, "stage", False) else ShapeError
        return cls("\n".join(lines))

    def _annotate(self, e: Exception, bound: dict[str, Any], env: Env) -> None:
        try:
            _stage_hint(e, self)
            args = ", ".join(
                f"{n}: {describe(v)}" for n, v in bound.items() if self.params[n].spec.leafy
            )
            binds = env.summary()
            note = f"barnacle: in {self.name}({args})" + (f"  [{binds}]" if binds else "")
            if note not in getattr(e, "__notes__", ()):
                e.add_note(note)
        except Exception:
            pass

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
        if n > max(config.retrace_warn, 1):
            warnings.warn(
                f"{self.name}() has been traced {n} times; each trace recompiles.\n"
                + self.explain_retrace()
                + f"\n(inspect `{self.name}.traces`, or raise barnacle.config.retrace_warn)",
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
        gains the batch axes in front.

        A single PRNG key is split into one key per batch element, so examples
        do not share randomness.
        """
        __tracebackhide__ = True
        ba = self._sig.bind(*args, **kwargs)
        leaves, batches, bshape = self._batch_plan(ba.arguments, "lift")
        if not bshape:
            return self(*args, **kwargs)

        n = len(bshape)
        prepared: list[Any] = []
        levels: list[set[int]] = []
        if self.random:  # one key per example
            prepared.append(jax.random.split(_rng.fresh(f"{self.name}.lift()"), bshape))
            levels.append(set(range(n)))
        for (spec, v, path), b in zip(leaves, batches):
            if isinstance(spec, KeySpec) and not b:
                split: Callable[[Any], Any] = lambda k: jax.random.split(k, bshape)  # noqa: E731
                for _ in range(spec.core_rank or 0):
                    split = jax.vmap(split, out_axes=n)
                prepared.append(split(v))
                levels.append(set(range(n)))
                continue
            off = n - len(b)
            ones = tuple(i for i, s in enumerate(b) if _is_one(s))
            if ones:
                v = jnp.squeeze(jnp.asarray(v), axis=ones)
            prepared.append(v)
            levels.append({off + i for i in range(len(b)) if i not in ones})

        def core_fn(*lv: Any) -> Any:
            it = iter(lv)
            k = next(it) if self.random else None
            new = self._walk(ba.arguments, lambda s, x, p: next(it))
            b2 = inspect.BoundArguments(self._sig, new)  # type: ignore[arg-type]
            if self._jitted is not None and self.random:
                return self._jitted(*b2.args, __bn_key__=k, **b2.kwargs)
            if self._jitted is not None:
                return self._jitted(*b2.args, **b2.kwargs)
            return self._run(b2.args, b2.kwargs, key=k)

        g: Callable[..., Any] = core_fn
        for level in reversed(range(n)):
            in_axes = tuple(0 if level in lv else None for lv in levels)
            if any(a is not None for a in in_axes):
                g = jax.vmap(g, in_axes=in_axes)
            else:
                g = jax.vmap(g, in_axes=in_axes, axis_size=bshape[level])
        return g(*prepared)

    def _batch_plan(
        self, arguments: Any, what: str
    ) -> tuple[list[tuple[Spec, Any, str]], list[tuple[Any, ...]], tuple[Any, ...]]:
        """Split each array argument's shape into batch axes + declared axes."""
        leaves: list[tuple[Spec, Any, str]] = []

        def collect(spec: Spec, v: Any, path: str) -> Any:
            leaves.append((spec, v, path))
            return v

        self._walk(arguments, collect)
        batches: list[tuple[Any, ...]] = []
        for spec, v, path in leaves:
            core = spec.core_rank  # type: ignore[attr-defined]
            if core is None:
                raise TypeError(
                    f"{self.name}.{what}(): `{path}: {spec}` has a variadic shape, so its "
                    f"batch axes cannot be told apart from its own axes."
                )
            if isinstance(spec, KeySpec):
                shape = KeySpec.batch_shape(v) or ()
            else:
                shape = tuple(jnp.shape(v))
            nb = len(shape) - core
            batches.append(shape[:nb] if nb > 0 else ())
        try:
            bshape = tuple(jnp.broadcast_shapes(*batches)) if batches else ()
        except ValueError:
            listing = "\n".join(
                f"    {path}: {describe(v)}  = batch {b} + {spec}"
                for (spec, v, path), b in zip(leaves, batches)
                if b
            )
            raise ShapeError(
                f"{self.name}.{what}(): the batch axes of the arguments do not broadcast.\n\n"
                f"{listing}\n\n    {self.signature}"
            ) from None
        return leaves, batches, bshape

    def batched(self, chunk: int | None = None, *, sizes: tuple[int, ...] | None = None) -> "Batched":
        """A compiled, vectorised version that does not recompile per batch size.

        ``f.batched(chunk=64)`` compiles the lifted function once, for batches
        of 64, and runs any batch size by splitting it into chunks of 64 and
        padding the last one. Because lifted examples are independent, the
        padding cannot affect the results.

        ``sizes=(8, 64, 512)`` compiles one version per listed size and picks
        the largest that fits, which wastes less work on small batches.
        """
        from ._batch import Batched

        if (chunk is None) == (sizes is None):
            raise TypeError("pass exactly one of `chunk` or `sizes`")
        return Batched(self, (chunk,) if chunk is not None else tuple(sizes or ()))

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
        """Prove the declared shapes for all sizes. See :func:`barnacle.prove`."""
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


def _is_one(s: Any) -> bool:
    try:
        return bool(s == 1)
    except Exception:
        return False


def _trace_desc(leaf: Any) -> str:
    d = describe(leaf)
    aval = getattr(leaf, "aval", None)
    if getattr(aval, "weak_type", False):
        d = f"weak {d} (a Python scalar)"
    return d


# ---------------------------------------------------------------- stage hints

_PKG_DIRS = (os.path.dirname(os.path.abspath(jax.__file__)), os.path.dirname(os.path.abspath(__file__)))


def _user_frame(e: BaseException) -> tuple[str, int, str] | None:
    tb = e.__traceback__
    found = None
    while tb is not None:
        fname = tb.tb_frame.f_code.co_filename
        if not os.path.abspath(fname).startswith(_PKG_DIRS):
            found = (fname, tb.tb_lineno, linecache.getline(fname, tb.tb_lineno).strip())
        tb = tb.tb_next
    return found


def _stage_hint(e: Exception, f: Fn[Any, Any]) -> None:
    if not isinstance(e, jax.errors.JAXTypeError) or getattr(e, "_barnacle_hint", False):
        return
    E = jax.errors
    if isinstance(e, E.TracerBoolConversionError):
        need = "a concrete Python bool (for `if`, `while`, `and`/`or`, `assert`)"
        fixes = [
            "to choose between two values:        jnp.where(cond, a, b)",
            "to choose between two computations:  jax.lax.cond(cond, f, g, *operands)",
            "to loop until a condition:           jax.lax.while_loop(cond_fn, body_fn, init)",
        ]
    elif isinstance(e, E.TracerIntegerConversionError):
        need = "a concrete Python int (for `range`, a shape, or an index into a Python list)"
        fixes = [
            "if it is a size, read it from an array's .shape (shapes are always concrete)",
            "to loop a traced number of times:    jax.lax.fori_loop / jax.lax.scan",
        ]
    elif isinstance(e, E.TracerArrayConversionError):
        need = "a concrete NumPy array"
        fixes = ["use jax.numpy (jnp) instead of numpy (np) on traced values"]
    elif isinstance(e, E.NonConcreteBooleanIndexError):
        need = "a concrete boolean mask (the result's shape would depend on the data)"
        fixes = ["keep the shape fixed:  jnp.where(mask, x, fill)  or  x * mask"]
    elif isinstance(e, E.ConcretizationTypeError):
        need = "a concrete value"
        fixes = []
    else:
        return
    e._barnacle_hint = True  # type: ignore[attr-defined]
    frame = _user_frame(e)
    where = f"`{frame[2]}`  ({frame[0]}:{frame[1]})" if frame and frame[2] else "this code"
    traced = [n for n, p in f.params.items() if isinstance(p.spec, (ArraySpec, KeySpec))]
    lines = [
        f"barnacle: {where} needs {need},",
        "    but the value is traced: inside jit/vmap/grad, array arguments have no "
        "values, only shapes.",
    ]
    lines += [f"    - {x}" for x in fixes]
    lines.append(
        "    - if the value is configuration rather than data, declare the parameter "
        "Static[...]"
        + (f" (array parameters of {f.name}: {', '.join(traced)})" if traced else "")
    )
    e.add_note("\n".join(lines))


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
        Give the function an implicit PRNG key: ``barnacle.key()`` inside it
        returns a fresh key on every call.
    ``examples``
        Values the prover should use for ``Static`` or unannotated parameters,
        as ``{"name": [value, ...]}``.
    ``where``
        Constraints on dimensions, e.g. ``where="T >= 2"``.
    ``**dims``
        Derived dimensions, e.g. ``U="T + 1"``.
    """

    def wrap(raw: Callable[..., Any]) -> Fn[Any, Any]:
        if isinstance(raw, Fn):
            raw = raw.raw
        return Fn(
            raw, jit=jit, checked=checked, prove=prove, convert=convert, random=random,
            examples=examples, where=where, defs=dims,
        )  # fmt: skip

    if f is not None:
        return wrap(f)
    return wrap

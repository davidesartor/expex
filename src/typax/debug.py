"""Debugging tools that work on the traced program rather than on guesses.

``find_nan(f, *args)``
    Runs the function one operation at a time and reports the first operation
    that produces a NaN from NaN-free inputs: the source line, the operation,
    which loop iteration / branch it was in, and statistics of its inputs.

``summary(f, *args)``
    Size and shape of the traced program: operation counts by kind and by
    source line, the largest intermediates, and a few lints (unrolled loops,
    large captured constants, accidental 64-bit values).

``timing(f, *args)``
    How long tracing, compiling and running take, separately.

``shapes(f, *args)``
    The function's source annotated with the shape of every variable assigned
    on each line (symbolic sizes when called without arguments).
"""

from __future__ import annotations

import inspect
import linecache
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jax.core import eval_jaxpr

from . import _rng
from ._fn import BoundFn, Fn
from ._jaxpr import iter_eqns, source_key, source_of, sub_jaxprs
from ._spec import describe

__all__ = [
    "find_nan", "shapes", "summary", "timing", "memory",
    "NanReport", "ShapeTrace", "Summary", "Timing", "Memory",
]  # fmt: skip


def _plain(
    f: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
    """The un-jitted callable behind ``f`` and the arguments to trace it with.

    ``Static`` arguments of an ``@fn`` are closed over, so that tracing the
    result does not turn them into traced values.
    """
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


def _trace(f: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, list[Any]]:
    """Trace ``f``; return the closed jaxpr and the flat list of traced inputs."""
    g, args, kwargs = _plain(f, args, kwargs)
    key = _rng.current().next() if _rng.current() else jr.key(0)  # type: ignore[union-attr]

    def run(*a: Any, **k: Any) -> Any:
        with _rng.rng(key):
            return g(*a, **k)

    return jax.make_jaxpr(run)(*args, **kwargs), jax.tree.leaves((args, kwargs))


# ------------------------------------------------------------------ find_nan


def _is_float(x: Any) -> bool:
    try:
        return bool(jnp.issubdtype(x.dtype, jnp.inexact))
    except Exception:
        return False


def _has_nan(xs: Any) -> bool:
    return any(_is_float(x) and bool(jnp.isnan(x).any()) for x in xs)


def _has_inf(xs: Any) -> bool:
    return any(_is_float(x) and bool(jnp.isinf(x).any()) for x in xs)


def _stats(x: Any) -> str:
    d = describe(x)
    if not _is_float(x) or getattr(x, "size", 0) == 0:
        if getattr(x, "size", 0) == 1:
            return f"{d}  value={np.asarray(x).item()}"
        return d
    a = np.asarray(x)
    if np.iscomplexobj(a):
        a = np.abs(a)
    fin = a[np.isfinite(a)]
    parts = []
    if a.size == 1:
        parts.append(f"value={a.item():.6g}")
    elif fin.size:
        parts.append(f"min={fin.min():.6g}  max={fin.max():.6g}")
    counts = [
        (int(np.isnan(a).sum()), "nan"),
        (int(np.isposinf(a).sum()), "+inf"),
        (int(np.isneginf(a).sum()), "-inf"),
        (int((a == 0).sum()), "zero"),
        (int((fin < 0).sum()), "negative"),
    ]
    flags = ", ".join(f"{n} {w}" for n, w in counts if n and a.size > 1)
    if flags:
        parts.append(f"({flags}, of {a.size})")
    return f"{d}  " + "  ".join(parts)


@dataclass
class NanReport:
    """Where a NaN first appeared."""

    kind: str  # "op" or "input"
    op: str
    source: str
    line: str
    path: list[str]
    inputs: list[str]
    outputs: list[str]
    infs: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.kind == "input":
            return "NaN is already present in the inputs:\n" + "\n".join(
                f"    {s}" for s in self.inputs
            )
        out = [f"NaN first appears at {self.source}"]
        if self.line:
            out.append(f"    > {self.line}")
        out.append(f"  operation: {self.op}")
        if self.path:
            out.append(f"  inside:    {' > '.join(self.path)}")
        out.append("  inputs:")
        out += [f"    {s}" for s in self.inputs]
        out.append("  output:")
        out += [f"    {s}" for s in self.outputs]
        if self.infs:
            out.append("  infinities appeared earlier at:")
            out += [f"    {s}" for s in self.infs]
        return "\n".join(out)


class _Found(Exception):
    def __init__(self, report: NanReport):
        self.report = report


def _line_of(source: str) -> str:
    try:
        fname, rest = source.rsplit(":", 1)
        return linecache.getline(fname, int(rest.split()[0])).strip()
    except Exception:
        return ""


def _bind(eqn: Any, invals: list[Any]) -> list[Any]:
    r = eqn.primitive.get_bind_params(eqn.params)
    # older JAX returns (subfuns, params); newer returns the params alone
    subfuns: Any
    params: Any
    subfuns, params = r if isinstance(r, tuple) and len(r) == 2 else ((), r)
    ctx = getattr(getattr(eqn, "ctx", None), "manager", None)
    if ctx is not None:
        with ctx:
            out = eqn.primitive.bind(*subfuns, *invals, **params)
    else:
        out = eqn.primitive.bind(*subfuns, *invals, **params)
    return list(out) if eqn.primitive.multiple_results else [out]


class _Interp:
    def __init__(self) -> None:
        self.infs: list[str] = []

    def run(self, jaxpr: Any, args: list[Any], path: list[str]) -> list[Any]:
        """Evaluate one jaxpr op by op; raise ``_Found`` at the first NaN-producing op."""
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        env: dict[Any, Any] = {}

        def read(v: Any) -> Any:
            return v.val if hasattr(v, "val") else env[v]

        for v, a in zip(_invars(jaxpr), args):
            env[v] = a
        for eqn in jaxpr.eqns:
            invals = [read(v) for v in eqn.invars]
            outs = _bind(eqn, invals)
            if _has_nan(outs) and not _has_nan(invals):
                self.dive(eqn, invals, path)  # raises if it can localise further
                raise _Found(self.report(eqn, invals, outs, path))
            if len(self.infs) < 3 and _has_inf(outs) and not _has_inf(invals):
                where = source_of(eqn)
                self.infs.append(f"{where}  ({eqn.primitive.name})  > {_line_of(where)}")
            for v, o in zip(eqn.outvars, outs):
                env[v] = o
        return [read(v) for v in jaxpr.outvars]

    def report(self, eqn: Any, invals: list[Any], outs: list[Any], path: list[str]) -> NanReport:
        where = source_of(eqn)
        return NanReport(
            "op", eqn.primitive.name, where, _line_of(where), list(path),
            [f"#{i}: {_stats(x)}" for i, x in enumerate(invals)],
            [_stats(x) for x in outs],
            list(self.infs),
        )  # fmt: skip

    def dive(self, eqn: Any, invals: list[Any], path: list[str]) -> None:
        """Look inside control flow / nested calls for the offending operation."""
        name = eqn.primitive.name
        p = eqn.params
        try:
            if name == "cond":
                idx = int(invals[0])
                self.run(p["branches"][idx], invals[1:], path + [f"cond branch {idx}"])
            elif name == "while":
                self.dive_while(p, invals, path)
            elif name == "scan":
                self.dive_scan(p, invals, path)
            else:
                subs = list(sub_jaxprs(p))
                if len(subs) == 1:
                    sub = subs[0][1]
                    if len(_invars(sub)) == len(invals):
                        label = p.get("name") if name in ("jit", "pjit") else name
                        self.run(sub, invals, path + [f"{label}"] if label else path)
        except _Found:
            raise
        except Exception:
            return  # unknown structure: report the enclosing operation instead

    def dive_while(self, p: dict[str, Any], invals: list[Any], path: list[str]) -> None:
        cn, tx = p["cond_nconsts"], p["body_nconsts"]
        cconsts, bconsts, carry = invals[:cn], invals[cn : cn + tx], invals[cn + tx :]
        cond_j, body_j = p["cond_jaxpr"], p["body_jaxpr"]
        body = jax.jit(lambda c, x: list(eval_jaxpr(_jp(body_j), [], *c, *x)))
        cond = jax.jit(lambda c, x: list(eval_jaxpr(_jp(cond_j), [], *c, *x))[0])
        it = 0
        while bool(cond(cconsts, carry)):
            new = body(bconsts, carry)
            if _has_nan(new) and not _has_nan(carry):
                self.run(body_j, [*bconsts, *carry], path + [f"while iteration {it}"])
            carry = new
            it += 1

    def dive_scan(self, p: dict[str, Any], invals: list[Any], path: list[str]) -> None:
        body_j = p["jaxpr"]
        length, reverse = p["length"], p.get("reverse", False)
        if "num_consts" in p:  # older JAX
            nc, ncar = p["num_consts"], p["num_carry"]
        else:  # newer JAX: a flat-tree of (consts, carry, xs)
            parts = p["ft_in"].unpack()
            nc, ncar = len(parts[0]), len(parts[1])
            if nc + ncar + len(parts[2]) != len(invals):
                return
        consts, carry, xs = invals[:nc], invals[nc : nc + ncar], invals[nc + ncar :]
        if reverse:
            xs = [x[::-1] for x in xs]

        def step(c: Any, x: Any) -> Any:
            out = list(eval_jaxpr(_jp(body_j), [], *consts, *c, *x))
            bad = jnp.zeros((), bool)
            for o in out:
                if _is_float(o):
                    bad = bad | jnp.isnan(o).any()
            return tuple(out[:ncar]), bad

        _, flags = lax.scan(step, tuple(carry), tuple(xs), length=length)
        flags = np.asarray(flags)
        if not flags.any():
            return
        first = int(flags.argmax())
        if first:
            done, _ = lax.scan(step, tuple(carry), tuple(x[:first] for x in xs), length=first)
            carry = list(done)
        it = length - 1 - first if reverse else first
        self.run(body_j, [*consts, *carry, *[x[first] for x in xs]], path + [f"scan iteration {it}"])


def _invars(jaxpr: Any) -> list[Any]:
    if hasattr(jaxpr, "all_invars"):
        return list(jaxpr.all_invars)
    return list(getattr(jaxpr, "constvars", ())) + list(jaxpr.invars)


def _jp(j: Any) -> Any:
    return getattr(j, "jaxpr", j)


def find_nan(f: Callable[..., Any], *args: Any, **kwargs: Any) -> NanReport | None:
    """Find the first operation that turns NaN-free values into NaN.

    Returns ``None`` if the function's computation never produces a NaN for
    these arguments. Works on any JAX function, including ``jax.grad(f)``.
    """
    closed, flat = _trace(f, args, kwargs)
    bad = [f"argument leaf #{i}: {_stats(x)}" for i, x in enumerate(flat) if _has_nan([x])]
    if bad:
        return NanReport("input", "", "", "", [], bad, [])
    interp = _Interp()
    try:
        interp.run(closed.jaxpr, [*closed.consts, *flat], [])
    except _Found as e:
        return e.report
    return None


# ------------------------------------------------------------------- summary


def _nbytes(aval: Any) -> int:
    try:
        return int(np.prod(aval.shape, dtype=np.int64)) * np.dtype(aval.dtype).itemsize
    except Exception:
        return 0


def _is_wide(dtype: Any) -> bool:
    """A 64-bit numeric dtype (False for keys and other non-NumPy dtypes)."""
    try:
        dt = np.dtype(dtype)
    except TypeError:
        return False
    return dt.kind in "fiuc" and dt.itemsize >= 8


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} kB"


@dataclass
class Summary:
    """Statistics of a traced program. ``str()`` gives a readable report."""

    ops: int
    top_level_ops: int
    by_op: list[tuple[str, int]]
    by_line: list[tuple[str, int, str]]
    largest: list[tuple[int, str, str]]
    constants: list[tuple[int, str]]
    warnings: list[str]

    def __str__(self) -> str:
        out = [f"{self.ops} operations ({self.top_level_ops} at top level)"]
        out.append("  by kind:  " + ", ".join(f"{n} {k}" for k, n in self.by_op[:8]))
        out.append("  by source line:")
        for where, n, text in self.by_line[:6]:
            out.append(f"    {n:6d}  {where}  {text[:70]}")
        if self.largest:
            out.append("  largest intermediates:")
            for size, desc, where in self.largest[:4]:
                out.append(f"    {_mb(size):>9}  {desc}  {where}")
        for w in self.warnings:
            out.append(f"  ! {w}")
        return "\n".join(out)


def summary(f: Callable[..., Any], *args: Any, **kwargs: Any) -> Summary:
    """Describe the program JAX traces for ``f(*args, **kwargs)``."""
    closed, _ = _trace(f, args, kwargs)
    top = closed.jaxpr.eqns
    by_op: Counter[str] = Counter()
    by_line: Counter[tuple[str, int, str]] = Counter()
    largest: list[tuple[int, str, str]] = []
    wide = None
    total = 0
    in_wide = any(_is_wide(getattr(v.aval, "dtype", None)) for v in closed.jaxpr.invars)
    for eqn in iter_eqns(closed.jaxpr):
        total += 1
        by_op[eqn.primitive.name] += 1
        by_line[source_key(eqn)] += 1
        for v in eqn.outvars:
            aval = getattr(v, "aval", None)
            if aval is None or not hasattr(aval, "dtype"):
                continue
            n = _nbytes(aval)
            if n >= 1024:
                largest.append((n, str(aval), source_of(eqn)))
            try:
                dt = np.dtype(aval.dtype)
            except TypeError:
                continue
            if wide is None and not in_wide and dt.kind in "fc" and dt.itemsize >= 8 and dt.kind == "f":
                wide = source_of(eqn)
    largest.sort(key=lambda t: -t[0])
    lines = [
        (f"{fn}:{ln}", n, linecache.getline(fn, ln).strip()) for (fn, ln, _), n in by_line.most_common()
    ]
    consts = sorted(
        ((_nbytes(c), describe(c)) for c in closed.consts if hasattr(c, "shape")), reverse=True
    )

    warnings: list[str] = []
    if lines and lines[0][1] >= 50 and lines[0][1] >= 0.2 * total:
        warnings.append(
            f"{lines[0][1]} of {total} operations come from one line ({lines[0][0]}): "
            f"probably a Python loop unrolled during tracing. A loop over an array or "
            f"range inside @fn is turned into lax.scan automatically; check `f.flow`."
        )
    if total >= 5000:
        warnings.append(
            f"the program has {total} operations; compile time grows with program size."
        )
    big = [c for c in consts if c[0] >= 1_000_000]
    if big:
        warnings.append(
            f"{len(big)} array(s) are captured as constants, the largest {big[0][1]} "
            f"({_mb(big[0][0])}). Constants are embedded in the compiled program; pass large "
            f"arrays as arguments instead."
        )
    if wide:
        warnings.append(f"64-bit floats appear although no input is 64-bit, first at {wide}.")
    return Summary(total, len(top), by_op.most_common(), lines, largest[:8], consts[:8], warnings)


# -------------------------------------------------------------------- timing


@dataclass
class Timing:
    """Seconds spent in each stage. ``str()`` gives a one-paragraph report."""

    trace: float
    compile: float
    first_run: float
    run: float
    ops: int

    def __str__(self) -> str:
        def ms(s: float) -> str:
            if s < 1e-3:
                return f"{s * 1e6:.0f} us"
            return f"{s * 1e3:.1f} ms" if s < 1 else f"{s:.2f} s"

        out = (
            f"trace {ms(self.trace)} | compile {ms(self.compile)} | "
            f"run {ms(self.run)} (first run {ms(self.first_run)}) | {self.ops} operations"
        )
        setup = self.trace + self.compile
        if self.run > 0 and setup > 10 * self.run:
            out += (
                f"\n  setup costs as much as {setup / self.run:,.0f} runs: avoid retracing "
                f"(see `.traces` on a jit=True @fn) and keep the program small."
            )
        return out


def timing(f: Callable[..., Any], *args: Any, repeat: int = 5, **kwargs: Any) -> Timing:
    """Time tracing, compilation and execution of ``jit(f)(*args, **kwargs)`` separately."""
    g, args, kwargs = _plain(f, args, kwargs)
    key = jr.key(0)

    def run(key: Any, *a: Any, **k: Any) -> Any:
        with _rng.rng(key):
            return g(*a, **k)

    jf = jax.jit(run)
    t0 = time.perf_counter()
    traced = jf.trace(key, *args, **kwargs)
    t1 = time.perf_counter()
    compiled = traced.lower().compile()
    t2 = time.perf_counter()
    jax.block_until_ready(compiled(key, *args, **kwargs))
    t3 = time.perf_counter()
    runs = []
    for _ in range(max(1, repeat)):
        s = time.perf_counter()
        jax.block_until_ready(compiled(key, *args, **kwargs))
        runs.append(time.perf_counter() - s)
    ops = sum(1 for _ in iter_eqns(traced.jaxpr))
    return Timing(t1 - t0, t2 - t1, t3 - t2, statistics.median(runs), ops)


# -------------------------------------------------------------------- shapes


@dataclass
class ShapeTrace:
    """The source of a function annotated with the shapes assigned on each line."""

    title: str
    lines: list[tuple[int, str, list[str]]]  # (line number, source, annotations)

    def __str__(self) -> str:
        width = max((len(src.rstrip()) for _, src, notes in self.lines if notes), default=0)
        width = min(max(width, 20), 72)
        out = [self.title]
        for n, src, notes in self.lines:
            text = src.rstrip()
            if notes:
                out.append(f"{n:5d} | {text:<{width}}   # {', '.join(notes)}")
            else:
                out.append(f"{n:5d} | {text}")
        return "\n".join(out)

    __repr__ = __str__


def _shape_of(v: Any) -> str | None:
    """A short description of an array-carrying value, or None for anything else."""
    from ._spec import show

    if isinstance(v, (jax.Array, np.ndarray, jax.ShapeDtypeStruct)) or hasattr(v, "aval"):
        try:
            return describe(v)
        except Exception:
            return None
    if callable(v) or isinstance(v, (type, str, bytes)) or v is None:
        return None
    try:
        leaves = jax.tree.leaves(v)
    except Exception:
        return None
    if leaves and any(hasattr(x, "shape") and hasattr(x, "dtype") for x in leaves):
        text = show(v)
        return text if len(text) <= 80 else text[:77] + "..."
    return None


def shapes(f: Callable[..., Any], *args: Any, **kwargs: Any) -> ShapeTrace:
    """Show the shape and dtype of every variable assigned on each line of ``f``.

    With arguments, ``f`` is traced abstractly on them (nothing is computed).
    Without arguments, an ``@fn`` is traced with symbolic sizes from its
    annotations, so the shapes read ``f32[B, T - 1]`` and hold for all sizes.
    Loop bodies show the shapes of one iteration.
    """
    import sys

    from . import _check

    target = f.fn if isinstance(f, BoundFn) else f
    raw: Any = getattr(target, "raw", target)
    raw = getattr(raw, "__func__", raw)
    try:
        src_lines, start = inspect.getsourcelines(raw)
        code = raw.__code__
    except (OSError, TypeError, AttributeError):
        raise TypeError("shapes() needs a Python function whose source is available") from None
    fname, end = code.co_filename, start + len(src_lines)
    seen: dict[int, dict[str, list[str]]] = {}
    frames: dict[int, tuple[int, dict[str, int]]] = {}

    def note(line: int, name: str, desc: str) -> None:
        bucket = seen.setdefault(line, {}).setdefault(name, [])
        if desc not in bucket:
            bucket.append(desc)

    def watched(frame: Any) -> bool:
        c = frame.f_code
        return c.co_filename == fname and start <= c.co_firstlineno < end

    def visible(name: str) -> bool:
        return not name.startswith("__") and name != "self"

    def changes(frame: Any, line: int, snap: dict[str, int]) -> dict[str, int]:
        now = {k: id(v) for k, v in frame.f_locals.items() if visible(k)}
        for k, ident in now.items():
            if snap.get(k) != ident:
                desc = _shape_of(frame.f_locals[k])
                if desc is not None:
                    note(line, k, desc)
        return now

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if not watched(frame):
            return None
        key = id(frame)
        if event == "call":
            frames[key] = (frame.f_lineno, {k: id(v) for k, v in frame.f_locals.items() if visible(k)})
        elif event == "line":
            last, snap = frames.get(key, (frame.f_lineno, {}))
            frames[key] = (frame.f_lineno, changes(frame, last, snap))
        elif event == "return":
            last, snap = frames.pop(key, (frame.f_lineno, {}))
            changes(frame, last, snap)
            if frame.f_code is code or frame.f_code.co_name == raw.__name__:
                desc = _shape_of(arg)
                if desc is not None:
                    note(last, "->", desc)
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        if args or kwargs or not isinstance(target, Fn):
            g, a, k = _plain(f, args, kwargs)
            with _rng.rng(jr.key(0)):
                jax.eval_shape(lambda *x, **y: g(*x, **y), *a, **k)
            title = f"{getattr(raw, '__qualname__', raw.__name__)} at {fname}:{start}"
        else:
            title = _symbolic_run(target, _check)
    finally:
        sys.settrace(old)

    rows = []
    for i, text in enumerate(src_lines):
        n = start + i
        notes = [
            (f"-> {' | '.join(d)}" if name == "->" else f"{name}: {' | '.join(d)}")
            for name, d in seen.get(n, {}).items()
        ]
        rows.append((n, text, notes))
    return ShapeTrace(title, rows)


def _symbolic_run(f: Fn[Any, Any], check: Any) -> str:
    """Trace ``f`` once on symbolic inputs (the prover's first case)."""
    from jax import export

    if f._lazy:
        f._resolve_owner()
    choices = check._choices(f, f.examples)
    abstract = [n for n, c in choices.items() if c == [check._ABSTRACT]]
    fixed = {n: c[0] for n, c in choices.items() if n not in abstract}
    statics = check._statics(f, fixed)
    constraints, _ = check._jax_constraints(f, statics)
    var_rank = {n[1:]: 1 for n in f.input_names if n.startswith("*")}
    ctx = check._Ctx(export.SymbolicScope(constraints), None, var_rank, statics)
    check._trace(f, fixed, abstract, ctx)
    sizes = ", ".join(sorted(n for n in f.input_names if not n.startswith("*")))
    label = ", ".join(f"{k}={v!r}" for k, v in fixed.items())
    return f"{f.signature}   [symbolic {sizes}{'; ' + label if label else ''}]"


# -------------------------------------------------------------------- memory


@dataclass
class Memory:
    """Memory use of a compiled function. ``str()`` gives a readable report."""

    arguments: int
    outputs: int
    temporaries: int
    live_peak: int
    live_at_peak: list[tuple[int, str, str]]  # (bytes, value, created at)
    peak_after: str

    @property
    def total(self) -> int:
        """XLA's figure: arguments + outputs + temporaries."""
        return self.arguments + self.outputs + self.temporaries

    def __str__(self) -> str:
        out = [
            f"compiled: {_mb(self.total)} = arguments {_mb(self.arguments)} + outputs "
            f"{_mb(self.outputs)} + temporaries {_mb(self.temporaries)}",
            f"before optimisation, at most {_mb(self.live_peak)} is alive at once, after "
            f"{self.peak_after}; the largest values then:",
        ]
        for size, desc, where in self.live_at_peak[:6]:
            out.append(f"    {_mb(size):>9}  {desc:<24} {where}")
        return "\n".join(out)

    __repr__ = __str__


def _aval_bytes(v: Any) -> int:
    aval = getattr(v, "aval", None)
    return _nbytes(aval) if aval is not None and hasattr(aval, "dtype") else 0


def memory(f: Callable[..., Any], *args: Any, **kwargs: Any) -> Memory:
    """Estimate how much memory ``jit(f)(*args)`` needs, and what holds it.

    Reports XLA's own numbers for the compiled program, and a simple liveness
    analysis of the traced program (before XLA's optimisations): the largest
    amount of data alive at once and which values, created on which source
    lines, make it up. Pass ``jax.grad(f)`` to see the cost of a backward pass;
    ``@fn(remat=...)`` reduces it.
    """
    g, args, kwargs = _plain(f, args, kwargs)
    key = jr.key(0)

    def run(key: Any, *a: Any, **k: Any) -> Any:
        with _rng.rng(key):
            return g(*a, **k)

    traced = jax.jit(run).trace(key, *args, **kwargs)
    stats = traced.lower().compile().memory_analysis()
    arg_b = int(getattr(stats, "argument_size_in_bytes", 0) or 0)
    out_b = int(getattr(stats, "output_size_in_bytes", 0) or 0)
    tmp_b = int(getattr(stats, "temp_size_in_bytes", 0) or 0)

    jaxpr = traced.jaxpr.jaxpr
    last_use: dict[Any, int] = {}
    for i, eqn in enumerate(jaxpr.eqns):
        for v in eqn.invars:
            if not hasattr(v, "val"):
                last_use[v] = i
    for v in jaxpr.outvars:
        if not hasattr(v, "val"):
            last_use[v] = len(jaxpr.eqns)
    origin: dict[Any, str] = {v: "an input" for v in list(jaxpr.invars) + list(jaxpr.constvars)}
    live = {v: _aval_bytes(v) for v in origin if v in last_use}
    peak, peak_set, peak_after = sum(live.values()), dict(live), "the start"
    for i, eqn in enumerate(jaxpr.eqns):
        for v in eqn.outvars:
            if v in last_use:
                live[v] = _aval_bytes(v)
                origin[v] = source_of(eqn)
        total = sum(live.values())
        if total > peak:
            peak, peak_set = total, dict(live)
            peak_after = f"{eqn.primitive.name} at {source_of(eqn)}"
        for v in eqn.invars:
            if not hasattr(v, "val") and last_use.get(v) == i:
                live.pop(v, None)
    largest = sorted(
        ((b, str(v.aval), origin.get(v, "?")) for v, b in peak_set.items() if b),
        key=lambda t: -t[0],
    )
    return Memory(arg_b, out_b, tmp_b, peak, largest, peak_after)

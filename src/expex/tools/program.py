"""``expex.debug.ops``, ``timing`` and ``memory``: the traced and compiled program of a call.

``ops`` counts the operations of the jaxpr by kind and source line and flags
unrolled loops, large captured constants and accidental 64-bit values;
``timing`` times tracing, compiling and running separately; ``memory`` gives
XLA's memory figures and a liveness analysis of the jaxpr.
"""

import linecache
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.random as jr
import numpy as np

from ..annotations.dims import describe
from .trace import iter_eqns, keyed, make_jaxpr, source_key, source_of


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
class Ops:
    """The operations of a traced program. ``str()`` gives a readable report."""

    count: int
    top_level_ops: int
    by_op: list[tuple[str, int]]
    by_line: list[tuple[str, int, str]]
    largest: list[tuple[int, str, str]]
    constants: list[tuple[int, str]]
    warnings: list[str]

    def __str__(self) -> str:
        out = [f"{self.count} operations ({self.top_level_ops} at top level)"]
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


def ops(f: Callable[..., Any], *args: Any, **kwargs: Any) -> Ops:
    """The operations JAX traces for ``f(*args, **kwargs)``: by kind and source line.

    Flags unrolled loops and large constants baked into the program.
    """
    closed, _ = make_jaxpr(f, args, kwargs)
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
    return Ops(total, len(top), by_op.most_common(), lines, largest[:8], consts[:8], warnings)


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
                f"(JAX_EXPLAIN_CACHE_MISSES=1 says why one happens) and keep the program small."
            )
        return out


def timing(f: Callable[..., Any], *args: Any, repeat: int = 5, **kwargs: Any) -> Timing:
    """Time tracing, compilation and execution of ``jit(f)(*args, **kwargs)`` separately."""
    run, args, kwargs = keyed(f, args, kwargs)
    key = jr.key(0)
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
    ``jax.checkpoint`` on a layer reduces it.
    """
    run, args, kwargs = keyed(f, args, kwargs)
    key = jr.key(0)

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

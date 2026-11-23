"""Timing and environment helpers shared by every benchmark in benchmarks/ports/.

A port's benchmark file builds, for one config, a list of `Measure`s: the same
computation as the original runs it and as the port runs it, on the same
parameters and inputs. `run_measure` then times both the same way:

* compile: `jit(f).lower(*args).compile()`, timed as tracing + lowering ("trace")
  and XLA compilation ("xla"), from cold caches (`jax.clear_caches()`), repeated
  `compile_repeats` times; the runs then call the compiled executable. A
  callable that is not a single `jax.jit` (the original Llama's two-stage
  generation) is called as is, and its compile time is estimated as first call
  minus the median steady-state call.
* run: one untimed warm-up call each, then `rounds` interleaved rounds (the
  order alternates every round), each sample `inner` calls ending in
  `jax.block_until_ready`. `inner` is calibrated so that a sample lasts at least
  `min_sample_s` (it is 1 for anything slower). Best, median, quartiles and
  every sample are kept.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Callable

import jax

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
EXAMPLES = ROOT / "examples"


# ----------------------------------------------------------------- loading


def use_original(name: str) -> pathlib.Path:
    """Put the original `name` (from originals.toml) on sys.path; fail clearly if it is not fetched."""
    sys.path.insert(0, str(HERE))
    import fetch

    pin = fetch.pins()[name]
    if fetch.head(fetch.location(name)) != pin["commit"]:
        raise SystemExit(f"original {name!r} is not fetched at {pin['commit'][:9]}: run python benchmarks/fetch.py")
    path = fetch.import_path(name)
    sys.path.insert(0, str(path))
    return path


def use_example(port: str) -> pathlib.Path:
    """Put examples/<port>/ on sys.path, so its model and compare.py import as in its own README."""
    path = EXAMPLES / port
    sys.path.insert(0, str(path))
    return path


def load_module(name: str, path: pathlib.Path):
    """Import a file under a unique module name (several examples have a `compare.py`)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- measures


@dataclasses.dataclass
class Fn:
    """A callable with its arguments. If `call` is a `jax.jit` function, its compile time is measured exactly."""

    call: Callable[..., Any]
    args: tuple = ()


@dataclasses.dataclass
class Measure:
    """One quantity timed for the original and the port, e.g. "forward" or "rollout"."""

    name: str
    original: Fn | None
    port: Fn
    items: float = 1.0  # units of work per call, for the throughput column
    unit: str = "calls"  # tokens, steps, games, ...
    check: Callable[[Any, Any], dict] | None = None  # (original output, port output) -> agreement numbers
    note: str = ""
    labels: tuple[str, str] = ("original", "port")
    compile: bool = True  # False: call `call` itself, never its AOT executable (to time its dispatch)


@dataclasses.dataclass
class Settings:
    rounds: int = 15
    min_sample_s: float = 0.05
    compile_repeats: int = 3
    budget_s: float = 240.0  # per measure: stop adding rounds after this (at least 3 rounds)

    @classmethod
    def quick(cls) -> Settings:
        return cls(rounds=5, min_sample_s=0.02, compile_repeats=1, budget_s=20.0)


def _block(x):
    return jax.block_until_ready(x)


def _sample(f: Fn, inner: int) -> float:
    t = time.perf_counter()
    out = None
    for _ in range(inner):
        out = f.call(*f.args)
    _block(out)
    return (time.perf_counter() - t) / inner


def _compile(f: Fn) -> tuple[dict, Any] | None:
    """Trace + lower and XLA compile times from cold caches, and the executable; None if `f.call` is not a jit."""
    if not hasattr(f.call, "lower"):
        return None
    jax.clear_caches()
    t0 = time.perf_counter()
    lowered = f.call.lower(*f.args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    return {"trace": t1 - t0, "xla": t2 - t1, "total": t2 - t0}, compiled


def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    q = statistics.quantiles(s, n=4, method="inclusive") if len(s) >= 2 else [s[0]] * 3
    med = statistics.median(s)
    return {
        "best": s[0],
        "median": med,
        "p25": q[0],
        "p75": q[2],
        "worst": s[-1],
        "iqr_pct": 100 * (q[2] - q[0]) / med if med else 0.0,
        "n": len(s),
        "samples": s,
    }


def run_measure(m: Measure, settings: Settings, log=print) -> dict:
    impls = {m.labels[0]: m.original, m.labels[1]: m.port}
    impls = {k: v for k, v in impls.items() if v is not None}
    out: dict[str, Any] = {"name": m.name, "items": m.items, "unit": m.unit, "note": m.note, "impls": {}}

    # compile, from cold caches, interleaved
    compiles: dict[str, list[dict]] = {k: [] for k in impls}
    for r in range(settings.compile_repeats if m.compile else 0):
        order = list(impls) if r % 2 == 0 else list(reversed(impls))
        for label in order:
            c = _compile(impls[label])
            if c is not None:
                compiles[label].append(c[0])
                impls[label] = Fn(c[1], impls[label].args)  # run the executable just compiled

    # warm up, record the first call
    first: dict[str, float] = {}
    outputs: dict[str, Any] = {}
    for label, f in impls.items():
        t = time.perf_counter()
        outputs[label] = _block(f.call(*f.args))
        first[label] = time.perf_counter() - t
    if m.check is not None and len(impls) == 2:
        try:
            out["check"] = m.check(outputs[m.labels[0]], outputs[m.labels[1]])
        except Exception as e:  # a failed check is reported, not fatal
            out["check"] = {"error": f"{type(e).__name__}: {e}"}
    del outputs

    # calibrate the inner loop
    inner = {}
    for label, f in impls.items():
        t = _sample(f, 1)
        inner[label] = max(1, min(100_000, math.ceil(settings.min_sample_s / max(t, 1e-9))))

    samples: dict[str, list[float]] = {k: [] for k in impls}
    start = time.perf_counter()
    for r in range(settings.rounds):
        order = list(impls) if r % 2 == 0 else list(reversed(impls))
        for label in order:
            samples[label].append(_sample(impls[label], inner[label]))
        if r >= 2 and time.perf_counter() - start > settings.budget_s:
            break

    for label in impls:
        stats = summarize(samples[label])
        entry: dict[str, Any] = {"run": stats, "inner": inner[label], "first_call_s": first[label]}
        if compiles[label]:
            totals = [c["total"] for c in compiles[label]]
            entry["compile"] = {
                "method": "aot",
                "cold_s": totals[0],
                "best_s": min(totals),
                "trace_s": min(c["trace"] for c in compiles[label]),
                "xla_s": min(c["xla"] for c in compiles[label]),
                "all_s": totals,
            }
        elif m.compile:
            entry["compile"] = {"method": "first call - median", "cold_s": first[label] - stats["median"]}
        entry["throughput"] = m.items / stats["best"]
        entry["throughput_median"] = m.items / stats["median"]
        out["impls"][label] = entry

    if len(impls) == 2:
        a, b = (out["impls"][k]["run"] for k in m.labels)
        out["speedup"] = {
            "best": a["best"] / b["best"],
            "median": a["median"] / b["median"],
            # the port's p25..p75 against the original's: a conservative range
            "low": a["p25"] / b["p75"],
            "high": a["p75"] / b["p25"],
        }
    line = f"  {m.name:22s}"
    for label in impls:
        e = out["impls"][label]
        line += f" {label} {fmt_time(e['run']['best'])} (med {fmt_time(e['run']['median'])}, IQR {e['run']['iqr_pct']:.0f}%)"
    if "speedup" in out:
        line += f"  -> {out['speedup']['best']:.2f}x best, {out['speedup']['median']:.2f}x median"
    log(line)
    return out


def fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:.1f} us"
    if s < 1:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"


# ----------------------------------------------------------------- environment


def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:  # macOS
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor() or "unknown"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment() -> dict:
    import jaxlib
    import numpy

    devices = jax.devices()
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    status = _git("status", "--porcelain", "--", "src")
    sys.path.insert(0, str(HERE))
    import fetch

    originals = {name: {"url": p["url"], "commit": p["commit"], "fetched": fetch.head(fetch.location(name)) == p["commit"]}
                 for name, p in fetch.pins().items()}  # fmt: skip
    gpu = None
    if devices[0].platform == "gpu":
        try:
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()  # fmt: skip
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": numpy.__version__,
        "backend": jax.default_backend(),
        "devices": [f"{d.platform}:{d.device_kind}" for d in devices],
        "device_kind": devices[0].device_kind,
        "nvidia_smi": gpu,
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "expex_commit": _git("rev-parse", "HEAD"),
        "expex_src_dirty": bool(status),
        "originals": originals,
    }

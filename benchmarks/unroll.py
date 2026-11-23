"""Compile and run time of each port under several `expex.config(unroll_layers=...)` settings, in one process.

    python benchmarks/unroll.py                              # every port with a list of layers, 0 / 4 / 16
    python benchmarks/unroll.py --only vit,llama --values 0,4,8,16
    python benchmarks/unroll.py --out results/unroll-cpu.json

`run.py` runs one setting per process, so on a shared machine the load changes
between the settings being compared. Here every setting of a (port, config) is
compiled in the same process, and their timed rounds are interleaved with the
original's (the order rotates every round): what the load does to one, it does
to all. Reported per setting: trace and XLA compile time (best of
`--repeats`, from cold caches), best and median run time, and the speedup
against the original (best and median of the per-round ratios).

The setting is read when a loop over a list of layers is traced, so each
setting is lowered and compiled inside `expex.config(unroll_layers=...)`.
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: Ports whose models loop over a list of identical layers (the others are unaffected by the setting).
LAYERED = ["llama", "mamba", "nerf", "ppo", "vit"]


def compile_once(fn, settings: dict) -> tuple[dict, object]:
    """Lower and compile from cold caches: wall and process CPU seconds of each phase, and the executable."""
    import jax

    import expex

    jax.clear_caches()
    with expex.config(**settings):
        t0, c0 = time.perf_counter(), time.process_time()
        lowered = fn.call.lower(*fn.args)
        t1, c1 = time.perf_counter(), time.process_time()
        compiled = lowered.compile()
        t2, c2 = time.perf_counter(), time.process_time()
    return {"trace": t1 - t0, "xla": t2 - t1, "trace_cpu": c1 - c0, "xla_cpu": c2 - c1}, compiled


def summarize_compiles(times: list[dict]) -> dict:
    return {
        "trace_s": min(t["trace"] for t in times),
        "xla_s": min(t["xla"] for t in times),
        "best_s": min(t["trace"] + t["xla"] for t in times),
        "trace_cpu_s": min(t["trace_cpu"] for t in times),
        "xla_cpu_s": min(t["xla_cpu"] for t in times),
        "all": times,
    }


def time_call(call, args, inner: int) -> float:
    import jax

    t = time.perf_counter()
    out = None
    for _ in range(inner):
        out = call(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t) / inner


def measure(m, values: list[str], rounds: int, repeats: int, min_sample_s: float, budget_s: float, log) -> dict:
    """One measure: the original and each setting, compiled, then timed in interleaved rounds."""
    import jax

    variants: dict[str, dict] = {}
    calls: dict[str, tuple] = {}
    todo: dict[str, tuple] = {}
    if m.original is not None and hasattr(m.original.call, "lower"):
        todo["original"] = (m.original, {}, "")
    for v in values:
        value, _, mode = v.partition("@")  # EXPLORATION ONLY
        todo[f"unroll_layers={v}"] = (m.port, {"unroll_layers": int(value)}, mode)
    times: dict[str, list[dict]] = {k: [] for k in todo}
    for r in range(repeats):  # compiles interleaved too, the order rotating
        order = list(todo)[r % len(todo):] + list(todo)[: r % len(todo)]
        for label in order:
            fn, settings, mode = todo[label]
            from expex.staging import loops
            loops.VARIANT["mode"] = mode or "scan"
            t, exe = compile_once(fn, settings)
            times[label].append(t)
            calls[label] = (exe, fn.args)
    for label in todo:
        variants[label] = {"compile": summarize_compiles(times[label])}
    for label, (exe, args) in calls.items():
        jax.block_until_ready(exe(*args))  # warm up
    inner = {}
    for label, (exe, args) in calls.items():
        t = time_call(exe, args, 1)
        inner[label] = max(1, min(10_000, math.ceil(min_sample_s / max(t, 1e-9))))
    samples: dict[str, list[float]] = {k: [] for k in calls}
    labels = list(calls)
    start = time.perf_counter()
    for r in range(rounds):
        order = labels[r % len(labels):] + labels[: r % len(labels)]
        if r % 2:
            order.reverse()
        for label in order:
            exe, args = calls[label]
            samples[label].append(time_call(exe, args, inner[label]))
        if r >= 2 and time.perf_counter() - start > budget_s:
            break
    for label, s in samples.items():
        variants[label]["run"] = {"best": min(s), "median": statistics.median(s), "samples": s}
        if "original" in samples and label != "original":
            ratios = [o / p for o, p in zip(samples["original"], s)]
            variants[label]["speedup"] = {
                "best": min(samples["original"]) / min(s),
                "median_of_rounds": statistics.median(ratios),
            }
    line = f"  {m.name:14s}"
    for label, e in variants.items():
        c, r = e["compile"], e["run"]
        sp = e.get("speedup")
        line += (f"\n    {label:18s} compile {c['best_s']:.2f} s (trace {c['trace_s']:.2f}, xla {c['xla_s']:.2f};"
                 f" cpu {c['trace_cpu_s']:.2f} + {c['xla_cpu_s']:.2f})"
                 f"  run best {r['best'] * 1e3:.2f} ms, median {r['median'] * 1e3:.2f} ms"
                 + (f"  speedup {sp['best']:.2f}x best, {sp['median_of_rounds']:.2f}x median of rounds" if sp else ""))  # fmt: skip
    log(line)
    return {"name": m.name, "note": m.note, "variants": variants}


def worker(port: str, config: str, values: list[str], rounds: int, repeats: int, out: str) -> None:
    import importlib

    import harness

    module = importlib.import_module(f"ports.{port}")
    cfg = module.CONFIGS[config]
    print(f"[{port}/{config}] {cfg}", flush=True)
    env = harness.environment()
    results = []
    for m in module.setup(dict(cfg)):
        if not m.compile:
            continue
        results.append(measure(m, values, rounds, repeats, 0.05, 120.0, lambda s: print(s, flush=True)))
    record = {"port": port, "config": config, "params": cfg, "values": values, "environment": env,
              "load_average_after": os.getloadavg(), "measures": results}  # fmt: skip
    pathlib.Path(out).write_text(json.dumps(record, indent=1))


def render(result: dict) -> str:
    import report

    lines = [report.environment_line(result.get("environment")), "", f"`{result['command']}`", ""]
    lines += [
        "| Port | Measure | Setting | Trace | XLA | Compile | Run (best) | Run (median) | Speedup vs original (best; median of rounds) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        if "error" in run:
            lines.append(f"| {run['port']} | **failed** | {' '.join(run['error'][-1:])} | | | | | | |")
            continue
        for m in run["measures"]:
            for label, e in m["variants"].items():
                c, r, s = e["compile"], e["run"], e.get("speedup")
                lines.append(
                    f"| {run['port']} ({run['config']}) | {m['name']} | {label} | {c['trace_s']:.2f} s | {c['xla_s']:.2f} s | "
                    f"**{c['best_s']:.2f} s** | {r['best'] * 1e3:.2f} ms | {r['median'] * 1e3:.2f} ms | "
                    + (f"{s['best']:.2f}x; {s['median_of_rounds']:.2f}x |" if s else "|")
                )
    lines += ["", "Every setting of a port is compiled and timed in one process, its rounds interleaved with the "
              "original's. Compile = trace + XLA from cold caches, best of the repeats."]  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated ports (default: " + ", ".join(LAYERED) + ")")
    parser.add_argument("--config", help="config name (default: the port's first CPU default)")
    parser.add_argument("--values", default="0,4,16", help="unroll_layers settings to compare (default 0,4,16)")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=3, help="cold compiles per setting (best kept)")
    parser.add_argument("--out", help="JSON file (default: benchmarks/results/unroll-<date>.json)")
    parser.add_argument("--worker", nargs=3, metavar=("PORT", "CONFIG", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    values = args.values.split(",")
    if args.worker:
        worker(args.worker[0], args.worker[1], values, args.rounds, args.repeats, args.worker[2])
        return
    import run

    out = pathlib.Path(args.out) if args.out else HERE / "results" / f"unroll-{time.strftime('%Y-%m-%d')}.json"
    runs = []
    for port in (args.only.split(",") if args.only else LAYERED):
        module = run.load_port(port)
        config = args.config or module.DEFAULTS["cpu"][0]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        cmd = [sys.executable, str(HERE / "unroll.py"), "--worker", port, config, tmp, "--values", args.values,
               "--rounds", str(args.rounds), "--repeats", str(args.repeats)]  # fmt: skip
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            runs.append({"port": port, "config": config, "error": proc.stderr.strip().splitlines()[-15:]})
            print(f"[{port}] FAILED:\n  " + "\n  ".join(runs[-1]["error"]))
        else:
            runs.append(json.loads(pathlib.Path(tmp).read_text()))
        pathlib.Path(tmp).unlink(missing_ok=True)
    environment = next((r.pop("environment") for r in runs if "environment" in r), None)
    for r in runs:
        r.pop("environment", None)
    result = {"command": " ".join(["python", "benchmarks/unroll.py", *sys.argv[1:]]), "environment": environment, "runs": runs}
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out}\n")
    print(render(result))


if __name__ == "__main__":
    main()

"""Render benchmark results (from run.py) as markdown tables.

    python benchmarks/report.py benchmarks/results/*.json            # to stdout
    python benchmarks/report.py results/cpu.json results/gpu.json > table.md

One section per file: the environment, then a row per (port, config, measure)
with the best time and the median and interquartile spread of each side, the
speedup (original time / port time: above 1 the port is faster), compile
times, and the speed claimed in the README for the config it was measured on.
"""

import json
import pathlib
import sys

from harness import fmt_time


def _rate(x: float, unit: str) -> str:
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= scale:
            return f"{x / scale:.3g} {suffix} {unit}/s"
    return f"{x:.3g} {unit}/s"


def _cell(e: dict) -> str:
    r = e["run"]
    return f"**{fmt_time(r['best'])}** (med {fmt_time(r['median'])}, IQR {r['iqr_pct']:.0f}%)"


def _compile(e: dict) -> str:
    c = e.get("compile")
    if not c:
        return "–"
    if c["method"] == "aot":
        return f"{c['best_s']:.2f} s"
    return f"~{max(c['cold_s'], 0):.2f} s*"


def _check(m: dict) -> str:
    c = m.get("check")
    if not c:
        return ""
    parts = []
    for k, v in c.items():
        parts.append(f"{k} {v:.2g}" if isinstance(v, float) else f"{k} {v}")
    return "; ".join(parts)


def environment_line(env: dict | None) -> str:
    if not env:
        return "_no environment recorded (every run failed)_"
    device = env["device_kind"] if env["backend"] != "cpu" else f"{env['cpu']} ({env['cpu_affinity']} of {env['cpu_count']} cores usable)"
    if env.get("nvidia_smi"):
        device += f" — {env['nvidia_smi']}"
    dirty = ", src/ modified" if env.get("expex_src_dirty") else ""
    load = env.get("load_average")
    load_s = f"; load average at start {load[0]:.1f}" if load else ""
    return (
        f"**{env['backend'].upper()}**: {device}. JAX {env['jax']} (jaxlib {env['jaxlib']}), Python {env['python']}, "
        f"expex {str(env.get('expex_commit'))[:9]}{dirty}; {env['date']}{load_s}."
    )


def render_file(result: dict) -> str:
    lines = [environment_line(result.get("environment")), ""]
    lines.append(f"`{result['command']}` ({result['wall_s'] / 60:.1f} min)")
    lines.append("")
    ports = [r for r in result["runs"] if r["port"] != "overhead"]
    compile_only = bool(ports) and all(
        "run" not in e for r in ports if "error" not in r for m in r["measures"] for e in m["impls"].values()
    )
    if compile_only:
        lines += compile_table(ports)
        return "\n".join(lines)
    if ports:
        lines += [
            "| Port | Config | Measure | Original | Port | Speedup best (median; IQR range) | Port throughput | Compile orig → port | README claim | Same result |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in ports:
            if "error" in run:
                lines.append(f"| {run['port']} | {run['config']} | **failed** | {' '.join(run['error'][-1:])} | | | | | | |")
                continue
            for m in run["measures"]:
                labels = list(m["impls"])
                if len(labels) != 2:
                    continue
                o, p = (m["impls"][k] for k in labels)
                s = m["speedup"]
                claim = run["claim"].get(m["name"], "") if run["config"] == "readme" else ""
                name = m["name"] + (f" <sub>{m['note']}</sub>" if m.get("note") else "")
                lines.append(
                    f"| {run['port']} | {run['config']} | {name} | {_cell(o)} | {_cell(p)} | "
                    f"**{s['best']:.2f}x** ({s['median']:.2f}x; {s['low']:.2f}–{s['high']:.2f}) | "
                    f"{_rate(p['throughput'], m['unit'])} | {_compile(o)} → {_compile(p)} | {claim} | {_check(m)} |"
                )
        lines.append("")
    overhead = [r for r in result["runs"] if r["port"] == "overhead"]
    for run in overhead:
        if "error" in run:
            lines.append(f"Overhead benchmark failed: {' '.join(run['error'][-1:])}")
            continue
        lines += [
            f"Per-call overhead of expex against the same thing written in plain JAX ({run['config']}):",
            "",
            "| Case | Plain JAX | expex | expex − JAX | Ratio |",
            "|---|---|---|---|---|",
        ]
        for m in run["measures"]:
            j, x = (m["impls"][k] for k in m["impls"])
            jb, xb = j["run"]["best"] / m["items"], x["run"]["best"] / m["items"]
            name = m["name"] + (f" <sub>{m['note']}</sub>" if m.get("note") else "")
            lines.append(
                f"| {name} | {fmt_time(jb)} (IQR {j['run']['iqr_pct']:.0f}%) | {fmt_time(xb)} (IQR {x['run']['iqr_pct']:.0f}%) | "
                f"{(xb - jb) * 1e6:+.1f} us | {xb / jb:.2f}x |"
            )
        lines.append("")
    lines.append(
        "Best of N interleaved runs, each after a warm-up, ending in `block_until_ready`. "
        "Speedup = original time / port time (above 1: the port is faster); the range pairs the "
        "original's quartiles with the port's opposite ones. Compile = trace + lower + XLA compile from cold "
        "caches, best of the repeats; * = estimated as first call minus the median call "
        "(for callables that are not one `jax.jit`)."
    )
    return "\n".join(lines)


def compile_table(runs: list[dict]) -> list[str]:
    """Compile times only (from `run.py --compile-only`): trace + lower and XLA, original and port."""
    lines = [
        "| Port | Config | Measure | Original trace | Original XLA | Original total | Port trace | Port XLA | Port total | Port / original |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        if "error" in run:
            lines.append(f"| {run['port']} | {run['config']} | **failed** | {' '.join(run['error'][-1:])} | | | | | | |")
            continue
        for m in run["measures"]:
            cells, totals = [], []
            for label in m.get("labels", ["original", "port"]):
                c = m["impls"].get(label, {}).get("compile")
                if not c:
                    cells += ["–"] * 3
                    totals.append(None)
                    continue
                cells += [f"{c['trace_s']:.2f} s", f"{c['xla_s']:.2f} s", f"**{c['best_s']:.2f} s**"]
                totals.append(c["best_s"])
            ratio = f"{totals[1] / totals[0]:.2f}x" if None not in totals and totals[0] else ""
            lines.append(f"| {run['port']} | {run['config']} | {m['name']} | " + " | ".join(cells) + f" | {ratio} |")
    lines.append("")
    lines.append("Trace = `jit(f).lower(*args)` (tracing and lowering to StableHLO), XLA = `.compile()`, each from "
                 "cold caches (`jax.clear_caches()`), best of the repeats.")
    lines.append("")
    return lines


def render(results: list[dict]) -> str:
    return "\n\n".join(render_file(r) for r in results)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results = [json.loads(pathlib.Path(p).read_text()) for p in sys.argv[1:]]
    print(render(results))


if __name__ == "__main__":
    main()

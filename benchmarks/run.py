"""Time each expex port against its original, the same way, and write the results as JSON.

    python benchmarks/run.py                          # every port, the configs for this device
    python benchmarks/run.py --only mamba,llama       # some ports
    python benchmarks/run.py --device gpu             # fail unless JAX is on a GPU; GPU configs
    python benchmarks/run.py --quick                  # small configs, about a minute in all on CPU
    python benchmarks/run.py --config readme,realistic --only llama
    python benchmarks/run.py --out results/a100.json  # default: results/<device>-<date>[-quick].json
    python benchmarks/run.py --list                   # ports, configs and claims

Each (port, config) runs in its own Python process, so that jit caches,
monkey-patches made by the originals' loaders, and memory do not leak from one
benchmark into the next. Then `python benchmarks/report.py <json>` renders the
markdown table (it is also printed at the end of a run).
"""

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PLATFORMS = {"cpu": "cpu", "gpu": "cuda"}


def port_names() -> list[str]:
    return sorted(p.stem for p in (HERE / "ports").glob("*.py") if not p.stem.startswith("_"))


def load_port(name: str):
    return importlib.import_module(f"ports.{name}")


# ------------------------------------------------------------------ worker


def worker(port: str, config: str, out: str, quick: bool, rounds: int | None, device: str) -> None:
    import harness

    if device != "auto":
        import jax

        want = {"cpu": "cpu", "gpu": "gpu"}[device]
        if jax.default_backend() != want:
            raise SystemExit(f"asked for --device {device}, but JAX is on {jax.default_backend()}")
    module = load_port(port)
    cfg = module.CONFIGS[config]
    settings = harness.Settings.quick() if quick else harness.Settings()
    if rounds:
        settings.rounds = rounds
    env = harness.environment()
    print(f"[{port}/{config}] {cfg}", flush=True)
    start = time.perf_counter()
    t = time.perf_counter()
    measures = module.setup(dict(cfg))
    setup_s = time.perf_counter() - t
    results = []
    for m in measures:
        results.append(harness.run_measure(m, settings, log=lambda s: print(s, flush=True)))
    record = {
        "port": port,
        "config": config,
        "params": cfg,
        "claim": getattr(module, "CLAIM", {}),
        "settings": vars(settings),
        "setup_s": setup_s,
        "wall_s": time.perf_counter() - start,
        "load_average_before": env["load_average"],
        "load_average_after": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "environment": env,
        "measures": results,
    }
    pathlib.Path(out).write_text(json.dumps(record, indent=1))


# ------------------------------------------------------------------ parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated ports (default: all): " + ", ".join(port_names()))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--quick", action="store_true", help="small configs and few rounds: a smoke test")
    parser.add_argument("--config", help="comma-separated config names, instead of the device's defaults")
    parser.add_argument("--rounds", type=int, help="timed rounds per measure (default 15, quick 5)")
    parser.add_argument("--out", help="JSON file (default: benchmarks/results/<device>-<date>[-quick].json)")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds per (port, config) process")
    parser.add_argument("--list", action="store_true", help="list ports, configs and claims")
    parser.add_argument("--worker", nargs=3, metavar=("PORT", "CONFIG", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(*args.worker, quick=args.quick, rounds=args.rounds, device=args.device)
        return

    names = args.only.split(",") if args.only else port_names()
    unknown = set(names) - set(port_names())
    if unknown:
        sys.exit(f"unknown ports: {', '.join(sorted(unknown))} (have {', '.join(port_names())})")
    ports = {n: load_port(n) for n in names}

    if args.list:
        for n, m in ports.items():
            print(f"{n}: originals {getattr(m, 'ORIGINALS', [])}; claim {getattr(m, 'CLAIM', {})}")
            for c, cfg in m.CONFIGS.items():
                runs_on = [k for k, v in m.DEFAULTS.items() if c in v]
                print(f"   {c:12s} {cfg}  (default for: {', '.join(runs_on) or '-'})")
        return

    env = dict(os.environ)
    if args.device != "auto":
        env["JAX_PLATFORMS"] = PLATFORMS[args.device]
    kind = "quick" if args.quick else ("gpu" if args.device == "gpu" else "cpu")
    if args.device == "auto" and not args.quick:
        probe = subprocess.run([sys.executable, "-c", "import jax; print(jax.default_backend())"],
                               capture_output=True, text=True, env=env)  # fmt: skip
        kind = "gpu" if probe.stdout.strip() == "gpu" else "cpu"
    label = "gpu" if (args.device == "gpu" or kind == "gpu") else args.device if args.device != "auto" else "cpu"
    out = pathlib.Path(args.out) if args.out else HERE / "results" / f"{label}-{time.strftime('%Y-%m-%d')}{'-quick' if args.quick else ''}.json"
    if not out.is_absolute() and not out.parent.exists() and (HERE / out).parent.exists():
        out = HERE / out  # `--out results/x.json` from the repository root
    out.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    start = time.perf_counter()
    for name, module in ports.items():
        configs = args.config.split(",") if args.config else module.DEFAULTS[kind]
        for config in configs:
            if config not in module.CONFIGS:
                print(f"[{name}] no config {config!r}; skipped")
                continue
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                tmp = f.name
            cmd = [sys.executable, str(HERE / "run.py"), "--worker", name, config, tmp, "--device", args.device]
            if args.quick:
                cmd.append("--quick")
            if args.rounds:
                cmd += ["--rounds", str(args.rounds)]
            t = time.perf_counter()
            try:
                proc = subprocess.run(cmd, env=env, timeout=args.timeout, stderr=subprocess.PIPE, text=True)
                error = None if proc.returncode == 0 else proc.stderr.strip().splitlines()[-15:]
            except subprocess.TimeoutExpired:
                error = [f"timed out after {args.timeout:.0f} s"]
            if error:
                print(f"[{name}/{config}] FAILED:\n  " + "\n  ".join(error))
                runs.append({"port": name, "config": config, "params": module.CONFIGS[config], "error": error,
                             "claim": getattr(module, "CLAIM", {}), "wall_s": time.perf_counter() - t})  # fmt: skip
            else:
                runs.append(json.loads(pathlib.Path(tmp).read_text()))
            pathlib.Path(tmp).unlink(missing_ok=True)

    environment = next((r["environment"] for r in runs if "environment" in r), None)
    for r in runs:
        r.pop("environment", None)  # identical in every process: kept once
    result = {
        "format": 1,
        "command": " ".join(["python", "benchmarks/run.py", *sys.argv[1:]]),
        "kind": kind,
        "wall_s": time.perf_counter() - start,
        "environment": environment,
        "runs": runs,
    }
    out.write_text(json.dumps(result, indent=1))
    print(f"\nwrote {out}  ({result['wall_s']:.0f} s)\n")
    import report

    print(report.render([result]))
    if any("error" in r for r in runs):
        sys.exit(1)


if __name__ == "__main__":
    main()

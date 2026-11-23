"""The long fuzzing run: ``python -m tests.fuzz.run --n 20000``.

Generates programs (``fuzzgen``), runs each through the oracle (``fuzzoracle``)
in every mode, shrinks each failure to a minimal program, and prints the
distinct failures (grouped by mode and kind) with their shrunk sources. Runs in
``--jobs`` worker processes, restarted every few hundred programs to bound
memory. ``--seed`` picks the first seed; program ``i`` uses seed ``seed + i``,
so any failure is reproduced by its seed alone: ``--only-seed 1234``.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fuzzgen  # noqa: E402


def run_one(seed: int, size: int, shrink: bool = True, modes: tuple[str, ...] | None = None) -> dict[str, Any]:
    import fuzzoracle

    prog = fuzzgen.generate(seed, size)
    kw = {} if modes is None else {"modes": modes}
    rep = fuzzoracle.check(prog.source(), prog.source("truth"), seed, **kw)
    out: dict[str, Any] = {
        "seed": seed,
        "truth_failed": rep.truth_failed,
        "outcomes": [(o.mode, o.kind, o.code) for o in rep.outcomes],
        "coverage": dict(fuzzgen.COVERAGE),
        "failures": [],
    }
    fuzzgen.COVERAGE.clear()
    for o in rep.failures[:1]:
        sig = o.signature()
        small = prog
        if shrink:

            def still(q: fuzzgen.Program) -> bool:
                r = fuzzoracle.check(q.source(), q.source("truth"), seed, modes=(o.mode if o.mode != "batch2" else "batch",) if o.mode != "define" else ("eager",))
                return r.truth_failed is None and any(x.signature() == sig for x in r.failures)

            small = fuzzgen.shrink(prog, still)
            fuzzgen.COVERAGE.clear()
        out["failures"].append(
            {"mode": o.mode, "kind": o.kind, "signature": sig, "detail": o.detail[:2000], "source": small.source(), "truth": small.source("truth")}
        )
    return out


def _worker(args: tuple[int, int, bool]) -> dict[str, Any]:
    seed, size, shrink = args
    try:
        return run_one(seed, size, shrink)
    except Exception as e:  # the harness itself failed: report, do not stop the run
        import traceback

        return {"seed": seed, "truth_failed": None, "outcomes": [], "coverage": {}, "failures": [
            {"mode": "harness", "kind": "error", "signature": "harness", "detail": traceback.format_exc()[-2000:], "source": "", "truth": ""}
        ]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1, help="program size (statements ~ 6 + 6 * size)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--only-seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the failures as JSON here")
    ap.add_argument("--show", type=int, default=3, help="examples printed per distinct failure")
    a = ap.parse_args(argv)
    if a.only_seed is not None:
        r = run_one(a.only_seed, a.size, not a.no_shrink)
        prog = fuzzgen.generate(a.only_seed, a.size)
        print(prog.source())
        print(json.dumps(r["outcomes"]))
        for f in r["failures"]:
            print(f["signature"], f["detail"], "\n", f["source"], sep="\n")
        return 1 if r["failures"] else 0
    seeds = [(a.seed + i, a.size, not a.no_shrink) for i in range(a.n)]
    t0 = time.time()
    stats: collections.Counter[str] = collections.Counter()
    coverage: collections.Counter[str] = collections.Counter()
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.jobs, maxtasksperchild=200) as pool:
        for r in pool.imap_unordered(_worker, seeds, chunksize=4):
            done += 1
            coverage.update(r["coverage"])
            if r["truth_failed"]:
                stats["truth raised (discarded)"] += 1
            for mode, kind, code in r["outcomes"]:
                stats[f"{mode}:{kind}" + (f":{code}" if code else "")] += 1
            for f in r["failures"]:
                f["seed"] = r["seed"]
                groups[f["signature"] + " | " + f["detail"].splitlines()[0][:90] if f["detail"] else f["signature"]].append(f)
            if done % 200 == 0:
                print(f"[{done}/{a.n}] {time.time() - t0:.0f}s, {sum(len(v) for v in groups.values())} failing programs, {len(groups)} kinds", flush=True)
    print(f"\n{a.n} programs in {time.time() - t0:.0f}s\n")
    print("outcomes:")
    for k, v in sorted(stats.items()):
        print(f"  {k:40s} {v}")
    print("\ncoverage (constructs drawn):")
    for k, v in sorted(coverage.items()):
        print(f"  {k:48s} {v}")
    print(f"\n{len(groups)} distinct failures:")
    for key, fs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n=== {len(fs)}x {key}")
        for f in sorted(fs, key=lambda f: len(f["source"]))[: a.show]:
            print(f"--- seed {f['seed']}: {f['detail'][:600]}")
            print(f["source"])
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({k: v for k, v in groups.items()}, fh, indent=1)
    return 1 if groups else 0


if __name__ == "__main__":
    raise SystemExit(main())

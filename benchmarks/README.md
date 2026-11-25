# Benchmarks: the ports against their originals

Every port in `examples/` claims a speed against the project it was ported from.
This directory measures those claims one reproducible way, on CPU or GPU:

- the original and the port run **the same config, parameters and inputs** (the
  port is loaded with the original's parameters, through each example's
  `compare.py`), each jit-compiled, and every benchmark checks that both
  give the same result (max logit difference, same tokens, same final boards...);
- for each measure (forward pass, gradient, generation, rollout, training run):
  **compile time** (trace + lower + XLA, from cold caches) and **run time**: a
  warm-up, then interleaved rounds (original, port, port, original, ...), each
  ending in `block_until_ready`. Reported: the **best** of N, and the **median**
  and **interquartile spread** of both sides, so a noisy machine shows as noise;
- each (port, config) runs in its own process;
- the environment (JAX version and backend, device, CPU model and cores, load
  average, Python, the expex commit, the originals' commits) goes in the JSON.

A shared machine is noisy: other jobs change the timings by tens of percent
from one round to the next. Read the best-of-N speedup with the median next to
it, and trust a difference only when the IQR range (the original's quartiles
against the port's opposite ones) stays on one side of 1.

## Results

The current numbers, which the README and the docs quote (CPU only; no GPU run yet):

| File | What |
|---|---|
| [`cpu-2026-11-25-v0.1.md`](results/cpu-2026-11-25-v0.1.md) / [`.json`](results/cpu-2026-11-25-v0.1.json) | every port at its default CPU config, 25 rounds, on a quiet 4-core VM (load average 0.8 at the start) |
| [`compile-cpu-2026-11-25-v0.1.md`](results/compile-cpu-2026-11-25-v0.1.md) / [`.json`](results/compile-cpu-2026-11-25-v0.1.json) | compile times, best of 5 cold compiles |
| [`overhead-cpu-2026-11-25-v0.1.md`](results/overhead-cpu-2026-11-25-v0.1.md) / [`.json`](results/overhead-cpu-2026-11-25-v0.1.json) | per-call overhead against plain JAX, 40 rounds |

History, kept for comparison and no longer quoted. These were taken on a shared VM under heavy
load (load average 9 to 20), before the lowering fixes of later commits, so their spreads are
wide and some speedups differ a lot from the v0.1 run (Mamba's forward pass measured 14x there,
5.2x on the quiet machine):

| File | What |
|---|---|
| [`cpu-2026-11-23.md`](results/cpu-2026-11-23.md) / [`.json`](results/cpu-2026-11-23.json), [`cpu-2026-11-23-vit-mctx-nerf.json`](results/cpu-2026-11-23-vit-mctx-nerf.json) | the first full runs (ViT, mctx and NeRF in a separate run, appended to the same `.md`) |
| [`cpu-2026-11-23-quick.md`](results/cpu-2026-11-23-quick.md) / [`.json`](results/cpu-2026-11-23-quick.json) | a `--quick` smoke run |
| [`compile-cpu-2026-11-23.md`](results/compile-cpu-2026-11-23.md) / [`.json`](results/compile-cpu-2026-11-23.json), [`unroll-cpu-2026-11-23.json`](results/unroll-cpu-2026-11-23.json) | the compile-time investigation (before and after the layer barrier) and `unroll_layers` settings |

## Running it on a GPU box, from scratch

```bash
git clone https://github.com/davidesartor/expex && cd expex
python3.14 -m venv .venv && source .venv/bin/activate       # expex needs Python 3.14
pip install -e . "jax[cuda12]==0.12.1"                     # or jax[cuda13]: match the box's CUDA
pip install -r benchmarks/requirements.txt
python benchmarks/fetch.py                                 # clones the pinned originals into benchmarks/.originals/
python -c "import jax; print(jax.devices())"               # check: CudaDevice(...)
python benchmarks/run.py --device gpu --quick              # a few minutes: does everything load?
python benchmarks/run.py --device gpu                      # the README's configs and the "realistic" ones
python benchmarks/report.py benchmarks/results/gpu-*.json > benchmarks/results/gpu.md
```

Then paste the table from `benchmarks/results/gpu.md` (it is also printed at the
end of the run) and keep the JSON next to it. With `uv`: `uv venv -p 3.14 && uv pip
install -e . "jax[cuda12]" -r benchmarks/requirements.txt`.

`--device gpu` sets `JAX_PLATFORMS=cuda` and stops if JAX is not on a GPU. On
GPU, XLA preallocates 75% of memory per process, which is fine since each
benchmark has its own process; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if
the box is shared.

## Options

```bash
python benchmarks/run.py --list                           # ports, configs, which run by default, README claims
python benchmarks/run.py --only mamba,llama               # some ports
python benchmarks/run.py --config realistic --only pgx    # named configs instead of the defaults
python benchmarks/run.py --rounds 30                      # more rounds (default 15; --quick 5)
python benchmarks/run.py --out results/a100.json          # default results/<cpu|gpu>-<date>[-quick].json
python benchmarks/fetch.py --list                         # the pinned originals and whether they are fetched
python benchmarks/run.py --compile-only --repeats 5       # compile times only: trace and XLA, original and port
python benchmarks/unroll.py --values 0,4,16               # each unroll_layers setting, compiled and timed in one process
```

`--compile-only` skips the timed runs: each measure is compiled `--repeats`
times from cold caches, original and port alternating, and the table splits
the best time into tracing + lowering ("trace") and XLA. The JSON also keeps
process CPU seconds of each phase, which change less than wall time when other
jobs share the machine.

`unroll.py` compares `expex.config(unroll_layers=...)` settings for the ports
whose models loop over a list of identical layers (Llama, Mamba, NeRF, ViT).
Every setting of a port is compiled and timed in the same process, its rounds
interleaved with the original's, so that a change in load hits all of them.

Default configs: `--quick` runs each port's `quick` config (small: three to
four minutes in all on a busy 4-core CPU, nearly all of it compiling; for
CI-style smoke runs); on CPU, the `readme` config (the one the README's claim was measured
on; `tiny` / `default` for ports without a claim); on GPU, `readme` and `realistic` (a GPU-sized model or batch).

| Port | Original (pinned in `originals.toml`) | `readme` config | `realistic` config |
|---|---|---|---|
| `ppo` | PureJaxRL `ppo.py` + gymnax 0.0.8 | CartPole, 976 updates of 4 envs x 128 steps (a full training run) | 50 updates of 1,024 envs x 128 steps, 32 minibatches |
| `mamba` | mamba-jax (Equinox) | width 128, 4 layers, vocab 1,000, T 2,048 | Mamba-130m's shape: width 768, 24 layers, vocab 50,280, T 2,048 |
| `llama` | llama-2-jax | width 256, 4 layers, 8 query / 2 kv heads, batch 4 x 256, vocab 32,000 | width 1,024, 8 layers, 16 / 4 heads, batch 4 x 2,048 |
| `pgx` | Pgx v2.6.0 Connect Four, Othello | 1,024 games x 100 random moves | 8,192 games x 100 moves |
| `vit` | vision_transformer (Flax) | `tiny`: ViT-Ti/16, batch 8 of 224 x 224 | ViT-B/16, batch 64 |
| `mctx` | mctx | `default`: batch 64 x 64 simulations, MuZero and Gumbel MuZero | batch 1,024 x 128 simulations |
| `nerf` | JaxNeRF | `default`: JaxNeRF's 8 x 256 MLPs, 1,024 rays, 64 + 128 samples | 4,096 rays |
| `overhead` | plain JAX | per-call cost of a typed call, a jitted call, a jitted method, a `donate="auto"` step, an eager `.vectorize` | |

## Layout

```
benchmarks/
  run.py            the runner: one process per (port, config), writes results/<name>.json
  unroll.py         compile and run time per unroll_layers setting, all settings in one process
  report.py         JSON -> markdown table
  harness.py        timing (compile, interleaved best-of-N, stats) and environment capture
  fetch.py          clones the originals pinned in originals.toml into .originals/ (gitignored)
  originals.toml    repository URL and commit of every original
  requirements.txt  what the originals need on top of expex and JAX
  ports/<name>.py   one file per port: CONFIGS, DEFAULTS, CLAIM and setup(cfg) -> [Measure]
  results/          cpu-<date>.json / .md, compile-..., overhead-... (see Results above)
```

## Adding a port

Write `benchmarks/ports/<name>.py` with `NAME`, `ORIGINALS` (entries of
`originals.toml`), `CLAIM` ({measure: the README's words}), `CONFIGS` (with a
`quick` one), `DEFAULTS` (`{"quick": [...], "cpu": [...], "gpu": [...]}`), and
`setup(cfg)`, which loads the original with `harness.use_original(...)`, the
port and its `compare.py` with `harness.use_example` / `harness.load_module`,
and returns `Measure(name, Fn(jax.jit(original), args), Fn(jax.jit(port), args),
items=..., unit=..., check=...)`s. Keep model code out of it: load parameters
through the example's `compare.py`.

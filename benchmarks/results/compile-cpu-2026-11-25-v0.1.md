# CPU compile times, 2026-11-25 (v0.1)

Compile time only (no timed runs), each measure compiled 5 times from cold caches, original and port alternating; the best of 5 is kept. Same quiet 4-core VM as [cpu-2026-11-25-v0.1.md](cpu-2026-11-25-v0.1.md), right after it (load average 2.3 at the start, from the run before it).

**CPU**: Intel(R) Xeon(R) Processor @ 2.30GHz (4 of 4 cores usable). JAX 0.12.1 (jaxlib 0.12.1), Python 3.14.6, expex c9fe6a12a; 2026-11-25 11:36:41 UTC; load average at start 2.3.

`python benchmarks/run.py --device cpu --compile-only --repeats 5 --out benchmarks/results/compile-cpu-2026-11-25-v0.1.json` (1.2 min)

| Port | Config | Measure | Original trace | Original XLA | Original total | Port trace | Port XLA | Port total | Port / original |
|---|---|---|---|---|---|---|---|---|---|
| llama | readme | forward | 0.06 s | 0.27 s | **0.32 s** | 0.11 s | 0.35 s | **0.46 s** | 1.41x |
| llama | readme | gradient | 0.22 s | 0.92 s | **1.14 s** | 0.24 s | 0.61 s | **0.85 s** | 0.75x |
| llama | readme | generate | – | – | – | 0.20 s | 0.74 s | **0.94 s** |  |
| mamba | readme | forward | 0.07 s | 0.19 s | **0.26 s** | 0.07 s | 0.19 s | **0.26 s** | 1.00x |
| mamba | readme | gradient | 0.27 s | 0.61 s | **0.88 s** | 0.23 s | 0.65 s | **0.88 s** | 1.00x |
| mamba | readme | generate | 0.10 s | 0.77 s | **0.87 s** | 0.11 s | 0.79 s | **0.90 s** | 1.04x |
| mctx | default | muzero | 0.42 s | 1.72 s | **2.14 s** | 0.36 s | 2.48 s | **2.84 s** | 1.32x |
| mctx | default | gumbel | 0.32 s | 0.83 s | **1.15 s** | 0.26 s | 0.81 s | **1.08 s** | 0.94x |
| nerf | default | render | 0.18 s | 0.94 s | **1.12 s** | 0.24 s | 1.02 s | **1.25 s** | 1.12x |
| nerf | default | gradient | 0.43 s | 2.03 s | **2.46 s** | 0.64 s | 1.80 s | **2.44 s** | 0.99x |
| pgx | readme | connect_four | 0.12 s | 0.65 s | **0.77 s** | 0.32 s | 1.03 s | **1.35 s** | 1.76x |
| pgx | readme | othello | 0.16 s | 1.08 s | **1.23 s** | 0.17 s | 1.00 s | **1.17 s** | 0.95x |
| ppo | readme | train | 0.61 s | 2.18 s | **2.79 s** | 0.34 s | 1.22 s | **1.56 s** | 0.56x |
| vit | tiny | forward | 0.25 s | 0.21 s | **0.46 s** | 0.12 s | 0.24 s | **0.36 s** | 0.78x |
| vit | tiny | gradient | 0.63 s | 0.81 s | **1.44 s** | 0.31 s | 1.03 s | **1.34 s** | 0.94x |

Trace = `jit(f).lower(*args)` (tracing and lowering to StableHLO), XLA = `.compile()`, each from cold caches (`jax.clear_caches()`), best of the repeats.


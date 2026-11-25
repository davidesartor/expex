# CPU compile time, 2026-11-23: where it goes, what changed, and `unroll_layers`

**CPU**: Intel Xeon @ 2.10GHz, 4 of 4 cores. JAX 0.12.1, Python 3.14.6. **The machine was shared and heavily loaded**: the 1-minute load average was 12–20 during the compile runs (other test and fuzzing jobs). Wall times are therefore noisy, by up to ±30% from one run to the next. Read the process CPU seconds, and the deterministic program sizes below, alongside them.

"Before" is the source at 0fe5818. "After" is the working tree with these changes (it also holds other in-progress work). For each port, the two ran back to back:

```
python benchmarks/run.py --compile-only --device cpu --only <port> --repeats 5     # once per source tree
python benchmarks/unroll.py --values 0,4,16 --rounds 15
```

## Compile time per port: before and after

Each cell is trace (tracing + lowering, `jit(f).lower`) + XLA (`.compile()`) = total, in seconds, from cold caches, best of 5. The original is the better of its two runs.

| Port (config) | Measure | Original: trace + XLA = total (s) | Port before | Port after | Port / original, before → after | CPU s: original / port before → after | Port trace / XLA, after / before |
|---|---|---|---|---|---|---|---|
| pgx (readme) | connect_four | 0.16 + 1.60 = **1.75** | 0.62 + 2.85 = **3.46** | 0.38 + 2.64 = **3.02** | 1.98x → 1.72x | 1.21 / 2.14 → 2.00 | 0.62x / 0.93x |
| pgx (readme) | othello | 0.22 + 2.34 = **2.57** | 0.31 + 2.26 = **2.57** | 0.22 + 2.60 = **2.81** | 1.00x → 1.10x | 2.05 / 1.99 → 1.83 | 0.71x / 1.15x |
| mctx (default) | muzero | 0.70 + 6.00 = **6.70** | 0.48 + 9.13 = **9.61** | 0.44 + 7.41 = **7.85** | 1.43x → 1.17x | 5.21 / 6.72 → 5.81 | 0.91x / 0.81x |
| mctx (default) | gumbel | 0.40 + 2.49 = **2.89** | 0.31 + 1.99 = **2.30** | 0.48 + 2.63 = **3.11** | 0.80x → 1.08x | 2.19 / 1.80 → 2.34 | 1.53x / 1.32x |
| vit (tiny) | forward | 0.27 + 0.42 = **0.69** | 0.11 + 0.42 = **0.53** | 0.11 + 0.43 = **0.55** | 0.77x → 0.79x | 0.64 / 0.48 → 0.51 | 0.97x / 1.05x |
| vit (tiny) | gradient | 0.70 + 1.73 = **2.43** | 0.37 + 3.50 = **3.87** | 0.37 + 1.80 = **2.17** | 1.59x → 0.89x | 2.23 / 3.69 → 1.96 | 0.98x / 0.52x |
| nerf (default) | render | 0.31 + 2.05 = **2.36** | 0.29 + 2.02 = **2.31** | 0.28 + 2.61 = **2.89** | 0.98x → 1.23x | 2.64 / 2.78 → 2.64 | 0.99x / 1.29x |
| nerf (default) | gradient | 0.49 + 4.11 = **4.60** | 0.67 + 5.22 = **5.88** | 0.59 + 5.63 = **6.22** | 1.28x → 1.35x | 4.95 / 5.47 → 5.48 | 0.89x / 1.08x |
| llama (readme) | forward | 0.07 + 0.65 = **0.73** | 0.16 + 0.96 = **1.12** | 0.12 + 0.93 = **1.05** | 1.54x → 1.45x | 0.69 / 0.95 → 0.91 | 0.76x / 0.97x |
| llama (readme) | gradient | 0.29 + 3.25 = **3.55** | 0.34 + 1.73 = **2.08** | 0.28 + 1.69 = **1.97** | 0.59x → 0.56x | 3.06 / 1.98 → 1.90 | 0.82x / 0.97x |
| llama (readme) | generate | – | 0.26 + 2.56 = **2.82** | 0.24 + 2.54 = **2.78** |  →  | – / 2.36 → 2.55 | 0.90x / 0.99x |
| mamba (readme) | forward | 0.09 + 0.49 = **0.58** | 0.10 + 0.52 = **0.62** | 0.12 + 0.51 = **0.63** | 1.06x → 1.07x | 0.57 / 0.57 → 0.57 | 1.18x / 0.98x |
| mamba (readme) | gradient | 0.41 + 2.07 = **2.48** | 0.29 + 2.14 = **2.43** | 0.30 + 1.75 = **2.06** | 0.98x → 0.83x | 2.23 / 2.00 → 1.90 | 1.04x / 0.82x |
| mamba (readme) | generate | 0.16 + 1.46 = **1.62** | 0.12 + 1.60 = **1.72** | 0.13 + 1.82 = **1.95** | 1.06x → 1.20x | 1.54 / 1.67 → 1.72 | 1.07x / 1.14x |
| ppo (readme) | train | 0.70 + 7.14 = **7.84** | 0.39 + 3.46 = **3.86** | 0.36 + 3.26 = **3.62** | 0.49x → 0.46x | 7.34 / 3.60 → 3.46 | 0.93x / 0.94x |

Load average (1 min) around each pair of runs: pgx 20.6→20.0, mctx 19.6→19.6, vit 19.6→18.7, nerf 18.7→14.7, llama 14.4→13.6, mamba 13.6→12.2, ppo 11.8→11.8

## Program size (deterministic: not affected by load)

Jaxpr equations (counted inside loops and branches too), and the instructions of XLA's optimized HLO (fusion bodies included).

| Measure | Original: jaxpr eqns / optimized HLO instrs | Port before | Port after |
|---|---|---|---|
| pgx/connect_four | 377 / 1144 | 1690 / 2555 | 871 / 2377 |
| pgx/othello | 927 / 1565 | 979 / 1783 | 766 / 1757 |
| mctx/muzero | 877 / 4526 | 842 / 4973 | 805 / 4944 |
| mctx/gumbel | 1009 / 1906 | 922 / 2023 | 894 / 2007 |
| vit/forward | 979 / 2918 | 191 / 2942 | 192 / 2942 |
| vit/gradient | 2862 / 10250 | 497 / 24062 | 499 / 11343 |
| nerf/render | 447 / 1928 | 446 / 1990 | 450 / 1990 |
| nerf/gradient | 1123 / 4659 | 1126 / 4858 | 1134 / 4866 |
| llama/forward | 140 / 412 | 255 / 1256 | 256 / 1256 |
| llama/gradient | 345 / 1846 | 476 / 3597 | 478 / 3597 |
| llama/generate | – | 772 / 3164 | 774 / 3163 |
| mamba/forward | 289 / 1115 | 163 / 1002 | 164 / 1002 |
| mamba/gradient | 852 / 3484 | 394 / 3938 | 396 / 3938 |
| mamba/generate | 541 / 2080 | 274 / 2062 | 276 / 2062 |
| ppo/train | 1550 / 11843 | 1235 / 6279 | 1116 / 6137 |

## Where the time went

* **ViT gradient (1.6x the original's compile time, now 0.9x): all in XLA.** The 12 identical blocks are one `lax.scan`, unrolled for XLA (`unroll_layers=16`). Unrolled, XLA:CPU fuses each layer's element-wise gradient work into the fusions of the layers around it, and recomputes the residual stream's gradient inside each of them. The optimized program had 24,062 instructions against the original's 10,250, and that size grows with the square of the depth. In a 16-wide pre-norm residual tower, multiplies go from 139 at 4 layers to 923 at 12. **Fix:** an `optimization_barrier` on what one layer passes to the next (`staging/loops.py: _between_layers`). The values are the same. The ViT gradient program drops to 11,343 instructions and grows linearly with depth (110 multiplies at 4 layers, 374 at 12).
* **Connect Four (4x the original's trace time): `and` / `or` / `if` on traced values.** Each became a `lax.cond`. `_cond` traces both branches to jaxprs and replays them inside `lax.cond`, then, under `.vectorize`, JAX's batching rule batches both branch jaxprs again. It also wraps every operand of each branch in `select(p, x, stop_gradient(x))`, booleans and integers included. That came to 32 `cond`s for the `and`s of the eight `while` tests alone. 64% of the port's 1,690 equations were `select_n` and `stop_gradient` from batched `cond`s. **Fixes** (`staging/flow.py: _selectable / _select`):
  * When the condition is mapped over (a `BatchTracer`), emit the select JAX would make, directly: one replay per branch, and masks only on float inputs (only they carry gradients).
  * When both branches are small (≤ 32 equations, ≤ 4,096 elements, no loops, branches or dots), effect-free, and give only booleans and integers (the `a and b` of a `while` test), select between them without a `cond`. No gradient flows through integers, so NaN-safe gradients are unaffected.
  * Don't build the untaken path's `not pred` unless an `assert`/`raise` in the branch reads it.

  Connect Four's jaxpr: 1,690 → 871 equations. Trace: 0.62 → 0.38 s.
* **The rest of Connect Four's gap is its design**: eight batched `while` loops walking from the new stone, where Pgx gathers from a table. Carrying the loop's condition instead of re-evaluating it emitted the same number of equations, so it was not kept.
* **mctx**: the gap is in XLA. The port's jaxpr is already smaller than the original's (805 vs 877 equations), and its optimized HLO is within 9% of it (4,944 vs 4,526). The changes above make muzero 9.6 → 7.9 s (CPU 6.7 → 5.8 s).
* **Not where the time goes** (cProfile of the traces):
  * Shape checks: 469 `spec.match` calls take 6 ms of a 449 ms ViT forward trace, about 1%, nested typed calls included. Caching them, or a switch that leaves them out of traced code, would gain about 1%, so neither was added.
  * Loop analysis: every staged loop body in every port is traced exactly once per execution of the loop.
  * Probes: `_probe` (a variable first assigned inside a loop) never runs in any port.
  * The dot-batching rule (`_flat_dot`): 11 ms of ViT's 449 ms.
* **Remaining gaps not addressed**:
  * Llama forward compiles 1.45x the original's time, because its optimized program is 3x the size (1,256 vs 412 instructions); its gradient is 0.56x.
  * NeRF compiles about 1.2–1.35x the original's time with almost the same program size (4,866 vs 4,659), so that gap is probably noise plus layout.

## `unroll_layers`: compile and run time per port, 0 / 4 / 16

Each port's settings were compiled and timed in one process, rounds interleaved with the original's ("vs original" > 1: the port is faster). The last column is the same setting before the layer barrier existed (from a separate run under different load: compare its CPU seconds and the ratios, not its wall times). Ports without a list of identical layers (mctx, pgx, PPO) are not affected by the setting.

Layers per model: Llama 4, Mamba 4, NeRF 8 (two loops: 2 and 4 identical layers), ViT-Ti 12.

| Port (layers) | Measure | Setting | Compile: trace + XLA (s) | CPU s | Run, best (median) | vs original: best; median of rounds | Before the barrier: compile (CPU s); run vs original |
|---|---|---|---|---|---|---|---|
| llama (readme) | forward | original | 0.06 + 0.68 = **0.80** | 0.68 | 216.6 ms (262.3) |  |  |
| llama (readme) | forward | 0 | 0.15 + 1.24 = **1.40** | 1.18 | 206.5 ms (243.0) | **1.05x**; 1.09x | 1.24 s (1.45); 0.94x; 1.06x |
| llama (readme) | forward | 4 | 0.13 + 1.04 = **1.18** | 0.92 | 171.5 ms (288.1) | **1.26x**; 1.03x | 0.93 s (1.02); 0.99x; 1.02x |
| llama (readme) | forward | 16 | 0.15 + 1.00 = **1.19** | 0.91 | 183.0 ms (250.2) | **1.18x**; 1.23x | 0.99 s (1.00); 0.94x; 0.94x |
| llama (readme) | gradient | original | 0.19 + 2.40 = **2.60** | 3.28 | 1610.4 ms (1958.3) |  |  |
| llama (readme) | gradient | 0 | 0.35 + 3.49 = **3.87** | 4.20 | 1215.9 ms (1499.9) | **1.32x**; 1.34x | 3.76 s (4.37); 1.45x; 1.55x |
| llama (readme) | gradient | 4 | 0.31 + 1.40 = **1.70** | 2.16 | 1041.4 ms (1388.4) | **1.55x**; 1.43x | 2.03 s (2.27); 1.43x; 1.59x |
| llama (readme) | gradient | 16 | 0.29 + 1.67 = **1.97** | 2.01 | 1087.9 ms (1369.6) | **1.48x**; 1.47x | 1.93 s (2.22); 1.32x; 1.39x |
| llama (readme) | generate | 0 | 0.34 + 3.36 = **3.95** | 3.22 | 410.3 ms (684.1) |  | 2.80 s (3.02); 318.4 ms |
| llama (readme) | generate | 4 | 0.29 + 2.71 = **3.03** | 2.85 | 269.4 ms (454.1) |  | 2.53 s (2.75); 221.3 ms |
| llama (readme) | generate | 16 | 0.29 + 2.81 = **3.27** | 2.86 | 274.0 ms (603.7) |  | 2.41 s (2.74); 228.3 ms |
| mamba (readme) | forward | original | 0.12 + 0.49 = **0.62** | 0.60 | 890.2 ms (1028.0) |  |  |
| mamba (readme) | forward | 0 | 0.11 + 0.57 = **0.72** | 0.79 | 115.0 ms (130.7) | **7.74x**; 7.97x | 0.58 s (0.71); 7.46x; 8.00x |
| mamba (readme) | forward | 4 | 0.09 + 0.60 = **0.70** | 0.62 | 111.3 ms (132.9) | **8.00x**; 8.06x | 0.56 s (0.58); 7.66x; 8.19x |
| mamba (readme) | forward | 16 | 0.11 + 0.50 = **0.61** | 0.64 | 115.2 ms (131.6) | **7.73x**; 7.71x | 0.45 s (0.54); 8.28x; 8.84x |
| mamba (readme) | gradient | original | 0.26 + 2.10 = **2.36** | 2.32 | 7045.7 ms (8020.6) |  |  |
| mamba (readme) | gradient | 0 | 0.30 + 3.67 = **4.10** | 3.60 | 1232.4 ms (1383.4) | **5.72x**; 5.81x | 2.91 s (3.18); 6.33x; 5.63x |
| mamba (readme) | gradient | 4 | 0.30 + 1.93 = **2.23** | 2.16 | 891.7 ms (993.5) | **7.90x**; 7.90x | 1.60 s (1.80); 7.85x; 7.50x |
| mamba (readme) | gradient | 16 | 0.31 + 1.57 = **1.87** | 2.16 | 931.8 ms (1095.6) | **7.56x**; 7.00x | 1.69 s (1.95); 8.82x; 8.02x |
| mamba (readme) | generate | original | 0.13 + 1.45 = **1.61** | 1.76 | 9.5 ms (13.9) |  |  |
| mamba (readme) | generate | 0 | 0.16 + 1.51 = **1.66** | 1.85 | 17.4 ms (20.2) | **0.55x**; 0.65x | 1.54 s (1.67); 0.58x; 0.51x |
| mamba (readme) | generate | 4 | 0.12 + 1.60 = **1.73** | 1.73 | 9.7 ms (11.7) | **0.98x**; 1.06x | 1.55 s (1.68); 0.96x; 0.89x |
| mamba (readme) | generate | 16 | 0.13 + 1.69 = **1.86** | 1.89 | 9.3 ms (11.9) | **1.03x**; 1.08x | 1.47 s (1.51); 0.99x; 0.99x |
| nerf (default) | render | original | 0.22 + 2.15 = **2.47** | 2.81 | 3842.6 ms (3916.0) |  |  |
| nerf (default) | render | 0 | 0.30 + 2.56 = **2.89** | 3.29 | 3883.2 ms (4686.3) | **0.99x**; 0.98x | 2.44 s (2.78); 1.04x; 1.01x |
| nerf (default) | render | 4 | 0.30 + 2.11 = **2.41** | 2.87 | 3861.6 ms (4548.0) | **1.00x**; 0.95x | 2.43 s (2.68); 0.96x; 0.90x |
| nerf (default) | render | 16 | 0.28 + 2.48 = **2.77** | 2.99 | 3860.0 ms (4363.9) | **1.00x**; 0.90x | 2.25 s (2.65); 0.95x; 0.95x |
| nerf (default) | gradient | original | 0.48 + 4.01 = **4.49** | 5.25 | 9739.4 ms (10231.1) |  |  |
| nerf (default) | gradient | 0 | 0.59 + 5.21 = **5.80** | 6.62 | 10891.1 ms (11115.7) | **0.89x**; 0.92x | 3.84 s (6.03); 0.83x; 0.85x |
| nerf (default) | gradient | 4 | 0.61 + 4.16 = **4.77** | 5.79 | 9601.8 ms (9915.8) | **1.01x**; 1.03x | 3.28 s (5.13); 0.97x; 0.94x |
| nerf (default) | gradient | 16 | 0.59 + 4.54 = **5.13** | 5.97 | 10664.5 ms (11307.7) | **0.91x**; 0.96x | 3.27 s (5.23); 0.98x; 0.95x |
| vit (tiny) | forward | original | 0.33 + 0.32 = **0.69** | 0.69 | 153.9 ms (222.9) |  |  |
| vit (tiny) | forward | 0 | 0.11 + 0.28 = **0.39** | 0.53 | 144.0 ms (221.3) | **1.07x**; 0.95x | 0.36 s (0.47); 0.89x; 0.92x |
| vit (tiny) | forward | 4 | 0.11 + 0.36 = **0.49** | 0.57 | 177.8 ms (235.2) | **0.87x**; 0.89x | 0.40 s (0.49); 0.85x; 1.02x |
| vit (tiny) | forward | 16 | 0.12 + 0.30 = **0.47** | 0.52 | 162.5 ms (237.4) | **0.95x**; 0.90x | 0.44 s (0.46); 1.01x; 0.97x |
| vit (tiny) | gradient | original | 0.68 + 2.02 = **2.87** | 2.44 | 677.7 ms (922.0) |  |  |
| vit (tiny) | gradient | 0 | 0.32 + 3.57 = **3.90** | 3.74 | 959.3 ms (1047.1) | **0.71x**; 0.84x | 2.28 s (3.53); 0.79x; 0.85x |
| vit (tiny) | gradient | 4 | 0.29 + 3.06 = **3.35** | 3.93 | 820.1 ms (1091.6) | **0.83x**; 0.86x | 2.19 s (3.44); 0.82x; 0.80x |
| vit (tiny) | gradient | 16 | 0.40 + 2.33 = **2.74** | 2.39 | 621.7 ms (847.3) | **1.09x**; 1.20x | 2.91 s (3.72); 1.00x; 0.94x |

### Depth scan: a ViT-Ti-shaped stack, gradient of a batch of 8, 32 × 32 images

Process CPU seconds of XLA compile (best of 3, the three variants interleaved), and run time (best of 9, interleaved). "Unrolled" is the scan unrolled without a barrier (as before); "barrier" is what `unroll_layers` now does.

| Layers | Rolled (0): XLA CPU s / run | Unrolled, no barrier | Unrolled + barrier |
|---|---|---|---|
| 2 | 2.29 / 42.0 ms | 1.32 / 24.1 ms | **1.18 / 21.9 ms** |
| 4 | 2.52 / 82.7 ms | 1.93 / 63.4 ms | **1.22 / 64.1 ms** |
| 8 | 3.54 / 181 ms | 2.94 / 148 ms | **1.53 / 132 ms** |
| 12 | 3.22 / 268 ms | 3.56 / 216 ms | **1.86 / 196 ms** |
| 16 | 3.45 / 367 ms | 3.95 / 284 ms | **1.97 / 268 ms** |
| 24 | 4.38 / 589 ms | 5.10 / 387 ms | **2.63 / 377 ms** |
| 32 | 5.32 / 732 ms | 6.74 / 601 ms | **3.48 / 643 ms** |
| 64 | 8.78 / 1569 ms | 13.40 / 1265 ms | **7.66 / 1129 ms** |

Width 48 instead of 192 gives the same picture: at 8 layers 3.52 / 1.57 CPU s, at 32 layers 6.04 / 3.88 CPU s (rolled / barrier). At 32 layers the barrier version runs 1.35x faster than rolled. The forward pass alone (width 192) compiles a little slower unrolled, and runs faster:

| Layers | Rolled: trace + XLA CPU s / run | Unrolled + barrier |
|---|---|---|
| 8 | 1.21 / 33.8 ms | 0.44 / 28.7 ms |
| 32 | 0.62 / 151 ms | 0.86 / 119 ms |

### Recommendation

Keep unrolling on by default, now with the barrier between layers, which is what this change does. With it, for every port and every depth measured up to 64 layers:

* Unrolled gradients compile faster than the rolled loop: ViT-Ti 2.4 CPU s against 3.7, Llama 2.0 against 4.2, Mamba 2.2 against 3.6.
* Unrolled code runs as fast or faster:
  * gradients: ViT-Ti 1.09x of the original against 0.71x rolled, Llama 1.48x against 1.32x, Mamba 7.6x against 5.7x;
  * generation: Mamba 9.3 ms against 17.4 ms rolled, Llama 274 ms against 410 ms.
* A forward pass alone is the only case where unrolling compiles slower: +0.25 s at 32 layers. At 12 layers or fewer it is a wash.

The data does not support a rule by layer size: narrow (48) and wide (192) layers behave the same way. Depth only starts to favour the rolled loop's compile time beyond 64 layers. The default stays at **16**. Raising it to **32** is safe on this data, and 64 is about break-even on compile while still running faster. `unroll_layers=4` is never better than 16 here, except that it rolls ViT's 12 layers, and with the barrier that is now the slower choice for both compile and run.

## Per-call overhead (hot path), before and after

`python benchmarks/run.py --only overhead`, two runs each, alternating; best of N per call. Unchanged within noise. The changes are trace-time only.

| Case | Plain JAX (µs) | expex before (µs) | expex after (µs) |
|---|---|---|---|
| typed eager call | 0.0 | 5.0 / 5.4 | 5.8 / 5.0 |
| jitted call | 5.3 | 5.7 / 5.4 | 6.1 / 5.3 |
| jitted method | 5.7 | 6.5 / 6.3 | 5.9 / 5.8 |
| donate step | 4.6 | 6.7 / 6.0 | 6.3 / 7.0 |
| eager vectorize | 319.5 | 19.9 / 23.9 | 18.8 / 19.9 |

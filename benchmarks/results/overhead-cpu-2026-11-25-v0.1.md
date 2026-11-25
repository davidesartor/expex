# CPU per-call overhead, 2026-11-25 (v0.1)

The overhead benchmark alone, with 40 rounds (each round times 100 calls), right after [cpu-2026-11-25-v0.1.md](cpu-2026-11-25-v0.1.md) and the [compile run](compile-cpu-2026-11-25-v0.1.md) on the same quiet 4-core VM. The design page's overhead table quotes this file.

**CPU**: Intel(R) Xeon(R) Processor @ 2.30GHz (4 of 4 cores usable). JAX 0.12.1 (jaxlib 0.12.1), Python 3.14.6, expex c9fe6a12a; 2026-11-25 11:37:54 UTC; load average at start 2.0.

`python benchmarks/run.py --device cpu --only overhead --rounds 40 --out benchmarks/results/overhead-cpu-2026-11-25-v0.1.json` (0.1 min)

Per-call overhead of expex against the same thing written in plain JAX (default):

| Case | Plain JAX | expex | expex − JAX | Ratio |
|---|---|---|---|---|
| typed eager call <sub>f(x: f32[B, D], w: f32[D, D]) returning x</sub> | 0.0 us (IQR 5%) | 4.6 us (IQR 3%) | +4.6 us | 173.25x |
| jitted call <sub>x @ w, 4 x 8</sub> | 4.8 us (IQR 1%) | 4.9 us (IQR 2%) | +0.1 us | 1.01x |
| jitted method <sub>x * self.w on a struct</sub> | 5.1 us (IQR 5%) | 5.3 us (IQR 10%) | +0.2 us | 1.03x |
| donate step <sub>donate="auto" vs donate_argnums=0, per step</sub> | 4.4 us (IQR 3%) | 5.5 us (IQR 2%) | +1.1 us | 1.26x |
| eager vectorize <sub>2x + 1 over 16 rows, not jitted</sub> | 248.0 us (IQR 2%) | 16.2 us (IQR 98%) | -231.8 us | 0.07x |

Best of N interleaved runs, each after a warm-up, ending in `block_until_ready`. Speedup = original time / port time (above 1: the port is faster); the range pairs the original's quartiles with the port's opposite ones. Compile = trace + lower + XLA compile from cold caches, best of the repeats; * = estimated as first call minus the median call (for callables that are not one `jax.jit`).

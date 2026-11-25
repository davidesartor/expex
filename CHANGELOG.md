# Changelog

## 0.1.0 (unreleased)

The first release. expex is a thin layer over JAX: arrays are annotated with their
sizes in function and class signatures, and the library reads those annotations to
check calls, prove shapes, convert plain Python control flow and drive its tools.
Requires Python 3.14 and JAX 0.12.

### Checked calls and proofs

- `@expex.fn` checks every argument and the result against annotations such as
  `f32[B, T, D]`; a mismatch raises `ShapeError` naming the axis and where its size
  was bound.
- Under `jax.jit` the checks run once, while tracing: the compiled program is the one
  plain JAX gives.
- `expex.prove` checks that the declared result shape follows from the declared
  inputs for every size, using JAX's shape polymorphism; also `python -m expex pkg/`
  (with `--watch`) and `pytest --expex pkg`. Undecidable size comparisons split the
  proof into regions; a `spot-checked` verdict is evidence, not proof.
- Every error has a code (`EX101` to `EX804`), `e.code`, and a link to its section of
  `docs/errors.md`.

### Sizes and axes by name

- Sizes are type parameters (`def f[B, T, D]`), usable by name in the body
  (`jnp.zeros(D)`, `x.sum(T)`, `axis=T`).
- Derived sizes (`f32[T + 1]`, `f32[heads, D // heads]`, `f32[self.heads]`),
  constraints (`assert D % heads == 0`), fixed sizes (`def f[Rows = 6]`), `*Rest`,
  `Literal`, `Dim` classes; `/` and `**` on sizes are refused.
- `expex.einsum(q, k, to=(B, T, S))` and `expex.rearrange(x, to=...)` by axis name.
- Helpers and lambdas defined inside a call see its sizes.

### Plain control flow

- `if` / `else`, ternaries, `and` / `or`, `while`, `for`, `break` / `continue` /
  `return` on traced values are converted from source to `lax.cond` / `select`,
  `lax.while_loop` and `lax.scan`; `for t in range(T)` with `.append` becomes a scan.
- In-place updates of locals (`board[r, c] = v`, `visits[i] += 1`), `assert`,
  `print`, `min` / `max` work on traced values.
- Checked with a differential fuzzer (about 26,000 generated programs run as plain
  Python, eagerly, jitted, batched, under `vmap` and `grad`).

### Batching

- A typed call given extra leading axes maps over them, so a layer written for one
  token takes a sequence and a model written for one example takes a batch;
  `vectorize=False` (or `expex.config.vectorize = False`) opts out.
- `f.vectorize(...)` maps explicitly; eager mapped calls are compiled from their
  second call; batch axes of mapped matmuls are merged so XLA sees one dot.

### Modules

- `@expex.struct` classes are pytrees whose methods are typed calls; they build from
  their annotations (`Stack(key, D=64, blocks=12)`), from a config
  (`Stack(key, cfg)`), or from values; `model.sizes` reads the sizes back.
- A loop over a list of identical layers is one `lax.scan` (unrolled for XLA up to
  `expex.config.unroll_layers`, 32 by default).
- `model.at.blocks[5].w.set(v)` (also `add`, `multiply`, `apply`), checked.
- `expex.save` / `expex.load` (checked on load, onto a mesh too), `expex.diff`.
- Mixed precision: `@expex.fn(compute=jnp.bfloat16)`, `@expex.struct(compute=...)`.
- Random keys (`random=True`, `expex.key()`), `jit=`, `remat=`, `donate="auto"`,
  and `expex.shard(f, mesh, B="data", D="model")` with shardings from axis names.

### Tools

- `expex.cost`: FLOPs, parameters and the largest array as formulas in the sizes, or
  numbers for given sizes; `expex.debug.profile` per source line.
- `expex.summary`: parameters, bytes and FLOPs per module.
- `expex.diagram`: architecture diagrams drawn from the code, as SVG, PNG or an
  interactive HTML page; `expex.diagram.timed` colours them by measured time, and a
  `ShapeError` carries `e.diagram` with the failing call in red.
- `expex.inspect`: summary, diagram, cost, activations and line profile on one page.
- `expex.debug`: `shapes` (line by line), `find_nan`, `watch` (activations and
  gradients per layer), `ops`, `timing`, `memory`.
- `python -m expex shapes file.py --json` for editors, and a VS Code extension in
  `editors/vscode/` (not published).
- Notebook displays for structs, errors, proofs and costs.

### Interop and migration

- `@expex.fn` on Equinox and Flax NNX methods (sizes from annotated fields), and
  inside Flax linen; structs and Equinox modules nest both ways.
- Optax optimizes structs as they are; `expex.mask` builds optimizer masks and
  Equinox partitions from paths and annotations. Orbax checkpoints structs.
- jaxtyping annotations (`Float[Array, "b t"]`) are read as they are;
  `python -m expex migrate jaxtyping` rewrites them in expex's style.
- `expex.annotate(f, *examples)` proposes annotations for plain JAX code from one
  run, with `.diff()` and `.apply()`.
- Seven projects ported as examples, each checked against the original and with
  every shape proved: PureJaxRL PPO, mamba-jax, llama-2-jax, Pgx (Connect Four,
  Othello), Vision Transformer, mctx (MuZero, Gumbel MuZero), JaxNeRF; a multi-device
  training example; notebooks for each.

### Performance

- A compiled call costs what `jax.jit` costs: 4.9 µs against 4.8 µs for a small jitted
  call, 5.3 µs against 5.1 µs for a jitted method; an eager typed call adds 4.6 µs of
  checks; `donate="auto"` adds 1.1 µs per step.
- The ports against their originals on a quiet 4-core CPU, best of 25 rounds (median):
  Mamba 5.2x forward (5.8x) and 5.4x gradients (6.0x), from writing the scan one step
  at a time; Llama 1.08x forward, 1.61x gradients, 1.48x generation; Connect Four
  1.50x, Othello 1.07x; PPO, mctx and Mamba generation the same; ViT slightly slower
  (0.92x forward, 0.93x gradients) and NeRF's render slightly slower (0.96x).
  Source: `benchmarks/results/cpu-2026-11-25-v0.1.md`.
- Mapped matmuls are given to XLA with one batch axis, and stacks of up to 32
  identical layers are unrolled with a barrier between layers, which keeps compile
  time linear in the depth.

### Known limits

- Python 3.14 only (annotations are read as text, PEP 649 / 749); Colab's hosted
  runtimes do not have it yet.
- Conversion needs the function's source; code inside `try` / `with`, loops with an
  `else`, in-place updates of anything but a local, and variables shared with a
  nested function stay plain Python.
- Automatic batching needs the extra axes in front, never maps over `self`, and can
  hide mismatches in code whose annotations were never enforced.
- Stateful Flax NNX modules cannot update state inside a mapped call or a loop over
  layers.
- `debug.find_nan`, parts of the prover and the merged dots read JAX internals; they
  are tested on JAX 0.12.1 only.
- Static type checkers need some `# type: ignore`s (see `docs/design.md`).
- No GPU benchmarks yet; compile times are often longer than the originals'.

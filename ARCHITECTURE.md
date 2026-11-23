# How expex is put together

Everything starts from `@expex.fn` (`decorators.py`), which makes an `Fn` (`function.py`);
`@expex.vectorize` and `@expex.random` set flags on the same `Fn`, so they stack in any order.
`@expex.jit` is `jax.jit` with `static_argnames` read from the signature. Gradients and
remat are JAX's own. Each module has one job:

| Module | Job |
|---|---|
| **`annotations/`** | **What users write, and what it means.** |
| `types.py` | The annotation objects: `f32[B, T]`, `Static[...]`, `Key`, `Batch[...]`, `Dim`. |
| `dims.py` | Dimensions (`B`, `T + 1`, `*Rest`), binding sizes during a call (`Env`), printing values as `f32[3, 4]`. |
| `expr.py` | Size arithmetic and constraints (`"T + 1"`, `"D % heads == 0"`). |
| `sym.py` | Size arithmetic written in annotations (`f32[T + 1]`), read from their text. |
| `spec.py` | Annotations parsed into specs, values matched against them, structs included; `fields()` walks a value's parts with their paths (`blocks[2].mlp.w1`). |
| `errors.py` | `ShapeError`, `StageError`. |
| `messages.py` | Error messages, hints and the notes added to exceptions. |
| `function.py` | `Fn`: binds arguments, checks them and the result, runs the body; `config`. |
| `decorators.py` | `@fn`, `@jit` (`donate="auto"` too), `@vectorize`, `@random`, `@remat`. |
| `batching.py` | `f.vectorize(...)`: mapping over extra leading axes. |
| `keys.py` | Implicit random keys (`expex.key()`). |
| `modules.py` | `@expex.struct`: modules registered as pytrees whose methods are `@fn`s; `Module.init(key, D=64)`. |
| `io.py` | `expex.save` / `expex.load`: one `.npz` per model, checked against the annotations on load. |
| `sharding.py` | `expex.shard`, `expex.shardings`: shardings from axis names mapped to mesh axes. |
| `consts.py` | `expex.eps`, `expex.tiny`, `expex.huge`: dtype limits resolved against the array they meet. |
| `debug.py` | `expex.debug`: the debugging tools, gathered from `tools/`. |
| `_diagram.py` | `expex.diagram` (being rewritten; it moves to `tools/` after). |
| `pytest_plugin.py` | `pytest --expex pkg`. |
| **`staging/`** | **Plain Python control flow on traced values.** |
| `analysis.py` | What the converter reads off the syntax tree: names assigned and read, jumps, sizes used by name. |
| `convert.py` | Rewrites a function's source so plain Python works on traced values. |
| `runtime.py` | The namespace the rewritten code calls into, backed by: |
| `flow.py` | `if` / `while` and the machinery they share, |
| `loops.py` | `for`, iterables, `expex.rows` / `expex.stack`, |
| `ops.py` | operators, in-place updates, `assert` / `raise` / `print` / `min` / `max`, checked locals, |
| `axes.py` | axes by name: `x.sum(T)`, `axis=T`, `expex.einsum`, `expex.rearrange`. |
| **`tools/`** | **Everything that looks at a function rather than runs it.** |
| `trace.py` | The trace engine: a function (with its examples, size keywords, list lengths) traced on symbolic sizes, or on real arguments; walking jaxprs. |
| `prove.py` | The prover (`expex.prove`) and the command line. |
| `cost.py` | `expex.cost`: FLOPs and parameters as polynomials in the sizes. |
| `profile.py` | `expex.debug.profile`: FLOPs and bytes per source line. |
| `summary.py` | `expex.summary`: one row per module with parameters and FLOPs. |
| `diff.py` | `expex.diff`: two models or checkpoints compared field by field. |
| `watch.py` | `expex.debug.watch`: per-call output and gradient statistics, with flags. |
| `nan.py`, `shapes.py`, `program.py` | `debug.find_nan`; `debug.shapes`; `debug.ops`, `timing`, `memory`. |
| `pages.py` | The HTML the tools write: page shell, light/dark palette, tree tables, JSON data. |

A call to an `@fn` goes `Fn.__call__` → (`batching.vectorize` if vectorized) → `_run`
(opens a key scope if random) → `_run_checked` (through `CALL_HOOK` when a tool installed
one) → `_checked`, which matches arguments with `annotations/spec.py`, runs the converted
body (`staging/convert.py` output calling `staging/runtime.py`), and matches the result.

The tools that need no data (`prove`, `cost`, `profile`, `summary`, `diagram`,
`debug.shapes`) all go through `tools/trace.py`: `resolve(target, sizes)` then
`.trace(n)`, with lists of modules given `n` elements.

Tests: `tests/` per feature, `tests/regressions/` for bugs found by reviews,
`tests/ports/` for ports checked against their originals.

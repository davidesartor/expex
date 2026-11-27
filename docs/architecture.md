# Architecture

[← The design of expex](design.md) · [Contents](README.md)

How expex is put together, for contributors.

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
| `jaxtyping.py` | jaxtyping's annotations (`Float[Array, "b t"]`) as expex dimensions and dtypes, from its classes or, without it, from the text; parsed once, checked by the same specs. |
| `spec.py` | Annotations parsed into specs, values matched against them, structs included; `fields()` walks a value's parts with their paths (`blocks[2].mlp.w1`). |
| `errors.py` | `ShapeError`, `StageError`; the error codes (`CODES`), and `coded`, which gives an error its code and the link to its section of `docs/errors.md`. |
| `messages.py` | Error messages, hints and the notes added to exceptions. |
| `function.py` | `Fn`: binds arguments, checks them and the result, runs the body; `config`. |
| `decorators.py` | `@fn`, `@jit` (`donate="auto"` too), `@vectorize`, `@random`, `@remat`. |
| `batching.py` | Mapping over extra leading axes: `f.vectorize(...)`, and plain calls that need it (not for jaxtyping-annotated functions unless asked: `Fn._auto`); eager mapped calls compiled from the second call; several batch axes, and nested mapped calls, lowered so that dots see one batch axis. |
| `keys.py` | Implicit random keys (`expex.key()`). |
| `modules.py` | `@expex.struct`: modules registered as pytrees whose methods are `@fn`s; the generated constructor (`Module(key, D=64)` builds, `Module(w, b)` takes values, `Module(key, cfg)` takes a config), `expex.build`, `model.sizes`, `model.at[...]` updates. |
| `io.py` | `expex.save` / `expex.load`: one `.npz` per model, checked against the annotations on load. |
| `masking.py` | `expex.mask`: a pytree of booleans chosen by path and annotation, for optax masks and Equinox's `partition`. |
| `sharding.py` | `expex.shard`, `expex.shardings`: shardings from axis names mapped to mesh axes. |
| `precision.py` | Mixed precision (`compute=jnp.bfloat16`): bodies wrapped to cast float inputs in and results back to their declared dtype; the compute dtype accepted for float annotations inside such a body. |
| `consts.py` | `expex.eps`, `expex.tiny`, `expex.huge`: dtype limits resolved against the array they meet. |
| `debug.py` | `expex.debug`: the debugging tools, gathered from `tools/`. |
| `_diagram.py`, `_diagram.html` | `expex.diagram`: the figure (layout, SVG, PNG) and its interactive page; it traces through `tools/trace.py`. |
| `pytest_plugin.py` | `pytest --expex pkg`. |
| `__main__.py`, **`cli/`** | **`python -m expex`**: `__init__.py` dispatches the subcommands (a path or module name alone is `prove`); `watch.py` re-proves on save (`prove --watch`); `shapes.py` reports every line's shapes and failures as JSON for editors (`editors/`). |
| **`staging/`** | **Plain Python control flow on traced values.** |
| `analysis.py` | What the converter reads off the syntax tree: names assigned and read, jumps, sizes used by name. |
| `convert.py` | Rewrites a function's source so plain Python works on traced values. |
| `runtime.py` | The namespace the rewritten code calls into, backed by: |
| `flow.py` | `if` / `while` and the machinery they share, |
| `loops.py` | `for`, iterables, `expex.rows` / `expex.stack`, |
| `ops.py` | operators, in-place updates, `assert` / `raise` / `print` / `min` / `max`, checked locals, |
| `axes.py` | axes by name: `x.sum(T)`, `axis=T`, `expex.einsum`, `expex.rearrange`. |
| **`tools/`** | **Everything that looks at a function rather than runs it.** |
| `trace.py` | The trace engine: a function (with its examples, size keywords, list lengths) traced on symbolic sizes (those a constraint fixes, `V == 3 * (1 + 2 * self.deg_view)`, as numbers), or on real arguments; walking jaxprs. |
| `prove.py` | The prover (`expex.prove`), and the `prove` subcommand: a module, a package, a file, or each file of a plain folder. |
| `annotate.py` | `expex.annotate`: annotations for plain JAX code, read off one run on example inputs; also the source edit, diff and write helpers. |
| `migrate.py` | `python -m expex migrate jaxtyping`: jaxtyping annotations rewritten in expex's style, with `annotate.py`'s edit, diff and write helpers. |
| `cost.py` | `expex.cost`: FLOPs and parameters as polynomials in the sizes. |
| `profile.py` | `expex.debug.profile`: FLOPs and bytes per source line. |
| `summary.py` | `expex.summary`: one row per module with parameters and FLOPs. |
| `diff.py` | `expex.diff`: two models or checkpoints compared field by field. |
| `watch.py` | `expex.debug.watch`: per-call output and gradient statistics, with flags. |
| `nan.py`, `shapes.py`, `program.py` | `debug.find_nan`; `debug.shapes`; `debug.ops`, `timing`, `memory`. |
| `inspect.py` | `expex.inspect`: one page with every tool's view of a model (summary, diagram, cost, watch), over one module tree. |
| `display.py` | Notebook displays: structs, shape errors, proofs and cost estimates as HTML and text. |
| `pages.py` | The HTML the tools write: page shell, light/dark palette, tree tables, JSON data. |

A call to an `@fn` goes `Fn.__call__` → `_run` (opens a key scope if random) →
`_run_checked` (through `CALL_HOOK` when a tool installed one) → `_checked`, which matches
arguments with `annotations/spec.py`, runs the converted body (`staging/convert.py` output
calling `staging/runtime.py`), and matches the result. When arguments fail to match only
by extra leading axes, `_checked` raises `batching.Rebatch` and `_run` maps the call over
them with `batching.vectorize`.

The tools that need no data (`prove`, `cost`, `profile`, `summary`, `diagram`,
`debug.shapes`) all go through `tools/trace.py`: `resolve(target, sizes)` then
`.trace(n)`, with lists of modules given `n` elements.

Tests, one file per feature, by area: `tests/core/` (checked calls, annotations, sizes,
staging, batching, modules, interop), `tests/tools/` (the prover, cost, summary, diagrams,
debugging tools, checkpoints, sharding, the command line and the pytest plugin), `tests/docs/`
(every code block of the README and the guide, the notebooks, `expex.prove` on every
example), `tests/typing/` (pyright and mypy on `usage.py`), `tests/ports/` for ports checked
against their originals (with `tests/data/`), `tests/regressions/` for bugs found by reviews,
ports and fuzzing, and `tests/fuzz/` for the differential fuzzer (`test_fuzz_smoke.py` in CI,
`python -m tests.fuzz.run` for long runs).

---

[← The design of expex](design.md) · [Contents](README.md)

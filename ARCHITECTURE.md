# How typax is put together

Everything starts from `@tx.fn` (`_fn.py`); `@tx.lift` and `@tx.random` set flags
on the same `Fn` object, so they stack in any order. `@tx.jit` is `jax.jit` with
`static_argnames` read from the signature. Gradients and remat are JAX's own. Each module has one job:

| Module | Job |
|---|---|
| `_types.py` | The annotation objects users write: `f32[B, T]`, `Static[...]`, `Key`, `Batch[...]`, `Dim`. |
| `_diff.py` | `tx.diff`: two models or checkpoints compared field by field, as text or HTML. |
| `_dims.py` | Dimensions (`B`, `T + 1`, `*Rest`), binding sizes during a call (`Env`), printing values as `f32[3, 4]`. |
| `_spec.py` | Turning annotations into specs and matching values against them, structs included. |
| `_summary.py` | `tx.summary`: one row per module with parameters and FLOPs, as formulas and numbers. |
| `_sym.py` | Size arithmetic written in annotations (`f32[T + 1]`), read from their text. |
| `_expr.py` | Size arithmetic and constraints (`"T + 1"`, `"D % heads == 0"`). |
| `_fn.py` | `@tx.fn`, `@tx.lift`, `@tx.random`, `@tx.jit`: binds arguments, checks them and the result, runs the body. |
| `_axes.py` | Axes by name: `x.sum(T)`, `axis=T`, resolved from annotations or sizes. |
| `_lift.py` | `f.lift(...)`: mapping over extra leading axes. |
| `_messages.py` | Error messages, hints and the notes added to exceptions. |
| `_consts.py` | `tx.eps`, `tx.tiny`, `tx.huge`: dtype limits resolved against the array they meet. |
| `_cost.py` | `tx.cost`: FLOPs and parameters as polynomials in the sizes, from a symbolic trace. |
| `_init.py` | `Module.init(key, D=64)`: parameters built from the field annotations. |
| `_io.py` | `tx.save` / `tx.load`: one `.npz` per model, checked against the annotations on load. |
| `_shard.py` | `tx.shard`, `tx.shardings`: shardings from axis names mapped to mesh axes. |
| `_struct.py` | `@tx.struct`: modules registered as pytrees whose methods are `@fn`s. |
| `_rng.py` | Implicit random keys (`tx.key()`). |
| `_convert.py` | Rewrites a function's source so plain Python works on traced values. |
| `_runtime.py` | The namespace the rewritten code calls into, backed by: |
| `_flow.py` | `if` / `while` and the machinery they share, |
| `_loops.py` | `for`, iterables, `tx.rows` / `tx.stack`, |
| `_ops.py` | operators, in-place updates, `assert` / `raise` / `print` / `min` / `max`, checked locals. |
| `_check.py` | The prover (`tx.prove`) and the command line. |
| `debug.py` | `find_nan`, `shapes`, `memory`, `summary`, `timing`. |
| `pytest_plugin.py` | `pytest --typax pkg`. |

A call to an `@fn` goes `_fn.Fn.__call__` → (`_lift` if lifted) → `_run`
(opens a key scope if random) → `_run_checked`, which matches arguments with `_spec`, runs the
converted body (`_convert` output calling `_runtime`), and matches the result.

Tests: `tests/` per feature, `tests/regressions/` for bugs found by reviews,
`tests/ports/` for ports checked against their originals.

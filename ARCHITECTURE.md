# How typax is put together

Everything starts from `@fn` (`_fn.py`). Each module has one job:

| Module | Job |
|---|---|
| `_types.py` | The annotation objects users write: `f32[B, T]`, `Static[...]`, `Key`, `Batch[...]`, `Dim`. |
| `_dims.py` | Dimensions (`B`, `T + 1`, `*Rest`), binding sizes during a call (`Env`), printing values as `f32[3, 4]`. |
| `_spec.py` | Turning annotations into specs and matching values against them, structs included. |
| `_expr.py` | Size arithmetic and constraints (`"T + 1"`, `"D % heads == 0"`). |
| `_fn.py` | The `@fn` wrapper: binds arguments, checks them and the result, runs the body. |
| `_jit.py` | How `@fn(jit=True)` splits arguments into traced arrays and static values. |
| `_lift.py` | `f.lift(...)`: mapping over extra leading axes. |
| `_messages.py` | Error messages, hints and the notes added to exceptions. |
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

A call to an `@fn` goes `_fn.Fn.__call__` → (`_jit` if compiled, `_lift` if
lifted) → `_run_checked`, which matches arguments with `_spec`, runs the
converted body (`_convert` output calling `_runtime`), and matches the result.

Tests: `tests/` per feature, `tests/regressions/` for bugs found by reviews,
`tests/ports/` for the PPO and bofus ports checked against their originals.

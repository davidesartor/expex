# barnacle

Shape-typed functions for JAX. You write the shapes in the signature, and one
decorator puts them to work:

```python
import jax
import jax.numpy as jnp
import barnacle as bn
from barnacle import fn, f32, i32, Static, Key

@fn
def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    return jnp.einsum("btd,bsd->bts", q, k)
```

From that signature, `@fn` gives you:

| | |
|---|---|
| **Checked calls** | Arguments and results are matched against the annotations, with errors that name the axis and where its size came from. |
| **Shape proofs** | `bn.prove(attend)` traces the function with *symbolic* sizes and shows the declared output shape holds for every `B`, `T`, `D`. No test data needed. |
| **Plain control flow** | `if`, `while` and `for` on traced values become `lax.cond`, `lax.while_loop`, `lax.scan` or `vmap` automatically. |
| **Vectorising** | `attend.lift(...)` maps over any extra leading axes, worked out from the declared ranks. `attend.batched(chunk=64)` does it without recompiling per batch size. |
| **Implicit random keys** | `@fn(random=True)` functions get a fresh key from `bn.key()`, safely under `jit`. |
| **Debugging** | Retrace explanations, a NaN finder that points at the source line, program size and timing breakdowns. |

It is a plain Python library (3.12+). The annotations are ordinary names, so
pyright and mypy accept them without plugins.

```
pip install -e .          # from a checkout
python examples/tour.py                        # see everything run
python -m barnacle examples/transformer.py     # prove a small transformer's shapes
```

## Annotations

```python
from typing import Any, Literal, NamedTuple
from barnacle import Dim, Etc, Float, Int

class RGB(Dim, size=3): ...      # a named fixed size
class Hidden(Dim): ...           # a named axis shared across functions

class Params[D](NamedTuple):     # dataclasses work too
    w: f32[D, D]
    b: f32[D]

@fn
def example[*Batch, H, W](
    img: f32[*Batch, H, W, RGB],     # *Batch: zero or more leading axes
    idx: Int[*Batch],                # any integer dtype
    feats: Float[Etc, Hidden],       # Etc: any leading axes, unchecked
    pair: f32[Literal[2], Any],      # a literal size; Any: one axis of any size
    temperature: f32,                # a scalar (also accepts a Python number)
    p: Params[Hidden],               # fields are checked, D is bound to Hidden
    mask: f32[H, W] | None = None,
    steps: Static[int] = 1,          # a trace-time constant, not an array
) -> tuple[f32[*Batch, H, W], i32]:
    return img.mean(-1), jnp.int32(steps)
```

- **Dimensions** are the function's own type parameters (`def f[B, T](...)`).
  Within one call each name has one size; different calls are independent.
- **Dtypes**: `f16 f32 f64 i8 i16 i32 i64 u8 u16 u32 u64 c64 c128 Bool`, and
  the categories `Float Int UInt Complex Inexact Num Arr` (`Arr` is any dtype).
- **Containers**: `tuple[...]`, `list[X]`, `dict[str, X]`, `X | None`, unions,
  NamedTuples and dataclasses with annotated fields.
- **Derived sizes and constraints** go on the decorator, because type checkers
  reject arithmetic in annotations:

```python
@fn(U="T + 1", where="T >= 2")
def pad[B, T, U](x: f32[B, T]) -> f32[B, U]:
    return jnp.pad(x, ((0, 0), (0, 1)))

# Static int parameters can be used by name:
@fn(H="heads", K="D // heads", where="D % heads == 0", examples={"heads": [1, 2]})
def split_heads[T, D, H, K](x: f32[T, D], heads: Static[int]) -> f32[H, T, K]:
    t, d = x.shape
    return x.reshape(t, heads, d // heads).transpose(1, 0, 2)
```

Methods work too. If the class is a dataclass or NamedTuple with annotated
fields, `self` is checked and the class's dimensions are shared with the
method, which also lets the prover check it without an instance:

```python
from dataclasses import dataclass

@dataclass
class Linear[I, O]:
    w: f32[O, I]
    b: f32[O]

    @fn
    def __call__[B](self, x: f32[B, I]) -> f32[B, O]:
        return x @ self.w.T + self.b

assert bn.prove(Linear.__call__).ok
```

## Checked calls

```python
x = jnp.ones((2, 3, 4))
try:
    attend(x, jnp.ones((5, 3, 4)))
except bn.ShapeError as e:
    print(e)
```

```
attend(): argument `k` does not match its annotation.

    declared  k: f32[B, T, D]
    received     f32[5, 3, 4]

    B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5.
```

Checks are shape and dtype comparisons, so under `jit` they run once per
trace and cost nothing in the compiled function. Under `vmap` and `grad` they
see the per-example shapes, which is what the annotations describe. Any
exception passing through an `@fn` also gets a note with the argument shapes
of each frame. Switch checks off with `@fn(checked=False)` or
`bn.config.checked = False`.

## Shape proofs

```python
@fn
def diff[T, D](xs: f32[T, D]) -> f32[T, D]:
    return xs[1:] - xs[:-1]

print(bn.prove(diff))
```

```
diff: failed
    diff(): the return value does not match its annotation.

        declared  -> f32[T, D]
        returned     f32[T - 1, D]

        axis 0 of `return` should be T, but is T - 1.
    confirmed at sizes D=2, T=3.
```

The prover builds abstract inputs from the annotations alone, using JAX's
symbolic dimensions, and traces the function once. Nothing is compiled and no
data is needed. Calls to other `@fn` functions are checked at the call site,
so a proof covers the whole call tree.

Run it over a package:

```
python -m barnacle mypackage            # or: barnacle mypackage
pytest --barnacle mypackage             # one test per @fn
```

or set `barnacle = ["mypackage"]` under `[tool.pytest.ini_options]`.

Each function gets one verdict:

| Verdict | Meaning |
|---|---|
| `proved` | The output matches the declaration for all sizes, and the function also traced at several concrete sizes (including size 1). |
| `spot-checked` | It holds at the concrete sizes tried, but could not be shown symbolically (for example, Python control flow that depends on a shape). |
| `failed` | The declared and actual shapes differ. |
| `error` | The function raised while being traced. If only some sizes fail, say which are supported with `where=`. |
| `unchecked` | The prover cannot invent an argument: a `Static` or unannotated parameter with no default. Give `@fn(examples={"n": [1, 4]})`. |

What a proof covers: output shapes and dtypes, and shape agreement at every
operation and `@fn` call inside. It does not cover values, and it is only as
strict as JAX's own shape rules; where JAX defers a check for symbolic sizes
(`reshape`), the prover checks it itself. Methods are proved on an instance:
`bn.prove(model.call)`.

## Plain control flow

```python
@fn
def collatz(n: i32) -> i32:
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

print(jax.vmap(collatz)(jnp.array([6, 7, 27])))     # [  8  16 111]

@fn
def rnn[T, D](xs: f32[T, D], w: f32[D, D]) -> f32[T, D]:
    h = jnp.zeros(xs.shape[1])
    hs = []
    for x in xs:
        h = jnp.tanh(w @ h + x)
        hs.append(h)
    return jnp.asarray(hs)

jax.jit(rnn)(jnp.ones((100, 3)), jnp.eye(3))
print(rnn.flow)        # {<line>: 'for -> lax.scan (carries h)'}
```

`@fn` re-compiles the function so that each statement decides *when it runs*:

- condition or iterable is an ordinary Python value → exactly the original
  Python behaviour (this includes everything in eager mode, `Static`
  arguments and shapes);
- it is traced → `lax.cond`, `lax.while_loop`, or for a `for` loop over an
  array / `range` / `enumerate` / `zip`: `vmap` if no iteration reads what an
  earlier one wrote, otherwise `lax.scan`. A traced `range(n)` becomes a
  `while_loop`.

`ys.append(v)` inside a loop is how you collect per-iteration results; after a
staged loop `ys` is a read-only sequence backed by one stacked array
(`jnp.asarray(ys)` is free). `and`, `or`, `not` and `a if c else b` are
converted too. The 100-step loop above traces to a handful of operations
instead of hundreds, which is what keeps compile times down.

Rules and limits:

- Under a traced `if`, both branches must give each variable the same shape
  and dtype. Errors name the variable.
- Under a traced loop, carried variables must keep their shape and dtype.
- Loops containing `break`, `continue` or `return`, and loops with an `else`
  clause, are left as plain Python (so they need concrete conditions).
- If a `for` loop cannot be staged (the index is used as a Python int, a
  shape changes between iterations, ...) it is unrolled as in plain JAX, and
  `f.flow` says why.
- Functions must be free of side effects, as JAX already requires: a staged
  body is traced once, not run once per iteration.
- Opt out per function with `@fn(convert=False)`, or keep conversion but
  unroll loops with `bn.config.lower_loops = False`. `f.converted_source`
  shows the generated code; `f.convert_note` says why a function was skipped.

## Vectorising

```python
@fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return jnp.tanh(x @ w)

xs, w = jnp.ones((32, 5, 3)), jnp.ones(3)
score.lift(xs, w).shape                              # (32, 5): xs mapped, w shared
score.lift(jnp.ones((8, 1, 5, 3)), jnp.ones((7, 3))).shape   # (8, 7, 5): batch axes broadcast
```

`lift` is explicit on purpose: a plain call with a rank-3 `xs` is still an
error. Batch axes must lead. A single `Key` argument is split into one key
per example, so examples never share randomness.

`jit` compiles per exact shape, so every new batch size normally recompiles.
Because lifted examples are independent, padding a batch cannot change the
results, which makes compile-once batching safe:

```python
run = score.batched(chunk=64)      # compiled once, for batches of 64
for n in (5, 64, 1000):
    run(jnp.ones((n, 5, 3)), w)    # chunked and padded as needed
print(run.traces)                  # 1
```

`batched(sizes=(8, 64, 512))` compiles one version per size and uses the
largest that fits, wasting less work on small batches.

## Implicit random keys

```python
@fn(random=True)
def dropout[D](x: f32[D], rate: Static[float] = 0.5) -> f32[D]:
    keep = jax.random.bernoulli(bn.key(), 1 - rate, x.shape)
    return jnp.where(keep, x / (1 - rate), 0.0)

bn.seed(0)
dropout(jnp.ones(6))     # different on every call, reproducible after bn.seed
```

Each `bn.key()` call returns a new key. Random functions calling random
functions derive their keys from the caller's, and each branch and loop
iteration gets its own, so eager, `jit`, `scan` and `vmap` runs draw the same
numbers.

The global key is Python state, so it must not be read while tracing: it
would be frozen into the compiled function. barnacle enforces this:

```python
@fn(jit=True, random=True)          # barnacle owns the jit and passes a fresh
def noisy[D](x: f32[D]) -> f32[D]:  # key in as an argument on every call
    return x + jax.random.normal(bn.key(), x.shape)

@jax.jit
def step(x, key):                   # or bring your own key
    with bn.rng(key):
        return dropout(x)
```

Calling a random function under a plain `jax.jit` with no key in scope raises
a `StageError` explaining these two options. Explicit `Key` parameters keep
working as usual.

## Debugging

```python
from barnacle import debug

@fn
def entropy[N](logits: f32[N]) -> f32:
    p = jax.nn.softmax(logits)
    return -(p * jnp.log(p)).sum()

print(debug.find_nan(entropy, jnp.array([0.0, 200.0, -200.0])))
```

```
NaN first appears at model.py:4 (entropy)
    > return -(p * jnp.log(p)).sum()
  operation: mul
  inputs:
    #0: f32[3]  min=0  max=1  (2 zero, of 3)
    #1: f32[3]  min=0  max=0  (2 -inf, 1 zero, of 3)
  ...
  infinities appeared earlier at:
    model.py:4 (entropy)  (log)  > return -(p * jnp.log(p)).sum()
```

`find_nan` replays the traced program one operation at a time. Inside loops
it reports the iteration (`scan iteration 37`), and it works through `jit`,
`grad` and converted control flow.

```python
print(debug.summary(entropy, jnp.ones(4)))   # op counts by kind and source line, largest
                                             # intermediates, lints (unrolled loops, big
                                             # captured constants, stray 64-bit values)
print(debug.timing(entropy, jnp.ones(4)))    # trace | compile | run, separately
```

`bn.compile_cache("~/.cache/myproject")` turns on JAX's on-disk compilation
cache, so that later runs of the same program skip compilation.

With `@fn(jit=True)`, `Static` parameters become static jit arguments and
every trace is recorded:

```python
@fn(jit=True)
def scale[N](x: f32[N], by: f32) -> f32[N]:
    return x * by

scale(jnp.ones(3), 2.0); scale(jnp.ones(4), 2.0)
print(scale.explain_retrace())
# scale(): trace #2 differs from trace #1 in:
#     x: f32[3]  ->  f32[4]
```

A `RetraceWarning` with that explanation is issued once a function has been
traced more than `bn.config.retrace_warn` times (default 4).

Common tracer errors raised inside an `@fn` (`TracerBoolConversionError` and
friends) get a note quoting the offending line and listing the fixes.

## Static type checking

For pyright and mypy, `f32[B, T]` is `jax.Array` and `Static[int]` is `int`,
so annotated code checks cleanly, wrong argument types and counts are still
caught, and `.lift` keeps the function's signature. What checkers cannot see:

- Shapes. They are checked by barnacle (at call time and by the prover), not
  statically.
- Sizes must be names: use `Literal[3]` or a `Dim` subclass for constants,
  and `@fn(U="T + 1")` for arithmetic.
- pyright warns that a type parameter "appears only once" for dimensions used
  in a single position (`reportInvalidTypeVarUse`); that warning can be
  disabled.

## Limitations

- Requires Python 3.12+ (for `def f[B, T]`) and a recent JAX.
- Conversion needs the function's source, so functions defined with `exec` or
  typed at a bare `python` prompt are not converted (`f.convert_note` says so).
- `debug.find_nan` and the prover's `reshape` check read JAX's internal
  program representation; they are tested against JAX 0.12 and may need
  adjusting for other versions.
- `lift` and `batched` need batch axes in front, and cannot be used on
  arguments declared with a variadic shape.
- A `spot-checked` verdict is evidence, not proof.

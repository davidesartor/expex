# typax

Shape-typed JAX functions in plain Python. Write the shapes in the signature
and decorate with `@fn`:

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import typax as tx
from typax import fn, f32, i32

@fn
def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    return jnp.einsum("btd,bsd->bts", q, k)
```

That signature then does four jobs:

- **Checks** every call, with errors that name the axis and where its size came from.
- **Proves** the shapes for all sizes: `tx.prove(attend)`, with no test data.
- **Vectorises**: `attend.lift(...)` maps over any extra leading axes.
- **Compiles** with `@fn(jit=True)`, with no `static_argnames`.

`@tx.struct` does the same for classes: typed, immutable modules that JAX can
trace.

Inside `@fn`, plain Python works on traced values: `if`, `for`, `while` (with
`break` / `continue` / `return`), `x[i] += 1`, `assert`, `print`, `min` / `max`.

```
pip install -e .                               # Python 3.12+, recent JAX
python examples/tour.py                        # see everything run
python -m typax examples/transformer.py        # prove a small transformer's shapes
```

## Annotations

```python
from typing import Literal, NamedTuple
from typax import Dim, Float, Int, Key

class RGB(Dim, size=3): ...          # a named fixed size

class Params[D](NamedTuple):         # dataclasses work too
    w: f32[D, D]
    b: f32[D]

@fn
def example[*Batch, H, W](
    img: f32[*Batch, H, W, RGB],     # *Batch: zero or more axes
    idx: Int[*Batch],                # any integer dtype
    p: Params[W],                    # fields are checked, D is bound to W
    pair: f32[Literal[2], W],        # a literal size
    scale: f32,                      # a traced scalar (also takes a Python number)
    steps: int = 1,                  # a plain Python type: static, not traced
    key: Key | None = None,
) -> f32[*Batch, H]:
    return img.mean((-1, -2)) * scale * steps
```

| Kind | Annotations |
|---|---|
| Arrays | `f16 f32 f64 i8 … i64 u8 … u64 c64 c128`, categories `Float Int UInt Bool Complex Num Arr` |
| Sizes | the function's type parameters (`def f[B, T]`), `Literal[3]`, `class RGB(Dim, size=3)`, `*Rest`, `Any` |
| Containers | `tuple[...]`, `list[X]`, `dict[str, X]`, `X \| None`, NamedTuples and dataclasses, generic ones too (`class Mix[*B, M, D]`) |
| Stacked | `Batch[Transition, T, N]`: a `Transition` whose arrays all have leading axes `T, N` |
| Static | `int`, `float`, `bool`, `str`, `Literal`, enums, callables, configuration NamedTuples. In a struct field, wrap numbers in `Static[...]` |

Each size name has one value per call. Arithmetic on sizes and constraints go
on the decorator (type checkers reject them in annotations), and may use
static parameters by name:

```python
@fn(U="T + 1")
def pad[B, T, U](x: f32[B, T]) -> f32[B, U]:
    return jnp.pad(x, ((0, 0), (0, 1)))

@fn(H="heads", K="D // heads", where="D % heads == 0")
def split_heads[T, D, H, K](x: f32[T, D], heads: int) -> f32[H, T, K]:
    t, d = x.shape
    return x.reshape(t, heads, d // heads).transpose(1, 0, 2)
```

To pyright and mypy, `f32[B, T]` is `jax.Array`, so annotated code type-checks
without plugins.

## Modules

```python
from typing import Callable

@tx.struct
class Block[D]:
    w: f32[D, D]
    b: f32[D]
    act: Callable = jax.nn.gelu          # plain Python types are static fields

    def __init__(self, key, d: int):
        self.w = 0.1 * jr.normal(key, (d, d))
        self.b = jnp.zeros(d)

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        return x + self.act(x @ self.w + self.b)

@tx.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __init__(self, key, d: int, depth: int):
        self.blocks = [Block(k, d) for k in jr.split(key, depth)]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for block in self.blocks:      # identical blocks: one lax.scan, not 24 copies
            x = block(x)
        return x

model = Stack(jr.key(0), d=16, depth=24)
y = jax.jit(lambda m, x: m(x))(model, jnp.ones((4, 16)))
grads = jax.grad(lambda m: m(jnp.ones((4, 16))).sum())(model)   # a Stack too
```

A struct is a frozen dataclass registered as a pytree:
- Fields with array annotations are traced; fields with plain Python types are static.
- `__init__` is optional: without one you get the dataclass constructor; with
  one, its fields are assigned as usual and the result is checked against the
  annotations. JAX and `.replace(...)` rebuild objects without calling it.
- Every method is an `@fn` that checks `self`'s fields and shares the class's
  sizes, so `tx.prove(Block.__call__)` works without an instance.
- `model.lift(x)` maps `__call__` over extra leading axes, and
  `@tx.struct(jit=True)` compiles every method.
- `@tx.struct(where="D % heads == 0")` states an invariant. It is checked
  whenever an instance is built or passed in, and assumed by every function
  that receives one; a method's derived sizes can name its int fields
  (`@fn(H="heads")`).

A `for` loop over a list of pytrees with identical structure and shapes, such
as layers, is stacked and lowered to `lax.scan`. Deep models therefore compile
in the time one layer takes.

## Checks and proofs

```python
try:
    attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))
except tx.ShapeError as e:
    print(e)
```

```
attend(): argument `k` does not match its annotation.

    declared  k: f32[B, T, D]
    received     f32[5, 3, 4]

    B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5.
```

Checks are shape comparisons, so under `jit` they run once per trace. Annotated
local variables (`h: f32[B, D] = ...`) are checked too.

`tx.prove` traces a function with symbolic sizes, so a passing proof holds for
every size:

```python
@fn
def diff[T, D](xs: f32[T, D]) -> f32[T, D]:
    return xs[1:] - xs[:-1]

print(tx.prove(diff))   # failed: returned f32[T - 1, D]
```

Static parameters without defaults are tried at a few values (`int`: 1 and 2,
`bool`: both, `Literal` and enums: every value); others need
`@fn(examples={"name": [...]})`. Run it over a package with
`python -m typax pkg` or `pytest --typax pkg`.

## Plain Python, staged

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

Each statement decides when it runs. On Python values it behaves exactly as
written; on traced values `if` becomes `lax.cond`, `while` becomes
`lax.while_loop`, and a `for` over an array or `range` becomes `vmap` when
iterations are independent and `lax.scan` otherwise. A 1000-step loop compiles
in 0.06 s instead of 8 s unrolled.

- `ys.append(v)` collects per-iteration values; `tx.stack(ys)` stacks pytrees.
  `for row in tx.rows(tree)` iterates a pytree; `tx.rows(tree, chunk=k)` runs
  `k` independent iterations at a time.
- `x[i] = v` and `x[i] += v` on a JAX array become `x.at[i]...`;
  `s.field = v` replaces a NamedTuple. `assert` and `raise` in a traced branch
  fire when the compiled code reaches them.
- Gradients through a traced `if` are NaN-free where a branch is not taken, so
  `jnp.where(x > 0, jnp.sqrt(x), 0.)` with its sanitised inputs becomes a plain `if`.
- `f.flow` shows how each statement ran; `@fn(convert=False)` opts out.

Left as plain Python (so they need concrete values, and loops unroll): code
inside `try` / `with`, loops with an `else`, in-place updates of anything but a
local variable, and variables shared with a nested function. A variable
assigned on only one traced path reads as zeros on the other.

## Vectorising and compiling

```python
@fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return jnp.tanh(x @ w)

score.lift(jnp.ones((32, 5, 3)), jnp.ones(3)).shape                # (32, 5)
score.lift(jnp.ones((8, 1, 5, 3)), jnp.ones((7, 3))).shape         # (8, 7, 5)

@fn(lift=True)                       # written for one element, takes batches
def safe_log(x: f32) -> f32:
    if x > 0:
        return jnp.log(x)
    return -jnp.inf

safe_log(jnp.ones((4, 5)))
```

`lift` aligns batch axes on the left and broadcasts them like NumPy. A plain
call with extra axes stays an error unless the function says `lift=True`.

`@fn(jit=True)` traces arrays and treats every other value as static, in
arguments and results, including functions and strings inside a NamedTuple or
dataclass. Each retrace is recorded with the reason:

```python
@fn(jit=True)
def scale[N](x: f32[N], by: f32) -> f32[N]:
    return x * by

scale(jnp.ones(3), 2.0)
scale(jnp.ones(4), 2.0)
print(scale.explain_retrace())       # x: f32[3]  ->  f32[4]
```

Compile time does not grow with batch size (measured from 1 to 4096) as long as
loops are staged. `tx.compile_cache(path)` keeps compiled functions on disk
across runs.

`@fn(remat=True)` recomputes a function's intermediates in the backward pass
instead of storing them (`remat="dots"` keeps matrix products). Put it on a
layer: with the layer scan, a 12-layer backward pass went from 57 MB to 22 MB.

## Random keys

```python
@fn(random=True)
def dropout[D](x: f32[D], rate: float = 0.5) -> f32[D]:
    keep = jr.bernoulli(tx.key(), 1 - rate, x.shape)
    return jnp.where(keep, x / (1 - rate), 0.0)

tx.seed(0)
dropout(jnp.ones(6))         # different on every call, reproducible after tx.seed
```

Each `tx.key()` returns a new key. Nested calls, branches and loop iterations
derive their own keys, so eager, `jit`, `scan` and `vmap` runs draw the same
numbers. Under a plain `jax.jit`, put a key in scope with `with tx.rng(key):`,
or use `@fn(jit=True, random=True)`, which passes a fresh key on every call.
`f.lift` gives each element its own key.

## Debugging

```python
@fn
def entropy[N](logits: f32[N]) -> f32:
    p = jax.nn.softmax(logits)
    return -(p * jnp.log(p)).sum()

print(tx.debug.find_nan(entropy, jnp.array([0.0, 200.0, -200.0])))
```

```
NaN first appears at model.py:4 (entropy)
    > return -(p * jnp.log(p)).sum()
  operation: mul
  inputs:
    #0: f32[3]  min=0  max=1  (2 zero, of 3)
    #1: f32[3]  min=0  max=0  (2 -inf, 1 zero, of 3)
```

`find_nan` replays the program one operation at a time, reporting the loop
iteration or branch too.

```python
@fn
def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    scores = q @ k.swapaxes(-1, -2)
    return jax.nn.softmax(scores, axis=-1)

print(tx.debug.shapes(attention))     # no arguments: symbolic sizes
```

```
   ... | def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
   ... |     scores = q @ k.swapaxes(-1, -2)            # scores: f32[B, T, T]
   ... |     return jax.nn.softmax(scores, axis=-1)     # -> f32[B, T, T]
```

`tx.debug.shapes(f, *args)` does the same on concrete arguments, for any
function. `tx.debug.memory(jax.grad(loss), params, x)` reports XLA's memory
numbers for the compiled program and which values, created on which lines,
make up the peak. `tx.debug.summary(f, ...)` counts operations by kind
and source line and flags unrolled loops and large constants;
`tx.debug.timing(f, ...)` splits trace, compile and run time. `f.info()` shows
everything typax knows about a function, and `tx.show(tree)` prints a pytree as
shapes.

## Ported projects

Two existing codebases rewritten in this style, checked against the originals:

- **[PureJaxRL](https://github.com/luchris429/purejaxrl) PPO** (`examples/ppo/`):
  plain loops instead of four nested `lax.scan`s, `if done:` for the
  auto-reset. Identical to the original for the first 21 of 976 updates, then
  drifting as float32 programs do; same speed; all shapes proved.
- **[bofus](https://github.com/davidesartor/bofus)** (`examples/bofus/`): the
  original's 74 tests pass against the port, and 164 arrays of values and
  gradients match it to 1e-13 (2.6e-7 for the L-BFGS acquisition optimizer).
  42 `jnp.where` calls became 13; speed is on par.

## Limitations

- Conversion needs the function's source, so functions defined with `exec` or
  at a bare `python` prompt run unconverted (`f.convert_note` says why).
- `debug.find_nan` and parts of the prover read JAX internals; they are tested
  on JAX 0.12.
- `lift` needs batch axes in front, and not on arguments with a `*Rest` shape.
- A `spot-checked` proof verdict is evidence, not proof.

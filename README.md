# typax

Shape-typed JAX in plain Python. The idea: be explicit about the shapes of
arrays, and of collections of arrays, and the library does the rest by reading
your annotations. Put the shapes in the signature:

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

and that signature is checked on every call, proved for all sizes without test
data, and used to vectorise the function. Inside `@fn`, plain `if` / `for` /
`while` work on traced values. It is ordinary JAX underneath: `jax.jit`,
`jax.grad`, `jax.vmap` and `jax.checkpoint` take an `@fn` like any function.

```
pip install -e .                               # Python 3.12+, JAX 0.12+
python examples/tour.py                        # see everything run
python -m typax examples/transformer.py        # prove a small transformer's shapes
```

| | |
|---|---|
| [Checked calls](#checked-calls) | errors that name the axis and where its size came from |
| [Axes by name](#axes-by-name) | `x.sum(T)`, `softmax(x, axis=T)`, `x.transpose(B, D, T)` |
| [Proofs](#proofs-without-data) | `tx.prove(f)`: the shapes hold for every size |
| [Plain Python control flow](#plain-python-on-traced-values) | `if` → `lax.cond`, `for` → `lax.scan` / `vmap`, `while` → `lax.while_loop` |
| [Vectorising](#vectorising) | `f.lift(...)` and `@tx.lift` map over extra leading axes |
| [Modules](#modules) | `@tx.struct`: typed, immutable classes that JAX can trace |
| [Random keys](#random-keys) | `@tx.random` and `tx.key()`, no key threading |
| [Compiling and gradients](#compiling-and-gradients) | `@tx.jit` reads static arguments from the signature; `jax.grad`, `jax.checkpoint` unchanged |
| [Debugging](#debugging) | where a NaN appeared, shapes line by line, memory, timing |

## Checked calls

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

    attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]
    defined at model.py:7
```

The result is checked too, and so are annotated local variables
(`h: f32[B, D] = ...`). Under `jax.jit` the checks run once, while tracing.

Sizes that depend on others, and constraints, go on the decorator:

```python
@fn(U="T + 1")
def pad[B, T, U](x: f32[B, T]) -> f32[B, U]:
    return jnp.pad(x, ((0, 0), (0, 1)))

@fn(H="heads", K="D // heads", where="D % heads == 0")
def split_heads[T, D, H, K](x: f32[T, D], heads: int) -> f32[H, T, K]:
    t, d = x.shape
    return x.reshape(t, heads, d // heads).transpose(1, 0, 2)

split_heads(jnp.ones((5, 8)), heads=2).shape     # (2, 5, 4)
```

## Axes by name

The sizes in the signature are names, and inside the function they name axes:

```python
@fn
def pool[B, T, D](x: f32[B, T, D]) -> f32[B, D]:
    return x.mean(T)

@fn
def summarise[B, T, D](x: f32[B, T, D]) -> f32[D, B]:
    h = jnp.tanh(x)
    return h.max(T).transpose(D, B)      # h is unannotated: found by size

pool(jnp.ones((2, 3, 4))).shape          # (2, 4)
summarise(jnp.ones((2, 3, 4))).shape     # (4, 2)
print(tx.prove(summarise))               # proved for all sizes
```

An axis name works as a positional argument of array methods (`sum`, `mean`,
`max`, `argmax`, `transpose`, `swapaxes`, ...) and as `axis=` / `axes=` of any
function (`jax.nn.softmax(x, axis=T)`). `Dim` classes and variadics (`*B`) are
names too. The position comes from the array's annotation when it has one, a
parameter or an annotated local (`h: f32[B, T, D] = ...`), and otherwise from
its sizes. When two axes have the same size and nothing says which is which,
the call is an error asking for an annotation.

## Proofs without data

`tx.prove` traces a function with symbolic sizes, so a passing proof holds for
every size:

```python
@fn
def diff[T, D](xs: f32[T, D]) -> f32[T, D]:
    return xs[1:] - xs[:-1]

print(tx.prove(diff))
```

```
diff: failed
    diff(): the return value does not match its annotation.

        declared  -> f32[T, D]
        returned     f32[T - 1, D]

        axis 0 of `return` should be T, but is T - 1.
    ...
```

`tx.prove(split_heads)` proves it for every `D` and `heads` with
`D % heads == 0`. Parameters such as `heads: int` are tried at a few values
(ints at 1 and 2, bools both ways, every `Literal` and enum value); others take
`@fn(examples={"name": [...]})`. Prove a whole package with `python -m typax pkg`
or `pytest --typax pkg`.

## Plain Python on traced values

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
```

```python
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

On Python values every statement runs exactly as written. On traced values:

- `if` becomes `lax.cond`, and `while` becomes `lax.while_loop`, with `break`,
  `continue` and `return` inside.
- A `for` over an array or a `range` becomes `vmap` when the iterations are
  independent, `lax.scan` otherwise. A 1000-step loop compiles in 0.06 s
  instead of 8 s unrolled.
- `x[i] = v` and `x[i] += v` become `x.at[i].set(v)` / `.add(v)`.
- `assert`, `raise` and `print` fire when the compiled code reaches them.
- Gradients through a traced `if` are NaN-free where a branch is not taken,
  so the `jnp.where`-with-sanitised-inputs trick becomes a plain `if`.

`f.flow` says how each statement ran; `@fn(convert=False)` opts out.

## Vectorising

A function's annotations say which axes are its own, so extra leading axes can
be mapped over without `in_axes`:

```python
@fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return jnp.tanh(x @ w)

score.lift(jnp.ones((32, 5, 3)), jnp.ones(3)).shape                # (32, 5)
score.lift(jnp.ones((8, 1, 5, 3)), jnp.ones((7, 3))).shape         # (8, 7, 5)
```

Batch axes broadcast like NumPy. `@tx.lift` makes plain calls do the same, so
a function written for one element takes batches:

```python
@tx.lift
def safe_log(x: f32) -> f32:
    if x > 0:
        return jnp.log(x)
    return -jnp.inf

safe_log(jnp.array([[1.0, -1.0], [jnp.e, 0.0]]))     # [[0, -inf], [1, -inf]]
```

Compile time does not depend on the batch size: one example is traced.

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

- A struct is a frozen dataclass and a pytree. Fields with array annotations
  are traced, fields with plain Python types are static.
- `__init__` is optional. With one, assign the fields as usual. The result is
  checked against the annotations, and JAX and `.replace(...)` rebuild objects
  without calling it.
- Every method is an `@fn` sharing the class's sizes, so
  `tx.prove(Block.__call__)` works without an instance.
- A loop over identical layers is one `lax.scan`, so a deep model compiles in
  the time one layer takes.
- `model.lift(x)` maps `__call__` over extra axes. Other decorators go on the
  methods themselves.

Invariants state what every instance satisfies. They are checked when one is
built or passed in, and the prover assumes them. A method's sizes can name the
struct's int fields:

```python
@tx.struct(where="D % heads == 0")
class Heads[D]:
    w: f32[D, D]
    heads: int = 4

    @fn(H="heads", K="D // heads")
    def split[T, H, K](self, x: f32[T, D]) -> f32[H, T, K]:
        t, d = x.shape
        return (x @ self.w).reshape(t, self.heads, d // self.heads).transpose(1, 0, 2)

Heads(jnp.eye(8)).split(jnp.ones((5, 8))).shape      # (4, 5, 2)
print(tx.prove(Heads.split).ok)                      # True
```

## Random keys

```python
@tx.random
@fn
def dropout[D](x: f32[D], rate: float = 0.5) -> f32[D]:
    keep = jr.bernoulli(tx.key(), 1 - rate, x.shape)
    return jnp.where(keep, x / (1 - rate), 0.0)

tx.seed(0)
dropout(jnp.ones(6))                        # different on every call
dropout(jnp.ones(6), key=jr.key(1))         # or choose the key
dropout.lift(jnp.ones((4, 6)))              # each row gets its own key
```

Every `tx.key()` is a new key. Nested calls, branches and loop iterations
derive their own, so eager, `jit`, `scan` and `vmap` runs draw the same
numbers. Under `jax.jit`, pass the key: the global key is Python state and
would be frozen into the compiled function.

## Compiling and gradients

`@tx.jit` is `jax.jit` with the static arguments read from the signature:
plain Python types (`int`, `str`, tuples, callables, enums) and `Static[...]`
are static; arrays and `float` numbers are traced, so a new learning rate does
not recompile.

```python
@tx.jit
@fn(K="N * n")
def repeat[N, K](x: f32[N], n: int, scale: float) -> f32[K]:
    return jnp.tile(x, n) * scale

repeat(jnp.ones(3), 4, 0.5).shape                       # (12,)
noisy = tx.jit(dropout)                                 # a fresh key on every call
noisy(jnp.ones(6)), noisy(jnp.ones(6), key=jr.key(2))
```

Everything else is JAX's own. `jax.jit(f, static_argnames=...)` works as usual
(a random function then takes `key=`), and so do `jax.grad`, `jax.vmap` and
`jax.checkpoint`:

```python
layer = jax.checkpoint(Block.__call__)                   # remat a layer
jax.grad(lambda m, x: layer(m, x).sum())(Block(jr.key(0), 16), jnp.ones((4, 16)))
```

`tx.compile_cache(path)` keeps compiled programs on disk across runs.

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

`find_nan` replays the program one operation at a time, including inside loops
and branches. `tx.debug.shapes` annotates every line with its shapes, here for
all sizes at once:

```python
@fn
def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    scores = q @ k.swapaxes(-1, -2)
    return jax.nn.softmax(scores, axis=-1)

print(tx.debug.shapes(attention))
```

```
attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]   [symbolic B, D, T]
   ... | def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
   ... |     scores = q @ k.swapaxes(-1, -2)            # scores: f32[B, T, T]
   ... |     return jax.nn.softmax(scores, axis=-1)     # -> f32[B, T, T]
```

| Tool | |
|---|---|
| `tx.debug.find_nan(f, *args)` | the first operation that made a NaN or inf, with its inputs |
| `tx.debug.shapes(f[, *args])` | every line with its shapes, symbolic or concrete |
| `tx.debug.memory(f, *args)` | peak memory, and which values on which lines make it up |
| `tx.debug.timing(f, *args)` | trace, compile and run time |
| `tx.debug.summary(f, *args)` | operations by kind and line; flags unrolled loops and big constants |
| `f.info()` | everything typax knows about a function |
| `tx.show(tree)` | a pytree as shapes: `Params(w=f32[3, 3], b=f32[3])` |

## Annotations

```python
from typing import Literal, NamedTuple
from typax import Dim, Int, Key

class RGB(Dim, size=3): ...          # a named fixed size

class Params[D](NamedTuple):         # dataclasses work too
    w: f32[D, D]
    b: f32[D]

@fn
def example[*Batch, H, W](
    img: f32[*Batch, H, W, RGB],     # *Batch: any number of axes
    idx: Int[*Batch],                # any integer dtype
    p: Params[W],                    # fields are checked, D is bound to W
    pair: f32[Literal[2], ...],      # a literal size, then any axes
    scale: f32,                      # a scalar array or a Python number
    steps: int = 1,                  # a plain Python value
    key: Key | None = None,
) -> f32[*Batch, H]:
    return img.mean((-1, -2)) * scale * steps
```

| Kind | Annotations |
|---|---|
| Arrays | `f16 f32 f64 i8 … i64 u8 … u64 c64 c128`, any of a kind: `Float Int UInt Bool Complex Num Arr` |
| Sizes | type parameters (`def f[B, T]`), `Literal[3]`, `class RGB(Dim, size=3)`, `*Rest` (any number of axes, the same everywhere), `...` (any number, unchecked), `Any` (one, unchecked) |
| Derived sizes | `@fn(U="T + 1")`, constraints `@fn(where="T >= 2")` |
| Containers | `tuple[...]`, `list[X]`, `dict[str, X]`, `X \| None`, NamedTuples and dataclasses, generic ones too |
| Stacked | `Batch[Transition, T, N]`: a `Transition` whose arrays all have leading axes `T, N` |
| Plain values | `int`, `float`, `bool`, `str`, `Literal`, enums, callables, config classes |

## Ported projects

Two existing codebases rewritten in this style and checked against the
originals:

- **[PureJaxRL](https://github.com/luchris429/purejaxrl) PPO** (`examples/ppo/`):
  plain loops instead of four nested `lax.scan`s, `if done:` for the
  auto-reset. Identical to the original for the first 21 of 976 updates, then
  drifting as float32 programs do; same speed; all shapes proved.
- **[bofus](https://github.com/davidesartor/bofus)** (`examples/bofus/`):
  mixtures and the Gaussian process become modules with methods. The
  original's 74 tests pass against the port, and 164 arrays of values and
  gradients match to 1e-13 (6e-7 for the L-BFGS acquisition optimizer).

## Limitations

- Conversion needs the function's source, so functions defined with `exec` or
  at a bare `python` prompt run unconverted (`f.info()` says why).
- Left as plain Python: code inside `try` / `with`, loops with an `else`,
  in-place updates of anything but a local variable, and variables shared with
  a nested function. A variable assigned on only one traced path reads as
  zeros on the other.
- `lift` needs batch axes in front, and not on arguments with a `*Rest` shape.
- `debug.find_nan` and parts of the prover read JAX internals; they are tested
  on JAX 0.12.
- A `spot-checked` proof verdict is evidence, not proof.

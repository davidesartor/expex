# Functions

[Contents](README.md) · [Modules →](modules.md)

The pages build on each other: these imports, and what earlier pages define.

```python
import jax
import jax.numpy as jnp
import jax.random as jr
import expex
from expex import f32, i32

@expex.fn
def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    return jnp.einsum("btd,bsd->bts", q, k)
```

## Checked calls

```python
try:
    attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))
except expex.ShapeError as e:
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

Sizes can be computed from other sizes and from int parameters, right in the
annotation, and constraints are `assert`s at the top of the body:

```python
@expex.fn
def pad[B, T](x: f32[B, T]) -> f32[B, T + 1]:
    return jnp.pad(x, ((0, 0), (0, 1)))

@expex.fn
def split_heads[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:
    assert D % heads == 0
    return x.reshape(T, heads, D // heads).transpose(1, 0, 2)

split_heads(jnp.ones((5, 8)), heads=2).shape     # (2, 5, 4)
```

```
>>> split_heads(jnp.ones((5, 8)), heads=3)
ShapeError: split_heads(): constraint `D % heads == 0` is violated (D=8, heads=3).
```

A leading `assert` on sizes is checked before the body runs, and the prover
assumes it. A leading assignment names a size for the signature and the body:

```python
@expex.fn
def heads_first[T, D](x: f32[T, D], heads: int) -> f32[H, T, K]:
    H, K = heads, D // heads
    assert D % heads == 0
    h: f32[T, H, K] = x.reshape(T, H, K)
    return h.transpose(H, T, K)
```

Only names used in the signature are definitions; other assignments are
ordinary local variables. (Arithmetic in annotations works because Python 3.14
evaluates annotations lazily: expex reads them as written.)

## Sizes and axes by name

The sizes in the signature are names, and the body can use them. As a value,
a name is its size in this call; as an axis argument, it is that axis:

```python
@expex.fn
def pool[B, T, D](x: f32[B, T, D]) -> f32[B, D]:
    return x.mean(T)                     # an axis

@expex.fn
def swap[B, T, D](x: f32[B, T, D]) -> f32[T, B, D]:
    y = jnp.tanh(x).reshape(B, T, D)     # sizes
    return y.transpose(T, B, D)          # axes

@expex.fn
def running_mean[N, D](xs: f32[N, D]) -> f32[N, D]:
    total = jnp.zeros(D)
    out = []
    for i in range(N):
        total = total + xs[i]
        out.append(total / (i + 1))
    return jnp.asarray(out)

pool(jnp.ones((2, 3, 4))).shape                  # (2, 4)
swap(jnp.ones((2, 3, 4))).shape                  # (3, 2, 4)
print(expex.prove(running_mean))                    # proved for all sizes
```

- **Axes**: positional arguments of array methods (`sum`, `mean`, `max`,
  `argmax`, `transpose`, `swapaxes`, ...) and `axis=` / `axes=` of any function
  (`jax.nn.softmax(x, axis=T)`). The position comes from the array's annotation
  (a parameter, or an annotated local `h: f32[B, T, D] = ...`), else from its sizes.
  When two axes have the same size and nothing says which is which, the call
  is an error asking for an annotation.
- **Sizes**: everywhere else: shapes, `range(N)`, arithmetic, slices. Under the
  prover the sizes are symbolic, so code built from them is proved for all sizes.
- `Dim` classes and variadics (`*B`, a tuple of sizes) work the same way.

A fixed size can be named where it is first used: `Rows:6` in a subscript
reads "Rows, which is 6". (`f32[Rows=6]` is not valid Python: subscripts take
no keywords.) The name is then a size of the function, like a type parameter
but fixed, and other annotations use it plainly:

```python
@expex.fn
def drop[B](board: i32[B, Rows:6, Cols:7], col: i32[B]) -> i32[B, Rows, Cols]:
    def one(b, c):
        row = Rows - 1 - (b != 0).sum(Rows)[c]       # a size, and an axis
        return b.at[row, c].set(1)
    return jax.vmap(one)(board, col)

drop(jnp.zeros((2, 6, 7), jnp.int32), jnp.array([3, 0])).sum()   # 2
```

```
>>> drop(jnp.zeros((2, 5, 7), jnp.int32), jnp.array([3, 0]))
ShapeError: drop(): argument `board` does not match its annotation.
    ...
    axis 1 should be Rows, which is fixed at 6, but is 5.
```

The prover treats `Rows` as the constant 6, and `expex.cost` and the diagrams
list it as `Rows (= 6)`. In a struct's fields (`cells: i32[Rows:6, Cols:7]`)
the names are sizes of the class, for its methods too. A name declared twice
with different sizes, or one that is also a type parameter, is an error when
the function is defined. Type checkers accept neither the slice nor the new
name, so the line needs a `# type: ignore`; where code must type-check as
is, use `class Rows(Dim, size=6)`.

Functions defined in the body see the sizes of the call they are made in, as
values and as axes, like `one` above. So do lambdas and comprehensions,
even when they run later or are passed to another `@fn`. A helper defined
outside the function does not see them: calling one that uses `D` is a
`NameError`, with a note suggesting to define it inside, or to make it an
`@expex.fn` of its own.

Contractions and reshapes need only the names of the result:

```python
@expex.fn
def attention[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]:
    return expex.einsum(q, k, to=(B, T, S))      # D is not in `to`: summed over

@expex.fn
def heads[T, D](x: f32[T, D], n: int) -> f32[H, T, K]:
    H, K = n, D // n
    assert D % n == 0
    return expex.rearrange(x, to=(H, T, K))      # D split into H, K; T moved
```

Each operand's names come from its annotation, or from its sizes when they
are unambiguous; repeated names are matched, as in `jnp.einsum`. To give an
axis a second name (self-attention: keys along `S`), annotate a copy:
`keys: f32[B, S, D] = x`. `expex.rearrange` moves axes, splits one into several
new ones, or merges several into one new one.

## Proofs without data

`expex.prove` traces a function with symbolic sizes, so a passing proof holds for
every size:

```python
@expex.fn
def diff[T, D](xs: f32[T, D]) -> f32[T, D]:
    return xs[1:] - xs[:-1]

print(expex.prove(diff))
```

```
diff: failed
    diff(): the return value does not match its annotation.

        declared  -> f32[T, D]
        returned     f32[T - 1, D]

        axis 0 of `return` should be T, but is T - 1.
    ...
```

`expex.prove(split_heads)` proves it for every `D` and `heads` with
`D % heads == 0`. Parameters such as `heads: int` are tried at a few values
(ints at 1 and 2, bools both ways, every `Literal` and enum value); others take
`@expex.fn(examples={"name": [...]})`. Prove a whole package with `python -m expex pkg`
or `pytest --expex pkg`.

## Plain Python on traced values

```python
@expex.fn
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
@expex.fn
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

`f.flow` says how each statement ran; `@expex.fn(convert=False)` opts out.

## Vectorising

A function's annotations say which axes are its own, so a call whose
arguments have extra leading axes is mapped over them, with no `vmap` and no
`in_axes`:

```python
@expex.fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return jnp.tanh(x @ w)

score(jnp.ones((32, 5, 3)), jnp.ones(3)).shape                # (32, 5)
score(jnp.ones((8, 1, 5, 3)), jnp.ones((7, 3))).shape         # (8, 7, 5)
```

Batch axes broadcast like NumPy, and an argument without them is passed whole
to every element. So a function written for one element takes batches, and a
per-token layer takes a sequence (`self.norm(h)` with `norm: f32[D] -> f32[D]`
and `h: f32[T, D]` is `f32[T, D]`):

```python
@expex.fn
def safe_log(x: f32) -> f32:
    if x > 0:
        return jnp.log(x)
    return -jnp.inf

safe_log(jnp.array([[1.0, -1.0], [jnp.e, 0.0]]))     # [[0, -inf], [1, -inf]]
```

A call is mapped only when extra leading axes are all that keep its arguments
from matching, so a call with the declared shapes runs exactly as before. If
each element still does not match, the error says what was mapped:

```python
@expex.fn
def normalize[D](x: f32[D], scale: f32[D]) -> f32[D]:
    return (x - x.mean()) / (x.std() + 1e-5) * scale

normalize(jnp.ones((32, 8)), jnp.ones(8)).shape     # (32, 8)
```

```
>>> normalize(jnp.ones((32, 8)), jnp.ones(4))
ShapeError: normalize(): argument `scale` does not match its annotation, even with the extra leading axes mapped over.

    declared  scale: f32[D]
    received         f32[4]

    D is 8 (from `x: f32[8]`), but axis 0 of `scale` is 4.

    Extra leading axes are mapped over automatically: `x: f32[32, 8]` is a batch [32] of f32[D].
```

- Only an argument whose annotation has a fixed number of axes is mapped over;
  one annotated with a variadic (`f32[*B, D]`) is passed whole. Extra axes
  that do not broadcast against each other are an error that lists them.
- `self` is never mapped over: a struct holding a stack of weights, called
  directly, is an error. `model.vectorize(x)` and `f.vectorize(...)` say it
  explicitly, and map over `self` too.
- Each element of a random function gets its own key, and so does each
  element when a single `key: Key` argument is passed.
- It works the same under `jax.jit`, `jax.grad` and `jax.vmap`, and the
  prover proves such calls for all sizes. Compile time does not depend on the
  batch size: one element is traced.
- `@expex.fn(vectorize=False)` turns it off for one function (extra axes are
  then a `ShapeError`, whose hint names `f.vectorize`), and
  `expex.config(vectorize=False)` (or `EXPEX_VECTORIZE=0`) for all of them.
  `@expex.fn(vectorize=True)`, or `@expex.vectorize`, keeps it on for one
  function regardless.

Outside `jax.jit`, a mapped call would be traced again and run operation by
operation on every call. So from the second call with the same shapes, dtypes
and static arguments, it runs compiled instead: one `jax.jit` of the mapped
call per function. A game step that took a second per call eagerly takes a
millisecond from the third call on. This changes what you see in Python:

- Python side effects in the body (`print`, appending to a list, a counter)
  happen while the body is traced: on the first call, on the second (traced
  for `jax.jit`), and not on later calls with the same shapes. `print` on an
  array inside a converted body prints from the compiled code, every call.
- While a tool watches calls (`expex.diagram`, `expex.debug.watch`, ...) and
  inside any JAX transformation, nothing is compiled this way: tools see every
  call.
- A body that works only outside `jax.jit` (one that reads an array's value
  with `float(x)`, say) runs uncompiled, and is not compiled again.
- `expex.config(compile_vectorized=False)` (or `EXPEX_COMPILE_VECTORIZED=0`)
  turns it off.

---

[Contents](README.md) · [Modules →](modules.md)

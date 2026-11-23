# Porting JAX code

[← Command line and editors](tools.md) · [Contents](README.md)

Porting a JAX codebase means working out the shape of every array and naming
its sizes. `expex.annotate` does that part: it runs a plain JAX function on
example inputs and proposes the annotated signature, without changing the
function.

```python
def attend_jax(q, k, v):
    scores = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(q.shape[-1])
    return jax.nn.softmax(scores, axis=-1) @ v

q, k, v = jnp.ones((2, 3, 8)), jnp.ones((2, 5, 8)), jnp.ones((2, 5, 4))
print(expex.annotate(attend_jax, q, k, v))
```

```
def attend_jax[B, T, S, D, A](q: f32[B, T, D], k: f32[B, S, D], v: f32[B, S, A]) -> f32[B, T, A]:

sizes in the example: B=2, T=3, S=5, D=8, A=4
```

`.diff()` shows the change to the source file as a unified diff, and
`.apply()` writes it (never by default): the signature, any `assert` it
needs, the annotated locals, and the `from expex import ...` line. Decorate
the result with `@expex.fn` and `expex.prove` checks it for every size.

## How sizes are grouped and named

Sizes come from the dataflow, not from the numbers. An axis contracted,
broadcast, concatenated or scanned together with another is the same size;
two axes that merely have the same length are not. Above, `v`'s last axis is
`A`, not `D`: nothing joins it to the axis `q` and `k` contract over.

Names follow the axis's role where it is clear (`B` a leading axis every
operation carries along, `T` and `S` sequences, `D` features, `V` a table
indexed by integers, `H` and `K` heads and their size), the names the body
gives them (`T, D = x.shape`), and otherwise letters in order. Fix some with
`names=`; the rest follow:

```python
print(expex.annotate(attend_jax, q, k, v, names={"q": "N Q E"}).signature)
```

```
def attend_jax[N, Q, S, E, D](q: f32[N, Q, E], k: f32[N, S, E], v: f32[N, S, D]) -> f32[N, Q, D]:
```

An array built from shapes alone (`jnp.zeros((x.shape[0], w.shape[1]))`) has
no dataflow to follow, so its sizes are matched by value, and sizes that
coincide in the example cannot be told apart. annotate warns, and
`vary=True` traces again with every size changed to settle it:

```python
def init_state(x, w):
    return jnp.zeros((x.shape[0], w.shape[1]))

print(expex.annotate(init_state, jnp.ones((2, 3)), jnp.ones((3, 2))).warnings[0])
print(expex.annotate(init_state, jnp.ones((2, 3)), jnp.ones((3, 2)), vary=True).returns)
```

```
axis 0 of the result is 2, which could be B or C: these sizes coincide in the example and the dataflow does not say which; guessed B. Use distinct sizes or vary=True
f32[B, C]
```

Examples with distinct sizes avoid the question.

## Derived sizes

A pad or concatenate adds, a reshape splits or merges, and the result says
so in the sizes, with the constraint a split needs. Python numbers among the
arguments are `int` / `float`:

```python
def split(x, n):
    T, D = x.shape
    return x.reshape(T, n, D // n).transpose(1, 0, 2)

def shift_right(tokens):
    return jnp.concatenate([jnp.zeros_like(tokens[:, :1]), tokens], axis=1)

print(expex.annotate(split, jnp.ones((5, 8)), 2).signature)
print(expex.annotate(shift_right, jnp.ones((2, 7), jnp.int32)).signature)
```

```
def split[T, D](x: f32[T, D], n: int) -> f32[n, T, D // n]:
    assert D % n == 0
def shift_right[B, T](tokens: Int[B, T]) -> Int[B, T + 1]:
```

## Parameters and results

A dict of parameters becomes an `@expex.struct`, and the body reads its
fields (`params.up` for `params["up"]`); callers then pass `Params(...)`.
NamedTuples are made generic, tuples and lists annotated element by element,
and a result that is a dict becomes a struct too.

```python
def ffn(params, x):
    h = jax.nn.gelu(x @ params["up"] + params["b"])
    return x + h @ params["down"]

ffn_params = {"up": jnp.ones((4, 16)), "b": jnp.ones(16), "down": jnp.ones((16, 4))}
print(expex.annotate(ffn, ffn_params, jnp.ones((2, 4))))
```

```
@expex.struct
class Params[D, A]:
    up: f32[D, A]
    b: f32[A]
    down: f32[A, D]

def ffn[B, D, A](params: Params[D, A], x: f32[B, D]) -> f32[B, D]:

sizes in the example: B=2, D=4, A=16
note: `params` is a dict: it becomes the struct Params (the body reads `params.up`); callers pass Params(up=..., b=..., down=...)
```

## Locals

`locals=True` adds the assignments worth annotating: those with the same size
on two axes, where expex needs the annotation to tell the axes apart by name
(`locals="all"` for every one):

```python
def self_attention(x):
    scores = x @ x.swapaxes(1, 2)
    return jax.nn.softmax(scores, axis=-1) @ x

print(expex.annotate(self_attention, jnp.ones((2, 5, 4)), locals=True))
```

```
def self_attention[B, T, D](x: f32[B, T, D]) -> f32[B, T, D]:
    ...
42 |     scores: f32[B, T, T] = x @ x.swapaxes(1, 2)

sizes in the example: B=2, T=5, D=4
```

## Limits

- It reads one trace: code that branches on shapes in Python is annotated for
  the branch the example takes.
- Grouping follows JAX's primitives (elementwise, contractions, reshapes,
  pads, gathers, scans, loops, conditionals, calls); sizes that pass only
  through Python ints are matched by value.
- Dicts used as a whole (`jax.tree.map(..., params)`) are left unannotated
  with a warning, and Equinox or Flax modules are not looked inside.
- Names are guesses from roles; check them, or pass `names=`.

## Coming from jaxtyping

Code annotated with [jaxtyping](https://github.com/patrick-kidger/jaxtyping)
needs no rewrite: put `@expex.fn` on it, and `Float[Array, "batch seq d"]`
means to expex what `Float[Batch, Seq, D]` does. The names in the strings
are the function's sizes, as if they were its type parameters, spelled as
written. Calls are checked, extra leading axes are mapped over, and
`expex.prove`, `expex.cost` and the diagrams work as for any `@expex.fn`.

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # or import jaxtyping as usual, if it is installed
    from jaxtyping import Array, Float, Int

@expex.fn
def lookup(tokens: Int[Array, "batch seq"], table: Float[Array, "vocab d"]) -> Float[Array, "batch seq d"]:
    return table[tokens]

@expex.fn
def extend(x: Float[Array, "*batch t"]) -> Float[Array, "*batch t+1"]:
    return jnp.concatenate([jnp.zeros(x.shape[:-1] + (1,)), x], axis=-1)

print(lookup.signature)
print(lookup(jnp.zeros((4, 2, 5), jnp.int32), jnp.ones((10, 8))).shape)  # 4 batches: mapped over
print(expex.prove(extend).ok)
try:
    lookup(jnp.zeros((2, 5)), jnp.ones((10, 8)))  # float tokens
except expex.ShapeError as e:
    print(e.code, str(e).splitlines()[5].strip())
```

```
lookup(tokens: Int[batch, seq], table: Float[vocab, d]) -> Float[batch, seq, d]
(4, 2, 5, 8)
True
EX101 expected dtype Int, got f32.
```

jaxtyping need not be installed. With it, expex reads its classes; without it
(imported only for type checkers, as above), expex reads the annotation's
text. Either way it is the same annotation, and expex never imports jaxtyping.

| jaxtyping | expex |
|---|---|
| `"b t d"`, `"batch seq"` | the sizes `b`, `t`, `d`, `batch`, `seq`: bound by the first array that has them, checked against the others |
| `"*batch d"` | a variadic, `*batch` |
| `"..."`, `"*_"` | `...`: any number of axes, unchecked |
| `"3"`, `"t=1"` | the fixed sizes 3 and 1 (in `t=1`, `t` is a name for the reader) |
| `""` | a scalar |
| `"t+1"`, `"dim-1"`, `"2*d"`, `"d//h"` | the size expressions `t + 1`, `dim - 1`, ...; a size only in one is solved for |
| `"{n}"`, `"{self.n}"` | the int parameter `n`, the int field `n` of `self` (as `f32[n]`, `f32[self.n]`) |
| `"_"`, `"_t"` | one axis, unchecked (`Any`) |
| `"#b"`, `"#4"` | `b` (or 4), or 1: a size-1 axis also matches, and does not bind `b` |
| `"*#batch"` | axes that need only broadcast together; `*batch` is their broadcast |
| `"?b"`, `"#t+1"` | not read: [EX411](errors.md#ex411) |
| `Float`, `Complex`, `Inexact`, `Num`, `UInt`, `Bool` | expex's dtypes of the same names |
| `Shaped` | `Arr`: any dtype |
| `Int`, `Integer`, `Real` | signed integers (expex's own `Int` is any integer); any integer; float or integer |
| `Float32`, `Int8`, `UInt8`, `Complex64`, ... | `f32`, `i8`, `u8`, `c64`, ... |
| `BFloat16`, `Float8e4m3fn`, `Int4`, ... | that dtype exactly (`bf16`, ...) |
| `Key[Array, "n"]`, `PRNGKeyArray` | `Key[n]`, `Key` (new-style keys and legacy `uint32[2]` ones) |
| `Float[np.ndarray, "n"]`, `Float[Array \| np.ndarray, "n"]` | `Float[n]`: expex checks the shape and dtype, not the array's class |
| `Scalar`, `ScalarLike` | a scalar of any dtype |
| `PyTree[...]` | not checked |

A function's sizes are shared by its whole signature, as under jaxtyping's
`@jaxtyped`, which jaxtyping needs for that (without it, each annotation is
checked alone). So expex can find annotations that disagree although
jaxtyping never complained. expex's own `Float`, `Int`, `Bool` and `Key`
written the jaxtyping way (`Float[jax.Array, "b t"]`) mean the same.

**In the body**, the sizes are not names: jaxtyping code reads them as
`x.shape[0]`, and keeps doing so. Making `batch` mean a size in the body would
change what that name means in code written without it, where `b`, `t` or
`d` may be a global or a loop variable. The rewrite below makes them type
parameters, which the body can then use by name.

**Annotations nobody checked** (jaxtyping checks only under `@jaxtyped` or
its import hook) can be wrong, and when an argument has more axes than its
annotation says, expex maps over them rather than failing. Turn that off
while you first run such code, with `expex.config.vectorize = False` (or
`@expex.fn(vectorize=False)`), so that a wrong annotation is reported where
it is, then turn it back on.

A module inside another one (a layer in a model) has its own sizes: in a
stack of layers of different widths, each layer's `in` and `out` are its
own, as jaxtyping checks a module's fields when that module is built. A
module's sizes are shared with the function only for `self` and for an
argument annotated with it.

**With jaxtyping's own checks**, put `@expex.fn` on top:
`@expex.fn` over `@jaxtyped(typechecker=beartype)` runs both checks, and
maps over extra leading axes before jaxtyping sees one element. The other
order works too, but jaxtyping then refuses the extra axes. Either way the
body is not converted (`f.info()` says it wraps another function), so Python
`if` / `for` on traced values stay as JAX has them; jaxtyping code uses
`lax` for those anyway.

### Rewriting in expex's style

`python -m expex migrate jaxtyping` rewrites the annotations of a file (or of
every `.py` file in a directory): the sizes become type parameters, the
function gets `@expex.fn`, and `@jaxtyped` goes (it cannot read expex's
annotations). It prints a diff; `--write` applies it. The file is read, not
imported, so jaxtyping need not be installed.

```
$ python -m expex migrate jaxtyping model.py
-def attend(q: Float[Array, "batch seq_len d"], k: Float[Array, "batch s d"]) -> Float[Array, "batch seq_len s"]:
+@expex.fn
+def attend[Batch, SeqLen, D, S](q: Float[Batch, SeqLen, D], k: Float[Batch, S, D]) -> Float[Batch, SeqLen, S]:
```

Sizes are renamed in expex's style (`b` to `B`, `seq_len` to `SeqLen`),
unless the new name is taken: by a global, an import, a parameter or a name
the body uses (`D = 64` at the top of the file keeps the size `d` as `d`).
Field annotations of a class (an Equinox module, a dataclass) give the class
its type parameters, which its methods share. A class with a field of such a
class takes its sizes too, and so do the functions that take it
(`params: Params[StateDim, EmissionDim]`), since jaxtyping's names mean one
size throughout a file; in `list[Layer]` or `tuple[Layer, ...]`, whose
elements may differ, `Layer` is left as it is. `Float` stays `Float`, as
jaxtyping's accepts any float dtype; `--float f32` writes `f32` (and `Int` as
`i32`). jaxtyping's imports that are no longer used are removed.

What has no expex spelling is left as jaxtyping's, and still checked, since
expex reads it: broadcastable sizes (`#b`), `PyTree[...]`, dtypes expex has
no name for (`BFloat16`), and aliases defined at the top of a module
(`Image = Float[Array, "h w 3"]`). A function or class with one is left whole,
so that its sizes keep one name, and gets `@expex.fn` all the same.

---

[← Command line and editors](tools.md) · [Contents](README.md)

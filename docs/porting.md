# Porting JAX code

[← Command line and editors](tools.md) · [Contents](README.md)

Porting a JAX codebase means working out the shape of every array and naming
its sizes. `expex.annotate` does that part: it runs a plain JAX function on
example inputs and proposes the annotated signature, without changing the
function.

```python
def plain_attention(q, k, v):
    scores = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(q.shape[-1])
    return jax.nn.softmax(scores, axis=-1) @ v

q, k, v = jnp.ones((2, 3, 8)), jnp.ones((2, 5, 8)), jnp.ones((2, 5, 4))
print(expex.annotate(plain_attention, q, k, v))
```

```
def plain_attention[B, T, S, D, A](q: f32[B, T, D], k: f32[B, S, D], v: f32[B, S, A]) -> f32[B, T, A]:

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
print(expex.annotate(plain_attention, q, k, v, names={"q": "N Q E"}).signature)
```

```
def plain_attention[N, Q, S, E, D](q: f32[N, Q, E], k: f32[N, S, E], v: f32[N, S, D]) -> f32[N, Q, D]:
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

---

[← Command line and editors](tools.md) · [Contents](README.md)

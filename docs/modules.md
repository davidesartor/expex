# Modules

[← Functions](functions.md) · [Contents](README.md) · [Keys, compiling and sharding →](compiling.md)


```python
from typing import Callable

@expex.struct
class Block[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)   # how to make it
    b: f32[D] = 0.0
    act: Callable = jax.nn.gelu          # plain Python types are static fields

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        return x + self.act(x @ self.w + self.b)

@expex.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __call__[B](self, x: f32[B, D]) -> f32[B, D]:
        for block in self.blocks:      # identical blocks: one lax.scan, not 24 copies
            x = block(x)
        return x

model = Stack(jr.key(0), D=16, blocks=24)  # a key first: every array built from its annotation
y = jax.jit(lambda m, x: m(x))(model, jnp.ones((4, 16)))
grads = jax.grad(lambda m: m(jnp.ones((4, 16))).sum())(model)   # a Stack too
```

- A struct is a frozen dataclass and a pytree. Fields with array annotations
  are traced, fields with plain Python types are static.
- `Stack(key, <sizes>)`, with a PRNG key first (`jr.key(0)`, or a raw
  `jr.PRNGKey(0)`), builds every array from its annotation. An array field's
  default says how: a JAX initializer (any function of `(key, shape, dtype)`)
  or a number to fill it with. The rest is by name: the sizes (`D=16`, any size
  the class or a module in it uses), the length of a list of modules by its
  field name (`blocks=24`), and fields, of the class or of nested modules
  (`heads=8`), which are taken instead of built. Nested modules are built the
  same way.
- Without a key first, the constructor takes the values themselves, as any
  dataclass does: `Block(w, b)` or `Block(w=w, b=b)`. (If the first field is
  itself annotated as a key, a key first is its value. A class with nothing to
  build, no initializers and no nested modules, such as a game state, keeps the
  plain dataclass constructor.)
- A hand-written `__init__` also works: assign the fields as usual, and the
  result is checked against the annotations. It replaces both forms above;
  `expex.build(Stack, key, D=16, blocks=24)` still builds from the annotations.
  JAX and `.replace(...)` rebuild objects without calling `__init__`.
- Every method is an `@expex.fn` sharing the class's sizes, so
  `expex.prove(Block.__call__)` works without an instance.
- A loop over identical layers is one `lax.scan`, so a deep model compiles in
  the time one layer takes.
- A method called with extra leading axes on its arguments maps over them, so
  a per-token layer takes a sequence: `self.norm(h)`. `model.vectorize(x)`
  says so explicitly, and also maps over a batch of models (fields with extra
  axes), which a plain call never does. Other decorators go on the methods
  themselves.

```
>>> Stack(jr.key(0), blocks=24)
TypeError: Stack(key, ...) needs D: the size of `blocks[*].w` (f32[D, D]). Pass it by name, e.g. Stack(key, D=64, ...).
```

Annotations can use the struct's fields through `self`, and a method's
leading `assert` on the class's sizes and fields is an invariant of the class:
checked whenever an instance is passed in, and assumed by every function that
receives one, including the prover:

```python
@expex.struct
class Heads[D]:
    w: f32[D, D]
    heads: int = 4

    def split[T](self, x: f32[T, D]) -> f32[self.heads, T, D // self.heads]:
        assert D % self.heads == 0
        return (x @ self.w).reshape(T, self.heads, D // self.heads).transpose(1, 0, 2)

Heads(jnp.eye(8)).split(jnp.ones((5, 8))).shape      # (4, 5, 2)
print(expex.prove(Heads.split).ok)                      # True
```


## Changing part of a model

`model.at` reaches into a model and returns a copy with one part changed:

```python
edited = model.at.blocks[5].w.multiply(1e3)           # one block's weights
edited = edited.at.blocks[0].b.set(jnp.ones(16))      # checked against `b: f32[D]`
edited = edited.at.blocks[0].w[0, :3].set(0.0)        # arrays are indexed as with x.at[...]
edited = edited.at.blocks[2].apply(lambda b: b.replace(act=jax.nn.relu))
print(edited.at.blocks[2].act.get())                  # reading works too: <function relu ...>
assert edited.at.blocks[0].b.get().tolist() == [1.0] * 16 and model.blocks[0].b.sum() == 0
try:
    model.at.blocks[5].w.set(jnp.zeros((8, 8)))
except expex.ShapeError as e:
    print(e)
```

- `.set(v)`, `.add(v)`, `.multiply(v)`, `.apply(f)` return a new model of the
  same structure (also under `jax.jit`); `.get()` returns the part. `add` and
  `multiply` on a module apply to each of its arrays.
- The path is fields (`.blocks`, or `["set"]` for a field named like a
  method), list and dict elements (`[5]`), then indices into an array
  (`[0, :3]`). Static fields can be read and replaced like the others.
- The new value is checked against its annotation, with the sizes the rest of
  the model fixes, so a mistake names the path (the `try` above):

```
Stack.at.blocks[5].w.set(...): `blocks[5].w` is f32[8, 8], but is declared f32[D, D].

    D is 16 (from `blocks[0].w: f32[16, 16]`), but axis 0 of `blocks[5].w` is 8.
```

- A list of identical modules held stacked, one array per field with a
  leading layer axis (as a loop collects them, or `jax.vmap` builds them:
  `jax.vmap(lambda k: Block(k, D=16))(keys)`), is indexed by row: `[5]` changes
  row 5 of every array, as it changes the sixth module of a list.


## Model summary

```python
print(expex.summary(model, B=64))     # B: a size the call needs that the model does not have
```

```
Stack   (B=64, D=16, L=24)
module     class              params  formula       bytes   flops  flops formula
────────────────────────────────────────────────────────────────────────────────
blocks[*]  Block[D=16]  × 24     272  D² + D      1.09 kB    43 k  2·B·D² + 10·B·D
  w        f32[D, D]             256  D²          1.02 kB
  b        f32[D]                 16  D              64 B
────────────────────────────────────────────────────────────────────────────────
total      Stack               6,528  D²·L + D·L  26.1 kB  1.03 M  2·B·D²·L + 10·B·D·L
```

One row per module, a list of identical layers as one row; parameters and
FLOPs of a call, as formulas in the sizes and as numbers. `expex.summary(Stack,
D=16, blocks=24)` gives the same without building the model. In a notebook,
or with `.save("summary.html")`, it is a collapsible, sortable table.

## Saving and loading

```python
expex.save(model, "stack.npz")
model = expex.load(Stack, "stack.npz")            # no __init__ call, no key needed
model = expex.load(Stack, "stack.npz", D=16)      # and D must be 16
```

```
>>> expex.load(Mamba, "mamba.npz")
ShapeError: expex.load(Mamba, 'mamba.npz'):

    layers[2].mixer.x_proj: declared f32[R + 2 * N, E], file has f32[40, 128]

    axis 0 should be R + 2 * N = 36 (with N=16, R=4), but is 40.
```

- One `.npz` file, one array per field under a readable key
  (`layers[2].mixer.x_proj`), so other tools can read it with `np.load`.
- The structure comes from the class's annotations: nested structs, lists of
  structs, NamedTuples, `X | None`. Every array is checked against its
  annotation, with sizes shared across the whole model, as in a call.
- Static fields that are plain data (numbers, strings, `None`, tuples and
  lists of those) are saved too. Others, such as `act: Callable`, take the
  class default. Without one, load into an existing model, which supplies the
  structure and static fields: `expex.load(model, "stack.npz")`.
- Keys missing from the file are an error, and so are extra keys unless you
  pass `strict=False`.


## Comparing models and checkpoints

`expex.diff(a, b)` compares two models, or files written by `expex.save`, field by
field:

```python
small = Stack(jr.key(0), D=16, blocks=3)
expex.save(small, "base.npz")
grads = jax.grad(lambda m: (m(jnp.ones((4, 16))) ** 2).mean())(small)
tuned = jax.tree.map(lambda p, g: p - 0.1 * g, small, grads)   # one SGD step
print(expex.diff("base.npz", tuned))
```

```
6 changed, 0 unchanged

                                   max |a−b|  relative  changed
changed  blocks[1].b  f32[16]       2.50e-02  1.00e+00   100.0%
changed  blocks[2].b  f32[16]       2.10e-02  1.00e+00   100.0%
changed  blocks[0].b  f32[16]       1.89e-02  1.00e+00   100.0%
changed  blocks[1].w  f32[16, 16]   2.96e-02  1.15e-01   100.0%
changed  blocks[2].w  f32[16, 16]   3.45e-02  1.10e-01   100.0%
changed  blocks[0].w  f32[16, 16]   1.89e-02  1.04e-01   100.0%
```

(The biases started at zero, so their relative change is 1.)

Changes in structure come first (fields added or removed, such as a different
number of layers, changed shapes or dtypes, changed static fields), then
changed values, largest relative change ‖a−b‖ / max(‖a‖, ‖b‖) first, with the
fraction of elements beyond `atol` / `rtol`. In a notebook (or with
`.save("diff.html")`) it is a collapsible tree of the model, coloured by
change, with a filter for unchanged fields.

---

[← Functions](functions.md) · [Contents](README.md) · [Keys, compiling and sharding →](compiling.md)

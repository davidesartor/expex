# Keys, compiling and sharding

[← Modules](modules.md) · [Contents](README.md) · [Cost and diagrams →](cost-and-diagrams.md)

## Random keys

```python
@expex.random
@expex.fn
def dropout[D](x: f32[D], rate: float = 0.5) -> f32[D]:
    keep = jr.bernoulli(expex.key(), 1 - rate, x.shape)
    return jnp.where(keep, x / (1 - rate), 0.0)

expex.seed(0)
dropout(jnp.ones(6))                        # different on every call
dropout(jnp.ones(6), key=jr.key(1))         # or choose the key
dropout.vectorize(jnp.ones((4, 6)))              # each row gets its own key
```

Every `expex.key()` is a new key. Nested calls, branches and loop iterations
derive their own, so eager, `jit`, `scan` and `vmap` runs draw the same
numbers. Under `jax.jit`, pass the key: the global key is Python state and
would be frozen into the compiled function.

## Compiling and gradients

The common transformations are options of `@expex.fn`, in any combination:

```python
@expex.fn(jit=True, remat="dots")
def repeat[N](x: f32[N], n: int, scale: float) -> f32[N * n]:
    return jnp.tile(x, n) * scale

repeat(jnp.ones(3), 4, 0.5).shape                       # (12,)
```

| Option | |
|---|---|
| `jit=True` | compile; static arguments are read from the signature |
| `vectorize=True` | plain calls take extra leading batch axes |
| `random=True` | `expex.key()` inside gives a fresh key on every call |
| `remat=True` | recompute in the backward pass (`jax.checkpoint`); or a policy: `"dots"`, `"dots_no_batch"`, `"everything"` |
| `donate=("state",)` | with `jit`: reuse these arguments' buffers |

Each is also a decorator of its own (`@expex.jit`, `@expex.vectorize`, `@expex.random`,
`@expex.remat`). Static arguments are the ones with plain Python types (`int`,
`str`, tuples, callables, enums) and `Static[...]`; arrays and `float`
numbers are traced, so a new learning rate does not recompile. A compiled
random function draws a fresh key on every call, or takes `key=`.

JAX's own transformations work on any `@expex.fn` too: `jax.jit(f,
static_argnames=...)`, `jax.grad`, `jax.vmap`, `jax.checkpoint`.

`expex.compile_cache(path)` keeps compiled programs on disk across runs.

## Sharding

Every array's axes are named in its annotation, so say once which sizes are
split over which mesh axes, and every array gets its sharding: no
`PartitionSpec`s by hand.

```python
from jax.sharding import AxisType

n = jax.device_count()                    # e.g. 8 devices: make_mesh((2, 4), ...)
mesh = jax.make_mesh((n, 1), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
names = dict(B="data", D="model")

print(expex.shardings(model, mesh, **names).blocks[0].w.spec)   # P('model',): w is f32[D, D]

model = expex.device_put(model, mesh, **names)
forward = expex.shard(Stack.__call__, mesh, **names)
out = forward(model, jnp.ones((8 * n, 16))) # x: f32[B, D], and the result, split as B, D
```

- `expex.shardings(tree, mesh, **names)` gives the `NamedSharding`s of a model
  (or of `jax.eval_shape(...)` of one, before any memory is used), with lists
  of layers and nested modules; `expex.device_put` places a tree by them.
- `expex.shard(f, mesh, **names)` is `expex.jit(f)` with the arguments placed and
  the result constrained by their annotations. Intermediates need nothing:
  XLA propagates on `Auto` mesh axes. (`jax.make_mesh` makes `Explicit` ones
  by default, where JAX asks for an `out_sharding` at contractions over a
  split axis.)
- Sizes not named are replicated, and so is a repeated one after its first
  axis (`f32[D, D]` is `P("model", None)`). A mapping is a mesh axis, a tuple
  of them, or `None`.
- A size that its mesh axis cannot split evenly is an error naming the array:
  `Stack.blocks[0].w: f32[D, D] has D = 6 on axis 0, which mesh axis 'model'
  cannot split evenly (4 devices do not divide 6).`

---

[← Modules](modules.md) · [Contents](README.md) · [Cost and diagrams →](cost-and-diagrams.md)

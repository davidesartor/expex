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
dropout(jnp.ones((4, 6)))                   # mapped over the rows: each gets its own key
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
| `vectorize=False` | extra leading axes are an error instead of being [mapped over](functions.md#vectorising) (`True`, the default, keeps them mapped even with `expex.config(vectorize=False)`) |
| `random=True` | `expex.key()` inside gives a fresh key on every call |
| `remat=True` | recompute in the backward pass (`jax.checkpoint`); or a policy: `"dots"`, `"dots_no_batch"`, `"everything"` |
| `donate=("state",)` | with `jit`: reuse these arguments' buffers |
| `compute=jnp.bfloat16` | [mixed precision](#mixed-precision): the body computes in bf16, the annotations keep the storage dtype |

Each but `compute` is also a decorator of its own (`@expex.jit`, `@expex.vectorize`, `@expex.random`,
`@expex.remat`). Static arguments are the ones with plain Python types (`int`,
`str`, tuples, callables, enums) and `Static[...]`; arrays and `float`
numbers are traced, so a new learning rate does not recompile. A compiled
random function draws a fresh key on every call, or takes `key=`.

JAX's own transformations work on any `@expex.fn` too: `jax.jit(f,
static_argnames=...)`, `jax.grad`, `jax.vmap`, `jax.checkpoint`.

`expex.compile_cache(path)` keeps compiled programs on disk across runs.

## Mixed precision

`compute=` makes a function, or every method of a struct, compute in another
float dtype than the one its arrays are stored in:

```python
import re

@expex.struct(compute=jnp.bfloat16)
class Stack16(Stack):          # the same model and annotations, computing in bfloat16
    pass

model16 = Stack16(jr.key(0), D=16, blocks=4)
inputs = jnp.ones((8, 16))
print(model16(inputs).dtype, model16.blocks[0].w.dtype)       # stored and returned as annotated
grads16 = jax.grad(lambda m: m(inputs).sum())(model16)
print(grads16.blocks[0].w.dtype)                             # gradients of f32 parameters: f32
program = jax.jit(Stack16.__call__).lower(model16, inputs).as_text()
print(re.findall(r"dot_general .*-> (tensor<\S+>)", program))  # the matmuls XLA is given

@expex.fn(compute=jnp.bfloat16)
def mlp[D, F](x: f32[D], w1: f32[D, F], w2: f32[F, D]) -> f32[D]:
    return jax.nn.gelu(x @ w1) @ w2        # x, w1 and w2 are bf16 here

with expex.config(compute=jnp.bfloat16):   # every @fn without a setting of its own
    assert Stack(jr.key(0), D=16, blocks=4)(inputs).dtype == jnp.float32
```

```
float32 float32
float32
['tensor<8x16xbf16>', 'tensor<8x16xbf16>', 'tensor<8x16xbf16>', 'tensor<8x16xbf16>']
```

- The annotations keep meaning the storage dtype. The arguments are checked
  as they are passed (`f32`, with their shapes), then the float arrays among
  them, struct fields such as parameters included, are cast to the compute
  dtype. Float results are cast back to the dtype their annotation declares:
  `-> f32[D]` returns f32. Integer and bool arrays, keys and Python numbers are
  left alone, and `expex.eps` / `expex.tiny` follow the dtype they meet.
- Modules and functions called from a bf16 body get bf16 values and compute
  in bf16; there, an annotation that declares a float dtype accepts the
  compute dtype too, so `Block.__call__(x: f32[B, D])` takes them unchanged.
- The setting comes from, innermost first: the function
  (`@expex.fn(compute=...)`), its struct (`@expex.struct(compute=...)`, for the
  methods it defines or, on a subclass, inherits), the body it is called from,
  and `expex.config.compute` (or `EXPEX_COMPUTE=bfloat16`). A setting of its
  own wins: a `@expex.struct(compute=jnp.float32)` layer norm inside a bf16
  model computes in f32, and its results come back in bf16 for the caller.
- It composes with `jit`, `vectorize` and automatic batching, `remat` and
  `jax.grad`: the casts are ordinary JAX operations, so the gradients of f32
  parameters are f32. [`examples/transformer_bf16.py`](../examples/transformer_bf16.py)
  trains the transformer both ways: the losses agree to 0.2%, and every
  `dot_general` of the bf16 program takes and returns bf16. (XLA's CPU backend
  may still run batched bf16 products in f32 internally; GPUs and TPUs run
  them in bf16.)
- Nothing changes for functions without a setting: no wrapper, no cost. With
  `jit`, the setting is read when the function is traced.

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
- Extra leading axes, which a call [maps over](functions.md#vectorising),
  have no name, so they are replicated: annotate the batch (`x: f32[B, D]`) on
  the function you shard to split it.

### Training on a mesh

A training step with an optax optimizer, whose state holds the model's moments,
sharded like the model:

```python optax
import optax
from typing import Any

opt = optax.adam(1e-3)

@expex.fn
def train_step[B, D](model: Stack, opt_state, x: f32[B, D]) -> tuple[Stack, Any, f32]:
    loss, grads = jax.value_and_grad(lambda m: (m(x) ** 2).mean())(model)
    updates, opt_state = opt.update(grads, opt_state)
    return optax.apply_updates(model, updates), opt_state, loss

step = expex.shard(train_step, mesh, donate="auto", **names)
opt_state = opt.init(model)
for _ in range(3):
    model, opt_state, loss = step(model, opt_state, jnp.ones((8 * n, 16)))  # both donated

expex.save((model, opt_state), "state.npz")
template = jax.eval_shape(lambda: (model, opt_state))       # in a new process: from the config
model, opt_state = expex.load(template, "state.npz", mesh=mesh, names=names)
print(opt_state[0].mu.blocks[0].w.sharding.spec)            # as the model's w
```

```
P('model',)
```

- Unannotated parameters and results (`opt_state`, `Any`) are searched for
  structs, which are placed and constrained by their own annotations; their
  other arrays (the step count) keep their placement.
- `donate=` is as for `@expex.fn(jit=True)`: parameter names, or `"auto"` for
  what each call site reassigns.
- `expex.save` and `expex.load` take the whole training state
  ([Saving and loading](modules.md#saving-and-loading)); with `mesh=` and
  `names=`, each array goes straight from the file to its devices.

[`examples/multidevice/train.py`](../examples/multidevice/train.py) trains the
transformer of `examples/` on a 4 × 2 data × model mesh of 8 simulated CPU
devices with `names = dict(B="data", D="model")`: it prints each array's
per-device shape, resumes from a checkpoint in a new process, and checks the
losses against a 1-device run:

```
  array                      shape          spec                       per device
  blocks[0].attn.wq [D,D]    (64, 64)       P('model',)                (32, 64)
  opt mu.blocks[0].attn.wq   (64, 64)       P('model',)                (32, 64)
  tokens  Int[B, T]          (32, 33)       P('data',)                 (8, 33)
  hidden  f32[B, T, D]       (32, 33, 64)   P('data', None, 'model')   (8, 33, 32)

step   8 devices   resumed (new process)    1 device   |8 - 1|
   9    3.264118                            3.264118   2.4e-07
  10    3.219189         3.219189  exact    3.219189   2.4e-07
  11    3.016314         3.016314  exact    3.016314   0.0e+00
```

---

[← Modules](modules.md) · [Contents](README.md) · [Cost and diagrams →](cost-and-diagrams.md)

# Equinox, Flax, Optax and Orbax

[← Errors](errors.md) · [Contents](README.md) · [The design of expex →](design.md)

expex works on pytrees and plain JAX calls, so it can be adopted one module or
one function at a time: annotate an Equinox or Flax module's arrays and
decorate its methods, and the rest of the codebase keeps working. Optax
optimizes `@expex.struct` models as they are, and Orbax checkpoints them.

| | Works | With a caveat | Does not work |
|---|---|---|---|
| **Equinox** | `@expex.fn` methods: sizes from the fields, checks, batches, proofs, `filter_jit` / `filter_grad` / `filter_vmap`; structs inside Equinox modules and Equinox modules inside structs; `summary`, `cost`, `diagram` | layers whose arrays are not annotated (`eqx.nn.Linear`): run and are checked at their edges, but are not proved, summarised or saved by `expex.save` | |
| **Flax NNX** | `@expex.fn` methods on `nnx.Param` attributes annotated in the class, nested modules, `nnx.List`, `nnx.jit` / `nnx.grad` / `nnx.Optimizer`, proofs | state updated in a call (`Dropout`, BatchNorm statistics, counters): write the method for the batch (`vectorize=False`) | state updates inside an automatically mapped call or a loop over layers |
| **Flax linen** | `@expex.fn` functions called in `__call__`, under `init` / `apply` / `jit` / `grad` | `@expex.fn` on a linen method runs, but is not proved | sizes from parameters (they are not attributes) |
| **Optax** | any optimizer on structs; `expex.mask` for weight decay and freezing; `multi_transform` | a struct of labels is callable: pass `multi_transform` a function | |
| **Orbax** | structs and their optimizer states, restored from `jax.eval_shape` targets, onto shardings from `expex.shardings` | | |

Each example below needs its library (`pip install equinox flax optax
orbax-checkpoint`); they are tested with Equinox 0.13, Flax 0.12, Optax 0.2
and Orbax 0.12.

## Equinox

Annotate an `eqx.Module`'s array fields and decorate its methods with
`@expex.fn`:

```python equinox
import equinox as eqx

class Linear[I, O](eqx.Module):
    w: f32[I, O]
    b: f32[O]

    def __init__(self, key, i: int, o: int):
        self.w = jr.normal(key, (i, o)) / i**0.5
        self.b = jnp.zeros(o)

    @expex.fn
    def __call__(self, x: f32[I]) -> f32[O]:
        return x @ self.w + self.b

class MLP[D, F](eqx.Module):
    up: Linear[D, F]
    down: Linear[F, D]

    def __init__(self, key, d: int, f: int):
        k1, k2 = jr.split(key)
        self.up, self.down = Linear(k1, d, f), Linear(k2, f, d)

    @expex.fn
    def __call__(self, x: f32[D]) -> f32[D]:
        return x + self.down(jax.nn.gelu(self.up(x)))

mlp = MLP(jr.key(0), 8, 32)
print(mlp(jnp.ones((4, 8))).shape)                  # written for one input, called on 4
loss_grads = eqx.filter_grad(lambda m, x: (m(x) ** 2).mean())(mlp, jnp.ones((4, 8)))
print(eqx.filter_jit(mlp.__call__)(jnp.ones(8)).shape)
print(expex.prove(MLP.__call__))
try:
    mlp(jnp.ones(6))
except expex.ShapeError as e:
    print(e)
```

```
(4, 8)
(8,)
__main__.MLP.__call__: proved for all sizes
MLP.__call__(): argument `x` does not match its annotation.

    declared  x: f32[D]
    received     f32[6]

    D is 8 (from `self.up.w: f32[8, 32]`), but axis 0 of `x` is 6.
    ...
```

- Sizes come from the fields, as in a struct: `D` is bound from `self.up.w`, and
  a wrong input names the field it disagrees with. `self` is never mapped over;
  a batch of inputs is.
- `eqx.filter_jit`, `filter_grad`, `filter_vmap` and `jax.jit` work as on any
  Equinox module. A bound method (`mlp.__call__`) is a pytree, as Equinox's own
  are, so `eqx.filter_jit(mlp.__call__)` traces the weights rather than baking
  them in, and does not compile again for a new model of the same shapes.
- A hand-written `__init__` is fine: the prover and the other tools make
  abstract modules without calling it, as JAX does when it rebuilds one.
- Static fields (`heads: int = eqx.field(static=True, default=4)`) can be used
  in sizes and constraints (`f32[self.heads, T, D // self.heads]`). The prover
  needs a default for them, or a bound method: `expex.prove(model.split)`.
- `expex.summary(mlp)`, `expex.cost(MLP.__call__)` and
  `expex.diagram(MLP.__call__)` work as on structs.
- Fields annotated with jaxtyping (`w: Float[Array, "i o"]`), as much Equinox
  code is, work the same way: see [Coming from
  jaxtyping](porting.md#coming-from-jaxtyping).

### Structs and Equinox modules inside each other

```python equinox
@expex.struct
class Norm[D]:
    scale: f32[D] = 1.0

    def __call__(self, x: f32[D]) -> f32[D]:
        return (x - x.mean()) / (x.std() + 1e-5) * self.scale

class PreNorm[D, F](eqx.Module):                # an @expex.struct inside an Equinox module
    norm: Norm[D]
    mlp: MLP[D, F]

    def __init__(self, key, d: int, f: int):
        self.norm, self.mlp = Norm(key, D=d), MLP(key, d, f)

    @expex.fn
    def __call__(self, x: f32[D]) -> f32[D]:
        return self.mlp(self.norm(x))

@expex.struct
class Head[D, V]:                               # an Equinox layer inside an @expex.struct
    norm: Norm[D]
    out: eqx.nn.Linear                          # Equinox's own: its arrays are not annotated

    def __call__(self, x: f32[D]) -> f32[V]:
        return self.out(self.norm(x))

pre = PreNorm(jr.key(1), 8, 32)
head = Head(jr.key(2), D=8, out=eqx.nn.Linear(8, 50, key=jr.key(3)))   # built by Equinox, passed by name
print(head(pre(jnp.ones((4, 8)))).shape)
print(expex.prove(PreNorm.__call__))
print(expex.summary(head))
print(expex.cost(head.__call__))
```

```
(4, 50)
__main__.PreNorm.__call__: proved for all sizes
Head   (D=8)
module   class        params  formula  bytes  flops  flops formula
──────────────────────────────────────────────────────────────────
norm     Norm[D=8]         8  D         32 B     72  8·D + 8
  scale  f32[D]            8  D         32 B
──────────────────────────────────────────────────────────────────
total    Head              8  D         32 B    922  922
note: 450 parameters in parts without array annotations are not counted
note: Head.__call__: no cost estimate (Unsynthesizable: self.out)
Head.__call__   (sizes: D=8)
  flops        922   = 922 FLOP
  parameters   458   = 458
  largest      f32[50] (50 elements)
```

- An Equinox layer is a field like any other: the constructor cannot build it
  from annotations, so it is passed by name (`Head(key, D=8, out=...)`).
- Its arrays have no annotations, so expex checks what goes in and out of it
  but cannot see inside: `expex.prove(Head.__call__)` cannot invent one,
  `expex.summary` leaves its parameters out (and says so), `expex.cost` of the
  bound method counts it at the instance's sizes, as numbers, and `expex.save`
  refuses the model rather than leave the layer out of the file: save it with
  Orbax ([below](#orbax)), or annotate the layer (a five-line `eqx.Module`
  like `Linear` above).

## Flax NNX

Annotate the attributes in the class body, as Equinox fields are; at run time
they are `nnx.Param`s, checked by their shape and dtype:

```python flax optax
from flax import nnx
import optax

class Dense[I, O](nnx.Module):
    w: f32[I, O]
    b: f32[O]

    def __init__(self, i: int, o: int, rngs: nnx.Rngs):
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (i, o)))
        self.b = nnx.Param(jnp.zeros(o))

    @expex.fn
    def __call__(self, x: f32[I]) -> f32[O]:
        return x @ self.w + self.b

class FeedForward[D, F](nnx.Module):
    up: Dense[D, F]
    down: Dense[F, D]

    def __init__(self, d: int, f: int, rngs: nnx.Rngs):
        self.up, self.down = Dense(d, f, rngs), Dense(f, d, rngs)

    @expex.fn
    def __call__(self, x: f32[D]) -> f32[D]:
        return x + self.down(jax.nn.gelu(self.up(x)))

ff = FeedForward(8, 32, nnx.Rngs(0))
optimizer = nnx.Optimizer(ff, optax.adamw(1e-3), wrt=nnx.Param)

@nnx.jit
def nnx_step(model, optimizer, x):
    loss, g = nnx.value_and_grad(lambda m: (m(x) ** 2).mean())(model)
    optimizer.update(model, g)
    return loss

print(nnx_step(ff, optimizer, jnp.ones((4, 8))).shape)   # x: 4 inputs, mapped over
print(expex.prove(FeedForward.__call__))
```

```
()
__main__.FeedForward.__call__: proved for all sizes
```

- Nested modules (`up: Dense[D, F]`) are checked through, and a list of them
  is annotated `list[Block[D]]` and held in an `nnx.List`. An int attribute
  annotated in the class (`heads: int = 4`) can be used in sizes and
  constraints (`f32[self.heads, T, D // self.heads]`, `assert D % self.heads
  == 0`); `expex.prove(Class.method)` takes the class default, and
  `expex.prove(module.method)` the module's value.
- Only the annotated attributes are checked; Flax's own layers (`nnx.Linear`,
  `nnx.LayerNorm`) held by a module run as usual, unchecked inside, and the
  prover cannot make them for other sizes: proofs need the arrays a method
  reads annotated.
- State that changes during a call (`nnx.Dropout` drawing from `nnx.Rngs`,
  BatchNorm's statistics in training, a counter) cannot change inside an
  automatically mapped call or a loop over identical layers: both run the body
  under `vmap` or `lax.scan`, where NNX refuses to update its variables
  (`TraceContextError`). Write such a method for the batch, with
  `vectorize=False`:

```python flax
class Noisy[D, F](nnx.Module):
    w: f32[D, F]

    def __init__(self, d: int, f: int, rngs: nnx.Rngs):
        self.w = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (d, f)))
        self.drop = nnx.Dropout(0.1, rngs=rngs)       # draws from rngs: state

    @expex.fn(vectorize=False)                        # written for the batch: no vmap around the state
    def __call__[B](self, x: f32[B, D]) -> f32[B, F]:
        return self.drop(x @ self.w)

print(Noisy(8, 16, nnx.Rngs(0))(jnp.ones((4, 8))).shape)
```

```
(4, 16)
```

- `expex.save(module, path)` writes every array of an NNX module under its
  path (`.up.w.value`), and `expex.load(template, path)` reads them back into
  `nnx.eval_shape(lambda: FeedForward(8, 32, nnx.Rngs(0)))`, checked against
  the template's shapes. Flax's own way, `nnx.state` with Orbax, works too.

## Flax linen

Linen's parameters live in a dict that `self.param` reads, not in attributes,
so there is nothing on `self` for sizes to come from. Write the typed part as
functions and call them from `__call__`:

```python flax
import flax.linen as nn

@expex.fn
def mix[T, S, D](q: f32[T, D], k: f32[S, D], v: f32[S, D]) -> f32[T, D]:
    return jax.nn.softmax(q @ k.T / jnp.sqrt(D), axis=-1) @ v

class SelfAttention(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):                      # x: [batch, tokens, features]
        q, k, v = (nn.Dense(self.features)(x) for _ in range(3))
        return mix(q, k, v)                     # written for one sequence, mapped over the batch

layer = SelfAttention(16)
tokens = jnp.ones((2, 5, 16))
variables = layer.init(jr.key(0), tokens)
print(jax.jit(layer.apply)(variables, tokens).shape)
print(expex.prove(mix))
```

```
(2, 5, 16)
__main__.mix: proved for all sizes
```

- The functions are checked under `init`, `apply`, `jax.jit` and `jax.grad`,
  and a wrong shape fails there with expex's message.
- `@expex.fn` on a linen `__call__` itself also runs, checked and mapped, but
  cannot be proved: `self.param` needs a module bound by `init` or `apply`.

## Optax

Optax treats a struct as the pytree it is: `init`, `update` and
`optax.apply_updates` return structs of the same class.

```python optax
import optax

@expex.struct
class Layer[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0
    scale: f32[D] = 1.0

    def __call__(self, x: f32[D]) -> f32[D]:
        return x + jax.nn.gelu(x @ self.w + self.b) * self.scale

@expex.struct
class Net[D]:
    layers: list[Layer[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for layer in self.layers:
            x = layer(x)
        return x

def make_step(tx):
    @jax.jit
    def step(net, tx_state, x):
        loss, g = jax.value_and_grad(lambda m: (m(x) ** 2).mean())(net)
        updates, tx_state = tx.update(g, tx_state, net)
        return optax.apply_updates(net, updates), tx_state, loss
    return step

net = Net(jr.key(0), D=16, layers=4)
decay = expex.mask(net, lambda path, spec: spec.ndim >= 2)     # weight decay on the matrices only
print(decay.layers[0])
tx = optax.adamw(1e-3, weight_decay=0.1, mask=decay)
tx_state = tx.init(net)
net, tx_state, loss = make_step(tx)(net, tx_state, jnp.ones((8, 16)))
print(type(net).__name__, type(tx_state[0].mu).__name__)
```

```
Layer(w=True, b=False, scale=False)
Net Net
```

`expex.mask(tree, where)` is a pytree of booleans of the structure of `tree`:
`where(path, spec)` for each annotated array, with its path as errors and
checkpoints name it (`layers[2].w`) and its annotation (`spec.ndim`,
`spec.dtype`, `spec.dims`). Arrays that no annotation describes are `False`.
`tree` may be a struct, an Equinox module, or any pytree holding them. Masks
by shape (`jax.tree.map(lambda a: a.ndim >= 2, net)`) give the same here, but
not for arrays with axes their annotations do not declare, such as an ensemble
built by `jax.vmap`:

```python optax
ensemble = jax.vmap(lambda k: Net(k, D=16, layers=2))(jr.split(jr.key(1), 4))   # 4 models
print(jax.tree.map(lambda a: a.ndim >= 2, ensemble).layers[0])                  # by shape: all of them
print(expex.mask(ensemble, lambda path, spec: spec.ndim >= 2).layers[0])        # by annotation
```

```
Layer(w=True, b=True, scale=True)
Layer(w=True, b=False, scale=False)
```

### Freezing part of a model

`optax.multi_transform` trains some arrays and leaves others alone. Its labels
come from a function of the parameters: a struct of labels would do, but
structs with a `__call__` are callable, and optax calls a callable to get the
labels.

```python optax
def labels(params):
    last = expex.mask(params, lambda path, spec: path.startswith("layers[3]"))
    return jax.tree.map(lambda t: "train" if t else "freeze", last)

tx = optax.multi_transform({"train": optax.adam(1e-3), "freeze": optax.set_to_zero()}, labels)
tuned, _, _ = make_step(tx)(net, tx.init(net), jnp.ones((8, 16)))
print([bool((a != b).any()) for a, b in zip(jax.tree.leaves(net), jax.tree.leaves(tuned))])
```

```
[False, False, False, False, False, False, False, False, False, True, True, True]
```

- `optax.masked(inner, mask)` is not a freeze on its own: it applies `inner` to
  the masked arrays and passes the others' updates (their raw gradients) through.
- To compute gradients for the trainable part only, split the model with the
  same mask: `eqx.partition(net, mask)` and `eqx.combine` work on structs, and
  so does plain JAX:

```python optax
train = expex.mask(net, lambda path, spec: path.startswith("layers[3]"))
trainable = jax.tree.map(lambda t, a: a if t else None, train, net)
frozen = jax.tree.map(lambda t, a: None if t else a, train, net)
merge = lambda p, f: jax.tree.map(lambda a, b: b if a is None else a, p, f, is_leaf=lambda v: v is None)
g = jax.grad(lambda p: (merge(p, frozen)(jnp.ones(16)) ** 2).sum())(trainable)
print(g.layers[0].w, g.layers[3].w.shape)
```

```
None (16, 16)
```

## Orbax

A struct and its optimizer state are pytrees, so Orbax saves them as they
are. To restore, give it the structure: `jax.eval_shape` builds it from the
config without using memory.

```python orbax optax
import orbax.checkpoint as ocp
from pathlib import Path

tx = optax.adamw(1e-3)

def fresh():                                    # what a new process builds from the config
    net = Net(jr.key(0), D=16, layers=4)
    return {"net": net, "opt": tx.init(net)}

checkpointer = ocp.StandardCheckpointer()
where = Path("checkpoints").absolute() / "step_100"        # Orbax takes absolute paths
checkpointer.save(where, {"net": net, "opt": tx.init(net)})
checkpointer.wait_until_finished()

abstract = jax.eval_shape(fresh)
restored = checkpointer.restore(where, abstract)
print(type(restored["net"]).__name__, jnp.array_equal(restored["net"].layers[2].w, net.layers[2].w))
```

```
Net True
```

Onto devices: `expex.shardings` gives the shardings of the whole state, the
optimizer's moments included, from the axis names, and Orbax reads each array
straight onto its devices:

```python orbax optax
from jax.sharding import AxisType

ckpt_mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
placed = expex.shardings(abstract, ckpt_mesh, D="model")
target = jax.tree.map(lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s), abstract, placed)
restored = checkpointer.restore(where, target)
print(restored["net"].layers[0].w.sharding.spec, restored["opt"][0].mu.layers[0].w.sharding.spec)

manager = ocp.CheckpointManager(Path("run").absolute(), options=ocp.CheckpointManagerOptions(max_to_keep=2))
for step in range(3):                                     # a training loop's checkpoints
    manager.save(step, args=ocp.args.StandardSave(restored))
manager.wait_until_finished()
print(manager.all_steps())
restored = manager.restore(manager.latest_step(), args=ocp.args.StandardRestore(abstract))
```

```
P('model',) P('model',)
[1, 2]
```

A restored model has the target's shapes, so the wrong config is an error
from Orbax (`Requested shape: (32, 32) is not compatible with the stored
shape: (16, 16)`); its annotations are checked again on the first call.

### `expex.save` or Orbax

| | `expex.save` / `expex.load` | Orbax |
|---|---|---|
| Format | one `.npz` file, one array per field under its path (`layers[2].w`): `np.load` reads it | a directory per checkpoint (TensorStore) |
| Checks | every array against its annotation, with sizes shared across the model: `axis 0 should be R + 2 * N = 36 (with N=16, R=4), but is 40` | shapes and dtypes against the target |
| Restoring | `expex.load(Net, path)` builds the structure from the annotations; no template needed | needs a target (`jax.eval_shape(...)`) |
| Devices | `load(..., mesh=, names=)` places each array; written from one process | each host writes and reads its own shards, in parallel; saves in the background while training goes on |
| Contents | structs, and pytrees holding them (`(model, opt_state)`, see [Training on a mesh](compiling.md#training-on-a-mesh)) | any pytree: also Equinox and Flax layers whose arrays are not annotated, linen variables, NNX state |
| Runs | | `CheckpointManager`: steps, how many to keep, the latest |

Use `expex.save` for a model you share, inspect, diff (`expex.diff`) or load
in another program: weights to publish, a fine-tuning starting point. Use Orbax
for the checkpoints of a training run, and for models too large for one host,
sharded over several, or holding layers whose arrays expex cannot see.

---

[← Errors](errors.md) · [Contents](README.md) · [The design of expex →](design.md)

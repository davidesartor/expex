# expex

**Write the shapes in the signature; get checks, proofs, loops and model tools for free.**

expex is a thin layer over JAX. Annotate arrays with their sizes and the
library reads those annotations: it checks every call, proves the shapes for
every size, turns plain Python loops into `lax.scan`, and draws, costs and
debugs your model. It is still JAX underneath: `jax.jit` and `jax.grad` work as
usual, and a compiled call costs the same as plain `jax.jit`.

```
pip install expex          # Python 3.14+, JAX 0.12+
```

```python
import jax, jax.numpy as jnp, jax.random as jr
import expex
from expex import f32

key = jr.key(0)
```

## JAX vs expex

### Shapes are checked, not commented

```python
# JAX
def attend(q, k):
    # q: [B, T, D], k: [B, S, D] -> [B, T, S]
    assert q.shape[0] == k.shape[0]
    assert q.shape[2] == k.shape[2]
    return jnp.einsum("btd,bsd->bts", q, k)

# expex
@expex.fn
def attend[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]:
    return expex.einsum(q, k, to=(B, T, S))
```

### Loops are loops

```python
# JAX
def rnn(w, xs):  # w: [D, D], xs: [T, D]
    def step(h, x):
        h = jnp.tanh(w @ h + x)
        return h, h
    h0 = jnp.zeros(xs.shape[1])
    _, hs = jax.lax.scan(step, h0, xs)
    return hs

# expex
@expex.fn
def rnn[T, D](w: f32[D, D], xs: f32[T, D]) -> f32[T, D]:
    h, hs = jnp.zeros(D), []
    for t in range(T):  # a lax.scan carrying h
        h = jnp.tanh(w @ h + xs[t])
        hs.append(h)
    return jnp.asarray(hs)
```

### Branches are `if` / `else`

```python
# JAX
@jax.jit
def safe_log(x):
    ok = x > 0
    safe = jnp.where(ok, x, 1.0)
    return jnp.where(ok, jnp.log(safe), -jnp.inf)

# expex
@expex.fn(jit=True, vectorize=True)
def safe_log(x: f32) -> f32:
    return jnp.log(x) if x > 0 else -jnp.inf  # a select per element
```

### Models are classes; identical layers scan themselves

```python
# JAX
def init(key, D, L):
    # stacked by hand so lax.scan can loop
    w = jr.normal(key, (L, D, D)) * 0.1
    return {"w": w, "b": jnp.zeros((L, D))}

def stack(params, x):
    def layer(x, p):
        y = p["w"] @ x + p["b"]
        return x + jax.nn.gelu(y), None
    x, _ = jax.lax.scan(layer, x, params)
    return x

params = init(key, D=64, L=12)

# expex
@expex.struct
class Block[D]:
    w: f32[D, D] = jax.nn.initializers.normal(0.1)
    b: f32[D] = 0.0

    def __call__(self, x: f32[D]) -> f32[D]:
        y = self.w @ x + self.b
        return x + jax.nn.gelu(y)

@expex.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for block in self.blocks:  # identical layers: one lax.scan
            x = block(x)
        return x

model = Stack.init(key, D=64, blocks=12)
```

A wrong call says which axis is wrong and where its size came from:

```python
try:
    attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))
except expex.ShapeError as e:
    print(e)
```

```
attend(): argument `k` does not match its annotation.

    declared  k: f32[B, S, D]
    received     f32[5, 3, 4]

    B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5.
```

## What else the annotations give you

None of this needs example data, unless it measures something.

**Proofs for every size**, also from the command line (`python -m expex pkg/`) and
in pytest (`pytest --expex pkg`):

```python
print(expex.prove(Stack.__call__))
```

```
__main__.Stack.__call__: proved for all sizes
```

**Cost as formulas in the sizes**, or as numbers for given sizes:

```python
print(expex.cost(Stack.__call__))
```

```
Stack.__call__   (sizes: D, L)
  flops        2·D²·L + 10·D·L
  parameters   D²·L + D·L
  largest      f32[D] (D elements)
  note: L = len(self.blocks)
```

**Model summaries:**

```python
print(expex.summary(model))
```

```
Stack   (D=64, L=12)
module     class              params  formula       bytes   flops  flops formula
────────────────────────────────────────────────────────────────────────────────
blocks[*]  Block[D=64]  × 12   4,160  D² + D      16.6 kB  8.83 k  2·D² + 10·D
  w        f32[D, D]           4,096  D²          16.4 kB
  b        f32[D]                 64  D             256 B
────────────────────────────────────────────────────────────────────────────────
total      Stack              49,920  D²·L + D·L   200 kB   106 k  2·D²·L + 10·D·L
```

**Architecture diagrams**, drawn from the code. `expex.diagram(Transformer.__call__)`
for the transformer in [`examples/`](examples/transformer.py); save it as `.svg`,
`.png`, or an interactive `.html` page where you click a call to open it:

<img src="docs/images/transformer.png" alt="Diagram of a transformer: tokens, index, Block × L, LayerNorm, matmul, with each block's internals on the right" width="720">

The same diagram timed on real inputs (`expex.diagram.timed(model, tokens)`), and
the one a `ShapeError` carries (`e.diagram`), with the failing call and the path
into it in red:

<table><tr>
<td><img src="docs/images/transformer_timed.png" alt="The transformer diagram with each call coloured by its share of the measured time"></td>
<td><img src="docs/images/transformer_error.png" alt="The transformer diagram with the MLP call that failed in red"></td>
</tr></table>

**Debugging:** shapes line by line, and where a NaN first appeared:

```python
print(expex.debug.shapes(Block.__call__))
```

```
__call__(self: Block, x: f32[D]) -> f32[D]   [symbolic D]
   75 |         y = self.w @ x + self.b     # y: f32[D]
   76 |         return x + jax.nn.gelu(y)   # -> f32[D]
```

```python
@expex.fn
def entropy[N](logits: f32[N]) -> f32:
    p = jax.nn.softmax(logits)
    return -(p * jnp.log(p)).sum()

print(expex.debug.find_nan(entropy, jnp.array([0.0, 200.0, -200.0])))
```

```
NaN first appears at example.py:4 (entropy)
    > return -(p * jnp.log(p)).sum()
  operation: mul
  inputs:
    #0: f32[3]  min=0  max=1  (2 zero, of 3)
    #1: f32[3]  min=0  max=0  (2 -inf, 1 zero, of 3)
  output:
    f32[3]  min=0  max=0  (2 nan, 1 zero, of 3)
  infinities appeared earlier at:
    example.py:4 (entropy)  (log)  > return -(p * jnp.log(p)).sum()
```

**And the rest:**

| | |
|---|---|
| `@expex.fn(jit=, vectorize=, random=, remat=, donate="auto")` | compile, batch, split keys, rematerialise, donate the buffers a call site overwrites |
| `expex.save` / `expex.load` | checkpoints checked against the annotations |
| `expex.diff(a, b)` | two models or checkpoints compared field by field |
| `expex.shard(f, mesh, B="data", D="model")` | shardings from axis names |
| `expex.eps`, `expex.tiny`, `expex.huge` | constants in the dtype of the array they meet |
| `expex.debug.watch(loss, model)` | activations and gradients of every layer, flagging exploding or dead ones |
| `expex.debug.profile(f, ...)` | FLOPs and bytes per source line |

## Ported projects

Popular JAX projects rewritten in this style and checked against the originals:
[PureJaxRL](https://github.com/luchris429/purejaxrl)'s PPO (`examples/ppo/`) and
[mamba-jax](https://github.com/vvvm23/mamba-jax) (`examples/mamba/`: same logits
to 6e-7, 4.5x faster forward, 3x faster gradients).

## More

The full guide, with every feature and its examples: [docs/](docs/README.md).

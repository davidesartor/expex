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

<table>
<tr><th>JAX</th><th>expex</th></tr>

<tr><td colspan="2"><b>Shapes are checked, not commented</b></td></tr>
<tr><td>

```python
def attend(q, k):
    # q: [B,T,D], k: [B,S,D] -> [B,T,S]
    assert q.shape[0] == k.shape[0]
    assert q.shape[2] == k.shape[2]
    return jnp.einsum("btd,bsd->bts", q, k)
```

</td><td>

```python
@expex.fn
def attend[B, T, S, D](
    q: f32[B, T, D],
    k: f32[B, S, D],
) -> f32[B, T, S]:
    return expex.einsum(q, k, to=(B, T, S))
```

</td></tr>

<tr><td colspan="2"><b>Loops are loops</b></td></tr>
<tr><td>

```python
def rnn(w, xs):  # w: [D, D], xs: [T, D]
    def step(h, x):
        h = jnp.tanh(w @ h + x)
        return h, h
    h0 = jnp.zeros(xs.shape[1])
    _, hs = jax.lax.scan(step, h0, xs)
    return hs
```

</td><td>

```python
@expex.fn
def rnn[T, D](
    w: f32[D, D], xs: f32[T, D]
) -> f32[T, D]:
    h, hs = jnp.zeros(D), []
    for t in range(T):  # a lax.scan
        h = jnp.tanh(w @ h + xs[t])
        hs.append(h)
    return jnp.asarray(hs)
```

</td></tr>

<tr><td colspan="2"><b>Branches are <code>if</code></b></td></tr>
<tr><td>

```python
@jax.jit
def safe_log(x):
    ok = x > 0
    safe = jnp.where(ok, x, 1.0)
    return jnp.where(
        ok, jnp.log(safe), -jnp.inf)
```

</td><td>

```python
@expex.fn(jit=True, vectorize=True)
def safe_log(x: f32) -> f32:
    if x > 0:  # lax.cond, any shape
        return jnp.log(x)
    return -jnp.inf
```

</td></tr>

<tr><td colspan="2"><b>Models are classes; identical layers scan themselves</b></td></tr>
<tr><td>

```python
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
```

</td><td>

```python
normal = jax.nn.initializers.normal(0.1)

@expex.struct
class Block[D]:
    w: f32[D, D] = normal
    b: f32[D] = 0.0

    def __call__(self, x: f32[D]) -> f32[D]:
        y = self.w @ x + self.b
        return x + jax.nn.gelu(y)

@expex.struct
class Stack[D]:
    blocks: list[Block[D]]

    def __call__(self, x: f32[D]) -> f32[D]:
        for block in self.blocks:  # a scan
            x = block(x)
        return x

model = Stack.init(key, D=64, blocks=12)
```

</td></tr>
</table>

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

## Tools

They all read the same annotations, so none needs example data unless it
measures something.

| | |
|---|---|
| `expex.prove(f)` | the shapes hold for every size (also `python -m expex pkg`, `pytest --expex pkg`) |
| `expex.cost(f)` | FLOPs and parameters as formulas in the sizes |
| `expex.summary(model)` | one row per module: parameters, FLOPs, shapes |
| `expex.diagram(f)` | an interactive architecture diagram; `.timed(...)` colours it by measured time |
| `expex.diff(a, b)` | two models or checkpoints compared field by field |
| `expex.save` / `expex.load` | checkpoints checked against the annotations |
| `expex.shard(f, mesh, B="data")` | shardings from axis names |
| `@expex.fn(jit=, vectorize=, random=, remat=, donate="auto")` | compile, batch, keys, rematerialise, donate buffers |
| `expex.eps`, `expex.tiny` | constants that take the dtype of the array they meet |
| `expex.debug.find_nan` / `watch` / `profile` | where a NaN appeared; activations and gradients per layer; FLOPs per line |

## Ported projects

Popular JAX projects rewritten in this style and checked against the originals:
[PureJaxRL](https://github.com/luchris429/purejaxrl)'s PPO (`examples/ppo/`) and
[mamba-jax](https://github.com/vvvm23/mamba-jax) (`examples/mamba/`: same logits
to 6e-7, 4.5x faster forward, 3x faster gradients).

## More

The full guide, with every feature and its examples: [docs/](docs/README.md).

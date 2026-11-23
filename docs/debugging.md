# Precision and debugging

[← Cost and diagrams](cost-and-diagrams.md) · [Contents](README.md) · [Reference →](reference.md)

## Precision constants

`expex.eps`, `expex.tiny` and `expex.huge` are the limits of whatever floating dtype
they meet, so the same code is right in bfloat16, float32 and float64:

```python
@expex.fn
def entropy_safe[N](p: f32[N]) -> f32:
    return -(p * jnp.log(p + expex.tiny)).sum()       # tiny of p's dtype

x = jnp.ones(3, jnp.bfloat16)
x + expex.eps                                         # bfloat16's epsilon: 0.0078125
x / (x.sum() + 1e3 * expex.eps)                       # scaled, still in x's dtype
jnp.maximum(x, expex.tiny(x))                         # as an argument: give it the array
```

## Debugging

```python
@expex.fn
def entropy[N](logits: f32[N]) -> f32:
    p = jax.nn.softmax(logits)
    return -(p * jnp.log(p)).sum()

print(expex.debug.find_nan(entropy, jnp.array([0.0, 200.0, -200.0])))
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
and branches. `expex.debug.shapes` annotates every line with its shapes, here for
all sizes at once:

```python
@expex.fn
def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    scores = q @ k.swapaxes(-1, -2)
    return jax.nn.softmax(scores, axis=-1)

print(expex.debug.shapes(attention))
```

```
attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]   [symbolic B, D, T]
   ... | def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
   ... |     scores = q @ k.swapaxes(-1, -2)            # scores: f32[B, T, T]
   ... |     return jax.nn.softmax(scores, axis=-1)     # -> f32[B, T, T]
```

| Tool | |
|---|---|
| `expex.debug.find_nan(f, *args)` | the first operation that made a NaN or inf, with its inputs |
| `expex.debug.shapes(f[, *args])` | every line with its shapes, symbolic or concrete |
| `expex.debug.memory(f, *args)` | peak memory, and which values on which lines make it up |
| `expex.debug.timing(f, *args)` | trace, compile and run time |
| `expex.debug.ops(f, *args)` | operations by kind and line; flags unrolled loops and big constants |
| `f.info()` | everything expex knows about a function |
| `expex.show(tree)` | a pytree as shapes: `Params(w=f32[3, 3], b=f32[3])` |


### Activations and gradients, call by call

`expex.debug.watch` runs one forward and backward pass of a loss and reports
every typed call made on the way: statistics of its output, and of the
gradient of the loss with respect to that output.

```python
x = jr.normal(jr.key(1), (8, 16))
loss = lambda m: (m(x) ** 2).mean()
print(expex.debug.watch(loss, model))       # one row per call, each layer its own row

broken = model.at.blocks[5].w.multiply(1e3)            # plant a problem
w = expex.debug.watch(loss, broken, growth=10.0)
for row in w.flagged:
    print(row.name, row.flags)
w.save("watch.html")                     # collapsible, sortable, RMS per layer as bars
```

```
watch <lambda>: loss 96.568, 25 calls, 0 flagged  (grad_ratio=100.0, growth=10.0)
                                         ──────────────output──────────────  ─────────d loss / d output─────────
  call                   shape       ×n    mean   std  |max|   rms  nan/inf   mean    std  |max|    rms  nan/inf  flags
  Stack.__call__         f32[8, 16]   1    7.05  6.85   30.3  9.83            0.11  0.107  0.473  0.154
    blocks[0].__call__   f32[8, 16]   1  0.0582  1.06   2.82  1.07            2.44   5.52   39.1   6.04
    blocks[1].__call__   f32[8, 16]   1    0.13  1.09    3.2   1.1            2.32   4.71   33.2   5.25
    ...
    blocks[23].__call__  f32[8, 16]   1    7.05  6.85   30.3  9.83            0.11  0.107  0.473  0.154
blocks[0].__call__ ["gradient exploding: RMS 1.8e+03× the layers' median"]
...
blocks[4].__call__ ["gradient exploding: RMS 1.3e+03× the layers' median"]
blocks[5].__call__ ['activations growing: RMS ×2.8e+02 from blocks[4].__call__']
```

- The loss function returns a scalar, or `(loss, aux)` (kept as `w.aux`).
  Its first argument is usually the model: methods of the modules in it are
  named by path (`blocks[3].attn.__call__`), other functions by name, with a
  call index when one is called more than once (`Attention.causal#3`).
- Each row has the output's shape, mean, std, largest magnitude, RMS and
  NaN / inf counts, and the same for `d loss / d output`. A mapped call (by
  itself or with `.vectorize`) is one row over the whole batch: `×n` counts
  the outputs it covers.
- A row is flagged (`!` in the text) when its output or gradient is not
  finite, when its gradient RMS is more than `grad_ratio` (100) times above
  or below the median of the same module over the layers, or when its output
  RMS grows by more than `growth` (10) from the previous layer.
- `w.rows` gives the numbers (`w.rows[1].grad["rms"]`). In a notebook, or
  with `w.save(...)`, it is a page with a bar per layer for repeated modules.
- Loops over layers run unrolled while watching, so each layer is traced on
  its own; `unroll=False` keeps them as one `lax.scan` and gives one row for
  all layers. JAX's caches are cleared before and after, so compiled
  functions are traced again: they compile again on their next call.

---

[← Cost and diagrams](cost-and-diagrams.md) · [Contents](README.md) · [Reference →](reference.md)

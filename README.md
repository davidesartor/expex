# expex

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
· Docs: [docs/](docs/README.md) (site: https://davidesartor.github.io/expex/, pending its first deploy)
· [Changelog](CHANGELOG.md)

**Write the shapes in the signature; get checks, proofs, loops and model tools for free.**

expex is a thin layer over JAX. Annotate arrays with their sizes and the
library reads those annotations: it checks every call, proves the shapes for
every size, turns plain Python loops into `lax.scan`, and draws, costs and
debugs your model. It is still JAX underneath: `jax.jit` and `jax.grad` work as
usual, and a compiled call costs the same as plain `jax.jit`.

Install with `pip install expex` (Python 3.14+, JAX 0.12+).

```python
import jax, jax.numpy as jnp, jax.random as jr
import expex
from expex import f32
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

A wrong call, `attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))`, raises a `ShapeError`
that says which axis is wrong and where its size came from, with a code
(`e.code == "EX101"`) and a link to its section in [docs/errors.md](docs/errors.md):

```
attend(): argument `k` does not match its annotation.

    declared  k: f32[B, S, D]
    received     f32[5, 3, 4]

    B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5.

    attend[B, T, S, D](q: f32[B, T, D], k: f32[B, S, D]) -> f32[B, T, S]
    defined at example.py:2
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex101
```

### Models are classes, loops are loops, batches map themselves

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

params = init(jr.key(0), D=64, L=12)
ys = jax.vmap(stack, in_axes=(None, 0))(params, jnp.ones((8, 64)))

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

model = Stack(jr.key(0), D=64, blocks=12)  # a key first: built from the annotations
ys = model(jnp.ones((8, 64)))  # extra leading axes are mapped over
```

## What the annotations give you

None of this needs example data, unless it measures something:

```python
print(expex.prove(Stack.__call__))
print(expex.cost(Stack.__call__))
```

```
__main__.Stack.__call__: proved for all sizes
Stack.__call__   (sizes: D, L)
  flops        2·D²·L + 10·D·L
  parameters   D²·L + D·L
  largest      f32[D] (D elements)
  note: L = len(self.blocks)
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/transformer_dark.png">
  <img src="docs/images/transformer.png" alt="The transformer of examples/transformer.py drawn from its code, left to right: tokens, index, then a Block × L container holding LayerNorm, Attention, a residual add, LayerNorm, MLP and a second residual add, then the final LayerNorm and matmul">
</picture>

| You write | You get | Guide |
|---|---|---|
| `f32[T + 1]`, `f32[heads, D // heads]`, `def f[R = 6]`, `x.sum(T)` | derived and fixed sizes, axes by name; helpers defined inside a call see its sizes | [Functions](docs/functions.md#sizes-and-axes-by-name) |
| plain `if`, `while`, `for` | on traced values, become `lax.cond`, `lax.while_loop`, `lax.scan` | [Functions](docs/functions.md#plain-python-on-traced-values) |
| `expex.prove(f)`, `python -m expex pkg/ --watch`, `pytest --expex pkg` | shapes proved for every size, also on save and in CI | [Functions](docs/functions.md#proofs-without-data), [CLI](docs/tools.md) |
| `expex.cost(f)`, `expex.debug.profile(f)` | FLOPs and parameters as formulas in the sizes, also per source line | [Cost](docs/cost-and-diagrams.md) |
| `expex.diagram(f)`, `.timed(model, x)`, `e.diagram` | the diagram above, drawn from the code (SVG, PNG, interactive HTML); timed; a `ShapeError`'s failing call in red | [Diagrams](docs/cost-and-diagrams.md#diagrams) |
| `expex.summary(model)`, `expex.inspect(model, x)` | one row per module with parameters and FLOPs; every tool on one page | [Modules](docs/modules.md#model-summary) |
| `model.at.blocks[5].w.set(v)` | a copy with one part changed, checked against its annotation | [Modules](docs/modules.md#changing-part-of-a-model) |
| `expex.save`, `expex.load`, `expex.diff(a, b)` | checkpoints checked on load; models compared field by field | [Modules](docs/modules.md#saving-and-loading) |
| `expex.debug.shapes`, `.find_nan`, `.watch` | shapes line by line, where a NaN first appeared, activations and gradients per layer | [Debugging](docs/debugging.md) |
| `@expex.fn(jit=, random=, remat=, donate="auto", compute=jnp.bfloat16)` | compile, implicit keys, remat, buffer donation, mixed precision | [Compiling](docs/compiling.md) |
| `expex.shard(f, mesh, B="data", D="model")` | shardings from axis names | [Sharding](docs/compiling.md#sharding) |
| `expex.annotate(f, *examples)`, `python -m expex migrate jaxtyping` | annotations proposed for plain JAX code; jaxtyping code runs as it is | [Porting](docs/porting.md) |
| Equinox, Flax, Optax, Orbax | `@expex.fn` on their modules; `expex.mask` for optimizer masks | [Interop](docs/interop.md) |
| `python -m expex shapes file.py --json` | shapes and failures for editors; [VS Code](editors/vscode/) shows them inline | [Editors](docs/tools.md#vs-code) |

## Ported projects

Popular JAX projects rewritten in this style, each checked against the original,
with every shape proved and a [notebook](examples/notebooks/) with outputs
([details](docs/design.md#the-seven-ports-in-detail)). Lines count code only. Speed
is the port's against the original's, jit-compiled on a quiet 4-core CPU: best of 25
interleaved rounds, median in brackets; above 1x the port is faster (source:
[`benchmarks/results/cpu-2026-11-25-v0.1.md`](benchmarks/results/cpu-2026-11-25-v0.1.md);
no GPU numbers yet).

| Project | Port | Lines, port vs original | Matches the original | Speed (CPU) |
|---|---|---|---|---|
| [PureJaxRL](https://github.com/luchris429/purejaxrl) PPO | [`examples/ppo/`](examples/ppo/) | | first 55–64 of 976 updates identical (depends on the CPU), then float32 drift | the same: 1.01x (1.02x) |
| [mamba-jax](https://github.com/vvvm23/mamba-jax) | [`examples/mamba/`](examples/mamba/) | | logits to 6e-7, tokens exact | forward 5.2x (5.8x), gradients 5.4x (6.0x); generation the same, 1.03x (1.00x) |
| [llama-2-jax](https://github.com/ayaka14732/llama-2-jax) | [`examples/llama/`](examples/llama/) | 101 in 1 file vs 478 in 15 | logits to 8e-7, tokens exact | forward 1.08x (1.14x), gradients 1.61x (1.71x), generation 1.48x (1.58x) |
| [Pgx](https://github.com/sotetsuk/pgx) Connect Four, Othello | [`examples/pgx/`](examples/pgx/) | 81 vs 161; 107 vs 204 | every state identical | Connect Four 1.50x (1.61x); Othello 1.07x (1.10x) |
| [Vision Transformer](https://github.com/google-research/vision_transformer) | [`examples/vit/`](examples/vit/) | 85 vs 183 | logits to 5e-7, gradients to 9e-7 | slightly slower: forward 0.92x (0.96x), gradients 0.93x (1.01x) |
| [mctx](https://github.com/google-deepmind/mctx) MuZero, Gumbel MuZero | [`examples/mctx/`](examples/mctx/) | 340 in 1 file vs 691 in 7 | every tree bit-identical | the same: MuZero 1.02x (0.98x), Gumbel 1.08x (1.11x) |
| [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) | [`examples/nerf/`](examples/nerf/) | 138 vs 334 | pixels to 2.4e-7, loss exact | render slightly slower, 0.96x (0.93x); gradients 1.08x (1.02x) |

## More

[The guide](docs/README.md), one page per topic with every feature and its examples ·
[the design](docs/design.md): why, how, what the ports measured, limits ·
[errors](docs/errors.md) by code · [changelog](CHANGELOG.md)

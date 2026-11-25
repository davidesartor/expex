# Reference

[← Precision and debugging](debugging.md) · [Contents](README.md) · [Command line and editors →](tools.md)

## Annotations

```python
from typing import Literal, NamedTuple
from expex import Dim, Int, Key

class RGB(Dim, size=3): ...          # a named fixed size

class Params[D](NamedTuple):         # dataclasses work too
    w: f32[D, D]
    b: f32[D]

@expex.fn
def example[*Batch, H, W, Rows = 6, Cols = 7](   # Rows, Cols: sizes fixed at 6 and 7
    img: f32[*Batch, H, W, RGB],     # *Batch: any number of axes
    idx: Int[*Batch],                # any integer dtype
    p: Params[W],                    # fields are checked, D is bound to W
    pair: f32[Literal[2], ...],      # a literal size, then any axes
    grid: f32[Rows, Cols],           # fixed sizes: 6 x 7, checked as Rows, Cols
    scale: f32,                      # a scalar array or a Python number
    steps: int = 1,                  # a plain Python value
    key: Key | None = None,
) -> f32[*Batch, H]:
    return img.mean((-1, -2)) * scale * steps
```

| Kind | Annotations |
|---|---|
| Arrays | `f16 f32 f64 i8 … i64 u8 … u64 c64 c128`, any of a kind: `Float Int UInt Bool Complex Num Arr` |
| Sizes | type parameters (`def f[B, T]`), `Literal[3]`, `class RGB(Dim, size=3)`, a type parameter fixed at a size (`def f[R = 3]`, `R = Literal[3]` or `R: Literal[3]`, then plain `R`), `*Rest` (any number of axes, the same everywhere), `...` (any number, unchecked), `Any` (one, unchecked) |
| Derived sizes | `f32[T + 1]`, `f32[heads, D // heads]`, `f32[self.heads]`; constraints: a leading `assert T >= 2` |
| Containers | `tuple[...]`, `list[X]`, `dict[str, X]`, `X \| None`, NamedTuples and dataclasses, generic ones too |
| Stacked | `Batch[Transition, T, N]`: a `Transition` whose arrays all have leading axes `T, N` |
| Plain values | `int`, `float`, `bool`, `str`, `Literal`, enums, callables, config classes |

## Ported projects

Existing codebases rewritten in this style and checked against the
originals. Speeds are from [`benchmarks/`](../benchmarks/) on a shared 4-core
CPU under heavy load (results in `benchmarks/results/`): best of interleaved
runs, with medians where they differ; above 1x the port is faster.

- **[PureJaxRL](https://github.com/luchris429/purejaxrl) PPO** (`examples/ppo/`):
  plain loops instead of four nested `lax.scan`s, `if done:` for the
  auto-reset, and the network and advantage estimate written for one
  environment and called on all of them. Identical to the original for the
  first 64 of 976 updates, then drifting as float32 programs do; about the
  same speed (1.05x median, 1.4x best); all shapes proved.
- **[mamba-jax](https://github.com/vvvm23/mamba-jax) Mamba** (`examples/mamba/`):
  the selective scan is a plain `for` loop over time, sizes are used by name
  (`xz[:, :E]`, `jnp.zeros((E, N))`), and generation is a loop carrying the
  layers' states. Logits match the original to 6e-7, gradients to 8e-7
  (relative) and sampled tokens exactly; all 12 functions are proved. On a
  2048-token sequence on CPU it runs 14x faster forward (10x median) and 6x
  faster for gradients (5x median): the original materialises the full
  `T x E x N` decay before scanning, and recomputes it under `jax.checkpoint`
  to save memory. Generation, one token at a time, runs at the original's
  speed (1.05x), with the layers unrolled when compiled as the original's are.
- **[llama-2-jax](https://github.com/ayaka14732/llama-2-jax) Llama 2** (`examples/llama/`):
  grouped-query attention contracted by axis name (`expex.einsum(q, cache.k,
  to=(G, R, T, S))`) with the head layout in the weights' annotations
  (`wq: f32[D, G, R, K]`), norms and the SwiGLU MLP written for one token and
  called on the sequence, and one code path for training and generation: a
  forward pass runs over an empty KV cache, and greedy decoding is a plain
  loop carrying the layers' caches. Logits match the original to 8e-7,
  gradients to 7e-7 (relative) and the tokens from its own KV-cache
  generation loop exactly; all 11 functions are proved. 101 lines of code in
  one file, against 478 in 15 (the model with its shape checks and
  initialisers, and the generation loop). On CPU (4 layers, width 256, 8
  query and 2 key/value heads, batch 4) it runs at about the same speed:
  0.97-1.3x for the forward pass over 256 tokens, 1.0-1.1x for gradients and
  0.8-1.2x for generation.
- **[Pgx](https://github.com/sotetsuk/pgx) Connect Four and Othello** (`examples/pgx/`):
  one game per `@expex.struct`, with the rules as plain `if` / `for` / `while`.
  A Connect Four win is found by a `while` walk from the new stone, and a move
  is `board[row, col] = self.color`. Othello's eight directions are one `for`
  loop, staged as a `lax.scan`. A batch of games is `Othello.new(keys)`,
  mapped over the keys, and `state.step.vectorize(actions)` steps it (a
  plain call never maps over `self`). Over 1,024 random games of each,
  illegal moves and passes included, every field of every state is identical
  to Pgx's: observation, rewards, termination, legal-action mask and board.
  The shapes of all 15 functions are proved. On CPU with 1,024 parallel
  games, Connect Four steps 1.4-1.65x faster than Pgx but takes about 1 s
  longer to compile; Othello runs at Pgx's speed (0.93-1.0x), with its edge
  masks computed once rather than per shift. The rules take half the lines:
  81 vs 161 for Connect Four, 107 vs 204 for Othello.
- **[Vision Transformer](https://github.com/google-research/vision_transformer) ViT** (`examples/vit/`):
  the image size is in the model's signature (`image: f32[GH * P, GW * Q, C]`:
  a grid of GH x GW patches of P x Q pixels), and the image is cut into patches
  by naming the axes it splits into (`expex.rearrange(image, to=(GH, P, Wpx, C))`,
  then `to=(GH, GW, P, Q, C)`) and contracted with the patch kernel by name,
  with no convolution, reshape chain or einops string. A leading `assert Hpx % P
  == 0 and Wpx % Q == 0` states that patches are whole. Heads are contracted by
  axis name (`expex.einsum(q, k, to=(H, T, S))`), the norms and MLP are
  written for one token and the model for one image, the loop over layers is
  a plain `for` (one `lax.scan`), and a batch of images is `model(images)`.
  On non-square images in non-square patches (32 x 24 in 4 x 2), logits match
  the original to 5e-7 and gradients to 9e-7 (relative); all 8 functions are
  proved. 85 lines of code against 183 for `vit_jax/models_vit.py`, or 157
  without its ResNet stem (lines of code only, no blank lines, comments or
  docstrings; the original still has more classifier options, a pre-logits
  layer and dropout, and uses Flax's attention, dense and norm layers, which
  the port writes out). On CPU (ViT-Ti/16, 224 x 224, batch 8, jit-compiled)
  gradients run at about the original's speed (0.92x best, 0.99x median; they
  were 0.7x until expex merged the batch axes of mapped matmuls and unrolled
  short layer lists when compiling, after which the compiled programs match
  the original's operation counts).
- **[mctx](https://github.com/google-deepmind/mctx) MuZero and Gumbel MuZero** (`examples/mctx/search.py`):
  tree search, the hard case for plain control flow. mctx writes the search for
  a batch of trees with `lax.fori_loop`, `jax.vmap`ped `lax.while_loop`s and
  `lax.cond`. The port writes it for one tree, `Tree[N, A, E]`, as plain Python:
  `simulate` takes the root step and then descends `while` the child is
  expanded; `backward` is `while node != ROOT:` with `visits[parent] += 1`;
  `search` is `for sim in range(S):` with `if child == UNVISITED:`. Called on a
  batch of roots, `search` maps over the trees, and the per-tree key split of
  a mapped call reproduces mctx's `split(key, B)`. The recurrent function is
  written for one state and returns a `Step[A]`.
  Over four batched searches of 8 trees with 32 simulations (both policies,
  with and without an invalid-action mask and `max_depth`), every array of
  every tree is bit-identical to mctx's, and so are the action weights, actions
  and root values. One tree searched eagerly in plain Python (no `vmap`, no
  `jit`) matches to one float32 ulp. All 26 functions are proved, including
  the search and both policies: their model and selection callables are typed
  by what they return (`Callable[..., tuple[Step[A], f32[E]]]`). The ported
  parts take 340 lines of code in one file, against 691 in 7 files. That
  counts code lines only (no blanks, comments or docstrings) of `tree`,
  `search`, `action_selection`, `qtransforms` (without `qtransform_by_min_max`),
  `seq_halving`, the two policies with their helpers, and their types in
  `base`. On CPU (batch 64 x 64 simulations, jit-compiled) the port runs at
  the same speed as mctx, within the noise of the runs (0.8-1.3x for both
  policies), and compiles in about the same time.
- **[JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) NeRF** (`examples/nerf/`):
  everything written for one ray and called on a batch of rays: stratified
  sampling, alpha compositing (`weights @ rgb`), the piecewise-constant PDF
  (a `jnp.cumsum` and a `jnp.searchsorted` in place of the original's
  `[..., None, :]` mask trick), with no batch axis and no `[..., None, :]`
  anywhere. The MLP is written for one point, with its sizes in the class
  (`MLP[P, V, W, C]`), and the skip connection is in the structure
  (`before_skip`, `skip`, `after_skip`) rather than an `i % skip_layer` test.
  Random numbers are drawn per batch, as the original draws them, so the
  samples are the same for the same keys. On 64 camera rays (64 coarse + 128
  fine samples, randomized and not), rgb, disparity and opacity match the
  original to 2.4e-7, the training loss exactly, and gradients to 1.6e-6
  (relative); all 12 functions are proved. 138 lines of code in one file,
  against 334 (`model_utils.py` without the noise helper, `models.py` without
  the model-zoo helper, and the dataset's ray generation; 317 against 138
  when both are formatted alike). On CPU (1,024 rays, width 256, depth 8),
  the render and its gradients run at about the original's speed (gradients
  1.02x best, 0.95x median). Written per ray, each layer's weight gradient
  sums over rays and samples; expex merges those batch axes for the matmul,
  as the original's `[rays * samples, features]` reshape does by hand.

## Limitations

- Conversion needs the function's source, so functions defined with `exec` or
  at a bare `python` prompt run unconverted (`f.info()` says why).
- Left as plain Python: code inside `try` / `with`, loops with an `else`,
  in-place updates of anything but a local variable, and variables shared with
  a nested function. A variable assigned on only one traced path reads as
  zeros on the other.
- Mapping over extra axes (by itself or with `.vectorize`) needs them in front,
  and never maps an argument annotated with a `*Rest` shape. A plain call never
  maps over `self`.
- `debug.find_nan` and parts of the prover read JAX internals; they are tested
  on JAX 0.12.
- A `spot-checked` proof verdict is evidence, not proof.

---

[← Precision and debugging](debugging.md) · [Contents](README.md) · [Command line and editors →](tools.md)

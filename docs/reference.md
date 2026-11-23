# Reference

[← Precision and debugging](debugging.md) · [Contents](README.md)

## Annotations

```python
from typing import Literal, NamedTuple
from expex import Dim, Int, Key

class RGB(Dim, size=3): ...          # a named fixed size

class Params[D](NamedTuple):         # dataclasses work too
    w: f32[D, D]
    b: f32[D]

@expex.fn
def example[*Batch, H, W](
    img: f32[*Batch, H, W, RGB],     # *Batch: any number of axes
    idx: Int[*Batch],                # any integer dtype
    p: Params[W],                    # fields are checked, D is bound to W
    pair: f32[Literal[2], ...],      # a literal size, then any axes
    scale: f32,                      # a scalar array or a Python number
    steps: int = 1,                  # a plain Python value
    key: Key | None = None,
) -> f32[*Batch, H]:
    return img.mean((-1, -2)) * scale * steps
```

| Kind | Annotations |
|---|---|
| Arrays | `f16 f32 f64 i8 … i64 u8 … u64 c64 c128`, any of a kind: `Float Int UInt Bool Complex Num Arr` |
| Sizes | type parameters (`def f[B, T]`), `Literal[3]`, `class RGB(Dim, size=3)`, `*Rest` (any number of axes, the same everywhere), `...` (any number, unchecked), `Any` (one, unchecked) |
| Derived sizes | `f32[T + 1]`, `f32[heads, D // heads]`, `f32[self.heads]`; constraints: a leading `assert T >= 2` |
| Containers | `tuple[...]`, `list[X]`, `dict[str, X]`, `X \| None`, NamedTuples and dataclasses, generic ones too |
| Stacked | `Batch[Transition, T, N]`: a `Transition` whose arrays all have leading axes `T, N` |
| Plain values | `int`, `float`, `bool`, `str`, `Literal`, enums, callables, config classes |

## Ported projects

Existing codebases rewritten in this style and checked against the
originals:

- **[PureJaxRL](https://github.com/luchris429/purejaxrl) PPO** (`examples/ppo/`):
  plain loops instead of four nested `lax.scan`s, `if done:` for the
  auto-reset. Identical to the original for the first 21 of 976 updates, then
  drifting as float32 programs do; same speed; all shapes proved.
- **[mamba-jax](https://github.com/vvvm23/mamba-jax) Mamba** (`examples/mamba/`):
  the selective scan is a plain `for` loop over time, sizes are used by name
  (`xz[:, :E]`, `jnp.zeros((E, N))`), and generation is a loop carrying the
  layers' states. Logits match the original to 6e-7, gradients to 8e-7
  (relative) and sampled tokens exactly; all 11 functions are proved. On a
  2048-token sequence on CPU it runs 4.5x faster forward and 3x faster for
  gradients (the original materialises the full `T x E x N` decay before
  scanning, and recomputes it under `jax.checkpoint` to save memory).
- **[llama-2-jax](https://github.com/ayaka14732/llama-2-jax) Llama 2** (`examples/llama/`):
  grouped-query attention contracted by axis name (`expex.einsum(q, cache.k,
  to=(G, R, T, S))`) with the head layout in the weights' annotations
  (`wq: f32[D, G, R, K]`), norms and the SwiGLU MLP written for one token and
  `.vectorize`d, and one code path for training and generation: a forward pass
  runs over an empty KV cache, and greedy decoding is a plain loop carrying the
  layers' caches. Logits match the original to 8e-7, gradients to 7e-7
  (relative) and the tokens from its own KV-cache generation loop exactly; all
  11 functions are proved. 101 lines of code in one file, against 478 in 15
  (the model with its shape checks and initialisers, and the generation loop).
  On CPU (4 layers, width 256, 8 query and 2 key/value heads, batch 4), the
  forward pass over 256 tokens is 1.1-1.4x faster and generation runs at the
  same speed (about 1,000-1,400 tokens/s for both).
- **[Pgx](https://github.com/sotetsuk/pgx) Connect Four and Othello** (`examples/pgx/`):
  one game per `@expex.struct`, with the rules as plain `if` / `for` / `while`.
  A Connect Four win is found by a `while` walk from the new stone, and a move
  is `board[row, col] = self.color`. Othello's eight directions are one `for`
  loop, staged as a `lax.scan`. A batch of games is `.vectorize`. Over 1,024
  random games of each, illegal moves and passes included, every field of
  every state is identical to Pgx's: observation, rewards, termination,
  legal-action mask and board. The shapes of all 14 functions are proved. On
  CPU with 1,024 parallel games, Connect Four steps 1.3x faster than Pgx (2.7 M
  vs 2.0 M steps/s) but takes about 0.6 s longer to compile. Othello runs at
  0.9-1.0x Pgx's speed and compiles in the same time. The rules take half the
  lines: 81 vs 161 for Connect Four, 101 vs 204 for Othello.

## Limitations

- Conversion needs the function's source, so functions defined with `exec` or
  at a bare `python` prompt run unconverted (`f.info()` says why).
- Left as plain Python: code inside `try` / `with`, loops with an `else`,
  in-place updates of anything but a local variable, and variables shared with
  a nested function. A variable assigned on only one traced path reads as
  zeros on the other.
- `vectorize` needs batch axes in front, and not on arguments with a `*Rest` shape.
- `debug.find_nan` and parts of the prover read JAX internals; they are tested
  on JAX 0.12.
- A `spot-checked` proof verdict is evidence, not proof.

---

[← Precision and debugging](debugging.md) · [Contents](README.md)

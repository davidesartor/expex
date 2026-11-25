# Cost and diagrams

[← Keys, compiling and sharding](compiling.md) · [Contents](README.md) · [Precision and debugging →](debugging.md)

## Cost estimates

`expex.cost` counts a function's FLOPs, and a module's parameters, as formulas in
its sizes. Like the prover, it needs no data:

```python
print(expex.cost(Stack.__call__))
print(expex.cost(Stack.__call__, B=64, D=1024, blocks=24))
```

```
Stack.__call__   (sizes: B, D, L)
  flops        2·B·D²·L + 10·B·D·L
  parameters   D²·L + D·L
  largest      f32[B,D] (B·D elements)
  note: L = len(self.blocks)
Stack.__call__   (sizes: B=64, D=1024, L=24)
  flops        3236954112   = 3.24 GFLOP
  parameters   25190400   = 25.2 M
```

Dominant terms come first. For the transformer in `examples/`, the four
`D × D` projections give `8·T·D²·L`, the MLP `4·T·D·F·L`, attention
`4·T²·D·L` and the unembedding `2·T·D·Vocab`. Matrix products and loops are
exact (a `for` over `T` multiplies its body by `T`); elementwise operations
count one flop per element, reshapes and gathers are free, and a `while`
loop counts one iteration and says so.

### Profiles, line by line

`expex.debug.profile` puts `expex.cost`'s numbers on the source. It traces the function with
symbolic sizes, so you don't need data. Each line, and each line of the functions it
calls, shows the shape it produces, its FLOPs as a formula and, once sizes are given,
as a number and a share of the total:

```python
report = expex.debug.profile(Stack.__call__, B=64, D=1024, blocks=24)
report.save("profile.html")           # the source as a heat map, FLOPs or bytes
```

For the transformer in `examples/` (`T=1024, D=512, F=2048, Vocab=32000, blocks=12`):

```
Transformer.__call__   (sizes: T=1024, D=512, F=2048, Vocab=32000, L=12)
  flops  8·T·D²·L + 4·T·D·F·L + 4·T²·D·L + 31·T²·L + 20·T·D·L + 8·T·F·L + 2·T·D·Vocab + 16·T·L + 9·T·D + 9·T + 5·L + 2  = 137 GFLOP
  bytes  136·T²·L + …  = 3.77 GB

── Transformer.__call__   transformer.py:86   self 24.4%
   86 | def __call__[T](self, tokens: Int[T]) -> f32[T, Vocab]:
   87 |     x = self.embed[tokens]                                f32[T,D]       3·T                           3.07 kFLOP   0.0%             2.11 MB
   88 |     for block in self.blocks:  # identical blocks: one …
   89 |         x = block(x)                                      f32[T,D]       ×L -                                       0.0%             calls 75.6%  277 MB
   90 |     return self.norm(x) @ self.unembed                    f32[T,Vocab]   2·T·D·Vocab                   33.6 GFLOP  24.4% ██          131 MB

── MLP.__call__   transformer.py:63   self 37.7%
   64 |     return jax.nn.gelu(x @ self.w1) @ self.w2             f32[T,D]       ×L 4·T·D·F·L + 8·T·F·L        51.7 GFLOP  37.7% ████        931 MB

── Attention.causal   transformer.py:44   self 19.0%
   47 |     scores = q @ k.T / jnp.sqrt(q.shape[-1])              f32[4,T,T]     ×L 2·T²·D·L + 4·T²·L + L      12.9 GFLOP   9.4% █           428 MB
   49 |     weights = jax.nn.softmax(jnp.where(mask, scores, -j…  f32[4,T,T]     ×L 24·T²·L + 4·T·L             302 MFLOP   0.2%             1.11 GB
   50 |     return weights @ v                                    f32[4,T,D//4]  ×L 2·T²·D·L                   12.9 GFLOP   9.4% █           25.2 MB
...
```

- Each operation is counted once, on the innermost line that created it. A line that
  calls into other functions shows their share as `calls`. The per-line counts add up
  to `expex.cost(f).flops`.
- Lines inside a staged loop are multiplied by the loop's trip count and marked: `×L`
  for a loop over a list of modules, `×T` for `range(T)`.
- The last column is the bytes the line's operations produce (data written, not peak
  memory; for peak memory, use `expex.debug.memory`).
- `profile.save("profile.html")`, or a notebook's rich display, gives a self-contained
  page. Each line has a heat map you can switch between FLOPs and bytes, and hovering
  over a line shows its formula. Each function is a collapsible section.

## Diagrams

`expex.diagram` draws a function or a module from a trace with symbolic sizes,
so like `expex.cost` and `expex.prove` it needs no data. The result is a figure
in the style of an architecture schematic from a paper:

```python
d = expex.diagram(Stack.__call__)                  # depth=1: the calls it makes
d.save("stack.svg")                             # the figure (also d.svg, as text)
d.save("stack.html")                            # an interactive page
print(d)                                        # an outline of the calls
expex.diagram(Stack.__call__, depth=2)             # every call opened one level
expex.diagram(Stack.__call__, B=64, blocks=24)     # sizes and list lengths, as for expex.cost
```

For the transformer in `examples/`, `expex.diagram(Transformer.__call__, depth=2)`
draws the flow from left to right: the input `tokens` (`T (i32)`), `index`,
then a rounded container titled `Block × L` holding `LayerNorm → Attention → ⊕
→ LayerNorm → MLP → ⊕`, then the final `LayerNorm` opened (`mean`, `var` and
the `elementwise` arithmetic that normalises), `matmul` and `return`
(`T × Vocab`). Each box has its name in bold and, in grey, what it produces
(`per element · T × D` for a call mapped over extra axes, by itself or with
`.vectorize`). A header names the
function and its sizes (`T · D · Vocab · L = len(blocks)`). `print(d)` gives
an outline:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/transformer_dark.png">
  <img src="images/transformer.png" alt="The transformer drawn at depth 2">
</picture>

```
Transformer.__call__
  × L
    Block.__call__                     f32[T, D] → f32[T, D]
      LayerNorm.__call__ · vectorized  f32[D] → f32[D]
      Attention.__call__               f32[T, D] → f32[T, D]
      ...
  LayerNorm.__call__ · vectorized      f32[D] → f32[D]
```

- Each call to an `@expex.fn` is a rounded box. `depth=2`, or a click on the
  page, opens it into a container holding the calls inside. A loop over
  identical layers is `Name × L`. While closed, it is drawn as a stack of
  layers. The selective scan in `examples/mamba/` is a container `loop × T`.
  A traced `if` or `while` is a dashed container too.
- Between calls, only operations that say what the model does are drawn
  (`matmul`, `index`, `sum`, `conv`, ...). A `sum` divided by the number of
  elements it adds is drawn as `mean`, and one inside `jnp.var` or `jnp.std`
  as `var`. A `⊕` is drawn where two computed values of one shape are added.
  When that is a residual, its skip is a thin line in the stream's colour that
  loops over the row into the `⊕`. Arithmetic that only transforms one value
  (reshapes, `gelu`, ...) is skipped, with arrows drawn straight across it.
  Arithmetic where values meet becomes one `elementwise` box that names its
  operations, so `(x - mean) / std * scale + bias` is `elementwise` with
  `sub · div · mul · add` and the dataflow stays whole. An opened call has one
  entry per value that comes in and one exit per value that leaves.
- Where an array is sliced on its way into a call, the shape it enters with is
  written by the arrow.
- Each input is a stream with its own colour (blue, violet, green, ...), and
  the boxes it reaches take it. A box reached by several streams is grey. An
  extra input that feeds several boxes is a side input (conditioning, say). It
  is orange, and its arrows are dashed.
- The layout is computed in Python, so it needs no browser and no CDN. The SVG
  carries a light and a dark palette and follows the viewer's colour scheme.
  `d.save("x.png")` rasterises the figure at 2x with Playwright or a headless
  Chromium if one is installed, and otherwise with matplotlib. Pass
  `theme="dark"` for a dark PNG.
- The page embeds the figure for each depth, and for the views a click
  reaches. You can pan and zoom it. A click on a call opens it in place, and a
  click on a container's title folds it. Hovering shows the signature, types
  and file and line, and a click also shows the source. It follows the
  system's light or dark mode, or a toggle.

Here is that page for the transformer of `examples/`, saved with
`expex.diagram(Transformer.__call__, depth=2).save("transformer.html")`. On the
[docs site](https://davidesartor.github.io/expex/cost-and-diagrams.html) it is live: click a
call to open it, drag to pan, scroll to zoom. GitHub does not show embedded pages, so here the
frame below stays empty ([the file](assets/transformer.html)).

<iframe class="diagram" src="assets/transformer.html" title="The transformer's interactive diagram" loading="lazy"></iframe>


### Where the time goes, and where shapes fail

`expex.diagram.timed` draws the same diagram from a run on real arguments. It
compiles the function with `jax.jit`, runs it under `jax.profiler`, and colours
each call and loop by its share of the measured time (amber, darker for more). Shape errors get a
diagram too. A `ShapeError` raised inside a typed call carries `e.diagram`,
which is the trace up to the call that failed:

```python
d = expex.diagram.timed(model.__call__, jnp.ones((256, 16)))  # runs it, under jax.jit and jax.profiler
print(d.text)
d.save("stack_timed.svg")                        # each box coloured by its share of the time

@expex.struct
class Probe[D]:
    stack: Stack[D]
    w: f32[D] = 1.0

    def __call__[B](self, x: f32[B, D]) -> f32[B]:
        return self.stack(x[:, 1:]) @ self.w     # a bug: x loses a feature

probe = Probe(jr.key(1), D=16, blocks=2)
try:
    probe(jnp.ones((4, 16)))
except expex.ShapeError as e:
    print(e.diagram.text)                        # the trace up to the call that failed
    e.diagram.save("error.html")                 # that call in red, the path into it in red
```

```
Stack.__call__   measured with jax.profiler (CPU): 285 µs of XLA op time per call (summed over threads), 320 µs wall, median of 5 runs
  × 24                  273 µs   95.8%  ███████████████████
    Block.__call__      250 µs   87.8%  ██████████████████
  (outside the calls)   11.9 µs    4.2%
Probe.__call__   ShapeError: traced up to the failing call
  Stack.__call__  ✗ received f32[4, 15], declared x: f32[B, D]

  path:  x: f32[4, 16] → ✗ Stack.__call__

  Stack.__call__(): argument `x` does not match its annotation.
  ...
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/transformer_timed_dark.png">
  <img src="images/transformer_timed.png" alt="The transformer timed: each call coloured by its share of the time">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="images/transformer_error_dark.png">
  <img src="images/transformer_error.png" alt="A transformer whose attention does not split the keys into heads: the call to Attention.causal that received keys of 24 × 32 where it declared k: f32[T, K], and the path into it, in red">
</picture>

- The profiler names XLA instructions, not Python calls. expex looks each one
  up in the compiled program, where it carries the scopes of the calls it came
  from. A fused instruction that spans calls is split between them. A loop's
  body runs once per iteration, so `× 24` adds up all 24. Each call's time
  includes the calls inside it. Shares are of XLA op time summed over threads,
  so on CPU they can add up to more than the wall time.
- When the profiler can't run, the shares are estimated from FLOPs per box
  (counted as `expex.cost` counts them). The diagram is then labelled `ESTIMATED`.
- `d.timings` maps boxes to (seconds per call, share), and `d.at(2)` opens the
  diagram one level further.
- `e.diagram` is built only when you use it, so calls that succeed cost
  nothing extra. It traces the function again on the arguments' shapes, with
  loops unrolled, up to the call that fails. That call is drawn in red, with
  the type it received and the one it declared, and the edges from the inputs
  to it are red too, with the panels around it open. The error's message is
  unchanged. A note under it says `(see e.diagram.save("error.html"))`.
  `expex.diagram.explain(f, *args)` draws the same thing without running `f`.

---

[← Keys, compiling and sharding](compiling.md) · [Contents](README.md) · [Precision and debugging →](debugging.md)

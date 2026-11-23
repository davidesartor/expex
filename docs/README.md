# expex guide

The full reference, one page per topic. Every example on these pages runs
(they are tested, in this order, as one program).

| Page | Covers |
|---|---|
| [Functions](functions.md) | checked calls, sizes and axes by name (`f32[R:6, 7]` too), proofs, plain Python control flow, automatic batching |
| [Modules](modules.md) | `@expex.struct`, building from the annotations, `model.at` updates, summaries, checkpoints, comparing models |
| [Keys, compiling and sharding](compiling.md) | random keys, `jit` / `remat` / `donate`, gradients, sharding |
| [Cost and diagrams](cost-and-diagrams.md) | FLOPs and parameters as formulas, line profiles, architecture diagrams |
| [Precision and debugging](debugging.md) | `eps` / `tiny` / `huge`, NaN hunting, shapes, memory, watching layers |
| [Reference](reference.md) | annotation syntax, ported projects, limitations |
| [Command line and editors](tools.md) | `python -m expex`, watch mode, shapes in the editor (VS Code) |
| [Porting JAX code](porting.md) | `expex.annotate`: annotations proposed for plain JAX functions from a run on examples |
| [Errors](errors.md) | every error by its code (`EX101`): what triggers it, an example, and the fix |
| [Equinox, Flax, Optax and Orbax](interop.md) | expex in Equinox and Flax modules, optimizers and masks (`expex.mask`), checkpoints with Orbax or `expex.save` |
| [The design of expex](design.md) | the essay behind the library: why shapes in the signature, how it works, what the ports measured, limits, related work |

The short version is the [front page](../README.md).

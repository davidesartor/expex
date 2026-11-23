# Command line and editors

[← Reference](reference.md) · [Contents](README.md) · [Porting JAX code →](porting.md)

The prover and `expex.debug.shapes` from a terminal, while you edit, and in
your editor. None of it needs example data.

## Proving

```
python -m expex pkg/                  # every @fn in a package, a module or a .py file
python -m expex prove pkg/            # the same
pytest --expex pkg                    # one test per function
```

## Watch mode

`--watch` proves everything once, then keeps watching the files. On each save
it re-imports the changed modules and the modules that import them, proves
again only the functions defined in those, and prints only what changed:
functions newly failing, with the usual report, functions newly fixed, and
one summary line.

```
python -m expex prove examples/transformer.py --watch
```

```
ok    transformer.LayerNorm.__call__
...
8 functions: 8 proved
watching 1 file for changes (Ctrl-C to stop)
ERR   transformer.Block.__call__
        the function raised while being traced on abstract inputs (D=8, F=8, T=9):
            TypeError: add got incompatible shapes for broadcasting: (9, 8), (8, 8).
            expex: in Block.__call__(self: Block, x: f32[9, 8])  [D=8, F=8, T=9]
...
[09:04:46] transformer: 8 re-proved in 0.96 s, 3 failing
fixed transformer.Block.__call__
fixed transformer.Transformer.__call__
fixed transformer.Transformer.loss
[09:04:47] transformer: 8 re-proved in 0.56 s, all clear: 8 functions
```

- It is one long-lived process: JAX is imported once, so a save of
  `examples/transformer.py` shows its result in about 0.6 s, where starting
  `python -m expex` again takes 1.5 to 2 s.
- Files are polled by modification time every 0.1 s (`--interval`), or watched
  with [watchfiles](https://pypi.org/project/watchfiles/) when it is installed
  (`--poll` to poll anyway). A cycle starts once the files have been quiet for
  0.1 s (`--debounce`), so an editor's several writes make one cycle.
- A module that fails to import (a syntax error, say) is reported and watched; the
  save that fixes it re-imports it. New files in a watched package are picked up.
- Modules are re-imported in the same process. Code that keeps a reference to an
  old module or class outside the watched folders still sees the old one: restart
  the watch if that matters.
- Ctrl-C stops it.

## Shapes of a file

`python -m expex shapes file.py` lists, for every typed function in the file,
the shape of each assignment and return (`expex.debug.shapes`, for all sizes),
and the proof of each function that is not proved. For this file:

```python
import jax
import expex
from expex import f32


@expex.fn
def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    scores = q @ k.swapaxes(-1, -2)
    return jax.nn.softmax(scores, axis=-1)


@expex.fn
def pool[T, D](x: f32[T, D]) -> f32[D]:
    weights = jax.nn.softmax(x.sum(-1))
    return weights @ x[1:]
```

```
attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]   [symbolic B, D, T]
    6 | @expex.fn
    7 | def attention[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    8 |     scores = q @ k.swapaxes(-1, -2)          # scores: f32[B, T, T]
    9 |     return jax.nn.softmax(scores, axis=-1)   # -> f32[B, T, T]

pool: stopped by the error
   12 | @expex.fn
   13 | def pool[T, D](x: f32[T, D]) -> f32[D]:
   14 |     weights = jax.nn.softmax(x.sum(-1))   # weights: f32[T]
   15 |     return weights @ x[1:]

model.py:15: model.pool: error
    the function raised while being traced on abstract inputs (D=2, T=3):
        TypeError: dot_general requires contracting dimensions to have the same shape, got (3,) and (2,).
        expex: in pool(x: f32[3, 2])  [T=3, D=2]
    see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex704
```

With `--json` the same is data for an editor: one hint per assigned name (its
shape) and per `return` (`-> ` and its shape), spanning the name or the returned
expression; and one diagnostic per failure, on the line where it happened (else
on the `def` line), with severity `error` for failures, `warning` for spot
checks and `information` for functions that could not be checked. Each diagnostic
has the `code` of its error ([errors](errors.md)) and `href`, the link to its section. Lines count
from 1 and columns (in characters) from 0; `file` is the absolute path. The command exits with 1
when there is an error, and the file's own output goes to stderr.

```
python -m expex shapes model.py --json
```

```
{"hints": [
  {"file": "/path/to/model.py", "line": 8, "column": 4, "end_column": 10, "text": "f32[B, T, T]"},
  {"file": "/path/to/model.py", "line": 9, "column": 11, "end_column": 42, "text": "-> f32[B, T, T]"},
  {"file": "/path/to/model.py", "line": 14, "column": 4, "end_column": 11, "text": "f32[T]"}
], "diagnostics": [
  {"file": "/path/to/model.py", "line": 15, "column": 4, "end_column": 26, "severity": "error", "message": "model.pool: error\nthe function raised while being traced on abstract inputs (D=2, T=3):\n    TypeError: dot_general requires contracting dimensions to have the same shape, got (3,) and (2,).\n    expex: in pool(x: f32[3, 2])  [T=3, D=2]", "code": "EX704", "href": "https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex704"}
]}
```

## VS Code

[`editors/vscode/`](../editors/vscode/) is a small extension that runs this
command on every save of a Python file, with the interpreter the Python extension
has selected, and shows the shapes as inlay hints at the end of the lines and the
failures as diagnostics. It is not published: its README says how to build and
install it from source.

---

[← Reference](reference.md) · [Contents](README.md) · [Porting JAX code →](porting.md)

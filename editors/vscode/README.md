# expex shapes for VS Code

Shows what `python -m expex shapes` knows about a file, each time you save it:

- the shape of every assignment and return of each typed function, as an inlay
  hint at the end of its line (`f32[B, T, D]`, `-> f32[B, T]`), for all sizes;
- shape errors and proof failures as diagnostics (squiggles and the Problems panel),
  on the line where the error happened.

It runs `python -m expex shapes <file> --json` with the interpreter the Python
extension has selected for the file, or `expex.python` when there is none, so expex
has to be installed in that environment. It is not published; install it from
source.

## Install from source

Needs Node.js 18 or later.

```
cd editors/vscode
npm install && npm run compile
```

Then either

- package it and install the package: `npx vsce package` writes `expex-shapes-0.1.0.vsix`;
  in VS Code, run **Extensions: Install from VSIX...** and pick it
  (or `code --install-extension expex-shapes-0.1.0.vsix`); or
- run it without installing: open `editors/vscode` in VS Code and press F5
  (**Run Extension**), which starts an Extension Development Host with it loaded.

## Settings

| Setting | Default | |
|---|---|---|
| `expex.hints.enabled` | `true` | show the shapes as inlay hints (also: **expex: Toggle shape hints**) |
| `expex.runOnSave` | `true` | check a Python file when it is saved (or run **expex: Show shapes of this file**) |
| `expex.python` | `""` | interpreter to use when the Python extension has none selected; empty means `python` |
| `expex.timeout` | `120` | seconds before a check is stopped |

Inlay hints also follow VS Code's own `editor.inlayHints.enabled`.

The output channel **expex** shows each command run, how long it took, and the
errors of runs that failed (for example, expex not installed in that interpreter).

## Notes

- A check imports the file, as `python -m expex` does, and takes about a second
  plus the proofs (2.4 s for `examples/transformer.py`).
- Hints are by line: once lines are added or removed, the file's hints are hidden
  until the next save.

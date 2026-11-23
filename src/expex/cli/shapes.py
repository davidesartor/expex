"""``python -m expex shapes file.py [--json]``: shapes and failures of a file, for editors.

For every provable ``@fn`` defined in the file (functions and struct methods)
this runs ``expex.debug.shapes`` (the shape of every assignment and return,
for all sizes) and ``expex.prove`` (the proof), and reports:

hints
    ``{file, line, column, end_column, text}``: one per assigned name, ``text``
    being its shape (``f32[B, T, D]``), and one per ``return`` (``-> f32[B, T]``).
    ``column`` / ``end_column`` span the name, or the returned expression.
diagnostics
    ``{file, line, column, end_column, severity, message, code}``: shape errors and
    proof failures (``error``), spot checks (``warning``) and functions the
    prover could not check (``information``), on the line where the error
    happened when it is known, else on the ``def`` line. ``code`` is the error's
    code (``"EX102"``; ``null`` when it has none, as for a syntax error) and
    ``href`` the section of the guide that explains it, when there is a code.

Lines are 1-based and columns 0-based, in characters. With ``--json``
the output is one object ``{"hints": [...], "diagnostics": [...]}``; without,
the ``debug.shapes`` listings and the failing reports.
"""

import argparse
import ast
import contextlib
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

SEVERITY = {"failed": "error", "error": "error", "spot-checked": "warning", "unchecked": "information"}


@dataclass
class FileShapes:
    """The hints and diagnostics of one file, and what was printed for it."""

    file: str
    hints: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        """``{"hints": [...], "diagnostics": [...]}``, one entry per line."""

        def rows(xs: list[dict[str, Any]]) -> str:
            return "[\n" + ",\n".join("  " + json.dumps(x) for x in xs) + "\n]" if xs else "[]"

        return f'{{"hints": {rows(self.hints)}, "diagnostics": {rows(self.diagnostics)}}}\n'


def _span(src: list[str], line: int) -> tuple[int, int]:
    text = src[line - 1] if 0 < line <= len(src) else ""
    body = text.rstrip()
    return len(body) - len(body.lstrip()), len(body)


def _chars(src: list[str], line: int, offset: int) -> int:
    """A column of the syntax tree (UTF-8 bytes) in characters."""
    text = src[line - 1] if 0 < line <= len(src) else ""
    return len(text.encode()[:offset].decode(errors="ignore"))


def _functions_in(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _node_of(f: Any, defs: list[ast.FunctionDef | ast.AsyncFunctionDef]) -> Any:
    """The syntax-tree node of ``f``'s ``def`` (its code starts at the first decorator)."""
    first = getattr(getattr(f.raw, "__code__", None), "co_firstlineno", None)
    for node in defs:
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        if start == first and node.name == f.raw.__name__:
            return node
    return None


def _hints(
    file: str, trace: Any, node: Any, src: list[str]
) -> list[dict[str, Any]]:
    """One hint per name and per return that ``debug.shapes`` annotated."""
    stores: dict[tuple[int, str], ast.Name] = {}
    returns: dict[int, ast.Return] = {}
    for n in ast.walk(node) if node is not None else ():
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            stores.setdefault((n.lineno, n.id), n)
        elif isinstance(n, ast.Return):
            returns.setdefault(n.lineno, n)
    out = []
    for line, _, notes in trace.lines:
        for note in notes:
            if note.startswith("-> "):
                text, ret = note, returns.get(line)
                target: ast.expr | ast.stmt | None = ret.value if ret is not None and ret.value is not None else ret
            else:
                name, _, shape = note.partition(": ")
                text, target = shape, stores.get((line, name))
            if target is not None and target.end_col_offset is not None and target.end_lineno == line:
                col, end = _chars(src, line, target.col_offset), _chars(src, line, target.end_col_offset)
            else:
                col, end = _span(src, line)
            out.append({"file": file, "line": line, "column": col, "end_column": end, "text": text})
    return out


def _error_line(file: str, f: Any, node: Any, err: BaseException | None, message: str) -> int:
    """Where a failure of ``f`` happened: the innermost line of ``f`` in the traceback,
    the ``return`` for a return value that does not match, else the ``def`` line."""
    if node is None:
        return getattr(getattr(f.raw, "__code__", None), "co_firstlineno", 1)
    lo, hi = node.lineno, node.end_lineno or node.lineno
    seen: set[int] = set()
    while err is not None and id(err) not in seen:
        seen.add(id(err))
        for fr in reversed(traceback.extract_tb(err.__traceback__)):
            if fr.lineno and lo < fr.lineno <= hi and os.path.abspath(fr.filename) == file:
                return fr.lineno
        err = err.__cause__ or err.__context__
    if "the return value does not match" in (message.splitlines() or [""])[0]:
        rets = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        if len(rets) == 1:
            return rets[0].lineno
    for m in re.finditer(re.escape(os.path.basename(file)) + r":(\d+)", message):
        if lo < int(m.group(1)) <= hi:
            return int(m.group(1))
    return node.lineno


def _diagnostic(file: str, line: int, col: int, end: int, severity: str, message: str, code: str | None) -> dict[str, Any]:
    from ..annotations.errors import DOCS_URL, strip_see

    # the link is in `href`, for the editor to show with the code
    out: dict[str, Any] = {"file": file, "line": line, "column": col, "end_column": end,
                           "severity": severity, "message": strip_see(message), "code": code}  # fmt: skip
    if code:
        out["href"] = f"{DOCS_URL}#{code.lower()}"
    return out


def analyse(path: str) -> FileShapes:
    """The hints and diagnostics of every typed function in the file at ``path``."""
    from ..annotations.errors import code_of
    from ..debug import shapes
    from ..tools.prove import ERROR, FAILED, PROVED, Report, _summ, collect, import_target, prove

    file = os.path.abspath(path)
    shown = os.path.relpath(file) if not os.path.relpath(file).startswith("..") else file
    out = FileShapes(file)
    try:
        with open(file) as fh:
            text = fh.read()
        tree = ast.parse(text, file)
    except SyntaxError as e:
        line = e.lineno or 1
        col = max((e.offset or 1) - 1, 0)
        out.diagnostics.append(_diagnostic(file, line, col, col + 1, "error", f"SyntaxError: {e.msg}", None))
        out.text.append(f"{shown}:{line}: SyntaxError: {e.msg}")
        return out
    src = text.splitlines()
    defs = _functions_in(tree)

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    try:
        mod = import_target(path)
    except Exception as e:
        frames = [fr for fr in traceback.extract_tb(e.__traceback__) if os.path.abspath(fr.filename) == file]
        line = (frames[-1].lineno if frames else None) or 1
        col, end = _span(src, line)
        msg = f"cannot import the file: {_summ(e)}"
        out.diagnostics.append(_diagnostic(file, line, col, end, "error", msg, code_of(e)))
        out.text.append(f"{shown}:{line}: {msg}")
        return out

    fns = [
        f for f in collect(mod)
        if os.path.abspath(getattr(getattr(f.raw, "__code__", None), "co_filename", "")) == file
    ]  # fmt: skip
    seen: set[tuple[int, int, str]] = set()
    for f in fns:
        node = _node_of(f, defs)
        try:
            report = prove(f)
        except Exception as e:
            report = Report(f, ERROR, f"the prover raised:\n    {_summ(e)}", code=code_of(e) or "EX704")
        err: BaseException | None = None
        trace: Any
        try:
            trace = shapes(f)
        except Exception as e:  # the shape error itself, with the line it happened on
            err, trace = e, getattr(e, "shapes", None)  # the lines before the error
        if trace is not None:
            out.text.append(str(trace))
            for h in _hints(file, trace, node, src):
                key = (h["line"], h["column"], h["text"])
                if key not in seen:  # a nested function is in its parent's listing too
                    seen.add(key)
                    out.hints.append(h)
        if report.status == PROVED and err is None:
            continue
        code = report.code
        if report.status == PROVED:  # proved, but shapes() could not run it
            severity, message = "warning", f"expex.debug.shapes: {_summ(err)}" if err else ""
            code = code_of(err)
        else:
            severity = SEVERITY[report.status]
            message = "\n".join([report.detail, *report.notes]).strip()
            if report.status in (FAILED, ERROR) and not message and err is not None:
                message = _summ(err)
                code = code or code_of(err)
        line = _error_line(file, f, node, err, message)
        col, end = _span(src, line)
        message = f"{report.name}: {report.status}\n{message}".strip()
        out.diagnostics.append(_diagnostic(file, line, col, end, severity, message, code))
        out.text.append(f"{shown}:{line}: {report}")
    out.hints.sort(key=lambda h: (h["line"], h["column"]))
    out.diagnostics.sort(key=lambda d: d["line"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="expex shapes",
        description="The shape of every assignment and return of each typed function in a file, "
        "and its shape errors and proof failures.",
    )
    ap.add_argument("file", help="a .py file")
    ap.add_argument("--json", action="store_true", help="hints and diagnostics as JSON, for editors")
    ap.add_argument("--x64", action="store_true", help="enable 64-bit floats before importing")
    args = ap.parse_args(argv)
    if args.x64:
        import jax

        jax.config.update("jax_enable_x64", True)
    real = sys.stdout
    # the file's own prints must not end up in the JSON
    with contextlib.redirect_stdout(sys.stderr if args.json else real):
        result = analyse(args.file)
    if args.json:
        real.write(result.to_json())
    else:
        print("\n\n".join(result.text))
    return 1 if any(d["severity"] == "error" for d in result.diagnostics) else 0

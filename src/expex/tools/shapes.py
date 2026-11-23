"""``expex.debug.shapes``: a function's source with the shape of every variable on each line.

The function runs under ``sys.settrace`` while it is traced: abstractly on
the arguments given, or, without arguments, on symbolic sizes from its
annotations (the trace engine's first case), so the shapes hold for all sizes.
"""

import contextlib
import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np

from ..annotations.dims import describe, show
from ..function import BoundFn, Fn, unwrap
from .trace import keyed, resolve, with_key


@dataclass
class ShapeTrace:
    """The source of a function annotated with the shapes assigned on each line."""

    title: str
    lines: list[tuple[int, str, list[str]]]  # (line number, source, annotations)

    def __str__(self) -> str:
        width = max((len(src.rstrip()) for _, src, notes in self.lines if notes), default=0)
        width = min(max(width, 20), 72)
        out = [self.title]
        for n, src, notes in self.lines:
            text = src.rstrip()
            if notes:
                out.append(f"{n:5d} | {text:<{width}}   # {', '.join(notes)}")
            else:
                out.append(f"{n:5d} | {text}")
        return "\n".join(out)

    __repr__ = __str__


def _shape_of(v: Any) -> str | None:
    """A short description of an array-carrying value, or None for anything else."""
    if isinstance(v, (jax.Array, np.ndarray, jax.ShapeDtypeStruct)) or hasattr(v, "aval"):
        try:
            return describe(v)
        except Exception:
            return None
    if callable(v) or isinstance(v, (type, str, bytes)) or v is None:
        return None
    try:
        leaves = jax.tree.leaves(v)
    except Exception:
        return None
    if leaves and any(hasattr(x, "shape") and hasattr(x, "dtype") for x in leaves):
        text = show(v)
        return text if len(text) <= 80 else text[:77] + "..."
    return None


def shapes(f: Callable[..., Any], *args: Any, **kwargs: Any) -> ShapeTrace:
    """Show the shape and dtype of every variable assigned on each line of ``f``.

    With arguments, ``f`` is traced abstractly on them (nothing is computed).
    Without arguments, an ``@fn`` is traced with symbolic sizes from its
    annotations, so the shapes read ``f32[B, T - 1]`` and hold for all sizes.
    Loop bodies show the shapes of one iteration.
    """
    f = unwrap(f)
    target = f.fn if isinstance(f, BoundFn) else f
    raw: Any = getattr(target, "raw", target)
    raw = getattr(raw, "__func__", raw)
    try:
        src_lines, start = inspect.getsourcelines(raw)
        code = raw.__code__
    except (OSError, TypeError, AttributeError):
        raise TypeError("shapes() needs a Python function whose source is available") from None
    fname, end = code.co_filename, start + len(src_lines)
    seen: dict[int, dict[str, list[str]]] = {}
    frames: dict[int, tuple[int, dict[str, int]]] = {}

    def note(line: int, name: str, desc: str) -> None:
        bucket = seen.setdefault(line, {}).setdefault(name, [])
        if desc not in bucket:
            bucket.append(desc)

    def watched(frame: Any) -> bool:
        c = frame.f_code
        return c.co_filename == fname and start <= c.co_firstlineno < end

    def visible(name: str) -> bool:
        return not name.startswith("__") and name != "self"

    def changes(frame: Any, line: int, snap: dict[str, int]) -> dict[str, int]:
        now = {k: id(v) for k, v in frame.f_locals.items() if visible(k)}
        for k, ident in now.items():
            if snap.get(k) != ident:
                desc = _shape_of(frame.f_locals[k])
                if desc is not None:
                    note(line, k, desc)
        return now

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if not watched(frame):
            return None
        key = id(frame)
        if event == "call":
            frames[key] = (frame.f_lineno, {k: id(v) for k, v in frame.f_locals.items() if visible(k)})
        elif event == "line":
            last, snap = frames.get(key, (frame.f_lineno, {}))
            frames[key] = (frame.f_lineno, changes(frame, last, snap))
        elif event == "return":
            last, snap = frames.pop(key, (frame.f_lineno, {}))
            changes(frame, last, snap)
            if frame.f_code is code or frame.f_code.co_name == raw.__name__:
                desc = _shape_of(arg)
                if desc is not None:
                    note(last, "->", desc)
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        if args or kwargs or not isinstance(target, Fn):
            run, a, k = keyed(f, args, kwargs)
            with_key(lambda key: jax.eval_shape(lambda *x, **y: run(key, *x, **y), *a, **k), f)
            title = f"{getattr(raw, '__qualname__', raw.__name__)} at {fname}:{start}"
        else:
            title = _symbolic_run(target)
    except Exception as e:  # the lines before the error, for `python -m expex shapes`
        with contextlib.suppress(AttributeError):
            setattr(e, "shapes", _listing(f"{raw.__qualname__}: stopped by the error", src_lines, start, seen))
        raise
    finally:
        sys.settrace(old)
    return _listing(title, src_lines, start, seen)


def _listing(title: str, src_lines: list[str], start: int, seen: dict[int, dict[str, list[str]]]) -> ShapeTrace:
    rows = []
    for i, text in enumerate(src_lines):
        n = start + i
        notes = [
            (f"-> {' | '.join(d)}" if name == "->" else f"{name}: {' | '.join(d)}")
            for name, d in seen.get(n, {}).items()
        ]
        rows.append((n, text, notes))
    return ShapeTrace(title, rows)


def _symbolic_run(f: Fn[Any, Any]) -> str:
    """Trace ``f`` once on symbolic inputs (the prover's first case)."""
    t = resolve(f, what="show the shapes of")
    t.trace()
    sizes = ", ".join(sorted(n for n in f.input_names if not n.startswith("*")))
    label = ", ".join(f"{k}={v!r}" for k, v in t.fixed.items())
    return f"{f.signature}   [symbolic {sizes}{'; ' + label if label else ''}]"

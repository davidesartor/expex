"""Helpers for walking jaxprs (JAX's intermediate representation)."""

from __future__ import annotations

import os
from typing import Any, Iterator


_HERE = os.path.dirname(os.path.abspath(__file__))


def sub_jaxprs(params: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    """Yield ``(param_name, jaxpr)`` for every jaxpr nested in an equation's params."""
    for k, v in params.items():
        for i, j in enumerate(_as_jaxprs(v)):
            yield (k if i == 0 else f"{k}[{i}]"), j


def _as_jaxprs(v: Any) -> list[Any]:
    if hasattr(v, "eqns"):
        return [v]
    if hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
        return [v.jaxpr]
    if isinstance(v, (tuple, list)):
        out: list[Any] = []
        for x in v:
            out.extend(_as_jaxprs(x))
        return out
    return []


def iter_eqns(jaxpr: Any) -> Iterator[Any]:
    """All equations of a jaxpr, including those in nested jaxprs."""
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    for eqn in jaxpr.eqns:
        yield eqn
        for _, sub in sub_jaxprs(eqn.params):
            yield from iter_eqns(sub)


def source_of(eqn: Any) -> str:
    """``file:line (function)`` of the user code that created an equation."""
    try:
        from jax._src import source_info_util

        tb = eqn.source_info.traceback
        if tb is not None:
            for frame in source_info_util.user_frames(tb):
                fname = frame.file_name
                if not os.path.abspath(fname).startswith(_HERE):
                    return f"{fname}:{frame.start_line} ({frame.function_name})"
        return str(source_info_util.summarize(eqn.source_info))
    except Exception:
        return "<unknown>"


def source_key(eqn: Any) -> tuple[str, int, str]:
    """``(file, line, function)`` for grouping equations by source line."""
    try:
        from jax._src import source_info_util

        tb = eqn.source_info.traceback
        if tb is not None:
            for frame in source_info_util.user_frames(tb):
                fname = frame.file_name
                if not os.path.abspath(fname).startswith(_HERE):
                    return fname, frame.start_line, frame.function_name
    except Exception:
        pass
    return "<unknown>", 0, ""

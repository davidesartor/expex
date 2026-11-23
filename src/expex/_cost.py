"""Old import path kept for ``_diagram.py`` while it is being rewritten: see ``tools/trace.py``."""

from typing import Any

from .tools.cost import flops
from .tools.trace import Target, resolve

_setup = resolve


def _trace(s: Target, n: int, preset: dict[str, int] | None = None) -> tuple[Any, dict[str, Any], list[str]]:
    return s.trace(n, preset)


__all__ = ["_setup", "_trace", "flops"]

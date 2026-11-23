"""Debugging tools that work on the traced program rather than on guesses.

``find_nan(f, *args)``
    Runs the function one operation at a time and reports the first operation
    that produces a NaN from NaN-free inputs: the source line, the operation,
    which loop iteration / branch it was in, and statistics of its inputs.

``ops(f, *args)``
    Size and shape of the traced program: operation counts by kind and by
    source line, the largest intermediates, and a few lints (unrolled loops,
    large captured constants, accidental 64-bit values).

``timing(f, *args)``
    How long tracing, compiling and running take, separately.

``shapes(f, *args)``
    The function's source annotated with the shape of every variable assigned
    on each line (symbolic sizes when called without arguments).

``memory(f, *args)``
    Peak memory of the compiled call, and which values on which lines make it up.

``profile(f, **sizes)``
    FLOPs and bytes per source line, as formulas in the sizes (no data needed).

``watch(loss_fn, *args)``
    Statistics of every typed call's output and gradient in one training step.

Each lives in ``expex/tools/``; this module is where they are found.
"""

from .tools.nan import NanReport, find_nan
from .tools.profile import profile
from .tools.program import Memory, Ops, Timing, memory, ops, timing
from .tools.shapes import ShapeTrace, shapes
from .tools.watch import watch

__all__ = [
    "find_nan", "shapes", "ops", "timing", "memory",
    "NanReport", "ShapeTrace", "Ops", "Timing", "Memory",
    "profile", "watch",
]  # fmt: skip

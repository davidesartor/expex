"""typax: shape-typed functions for JAX.

Annotate with ``f32[B, T]`` and decorate with ``@fn``; the annotations are then
checked on every call, provable for all sizes, and usable for vectorising.
"""

from . import debug
from ._check import Report, prove
from ._errors import RetraceWarning, ShapeError, StageError
from ._flow import rows, stack
from ._fn import Fn, compile_cache, config, fn
from ._rng import key, rng, seed
from ._spec import show
from ._types import (
    Arr, Batch, Bool, Complex, Dim, Etc, Float, Inexact, Int, Key, Num, Static, UInt,
    c64, c128, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
)  # fmt: skip

__all__ = [
    # decorate
    "fn", "prove",
    # annotate
    "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "c64", "c128",
    "Arr", "Bool", "Int", "UInt", "Float", "Complex", "Inexact", "Num",
    "Key", "Static", "Batch", "Dim", "Etc",
    # inside functions
    "key", "rng", "seed", "rows", "stack",
    # tools
    "show", "debug", "config", "compile_cache",
    # types and errors
    "Fn", "Report", "ShapeError", "StageError", "RetraceWarning",
]  # fmt: skip

__version__ = "0.1.0"

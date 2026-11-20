"""typax: shape-typed functions for JAX.

Annotate with ``f32[B, T]`` and decorate with ``@fn``; the annotations are then
checked on every call, provable for all sizes, and usable for vectorising.
"""

from . import debug
from ._check import Report, collect, format_reports, prove
from ._errors import RetraceWarning, ShapeError, StageError
from ._flow import Stacked, rows, stack
from ._fn import BoundFn, Fn, compile_cache, config, fn
from ._rng import key, keys, rng, seed
from ._types import (
    Arr, Batch, Bool, Complex, Dim, Etc, Float, Inexact, Int, Key, Num, Static, UInt,
    c64, c128, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
)  # fmt: skip

__all__ = [
    "fn", "Fn", "BoundFn", "config", "compile_cache",
    "prove", "collect", "Report", "format_reports",
    "key", "keys", "rng", "seed", "Stacked", "rows", "stack", "debug",
    "ShapeError", "StageError", "RetraceWarning",
    "Dim", "Etc", "Static", "Key", "Batch",
    "Arr", "Bool", "Int", "UInt", "Float", "Complex", "Inexact", "Num",
    "f16", "f32", "f64", "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64", "c64", "c128",
]  # fmt: skip

__version__ = "0.1.0"

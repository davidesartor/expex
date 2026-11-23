"""typax: shape-typed functions for JAX.

Annotate with ``f32[B, T]`` and decorate with ``@fn``; the annotations are then
checked on every call, provable for all sizes, and usable for vectorising.
"""

from . import debug
from ._check import Report, prove
from ._dims import show
from ._errors import ShapeError, StageError
from ._fn import Fn, compile_cache, config, fn, jit, lift, random
from ._loops import rows, stack
from ._rng import key, rng, seed
from ._struct import struct
from ._types import (
    Arr,
    Batch,
    Bool,
    Complex,
    Dim,
    Float,
    Inexact,
    Int,
    Key,
    Num,
    Static,
    UInt,
    c64,
    c128,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)  # fmt: skip

__all__ = [
    # decorate
    "fn", "jit", "lift", "random", "struct", "prove",
    # annotate
    "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "c64", "c128",
    "Arr", "Bool", "Int", "UInt", "Float", "Complex", "Inexact", "Num",
    "Key", "Static", "Batch", "Dim",
    # inside functions
    "key", "rng", "seed", "rows", "stack",
    # tools
    "show", "debug", "config", "compile_cache",
    # types and errors
    "Fn", "Report", "ShapeError", "StageError",
]  # fmt: skip

__version__ = "0.1.0"

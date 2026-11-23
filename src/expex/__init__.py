"""expex: shape-typed functions for JAX.

Annotate with ``f32[B, T]`` and decorate with ``@fn``; the annotations are then
checked on every call, provable for all sizes, and usable for vectorising.
"""

from . import debug
from .staging.axes import einsum, rearrange
from .tools.prove import Report, prove
from .consts import eps, huge, tiny
from .tools.cost import cost
from ._diagram import Diagram, diagram
from .tools.diff import Diff, diff
from .tools.summary import Summary, summary
from .io import load, save
from .sharding import device_put, shard, shardings
from .annotations.dims import show
from .annotations.errors import ShapeError, StageError
from .function import Fn, compile_cache, config, fn, jit, vectorize, random, remat
from .staging.loops import rows, stack
from .keys import key, rng, seed
from .modules import struct
from .annotations.types import (
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
    "fn", "jit", "vectorize", "random", "remat", "struct", "prove",
    "save", "load", "shard", "shardings", "device_put", "cost", "diff", "summary", "diagram", "eps", "tiny", "huge",
    # annotate
    "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "c64", "c128",
    "Arr", "Bool", "Int", "UInt", "Float", "Complex", "Inexact", "Num",
    "Key", "Static", "Batch", "Dim",
    # inside functions
    "key", "rng", "seed", "rows", "stack", "einsum", "rearrange",
    # tools
    "show", "debug", "config", "compile_cache",
    # types and errors
    "Fn", "Report", "Diff", "Summary", "Diagram", "ShapeError", "StageError",
]  # fmt: skip

__version__ = "0.1.0"

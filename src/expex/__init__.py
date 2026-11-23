"""expex: shape-typed functions for JAX.

Annotate with ``f32[B, T]`` and decorate with ``@fn``; the annotations are then
checked on every call, provable for all sizes, and usable for vectorising.
"""

from . import debug
from ._diagram import Diagram, diagram
from .annotations.dims import show
from .annotations.errors import ShapeError, StageError
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
from .consts import eps, huge, tiny
from .decorators import fn, jit, random, remat, vectorize
from .function import Fn, compile_cache, config
from .io import load, save
from .keys import key, rng, seed
from .masking import mask
from .modules import ConfigWarning, build, struct
from .sharding import device_put, shard, shardings
from .staging.axes import einsum, rearrange
from .staging.loops import rows, stack
from .tools.cost import cost
from .tools.diff import Diff, diff
from .tools.inspect import Inspect, inspect
from .tools.prove import Report, prove
from .tools.summary import Summary, summary
from .tools.annotate import annotate

__all__ = [
    # decorate
    "fn", "jit", "vectorize", "random", "remat", "struct", "build", "prove",
    "save", "load", "mask", "shard", "shardings", "device_put", "cost", "diff", "summary", "diagram", "inspect", "eps", "tiny", "huge",
    # annotate
    "f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "c64", "c128",
    "Arr", "Bool", "Int", "UInt", "Float", "Complex", "Inexact", "Num",
    "Key", "Static", "Batch", "Dim",
    # inside functions
    "key", "rng", "seed", "rows", "stack", "einsum", "rearrange",
    # tools
    "show", "debug", "config", "compile_cache",
    # types and errors
    "Fn", "Report", "Diff", "Summary", "Diagram", "Inspect", "ShapeError", "StageError", "ConfigWarning",
]  # fmt: skip

__version__ = "0.1.0"

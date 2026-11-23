"""The objects you write in annotations: ``f32[B, T]``, ``Static[int]``, ``Key``.

For static type checkers these are plain aliases (``f32[B, T]`` is a
``jax.Array``, ``Static[int]`` is an ``int``), so annotated code type-checks
without plugins. At runtime they are small marker objects that ``@fn`` reads.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from .errors import coded

__all__ = [
    "Dim", "DType", "ArrayAnn", "StaticAnn", "KeyAnn",
    "Arr", "Bool", "Int", "UInt", "Float", "Complex", "Inexact", "Num",
    "f16", "f32", "f64", "i8", "i16", "i32", "i64",
    "u8", "u16", "u32", "u64", "c64", "c128",
    "Static", "Key", "Batch", "BatchAnn",
]  # fmt: skip


class Dim:
    """Base class for module-level dimensions.

    ``class Hidden(Dim): ...`` declares a named axis that can be shared by many
    functions. ``class RGB(Dim, size=3): ...`` declares a named fixed size.
    """

    size: ClassVar[int | None] = None

    def __init_subclass__(cls, size: int | None = None, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if size is not None:
            cls.size = int(size)


class ArrayAnn:
    """Runtime value of ``f32[B, T]``: a dtype plus unparsed dimension items."""

    __slots__ = ("dtype", "items")

    def __init__(self, dtype: "DType", items: tuple[Any, ...]):
        self.dtype = dtype
        self.items = items

    def __repr__(self) -> str:
        return f"{self.dtype.name}[{', '.join(map(_item_repr, self.items))}]"

    def __or__(self, other: Any) -> Any:
        from typing import Union

        return Union[self, other]

    def __ror__(self, other: Any) -> Any:
        from typing import Union

        return Union[other, self]

    # Lets these objects sit inside typing constructs that insist on callables.
    def __call__(self, *a: Any, **k: Any) -> Any:
        raise coded("EX410", TypeError(f"{self!r} is an annotation, not a constructor"))


def _item_repr(item: Any) -> str:
    return getattr(item, "__name__", None) or repr(item)


class DType:
    """A dtype (``f32``) or dtype category (``Float``) usable as ``f32[B, T]``."""

    def __init__(
        self,
        name: str,
        dtype: Any = None,
        category: Any = None,
        example: Any = None,
        weak: tuple[type, ...] = (),
    ):
        self.name = name
        self.dtype = None if dtype is None else np.dtype(dtype)
        self.category = category
        self.example = np.dtype(example if example is not None else dtype)
        self.weak = weak  # Python scalar types accepted for rank-0 values

    def accepts(self, dtype: Any) -> bool:
        if self.dtype is not None:
            return dtype == self.dtype
        if self.category is None:
            return True
        return any(jnp.issubdtype(dtype, c) for c in self.category)

    def __getitem__(self, items: Any) -> ArrayAnn:
        if not isinstance(items, tuple):
            items = (items,)
        return ArrayAnn(self, items)

    def __repr__(self) -> str:
        return self.name

    def __or__(self, other: Any) -> Any:
        from typing import Union

        return Union[self, other]

    def __ror__(self, other: Any) -> Any:
        from typing import Union

        return Union[other, self]

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise coded("EX410", TypeError(f"{self.name} is an annotation; use jnp.{self.example.name} to convert"))


class StaticAnn:
    """Runtime value of ``Static[int]``."""

    __slots__ = ("inner",)

    def __init__(self, inner: Any):
        self.inner = inner

    def __repr__(self) -> str:
        return f"Static[{_item_repr(self.inner)}]"

    def __or__(self, other: Any) -> Any:
        from typing import Union

        return Union[self, other]

    def __ror__(self, other: Any) -> Any:
        from typing import Union

        return Union[other, self]

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise coded("EX410", TypeError("Static[...] is an annotation"))


class _StaticRuntime:
    def __getitem__(self, inner: Any) -> StaticAnn:
        return StaticAnn(inner)

    def __repr__(self) -> str:
        return "Static"


class BatchAnn:
    """Runtime value of ``Batch[T, B, ...]``: ``T`` with leading axes on every array."""

    __slots__ = ("inner", "items")

    def __init__(self, inner: Any, items: tuple[Any, ...]):
        self.inner = inner
        self.items = items

    def __repr__(self) -> str:
        return f"Batch[{_item_repr(self.inner)}, {', '.join(map(_item_repr, self.items))}]"

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise coded("EX410", TypeError("Batch[...] is an annotation"))

    def __or__(self, other: Any) -> Any:
        from typing import Union

        return Union[self, other]

    def __ror__(self, other: Any) -> Any:
        from typing import Union

        return Union[other, self]


class _BatchRuntime:
    def __getitem__(self, args: Any) -> BatchAnn:
        if not isinstance(args, tuple) or len(args) < 2:
            raise coded("EX401", TypeError("use Batch[Type, Dim, ...], e.g. Batch[Transition, T]"))
        return BatchAnn(args[0], tuple(args[1:]))

    def __repr__(self) -> str:
        return "Batch"


class KeyAnn:
    """Runtime value of ``Key`` / ``Key[B]``: a PRNG key or a batch of them."""

    __slots__ = ("items",)

    def __init__(self, items: tuple[Any, ...] = ()):
        self.items = items

    def __getitem__(self, items: Any) -> "KeyAnn":
        if not isinstance(items, tuple):
            items = (items,)
        return KeyAnn(items)

    def __repr__(self) -> str:
        return "Key" + (f"[{', '.join(map(_item_repr, self.items))}]" if self.items else "")

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise coded("EX410", TypeError("Key is an annotation; use jax.random.key(seed)"))

    def __or__(self, other: Any) -> Any:
        from typing import Union

        return Union[self, other]

    def __ror__(self, other: Any) -> Any:
        from typing import Union

        return Union[other, self]


_INT = (int,)
_FLT = (int, float)
_CPX = (int, float, complex)

_runtime = {
    "Arr": DType("Arr", example=np.float32, weak=(bool, int, float, complex)),
    "Bool": DType("Bool", dtype=np.bool_, weak=(bool,)),
    "Int": DType("Int", category=(jnp.integer,), example=np.int32, weak=_INT),
    "UInt": DType("UInt", category=(jnp.unsignedinteger,), example=np.uint32, weak=_INT),
    "Float": DType("Float", category=(jnp.floating,), example=np.float32, weak=_FLT),
    "Complex": DType("Complex", category=(jnp.complexfloating,), example=np.complex64, weak=_CPX),
    "Inexact": DType("Inexact", category=(jnp.inexact,), example=np.float32, weak=_CPX),
    "Num": DType("Num", category=(jnp.number,), example=np.float32, weak=_CPX),
}
for _n, _d, _w in [
    ("f16", np.float16, _FLT), ("f32", np.float32, _FLT), ("f64", np.float64, _FLT),
    ("i8", np.int8, _INT), ("i16", np.int16, _INT), ("i32", np.int32, _INT),
    ("i64", np.int64, _INT), ("u8", np.uint8, _INT), ("u16", np.uint16, _INT),
    ("u32", np.uint32, _INT), ("u64", np.uint64, _INT),
    ("c64", np.complex64, _CPX), ("c128", np.complex128, _CPX),
]:  # fmt: skip
    _runtime[_n] = DType(_n, dtype=_d, weak=_w)

#: numpy dtype -> short name, for printing received values.
SHORT_NAMES: dict[Any, str] = {
    d.dtype: n for n, d in _runtime.items() if d.dtype is not None
}

if TYPE_CHECKING:
    type Arr[*Shape] = jax.Array
    type Bool[*Shape] = jax.Array
    type Int[*Shape] = jax.Array
    type UInt[*Shape] = jax.Array
    type Float[*Shape] = jax.Array
    type Complex[*Shape] = jax.Array
    type Inexact[*Shape] = jax.Array
    type Num[*Shape] = jax.Array
    type f16[*Shape] = jax.Array
    type f32[*Shape] = jax.Array
    type f64[*Shape] = jax.Array
    type i8[*Shape] = jax.Array
    type i16[*Shape] = jax.Array
    type i32[*Shape] = jax.Array
    type i64[*Shape] = jax.Array
    type u8[*Shape] = jax.Array
    type u16[*Shape] = jax.Array
    type u32[*Shape] = jax.Array
    type u64[*Shape] = jax.Array
    type c64[*Shape] = jax.Array
    type c128[*Shape] = jax.Array
    type Static[T] = T
    type Key[*Shape] = jax.Array
    type Batch[T, *Shape] = T
else:
    Batch = _BatchRuntime()
    globals().update(_runtime)
    Static = _StaticRuntime()
    Key = KeyAnn()

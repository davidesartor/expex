"""jaxtyping annotations, read as expex ones: ``Float[Array, "b t d"]`` is ``Float[b, t, d]``.

Code annotated for `jaxtyping <https://github.com/patrick-kidger/jaxtyping>`_
gets expex's checks, proofs, mapping over extra axes and diagrams as it is.
The names in the shape strings are the function's sizes, as if they were its
type parameters, kept as written (``batch`` stays ``batch``).

Two ways in, one meaning:

* jaxtyping installed: ``Float[Array, "b t"]`` is one of its classes, read
  through its ``dtype``, ``dtypes`` and ``dim_str`` (:func:`from_class`);
* jaxtyping absent (imported under ``if TYPE_CHECKING:``): Python 3.14 keeps
  the annotation's text, and :func:`stand_ins` gives the missing names
  (``Float``, ``Array``, ``PRNGKeyArray``, ...) stand-ins that build the same
  :class:`JtArray`. expex never imports jaxtyping.

Only parsing happens here, at decoration time: a call is checked by the same
specs as ``f32[B, T]``, so expex-style annotations pay nothing for this.
"""

import builtins
import typing
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .dims import (
    _ANY,
    _ELLIPSIS,
    DBroad,
    DBroadConst,
    DConst,
    DExpr,
    DimError,
    DName,
    DVar,
    DVarBroad,
)
from .errors import coded
from .expr import Expr
from .types import DType, KeyAnn, _runtime

# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------

_INT = (int,)
_FLT = (int, float)
_CPX = (int, float, complex)


def _exact(name: str, dtype: Any, weak: tuple[type, ...]) -> DType | None:
    try:
        return DType(name, dtype=jnp.dtype(dtype), weak=weak)
    except Exception:  # a dtype this NumPy / ml_dtypes does not have (int2, float4, ...)
        return None


class _ByName(DType):
    """A set of dtypes given by their names, as jaxtyping matches them (a custom ``AbstractDtype``)."""

    def __init__(self, name: str, names: tuple[str, ...]):
        example = next((n for n in ("float32", "int32", "uint32", "bool", "complex64") if n in names), None)
        try:
            ex = np.dtype(example or names[0])
        except Exception:
            ex = np.dtype(np.float32)
        super().__init__(name, example=ex, weak=(bool, int, float, complex))
        self.category = ()  # not None: `accepts` is overridden
        self.names = frozenset(names)

    def accepts(self, dtype: Any) -> bool:
        return getattr(dtype, "name", str(dtype)) in self.names


#: jaxtyping dtype class name -> expex dtype. ``"key"`` is a PRNG key (``Key[Array, ""]``).
DTYPES: dict[str, Any] = {
    "Shaped": _runtime["Arr"],
    "Bool": _runtime["Bool"],
    "Float": _runtime["Float"],
    "Complex": _runtime["Complex"],
    "Inexact": _runtime["Inexact"],
    "Num": _runtime["Num"],
    "UInt": _runtime["UInt"],
    "Integer": _runtime["Int"],  # signed and unsigned, as expex's Int
    # jaxtyping's Int is signed only, unlike expex's
    "Int": DType("Int", category=(jnp.signedinteger,), example=np.int32, weak=_INT),
    "Real": DType("Real", category=(jnp.floating, jnp.integer), example=np.float32, weak=_FLT),
    "Key": "key",
}
for _jt, _ex in [
    ("Float16", "f16"), ("Float32", "f32"), ("Float64", "f64"),
    ("Int8", "i8"), ("Int16", "i16"), ("Int32", "i32"), ("Int64", "i64"),
    ("UInt8", "u8"), ("UInt16", "u16"), ("UInt32", "u32"), ("UInt64", "u64"),
    ("Complex64", "c64"), ("Complex128", "c128"),
]:  # fmt: skip
    DTYPES[_jt] = _runtime[_ex]
for _jt, _np, _w in [
    ("BFloat16", "bfloat16", _FLT), ("Int2", "int2", _INT), ("Int4", "int4", _INT),
    ("UInt2", "uint2", _INT), ("UInt4", "uint4", _INT), ("Float4e2m1fn", "float4_e2m1fn", _FLT),
    ("Float8e4m3b11fnuz", "float8_e4m3b11fnuz", _FLT), ("Float8e4m3fn", "float8_e4m3fn", _FLT),
    ("Float8e4m3fnuz", "float8_e4m3fnuz", _FLT), ("Float8e5m2", "float8_e5m2", _FLT),
    ("Float8e5m2fnuz", "float8_e5m2fnuz", _FLT),
]:  # fmt: skip
    _d = _exact("bf16" if _jt == "BFloat16" else _jt, _np, _w)
    DTYPES[_jt] = _d if _d is not None else _ByName(_jt, (_np,))


# --------------------------------------------------------------------------
# What a jaxtyping annotation becomes before it is a spec
# --------------------------------------------------------------------------


class JtArray:
    """``Float[Array, "b t"]``: an expex dtype (or ``"key"``) and jaxtyping's shape string."""

    __slots__ = ("dtype", "dim_str", "array_type", "written")

    def __init__(self, dtype: Any, dim_str: str, array_type: Any = None, written: str = ""):
        self.dtype = dtype
        self.dim_str = dim_str
        self.array_type = array_type
        #: As written: ``Float[Array, 'b t']``, for messages.
        self.written = written or f"{getattr(dtype, 'name', dtype)}[..., {dim_str!r}]"

    def __repr__(self) -> str:
        return self.written

    def __or__(self, other: Any) -> Any:
        return typing.Union[self, other]

    def __ror__(self, other: Any) -> Any:
        return typing.Union[other, self]

    def __call__(self, *a: Any, **k: Any) -> Any:  # typing wants callables in unions
        raise coded("EX410", TypeError(f"{self.written} is an annotation, not a constructor"))


class JtPyTree:
    """``PyTree[...]``: expex has no "any pytree of X" annotation, so it is not checked."""

    def __init__(self, written: str = "PyTree"):
        self.written = written

    def __repr__(self) -> str:
        return self.written

    def __getitem__(self, item: Any) -> "JtPyTree":
        return JtPyTree(f"PyTree[{item!r}]" if not isinstance(item, tuple) else f"PyTree[{', '.join(map(repr, item))}]")

    def __or__(self, other: Any) -> Any:
        return typing.Union[self, other]

    def __ror__(self, other: Any) -> Any:
        return typing.Union[other, self]

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise coded("EX410", TypeError(f"{self.written} is an annotation"))


_ARRAY_LIKE = (bool, int, float, complex)


def _written(dtype_name: str, array_type: Any, dim_str: str) -> str:
    t = getattr(array_type, "__name__", None) or repr(array_type)
    return f"{dtype_name}[{t}, {dim_str!r}]"


def _subscript(dtype_name: str, dtype: Any, item: Any) -> Any:
    if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[1], str):
        raise coded("EX401", DimError(
            f"{dtype_name}[...] takes an array type and a shape string, as in jaxtyping: "
            f"{dtype_name}[Array, \"b t\"] (\"...\" for any shape, \"\" for a scalar)"
        ))
    array_type, dim_str = item[0], item[1].strip()
    if array_type in _ARRAY_LIKE:  # Float[float, ""] is `float`, as jaxtyping has it
        if dim_str:
            raise coded("EX401", DimError(
                f"{dtype_name}[{array_type.__name__}, {dim_str!r}]: a Python {array_type.__name__} "
                f"has no axes; write {dtype_name}[{array_type.__name__}, \"\"] or use an array type"
            ))
        return array_type
    return JtArray(dtype, dim_str, array_type, _written(dtype_name, array_type, dim_str))


class _StandIn:
    """``Float`` and friends when jaxtyping is not imported: ``Float[Array, "b"]`` is a :class:`JtArray`."""

    def __init__(self, name: str):
        self.name = name

    def __getitem__(self, item: Any) -> Any:
        return _subscript(self.name, DTYPES[self.name], item)

    def __repr__(self) -> str:
        return self.name


class _Module:
    """``jaxtyping`` (or ``jt``) itself, for ``jaxtyping.Float[jaxtyping.Array, "b"]``."""

    def __getattr__(self, name: str) -> Any:
        if name in STAND_INS:
            return STAND_INS[name]
        return object.__getattribute__(self, name)  # AttributeError, as for a missing name


STAND_INS: dict[str, Any] = {name: _StandIn(name) for name in DTYPES}
STAND_INS.update({
    "Array": jax.Array,
    "ArrayLike": jax.Array,
    "PRNGKeyArray": KeyAnn(),
    "Scalar": JtArray(_runtime["Arr"], "", jax.Array, "Scalar"),
    "ScalarLike": JtArray(_runtime["Arr"], "", jax.Array, "ScalarLike"),
    "PyTree": JtPyTree(),
})  # fmt: skip
#: Names a module is commonly imported as.
MODULES = ("jaxtyping", "jt", "jtyping")


def stand_ins(text: str, glb: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    """Stand-ins for the jaxtyping names ``text`` uses that are not defined: ``{"Float": ..., "Array": ...}``.

    Empty if ``text`` uses no such name, so that other annotations are left alone.
    """
    import re

    out: dict[str, Any] = {}
    for name in set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)", text)):
        if name in scope or name in glb or hasattr(builtins, name):
            continue
        if name in STAND_INS:
            out[name] = STAND_INS[name]
        elif name in MODULES:
            out[name] = _Module()
    return out


# --------------------------------------------------------------------------
# From annotation objects
# --------------------------------------------------------------------------


def _is_jaxtyping(obj: Any, kind: str) -> bool:
    meta = type(obj)
    return meta.__module__.startswith("jaxtyping") and meta.__name__ == kind


def from_class(cls: Any) -> Any:
    """A jaxtyping class (``jaxtyping.Float[Array, "b"]``) as a :class:`JtArray` or :class:`JtPyTree`; else None."""
    if not isinstance(cls, type):
        return None
    if _is_jaxtyping(cls, "_MetaAbstractArray"):
        dt = cls.dtype  # type: ignore[attr-defined]
        name = getattr(dt, "__name__", "Shaped")
        dtype = DTYPES.get(name)
        if dtype is None:  # a custom AbstractDtype: its dtypes are names
            names = cls.dtypes  # type: ignore[attr-defined]
            if isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names):
                dtype = _ByName(name, tuple(names))
            elif isinstance(names, str):
                dtype = _ByName(name, (names,))
            else:  # regexes, or any dtype
                dtype = _runtime["Arr"]
        return JtArray(dtype, cls.dim_str, cls.array_type, cls.__name__.removeprefix("jaxtyping."))  # type: ignore[attr-defined]
    if _is_jaxtyping(cls, "_MetaPyTree"):
        return JtPyTree(cls.__name__.removeprefix("jaxtyping."))
    return None


def from_expex(dtype: Any, items: tuple[Any, ...]) -> JtArray | None:
    """expex's own ``Float`` / ``Key`` written jaxtyping's way: ``Float[Array, "b t"]``; else None."""
    if len(items) != 2 or not isinstance(items[1], str):
        return None
    if not (isinstance(items[0], type) or typing.get_origin(items[0]) is not None or items[0] is typing.Any):
        return None
    name = dtype.name if isinstance(dtype, DType) else "Key"
    return JtArray(dtype if isinstance(dtype, DType) else "key", items[1].strip(), items[0], _written(name, items[0], items[1]))


_SCALAR_TYPES = (bool, int, float, complex, np.bool_, np.number, np.generic)


def union(args: tuple[Any, ...]) -> JtArray | None:
    """A union jaxtyping builds, as one annotation; else None.

    ``PRNGKeyArray`` (``Key[Array, ""] | UInt32[Array, "2"] | ...``) is a key;
    ``Float[Array | np.ndarray, "b"]`` and ``ScalarLike`` (array types and Python
    scalars with one shape) are that shape: expex does not check the array type.
    """
    jts = [from_class(a) for a in args]
    arrays = [j for j in jts if isinstance(j, JtArray)]
    if not arrays or any(j is None and a not in _SCALAR_TYPES for j, a in zip(jts, args)):
        return None
    keys = [j for j in arrays if j.dtype == "key"]
    if len(keys) == 1:
        k = keys[0]
        return JtArray("key", k.dim_str, k.array_type, "PRNGKeyArray" if not k.dim_str else k.written)
    if len({(id(j.dtype), " ".join(j.dim_str.split())) for j in arrays}) == 1:
        return arrays[0]
    return None


# --------------------------------------------------------------------------
# Shape strings
# --------------------------------------------------------------------------


def _unsupported(written: str, elem: str, why: str) -> TypeError:
    return coded("EX411", DimError(f"{written}: `{elem}` {why}"))


def parse_shape(dim_str: str, written: str = "") -> tuple[tuple[Any, ...], int | None]:
    """jaxtyping's shape string as expex dimensions, and the index of the variadic one.

    ``"b t"`` -> ``b, t``; ``"*batch d"`` -> ``*batch, d``; ``"..."`` -> ``...``;
    ``"3"`` -> ``3``; ``"t=1"`` -> ``1``; ``"t+1"`` -> the expression ``t + 1``;
    ``"_"`` / ``"_t"`` -> one unchecked axis; ``"#b"`` -> ``b``, or 1.
    """
    written = written or repr(dim_str)
    dims: list[Any] = []
    var: int | None = None
    for index, raw in enumerate(dim_str.split()):
        if "," in raw and "(" not in raw:
            raise coded("EX401", DimError(f"{written}: separate axes with spaces, not commas"))
        elem = raw
        broad = star = anon = tree = False
        if elem == "...":
            star = anon = True
            elem = ""
        else:
            while elem:
                c = elem[0]
                if c == "#" and not broad:
                    broad, elem = True, elem[1:]
                elif c == "*" and not star:
                    star, elem = True, elem[1:]
                elif c == "_" and not anon:  # `_` and `_t`: unchecked, the name is for the reader
                    anon, elem = True, elem[1:]
                elif c == "?" and not tree:
                    tree, elem = True, elem[1:]
                elif elem.count("=") == 1:  # `rows=3` is `3`, a name for the reader only
                    elem = elem.split("=")[1]
                else:
                    break
        if tree:
            raise _unsupported(
                written, raw,
                "depends on the position in a PyTree, and expex has no PyTree[...] annotation. "
                "Give the axis a plain name, or `_` to leave it unchecked.",
            )  # fmt: skip
        if star:
            if var is not None:
                raise coded("EX401", DimError(f"{written}: at most one variadic (`*name` or `...`)"))
            var = index
        if not elem or elem.isidentifier():
            if anon:
                if broad:
                    raise coded("EX401", DimError(f"{written}: `{raw}`: an unchecked axis cannot also broadcast"))
                dims.append(_ELLIPSIS if star else _ANY)
            elif star:
                dims.append(DVarBroad(elem) if broad else DVar(elem))
            else:
                dims.append(DBroad(elem) if broad else DName(elem))
            continue
        try:
            size = int(elem)
        except ValueError:
            size = None
        if size is not None:
            if star or anon:
                raise coded("EX401", DimError(f"{written}: `{raw}`: a fixed size cannot be `*` or `_`"))
            dims.append(DBroadConst(size) if broad else DConst(size))
            continue
        if star or anon:
            raise coded("EX401", DimError(f"{written}: `{raw}`: an expression cannot be `*` or `_`"))
        if broad:
            raise _unsupported(
                written, raw,
                "is an expression that may broadcast; expex's sizes computed from others are exact. "
                "Drop the `#`, or name the axis and leave it unchecked with `_`.",
            )  # fmt: skip
        if "{" in elem:  # an f-string: "{n}" is the int parameter n, "{self.heads}" the int field heads
            import re

            plain = re.sub(r"\{\s*(?:self\s*\.\s*)?([A-Za-z_]\w*)\s*\}", r"(\1)", elem)
            if "{" in plain or "}" in plain:
                raise _unsupported(
                    written, raw,
                    "uses an f-string expression; only `{n}` of an int parameter and `{self.n}` of an int field are read.",
                )  # fmt: skip
            elem = plain
        import ast

        try:
            src = ast.unparse(ast.parse(elem, mode="eval"))
        except SyntaxError:
            src = elem  # Expr reports it, with its code
        dims.append(DExpr(Expr(src)))
    return tuple(dims), var

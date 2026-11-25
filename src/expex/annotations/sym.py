"""Size arithmetic written directly in annotations: ``f32[T + 1]``, ``f32[D // heads]``.

Python cannot evaluate ``T + 1`` when ``T`` is a type parameter. Python 3.14
evaluates annotations lazily, so such an annotation is still available as
text; expex evaluates that text again with every size, parameter and field of
``self`` replaced by a :class:`Sym`, which records the arithmetic as an
expression.
"""

import ast
import functools
import inspect
import re
import types
import typing
import weakref
from typing import Any

from .errors import coded

_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "//": 2, "%": 2}


class Sym:
    """A size, a parameter or a field, as it appears in an annotation."""

    __slots__ = ("src", "prec", "is_dim")

    def __init__(self, src: str, prec: int = 9, is_dim: bool = False):
        self.src = src
        self.prec = prec
        #: A plain size name (a type parameter), as opposed to a parameter or an expression.
        self.is_dim = is_dim

    def _op(self, op: str, other: Any, swap: bool = False) -> Sym:
        p = _PRECEDENCE[op]
        a, b = (other, self) if swap else (self, other)

        def side(x: Any, right: bool) -> str:
            text = x.src if isinstance(x, Sym) else repr(x)
            prec = x.prec if isinstance(x, Sym) else 9
            return f"({text})" if prec < p or (right and prec == p) else text

        if not isinstance(other, (Sym, int)) or isinstance(other, bool):
            return NotImplemented
        return Sym(f"{side(a, False)} {op} {side(b, True)}", p)

    def __add__(self, o: Any) -> Sym:
        return self._op("+", o)

    def __radd__(self, o: Any) -> Sym:
        return self._op("+", o, True)

    def __sub__(self, o: Any) -> Sym:
        return self._op("-", o)

    def __rsub__(self, o: Any) -> Sym:
        return self._op("-", o, True)

    def __mul__(self, o: Any) -> Sym:
        return self._op("*", o)

    def __rmul__(self, o: Any) -> Sym:
        return self._op("*", o, True)

    def __floordiv__(self, o: Any) -> Sym:
        return self._op("//", o)

    def __rfloordiv__(self, o: Any) -> Sym:
        return self._op("//", o, True)

    def __mod__(self, o: Any) -> Sym:
        return self._op("%", o)

    def __rmod__(self, o: Any) -> Sym:
        return self._op("%", o, True)

    # Not size arithmetic: refused loudly, or the annotation would silently go unchecked.
    def _refuse(self, op: str, o: Any, right: bool, fix: str) -> Any:
        from .expr import ExprError

        raise coded("EX403", ExprError(f"`{op}` in a size annotation (with {self.src}) is not size arithmetic: {fix}"))

    def __truediv__(self, o: Any) -> Any:
        return self._refuse("/", o, False, "sizes are integers, divide with `//` (and add a leading `assert ... % ... == 0` if it must be exact).")

    def __rtruediv__(self, o: Any) -> Any:
        return self._refuse("/", o, True, "sizes are integers, divide with `//`.")

    def __pow__(self, o: Any) -> Any:
        return self._refuse("**", o, False, "write the product out, e.g. `T * T`.")

    def __rpow__(self, o: Any) -> Any:
        return self._refuse("**", o, True, "powers of sizes are not supported; give the result its own size name.")

    def __call__(self, *a: Any) -> Any:  # lets typing accept it where it wants a type
        raise coded("EX410", TypeError("a size is not callable"))

    def __repr__(self) -> str:
        return self.src


class _Self:
    """``self`` in an annotation: its attributes are fields, e.g. ``self.heads``."""

    def __getattr__(self, name: str) -> Sym:
        if name.startswith("__"):
            raise AttributeError(name)
        return Sym(name)


def caller_locals() -> dict[str, Any]:
    """The local variables where a decorator is being applied (outside expex).

    An annotation expex evaluates from its text sees only the module's globals,
    so a class defined inside a function would not be found. This is what lets
    expex find it. Inside a class body, the enclosing function's locals are
    included too.
    """
    import sys

    f = sys._getframe(1)
    while f is not None and f.f_globals.get("__name__", "").split(".")[0] == "expex":
        f = f.f_back  # type: ignore[assignment]
    out: dict[str, Any] = {}
    for _ in range(2):
        if f is None or f.f_code.co_name == "<module>":
            break
        out = {**f.f_locals, **out}
        if "__qualname__" not in f.f_locals:  # not a class body: done
            break
        f = f.f_back  # type: ignore[assignment]
    return out


def signature(f: Any) -> inspect.Signature:
    """``inspect.signature``, also for functions whose annotations do size arithmetic."""
    try:
        return inspect.signature(f)
    except (TypeError, NameError):
        import annotationlib  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]  # Python 3.14

        return inspect.signature(f, annotation_format=annotationlib.Format.STRING)  # type: ignore[call-arg]


def texts(obj: Any) -> dict[str, Any]:
    """An object's annotations, as values where Python can evaluate them, else as text.

    Always as text where they declare sizes (``f32[R:6]``): a global ``R`` must not be used.
    """
    try:
        if not _declares(obj):
            return dict(inspect.get_annotations(obj))
    except Exception:
        pass
    try:  # Python 3.14: an annotation could not be evaluated; ask for the source text
        import annotationlib  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        return dict(annotationlib.get_annotations(obj, format=annotationlib.Format.STRING))
    except Exception:
        return {}


def evaluate(text: str, glb: dict[str, Any], scope: dict[str, Any], params: Any = ()) -> Any:
    """Evaluate an annotation's text; with sizes as :class:`Sym` if it does arithmetic on them.

    Raises if a name is unknown (e.g. a class that is not defined yet).
    """
    if not isinstance(text, str):
        return text  # evaluated already: a forward reference that Python can now resolve
    try:
        out = eval(text, glb, scope)  # noqa: S307 - as typing.get_type_hints
        # an annotation written in quotes, read from its text: quoted twice
        return evaluate(out, glb, scope, params) if isinstance(out, str) else out
    except NameError:
        pass  # perhaps a parameter, or self: try symbolically
    except TypeError:
        pass  # arithmetic on a type parameter
    sym: dict[str, Any] = {}
    for k, v in scope.items():
        if isinstance(v, typing.TypeVar):
            sym[k] = Sym(k, is_dim=True)
        else:
            sym[k] = v
    for p in params:
        sym.setdefault(p, Sym(p))
    sym["self"] = _Self()
    from .jaxtyping import stand_ins

    sym.update(stand_ins(text, glb, scope))  # Float[Array, "b t"] with jaxtyping imported only for type checkers
    out = eval(text, glb, sym)  # noqa: S307
    return evaluate(out, glb, scope, params) if isinstance(out, str) else out


#: Names that only mean an array annotation: expex's dtypes and categories, jaxtyping's.
ANNOTATION_NAMES = frozenset(
    "f16 f32 f64 bf16 i8 i16 i32 i64 u8 u16 u32 u64 c64 c128 Arr Bool Int UInt Float Complex "
    "Inexact Num Key Static Batch Shaped Real Integer Array Float16 Float32 Float64 Int32 Int64 "
    "BFloat16 PRNGKeyArray".split()
)
DTYPE_LIKE = re.compile(r"(bf|[fiuc])(8|16|32|64|128)")


def dropped(text: str, glb: dict[str, Any], scope: dict[str, Any], params: Any = ()) -> set[str]:
    """The undefined names that keep an array annotation from being checked.

    For an annotation that did not evaluate to an array annotation: ``f32[T]`` with
    ``f32`` not imported, ``Float[Array, "b t"]`` with neither jaxtyping nor its
    names, or ``f32[*B]`` with ``B`` not declared. It was meant to be checked, so it
    must not be skipped as an unknown class is (a ``TYPE_CHECKING``-only import, or
    a class defined further down). Empty for those.
    """
    from .types import DType

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return set()
    if isinstance(tree.body, ast.Constant) and isinstance(tree.body.value, str):  # quoted
        return dropped(tree.body.value, glb, scope, params)
    missing = undefined(text, glb, scope, params)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in missing:
            if node.id in ANNOTATION_NAMES or DTYPE_LIKE.fullmatch(node.id):
                out.add(node.id)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            head = scope.get(node.value.id, glb.get(node.value.id))
            if isinstance(head, DType):  # f32[*B]: an array annotation with an unknown size
                out |= {m.id for m in ast.walk(node.slice) if isinstance(m, ast.Name) and m.id in missing}
    return out


def undefined(text: str, glb: dict[str, Any], scope: dict[str, Any], params: Any = ()) -> set[str]:
    """The names ``text`` reads that are defined nowhere it could take them from."""
    import builtins

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return set()
    known = set(glb) | set(scope) | set(params) | {"self"}
    return {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id not in known and not hasattr(builtins, n.id)
    }  # fmt: skip


def import_hint(names: set[str]) -> str:
    """Where the undefined annotation names come from: ``: from expex import f32``."""
    from . import types as t

    ours = sorted(n for n in names if n in t.__all__)
    return f": from expex import {', '.join(ours)}" if ours and len(ours) == len(names) else ""


def unknown_names(text: str, glb: dict[str, Any], scope: dict[str, Any], params: Any = ()) -> tuple[Any, set[str]]:
    """Evaluate with every unknown name as a size; return the value and those names."""
    import builtins

    missing: set[str] = set()

    class Loose(dict):  # type: ignore[type-arg]
        def __missing__(self, name: str) -> Any:
            if name in glb or hasattr(builtins, name):
                raise KeyError(name)
            missing.add(name)
            return Sym(name)

    ns = Loose({k: (Sym(k, is_dim=True) if isinstance(v, typing.TypeVar) else v) for k, v in scope.items()})
    for p in params:
        ns.setdefault(p, Sym(p))
    ns["self"] = _Self()
    out = eval(text, glb, ns)  # noqa: S307
    while isinstance(out, str):  # an annotation written in quotes, read from its text: quoted twice
        out = eval(out, glb, ns)  # noqa: S307
    return out, missing


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Fixed sizes: def f[R = 6], class Board[Rows = 6, Cols = 7]
# --------------------------------------------------------------------------
#
# A type parameter with a default (PEP 696), or with a ``Literal`` bound, is a
# size fixed at that value: ``R = 6``, ``R = Literal[6]`` or ``R: Literal[6]``.
# R then reads as a ``Dim`` class with ``size=6``, so everything that works for
# ``class R(Dim, size=6)`` works for it: plain ``R`` in the annotations, R in
# the body as a size or an axis, the prover treating it as the constant 6.

_FIXED: "weakref.WeakKeyDictionary[Any, type | None]" = weakref.WeakKeyDictionary()


@functools.cache
def fixed(name: str, size: int) -> type:
    """The ``Dim`` class of the type parameter ``name`` fixed at ``size`` (``def f[name = size]``)."""
    from .types import Dim

    def body(ns: dict[str, Any]) -> None:
        ns["__module__"] = "expex"
        ns["__doc__"] = f"The size {name}, fixed at {size} (declared as the type parameter `{name} = {size}`)."

    return types.new_class(name, (Dim,), {"size": size}, body)


def _shown(v: Any) -> str:
    if isinstance(v, type):
        return v.__name__
    return repr(v).replace("typing.", "")


def _literal(v: Any) -> int | None:
    """``6`` for ``Literal[6]``; None for anything else."""
    if typing.get_origin(v) is typing.Literal:
        args = typing.get_args(v)
        if len(args) == 1 and type(args[0]) is int and args[0] >= 0:
            return args[0]
    return None


def fixed_param(tp: Any, owner: str = "") -> type | None:
    """The ``Dim`` class of a type parameter fixed at a size, None for a free size.

    ``R = 6``, ``R = Literal[6]`` and ``R: Literal[6]`` fix R at 6; a type
    parameter with no default and no bound is a free size. Any other default or
    bound is an EX404 ``TypeError``. ``owner`` names the function or class in it.
    """
    try:
        return _FIXED[tp]
    except (KeyError, TypeError):
        pass
    if not isinstance(tp, (typing.TypeVar, typing.TypeVarTuple)):
        return None
    name = tp.__name__
    where = f"{owner}: " if owner else ""
    has_default = getattr(tp, "has_default", lambda: False)()
    if isinstance(tp, typing.TypeVarTuple):
        if has_default:
            raise coded("EX404", TypeError(
                f"{where}the variadic type parameter *{name} has a default, but only a single size "
                f"can be fixed. Drop the default: *{name} takes its sizes from the arguments."
            ))
        return None
    size: int | None = None
    how = ""
    if has_default:
        try:
            default = tp.__default__
        except Exception as e:
            raise coded("EX404", TypeError(
                f"{where}the default of the type parameter {name} cannot be evaluated "
                f"({type(e).__name__}: {e}). Fix it with an int: {name} = 6."
            )) from None
        size = default if type(default) is int and default >= 0 else _literal(default)
        if size is None:
            raise coded("EX404", TypeError(
                f"{where}the type parameter {name} has the default {_shown(default)}, but a default "
                f"fixes a size, so it must be an int (a literal, not negative): write {name} = 6, "
                f"or {name} = Literal[6] for type checkers. A size taken from the arguments has no default."
            ))
        how = "default"
    if tp.__constraints__:
        shown = ", ".join(_shown(c) for c in tp.__constraints__)
        raise coded("EX404", TypeError(
            f"{where}the type parameter {name} is constrained to ({shown}), but a size takes no "
            f"constraints. Write {name} for a size taken from the arguments, or {name} = 6 to fix it."
        ))
    bound = tp.__bound__
    if bound is not None:
        fixed_at = _literal(bound)
        if fixed_at is None:
            raise coded("EX404", TypeError(
                f"{where}the type parameter {name} has the bound {_shown(bound)}, but the only bound "
                f"a size takes is Literal[<int>], which fixes it: {name}: Literal[6]. Write {name} "
                f"for a size taken from the arguments, or {name} = 6 to fix it."
            ))
        if size is not None and size != fixed_at:
            raise coded("EX404", TypeError(
                f"{where}the type parameter {name} is fixed twice, with different values: at "
                f"{fixed_at} by its bound, and at {size} by its {how}. Keep one."
            ))
        size = fixed_at
    out = None if size is None else fixed(name, size)
    try:
        _FIXED[tp] = out
    except TypeError:
        pass
    return out


def fixed_params(obj: Any) -> dict[str, type]:
    """``{"R": <Dim R, size 6>}`` for the type parameters of ``obj`` fixed at a size.

    For ``def f[B, R = 6]`` or ``class Board[Rows = 6]``. Raises EX404 for a
    default or bound that is not a size, naming ``obj``.
    """
    owner = str(getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?"))
    out: dict[str, type] = {}
    for tp in getattr(obj, "__type_params__", ()):
        dim = fixed_param(tp, owner)
        if dim is not None:
            out[tp.__name__] = dim
    return out


def param_head(tp: Any) -> str:
    """A type parameter as written in a signature: ``B``, ``*B``, ``R = 6``."""
    if isinstance(tp, typing.TypeVarTuple):
        return "*" + tp.__name__
    try:
        dim = fixed_param(tp)
    except TypeError:
        dim = None
    return f"{tp.__name__} = {getattr(dim, 'size')}" if dim is not None else tp.__name__


# ``f32[R:6, 7]`` declared a fixed size in the annotation, before fixed sizes
# were type parameters. It is refused with a message that says what to write.


def strings(obj: Any) -> dict[str, str]:
    """An object's own annotations as source text, without evaluating them."""
    try:
        import annotationlib  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        out = annotationlib.get_annotations(obj, format=annotationlib.Format.STRING)
    except Exception:
        return {}
    return {k: v for k, v in out.items() if isinstance(v, str)}


_SLICE = re.compile(r"[\[,]\s*[A-Za-z_]\w*\s*:")


def _declares(obj: Any) -> bool:
    """Might ``obj``'s own annotations use the old ``f32[R:6]``? Cheap: no evaluation."""
    return any(":" in t and _SLICE.search(t) for t in strings(obj).values())


def reject_slices(obj: Any) -> None:
    """Raise EX404 for ``f32[R:6]`` in the annotations of ``obj`` (a function or a class)."""
    if not _declares(obj):
        return
    owner = str(getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?"))
    head = "class " + owner.rsplit(".", 1)[-1] if isinstance(obj, type) else "def " + owner.rsplit(".", 1)[-1]
    for key, text in strings(obj).items():
        if not (":" in text and _SLICE.search(text)):
            continue
        try:
            tree: ast.AST = ast.parse(text, mode="eval").body
            while isinstance(tree, ast.Constant) and isinstance(tree.value, str):  # quoted
                tree = ast.parse(tree.value, mode="eval").body
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            for it in node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]:
                if isinstance(it, ast.Slice) and isinstance(it.lower, ast.Name):
                    name = it.lower.id
                    size = ast.unparse(it.upper) if it.upper is not None else "6"
                    where = f"`-> {text}`" if key == "return" else f"`{key}: {text}`"
                    raise coded("EX404", TypeError(
                        f"{owner}: `{ast.unparse(it)}` in {where} declares a size in the annotation, "
                        f"which expex no longer reads. Declare the fixed size with the type parameters, "
                        f"{head}[{name} = {size}] (or {name} = Literal[{size}] for type checkers), "
                        f"and write plain {name} in the annotations."
                    ))

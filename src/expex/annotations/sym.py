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
# Sizes declared in annotations: f32[R:6, 7]
# --------------------------------------------------------------------------
#
# ``R:6`` in a subscript is a slice, ``slice(R, 6)``: it reads "R, which is 6".
# Anywhere in a function's annotations (or a class's field annotations) it
# declares R as a fixed size of the function (or class). R is then a ``Dim``
# class with ``size=6``, so everything that works for ``class R(Dim, size=6)``
# works for it: plain ``R`` in the other annotations, R in the body as a size
# or an axis, the prover treating it as the constant 6.

_NAMED: "weakref.WeakKeyDictionary[Any, dict[str, type]]" = weakref.WeakKeyDictionary()


@functools.cache
def fixed(name: str, size: int) -> type:
    """The ``Dim`` class of a size declared as ``name:size``."""
    from .types import Dim

    def body(ns: dict[str, Any]) -> None:
        ns["__module__"] = "expex"
        ns["__doc__"] = f"The size {name}, fixed at {size} (declared as `{name}:{size}` in an annotation)."

    return types.new_class(name, (Dim,), {"size": size}, body)


def strings(obj: Any) -> dict[str, str]:
    """An object's own annotations as source text, without evaluating them."""
    try:
        import annotationlib  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        out = annotationlib.get_annotations(obj, format=annotationlib.Format.STRING)
    except Exception:
        return {}
    return {k: v for k, v in out.items() if isinstance(v, str)}


def _declares(obj: Any) -> bool:
    """Do ``obj``'s own annotations declare a size (``R:6``)? Cheap: no evaluation."""
    return any(":" in t and _SLICE.search(t) for t in strings(obj).values())


_SLICE = re.compile(r"[\[,]\s*[A-Za-z_]\w*\s*:")


def _where(key: str, text: str) -> str:
    return f"`-> {text}`" if key == "return" else f"`{key}: {text}`"


def _scan(owner: str, anns: dict[str, str], glb: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """``{name: (size, where)}`` for each ``name:size`` in the annotation texts ``anns``."""
    import builtins

    out: dict[str, tuple[int, str]] = {}
    for key, text in anns.items():
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
            items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            for it in items:
                if not (isinstance(it, ast.Slice) and isinstance(it.lower, ast.Name)):
                    continue
                name, where = it.lower.id, _where(key, text)
                size: Any = None
                if it.upper is not None and it.step is None:
                    try:
                        size = eval(  # noqa: S307 - an annotation, as typing.get_type_hints
                            compile(ast.Expression(it.upper), "<annotation>", "eval"),
                            {"__builtins__": builtins, **glb},
                        )
                    except Exception:
                        size = None
                if type(size) is not int or size < 0:
                    shown = ast.unparse(it)
                    raise coded("EX404", TypeError(
                        f"{owner}: `{shown}` in {where} declares the size {name}, but "
                        + ("its size is missing" if it.upper is None else "a size takes no step" if it.step is not None
                           else "its size is not an int known where it is declared")
                        + f". Write it as {name}:6 (a literal, or a global int)."
                    ))
                if name in out and out[name][0] != size:
                    raise coded("EX404", TypeError(
                        f"{owner}: the size {name} is declared twice, with different values: "
                        f"{name}:{out[name][0]} in {out[name][1]}, and {name}:{size} in {where}."
                    ))
                out.setdefault(name, (size, where))
    return out


def _merge(owner: str, into: dict[str, tuple[int, str]], more: dict[str, tuple[int, str]], whose: str) -> None:
    for name, (size, where) in more.items():
        if name in into and into[name][0] != size:
            raise coded("EX404", TypeError(
                f"{owner}: the size {name} is declared twice, with different values: "
                f"{name}:{into[name][0]} in {into[name][1]}, and {name}:{size} in {where} ({whose})."
            ))
        into.setdefault(name, (size, where))


def named_sizes(obj: Any) -> dict[str, type]:
    """The sizes declared as ``R:6`` in the annotations of ``obj``: ``{"R": <Dim R, size 6>}``.

    For a function, its own annotations and those of the class it is defined
    in (a method sees its class's sizes); for a class, its field annotations
    and its bases'. Raises ``TypeError`` for a name declared with two values,
    or one that is also a type parameter or a parameter.
    """
    try:
        return _NAMED[obj]
    except (KeyError, TypeError):
        pass
    found = _declared(obj)
    out = {name: fixed(name, size) for name, (size, _) in found.items()}
    try:
        _NAMED[obj] = out
    except TypeError:
        pass
    return out


def _declared(obj: Any) -> dict[str, tuple[int, str]]:
    import sys

    owner = str(getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?"))
    if isinstance(obj, type):
        glb = getattr(sys.modules.get(obj.__module__), "__dict__", {})
        found = _scan(owner, strings(obj), glb) if _declares(obj) else {}
        for base in obj.__bases__:
            if base is not object:
                more = {n: (_size(c), f"the class {base.__name__}") for n, c in named_sizes(base).items()}
                _merge(owner, found, more, f"inherited from {base.__name__}")
        params = {p.__name__ for p in getattr(obj, "__type_params__", ())}
        _not_params(owner, found, params, set())
        return found
    if not hasattr(obj, "__code__"):
        return {}
    glb = getattr(obj, "__globals__", {})
    found = _scan(owner, strings(obj), glb) if _declares(obj) else {}
    params = {p.__name__ for p in getattr(obj, "__type_params__", ())}
    code = obj.__code__
    n = code.co_argcount + code.co_kwonlyargcount
    n += bool(code.co_flags & inspect.CO_VARARGS) + bool(code.co_flags & inspect.CO_VARKEYWORDS)
    _not_params(owner, found, params, set(code.co_varnames[:n]))
    cls_sizes, cls_params = _class_of(obj)
    if cls_sizes:
        _merge(owner, found, cls_sizes, "the class's")
    _not_params(owner, found, cls_params, set())
    return found


def _size(dim: type) -> int:
    return int(getattr(dim, "size"))


def _not_params(owner: str, found: dict[str, tuple[int, str]], type_params: set[str], params: set[str]) -> None:
    for name, (size, where) in found.items():
        if name in type_params:
            raise coded("EX404", TypeError(
                f"{owner}: {name}:{size} in {where} declares {name} as a size fixed at {size}, "
                f"but {name} is also a type parameter, a size bound by the arguments. "
                f"Use one or the other: drop {name} from the type parameters, or write {name} "
                f"without `:{size}`."
            ))
        if name in params:
            raise coded("EX404", TypeError(
                f"{owner}: {name}:{size} in {where} declares the size {name}, but {name} is "
                f"also a parameter. Rename one of them."
            ))


# A method's body may use its class's sizes, but the class does not exist yet
# when the method is decorated: they are read from the class's source.
_TREES: dict[str, tuple[int, ast.Module | None]] = {}


def _class_of(fn: Any) -> tuple[dict[str, tuple[int, str]], set[str]]:
    """The sizes declared in the fields of the class ``fn`` is defined in, and its type parameters."""
    qual = getattr(fn, "__qualname__", "").split(".")
    if len(qual) < 2 or qual[-2] == "<locals>":
        return {}, set()
    import linecache

    code = fn.__code__
    lines = linecache.getlines(code.co_filename)
    if not lines:
        return {}, set()
    key = hash(tuple(lines))
    have = _TREES.get(code.co_filename)
    if have is None or have[0] != key:
        try:
            have = (key, ast.parse("".join(lines)))
        except SyntaxError:
            have = (key, None)
        _TREES[code.co_filename] = have
    tree = have[1]
    if tree is None:
        return {}, set()
    best: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ClassDef) and node.name == qual[-2]
            and node.lineno <= code.co_firstlineno <= (node.end_lineno or 0)
            and (best is None or node.lineno > best.lineno)
        ):  # fmt: skip
            best = node
    if best is None:
        return {}, set()
    anns = {
        s.target.id: ast.unparse(s.annotation)
        for s in best.body
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
    }
    glb = getattr(fn, "__globals__", {})
    try:
        found = _scan(best.name, anns, glb)
    except TypeError:
        found = {}  # reported where the class is used
    for b in best.bases:
        base = glb.get(b.id) if isinstance(b, ast.Name) else None
        if isinstance(base, type):
            for n, c in named_sizes(base).items():
                found.setdefault(n, (_size(c), f"the class {base.__name__}"))
    return found, {getattr(t, "name", "") for t in getattr(best, "type_params", ())}

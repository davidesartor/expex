"""Size arithmetic written directly in annotations: ``f32[T + 1]``, ``f32[D // heads]``.

Python cannot evaluate ``T + 1`` when ``T`` is a type parameter. Python 3.14
evaluates annotations lazily, so such an annotation is still available as
text; typax evaluates that text again with every size, parameter and field of
``self`` replaced by a :class:`Sym`, which records the arithmetic as an
expression.
"""

import inspect
import typing
from typing import Any

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

    def __call__(self, *a: Any) -> Any:  # lets typing accept it where it wants a type
        raise TypeError("a size is not callable")

    def __repr__(self) -> str:
        return self.src


class _Self:
    """``self`` in an annotation: its attributes are fields, e.g. ``self.heads``."""

    def __getattr__(self, name: str) -> Sym:
        if name.startswith("__"):
            raise AttributeError(name)
        return Sym(name)


def caller_locals() -> dict[str, Any]:
    """The local variables where a decorator is being applied (outside typax).

    An annotation typax evaluates from its text sees only the module's globals,
    so a class defined inside a function would not be found. This is what lets
    typax find it. Inside a class body, the enclosing function's locals are
    included too.
    """
    import sys

    f = sys._getframe(1)
    while f is not None and f.f_globals.get("__name__", "").split(".")[0] == "typax":
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
    """An object's annotations, as values where Python can evaluate them, else as text."""
    try:
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
    out = eval(text, glb, sym)  # noqa: S307
    return evaluate(out, glb, scope, params) if isinstance(out, str) else out


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
    return eval(text, glb, ns), missing  # noqa: S307

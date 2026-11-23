"""``python -m expex migrate jaxtyping FILE``: jaxtyping annotations rewritten in expex's style.

``Float[Array, "b t d"]`` becomes ``Float[B, T, D]``, the sizes become the
function's type parameters (``def f[B, T, D](...)``), and the function gets
``@expex.fn``. The file is only read: a diff is printed, and ``--write``
applies it. Nothing is imported, so jaxtyping need not be installed and the
module's import-time code does not run.

expex also reads jaxtyping annotations as they are (``annotations/jaxtyping.py``),
so an annotation this cannot rewrite (a broadcastable ``#b``, a ``PyTree[...]``,
an alias defined at module level) is left as it is and still checked; a
function or class that has one is left whole, so that its sizes keep one name.
"""

import ast
import builtins
import keyword
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ..annotations import jaxtyping as jt
from ..annotations.dims import DBroad, DBroadConst, DConst, DEllipsis, DExpr, DName, DVar, DVarBroad, _ANY
from .annotate import EXPEX_NAMES, apply_edits, defined, import_edits, offset, unified_diff, write_back

__all__ = ["migrate", "Migration", "main"]

#: jaxtyping dtype -> expex. Those not here (BFloat16, Real, Float8..., Int4, ...) have no expex name.
DTYPES = {
    "Shaped": "Arr", "Bool": "Bool", "Float": "Float", "Complex": "Complex", "Inexact": "Inexact",
    "Num": "Num", "UInt": "UInt", "Integer": "Int", "Int": "Int", "Key": "Key",
    "Float16": "f16", "Float32": "f32", "Float64": "f64",
    "Int8": "i8", "Int16": "i16", "Int32": "i32", "Int64": "i64",
    "UInt8": "u8", "UInt16": "u16", "UInt32": "u32", "UInt64": "u64",
    "Complex64": "c64", "Complex128": "c128",
}  # fmt: skip
#: jaxtyping names that are whole annotations.
WHOLE = {"PRNGKeyArray": "Key", "Scalar": "Arr", "ScalarLike": "Arr"}
_SCALARS = {"float", "int", "bool", "complex"}
#: jaxtyping's decorators: redundant once the annotations are expex's (and beartype cannot read those).
DECORATORS = {"jaxtyped"}
#: Methods whose annotations are rewritten but that do not get ``@expex.fn``: they build ``self``.
CONSTRUCTORS = {"__init__", "__post_init__", "__check_init__", "__new__"}


class _Keep(Exception):
    """An annotation that stays jaxtyping's: why, and whether expex reads it (so ``@expex.fn`` may go on)."""

    def __init__(self, why: str, readable: bool = True):
        super().__init__(why)
        self.readable = readable


@dataclass
class Migration:
    """The rewrite of one file: ``str(m)`` / :meth:`diff` show it, :meth:`apply` writes it."""

    path: pathlib.Path
    text: str
    new: str
    #: One line per function or class rewritten.
    changes: list[str] = field(default_factory=list)
    #: What was left as jaxtyping (expex still reads it), and why.
    kept: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.new != self.text

    def diff(self) -> str:
        """The rewrite as a unified diff."""
        return unified_diff(self.path, self.text, self.new)

    def apply(self) -> pathlib.Path:
        """Write the rewrite into the file (refused if the file changed since it was read)."""
        write_back(self.path, self.text, self.new, "migrate")
        self.text = self.new
        return self.path

    def __str__(self) -> str:
        notes = [f"# kept as jaxtyping (expex reads it as is): {k}" for k in self.kept]
        return "\n".join([self.diff().rstrip("\n"), *notes]).strip("\n")


def migrate(path: str | pathlib.Path, *, float_as: str | None = None) -> Migration:
    """The rewrite of the jaxtyping annotations in the file ``path``. Nothing is written.

    ``float_as="f32"`` writes ``Float`` as ``f32`` (and ``Int`` / ``Integer`` as
    ``i32``): stricter than jaxtyping's, which accept any float (integer) dtype.
    """
    p = pathlib.Path(path)
    text = p.read_text()
    return _Rewriter(p, text, float_as).run()


# ------------------------------------------------------------------ the rewrite


def _camel(name: str) -> str:
    """``b`` -> ``B``, ``batch`` -> ``Batch``, ``seq_len`` -> ``SeqLen``."""
    parts = [x for x in name.split("_") if x]
    return "".join(x[:1].upper() + x[1:] for x in parts) or name


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.arg for n in ast.walk(node) if isinstance(n, ast.arg)
    }


@dataclass
class _Ann:
    node: ast.expr  # a jaxtyping subscript, or a name such as PRNGKeyArray
    dtype: str | None  # expex's, or None for an array type that is a Python scalar
    dims: tuple[Any, ...]
    whole: str | None = None  # a whole-annotation replacement (PRNGKeyArray -> Key)


class _Rewriter:
    def __init__(self, path: pathlib.Path, text: str, float_as: str | None):
        self.path, self.text = path, text
        self.lines = text.splitlines(keepends=True)
        self.tree = ast.parse(text)
        self.float_as = float_as
        self.edits: list[tuple[int, int, str]] = []
        self.changes: list[str] = []
        self.kept: list[str] = []
        self.taken = defined(self.tree) | set(dir(builtins)) | set(EXPEX_NAMES) | {"Any", "expex"}
        # what the file calls jaxtyping's names: `from jaxtyping import Float as F`, `import jaxtyping as jt`
        self.names: dict[str, str] = {}
        self.modules: set[str] = set()
        self.imports: list[ast.ImportFrom | ast.Import] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module == "jaxtyping":
                self.imports.append(node)
                for a in node.names:
                    self.names[a.asname or a.name] = a.name
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "jaxtyping":
                        self.modules.add(a.asname or a.name)
                        self.imports.append(node)

    # -------------------------------------------------------------- reading annotations

    def ref(self, node: ast.AST) -> str | None:
        """The jaxtyping name ``node`` refers to (``Float`` for ``F`` or ``jt.Float``), or None."""
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.modules:
            return node.attr
        return None

    def anns(self, ann: ast.expr | None) -> Iterator[_Ann]:
        """The jaxtyping annotations inside the annotation ``ann``. Raises :class:`_Keep`."""
        if ann is None:
            return
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            raise _Keep(f"a quoted annotation, {ann.value!r}")
        for node in ast.walk(ann):
            name = self.ref(node.value) if isinstance(node, ast.Subscript) else self.ref(node)
            if isinstance(node, ast.Subscript) and name is not None:
                if name == "PyTree":
                    raise _Keep(f"`{ast.unparse(node)}`: expex has no PyTree[...] annotation")
                if name not in jt.DTYPES:
                    continue
                if name not in DTYPES:
                    raise _Keep(f"`{ast.unparse(node)}`: {name} has no expex name")
                sl = node.slice
                if not (isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and isinstance(sl.elts[1], ast.Constant)
                        and isinstance(sl.elts[1].value, str)):  # fmt: skip
                    raise _Keep(f"`{ast.unparse(node)}`: not an array type and a shape string", readable=False)
                array_type, dim_str = sl.elts[0], sl.elts[1].value
                if isinstance(array_type, ast.Subscript) and self.ref(array_type.value) in jt.DTYPES:
                    raise _Keep(f"`{ast.unparse(node)}`: a jaxtyping type extended with more axes")
                try:
                    dims, _ = jt.parse_shape(dim_str, ast.unparse(node))
                except Exception as e:
                    raise _Keep(f"`{ast.unparse(node)}`: {str(e).splitlines()[0]}", readable=False) from None
                for d in dims:
                    if isinstance(d, (DBroad, DBroadConst, DVarBroad)):
                        raise _Keep(f"`{ast.unparse(node)}`: `{d}` broadcasts, which expex's own annotations do not say")
                scalar = isinstance(array_type, ast.Name) and array_type.id in _SCALARS
                yield _Ann(node, None if scalar else DTYPES[name], dims, array_type.id if scalar else None)  # type: ignore[union-attr]
            elif name in WHOLE and not isinstance(node, ast.Subscript):
                yield _Ann(node, None, (), WHOLE[name])
            elif name == "PyTree":
                raise _Keep(f"`{ast.unparse(node)}`: expex has no PyTree annotation")

    def sizes(self, found: Sequence[_Ann], params: set[str]) -> list[tuple[str, bool]]:
        """The sizes the annotations name, in order: ``(name, variadic)``; int parameters (``{n}``) excluded."""
        out: dict[str, bool] = {}
        for a in found:
            for d in a.dims:
                if isinstance(d, DVar):
                    out.setdefault(d.name, True)
                elif isinstance(d, DName):
                    out.setdefault(d.name, False)
                elif isinstance(d, DExpr):
                    for n in sorted(d.expr.names, key=d.expr.src.find):
                        if n not in params:
                            out.setdefault(n, False)
        return list(out.items())

    def rename(self, sizes: Sequence[tuple[str, bool]], avoid: set[str], given: dict[str, str]) -> dict[str, str]:
        """New names for ``sizes``: ``b`` -> ``B``, ``seq_len`` -> ``SeqLen``; as written, or with a ``_``, if taken."""
        out = dict(given)
        for name, _ in sizes:
            if name in out:
                continue
            used = avoid | set(out.values())
            for cand in (_camel(name), name, _camel(name) + "_", name + "_"):
                if cand.isidentifier() and not keyword.iskeyword(cand) and cand not in used:
                    out[name] = cand
                    break
        return out

    # -------------------------------------------------------------- writing annotations

    def render(self, a: _Ann, names: dict[str, str]) -> str:
        if a.whole is not None and a.dtype is None:
            return f"\x00{a.whole}\x00"
        dtype = a.dtype or "Arr"
        if self.float_as == "f32":
            dtype = {"Float": "f32", "Int": "i32"}.get(dtype, dtype)
        dtype = f"\x00{dtype}\x00"  # expex's name, or expex.Float if jaxtyping's Float stays (see finish)
        if not a.dims:
            return dtype  # Float[Array, ""]: a scalar
        items = []
        for d in a.dims:
            if isinstance(d, DVar):
                items.append("*" + names[d.name])
            elif isinstance(d, DName):
                items.append(names[d.name])
            elif isinstance(d, DConst):
                items.append(str(d.value))
            elif isinstance(d, DEllipsis):
                items.append("...")
            elif d is _ANY:
                items.append("Any")
            elif isinstance(d, DExpr):
                tree = ast.parse(d.expr.src, mode="eval")
                for n in ast.walk(tree):
                    if isinstance(n, ast.Name) and n.id in names:
                        n.id = names[n.id]
                items.append(ast.unparse(tree))
        return f"{dtype}[{', '.join(items)}]"

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            offset(self.lines, node.lineno, node.col_offset),  # type: ignore[attr-defined]
            offset(self.lines, node.end_lineno or node.lineno, node.end_col_offset or 0),  # type: ignore[attr-defined]
        )

    def replace(self, found: Sequence[_Ann], names: dict[str, str]) -> list[tuple[int, int, str]]:
        out = []
        for a in found:
            st, en = self.span(a.node)
            if a.dtype is None and a.whole is not None and isinstance(a.node, ast.Subscript):
                out.append((st, en, a.whole))  # Float[float, ""] -> float, a Python type
            else:
                out.append((st, en, self.render(a, names)))
        return out

    def type_params(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, params: list[str]) -> list[tuple[int, int, str]]:
        """Insert ``[B, T]`` after the name of ``def`` / ``class`` (or extend its type parameters)."""
        if not params:
            return []
        if node.type_params:
            last = node.type_params[-1]
            at = self.span(last)[1]
            return [(at, at, ", " + ", ".join(params))]
        start = offset(self.lines, node.lineno, node.col_offset)
        m = re.compile(rf"\b(def|class)\s+{re.escape(node.name)}\b").search(self.text, start)
        assert m is not None
        return [(m.end(), m.end(), f"[{', '.join(params)}]")]

    def decorate(self, node: ast.FunctionDef | ast.AsyncFunctionDef, drop: bool = True) -> list[tuple[int, int, str]]:
        """``@expex.fn`` directly above the ``def`` (unless it has one); jaxtyping's ``@jaxtyped`` removed if ``drop``."""
        out: list[tuple[int, int, str]] = []
        has = False
        for d in node.decorator_list:
            target = d.func if isinstance(d, ast.Call) else d
            text = ast.unparse(target)
            if text in ("expex.fn", "fn", "expex.jit"):
                has = True
            if drop and (self.ref(target) in DECORATORS or text in DECORATORS):
                st = offset(self.lines, d.lineno, 0)
                en = offset(self.lines, (d.end_lineno or d.lineno) + 1, 0)
                out.append((st, en, ""))
        if not has:
            at = offset(self.lines, node.lineno, 0)
            out.append((at, at, " " * node.col_offset + "@expex.fn\n"))
        return out

    # -------------------------------------------------------------- functions and classes

    def function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, outer: dict[str, str], where: str, keep: bool = False
    ) -> bool:
        """Rewrite one function; with ``keep`` (its class is kept), only decorate it."""
        a = node.args
        args = [*a.posonlyargs, *a.args, *a.kwonlyargs, *[x for x in (a.vararg, a.kwarg) if x]]
        params = {x.arg for x in args}
        locals_ = [
            s for s in ast.walk(node)
            if isinstance(s, ast.AnnAssign) and s is not node and _inside(s, node)
        ]  # fmt: skip
        try:
            found = [f for x in args for f in self.anns(x.annotation)] + list(self.anns(node.returns))
            found_locals = [f for s in locals_ for f in self.anns(s.annotation)]
        except _Keep as e:
            # expex reads jaxtyping as it is: still checked, if expex can read it
            decorated = e.readable and node.name not in CONSTRUCTORS
            if decorated:
                self.edits += self.decorate(node, drop=False)
            if not keep:
                self.kept.append(f"{where}{node.name}: {e}" + ("; decorated with @expex.fn" if decorated else ""))
            return False
        if not found and not found_locals:
            return False
        if keep:
            if node.name not in CONSTRUCTORS:
                self.edits += self.decorate(node, drop=False)
            return False
        sizes = self.sizes(found + found_locals, params)
        own = [(n, v) for n, v in sizes if n not in outer]
        names = self.rename(own, self.taken | _names_in(node), outer)
        self.edits += self.replace(found + found_locals, names)
        tps = [("*" if v else "") + names[n] for n, v in own]
        self.edits += self.type_params(node, tps)
        if node.name not in CONSTRUCTORS:
            self.edits += self.decorate(node)
        shown = ", ".join(f"{n}->{names[n]}" for n, _ in own if names[n] != n)
        self.changes.append(f"{where}{node.name}" + (f"[{', '.join(tps)}]" if tps else "") + (f" (renamed {shown})" if shown else ""))
        return True

    def cls(self, node: ast.ClassDef) -> None:
        fields = [s for s in node.body if isinstance(s, ast.AnnAssign)]
        methods = [s for s in node.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        where = f"{node.name}."

        def kept(why: str) -> None:  # the class and its methods stay jaxtyping's, their sizes having one name
            self.kept.append(f"class {node.name}: {why}; its methods too (decorated with @expex.fn where readable)")
            for m in methods:
                self.function(m, {}, where, keep=True)

        try:
            found = [f for s in fields for f in self.anns(s.annotation)]
        except _Keep as e:
            return kept(str(e))
        if not found:
            for m in methods:
                self.function(m, {}, where)
            return
        sizes = self.sizes(found, set())
        body_names = set().union(*(_names_in(m) for m in methods)) if methods else set()
        names = self.rename(sizes, self.taken | body_names | {node.name}, {})
        saved = (list(self.edits), list(self.changes), len(self.kept))
        for m in methods:
            if not self.function(m, names, where) and len(self.kept) > saved[2]:
                why = self.kept[saved[2]].split(": ", 1)[1].split("; decorated")[0]
                self.edits, self.changes = saved[0], saved[1]
                del self.kept[saved[2] :]
                return kept(f"its method {m.name} has {why}")
        self.edits += self.replace(found, names)
        tps = [("*" if v else "") + names[n] for n, v in sizes]
        self.edits += self.type_params(node, tps)
        shown = ", ".join(f"{n}->{names[n]}" for n, _ in sizes if names[n] != n)
        self.changes.append(f"class {node.name}[{', '.join(tps)}]" + (f" (renamed {shown})" if shown else ""))

    def visit(self, body: list[ast.stmt], where: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                self.cls(node)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function(node, {}, where)
                self.visit(node.body, f"{where}{node.name}.<locals>.")  # nested functions are their own
            elif isinstance(node, (ast.If, ast.Try, ast.With)):
                self.visit(getattr(node, "body", []), where)
                self.visit(getattr(node, "orelse", []), where)
            elif isinstance(node, (ast.AnnAssign, ast.Assign)) and where == "":
                value = node.value if isinstance(node, ast.Assign) else node.annotation
                try:
                    if value is not None and any(True for _ in self.anns(value)):
                        target = ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
                        self.kept.append(f"{target}: an alias at module level, whose sizes no function declares")
                except _Keep:
                    pass

    # -------------------------------------------------------------- imports

    def finish(self) -> None:
        """Settle expex's names and the imports.

        Edits write expex's names between NUL characters, as placeholders: ``Float``
        is expex's unless the file still uses jaxtyping's ``Float`` (in what was
        kept), in which case it is ``expex.Float``. jaxtyping's names that are no
        longer used are dropped from its imports, and its import if none is.
        """
        spans = [(st, en, "") for st, en, _ in self.edits] + [(*self.span(i), "") for i in self.imports]
        rest = apply_edits(self.text, spans)  # the code that stays as it was, imports aside

        def used(name: str, attr: bool = False) -> bool:
            return re.search(rf"(?<![\w.]){re.escape(name)}" + (r"\s*\." if attr else r"\b"), rest) is not None

        dropped = {n for n in self.names if not used(n)}
        have = defined(self.tree) - dropped
        from_expex = {
            a.asname or a.name for n in self.tree.body if isinstance(n, ast.ImportFrom) and n.module == "expex" for a in n.names
        }

        def settle(m: re.Match[str]) -> str:
            name = m.group(1)
            return f"expex.{name}" if name in have and name not in from_expex else name

        self.edits = [(st, en, re.sub(r"\x00(\w+)\x00", settle, t)) for st, en, t in self.edits]
        new_text = " ".join(t for _, _, t in self.edits)
        self.edits += import_edits(self.tree, self.text, new_text, need_expex=True, have=have)
        if re.search(r"(?<![\w.])Any\b", new_text) and "Any" not in have:
            at = self._after_imports()
            self.edits.append((at, at, "from typing import Any\n"))
        for imp in self.imports:
            if isinstance(imp, ast.ImportFrom):
                keep = [a for a in imp.names if a.name == "*" or (a.asname or a.name) not in dropped]
                if len(keep) == len(imp.names):
                    continue
                if keep:
                    names = ", ".join(a.name if a.asname is None else f"{a.name} as {a.asname}" for a in keep)
                    self.edits.append((*self.span(imp), f"from jaxtyping import {names}"))
                else:
                    self.edits.append(self._drop_statement(imp))
            else:
                alias = next(a.asname or a.name for a in imp.names if a.name == "jaxtyping")
                if len(imp.names) == 1 and not used(alias, attr=True):
                    self.edits.append(self._drop_statement(imp))

    def _drop_statement(self, stmt: ast.stmt) -> tuple[int, int, str]:
        """Remove a whole-line statement; ``pass`` if it was the only one in its block."""
        st = offset(self.lines, stmt.lineno, 0)
        en = offset(self.lines, (stmt.end_lineno or stmt.lineno) + 1, 0)
        for parent in ast.walk(self.tree):
            for name in ("body", "orelse"):
                block = getattr(parent, name, None)
                if isinstance(block, list) and block == [stmt] and parent is not self.tree:
                    return (st, en, " " * stmt.col_offset + "pass\n")
        return (st, en, "")

    def _after_imports(self) -> int:
        last = None
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last = node
            elif last is not None or not (isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)):
                break
        return offset(self.lines, (last.end_lineno or last.lineno) + 1, 0) if last is not None else 0

    def run(self) -> Migration:
        if self.names or self.modules:
            self.visit(self.tree.body)
        if self.edits:
            self.finish()
        new = apply_edits(self.text, self.edits)
        return Migration(self.path, self.text, new, self.changes, self.kept)


def _inside(stmt: ast.AST, fn: ast.AST) -> bool:
    """Is ``stmt`` in ``fn``'s own body (not in a nested function or class)?"""
    stack = list(getattr(fn, "body", []))
    while stack:
        n = stack.pop()
        if n is stmt:
            return True
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return False


# ------------------------------------------------------------------ the command line

USAGE = """\
usage: python -m expex migrate jaxtyping PATH [PATH ...] [--write] [--float f32]

  Rewrite jaxtyping annotations (Float[Array, "b t d"]) in expex's style
  (Float[B, T, D], with the sizes as type parameters and @expex.fn on the
  function). Prints a diff; --write applies it. Directories are searched for
  .py files. --float f32 writes Float as f32 and Int as i32 (stricter than
  jaxtyping, which accepts any float or integer dtype).
"""


def _files(paths: Sequence[str]) -> Iterator[pathlib.Path]:
    for p in map(pathlib.Path, paths):
        if p.is_dir():
            yield from sorted(x for x in p.rglob("*.py") if not any(part.startswith(".") for part in x.parts))
        else:
            yield p


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    if not args or args[0] in ("-h", "--help") or args[0] != "jaxtyping":
        print(USAGE, end="", file=sys.stdout if args[:1] in (["-h"], ["--help"]) else sys.stderr)
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    args = args[1:]
    write = "--write" in args
    float_as = None
    if "--float" in args:
        i = args.index("--float")
        float_as = args[i + 1] if i + 1 < len(args) else None
        if float_as not in ("f32", "Float"):
            print("--float takes f32 or Float", file=sys.stderr)
            return 2
        del args[i : i + 2]
    paths = [a for a in args if a != "--write"]
    if not paths or any(a.startswith("-") for a in paths):
        print(USAGE, end="", file=sys.stderr)
        return 2
    n = 0
    for f in _files(paths):
        try:
            m = migrate(f, float_as=float_as)
        except (OSError, SyntaxError, UnicodeDecodeError) as e:
            print(f"{f}: skipped ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        if m.changed:
            n += 1
            if write:
                m.apply()
                print(f"rewrote {f}: " + "; ".join(m.changes))
            else:
                print(m.diff(), end="")
        for k in m.kept:
            print(f"{f}: kept as jaxtyping (expex reads it as is): {k}")
    if write:
        print(f"{n} file{'s' if n != 1 else ''} rewritten")
    return 0

"""What the converter reads off a function's syntax tree, without changing it.

Which names a block assigns and reads (``assigned``, ``reads``,
``read_before_written``), the jumps and returns it contains, the free
names of nested scopes, which statements are worth rewriting, and the sizes a
function uses by name (``size_asserts``, ``axis_names``). ``convert.py``
builds the new tree from these answers.
"""

import ast
import copy
import re
import types
from typing import Any, Iterable

from . import axes

SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


LOOPS = (ast.For, ast.AsyncFor, ast.While)


UNTOUCHED = (ast.Try, ast.TryStar, ast.With, ast.AsyncWith)  # never converted, nor their contents


JUMPS = (ast.Break, ast.Continue, ast.Return)


def walk_scope(node: ast.AST, into_loops: bool = True) -> Iterable[ast.AST]:
    """Walk ``node`` without entering nested function/class scopes."""
    stack = [node]
    first = True
    while stack:
        n = stack.pop()
        yield n
        if not first and isinstance(n, SCOPES):
            continue
        if not first and not into_loops and isinstance(n, LOOPS):
            continue
        first = False
        stack.extend(ast.iter_child_nodes(n))


def walk_all(nodes: Iterable[ast.AST], into_loops: bool = True) -> Iterable[ast.AST]:
    for n in nodes:
        if isinstance(n, SCOPES) or (not into_loops and isinstance(n, LOOPS)):
            yield n
            continue
        yield from walk_scope(n, into_loops)


def assigned(nodes: Iterable[ast.AST]) -> set[str]:
    """Names bound by ``nodes`` in the enclosing function's scope."""
    out: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            return
        if isinstance(n, ast.Lambda) or isinstance(n, COMPS):
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
        for c in ast.iter_child_nodes(n):
            visit(c)

    for n in nodes:
        visit(n)
    return out


def reads(nodes: Iterable[ast.AST]) -> set[str]:
    """Names read anywhere in ``nodes``, including inside nested scopes."""
    out: set[str] = set()
    for n in nodes:
        for m in ast.walk(n):
            if isinstance(m, ast.Name) and not isinstance(m.ctx, ast.Store):
                out.add(m.id)
    return out


def has(nodes: Iterable[ast.AST], kinds: tuple[type, ...], into_loops: bool = True) -> bool:
    return any(isinstance(m, kinds) for m in walk_all(nodes, into_loops))


def has_return(nodes: Iterable[ast.AST]) -> bool:
    return has(nodes, (ast.Return,))


def has_jump(nodes: Iterable[ast.AST]) -> bool:
    """``break`` / ``continue`` that belong to an enclosing loop."""
    return has(nodes, (ast.Break, ast.Continue), into_loops=False)


DECL = (ast.Global, ast.Nonlocal, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await)


def always(stmts: list[ast.stmt], kinds: tuple[type, ...]) -> bool:
    """Does this block always end in one of ``kinds`` (looking through ``if`` / ``else``)?"""
    return any(
        isinstance(s, kinds)
        or (isinstance(s, ast.If) and always(s.body, kinds) and always(s.orelse, kinds))
        for s in stmts
    )


def count_via_ifs(stmts: list[ast.stmt]) -> int:
    """How many ``break`` / ``continue`` / ``return`` are reachable through ``if`` only."""
    n = 0
    for s in stmts:
        if isinstance(s, (ast.Break, ast.Continue, ast.Return)):
            n += 1
        elif isinstance(s, ast.If):
            n += count_via_ifs(s.body) + count_via_ifs(s.orelse)
    return n


def returns_via_ifs(stmts: list[ast.stmt]) -> bool:
    """Is there a ``return`` reachable through ``if`` statements only?"""
    return any(
        isinstance(s, ast.Return)
        or (isinstance(s, ast.If) and (returns_via_ifs(s.body) or returns_via_ifs(s.orelse)))
        for s in stmts
    )


def free_names(scope: ast.AST) -> set[str]:
    """Names a nested function / lambda / generator takes from the enclosing scope."""
    bound: set[str] = set()
    used: set[str] = set()
    declared: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            bound.add(arg.arg)
        # defaults and decorators are evaluated outside, so they count as uses
        outer = list(a.defaults) + [d for d in a.kw_defaults if d is not None]
        outer += list(getattr(scope, "decorator_list", []))
        for default in outer:
            node: ast.AST = default
            used |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        body: list[ast.AST] = list(scope.body) if isinstance(scope.body, list) else [scope.body]
    elif isinstance(scope, ast.ClassDef):
        return {n.id for n in ast.walk(scope) if isinstance(n, ast.Name)}
    else:  # generator expression: targets are its own
        body = [scope]
        for gen in scope.generators:  # type: ignore[attr-defined]
            bound |= set(target_names(gen.target) or ())
    for node in body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                if isinstance(sub.ctx, ast.Store):
                    bound.add(sub.id)
                used.add(sub.id)
            elif isinstance(sub, (ast.Nonlocal, ast.Global)):
                declared |= set(sub.names)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(sub.name)
            elif isinstance(sub, ast.arg):
                # parameters of functions nested deeper: bound there, not free here
                bound.add(sub.arg)
    return (used - bound) | declared


def is_rewritable(m: ast.AST) -> bool:
    """Statements/calls worth converting even without control flow around them."""
    if isinstance(m, ast.Assert):
        return True
    if isinstance(m, (ast.Assign, ast.AugAssign)):
        targets = m.targets if isinstance(m, ast.Assign) else [m.target]
        return any(
            isinstance(t, (ast.Subscript, ast.Attribute)) and isinstance(t.value, ast.Name)
            for t in targets
        )
    return (
        isinstance(m, ast.Call)
        and isinstance(m.func, ast.Name)
        and m.func.id in ("print", "max", "min")
    )


def is_local_annotation(m: ast.AST) -> bool:
    return (
        isinstance(m, ast.AnnAssign)
        and m.value is not None
        and isinstance(m.target, ast.Name)
        and not isinstance(m.annotation, ast.Constant)
    )


MUTATORS = frozenset(
    "append extend insert pop remove clear sort reverse update setdefault popitem add discard "
    "appendleft popleft".split()
)


def mutates(s: ast.stmt) -> bool:
    """Does ``s`` update an object in place (``d[k] = v``, ``o.a = v``, ``xs.sort()``)?"""
    for m in walk_scope(s):
        if isinstance(m, (ast.Subscript, ast.Attribute)) and isinstance(m.ctx, (ast.Store, ast.Del)):
            return True
        if (
            isinstance(m, ast.Expr)
            and isinstance(m.value, ast.Call)
            and isinstance(m.value.func, ast.Attribute)
            and m.value.func.attr in MUTATORS
        ):
            return True
    return False


def target_names(t: ast.AST) -> list[str] | None:
    """Names bound by a simple assignment target, or None if it is not simple."""
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            if isinstance(e, ast.Starred):
                e = e.value
            sub = target_names(e)
            if sub is None:
                return None
            out += sub
        return out
    return None


def read_before_written(stmts: list[ast.stmt], fresh: set[str]) -> tuple[set[str], set[str]]:
    """Names that may be read before being written when ``stmts`` run once.

    ``fresh`` are names known to have been assigned already. Returns the
    read-before-written names and the names definitely assigned afterwards.
    """
    fresh = set(fresh)
    rbw: set[str] = set()

    def read(names: set[str]) -> None:
        rbw.update(names - fresh)

    for s in stmts:
        if isinstance(s, ast.Assign):
            read(reads([s.value]))
            stores: set[str] = set()
            for t in s.targets:
                tn = target_names(t)
                if tn is None:
                    read(reads([t]))
                else:
                    stores |= set(tn)
            fresh |= stores
        elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            if s.value is not None:
                read(reads([s.value]))
                fresh.add(s.target.id)
        elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            read(reads([s.value]) | {s.target.id})
            fresh.add(s.target.id)
        elif isinstance(s, ast.If):
            read(reads([s.test]))
            r1, f1 = read_before_written(s.body, fresh)
            r2, f2 = read_before_written(s.orelse, fresh)
            rbw |= r1 | r2
            fresh = f1 & f2
        elif isinstance(s, ast.For) and target_names(s.target) is not None:
            read(reads([s.iter]))
            r1, _ = read_before_written(s.body, fresh | set(target_names(s.target) or ()))
            r2, _ = read_before_written(s.orelse, fresh)
            rbw |= r1 | r2  # the loop may run zero times: nothing new is definite
        elif isinstance(s, ast.While):
            read(reads([s.test]))
            r1, _ = read_before_written(s.body, fresh)
            r2, _ = read_before_written(s.orelse, fresh)
            rbw |= r1 | r2
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            read(reads([s]))
            fresh.add(s.name)
        else:
            read(reads([s]))
    return rbw, fresh


INTERESTING = (ast.If, ast.While, ast.For, ast.IfExp, ast.BoolOp)


MANGLED = re.compile(r"(?<![\w.])__[A-Za-z0-9]\w*(?<!__)\b|\.__[A-Za-z0-9]\w*(?<!__)\b")


def einsum_spellings(raw: Any) -> tuple[dict[str, str], set[str]]:
    """How ``raw`` reaches ``expex.einsum`` / ``expex.rearrange``: bare names, and modules."""
    from .axes import einsum, rearrange

    glb = getattr(raw, "__globals__", {})
    names = {k: v.__name__ for k, v in glb.items() if v is einsum or v is rearrange}
    modules = {
        k for k, v in glb.items()
        if isinstance(v, types.ModuleType) and getattr(v, "einsum", None) is einsum
    }  # fmt: skip
    return names, modules


def axis_names(raw: Any, fdef: ast.FunctionDef) -> set[str]:
    """Names that stand for axes in ``raw``: type parameters and ``Dim`` classes."""
    out = {tp.__name__ for tp in getattr(raw, "__type_params__", ())}
    code = raw.__code__
    for name, cell in zip(code.co_freevars, raw.__closure__ or ()):
        try:
            if axes.axis_name(cell.cell_contents) is not None:  # e.g. a class's type parameter
                out.add(name)
        except ValueError:  # an empty cell
            pass
    for name in code.co_names:
        if axes.axis_name(raw.__globals__.get(name)) is not None:
            out.add(name)
    return out


def signature_names(fdef: ast.FunctionDef) -> set[str]:
    """Names used in the signature's annotations, and the type parameters."""
    a = fdef.args
    anns = [x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg] if x]
    out = {n.id for ann in [*anns, fdef.returns] if ann is not None for n in ast.walk(ann) if isinstance(n, ast.Name)}
    out |= {getattr(t, "name", "") for t in getattr(fdef, "type_params", ())}
    for ann in [*anns, fdef.returns]:  # names inside a quoted annotation
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            try:
                out |= {n.id for n in ast.walk(ast.parse(ann.value, mode="eval")) if isinstance(n, ast.Name)}
            except SyntaxError:
                pass
    return out


def size_asserts(
    fdef: ast.FunctionDef,
    sizes: set[str],
    defs: dict[str, str] | None = None,
    def_stmts: list[ast.Assign] | None = None,
) -> list[Any]:
    """The leading ``assert``s and size definitions of the body.

    ``assert D % heads == 0`` becomes the constraint ``D % heads == 0``: checked
    before the body runs, and assumed by the prover. ``H, K = heads, D // heads``,
    for names used in the signature, defines those sizes (recorded in ``defs``).
    Both may only involve sizes, int parameters and fields of ``self``.
    """
    a = fdef.args
    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    in_signature = signature_names(fdef)
    allowed = sizes | params

    def plain(e: ast.AST) -> bool:
        if isinstance(e, ast.Attribute):
            return isinstance(e.value, ast.Name) and e.value.id == "self"
        if isinstance(e, ast.Name):
            return e.id in allowed and e.id != "self"
        if isinstance(e, ast.Constant):
            return type(e.value) is int
        if isinstance(e, ast.BinOp):
            return plain(e.left) and plain(e.right)
        if isinstance(e, ast.UnaryOp):
            return plain(e.operand)
        if isinstance(e, ast.Compare):
            return plain(e.left) and all(plain(c) for c in e.comparators)
        return False

    class Unself(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> Any:
            return ast.Name(id=node.attr, ctx=ast.Load())

    out: list[tuple[ast.Assert, list[str]]] = []
    body = fdef.body
    start = 1 if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) else 0
    for s in body[start:]:
        if isinstance(s, ast.Assign) and defs is not None and len(s.targets) == 1:
            target, value = s.targets[0], s.value
            names = [target] if isinstance(target, ast.Name) else getattr(target, "elts", None)
            values = [value] if isinstance(target, ast.Name) else getattr(value, "elts", None)
            if (
                names and values and len(names) == len(values)
                and all(isinstance(n, ast.Name) and n.id in in_signature for n in names)
                and all(plain(v) for v in values)
            ):  # fmt: skip
                for n, v in zip(names, values):
                    defs[n.id] = ast.unparse(Unself().visit(copy.deepcopy(v)))
                    allowed.add(n.id)
                if def_stmts is not None:
                    def_stmts.append(s)
                continue
            break
        if not isinstance(s, ast.Assert):
            break
        tests = s.test.values if isinstance(s.test, ast.BoolOp) and isinstance(s.test.op, ast.And) else [s.test]
        if not all(isinstance(t, ast.Compare) and plain(t) for t in tests):
            break
        out.append((s, [ast.unparse(Unself().visit(copy.deepcopy(t))) for t in tests]))
    return out


def named_axes_call(node: ast.Call, names: set[str] | frozenset[str] | None) -> bool:
    """Does the call pass an axis by name? (``names=None``: any bare name might be one.)"""
    if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
        return False

    def is_axis(e: ast.expr) -> bool:
        if isinstance(e, ast.Name):
            return names is None or e.id in names
        if isinstance(e, ast.Tuple) and e.elts:
            return all(isinstance(x, ast.Constant) or is_axis(x) for x in e.elts) and any(
                isinstance(x, ast.Name) for x in e.elts
            )
        return False

    if any(k.arg in axes.AXIS_KEYWORDS and is_axis(k.value) for k in node.keywords):
        return True
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in axes.AXIS_METHODS
        and any(is_axis(a) for a in node.args)
    )

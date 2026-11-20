"""Source conversion: plain ``if`` / ``while`` / ``for`` into staged control flow.

``convert(f)`` re-compiles a function so that its control-flow statements call
the helpers in ``_flow.py``. The converted function behaves exactly like the
original when conditions and iterables are ordinary Python values, and uses
``lax.cond`` / ``lax.while_loop`` / ``lax.scan`` / ``vmap`` when they are
traced.

A statement becomes::

    def __bn_true_1__(x, y):        # one parameter per variable the statement assigns
        ...original body...
        return (x, y)
    def __bn_false_1__(x, y):
        ...original else...
        return (x, y)
    (x, y) = __barnacle_flow__.if_(test, __bn_true_1__, __bn_false_1__, (x, y), ...)

What is left alone (and keeps plain Python semantics): loops containing
``break`` / ``continue`` / ``return``, loops with an ``else`` clause,
statements using ``global`` / ``nonlocal`` / ``del`` / ``yield``, nested
function definitions, and generators / async functions.
"""

from __future__ import annotations

import ast
import copy
import inspect
import re
import textwrap
import types
from typing import Any, Callable, Iterable

from . import _flow

FLOW = "__barnacle_flow__"
RET = "__bn_ret__"

_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)


class NotConverted(Exception):
    """Conversion was skipped; the message says why."""


# ------------------------------------------------------------------ analysis


def _walk_scope(node: ast.AST, into_loops: bool = True) -> Iterable[ast.AST]:
    """Walk ``node`` without entering nested function/class scopes."""
    stack = [node]
    first = True
    while stack:
        n = stack.pop()
        yield n
        if not first and isinstance(n, _SCOPES):
            continue
        if not first and not into_loops and isinstance(n, _LOOPS):
            continue
        first = False
        stack.extend(ast.iter_child_nodes(n))


def _walk_all(nodes: Iterable[ast.AST], into_loops: bool = True) -> Iterable[ast.AST]:
    for n in nodes:
        if isinstance(n, _SCOPES) or (not into_loops and isinstance(n, _LOOPS)):
            yield n
            continue
        yield from _walk_scope(n, into_loops)


def _assigned(nodes: Iterable[ast.AST]) -> set[str]:
    """Names bound by ``nodes`` in the enclosing function's scope."""
    out: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            return
        if isinstance(n, ast.Lambda) or isinstance(n, _COMPS):
            return
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        for c in ast.iter_child_nodes(n):
            visit(c)

    for n in nodes:
        visit(n)
    return out


def _reads(nodes: Iterable[ast.AST]) -> set[str]:
    """Names read anywhere in ``nodes``, including inside nested scopes."""
    out: set[str] = set()
    for n in nodes:
        for m in ast.walk(n):
            if isinstance(m, ast.Name) and not isinstance(m.ctx, ast.Store):
                out.add(m.id)
    return out


def _has(nodes: Iterable[ast.AST], kinds: tuple[type, ...], into_loops: bool = True) -> bool:
    return any(isinstance(m, kinds) for m in _walk_all(nodes, into_loops))


def _has_return(nodes: Iterable[ast.AST]) -> bool:
    return _has(nodes, (ast.Return,))


def _has_jump(nodes: Iterable[ast.AST]) -> bool:
    """``break`` / ``continue`` that belong to an enclosing loop."""
    return _has(nodes, (ast.Break, ast.Continue), into_loops=False)


_DECL = (ast.Global, ast.Nonlocal, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await)


def _returns_via_ifs(stmts: list[ast.stmt]) -> bool:
    """Is there a ``return`` reachable through ``if`` statements only?"""
    return any(
        isinstance(s, ast.Return)
        or (isinstance(s, ast.If) and (_returns_via_ifs(s.body) or _returns_via_ifs(s.orelse)))
        for s in stmts
    )


def _target_names(t: ast.AST) -> list[str] | None:
    """Names bound by a simple assignment target, or None if it is not simple."""
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            if isinstance(e, ast.Starred):
                e = e.value
            sub = _target_names(e)
            if sub is None:
                return None
            out += sub
        return out
    return None


def _rbw(stmts: list[ast.stmt], fresh: set[str]) -> tuple[set[str], set[str]]:
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
            read(_reads([s.value]))
            stores: set[str] = set()
            for t in s.targets:
                tn = _target_names(t)
                if tn is None:
                    read(_reads([t]))
                else:
                    stores |= set(tn)
            fresh |= stores
        elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            if s.value is not None:
                read(_reads([s.value]))
                fresh.add(s.target.id)
        elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            read(_reads([s.value]) | {s.target.id})
            fresh.add(s.target.id)
        elif isinstance(s, ast.If):
            read(_reads([s.test]))
            r1, f1 = _rbw(s.body, fresh)
            r2, f2 = _rbw(s.orelse, fresh)
            rbw |= r1 | r2
            fresh = f1 & f2
        elif isinstance(s, ast.For) and _target_names(s.target) is not None:
            read(_reads([s.iter]))
            r1, _ = _rbw(s.body, fresh | set(_target_names(s.target) or ()))
            r2, _ = _rbw(s.orelse, fresh)
            rbw |= r1 | r2  # the loop may run zero times: nothing new is definite
        elif isinstance(s, ast.While):
            read(_reads([s.test]))
            r1, _ = _rbw(s.body, fresh)
            r2, _ = _rbw(s.orelse, fresh)
            rbw |= r1 | r2
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            read(_reads([s]))
            fresh.add(s.name)
        else:
            read(_reads([s]))
    return rbw, fresh


# ---------------------------------------------------------------- generation


def _name(n: str, store: bool = False) -> ast.Name:
    return ast.Name(id=n, ctx=ast.Store() if store else ast.Load())


def _flow_attr(attr: str) -> ast.Attribute:
    return ast.Attribute(value=_name(FLOW), attr=attr, ctx=ast.Load())


def _call(attr: str, *args: ast.expr) -> ast.Call:
    return ast.Call(func=_flow_attr(attr), args=list(args), keywords=[])


def _thunk(e: ast.expr) -> ast.Lambda:
    return ast.Lambda(
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=e,
    )


def _tuple(names: list[str], store: bool = False) -> ast.Tuple:
    return ast.Tuple(elts=[_name(n, store) for n in names], ctx=ast.Store() if store else ast.Load())


def _strs(names: Iterable[str]) -> ast.Tuple:
    return ast.Tuple(elts=[ast.Constant(n) for n in names], ctx=ast.Load())


def _state(names: list[str]) -> ast.Tuple:
    return ast.Tuple(elts=[_call("ld", _thunk(_name(n))) for n in names], ctx=ast.Load())


def _def(name: str, params: list[str], body: list[ast.stmt], ret: ast.expr) -> ast.FunctionDef:
    f = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=p) for p in params],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[*body, ast.Return(value=ret)],
        decorator_list=[],
        type_params=[],
    )
    return f


def _bind(names: list[str], call: ast.expr) -> ast.stmt:
    if not names:
        return ast.Expr(value=call)
    return ast.Assign(targets=[_tuple(names, store=True)], value=call)


class _Exprs(ast.NodeTransformer):
    """``and`` / ``or`` / ``not`` / ``a if c else b`` -> staged helpers."""

    def __init__(self, site: Callable[[ast.AST], ast.expr]):
        self.site = site

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return node  # nested functions are left alone

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]
    visit_ClassDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.generic_visit(node)
        attr = "and_" if isinstance(node.op, ast.And) else "or_"
        acc = node.values[0]
        for v in node.values[1:]:
            acc = ast.copy_location(_call(attr, _thunk(acc), _thunk(v)), node)
        return acc

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.copy_location(_call("not_", node.operand), node)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.generic_visit(node)
        return ast.copy_location(
            _call("ifexp", node.test, _thunk(node.body), _thunk(node.orelse), self.site(node)), node
        )


class _Converter:
    def __init__(self, fdef: ast.FunctionDef, filename: str, shadowed: set[str]):
        self.fdef = fdef
        self.filename = filename
        self.n = 0
        self.count = 0
        args = fdef.args
        params = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        if args.vararg:
            params.append(args.vararg.arg)
        if args.kwarg:
            params.append(args.kwarg.arg)
        declared: set[str] = set()
        for m in _walk_all(fdef.body):
            if isinstance(m, (ast.Global, ast.Nonlocal)):
                declared |= set(m.names)
        self.locals = (set(params) | _assigned(fdef.body)) - declared
        self.builtins_ok = {
            b for b in ("range", "enumerate", "zip") if b not in self.locals and b not in shadowed
        }

    def site(self, node: ast.AST) -> ast.expr:
        return ast.Constant(f"{self.filename}:{getattr(node, 'lineno', 0)}")

    def fresh(self, kind: str) -> str:
        self.n += 1
        return f"__bn_{kind}_{self.n}__"

    # -- pre-passes

    def rewrite_appends(self, stmts: list[ast.stmt]) -> None:
        """``xs.append(v)`` -> ``xs = append_(xs, v)`` for local lists (lets loops collect)."""
        holder = ast.Module(body=stmts, type_ignores=[])
        for m in [holder, *_walk_all(stmts)]:
            if isinstance(m, _SCOPES):
                continue
            for field in ("body", "orelse", "finalbody"):
                seq = getattr(m, field, None)
                if not isinstance(seq, list):
                    continue
                for i, s in enumerate(seq):
                    if (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Call)
                        and isinstance(s.value.func, ast.Attribute)
                        and s.value.func.attr == "append"
                        and isinstance(s.value.func.value, ast.Name)
                        and s.value.func.value.id in self.locals
                        and len(s.value.args) == 1
                        and not s.value.keywords
                    ):
                        n = s.value.func.value.id
                        seq[i] = ast.copy_location(
                            ast.Assign(
                                targets=[_name(n, store=True)],
                                value=_call("append_", _name(n), s.value.args[0]),
                            ),
                            s,
                        )

    def lower_returns(self, stmts: list[ast.stmt]) -> tuple[list[ast.stmt], bool]:
        """Rewrite early returns inside ``if`` so that the function has a single exit."""
        out: list[ast.stmt] = []
        for i, s in enumerate(stmts):
            if isinstance(s, ast.Return):
                out.append(
                    ast.copy_location(
                        ast.Assign(
                            targets=[_name(RET, store=True)], value=s.value or ast.Constant(None)
                        ),
                        s,
                    )
                )
                return out, True
            if isinstance(s, ast.If) and (_returns_via_ifs(s.body) or _returns_via_ifs(s.orelse)):
                rest = stmts[i + 1 :]
                body, rb = self.lower_returns(s.body + copy.deepcopy(rest))
                orelse, ro = self.lower_returns(s.orelse + rest)
                new = ast.copy_location(ast.If(test=s.test, body=body, orelse=orelse), s)
                out.append(new)
                return out, rb and ro
            out.append(s)
        return out, False

    # -- statements

    def block(self, stmts: list[ast.stmt], live_out: set[str]) -> list[ast.stmt]:
        suffix: list[set[str]] = [set()]
        for s in reversed(stmts):
            suffix.append(suffix[-1] | _reads([s]))
        suffix.reverse()
        out: list[ast.stmt] = []
        for i, s in enumerate(stmts):
            out.extend(self.stmt(s, live_out | suffix[i + 1]))
        return out

    def stmt(self, s: ast.stmt, live: set[str]) -> list[ast.stmt]:
        if isinstance(s, ast.If):
            return self.if_(s, live)
        if isinstance(s, ast.While):
            return self.while_(s, live)
        if isinstance(s, ast.For):
            return self.for_(s, live)
        if isinstance(s, _SCOPES):
            return [s]
        inner = live | _reads([s])
        for field in ("body", "orelse", "finalbody"):
            seq = getattr(s, field, None)
            if isinstance(seq, list) and seq and isinstance(seq[0], ast.stmt):
                setattr(s, field, self.block(seq, inner))
        for h in getattr(s, "handlers", []) or []:
            h.body = self.block(h.body, inner)
        for c in getattr(s, "cases", []) or []:
            c.body = self.block(c.body, inner)
        return [s]

    def iterable(self, it: ast.expr) -> ast.expr:
        """``range`` / ``enumerate`` / ``zip`` calls -> their staged versions (nested too)."""
        if (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id in self.builtins_ok
        ):
            args = it.args if it.func.id == "range" else [self.iterable(a) for a in it.args]
            return ast.copy_location(
                ast.Call(func=_flow_attr(it.func.id + "_"), args=args, keywords=it.keywords), it
            )
        return it

    def _blocked(self, s: ast.stmt) -> bool:
        return _has([s], _DECL + (ast.NamedExpr,)) or _has_return([s])

    def if_(self, s: ast.If, live: set[str]) -> list[ast.stmt]:
        names = sorted((_assigned(s.body) | _assigned(s.orelse)) & self.locals)
        ok = not self._blocked(s) and not _has_jump(s.body + s.orelse)
        body = self.block(s.body, live)
        orelse = self.block(s.orelse, live)
        if not ok:
            s.body, s.orelse = body, orelse
            return [s]
        self.count += 1
        t, f = self.fresh("true"), self.fresh("false")
        ret = _tuple(names)
        call = _call("if_", s.test, _name(t), _name(f), _state(names), _strs(names), self.site(s))
        new = [_def(t, names, body, ret), _def(f, names, orelse, copy.deepcopy(ret)), _bind(names, call)]
        return [ast.copy_location(n, s) for n in new]

    def while_(self, s: ast.While, live: set[str]) -> list[ast.stmt]:
        names = sorted(_assigned(s.body) & self.locals)
        ok = not s.orelse and not self._blocked(s) and not _has_jump(s.body)
        back = _rbw(s.body, set())[0] | _reads([s.test])
        body = self.block(s.body, live | back)
        if not ok:
            s.body = body
            return [s]
        self.count += 1
        t, b = self.fresh("test"), self.fresh("body")
        call = _call(
            "while_", _name(t), _name(b), _state(names), _strs(names),
            _strs(sorted(set(names) & live)), self.site(s),
        )  # fmt: skip
        new = [_def(t, names, [], s.test), _def(b, names, body, _tuple(names)), _bind(names, call)]
        return [ast.copy_location(n, s) for n in new]

    def for_(self, s: ast.For, live: set[str]) -> list[ast.stmt]:
        targets = _target_names(s.target)
        ok = (
            targets is not None
            and not s.orelse
            and not self._blocked(s)
            and not _has_jump(s.body)
            and set(targets) <= self.locals
        )
        rbw, definite = _rbw(s.body, set(targets or ()))
        if not ok:
            s.body = self.block(s.body, live | rbw)
            return [s]
        assert targets is not None
        nameset = (_assigned(s.body) | set(targets)) & self.locals
        names = sorted(nameset)
        # Carried from one iteration to the next: read before being written, or
        # only conditionally written and needed after the loop.
        carried = (rbw | ((nameset - definite) & live)) & nameset
        body = self.block(s.body, live | rbw)
        self.count += 1
        b, item = self.fresh("body"), "__bn_item__"
        unpack = ast.Assign(targets=[s.target], value=_name(item))
        it = self.iterable(s.iter)
        call = _call(
            "for_", it, _name(b), _state(names), _strs(names),
            _strs(sorted(carried)), _strs(sorted(nameset & live)), self.site(s),
        )  # fmt: skip
        new = [_def(b, [item, *names], [unpack, *body], _tuple(names)), _bind(names, call)]
        return [ast.copy_location(n, s) for n in new]


# ----------------------------------------------------------------- top level

_INTERESTING = (ast.If, ast.While, ast.For, ast.IfExp, ast.BoolOp)
_MANGLED = re.compile(r"(?<![\w.])__[A-Za-z0-9]\w*(?<!__)\b|\.__[A-Za-z0-9]\w*(?<!__)\b")


def convert(raw: Callable[..., Any]) -> Callable[..., Any]:
    """Return a control-flow-converted copy of ``raw``, or raise :class:`NotConverted`."""
    if not isinstance(raw, types.FunctionType):
        raise NotConverted("not a plain Python function")
    code = raw.__code__
    if code.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR):
        raise NotConverted("generators and async functions are not converted")
    try:
        src = textwrap.dedent(inspect.getsource(raw))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError) as e:
        raise NotConverted(f"source not available ({type(e).__name__})") from None
    fdef = tree.body[0] if tree.body else None
    if not isinstance(fdef, ast.FunctionDef) or fdef.name != raw.__name__:
        raise NotConverted("could not locate the function definition in its source")
    if not any(
        isinstance(m, _INTERESTING) or (isinstance(m, ast.UnaryOp) and isinstance(m.op, ast.Not))
        for m in _walk_all(fdef.body)
    ):
        raise NotConverted("no control flow to convert")
    if "." in raw.__qualname__.replace(".<locals>.", "") and _MANGLED.search(src):
        raise NotConverted("uses private (name-mangled) attributes")

    ast.increment_lineno(tree, code.co_firstlineno - 1)
    conv = _Converter(fdef, code.co_filename, set(raw.__globals__))
    conv.rewrite_appends(fdef.body)
    exprs = _Exprs(conv.site)
    fdef.body = [exprs.visit(s) for s in fdef.body]
    if any(
        isinstance(s, ast.If) and (_returns_via_ifs(s.body) or _returns_via_ifs(s.orelse))
        for s in fdef.body
    ):
        body, always = conv.lower_returns(fdef.body)
        if not always:
            body.insert(0, ast.Assign(targets=[_name(RET, store=True)], value=ast.Constant(None)))
        body.append(ast.Return(value=_name(RET)))
        fdef.body = body
        conv.locals.add(RET)
    fdef.body = conv.block(fdef.body, set())

    # Re-create the function: same globals, defaults and closure cells, new code.
    fdef.decorator_list = []
    fdef.returns = None
    fdef.type_params = []
    a = fdef.args
    for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
        arg.annotation = None
    a.defaults = [ast.Constant(None) for _ in a.defaults]
    a.kw_defaults = [None if d is None else ast.Constant(None) for d in a.kw_defaults]
    cells = dict(zip(code.co_freevars, raw.__closure__ or ()))
    pre: list[ast.stmt] = [
        ast.Assign(targets=[_name(v, store=True)], value=ast.Constant(None)) for v in cells
    ]
    factory = ast.FunctionDef(
        name="__bn_factory__",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[*pre, fdef, ast.Return(value=_name(fdef.name))],
        decorator_list=[],
        type_params=[],
    )
    module = ast.Module(body=[factory], type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        compiled = compile(module, code.co_filename, "exec")
    except (SyntaxError, ValueError) as e:
        raise NotConverted(f"generated code did not compile ({e})") from None
    fcode = next(c for c in compiled.co_consts if isinstance(c, types.CodeType))
    inner = next(
        c for c in fcode.co_consts if isinstance(c, types.CodeType) and c.co_name == fdef.name
    )
    try:
        closure = tuple(cells[v] for v in inner.co_freevars)
    except KeyError as e:
        raise NotConverted(f"free variable {e} could not be resolved") from None
    raw.__globals__.setdefault(FLOW, _flow)
    new = types.FunctionType(inner, raw.__globals__, raw.__name__, raw.__defaults__, closure or None)
    new.__kwdefaults__ = raw.__kwdefaults__
    new.__qualname__ = raw.__qualname__
    new.__module__ = raw.__module__
    new.__doc__ = raw.__doc__
    new.__dict__.update(raw.__dict__)
    new.__barnacle_converted__ = conv.count  # type: ignore[attr-defined]
    return new


def source(raw: Callable[..., Any]) -> str:
    """The converted source of ``raw`` (for inspection)."""
    src = textwrap.dedent(inspect.getsource(raw))
    tree = ast.parse(src)
    fdef = tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)
    ast.increment_lineno(tree, raw.__code__.co_firstlineno - 1)
    conv = _Converter(fdef, raw.__code__.co_filename, set(getattr(raw, "__globals__", {})))
    conv.rewrite_appends(fdef.body)
    fdef.body = [_Exprs(conv.site).visit(s) for s in fdef.body]
    if any(
        isinstance(s, ast.If) and (_returns_via_ifs(s.body) or _returns_via_ifs(s.orelse))
        for s in fdef.body
    ):
        body, always = conv.lower_returns(fdef.body)
        if not always:
            body.insert(0, ast.Assign(targets=[_name(RET, store=True)], value=ast.Constant(None)))
        body.append(ast.Return(value=_name(RET)))
        fdef.body = body
        conv.locals.add(RET)
    fdef.body = conv.block(fdef.body, set())
    fdef.decorator_list = []
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

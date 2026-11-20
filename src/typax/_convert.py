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
    (x, y) = __typax_flow__.if_(test, __bn_true_1__, __bn_false_1__, (x, y), ...)

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

FLOW = "__typax_flow__"
RET = "__bn_ret__"
RETF = "__bn_returned__"

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
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
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


def _count_via_ifs(stmts: list[ast.stmt]) -> int:
    """How many ``break`` / ``continue`` / ``return`` are reachable through ``if`` only."""
    n = 0
    for s in stmts:
        if isinstance(s, (ast.Break, ast.Continue, ast.Return)):
            n += 1
        elif isinstance(s, ast.If):
            n += _count_via_ifs(s.body) + _count_via_ifs(s.orelse)
    return n


def _returns_via_ifs(stmts: list[ast.stmt]) -> bool:
    """Is there a ``return`` reachable through ``if`` statements only?"""
    return any(
        isinstance(s, ast.Return)
        or (isinstance(s, ast.If) and (_returns_via_ifs(s.body) or _returns_via_ifs(s.orelse)))
        for s in stmts
    )


def _free_names(scope: ast.AST) -> set[str]:
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
            bound |= set(_target_names(gen.target) or ())
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


def _index_expr(node: ast.expr) -> ast.expr:
    """A subscript's index as an ordinary expression (``1:3`` -> ``slice(1, 3)``)."""
    if isinstance(node, ast.Slice):
        none = ast.Constant(None)
        return ast.Call(
            func=_name("slice"),
            args=[node.lower or none, node.upper or none, node.step or none],
            keywords=[],
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple(elts=[_index_expr(e) for e in node.elts], ctx=ast.Load())
    return node


def _is_rewritable(m: ast.AST) -> bool:
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


def _is_local_annotation(m: ast.AST) -> bool:
    return (
        isinstance(m, ast.AnnAssign)
        and m.value is not None
        and isinstance(m.target, ast.Name)
        and not isinstance(m.annotation, ast.Constant)
    )


_MUTATORS = frozenset(
    "append extend insert pop remove clear sort reverse update setdefault popitem add discard "
    "appendleft popleft".split()
)


def _mutates(s: ast.stmt) -> bool:
    """Does ``s`` update an object in place (``d[k] = v``, ``o.a = v``, ``xs.sort()``)?"""
    for m in _walk_scope(s):
        if isinstance(m, (ast.Subscript, ast.Attribute)) and isinstance(m.ctx, (ast.Store, ast.Del)):
            return True
        if (
            isinstance(m, ast.Expr)
            and isinstance(m.value, ast.Call)
            and isinstance(m.value.func, ast.Attribute)
            and m.value.func.attr in _MUTATORS
        ):
            return True
    return False


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


def _thunk(e: ast.expr, params: Iterable[str] = ()) -> ast.Lambda:
    return ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=p) for p in params],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=e,
    )


def _tuple(names: list[str], store: bool = False) -> ast.Tuple:
    return ast.Tuple(elts=[_name(n, store) for n in names], ctx=ast.Store() if store else ast.Load())


def _strs(names: Iterable[str]) -> ast.Tuple:
    return ast.Tuple(elts=[ast.Constant(n) for n in names], ctx=ast.Load())


def _state(names: list[str]) -> ast.Tuple:
    return ast.Tuple(elts=[_call("ld", _thunk(_name(n))) for n in names], ctx=ast.Load())


def _unbind(names: list[str]) -> list[ast.stmt]:
    """``if x is UNDEF: del x`` for each name: undefined variables stay unbound."""
    return [
        ast.If(
            test=ast.Compare(left=_name(n), ops=[ast.Is()], comparators=[_flow_attr("UNDEF")]),
            body=[ast.Delete(targets=[ast.Name(id=n, ctx=ast.Del())])],
            orelse=[],
        )
        for n in names
    ]


def _def(
    name: str, params: list[str], body: list[ast.stmt], ret: ast.expr, state: list[str] | None = None
) -> ast.FunctionDef:
    body = [*_unbind(params if state is None else state), *body]
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


def _bind(names: list[str], call: ast.expr) -> list[ast.stmt]:
    """Assign the helper's result back, leaving undefined variables unbound."""
    if not names:
        return [ast.Expr(value=call)]
    return [ast.Assign(targets=[_tuple(names, store=True)], value=call), *_unbind(names)]


def _has_walrus(node: ast.AST) -> bool:
    return any(isinstance(m, ast.NamedExpr) for m in ast.walk(node))


def _locate(new: ast.stmt, old: ast.stmt) -> ast.stmt:
    """Give generated code the location of the statement it replaces."""
    for m in ast.walk(new):
        if not hasattr(m, "lineno") and isinstance(m, (ast.stmt, ast.expr)):
            ast.copy_location(m, old)
    return new


class _Exprs(ast.NodeTransformer):
    """``and`` / ``or`` / ``not`` / ``a if c else b`` -> staged helpers."""

    def __init__(
        self,
        site: Callable[[ast.AST], ast.expr],
        first_param: str | None = None,
        builtins_ok: set[str] | frozenset[str] = frozenset(),
        readable: set[str] | frozenset[str] = frozenset(),
    ):
        self.site = site
        self.first_param = first_param
        self.builtins_ok = builtins_ok
        self.readable = readable

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return node  # nested functions are left alone

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]
    visit_ClassDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "super"
            and not node.args
            and not node.keywords
            and self.first_param
        ):
            # zero-argument super() reads the *current* function's first argument,
            # which is wrong once the call sits inside a generated branch function
            node.args = [_name("__class__"), _name(self.first_param)]
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id in ("print", "max", "min")
            and node.func.id in self.builtins_ok
        ):
            # builtins that need a concrete value: staged versions handle traced ones
            node.func = ast.copy_location(_flow_attr(node.func.id + "_"), node.func)
        return node

    def visit_Compare(self, node: ast.Compare) -> Any:
        self.generic_visit(node)
        if len(node.ops) < 2 or _has_walrus(node):
            return node
        ops = ast.Tuple(elts=[ast.Constant(type(o).__name__) for o in node.ops], ctx=ast.Load())
        return ast.copy_location(
            _call("cmp_", node.left, ops, *[_thunk(c) for c in node.comparators]), node
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.generic_visit(node)
        if _has_walrus(node):
            return node  # `:=` inside a thunk would bind in the wrong scope
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
        if _has_walrus(node):
            return node
        reads = sorted((_reads([node.body]) | _reads([node.orelse])) & self.readable)
        return ast.copy_location(
            _call(
                "ifexp", node.test, _thunk(node.body, reads), _thunk(node.orelse, reads),
                self.site(node), _state(reads),
            ),
            node,
        )  # fmt: skip


class _Converter:
    def __init__(
        self,
        fdef: ast.FunctionDef,
        filename: str,
        shadowed: set[str],
        freevars: Iterable[str] = (),
    ):
        self.freevars = set(freevars)
        self.fdef = fdef
        self.filename = filename
        self.n = 0
        self.count = 0
        self.uses_return_flag = False
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
        self.declared = declared
        self.first_param = (args.posonlyargs + args.args)[0].arg if args.posonlyargs + args.args else None
        self.locals = (set(params) | _assigned(fdef.body)) - declared
        # Names used inside nested functions, lambdas and generator expressions.
        # A statement assigning one of them is not converted: conversion moves
        # the assignment into a helper function, and the closure would stop
        # seeing the updates.
        self.captured: set[str] = set()
        for m in ast.walk(fdef):
            if m is not fdef and isinstance(m, _SCOPES[:3] + (ast.GeneratorExp,)):
                self.captured |= _free_names(m)
        #: names whose values may be traced: locals and enclosing-function variables
        self.readable = (self.locals | self.freevars) - declared
        self.builtins_ok = {
            b for b in ("range", "enumerate", "zip", "reversed", "print", "max", "min") if b not in self.locals and b not in shadowed
        }

    def site(self, node: ast.AST) -> ast.expr:
        extra = getattr(node, "_bn_site", "")
        return ast.Constant(f"{self.filename}:{getattr(node, 'lineno', 0)}{extra}")

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

    def rewrite_inplace(self, stmts: list[ast.stmt]) -> None:
        """In-place updates of a local variable become rebinding assignments.

        ``x[i] = v`` -> ``x = setitem_(v, x, i)``, which is ``x.at[i].set(v)`` for
        a JAX array and a real item assignment for anything else. Likewise
        ``x[i] += v``, ``s.field = v`` (``_replace`` for NamedTuples) and
        ``assert``.
        """
        holder = ast.Module(body=stmts, type_ignores=[])
        for m in [holder, *_walk_all(stmts)]:
            if isinstance(m, _SCOPES):
                continue
            for fld in ("body", "orelse", "finalbody"):
                seq = getattr(m, fld, None)
                if not isinstance(seq, list):
                    continue
                for i, s in enumerate(seq):
                    new = self._inplace(s)
                    if new is not None:
                        seq[i] = _locate(new, s)

    def _inplace(self, s: ast.stmt) -> ast.stmt | None:
        if isinstance(s, ast.Assert):
            msg = _thunk(s.msg) if s.msg is not None else ast.Constant(None)
            return ast.Expr(value=_call("assert_", s.test, msg, self.site(s)))
        if isinstance(s, ast.Assign) and len(s.targets) == 1:
            target, op = s.targets[0], None
        elif isinstance(s, ast.AugAssign):
            target, op = s.target, type(s.op).__name__
        else:
            return None
        base = getattr(target, "value", None)
        if not (isinstance(base, ast.Name) and base.id in self.locals):
            return None
        if base.id in self.declared or base.id in self.captured or _has_walrus(s):
            return None
        if isinstance(target, ast.Subscript):
            args = [s.value, _name(base.id), _index_expr(target.slice)]
            helper = "setitem_" if op is None else "augitem_"
        elif isinstance(target, ast.Attribute):
            args = [s.value, _name(base.id), ast.Constant(target.attr)]
            helper = "setattr_" if op is None else "augattr_"
        else:
            return None
        if op is not None:
            args.append(ast.Constant(op))
        return ast.Assign(targets=[_name(base.id, store=True)], value=_call(helper, *args))

    def rewrite_annotations(self, stmts: list[ast.stmt]) -> int:
        """``h: f32[B, T] = v`` -> ``h = local_(v, lambda: f32[B, T], ...)`` (a checked local)."""
        count = 0
        holder = ast.Module(body=stmts, type_ignores=[])
        for m in [holder, *_walk_all(stmts)]:
            if isinstance(m, _SCOPES):
                continue
            for fld in ("body", "orelse", "finalbody"):
                seq = getattr(m, fld, None)
                if not isinstance(seq, list):
                    continue
                for i, s in enumerate(seq):
                    if _is_local_annotation(s) and s.target.id in self.locals:
                        call = _call(
                            "local_", s.value, _thunk(s.annotation),
                            ast.Constant(s.target.id), self.site(s),
                        )  # fmt: skip
                        seq[i] = _locate(
                            ast.Assign(targets=[_name(s.target.id, store=True)], value=call), s
                        )
                        count += 1
        return count

    # -- break / continue / return inside loops

    def lower_jumps(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        """Rewrite loops so that they contain no ``break`` / ``continue`` / ``return``.

        ``break`` sets a flag that guards the rest of the loop; ``continue``
        moves the rest of the body into the branches that do not skip it;
        ``return v`` stores the value, sets a flag and breaks, and the loop is
        followed by ``if returned: return value``. Innermost loops first.
        """
        out: list[ast.stmt] = []
        for s in stmts:
            if not isinstance(s, _SCOPES):
                for fld in ("body", "orelse", "finalbody"):
                    seq = getattr(s, fld, None)
                    if isinstance(seq, list) and seq and isinstance(seq[0], ast.stmt):
                        setattr(s, fld, self.lower_jumps(seq))
                for h in getattr(s, "handlers", []) or []:
                    h.body = self.lower_jumps(h.body)
                for c in getattr(s, "cases", []) or []:
                    c.body = self.lower_jumps(c.body)
            new = self._lower_loop(s) if isinstance(s, (ast.For, ast.While)) else None
            out.extend(new if new is not None else [s])
        return out

    def _lower_loop(self, s: ast.For | ast.While) -> list[ast.stmt] | None:
        body = s.body
        jumps = sum(isinstance(m, (ast.Break, ast.Continue)) for m in _walk_all(body, False))
        returns = sum(isinstance(m, ast.Return) for m in _walk_all(body))
        if not jumps and not returns:
            return None
        if s.orelse or _count_via_ifs(body) != jumps + returns:
            return None  # a jump inside try/with/match, or in a loop that was left alone
        assigned = _assigned([s])
        if (
            assigned & (self.declared | self.captured)
            or _mutates(s)
            or _has([s], _DECL + (ast.NamedExpr,))
        ):
            return None  # this loop will not be converted, so it keeps its real jumps
        if isinstance(s, ast.For):
            targets = _target_names(s.target)
            if targets is None or not set(targets) <= self.locals:
                return None

        self.n += 1
        brk = f"__bn_break_{self.n}__"
        state = {"break": False, "return": False}
        new_body, _ = self._skip(body, brk, state)
        pre: list[ast.stmt] = []
        post: list[ast.stmt] = []
        if state["break"]:
            self.locals.add(brk)
            pre.append(ast.Assign(targets=[_name(brk, store=True)], value=ast.Constant(False)))
            if isinstance(s, ast.For):
                # the loop variable must not advance once the loop has been left
                item = f"__bn_item_{self.n}__"
                self.locals.add(item)
                unpack = ast.Assign(targets=[s.target], value=_name(item))
                guard = ast.If(test=_call("not_", _name(brk)), body=[unpack, *new_body], orelse=[])
                guard._bn_site = "#break guard"  # type: ignore[attr-defined]
                new_body = [guard]
                s.target = _name(item, store=True)
                s._bn_break = brk  # type: ignore[attr-defined]
            else:
                s.test = _call("and_", _thunk(_call("not_", _name(brk))), _thunk(s.test))
        if state["return"]:
            self.uses_return_flag = True
            self.locals |= {RET, RETF}
            post.append(
                ast.If(test=_name(RETF), body=[ast.Return(value=_name(RET))], orelse=[])
            )
        final: list[ast.stmt] = list(new_body) or [ast.Pass()]
        s.body = final
        return [_locate(n, s) for n in [*pre, s, *post]]

    def _skip(
        self, stmts: list[ast.stmt], brk: str, state: dict[str, bool]
    ) -> tuple[list[ast.stmt], bool]:
        """Replace the jumps in a loop body; returns the new block and "always jumps"."""
        out: list[ast.stmt] = []
        for i, s in enumerate(stmts):
            if isinstance(s, ast.Continue):
                return out, True
            if isinstance(s, (ast.Break, ast.Return)):
                state["break"] = True
                if isinstance(s, ast.Return):
                    state["return"] = True
                    value = s.value or ast.Constant(None)
                    out.append(_locate(ast.Assign(targets=[_name(RET, store=True)], value=value), s))
                    out.append(
                        _locate(ast.Assign(targets=[_name(RETF, store=True)], value=ast.Constant(True)), s)
                    )
                out.append(_locate(ast.Assign(targets=[_name(brk, store=True)], value=ast.Constant(True)), s))
                return out, True
            if isinstance(s, ast.If) and (_count_via_ifs(s.body) or _count_via_ifs(s.orelse)):
                rest = stmts[i + 1 :]
                then, a = self._skip(s.body + copy.deepcopy(rest), brk, state)
                other, b = self._skip(s.orelse + rest, brk, state)
                out.append(ast.copy_location(ast.If(test=s.test, body=then or [ast.Pass()], orelse=other), s))
                return out, a and b
            out.append(s)
        return out, False

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
        out: list[ast.stmt] = []
        for i, s in enumerate(stmts):
            # live after s: read by the rest before being rewritten, or live at the
            # end of the block and not definitely rewritten by the rest
            reads, definite = _rbw(stmts[i + 1 :], set())
            out.extend(self.stmt(s, reads | (live_out - definite)))
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
        if isinstance(s, (ast.Try, ast.TryStar, ast.With, ast.AsyncWith)):
            # An exception caught here must see the assignments made before it
            # was raised, which a converted statement only writes back at its end.
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
        assigned = _assigned([s])
        if assigned & self.declared:
            return True  # assigns a global/nonlocal: must stay in this scope
        if assigned & self.captured:
            return True  # a closure shares the variable
        if _mutates(s):
            return True  # in-place updates must happen exactly as written
        return _has([s], _DECL + (ast.NamedExpr,)) or _has_return([s])

    def if_(self, s: ast.If, live: set[str]) -> list[ast.stmt]:
        names = sorted((_assigned(s.body) | _assigned(s.orelse)) & self.locals)
        reads_body, reads_else = _reads(s.body), _reads(s.orelse)
        ok = not self._blocked(s) and not _has_jump(s.body + s.orelse)
        body = self.block(s.body, live)
        orelse = self.block(s.orelse, live)
        if not ok:
            s.body, s.orelse = body, orelse
            return [s]
        self.count += 1
        t, f = self.fresh("true"), self.fresh("false")
        ret = _state(names)
        # Variables the branches only read are passed in too, so that the helper
        # can cut their gradients where a branch is not taken (see `_guard`).
        reads = sorted(((reads_body | reads_else) & self.readable) - set(names))
        call = _call(
            "if_", s.test, _name(t), _name(f), _state(names), _strs(names), self.site(s),
            _state(reads),
        )  # fmt: skip
        both = names + reads
        new = [
            _def(t, both, body, ret),
            _def(f, both, orelse, copy.deepcopy(ret)),
            *_bind(names, call),
        ]
        return [_locate(n, s) for n in new]

    def while_(self, s: ast.While, live: set[str]) -> list[ast.stmt]:
        names = sorted(_assigned(s.body) & self.locals)
        ok = not s.orelse and not self._blocked(s) and not _has_jump(s.body)
        back = _rbw(s.body, set())[0] | _reads([s.test]) | _reads(s.orelse)
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
        new = [_def(t, names, [], s.test), _def(b, names, body, _state(names)), *_bind(names, call)]
        return [_locate(n, s) for n in new]

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
            s.body = self.block(s.body, live | rbw | _reads(s.orelse))
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
            ast.Constant(getattr(s, "_bn_break", None)),
        )  # fmt: skip
        new = [_def(b, [item, *names], [unpack, *body], _state(names), names), *_bind(names, call)]
        return [_locate(n, s) for n in new]


# ----------------------------------------------------------------- top level

_INTERESTING = (ast.If, ast.While, ast.For, ast.IfExp, ast.BoolOp)
_MANGLED = re.compile(r"(?<![\w.])__[A-Za-z0-9]\w*(?<!__)\b|\.__[A-Za-z0-9]\w*(?<!__)\b")


def convert(raw: Callable[..., Any]) -> Callable[..., Any]:
    """Return a control-flow-converted copy of ``raw``, or raise :class:`NotConverted`."""
    if not isinstance(raw, types.FunctionType):
        raise NotConverted("not a plain Python function")
    if hasattr(raw, "__wrapped__"):
        raise NotConverted("wraps another function (its source is the inner function's)")
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
        isinstance(m, _INTERESTING)
        or (isinstance(m, ast.UnaryOp) and isinstance(m.op, ast.Not))
        or _is_local_annotation(m)
        or _is_rewritable(m)
        for m in _walk_all(fdef.body)
    ):
        raise NotConverted("no control flow to convert")
    if "." in raw.__qualname__.replace(".<locals>.", "") and _MANGLED.search(src):
        raise NotConverted("uses private (name-mangled) attributes")

    ast.increment_lineno(tree, code.co_firstlineno - 1)
    conv = _Converter(
        fdef, code.co_filename, set(raw.__globals__) | set(code.co_freevars), code.co_freevars
    )
    conv.rewrite_appends(fdef.body)
    conv.rewrite_inplace(fdef.body)
    n_locals = conv.rewrite_annotations(fdef.body)
    exprs = _Exprs(conv.site, conv.first_param, conv.builtins_ok, conv.readable)
    fdef.body = [exprs.visit(s) for s in fdef.body]
    fdef.body = conv.lower_jumps(fdef.body)
    if conv.uses_return_flag:
        fdef.body.insert(0, ast.Assign(targets=[_name(RETF, store=True)], value=ast.Constant(False)))
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
    orig_name = fdef.name
    fdef.name = "__bn_fn__"
    cells = dict(zip(code.co_freevars, raw.__closure__ or ()))
    if n_locals:  # local annotations may name the function's type parameters
        for tp in getattr(raw, "__type_params__", ()):
            cells.setdefault(tp.__name__, types.CellType(tp))
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
    inner = inner.replace(co_name=orig_name, co_qualname=raw.__qualname__)
    new = types.FunctionType(inner, raw.__globals__, raw.__name__, raw.__defaults__, closure or None)
    new.__kwdefaults__ = raw.__kwdefaults__
    new.__qualname__ = raw.__qualname__
    new.__module__ = raw.__module__
    new.__doc__ = raw.__doc__
    new.__dict__.update(raw.__dict__)
    new.__typax_converted__ = conv.count  # type: ignore[attr-defined]
    new.__typax_locals__ = n_locals  # type: ignore[attr-defined]
    return new


def source(raw: Callable[..., Any]) -> str:
    """The converted source of ``raw`` (for inspection)."""
    src = textwrap.dedent(inspect.getsource(raw))
    tree = ast.parse(src)
    fdef = tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)
    ast.increment_lineno(tree, raw.__code__.co_firstlineno - 1)
    conv = _Converter(
        fdef, raw.__code__.co_filename, set(getattr(raw, "__globals__", {})), raw.__code__.co_freevars
    )
    conv.rewrite_appends(fdef.body)
    conv.rewrite_inplace(fdef.body)
    conv.rewrite_annotations(fdef.body)
    fdef.body = [_Exprs(conv.site, conv.first_param, conv.builtins_ok, conv.readable).visit(s) for s in fdef.body]
    fdef.body = conv.lower_jumps(fdef.body)
    if conv.uses_return_flag:
        fdef.body.insert(0, ast.Assign(targets=[_name(RETF, store=True)], value=ast.Constant(False)))
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

"""Source conversion: plain ``if`` / ``while`` / ``for`` into staged control flow.

``convert(f)`` re-compiles a function so that its control-flow statements call
the helpers in ``runtime.py``. The converted function behaves exactly like the
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
    (x, y) = __expex_flow__.if_(test, __bn_true_1__, __bn_false_1__, (x, y), ...)

What is left alone (and keeps plain Python semantics): loops containing
``break`` / ``continue`` / ``return``, loops with an ``else`` clause,
statements using ``global`` / ``nonlocal`` / ``del`` / ``yield``, nested
function definitions, and generators / async functions.
"""

import ast
import copy
import inspect
import textwrap
import types
from typing import Any, Callable, Iterable

from ..annotations.sym import named_sizes
from . import analysis, axes, runtime

FLOW = "__expex_flow__"
CALL = "__bn_call__"  # the running call, for nested functions to see its sizes
RET = "__bn_ret__"
RETF = "__bn_returned__"

SKIP = "__bn_skip__"


class NotConverted(Exception):
    """Conversion was skipped; the message says why."""


# ------------------------------------------------------------------ analysis


def _rebuild_if(s: ast.If, then: list[ast.stmt], other: list[ast.stmt]) -> ast.If:
    new = ast.copy_location(ast.If(test=s.test, body=then or [ast.Pass()], orelse=other), s)
    if hasattr(s, "_bn_site"):
        new._bn_site = s._bn_site  # type: ignore[attr-defined]
    return new


def _index_expr(node: ast.expr) -> ast.expr:
    """A subscript's index as an ordinary expression (``1:3`` -> ``slice(1, 3)``)."""
    if isinstance(node, ast.Slice):
        none = ast.Constant(None)
        return ast.Call(
            func=_flow_attr("slice_"),
            args=[node.lower or none, node.upper or none, node.step or none],
            keywords=[],
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple(elts=[_index_expr(e) for e in node.elts], ctx=ast.Load())
    return node


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
        axis_names: set[str] | frozenset[str] = frozenset(),
        einsums: tuple[dict[str, str], set[str]] = ({}, set()),
    ):
        self.site = site
        self.first_param = first_param
        self.builtins_ok = builtins_ok
        self.axis_names = axis_names
        #: How `expex.einsum` is spelled here: bare names, and modules it is an attribute of.
        self.einsums = einsums
        #: Calls rewritten to take axes by name.
        self.named = 0
        #: Inside a nested function or lambda: the sizes are those of the call it was
        #: created in (``__bn_call__``), not of whatever call is running when it runs.
        self.call: ast.expr | None = None
        #: Some nested function or lambda uses sizes: the body must capture its call.
        self.uses_call = False
        #: Inside a nested ``def``: only sizes and axes are rewritten; the rest stays plain Python.
        self.plain = False
        #: Variables of nested scopes: not the enclosing function's annotated variables.
        self.hidden: set[str] = set()

    def _nested(self, node: ast.AST, body: Callable[[], Any], plain: bool) -> None:
        """Visit a nested scope's body with its own names hiding the enclosing sizes and variables."""
        bound = analysis.bound_names(node)
        names = self.axis_names - bound
        comp = isinstance(node, analysis.COMPS)  # runs right away, in the call
        parts = [] if comp else node.body if isinstance(node.body, list) else [node.body]  # type: ignore[attr-defined]
        used = not comp and any(isinstance(m, ast.Name) and m.id in names for p in parts for m in ast.walk(p))
        if plain and not used:
            return  # a nested function that uses no size is left alone
        saved = self.axis_names, self.call, self.plain, self.hidden
        self.axis_names, self.hidden = names, self.hidden | bound
        if used:
            self.call, self.plain, self.uses_call = _name(CALL), self.plain or plain, True
        try:
            body()
        finally:
            self.axis_names, self.call, self.plain, self.hidden = saved

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        # A nested function sees the sizes of the call it is defined in, as a closure
        # would: `def score(row): return row.sum(D)`. Its own control flow is left alone.
        node.decorator_list = [self.visit(d) for d in node.decorator_list]  # evaluated outside
        a = node.args
        a.defaults = [self.visit(d) for d in a.defaults]
        a.kw_defaults = [d if d is None else self.visit(d) for d in a.kw_defaults]

        def body() -> None:
            node.body = [self.visit(s) for s in node.body]

        self._nested(node, body, plain=True)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        return node  # nested classes are left alone

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        if getattr(node, "_expex_annotation", False):
            return node  # `f32[B, T]` in a local annotation: the names stay names
        self._nested(node, lambda: self.generic_visit(node), plain=False)
        return node

    def _comp(self, node: ast.AST) -> Any:
        self._nested(node, lambda: self.generic_visit(node), plain=False)
        return node

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

    def _with_call(self, call: ast.Call) -> ast.Call:
        if self.call is not None:
            call.args.append(self.call)
        return call

    def _var(self, v: Any) -> Any:
        """A variable's name, unless a nested scope's own variable hides the enclosing one."""
        if isinstance(v, str) and v.split(".")[0] in self.hidden:
            return None
        return v

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value is not None:
            node.value = self.visit(node.value)
        return node

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load) and node.id in self.axis_names:
            self.named += 1
            return ast.copy_location(self._with_call(_call("size_", node)), node)  # `T` as a value: its size
        return node

    def _by_name(self, f: ast.expr) -> str | None:
        """``"einsum"`` / ``"rearrange"`` if ``f`` is expex's, however it was imported."""
        names, modules = self.einsums
        if isinstance(f, ast.Name):
            return names.get(f.id)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in modules:
            return f.attr if f.attr in ("einsum", "rearrange") else None
        return None

    def visit_Call(self, node: ast.Call) -> Any:
        which = self._by_name(node.func)
        if which and not any(isinstance(a, ast.Starred) for a in node.args):
            # expex.einsum(q, k, to=(B, T, S)): the operands' variable names find their
            # annotations, and the names in `to` stay names
            to = next((k.value for k in node.keywords if k.arg == "to"), None)
            if to is not None and len(node.keywords) == 1:
                self.named += 1
                ops = [self.visit(a) for a in node.args]
                variables: list[ast.expr] = [
                    _constant(self._var(_operand_name(a)) or self._names(a)) for a in node.args
                ]
                if which == "rearrange" and len(ops) == 1:
                    call = _call("rearrange_", ops[0], variables[0], to, self.site(node))
                else:
                    call = _call(
                        "einsum_", ast.Tuple(elts=ops, ctx=ast.Load()),
                        ast.Tuple(elts=variables, ctx=ast.Load()), to, self.site(node),
                    )  # fmt: skip
                return ast.copy_location(self._with_call(call), node)
        if self.axis_names and analysis.named_axes_call(node, self.axis_names):
            # axis positions keep the names; everything else is visited as usual
            method = isinstance(node.func, ast.Attribute) and node.func.attr in axes.AXIS_METHODS
            node.func = self.visit(node.func)
            if not method:
                node.args = [self.visit(a) for a in node.args]
            for k in node.keywords:
                if k.arg not in axes.AXIS_KEYWORDS:
                    k.value = self.visit(k.value)
            return self._named(node)
        self.generic_visit(node)
        if self.plain:
            return node
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

    def _names(self, e: ast.expr | None) -> tuple[str, ...] | None:
        """The axis names of an expression, when its syntax says: an einsum or rearrange's ``to``,
        also through ``jnp.where`` and arithmetic (``jnp.where(m, expex.einsum(..., to=X), -inf)``)."""
        operands: list[ast.expr] = []
        if isinstance(e, ast.Call):
            to = next((k.value for k in e.keywords if k.arg == "to"), None) if self._by_name(e.func) else None
            if isinstance(e.func, ast.Attribute) and e.func.attr in ("einsum_", "rearrange_") and len(e.args) > 2:
                to = e.args[2]  # already rewritten
            if to is not None:
                elts = to.elts if isinstance(to, (ast.Tuple, ast.List)) else [to]
                names = tuple(x.id for x in elts if isinstance(x, ast.Name))
                return names if len(names) == len(elts) else None
            if isinstance(e.func, ast.Attribute) and e.func.attr == "where":
                operands = e.args[1:]
        elif isinstance(e, ast.BinOp):
            operands = [e.left, e.right]
        elif isinstance(e, ast.UnaryOp):
            operands = [e.operand]
        return next((n for o in operands if (n := self._names(o))), None)

    def _named(self, node: ast.Call) -> ast.expr:
        """``x.sum(T)`` -> ``named_(x, "sum", (T,), {}, site, "x")``."""
        self.named += 1
        if isinstance(node.func, ast.Attribute) and node.func.attr in axes.AXIS_METHODS:
            target: ast.expr = node.func.value
            attr: ast.expr = ast.Constant(node.func.attr)
            var = target.id if isinstance(target, ast.Name) else self._names(target)
        else:
            target, attr = node.func, ast.Constant(None)
            first = node.args[0] if node.args else None
            var = first.id if isinstance(first, ast.Name) else self._names(first)
        kwargs = ast.Dict(
            keys=[ast.Constant(k.arg) for k in node.keywords],
            values=[k.value for k in node.keywords],
        )
        args = ast.Tuple(elts=list(node.args), ctx=ast.Load())
        call = _call("named_", target, attr, args, kwargs, self.site(node), _constant(self._var(var)))
        return ast.copy_location(self._with_call(call), node)

    def visit_Compare(self, node: ast.Compare) -> Any:
        self.generic_visit(node)
        if self.plain or len(node.ops) < 2 or _has_walrus(node):
            return node
        ops = ast.Tuple(elts=[ast.Constant(type(o).__name__) for o in node.ops], ctx=ast.Load())
        return ast.copy_location(
            _call("cmp_", node.left, ops, *[_thunk(c) for c in node.comparators]), node
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.generic_visit(node)
        if self.plain or _has_walrus(node):
            return node  # `:=` inside a thunk would bind in the wrong scope
        attr = "and_" if isinstance(node.op, ast.And) else "or_"
        acc = node.values[0]
        for v in node.values[1:]:
            acc = ast.copy_location(_call(attr, _thunk(acc), _thunk(v)), node)
        return acc

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and not self.plain:
            return ast.copy_location(_call("not_", node.operand), node)
        return node

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        self.generic_visit(node)
        if self.plain or _has_walrus(node):
            return node
        return ast.copy_location(
            _call("ifexp", node.test, _thunk(node.body), _thunk(node.orelse), self.site(node)), node
        )


class _Converter:
    def __init__(
        self,
        fdef: ast.FunctionDef,
        filename: str,
        shadowed: set[str],
    ):
        self.fdef = fdef
        self.filename = filename
        self.n = 0
        self.count = 0
        self.sizes = 0  # uses of size names in the body
        self.lifted: list[list[str]] = []  # leading asserts on sizes, as constraints
        self.defs: dict[str, str] = {}  # leading size definitions, H = heads
        self.uses_return_flag = False
        args = fdef.args
        params = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        if args.vararg:
            params.append(args.vararg.arg)
        if args.kwarg:
            params.append(args.kwarg.arg)
        declared: set[str] = set()
        for m in analysis.walk_all(fdef.body):
            if isinstance(m, (ast.Global, ast.Nonlocal)):
                declared |= set(m.names)
        self.declared = declared
        self.first_param = (args.posonlyargs + args.args)[0].arg if args.posonlyargs + args.args else None
        self.locals = (set(params) | analysis.assigned(fdef.body)) - declared
        # Names used inside nested functions, lambdas and generator expressions.
        # A statement assigning one of them is not converted: conversion moves
        # the assignment into a helper function, and the closure would stop
        # seeing the updates.
        self.captured: set[str] = set()
        for m in ast.walk(fdef):
            if m is not fdef and isinstance(m, analysis.SCOPES[:3] + (ast.GeneratorExp,)):
                self.captured |= analysis.free_names(m)
        #: names whose values may be traced: locals and enclosing-function variables
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
        for m in [holder, *analysis.walk_all(stmts)]:
            if isinstance(m, analysis.SCOPES):
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
        for m in [holder, *analysis.walk_all(stmts)]:
            if isinstance(m, analysis.SCOPES):
                continue
            for fld in ("body", "orelse", "finalbody"):
                seq = getattr(m, fld, None)
                if not isinstance(seq, list):
                    continue
                out: list[ast.stmt] = []
                for s in seq:
                    new = self._inplace(s)
                    out += [s] if new is None else [_locate(n, s) for n in new]
                seq[:] = out

    def _inplace(self, s: ast.stmt) -> list[ast.stmt] | None:
        if isinstance(s, ast.Assign) and (len(s.targets) > 1 or isinstance(s.targets[0], (ast.Tuple, ast.List))):
            # `a[i] = b[j] = v`, `a[i], b[j] = v, w`: the value once, then each target in order
            parts = [n for t in s.targets for n in ast.walk(t)]
            if not any(self._updatable(n) for n in parts) or any(isinstance(n, ast.Starred) for n in parts):
                return None
            value = self.fresh("value")
            out: list[ast.stmt] = [ast.Assign(targets=[_name(value, store=True)], value=s.value)]
            for t in s.targets:
                out += self._assign(t, value)
            return out
        new = self._update(s)
        return None if new is None else [new]

    def _updatable(self, t: ast.AST) -> bool:
        """``x[i]`` or ``x.a`` of a local variable ``x``."""
        return isinstance(t, (ast.Subscript, ast.Attribute)) and isinstance(t.value, ast.Name) and t.value.id in self.locals

    def _assign(self, target: ast.expr, value: str) -> list[ast.stmt]:
        """``target = value``, with tuples unpacked one element at a time and in-place updates rewritten."""
        if isinstance(target, (ast.Tuple, ast.List)):
            items = [self.fresh("item") for _ in target.elts]
            out: list[ast.stmt] = [ast.Assign(targets=[_tuple(items, store=True)], value=_name(value))]
            for t, item in zip(target.elts, items):
                out += self._assign(t, item)
            return out
        plain = ast.Assign(targets=[target], value=_name(value))
        return [self._update(plain) or plain]

    def _update(self, s: ast.stmt) -> ast.stmt | None:
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
            where: ast.expr = _index_expr(target.slice)
            helper = "setitem_" if op is None else "augitem_"
        elif isinstance(target, ast.Attribute):
            where = ast.Constant(target.attr)
            helper = "setattr_" if op is None else "augattr_"
        else:
            return None
        if op is None:  # Python evaluates the value, then the target
            args = [s.value, _name(base.id), where]
        else:  # ... but for `op=` it reads the target first, then evaluates the value
            args = [_name(base.id), where, _thunk(s.value), ast.Constant(op)]
        return ast.Assign(targets=[_name(base.id, store=True)], value=_call(helper, *args))

    def rewrite_raises(self, stmts: list[ast.stmt]) -> None:
        """``raise X`` -> ``raise_(lambda: X)``: raised at once in plain Python, and
        when the compiled code reaches it if it sits in a traced branch."""
        for i, s in enumerate(stmts):
            if isinstance(s, analysis.SCOPES + analysis.UNTOUCHED):
                continue
            if isinstance(s, ast.Raise) and s.exc is not None and not _has_walrus(s):
                cause = _thunk(s.cause) if s.cause is not None else ast.Constant(None)
                call = _call("raise_", _thunk(s.exc), cause, self.site(s))
                stmts[i] = _locate(ast.Expr(value=call), s)
                continue
            for fld in ("body", "orelse"):
                seq = getattr(s, fld, None)
                if isinstance(seq, list) and seq and isinstance(seq[0], ast.stmt):
                    self.rewrite_raises(seq)
            for c in getattr(s, "cases", []) or []:
                self.rewrite_raises(c.body)

    def rewrite_annotations(self, stmts: list[ast.stmt]) -> int:
        """``h: f32[B, T] = v`` -> ``h = local_(v, lambda: f32[B, T], ...)`` (a checked local)."""
        count = 0
        holder = ast.Module(body=stmts, type_ignores=[])
        for m in [holder, *analysis.walk_all(stmts)]:
            if isinstance(m, analysis.SCOPES):
                continue
            for fld in ("body", "orelse", "finalbody"):
                seq = getattr(m, fld, None)
                if not isinstance(seq, list):
                    continue
                for i, s in enumerate(seq):
                    if analysis.is_local_annotation(s) and s.target.id in self.locals:
                        ann = _thunk(s.annotation)
                        ann._expex_annotation = True  # type: ignore[attr-defined]
                        call = _call(
                            "local_", s.value, ann,
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
            if isinstance(s, analysis.UNTOUCHED):
                out.append(s)  # left exactly as written, real jumps included
                continue
            if not isinstance(s, analysis.SCOPES):
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
        jumps = sum(isinstance(m, (ast.Break, ast.Continue)) for m in analysis.walk_all(body, False))
        returns = sum(isinstance(m, ast.Return) for m in analysis.walk_all(body))
        if not jumps and not returns:
            return None
        if s.orelse or analysis.count_via_ifs(body) != jumps + returns:
            return None  # a jump inside try/with/match, or in a loop that was left alone
        assigned = analysis.assigned([s])
        if (
            assigned & (self.declared | self.captured)
            or analysis.mutates(s)
            or analysis.has([s], analysis.DECL + (ast.NamedExpr,))
        ):
            return None  # this loop will not be converted, so it keeps its real jumps
        if isinstance(s, ast.For):
            targets = analysis.target_names(s.target)
            if targets is None or not set(targets) <= self.locals:
                return None

        self.n += 1
        brk = f"__bn_break_{self.n}__"
        state = {"break": False, "return": False, "skip": False}
        new_body, _ = self._skip(body, brk, state)
        if state["skip"]:  # reset at the start of every iteration
            self.locals.add(SKIP)
            new_body.insert(0, ast.Assign(targets=[_name(SKIP, store=True)], value=ast.Constant(False)))
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
            leave = ast.If(test=_name(RETF), body=[ast.Return(value=_name(RET))], orelse=[])
            leave._bn_site = "#return"  # type: ignore[attr-defined]
            post.append(leave)
        final: list[ast.stmt] = list(new_body) or [ast.Pass()]
        s.body = final
        return [_locate(n, s) for n in [*pre, s, *post]]

    def _skip(
        self, stmts: list[ast.stmt], brk: str, state: dict[str, bool], flagged: bool = False
    ) -> tuple[list[ast.stmt], bool]:
        """Replace the jumps in a loop body; returns the new block and "always jumps".

        Where possible the rest of a block simply moves into the branch that
        does not jump. Where both branches may or may not jump, a per-iteration
        flag is set instead and the rest is guarded by it (``flagged``), which
        keeps the generated code linear in the size of the original.
        """
        out: list[ast.stmt] = []

        def set_flag(name: str, at: ast.stmt) -> ast.stmt:
            return _locate(ast.Assign(targets=[_name(name, store=True)], value=ast.Constant(True)), at)

        for i, s in enumerate(stmts):
            if isinstance(s, ast.Continue):
                if flagged:
                    out.append(set_flag(SKIP, s))
                return out, True
            if isinstance(s, (ast.Break, ast.Return)):
                state["break"] = True
                if isinstance(s, ast.Return):
                    state["return"] = True
                    value = s.value or ast.Constant(None)
                    out.append(_locate(ast.Assign(targets=[_name(RET, store=True)], value=value), s))
                    out.append(set_flag(RETF, s))
                out.append(set_flag(brk, s))
                if flagged:
                    out.append(set_flag(SKIP, s))
                return out, True
            if isinstance(s, ast.If) and (analysis.count_via_ifs(s.body) or analysis.count_via_ifs(s.orelse)):
                rest = stmts[i + 1 :]
                ja, jb = analysis.always(s.body, analysis.JUMPS), analysis.always(s.orelse, analysis.JUMPS)
                if flagged or not (ja or jb):
                    state["skip"] = True
                    then, _ = self._skip(s.body, brk, state, True)
                    other, _ = self._skip(s.orelse, brk, state, True)
                    out.append(_rebuild_if(s, then, other))
                    if rest:
                        tail, _ = self._skip(rest, brk, state, flagged)
                        guard = ast.If(test=_call("not_", _name(SKIP)), body=tail or [ast.Pass()], orelse=[])
                        guard._bn_site = "#skip guard"  # type: ignore[attr-defined]
                        out.append(_locate(guard, s))
                    return out, False
                # exactly the branches that fall through continue with the rest
                then, a = self._skip(s.body if ja else s.body + rest, brk, state)
                other, b2 = self._skip(s.orelse if jb else s.orelse + rest, brk, state)
                out.append(_rebuild_if(s, then, other))
                return out, a and b2
            out.append(s)
        return out, False

    def lower_returns(self, stmts: list[ast.stmt], flagged: bool = False) -> tuple[list[ast.stmt], bool]:
        """Rewrite early returns inside ``if`` so that the function has a single exit.

        Same scheme as ``_skip``: the rest of a block moves into the branch that
        does not return, or, where both may or may not, is guarded by a flag.
        """
        out: list[ast.stmt] = []
        for i, s in enumerate(stmts):
            if isinstance(s, ast.Return):
                value = s.value or ast.Constant(None)
                out.append(ast.copy_location(ast.Assign(targets=[_name(RET, store=True)], value=value), s))
                if flagged:
                    self.uses_return_flag = True
                    out.append(_locate(ast.Assign(targets=[_name(RETF, store=True)], value=ast.Constant(True)), s))
                return out, True
            if isinstance(s, ast.If) and (analysis.returns_via_ifs(s.body) or analysis.returns_via_ifs(s.orelse)):
                rest = stmts[i + 1 :]
                ja, jb = analysis.always(s.body, (ast.Return,)), analysis.always(s.orelse, (ast.Return,))
                if flagged or not (ja or jb):
                    self.uses_return_flag = True
                    self.locals.add(RETF)
                    then, _ = self.lower_returns(s.body, True)
                    other, _ = self.lower_returns(s.orelse, True)
                    out.append(_rebuild_if(s, then, other))
                    if rest:
                        tail, _ = self.lower_returns(rest, flagged)
                        guard = ast.If(test=_call("not_", _name(RETF)), body=tail or [ast.Pass()], orelse=[])
                        guard._bn_site = "#return guard"  # type: ignore[attr-defined]
                        out.append(_locate(guard, s))
                    return out, False
                then, a = self.lower_returns(s.body if ja else s.body + rest)
                other, b2 = self.lower_returns(s.orelse if jb else s.orelse + rest)
                out.append(_rebuild_if(s, then, other))
                return out, a and b2
            out.append(s)
        return out, False

    # -- statements

    def block(self, stmts: list[ast.stmt], live_out: set[str]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for i, s in enumerate(stmts):
            # live after s: read by the rest before being rewritten, or live at the
            # end of the block and not definitely rewritten by the rest
            reads, definite = analysis.read_before_written(stmts[i + 1 :], set())
            out.extend(self.stmt(s, reads | (live_out - definite)))
        return out

    def stmt(self, s: ast.stmt, live: set[str]) -> list[ast.stmt]:
        if isinstance(s, ast.If):
            return self.if_(s, live)
        if isinstance(s, ast.While):
            return self.while_(s, live)
        if isinstance(s, ast.For):
            return self.for_(s, live)
        if isinstance(s, analysis.SCOPES):
            return [s]
        if isinstance(s, analysis.UNTOUCHED):
            # An exception caught here must see the assignments made before it
            # was raised, which a converted statement only writes back at its end.
            return [s]
        inner = live | analysis.reads([s])
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
        assigned = analysis.assigned([s])
        if assigned & self.declared:
            return True  # assigns a global/nonlocal: must stay in this scope
        if assigned & self.captured:
            return True  # a closure shares the variable
        if analysis.mutates(s):
            return True  # in-place updates must happen exactly as written
        return analysis.has([s], analysis.DECL + (ast.NamedExpr,)) or analysis.has_return([s])

    def if_(self, s: ast.If, live: set[str]) -> list[ast.stmt]:
        names = sorted((analysis.assigned(s.body) | analysis.assigned(s.orelse)) & self.locals)
        ok = not self._blocked(s) and not analysis.has_jump(s.body + s.orelse)
        body = self.block(s.body, live)
        orelse = self.block(s.orelse, live)
        if not ok:
            s.body, s.orelse = body, orelse
            return [s]
        self.count += 1
        t, f = self.fresh("true"), self.fresh("false")
        ret = _state(names)
        call = _call("if_", s.test, _name(t), _name(f), _state(names), _strs(names), self.site(s))
        new = [_def(t, names, body, ret), _def(f, names, orelse, copy.deepcopy(ret)), *_bind(names, call)]
        return [_locate(n, s) for n in new]

    def while_(self, s: ast.While, live: set[str]) -> list[ast.stmt]:
        names = sorted(analysis.assigned(s.body) & self.locals)
        ok = not s.orelse and not self._blocked(s) and not analysis.has_jump(s.body)
        back = analysis.read_before_written(s.body, set())[0] | analysis.reads([s.test]) | analysis.reads(s.orelse)
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
        targets = analysis.target_names(s.target)
        ok = (
            targets is not None
            and not s.orelse
            and not self._blocked(s)
            and not analysis.has_jump(s.body)
            and set(targets) <= self.locals
        )
        rbw, definite = analysis.read_before_written(s.body, set(targets or ()))
        if not ok:
            s.body = self.block(s.body, live | rbw | analysis.reads(s.orelse))
            return [s]
        assert targets is not None
        nameset = (analysis.assigned(s.body) | set(targets)) & self.locals
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


def _constant(v: str | tuple[str, ...] | None) -> ast.expr:
    """A name, or a tuple of names, as a literal."""
    return ast.Tuple(elts=[ast.Constant(x) for x in v], ctx=ast.Load()) if isinstance(v, tuple) else ast.Constant(v)


def _operand_name(a: ast.expr) -> str | None:
    """An einsum operand's name for finding its annotation: ``q``, or ``self.wq`` for a field."""
    if isinstance(a, ast.Name):
        return a.id
    if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name):
        return f"{a.value.id}.{a.attr}"
    return None


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
    names = analysis.axis_names(raw, fdef)  # size names, used as sizes or axes
    if not any(
        isinstance(m, analysis.INTERESTING)
        or (isinstance(m, ast.UnaryOp) and isinstance(m.op, ast.Not))
        or (isinstance(m, ast.Compare) and len(m.ops) > 1)  # a chained comparison
        or analysis.is_local_annotation(m)
        or analysis.is_rewritable(m)
        or (isinstance(m, ast.Name) and m.id in names)
        for m in analysis.walk_all(fdef.body)
    ) and not analysis.size_asserts(fdef, names, {}, []) and not any(
        isinstance(m, ast.Name) and m.id in names for s in fdef.body for m in ast.walk(s)
    ):  # a size used in a nested function
        raise NotConverted("no control flow to convert")
    if "." in raw.__qualname__.replace(".<locals>.", "") and analysis.MANGLED.search(src):
        raise NotConverted("uses private (name-mangled) attributes")

    conv, n_locals = _rewrite(tree, fdef, raw)

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
    for n, dim in named_sizes(raw).items():  # f32[R:6]: R is a name in the body
        cells[n] = types.CellType(dim)
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
    raw.__globals__.setdefault(FLOW, runtime)
    inner = inner.replace(co_name=orig_name, co_qualname=raw.__qualname__)
    new = types.FunctionType(inner, raw.__globals__, raw.__name__, raw.__defaults__, closure or None)
    new.__kwdefaults__ = raw.__kwdefaults__
    new.__qualname__ = raw.__qualname__
    new.__module__ = raw.__module__
    new.__doc__ = raw.__doc__
    new.__dict__.update(raw.__dict__)
    new.__expex_converted__ = conv.count  # type: ignore[attr-defined]
    new.__expex_locals__ = n_locals  # type: ignore[attr-defined]
    new.__expex_sizes__ = conv.sizes  # type: ignore[attr-defined]
    new.__expex_where__ = conv.lifted  # type: ignore[attr-defined]
    new.__expex_defs__ = conv.defs  # type: ignore[attr-defined]
    return new


def _rewrite(tree: ast.Module, fdef: ast.FunctionDef, raw: Any) -> tuple[_Converter, int]:
    """Rewrite ``fdef`` (parsed from ``raw``'s source) in place; the shared pipeline."""
    code = raw.__code__
    ast.increment_lineno(tree, code.co_firstlineno - 1)
    defs: dict[str, str] = {}
    def_stmts: list[ast.Assign] = []
    lifted = analysis.size_asserts(fdef, analysis.axis_names(raw, fdef), defs, def_stmts)
    # `H, K = heads, D // heads` defines sizes: H and K become size names, whose
    # value in the body is computed like any size of the call
    for s in def_stmts:
        names = [n.id for n in ast.walk(s.targets[0]) if isinstance(n, ast.Name)]
        new = [
            _locate(ast.Assign(targets=[_name(n, store=True)], value=_call("dim_", ast.Constant(n))), s)
            for n in names
        ]
        i = fdef.body.index(s)
        fdef.body[i : i + 1] = new
    # An assert that became a constraint is checked before the body runs (or
    # after, if it involves the result): the body skips it then.
    for i, (stmt, _) in enumerate(lifted):
        guard = ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=_call("lifted_", ast.Constant(i))),
            body=[stmt], orelse=[],
        )  # fmt: skip
        fdef.body[fdef.body.index(stmt)] = _locate(guard, stmt)
    conv = _Converter(fdef, code.co_filename, set(getattr(raw, "__globals__", {})) | set(code.co_freevars))
    conv.rewrite_appends(fdef.body)
    conv.rewrite_inplace(fdef.body)
    conv.rewrite_raises(fdef.body)
    n_locals = conv.rewrite_annotations(fdef.body)
    exprs = _Exprs(
        conv.site, conv.first_param, conv.builtins_ok, (analysis.axis_names(raw, fdef) - conv.locals) | set(defs),
        analysis.einsum_spellings(raw),
    )
    fdef.body = [exprs.visit(s) for s in fdef.body]
    if exprs.uses_call:  # nested functions see the sizes of the call they are defined in
        fdef.body.insert(0, _locate(ast.Assign(targets=[_name(CALL, store=True)], value=_call("call_")), fdef.body[0]))
    n_locals += exprs.named  # axes by name need the call's sizes, as checked locals do
    conv.sizes = exprs.named
    conv.lifted = [srcs for _, srcs in lifted]
    conv.defs = defs
    fdef.body = conv.lower_jumps(fdef.body)
    if any(
        isinstance(s, ast.If) and (analysis.returns_via_ifs(s.body) or analysis.returns_via_ifs(s.orelse))
        for s in fdef.body
    ):
        body, _ = conv.lower_returns(fdef.body)
        # a path that falls off the end leaves the variable unset: that is `None`
        body.append(ast.Return(value=_call("ret_", _thunk(_name(RET)))))
        fdef.body = body
        conv.locals.add(RET)
    if conv.uses_return_flag:
        fdef.body.insert(0, ast.Assign(targets=[_name(RETF, store=True)], value=ast.Constant(False)))
    fdef.body = conv.block(fdef.body, set())
    return conv, n_locals


def source(raw: Callable[..., Any]) -> str:
    """The converted source of ``raw`` (for inspection)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(raw)))
    fdef = tree.body[0]
    assert isinstance(fdef, ast.FunctionDef)
    _rewrite(tree, fdef, raw)
    fdef.decorator_list = []
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

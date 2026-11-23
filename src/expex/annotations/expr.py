"""Tiny, safe evaluator for dimension expressions such as ``"2 * T + 1"``.

Expressions are evaluated against an environment mapping dimension names to
sizes. Sizes are usually ints, but may be JAX symbolic dimensions, which
overload the same arithmetic operators.
"""

import ast
import operator
from typing import Any, Mapping

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
}
_FUNCS = {"max": max, "min": min}


class ExprError(ValueError):
    """A dimension expression could not be parsed."""


def _validate(node: ast.AST, src: str, names: set[str], allow_compare: bool) -> None:
    if isinstance(node, ast.Constant):
        if type(node.value) is not int:
            raise ExprError(f"only integer literals are allowed in {src!r}")
    elif isinstance(node, ast.Name):
        if node.id not in _FUNCS:
            names.add(node.id)
    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            raise ExprError(f"use `//` instead of `/` in {src!r}")
        if type(node.op) not in _BIN:
            raise ExprError(f"unsupported operator in {src!r}")
        _validate(node.left, src, names, False)
        _validate(node.right, src, names, False)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _validate(node.operand, src, names, False)
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        for a in node.args:
            _validate(a, src, names, False)
    elif isinstance(node, ast.Compare) and allow_compare:
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMP:
            raise ExprError(f"unsupported comparison in {src!r}")
        _validate(node.left, src, names, False)
        _validate(node.comparators[0], src, names, False)
    else:
        raise ExprError(f"unsupported syntax in dimension expression {src!r}")


def _eval(node: ast.AST, env: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _BIN[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call):
        return _FUNCS[node.func.id](*[_eval(a, env) for a in node.args])  # type: ignore[attr-defined]
    if isinstance(node, ast.Compare):
        return _CMP[type(node.ops[0])](
            _eval(node.left, env), _eval(node.comparators[0], env)
        )
    raise AssertionError(node)


class Expr:
    """A parsed arithmetic expression over dimension names."""

    __slots__ = ("src", "names", "_tree", "is_constraint")

    def __init__(self, src: str, *, constraint: bool = False):
        self.src = " ".join(src.split())
        try:
            tree = ast.parse(self.src, mode="eval").body
        except SyntaxError as e:
            raise ExprError(f"cannot parse dimension expression {src!r}") from e
        names: set[str] = set()
        _validate(tree, self.src, names, constraint)
        if constraint and not isinstance(tree, ast.Compare):
            raise ExprError(f"constraint {src!r} must be a comparison, e.g. 'T >= 2'")
        self._tree = tree
        self.names = frozenset(names)
        self.is_constraint = constraint

    def __call__(self, env: Mapping[str, Any]) -> Any:
        return _eval(self._tree, env)

    def __repr__(self) -> str:
        return self.src

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expr) and other.src == self.src

    def __hash__(self) -> int:
        return hash(self.src)


_JAX_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
_JAX_FUNCS = {ast.FloorDiv: "floordiv", ast.Mod: "mod"}


def _to_jax(node: ast.AST, consts: Mapping[str, int]) -> str:
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Name):
        return str(consts[node.id]) if node.id in consts else node.id
    if isinstance(node, ast.BinOp):
        l, r = _to_jax(node.left, consts), _to_jax(node.right, consts)
        if type(node.op) in _JAX_OPS:
            return f"({l} {_JAX_OPS[type(node.op)]} {r})"
        if type(node.op) in _JAX_FUNCS:
            return f"{_JAX_FUNCS[type(node.op)]}({l}, {r})"
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
            return f"{l}^{node.right.value!r}"
        raise ValueError
    if isinstance(node, ast.UnaryOp):
        v = _to_jax(node.operand, consts)
        return f"(0 - {v})" if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.Call):
        args = ", ".join(_to_jax(a, consts) for a in node.args)
        return f"{node.func.id}({args})"  # type: ignore[attr-defined]
    raise ValueError


def to_jax_constraint(expr: Expr, consts: Mapping[str, int] | None = None) -> str | None:
    """Translate a constraint to ``jax.export`` syntax, or None if not expressible.

    ``consts`` are names to replace by known integer values.
    """
    consts = consts or {}
    tree = expr._tree
    if not isinstance(tree, ast.Compare):
        return None
    try:
        l, r = _to_jax(tree.left, consts), _to_jax(tree.comparators[0], consts)
    except ValueError:
        return None
    op = type(tree.ops[0])
    # jax wants no outer parentheses on either side
    strip = lambda s: s[1:-1] if s.startswith("(") and s.endswith(")") and _balanced(s[1:-1]) else s  # noqa: E731
    l, r = strip(l), strip(r)
    if op is ast.Eq:
        return f"{l} == {r}"
    if op is ast.GtE:
        return f"{l} >= {r}"
    if op is ast.LtE:
        return f"{l} <= {r}"
    if op is ast.Gt:
        return f"{l} >= {r} + 1"
    if op is ast.Lt:
        return f"{l} + 1 <= {r}"
    return None


def divisibility(expr: Expr, consts: Mapping[str, int] | None = None) -> tuple[str, int] | None:
    """``("K", 2)`` for the constraint ``K % 2 == 0`` (the modulus may be a name in ``consts``), else None."""
    tree, consts = expr._tree, consts or {}
    if not (isinstance(tree, ast.Compare) and isinstance(tree.ops[0], ast.Eq)):
        return None
    left, right = tree.left, tree.comparators[0]
    if isinstance(left, ast.Constant):
        left, right = right, left
    if not (isinstance(right, ast.Constant) and right.value == 0 and isinstance(left, ast.BinOp)):
        return None
    if not (isinstance(left.op, ast.Mod) and isinstance(left.left, ast.Name)) or left.left.id in consts:
        return None
    m = left.right
    k = m.value if isinstance(m, ast.Constant) else consts.get(m.id) if isinstance(m, ast.Name) else None
    return (left.left.id, k) if isinstance(k, int) and k > 1 else None


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0

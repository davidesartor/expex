"""A generator of small random Python functions for differential fuzzing of expex.

Each program is a tree of statements and expressions (``Block``, ``If``, ``Loop``,
``Line``, ``E``) rendered twice:

- ``render("expex")``: the function as a user writes it under ``@expex.fn``, with
  sizes in the signature (``def f[N, D](x: f32[N], ...)``), sizes used by name as
  values and as axes (``m.sum(D)``), and in-place updates (``t[i] += v``);
- ``render("truth")``: the same function in plain Python, the ground truth: the
  sizes are read off the arguments (``N = x.shape[0]``), axes are positions
  (``m.sum(1)``), and an in-place update copies the array through NumPy and
  writes it back, which is what ``t[i] += v`` means on a mutable array that no
  other name shares.

Every function has the same signature::

    def f[N, D](x: f32[N], y: f32[N], m: f32[N, D], s: f32, k: i32, n: int)

and the generator tracks a type for every variable, so that programs are
well-typed in plain Python: they never raise there (the oracle discards the
few that do). Types:

    F  a float scalar (f32[], or a Python float)     V  f32[N]
    I  an int scalar (i32[], or an int, maybe traced) W  f32[D]
    P  a Python int (static: sizes, `n`, literals)   M  f32[N, D]
    B  a condition

The constructs, and how often each is drawn, are listed in ``COVERAGE``.
Values stay small and dyadic (inputs are multiples of 1/16, products are
clipped), so that every mode computes them exactly and a comparison never
flips on rounding.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# ------------------------------------------------------------------ the tree


@dataclass
class E:
    """An expression: ``tpl`` with ``{0}``, ``{1}``... filled by the children.

    ``alt`` is the truth rendering's template when it differs (axes by position).
    """

    t: str  # its type
    tpl: str
    kids: list["E"] = field(default_factory=list)
    alt: str | None = None

    def render(self, mode: str) -> str:
        tpl = self.alt if (mode == "truth" and self.alt is not None) else self.tpl
        return tpl.format(*[k.render(mode) for k in self.kids])

    def size(self) -> int:
        return 1 + sum(k.size() for k in self.kids)


class Stmt:
    def lines(self, mode: str, ind: int) -> list[str]:
        raise NotImplementedError

    def blocks(self) -> list["Block"]:
        return []

    def exprs(self) -> list[E]:
        return []


@dataclass
class Line(Stmt):
    """One simple statement: ``tpl`` filled with the rendered expressions ``es``."""

    tpl: str
    es: list[E] = field(default_factory=list)
    truth: str | None = None  # the truth's template, when it differs
    tag: str = ""

    def lines(self, mode: str, ind: int) -> list[str]:
        tpl = self.truth if (mode == "truth" and self.truth is not None) else self.tpl
        text = tpl.format(*[e.render(mode) for e in self.es])
        return [" " * ind + ln for ln in text.split("\n")]

    def exprs(self) -> list[E]:
        return self.es


@dataclass
class Block:
    body: list[Stmt] = field(default_factory=list)

    def lines(self, mode: str, ind: int) -> list[str]:
        out: list[str] = []
        for s in self.body:
            out += s.lines(mode, ind)
        return out or [" " * ind + "pass"]


@dataclass
class If(Stmt):
    cond: E
    then: Block
    other: Block | None = None
    elif_: bool = False  # render `other` (a lone If) as `elif`

    def lines(self, mode: str, ind: int, kw: str = "if") -> list[str]:
        out = [" " * ind + f"{kw} {self.cond.render(mode)}:"] + self.then.lines(mode, ind + 4)
        if self.other is not None and self.other.body:
            o = self.other.body
            if self.elif_ and len(o) == 1 and isinstance(o[0], If):
                out += o[0].lines(mode, ind, "elif")
            else:
                out += [" " * ind + "else:"] + self.other.lines(mode, ind + 4)
        return out

    def blocks(self) -> list[Block]:
        return [self.then] + ([self.other] if self.other is not None else [])

    def exprs(self) -> list[E]:
        return [self.cond]


@dataclass
class Loop(Stmt):
    """``for <head>:`` or ``while <head>:``; ``pre`` runs before (a counter, an empty list)."""

    head: str
    es: list[E]
    body: Block
    pre: list[Stmt] = field(default_factory=list)
    truth: str | None = None

    def lines(self, mode: str, ind: int) -> list[str]:
        out: list[str] = []
        for s in self.pre:
            out += s.lines(mode, ind)
        tpl = self.truth if (mode == "truth" and self.truth is not None) else self.head
        out.append(" " * ind + tpl.format(*[e.render(mode) for e in self.es]) + ":")
        return out + self.body.lines(mode, ind + 4)

    def blocks(self) -> list[Block]:
        return [self.body]

    def exprs(self) -> list[E]:
        return self.es


@dataclass
class Def(Stmt):
    """A nested helper: ``def name(a): <body>``."""

    name: str
    params: str
    body: Block

    def lines(self, mode: str, ind: int) -> list[str]:
        return [" " * ind + f"def {self.name}({self.params}):"] + self.body.lines(mode, ind + 4)

    def blocks(self) -> list[Block]:
        return [self.body]


SIGNATURE = "def f[N, D](x: f32[N], y: f32[N], m: f32[N, D], s: f32, k: i32, n: int):"
TRUTH_SIGNATURE = "def f(x, y, m, s, k, n):"
TRUTH_PROLOGUE = ["N = x.shape[0]", "D = m.shape[1]"]
PARAMS = ("x", "y", "m", "s", "k", "n")
FLOAT_PARAMS = ("x", "y", "m", "s")


@dataclass
class Program:
    body: Block
    ret: str  # the type of the result: "F", "V", "I", "FI", "L"
    seed: int = 0

    def source(self, mode: str = "expex") -> str:
        if mode == "truth":
            lines = [TRUTH_SIGNATURE] + ["    " + ln for ln in TRUTH_PROLOGUE]
        else:
            lines = [SIGNATURE]
        return "\n".join(lines + self.body.lines(mode, 4)) + "\n"

    def size(self) -> int:
        n = 0
        stack: list[Block] = [self.body]
        while stack:
            b = stack.pop()
            for s in b.body:
                n += 1
                stack += s.blocks()
        return n


# ------------------------------------------------------------------ helpers


def ax(name: str, pos: int) -> E:
    """A size used as an axis: ``D`` under expex, its position in the truth."""
    return E("A", name, alt=str(pos))


def lit(t: str, text: str) -> E:
    return E(t, text)


#: How each construct was drawn, summed over every program generated in this process.
COVERAGE: dict[str, int] = {}


def _cov(tag: str) -> None:
    COVERAGE[tag] = COVERAGE.get(tag, 0) + 1


class Ctx:
    """Where a statement is generated: the variables in scope, and what jumps are allowed."""

    def __init__(self, env: dict[str, str], depth: int = 0, loops: int = 0, in_while: bool = False):
        self.env = env
        self.depth = depth  # nesting of blocks
        self.loops = loops  # enclosing loops (break / continue allowed when > 0)
        self.in_while = in_while
        self.staged = depth > 0  # inside a branch or loop: Python ints here may become traced
        #: Variables of the enclosing loop's iterations before this one: their type may not change.
        self.outer: set[str] = set()

    def child(self, loop: bool = False, in_while: bool = False) -> "Ctx":
        c = Ctx(dict(self.env), self.depth + 1, self.loops + (1 if loop else 0), in_while or (self.in_while and not loop))
        c.outer = set(self.env) if loop else set(self.outer)
        return c


def join(a: str, b: str) -> str | None:
    """The type a variable has after paths that left it ``a`` and ``b``."""
    if a == b:
        return a
    nums = {"P": 0, "I": 1, "F": 2}
    if a in nums and b in nums:
        return max(a, b, key=nums.__getitem__) if "P" not in (a, b) or True else None
    return None


class Gen:
    def __init__(self, seed: int, size: int = 1):
        self.rng = random.Random(seed)
        self.seed = seed
        self.n = 0
        self.budget = 6 + 6 * size  # statements left
        self.ret = self.rng.choice(["F", "F", "V", "I", "FI", "F"])
        self.helpers: list[tuple[str, str]] = []  # (name, type of the result) of nested defs

    # -- names

    def fresh(self, p: str = "v") -> str:
        self.n += 1
        return f"{p}{self.n}"

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def pick(self, xs: list[Any]) -> Any:
        return self.rng.choice(xs)

    def vars_of(self, env: dict[str, str], t: str) -> list[str]:
        return [k for k, v in env.items() if v == t]

    # -- expressions

    def expr(self, t: str, env: dict[str, str], d: int = 2) -> E:
        fn = getattr(self, "e_" + t)
        return fn(env, d)

    def idx(self, env: dict[str, str], d: int, size: str = "N") -> E:
        """An index into an axis of size ``size``: in range in plain Python (NumPy raises otherwise)."""
        r = self.rng.random()
        if r < 0.35:
            return lit("I", self.pick(["0", "-1", f"{size} - 1", f"{size} // 2"]))
        if r < 0.75:
            ints = self.vars_of(env, "I") + self.vars_of(env, "P")
            if ints:
                _cov("index:variable")
                return E("I", "{0} % " + size, [lit("I", self.pick(ints))])
        _cov("index:expression")
        return E("I", "({0}) % " + size, [self.e_I(env, d - 1)])

    def e_F(self, env: dict[str, str], d: int) -> E:
        vs = self.vars_of(env, "F")
        leaves: list[Callable[[], E]] = [
            lambda: lit("F", "s"),
            lambda: E("F", "x[{0}]", [self.idx(env, d)]),
            lambda: E("F", "y[{0}]", [self.idx(env, d)]),
            lambda: lit("F", self.pick(["0.5", "1.0", "-2.0", "0.0", "3.0"])),
            lambda: lit("F", self.pick(["x.sum()", "y.max()", "m.sum()", "x.min()", "jnp.dot(x, y)"])),
            lambda: E("F", "m[{0}, {1}]", [self.idx(env, d), self.idx(env, d, "D")]),
        ]
        if vs:
            leaves += [lambda: lit("F", self.pick(vs))] * 3
        if self.helpers and self.chance(0.3):
            name, ht = self.pick(self.helpers)
            if ht == "F" and d > 0:
                _cov("helper call")
                return E("F", name + "({0})", [self.e_F(env, d - 1)])
        if d <= 0 or self.chance(0.3):
            return self.pick(leaves)()
        r = self.rng.randrange(14)
        a = lambda: self.e_F(env, d - 1)  # noqa: E731
        if r == 0:
            return E("F", "({0} + {1})", [a(), a()])
        if r == 1:
            return E("F", "({0} - {1})", [a(), a()])
        if r == 2:
            return E("F", "jnp.clip({0} * {1}, -64.0, 64.0)", [a(), a()])
        if r == 3:
            return E("F", "jnp.abs({0})", [a()])
        if r == 4:
            _cov("builtin max/min")
            return E("F", self.pick(["max({0}, {1})", "min({0}, {1})", "jnp.maximum({0}, {1})"]), [a(), a()])
        if r == 5:
            _cov("ternary")
            return E("F", "({0} if {1} else {2})", [a(), self.e_B(env, d - 1), a()])
        if r == 6:
            return E("F", "({0} + {1})", [a(), self.e_I(env, d - 1)])
        if r == 7:
            return E("F", "({0} * {1})", [a(), lit("P", self.pick(["N", "D", "n", "2"]))])
        if r == 8:
            _cov("size as axis")
            return E("F", self.pick(["m.sum({0})[{1}]", "m.max({0})[{1}]"]), [ax("D", 1), self.idx(env, d)])
        if r == 9:
            return E("F", "jnp.where({0}, {1}, {2})", [self.e_B(env, d - 1), a(), a()])
        if r == 10:
            return E("F", "({0} / 2.0)", [a()])
        if r == 11:
            vs_v = self.vars_of(env, "V")
            if vs_v:
                return E("F", "{0}[{1}]", [lit("V", self.pick(vs_v)), self.idx(env, d)])
            return E("F", "(-{0})", [a()])
        if r == 12:
            _cov("comprehension")
            j = self.fresh("j")
            if self.chance(0.5):
                return E("F", f"sum([{{0}} for {j} in range({{1}})])", [self._with(env, j, "I", "F", d - 1), self.pick([lit("P", "N"), lit("P", "n"), lit("P", "3")])])
            if self.chance(0.3):
                _cov("comprehension with a condition")
                return E("F", f"sum([{{0}} for {j} in range(N) if {{1}}], 0.0)", [self._with(env, j, "I", "F", d - 1), self._with(env, j, "I", "B", d - 1)])
            return E("F", f"jnp.asarray([{{0}} for {j} in x]).sum()", [self._with(env, j, "F", "F", d - 1)])
        ws = self.vars_of(env, "W")
        if ws:
            return E("F", "{0}[{1}]", [lit("W", self.pick(ws)), self.idx(env, d, "D")])
        return E("F", "float({0})", [lit("P", self.pick(["N", "D", "n"]))])

    def _with(self, env: dict[str, str], name: str, t: str, want: str, d: int) -> E:
        inner = dict(env)
        inner[name] = t
        return self.expr(want, inner, d)

    def e_I(self, env: dict[str, str], d: int) -> E:
        vs = self.vars_of(env, "I")
        leaves: list[Callable[[], E]] = [
            lambda: lit("I", "k"),
            lambda: lit("I", self.pick(["0", "1", "2", "-1"])),
            lambda: lit("I", self.pick(["jnp.argmax(x)", "(x > 0).sum()", "jnp.argmin(y)", "(m > 0).sum()"])),
        ]
        if vs:
            leaves += [lambda: lit("I", self.pick(vs))] * 3
        ps = self.vars_of(env, "P")
        if ps:
            leaves.append(lambda: lit("I", self.pick(ps)))
        if d <= 0 or self.chance(0.35):
            return self.pick(leaves)()
        r = self.rng.randrange(8)
        a = lambda: self.e_I(env, d - 1)  # noqa: E731
        if r == 0:
            return E("I", "({0} + {1})", [a(), a()])
        if r == 1:
            return E("I", "({0} - {1})", [a(), a()])
        if r == 2:
            return E("I", "({0} % 3)", [a()])
        if r == 3:
            return E("I", "({0} // 2)", [a()])
        if r == 4:
            _cov("ternary")
            return E("I", "({0} if {1} else {2})", [a(), self.e_B(env, d - 1), a()])
        if r == 5:
            return E("I", "jnp.clip({0} * {1}, -50, 50)", [a(), a()])
        if r == 6:
            return E("I", "({0} + {1})", [a(), self.e_P(env, d - 1)])
        return E("I", "jnp.minimum({0}, {1})", [a(), a()])

    def e_P(self, env: dict[str, str], d: int) -> E:
        vs = self.vars_of(env, "P")
        leaves = [lambda: lit("P", self.pick(["0", "1", "2", "3"])), lambda: lit("P", self.pick(["N", "D", "n", "len(x)"]))]
        if vs:
            leaves.append(lambda: lit("P", self.pick(vs)))
        if d <= 0 or self.chance(0.5):
            return self.pick(leaves)()
        a = lambda: self.e_P(env, d - 1)  # noqa: E731
        _cov("size as value")
        return E("P", self.pick(["({0} + {1})", "({0} - {1})", "max({0}, {1})", "({0} * 2)"]), [a(), a()])

    def e_V(self, env: dict[str, str], d: int) -> E:
        vs = self.vars_of(env, "V")
        leaves = [lambda: lit("V", self.pick(["x", "y", "(x + y)", "x[::-1]", "jnp.cumsum(x)", "jnp.zeros(N)", "jnp.ones(N)"]))]
        if vs:
            leaves += [lambda: lit("V", self.pick(vs))] * 2
        if d <= 0 or self.chance(0.4):
            return self.pick(leaves)()
        a = lambda: self.e_V(env, d - 1)  # noqa: E731
        r = self.rng.randrange(7)
        if r == 0:
            return E("V", "jnp.clip({0} * {1}, -64.0, 64.0)", [a(), self.e_F(env, d - 1)])
        if r == 1:
            return E("V", "({0} + {1})", [a(), a()])
        if r == 2:
            _cov("size as axis")
            return E("V", self.pick(["m.sum({0})", "m.max({0})", "m.mean({0})"]), [ax("D", 1)])
        if r == 3:
            return E("V", "jnp.where({0} > {1}, {0}, {2})", [a(), self.e_F(env, d - 1), a()])
        if r == 4:
            _cov("ternary")
            return E("V", "({0} if {1} else {2})", [a(), self.e_B(env, d - 1), a()])
        if r == 5:
            _cov("size as value")
            return E("V", "(jnp.arange(N) * {0})", [self.e_F(env, d - 1)])
        return E("V", "jnp.maximum({0}, 0.0)", [a()])

    def e_W(self, env: dict[str, str], d: int) -> E:
        vs = self.vars_of(env, "W")
        r = self.rng.randrange(5)
        if vs and r == 0:
            return lit("W", self.pick(vs))
        if r == 1:
            _cov("size as axis")
            return E("W", self.pick(["m.sum({0})", "m.min({0})"]), [ax("N", 0)])
        if r == 2:
            return E("W", "(jnp.ones(D) * {0})", [self.e_F(env, d - 1)])
        return E("W", "m[{0}]", [self.idx(env, d)])

    def e_B(self, env: dict[str, str], d: int) -> E:
        r = self.rng.randrange(14)
        f = lambda: self.e_F(env, d - 1)  # noqa: E731
        i = lambda: self.e_I(env, d - 1)  # noqa: E731
        if r <= 2 or d <= 0:
            return E("B", "({0} " + self.pick([">", "<", ">=", "<="]) + " {1})", [f(), f()])
        if r == 3:
            return E("B", "({0} " + self.pick(["==", "!=", "<", ">"]) + " {1})", [i(), i()])
        if r == 4:
            _cov("chained comparison")
            return E("B", "({0} < {1} " + self.pick(["<", "<="]) + " {2})", [f(), f(), f()])
        if r == 5:
            _cov("chained comparison")
            return E("B", "({0} <= {1} < {2})", [self.e_P(env, 0), i(), self.e_P(env, 1)])
        if r == 6:
            _cov("not / and / or")
            return E("B", "(not {0})", [self.e_B(env, d - 1)])
        if r == 7:
            _cov("not / and / or")
            return E("B", "({0} and {1})", [self.e_B(env, d - 1), self.e_B(env, d - 1)])
        if r == 8:
            _cov("not / and / or")
            return E("B", "({0} or {1})", [self.e_B(env, d - 1), self.e_B(env, d - 1)])
        if r == 9:
            _cov("static comparison")
            return E("B", "({0} " + self.pick([">", "==", "<", ">="]) + " {1})", [self.e_P(env, 1), self.e_P(env, 1)])
        if r == 10:
            _cov("traced vs static comparison")
            return E("B", "({0} " + self.pick([">", "==", "<"]) + " {1})", [i(), self.e_P(env, 1)])
        if r == 11:
            return E("B", "({0} % 2 == 0)", [i()])
        if r == 12:
            return E("B", "(({0}) > 0).any()" if self.chance(0.5) else "(({0}) > 0).all()", [self.e_V(env, d - 1)])
        return E("B", "({0} > {1})", [f(), i()])

    # -- statements

    def block(self, ctx: Ctx, n: int | None = None) -> Block:
        b = Block()
        k = n if n is not None else self.rng.randint(1, 3)
        for _ in range(k):
            if self.budget <= 0:
                break
            stops = self.stmt(ctx, b)
            if stops:
                break  # nothing after an unconditional jump
        return b

    def ret_expr(self, env: dict[str, str], final: bool = False) -> E:
        if self.ret == "FI":
            return E("T", "({0}, {1})", [self.ret_one("F", env, final), self.ret_one("I", env, final)])
        return self.ret_one(self.ret, env, final)

    def ret_one(self, t: str, env: dict[str, str], final: bool) -> E:
        """A result that uses the variables computed so far (else most programs compute nothing)."""
        e = self.expr(t, env, 2)
        if not final:
            return e
        ok = {"F": ("F", "I", "P"), "I": ("I", "P"), "V": ("V", "F", "I", "P"), "W": ("W",)}[t]
        used = [n for n, vt in env.items() if vt in ok and n not in PARAMS]
        self.rng.shuffle(used)
        for n in used[:3]:
            e = E(t, "({0} + {1})", [e, lit(env[n], n)])
        return e

    def assign_new(self, ctx: Ctx, b: Block) -> None:
        t = self.pick(["F", "F", "F", "I", "I", "P", "V", "V", "W"])
        if t == "P" and ctx.staged:
            t = "I"
        name = self.fresh()
        ann = {"V": "f32[N]", "W": "f32[D]", "F": "f32"}.get(t)
        if ann is not None and self.chance(0.2):
            _cov("annotated local variable")
            b.body.append(Line(f"{name}: {ann} = jnp.asarray({{0}}, jnp.float32)", [self.expr(t, ctx.env)]))
        else:
            b.body.append(Line(f"{name} = {{0}}", [self.expr(t, ctx.env)]))
        ctx.env[name] = t

    def stmt(self, ctx: Ctx, b: Block) -> bool:
        """Append one statement; True if it always jumps (nothing may follow)."""
        self.budget -= 1
        env = ctx.env
        r = self.rng.random()
        deep = ctx.depth >= 3
        if ctx.loops and self.chance(0.12):
            return self.jump(ctx, b)
        if ctx.depth and self.chance(0.06):
            _cov("return in branch/loop")
            b.body.append(Line("return {0}", [self.ret_expr(env)]))
            return True
        if r < 0.22 or deep:
            if self.chance(0.4) and env:
                return self.reassign(ctx, b)
            self.assign_new(ctx, b)
            return False
        if r < 0.32:
            return self.aug(ctx, b)
        if r < 0.42:
            return self.subscript(ctx, b)
        if r < 0.60:
            return self.if_(ctx, b)
        if r < 0.80:
            return self.for_(ctx, b)
        if r < 0.88:
            return self.while_(ctx, b)
        if r < 0.94 and not ctx.depth:
            return self.helper(ctx, b)
        self.assign_new(ctx, b)
        return False

    def jump(self, ctx: Ctx, b: Block) -> bool:
        kind = self.pick(["break", "continue", "return"])
        _cov(kind + " in loop")
        if kind == "return":
            line = Line("return {0}", [self.ret_expr(ctx.env)])
        else:
            line = Line(kind)
        if self.chance(0.85):
            cond = self.e_B(ctx.env, 2)
            b.body.append(If(cond, Block([line])))
            return False
        b.body.append(line)
        return True

    def reassign(self, ctx: Ctx, b: Block) -> bool:
        name = self.pick([v for v in ctx.env if not v.startswith("w")] or ["s"])
        t = ctx.env[name]
        if t not in ("F", "I", "P", "V", "W"):
            self.assign_new(ctx, b)
            return False
        new = t
        if t in ("P", "I") and not ctx.loops and name not in PARAMS and self.chance(0.25):
            _cov("int accumulator becomes float")
            new = "F"
        if t == "P" and ctx.staged:
            new = "I" if new == "P" else new
        b.body.append(Line(f"{name} = {{0}}", [self.expr(new if new != "P" else "P", ctx.env)]))
        ctx.env[name] = new
        return False

    def aug(self, ctx: Ctx, b: Block) -> bool:
        cands = [n for n, t in ctx.env.items() if t in ("F", "I", "P", "V") and not n.startswith("w")]
        if not cands:
            name = self.fresh("acc")
            init = self.pick([("0", "P"), ("0.0", "F"), ("s", "F"), ("k", "I")])
            if init[1] == "P" and ctx.staged:
                init = ("0", "I")
            b.body.append(Line(f"{name} = {init[0]}"))
            ctx.env[name] = init[1]
            cands = [name]
        name = self.pick(cands)
        t = ctx.env[name]
        op = self.pick(["+=", "-=", "+=", "*="])
        if t == "F":
            val = self.e_F(ctx.env, 1) if op != "*=" else lit("F", self.pick(["0.5", "-1.0", "2.0"]))
        elif t in ("I", "P"):
            if self.chance(0.25) and (not ctx.loops or name not in ctx.outer) and name not in PARAMS:
                _cov("int accumulator becomes float")
                val, t = self.e_F(ctx.env, 1), "F"
                op = "+="
            else:
                val = self.e_I(ctx.env, 1) if t == "I" or ctx.staged else self.e_P(ctx.env, 1)
                if op == "*=":
                    val = lit("P", self.pick(["2", "-1"]))
                if t == "P" and (ctx.staged or val.t == "I"):
                    t = "I"
        else:
            val = self.e_F(ctx.env, 1)
        _cov("augmented assignment")
        b.body.append(Line(f"{name} {op} {{0}}", [val]))
        if t == "F" and ctx.env[name] in ("I", "P") or ctx.env[name] == "P":
            ctx.env[name] = t
        return False

    def subscript(self, ctx: Ctx, b: Block) -> bool:
        """An in-place update of a local array: ``t[i] += v`` (and the truth's NumPy copy)."""
        arrays = [n for n, t in ctx.env.items() if t in ("V", "W", "M") and n not in PARAMS]
        if not arrays or self.chance(0.25):
            name = self.fresh("t")
            t = self.pick(["V", "V", "W", "M"])
            init = {"V": "x * 1.0", "W": "m[0] * 1.0", "M": "m * 1.0"}[t]
            if self.chance(0.4):
                init = {"V": "jnp.zeros(N)", "W": "jnp.zeros(D)", "M": "jnp.zeros((N, D))"}[t]
            b.body.append(Line(f"{name} = {init}"))
            ctx.env[name] = t
            arrays = [name]
        name = self.pick(arrays)
        t = ctx.env[name]
        size = {"V": "N", "W": "D", "M": "N"}[t]
        op = self.pick(["+=", "=", "-=", "*=", "+="])
        r = self.rng.random()
        if r < 0.6:
            index: E = self.idx(ctx.env, 1, size)
            _cov("subscript update: index")
            if t == "M":
                if self.chance(0.5):
                    index = E("I", "{0}, {1}", [index, self.idx(ctx.env, 1, "D")])
                    val = self.e_F(ctx.env, 1)
                else:
                    val = self.e_W(ctx.env, 1)
            else:
                val = self.e_F(ctx.env, 1)
        elif r < 0.8:
            _cov("subscript update: slice")
            index = lit("I", self.pick(["1:", ":-1", "::2", ":", f"jnp.array([0, 0, {size} - 1])", f"{size} // 2:"]))
            val = self.e_F(ctx.env, 1)
        else:
            _cov("subscript update: mask")
            index = E("B", f"{name} > {{0}}", [self.e_F(ctx.env, 1)])
            val = self.e_F(ctx.env, 1)
            op = self.pick(["=", "+="])
        if op == "*=":
            val = lit("F", self.pick(["2.0", "0.5", "-1.0"]))
        pyop = {"+=": "+", "-=": "-", "*=": "*", "=": "="}[op]
        b.body.append(
            Line(
                f"{name}[{{0}}] {op} {{1}}",
                [index, val],
                truth=f"{name} = __upd__({name}, lambda: np.s_[{{0}}], {pyop!r}, {{1}})",
            )
        )
        return False

    def merge(self, ctx: Ctx, envs: list[dict[str, str]]) -> None:
        """After branches: a variable defined on every path, with the joined type."""
        names = set(envs[0])
        for e in envs[1:]:
            names &= set(e)
        for n in names:
            t = envs[0][n]
            for e in envs[1:]:
                j = join(t, e[n])
                t = j if j is not None else "?"
            if t == "?":
                ctx.env.pop(n, None)
            else:
                ctx.env[n] = t

    def if_(self, ctx: Ctx, b: Block) -> bool:
        _cov("if")
        cond = self.e_B(ctx.env, 2)
        c1 = ctx.child()
        then = self.block(c1)
        envs = [c1.env]
        other = None
        elif_ = False
        if self.chance(0.6):
            c2 = ctx.child()
            if self.chance(0.25):
                _cov("elif")
                elif_ = True
                other = Block()
                self.if_(c2, other)
            else:
                other = self.block(c2)
            envs.append(c2.env)
        else:
            envs.append(dict(ctx.env))
        # a branch that always jumps does not reach what follows
        live = [e for e, blk in zip(envs, [then, other or Block()]) if not _always_jumps(blk)]
        if live:
            self.merge(ctx, live)
        else:
            self.merge(ctx, envs)
        b.body.append(If(cond, then, other, elif_))
        if other is not None and _always_jumps(then) and _always_jumps(other):
            return True
        return False

    def for_(self, ctx: Ctx, b: Block) -> bool:
        i, v = self.fresh("i"), self.fresh("e")
        r = self.rng.randrange(17)
        nonempty = False
        binds: dict[str, str] = {}
        pre: list[Stmt] = []
        es: list[E] = []
        truth = None
        if r == 0:
            head, binds, nonempty = f"for {i} in range(N)", {i: "I"}, True
        elif r == 1:
            head, binds = f"for {i} in range(n)", {i: "I"}
        elif r == 2:
            head, binds = f"for {i} in range(1, N)", {i: "I"}
        elif r == 3:
            _cov("range with step")
            head, binds = f"for {i} in range(0, N, 2)", {i: "I"}
            nonempty = True
        elif r == 4:
            _cov("range with step")
            head, binds, nonempty = f"for {i} in range(N - 1, -1, -1)", {i: "I"}, True
        elif r == 5:
            _cov("range over a traced bound")
            head, binds = f"for {i} in range(jnp.clip(k, 0, 4))", {i: "I"}
        elif r == 6:
            head, binds, nonempty = f"for {v} in x", {v: "F"}, True
        elif r == 7:
            _cov("enumerate")
            head, binds, nonempty = f"for {i}, {v} in enumerate(x)", {i: "I", v: "F"}, True
        elif r == 8:
            _cov("enumerate")
            head, binds, nonempty = f"for {i}, {v} in enumerate(y, 1)", {i: "I", v: "F"}, True
        elif r == 9:
            _cov("zip")
            head, binds, nonempty = f"for {i}, {v} in zip(x, y)", {i: "F", v: "F"}, True
        elif r == 10:
            _cov("reversed")
            head, binds, nonempty = f"for {v} in reversed(x)", {v: "F"}, True
        elif r == 11:
            _cov("reversed")
            head, binds, nonempty = f"for {i} in reversed(range(N))", {i: "I"}, True
        elif r == 12:
            head, binds, nonempty = f"for {v} in m", {v: "W"}, True
        elif r == 13:
            _cov("zip")
            head, binds, nonempty = f"for {i}, {v} in zip(range(D), m[0])", {i: "I", v: "F"}, True
        elif r == 14:
            head, binds = f"for {v} in x[1:]", {v: "F"}
        elif r == 15:
            _cov("range with step")
            head, binds = f"for {i} in range(n, 0, -1)", {i: "I"}
        else:
            head, binds, nonempty = f"for {i} in range(D)", {i: "I"}, True
        _cov("for")
        collect = self.chance(0.3)
        lst = None
        c = ctx.child(loop=True)
        c.env.update(binds)
        if collect:
            lst = self.fresh("out")
            pre.append(Line(f"{lst} = []"))
        promote = self.promoted(ctx, c)
        if collect:  # from the variables at the top of the body: the append goes there or last
            assert lst is not None
            item_t = self.pick(["F", "F", "W", "I"])
            line = Line(f"{lst}.append({{0}})", [self.expr(item_t, c.env, 1)])
            cond = self.e_B(c.env, 1)
        body = self.block(c)
        if promote is not None:
            body.body.insert(0, promote)
        maybe_empty = False
        if collect:
            if self.chance(0.15):
                maybe_empty = True
                _cov("append under a condition")
                body.body.insert(0, If(cond, Block([line])))
            else:
                _cov("list grown and stacked")
                body.body.insert(0 if self.chance(0.5) else len(body.body), line)
                if _always_jumps(Block(body.body[:-1])):
                    body.body.insert(0, body.body.pop())
        b.body.append(Loop(head, es, body, pre, truth))
        jumped = _has_jump(body) or any("return" in ln for ln in body.lines("expex", 0))  # a return in a nested loop too
        if collect:
            # use the collected values right after the loop
            use = self.pick(
                [
                    ("jnp.asarray({}).sum()", "F"),
                    ("float(len({}))", "F"),
                ]
                + ([("sum({}, 0.0)", "F")] if item_t != "W" else [])
                + ([("jnp.stack({}).sum()", "F")] if nonempty and not jumped and not maybe_empty else [])
            )
            name = self.fresh()
            b.body.append(Line(f"{name} = " + use[0].format(lst)))
            ctx.env[name] = use[1]
        if nonempty and not jumped:
            for nm, t in c.env.items():
                if nm not in ctx.env and nm not in binds and t in ("F", "I", "V", "W"):
                    _cov("variable first assigned in a loop, used after")
                    ctx.env[nm] = t
        self.merge(ctx, [ctx.env, {k: c.env.get(k, ctx.env[k]) for k in ctx.env}])
        return False

    def while_(self, ctx: Ctx, b: Block) -> bool:
        _cov("while")
        j = self.fresh("w")
        bound = self.pick(["3", "N", "D + 1", "4"])
        pre: list[Stmt] = [Line(f"{j} = 0")]
        cond = self.e_B(ctx.env, 1)
        if self.chance(0.4):
            _cov("while with complex condition")
            cond2 = self.e_B(ctx.env, 1)
            head = f"while ({j} < {bound} and {{0}}) or ({j} < 2 and {{1}})"
            es = [cond, cond2]
        else:
            head = f"while {j} < {bound} and {{0}}"
            es = [cond]
        c = ctx.child(loop=True, in_while=True)
        c.env[j] = "I"
        promote = self.promoted(ctx, c, " ".join(e.render("expex") for e in es))
        body = self.block(c)
        if promote is not None:
            body.body.insert(0, promote)
        body.body.insert(0, Line(f"{j} += 1"))
        b.body.append(Loop(head, es, body, pre))
        ctx.env[j] = "I"
        self.merge(ctx, [ctx.env, {k: c.env.get(k, ctx.env[k]) for k in ctx.env}])
        return False

    def promoted(self, ctx: Ctx, c: Ctx, avoid: str = "") -> Line | None:
        """``acc += <float>`` first in a loop body, for an int ``acc`` from before the loop:
        a float from the first iteration on (the body sees it as one throughout)."""
        ints = [
            n for n, t in ctx.env.items()
            if t in ("P", "I") and n not in PARAMS and n not in ctx.outer and not n.startswith(("w", "i"))
            and not re.search(rf"\b{n}\b", avoid)
        ]  # fmt: skip (not one an enclosing loop's earlier iterations, or the loop's test, use as an int)
        if not ints or not self.chance(0.4):
            return None
        _cov("int accumulator becomes float in a loop")
        name = self.pick(ints)
        c.env[name] = "F"
        return Line(f"{name} += {{0}}", [self.e_F(c.env, 1)])

    def helper(self, ctx: Ctx, b: Block) -> bool:
        _cov("nested helper")
        name = self.fresh("h")
        a = self.fresh("a")
        inner = dict(ctx.env)
        inner[a] = "F"
        if self.chance(0.4):
            _cov("lambda")
            b.body.append(Line(f"{name} = lambda {a}: {{0}}", [self.e_F(inner, 1)]))
        elif self.chance(0.3):
            _cov("nested helper with its own if")
            body = Block([If(self.e_B(inner, 1), Block([Line("return {0}", [self.e_F(inner, 1)])])), Line("return {0}", [self.e_F(inner, 1)])])
            b.body.append(Def(name, a, body))
        else:
            body = Block([Line("return {0}", [self.e_F(inner, 2)])])
            b.body.append(Def(name, a, body))
        self.helpers.append((name, "F"))
        return False

    def program(self) -> Program:
        ctx = Ctx({"x": "V", "y": "V", "m": "M", "s": "F", "k": "I", "n": "P"})
        body = Block()
        while self.budget > 0:
            if self.stmt(ctx, body):
                break
        if self.chance(0.05) and body.body and _has_jump(body):
            _cov("falls off the end on some paths")  # returns None there
        elif not body.body or not _always_jumps(body):
            body.body.append(Line("return {0}", [self.ret_expr(ctx.env, final=True)]))
        return Program(body, self.ret, self.seed)


def _always_jumps(b: Block) -> bool:
    for s in b.body:
        if isinstance(s, Line) and (s.tpl.startswith(("return", "break", "continue"))):
            return True
        if isinstance(s, If) and s.other is not None and _always_jumps(s.then) and _always_jumps(s.other):
            return True
    return False


def _has_jump(b: Block) -> bool:
    for s in b.body:
        if isinstance(s, Line) and s.tpl.startswith(("return", "break", "continue")):
            return True
        if isinstance(s, If) and any(_has_jump(x) for x in s.blocks()):
            return True
    return False


def generate(seed: int, size: int = 1) -> Program:
    return Gen(seed, size).program()


# ------------------------------------------------------------------ shrinking


def _all_blocks(p: Program) -> list[Block]:
    out, stack = [], [p.body]
    while stack:
        b = stack.pop()
        out.append(b)
        for s in b.body:
            stack += s.blocks()
    return out


SIMPLE = {  # constants first: a variable may have another type where the expression was
    "F": ["1.0", "x[0]"],
    "I": ["1", "0"],
    "P": ["1", "N", "0"],
    "V": ["jnp.ones(N)"],
    "W": ["jnp.ones(D)"],
    "M": ["jnp.ones((N, D))"],
    "B": ["(x[0] > 0)", "True"],
}


def candidates(p: Program) -> list[Program]:
    """Smaller programs: a statement dropped, a branch or loop inlined, an expression made a leaf."""
    out: list[Program] = []
    nb = len(_all_blocks(p))
    for bi in range(nb):
        blk = _all_blocks(p)[bi]
        for si in range(len(blk.body)):
            # drop statement
            q = copy.deepcopy(p)
            qb = _all_blocks(q)[bi]
            del qb.body[si]
            out.append(q)
            s = blk.body[si]
            # inline a block of the statement
            for ki in range(len(s.blocks())):
                q = copy.deepcopy(p)
                qb = _all_blocks(q)[bi]
                st = qb.body[si]
                inner = st.blocks()[ki]
                pre = list(getattr(st, "pre", []))
                qb.body[si : si + 1] = pre + inner.body
                out.append(q)
            if isinstance(s, If) and s.other is not None:
                q = copy.deepcopy(p)
                st = _all_blocks(q)[bi].body[si]
                assert isinstance(st, If)
                st.other = None
                out.append(q)
    # expressions -> leaves
    exprs: list[tuple[int, int, list[int]]] = []

    def walk(e: E, path: list[int], bi: int, si: int) -> None:
        if e.kids or e.alt is None:
            exprs.append((bi, si, list(path)))
        for ki, kid in enumerate(e.kids):
            walk(kid, path + [ki], bi, si)

    for bi, blk in enumerate(_all_blocks(p)):
        for si, s in enumerate(blk.body):
            for ei, e in enumerate(s.exprs()):
                walk(e, [ei], bi, si)
    for bi, si, path in exprs:
        e0 = _get(p, bi, si, path)
        if e0.size() <= 1 and e0.tpl in SIMPLE.get(e0.t, []):
            continue
        for leaf in SIMPLE.get(e0.t, []):
            q = copy.deepcopy(p)
            e = _get(q, bi, si, path)
            e.tpl, e.kids, e.alt = leaf, [], None
            out.append(q)
        # replace by one of its children of the same type
        for kid in range(len(e0.kids)):
            if e0.kids[kid].t == e0.t:
                q = copy.deepcopy(p)
                e = _get(q, bi, si, path)
                c = e.kids[kid]
                e.tpl, e.kids, e.alt = c.tpl, c.kids, c.alt
                out.append(q)
    return out


def _get(p: Program, bi: int, si: int, path: list[int]) -> E:
    s = _all_blocks(p)[bi].body[si]
    e = s.exprs()[path[0]]
    for k in path[1:]:
        e = e.kids[k]
    return e


def shrink(p: Program, still_fails: Callable[[Program], bool], budget: int = 400) -> Program:
    """Greedy: take the first smaller candidate that still fails, until none does."""
    tries = 0
    progress = True
    while progress and tries < budget:
        progress = False
        for q in candidates(p):
            tries += 1
            if tries >= budget:
                break
            try:
                ok = still_fails(q)
            except Exception:
                ok = False
            if ok:
                p = q
                progress = True
                break
    return p

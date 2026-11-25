"""The prover: check a function's declared shapes for *all* sizes.

For each ``@fn`` the prover invents abstract inputs straight from the
annotations, using JAX symbolic dimensions (``B``, ``T``, ...) instead of
numbers, and traces the function once with ``jax.eval_shape``. No data is
needed and nothing is compiled. If the traced result has the declared shape,
the annotation holds for every size.

When a function cannot be traced symbolically (some shape operations are
inconclusive for symbolic sizes) the prover falls back to a few concrete
sizes and says so: that is a spot check, not a proof.
"""

import ast
import importlib
import importlib.util
import itertools
import math
import pkgutil
import random
import re
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, overload

import jax
from jax import export

from ..annotations.errors import ShapeError, StageError, code_of, coded, see
from ..annotations.expr import Expr
from ..annotations.spec import Unsynthesizable
from ..function import REGISTRY, BoundFn, Fn, unwrap
from . import trace
from .trace import PRIMES, iter_eqns, source_of

PROVED = "proved"
SPOT = "spot-checked"
FAILED = "failed"
ERROR = "error"
UNCHECKED = "unchecked"

_RANK = {PROVED: 0, SPOT: 1, UNCHECKED: 2, ERROR: 3, FAILED: 4}
_MAX_CASES = 24


@dataclass
class Report:
    """Outcome of proving one function."""

    fn: Fn[Any, Any]
    status: str
    detail: str = ""
    cases: int = 0
    notes: list[str] = field(default_factory=list)
    #: The code of what went wrong (``EX102``, ``EX702``), the section of ``docs/errors.md`` on it.
    code: str | None = None

    @property
    def ok(self) -> bool:
        """True if nothing is wrong (proved or spot-checked)."""
        return self.status in (PROVED, SPOT)

    @property
    def name(self) -> str:
        mod = getattr(self.fn.raw, "__module__", None)
        return f"{mod}.{self.fn.name}" if mod else self.fn.name

    def _repr_html_(self) -> str:
        from .display import reports_html

        return reports_html([self])

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        p.text(str(self))

    def raise_if_failed(self) -> None:
        if self.status in (FAILED, ERROR):
            err = AssertionError(str(self))
            err.code = self.code  # type: ignore[attr-defined]
            raise err

    def __str__(self) -> str:
        head = f"{self.name}: {self.status}"
        if self.status == PROVED:
            n = self.cases
            head += " for all sizes" + (f" ({n} cases)" if n > 1 else "")
        out = [head]
        if self.detail:
            out.append(_indent(self.detail))
        for note in self.notes:
            out.append(_indent(note))
        if self.code and self.status != PROVED and see(self.code) not in self.detail:
            out.append(_indent(see(self.code).strip()))
        return "\n".join(out)


def _indent(s: str, by: str = "    ") -> str:
    return "\n".join(by + line if line else line for line in s.splitlines())


def _unproven_reshapes(jaxpr: Any) -> list[str]:
    """Reshapes whose element count is not provably preserved.

    JAX does not validate ``reshape`` sizes for symbolic dimensions, so the
    prover does it here; otherwise ``x.reshape(b, d + 1)`` would be "proved".
    """
    out = []
    for eqn in iter_eqns(jaxpr):
        if eqn.primitive.name != "reshape":
            continue
        src = tuple(eqn.invars[0].aval.shape)
        dst = tuple(eqn.outvars[0].aval.shape)
        try:
            same = bool(math.prod(src) == math.prod(dst))
        except Exception:
            same = False
        if not same:
            out.append(
                f"reshape {_fmt(src)} -> {_fmt(dst)} at {source_of(eqn)}: cannot show that "
                f"{math.prod(src)} == {math.prod(dst)}"
            )
    return out


def _fmt(shape: tuple[Any, ...]) -> str:
    return "[" + ", ".join(map(str, shape)) + "]"


def _concrete_sizes(
    f: Fn[Any, Any], attempt: int, statics: dict[str, int]
) -> dict[str, int] | None:
    names = sorted(n for n in f.input_names if not n.startswith("*"))
    rng = random.Random(attempt)
    for tries in range(400):
        if attempt == 2:  # the smallest sizes: catches squeeze() and empty-axis bugs
            if tries:
                return None
            sizes = {n: 1 for n in names}
        elif tries == 0:
            off = 4 if attempt == 3 else attempt  # attempt 3: larger sizes (11, 13, ...)
            sizes = {n: PRIMES[(i + off) % len(PRIMES)] for i, n in enumerate(names)}
        else:
            sizes = {n: rng.randint(1, 12) for n in names}
        try:
            env = {**sizes, **statics}
            if all(bool(w(env)) for w in [*f.where, *f.assumed] if w.names <= env.keys()):
                return sizes
        except Exception:
            continue
    return None


def _summ(e: BaseException, limit: int = 6) -> str:
    lines = str(e).strip().splitlines() or [""]
    text = "\n".join(lines[:limit]) + ("\n..." if len(lines) > limit else "")
    return f"{type(e).__name__}: {text}"


def _full(e: BaseException) -> str:
    notes = [n for n in getattr(e, "__notes__", ()) if n.startswith("expex:")]
    return "\n".join([str(e).strip(), *notes])


@overload
def prove(target: "Fn[Any, Any] | BoundFn[Any, Any]", examples: dict[str, Any] | None = None) -> Report: ...
@overload
def prove(target: "str | types.ModuleType", examples: None = None) -> "Reports": ...
def prove(target: Any, examples: dict[str, Any] | None = None) -> Any:
    """Prove the shape annotations of an ``@fn`` function for all sizes.

    ``target`` may be an ``@fn`` function (returns a :class:`Report`), or a
    module, package or dotted name (returns a list of reports, one for every
    ``@fn`` defined inside it). A jitted ``@fn`` is proved without compiling it.
    """
    target = unwrap(target)
    if isinstance(target, BoundFn):
        f = target.fn
        if f._lazy:
            f._resolve_owner()
        first = next(iter(f.params))
        obj = trace.Template(target.obj) if f.params[first].spec.leafy else target.obj
        return _prove_fn(f, {first: [obj], **(examples or {})})
    if isinstance(target, Fn):
        return _prove_fn(target, examples)
    return Reports(_prove_fn(f, None) for f in collect(target))


def collect(target: Any) -> list[Fn[Any, Any]]:
    """All provable ``@fn`` functions defined in a module/package (by name or object)."""
    if isinstance(target, str):
        target = importlib.import_module(target)
    if not isinstance(target, types.ModuleType):
        raise coded("EX705", TypeError(f"cannot prove {target!r}: expected an @fn function, a module or a name"))
    prefix = target.__name__
    if hasattr(target, "__path__"):
        for info in pkgutil.walk_packages(target.__path__, prefix + "."):
            if not info.name.endswith("__main__"):
                importlib.import_module(info.name)
    out = []
    for f in list(REGISTRY):
        m = getattr(f.raw, "__module__", None) or ""
        if not f.prove or not (m == prefix or m.startswith(prefix + ".")):
            continue
        # Skip functions from a stale copy of the module (e.g. after a reload).
        live = sys.modules.get(m)
        if live is None or getattr(f.raw, "__globals__", None) is not vars(live):
            continue
        out.append(f)
    out.sort(key=lambda f: (getattr(f.raw, "__module__", ""), _lineno(f), f.name))
    return out


def _lineno(f: Fn[Any, Any]) -> int:
    try:
        return f.raw.__code__.co_firstlineno  # type: ignore[attr-defined]
    except AttributeError:
        return 0


def _prove_fn(f: Fn[Any, Any], examples: dict[str, Any] | None) -> Report:
    if f._lazy:
        f._resolve_owner()
    ex = {**f.examples, **(examples or {})}
    try:
        choices = trace.choices(f, ex)
    except trace.Unchecked as u:
        return Report(f, UNCHECKED, str(u), code="EX701")

    abstract = [n for n, c in choices.items() if c == [trace.ABSTRACT]]
    fixed_names = [n for n in choices if n not in abstract]
    var_names = sorted(n[1:] for n in f.input_names if n.startswith("*"))
    ranks: Iterable[tuple[int, ...]] = itertools.product(*[(0, 1, 2)] * len(var_names))
    combos = itertools.product(*[choices[n] for n in fixed_names])
    cases = list(itertools.islice(itertools.product(combos, ranks), _MAX_CASES))
    dropped: list[str] = []

    worst = Report(f, PROVED, cases=len(cases))
    skipped = 0
    for combo, rank in cases:
        fixed = dict(zip(fixed_names, combo))
        var_rank = dict(zip(var_names, rank))
        found = trace.statics(f, fixed)
        found.update(_pinned(f, found))
        if not _admissible(f, found):  # e.g. min_deg=2, max_deg=1 against `assert min_deg <= max_deg`
            skipped += 1
            continue
        scope, dropped = trace.constraints(f, found)
        r = _prove_case(f, fixed, abstract, var_rank, scope, found)
        label = _case_label(fixed, var_rank)
        if label and r.status != PROVED:
            r.detail = f"with {label}:\n{r.detail}"
        if _RANK[r.status] > _RANK[worst.status]:
            r.cases = len(cases)
            worst = r
            if r.status == FAILED:
                break
    if skipped == len(cases):
        return Report(f, UNCHECKED, "no combination of the values tried satisfies the leading asserts.", code="EX703")
    worst.cases = len(cases) - skipped
    if dropped and worst.status == PROVED:
        worst.notes.append(
            "constraints not expressible symbolically were only checked at run time: "
            + ", ".join(f"`{d}`" for d in dropped)
        )
    return worst


def _pinned(f: Fn[Any, Any], statics: dict[str, int]) -> dict[str, int]:
    """Sizes a leading assert fixes from static ints: ``{"P": 63}`` for ``assert P == 3 * (1 + 2 * self.deg)``."""
    out: dict[str, int] = {}
    for w in [*f.where, *f.assumed]:
        tree = w._tree
        if not (isinstance(tree, ast.Compare) and isinstance(tree.ops[0], ast.Eq)):
            continue
        for name, other in ((tree.left, tree.comparators[0]), (tree.comparators[0], tree.left)):
            if isinstance(name, ast.Name) and name.id not in statics and name.id not in out:
                value = Expr(ast.unparse(other))
                try:
                    if value.names <= statics.keys():
                        out[name.id] = int(value(statics))
                        break
                except ArithmeticError:  # e.g. a division by zero: the trace reports it
                    pass
    return out


def _admissible(f: Fn[Any, Any], statics: dict[str, int]) -> bool:
    """Do the static ints satisfy the leading asserts that mention only them?"""
    try:
        return all(bool(w(statics)) for w in [*f.where, *f.assumed] if w.names <= statics.keys())
    except Exception:
        return True  # cannot tell here: the trace will


def _case_label(fixed: dict[str, Any], var_rank: dict[str, int]) -> str:
    parts = [f"{k}={v!r}"[:60] for k, v in fixed.items()]
    parts += [f"*{k} of rank {v}" for k, v in var_rank.items()]
    return ", ".join(parts)


def _at(sizes: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sizes.items() if not k.startswith("_")) or "no free sizes"


def _expex_notes(e: BaseException) -> str:
    return "".join(
        "\n" + _indent(n) for n in getattr(e, "__notes__", ()) if n.startswith("expex:")
    )


def _prove_case(
    f: Fn[Any, Any],
    fixed: dict[str, Any],
    abstract: list[str],
    var_rank: dict[str, int],
    constraints: list[str],
    statics: dict[str, int],
) -> Report:
    # 1. Symbolic sizes: a proof of the declared output, for all sizes at once.
    try:
        jaxprs, sym_err, witness = _symbolic(f, fixed, abstract, var_rank, constraints, statics)
    except Unsynthesizable as u:
        return Report(
            f, UNCHECKED,
            f"the prover cannot invent a value for `{u}`. "
            f"Add @fn(examples={{...}}) for that parameter.", code="EX701",
        )  # fmt: skip
    unproven = list(dict.fromkeys(u for j in jaxprs for u in _unproven_reshapes(j)))

    # 2. Concrete sizes: confirms failures, and validates what symbolic tracing
    #    cannot (JAX defers some checks when sizes are symbolic).
    used: list[dict[str, int]] = []
    for attempt in range(5):
        sizes = _concrete_sizes(f, attempt, statics) if attempt < 4 else _witness_sizes(f, witness, statics)
        if sizes is None:
            continue
        try:
            _, ctx, err = trace.attempt(
                f, fixed, abstract, lambda s=sizes: trace.Sizes(None, dict(s), var_rank, statics)
            )
        except (trace.BadSizes, Unsynthesizable):
            continue
        assert ctx.sizes is not None
        if err is None:
            used.append(dict(ctx.sizes))
            continue
        at = _at(ctx.sizes)
        if isinstance(err, (ShapeError, StageError)):
            shown = sym_err if isinstance(sym_err, ShapeError) else err
            return Report(f, FAILED, _full(shown), notes=[f"confirmed at sizes {at}."], code=code_of(shown))
        hint = ""
        if sym_err is None:
            hint = (
                "\nThe function is fine for symbolic sizes but not for these. If it is not "
                "meant to support them, say so with an `assert` at the top of the body."
            )
        return Report(
            f, ERROR,
            f"the function raised while being traced on abstract inputs ({at}):\n"
            + _indent(_summ(err, 12)) + _expex_notes(err) + hint, code=code_of(err) or "EX704",
        )  # fmt: skip

    if not used:
        why = f"\nsymbolic trace failed with: {_summ(sym_err)}" if sym_err else ""
        return Report(
            f, UNCHECKED, "could not find sizes satisfying the `where` constraints." + why, code="EX703"
        )
    if sym_err is None and not unproven:
        return Report(f, PROVED)
    at = "; ".join(map(_at, used))
    reason = _summ(sym_err) if sym_err is not None else "\n".join(unproven)
    return Report(
        f, SPOT,
        f"holds at sizes {at}, but could not be proved for all sizes:\n" + _indent(reason),
        code=code_of(sym_err) or "EX702",  # the reason's own code, when it has one
    )  # fmt: skip


_INCONCLUSIVE = re.compile(r"comparison '(.+?)' (>=|>|<=|<|==|!=) '(.+?)' is inconclusive")
_MAX_SPLITS = 16


def _regions(a: str, op: str, b: str) -> list[list[str]]:
    """Constraints (in ``jax.export`` syntax) for regions that together cover every size,
    in each of which ``a op b`` has one answer."""
    if op in ("<", "<="):
        a, b, op = b, a, {"<": ">", "<=": ">="}[op]
    if op == ">":
        return [[f"{a} >= {b} + 1"], [f"{b} >= {a}"]]
    if op == ">=":
        return [[f"{a} >= {b}"], [f"{b} >= {a} + 1"]]
    # == / !=: above, below, and equal (JAX takes `term == expression` only)
    return [[f"{a} >= {b} + 1"], [f"{b} >= {a} + 1"], [f"{a} == {b}", f"{b} == {a}"]]


def _pinned_equal(cs: list[str], a: str, b: str) -> list[str]:
    """``cs``, with ``a == b`` added where ``cs`` already forces it (``1 >= T`` for ``T >= 1``).

    JAX compares sizes by their expressions: ``T - 1`` is not ``0`` unless the scope
    says ``T == 1`` in so many words.
    """
    try:
        scope = export.SymbolicScope(cs)
        ea, eb = export.symbolic_shape(f"{a}, {b}", scope=scope)
        if not (ea >= eb and eb >= ea):
            return cs
    except Exception:
        return cs
    for c in (f"{a} == {b}", f"{b} == {a}"):
        try:
            export.SymbolicScope([*cs, c])
            return [*cs, c]
        except ValueError:
            continue
    return cs


def _symbolic(
    f: Fn[Any, Any],
    fixed: dict[str, Any],
    abstract: list[str],
    var_rank: dict[str, int],
    constraints: list[str],
    statics: dict[str, int],
) -> tuple[list[Any], BaseException | None, dict[str, int]]:
    """Trace on symbolic sizes: (jaxprs, error).

    A comparison of sizes that is true for some sizes and false for others
    (``if T > 1``) splits the sizes into regions (``T >= 2``, ``1 >= T``), each traced
    with its constraint added, where the comparison has one answer. The function is
    proved when every region is; a region with no sizes at all is vacuous.
    """
    jaxprs: list[Any] = []
    todo = [list(constraints)]
    splits = 0
    while todo:
        cs = todo.pop()
        jaxpr, _, err = trace.attempt(
            f, fixed, abstract,
            lambda cs=cs: trace.Sizes(export.SymbolicScope(cs), None, var_rank, statics, multiples=trace.multiples(f, statics)),
        )  # fmt: skip
        if err is None:
            jaxprs.append(jaxpr)
            continue
        if isinstance(err, ValueError) and "Unsatisfiable constraint" in str(err):
            continue  # no sizes in this region
        m = _INCONCLUSIVE.search(str(err))
        if m is None or splits >= _MAX_SPLITS:
            return jaxprs, err, _pins(cs) if isinstance(err, ShapeError) else {}
        splits += 1
        for region in _regions(*m.groups()):
            for c in region:  # the first form of the constraint that JAX accepts
                try:
                    export.SymbolicScope([*cs, c])
                except ValueError as e:
                    if "Unsatisfiable" in str(e):
                        break  # an empty region
                    continue
                todo.append(_pinned_equal([*cs, c], *m.group(1, 3)))
                break
            else:
                return jaxprs, err, {}  # a region that cannot be written as a constraint
    return jaxprs, None, {}


_PIN = re.compile(r"\s*([A-Za-z_]\w*)\s*==\s*(-?\d+)\s*$|\s*(-?\d+)\s*==\s*([A-Za-z_]\w*)\s*$")


def _pins(constraints: list[str]) -> dict[str, int]:
    """The sizes a region's constraints fix (``T == 6``): where a symbolic failure can be confirmed."""
    out: dict[str, int] = {}
    for c in constraints:
        if m := _PIN.match(c):
            name, value = (m[1], m[2]) if m[1] else (m[4], m[3])
            out[name] = int(value)
    return out


def _witness_sizes(f: Fn[Any, Any], pins: dict[str, int], statics: dict[str, int]) -> dict[str, int] | None:
    """The sizes of the first attempt, with those a failing region fixes; None if there are none."""
    if not pins:
        return None
    base = _concrete_sizes(f, 0, statics) or {}
    sizes = {**base, **{k: v for k, v in pins.items() if k in base or k in f.input_names}}
    try:
        env = {**sizes, **statics}
        if not all(bool(w(env)) for w in [*f.where, *f.assumed] if w.names <= env.keys()):
            return None
    except Exception:
        return None
    return sizes


class Reports(list):  # type: ignore[type-arg]
    """The reports for a whole module or package; prints as a summary."""

    @property
    def ok(self) -> bool:
        return all(r.ok or r.status == UNCHECKED for r in self)

    def __str__(self) -> str:
        return format_reports(self)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        from .display import reports_html

        return reports_html(self)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        p.text(str(self))


def format_reports(reports: list[Report], verbose: bool = False) -> str:
    """A human-readable summary of many reports."""
    mark = {PROVED: "ok  ", SPOT: "spot", UNCHECKED: "skip", ERROR: "ERR ", FAILED: "FAIL"}
    lines = []
    for r in reports:
        lines.append(f"{mark[r.status]}  {r.name}")
        if r.status in (FAILED, ERROR) or (verbose and (r.detail or r.notes)) or r.status == UNCHECKED:
            body = "\n".join([r.detail, *r.notes]).strip()
            if body:
                lines.append(_indent(body, "        "))
    counts = {s: sum(r.status == s for r in reports) for s in mark}
    lines.append("")
    lines.append(
        f"{len(reports)} functions: "
        + ", ".join(f"{n} {s}" for s, n in counts.items() if n)
    )
    return "\n".join(lines)


def import_target(t: str) -> types.ModuleType:
    """Import a module/package name, or a path to a .py file or package folder."""
    import os

    if t.endswith(".py") or os.sep in t or os.path.isdir(t):
        # a file or folder: find the top of its package, so relative imports work
        path = os.path.abspath(t)
        parts = [] if os.path.isdir(path) else [os.path.splitext(os.path.basename(path))[0]]
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        while os.path.exists(os.path.join(folder, "__init__.py")):
            parts.insert(0, os.path.basename(folder))
            folder = os.path.dirname(folder)
        if parts and parts[-1] == "__init__":
            parts.pop()
        sys.path.insert(0, folder)
        return importlib.import_module(".".join(parts))
    return importlib.import_module(t)


def import_folder(folder: str) -> list[tuple[str, types.ModuleType | BaseException]]:
    """Each ``.py`` file under a folder that is not a package, imported: ``(path, module or the error)``.

    Packages inside it are imported as packages; plain subfolders are walked the
    same way. A file's folder goes on ``sys.path`` (its own helper modules), and
    it is imported under its file name, or under ``<folder>_<name>`` when another
    file took that name (``llama/compare.py``, ``mamba/compare.py``).
    """
    import os

    out: list[tuple[str, types.ModuleType | BaseException]] = []
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        if entry.name.startswith((".", "_")) and entry.name != "__init__.py" or entry.name == "__pycache__":
            continue
        path = entry.path
        try:
            if entry.is_dir():
                if os.path.exists(os.path.join(path, "__init__.py")):
                    out.append((path, import_target(path)))
                else:
                    out += import_folder(path)
                continue
            if not entry.name.endswith(".py"):
                continue
            here = os.path.dirname(os.path.abspath(path))
            if here not in sys.path:
                sys.path.insert(0, here)
            name = entry.name[:-3]
            have = sys.modules.get(name)
            if have is not None and os.path.abspath(getattr(have, "__file__", "") or "") != os.path.abspath(path):
                name = f"{os.path.basename(here)}_{name}"
            if name in sys.modules:
                out.append((path, sys.modules[name]))
                continue
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[name]
                raise
            out.append((path, module))
        except (Exception, SystemExit) as e:
            out.append((path, e))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(
        prog="expex", description="Prove the shape annotations of every @fn in a package."
    )
    ap.add_argument(
        "targets", nargs="+", help="module/package names, or paths to .py files or package folders"
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--x64", action="store_true", help="enable 64-bit floats before importing")
    args = ap.parse_args(argv)
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    reports: list[Report] = []
    sys.path.insert(0, os.getcwd())
    failed: list[str] = []
    for t in args.targets:
        if os.path.isdir(t) and not os.path.exists(os.path.join(t, "__init__.py")):
            for path, m in import_folder(t):  # not a package: each file in it
                if isinstance(m, BaseException):
                    failed.append(f"{path}: could not import it: {type(m).__name__}: {m}")
                else:
                    reports += prove(m)
            continue
        reports += prove(import_target(t))
    print(format_reports(reports, args.verbose))
    for line in failed:
        print(f"error: {line}")
    return 1 if failed or any(r.status in (FAILED, ERROR) for r in reports) else 0

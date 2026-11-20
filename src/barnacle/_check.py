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

from __future__ import annotations

import importlib
import inspect
import itertools
import math
import pkgutil
import random
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, overload

import numpy as np

import jax
from jax import export

from . import _rng
from ._errors import ShapeError, StageError
from ._expr import to_jax_constraint
from ._fn import REGISTRY, BoundFn, Fn
from ._jaxpr import iter_eqns, source_of
from ._spec import (
    DAny,
    DConst,
    DEtc,
    DExpr,
    DName,
    DVar,
    Env,
    Spec,
    StaticSpec,
    Unsynthesizable,
)

PROVED = "proved"
SPOT = "spot-checked"
FAILED = "failed"
ERROR = "error"
UNCHECKED = "unchecked"

_RANK = {PROVED: 0, SPOT: 1, UNCHECKED: 2, ERROR: 3, FAILED: 4}
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_MAX_CASES = 24


@dataclass
class Report:
    """Outcome of proving one function."""

    fn: Fn[Any, Any]
    status: str
    detail: str = ""
    cases: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if nothing is wrong (proved or spot-checked)."""
        return self.status in (PROVED, SPOT)

    @property
    def name(self) -> str:
        mod = getattr(self.fn.raw, "__module__", None)
        return f"{mod}.{self.fn.name}" if mod else self.fn.name

    def raise_if_failed(self) -> None:
        if self.status in (FAILED, ERROR):
            raise AssertionError(str(self))

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
        return "\n".join(out)


def _indent(s: str, by: str = "    ") -> str:
    return "\n".join(by + line if line else line for line in s.splitlines())


class _Ctx:
    """Supplies sizes for abstract inputs: symbolic or concrete."""

    def __init__(
        self,
        scope: Any | None,
        sizes: dict[str, int] | None,
        var_rank: dict[str, int],
        preset: dict[str, int] | None = None,
    ):
        self.scope = scope
        self.sizes = sizes
        self.var_rank = var_rank
        self.dims: dict[str, Any] = dict(preset or {})
        self.origin: dict[str, str] = {}
        self.vars: dict[str, tuple[Any, ...]] = {}
        self._anon = 0
        self.need_x64 = False
        self.legacy_keys = False

    def _sym(self, name: str) -> Any:
        if self.scope is not None:
            return export.symbolic_shape(name, scope=self.scope)[0]
        assert self.sizes is not None
        if name not in self.sizes:
            self.sizes[name] = _PRIMES[len(self.sizes) % len(_PRIMES)]
        return self.sizes[name]

    def dim(self, name: str) -> Any:
        if name not in self.dims:
            self.dims[name] = self._sym(name)
        return self.dims[name]

    def fresh(self) -> Any:
        self._anon += 1
        return self._sym(f"_a{self._anon}")

    def var(self, name: str) -> tuple[Any, ...]:
        if name not in self.vars:
            self.vars[name] = tuple(self._sym(f"{name}_{i}") for i in range(self.var_rank.get(name, 1)))
        return self.vars[name]

    def shape(self, dims: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for d in dims:
            if isinstance(d, DName):
                out.append(self.dim(d.name))
            elif isinstance(d, DConst):
                out.append(d.value)
            elif isinstance(d, DAny):
                out.append(self.fresh())
            elif isinstance(d, DVar):
                out.extend(self.var(d.name))
            elif isinstance(d, DEtc):
                out.append(self.fresh())
            elif isinstance(d, DExpr):
                v = d.expr({n: self.dim(n) for n in d.expr.names})
                if isinstance(v, (int, np.integer)) and v < 0:
                    raise _BadSizes(f"`{d}` is negative")
                out.append(v)
        return tuple(out)

    def dtype(self, dt: Any) -> Any:
        d = dt.example
        if d.itemsize * (1 if d.kind != "c" else 0.5) >= 8 and not jax.config.jax_enable_x64:
            self.need_x64 = True
        return d

    def key(self, shape: tuple[Any, ...]) -> Any:
        if self.legacy_keys:
            return jax.ShapeDtypeStruct(shape + (2,), np.uint32)
        return jax.ShapeDtypeStruct(shape, jax.random.key(0).dtype)


class _BadSizes(Exception):
    pass


class _Unchecked(Exception):
    pass


_ABSTRACT = object()


def _choices(f: Fn[Any, Any], examples: dict[str, Any]) -> dict[str, list[Any]]:
    """For each parameter: ``[_ABSTRACT]`` or the list of concrete values to try."""
    for k in examples:
        if k not in f.params:
            raise TypeError(f"{f.name}: examples given for unknown parameter `{k}`")
    out: dict[str, list[Any]] = {}
    for name, p in f.params.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in examples:
            ex = examples[name]
            out[name] = list(ex) if isinstance(ex, (list, tuple)) else [ex]
            if not out[name]:
                raise _Unchecked(f"`examples` for `{name}` is empty.")
        elif isinstance(p.spec, StaticSpec) or not p.spec.leafy:
            if p.default is inspect.Parameter.empty:
                if name in ("self", "cls") and name == next(iter(f.params)):
                    raise _Unchecked(
                        "it is a method, so it needs an instance: "
                        "prove it with barnacle.prove(instance.method)."
                    )
                what = "Static parameter" if isinstance(p.spec, StaticSpec) else "parameter"
                why = "" if isinstance(p.spec, StaticSpec) else " and no shape annotation"
                raise _Unchecked(
                    f"{what} `{name}` has no default{why}, so the prover cannot invent "
                    f'a value for it. Add @fn(examples={{"{name}": [...]}}).'
                )
            # otherwise: leave it out and let the default apply
        else:
            out[name] = [_ABSTRACT]
    return out


def _trace(
    f: Fn[Any, Any],
    fixed: dict[str, Any],
    abstract: list[str],
    ctx: _Ctx,
) -> Any:
    """Trace ``f`` once on abstract inputs and return the jaxpr.

    Raises whatever the function raises, including ``ShapeError`` from its own
    (and nested) annotation checks.
    """
    arguments: dict[str, Any] = dict(fixed)
    for name in abstract:
        arguments[name] = f.params[name].spec.abstract(ctx, name)

    leaves: list[Any] = []

    def collect(spec: Spec, v: Any, path: str) -> Any:
        leaves.append(v)
        return v

    abstract_args = {n: arguments[n] for n in abstract}
    f._walk(abstract_args, collect)
    leaves.append(ctx.key(()))  # a key, in case anything inside draws random numbers

    def g(*lv: Any) -> Any:
        it = iter(lv[:-1])
        new = dict(arguments)
        new.update(f._walk(abstract_args, lambda s, x, p: next(it)))
        ordered = {n: new[n] for n in f.params if n in new}
        ba = inspect.BoundArguments(f._sig, ordered)  # type: ignore[arg-type]
        env = Env()
        for n, v in ctx.dims.items():
            env.dims[n] = v
            env.origin[n] = "the declared inputs"
        with _rng.rng(lv[-1]):
            f._run(ba.args, ba.kwargs, env, force=True)
        return ()

    if ctx.need_x64:
        with jax.enable_x64(True):
            return jax.make_jaxpr(g)(*leaves)
    return jax.make_jaxpr(g)(*leaves)


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


def _statics(f: Fn[Any, Any], fixed: dict[str, Any]) -> dict[str, int]:
    """Values of the Static int parameters that dimension expressions refer to."""
    out = {}
    for n in f.static_refs:
        v = fixed[n] if n in fixed else f.params[n].default
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out[n] = int(v)
    return out


def _jax_constraints(f: Fn[Any, Any], statics: dict[str, int]) -> tuple[list[str], list[str]]:
    """``where`` constraints in jax.export syntax, and those that cannot be expressed."""
    good, dropped = [], []
    inputs = {n for n in f.input_names if not n.startswith("*")} | statics.keys()
    for w in f.where:
        if not w.names <= inputs:
            continue
        if w.names <= statics.keys():
            continue  # no free sizes: checked directly when the function is called
        j = to_jax_constraint(w, statics)
        try:
            if j is None:
                raise ValueError
            export.SymbolicScope([j])
            good.append(j)
        except Exception:
            dropped.append(w.src)
    return good, dropped


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
            sizes = {n: _PRIMES[(i + off) % len(_PRIMES)] for i, n in enumerate(names)}
        else:
            sizes = {n: rng.randint(1, 12) for n in names}
        try:
            env = {**sizes, **statics}
            if all(bool(w(env)) for w in f.where if w.names <= env.keys()):
                return sizes
        except Exception:
            continue
    return None


def _summ(e: BaseException, limit: int = 6) -> str:
    lines = str(e).strip().splitlines() or [""]
    text = "\n".join(lines[:limit]) + ("\n..." if len(lines) > limit else "")
    return f"{type(e).__name__}: {text}"


def _full(e: BaseException) -> str:
    notes = [n for n in getattr(e, "__notes__", ()) if n.startswith("barnacle:")]
    return "\n".join([str(e).strip(), *notes])


@overload
def prove(target: "Fn[Any, Any] | BoundFn[Any, Any]", examples: dict[str, Any] | None = None) -> Report: ...
@overload
def prove(target: "str | types.ModuleType", examples: None = None) -> list[Report]: ...
def prove(target: Any, examples: dict[str, Any] | None = None) -> Any:
    """Prove the shape annotations of an ``@fn`` function for all sizes.

    ``target`` may be an ``@fn`` function (returns a :class:`Report`), or a
    module, package or dotted name (returns a list of reports, one for every
    ``@fn`` defined inside it).
    """
    if isinstance(target, BoundFn):
        ex = {"self": [target.obj], **(examples or {})}
        first = next(iter(target.fn.params), None)
        if first and first != "self":
            ex[first] = ex.pop("self")
        return _prove_fn(target.fn, ex)
    if isinstance(target, Fn):
        return _prove_fn(target, examples)
    return [_prove_fn(f, None) for f in collect(target)]


def collect(target: Any) -> list[Fn[Any, Any]]:
    """All provable ``@fn`` functions defined in a module/package (by name or object)."""
    if isinstance(target, str):
        target = importlib.import_module(target)
    if not isinstance(target, types.ModuleType):
        raise TypeError(f"cannot prove {target!r}: expected an @fn function, a module or a name")
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
    ex = {**f.examples, **(examples or {})}
    try:
        choices = _choices(f, ex)
    except _Unchecked as u:
        return Report(f, UNCHECKED, str(u))

    abstract = [n for n, c in choices.items() if c == [_ABSTRACT]]
    fixed_names = [n for n in choices if n not in abstract]
    var_names = sorted(n[1:] for n in f.input_names if n.startswith("*"))
    ranks: Iterable[tuple[int, ...]] = itertools.product(*[(0, 1, 2)] * len(var_names))
    combos = itertools.product(*[choices[n] for n in fixed_names])
    cases = list(itertools.islice(itertools.product(combos, ranks), _MAX_CASES))
    dropped: list[str] = []

    worst = Report(f, PROVED, cases=len(cases))
    for combo, rank in cases:
        fixed = dict(zip(fixed_names, combo))
        var_rank = dict(zip(var_names, rank))
        statics = _statics(f, fixed)
        constraints, dropped = _jax_constraints(f, statics)
        r = _prove_case(f, fixed, abstract, var_rank, constraints, statics)
        label = _case_label(fixed, var_rank)
        if label and r.status != PROVED:
            r.detail = f"with {label}:\n{r.detail}"
        if _RANK[r.status] > _RANK[worst.status]:
            r.cases = len(cases)
            worst = r
            if r.status == FAILED:
                break
    if dropped and worst.status == PROVED:
        worst.notes.append(
            "constraints not expressible symbolically were only checked at run time: "
            + ", ".join(f"`{d}`" for d in dropped)
        )
    return worst


def _case_label(fixed: dict[str, Any], var_rank: dict[str, int]) -> str:
    parts = [f"{k}={v!r}"[:60] for k, v in fixed.items()]
    parts += [f"*{k} of rank {v}" for k, v in var_rank.items()]
    return ", ".join(parts)


def _attempt(
    f: Fn[Any, Any],
    fixed: dict[str, Any],
    abstract: list[str],
    make_ctx: Any,
) -> tuple[Any, _Ctx, BaseException | None]:
    """Trace with new-style keys, retrying with legacy keys if that fails."""
    first: tuple[_Ctx, BaseException] | None = None
    for legacy in (False, True):
        ctx = make_ctx()
        ctx.legacy_keys = legacy
        try:
            return _trace(f, fixed, abstract, ctx), ctx, None
        except (Unsynthesizable, _BadSizes):
            raise
        except Exception as e:
            if first is None:
                first = (ctx, e)
            if isinstance(e, (ShapeError, StageError)) or not _mentions_key(f):
                break
    assert first is not None
    return None, first[0], first[1]


def _at(sizes: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sizes.items() if not k.startswith("_")) or "no free sizes"


def _barnacle_notes(e: BaseException) -> str:
    return "".join(
        "\n" + _indent(n) for n in getattr(e, "__notes__", ()) if n.startswith("barnacle:")
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
        jaxpr, _, sym_err = _attempt(
            f,
            fixed,
            abstract,
            lambda: _Ctx(export.SymbolicScope(constraints), None, var_rank, statics),
        )
    except Unsynthesizable as u:
        return Report(
            f, UNCHECKED,
            f"the prover cannot invent a value for `{u}`. "
            f"Add @fn(examples={{...}}) for that parameter.",
        )  # fmt: skip
    unproven = _unproven_reshapes(jaxpr) if jaxpr is not None else []

    # 2. Concrete sizes: confirms failures, and validates what symbolic tracing
    #    cannot (JAX defers some checks when sizes are symbolic).
    used: list[dict[str, int]] = []
    for attempt in range(4):
        sizes = _concrete_sizes(f, attempt, statics)
        if sizes is None:
            continue
        try:
            _, ctx, err = _attempt(
                f, fixed, abstract, lambda s=sizes: _Ctx(None, dict(s), var_rank, statics)
            )
        except (_BadSizes, Unsynthesizable):
            continue
        assert ctx.sizes is not None
        if err is None:
            used.append(dict(ctx.sizes))
            continue
        at = _at(ctx.sizes)
        if isinstance(err, (ShapeError, StageError)):
            detail = _full(sym_err) if isinstance(sym_err, ShapeError) else _full(err)
            return Report(f, FAILED, detail, notes=[f"confirmed at sizes {at}."])
        hint = ""
        if sym_err is None:
            hint = (
                "\nThe function is fine for symbolic sizes but not for these. If it is not "
                "meant to support them, say so: @fn(where=\"...\")."
            )
        return Report(
            f, ERROR,
            f"the function raised while being traced on abstract inputs ({at}):\n"
            + _indent(_summ(err, 12)) + _barnacle_notes(err) + hint,
        )  # fmt: skip

    if not used:
        why = f"\nsymbolic trace failed with: {_summ(sym_err)}" if sym_err else ""
        return Report(
            f, UNCHECKED, "could not find sizes satisfying the `where` constraints." + why
        )
    if sym_err is None and not unproven:
        return Report(f, PROVED)
    at = "; ".join(map(_at, used))
    reason = _summ(sym_err) if sym_err is not None else "\n".join(unproven)
    return Report(
        f, SPOT,
        f"holds at sizes {at}, but could not be proved for all sizes:\n" + _indent(reason),
    )  # fmt: skip


def _mentions_key(f: Fn[Any, Any]) -> bool:
    from ._spec import KeySpec

    found = False

    def visit(spec: Spec, v: Any, path: str) -> Any:
        nonlocal found
        found = found or isinstance(spec, KeySpec)
        return v

    try:
        ctx = _Ctx(None, {}, {})
        for p in f.params.values():
            if p.spec.leafy and not isinstance(p.spec, StaticSpec):
                try:
                    p.spec.walk(p.spec.abstract(ctx, p.name), p.name, visit)
                except Exception:
                    pass
    except Exception:
        pass
    return found


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


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(
        prog="barnacle", description="Prove the shape annotations of every @fn in a package."
    )
    ap.add_argument("targets", nargs="+", help="module/package names, or paths to .py files")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    reports: list[Report] = []
    sys.path.insert(0, os.getcwd())
    for t in args.targets:
        if t.endswith(".py") or os.sep in t:
            path = os.path.abspath(t)
            name = os.path.splitext(os.path.basename(path))[0]
            sys.path.insert(0, os.path.dirname(path))
            mod = importlib.import_module(name)
        else:
            mod = importlib.import_module(t)
        reports += prove(mod)
    print(format_reports(reports, args.verbose))
    return 1 if any(r.status in (FAILED, ERROR) for r in reports) else 0

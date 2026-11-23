"""``expex.debug.find_nan``: the first operation that turns NaN-free values into NaN.

The function is traced on its arguments and the jaxpr replayed one operation
at a time. At the first operation whose outputs have a NaN its inputs do not,
the replay looks inside control flow (the branch taken, the loop iteration)
and nested calls for the innermost operation, and reports its source line,
the path to it and statistics of its inputs.
"""

import linecache
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.core import eval_jaxpr

from ..annotations.dims import describe
from .trace import make_jaxpr, source_of, sub_jaxprs


def _is_float(x: Any) -> bool:
    try:
        return bool(jnp.issubdtype(x.dtype, jnp.inexact))
    except Exception:
        return False


def _has_nan(xs: Any) -> bool:
    return any(_is_float(x) and bool(jnp.isnan(x).any()) for x in xs)


def _has_inf(xs: Any) -> bool:
    return any(_is_float(x) and bool(jnp.isinf(x).any()) for x in xs)


def _stats(x: Any) -> str:
    d = describe(x)
    if not _is_float(x) or getattr(x, "size", 0) == 0:
        if getattr(x, "size", 0) == 1:
            return f"{d}  value={np.asarray(x).item()}"
        return d
    a = np.asarray(x)
    if np.iscomplexobj(a):
        a = np.abs(a)
    fin = a[np.isfinite(a)]
    parts = []
    if a.size == 1:
        parts.append(f"value={a.item():.6g}")
    elif fin.size:
        parts.append(f"min={fin.min():.6g}  max={fin.max():.6g}")
    counts = [
        (int(np.isnan(a).sum()), "nan"),
        (int(np.isposinf(a).sum()), "+inf"),
        (int(np.isneginf(a).sum()), "-inf"),
        (int((a == 0).sum()), "zero"),
        (int((fin < 0).sum()), "negative"),
    ]
    flags = ", ".join(f"{n} {w}" for n, w in counts if n and a.size > 1)
    if flags:
        parts.append(f"({flags}, of {a.size})")
    return f"{d}  " + "  ".join(parts)


@dataclass
class NanReport:
    """Where a NaN first appeared."""

    kind: str  # "op" or "input"
    op: str
    source: str
    line: str
    path: list[str]
    inputs: list[str]
    outputs: list[str]
    infs: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.kind == "input":
            return "NaN is already present in the inputs:\n" + "\n".join(
                f"    {s}" for s in self.inputs
            )
        out = [f"NaN first appears at {self.source}"]
        if self.line:
            out.append(f"    > {self.line}")
        out.append(f"  operation: {self.op}")
        if self.path:
            out.append(f"  inside:    {' > '.join(self.path)}")
        out.append("  inputs:")
        out += [f"    {s}" for s in self.inputs]
        out.append("  output:")
        out += [f"    {s}" for s in self.outputs]
        if self.infs:
            out.append("  infinities appeared earlier at:")
            out += [f"    {s}" for s in self.infs]
        return "\n".join(out)


class _Found(Exception):
    def __init__(self, report: NanReport):
        self.report = report


def _line_of(source: str) -> str:
    try:
        fname, rest = source.rsplit(":", 1)
        return linecache.getline(fname, int(rest.split()[0])).strip()
    except Exception:
        return ""


def _bind(eqn: Any, invals: list[Any]) -> list[Any]:
    r = eqn.primitive.get_bind_params(eqn.params)
    # older JAX returns (subfuns, params); newer returns the params alone
    subfuns: Any
    params: Any
    subfuns, params = r if isinstance(r, tuple) and len(r) == 2 else ((), r)
    ctx = getattr(getattr(eqn, "ctx", None), "manager", None)
    if ctx is not None:
        with ctx:
            out = eqn.primitive.bind(*subfuns, *invals, **params)
    else:
        out = eqn.primitive.bind(*subfuns, *invals, **params)
    return list(out) if eqn.primitive.multiple_results else [out]


class _Interp:
    def __init__(self) -> None:
        self.infs: list[str] = []

    def run(self, jaxpr: Any, args: list[Any], path: list[str]) -> list[Any]:
        """Evaluate one jaxpr op by op; raise ``_Found`` at the first NaN-producing op."""
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        env: dict[Any, Any] = {}

        def read(v: Any) -> Any:
            return v.val if hasattr(v, "val") else env[v]

        for v, a in zip(_invars(jaxpr), args):
            env[v] = a
        for eqn in jaxpr.eqns:
            invals = [read(v) for v in eqn.invars]
            outs = _bind(eqn, invals)
            if _has_nan(outs) and not _has_nan(invals):
                self.dive(eqn, invals, path)  # raises if it can localise further
                raise _Found(self.report(eqn, invals, outs, path))
            if len(self.infs) < 3 and _has_inf(outs) and not _has_inf(invals):
                where = source_of(eqn)
                self.infs.append(f"{where}  ({eqn.primitive.name})  > {_line_of(where)}")
            for v, o in zip(eqn.outvars, outs):
                env[v] = o
        return [read(v) for v in jaxpr.outvars]

    def report(self, eqn: Any, invals: list[Any], outs: list[Any], path: list[str]) -> NanReport:
        where = source_of(eqn)
        return NanReport(
            "op", eqn.primitive.name, where, _line_of(where), list(path),
            [f"#{i}: {_stats(x)}" for i, x in enumerate(invals)],
            [_stats(x) for x in outs],
            list(self.infs),
        )  # fmt: skip

    def dive(self, eqn: Any, invals: list[Any], path: list[str]) -> None:
        """Look inside control flow / nested calls for the offending operation."""
        name = eqn.primitive.name
        p = eqn.params
        try:
            if name == "cond":
                idx = int(invals[0])
                self.run(p["branches"][idx], invals[1:], path + [f"cond branch {idx}"])
            elif name == "while":
                self.dive_while(p, invals, path)
            elif name == "scan":
                self.dive_scan(p, invals, path)
            else:
                subs = list(sub_jaxprs(p))
                if len(subs) == 1:
                    sub = subs[0][1]
                    if len(_invars(sub)) == len(invals):
                        label = p.get("name") if name in ("jit", "pjit") else name
                        self.run(sub, invals, path + [f"{label}"] if label else path)
        except _Found:
            raise
        except Exception:
            return  # unknown structure: report the enclosing operation instead

    def dive_while(self, p: dict[str, Any], invals: list[Any], path: list[str]) -> None:
        cn, tx = p["cond_nconsts"], p["body_nconsts"]
        cconsts, bconsts, carry = invals[:cn], invals[cn : cn + tx], invals[cn + tx :]
        cond_j, body_j = p["cond_jaxpr"], p["body_jaxpr"]
        body = jax.jit(lambda c, x: list(eval_jaxpr(_jp(body_j), [], *c, *x)))
        cond = jax.jit(lambda c, x: list(eval_jaxpr(_jp(cond_j), [], *c, *x))[0])
        it = 0
        while bool(cond(cconsts, carry)):
            new = body(bconsts, carry)
            if _has_nan(new) and not _has_nan(carry):
                self.run(body_j, [*bconsts, *carry], path + [f"while iteration {it}"])
            carry = new
            it += 1

    def dive_scan(self, p: dict[str, Any], invals: list[Any], path: list[str]) -> None:
        body_j = p["jaxpr"]
        length, reverse = p["length"], p.get("reverse", False)
        if "num_consts" in p:  # older JAX
            nc, ncar = p["num_consts"], p["num_carry"]
        else:  # newer JAX: a flat-tree of (consts, carry, xs)
            parts = p["ft_in"].unpack()
            nc, ncar = len(parts[0]), len(parts[1])
            if nc + ncar + len(parts[2]) != len(invals):
                return
        consts, carry, xs = invals[:nc], invals[nc : nc + ncar], invals[nc + ncar :]
        if reverse:
            xs = [x[::-1] for x in xs]

        def step(c: Any, x: Any) -> Any:
            out = list(eval_jaxpr(_jp(body_j), [], *consts, *c, *x))
            bad = jnp.zeros((), bool)
            for o in out:
                if _is_float(o):
                    bad = bad | jnp.isnan(o).any()
            return tuple(out[:ncar]), bad

        _, flags = lax.scan(step, tuple(carry), tuple(xs), length=length)
        flags = np.asarray(flags)
        if not flags.any():
            return
        first = int(flags.argmax())
        if first:
            done, _ = lax.scan(step, tuple(carry), tuple(x[:first] for x in xs), length=first)
            carry = list(done)
        it = length - 1 - first if reverse else first
        self.run(body_j, [*consts, *carry, *[x[first] for x in xs]], path + [f"scan iteration {it}"])


def _invars(jaxpr: Any) -> list[Any]:
    if hasattr(jaxpr, "all_invars"):
        return list(jaxpr.all_invars)
    return list(getattr(jaxpr, "constvars", ())) + list(jaxpr.invars)


def _jp(j: Any) -> Any:
    return getattr(j, "jaxpr", j)


def find_nan(f: Callable[..., Any], *args: Any, **kwargs: Any) -> NanReport | None:
    """Find the first operation that turns NaN-free values into NaN.

    Returns ``None`` if the function's computation never produces a NaN for
    these arguments. Works on any JAX function, including ``jax.grad(f)``.
    """
    closed, flat = make_jaxpr(f, args, kwargs)
    bad = [f"argument leaf #{i}: {_stats(x)}" for i, x in enumerate(flat) if _has_nan([x])]
    if bad:
        return NanReport("input", "", "", "", [], bad, [])
    interp = _Interp()
    try:
        interp.run(closed.jaxpr, [*closed.consts, *flat], [])
    except _Found as e:
        return e.report
    return None

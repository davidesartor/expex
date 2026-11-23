"""``expex.annotate``: annotations for plain JAX code, read off a run on example inputs.

``annotate(f, *args, **kwargs)`` traces ``f`` (which stays as it is) to a
jaxpr and follows every axis of every input through it. Two axes that the
program joins (an elementwise op, a contraction, a batch dimension, a scan's
carry) are the same size; two axes that only happen to have the same length
in the example are not. Axes of other values are expressed in those sizes
when the dataflow says how (a pad or concatenate adds, a reshape splits or
merges), and matched by value otherwise; ``vary=True`` traces again with
every size changed, to tell coincidences apart.

The result prints as the annotated ``def`` line, and ``.diff()`` /
``.apply()`` turn it into an edit of the function's source file. Nothing here
runs when an annotated function is called.
"""

import ast
import dataclasses
import difflib
import inspect
import io
import itertools
import keyword
import pathlib
import re
import sys
import tokenize
from collections import defaultdict
from typing import Any, Callable, Iterator, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["annotate", "Annotation"]

# ------------------------------------------------------------------ sizes as polynomials


@dataclasses.dataclass(frozen=True)
class Div:
    """``num // den``, kept whole when it does not divide out."""

    num: "Size"
    den: "Size"

    def __str__(self) -> str:
        return f"{self.num.text(paren=True)} // {self.den.text(paren=True)}"


Mono = tuple[Any, ...]  # sorted atoms: names (str) and Div


def _atom_key(a: Any) -> tuple[int, str]:
    return (0, a) if isinstance(a, str) else (1, str(a))


@dataclasses.dataclass(frozen=True)
class Size:
    """A sum of integer multiples of products of names: ``2 * T + 1``, ``H * K``, ``D // heads``."""

    terms: tuple[tuple[Mono, int], ...] = ()

    @staticmethod
    def make(d: Mapping[Mono, int]) -> "Size":
        items = [(m, c) for m, c in d.items() if c != 0]
        items.sort(key=lambda mc: (-len(mc[0]), [_atom_key(a) for a in mc[0]]))
        return Size(tuple(items))

    @staticmethod
    def name(n: str) -> "Size":
        return Size((((n,), 1),))

    @staticmethod
    def const(c: int) -> "Size":
        return Size.make({(): int(c)})

    @property
    def value(self) -> int | None:
        """The number, if constant."""
        if not self.terms:
            return 0
        if len(self.terms) == 1 and self.terms[0][0] == ():
            return self.terms[0][1]
        return None

    def __add__(self, o: "Size | int") -> "Size":
        o = Size.const(o) if isinstance(o, int) else o
        d: dict[Mono, int] = defaultdict(int)
        for m, c in self.terms + o.terms:
            d[m] += c
        return Size.make(d)

    def __mul__(self, o: "Size | int") -> "Size":
        o = Size.const(o) if isinstance(o, int) else o
        d: dict[Mono, int] = defaultdict(int)
        for (m1, c1), (m2, c2) in itertools.product(self.terms, o.terms):
            d[tuple(sorted(m1 + m2, key=_atom_key))] += c1 * c2
        return Size.make(d)

    def __floordiv__(self, o: "Size | int") -> "Size":
        o = Size.const(o) if isinstance(o, int) else o
        c = o.value
        if c is not None and c != 0:
            if all(k % c == 0 for _, k in self.terms):
                return Size.make({m: k // c for m, k in self.terms})
            if (v := self.value) is not None:
                return Size.const(v // c)
        elif len(o.terms) == 1 and o.terms[0][1] == 1:
            den = list(o.terms[0][0])
            out: dict[Mono, int] = {}
            for m, k in self.terms:
                rest = list(m)
                for a in den:
                    if a not in rest:
                        break
                    rest.remove(a)
                else:
                    out[tuple(rest)] = k
                    continue
                break
            else:
                return Size.make(out)
        return Size.make({(Div(self, o),): 1})

    def eval(self, env: Mapping[str, int]) -> int:
        total = 0
        for m, c in self.terms:
            p = c
            for a in m:
                p *= env[a] if isinstance(a, str) else a.num.eval(env) // a.den.eval(env)
            total += p
        return total

    def names(self) -> set[str]:
        out: set[str] = set()
        for m, _ in self.terms:
            for a in m:
                out |= {a} if isinstance(a, str) else a.num.names() | a.den.names()
        return out

    def divs(self) -> Iterator[Div]:
        for m, _ in self.terms:
            for a in m:
                if isinstance(a, Div):
                    yield from a.num.divs()
                    yield a

    def text(self, paren: bool = False) -> str:
        s = str(self)
        single = len(self.terms) <= 1 and (not self.terms or (len(self.terms[0][0]) <= 1 and self.terms[0][1] >= 0))
        single = single and not (self.terms and self.terms[0][1] not in (1,) and self.terms[0][0])
        return f"({s})" if paren and not single else s

    def __str__(self) -> str:
        out = ""
        for i, (m, c) in enumerate(self.terms):
            k = abs(c)
            parts = [str(a) if isinstance(a, str) else f"({a})" if len(m) > 1 or k != 1 else str(a) for a in m]
            if k != 1 or not m:
                parts.insert(0, str(k))
            s = " * ".join(parts)
            out = ("-" + s if c < 0 else s) if i == 0 else out + (" - " if c < 0 else " + ") + s
        return out or "0"


# ------------------------------------------------------------------ dataflow over a jaxpr

_SEQ, _CONTRACT, _BATCH, _REDUCE, _SCAN, _GATHER = "seq", "contracted", "dot_batch", "reduced", "scanned", "gathered"
_SPLIT_SRC, _SPLIT_HEAD, _SPLIT_REST = "split_src", "split_head", "split_rest"


def _is_literal(v: Any) -> bool:
    return type(v).__name__ == "Literal"


def _sub_jaxprs(params: Mapping[str, Any]) -> Iterator[Any]:
    def visit(v: Any) -> Iterator[Any]:
        if hasattr(v, "eqns"):
            yield v
        elif hasattr(v, "jaxpr") and hasattr(v.jaxpr, "eqns"):
            yield v.jaxpr
        elif isinstance(v, (tuple, list)):
            for x in v:
                yield from visit(x)

    for v in params.values():
        yield from visit(v)


class Flow:
    """Axes of a jaxpr's values, joined where the program says they are the same size.

    A node is one axis of one value. ``facts`` record what the program does
    with an axis (contracted, scanned over, ...), for naming; ``rels`` record
    sizes derived from others (``out = in + 1``, ``in = H * K``).
    """

    def __init__(self, closed: Any):
        self.parent: list[int] = []
        self.size: list[int] = []
        self.index: dict[int, list[int]] = {}
        self._keep: list[Any] = []
        self.facts: dict[int, set[str]] = defaultdict(set)
        self.rels: list[tuple[Any, ...]] = []
        self.reshape_notes: list[str] = []
        self.walk(closed.jaxpr)

    # -- nodes
    def new(self, size: int) -> int:
        self.parent.append(len(self.parent))
        self.size.append(int(size))
        return len(self.parent) - 1

    def axes(self, v: Any) -> list[int]:
        aval = getattr(v, "aval", None)
        shape = getattr(aval, "shape", None)
        if shape is None:
            return []
        if id(v) not in self.index:
            self._keep.append(v)
            self.index[id(v)] = [self.new(int(s)) for s in shape]
        return self.index[id(v)]

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        if self.size[a] != self.size[b] or self.size[a] == 1:
            return
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def link(self, a: Any, b: Any) -> None:
        """All axes of two values of the same shape."""
        xa, xb = self.axes(a), self.axes(b)
        if len(xa) == len(xb):
            for p, q in zip(xa, xb):
                self.union(p, q)

    def mark(self, node: int, fact: str) -> None:
        self.facts[node].add(fact)

    # -- equations
    def walk(self, jaxpr: Any) -> None:
        for eqn in jaxpr.eqns:
            name = eqn.primitive.name
            handler = getattr(self, "_" + name.replace("-", "_"), None)
            if handler is not None:
                handler(eqn, eqn.params)
            elif name.startswith("reduce_") or name in ("argmax", "argmin", "reduce"):
                self._reduce(eqn, eqn.params)
            elif name.startswith("scatter"):
                self._scatter(eqn, eqn.params)
            elif any(True for _ in _sub_jaxprs(eqn.params)):
                self._call(eqn, eqn.params)
            else:
                self._elementwise(eqn)

    def _elementwise(self, eqn: Any) -> None:
        vals = [v for v in [*eqn.invars, *eqn.outvars] if getattr(getattr(v, "aval", None), "shape", ())]
        if vals and all(v.aval.shape == vals[0].aval.shape for v in vals):
            for v in vals[1:]:
                self.link(vals[0], v)

    def _call(self, eqn: Any, params: Mapping[str, Any]) -> None:
        for sub in _sub_jaxprs(params):
            if len(sub.invars) == len(eqn.invars) and len(sub.outvars) == len(eqn.outvars):
                for a, b in zip(eqn.invars, sub.invars):
                    self.link(a, b)
                for a, b in zip(eqn.outvars, sub.outvars):
                    self.link(a, b)
            self.walk(sub)

    def _transpose(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        for i, p in enumerate(params["permutation"]):
            self.union(out[i], x[p])

    def _broadcast_in_dim(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        for i, d in enumerate(params["broadcast_dimensions"]):
            self.union(x[i], out[d])

    def _squeeze(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        kept = [a for i, a in enumerate(x) if i not in set(params["dimensions"])]
        for a, b in zip(kept, out):
            self.union(a, b)

    def _reduce(self, eqn: Any, params: Mapping[str, Any]) -> None:
        axes = params.get("axes", params.get("dimensions"))
        if axes is None:
            return self._elementwise(eqn)
        axes = set(axes)
        for v in eqn.invars[: len(eqn.outvars)]:
            x = self.axes(v)
            for i in axes:
                if i < len(x):
                    self.mark(x[i], _REDUCE)
            kept = [a for i, a in enumerate(x) if i not in axes]
            for o in eqn.outvars:
                out = self.axes(o)
                if len(out) == len(kept):
                    for a, b in zip(kept, out):
                        self.union(a, b)

    def _dot_general(self, eqn: Any, params: Mapping[str, Any]) -> None:
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        lhs, rhs, out = self.axes(eqn.invars[0]), self.axes(eqn.invars[1]), self.axes(eqn.outvars[0])
        for a, b in zip(lc, rc):
            self.union(lhs[a], rhs[b])
            self.mark(lhs[a], _CONTRACT)
            self.mark(rhs[b], _CONTRACT)
        for a, b in zip(lb, rb):
            self.union(lhs[a], rhs[b])
            self.mark(lhs[a], _BATCH)
            self.mark(rhs[b], _BATCH)
        free_l = [lhs[i] for i in range(len(lhs)) if i not in lc and i not in lb]
        free_r = [rhs[i] for i in range(len(rhs)) if i not in rc and i not in rb]
        for a, b in zip([lhs[i] for i in lb] + free_l + free_r, out):
            self.union(a, b)

    def _reshape(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        if params.get("dimensions") not in (None, ()):
            return
        a, b = [self.size[n] for n in x], [self.size[n] for n in out]
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i] == 1:
                i += 1
                continue
            if b[j] == 1:
                j += 1
                continue
            ii, jj, pa, pb = [i], [j], a[i], b[j]
            i, j = i + 1, j + 1
            while pa != pb:
                if pa < pb and i < len(a):
                    pa *= a[i]
                    ii.append(i)
                    i += 1
                elif pb < pa and j < len(b):
                    pb *= b[j]
                    jj.append(j)
                    j += 1
                else:
                    return
            ins = [x[k] for k in ii if a[k] != 1]
            outs = [out[k] for k in jj if b[k] != 1]
            if len(ins) == 1 and len(outs) == 1:
                self.union(ins[0], outs[0])
            elif len(ins) == 1:
                self.rels.append(("split", ins[0], outs))
                self.mark(ins[0], _SPLIT_SRC)
                self.mark(outs[0], _SPLIT_HEAD)
                for o in outs[1:]:
                    self.mark(o, _SPLIT_REST)
            elif len(outs) == 1:
                self.rels.append(("prod", outs[0], ins))
            else:
                self.reshape_notes.append(f"a reshape from {tuple(a)} to {tuple(b)} regroups axes; their sizes are matched by value")

    def _concatenate(self, eqn: Any, params: Mapping[str, Any]) -> None:
        d = params["dimension"]
        out = self.axes(eqn.outvars[0])
        parts: list[Any] = []
        for v in eqn.invars:
            x = self.axes(v)
            for i, (p, q) in enumerate(zip(x, out)):
                if i != d:
                    self.union(p, q)
            parts.append(x[d])
            self.mark(x[d], _SEQ)
        self.rels.append(("sum", out[d], parts))

    def _pad(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        for i, (lo, hi, interior) in enumerate(params["padding_config"]):
            if lo == hi == interior == 0:
                self.union(x[i], out[i])
            elif interior == 0:
                self.rels.append(("sum", out[i], [x[i], ("const", int(lo) + int(hi))]))
                self.mark(x[i], _SEQ)

    def _slice(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        strides = params.get("strides") or (1,) * len(x)
        for i, (s, e, st) in enumerate(zip(params["start_indices"], params["limit_indices"], strides)):
            if st not in (None, 1):
                continue
            if s == 0 and e == self.size[x[i]]:
                self.union(x[i], out[i])
            else:
                self.rels.append(("sum", out[i], [x[i], ("const", int(e) - int(s) - self.size[x[i]])]))
                self.mark(x[i], _SEQ)

    def _split(self, eqn: Any, params: Mapping[str, Any]) -> None:
        d = params["axis"]
        x = self.axes(eqn.invars[0])
        for o in eqn.outvars:
            out = self.axes(o)
            for i, (p, q) in enumerate(zip(x, out)):
                if i != d:
                    self.union(p, q)
        outs = [self.axes(o)[d] for o in eqn.outvars]
        if len({self.size[o] for o in outs}) == 1 and len(outs) > 1:
            for o in outs:
                self.rels.append(("div", o, x[d], len(outs)))

    def _dynamic_slice(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x, out = self.axes(eqn.invars[0]), self.axes(eqn.outvars[0])
        for p, q in zip(x, out):
            self.union(p, q)  # only joins axes taken whole

    def _dynamic_update_slice(self, eqn: Any, params: Mapping[str, Any]) -> None:
        self.link(eqn.invars[0], eqn.outvars[0])
        x, upd = self.axes(eqn.invars[0]), self.axes(eqn.invars[1])
        for p, q in zip(x, upd):
            self.union(p, q)

    def _gather(self, eqn: Any, params: Mapping[str, Any]) -> None:
        dn = params["dimension_numbers"]
        x, idx, out = self.axes(eqn.invars[0]), self.axes(eqn.invars[1]), self.axes(eqn.outvars[0])
        sizes = params["slice_sizes"]
        skip = set(dn.collapsed_slice_dims) | set(getattr(dn, "operand_batching_dims", ()))
        offset_src = [d for d in range(len(x)) if d not in skip]
        for pos, d in zip(dn.offset_dims, offset_src):
            if sizes[d] == self.size[x[d]]:
                self.union(out[pos], x[d])
        batch_out = [i for i in range(len(out)) if i not in set(dn.offset_dims)]
        for pos, d in zip(batch_out, idx[:-1]):
            self.union(out[pos], d)
        for a, b in zip(getattr(dn, "operand_batching_dims", ()), getattr(dn, "start_indices_batching_dims", ())):
            self.union(x[a], idx[b])
        for d in dn.collapsed_slice_dims:
            if d in dn.start_index_map:
                self.mark(x[d], _GATHER)

    def _scatter(self, eqn: Any, params: Mapping[str, Any]) -> None:
        dn = params.get("dimension_numbers")
        self.link(eqn.invars[0], eqn.outvars[0])
        if dn is None or len(eqn.invars) < 3:
            return
        x, idx, upd = self.axes(eqn.invars[0]), self.axes(eqn.invars[1]), self.axes(eqn.invars[2])
        skip = set(dn.inserted_window_dims) | set(getattr(dn, "operand_batching_dims", ()))
        window_src = [d for d in range(len(x)) if d not in skip]
        for pos, d in zip(dn.update_window_dims, window_src):
            self.union(upd[pos], x[d])
        scatter_pos = [i for i in range(len(upd)) if i not in set(dn.update_window_dims)]
        for pos, d in zip(scatter_pos, idx[:-1]):
            self.union(upd[pos], d)

    def _top_k(self, eqn: Any, params: Mapping[str, Any]) -> None:
        x = self.axes(eqn.invars[0])
        for o in eqn.outvars:
            for p, q in zip(x[:-1], self.axes(o)[:-1]):
                self.union(p, q)

    def _conv_general_dilated(self, eqn: Any, params: Mapping[str, Any]) -> None:
        dn = params["dimension_numbers"]
        lhs, rhs, out = self.axes(eqn.invars[0]), self.axes(eqn.invars[1]), self.axes(eqn.outvars[0])
        ls, rs, os_ = dn.lhs_spec, dn.rhs_spec, dn.out_spec
        self.union(lhs[ls[0]], out[os_[0]])
        self.union(rhs[rs[0]], out[os_[1]])
        if params.get("feature_group_count", 1) == 1 and params.get("batch_group_count", 1) == 1:
            self.union(lhs[ls[1]], rhs[rs[1]])
            self.mark(lhs[ls[1]], _CONTRACT)
        strides = params.get("window_strides") or (1,) * (len(ls) - 2)
        pads = params.get("padding") or ((0, 0),) * (len(ls) - 2)
        for k, (a, b, c) in enumerate(zip(ls[2:], rs[2:], os_[2:])):
            if self.size[lhs[a]] == self.size[out[c]]:
                self.union(lhs[a], out[c])
            elif strides[k] == self.size[rhs[b]] and tuple(pads[k]) == (0, 0):
                self.rels.append(("div", out[c], lhs[a], int(strides[k])))

    def _cond(self, eqn: Any, params: Mapping[str, Any]) -> None:
        for br in params["branches"]:
            sub = br.jaxpr
            for a, b in zip(eqn.invars[1:], sub.invars):
                self.link(a, b)
            for a, b in zip(eqn.outvars, sub.outvars):
                self.link(a, b)
            self.walk(sub)

    def _while(self, eqn: Any, params: Mapping[str, Any]) -> None:
        cn, bn = params["cond_nconsts"], params["body_nconsts"]
        body, cond = params["body_jaxpr"].jaxpr, params["cond_jaxpr"].jaxpr
        cc, bc, init = eqn.invars[:cn], eqn.invars[cn : cn + bn], eqn.invars[cn + bn :]
        for a, b in zip(cc, cond.invars[:cn]):
            self.link(a, b)
        for a, b in zip(bc, body.invars[:bn]):
            self.link(a, b)
        for i, v in enumerate(init):
            for other in (body.invars[bn + i], body.outvars[i], eqn.outvars[i], cond.invars[cn + i]):
                self.link(v, other)
        self.walk(body)
        self.walk(cond)

    def _scan(self, eqn: Any, params: Mapping[str, Any]) -> None:
        body = params["jaxpr"].jaxpr
        if "num_consts" in params:
            nconst, ncarry = params["num_consts"], params["num_carry"]
        else:
            ft = params.get("ft_in")
            if ft is None or len(ft) != 3:
                return self._call(eqn, params)
            nconst, ncarry = len(ft[0]), len(ft[1])
        length = self.new(int(params["length"]))
        self.mark(length, _SCAN)
        ins = eqn.invars
        for a, b in zip(ins[: nconst + ncarry], body.invars[: nconst + ncarry]):
            self.link(a, b)
        xs = ins[nconst + ncarry :]
        for a, b in zip(xs, body.invars[nconst + ncarry :]):
            outer, inner = self.axes(a), self.axes(b)
            if outer:
                self.union(outer[0], length)
                self.mark(outer[0], _SCAN)
            for p, q in zip(outer[1:], inner):
                self.union(p, q)
        for i in range(ncarry):
            for other in (body.outvars[i], eqn.outvars[i]):
                self.link(body.invars[nconst + i], other)
        ys = eqn.outvars[ncarry:]
        inner_ys = list(body.outvars[ncarry:])
        kinds = list(params.get("ft_out", (None, ()))[1]) if len(inner_ys) != len(ys) else [None] * len(ys)
        for y, kind in zip(ys, kinds):
            out = self.axes(y)
            if out:
                self.union(out[0], length)
            if kind is not None and "Left" in repr(kind):  # an xs forwarded unchanged
                for x in xs:
                    if getattr(x, "aval", None) is not None and x.aval.shape == y.aval.shape:
                        self.link(x, y)
                        break
                continue
            if inner_ys:
                inner = self.axes(inner_ys.pop(0))
                for p, q in zip(out[1:], inner):
                    self.union(p, q)
        self.walk(body)


# ------------------------------------------------------------------ running the function


class _Watch:
    """Records the traced values assigned by a function's statements, without settrace.

    Uses ``sys.monitoring`` line events on the one code object; switched off on exit.
    """

    def __init__(self, code: Any, stmt_of_line: Mapping[int, tuple[int, str]]):
        self.code = code
        self.stmt_of_line = stmt_of_line
        self.found: dict[tuple[int, str], list[Any]] = {}
        self._last: dict[str, Any] = {}
        self._prev: int | None = None
        self._frame: Any = None
        self.tool: int | None = None

    def _see(self, frame: Any, line: int | None) -> None:
        if frame.f_code is not self.code:
            return
        if self._frame is None:
            self._frame = frame
        elif frame is not self._frame:
            return
        stmt = self.stmt_of_line.get(self._prev) if self._prev is not None else None
        for k, v in list(frame.f_locals.items()):
            if self._last.get(k) is v:
                continue
            self._last[k] = v
            if stmt is not None and stmt[1] == k and isinstance(v, jax.core.Tracer):
                self.found.setdefault(stmt, []).append(v)
        self._prev = line

    def __enter__(self) -> "_Watch":
        mon = getattr(sys, "monitoring", None)
        if mon is None or not self.stmt_of_line:
            return self
        free = [i for i in range(6) if mon.get_tool(i) is None]
        if not free:
            return self
        self.tool = free[-1]
        mon.use_tool_id(self.tool, "expex.annotate")
        mon.register_callback(self.tool, mon.events.LINE, lambda code, line: self._see(sys._getframe(1), line))
        mon.register_callback(
            self.tool, mon.events.PY_RETURN, lambda code, off, ret: self._see(sys._getframe(1), None)
        )
        mon.set_local_events(self.tool, self.code, mon.events.LINE | mon.events.PY_RETURN)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.tool is None:
            return
        mon = sys.monitoring
        mon.set_local_events(self.tool, self.code, 0)
        mon.register_callback(self.tool, mon.events.LINE, None)
        mon.register_callback(self.tool, mon.events.PY_RETURN, None)
        mon.free_tool_id(self.tool)
        self.tool = None


@dataclasses.dataclass
class _Run:
    jaxpr: Any
    out: Any  # pytree of ShapeDtypeStruct
    out_leaves: list[tuple[str, Any]]  # (path, ShapeDtypeStruct)
    locals: dict[tuple[int, str], list[Any]]  # (line, name) -> traced values (run 0) or shapes


def _is_array(v: Any) -> bool:
    return isinstance(v, (jax.Array, np.ndarray, jax.ShapeDtypeStruct)) and hasattr(v, "shape")


def _keystr(path: Any) -> str:
    out = ""
    for k in path:
        if hasattr(k, "key"):
            out += f"[{k.key!r}]"
        elif hasattr(k, "idx"):
            out += f"[{k.idx}]"
        elif hasattr(k, "name"):
            out += f".{k.name}"
        else:
            out += str(k)
    return out


# ------------------------------------------------------------------ names


_LETTERS = "ACEFGJLMNPQRUWXYZ"
_NUMERIC_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]


def _dtype_name(dt: Any) -> str:
    dt = np.dtype(dt) if not jax.dtypes.issubdtype(dt, jax.dtypes.prng_key) else dt
    if jax.dtypes.issubdtype(dt, jax.dtypes.prng_key):
        return "Key"
    if dt == np.bool_:
        return "Bool"
    kind, bits = dt.kind, dt.itemsize * 8
    if kind == "f":
        return f"f{bits}" if bits in (16, 32, 64) and dt.name.startswith("float") else "Float"
    if kind == "i":
        return "Int"
    if kind == "u":
        return f"u{bits}"
    if kind == "c":
        return f"c{bits}"
    return "Arr"


def _camel(name: str) -> str:
    out = "".join(p[:1].upper() + p[1:] for p in re.split(r"[_\W]+", name) if p)
    return out if out.isidentifier() else "Struct"


def _norm_path(p: str) -> str:
    return re.sub(r"\[['\"]?([^'\"\]]+)['\"]?\]", r".\1", p)


@dataclasses.dataclass
class Local:
    """A proposed annotation for an assignment in the body."""

    line: int
    name: str
    annotation: str
    reason: str
    source: str = ""

    def __str__(self) -> str:
        return f"line {self.line}: {self.name}: {self.annotation}   ({self.reason})"


@dataclasses.dataclass
class _Struct:
    name: str
    params: list[str]
    fields: list[tuple[str, str]]
    kind: str  # "struct" (a new @expex.struct), "namedtuple" or "dataclass" (an existing class, made generic)
    cls: Any = None
    dict_path: str = ""  # for a dict turned into a struct: where it was

    def ref(self) -> str:
        return f"{self.name}[{', '.join(self.params)}]" if self.params else self.name

    def source(self) -> str:
        tp = f"[{', '.join(self.params)}]" if self.params else ""
        base = {"namedtuple": "(NamedTuple)"}.get(self.kind, "")
        head = "@expex.struct\n" if self.kind == "struct" else "@dataclasses.dataclass\n" if self.kind == "dataclass" else ""
        body = "".join(f"\n    {n}: {a}" for n, a in self.fields) or "\n    pass"
        return f"{head}class {self.name}{tp}{base}:{body}"


# ------------------------------------------------------------------ the result


class Annotation:
    """Annotations proposed for a plain JAX function by :func:`annotate`.

    ``str(a)`` is the annotated ``def`` (with any struct classes it needs, the
    constraints and the locals); ``a.diff()`` the edit to the source file as a
    unified diff; ``a.apply()`` writes it. ``a.sizes`` maps each size to its
    value in the example; ``a.warnings`` says what could not be told apart.
    """

    def __init__(self) -> None:
        self.function: Any = None
        self.name = ""
        self.type_params: list[str] = []
        self.params: dict[str, str | None] = {}
        self.returns: str | None = None
        self.asserts: list[str] = []
        self.structs: list[_Struct] = []
        self.locals: list[Local] = []
        self.sizes: dict[str, int] = {}
        self.warnings: list[str] = []
        self.notes: list[str] = []
        self._rewrites: dict[str, _Struct] = {}  # dict parameters to read as struct fields
        self._return_struct: _Struct | None = None
        self._header: str = ""
        self._src: "_Source | None" = None

    @property
    def signature(self) -> str:
        """The annotated ``def`` line(s), with the leading ``assert``s."""
        lines = [self._header] + [f"    assert {a}" for a in self.asserts]
        return "\n".join(lines)

    def __str__(self) -> str:
        parts = [s.source() for s in self.structs]
        sig = self.signature
        if self.locals:
            width = max(len(str(loc.line)) for loc in self.locals)
            sig += "\n    ..." + "".join(
                f"\n{loc.line:>{width}} | {loc.source}" for loc in self.locals
            )
        parts.append(sig)
        tail = []
        if self.sizes:
            tail.append("sizes in the example: " + ", ".join(f"{k}={v}" for k, v in self.sizes.items()))
        tail += [f"warning: {w}" for w in self.warnings] + [f"note: {n}" for n in self.notes]
        return "\n\n".join(parts) + ("\n\n" + "\n".join(tail) if tail else "")

    def __repr__(self) -> str:
        return f"<expex.annotate of {self.name}: {self._header!r}>"

    def _repr_markdown_(self) -> str:
        return f"```python\n{self}\n```"

    def source(self) -> str:
        """The function's file as it would be after :meth:`apply`."""
        if self._src is None:
            raise ValueError(f"no source file for {self.name}: annotate can only print its proposal")
        return self._src.edited(self)

    def diff(self) -> str:
        """A unified diff of the function's source file against the annotated version."""
        new = self.source()
        assert self._src is not None
        name = str(self._src.path)
        return "".join(
            difflib.unified_diff(
                self._src.text.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{name}", f"b/{name}"
            )
        )

    def apply(self) -> pathlib.Path:
        """Write the annotations into the function's source file; returns its path."""
        new = self.source()
        assert self._src is not None
        if self._src.path.read_text() != self._src.text:
            raise RuntimeError(f"{self._src.path} changed since annotate read it; run annotate again")
        self._src.path.write_text(new)
        self._src.text = new
        return self._src.path


# ------------------------------------------------------------------ the source


class _Source:
    """The file and syntax tree of the function being annotated."""

    def __init__(self, f: Any):
        file = inspect.getsourcefile(f)
        if file is None or not pathlib.Path(file).is_file():
            raise OSError("no source file")
        self.path = pathlib.Path(file)
        self.text = self.path.read_text()
        self.lines = self.text.splitlines(keepends=True)
        self.tree = ast.parse(self.text)
        code = f.__code__
        self.node: Any = None
        self.top: Any = None  # the module-level statement that holds the function
        for top in self.tree.body:
            for n in ast.walk(top):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == code.co_name:
                    first = min([n.lineno] + [d.lineno for d in n.decorator_list])
                    if first == code.co_firstlineno or n.lineno == code.co_firstlineno:
                        self.node, self.top = n, top
                        break
            if self.node is not None:
                break
        if self.node is None:
            raise OSError(f"cannot find `def {code.co_name}` in {file}")

    def seg(self, node: Any) -> str:
        return ast.get_source_segment(self.text, node) or ast.unparse(node)

    def offset(self, line: int, col: int) -> int:
        """Character offset of an ast (line, byte column) position."""
        before = sum(len(x) for x in self.lines[: line - 1])
        return before + len(self.lines[line - 1].encode()[:col].decode(errors="ignore"))

    def header_span(self) -> tuple[int, int]:
        """Character offsets of ``def ...:`` (from ``def`` to its colon)."""
        n = self.node
        start = self.offset(n.lineno, n.col_offset)
        if isinstance(n, ast.AsyncFunctionDef):
            start = self.text.index("def", start)
        tokens = tokenize.generate_tokens(io.StringIO(self.text[start:]).readline)
        depth = 0
        base_line = n.lineno
        for tok in tokens:
            if tok.type == tokenize.OP:
                if tok.string in "([{":
                    depth += 1
                elif tok.string in ")]}":
                    depth -= 1
                elif tok.string == ":" and depth == 0:
                    line = base_line + tok.start[0] - 1
                    col = tok.start[1] + (n.col_offset if tok.start[0] == 1 else 0)
                    if isinstance(n, ast.AsyncFunctionDef) and tok.start[0] == 1:
                        col = tok.start[1] + (start - self.offset(n.lineno, 0))
                    return start, self.offset(line, 0) + col + 1
        raise OSError("cannot find the end of the def line")

    def stmts(self) -> dict[int, tuple[int, str]]:
        """Line -> (first line, name) of the single-name assignments in the function's own body."""
        out: dict[int, tuple[int, str]] = {}

        def visit(body: list[Any]) -> None:
            for s in body:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                target = None
                if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name):
                    target = s.targets[0].id
                elif isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.value is not None:
                    target = s.target.id
                if target is not None:
                    for ln in range(s.lineno, (s.end_lineno or s.lineno) + 1):
                        out[ln] = (s.lineno, target)
                for field in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(s, field, None)
                    if isinstance(sub, list):
                        visit(sub)
                if isinstance(s, ast.Match):
                    for case in s.cases:
                        visit(case.body)

        visit(self.node.body)
        return out

    def own_nodes(self) -> Iterator[Any]:
        """Nodes of the function's body, not entering nested functions or classes."""
        stack = list(self.node.body)
        while stack:
            n = stack.pop()
            yield n
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.extend(ast.iter_child_nodes(n))

    def edited(self, a: Annotation) -> str:
        edits: list[tuple[int, int, str]] = []
        n = self.node
        s, e = self.header_span()
        indent = " " * n.col_offset
        edits.append((s, e, a._header.replace("\n", "\n" + indent)))
        body_indent = " " * n.body[0].col_offset
        if a.asserts:
            first = n.body[0]
            doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) and isinstance(first.value.value, str)
            if first.lineno == n.lineno:
                a.notes.append("the body starts on the def line, so the constraints are not written; add them by hand")
            else:
                at = self.offset((first.end_lineno or first.lineno) + 1, 0) if doc else self.offset(first.lineno, 0)
                edits.append((at, at, "".join(f"{body_indent}assert {x}\n" for x in a.asserts)))
        for loc in a.locals:
            for stmt in self.own_nodes():
                if isinstance(stmt, ast.Assign) and stmt.lineno == loc.line and len(stmt.targets) == 1:
                    t = stmt.targets[0]
                    if isinstance(t, ast.Name) and t.id == loc.name:
                        p = self.offset(t.lineno, t.end_col_offset or 0)
                        edits.append((p, p, f": {loc.annotation}"))
        for name, struct in a._rewrites.items():
            for node in ast.walk(n):
                chain = _subscript_chain(node, name)
                if chain is not None and not _inside_chain(node, n, name):
                    st = self.offset(node.lineno, node.col_offset)
                    en = self.offset(node.end_lineno or node.lineno, node.end_col_offset or 0)
                    edits.append((st, en, name + "".join("." + k for k in chain)))
        if a._return_struct is not None:
            for node in self.own_nodes():
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
                    d = node.value
                    args = ", ".join(
                        f"{k.value}={self.seg(v)}" for k, v in zip(d.keys, d.values) if isinstance(k, ast.Constant)
                    )
                    st = self.offset(d.lineno, d.col_offset)
                    en = self.offset(d.end_lineno or d.lineno, d.end_col_offset or 0)
                    edits.append((st, en, f"{a._return_struct.name}({args})"))
        top_at = self.offset(min([self.top.lineno] + [d.lineno for d in getattr(self.top, "decorator_list", [])]), 0)
        new_classes = [st for st in a.structs if st.kind == "struct"]
        if new_classes:
            edits.append((top_at, top_at, "".join(st.source() + "\n\n\n" for st in new_classes)))
        for st in a.structs:
            if st.kind != "struct":
                edits += self._class_edits(st)
        edits += self._imports(a)
        out = self.text
        for st_, en_, txt in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
            out = out[:st_] + txt + out[en_:]
        return out

    def _class_edits(self, st: _Struct) -> list[tuple[int, int, str]]:
        """Type parameters and field annotations for a NamedTuple / dataclass defined in this file."""
        cls = None
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef) and node.name == st.name:
                cls = node
        if cls is None:
            return []
        edits: list[tuple[int, int, str]] = []
        start = self.offset(cls.lineno, cls.col_offset)
        name_at = self.text.index(st.name, start) + len(st.name)
        if st.params and not cls.type_params:
            edits.append((name_at, name_at, f"[{', '.join(st.params)}]"))
        fields = dict(st.fields)
        for s in cls.body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name) and s.target.id in fields:
                a = s.annotation
                edits.append(
                    (
                        self.offset(a.lineno, a.col_offset),
                        self.offset(a.end_lineno or a.lineno, a.end_col_offset or 0),
                        fields[s.target.id],
                    )
                )
        return edits

    def _imports(self, a: Annotation) -> list[tuple[int, int, str]]:
        text = a.signature + "".join(s.source() for s in a.structs) + "".join(loc.annotation for loc in a.locals)
        glb = getattr(a.function, "__globals__", {})
        wanted = ["f16", "f32", "f64", "u8", "u16", "u32", "u64", "c64", "c128", "Int", "Float", "Bool", "Key", "Arr"]
        names = [w for w in wanted if re.search(rf"\b{w}\[|\b{w}\b(?!\w)", text) and w not in glb]
        names = [w for w in names if re.search(rf"(?<![\w.]){w}\b", text)]
        need_expex = any(s.kind == "struct" for s in a.structs) and "expex" not in glb
        need_nt = any(s.kind == "namedtuple" for s in a.structs) and "NamedTuple" not in glb and False
        lines = []
        if need_expex:
            lines.append("import expex\n")
        if names:
            lines.append(f"from expex import {', '.join(names)}\n")
        if need_nt:
            lines.append("from typing import NamedTuple\n")
        if not lines:
            return []
        last = None
        for node in self.tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last = node
            elif last is not None or not (isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant)):
                break
        at = self.offset((last.end_lineno or last.lineno) + 1, 0) if last is not None else 0
        return [(at, at, "".join(lines))]


def _subscript_chain(node: Any, name: str) -> list[str] | None:
    """``["a", "b"]`` for ``name["a"]["b"]``; None for anything else."""
    keys: list[str] = []
    while isinstance(node, ast.Subscript):
        sl = node.slice
        if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str)):
            return None
        keys.append(sl.value)
        node = node.value
    if isinstance(node, ast.Name) and node.id == name and keys:
        return keys[::-1]
    return None


def _inside_chain(node: Any, root: Any, name: str) -> bool:
    """Whether ``node`` is the inner part of a longer ``name[...][...]`` chain."""
    for parent in ast.walk(root):
        if isinstance(parent, ast.Subscript) and parent.value is node and _subscript_chain(parent, name) is not None:
            return True
    return False


def _dict_uses_ok(src: _Source, name: str, keys_ok: Callable[[list[str]], bool]) -> str | None:
    """None if ``name`` is only read as ``name["key"]...`` in the function; else where it is not."""
    for node in ast.walk(src.node):
        if isinstance(node, ast.Name) and node.id == name:
            parent = _parent_chain_top(src.node, node, name)
            if parent is None:
                return f"line {node.lineno}"
            chain = _subscript_chain(parent, name)
            if chain is None or not keys_ok(chain) or not isinstance(getattr(parent, "ctx", None), ast.Load):
                return f"line {node.lineno}"
    for arg in src.node.args.args + src.node.args.kwonlyargs + src.node.args.posonlyargs:
        pass
    return None


def _parent_chain_top(root: Any, name_node: Any, name: str) -> Any:
    parents: dict[int, Any] = {}
    for p in ast.walk(root):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    if isinstance(parents.get(id(name_node)), ast.arguments) or isinstance(parents.get(id(name_node)), ast.arg):
        return name_node
    node = name_node
    while isinstance(parents.get(id(node)), ast.Subscript) and parents[id(node)].value is node:
        node = parents[id(node)]
    return node if node is not name_node else None


# ------------------------------------------------------------------ annotate


def annotate(
    f: Callable[..., Any],
    *args: Any,
    vary: bool = False,
    locals: bool | str = False,
    names: Mapping[str, str | Sequence[str]] | None = None,
    **kwargs: Any,
) -> Annotation:
    """Propose expex annotations for a plain JAX function, from a run on example inputs.

    ``f`` is traced (not changed) on ``args`` / ``kwargs``: arrays (or
    ``jax.ShapeDtypeStruct``) become annotated arrays, dicts, tuples and
    NamedTuples of them become structs, Python numbers ``int`` / ``float``,
    anything else keeps its annotation. Sizes are grouped by the program's
    dataflow, not by their values, and named by their role where it is clear
    (``B`` batch, ``T`` / ``S`` sequence, ``D`` features, ``V`` vocabulary,
    ``H`` heads, ``K`` head size); ``names={"q": "B T D"}`` fixes some.

    Args:
        vary: trace again with every size changed, to tell apart sizes that
            coincide in the example and to check the derived ones.
        locals: also propose annotations for assignments in the body: ``True``
            for those with the same size on two axes (the ones expex needs to
            tell axes apart by name), ``"all"`` for every one.
        names: sizes to use, by argument (``"q"``, or a leaf: ``"params.w"``),
            as a string (``"B T D"``, ``_`` to leave one out) or a sequence.
    """
    return _Annotator(f, args, kwargs, vary=vary, locals=locals, names=names or {}).run()


class _Annotator:
    def __init__(self, f: Any, args: tuple[Any, ...], kwargs: dict[str, Any], vary: bool, locals: bool | str, names: Mapping[str, Any]):
        self.orig = f
        self.f = inspect.unwrap(f) if not hasattr(f, "__code__") else f
        while hasattr(self.f, "__wrapped__") and not hasattr(self.f, "__code__"):
            self.f = self.f.__wrapped__
        fn = getattr(self.f, "fn", None)  # an expex Fn: annotate the function inside
        if fn is not None and hasattr(fn, "__code__"):
            self.f = fn
        if not hasattr(self.f, "__code__"):
            raise TypeError(f"annotate needs a Python function, got {f!r}")
        self.sig = inspect.signature(self.f)
        self.bound = self.sig.bind(*args, **kwargs)
        self.vary = vary
        self.want_locals = locals
        self.names = {_norm_path(k): v for k, v in names.items()}
        self.out = Annotation()
        self.out.function = self.f
        self.out.name = self.f.__name__
        try:
            self.src: _Source | None = _Source(self.f)
        except (OSError, TypeError, SyntaxError):
            self.src = None
        self.out._src = self.src

    # -- leaves
    def flatten(self) -> None:
        self.leaves: list[tuple[str, str, Any]] = []  # (param, path, value)
        self.static: dict[str, Any] = {}  # Python numbers by parameter
        self.treedefs: dict[str, Any] = {}
        for name, v in self.bound.arguments.items():
            kind = self.sig.parameters[name].kind
            if kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                flat, tree = jax.tree_util.tree_flatten_with_path(v, is_leaf=lambda x: not _is_array(x) and not _has_arrays(x))
                for p, leaf in flat:
                    if _is_array(leaf):
                        self.leaves.append((name, name + _keystr(p), leaf))
                continue
            if _is_array(v):
                self.leaves.append((name, name, v))
            elif isinstance(v, (bool, int, float)) and not isinstance(v, np.generic):
                self.static[name] = v
            elif _has_arrays(v):
                flat, tree = jax.tree_util.tree_flatten_with_path(v)
                for p, leaf in flat:
                    if _is_array(leaf):
                        self.leaves.append((name, name + _keystr(p), leaf))

    def rebuild(self, values: Sequence[Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        it = iter(values)
        new = dict(self.bound.arguments)
        for name, v in self.bound.arguments.items():
            if _is_array(v):
                new[name] = next(it)
            elif _has_arrays(v):
                new[name] = jax.tree.map(lambda x: next(it) if _is_array(x) else x, v)
        ba = inspect.BoundArguments(self.sig, new)  # type: ignore[arg-type]
        return ba.args, ba.kwargs

    def trace(self, shapes: Sequence[tuple[int, ...]], watch: bool) -> _Run:
        specs = [jax.ShapeDtypeStruct(s, leaf.dtype) for s, (_, _, leaf) in zip(shapes, self.leaves)]

        def g(*vals: Any) -> Any:
            a, k = self.rebuild(vals)
            return self.f(*a, **k)

        stmts = self.src.stmts() if (self.src is not None and watch) else {}
        with _Watch(self.f.__code__, stmts) as w:
            closed, out = jax.make_jaxpr(g, return_shape=True)(*specs)
        flat, _ = jax.tree_util.tree_flatten_with_path(out)
        return _Run(closed, out, [(_keystr(p), leaf) for p, leaf in flat], w.found)

    # -- main
    def run(self) -> Annotation:
        self.flatten()
        shapes = [tuple(int(s) for s in leaf.shape) for _, _, leaf in self.leaves]
        run0 = self.trace(shapes, watch=True)
        flow = Flow(run0.jaxpr)
        self.flow, self.run0 = flow, run0
        invars = run0.jaxpr.jaxpr.invars
        self.inputs = [flow.axes(v) for v in invars]  # nodes of each leaf's axes
        self.groups()
        self.note_units()
        self.name_groups()
        runs = [run0]
        if self.vary:
            runs += self.varied(shapes)
        self.runs = runs
        self.envs = [self.env(i) for i in range(len(runs))]
        self.resolve_sites()
        self.render()
        return self.out

    def groups(self) -> None:
        flow = self.flow
        self.group_of: dict[int, int] = {}  # root -> group index
        self.members: list[list[tuple[int, int]]] = []  # (leaf, axis)
        self.unit_axes: list[tuple[int, int]] = []
        for li, nodes in enumerate(self.inputs):
            for ax, n in enumerate(nodes):
                if flow.size[n] == 1:
                    self.unit_axes.append((li, ax))
                    continue
                r = flow.find(n)
                if r not in self.group_of:
                    self.group_of[r] = len(self.members)
                    self.members.append([])
                self.members[self.group_of[r]].append((li, ax))
        self.facts: list[set[str]] = [set() for _ in self.members]
        for n in range(len(flow.parent)):
            g = self.group_of.get(flow.find(n))
            if g is not None:
                self.facts[g] |= flow.facts.get(n, set())
        self.gsize = [flow.size[self.inputs[m[0][0]][m[0][1]]] for m in self.members]

    def note_units(self) -> None:
        if self.unit_axes:
            where = ", ".join(f"axis {ax} of `{self.leaves[li][1]}`" for li, ax in self.unit_axes[:3])
            self.out.notes.append(
                f"{where} {'has' if len(self.unit_axes) == 1 else 'have'} size 1, annotated as a literal 1; "
                "use sizes above 1 in the example if they vary"
            )

    # -- naming
    def user_names(self) -> dict[int, str]:
        """Group -> name given by the user; two groups given one name are joined."""
        while True:
            assigned: dict[int, str] = {}
            by_name: dict[str, list[int]] = defaultdict(list)
            for li, (param, path, leaf) in enumerate(self.leaves):
                given = self.names.get(_norm_path(path))
                if given is None and path == param:
                    given = self.names.get(param)
                if given is None:
                    continue
                items = given.replace(",", " ").split() if isinstance(given, str) else list(given)
                if len(items) != len(leaf.shape):
                    raise ValueError(f"names for `{path}`: {len(items)} names for {len(leaf.shape)} axes {tuple(leaf.shape)}")
                for ax, nm in enumerate(items):
                    if nm in ("_", "...", "", None):
                        continue
                    nm = str(nm)
                    if not nm.isidentifier() or keyword.iskeyword(nm):
                        raise ValueError(f"names for `{path}`: `{nm}` is not a name")
                    g = self.group_at(li, ax)
                    if g is None:
                        continue
                    if assigned.get(g, nm) != nm:
                        raise ValueError(
                            f"names for `{path}`: axis {ax} is the same size as {assigned[g]} (the dataflow joins them), not {nm}"
                        )
                    assigned[g] = nm
                    if g not in by_name[nm]:
                        by_name[nm].append(g)
            merges = [gs for gs in by_name.values() if len(gs) > 1]
            if not merges:
                return assigned
            for gs in merges:
                for g in gs[1:]:
                    if self.gsize[g] != self.gsize[gs[0]]:
                        raise ValueError(f"names: {assigned[g]} is both {self.gsize[gs[0]]} and {self.gsize[g]} in the example")
                    a, b = self.members[gs[0]][0], self.members[g][0]
                    self.flow.union(self.inputs[a[0]][a[1]], self.inputs[b[0]][b[1]])
            self.groups()

    def name_groups(self) -> None:
        given = self.user_names()
        G = len(self.members)
        names: list[str | None] = [given.get(g) for g in range(G)]
        used = set(self.sig.parameters) | set(self.static) | {n for n in names if n}

        leaf_ndim = [len(leaf.shape) for _, _, leaf in self.leaves]
        is_int = [jnp.issubdtype(leaf.dtype, jnp.integer) for _, _, leaf in self.leaves]

        def free(g: int) -> bool:
            return names[g] is None

        def take(g: int, nm: str) -> bool:
            if names[g] is None and nm not in used:
                names[g] = nm
                used.add(nm)
                return True
            return False

        order = range(G)
        busy = {_CONTRACT, _REDUCE, _SCAN, _SEQ, _GATHER, _SPLIT_SRC, _SPLIT_HEAD, _SPLIT_REST}
        # V: a table's axis that integer indices pick from
        for g in order:
            if _GATHER in self.facts[g] and take(g, "V"):
                break
        # B: a leading axis that every operation carries along
        lead: dict[int, int] = defaultdict(int)
        for li in range(len(self.leaves)):
            g = self.group_at(li, 0)
            if g is not None and leaf_ndim[li] >= 2 and not (self.facts[g] & busy):
                lead[g] += 1
        if lead:
            best = max(lead.values())
            for g in order:
                if lead.get(g) == best and take(g, "B"):
                    break
        b = names.index("B") if "B" in names else None
        # H: a batch dimension of a contraction other than B; or the first factor of a split
        for g in order:
            if free(g) and ((_BATCH in self.facts[g] and g != b) or _SPLIT_HEAD in self.facts[g]):
                if take(g, "H"):
                    break
        h = names.index("H") if "H" in names else None
        # T, S: scanned, padded, sliced or concatenated axes; middle axes after B; token positions
        seq: list[int] = []
        for g in order:
            if free(g) and self.facts[g] & {_SCAN, _SEQ} and _CONTRACT not in self.facts[g]:
                seq.append(g)
        for g in order:
            if not free(g) or g in seq or self.facts[g] & {_CONTRACT, _GATHER, _BATCH}:
                continue
            for li, ax in self.members[g]:
                nd = leaf_ndim[li]
                first = self.group_at(li, 0)
                middle = ax >= 1 and ax < nd - 1 and (b is not None and first == b) and nd >= 3
                tokens = is_int[li] and (ax >= 1 or b is None) and nd >= 1
                if middle or tokens:
                    seq.append(g)
                    break
        for g, nm in zip(seq, ["T", "S"]):
            take(g, nm)
        # K: contracted, last axis of an array that has H; or the rest of a split
        for g in order:
            if not free(g):
                continue
            last_with_h = h is not None and any(
                ax == leaf_ndim[li] - 1 and any(self.group_at(li, a) == h for a in range(leaf_ndim[li]))
                for li, ax in self.members[g]
            )
            if (_CONTRACT in self.facts[g] and last_with_h) or _SPLIT_REST in self.facts[g]:
                if take(g, "K"):
                    break
        # D: contracted, split, or normalised over
        for g in order:
            if free(g) and self.facts[g] & {_CONTRACT, _SPLIT_SRC, _REDUCE} and take(g, "D"):
                break
        for g in order:  # otherwise the last axis of the first float input with two or more axes
            if "D" in names:
                break
            if free(g) and any(ax == leaf_ndim[li] - 1 and leaf_ndim[li] >= 2 and not is_int[li] for li, ax in self.members[g]):
                take(g, "D")
                break
        pool = iter([c for c in _LETTERS if c not in used] + [f"N{i}" for i in range(1, 100)])
        for g in order:
            if names[g] is None:
                nm = next(pool)
                while nm in used:
                    nm = next(pool)
                take(g, nm)
        self.gname: list[str] = [n or "?" for n in names]

    def group_at(self, li: int, ax: int) -> int | None:
        n = self.inputs[li][ax]
        return self.group_of.get(self.flow.find(n))

    # -- varied runs
    def varied(self, shapes: list[tuple[int, ...]]) -> list[_Run]:
        used = set(self.gsize) | {v for v in self.static.values() if isinstance(v, int)}
        new: dict[int, int] = {}
        for g, s in enumerate(self.gsize):
            for p in _NUMERIC_PRIMES:
                if s * p not in used:
                    new[g] = s * p
                    used.add(s * p)
                    break

        def shapes_for(groups: Mapping[int, int]) -> list[tuple[int, ...]]:
            out = []
            for li, sh in enumerate(shapes):
                out.append(
                    tuple(groups.get(g, d) if (g := self.group_at(li, ax)) is not None else d for ax, d in enumerate(sh))
                )
            return out

        self.varied_sizes: list[dict[int, int]] = []
        runs: list[_Run] = []
        try:
            r = self.trace(shapes_for(new), watch=True)
            if self.same_structure(r):
                self.varied_sizes.append(new)
                runs.append(r)
        except Exception as e:  # the function does not allow all of them at once: one at a time
            first_error = e
            failed = []
            for g, s in new.items():
                try:
                    r = self.trace(shapes_for({g: s}), watch=True)
                except Exception:
                    failed.append(g)
                    continue
                if self.same_structure(r):
                    self.varied_sizes.append({g: s})
                    runs.append(r)
            if failed:
                self.out.notes.append(
                    f"vary: the function fails when {', '.join(self.gname[g] for g in failed)} change "
                    f"({type(first_error).__name__}), so {'it was' if len(failed) == 1 else 'they were'} only tried at the example size"
                )
        return runs

    def same_structure(self, r: _Run) -> bool:
        a = [(p, len(x.shape)) for p, x in self.run0.out_leaves]
        b = [(p, len(x.shape)) for p, x in r.out_leaves]
        return a == b

    def env(self, i: int) -> dict[str, int]:
        env = {self.gname[g]: s for g, s in enumerate(self.gsize)}
        if i > 0:
            env.update({self.gname[g]: s for g, s in self.varied_sizes[i - 1].items()})
        env.update({k: int(v) for k, v in self.static.items() if isinstance(v, int) and not isinstance(v, bool)})
        return env

    # -- sizes of every axis
    def expr_of_root(self, r: int, visiting: frozenset[int] = frozenset(), splits: bool = True) -> Size | None:
        key = (r, splits)
        memo = self.__dict__.setdefault("_memo", {})
        if key in memo and not visiting:
            return memo[key]
        flow = self.flow
        if r in visiting:
            return None
        visiting = visiting | {r}
        out: Size | None = None
        if (g := self.group_of.get(r)) is not None:
            out = Size.name(self.gname[g])
        elif flow.size[r] == 1:
            out = Size.const(1)
        else:
            for rel in flow.rels:
                kind = rel[0]
                if kind in ("sum", "prod", "div") and flow.find(rel[1]) == r:
                    out = self._rel(rel, visiting)
                elif kind == "split" and flow.find(rel[1]) == r:
                    parts = [self.expr_of_root(flow.find(o), visiting, splits) for o in rel[2]]
                    if all(p is not None for p in parts):
                        out = _prod(parts)  # type: ignore[arg-type]
                elif kind == "split" and splits and any(flow.find(o) == r for o in rel[2]):
                    solved = self._split(rel, visiting)
                    out = solved.get(r)
                if out is not None:
                    break
        if out is not None and out.eval(self.envs[0]) != flow.size[r]:
            out = None
        if not visiting - {r}:
            memo[key] = out
        return out

    def _rel(self, rel: tuple[Any, ...], visiting: frozenset[int]) -> Size | None:
        flow = self.flow
        kind = rel[0]
        if kind == "div":
            e = self.expr_of_root(flow.find(rel[2]), visiting)
            return None if e is None else e // rel[3]
        parts: list[Size] = []
        for p in rel[2]:
            if isinstance(p, tuple):
                parts.append(Size.const(p[1]))
                continue
            e = self.expr_of_root(flow.find(p), visiting)
            if e is None:
                return None
            parts.append(e)
        if kind == "sum":
            total = Size()
            for p_ in parts:
                total = total + p_
            return total
        return _prod(parts)

    def _split(self, rel: tuple[Any, ...], visiting: frozenset[int]) -> dict[int, Size]:
        """Sizes of a reshape's split axes: a known factor, a static argument or a number, then the rest."""
        flow = self.flow
        src = self.expr_of_root(flow.find(rel[1]), visiting)
        outs = [flow.find(o) for o in rel[2]]
        es = [self.expr_of_root(o, visiting, splits=False) for o in outs]
        if src is None:
            return {}
        missing = [i for i, e in enumerate(es) if e is None]
        for i in missing[:-1]:
            es[i] = self.by_static(flow.size[outs[i]]) or Size.const(flow.size[outs[i]])
        if missing:
            rest = _prod([e for i, e in enumerate(es) if i != missing[-1]])  # type: ignore[misc]
            es[missing[-1]] = src // rest
        return {o: e for o, e in zip(outs, es) if e is not None}

    def by_static(self, v: int) -> Size | None:
        hits = [k for k, x in self.static.items() if isinstance(x, int) and not isinstance(x, bool) and x == v]
        return Size.name(hits[0]) if len(hits) == 1 else None

    def site(self, node: int | None, values: list[int | None], where: str) -> Size:
        """The size of one axis: from the dataflow, checked against every run, else by value."""
        e = self.expr_of_root(self.flow.find(node)) if node is not None else None
        ok = lambda x: all(v is None or x.eval(env) == v for env, v in zip(self.envs, values))  # noqa: E731
        if e is not None and ok(e):
            self._derive_constraints(e)
            return e
        v0 = values[0]
        assert v0 is not None
        if v0 == 1:
            return Size.const(1)
        atoms = [n for n, x in self.envs[0].items() if x != 1]
        levels: list[list[Size]] = [[Size.name(a) for a in atoms]]
        if len(self.runs) > 1:
            levels.append(
                [Size.name(a) + c for a in atoms for c in (1, -1, 2, -2)]
                + [Size.name(a) * 2 for a in atoms]
                + [Size.name(a) // 2 for a in atoms]
            )
            levels.append(
                [Size.name(a) * Size.name(b) for a, b in itertools.combinations(atoms, 2)]
                + [Size.name(a) + Size.name(b) for a, b in itertools.combinations(atoms, 2)]
                + [Size.name(a) // Size.name(b) for a, b in itertools.permutations(atoms, 2)]
            )
        for level in levels:
            hits = list(dict.fromkeys(x for x in level if ok(x)))
            if hits:
                if len(hits) > 1:
                    alts = " or ".join(str(x) for x in hits)
                    self.out.warnings.append(
                        f"{where} is {v0}, which could be {alts}: these sizes coincide in the example and the "
                        f"dataflow does not say which; guessed {hits[0]}. Use distinct sizes"
                        + (" or vary=True" if len(self.runs) == 1 else "")
                    )
                elif len(self.runs) == 1:
                    self.out.notes.append(f"{where} is {hits[0]} by value only (no operation joins it to an input)")
                return hits[0]
        if all(v == v0 for v in values):
            return Size.const(v0)
        nm = self.fresh()
        self.out.warnings.append(f"{where} depends on the inputs in a way annotate could not express; called it {nm}")
        return Size.name(nm)

    def fresh(self) -> str:
        used = set(self.gname) | set(self.sig.parameters) | set(self.extra_names)
        nm = next(c for c in _LETTERS if c not in used)
        self.extra_names.append(nm)
        return nm

    def _derive_constraints(self, e: Size) -> None:
        for d in e.divs():
            c = f"{d.num.text(paren=True)} % {d.den.text(paren=True)} == 0"
            if c not in self.out.asserts:
                self.out.asserts.append(c)

    def resolve_sites(self) -> None:
        self.extra_names: list[str] = []
        flow, run0 = self.flow, self.run0
        # every split whose sizes are all known: a constraint between them
        for rel in flow.rels:
            if rel[0] == "split":
                src = self.expr_of_root(flow.find(rel[1]), splits=False)
                parts = [self.expr_of_root(flow.find(o), splits=False) for o in rel[2]]
                if src is not None and all(p is not None for p in parts):
                    prod = _prod(parts)  # type: ignore[arg-type]
                    if prod != src and self.group_of.get(flow.find(rel[1])) is not None:
                        c = f"{src} == {prod}"
                        if c not in self.out.asserts:
                            self.out.asserts.append(c)
        # outputs
        outvars = run0.jaxpr.jaxpr.outvars
        self.out_ann: dict[str, str] = {}
        for i, (path, leaf) in enumerate(run0.out_leaves):
            var = outvars[i]
            nodes = flow.index.get(id(var)) if not _is_literal(var) else None
            dims = []
            for ax in range(len(leaf.shape)):
                values = [int(r.out_leaves[i][1].shape[ax]) for r in self.runs]
                dims.append(self.site(nodes[ax] if nodes else None, list(values), f"axis {ax} of the result{path}"))
            self.out_ann[path] = _array_ann(leaf.dtype, dims)
        # locals
        self.local_ann: list[tuple[int, str, list[Size], Any]] = []
        if self.want_locals:
            for (line, name), vals in sorted(run0.locals.items()):
                shapes = {tuple(v.shape) for v in vals}
                if len(shapes) != 1:
                    continue
                v = vals[0]
                var = getattr(v, "val", None)
                nodes = flow.index.get(id(var)) if var is not None else None
                dims = []
                for ax in range(len(v.shape)):
                    values: list[int | None] = []
                    for r in self.runs:
                        got = r.locals.get((line, name))
                        values.append(int(got[0].shape[ax]) if got and len(got[0].shape) == len(v.shape) else None)
                    dims.append(self.site(nodes[ax] if nodes else None, values, f"axis {ax} of `{name}` (line {line})"))
                self.local_ann.append((line, name, dims, v))

    # -- rendering
    def leaf_ann(self, li: int) -> str:
        _, _, leaf = self.leaves[li]
        dims = []
        for ax, n in enumerate(self.inputs[li]):
            g = self.group_at(li, ax)
            dims.append(Size.name(self.gname[g]) if g is not None else Size.const(self.flow.size[n]))
        return _array_ann(leaf.dtype, dims)

    def render(self) -> None:
        out = self.out
        leaf_ann = {path: self.leaf_ann(li) for li, (_, path, _) in enumerate(self.leaves)}
        self.order = list(dict.fromkeys(self.gname))
        self.struct_names: set[str] = set()
        glb = getattr(self.f, "__globals__", {})
        params: dict[str, str | None] = {}
        for name, p in self.sig.parameters.items():
            if name in self.bound.arguments:
                v = self.bound.arguments[name]
                if name in self.static:
                    params[name] = type(v).__name__
                elif _is_array(v):
                    params[name] = leaf_ann[name]
                elif _has_arrays(v):
                    params[name] = self.container(v, name, leaf_ann, glb, param=name)
                else:
                    params[name] = None
            else:
                params[name] = None
        out.params = params
        ret = self.run0.out
        out_leaves = {path: a for path, a in self.out_ann.items()}
        if ret is None:
            out.returns = "None"
        elif isinstance(ret, jax.ShapeDtypeStruct):
            out.returns = out_leaves[""]
        else:
            out.returns = self.container(ret, "", out_leaves, glb, param=None)
        # locals worth annotating
        for line, name, dims, v in self.local_ann:
            texts = [str(d) for d in dims]
            sizes = [int(s) for s in v.shape]
            dup = len(set(texts)) < len(texts)
            same = len({s for s in sizes if s != 1}) < len([s for s in sizes if s != 1])
            if self.want_locals != "all" and not (dup or same):
                continue
            reason = (
                "the same size on two axes" if dup else "two axes of the same length in the example" if same else "every assignment"
            )
            ann = _array_ann(v.dtype, dims)
            src_line = self.src.lines[line - 1].rstrip("\n") if self.src else f"{name} = ..."
            shown = re.sub(rf"^(\s*){re.escape(name)}(\s*)=", rf"\g<1>{name}: {ann}\g<2>=", src_line, count=1)
            out.locals.append(Local(line, name, ann, reason, shown))
        for line, name, dims, v in self.local_ann:
            pass
        # type parameters, in order of appearance
        text = " ".join(a or "" for a in params.values()) + " " + (out.returns or "")
        text += " " + " ".join(f for s in out.structs for _, f in s.fields)
        used_names = [n for n in self.order + self.extra_names if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)]
        statics = set(self.static)
        first_seen = sorted(used_names, key=lambda n: _first_index(text, n))
        out.type_params = [n for n in first_seen if n not in statics]
        out.sizes = {n: self.envs[0][n] for n in out.type_params if n in self.envs[0]}
        for a in list(out.asserts):  # constraints only on sizes that are in the signature
            if not Size_names(a) <= set(out.type_params) | statics:
                out.asserts.remove(a)
        self.coincidences()
        out._header = self.header()

    def coincidences(self) -> None:
        by_value: dict[int, list[str]] = defaultdict(list)
        for n in self.out.type_params:
            if n in self.envs[0]:
                by_value[self.envs[0][n]].append(n)
        for s in self.static:
            v = self.static[s]
            if isinstance(v, int) and not isinstance(v, bool) and v in by_value and v != 1:
                by_value[v].append(s)
        for v, ns in by_value.items():
            if len(ns) > 1 and not (self.vary and len(self.runs) > 1):
                self.out.notes.append(
                    f"{' and '.join(ns)} are all {v} in the example; they are kept apart because no operation joins "
                    "them. Use distinct sizes, or vary=True, to be sure"
                )

    def container(self, v: Any, path: str, leaf_ann: Mapping[str, str], glb: Mapping[str, Any], param: str | None) -> str:
        """The annotation of a pytree of arrays: a struct, NamedTuple, tuple, list or dict."""
        out = self.out
        if v is None:
            return "None"
        if _is_array(v):
            return leaf_ann.get(path, "Arr")
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, (int, float)):
            return type(v).__name__
        if isinstance(v, tuple) and hasattr(v, "_fields"):
            fields = [(f, self.container(getattr(v, f), f"{path}.{f}", leaf_ann, glb, None)) for f in v._fields]
            return self.add_struct(type(v).__name__, fields, "namedtuple", type(v)).ref()
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            fields = [(fl.name, self.container(getattr(v, fl.name), f"{path}.{fl.name}", leaf_ann, glb, None)) for fl in dataclasses.fields(v)]
            st = self.add_struct(type(v).__name__, fields, "dataclass", type(v))
            out.notes.append(f"{type(v).__name__} is a dataclass: annotate its fields as shown; applying edits it only if it is defined in the same file")
            return st.ref()
        if isinstance(v, tuple):
            return f"tuple[{', '.join(self.container(x, f'{path}[{i}]', leaf_ann, glb, None) for i, x in enumerate(v))}]"
        if isinstance(v, list):
            elems = [self.container(x, f"{path}[{i}]", leaf_ann, glb, None) for i, x in enumerate(v)]
            if elems and len(set(elems)) == 1:
                return f"list[{elems[0]}]"
            out.notes.append(f"`{path}` is a list whose elements differ, annotated as a tuple of them; pass a tuple")
            return f"tuple[{', '.join(elems)}]"
        if isinstance(v, dict):
            vals = {k: self.container(x, f"{path}[{k!r}]", leaf_ann, glb, None) for k, x in v.items()}
            if len(set(vals.values())) == 1 and not all(isinstance(k, str) and k.isidentifier() for k in v):
                return f"dict[str, {next(iter(vals.values()))}]"
            if not all(isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k) for k in v):
                if len(set(vals.values())) == 1:
                    return f"dict[str, {next(iter(vals.values()))}]"
                out.warnings.append(f"`{path}` is a dict whose keys are not all names; left unannotated")
                return "dict"
            if len(set(vals.values())) == 1 and len(vals) > 1 and param is None and path.count("[") > 0:
                return f"dict[str, {next(iter(vals.values()))}]"
            base = (param or path.rsplit("[", 1)[-1].strip("'\"]") or self.f.__name__ + "_output") if path else self.f.__name__ + "_output"
            if base.endswith("s") and param is None and path:
                base = base[:-1]
            st = self.add_struct(_camel(base), list(vals.items()), "struct", None)
            st.dict_path = path
            if param is not None:
                if self.src is None:
                    pass
                elif (where := _dict_uses_ok(self.src, param, lambda ks: True)) is None:
                    self._rewrite(param, st)
                    out.notes.append(
                        f"`{param}` is a dict: it becomes the struct {st.name} (the body reads `{param}.{next(iter(v))}`); "
                        f"callers pass {st.name}({', '.join(f'{k}=...' for k in v)})"
                    )
                else:
                    out.warnings.append(
                        f"`{param}` is a dict used as a whole ({where}), so it cannot become the struct {st.name} "
                        "automatically: it is left unannotated, and expex.prove cannot check this function until it is"
                    )
                    self._unannotated.add(param)
                return st.ref()
            if path == "":
                if self.src is not None and self._returns_dict_literals():
                    out._return_struct = st
                    out.notes.append(f"the result is a dict: it becomes the struct {st.name}, built by the return statements")
                else:
                    out.warnings.append(f"the result is a dict built elsewhere than in `return {{...}}`; make it a {st.name} by hand")
            return st.ref()
        out.warnings.append(f"`{path or 'the result'}` is a {type(v).__name__}, which annotate does not look inside")
        return type(v).__name__

    @property
    def _unannotated(self) -> set[str]:
        return self.__dict__.setdefault("_unann", set())

    def _rewrite(self, param: str, st: _Struct) -> None:
        self.out._rewrites[param] = st

    def _returns_dict_literals(self) -> bool:
        assert self.src is not None
        rets = [n for n in self.src.own_nodes() if isinstance(n, ast.Return)]
        return bool(rets) and all(
            isinstance(r.value, ast.Dict) and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in r.value.keys)
            for r in rets
        )

    def add_struct(self, name: str, fields: list[tuple[str, str]], kind: str, cls: Any) -> _Struct:
        text = " ".join(a for _, a in fields)
        params = [n for n in self.order + self.extra_names if re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)]
        params.sort(key=lambda n: _first_index(text, n))
        for st in self.out.structs:
            if st.name == name:
                if st.fields == fields:
                    return st
                name = name + str(sum(s.name.startswith(name) for s in self.out.structs) + 1)
        st = _Struct(name, params, fields, kind, cls)
        if kind == "struct":
            self.out.structs.insert(0, st) if False else self.out.structs.append(st)
        else:
            self.out.structs.append(st)
            if kind == "namedtuple":
                same_file = self.src is not None and _defined_in(cls, self.src)
                self.out.notes.append(
                    f"{name} is a NamedTuple: make it generic with field annotations as shown"
                    + ("" if same_file else " (it is defined in another file, so apply() does not edit it)")
                )
        return st

    def header(self) -> str:
        out = self.out
        node = self.src.node if self.src is not None else None
        defaults: dict[str, str] = {}
        old_ann: dict[str, str] = {}
        old_tp: list[str] = []
        if node is not None:
            a = node.args
            pos = a.posonlyargs + a.args
            for arg, d in zip(pos[len(pos) - len(a.defaults) :], a.defaults):
                defaults[arg.arg] = self.src.seg(d)  # type: ignore[union-attr]
            for arg, d in zip(a.kwonlyargs, a.kw_defaults):
                if d is not None:
                    defaults[arg.arg] = self.src.seg(d)  # type: ignore[union-attr]
            for arg in pos + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x is not None]:
                if arg.annotation is not None:
                    old_ann[arg.arg] = self.src.seg(arg.annotation)  # type: ignore[union-attr]
            old_tp = [self.src.seg(t) for t in getattr(node, "type_params", [])]  # type: ignore[union-attr]
        items: list[str] = []
        seen_kwonly = False
        last_posonly = None
        params = list(self.sig.parameters.values())
        for i, p in enumerate(params):
            if p.kind == p.POSITIONAL_ONLY:
                last_posonly = i
        for i, p in enumerate(params):
            ann = out.params.get(p.name)
            if p.name in self._unannotated:
                ann = None
            if ann is None:
                ann = old_ann.get(p.name)
            prefix = "*" if p.kind == p.VAR_POSITIONAL else "**" if p.kind == p.VAR_KEYWORD else ""
            if p.kind == p.KEYWORD_ONLY and not seen_kwonly:
                items.append("*")
            if p.kind in (p.KEYWORD_ONLY, p.VAR_POSITIONAL):
                seen_kwonly = True
            s = prefix + p.name
            if ann:
                s += f": {ann}"
            if p.default is not p.empty:
                d = defaults.get(p.name, repr(p.default))
                s += f" = {d}" if ann else f"={d}"
            items.append(s)
            if last_posonly == i:
                items.append("/")
        tps = list(dict.fromkeys(old_tp + out.type_params))
        tp = f"[{', '.join(tps)}]" if tps else ""
        ret = f" -> {out.returns}" if out.returns else ""
        if out.returns == "None" and node is not None and node.returns is None:
            ret = ""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        one = f"{prefix} {self.f.__name__}{tp}({', '.join(items)}){ret}:"
        indent = node.col_offset if node is not None else 0
        if len(one) + indent <= 100:
            return one
        body = "".join(f"\n    {x}," for x in items)
        return f"{prefix} {self.f.__name__}{tp}({body}\n){ret}:"


def _defined_in(cls: Any, src: _Source) -> bool:
    try:
        return pathlib.Path(inspect.getsourcefile(cls) or "") == src.path
    except TypeError:
        return False


def Size_names(constraint: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*", constraint))


def _first_index(text: str, n: str) -> int:
    m = re.search(rf"(?<![\w.]){re.escape(n)}(?!\w)", text)
    return m.start() if m else len(text)


def _prod(parts: Sequence[Size]) -> Size:
    out = Size.const(1)
    for p in parts:
        out = out * p
    return out


def _array_ann(dtype: Any, dims: Sequence[Size]) -> str:
    name = _dtype_name(dtype)
    return f"{name}[{', '.join(map(str, dims))}]" if dims else name


def _has_arrays(v: Any) -> bool:
    if _is_array(v):
        return True
    if isinstance(v, (str, bytes, int, float, complex, type(None))) or callable(v) and not isinstance(v, (tuple, list, dict)):
        return False
    try:
        return any(_is_array(x) for x in jax.tree.leaves(v))
    except Exception:
        return False

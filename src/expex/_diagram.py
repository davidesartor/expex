"""Architecture diagrams: an ``@fn`` or a module drawn as an SVG figure, like a paper's schematic.

``diagram(f)`` traces ``f`` like ``expex.cost`` and ``expex.prove`` do, with symbolic
sizes and no data, while every typed call runs inside a ``jax.named_scope``
named after it. Each equation of the jaxpr then knows which calls it is
nested in, and the calls become nodes::

    >>> d = diagram(Transformer.__call__)       # no data: sizes stay symbolic
    >>> d.nodes                                 # (id, kind, label, box)
    [('n0', 'in', 'tokens: i32[T]', -1), ('n7', 'op', 'index', -1),
     ('b1', 'call', 'Block.__call__\nf32[T, D] → f32[T, D]', 0), ...]
    >>> d.boxes
    {'b0': '× L'}
    >>> d.save("transformer.svg")               # the figure; .png (light or dark) and .html too
    >>> print(d)                                # an outline of the calls

- A call deeper than ``depth`` is one node, labelled with its input and output
  shapes (per element for ``.vectorize``). Shallower calls are boxes around their
  contents, so ``depth=2`` shows a block's attention and MLP.
- Operations outside any call at that depth are drawn only when they say what
  the model does: ``matmul``, ``index``, reductions (``sum``, ``max``),
  ``conv``, ``scatter``, ... and ``+`` where two computed values of one shape
  meet (a residual). Reshapes, broadcasts, other elementwise arithmetic and
  ``jit`` wrappers are passed through: edges go straight across them.
- A staged loop is a box labelled with its trip count: ``× L`` for a list of
  modules (``L`` unless given, like ``expex.cost``), ``× T`` for a loop over time.
  Traced ``if`` and ``while`` are boxes too.
- Edges follow the data (through passed-through operations), with the array's
  type as it leaves a node, or as it enters a call.
- Inputs are the parameters other than ``self``: a module's own arrays reach
  nearly every node, so they are left out.

Drawing (``_Scene``, ``_Figure``): the layout and the SVG are computed here, in
Python, so a figure needs no browser and the page no CDN (an earlier version
drew Mermaid text with Mermaid loaded from a CDN, which offline pages and
sandboxes cannot reach). The style is a paper's architecture figure: the flow
runs left to right in columns by dependency depth, each input a stream with its
own colour (pale boxes, inner ones paler), a box's name in bold over what it
produces in grey, residuals as ``⊕`` with the skip looping over the row, a loop
over layers as a ``Block × L`` container, side inputs as dashed orange lines.
The figure is a display list (``_Canvas``) written as SVG with a light and a dark
palette in CSS variables, or drawn with matplotlib when a PNG is wanted and no
browser is installed. The page (``_diagram.html``) embeds an SVG per view: each
depth, and the views clicks reach from them, up to a budget (a click past it
shows the nearest depth).

Measured time, ``diagram.timed(f, *args)``: the same trace, but on real
arguments under ``jax.jit``, compiled and run under ``jax.profiler``. The
profiler's events name XLA instructions (``fusion.12``), not scopes, so each
is looked up in the compiled HLO, whose ``op_name`` metadata carries the
``Name#uid`` scopes of the calls it came from. A fusion is split between the
calls of its instructions; a staged loop's body runs once per iteration and
each run is an event, so ``× L`` boxes add up their iterations. Without a
profiler, shares are estimated from FLOPs per box (as ``expex.cost`` counts them)
and labelled so.

Shape errors, ``diagram.explain(f, *args)`` or ``error.diagram``: the trace is
redone on the arguments' shapes with loops unrolled, and stops at the call that
raises: the hook turns the ``ShapeError`` into ``_Stop``, which carries the
failing call's arguments (or those of the nearest enclosing call traced at the
top level) out as the partial trace's outputs. The diagram draws what was
traced up to there, the failing call in red, and the path that feeds it.
"""

import gzip
import html
import inspect
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import textwrap
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

from . import function as _fn
from .annotations.dims import fmt_dim, short_dtype
from .annotations.errors import ShapeError, coded
from .annotations.spec import fn_fixed_sizes
from .tools import cost as _cost
from .tools.trace import sub_jaxprs

#: Operations that are drawn (as small nodes) when outside every closed call.
OPS = {
    "dot_general": "matmul", "gather": "index", "dynamic_update_slice": "update",
    "scatter": "scatter", "scatter-add": "scatter", "scatter_add": "scatter",
    "reduce_sum": "sum", "reduce_max": "max", "reduce_min": "min", "reduce_prod": "prod",
    "argmax": "argmax", "argmin": "argmin", "cumsum": "cumsum", "cumprod": "cumprod",
    "cummax": "cummax", "cumlogsumexp": "cumlogsumexp", "conv_general_dilated": "conv",
    "sort": "sort", "concatenate": "concat", "fft": "fft", "top_k": "top_k",
    "random_bits": "random", "all_gather": "all_gather", "psum": "psum",
}  # fmt: skip
_TOKEN = re.compile(r"(vmap\()?([^/()]+)#(\d+)")
_PAGE = Path(__file__).with_name("_diagram.html")
#: Share of the measured time from which a box is drawn one step hotter (5 steps).
HEAT = (0.02, 0.08, 0.2, 0.4)
#: XLA instructions whose events contain the events of the instructions they run.
_CONTROL = {"while", "conditional", "call", "async-start", "async-update", "async-done"}
_PLUMBING = {"parameter", "constant", "bitcast", "get-tuple-element", "tuple"}


def _type(x: Any) -> str | None:
    """``f32[T, D]`` for an array (or aval), None for keys and non-arrays."""
    dt, shape = getattr(x, "dtype", None), getattr(x, "shape", None)
    if dt is None or shape is None or jnp.issubdtype(dt, jax.dtypes.prng_key):
        return None
    return short_dtype(dt) + ("[" + ", ".join(map(fmt_dim, shape)) + "]" if len(shape) else "")


def _brief(v: Any) -> str | None:
    """An argument or result as text: its array type, or its class for a pytree of arrays."""
    if (t := _type(v)) is not None:
        return t
    if isinstance(v, (tuple, list)) and not hasattr(v, "_fields"):
        parts = [p for p in map(_brief, v) if p]
        return f"({', '.join(parts)})" if parts else None
    return type(v).__name__ if any(_type(x) for x in jax.tree.leaves(v)) else None


def _get(env: dict[Any, list[int]], v: Any) -> list[int]:
    """The atoms a variable comes from (none for a literal)."""
    return [] if type(v).__name__ == "Literal" else env.get(v, [])


def _short(name: str) -> str:
    return name.rsplit("<locals>.", 1)[-1]


def _barrier(v: Any) -> Any:
    """``v`` through ``optimization_barrier``: an equation that marks where it enters or leaves."""
    leaves, tree = jax.tree.flatten(v)
    idx = [i for i, x in enumerate(leaves) if _type(x) is not None]
    for i, x in zip(idx, jax.lax.optimization_barrier([leaves[i] for i in idx])):
        leaves[i] = x
    return jax.tree.unflatten(tree, leaves)


class _Stop(BaseException):
    """Raised by the hook (with ``stop=True``) in place of a call's ``ShapeError``.

    A ``BaseException``, so it passes through expex's and JAX's ``except
    Exception`` handlers to the trace's top, carrying arrays traced at the top
    level: the arguments of the failing call, or of the innermost call around it
    whose arguments are top-level tracers (a ``.vectorize``'s are unbatched).
    """

    def __init__(self) -> None:
        super().__init__()
        self.payload: list[Any] | None = None
        self.at = -1


def _arrays(v: Any) -> list[Any]:
    return [x for x in jax.tree.leaves(v) if _type(x) is not None]


class _Hook:
    """Installed as ``_fn.CALL_HOOK``: scopes every call and records what it was.

    With ``stop``, the first ``ShapeError`` a call raises is recorded and turned
    into ``_Stop``, so the trace ends there (see ``explain``).
    """

    def __init__(self, stop: bool = False) -> None:
        self.uid, self.depth, self.root = 0, 0, -1
        self.calls: dict[int, dict[str, Any]] = {}
        self.inputs: dict[str, str] = {}
        self.sources: dict[str, str] = {}
        self.stop = stop
        self.open: list[int] = []  # the calls running now, outermost first
        self.batched: list[bool] = []  # whether their arguments are batched (inside a vmap)
        self.lifted: dict[int, bool] = {}
        self.error: dict[str, Any] | None = None
        self.top: Any = None  # the trace the root call's arguments belong to
        self.leaves: dict[int, tuple[str, Any]] = {}  # id(array of a module) -> (its path, the array)
        self.modules: list[tuple[str | None, Any]] = []  # (path, module) of the open method calls

    def path(self, fn: Any, args: Any) -> str | None:
        """The module a method is called on, by path from the root's (``blocks[*].attn``); None if unknown.

        Found by the identity of its arrays, as ``debug.watch`` does. A layer
        sliced out of a staged loop has new arrays: it is the one field (or list)
        of its class in the enclosing module, and its own modules follow from it.
        """
        if next(iter(fn.params), None) != "self" or not args:
            return None
        me = args[0]
        leaves = jax.tree_util.tree_flatten_with_path(me)[0]
        rel = jax.tree_util.keystr(leaves[0][0]) if leaves else ""
        known = self.leaves.get(id(leaves[0][1])) if leaves else None
        if known is not None and known[0].endswith(rel) and known[1] is leaves[0][1]:
            return known[0][: len(known[0]) - len(rel)].lstrip(".")
        if not self.modules:
            if self.depth:  # inside a plain function: no module to start from
                return None
            path = ""
        else:
            up, parent = self.modules[-1]
            if up is None:
                return None
            names = []
            for f in getattr(type(parent), "__dataclass_fields__", {}):
                v = getattr(parent, f, None)
                if type(v) is type(me):
                    names.append(f)
                elif isinstance(v, (list, tuple)) and v and all(type(x) is type(me) for x in v):
                    names.append(f + "[*]")
            if len(names) != 1:
                return None
            path = f"{up}.{names[0]}".lstrip(".")
        for p, x in leaves:
            self.leaves.setdefault(id(x), ((f".{path}" if path else "") + jax.tree_util.keystr(p), x))
        return path

    def __call__(self, fn: Any, args: Any, kwargs: Any, run: Any) -> Any:
        self.uid += 1
        uid, top = self.uid, self.depth == 0
        try:
            bound = dict(fn._sig.bind(*args, **kwargs).arguments)
        except TypeError:
            bound = {}
        if (first := next(iter(fn.params), None)) in ("self", "cls"):
            bound.pop(first, None)
        ins = [b for b in map(_brief, bound.values()) if b]
        if top:
            self.root = uid
            tracers = [x for x in _arrays((args, kwargs)) if isinstance(x, Tracer)]
            self.top = getattr(tracers[0], "_trace", None) if tracers else None
        if top and bound:  # mark the inputs, so the diagram can name them
            full = fn._sig.bind(*args, **kwargs)
            for k in [k for k in bound if _arrays(bound[k])]:  # static arguments are not inputs
                with jax.named_scope(f"expex-in.{k}"):
                    full.arguments[k] = _barrier(full.arguments[k])
                self.inputs[k] = _brief(bound[k]) or ""
            args, kwargs = full.args, full.kwargs
        if fn.name not in self.sources:
            try:
                self.sources[fn.name] = textwrap.dedent(inspect.getsource(fn.raw))
            except (OSError, TypeError):
                self.sources[fn.name] = ""
        path = self.path(fn, args)
        self.calls[uid] = dict(name=_short(fn.name), sig=fn.signature, loc=fn.location,
                               src=fn.name, ins=", ".join(ins), outs="", path=path)  # fmt: skip
        method = path is not None or next(iter(fn.params), None) == "self"
        if method:
            self.modules.append((path, args[0] if args else None))
        batched = self.stop and any(isinstance(x, _fn._BatchTracer) for x in _arrays(list(bound.values())))
        self.lifted[uid] = batched and not (self.batched and self.batched[-1])
        self.open.append(uid)
        self.batched.append(batched)
        self.depth += 1
        try:
            with jax.named_scope(f"{_short(fn.name)}#{uid}"):
                out = run(args, kwargs)
        except _Stop as s:
            self._carry(s, uid, bound)
            raise
        except ShapeError as e:
            if not self.stop or self.error is not None:
                raise
            self.error = dict(uid=uid, path=list(self.open), message=str(e), notes=list(getattr(e, "__notes__", ())))
            stop = _Stop()
            self._carry(stop, uid, bound)
            raise stop from None
        finally:
            self.depth -= 1
            self.open.pop()
            self.batched.pop()
            if method:
                self.modules.pop()
        self.calls[uid]["outs"] = _brief(out) or ""
        if top:
            with jax.named_scope("expex-out"):
                out = _barrier(out)
        return out

    def _carry(self, s: _Stop, uid: int, bound: dict[str, Any]) -> None:
        """Let ``s`` carry this call's arguments if it carries none yet and they are top-level."""
        if s.payload is not None:
            return
        xs = []
        for x in _arrays(list(bound.values())):
            while isinstance(x, _fn._BatchTracer):
                x = x.val  # the batched array, traced one level up
            if isinstance(x, Tracer) and getattr(x, "_trace", None) is not self.top:
                return  # inside a scan, cond or while: cannot leave it
            xs.append(x)
        s.payload, s.at = xs, uid


class _Builder:
    """Walks the jaxpr into boxes (calls, loops), atoms (equations, inputs) and edges."""

    def __init__(self, hook: _Hook, length: Any, flops: bool = False):
        self.hook, self.length = hook, length
        self.flops: list[float] | None = [] if flops else None  # per atom, when estimating time
        self.mult = 1  # trip count of the loops around the equations being walked
        self.fail = -1  # the failing call's box (see ``error``)
        self.error: dict[str, Any] | None = None
        self.boxes: list[dict[str, Any]] = []
        self.atoms: list[list[Any]] = []  # [box, label or None (passed through), kind, primitive]
        self.sums: dict[Any, tuple[int, int]] = {}  # a reduce_sum's result (through reshapes) -> (atom, count)
        self.within: list[str] = []  # the names of the jit-ed functions being walked (``_var``, ...)
        self.made: dict[Any, str] = {}  # each variable's primitive
        self.data: list[bool] = []  # depends on an input (else on constants and sizes only)
        self.near: list[frozenset[int]] = []  # the nearest drawn atoms each atom is computed from
        self.edges: list[tuple[int, int, int]] = []  # (from atom, to atom, type index)
        self.types: dict[str, int] = {}
        self.by_uid: dict[int, int] = {}

    def box(self, parent: int, kind: str, label: str, **info: Any) -> int:
        up = parent
        while up >= 0 and self.boxes[up]["k"] != "call":
            up = self.boxes[up]["p"]
        depth = (self.boxes[up]["d"] if up >= 0 else 0) + (kind == "call")
        self.boxes.append(dict(k=kind, l=label, p=parent, d=depth, **info))
        return len(self.boxes) - 1

    def call(self, parent: int, uid: int, lifted: bool) -> int:
        if uid not in self.by_uid:
            lifted = lifted or self.hook.lifted.get(uid, False)
            c = self.hook.calls.get(uid, dict(name="?", sig="", loc="", src="", ins="", outs=""))
            self.by_uid[uid] = self.box(parent, "call", c["name"], vectorize=lifted, **c)
        return self.by_uid[uid]

    def atom(self, box: int, label: str | None, kind: str, ins: list[Any], env: dict[Any, list[int]], eqn: Any = None) -> int:  # fmt: skip
        me = len(self.atoms)
        self.atoms.append([box, label, kind, eqn.primitive.name if eqn is not None else ""])
        if self.flops is not None:
            n = _cost.flops(types.SimpleNamespace(eqns=[eqn]), []).value if eqn is not None else 0
            self.flops.append(float(n or 0) * self.mult)
        self.data.append(kind == "in" or any(self.data[a] for v in ins for a in _get(env, v)))
        self.near.append(frozenset({me}) if label else frozenset().union(*[self.near[a] for v in ins for a in _get(env, v)]))
        for v in ins:
            if (t := _type(v.aval)) is not None:
                ti = self.types.setdefault(t, len(self.types))
                self.edges.extend((src, me, ti) for src in _get(env, v))
        return me

    def walk(self, jaxpr: Any, base: int, env: dict[Any, list[int]]) -> None:
        for eqn in jaxpr.eqns:
            stack, name, p = str(eqn.source_info.name_stack), eqn.primitive.name, eqn.params
            b = base
            for vectorize, short, uid in _TOKEN.findall(stack):
                if int(uid) != self.hook.root:
                    b = self.call(b, int(uid), bool(vectorize))
            srcs = lambda vs: [_get(env, v) for v in vs]
            outs: list[list[int]] | None = None
            if name == "optimization_barrier" and "expex-" in stack:
                kind, what = ("in", stack.split("expex-in.")[-1].split("/")[0]) if "expex-in." in stack else ("out", "")
                text = f"{what}: {self.hook.inputs.get(what, '')}" if kind == "in" else "return " + " ".join(
                    filter(None, (_type(v.aval) for v in eqn.outvars)))  # fmt: skip
                me = self.atom(-1, text, kind, eqn.invars if kind == "out" else [], env)
                outs = [[me]] * len(eqn.outvars)
            elif name == "scan":
                loop = self.box(b, "loop", f"× {self.length(p['length'])}")
                mult, self.mult = self.mult, self.mult * (p["length"] if isinstance(p["length"], int) else 1)
                outs = self.sub(p["jaxpr"], loop, srcs(eqn.invars))
                self.mult = mult
            elif name == "cond":
                box = self.box(b, "loop", "if" if len(p["branches"]) == 2 else "switch")
                self.atom(box, "condition", "op", eqn.invars[:1], env)
                names = ["else", "then"] if len(p["branches"]) == 2 else [f"case {i}" for i in range(len(p["branches"]))]
                results = [self.sub(br, self.box(box, "loop", n), srcs(eqn.invars[1:])) for n, br in zip(names, p["branches"])]  # fmt: skip
                outs = [sum(r, []) for r in zip(*results)]
            elif name == "while":
                box, nc, nb = self.box(b, "loop", "while"), p["cond_nconsts"], p["body_nconsts"]
                init = srcs(eqn.invars[nc + nb :])
                cj = p["cond_jaxpr"].jaxpr
                cenv = dict(zip(cj.invars, srcs(eqn.invars[:nc]) + init))
                self.walk(cj, box, cenv)
                self.atom(box, "test", "op", cj.outvars, cenv)
                body = self.sub(p["body_jaxpr"], box, srcs(eqn.invars[nc : nc + nb]) + init)
                outs = [a + c for a, c in zip(init, body)]
            elif (subs := [j for _, j in sub_jaxprs(p)]) and len(subs[0].invars) == len(eqn.invars):
                self.within.append(str(p.get("name", "")))
                outs = self.sub(subs[0], b, srcs(eqn.invars))  # jit, custom_jvp, remat, ...: inline
                self.within.pop()
            if outs is None:
                label = OPS.get(name)
                if name == "reduce_sum" and self.within and self.within[-1].lstrip("_") == "var":
                    label = "var"  # inside jnp.var (or std): the sum of squared deviations
                self.mean(eqn, env)
                if name == "gather" and not any(self.data[a] for a in _get(env, eqn.invars[1])):
                    label = None  # a slice by sizes, not a lookup
                if name == "add" and self.joins(eqn, env):
                    label = "+"  # two computed values of one shape: a residual or skip connection
                outs = [[self.atom(b, label, "op", eqn.invars, env, eqn)]] * len(eqn.outvars)
                if name == "reduce_sum" and label:
                    shape = getattr(eqn.invars[0].aval, "shape", ())
                    self.sums[eqn.outvars[0]] = (outs[0][0], math.prod(shape[a] for a in p["axes"]) if shape else 1)
            env.update({o: s for o, s in zip(eqn.outvars, outs) if _type(o.aval) is not None})
            self.made.update((o, name) for o in eqn.outvars)

    def mean(self, eqn: Any, env: dict[Any, list[int]]) -> None:
        """Relabel a ``sum`` divided by the number of elements it adds up (``jnp.mean``) as ``mean``."""
        name, ins = eqn.primitive.name, eqn.invars
        if not ins or type(ins[0]).__name__ == "Literal" or ins[0] not in self.sums:
            return
        if name in ("broadcast_in_dim", "reshape", "squeeze", "expand_dims", "convert_element_type"):
            self.sums[eqn.outvars[0]] = self.sums[ins[0]]
        elif name == "div" and len(ins) == 2:
            atom, n = self.sums[ins[0]]
            d = ins[1]
            if type(d).__name__ == "Literal":
                try:
                    count = float(d.val) == n
                except (TypeError, ValueError):
                    count = False
            else:  # a symbolic size: the count itself as a float, not a sum of it with something (n - ddof)
                count = (self.made.get(d) == "convert_element_type" and getattr(d.aval, "shape", None) == ()
                         and not any(self.data[a] for a in _get(env, d)))  # fmt: skip
            if count and self.atoms[atom][1] in ("sum", "var"):
                self.atoms[atom][1] = "mean"

    def joins(self, eqn: Any, env: dict[Any, list[int]]) -> bool:
        """Whether an ``add`` joins two branches: both depend on the inputs, have one shape, and neither is
        computed only from what the other is (``x + x**3`` in ``gelu`` is not drawn, ``x + attn(x)`` is)."""
        shapes = {getattr(v.aval, "shape", None) for v in [*eqn.invars, *eqn.outvars]}
        if len(shapes) != 1 or not next(iter(shapes)) or any(type(v).__name__ == "Literal" for v in eqn.invars):
            return False
        if not all(any(self.data[a] for a in _get(env, v)) for v in eqn.invars):
            return False
        a, b = (frozenset().union(*[self.near[x] for x in _get(env, v)]) for v in eqn.invars)
        return not (a <= b or b <= a)

    def sub(self, jaxpr: Any, box: int, ins: list[list[int]]) -> list[list[int]]:
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        env = dict(zip(jaxpr.invars, ins))
        self.walk(jaxpr, box, env)
        return [_get(env, v) for v in jaxpr.outvars]

    def stopped(self, outvars: list[Any], env: dict[Any, list[int]]) -> None:
        """Draw the call the trace stopped at, fed by what ``_Stop`` carried out (``outvars``)."""
        err = self.hook.error
        if err is None:
            return
        box = -1
        for uid in err["path"]:  # the calls open when it failed, outermost first
            if uid != self.hook.root:
                box = self.call(box, uid, False)
        label = None if box >= 0 else "✗ ShapeError"  # the root itself failed: a node of its own
        me = self.atom(box, label, "op", outvars, env)
        self.fail = box
        lines = err["message"].splitlines()
        grab = lambda word: next((ln.split(word, 1)[1].strip() for ln in lines if ln.strip().startswith(word)), "")
        got = next((grab(w) for w in ("received", "returned", "assigned") if grab(w)), "")
        self.error = dict(atom=me, box=box, message=err["message"], head=lines[0] if lines else "",
                          declared=grab("declared"), got=got, notes=err["notes"])  # fmt: skip
        if box >= 0:
            self.boxes[box]["err"] = 2
            up = self.boxes[box]["p"]
            while up >= 0:
                self.boxes[up]["err"] = 1  # contains the failing call
                up = self.boxes[up]["p"]

    def model(self, name: str, depth: int) -> dict[str, Any]:
        has = [False] * len(self.boxes)  # anything to show inside (else a call cannot open)
        marks = [a[0] for a in self.atoms if a[1] and a[2] == "op"] + [b["p"] for b in self.boxes if b["k"] == "call"]
        for i in marks:
            while i >= 0 and not has[i]:
                has[i], i = True, self.boxes[i]["p"]
        for i, b in enumerate(self.boxes):
            b["o"] = b["k"] == "call" and has[i] and i != self.fail  # the failing call stays closed
        types = sorted(self.types, key=self.types.__getitem__)
        return dict(name=name, depth=depth, boxes=self.boxes, atoms=self.atoms, edges=self.edges,
                    types=types, sources=self.hook.sources, timing=None, error=self.error)  # fmt: skip


def _amount(x: float, unit: str) -> str:
    """``1.23 ms`` for seconds, ``4.5 GFLOP`` for FLOPs."""
    if unit == "s":
        for scale, name in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
            if x >= scale:
                return f"{x / scale:.3g} {name}"
        return f"{x * 1e9:.3g} ns"
    for scale, name in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if x >= scale:
            return f"{x / scale:.3g} {name}FLOP"
    return f"{x:.3g} FLOP"


def heat(share: float) -> int:
    """0 (cold) to 4 (hottest): the step of ``HEAT`` a share of the total time reaches."""
    return sum(share >= h for h in HEAT)


def _timing(m: dict[str, Any], b: dict[str, Any]) -> str:
    """`` · 1.23 ms · 45.6%`` for a timed box, else ''."""
    t = m.get("timing")
    if not t or "t" not in b:
        return ""
    return f"{_amount(b['t'], t['unit'])} · {b['s']:.1%}"


def _label(m: dict[str, Any], b: dict[str, Any]) -> str:
    """A closed call's node: name, types, and its time or what failed in it."""
    lines = [("✗ " if b.get("err") else "") + b["l"] + (" · vectorized" if b["vectorize"] else "")]
    err = m.get("error")
    if b.get("err") == 2 and err and err["got"]:  # argument (or local) received vs declared, or the result
        arg = "return value" not in err["head"]
        lines.append(f"received {err['got']}" if arg else f"{b['ins']} → {err['got']}")
        lines.append(f"declared {err['declared']}")
    else:
        lines.append(f"{b['ins']} → {b['outs'] or '?'}")
        if b.get("err") == 1:
            lines.append("fails inside")
    if t := _timing(m, b):
        lines.append(t)
    return "\n".join(lines)


def _view(m: dict[str, Any], expanded: set[int], full: bool = False) -> tuple[list[tuple[str, str, str, int]], list[Any], list[int], str | None]:  # fmt: skip
    """The visible graph: nodes (id, kind, label, parent box), edges (from, to, type), drawn boxes,
    and the node that failed (for ``explain``), if any.

    Atoms inside a closed call become that call's node; unlabelled atoms elsewhere
    are passed through. An edge is labelled with the type of the array as it
    leaves its source, or as it enters its target if that is a call (whose
    label says what it takes). With ``full``, edges also carry both types.
    """
    boxes, atoms = m["boxes"], m["atoms"]
    rep: list[int | None] = []  # the closed call that swallows each box
    for i, b in enumerate(boxes):
        up = rep[b["p"]] if b["p"] >= 0 else None
        rep.append(up if up is not None else i if b["k"] == "call" and not (b["o"] and i in expanded) else None)
    nodes: dict[str, tuple[str, str, str, int]] = {}
    key: list[str] = []
    for i, (box, label, kind, *_) in enumerate(atoms):
        r = rep[box] if box >= 0 else None
        if r is not None:
            key.append(f"b{r}")
            nodes.setdefault(key[-1], (key[-1], "call", _label(m, boxes[r]), boxes[r]["p"]))
        elif label:
            key.append(f"n{i}")
            nodes[key[-1]] = (key[-1], kind, label, box)
        else:
            key.append(f"h{i}")
    _elementwise(m, key, nodes)
    succ: dict[str, list[tuple[str, int]]] = {}
    for a, b, t in m["edges"]:
        if key[a] != key[b]:
            succ.setdefault(key[a], []).append((key[b], t))
    edges: dict[tuple[str, str], tuple[str, str, str]] = {}  # (shown, leaving u, reaching v)
    for u in nodes:
        seen, todo = {u}, [(v, t, t) for v, t in reversed(succ.get(u, []))]
        while todo:  # (node, type leaving u, type reaching the node)
            v, t0, t = todo.pop()
            if v in seen:
                continue
            seen.add(v)
            if v in nodes:
                edges.setdefault((u, v), (m["types"][t if nodes[v][1] == "call" else t0], m["types"][t0], m["types"][t]))
            else:
                todo.extend((w, t0, tw) for w, tw in reversed(succ.get(v, [])))
    drawn: set[int] = set()
    for _, _, _, box in nodes.values():
        while box >= 0 and box not in drawn:
            drawn.add(box)
            box = boxes[box]["p"]
    fail = key[m["error"]["atom"]] if m.get("error") else None
    if full:  # (from, to, type, type leaving the source, type entering the target)
        return list(nodes.values()), [(u, v, *t) for (u, v), t in edges.items()], sorted(drawn), fail
    return list(nodes.values()), [(u, v, t[0]) for (u, v), t in edges.items()], sorted(drawn), fail


#: Short names for the operations an ``elementwise`` node stands for; others (reshapes, ...) go unnamed.
_MATH = {"sub": "sub", "div": "div", "mul": "mul", "add": "add", "neg": "neg", "abs": "abs", "max": "max",
         "min": "min", "integer_pow": "pow", "pow": "pow", "square": "square", "sqrt": "sqrt", "rsqrt": "rsqrt",
         "exp": "exp", "log": "log", "log1p": "log1p", "tanh": "tanh", "logistic": "sigmoid", "erf": "erf",
         "select_n": "where", "sign": "sign", "exp2": "exp2", "sin": "sin", "cos": "cos"}  # fmt: skip


def _elementwise(m: dict[str, Any], key: list[str], nodes: dict[str, tuple[str, str, str, int]]) -> None:
    """Make the hidden operations that join values from different nodes visible, as ``elementwise`` nodes.

    Hidden operations are passed through, which is right for a chain (``gelu(x @ w)``) but loses the
    dataflow where two values meet: ``(x - mean) / std`` would make ``x``, ``mean`` and ``std`` each
    feed what follows. A hidden atom whose inputs come from different nodes is a join; a join and
    the hidden atoms after it (up to the next node) become one node, and joins whose runs meet are
    one node too. Changes ``key`` and ``nodes`` in place.
    """
    atoms = m["atoms"]
    preds: dict[int, list[int]] = defaultdict(list)
    for a, b, _ in m["edges"]:
        preds[b].append(a)
    hidden = lambda i: key[i][0] == "h"
    seen: list[frozenset[str]] = []  # the nodes each atom's value comes from (itself if it is one)
    joins = []
    for i in range(len(atoms)):
        sets = [seen[a] for a in preds[i]]
        seen.append(frozenset({key[i]}) if not hidden(i) else frozenset().union(*sets))
        if hidden(i) and len(set(sets)) >= 2 and len(seen[i]) >= 2:
            joins.append(i)
    if not joins:
        return
    succ: dict[int, list[int]] = defaultdict(list)
    for a, b, _ in m["edges"]:
        succ[a].append(b)
    parent = {j: j for j in joins}

    def find(j: int) -> int:
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = parent[j]
        return j

    owner: dict[int, int] = {}  # hidden atom -> a join whose run it is in
    for j in joins:
        todo = [j]
        while todo:
            a = todo.pop()
            if a in owner:
                if find(owner[a]) != find(j):
                    parent[find(owner[a])] = find(j)
                continue
            owner[a] = j
            todo += [b for b in succ[a] if hidden(b) and atoms[b][0] == atoms[j][0]]
    runs: dict[int, list[int]] = defaultdict(list)
    for a, j in owner.items():
        runs[find(j)].append(a)
    for root, members in runs.items():
        members.sort()
        names = list(dict.fromkeys(_MATH[atoms[a][3]] for a in members if len(atoms[a]) > 3 and atoms[a][3] in _MATH))
        text = " · ".join(names[:4]) + (" …" if len(names) > 4 else "")
        nid = f"e{members[0]}"
        for a in members:
            key[a] = nid
        nodes[nid] = (nid, "op", "elementwise" + (f"\n{text}" if text else ""), atoms[members[0]][0])
    order: dict[str, int] = {}  # nodes in the order their first atom comes
    for i, k in enumerate(key):
        order.setdefault(k, i)
    items = sorted(nodes.items(), key=lambda kv: order.get(kv[0], 0))
    nodes.clear()
    nodes.update(items)


def _feeds(edges: list[tuple[str, str, str]], node: str | None) -> tuple[set[str], set[int]]:
    """The nodes from which ``node`` is reached, and the indices of the edges between them."""
    if node is None:
        return set(), set()
    seen, todo = {node}, [node]
    while todo:
        v = todo.pop()
        for u, w, _ in edges:
            if w == v and u not in seen:
                seen.add(u)
                todo.append(u)
    return seen - {node}, {i for i, (u, w, _) in enumerate(edges) if w in seen and u in seen}


def _opened(m: dict[str, Any]) -> set[int]:
    """The calls open at the model's depth: those shallower, and those around a failing call."""
    return {i for i, b in enumerate(m["boxes"]) if b["k"] == "call" and (b["d"] < m["depth"] or b.get("err") == 1)}


def _outline(m: dict[str, Any]) -> str:
    """The calls and loops as an indented outline, with times or the failure."""
    boxes, t, err = m["boxes"], m.get("timing"), m.get("error")
    kids: dict[int, list[int]] = {}
    for i, b in enumerate(boxes):
        kids.setdefault(b["p"], []).append(i)
    rows: list[tuple[str, str, str]] = []

    def walk(box: int, ind: str) -> None:
        for i in kids.get(box, []):
            b = boxes[i]
            if b["k"] == "loop" and b["l"] in ("then", "else") and not kids.get(i):
                continue
            name = ind + b["l"] + (" · vectorized" if b.get("vectorize") else "")
            if b["k"] == "call":
                what = f"{b['ins']} → {b['outs'] or '?'}"
            else:
                what = ""
            if t and "t" in b:
                bar = "█" * round(b["s"] * 20)
                what = f"{_amount(b['t'], t['unit']):>10}  {b['s']:6.1%}  {bar}"
            if b.get("err") == 2 and err:
                verb = "returned" if "return value" in err["head"] else "received"
                what = f"✗ {verb} {err['got']}, declared {err['declared']}" if err["got"] else f"✗ {err['head']}"
            elif b.get("err") == 1:
                what += "   ✗ fails inside"
            rows.append((name, what, ""))
            walk(i, ind + "  ")

    walk(-1, "  ")
    head = m["name"]
    if t:
        if t["unit"] == "s":
            head += (f"   measured with jax.profiler ({t['backend']}): {_amount(t['total'], 's')} of XLA op time"
                     f" per call (summed over threads), {_amount(t['wall'], 's')} wall, median of {t['runs']} runs")  # fmt: skip
        else:
            head += f"   ESTIMATED from FLOPs, not measured ({t['why']}): {_amount(t['total'], 'flop')} per call"
    elif err:
        head += "   ShapeError: traced up to the failing call"
    if m.get("fixed"):  # f32[R:6]: R is shown as 6 in the types, so say which sizes are fixed
        types = [e[2] for e in _view(m, _opened(m))[1]]
        head += "   sizes " + " · ".join(
            [n for n in _names(types) if n not in m["fixed"]] + [f"{k} (= {v})" for k, v in m["fixed"].items()]
        )
    width = max([len(r[0]) for r in rows] + [len("  (outside the calls)") if t else 0]) + 2
    lines = [head] + [f"{name:<{width}}{what}".rstrip() for name, what, _ in rows]
    if t and t.get("outside"):
        lines.append(f"{'  (outside the calls)':<{width}}{_amount(t['outside'], t['unit']):>10}  {t['outside'] / t['total']:6.1%}")  # fmt: skip  # noqa: E501
    if err:
        nodes, edges, _, fail = _view(m, _opened(m))
        feeds, _ = _feeds(edges, fail)
        names = {n[0]: n[2].split("\n")[0] for n in nodes}
        order = [n[0] for n in nodes if n[0] in feeds or n[0] == fail]
        lines += ["", "  path:  " + " → ".join(names[n] for n in order), ""]
        lines += ["  " + ln if ln else "" for ln in err["message"].splitlines()]
    return "\n".join(lines)


# ------------------------------------------------------------------ drawing: colours, fonts, display list


def _mix(color: str, p: float, base: str) -> str:
    """``color`` at ``p`` over ``base`` (TikZ's ``color!p``), as hex."""
    a, b = (tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in (color, base))
    return "#" + "".join(f"{round(x * p + y * (1 - p)):02X}" for x, y in zip(a, b))


_INK = {"bg": ("#FFFFFF", "#151519"), "ink": ("#16161A", "#ECECEF"), "ink2": ("#55555C", "#A4A4AC"),
        "axis": ("#B8B8BF", "#4C4C55"), "wash": ("#F2F2F4", "#232329"), "panel2": ("#FAFAFB", "#1A1A1F")}  # fmt: skip
#: The streams' hues (light; dark mode lightens them): blue, violet, green, magenta; muted for a box
#: several streams reach, orange for side inputs (conditioning), red for errors, amber for time.
_HUES = {"s0": "#2A78D6", "s1": "#6250D6", "s2": "#1BAF7A", "s3": "#C2418F", "mix": "#909098",
         "side": "#EB6834", "err": "#D43A3A"}  # fmt: skip
_AMBER = "#F29D0C"


def _palette() -> dict[str, tuple[str, str]]:
    """Colour tokens, light then dark: each hue as a stroke, a box fill (12%) and a chip fill (6%)."""
    out = dict(_INK)
    white, dark = _INK["bg"]
    for k, hue in _HUES.items():
        lit = _mix(hue, 0.78, "#FFFFFF")  # lighter on a dark background
        out[k] = (hue, lit)
        out[k + "f"] = (_mix(hue, 0.12, white), _mix(lit, 0.2, dark))
        out[k + "c"] = (_mix(hue, 0.06, white), _mix(lit, 0.12, dark))
    out["mixf"], out["mixc"] = out["wash"], (white, _mix("#FFFFFF", 0.04, dark))
    for j, (p, q) in enumerate([(0.0, 0.0), (0.12, 0.16), (0.28, 0.3), (0.5, 0.48), (0.78, 0.7)]):
        out[f"h{j}"] = (_mix(_AMBER, p, white), _mix(_AMBER, q, dark))  # boxes by their share of the time
        out[f"p{j}"] = (_mix(_AMBER, p / 8, white), _mix(_AMBER, q / 6, dark))  # panels, fainter
    return out


COLORS = _palette()
_SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
_MONO = "'JetBrains Mono', 'DejaVu Sans Mono', Menlo, Consolas, monospace"
#: Text styles: size (px), weight, italic, monospace, colour token.
FONTS = {
    "tb": (13, 600, False, False, "ink"), "ts": (11.5, 400, False, False, "ink2"),
    "ti": (13.5, 400, True, False, "ink"), "tsh": (11.5, 400, False, False, "ink2"),
    "tp": (12, 600, False, False, "ink"), "tpm": (12, 400, False, False, "ink2"),
    "th": (13, 600, False, False, "ink"), "tl": (12, 400, False, False, "ink2"),
    "te": (12, 600, False, False, "err"), "tem": (10.5, 400, False, False, "err"),
    "tred": (12, 400, False, False, "err"), "tlab": (10, 400, False, False, "ink2"),
    "tplus": (12, 400, False, False, "ink"),
}  # fmt: skip
# Inter's advance widths (1/100 em) for ' ' to '~', regular and semi-bold: boxes are sized to their text.
_W = [int(w) / 100 for w in (
    "28 28 46 63 64 98 64 30 36 36 50 66 28 45 28 36 63 40 60 61 64 60 62 56 62 62 28 30 66 66 66 51 96 69 65 73 "
    "72 60 59 74 74 26 57 67 56 90 75 76 63 76 64 64 64 74 69 98 68 67 63 36 36 36 47 45 32 56 61 57 61 58 37 61 "
    "59 24 24 54 24 87 59 60 61 61 37 52 32 59 56 81 54 56 55 42 33 42 66").split()]
_WB = [int(w) / 100 for w in (
    "25 32 52 64 64 100 66 32 37 37 54 67 31 46 31 37 65 42 62 63 66 62 63 57 64 63 31 32 67 67 67 54 99 72 65 "
    "73 72 60 58 74 74 27 58 70 56 92 75 76 64 77 65 64 66 73 72 102 72 71 65 37 37 37 48 46 35 57 62 58 62 59 38 "
    "62 61 26 26 56 26 89 61 60 62 62 39 54 35 61 58 83 56 58 56 45 35 45 67").split()]
_WIDE = {"→": 0.96, "×": 0.67, "·": 0.32, "✗": 0.88, "‖": 0.5, "…": 0.7, "≤": 0.67, "≥": 0.67, "µ": 0.6, "⊕": 0.9}


def _tw(text: str, cls: str) -> float:
    """The width of ``text`` in the text style ``cls`` (estimated from Inter's metrics, 0.6 em when monospace)."""
    size, weight, _, mono, _ = FONTS[cls]
    if mono:
        return 0.6 * size * len(text)
    table = _WB if weight >= 600 else _W
    return size * sum(table[ord(c) - 32] if 32 <= ord(c) < 127 else _WIDE.get(c, 0.75) for c in text)


Run = tuple[str, str]  # (text, style)


class _Canvas:
    """A display list: rectangles, text, polylines and arrow heads in one coordinate system.

    ``_svg`` writes it as SVG; ``_png_matplotlib`` draws the same list with matplotlib.
    """

    def __init__(self) -> None:
        self.ops: list[tuple[Any, ...]] = []

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str | None = "ink2",
             sw: float = 1.0, r: float = 3.0, dash: str | None = None) -> None:  # fmt: skip
        self.ops.append(("rect", x, y, w, h, fill, stroke, sw, r, dash))

    def text(self, x: float, y: float, runs: list[Run], anchor: str = "middle", halo: str | None = None) -> None:
        self.ops.append(("text", x, y, runs, anchor, halo))

    def line(self, pts: list[tuple[float, float]], stroke: str = "ink2", sw: float = 1.0, dash: str | None = None,
             radius: float = 4.0) -> None:  # fmt: skip
        self.ops.append(("line", pts, stroke, sw, dash, radius))

    def head(self, x: float, y: float, dx: float, dy: float, color: str = "ink2", size: float = 5.0) -> None:
        """A filled arrow head with its tip at (x, y), pointing along (dx, dy)."""
        self.ops.append(("head", x, y, dx, dy, color, size))

    def group(self, **attrs: Any) -> None:
        self.ops.append(("g", attrs))

    def end(self) -> None:
        self.ops.append(("/g",))


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _rounded(pts: list[tuple[float, float]], r: float) -> str:
    """An SVG path through ``pts`` with its corners rounded by up to ``r``."""
    d = f"M{_num(pts[0][0])},{_num(pts[0][1])}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        a, b = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
        k = min(r, a / 2, b / 2)
        if k < 0.5:
            d += f"L{_num(x1)},{_num(y1)}"
            continue
        p = (x1 - (x1 - x0) / a * k, y1 - (y1 - y0) / a * k)
        q = (x1 + (x2 - x1) / b * k, y1 + (y2 - y1) / b * k)
        d += f"L{_num(p[0])},{_num(p[1])}Q{_num(x1)},{_num(y1)} {_num(q[0])},{_num(q[1])}"
    return d + f"L{_num(pts[-1][0])},{_num(pts[-1][1])}"


def _css() -> str:
    """The SVG's style sheet: the colour variables for light and dark, colour classes over them, text styles.

    Dark follows ``prefers-color-scheme``; ``data-theme="light"`` or ``"dark"`` on the SVG or on an
    element around it (the page's ``<html>``) overrides it.
    """
    light = ";".join(f"--xd-{k}:{v[0]}" for k, v in COLORS.items())
    dark = ";".join(f"--xd-{k}:{v[1]}" for k, v in COLORS.items())
    out = [f"svg.expex-diagram{{{light}}}",
           f"@media (prefers-color-scheme: dark){{svg.expex-diagram{{{dark}}}}}",
           f"[data-theme=light] svg.expex-diagram,svg.expex-diagram[data-theme=light]{{{light}}}",
           f"[data-theme=dark] svg.expex-diagram,svg.expex-diagram[data-theme=dark]{{{dark}}}"]  # fmt: skip
    out += [f".f-{k}{{fill:var(--xd-{k})}}.k-{k}{{stroke:var(--xd-{k})}}" for k in COLORS]
    for cls, (size, weight, italic, mono, color) in FONTS.items():
        out.append(f".{cls}{{font-family:{_MONO if mono else _SANS};font-size:{size}px;font-weight:{weight};"
                   f"{'font-style:italic;' if italic else ''}fill:var(--xd-{color})}}")  # fmt: skip
    out.append("text{white-space:pre}.halo{paint-order:stroke;stroke-width:4px;stroke-linejoin:round}")
    return "".join(out)


def _svg(c: _Canvas, w: float, h: float) -> str:
    """The display list as a standalone SVG document (light or dark with the viewer's colour scheme)."""
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_num(w)} {_num(h)}" width="{_num(w)}" '
           f'height="{_num(h)}" class="expex-diagram"><style>{_css()}</style>',
           f'<rect class="f-bg" x="0" y="0" width="{_num(w)}" height="{_num(h)}"/>']  # fmt: skip
    esc = lambda s: html.escape(s, quote=False)
    for op in c.ops:
        kind = op[0]
        if kind == "rect":
            _, x, y, rw, rh, fill, stroke, sw, r, dash = op
            cls = f"f-{fill}" + (f" k-{stroke}" if stroke else "")
            extra = (f' stroke-width="{sw}"' if stroke else "") + (f' stroke-dasharray="{dash}"' if dash else "")
            out.append(f'<rect class="{cls}" x="{_num(x)}" y="{_num(y)}" width="{_num(rw)}" height="{_num(rh)}" '
                       f'rx="{_num(r)}"{extra}/>')  # fmt: skip
        elif kind == "text":
            _, x, y, runs, anchor, halo = op
            cls = runs[0][1] + (f" halo k-{halo}" if halo else "")
            body = esc(runs[0][0]) + "".join(f'<tspan class="{c2}">{esc(t)}</tspan>' for t, c2 in runs[1:])
            out.append(f'<text class="{cls}" x="{_num(x)}" y="{_num(y)}" text-anchor="{anchor}">{body}</text>')
        elif kind == "line":
            _, pts, stroke, sw, dash, radius = op
            extra = f' stroke-dasharray="{dash}"' if dash else ""
            out.append(f'<path class="k-{stroke}" fill="none" stroke-width="{sw}"{extra} d="{_rounded(pts, radius)}"/>')
        elif kind == "head":
            out.append(f'<polygon class="f-{op[5]}" points="{" ".join(f"{_num(x)},{_num(y)}" for x, y in _head(op))}"/>')
        elif kind == "g":
            attrs = "".join(f' {k.rstrip("_").replace("_", "-")}="{html.escape(str(v))}"' for k, v in op[1].items())
            out.append(f"<g{attrs}>")
        else:
            out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


def _head(op: tuple[Any, ...]) -> list[tuple[float, float]]:
    """The points of an arrow head: a small stealth shape, tip first."""
    _, x, y, dx, dy, _, size = op
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    bx, by = x - ux * size, y - uy * size
    px, py = -uy * size * 0.45, ux * size * 0.45
    ix, iy = x - ux * size * 0.72, y - uy * size * 0.72  # the notch
    return [(x, y), (bx + px, by + py), (ix, iy), (bx - px, by - py)]


_DTYPE = re.compile(r"\b(f16|bf16|f32|f64|i8|i16|i32|i64|u8|u32|bool|c64|c128)(?:\[([^\]]*)\])?")


def _shape(t: str) -> str:
    """``T × D`` for ``f32[T, D]``; other dtypes are named (``T (i32)``), a 0-d array is ``scalar``."""
    def one(m: re.Match[str]) -> str:
        dims = " × ".join(d.strip() for d in m.group(2).split(",")) if m.group(2) else "scalar"
        return dims if m.group(1) == "f32" else f"{dims} ({m.group(1)})"

    return _DTYPE.sub(one, t)


# ------------------------------------------------------------------ drawing: the scene


def _display(name: str) -> str:
    """``Block`` for ``Block.__call__``; other names as they are."""
    return name[: -len(".__call__")] if name.endswith(".__call__") else name


_DIMS = re.compile(r"\[([^\]]*)\]")


def _names(types: list[str]) -> list[str]:
    """The size names in array types, in order: ``['T', 'D']`` from ``f32[T, D // 4]``."""
    out: list[str] = []
    for t in types:
        for inside in _DIMS.findall(t):
            out += [n for n in re.findall(r"[A-Za-z_]\w*", inside) if n not in out]
    return out


class _Scene:
    """One view of a diagram (the calls in ``expanded`` open), laid out and painted.

    The main flow is laid out container by container, inside out: the root, each
    open call and each loop or branch is a container whose children (nodes and
    inner containers, each a rectangle by then) are placed in layers by
    dependency depth, ordered to cross few edges (barycentre sweeps), and given
    x positions that keep edges straight (an order-preserving least-squares fit).
    An edge that crosses a container's border goes through a port on its
    leading (or trailing) edge, which takes part in the container's layout like
    a node; an edge that skips layers goes through one dummy per layer, shared
    by all edges from the same value, so fan-out draws as a tree, except a
    residual into a ``+``, which loops over the row. The layout's x runs across
    the flow and y along it; it is drawn transposed, left to right.
    """

    PAD, TOP, GAP, LANE, LANES = 12.0, 36.0, 26.0, 5.0, 24.0

    def __init__(self, m: dict[str, Any], expanded: set[int]):
        self.m, self.expanded = m, set(expanded)
        self.B = m["boxes"]
        nodes, edges, drawn, fail = _view(m, self.expanded, full=True)
        self.nodes = {n[0]: n for n in nodes}
        self.order = [n[0] for n in nodes]
        self.edges = edges
        self.fail = fail
        self.feeds, path = _feeds([(u, v, t) for u, v, t, *_ in edges], fail)
        self.path = {(edges[i][0], edges[i][1]) for i in path}
        self._containers(drawn)
        self._roles()

    # -- structure

    def _containers(self, drawn: list[int]) -> None:
        B = self.B
        members: dict[int, list[str]] = {i: [] for i in drawn}  # what each drawn box directly holds
        for n in self.order:
            if self.nodes[n][3] >= 0:
                members[self.nodes[n][3]].append(n)
        for i in drawn:
            if B[i]["p"] >= 0:
                members[B[i]["p"]].append(f"c{i}")
        self.card: dict[str, int] = {}  # a closed call alone in a loop: drawn as a stack of cards
        self.alias: dict[int, int] = {}  # an open call alone in a loop: one panel, "Name × L"
        for i in drawn:
            if B[i]["k"] == "loop" and B[i]["l"].startswith("×") and len(members[i]) == 1:
                (only,) = members[i]
                if only[0] == "b":
                    self.card[only] = i
                elif only[0] == "c" and B[int(only[1:])]["k"] == "call":
                    self.alias[int(only[1:])] = i
        self.conts = [i for i in drawn if i not in self.card.values() and i not in self.alias]
        self.cparent: dict[int, int] = {}
        for i in self.conts:
            self.cparent[i] = self._up(self._raw_parent(i))
        self.ncont = {n: self._up(self.nodes[n][3]) for n in self.order}

    def _raw_parent(self, i: int) -> int:
        p = self.B[i]["p"]
        if p in self.alias.values():  # the loop merged into this call: skip it
            p = self.B[p]["p"]
        return p

    def _up(self, b: int) -> int:
        """The container drawn for box ``b``: itself, or the nearest enclosing one (-1: the root)."""
        while b >= 0:
            if b in self.alias:
                return self.alias[b]
            if b in self.conts or (b not in self.card.values() and b in self.cparent):
                return b
            b = self.B[b]["p"]
        return -1

    def chain(self, c: int) -> list[int]:
        out = [c]
        while c != -1:
            c = self.cparent[c]
            out.append(c)
        return out

    # -- roles and labels

    def _roles(self) -> None:
        """Each node's colour: the input stream that reaches it, ``mix`` if several do, ``side`` for side inputs."""
        kinds = {n: self.nodes[n][1] for n in self.order}
        succ: dict[str, list[str]] = defaultdict(list)
        for u, v, *_ in self.edges:
            succ[u].append(v)

        def reach(n: str) -> set[str]:
            seen, todo = {n}, [n]
            while todo:
                for w in succ[todo.pop()]:
                    if w not in seen:
                        seen.add(w)
                        todo.append(w)
            return seen

        ins = [n for n in self.order if kinds[n] == "in"]
        reached = {n: reach(n) for n in ins}
        primary = max(ins, key=lambda n: len(reached[n])) if ins else None
        self.side = {n for n in ins if n != primary and len(succ[n]) >= 2}
        streams = [n for n in ins if n not in self.side]
        self.stream = {n: f"s{min(k, 3)}" for k, n in enumerate(streams)}
        self.role: dict[str, str] = {}
        for n in self.order:
            by = {s for s in streams if n in reached[s]}
            if n in self.side or (not by and any(n in reached[s] for s in self.side)):
                role = "side"
            elif len(by) == 1:
                role = self.stream[next(iter(by))]
            else:
                role = "mix"
            self.role[n] = role
        self.sidenodes = {n for n in self.order if self.role[n] == "side"}

    def outtypes(self, n: str) -> list[str]:
        """The types of the values a node produces, as its outgoing edges carry them."""
        out = []
        for u, _, _, t0, _ in self.edges:
            if u == n and t0 not in out:
                out.append(t0)
        if not out:
            kind, label = self.nodes[n][1], self.nodes[n][2]
            if kind == "out":
                out = [label.removeprefix("return ")]
            elif kind == "in":
                out = [label.split(": ", 1)[-1]]
            elif kind == "call":
                b = self.B[int(n[1:])]
                out = [b["outs"]] if b["outs"] and b.get("err") != 2 else []
        return out

    def heat_of(self, i: int) -> str | None:
        b = self.B[i]
        return f"h{heat(b['s'])}" if self.m.get("timing") and "s" in b else None

    # -- node and container sizes

    def lines(self, n: str) -> list[list[Run]]:
        """A node's text: a bold title, then grey lines (what it produces, how long it took, what failed)."""
        kind, label = self.nodes[n][1], self.nodes[n][2]
        shape = _shape(", ".join(self.outtypes(n)))
        if kind == "in":
            return [[(label.split(": ", 1)[0], "ti")], [(shape, "tsh")]]
        if kind == "out":
            return [[("return", "ti")], [(shape, "tsh")]]
        if kind == "op":
            if n == self.fail:
                return [[(label, "te")]]
            if label.startswith("elementwise"):
                ops = label.split("\n", 1)[1:] or [shape]
                return [[("elementwise", "tb")], [(ops[0], "ts")]]
            return [[(label, "tb")]] + ([[(shape, "ts")]] if shape else [])
        i = int(n[1:])
        b, err = self.B[i], self.m.get("error")
        name = _display(b["l"])
        if b.get("err") == 2 and err:
            out = [[("✗ " + name, "te")]]
            if err["got"]:
                arg = "return value" not in err["head"]
                out += [[(f"received {_shape(err['got'])}" if arg else f"returned {_shape(err['got'])}", "tem")],
                        [(f"declared {err['declared']}", "tem")]]  # fmt: skip
            return out
        first: list[Run] = [(name, "tb")]
        if n in self.card:
            first.append(("  " + self.B[self.card[n]]["l"], "tpm"))
        out = [first]
        sub = ("per element · " if b.get("vectorize") else "") + shape
        if sub:
            out.append([(sub, "ts")])
        if t := _timing(self.m, b):
            out.append([(t, "ts")])
        return out

    @staticmethod
    def _lh(line: list[Run]) -> float:
        return 16.0 if line[0][1] in ("tb", "te", "ti") else 14.0

    def node_size(self, n: str) -> tuple[float, float, float]:
        """(height, width, 0): the layout runs across, then along the flow (see ``place``)."""
        kind = self.nodes[n][1]
        if kind == "op" and self.nodes[n][2] == "+" and n != self.fail:
            return 17.0, 17.0, 0.0
        lines = self.lines(n)
        w = max(sum(_tw(t, c) for t, c in line) for line in lines)
        h = sum(self._lh(line) for line in lines)
        if kind in ("in", "out"):
            return h + 2, w + 4, 0.0
        w, h = max(w + 18, 52.0), max(h + 12, 34.0)
        if n in self.card:
            w, h = w + 4, h + 4
        return h, w, 0.0

    def title(self, c: int) -> list[Run]:
        b = self.B[c]
        if c in self.alias.values():
            inner = next(k for k, v in self.alias.items() if v == c)
            runs = [(_display(self.B[inner]["l"]) + "  " + b["l"], "tp")]
            b = self.B[inner]
        elif b["k"] == "call":
            runs = [(_display(b["l"]), "tp")]
        else:
            runs = [(("loop  " if b["l"].startswith("×") else "") + b["l"], "tp")]
        if b.get("vectorize"):
            runs.append(("  per element", "tpm"))
        if self.m.get("timing") and "s" in self.B[c]:
            runs.append(("  " + _timing(self.m, self.B[c]), "tpm"))
        if self.B[c].get("err") == 1:
            runs.append(("  ✗ fails inside", "tred"))
        return runs

    # -- layout

    def layout(self) -> None:
        self.size: dict[str, tuple[float, float, float]] = {}
        self.at: dict[int, dict[str, tuple[float, float]]] = {}
        self.items: dict[int, list[str]] = defaultdict(list)
        self.E: dict[int, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
        self.segs: dict[int, dict[tuple[str, str], list[tuple[float, float]]]] = {}
        self.layer: dict[int, dict[str, int]] = {}
        self.skipped: set[tuple[str, str]] = set()
        for n in self.order:
            self.items[self.ncont[n]].append(n)
            self.size[n] = self.node_size(n)
        seen: set[int] = set()
        for n in self.order:  # containers in the order their first node appears
            ch = self.chain(self.ncont[n])
            for a, b in zip(ch, ch[1:]):
                if a not in seen:
                    seen.add(a)
                    self.items[b].append(f"c{a}")
        for i in self.conts:  # containers without nodes (an empty branch) are left out
            self.items.setdefault(i, [])
        for u, v, label, t0, t in self.edges:
            cu, cv = self.chain(self.ncont[u]), self.chain(self.ncont[v])
            lca = next(c for c in cu if c in cv)
            info = dict(label=t if (t != t0 and t) else None, path=(u, v) in self.path, side=u in self.sidenodes)
            for c in cu[: cu.index(lca)]:
                self._add(c, (u, "o:" + u), info, v)
            self._add(lca, (u, self.item_of(v, lca)), info, v)
            for c in cv[: cv.index(lca)]:
                self._add(c, (u, self.item_of(v, c)), info, v)
        self.origin: dict[int, tuple[float, float]] = {}
        self.hints: dict[int, dict[str, float]] = {}
        self.place(-1)
        self.absolute(-1, 0.0, 0.0)
        if any(p.startswith("i:") for c in self.conts for p in self.at.get(c, {})):
            # again, with each container's entry ports in the order their values arrive from outside
            for c in self.conts:
                self.hints[c] = {p: self._centre(p[2:]) for p in self.at.get(c, {}) if p.startswith("i:")}
            self.skipped.clear()
            self.place(-1)
            self.absolute(-1, 0.0, 0.0)

    def _add(self, c: int, key: tuple[str, str], info: dict[str, Any], v: str) -> None:
        e = self.E[c].setdefault(key, dict(label=None, path=False, side=True, final=False))
        e["path"] |= info["path"]
        e["side"] &= info["side"]
        if key[1] == v:
            e["final"] = True
            e["label"] = info["label"]

    def item_of(self, n: str, c: int) -> str:
        """The item of container ``c`` that holds node ``n``: the node itself or a child container."""
        ch = self.chain(self.ncont[n])
        if ch[0] == c:
            return n
        return f"c{ch[ch.index(c) - 1]}"

    def src_of(self, u: str, c: int) -> str:
        ch = self.chain(self.ncont[u])
        return self.item_of(u, c) if c in ch else "i:" + u

    def place(self, c: int) -> tuple[float, float]:
        """Lay out container ``c`` (after its children): positions relative to its top left, and its size."""
        for it in self.items[c]:
            if it[0] == "c":
                self.size[it] = (*self.place(int(it[1:])), 0.0)
        panel = c >= 0
        items = list(self.items[c])
        edges = self.E[c]
        src = {key: self.src_of(key[0], c) for key in edges}
        ins = list(dict.fromkeys(s for s in src.values() if s.startswith("i:")))
        outs = list(dict.fromkeys(t for _, t in edges if t.startswith("o:")))
        for p in ins + outs:
            self.size[p] = (0.0, 0.0, 0.0)
        # layers: longest path from the top, inputs pinned to it, sources without inputs pulled down
        pos = {it: k for k, it in enumerate(items)}
        succ: dict[str, set[str]] = defaultdict(set)
        pred: dict[str, set[str]] = defaultdict(set)
        state: dict[str, int] = {}

        def dfs(a: str) -> None:  # drops the edges that close a cycle
            state[a] = 1
            for key in edges:
                if src[key] == a and key[1] in pos and key[1] != a:
                    b = key[1]
                    if state.get(b) == 1:
                        continue
                    succ[a].add(b)
                    pred[b].add(a)
                    if b not in state:
                        dfs(b)
            state[a] = 2

        for it in items:
            if it not in state:
                dfs(it)
        L: dict[str, int] = {}
        base = 1 if panel else 0
        for it in self._topo(items, pred):
            L[it] = max([L[p] + 1 for p in pred[it]] + [base])
        topo = self._topo(items, pred)
        for it in reversed(topo):  # what only needs the inputs is computed as late as it can be
            kind = self.nodes[it][1] if it in self.nodes else ""
            if not pred[it] and succ[it] and kind != "in":
                L[it] = max(base, min(L[s] for s in succ[it]) - 1)
        last = max(L.values(), default=base - 1) + 1
        for p in ins:
            L[p] = 0
        for p in outs:
            L[p] = last
        # one dummy per layer per value that skips layers: edges from one value share them
        hops: list[tuple[str, str, str]] = []  # (from, to, value) between consecutive layers
        dummies: set[str] = set()
        by_value: dict[str, list[str]] = defaultdict(list)
        skips = self._skips(c, src, L)
        for (u, t), s in src.items():
            if (u, t) not in skips:
                by_value[u].append(t)
        for u, targets in by_value.items():
            s = src[(u, targets[0])]
            for t in targets:
                prev = s
                for layer in range(L[s] + 1, L[t]):
                    d = f"d:{u}:{layer}:{c}"
                    L[d] = layer
                    if d not in dummies:
                        dummies.add(d)
                        self.size[d] = (0.0, 0.0, 0.0)
                    if (prev, d, u) not in hops:
                        hops.append((prev, d, u))
                    prev = d
                hops.append((prev, t, u))
        nl = max(L.values(), default=0) + 1
        layers: list[list[str]] = [[] for _ in range(nl)]
        rank: dict[str, float] = {it: k for k, it in enumerate(items)}
        for k, p in enumerate(ins):
            rank[p] = min([rank[t] for key, t in ((key, key[1]) for key in edges) if src[key] == p and t in rank] + [k]) - 0.5
        for k, p in enumerate(outs):
            rank[p] = len(items) + k
        for d in dummies:
            u = d.split(":")[1]
            rank[d] = rank.get(src[(u, by_value[u][0])], 0) + 0.5
        for (u, t) in skips:  # a skip keeps its source in the order, without dummies
            by_value.setdefault(u, [])
        for it in sorted(L, key=lambda x: rank.get(x, 0)):
            layers[L[it]].append(it)
        hint = self.hints.get(c)
        if hint:
            layers[0].sort(key=lambda p: hint.get(p, 0.0))
        self._order(layers, hops, fixed=bool(hint))
        xs = self._xs(c, layers, hops)
        ys, H, gaps = self._ys(c, layers, hops, xs, panel)
        self.layer[c] = L
        # the layout's x runs across the streams and y along the flow: drawn transposed, left to right
        at = {it: (xs[it] + (self.TOP if panel else 0.0), ys[L[it]]) for it in L}
        width = max([xs[it] + self.size[it][0] for it in L] + [0.0]) + (self.TOP + self.PAD if panel else 0.0)
        if panel:
            need = sum(_tw(t, s) for t, s in self.title(c)) + 2 * self.PAD + 12
            if H < need:  # a wide title: a longer panel, the flow centred in it
                shift = (need - H) / 2
                at = {it: (x, y + (shift if not it.startswith("i:") else 0.0)) for it, (x, y) in at.items()}
                at.update({it: (x, need) for it, (x, y) in at.items() if it.startswith("o:")})
                ys = [y + shift for y in ys]
                gaps = [(g + shift, {k: v + shift for k, v in d.items()}) for g, d in gaps]
                H = need
        self.at[c] = at
        self._route(c, layers, hops, L, at, gaps)
        self._skip_routes(c, skips, src, L, layers, at)
        return width, H

    @staticmethod
    def _topo(items: list[str], pred: dict[str, set[str]]) -> list[str]:
        out, seen = [], set()

        def visit(a: str) -> None:
            if a in seen:
                return
            seen.add(a)
            for p in sorted(pred[a], key=items.index):
                visit(p)
            out.append(a)

        for it in items:
            visit(it)
        return out

    @staticmethod
    def _crossings(layers: list[list[str]], hops: list[tuple[str, str, str]]) -> int:
        idx = {it: k for layer in layers for k, it in enumerate(layer)}
        lay = {it: n for n, layer in enumerate(layers) for it in layer}
        by: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for a, b, _ in hops:
            if lay[b] == lay[a] + 1:
                by[lay[a]].append((idx[a], idx[b]))
        n = 0
        for es in by.values():
            es = sorted(set(es))
            n += sum(1 for i in range(len(es)) for j in range(i + 1, len(es)) if es[i][0] < es[j][0] and es[i][1] > es[j][1])
        return n

    def _order(self, layers: list[list[str]], hops: list[tuple[str, str, str]], fixed: bool = False) -> None:
        """Barycentre sweeps (down, then up), keeping the order with the fewest crossings."""
        up: dict[str, list[str]] = defaultdict(list)
        down: dict[str, list[str]] = defaultdict(list)
        for a, b, _ in hops:
            down[a].append(b)
            up[b].append(a)
        best, best_n = [list(x) for x in layers], self._crossings(layers, hops)
        for sweep in range(8):
            rng = range(1, len(layers)) if sweep % 2 == 0 else range(len(layers) - 2, 0 if fixed else -1, -1)
            for k in rng:  # with ``fixed``, the entry ports (layer 0) keep their order
                ref = layers[k - 1] if sweep % 2 == 0 else layers[k + 1]
                where = {it: j for j, it in enumerate(ref)}
                nb = up if sweep % 2 == 0 else down
                cur = {it: j for j, it in enumerate(layers[k])}

                def key(it: str) -> float:
                    ps = [where[p] for p in nb[it] if p in where]
                    return sum(ps) / len(ps) if ps else cur[it]

                layers[k].sort(key=lambda it: (key(it), cur[it]))
            n = self._crossings(layers, hops)
            if n < best_n:
                best, best_n = [list(x) for x in layers], n
        layers[:] = best

    def attach(self, c: int, it: str, value: str, top: bool) -> float:
        """Where an edge carrying ``value`` meets item ``it`` of container ``c``, from its left edge."""
        if it[0] == "c":
            inner = int(it[1:])
            return self.at[inner][("i:" if top else "o:") + value][0]
        return self.size[it][0] / 2

    def _xs(self, c: int, layers: list[list[str]], hops: list[tuple[str, str, str]]) -> dict[str, float]:
        """x of each item: packed, then moved towards its neighbours' edge ends (order kept) a few times."""
        gap = lambda a, b: self.LANES if a[1] != ":" and b[1] != ":" else 10.0  # boxes apart, dummies and ports closer
        xs: dict[str, float] = {}
        for layer in layers:
            x = 0.0
            for k, it in enumerate(layer):
                xs[it] = x
                x += self.size[it][0] + (gap(it, layer[k + 1]) if k + 1 < len(layer) else 0)
        up: dict[str, list[tuple[str, str]]] = defaultdict(list)
        down: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for a, b, u in hops:
            down[a].append((b, u))
            up[b].append((a, u))

        def fit(layer: list[str], want: dict[str, float]) -> None:
            # least squares under x[k+1] >= x[k] + w[k] + gap (pool adjacent violators)
            off, o = [], 0.0
            for k, it in enumerate(layer):
                off.append(o)
                o += self.size[it][0] + (gap(it, layer[k + 1]) if k + 1 < len(layer) else 0)
            blocks: list[list[float]] = []  # [mean, weight, count]
            for k, it in enumerate(layer):
                w = 1.0 if it in want else 0.05
                blocks.append([want.get(it, xs[it]) - off[k], w, 1])
                while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                    m2, w2, n2 = blocks.pop()
                    m1, w1, n1 = blocks.pop()
                    blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
            k = 0
            for mean, _, n in blocks:
                for _ in range(int(n)):
                    xs[layer[k]] = mean + off[k]
                    k += 1

        for sweep in range(10):
            downward = sweep % 2 == 0
            rng = range(1, len(layers)) if downward else range(len(layers) - 2, -1, -1)
            for k in rng:
                want = {}
                for it in layers[k]:
                    nb = up[it] if downward else down[it]
                    if nb:
                        if downward:
                            ends = [xs[a] + self.attach(c, a, u, False) - self.attach(c, it, u, True) for a, u in nb]
                        else:
                            ends = [xs[b] + self.attach(c, b, u, True) - self.attach(c, it, u, False) for b, u in nb]
                        want[it] = sum(ends) / len(ends)
                fit(layers[k], want)
            if sweep == 0:  # the first layer follows the second
                want = {}
                for it in layers[0]:
                    if down[it]:
                        ends = [xs[b] + self.attach(c, b, u, True) - self.attach(c, it, u, False) for b, u in down[it]]
                        want[it] = sum(ends) / len(ends)
                fit(layers[0], want)
        lo = min([x for x in xs.values()] + [0.0]) if xs else 0.0
        return {it: x - lo for it, x in xs.items()}

    def _ys(self, c: int, layers: list[list[str]], hops: list[tuple[str, str, str]], xs: dict[str, float],
            panel: bool) -> tuple[list[float], float, list[tuple[float, dict[tuple[str, int], float]]]]:  # fmt: skip
        """y of each layer, the container's height, and for each gap its top and the y of each jog."""
        lay = {it: k for k, layer in enumerate(layers) for it in layer}
        heights = [max([self.size[it][1] + self.size[it][2] for it in layer] + [0.0]) for layer in layers]
        jogs: dict[int, dict[str, list[float]]] = defaultdict(dict)  # gap -> value -> x interval
        labelled: set[int] = set()
        for a, b, u in hops:
            k = lay[a]
            xa = xs[a] + self.attach(c, a, u, False)
            xb = xs[b] + self.attach(c, b, u, True)
            if abs(xa - xb) > 0.5:
                lo, hi = jogs[k].get(u, [min(xa, xb), max(xa, xb)])
                jogs[k][u] = [min(lo, xa, xb), max(hi, xa, xb)]
            if self.E[c].get((u, b), {}).get("label"):
                labelled.add(k)
        ys, gaps, y = [], [], 0.0
        for k, layer in enumerate(layers):
            ys.append(y)
            if k == len(layers) - 1:
                break
            # tracks: values whose jogs overlap get different heights
            tracks: list[float] = []
            slot: dict[tuple[str, int], float] = {}
            for u, (lo, hi) in sorted(jogs[k].items(), key=lambda kv: kv[1][0]):
                t = next((j for j, end in enumerate(tracks) if end < lo - 4), None)
                if t is None:
                    tracks.append(hi)
                    t = len(tracks) - 1
                else:
                    tracks[t] = hi
                slot[(u, k)] = t
            top = y + heights[k]
            lead = self.PAD if (panel and k == 0) else 9.0
            tail = 30.0 if k in labelled else 13.0
            span = lead + max(len(tracks) - 1, 0) * self.LANE + tail
            if not jogs[k] and not (panel and k == 0):
                span = max(span, self.GAP - 6)
            gaps.append((top, {key: top + lead + j * self.LANE for key, j in slot.items()}))
            y = top + span
        height = ys[-1] + heights[-1] if ys else 0.0
        has_out = panel and any(it.startswith("o:") for it in layers[-1]) if layers else False
        if panel and not has_out:
            height += self.PAD
        return ys, height, gaps

    def _route(self, c: int, layers: list[list[str]], hops: list[tuple[str, str, str]], L: dict[str, int],
               at: dict[str, tuple[float, float]], gaps: list[Any]) -> None:  # fmt: skip
        """The polyline of each edge (value, target) inside ``c``, relative to its top left."""
        def bottom(it: str, u: str) -> tuple[float, float]:
            x, y = at[it]
            if it.startswith("i:"):
                return x, y
            if it.startswith("d:"):
                return x, y + max([self.size[o][1] + self.size[o][2] for o in layers[L[it]]] + [0.0])
            w, h, below = self.size[it]
            return x + self.attach(c, it, u, False), y + h + below

        # spread the arrows into a node across its top, ordered by where they come from
        into: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for a, b, u in hops:
            if b[1] != ":" and b[0] != "c":
                into[b].append((bottom(a, u)[0], u))
        spread: dict[tuple[str, str], float] = {}
        for b, srcs in into.items():
            srcs = sorted(set(srcs))
            w = self.size[b][0]
            for k, (_, u) in enumerate(srcs):
                spread[(b, u)] = w / 2 if len(srcs) == 1 else min(max(w * (k + 1) / (len(srcs) + 1), 10), w - 10)

        def top(it: str, u: str) -> tuple[float, float]:
            x, y = at[it]
            if it.startswith("o:") or it.startswith("d:"):
                return x, y
            if it[0] == "c":
                return x + self.attach(c, it, u, True), y
            return x + spread.get((it, u), self.size[it][0] / 2), y

        nxt: dict[tuple[str, str], str] = {}  # (value, from) -> next hop towards each target
        hop_of: dict[tuple[str, str], list[str]] = defaultdict(list)
        for a, b, u in hops:
            hop_of[(u, a)].append(b)
        segs: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for (u, t) in self.E[c]:
            s = self.src_of(u, c)
            path, cur = [s], s
            while cur != t:
                options = hop_of[(u, cur)]
                cur = t if t in options else next((o for o in options if o.startswith("d:")), t)
                path.append(cur)
                if len(path) > len(layers) + 2:
                    break
            pts = [bottom(s, u)]
            for a, b in zip(path, path[1:]):
                x0, y0 = pts[-1]
                xb, yb = top(b, u)
                ka = L.get(a, 0)
                if abs(x0 - xb) > 0.5 and ka < len(gaps):
                    yj = gaps[ka][1].get((u, ka), gaps[ka][0] + 12)
                    pts += [(x0, yj), (xb, yj)]
                pts.append((xb, yb))
                if b.startswith("d:"):
                    pts.append(bottom(b, u))
            segs[(u, t)] = pts
        self.segs[c] = segs
        nxt.clear()

    def _skips(self, c: int, src: dict[tuple[str, str], str], L: dict[str, int]) -> set[tuple[str, str]]:
        """Edges into a ``+`` from further back than its other input: residual connections, drawn as loops."""
        into: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for (u, t), s in src.items():
            if t in self.nodes and self.nodes[t][2] == "+" and s in L:
                into[t].append((u, s))
        out = set()
        for t, srcs in into.items():
            if len(srcs) >= 2:
                last = max(L[s] for _, s in srcs)
                out |= {(u, t) for u, s in srcs if L[s] < last and L[s] < L[t]}
        return out

    def _skip_routes(self, c: int, skips: set[tuple[str, str]], src: dict[tuple[str, str], str], L: dict[str, int],
                     layers: list[list[str]], at: dict[str, tuple[float, float]]) -> None:  # fmt: skip
        """A residual leaves its source's arrow just after it, runs above the row, and drops into the ``+``."""
        for u, t in skips:
            s = src[(u, t)]
            sx, sy = at[s]
            if not s.startswith("i:"):
                sx += self.attach(c, s, u, False)
                sy += self.size[s][1]
            tx, ty = at[t]
            mid = ty + self.size[t][1] / 2
            between = [it for k in range(L[s] + 1, L[t]) for it in layers[k] if it[1] != ":"]
            lo, hi = min(sx, tx), max(sx, tx + self.size[t][0])
            over = [at[it][0] for it in between if at[it][0] < hi and at[it][0] + self.size[it][0] > lo]
            above = min(over + [min(sx, tx)]) - 9
            branch = sy + (4 if s.startswith("i:") else 5)
            self.segs[c][(u, t)] = [(sx, sy), (sx, branch), (above, branch), (above, mid), (tx, mid)]
            self.skipped.add((u, t))

    def _centre(self, n: str) -> float:
        """Where node ``n`` sits across the flow (absolute), from the previous layout."""
        c = self.ncont[n]
        o = self.origin[c][0] + self.at[c][n][0]
        return o + self.size[n][0] / 2

    def absolute(self, c: int, x0: float, y0: float) -> None:
        self.origin[c] = (x0, y0)
        for it, (x, y) in self.at[c].items():
            if it[0] == "c":
                self.absolute(int(it[1:]), x0 + x, y0 + y)

    def box_at(self, it: str) -> tuple[float, float, float, float]:
        """The absolute rectangle of a node or container item."""
        c = self.ncont[it] if it in self.nodes else self.cparent[int(it[1:])]
        ox, oy = self.origin[c]
        x, y = self.at[c][it]
        w, h, _ = self.size[it]
        return ox + x, oy + y, w, h

    def edge_path(self, u: str, v: str) -> list[tuple[float, float]]:
        cu, cv = self.chain(self.ncont[u]), self.chain(self.ncont[v])
        lca = next(c for c in cu if c in cv)
        parts = [(c, (u, "o:" + u)) for c in cu[: cu.index(lca)]] + [(lca, (u, self.item_of(v, lca)))]
        parts += [(c, (u, self.item_of(v, c))) for c in reversed(cv[: cv.index(lca)])]
        pts: list[tuple[float, float]] = []
        for c, key in parts:
            ox, oy = self.origin[c]
            for x, y in self.segs[c].get(key, []):
                p = (ox + x, oy + y)
                if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 0.01:
                    pts.append(p)
        # drop points in the middle of straight runs
        out = pts[:1]
        for k in range(1, len(pts) - 1):
            (ax, ay), (bx, by), (cx, cy) = out[-1], pts[k], pts[k + 1]
            if abs((bx - ax) * (cy - by) - (by - ay) * (cx - bx)) > 0.01:
                out.append(pts[k])
        return out + pts[-1:] if len(pts) > 1 else pts

    # -- painting (the layout's (x, y) is drawn as (y, x): left to right)

    def rect_of(self, it: str) -> tuple[float, float, float, float]:
        """The drawn rectangle (x, y, w, h) of a node or container item."""
        o, a, wo, ha = self.box_at(it)
        return a + self.x0, o + self.y0, ha, wo

    def paint(self, cv: _Canvas, x0: float, y0: float) -> None:
        """Draw the main flow with its top left at (x0, y0)."""
        self.x0, self.y0 = x0, y0
        self.titles: list[Any] = []
        self.fills: dict[int, str] = {-1: "bg"}
        self._panels(cv, -1, 0)
        rank = lambda e: ((e[0], e[1]) not in self.skipped, (e[0], e[1]) in self.path, e[0] in self.sidenodes)
        self.labels: list[Any] = []
        for u, v, *_ in sorted(self.edges, key=rank):
            self._edge(cv, u, v)
        for x, y, runs, fill in self.labels:  # the types arrays enter calls with, over the edges
            cv.text(x, y, runs, anchor="end", halo=fill)
        for x, y, runs, anchor, fill, inner, i in self.titles:  # over the edges that enter the panel
            cv.group(class_="pt", data_box=inner, data_loop=i)
            cv.text(x, y, runs, anchor=anchor, halo=fill)
            cv.end()
        for n in self.order:
            self._node(cv, n)

    def _panels(self, cv: _Canvas, c: int, depth: int) -> None:
        for it in self.items[c]:
            if it[0] != "c":
                continue
            i = int(it[1:])
            x, y, w, h = self.rect_of(it)
            b = self.B[i]
            inner = next((k for k, v in self.alias.items() if v == i), i)
            fill = "bg" if depth % 2 == 0 else "panel2"
            stroke, sw, dash = "axis", 1.0, None
            if (hk := self.heat_of(i)) is not None:
                fill = "p" + hk[1:]
            if b.get("err") == 1 or self.B[inner].get("err") == 1:
                stroke, sw = "err", 1.2
            if b["k"] != "call" and i not in self.alias.values():
                dash = "4 2.5"  # a loop or branch around its contents
            self.fills[i] = fill
            cv.group(class_="pn", data_box=inner, data_loop=i)
            cv.rect(x, y, w, h, fill, stroke, sw, r=6, dash=dash)
            cv.end()
            right = b["k"] == "loop" and b["l"].startswith("×")  # "block × N" goes top right, as in a paper
            tx = x + w - self.PAD + 4 if right else x + self.PAD - 4
            self.titles.append((tx, y + 16, self.title(i), "end" if right else "start", fill, inner, i))
            self._panels(cv, i, depth + 1)

    def _edge(self, cv: _Canvas, u: str, v: str) -> None:
        pts = [(a + self.x0, o + self.y0) for o, a in self.edge_path(u, v)]
        if len(pts) < 2:
            return
        red, side, skip = (u, v) in self.path, u in self.sidenodes, (u, v) in self.skipped
        color = "err" if red else "side" if side else self.role.get(v, "mix") if skip else "ink2"
        if skip and color == "mix":
            color = "mix"
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        n = math.hypot(xb - xa, yb - ya) or 1.0
        tip = pts[-1]
        pts[-1] = (xb - (xb - xa) / n * 3.5, yb - (yb - ya) / n * 3.5)
        width = 1.5 if red else 0.9 if skip else 1.0
        cv.line(pts, color, width, "4 2.4" if side and not red else None)
        cv.head(tip[0], tip[1], xb - xa, yb - ya, color, 5.5 if red else 5.0)
        info = next((e for e in self.edges if e[0] == u and e[1] == v), None)
        if info and info[4] and info[4] != info[3] and abs(yb - ya) < 0.5 and self.nodes[v][1] == "call":
            self.labels.append((xb - 7, yb - 4, [(_shape(info[4]), "tlab")], self.fills[self.ncont[v]]))

    def _node(self, cv: _Canvas, n: str) -> None:
        x, y, w, h = self.rect_of(n)
        kind, label = self.nodes[n][1], self.nodes[n][2]
        role = self.role[n]
        inside = self.ncont[n] >= 0
        fill, stroke, sw = role + ("c" if inside else "f"), role, 0.9 if inside else 1.1
        box = int(n[1:]) if kind == "call" else -1
        if box >= 0 and (hk := self.heat_of(box)) is not None:
            fill = hk
        if n == self.fail:
            fill, stroke, sw = "errf", "err", 1.8
        elif n in self.feeds:
            stroke, sw = "err", 1.4
        cv.group(class_="nd" + (" open" if box >= 0 and self.B[box]["o"] and n != self.fail else ""),
                 data_id=n, data_box=box)  # fmt: skip
        lines = self.lines(n)
        if kind in ("in", "out"):  # a symbol and its shape, no box
            ax = x + w if kind == "in" else x
            ty = y + 12
            for line in lines:
                cv.text(ax, ty, line, anchor="end" if kind == "in" else "start")
                ty += self._lh(line)
        elif kind == "op" and label == "+" and n != self.fail:
            cv.rect(x, y, w, h, "bg", stroke, sw, r=w / 2)
            cx, cy, k = x + w / 2, y + h / 2, w * 0.27
            cv.line([(cx - k, cy), (cx + k, cy)], "ink", 0.9, radius=0)
            cv.line([(cx, cy - k), (cx, cy + k)], "ink", 0.9, radius=0)
        else:
            r = 3.0 if inside else 4.0
            if n in self.card:  # a stack of layers
                cv.rect(x + 4, y, w - 4, h - 4, fill, stroke, sw, r=r)
                y, w, h = y + 4, w - 4, h - 4
            cv.rect(x, y, w, h, fill, stroke, sw, r=r)
            total = sum(self._lh(line) for line in lines)
            ty = y + (h - total) / 2
            for line in lines:
                ty += self._lh(line)
                cv.text(x + w / 2, ty - 4, line)
        cv.end()


# ------------------------------------------------------------------ drawing: the figure


class _Figure:
    """A whole figure: a one-line header (bold name, grey words) above the flow, as a display list."""

    M = 16.0

    def __init__(self, m: dict[str, Any], expanded: set[int]):
        self.m, self.expanded = m, expanded
        self.scene = _Scene(m, expanded)
        self.scene.layout()
        self.cv = _Canvas()
        self.paint()

    def sizes(self) -> str:
        """``T · D · Vocab · L = len(blocks)``: the sizes in the drawn types, list lengths, given sizes."""
        sc, m = self.scene, self.m
        types = [n[2] for n in sc.nodes.values()] + [t for e in sc.edges for t in e[2:]]
        loops = [b["l"][2:] for b in m["boxes"] if b["k"] == "loop" and b["l"].startswith("× ")]
        parts: list[str] = []
        for name in _names(types) + [n for n in loops if n.isidentifier()]:
            if any(p.split(" ")[0] == name for p in parts):
                continue
            parts.append(f"{name} = len({m['lists'][name]})" if name in m.get("lists", {}) else name)
        parts += [f"{k} = {v}" for k, v in m.get("given", {}).items()]
        parts += [f"{k} (= {v})" for k, v in m.get("fixed", {}).items()]  # f32[R:6]: drawn as 6
        return " · ".join(parts)

    def paint(self) -> None:
        cv, m, sc, M = self.cv, self.m, self.scene, self.M
        heading: list[Run] = [(m["name"], "th")]
        if sizes := self.sizes():
            heading.append(("   sizes " + sizes, "tl"))
        rows = [heading]
        t, err = m.get("timing"), m.get("error")
        if t and t["unit"] == "s":
            rows.append([(f"measured with jax.profiler ({t['backend']}): {_amount(t['total'], 's')} of XLA op time per call, "
                          f"{_amount(t['wall'], 's')} wall, median of {t['runs']} runs", "tl")])  # fmt: skip
        elif t:
            rows.append([(f"ESTIMATED from FLOPs, not measured ({t['why']}): {_amount(t['total'], 'flop')} per call", "tred")])
        if err:
            rows.append([("✗ " + err["head"], "tred")])
        y = M + 12
        widths = []
        for k, runs in enumerate(rows):
            cv.text(M, y, runs, anchor="start")
            widths.append(sum(_tw(a, b) for a, b in runs))
            y += 17
        if t:  # the heat scale, in the header's grey words
            x = M
            lo, hi = [0.0, *HEAT], [*HEAT, 1.0]
            cv.text(x, y, [("share of the time", "tl")], anchor="start")
            x += _tw("share of the time", "tl") + 10
            for k in range(len(HEAT) + 1):
                label = f"< {hi[0]:.0%}" if k == 0 else f"≥ {lo[k]:.0%}" if k == len(HEAT) else f"{lo[k]:.0%}–{hi[k]:.0%}"
                cv.rect(x, y - 9, 16, 10, f"h{k}", "ink2", 0.8, r=2)
                cv.text(x + 21, y, [(label, "tl")], anchor="start")
                x += 21 + _tw(label, "tl") + 12
            widths.append(x - M)
            y += 17
        top = y + 4
        sc.paint(cv, M, top)
        at = sc.at[-1]
        main_w = max([sc.size[it][1] + at[it][1] for it in at] + [100.0])
        main_h = max([sc.size[it][0] + at[it][0] for it in at] + [30.0])
        self.w = M + max([main_w] + widths) + M
        self.h = top + main_h + M


# ------------------------------------------------------------------ the interactive page and pictures


def _states(m: dict[str, Any], budget: int = 40) -> tuple[dict[str, str], dict[str, dict[str, str]], list[str]]:
    """SVGs for the views the page can show: each depth, and what clicks reach from them (up to ``budget``).

    Returns (key -> svg, key -> {box: key it goes to when clicked}, the key of each depth).
    """
    calls = [i for i, b in enumerate(m["boxes"]) if b["k"] == "call" and b["o"]]
    deepest = max([m["boxes"][i]["d"] for i in calls] + [0]) + 1
    key = lambda s: ",".join(map(str, sorted(s)))
    depths = []
    states: dict[str, set[int]] = {}
    for d in range(1, max(deepest, m["depth"]) + 1):
        s = _opened({**m, "depth": d})
        depths.append(key(s))
        states.setdefault(key(s), s)
    svgs: dict[str, str] = {}
    clicks: dict[str, dict[str, str]] = {}

    def below(i: int) -> set[int]:
        out = {i}
        for j, b in enumerate(m["boxes"]):
            p = b["p"]
            while p >= 0 and p not in out:
                p = m["boxes"][p]["p"]
            if p in out:
                out.add(j)
        return out

    todo = list(states)
    while todo:
        k = todo.pop(0)
        fig = _Figure(m, states[k])
        svgs[k] = _svg(fig.cv, fig.w, fig.h)
        nodes = [n for n in fig.scene.order if fig.scene.nodes[n][1] == "call"]
        clickable = {int(n[1:]) for n in nodes if m["boxes"][int(n[1:])]["o"]} | {i for i in states[k] if i in calls}
        for i in sorted(clickable):
            s = states[k] - below(i) if i in states[k] else states[k] | {i}
            if key(s) not in states and len(states) < budget:
                states[key(s)] = s
                todo.append(key(s))
            clicks.setdefault(k, {})[str(i)] = key(s)
    for targets in clicks.values():  # a click past the budget goes to the nearest depth view
        for box, t in targets.items():
            if t not in svgs:
                want = states.get(t) or {int(x) for x in t.split(",") if x}
                fits = [d for d in depths if want <= states[d]]
                targets[box] = fits[0] if fits else depths[-1]
    return svgs, clicks, depths


def _page(m: dict[str, Any]) -> str:
    """The interactive page: every view's SVG, and the boxes' details, in one HTML file."""
    svgs, clicks, depths = _states(m)
    start = ",".join(map(str, sorted(_opened(m))))
    boxes = [dict(name=b["l"], k=b["k"], sig=b.get("sig", ""), loc=b.get("loc", ""), ins=b.get("ins", ""),
                  outs=b.get("outs", ""), src=b.get("src", ""), vectorize=b.get("vectorize", False),
                  err=b.get("err", 0), t=b.get("t"), s=b.get("s"), n=b.get("n"), o=b.get("o", False),
                  p=b["p"], path=b.get("path"))
             for b in m["boxes"]]  # fmt: skip
    data = dict(name=m["name"], start=start, depths=depths, svgs=svgs, clicks=clicks, boxes=boxes, sources=m["sources"],
                timing=m.get("timing"), error=m.get("error"),
                text=_outline(m))  # fmt: skip
    blob = json.dumps(data).replace("</", "<\\/")
    page = _PAGE.read_text(encoding="utf-8")
    return page.replace("__TITLE__", html.escape(m["name"])).replace("/*DATA*/null", blob)


def _chrome() -> str | None:
    """A Chromium or Chrome binary to take screenshots with, if one is installed."""
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "chrome", "headless_shell"):
        if path := shutil.which(name):
            return path
    roots = [os.environ.get("PLAYWRIGHT_BROWSERS_PATH", ""), str(Path.home() / ".cache" / "ms-playwright")]
    for root in filter(None, roots):
        for pattern in ("chromium_headless_shell-*/*/chrome-headless-shell", "chromium-*/*/chrome"):
            found = sorted(Path(root).glob(pattern))
            if found:
                return str(found[-1])
    return None


def _png(fig: "_Figure", path: Path, theme: str = "light", scale: float = 2.0) -> None:
    """Rasterise the figure in ``theme``: with Playwright, else a headless Chromium, else matplotlib."""
    svg, w, h = _svg(fig.cv, fig.w, fig.h), fig.w, fig.h
    bg = COLORS["bg"][theme == "dark"]
    page = f'<!doctype html><html data-theme="{theme}"><body style="margin:0;background:{bg}">{svg}</body></html>'
    size = (math.ceil(w), math.ceil(h))
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        sync_playwright = None
    if sync_playwright is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                tab = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=scale,
                                       color_scheme=theme)  # fmt: skip
                tab.set_content(page)
                tab.screenshot(path=str(path), full_page=True)
                browser.close()
            return
        except Exception:
            pass
    if (chrome := _chrome()) is not None:
        with tempfile.TemporaryDirectory(prefix="expex-diagram-") as tmp:
            src = Path(tmp) / "diagram.html"
            src.write_text(page, encoding="utf-8")
            out = Path(tmp) / "shot.png"
            cmd = [chrome, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--mute-audio",
                   f"--force-device-scale-factor={scale}", f"--window-size={size[0]},{size[1]}",
                   f"--screenshot={out}", src.as_uri()]  # fmt: skip
            try:
                subprocess.run(cmd, capture_output=True, timeout=120, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if out.exists() and out.stat().st_size:
                shutil.copyfile(out, path)
                return
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise coded("EX708", RuntimeError(
            "saving a PNG needs a browser to draw the SVG (Playwright, or Chromium / Chrome on the PATH) "
            "or matplotlib; none was found. Save an .svg or .html instead."
        )) from None
    _png_matplotlib(fig, path, theme, scale)


def _png_matplotlib(fig: "_Figure", path: Path, theme: str, scale: float) -> None:
    """Draw the display list again with matplotlib: same layout, its own fonts (the fallback without a browser)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon

    col = lambda k: COLORS[k][theme == "dark"]
    dpi = 96 * scale
    f = plt.figure(figsize=(fig.w / 96, fig.h / 96), dpi=dpi)
    ax = f.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, fig.w)
    ax.set_ylim(fig.h, 0)
    ax.axis("off")
    f.patch.set_facecolor(col("bg"))
    z = 1
    for op in fig.cv.ops:
        z += 1
        if op[0] == "rect":
            _, x, y, w, h, fill, stroke, sw, r, dash = op
            r = min(r, w / 2, h / 2)
            ax.add_patch(FancyBboxPatch((x + r, y + r), max(w - 2 * r, 0.01), max(h - 2 * r, 0.01),
                                        boxstyle=f"round,pad={r}", facecolor=col(fill),
                                        edgecolor=col(stroke) if stroke else "none", linewidth=sw * 0.75,
                                        linestyle=(0, (4, 2.5)) if dash else "-", zorder=z))  # fmt: skip
        elif op[0] == "text":
            _, x, y, runs, anchor, _ = op
            text = "".join(t for t, _ in runs)
            size, weight, italic, mono, color = FONTS[runs[0][1]]
            ax.text(x, y, text, ha={"middle": "center", "start": "left", "end": "right"}[anchor], va="baseline",
                    fontsize=size * 0.75, fontweight="bold" if weight >= 600 else "normal",
                    style="italic" if italic else "normal", family="monospace" if mono else "sans-serif",
                    color=col(color), zorder=z)  # fmt: skip
        elif op[0] == "line":
            _, pts, stroke, sw, dash, _ = op
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col(stroke), linewidth=sw * 0.75,
                    linestyle=(0, (4, 2.5)) if dash else "-", zorder=z, solid_capstyle="butt")  # fmt: skip
        elif op[0] == "head":
            ax.add_patch(Polygon(_head(op), closed=True, facecolor=col(op[5]), edgecolor="none", zorder=z))
    f.savefig(path, dpi=dpi, facecolor=col("bg"))
    plt.close(f)


class Diagram:
    """A traced function's structure, drawn as a figure.

    ``svg`` is the figure at ``depth`` (SVG text); ``save`` writes it as ``.svg``,
    ``.png`` or an interactive ``.html`` page, also shown in notebooks. ``text``
    (and ``str()``) is an outline of every call, with ``timings`` (from
    ``diagram.timed``) or the ``error`` (from ``diagram.explain``). ``nodes``,
    ``edges`` and ``boxes`` describe the same view for programs.
    """

    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.depth: int = model["depth"]
        opened = _opened(model)
        nodes, edges, drawn, fail = _view(model, opened)
        #: (id, kind, label, parent box): kind is "call", "op", "in" or "out".
        self.nodes = nodes
        #: (from id, to id, array type)
        self.edges = edges
        #: box id -> label, for the boxes drawn (open calls, loops, branches)
        self.boxes = {f"b{i}": model["boxes"][i]["l"] for i in drawn}
        #: The id of the node that failed (``explain``), and the ids of the nodes that feed it.
        self.failed = fail
        self.feeds = _feeds(edges, fail)[0]
        self._figure: _Figure | None = None

    def _fig(self) -> "_Figure":
        if self._figure is None:
            self._figure = _Figure(self.model, _opened(self.model))
        return self._figure

    @property
    def svg(self) -> str:
        """The figure as SVG text (a light figure, sized in px)."""
        f = self._fig()
        return _svg(f.cv, f.w, f.h)

    @property
    def timings(self) -> dict[str, tuple[float, float]]:
        """For a timed diagram: box id -> (seconds per call, or FLOPs if estimated; share of the total)."""
        return {f"b{i}": (b["t"], b["s"]) for i, b in enumerate(self.model["boxes"]) if "t" in b}

    @property
    def error(self) -> str | None:
        """For ``explain``: the message of the ``ShapeError`` the diagram stops at."""
        return self.model["error"]["message"] if self.model.get("error") else None

    @property
    def text(self) -> str:
        """Every call and loop as an indented outline, with its time or failure."""
        return _outline(self.model)

    def at(self, depth: int) -> "Diagram":
        """The same diagram, opened ``depth`` levels."""
        return Diagram({**self.model, "depth": depth})

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"<Diagram of {self.model['name']}: {len(self.nodes)} nodes, depth {self.depth}>"

    def html(self) -> str:
        """The interactive page: pan and zoom, click a call to open or fold it, hover for details."""
        return _page(self.model)

    def save(self, path: str | Path, theme: str = "light") -> Path:
        """Write the figure: ``.svg``, ``.png`` (2x, needs a browser or matplotlib) or ``.html`` (interactive).

        An SVG and the page follow the viewer's light or dark mode; a PNG is drawn in ``theme``
        (``"light"`` or ``"dark"``).
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if theme not in ("light", "dark"):
            raise coded("EX708", ValueError(f"theme is 'light' or 'dark', not {theme!r}"))
        if suffix == ".svg":
            path.write_text(self.svg, encoding="utf-8")
        elif suffix == ".png":
            _png(self._fig(), path, theme)
        elif suffix in (".html", ".htm", ""):
            path.write_text(self.html(), encoding="utf-8")
        else:
            raise coded("EX708", ValueError(f"cannot save a diagram as {suffix!r}: use .svg, .png or .html"))
        return path

    def _repr_html_(self) -> str:
        return f'<iframe srcdoc="{html.escape(self.html())}" style="width:100%;height:640px;border:0"></iframe>'


def _draw(target: Any, depth: int, sizes: dict[str, Any]) -> Diagram:
    """``diagram(target, depth, **sizes)``: see ``DiagramFn.__call__``."""
    s = _cost._setup(target, dict(sizes), "draw")
    f, owner = s.f, getattr(s.f, "_owner_cls", None)
    known = set(f.params) | f.input_names | {"L"}
    known |= {t.__name__ for src in (f.raw, owner) for t in getattr(src, "__type_params__", ())}
    field = lambda path: path.rsplit(".", 1)[-1].split("[")[0]  # self.blocks[0] -> blocks

    def trace(n: int, aliases: set[str]) -> tuple[Any, list[str], _Hook]:
        preset = {k: int(v) for k, v in s.sizes.items() if k not in s.static_kw.keys() | aliases | {"L"}}
        hook = _Hook()
        token = _fn.CALL_HOOK.set(hook)
        try:
            jaxpr, _, lists = _cost._trace(s, n, preset)
        finally:
            _fn.CALL_HOOK.reset(token)
        return jaxpr, lists, hook

    jaxpr, lists, hook = trace(3, set())  # lists get an unusual length, read back as L
    aliases = {field(p) for p in lists} | {"L"}
    given = [int(v) for k, v in s.sizes.items() if k in aliases]
    if lists and given:
        jaxpr, lists, hook = trace(given[0], aliases)
    for k in sizes:
        if k not in known | aliases:
            raise coded("EX706", TypeError(f"{f.name}: unknown size or parameter `{k}`"))
    length = lambda x: "L" if lists and not given and x == 3 else fmt_dim(x)
    b = _Builder(hook, length)
    b.walk(jaxpr.jaxpr, -1, {})
    model = b.model(getattr(f.raw, "__qualname__", f.name), depth)
    if lists and not given:  # for the figure's line of sizes: L = len(blocks)
        model["lists"] = {"L": ", ".join(dict.fromkeys(field(p) for p in lists))}
    model["given"] = {k: int(v) for k, v in sizes.items() if k not in f.params and isinstance(v, int)}
    model["fixed"] = fn_fixed_sizes(f)  # f32[R:6]: R is drawn as 6
    return Diagram(model)


# ------------------------------------------------------------------ on real arguments


def _split(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Callable[[tuple[Any, ...]], Any]]:
    """The arrays in ``(args, kwargs)``, and a function putting new ones in their place."""
    leaves, tree = jax.tree.flatten((args, kwargs))
    at = [i for i, x in enumerate(leaves) if isinstance(x, (jax.Array, np.ndarray, jax.ShapeDtypeStruct))]

    def put(arrays: tuple[Any, ...]) -> Any:
        new = list(leaves)
        for i, x in zip(at, arrays):
            new[i] = x
        return jax.tree.unflatten(tree, new)

    return [leaves[i] for i in at], put


def _name(target: Any, hook: _Hook) -> str:
    root = hook.calls.get(hook.root)
    if root is not None:
        return root["name"]
    return getattr(target, "__qualname__", getattr(target, "__name__", type(target).__name__))


def _hlo(text: str) -> tuple[dict[str, tuple[str, str, str | None]], dict[str, list[str]]]:
    """From compiled HLO text: instruction -> (opcode, op_name, called computation), computation -> instructions."""
    instrs: dict[str, tuple[str, str, str | None]] = {}
    comps: dict[str, list[str]] = {}
    cur: str | None = None
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith("{"):
            words = line.split()
            cur = (words[1] if words[0] == "ENTRY" else words[0]).lstrip("%")
            comps[cur] = []
        elif cur is not None and " = " in line:
            lhs, rhs = line.split(" = ", 1)
            name = lhs.split()[-1].lstrip("%")
            rest = rhs
            if rest.startswith("("):  # a tuple type
                level = 0
                for j, c in enumerate(rest):
                    level += (c == "(") - (c == ")")
                    if level == 0:
                        rest = rest[j + 1 :]
                        break
            else:
                rest = rest.split(" ", 1)[1] if " " in rest else ""
            op = re.match(r"\s*([\w\-]+)", rest)
            calls = re.search(r"calls=%?([\w.\-]+)", rhs)
            meta = re.search(r'op_name="([^"]*)"', rhs)
            instrs[name] = (op.group(1) if op else "", meta.group(1) if meta else "", calls.group(1) if calls else None)
            comps[cur].append(name)
    return instrs, comps


def _profile(compiled: Any, arrays: list[Any], module: str, runs: int) -> tuple[dict[str, float], dict[str, float], float]:  # fmt: skip
    """Run ``compiled`` under ``jax.profiler``: (seconds and executions per instruction per run, wall time)."""
    jax.block_until_ready(compiled(*arrays))  # warm up
    walls = []
    for _ in range(runs):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(*arrays))
        walls.append(time.perf_counter() - t0)
    secs: dict[str, float] = defaultdict(float)
    count: dict[str, float] = defaultdict(float)
    with tempfile.TemporaryDirectory(prefix="expex-profile-") as tmp:
        jax.profiler.start_trace(tmp)
        try:
            for _ in range(runs):
                jax.block_until_ready(compiled(*arrays))
        finally:
            jax.profiler.stop_trace()
        files = sorted(Path(tmp).rglob("*.trace.json.gz"))
        if not files:
            raise RuntimeError("the profiler wrote no trace")
        events = json.loads(gzip.decompress(files[-1].read_bytes()))["traceEvents"]
    for e in events:
        a = e.get("args") or {}
        if e.get("ph") == "X" and a.get("hlo_module") == module and "hlo_op" in a:
            secs[a["hlo_op"]] += float(e.get("dur", 0)) * 1e-6 / runs
            count[a["hlo_op"]] += 1 / runs
    return secs, count, statistics.median(walls)


class _Credit:
    """Adds time to boxes: to a call's box and those around it, from an ``op_name`` scope."""

    def __init__(self, b: _Builder):
        self.b = b
        self.t = [0.0] * len(b.boxes)
        self.n = [0.0] * len(b.boxes)
        self.total = self.outside = 0.0
        # the loops directly in each call (-1: the root), for time spent in a loop outside its calls
        self.loops: dict[int, list[int]] = defaultdict(list)
        for i, box in enumerate(b.boxes):
            if box["k"] == "loop" and (box["p"] < 0 or b.boxes[box["p"]]["k"] == "call"):
                self.loops[box["p"]].append(i)

    def add(self, boxes: set[int], t: float, n: float) -> None:
        self.total += t
        if not boxes:
            self.outside += t
        for i in boxes:
            self.t[i] += t
            self.n[i] += n

    def boxes(self, op_name: str) -> set[int]:
        """The boxes an instruction with this ``op_name`` belongs to (each with its ancestors)."""
        found: set[int] = set()
        last, end = -1, 0
        for m in _TOKEN.finditer(op_name):
            uid = int(m.group(3))
            box = -1 if uid == self.b.hook.root else self.b.by_uid.get(uid)
            if box is None:
                continue
            last, end = box, m.end()
        rest = op_name[end:]
        if (rest.startswith("/while/") or rest.startswith("/cond/")) and len(self.loops.get(last, [])) == 1:
            last = self.loops[last][0]  # in the only loop of the call, outside its calls
        while last >= 0:
            found.add(last)
            last = self.b.boxes[last]["p"]
        return found


def _measured(b: _Builder, compiled: Any, arrays: list[Any], runs: int) -> dict[str, Any]:
    text = compiled.as_text()
    module = text.split()[1].rstrip(",") if text.startswith("HloModule") else ""
    instrs, comps = _hlo(text)
    secs, count, wall = _profile(compiled, arrays, module, runs)
    c = _Credit(b)
    for name, t in secs.items():
        op, op_name, calls = instrs.get(name, ("", "", None))
        if op in _CONTROL:
            continue  # its events contain those of the instructions it runs
        names = [instrs[x][1] for x in comps.get(calls or "", []) if instrs[x][0] not in _PLUMBING and "/" in instrs[x][1]]
        names = names if op == "fusion" and names else [op_name]
        for nm in names:  # a fusion's time is split between the calls of its instructions
            c.add(c.boxes(nm), t / len(names), count[name] / len(names))
    if c.total <= 0:
        raise RuntimeError("the profile has no events for the compiled function")
    for i, box in enumerate(b.boxes):
        box.update(t=c.t[i], s=c.t[i] / c.total, n=round(c.n[i], 2))
    backend = jax.default_backend().upper()
    return dict(unit="s", total=c.total, outside=c.outside, wall=wall, runs=runs, backend=backend, how="profiler")


def _estimated(b: _Builder, why: str) -> dict[str, Any]:
    flops = b.flops or []
    t = [0.0] * len(b.boxes)
    outside = 0.0
    for (box, *_), x in zip(b.atoms, flops):
        if box < 0:
            outside += x
        while box >= 0:
            t[box] += x
            box = b.boxes[box]["p"]
    total = sum(flops) or 1.0
    for i, box in enumerate(b.boxes):
        box.update(t=t[i], s=t[i] / total)
    return dict(unit="flop", total=total, outside=outside, how="estimate", why=why)


def _timed(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any], runs: int = 5) -> Diagram:
    arrays, put = _split(args, kwargs)

    def _expex_timed(*xs: Any) -> Any:
        a, k = put(xs)
        return target(*a, **k)

    hook = _Hook()
    token = _fn.CALL_HOOK.set(hook)
    try:
        traced = jax.jit(_expex_timed).trace(*arrays)
        lowered = traced.lower()
    finally:
        _fn.CALL_HOOK.reset(token)
    compiled = lowered.compile()
    b = _Builder(hook, fmt_dim)
    b.walk(traced.jaxpr.jaxpr, -1, {})
    model = b.model(_name(target, hook), 1)
    try:
        model["timing"] = _measured(b, compiled, arrays, runs)
    except Exception as e:  # no profiler (or one running already): estimate from FLOPs
        why = f"the profiler failed: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}".rstrip(": ")
        b = _Builder(hook, fmt_dim, flops=True)
        b.walk(traced.jaxpr.jaxpr, -1, {})
        model = b.model(_name(target, hook), 1)
        model["timing"] = _estimated(b, why)
    return Diagram(model)


def _explain(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any], depth: int = 1) -> Diagram:
    arrays, put = _split(args, kwargs)
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in arrays]
    hook = _Hook(stop=True)

    def _expex_explain(*xs: Any) -> Any:
        a, k = put(xs)
        try:
            return target(*a, **k)
        except _Stop as s:
            return s.payload or []

    token = _fn.CALL_HOOK.set(hook)
    try:
        with _fn.config(lower_loops=False):  # unrolled, the failing call is traced at the top level
            jaxpr = jax.make_jaxpr(_expex_explain)(*shapes)
    finally:
        _fn.CALL_HOOK.reset(token)
    b = _Builder(hook, fmt_dim)
    env: dict[Any, list[int]] = {}
    b.walk(jaxpr.jaxpr, -1, env)
    b.stopped(jaxpr.jaxpr.outvars, env)
    return Diagram(b.model(_name(target, hook), depth))


class _LazyDiagram(Diagram):
    """``error.diagram``: built from the arguments' shapes the first time it is used."""

    def __init__(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):  # noqa: D107
        object.__setattr__(self, "_todo", (target, args, kwargs))

    def __getattr__(self, name: str) -> Any:
        todo = self.__dict__.pop("_todo", None)
        if todo is None:
            raise AttributeError(name)
        self.__dict__.update(vars(_explain(*todo)))
        return getattr(self, name)


def attach(fn: Any, e: BaseException, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Give a ``ShapeError`` leaving the outermost ``@fn`` call an ``e.diagram`` (built when used)."""
    try:
        if hasattr(e, "diagram"):
            return
        abstract = lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if _type(x) is not None or (
            hasattr(x, "shape") and hasattr(x, "dtype")) else x  # fmt: skip
        args, kwargs = jax.tree.map(abstract, (args, kwargs))  # keep no arrays alive
        setattr(e, "diagram", _LazyDiagram(fn, args, kwargs))
        e.add_note('(see e.diagram.save("error.html"))')
    except Exception:
        pass


class DiagramFn:
    """``expex.diagram``: draw a function from a symbolic trace; ``.timed`` and ``.explain`` use arguments."""

    def __call__(self, target: Any, config: Any = None, /, depth: int = 1, **sizes: Any) -> Diagram:
        """Draw an ``@fn``, an unbound struct method or a bound one, without data.

        ``depth=1`` shows the calls ``target`` makes as nodes, ``depth=2`` opens
        those one level further, and so on. Keywords set sizes (``T=128``), static
        parameters (``heads=4``) and the length of lists of modules (``L=12`` or
        ``blocks=12``), as for ``expex.cost``; ``config``, a config object, as
        ``expex.cost`` takes it. See the module docstring.
        """
        if config is not None:
            from .modules import call_config

            target, sizes = call_config(target, config, sizes, "expex.diagram")
        return _draw(target, depth, sizes)

    def timed(self, target: Any, /, *args: Any, **kwargs: Any) -> Diagram:
        """``target(*args, **kwargs)`` under ``jax.jit`` and ``jax.profiler``: each box coloured by its share.

        The result's ``timings`` maps boxes to (seconds per call, share), its
        ``text`` lists them; ``.at(2)`` opens it further. If the profiler cannot
        run, shares are estimated from FLOPs and the diagram says so.
        """
        return _timed(target, args, kwargs)

    def explain(self, target: Any, /, *args: Any, **kwargs: Any) -> Diagram:
        """Where ``target(*args, **kwargs)`` fails its shapes: the diagram up to the failing call.

        ``explain(error)`` returns the diagram attached to a ``ShapeError`` that
        left a typed call (``error.diagram``). With a function, its trace on the
        arguments' shapes (no data is used) stops at the first call that raises
        ``ShapeError``, drawn in red with the path that feeds it. If nothing
        fails, the diagram is the plain one.
        """
        if isinstance(target, BaseException):
            d = getattr(target, "diagram", None)
            if not isinstance(d, Diagram):
                raise coded("EX708", ValueError("this error has no diagram: only a ShapeError leaving a typed call has one"))
            return d
        return _explain(target, args, kwargs)

    def __repr__(self) -> str:
        return "<function expex.diagram>"


diagram = DiagramFn()

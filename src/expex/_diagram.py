"""Architecture diagrams: an ``@fn`` or a module drawn as a Mermaid flowchart.

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
    >>> print(d)                                # Mermaid: flowchart TD ...
    >>> d.save("transformer.html")              # interactive

- A call deeper than ``depth`` is one node, labelled with its input and output
  shapes (per element for ``.vectorize``). Shallower calls are boxes around their
  contents, so ``depth=2`` shows a block's attention and MLP.
- Operations outside any call at that depth are drawn only when they say what
  the model does: ``matmul``, ``index``, reductions (``sum``, ``max``),
  ``conv``, ``scatter``, ... Reshapes, broadcasts, elementwise arithmetic and
  ``jit`` wrappers are passed through: edges go straight across them.
- A staged loop is a box labelled with its trip count: ``× L`` for a list of
  modules (``L`` unless given, like ``expex.cost``), ``× T`` for a loop over time.
  Traced ``if`` and ``while`` are boxes too.
- Edges follow the data (through passed-through operations), labelled with the
  array's type as it leaves a node, or as it enters a call.
- Inputs are the parameters other than ``self``: a module's own arrays reach
  nearly every node, so they are left out.
- In Mermaid labels, characters Mermaid would read as syntax or markdown
  (``[ ] " _ *`` ...) are written as entity codes (``#91;``), which it renders.

``diagram.save(path)`` writes an interactive page (pan, zoom, click a call to
open or fold it, hover for its signature and source); ``str(diagram)`` is the
Mermaid text, which GitHub renders in a ```mermaid block, and ``.text`` an
outline of the calls.

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
from .annotations.errors import ShapeError
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
        self.calls[uid] = dict(name=_short(fn.name), sig=fn.signature, loc=fn.location,
                               src=fn.name, ins=", ".join(ins), outs="")  # fmt: skip
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
        self.atoms: list[list[Any]] = []  # [box, label or None (passed through), kind]
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
        self.atoms.append([box, label, kind])
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
                outs = self.sub(subs[0], b, srcs(eqn.invars))  # jit, custom_jvp, remat, ...: inline
            if outs is None:
                label = OPS.get(name)
                if name == "gather" and not any(self.data[a] for a in _get(env, eqn.invars[1])):
                    label = None  # a slice by sizes, not a lookup
                if name == "add" and self.joins(eqn, env):
                    label = "+"  # two computed values of one shape: a residual or skip connection
                outs = [[self.atom(b, label, "op", eqn.invars, env, eqn)]] * len(eqn.outvars)
            env.update({o: s for o, s in zip(eqn.outvars, outs) if _type(o.aval) is not None})

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
    for i, (box, label, kind) in enumerate(atoms):
        r = rep[box] if box >= 0 else None
        if r is not None:
            key.append(f"b{r}")
            nodes.setdefault(key[-1], (key[-1], "call", _label(m, boxes[r]), boxes[r]["p"]))
        elif label:
            key.append(f"n{i}")
            nodes[key[-1]] = (key[-1], kind, label, box)
        else:
            key.append(f"h{i}")
    succ: dict[str, list[tuple[str, int]]] = {}
    for a, b, t in m["edges"]:
        if key[a] != key[b]:
            succ.setdefault(key[a], []).append((key[b], t))
    edges: dict[tuple[str, str], str] = {}
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

#: Colour tokens, light then dark. The SVG uses them through CSS variables (``--xd-<name>``) whose
#: fallbacks are the light colours, so a saved ``.svg`` is a light paper figure and the page can
#: switch to dark by setting the variables.
COLORS = {
    "bg": ("#ffffff", "#15171c"), "fg": ("#1a1a1a", "#e6e7ea"), "muted": ("#555555", "#9da3ad"),
    "edge": ("#333333", "#c3c7ce"), "shape": ("#0057b7", "#7cb7ff"), "rule": ("#bbbbbb", "#3a3f48"),
    "s0": ("#cfe8ff", "#1d3a5a"), "s1": ("#ffe0cc", "#5a3621"), "s2": ("#d2f0ea", "#1b4841"),
    "s3": ("#fbd8e6", "#552639"), "s4": ("#ecedc8", "#45481f"), "mix": ("#eceff3", "#2b3038"),
    "norm": ("#fff6cc", "#4a421c"), "op": ("#ffffff", "#262a31"), "io": ("#d6f5d6", "#1d4a29"),
    "side": ("#e5d4ff", "#3b2c5e"), "sideline": ("#8850cc", "#b48cf2"),
    "panel": ("#f2f2f2", "#1d2026"), "panel2": ("#fafafa", "#23272e"), "frame": ("#666666", "#5d636d"),
    "h0": ("#f5f5f5", "#252830"), "h1": ("#fff0d9", "#3b2d1a"), "h2": ("#ffd7a6", "#5b3915"),
    "h3": ("#ffae66", "#8a4912"), "h4": ("#ff7a1a", "#c4590b"),
    "p0": ("#f5f5f5", "#1f2228"), "p1": ("#fff9f1", "#28231d"), "p2": ("#fff3e5", "#30261b"),
    "p3": ("#ffecd6", "#3a2918"), "p4": ("#ffe3c4", "#462c15"),
    "err": ("#ffd9d9", "#5a1c1c"), "errline": ("#c62828", "#ff6b6b"),
}  # fmt: skip
_SANS = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
_MONO = "'JetBrains Mono', 'DejaVu Sans Mono', Menlo, Consolas, monospace"
#: Text styles: size (px), weight, italic, monospace, colour token.
FONTS = {
    "tn": (12, 400, False, False, "fg"), "tb": (12, 600, False, False, "fg"),
    "ts": (10, 400, True, False, "muted"), "tsh": (10.5, 400, False, True, "shape"),
    "tp": (13, 600, False, False, "fg"), "tpm": (12, 400, False, False, "muted"),
    "th": (21, 700, False, False, "fg"), "tl": (12, 400, False, False, "muted"),
    "tc": (16, 700, False, False, "fg"), "tst": (12.5, 600, False, False, "fg"),
    "tsg": (10, 400, False, True, "muted"), "tstep": (11, 400, False, False, "fg"),
    "tnote": (11, 400, True, False, "muted"), "te": (12, 600, False, False, "errline"),
    "tem": (10.5, 400, False, True, "errline"), "tplus": (15, 500, False, False, "fg"),
    "tred": (12, 400, False, False, "errline"), "tx": (10.5, 600, False, False, "fg"),
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
_WIDE = {"→": 0.96, "×": 0.67, "·": 0.32, "✗": 0.88, "‖": 0.5, "…": 0.7, "≤": 0.67, "≥": 0.67, "µ": 0.6, "⋯": 0.9}


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

    ``svg()`` writes it as SVG; ``_png_matplotlib`` draws the same list with matplotlib.
    """

    def __init__(self) -> None:
        self.ops: list[tuple[Any, ...]] = []

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str | None = "edge",
             sw: float = 1.1, r: float = 6.0, dash: str | None = None) -> None:  # fmt: skip
        self.ops.append(("rect", x, y, w, h, fill, stroke, sw, r, dash))

    def text(self, x: float, y: float, runs: list[Run], anchor: str = "middle", halo: str | None = None) -> None:
        self.ops.append(("text", x, y, runs, anchor, halo))

    def line(self, pts: list[tuple[float, float]], stroke: str = "edge", sw: float = 1.2, dash: str | None = None,
             radius: float = 5.0) -> None:  # fmt: skip
        self.ops.append(("line", pts, stroke, sw, dash, radius))

    def head(self, x: float, y: float, dx: float, dy: float, color: str = "edge", size: float = 7.5) -> None:
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
    """The SVG's style sheet: colour classes over the ``--xd-*`` variables, and the text styles."""
    out = [f".f-{k}{{fill:var(--xd-{k},{v[0]})}}.k-{k}{{stroke:var(--xd-{k},{v[0]})}}" for k, v in COLORS.items()]
    for cls, (size, weight, italic, mono, color) in FONTS.items():
        out.append(f".{cls}{{font-family:{_MONO if mono else _SANS};font-size:{size}px;font-weight:{weight};"
                   f"{'font-style:italic;' if italic else ''}fill:var(--xd-{color},{COLORS[color][0]})}}")  # fmt: skip
    out.append("text{white-space:pre}.halo{paint-order:stroke;stroke-width:5px;stroke-linejoin:round}")
    return "".join(out)


def _svg(c: _Canvas, w: float, h: float) -> str:
    """The display list as a standalone SVG document."""
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
            _, x, y, dx, dy, color, size = op
            n = math.hypot(dx, dy) or 1.0
            ux, uy = dx / n, dy / n
            bx, by = x - ux * size, y - uy * size
            px, py = -uy * size * 0.42, ux * size * 0.42
            pts = f"{_num(x)},{_num(y)} {_num(bx + px)},{_num(by + py)} {_num(bx - px)},{_num(by - py)}"
            out.append(f'<polygon class="f-{color}" points="{pts}"/>')
        elif kind == "g":
            attrs = "".join(f' {k.replace("_", "-")}="{html.escape(str(v))}"' for k, v in op[1].items())
            out.append(f"<g{attrs}>")
        else:
            out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


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
    An edge that crosses a container's border goes through a port on its top
    (or bottom) edge, which takes part in the container's layout like a node;
    an edge that skips layers goes through one dummy per layer, shared by all
    edges from the same value, so fan-out draws as a tree.
    """

    PAD, TITLE, GAP, LANE = 14.0, 30.0, 30.0, 6.0

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
        B, members = self.B, {i: [] for i in drawn}  # what each drawn box directly holds
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
        self.stream = {n: f"s{min(k, 4)}" for k, n in enumerate(streams)}
        self.role: dict[str, str] = {}
        for n in self.order:
            kind, label = kinds[n], self.nodes[n][2]
            by = {s for s in streams if n in reached[s]}
            from_side = not by and any(n in reached[s] for s in self.side)
            if kind in ("in", "out"):
                role = "side" if n in self.side else "io"
            elif kind == "op":
                role = "op"
            elif "norm" in self.B[int(n[1:])]["l"].lower():
                role = "norm"
            elif from_side:
                role = "side"
            elif len(by) == 1:
                role = self.stream[next(iter(by))]
            else:
                role = "mix"
            self.role[n] = role
        self.sidenodes = {n for n in self.order if self.role[n] == "side" and kinds[n] != "in"} | self.side

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
            elif kind == "call":
                b = self.B[int(n[1:])]
                out = [b["outs"]] if b["outs"] and b.get("err") != 2 else []
        return out

    def intypes(self, n: str) -> list[str]:
        return [t for u, v, _, _, t in self.edges if v == n]

    def heat_of(self, i: int) -> str | None:
        b = self.B[i]
        return f"h{heat(b['s'])}" if self.m.get("timing") and "s" in b else None

    # -- node and container sizes

    def lines(self, n: str) -> list[list[Run]]:
        kind, label = self.nodes[n][1], self.nodes[n][2]
        if kind == "in":
            return [[(label.split(": ", 1)[0], "tb")]]
        if kind == "out":
            return [[("return", "tb")]]
        if kind == "op":
            if n == self.fail:
                return [[(label, "te")]]
            return [[("+", "tplus")]] if label == "+" else [[(label, "tn")]]
        i = int(n[1:])
        b, err = self.B[i], self.m.get("error")
        name = _display(b["l"])
        if b.get("err") == 2 and err:
            out = [[("✗ " + name, "te")]]
            if err["got"]:
                arg = "return value" not in err["head"]
                out += [[(f"received {err['got']}" if arg else f"returned {err['got']}", "tem")],
                        [(f"declared {err['declared']}", "tem")]]  # fmt: skip
            return out
        first: list[Run] = [(name, "tn")]
        if n in self.card:
            first.append(("  " + self.B[self.card[n]]["l"], "tx"))
        out = [first]
        if b.get("vectorize"):
            out.append([("vectorized", "ts")])
        if t := _timing(self.m, b):
            out.append([(t, "ts")])
        return out

    def node_size(self, n: str) -> tuple[float, float, float]:
        """(width, height, room below for the shape)"""
        lines = self.lines(n)
        below = 15.0 if self.shown(n) else 0.0
        if self.nodes[n][2] == "+" and self.nodes[n][1] == "op":
            return 24.0, 24.0, below
        w = max(sum(_tw(t, c) for t, c in line) for line in lines) + 28
        h = 16 + sum(16 if line[0][1] in ("tn", "tb", "te") else 13 for line in lines)
        kind = self.nodes[n][1]
        w = max(w, {"call": 110, "in": 84, "out": 84}.get(kind, 64))
        h = max(h, 32)
        if n in self.card:
            w, h = w + 5, h + 5
        return w, h, below

    def shown(self, n: str) -> bool:
        """Whether a node's shape is written under it: inputs, outputs, and where the shape changes."""
        kind = self.nodes[n][1]
        outs = self.outtypes(n)
        if not outs or (kind == "op" and self.nodes[n][2] == "+"):
            return False
        return kind in ("in", "out") or not set(outs) <= set(self.intypes(n))

    def title(self, c: int) -> list[Run]:
        b = self.B[c]
        if c in self.alias.values():
            inner = next(k for k, v in self.alias.items() if v == c)
            runs = [(_display(self.B[inner]["l"]), "tp"), ("  " + b["l"], "tp")]
            b = self.B[inner]
        elif b["k"] == "call":
            runs = [(_display(b["l"]), "tp")]
        else:
            runs = [(("loop  " if b["l"].startswith("×") else "") + b["l"], "tp")]
        if b.get("vectorize"):
            runs.append(("  vectorized", "tpm"))
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
        fed = {key[1] for key in edges if src[key].startswith("i:")}
        topo = self._topo(items, pred)
        for it in reversed(topo):
            kind = self.nodes[it][1] if it in self.nodes else ""
            if not pred[it] and succ[it] and kind != "in" and it not in fed:
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
        for (u, t), s in src.items():
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
        for it in sorted(L, key=lambda x: rank.get(x, 0)):
            layers[L[it]].append(it)
        self._order(layers, hops)
        xs = self._xs(c, layers, hops)
        ys, H, gaps = self._ys(c, layers, hops, xs, panel)
        self.layer[c] = L
        at = {it: (xs[it] + (self.PAD if panel else 0.0), ys[L[it]]) for it in L}
        width = max([xs[it] + self.size[it][0] for it in L] + [0.0]) + (2 * self.PAD if panel else 0.0)
        if panel:
            width = max(width, sum(_tw(t, s) for t, s in self.title(c)) + 2 * self.PAD)
            content = max([xs[it] + self.size[it][0] for it in L] + [0.0]) + 2 * self.PAD
            shift = (width - content) / 2
            at = {it: (x + shift, y) for it, (x, y) in at.items()}
        self.at[c] = at
        self._route(c, layers, hops, L, at, gaps)
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

    def _order(self, layers: list[list[str]], hops: list[tuple[str, str, str]]) -> None:
        """Barycentre sweeps (down, then up), keeping the order with the fewest crossings."""
        up: dict[str, list[str]] = defaultdict(list)
        down: dict[str, list[str]] = defaultdict(list)
        for a, b, _ in hops:
            down[a].append(b)
            up[b].append(a)
        best, best_n = [list(x) for x in layers], self._crossings(layers, hops)
        for sweep in range(8):
            rng = range(1, len(layers)) if sweep % 2 == 0 else range(len(layers) - 2, -1, -1)
            for k in rng:
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
        gap = lambda a, b: 26.0 if a[1] != ":" and b[1] != ":" else 12.0  # boxes apart, dummies and ports closer
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
                for _ in range(n):
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
            lead = self.TITLE if (panel and k == 0) else 12.0
            tail = 24.0 if k in labelled else 16.0
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

    # -- painting

    def paint(self, cv: _Canvas, x0: float, y0: float) -> None:
        """Draw the main flow with its top left at (x0, y0)."""
        self.x0, self.y0 = x0, y0
        self.titles: list[Any] = []
        self.fills: dict[int, str] = {-1: "bg"}
        self._panels(cv, -1, 0)
        edges = sorted(self.edges, key=lambda e: ((e[0], e[1]) in self.path, e[0] in self.sidenodes))
        for u, v, *_ in edges:
            self._edge(cv, u, v)
        for x, y, runs, fill, inner, i in self.titles:  # over the edges that enter the panel
            cv.group(class_="pt", data_box=inner, data_loop=i)
            cv.text(x, y, runs, anchor="start", halo=fill)
            cv.end()
        for n in self.order:
            self._node(cv, n)

    def _panels(self, cv: _Canvas, c: int, depth: int) -> None:
        for it in self.items[c]:
            if it[0] != "c":
                continue
            i = int(it[1:])
            x, y, w, h = self.box_at(it)
            x, y = x + self.x0, y + self.y0
            b = self.B[i]
            inner = next((k for k, v in self.alias.items() if v == i), i)
            fill = "panel" if depth % 2 == 0 else "panel2"
            self.fills[i] = fill
            stroke, sw, dash = "frame", 1.0, None
            if (hk := self.heat_of(i)) is not None:
                fill = "p" + hk[1:]
                self.fills[i] = fill
            if b.get("err") == 1 or self.B[inner].get("err") == 1:
                stroke, sw = "errline", 1.4
            if b["k"] != "call" and i not in self.alias.values():
                dash = "5 3"  # a loop or branch around its contents
            openable = self.B[inner]["k"] == "call"
            cv.group(class_="pn" + (" fold" if openable else ""), data_box=inner, data_loop=i)
            cv.rect(x, y, w, h, fill, stroke, sw, r=10, dash=dash)
            cv.end()
            self.titles.append((x + self.PAD, y + 20, self.title(i), fill, inner, i))
            self._panels(cv, i, depth + 1)

    def _edge(self, cv: _Canvas, u: str, v: str) -> None:
        pts = [(x + self.x0, y + self.y0) for x, y in self.edge_path(u, v)]
        if len(pts) < 2:
            return
        red, side = (u, v) in self.path, u in self.sidenodes
        color = "errline" if red else "sideline" if side else "edge"
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        n = math.hypot(xb - xa, yb - ya) or 1.0
        tip = pts[-1]
        pts[-1] = (xb - (xb - xa) / n * 6.5, yb - (yb - ya) / n * 6.5)
        cv.line(pts, color, 1.8 if red else 1.2, "5 3" if side and not red else None)
        cv.head(tip[0], tip[1], xb - xa, yb - ya, color)
        info = next((e for e in self.edges if e[0] == u and e[1] == v), None)
        if info and info[4] and info[4] != info[3] and abs(xb - xa) < 0.5 and self.nodes[v][1] == "call":
            cv.text(xb + 5, yb - 9, [(info[4], "tsh")], anchor="start", halo=self.fills[self.ncont[v]])

    def _node(self, cv: _Canvas, n: str) -> None:
        x, y, w, h = self.box_at(n)
        x, y = x + self.x0, y + self.y0
        kind, label = self.nodes[n][1], self.nodes[n][2]
        fill, stroke, sw = self.role[n], "edge", 1.1
        box = int(n[1:]) if kind == "call" else -1
        if box >= 0 and (hk := self.heat_of(box)) is not None:
            fill = hk
        if n == self.fail:
            fill, stroke, sw = "err", "errline", 2.0
        elif n in self.feeds:
            stroke, sw = "errline", 1.5
        cv.group(class_="nd" + (" open" if box >= 0 and self.B[box]["o"] and n != self.fail else ""),
                 data_id=n, data_box=box)  # fmt: skip
        if kind == "op" and label == "+":
            cv.rect(x, y, w, h, fill, stroke, sw, r=w / 2)
            cv.text(x + w / 2, y + h / 2 + 5, [("+", "tplus")])
        else:
            if n in self.card:
                cv.rect(x + 5, y, w - 5, h - 5, fill, stroke, sw, r=6)
                y, w, h = y + 5, w - 5, h - 5
            cv.rect(x, y, w, h, fill, stroke, sw, r=6)
            lines = self.lines(n)
            sizes = [16 if line[0][1] in ("tn", "tb", "te") else 13 for line in lines]
            ty = y + (h - sum(sizes)) / 2
            for line, s in zip(lines, sizes):
                ty += s
                cv.text(x + w / 2, ty - (4 if s == 16 else 3), line)
        if self.shown(n):
            cv.text(x + w / 2, y + h + 12, [(", ".join(self.outtypes(n)), "tsh")], halo=self.fills[self.ncont[n]])
        cv.end()


# ------------------------------------------------------------------ drawing: the figure


def _steps(m: dict[str, Any], expanded: set[int], i: int, color: str = "mix") -> list[dict[str, Any]]:
    """The steps of call ``i`` opened one level: its nodes by dependency depth, alike ones merged."""
    nodes, edges, _, _ = _view(m, expanded | {i}, full=True)
    B = m["boxes"]

    def inside(box: int) -> int | None:  # the loop or branch right under i, or i itself; None if outside i
        last = None
        while box >= 0:
            if box == i:
                return last if last is not None else i
            if B[box]["k"] == "loop":
                last = box
            box = B[box]["p"]
        return None

    mine = {n[0]: n for n in nodes if inside(n[3]) is not None}
    rank: dict[str, int] = {}
    for n in mine:  # nodes come in a topological order
        rank[n] = max([rank[u] + 1 for u, v, *_ in edges if v == n and u in rank] + [0])
    groups: dict[tuple[int, int | None], list[str]] = {}
    for n in mine:
        groups.setdefault((rank[n], inside(mine[n][3])), []).append(n)
    steps = []
    for (r, loop), ns in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1)):
        names = []
        for n in ns:
            kind, label = mine[n][1], mine[n][2]
            name = _display(B[int(n[1:])]["l"]) if kind == "call" else ("residual +" if label == "+" else label)
            names.append(name)
        uniq = list(dict.fromkeys(names))
        text = f"{uniq[0]}  ×{len(names)}" if len(uniq) == 1 and len(names) > 1 else "  ‖  ".join(uniq)
        role = "op"
        first = mine[ns[0]]
        if first[1] == "call":
            b = B[int(first[0][1:])]
            role = "norm" if "norm" in b["l"].lower() else color
            if m.get("timing") and "s" in b:
                role = f"h{heat(b['s'])}"
        outs = list(dict.fromkeys(t0 for u, _, _, t0, _ in edges if u in ns))
        loop_label = B[loop]["l"] if loop is not None and loop != i else None
        steps.append(dict(text=text, role=role, shape=", ".join(outs[:2]) + (", …" if len(outs) > 2 else ""),
                          loop=loop if loop != i else None, loop_label=loop_label, nodes=ns))  # fmt: skip
    return steps


def _sig(b: dict[str, Any]) -> str:
    return f"{b['ins']} → {b['outs'] or '?'}" + ("  per element" if b.get("vectorize") else "")


class _Figure:
    """A whole figure: header, main flow, the internals column and the legend, as a display list."""

    M = 24.0

    def __init__(self, m: dict[str, Any], expanded: set[int]):
        self.m, self.expanded = m, expanded
        self.scene = _Scene(m, expanded)
        self.scene.layout()
        self.cv = _Canvas()
        self.paint()

    def legend_line(self) -> str:
        sc, m = self.scene, self.m
        types = [n[2] for n in sc.nodes.values()] + [t for e in sc.edges for t in e[2:]]
        loops = [b["l"][2:] for i, b in enumerate(m["boxes"]) if b["k"] == "loop" and b["l"].startswith("× ")]
        parts = []
        for name in _names(types) + [n for n in loops if n.isidentifier()]:
            if any(p.split(" ")[0] == name for p in parts):
                continue
            if name in m.get("lists", {}):
                parts.append(f"{name} = len({m['lists'][name]})")
            else:
                parts.append(name)
        parts += [f"{k} = {v}" for k, v in m.get("given", {}).items()]
        return "sizes:  " + "  ·  ".join(parts) if parts else ""

    def paint(self) -> None:
        cv, m, sc, M = self.cv, self.m, self.scene, self.M
        title = _display(m["name"])
        heading = [(title, "th")] + ([("  " + m["name"][len(title):], "tl")] if title != m["name"] else [])
        y = M + 20
        cv.text(M, y, heading, anchor="start")
        notes: list[list[Run]] = []
        if line := self.legend_line():
            notes.append([(line, "tl")])
        t, err = m.get("timing"), m.get("error")
        if t:
            if t["unit"] == "s":
                notes.append([(f"time per call measured with jax.profiler ({t['backend']}): {_amount(t['total'], 's')} of XLA op time, "
                               f"{_amount(t['wall'], 's')} wall, median of {t['runs']} runs", "tl")])  # fmt: skip
            else:
                notes.append([(f"ESTIMATED from FLOPs, not measured ({t['why']}): {_amount(t['total'], 'flop')} per call", "tred")])
        if err:
            notes.append([("✗ " + err["head"], "tred")])
        for runs in notes:
            y += 19
            cv.text(M, y, runs, anchor="start")
        top = y + 26
        sc.paint(cv, M, top)
        main_w = max([sc.size[it][0] + sc.at[-1][it][0] for it in sc.at[-1]] + [200.0])
        main_h = max([sc.size[it][1] + sc.size[it][2] + sc.at[-1][it][1] for it in sc.at[-1]] + [40.0])
        head_w = max(_tw(title, "th"), *[sum(_tw(a, b) for a, b in r) for r in notes], 0)
        right = M + max(main_w, head_w)
        bottom = top + main_h
        inner = self.internals(right + 36, M, max(bottom, M + 200))
        if inner is not None:
            right, bottom = max(right, inner[0]), max(bottom, inner[1])
        bottom = self.legend(M, bottom + 30, right)
        self.w, self.h = right + M, bottom + M

    def internals(self, x0: float, y0: float, main_bottom: float) -> tuple[float, float] | None:
        """The column of each call drawn closed and able to open, as a titled stack of its steps."""
        m, sc, cv = self.m, self.scene, self.cv
        seen, stacks = set(), []
        for n in sc.order:
            if sc.nodes[n][1] != "call" or n == sc.fail:
                continue
            i = int(n[1:])
            b = m["boxes"][i]
            if not b["o"] or (b["l"], b.get("vectorize")) in seen:
                continue
            seen.add((b["l"], b.get("vectorize")))
            color = sc.role[n] if sc.role[n] not in ("norm", "op", "io") else "mix"
            steps = _steps(m, self.expanded, i, color)
            if steps:
                stacks.append((i, steps))
        if not stacks:
            return None
        sized = []
        for i, steps in stacks:
            b = m["boxes"][i]
            sw = max([_tw(s["text"], "tstep") + 26 for s in steps] + [150.0])
            shw = max([_tw(s["shape"], "tsh") for s in steps] + [0.0])
            loops = any(s["loop"] is not None for s in steps)
            left = 34.0 if loops else 0.0
            tw = max(_tw(_display(b["l"]), "tst"), _tw(_sig(b), "tsg"))
            w = max(left + sw + (shw + 10 if shw else 0), tw)
            h = 42 + len(steps) * 26 + (len(steps) - 1) * 14
            sized.append((i, steps, w, h, sw, left))
        total = sum(s[3] + 26 for s in sized)
        ncol = 1 if total <= max(main_bottom - y0 - 60, 360) else 2 if total <= 2.2 * max(main_bottom - y0 - 60, 360) else 3
        cols: list[list[Any]] = [[] for _ in range(ncol)]
        heights = [0.0] * ncol
        for s in sized:  # in order, each into the shortest column
            k = heights.index(min(heights))
            cols[k].append(s)
            heights[k] += s[3] + 26
        cx = x0 + 28
        cv.text(cx, y0 + 20, [("Component internals", "tc")], anchor="start")
        sub = "each call on the left, opened one level: its steps from top to bottom"
        cv.text(cx, y0 + 39, [(sub, "tl")], anchor="start")
        ytop = y0 + 66
        bottom = ytop
        for col in cols:
            if not col:
                continue
            cw = max(s[2] for s in col)
            y = ytop
            for i, steps, w, h, sw, left in col:
                self.stack(cx, y, i, steps, sw, left)
                y += h + 26
            bottom = max(bottom, y - 26)
            cx += cw + 32
        cv.line([(x0, y0 + 4), (x0, max(bottom, main_bottom))], "rule", 1.0, radius=0)
        return max(cx - 32, x0 + 28 + _tw(sub, "tl")), bottom

    def stack(self, x: float, y: float, i: int, steps: list[dict[str, Any]], sw: float, left: float) -> None:
        cv, b = self.cv, self.m["boxes"][i]
        cv.group(class_="st", data_box=i)
        cv.text(x, y + 13, [(_display(b["l"]), "tst")], anchor="start")
        cv.text(x, y + 29, [(_sig(b), "tsg")], anchor="start")
        sx, yy = x + left, y + 42
        spans: dict[int, list[float]] = {}
        for k, s in enumerate(steps):
            cv.rect(sx, yy, sw, 26, s["role"], "edge", 1.0, r=5)
            cv.text(sx + sw / 2, yy + 17, [(s["text"], "tstep")])
            if s["shape"]:
                cv.text(sx + sw + 8, yy + 17, [(s["shape"], "tsh")], anchor="start")
            if s["loop"] is not None:
                lo, hi = spans.get(s["loop"], [yy, yy + 26])
                spans[s["loop"]] = [min(lo, yy), max(hi, yy + 26)]
            if k + 1 < len(steps):
                cv.line([(sx + sw / 2, yy + 26), (sx + sw / 2, yy + 26 + 14 - 6)], "edge", 1.1)
                cv.head(sx + sw / 2, yy + 40, 0, 1, "edge", 6)
            yy += 40
        for loop, (lo, hi) in spans.items():
            bx = sx - 9
            cv.line([(bx + 5, lo + 3), (bx, lo + 3), (bx, hi - 3), (bx + 5, hi - 3)], "frame", 1.1, radius=2)
            cv.text(bx - 4, (lo + hi) / 2 + 4, [(self.m["boxes"][loop]["l"], "tx")], anchor="end")
        cv.end()

    def legend(self, x0: float, y: float, right: float) -> float:
        """Swatches for the colours used, wrapped to the figure's width; returns the bottom."""
        m, sc, cv = self.m, self.scene, self.cv
        items: list[tuple[str, str, str | None]] = []  # (fill, text, stroke)
        roles = {op[5] for op in cv.ops if op[0] == "rect"}  # the fills drawn
        if m.get("timing"):
            lo, hi = [0.0, *HEAT], [*HEAT, 1.0]
            pct = lambda s: f"{s:.0%}"
            items += [(f"h{k}", f"< {pct(hi[0])}" if k == 0 else f"≥ {pct(lo[k])}" if k == len(HEAT) else f"{pct(lo[k])}–{pct(hi[k])}", None)
                      for k in range(len(HEAT) + 1)]  # fmt: skip
        for k, n in enumerate(sc.stream):
            if sc.stream[n] in roles:
                items.append((sc.stream[n], f"{sc.nodes[n][2].split(':')[0]} stream", None))
        if "mix" in roles:
            items.append(("mix", "mixes inputs", None))
        if "side" in roles:
            items.append(("side", "side input", None))
        if "norm" in roles:
            items.append(("norm", "normalization", None))
        if "op" in roles:
            items.append(("op", "primitive op", None))
        if "io" in roles:
            items.append(("io", "input / output", None))
        if sc.conts:
            items.append(("panel", "call opened · loop (dashed)", "frame"))
        if m.get("error"):
            items += [("err", "failing call", "errline"), ("", "path into it", "errline")]
        x = x0
        for fill, text, stroke in items:
            w = max(_tw(text, "tl") + 24, 64)
            if x + w > right and x > x0:
                x, y = x0, y + 30
            if not fill:  # a line, not a box
                w = _tw(text, "tl") + 40
                cv.line([(x, y + 11), (x + 26, y + 11)], stroke or "edge", 1.8)
                cv.text(x + 32, y + 15, [(text, "tl")], anchor="start")
            else:
                cv.rect(x, y, w, 22, fill, stroke or "edge", 1.4 if stroke == "errline" else 1.0, r=5)
                cv.text(x + w / 2, y + 15, [(text, "tl")])
            x += w + 8
        return y + 22


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
    for k, targets in clicks.items():  # a click past the budget goes to the nearest depth view
        for i, t in targets.items():
            if t not in svgs:
                want = states.get(t) or {int(x) for x in t.split(",") if x}
                fits = [d for d in depths if want <= states[d]]
                targets[i] = fits[0] if fits else depths[-1]
    return svgs, clicks, depths


def _page(m: dict[str, Any]) -> str:
    """The interactive page: every view's SVG, and the boxes' details, in one HTML file."""
    svgs, clicks, depths = _states(m)
    start = ",".join(map(str, sorted(_opened(m))))
    boxes = [dict(name=b["l"], k=b["k"], sig=b.get("sig", ""), loc=b.get("loc", ""), ins=b.get("ins", ""),
                  outs=b.get("outs", ""), src=b.get("src", ""), vectorize=b.get("vectorize", False),
                  err=b.get("err", 0), t=b.get("t"), s=b.get("s"), n=b.get("n"), o=b.get("o", False))
             for b in m["boxes"]]  # fmt: skip
    data = dict(name=m["name"], start=start, depths=depths, svgs=svgs, clicks=clicks, boxes=boxes, sources=m["sources"],
                timing=m.get("timing"), error=m.get("error"), colors={k: v[1] for k, v in COLORS.items()},
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


def _png(fig: "_Figure", path: Path, scale: float = 2.0) -> None:
    """Rasterise the figure: with Playwright, else a headless Chromium, else matplotlib (see ``_png_matplotlib``)."""
    svg, w, h = _svg(fig.cv, fig.w, fig.h), fig.w, fig.h
    page = f'<!doctype html><html><body style="margin:0;background:#fff">{svg}</body></html>'
    size = (math.ceil(w), math.ceil(h))
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        sync_playwright = None
    if sync_playwright is not None:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                tab = browser.new_page(viewport={"width": size[0], "height": size[1]}, device_scale_factor=scale)
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
        raise RuntimeError(
            "saving a PNG needs a browser to draw the SVG (Playwright, or Chromium / Chrome on the PATH) "
            "or matplotlib; none was found. Save an .svg or .html instead."
        ) from None
    _png_matplotlib(fig, path, scale)


def _png_matplotlib(fig: "_Figure", path: Path, scale: float) -> None:
    """Draw the display list again with matplotlib: same layout, its own fonts (the fallback without a browser)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon

    col = lambda k: COLORS[k][0]
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
            _, x, y, dx, dy, color, size = op
            n = math.hypot(dx, dy) or 1.0
            ux, uy = dx / n, dy / n
            bx, by = x - ux * size, y - uy * size
            px, py = -uy * size * 0.42, ux * size * 0.42
            ax.add_patch(Polygon([(x, y), (bx + px, by + py), (bx - px, by - py)], closed=True,
                                 facecolor=col(color), edgecolor="none", zorder=z))  # fmt: skip
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

    def save(self, path: str | Path) -> Path:
        """Write the figure: ``.svg``, ``.png`` (2x, needs a browser or matplotlib) or ``.html`` (interactive)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".svg":
            path.write_text(self.svg, encoding="utf-8")
        elif suffix == ".png":
            _png(self._fig(), path)
        elif suffix in (".html", ".htm", ""):
            path.write_text(self.html(), encoding="utf-8")
        else:
            raise ValueError(f"cannot save a diagram as {suffix!r}: use .svg, .png or .html")
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
            raise TypeError(f"{f.name}: unknown size or parameter `{k}`")
    length = lambda x: "L" if lists and not given and x == 3 else fmt_dim(x)
    b = _Builder(hook, length)
    b.walk(jaxpr.jaxpr, -1, {})
    model = b.model(getattr(f.raw, "__qualname__", f.name), depth)
    if lists and not given:  # for the figure's line of sizes: L = len(blocks)
        model["lists"] = {"L": ", ".join(dict.fromkeys(field(p) for p in lists))}
    model["given"] = {k: int(v) for k, v in sizes.items() if k not in f.params and isinstance(v, int)}
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
    for (box, _, _), x in zip(b.atoms, flops):
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

    def __call__(self, target: Any, /, depth: int = 1, **sizes: Any) -> Diagram:
        """Draw an ``@fn``, an unbound struct method or a bound one, without data.

        ``depth=1`` shows the calls ``target`` makes as nodes, ``depth=2`` opens
        those one level further, and so on. Keywords set sizes (``T=128``), static
        parameters (``heads=4``) and the length of lists of modules (``L=12`` or
        ``blocks=12``), as for ``expex.cost``. See the module docstring.
        """
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
                raise ValueError("this error has no diagram: only a ShapeError leaving a typed call has one")
            return d
        return _explain(target, args, kwargs)

    def __repr__(self) -> str:
        return "<function expex.diagram>"


diagram = DiagramFn()

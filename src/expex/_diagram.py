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
import re
import statistics
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

from . import _cost, _fn
from ._dims import fmt_dim, short_dtype
from ._errors import ShapeError
from ._jaxpr import sub_jaxprs

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
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"
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
                outs = [[self.atom(b, label, "op", eqn.invars, env, eqn)]] * len(eqn.outvars)
            env.update({o: s for o, s in zip(eqn.outvars, outs) if _type(o.aval) is not None})

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


def _esc(text: str) -> str:
    """Text for a quoted Mermaid label: brackets, quotes and tags as entity codes."""
    return "".join(f"#{ord(c)};" if c in '"[]<>#&|{}_*`~' else c for c in text)


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


def _view(m: dict[str, Any], expanded: set[int]) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str]], list[int], str | None]:  # fmt: skip
    """The visible graph: nodes (id, kind, label, parent box), edges (from, to, type), drawn boxes,
    and the node that failed (for ``explain``), if any.

    Atoms inside a closed call become that call's node; unlabelled atoms elsewhere
    are passed through. An edge is labelled with the type of the array as it
    leaves its source, or as it enters its target if that is a call (whose
    label says what it takes).

    Mirrored in ``_diagram.html`` (``view``), which redraws as calls are opened.
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
                edges.setdefault((u, v), m["types"][t if nodes[v][1] == "call" else t0])
            else:
                todo.extend((w, t0, tw) for w, tw in reversed(succ.get(v, [])))
    drawn: set[int] = set()
    for _, _, _, box in nodes.values():
        while box >= 0 and box not in drawn:
            drawn.add(box)
            box = boxes[box]["p"]
    fail = key[m["error"]["atom"]] if m.get("error") else None
    return list(nodes.values()), [(u, v, t) for (u, v), t in edges.items()], sorted(drawn), fail


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


#: Mermaid styles, light then dark (the page switches; the Mermaid text is light).
STYLES = {
    "calls": ("fill:#dbeafe,stroke:#2563eb,color:#0f172a", "fill:#1e3a8a,stroke:#60a5fa,color:#e2e8f0"),
    "ops": ("fill:#fef3c7,stroke:#d97706,color:#0f172a", "fill:#78350f,stroke:#f59e0b,color:#fef3c7"),
    "io": ("fill:#dcfce7,stroke:#16a34a,color:#0f172a", "fill:#14532d,stroke:#22c55e,color:#dcfce7"),
    "heat0": ("fill:#f1f5f9,stroke:#94a3b8,color:#334155", "fill:#1e293b,stroke:#475569,color:#cbd5e1"),
    "heat1": ("fill:#fff7ed,stroke:#fdba74,color:#0f172a", "fill:#2a1a10,stroke:#9a3412,color:#fde7d0"),
    "heat2": ("fill:#ffedd5,stroke:#fb923c,color:#0f172a", "fill:#47210c,stroke:#c2410c,color:#ffedd5"),
    "heat3": ("fill:#fdba74,stroke:#ea580c,color:#0f172a", "fill:#7c2d12,stroke:#f97316,color:#ffedd5"),
    "heat4": ("fill:#f97316,stroke:#9a3412,stroke-width:2px,color:#1c0a00", "fill:#c2410c,stroke:#fdba74,stroke-width:2px,color:#fff7ed"),  # noqa: E501
    "error": ("fill:#fee2e2,stroke:#dc2626,stroke-width:3px,color:#7f1d1d", "fill:#450a0a,stroke:#f87171,stroke-width:3px,color:#fee2e2"),  # noqa: E501
    "feeds": ("stroke:#dc2626,stroke-width:2px", "stroke:#f87171,stroke-width:2px"),
}  # fmt: skip
#: The edges on the path into a failing call, light then dark.
PATH = ("stroke:#dc2626,stroke-width:2.5px,color:#b91c1c", "stroke:#f87171,stroke-width:2.5px,color:#fca5a5")
#: Opened boxes: (calls, loops, failing) strokes, light then dark.
FRAME = (("#2563eb", "#9333ea", "#dc2626"), ("#60a5fa", "#c084fc", "#f87171"))


def _part(style: str, name: str) -> str:
    """``#9a3412`` from ``fill:...,stroke:#9a3412,...`` and ``"stroke"``."""
    return next(kv.split(":", 1)[1] for kv in style.split(",") if kv.split(":", 1)[0] == name)


def _mermaid(m: dict[str, Any], expanded: set[int]) -> str:
    nodes, edges, drawn, fail = _view(m, expanded)
    shape = {"call": '["{}"]', "op": '(["{}"])', "in": '[/"{}"/]', "out": '[\\"{}"\\]'}
    out = ["flowchart TD"]

    items: dict[int, list[Any]] = {}  # box -> its nodes and boxes, in the order they first appear
    for node in nodes:
        items.setdefault(node[3], []).append(node)
        box = node[3]
        while box >= 0 and box not in items.get(m["boxes"][box]["p"], []):
            items.setdefault(m["boxes"][box]["p"], []).append(box)
            box = m["boxes"][box]["p"]

    def emit(box: int, ind: str) -> None:
        for it in items.get(box, []):
            if isinstance(it, int):
                b = m["boxes"][it]
                share = f" · {b['s']:.1%}" if m.get("timing") and "s" in b else ""  # long titles wrap over the contents
                title = b["l"] + (" · vectorized" if b.get("vectorize") else "") + share
                out.append(f'{ind}subgraph b{it} ["{_esc(title)}"]')
                emit(it, ind + "  ")
                out.append(f"{ind}end")
            else:
                text = _esc(it[2])  # a line break in the quotes is a line break in the label
                out.append(ind + it[0] + shape[it[1]].format(text))

    emit(-1, "  ")
    out += [f'  {u} -->|"{_esc(t)}"| {v}' for u, v, t in edges]
    groups: dict[str, list[str]] = {}
    for n, k, _, _ in nodes:
        b = m["boxes"][int(n[1:])] if k == "call" else None
        cls = {"call": "calls", "op": "ops", "in": "io", "out": "io"}[k]
        if b is not None and "s" in b:
            cls = f"heat{heat(b['s'])}"
        groups.setdefault("error" if n == fail else cls, []).append(n)
    feeds, path = _feeds(edges, fail)
    if feeds:
        groups["feeds"] = sorted(feeds, key=[n[0] for n in nodes].index)
    for cls in STYLES:
        if cls in groups or cls in ("calls", "ops", "io"):
            out.append(f"  classDef {cls} {STYLES[cls][0]}")
    out += [f"  class {','.join(ids)} {cls}" for cls in STYLES if (ids := groups.get(cls))]
    for i in drawn:
        b = m["boxes"][i]
        color = FRAME[0][2] if b.get("err") else FRAME[0][0] if b["k"] == "call" else FRAME[0][1]
        fill = "transparent"
        if "s" in b:  # a tint of the heat step's stroke
            color = _part(STYLES[f"heat{heat(b['s'])}"][0], "stroke")
            fill = color + "26"
        out.append(f"  style b{i} fill:{fill},stroke:{color},stroke-dasharray:4 3")
    if path:
        out.append(f"  linkStyle {','.join(map(str, sorted(path)))} {PATH[0]}")
    return "\n".join(out)


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


class Diagram:
    """A traced function's structure. ``str()`` / ``.mermaid`` is Mermaid text at ``depth``.

    ``nodes``, ``edges`` and ``boxes`` describe the same view for programs;
    ``text`` is an outline of every call, with ``timings`` (from ``diagram.timed``)
    or the ``error`` (from ``diagram.explain``); ``save(path)`` writes the
    interactive page, also shown in notebooks.
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
        self.mermaid = _mermaid(model, opened)

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
        return self.mermaid

    def __repr__(self) -> str:
        return f"<Diagram of {self.model['name']}: {len(self.nodes)} nodes, depth {self.depth}>"

    def html(self, mermaid_url: str = MERMAID) -> str:
        """The interactive page: pan and zoom, click a call to open it, hover for details."""
        page = _PAGE.read_text(encoding="utf-8")
        model = {**self.model, "styles": STYLES, "path": PATH, "frame": FRAME, "heat": HEAT}
        data = json.dumps(model).replace("</", "<\\/")
        return (page.replace("__TITLE__", html.escape(self.model["name"]))
                .replace("__MERMAID__", mermaid_url).replace("/*DATA*/null", data))  # fmt: skip

    def save(self, path: str | Path) -> Path:
        """Write the interactive page (a standalone HTML file; Mermaid comes from a CDN)."""
        path = Path(path)
        path.write_text(self.html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return f'<iframe srcdoc="{html.escape(self.html())}" style="width:100%;height:600px;border:0"></iframe>'


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
    return Diagram(b.model(getattr(f.raw, "__qualname__", f.name), depth))


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

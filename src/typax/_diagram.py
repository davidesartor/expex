"""Architecture diagrams: an ``@fn`` or a module drawn as a Mermaid flowchart.

``diagram(f)`` traces ``f`` like ``tx.cost`` and ``tx.prove`` do, with symbolic
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
  shapes (per element for ``.lift``). Shallower calls are boxes around their
  contents, so ``depth=2`` shows a block's attention and MLP.
- Operations outside any call at that depth are drawn only when they say what
  the model does: ``matmul``, ``index``, reductions (``sum``, ``max``),
  ``conv``, ``scatter``, ... Reshapes, broadcasts, elementwise arithmetic and
  ``jit`` wrappers are passed through: edges go straight across them.
- A staged loop is a box labelled with its trip count: ``× L`` for a list of
  modules (``L`` unless given, like ``tx.cost``), ``× T`` for a loop over time.
  Traced ``if`` and ``while`` are boxes too.
- Edges follow the data (through passed-through operations), labelled with the
  array's type as it leaves a node, or as it enters a call.
- Inputs are the parameters other than ``self``: a module's own arrays reach
  nearly every node, so they are left out.
- In Mermaid labels, characters Mermaid would read as syntax or markdown
  (``[ ] " _ *`` ...) are written as entity codes (``#91;``), which it renders.

``diagram.save(path)`` writes an interactive page (pan, zoom, click a call to
open or fold it, hover for its signature and source); ``str(diagram)`` is the
Mermaid text, which GitHub renders in a ```mermaid block.
"""

import html
import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from . import _cost, _fn
from ._dims import fmt_dim, short_dtype
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


class _Hook:
    """Installed as ``_fn.CALL_HOOK``: scopes every call and records what it was."""

    def __init__(self) -> None:
        self.uid, self.depth, self.root = 0, 0, -1
        self.calls: dict[int, dict[str, Any]] = {}
        self.inputs: dict[str, str] = {}
        self.sources: dict[str, str] = {}

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
        if top and bound:  # mark the inputs, so the diagram can name them
            full = fn._sig.bind(*args, **kwargs)
            for k in bound:
                with jax.named_scope(f"typax-in.{k}"):
                    full.arguments[k] = _barrier(full.arguments[k])
                self.inputs[k] = _brief(bound[k]) or ""
            args, kwargs = full.args, full.kwargs
        self.depth += 1
        try:
            with jax.named_scope(f"{_short(fn.name)}#{uid}"):
                out = run(args, kwargs)
        finally:
            self.depth -= 1
        if fn.name not in self.sources:
            try:
                self.sources[fn.name] = textwrap.dedent(inspect.getsource(fn.raw))
            except (OSError, TypeError):
                self.sources[fn.name] = ""
        self.calls[uid] = dict(name=_short(fn.name), sig=fn.signature, loc=fn.location,
                               src=fn.name, ins=", ".join(ins), outs=_brief(out) or "")  # fmt: skip
        if top:
            with jax.named_scope("typax-out"):
                out = _barrier(out)
        return out


class _Builder:
    """Walks the jaxpr into boxes (calls, loops), atoms (equations, inputs) and edges."""

    def __init__(self, hook: _Hook, length: Any):
        self.hook, self.length = hook, length
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
            c = self.hook.calls.get(uid, dict(name="?", sig="", loc="", src="", ins="", outs=""))
            self.by_uid[uid] = self.box(parent, "call", c["name"], lift=lifted, **c)
        return self.by_uid[uid]

    def atom(self, box: int, label: str | None, kind: str, ins: list[Any], env: dict[Any, list[int]]) -> int:
        me = len(self.atoms)
        self.atoms.append([box, label, kind])
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
            for lift, short, uid in _TOKEN.findall(stack):
                if int(uid) != self.hook.root:
                    b = self.call(b, int(uid), bool(lift))
            srcs = lambda vs: [_get(env, v) for v in vs]
            outs: list[list[int]] | None = None
            if name == "optimization_barrier" and "typax-" in stack:
                kind, what = ("in", stack.split("typax-in.")[-1].split("/")[0]) if "typax-in." in stack else ("out", "")
                text = f"{what}: {self.hook.inputs.get(what, '')}" if kind == "in" else "return " + " ".join(
                    filter(None, (_type(v.aval) for v in eqn.outvars)))  # fmt: skip
                me = self.atom(-1, text, kind, eqn.invars if kind == "out" else [], env)
                outs = [[me]] * len(eqn.outvars)
            elif name == "scan":
                loop = self.box(b, "loop", f"× {self.length(p['length'])}")
                outs = self.sub(p["jaxpr"], loop, srcs(eqn.invars))
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
                outs = [[self.atom(b, label, "op", eqn.invars, env)]] * len(eqn.outvars)
            env.update({o: s for o, s in zip(eqn.outvars, outs) if _type(o.aval) is not None})

    def sub(self, jaxpr: Any, box: int, ins: list[list[int]]) -> list[list[int]]:
        jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
        env = dict(zip(jaxpr.invars, ins))
        self.walk(jaxpr, box, env)
        return [_get(env, v) for v in jaxpr.outvars]

    def model(self, name: str, depth: int) -> dict[str, Any]:
        has = [False] * len(self.boxes)  # anything to show inside (else a call cannot open)
        marks = [a[0] for a in self.atoms if a[1] and a[2] == "op"] + [b["p"] for b in self.boxes if b["k"] == "call"]
        for i in marks:
            while i >= 0 and not has[i]:
                has[i], i = True, self.boxes[i]["p"]
        for i, b in enumerate(self.boxes):
            b["o"] = b["k"] == "call" and has[i]
        types = sorted(self.types, key=self.types.__getitem__)
        return dict(name=name, depth=depth, boxes=self.boxes, atoms=self.atoms, edges=self.edges,
                    types=types, sources=self.hook.sources)  # fmt: skip


def _esc(text: str) -> str:
    """Text for a quoted Mermaid label: brackets, quotes and tags as entity codes."""
    return "".join(f"#{ord(c)};" if c in '"[]<>#&|{}_*`~' else c for c in text)


def _view(m: dict[str, Any], expanded: set[int]) -> tuple[list[tuple[str, str, str, int]], list[tuple[str, str, str]], list[int]]:  # fmt: skip
    """The visible graph: nodes (id, kind, label, parent box), edges (from, to, type), drawn boxes.

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
            b = boxes[r]
            text = b["l"] + (" · lift" if b["lift"] else "") + f"\n{b['ins']} → {b['outs']}"
            key.append(f"b{r}")
            nodes.setdefault(key[-1], (key[-1], "call", text, b["p"]))
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
    return list(nodes.values()), [(u, v, t) for (u, v), t in edges.items()], sorted(drawn)


def _mermaid(m: dict[str, Any], expanded: set[int]) -> str:
    nodes, edges, drawn = _view(m, expanded)
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
                out.append(f'{ind}subgraph b{it} ["{_esc(b["l"] + (" · lift" if b.get("lift") else ""))}"]')
                emit(it, ind + "  ")
                out.append(f"{ind}end")
            else:
                text = _esc(it[2])  # a line break in the quotes is a line break in the label
                out.append(ind + it[0] + shape[it[1]].format(text))

    emit(-1, "  ")
    out += [f'  {u} -->|"{_esc(t)}"| {v}' for u, v, t in edges]
    out += [
        "  classDef calls fill:#dbeafe,stroke:#2563eb,color:#0f172a",
        "  classDef ops fill:#fef3c7,stroke:#d97706,color:#0f172a",
        "  classDef io fill:#dcfce7,stroke:#16a34a,color:#0f172a",
    ]
    for cls, kinds in (("calls", ("call",)), ("ops", ("op",)), ("io", ("in", "out"))):
        if ids := [n for n, k, _, _ in nodes if k in kinds]:
            out.append(f"  class {','.join(ids)} {cls}")
    for i in drawn:
        color = "#2563eb" if m["boxes"][i]["k"] == "call" else "#9333ea"
        out.append(f"  style b{i} fill:transparent,stroke:{color},stroke-dasharray:4 3")
    return "\n".join(out)


class Diagram:
    """A traced function's structure. ``str()`` / ``.mermaid`` is Mermaid text at ``depth``.

    ``nodes``, ``edges`` and ``boxes`` describe the same view for programs;
    ``save(path)`` writes the interactive page, also shown in notebooks.
    """

    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.depth: int = model["depth"]
        opened = {i for i, b in enumerate(model["boxes"]) if b["k"] == "call" and b["d"] < self.depth}
        nodes, edges, drawn = _view(model, opened)
        #: (id, kind, label, parent box): kind is "call", "op", "in" or "out".
        self.nodes = nodes
        #: (from id, to id, array type)
        self.edges = edges
        #: box id -> label, for the boxes drawn (open calls, loops, branches)
        self.boxes = {f"b{i}": model["boxes"][i]["l"] for i in drawn}
        self.mermaid = _mermaid(model, opened)

    def __str__(self) -> str:
        return self.mermaid

    def __repr__(self) -> str:
        return f"<Diagram of {self.model['name']}: {len(self.nodes)} nodes, depth {self.depth}>"

    def html(self, mermaid_url: str = MERMAID) -> str:
        """The interactive page: pan and zoom, click a call to open it, hover for details."""
        page = _PAGE.read_text(encoding="utf-8")
        data = json.dumps(self.model).replace("</", "<\\/")
        return (page.replace("__TITLE__", html.escape(self.model["name"]))
                .replace("__MERMAID__", mermaid_url).replace("/*DATA*/null", data))  # fmt: skip

    def save(self, path: str | Path) -> Path:
        """Write the interactive page (a standalone HTML file; Mermaid comes from a CDN)."""
        path = Path(path)
        path.write_text(self.html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return f'<iframe srcdoc="{html.escape(self.html())}" style="width:100%;height:600px;border:0"></iframe>'


def diagram(target: Any, /, depth: int = 1, **sizes: Any) -> Diagram:
    """Draw an ``@fn``, an unbound struct method or a bound one, without data.

    ``depth=1`` shows the calls ``target`` makes as nodes, ``depth=2`` opens
    those one level further, and so on. Keywords set sizes (``T=128``), static
    parameters (``heads=4``) and the length of lists of modules (``L=12`` or
    ``blocks=12``), as for ``tx.cost``. See the module docstring.
    """
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

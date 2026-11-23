"""Rich display in notebooks: structs, shape errors, proofs and cost estimates as HTML (and text for IPython).

Each class keeps a one-line ``_repr_html_`` / ``_repr_pretty_`` that calls into
this module, so nothing here is imported, traced or computed until an object
is displayed:

- :func:`struct_html` / :func:`struct_text`: a struct as a collapsible tree of
  its fields, with shapes, dtypes, bytes and parameter counts. Arrays are not
  printed (a handful of numbers is); long lists of modules show their first
  and last elements.
- :func:`error_html`: a ``ShapeError``'s message and, below it, its diagram
  (``e.diagram``: traced when displayed, not when raised).
- :func:`reports_html`: ``expex.prove`` reports as a table, failures expandable.
- :func:`cost_html`: an ``expex.cost`` estimate, formulas and numbers.

The fragments share the palette and the light / dark switch of :mod:`.pages`.
"""

import dataclasses
import html
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import pages

#: Lists of modules longer than this show their first three elements and their last.
LONG = 8
#: Arrays with at most this many elements show their values.
SMALL = 8


@dataclass
class _Node:
    """One field of a struct, for display."""

    name: str
    kind: str  # struct, list, dict, array, static, more
    type: str  # f32[16, 16], Block[D=16], list[24], or a static value's repr
    declared: str = ""  # the field's annotation, when it differs from `type`
    params: int = 0
    nbytes: int = 0
    value: str = ""  # a small array's values
    children: list["_Node"] = field(default_factory=list)


def _array(x: Any) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, type)


def _is_key(x: Any) -> bool:
    try:
        import jax
        import jax.numpy as jnp

        return bool(jnp.issubdtype(x.dtype, jax.dtypes.prng_key))
    except Exception:
        return False


def _values(x: Any) -> str:
    """The values of a small concrete array (not of a tracer or ``ShapeDtypeStruct``)."""
    import jax
    from jax.core import Tracer

    if math.prod(x.shape) > SMALL or _is_key(x) or isinstance(x, (Tracer, jax.ShapeDtypeStruct)):
        return ""
    try:
        return np.array2string(np.asarray(x), precision=4, separator=", ", threshold=SMALL, max_line_width=10_000)
    except Exception:
        return ""


def _short(v: Any, n: int = 60) -> str:
    if callable(v) and hasattr(v, "__qualname__"):  # <function gelu>, not its address
        return f"<{'class' if isinstance(v, type) else 'function'} {v.__qualname__}>"
    try:
        text = repr(v)
    except Exception:
        text = f"<{type(v).__name__}>"
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _declared(owner: Any) -> dict[str, str]:
    """Each field's annotation as text (``f32[D, D]``), for a struct class."""
    try:
        from ..annotations.spec import StructSpec, parse_owner

        spec = parse_owner(owner)
        return {k: str(s) for k, s in spec.fields.items()} if isinstance(spec, StructSpec) else {}
    except Exception:
        return {}


def _sizes(obj: Any) -> dict[str, int]:
    """The sizes the struct's arrays fix (``D=16``), named by its class."""
    try:
        from ..annotations.dims import Env
        from ..annotations.spec import parse_owner

        env = Env()
        parse_owner(type(obj)).match(obj, type(obj).__name__, env)
        own = [p.__name__ for p in getattr(type(obj), "__type_params__", ())]
        return {k: int(env.dims[k]) for k in own if isinstance(env.dims.get(k), int)}
    except Exception:
        return {}


def _struct_type(obj: Any) -> str:
    sizes = _sizes(obj)
    return type(obj).__name__ + ("[" + ", ".join(f"{k}={v}" for k, v in sizes.items()) + "]" if sizes else "")


def _node(name: str, value: Any, declared: str = "") -> _Node:
    from ..annotations.dims import describe

    v: Any = value  # (not narrowed by the checks below)

    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        anns = _declared(type(v))
        kids = [_node(f.name, getattr(v, f.name, None), anns.get(f.name, "")) for f in dataclasses.fields(v)]
        n = _Node(name, "struct", _struct_type(v) if getattr(v, "__expex_struct__", False) else type(v).__name__, children=kids)
    elif isinstance(v, tuple) and hasattr(v, "_fields"):  # a NamedTuple
        n = _Node(name, "struct", type(v).__name__, children=[_node(k, getattr(v, k)) for k in v._fields])
    elif isinstance(v, (list, tuple)) and v and any(_array(x) or dataclasses.is_dataclass(x) for x in v):
        elem = declared[5:-1] if declared.startswith("list[") else ""
        idx = list(range(len(v)))
        shown = idx if len(v) <= LONG else [*idx[:3], -1, idx[-1]]
        kids = [_Node(f"⋯ {len(v) - 4} more", "more", "") if i < 0 else _node(f"[{i}]", v[i], elem) for i in shown]
        n = _Node(name, "list", f"{type(v).__name__}[{len(v)}]", children=kids)
        n.params = sum(_count(x)[0] for x in v)
        n.nbytes = sum(_count(x)[1] for x in v)
        return _decl(n, declared)
    elif isinstance(v, dict) and any(_array(x) or dataclasses.is_dataclass(x) for x in v.values()):
        n = _Node(name, "dict", f"dict[{len(v)}]", children=[_node(f"[{k!r}]", x) for k, x in v.items()])
    elif _array(v):
        key = _is_key(v)
        size = math.prod(v.shape)
        try:
            label = describe(v)
        except Exception:
            label = f"{v.dtype}{list(v.shape)}"
        n = _Node(name, "array", label, params=0 if key else size, nbytes=0 if key else size * np.dtype(v.dtype).itemsize,
                  value=_values(v))  # fmt: skip
        return _decl(n, declared)
    else:
        return _Node(name, "static", _short(v))
    n.params = sum(k.params for k in n.children if k.kind != "more")
    n.nbytes = sum(k.nbytes for k in n.children if k.kind != "more")
    return _decl(n, declared)


def _count(v: Any) -> tuple[int, int]:
    """(parameters, bytes) of every array in ``v``, keys left out."""
    import jax

    leaves = jax.tree.leaves(v, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
    arrays = [x for x in leaves if _array(x) and not _is_key(x)]
    return sum(math.prod(x.shape) for x in arrays), sum(math.prod(x.shape) * np.dtype(x.dtype).itemsize for x in arrays)


def _decl(n: _Node, declared: str) -> _Node:
    n.declared = declared if declared and declared != n.type and not declared.startswith(("list[", "tuple[")) else ""
    return n


def _flat(root: _Node) -> list[tuple[int, int, _Node]]:
    """(depth, parent index, node) in display order; the root is not included."""
    out: list[tuple[int, int, _Node]] = []

    def walk(n: _Node, depth: int, parent: int) -> None:
        for k in n.children:
            out.append((depth, parent, k))
            walk(k, depth + 1, len(out) - 1)

    walk(root, 0, -1)
    return out


def struct_text(obj: Any) -> str:
    """A struct as an indented outline: one line per field, arrays by type."""
    root = _node("", obj)
    rows = [("  " * d + n.name, n.type, n.declared, f"{n.params:,}" if n.params and n.kind != "static" else "", n.value)
            for d, _, n in _flat(root)]  # fmt: skip
    w = [max([len(r[i]) for r in rows] + [0]) for i in range(4)]
    lines = [f"{root.type}   {root.params:,} parameters · {_bytes(root.nbytes)}"]
    for r in rows:
        cells = [r[0].ljust(w[0]), r[1].ljust(w[1]), r[2].ljust(w[2]), r[3].rjust(w[3]), r[4]]
        lines.append("  " + "  ".join(cells).rstrip())
    return "\n".join(lines)


def _bytes(n: int) -> str:
    from .cost import si

    return si(n, "B")


def struct_html(obj: Any) -> str:
    """A struct as a collapsible tree table (see the module docstring)."""
    root = _node("", obj)
    flat = _flat(root)
    shut: set[int] = set()
    for i, (_, _, n) in enumerate(flat):
        if n.kind == "list" and len(n.children) > 3:
            shut.add(i)
    hidden = set()
    for i, (_, p, _) in enumerate(flat):
        if p in shut or p in hidden:
            hidden.add(i)
    body = []
    for i, (d, p, n) in enumerate(flat):
        kids = bool(n.children)
        tog = pages.toggle(kids)
        if i in shut:
            tog = tog.replace('aria-expanded="true"', 'aria-expanded="false"')
        cls = " ".join(c for c in ("shut" if i in shut else "", "more" if n.kind == "more" else "") if c)
        body.append(
            f'<tr data-i="{i}" data-p="{"" if p < 0 else p}" data-params="{n.params}" data-bytes="{n.nbytes}"'
            + (f' class="{cls}"' if cls else "") + (" hidden" if i in hidden else "") + ">"
            f'<td class="m" style="--d:{d}">{tog}{html.escape(n.name)}</td>'
            f'<td class="c k-{n.kind}">{html.escape(n.type)}</td><td class="f">{html.escape(n.declared)}</td>'
            f'<td class="n">{f"{n.params:,}" if n.params else ""}</td><td class="n">{_bytes(n.nbytes) if n.nbytes else ""}</td>'
            f'<td class="c v">{html.escape(n.value)}</td></tr>'
        )  # fmt: skip
    head = (f'<div class="hd"><b>{html.escape(root.type)}</b><span>{root.params:,} parameters</span>'
            f'<span>{_bytes(root.nbytes)}</span></div>')  # fmt: skip
    table = (f'{head}<div class="sc"><table><thead><tr><th>field</th><th>type</th><th class="f">declared</th>'
             '<th data-k="params" class="n">params</th><th data-k="bytes" class="n">bytes</th><th>value</th></tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table></div>')  # fmt: skip
    return pages.table(pages.uid("txt"), _STRUCT_CSS, table)


_STRUCT_CSS = """#ID .k-struct{font-weight:600}#ID .k-static{color:var(--mu)}#ID .v{color:var(--mu);max-width:40ch;overflow:hidden;text-overflow:ellipsis}
#ID tr.more td{color:var(--mu);font-style:italic}@container (max-width:560px){#ID .f,#ID .v{display:none}}"""


# ------------------------------------------------------------------ shape errors


def error_text(e: BaseException) -> str:
    notes = [n for n in getattr(e, "__notes__", ()) if "e.diagram" not in n]
    return "\n".join([f"{type(e).__name__}: {e}", *notes])


def error_html(e: BaseException) -> str:
    """The error's message and its diagram, which is traced now (once), not when the error was raised."""
    from .._diagram import Diagram

    uid = pages.uid("txe")
    notes = [n for n in getattr(e, "__notes__", ()) if "e.diagram" not in n]
    figure = ""
    d = getattr(e, "diagram", None)
    if isinstance(d, Diagram):
        try:
            figure = (f'<div class="dg">{d.svg}</div><details><summary>the trace, as text</summary>'
                      f"<pre>{html.escape(d.text)}</pre></details>")  # fmt: skip
        except Exception as x:  # the error is the point: a diagram that fails to draw is a footnote
            figure = f'<p class="mu">(no diagram: {html.escape(type(x).__name__)}: {html.escape(str(x))})</p>'
    css = (pages.tokens("#ID", _ERR_LIGHT, _ERR_DARK) + _ERR_CSS).replace("#ID", "#" + uid)
    return (f'<div class="txe" id="{uid}"><style>{css}</style><div class="hd"><b>{html.escape(type(e).__name__)}</b></div>'
            f'<pre class="msg">{html.escape(str(e))}</pre>'
            + "".join(f'<p class="mu">{html.escape(n)}</p>' for n in notes) + figure + "</div>")  # fmt: skip


_ERR_LIGHT, _ERR_DARK = "--bad:#b42318", "--bad:#ff7b72"
_ERR_CSS = f"""#ID{{color:var(--fg);background:var(--bg);font:13px/1.45 {pages.SANS};border:1px solid var(--ln);border-left:4px solid var(--bad);border-radius:8px;padding:10px 12px;max-width:100%}}
#ID .hd b{{color:var(--bad)}}#ID pre{{font:12px/1.45 {pages.MONO};white-space:pre-wrap;margin:6px 0;overflow-wrap:anywhere}}
#ID .mu{{color:var(--mu);margin:4px 0;font-size:12px}}#ID .dg{{overflow:auto;max-height:520px;margin-top:8px;border:1px solid var(--ln);border-radius:6px}}
#ID .dg svg{{display:block;max-width:none}}#ID summary{{cursor:pointer;color:var(--mu);font-size:12px;margin-top:6px}}"""


# ------------------------------------------------------------------ proofs


_MARK = {"proved": ("ok", "ok"), "spot-checked": ("spot", "spot"), "unchecked": ("skip", "skip"),
         "error": ("error", "bad"), "failed": ("fail", "bad")}  # fmt: skip


def reports_html(reports: list[Any]) -> str:
    """``expex.prove`` reports: one row per function, with what failed in a row that opens."""
    uid = pages.uid("txr")
    counts: dict[str, int] = {}
    rows = []
    for r in reports:
        counts[r.status] = counts.get(r.status, 0) + 1
        word, cls = _MARK.get(r.status, (r.status, "skip"))
        body = "\n".join(x for x in [r.detail, *r.notes] if x).strip()
        what = r.status + (f" for all sizes ({r.cases} cases)" if r.status == "proved" and r.cases > 1 else
                           " for all sizes" if r.status == "proved" else "")  # fmt: skip
        name = f"<code>{html.escape(r.name)}</code>"
        if body:
            first = html.escape(body.splitlines()[0])
            cell = (f"<details{' open' if cls == 'bad' and len(reports) == 1 else ''}><summary>{name} <span class=\"mu\">{first}</span>"
                    f"</summary><pre>{html.escape(body)}</pre></details>")  # fmt: skip
        else:
            cell = f'{name} <span class="mu">{html.escape(what)}</span>'
        rows.append(f'<tr class="{cls}"><td><span class="b {cls}">{word}</span></td><td>{cell}</td></tr>')
    total = f"{len(reports)} function{'s' if len(reports) != 1 else ''}: " + ", ".join(f"{n} {s}" for s, n in counts.items())
    css = (pages.tokens("#ID", _ERR_LIGHT + ";--good:#1a7f37", _ERR_DARK + ";--good:#3fb950") + _REP_CSS).replace("#ID", "#" + uid)
    return (f'<div class="txr" id="{uid}"><style>{css}</style><div class="hd"><b>expex.prove</b> <span class="mu">{html.escape(total)}</span></div>'
            f'<table><tbody>{"".join(rows)}</tbody></table></div>')  # fmt: skip


_REP_CSS = f"""#ID{{color:var(--fg);background:var(--bg);font:13px/1.45 {pages.SANS};border:1px solid var(--ln);border-radius:8px;padding:10px 12px;max-width:100%}}
#ID .hd{{margin-bottom:6px}}#ID .mu{{color:var(--mu)}}#ID table{{border-collapse:collapse;width:100%}}#ID td{{padding:3px 8px;border-bottom:1px solid var(--ln);vertical-align:top}}
#ID td:first-child{{width:1%}}#ID code{{font:12px {pages.MONO}}}#ID .b{{display:inline-block;min-width:3.4em;text-align:center;border-radius:9px;padding:0 6px;font-size:11px;font-weight:600;border:1px solid}}
#ID .b.ok{{color:var(--good)}}#ID .b.spot{{color:var(--ac)}}#ID .b.skip{{color:var(--mu)}}#ID .b.bad{{color:var(--bad)}}
#ID summary{{cursor:pointer}}#ID pre{{font:12px/1.45 {pages.MONO};white-space:pre-wrap;margin:6px 0 2px;color:var(--fg);overflow-wrap:anywhere}}"""


# ------------------------------------------------------------------ cost estimates


def cost_html(c: Any, at: Any = None, values: dict[str, int] | None = None) -> str:
    """An ``expex.cost`` estimate: FLOPs, parameters and the largest value, as formulas and numbers.

    ``at`` is the same estimate at given sizes (``values``), for the numbers next to ``c``'s formulas.
    """
    from .cost import si

    uid = pages.uid("txc")
    values = values or {}
    sizes = [*(f"{n}={values[n]}" if n in values else n for n in c.sizes), *(f"{k}={v}" for k, v in c.given.items()),
             *(f"{k} (= {v})" for k, v in c.fixed.items())]  # fmt: skip
    num = lambda p, q: (q if at is not None and q is not None else p).value  # noqa: E731

    def row(what: str, p: Any, q: Any, unit: str) -> str:
        v = num(p, q)
        return (f'<tr><th>{what}</th><td class="f">{html.escape(p.format(c.sizes))}</td>'
                f'<td class="n">{"" if v is None else html.escape(si(v, unit))}</td></tr>')  # fmt: skip

    rows = [row("FLOPs", c.flops, getattr(at, "flops", None), "FLOP")]
    if c.parameters is not None:
        rows.append(row("parameters", c.parameters, getattr(at, "parameters", None), ""))
    if c.largest is not None:
        t, n = c.largest
        v = num(n, at.largest[1] if at is not None and at.largest is not None else None)
        rows.append(f'<tr><th>largest value</th><td class="f">{html.escape(t)} <span class="mu">({html.escape(n.format(c.sizes))} elements)</span></td>'
                    f'<td class="n">{"" if v is None else html.escape(si(v))}</td></tr>')  # fmt: skip
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in dict.fromkeys(c.notes))
    css = (pages.tokens("#ID") + _COST_CSS).replace("#ID", "#" + uid)
    chips = "".join(f"<span>{html.escape(s)}</span>" for s in sizes)
    return (f'<div class="txc" id="{uid}"><style>{css}</style><div class="hd"><b>{html.escape(c.name)}</b>{chips}</div>'
            f'<table><tbody>{"".join(rows)}</tbody></table><ul>{notes}</ul></div>')  # fmt: skip


_COST_CSS = f"""#ID{{color:var(--fg);background:var(--bg);font:13px/1.45 {pages.SANS};border:1px solid var(--ln);border-radius:8px;padding:10px 12px;max-width:100%}}
#ID .hd{{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline;margin-bottom:6px}}#ID .hd span{{color:var(--mu);font:12px {pages.MONO}}}
#ID table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}#ID th,#ID td{{padding:3px 10px;border-bottom:1px solid var(--ln);text-align:left;vertical-align:baseline}}
#ID th{{color:var(--mu);font-weight:600;font-size:12px;white-space:nowrap;width:1%}}#ID .f{{font:12px {pages.MONO};overflow-wrap:anywhere}}#ID .n{{text-align:right;white-space:nowrap;font-weight:600}}
#ID .mu{{color:var(--mu)}}#ID ul{{margin:6px 0 0;padding-left:18px;color:var(--mu);font-size:12px}}#ID ul:empty{{display:none}}"""

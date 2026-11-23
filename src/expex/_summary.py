"""``expex.summary``: a model's modules in a table, with parameters and FLOPs as formulas.

::

    >>> print(expex.summary(model, T=128))      # an instance: sizes from its arrays
    >>> expex.summary(Transformer, Vocab=50, D=32, F=64, blocks=4)   # nothing allocated

One row per module, by path; a list of identical layers is one row
(``blocks[*]  × 4``) whose children are shown once. Array fields are leaf
rows. Rows under a list count one element. Parameter formulas come from the
field annotations, in the names of the root class; FLOPs are those of one call
of a module's ``__call__``, from :func:`expex.cost`. Sizes the call needs but
the model does not fix (a sequence length ``T``) are keywords; without them
only the formula is shown. ``str()`` is a text table, ``_repr_html_()`` an
interactive one (collapsible, sortable, formulas on hover), and
``save(path)`` writes it as a standalone page.
"""

import html
import itertools
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from ._cost import Poly, cost
from ._dims import DConst, DExpr, DName, Env
from ._fn import Fn, unwrap
from ._init import _all_fields
from ._spec import ArraySpec, SeqSpec, StructSpec, UnionSpec, parse_owner


@dataclass
class Row:
    """One line of the table. Numbers count one element of the enclosing lists."""

    path: str
    depth: int
    label: str  # `Attention[D=32]`, or an array's annotation `f32[Vocab, D]`
    count: int  # list length (1 outside a list row itself)
    params: int
    formula: Poly | None
    nbytes: int
    flops: Poly | None = None  # one call of __call__, in the root's names
    flops_value: int | None = None
    parent: int = -1
    weight: int = 1  # how many copies of this row the model holds


@dataclass
class Summary:
    """The rows of :func:`summary`, the total and the sizes. See the module docstring."""

    name: str
    rows: list[Row]
    total: Row
    sizes: dict[str, int]
    order: list[str]
    notes: list[str] = field(default_factory=list)

    def _cells(self, r: Row) -> list[str]:
        fmt = lambda p: "" if p is None else p.format(self.order)
        flops = "" if r.flops_value is None else _si(r.flops_value)
        return [r.label, f"× {r.count}" if r.count > 1 else "", f"{r.params:,}", fmt(r.formula),
                _bytes(r.nbytes), flops, fmt(r.flops)]  # fmt: skip

    def __str__(self) -> str:
        head = ["module", "class", "", "params", "formula", "bytes", "flops", "flops formula"]
        body = [["  " * r.depth + r.path.rsplit(".", 1)[-1], *self._cells(r)] for r in self.rows]
        foot = ["total", *self._cells(self.total)]
        w = [max(len(x[i]) for x in [head, *body, foot]) for i in range(len(head))]
        right = {2, 3, 5, 6}
        line = lambda x: "  ".join(c.rjust(w[i]) if i in right else c.ljust(w[i]) for i, c in enumerate(x)).rstrip()  # fmt: skip
        rule = "─" * len(line(head))
        sizes = ", ".join(f"{k}={v}" for k, v in self.sizes.items())
        out = [f"{self.name}   ({sizes})", line(head), rule, *map(line, body), rule, line(foot)]
        return "\n".join(out + [f"note: {n}" for n in self.notes])

    __repr__ = __str__

    def _repr_html_(self) -> str:
        uid = f"txs{next(_ids)}"
        total = max(self.total.params, 1)

        def tr(i: Any, r: Row) -> str:
            c = self._cells(r)
            share = r.params * r.weight / total if i != "" else 0
            kids = any(x.parent == i for x in self.rows)
            name = html.escape(r.path.rsplit(".", 1)[-1] if i != "" else "total")
            tog = '<button class="tg" aria-expanded="true" aria-label="collapse">▾</button>' if kids else '<span class="tg"></span>'  # fmt: skip
            tip = lambda s: f' title="{html.escape(s)}"' if s else ""
            ptip = f"{c[3]}" + (f"\n{r.weight} × {r.params:,} = {r.params * r.weight:,}" if r.weight > 1 else "") + (f"\n{share:.1%} of parameters" if share else "")  # fmt: skip
            return (
                f'<tr data-i="{i}" data-p="{"" if r.parent < 0 else r.parent}" data-params="{r.params * r.weight}" '
                f'data-flops="{r.flops_value if r.flops_value is not None else -1}"><td class="m" '
                f'style="--d:{r.depth}" title="{html.escape(r.path)}">{tog}{name}</td><td class="c">{html.escape(c[0])}'
                f'<span class="x">{html.escape(c[1])}</span></td><td class="n bar" style="--s:{share:.4f}"{tip(ptip)}>'
                f'{c[2]}</td><td class="f">{html.escape(c[3])}</td><td class="n b">{c[4]}</td>'
                f'<td class="n"{tip(c[6])}>{c[5] or ("—" if not c[6] else "ƒ")}</td><td class="f">{html.escape(c[6])}</td></tr>'
            )  # fmt: skip

        sizes = " ".join(f"<span>{html.escape(k)}={v}</span>" for k, v in self.sizes.items())
        body = "".join(tr(i, r) for i, r in enumerate(self.rows))
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in self.notes)
        return (
            f'<div class="txs" id="{uid}"><style>{_CSS.replace("#ID", "#" + uid)}</style>'
            f'<div class="hd"><b>{html.escape(self.name)}</b> {sizes}</div><div class="sc"><table><thead><tr>'
            '<th>module</th><th>class</th><th data-k="params" class="n" title="parameters (click to sort)">params</th>'
            '<th class="f">formula</th><th class="n b">bytes</th><th data-k="flops" class="n" title="FLOPs of one call of __call__ (click to sort)">'
            'FLOPs</th><th class="f">formula</th></tr></thead>'
            f'<tbody>{body}</tbody><tfoot>{tr("", self.total)}</tfoot></table></div>'
            f'<ul>{notes}</ul><script>{_JS.replace("ID", json.dumps(uid))}</script></div>'
        )

    def save(self, path: str | pathlib.Path) -> None:
        """Write the HTML table as a standalone page."""
        page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
                f"content='width=device-width, initial-scale=1'><title>{html.escape(self.name)} summary"
                f"</title><style>body{{margin:16px;background:Canvas;color:CanvasText;color-scheme:light dark}}"
                f"</style></head><body>{self._repr_html_()}</body></html>")  # fmt: skip
        pathlib.Path(path).write_text(page, encoding="utf-8")


def summary(model: Any, /, **sizes: int) -> Summary:
    """A table of a model's modules: parameters, bytes and FLOPs, as numbers and formulas.

    ``model`` is an ``@expex.struct`` instance, or a struct class with the sizes
    ``init`` needs (``summary(Transformer, Vocab=50, D=32, F=64, blocks=4)``,
    built under ``jax.eval_shape``, so nothing is allocated). Other keywords
    (``T=128``) are sizes of the ``__call__`` methods, used for FLOP counts.
    """
    if isinstance(model, type):
        spec = parse_owner(model)
        accepts = {p.__name__ for p in getattr(model, "__type_params__", ())} | _all_fields(spec)
        names: set[str] = set()
        spec.names(names)
        accepts |= {n for n in names if n.isidentifier()}
        built = {k: v for k, v in sizes.items() if k in accepts}
        sizes = {k: v for k, v in sizes.items() if k not in accepts}
        model = jax.eval_shape(lambda: model.init(jr.key(0), **built))
    cls = type(model)
    spec = parse_owner(cls)
    if not isinstance(spec, StructSpec):
        raise TypeError(f"expex.summary: {cls.__name__} is not a @expex.struct with array fields")
    env = Env()
    spec.match(model, cls.__name__, env)
    known = {k: int(v) for k, v in env.dims.items() if isinstance(v, int) and k.isidentifier()}
    rows: list[Row] = []
    lists: dict[str, int] = {}
    s = _Walk(rows, lists, known, sizes, [])
    s.fields(spec, model, "", 0, -1, 1)
    if lists:
        known["L"] = max(lists.values())
        s.notes.append(f"L = len({', '.join(lists)})")
    params = sum(r.params for r in rows if r.parent < 0 for _ in range(r.count))  # top-level rows
    formula = _sum([None if r.formula is None else r.formula * (Poly.of("L") if r.count > 1 else 1) for r in rows if r.parent < 0])  # fmt: skip
    nbytes = sum(r.nbytes * r.count for r in rows if r.parent < 0)
    total = Row("total", 0, cls.__name__, 1, params, formula, nbytes)
    s.flops(total, cls, model, {})
    if s.fallback:
        s.notes.append(f"FLOP formulas of {', '.join(s.fallback)} are in the call's sizes only: static fields differ from the class defaults")  # fmt: skip
    used = set().union(*(r.flops.names() for r in [*rows, total] if r.flops is not None))
    if unknown := set(sizes) - used:
        raise TypeError(f"expex.summary: unknown size {', '.join(sorted(unknown))} (the calls use {', '.join(sorted(used)) or 'none'})")  # fmt: skip
    order = [p.__name__ for p in getattr(cls, "__type_params__", ())]
    calls = sorted((used | set(sizes)) - set(known) - {"L"} - set(order))  # T first, as expex.cost does
    order = list(dict.fromkeys([*calls, *order, *sorted(known.keys() - {"L"}), "L"]))
    shown = {k: v for k, v in sorted({**known, **sizes}.items(), key=lambda kv: order.index(kv[0]))}
    return Summary(cls.__name__, rows, total, shown, order, s.notes)


@dataclass
class _Walk:
    rows: list[Row]
    lists: dict[str, int]
    known: dict[str, int]
    given: dict[str, int]
    notes: list[str]
    fallback: list[str] = field(default_factory=list)  # rows whose class formula does not apply
    _own: dict[type, Poly | None] = field(default_factory=dict)

    def fields(self, spec: StructSpec, obj: Any, path: str, depth: int, parent: int, weight: int) -> None:
        for name, s in spec.fields.items():
            value, here = getattr(obj, name), f"{path}.{name}" if path else name
            if (s := _pick(s, value)) is None:
                continue
            n = 1
            if isinstance(s, SeqSpec) and isinstance(value, (list, tuple)):
                if not value:
                    continue
                s, value, n, here = s.elem, value[0], len(value), here + "[*]"
                self.lists[here.removesuffix("[*]")] = n
            arrays = _arrays(value)
            row = Row(here, depth, str(s), n, sum(a.size for a in arrays), _formula(s, value),
                      sum(a.size * a.dtype.itemsize for a in arrays), parent=parent, weight=weight * n)  # fmt: skip
            self.rows.append(row)
            if isinstance(s, StructSpec):
                mapping: dict[str, Poly] = {}
                _pair(parse_owner(s.cls), s, mapping)
                env = Env()
                parse_owner(s.cls).match(value, here, env)
                own = [p.__name__ for p in getattr(s.cls, "__type_params__", ())]
                row.label = s.cls.__name__ + ("[" + ", ".join(f"{k}={env.dims.get(k, '?')}" for k in own) + "]" if own else "")  # fmt: skip
                self.flops(row, s.cls, value, mapping)
                self.fields(s, value, here, depth + 1, len(self.rows) - 1, weight * n)

    def flops(self, row: Row, cls: type, obj: Any, mapping: dict[str, Poly]) -> None:
        """FLOPs of one call of ``obj.__call__``: the class's formula, and a number when sizes allow."""
        if not isinstance(unwrap(getattr(cls, "__call__", None)), Fn):
            return
        if cls not in self._own:
            try:
                self._own[cls] = cost(cls.__call__).flops
            except Exception as e:
                self._own[cls] = None
                self.notes.append(f"{cls.__name__}.__call__: no cost estimate ({type(e).__name__}: {e})")
        try:
            bound = cost(obj.__call__).flops  # only the call's own sizes are left
        except Exception:
            return
        if (own := self._own[cls]) is not None:
            own = own.subs(mapping)
            if own.subs({**self.known, **self.given}) != bound.subs(self.given):
                self.fallback.append(row.path)
                own = None
        row.flops = own if own is not None else bound
        v = bound.subs(self.given).value
        row.flops_value = None if v is None else int(v)


def _pick(spec: Any, value: Any) -> Any:
    """The spec that describes ``value`` (the matching option of ``X | None``), or None to skip it."""
    if isinstance(spec, UnionSpec):
        if value is None:
            return None
        structs: list[Any] = [o for o in spec.options if isinstance(o, StructSpec) and isinstance(value, o.cls)]
        spec = (structs or [o for o in spec.options if o.leafy])[0]
    return spec if spec.leafy else None


def _arrays(x: Any) -> list[Any]:
    leaves = jax.tree.leaves(x, is_leaf=lambda v: isinstance(v, jax.ShapeDtypeStruct))
    return [a for a in leaves if hasattr(a, "dtype") and hasattr(a, "size") and not jnp.issubdtype(a.dtype, jax.dtypes.prng_key)]  # fmt: skip


def _dim(d: Any) -> Poly:
    if isinstance(d, DName):
        return Poly.of(d.name)
    if isinstance(d, DConst):
        return Poly.of(d.value)
    if isinstance(d, DExpr):
        try:
            return Poly.of(d.expr({n: Poly.of(n) for n in d.expr.names}))
        except Exception:  # floordiv, mod, max: an opaque name
            pass
    return Poly.of(str(d))


def _formula(spec: Any, value: Any) -> Poly | None:
    """Parameters of ``value`` as a polynomial, from its annotation; ``L`` for lists."""
    if isinstance(spec, ArraySpec):
        return math.prod(map(_dim, spec.dims), start=Poly.of(1))
    if isinstance(spec, StructSpec):
        parts = []
        for name, s in spec.fields.items():
            if (s := _pick(s, v := getattr(value, name))) is not None:
                parts.append(_formula(s, v))
        return _sum(parts)
    if isinstance(spec, SeqSpec) and isinstance(value, (list, tuple)):
        inner = _formula(spec.elem, value[0]) if value else Poly()
        return None if inner is None else inner * Poly.of("L")
    return None  # a tuple, a dict, a key: the number only


def _sum(parts: list[Poly | None]) -> Poly | None:
    return None if any(p is None for p in parts) else sum((p for p in parts if p is not None), Poly())


def _pair(own: Any, used: Any, out: dict[str, Poly]) -> None:
    """Map a class's own size names to the sizes of one use of it (``Linear[I, O]`` as ``Linear[D, F]``)."""
    if isinstance(own, ArraySpec) and isinstance(used, ArraySpec) and len(own.dims) == len(used.dims):
        for a, b in zip(own.dims, used.dims):
            if isinstance(a, DName):
                out.setdefault(a.name, _dim(b))
    elif isinstance(own, StructSpec) and isinstance(used, StructSpec):
        for k, s in own.fields.items():
            if k in used.fields:
                _pair(s, used.fields[k], out)
    elif isinstance(own, (SeqSpec, UnionSpec)) and type(used) is type(own):
        for a, b in zip(getattr(own, "options", [getattr(own, "elem", None)]), getattr(used, "options", [getattr(used, "elem", None)])):  # fmt: skip
            _pair(a, b, out)


def _si(v: float) -> str:
    for s, p in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= s:
            return f"{v / s:.3g} {p}"
    return f"{v:,}"


def _bytes(v: int) -> str:
    return _si(v) + "B" if v >= 1000 else f"{v} B"


_ids = itertools.count()

_CSS = """#ID{--fg:#1f2328;--mu:#656d76;--ln:#d0d7de;--bg:#fff;--hd:#f6f8fa;--ac:#2f6fdb;--br:rgba(47,111,219,.16);
color:var(--fg);background:var(--bg);font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;max-width:100%;container-type:inline-size;border:1px solid var(--ln);border-radius:8px;padding:10px 12px}
@media (prefers-color-scheme:dark){#ID{--fg:#e6edf3;--mu:#8d96a0;--ln:#30363d;--bg:#0d1117;--hd:#161b22;--ac:#58a6ff;--br:rgba(88,166,255,.22)}}
#ID .hd{margin-bottom:8px;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}#ID .hd span{color:var(--mu);font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
#ID .sc{overflow-x:auto}#ID table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
#ID th,#ID td{padding:3px 10px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap;vertical-align:baseline}
#ID th{background:var(--hd);font-weight:600;color:var(--mu);font-size:12px;position:sticky;top:0}#ID th[data-k]{cursor:pointer;color:var(--ac)}
#ID th[data-k]::after{content:" ↕";opacity:.5}#ID th[data-dir="1"]::after{content:" ↓";opacity:1}#ID th[data-dir="-1"]::after{content:" ↑";opacity:1}
#ID .n{text-align:right}#ID .m{padding-left:calc(6px + var(--d)*16px);font-weight:500}#ID .c,#ID .f{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
#ID .f{color:var(--mu);white-space:normal;min-width:12ch}#ID .x{color:var(--ac);margin-left:6px;font-weight:600}
#ID .bar{background:linear-gradient(to left,var(--br) calc(var(--s)*100%),transparent 0)}#ID tbody tr:hover{background:var(--hd)}
#ID tfoot td{font-weight:600;border-top:2px solid var(--ln);border-bottom:0}#ID .tg{display:inline-block;width:16px;margin-right:2px;border:0;background:none;color:var(--mu);cursor:pointer;padding:0;font:inherit}
#ID tr.shut .tg{transform:rotate(-90deg)}#ID ul{margin:6px 0 0;padding-left:18px;color:var(--mu);font-size:12px}#ID ul:empty{display:none}
@container (max-width:720px){#ID .f{display:none}#ID th,#ID td{padding:3px 6px}}@container (max-width:480px){#ID .b{display:none}#ID .c{white-space:normal}#ID .m{padding-left:calc(2px + var(--d)*10px)}}"""

_JS = """(()=>{const root=document.getElementById(ID),body=root.querySelector("tbody"),rows=[...body.rows],by={},kids={};
rows.forEach(r=>{by[r.dataset.i]=r;(kids[r.dataset.p]??=[]).push(r)});let key=null,dir=1;
const vis=()=>rows.forEach(r=>{let p=r.dataset.p,h=false;while(p!==""){if(by[p].classList.contains("shut")){h=true;break}p=by[p].dataset.p}r.hidden=h});
const order=()=>{const out=[],walk=p=>{const c=[...(kids[p]||[])];if(key)c.sort((a,b)=>dir*(b.dataset[key]-a.dataset[key]));c.forEach(r=>{out.push(r);walk(r.dataset.i)})};walk("");out.forEach(r=>body.appendChild(r))};
root.addEventListener("click",e=>{const b=e.target.closest("button.tg");if(b){const r=b.closest("tr"),shut=r.classList.toggle("shut");b.setAttribute("aria-expanded",String(!shut));vis();return}
const h=e.target.closest("th[data-k]");if(!h)return;if(key===h.dataset.k){if(dir===1)dir=-1;else key=null}else{key=h.dataset.k;dir=1}
root.querySelectorAll("th[data-k]").forEach(t=>t.dataset.dir=t.dataset.k===key?dir:"");order()})})();"""

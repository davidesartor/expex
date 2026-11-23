"""``expex.inspect``: one page with every tool's view of a model, over one module tree.

::

    page = expex.inspect(model, x)                    # example inputs: sizes, and activations
    page = expex.inspect(model, x, loss=loss_fn)      # activations and gradients of loss_fn(model, x)
    page = expex.inspect(Transformer, Vocab=50, D=32, F=64, blocks=4, T=24)   # no data
    page.save("model.html")                           # or display it in a notebook

The page has a tab per tool, each drawn by the tool's own rendering:

- **Summary**: :func:`expex.summary`'s table, one row per module.
- **Diagram**: :func:`expex.diagram`'s interactive page, in a frame (its page is
  gzipped into this one and unpacked by the browser).
- **Cost**: :func:`expex.cost` of each module's ``__call__``, with the sizes it is
  called with in the model's call; modules with the same estimate share a card.
- **Activations**: :func:`expex.debug.watch` of ``loss(model, *args)``, or of the
  forward call ``model(*args)`` alone (no gradients) when no loss is given.
- **Profile**: :func:`expex.debug.profile`'s FLOPs and bytes per source line.

Selection: modules are named by path (``blocks[*].attn``, a list's elements as
``[*]``). Clicking one in any tab selects it everywhere: its summary row is
highlighted (and its parents opened), the diagram opens the calls around it
and centres it, and the cost cards, activation rows and profile sections are
filtered to it and the modules inside it. The tools' fragments mark their rows
with ``data-path``; the diagram's frame takes ``{expex: "select", path}`` and
``{expex: "theme", theme}`` messages and sends ``select`` when a call is clicked.

A tool that fails leaves its tab with the error (a ``ShapeError`` with its
diagram); the others are unaffected. ``Inspect.errors`` lists them.
"""

import base64
import dataclasses
import gzip
import html
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from ..annotations.dims import Env
from ..annotations.errors import coded
from ..annotations.spec import join, parse_owner
from ..function import Fn, unwrap
from . import pages

TABS = (("summary", "Summary"), ("diagram", "Diagram"), ("cost", "Cost"), ("activations", "Activations"), ("profile", "Profile"))


class Inspect:
    """The parts of :func:`inspect`'s page; ``html()`` builds it, ``save(path)`` writes it.

    ``summary``, ``diagram``, ``watch`` and ``profile`` are the tools' own results
    (None when a tool failed or, for ``watch``, was not asked for); ``costs`` is a
    list of (module paths, ``Cost`` at their sizes, the class's symbolic ``Cost`` or None, the sizes);
    ``modules`` lists the module tree as dicts
    with ``path``, ``cls`` and ``label``; ``errors`` maps a tab to what went wrong.
    """

    def __init__(self, name: str, sizes: dict[str, int], modules: list[dict[str, str]], parts: dict[str, Any],
                 errors: dict[str, BaseException]):  # fmt: skip
        self.name, self.sizes, self.modules = name, sizes, modules
        self.summary = parts.get("summary")
        self.diagram = parts.get("diagram")
        self.costs: list[tuple[list[str], Any, Any, dict[str, int]]] = parts.get("costs", [])
        self.watch = parts.get("watch")
        self.profile = parts.get("profile")
        self.errors = errors
        self._html: str | None = None

    def __repr__(self) -> str:
        have = [t for t, _ in TABS if self._part(t) is not None]
        return f"<expex.inspect {self.name}: {', '.join(have)}{'; failed: ' + ', '.join(self.errors) if self.errors else ''}>"

    def _part(self, tab: str) -> Any:
        return {"summary": self.summary, "diagram": self.diagram, "cost": self.costs or None,
                "activations": self.watch, "profile": self.profile}[tab]  # fmt: skip

    def html(self) -> str:
        """The page: self-contained (inline CSS and JS, nothing from a CDN)."""
        if self._html is None:
            self._html = _page(self)
        return self._html

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write the page to ``path`` (``model.html``)."""
        path = pathlib.Path(path)
        path.write_text(self.html(), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        return f'<iframe srcdoc="{html.escape(self.html())}" style="width:100%;height:760px;border:0" title="expex.inspect {html.escape(self.name)}"></iframe>'  # fmt: skip


def inspect(model: Any, /, *args: Any, loss: Any = None, **sizes: int) -> Inspect:
    """Every tool's view of ``model`` in one page: summary, diagram, cost, activations and profile.

    ``model`` is an ``@expex.struct`` instance, or a struct class with the sizes
    building it needs (built under ``jax.eval_shape``: no arrays, no activations).
    ``args`` are example inputs of ``model(*args)``: they fix the call's sizes and
    give the activations. ``loss(model, *args)`` (a scalar, or ``(loss, aux)``)
    adds the gradients. Other keywords are sizes of the call (``T=128``), for the
    FLOP counts when no inputs are given. See the module docstring.
    """
    if isinstance(model, type):
        model, sizes = _build(model, sizes)
        if args or loss is not None:
            raise coded("EX705", TypeError("expex.inspect: a class has no arrays to run; pass an instance for activations"))
    cls = type(model)
    if not getattr(cls, "__expex_struct__", False):
        raise coded("EX705", TypeError(f"expex.inspect: {cls.__name__} is not an @expex.struct"))
    call = unwrap(getattr(cls, "__call__", None))
    sizes = {**_call_sizes(model, args), **sizes} if args else dict(sizes)
    parts: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}

    def attempt(tab: str, make: Any) -> None:
        try:
            parts[tab] = make()
        except Exception as e:  # the tab shows it (a ShapeError with its diagram); the others are drawn
            errors[tab] = e

    from .summary import _call_sizes as seen_sizes
    from .summary import summary

    attempt("summary", lambda: _with_sizes(lambda **s: summary(model, **s), sizes))
    modules = _modules(model)
    if isinstance(call, Fn):
        from .._diagram import diagram
        from .profile import profile

        attempt("diagram", lambda: diagram(model.__call__))
        seen = seen_sizes(model, _known(sizes, parts.get("summary")))
        known = dict(parts["summary"].sizes) if "summary" in parts else dict(sizes)
        attempt("costs", lambda: _costs(modules, seen, sizes, known))
        attempt("profile", lambda: _with_sizes(lambda **s: profile(model.__call__, **s), sizes))
    if args or loss is not None:
        attempt("activations", lambda: _watch(model, args, loss))
        if "activations" in parts:
            parts["watch"] = parts.pop("activations")
    if "costs" in errors:
        errors["cost"] = errors.pop("costs")
    shown = parts["summary"].sizes if "summary" in parts else sizes
    for m in modules:
        m.pop("obj")
    return Inspect(cls.__name__, dict(shown), modules, parts, errors)


# ------------------------------------------------------------------ the parts


def _build(cls: type, sizes: dict[str, int]) -> tuple[Any, dict[str, int]]:
    """A class built from its annotations under ``jax.eval_shape``, and the sizes left for its call."""
    from ..modules import _all_fields, build

    spec = parse_owner(cls)
    accepts = {p.__name__ for p in getattr(cls, "__type_params__", ())} | _all_fields(spec)
    names: set[str] = set()
    spec.names(names)
    accepts |= {n for n in names if n.isidentifier()}
    built = {k: v for k, v in sizes.items() if k in accepts}
    model = jax.eval_shape(lambda: build(cls, jr.key(0), **built))
    return model, {k: v for k, v in sizes.items() if k not in accepts}


def _call_sizes(model: Any, args: tuple[Any, ...]) -> dict[str, int]:
    """The sizes ``model(*args)`` binds that the model itself does not fix (``T``, a batch ``B``)."""
    f = unwrap(getattr(type(model), "__call__", None))
    if not isinstance(f, Fn):
        return {}
    try:
        if f._lazy:
            f._resolve_owner()
        bound = f._sig.bind(model, *args).arguments
        env, own = Env(), Env()
        for name, spec in f._check_inputs:
            if name in bound and name != "self":
                spec.match(bound[name], name, env)
        parse_owner(type(model)).match(model, type(model).__name__, own)
    except Exception:
        return {}
    return {k: int(v) for k, v in env.dims.items() if isinstance(v, int) and k.isidentifier() and k not in own.dims}


def _with_sizes(make: Any, sizes: dict[str, int]) -> Any:
    """``make(**sizes)``, or with fewer sizes when the tool says one is not its own (a batch ``B``)."""
    try:
        return make(**sizes)
    except TypeError as e:
        if not sizes or not (getattr(e, "code", None) == "EX706" or "unknown size" in str(e)):
            raise
        return make()


def _known(sizes: dict[str, int], summary: Any) -> dict[str, int]:
    return {k: v for k, v in sizes.items() if summary is None or k in summary.sizes}


def _modules(model: Any) -> list[dict[str, Any]]:
    """The model and every module in it, by path (a list's elements as ``[*]``, shown by its first)."""
    from .display import _struct_type

    out = [dict(path="", cls=type(model).__name__, label=_struct_type(model), obj=model)]

    def walk(obj: Any, path: str) -> None:
        for f in dataclasses.fields(obj):
            v, here = getattr(obj, f.name, None), join(path, f.name)
            if isinstance(v, (list, tuple)) and v and getattr(type(v[0]), "__expex_struct__", False):
                v, here = v[0], here + "[*]"
            if getattr(type(v), "__expex_struct__", False):
                out.append(dict(path=here, cls=type(v).__name__, label=_struct_type(v), obj=v))
                walk(v, here)

    walk(model, "")
    return out


def _costs(modules: list[dict[str, Any]], seen: dict[type, dict[str, int]], sizes: dict[str, int],
           known: dict[str, int]) -> list[tuple[list[str], Any, Any, dict[str, int]]]:  # fmt: skip
    """``expex.cost`` of each module's ``__call__`` at the sizes it is called with; equal estimates share a card.

    Each card is (paths, the bound method's cost, the class's symbolic one or None, the sizes).
    """
    from .cost import cost
    from .display import _sizes

    cards: dict[tuple[type, str], tuple[list[str], Any, Any, dict[str, int]]] = {}
    for m in modules:
        obj = m["obj"]
        if not isinstance(unwrap(getattr(type(obj), "__call__", None)), Fn):
            continue
        try:
            c = cost(obj.__call__)
            given = {**seen.get(type(obj), {}), **(sizes if m["path"] == "" else {})}
            if want := {k: v for k, v in given.items() if k in c.sizes}:
                c = cost(obj.__call__, **want)
        except Exception:
            continue  # the summary says which modules have no estimate
        values = {**known, **_sizes(obj), **c.given}
        try:  # the formula in the class's size names, if the instance's static fields do not change it
            sym = cost(type(obj).__call__)
            if sym.flops.subs(values).value != c.flops.value:
                sym = None
        except Exception:
            sym = None
        cards.setdefault((type(obj), str(c)), ([], c, sym, values))[0].append(m["path"])
    if not cards:
        raise ValueError("no module has a cost estimate")
    return list(cards.values())


def _watch(model: Any, args: tuple[Any, ...], loss: Any) -> Any:
    """``debug.watch`` of ``loss(model, *args)``, or of ``model(*args)`` alone (no gradients)."""
    from .watch import Watch, watch

    if loss is not None:
        return watch(loss, model, *args)

    def forward(m: Any, *a: Any) -> Any:  # the outputs times zero: a loss that leaves the statistics alone
        out = [x for x in jax.tree.leaves(m(*a)) if jnp.issubdtype(jnp.result_type(x), jnp.floating)]
        return sum((0.0 * x.astype(jnp.float32).sum() for x in out), jnp.zeros((), jnp.float32))

    forward.__qualname__ = f"{type(model).__name__}.__call__"
    w = watch(forward, model, *args)
    for r in w.rows:
        r.grad = None
        r.flags = [f for f in r.flags if not f.startswith("gradient")]
    return Watch(w.name, None, None, w.rows, w.options)


# ------------------------------------------------------------------ the page


def _panel(tab: str, page: Inspect) -> str:
    if tab in page.errors:
        e = page.errors[tab]
        if hasattr(e, "_repr_html_"):
            return e._repr_html_()
        return f'<p class="err"><b>{html.escape(type(e).__name__)}</b>: {html.escape(str(e))}</p>'
    part = page._part(tab)
    if tab == "activations" and part is None:
        return ('<p class="empty">Give example inputs, <code>expex.inspect(model, x)</code>, to see the activations of '
                'every call, or a loss, <code>expex.inspect(model, x, loss=loss_fn)</code>, for their gradients too.</p>')  # fmt: skip
    if part is None:
        return f'<p class="empty">{html.escape(page.name)} has no typed <code>__call__</code>: nothing to show here.</p>'
    if tab == "diagram":
        return ""  # filled from the data when the tab is first shown
    if tab == "cost":
        from .display import cost_html

        cards = "".join(f'<div class="card" data-paths="{html.escape(pages.data(paths))}"><div class="paths">'
                        f'{" · ".join(html.escape(p or page.name) for p in paths)}</div>'
                        f'{cost_html(sym, at=c, values=values) if sym is not None else c._repr_html_()}</div>'
                        for paths, c, sym, values in page.costs)  # fmt: skip
        return ('<p class="note">FLOPs of one call of each module\'s <code>__call__</code>, at the sizes the model\'s call '
                f"gives it; modules with the same estimate share a card.</p>{cards}")  # fmt: skip
    return part._repr_html_()


def _page(page: Inspect) -> str:
    data: dict[str, Any] = {"name": page.name, "modules": page.modules, "diagram": None}
    if page.diagram is not None:
        raw = gzip.compress(page.diagram.html().encode("utf-8"), compresslevel=9, mtime=0)
        data["diagram"] = base64.b64encode(raw).decode("ascii")
    sizes = "".join(f"<span>{html.escape(k)}={v}</span>" for k, v in page.sizes.items())
    tabs = "".join(f'<button role="tab" id="t-{t}" data-tab="{t}" aria-controls="tab-{t}" aria-selected="false"'
                   f'{" class=\"bad\"" if t in page.errors else ""}>{label}</button>' for t, label in TABS)  # fmt: skip
    panels = "".join(f'<section role="tabpanel" id="tab-{t}" aria-labelledby="t-{t}" hidden><div class="filter" hidden>'
                     f'Showing <code></code> and the modules in it · <button class="all">show all</button></div>'
                     f"{_panel(t, page)}</section>" for t, _ in TABS)  # fmt: skip
    body = (f'<header class="top"><h1>{html.escape(page.name)}</h1><span class="sizes">{sizes}</span>'
            f'<nav role="tablist" aria-label="views">{tabs}</nav><span class="selbar">module <code id="sel">none</code>'
            f'<span id="selcls"></span><button id="clear" title="Clear the selection (Esc)" hidden>✕</button></span>'
            f'<button id="theme" title="Colour scheme">Auto</button></header><main>{panels}</main>'
            f'<script type="application/json" id="expex-data">{pages.data(data)}</script><script>{_JS}</script>')  # fmt: skip
    return pages.document(f"{page.name} · expex.inspect", body, pages.tokens(":root") + _CSS)


_CSS = f"""body{{padding:0;background:var(--hd);color:var(--fg);font:14px/1.45 {pages.SANS}}}
header.top{{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--ln);display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;padding:6px 16px}}
header h1{{font-size:15px;margin:0}}.sizes span{{color:var(--mu);font:12px {pages.MONO};margin-right:8px}}
nav{{display:flex;flex-wrap:wrap}}nav button{{font:inherit;border:0;border-bottom:2px solid transparent;background:none;color:var(--mu);padding:6px 12px;cursor:pointer}}
nav button:hover{{color:var(--fg)}}nav button[aria-selected=true]{{color:var(--fg);border-bottom-color:var(--ac);font-weight:600}}nav button.bad::after{{content:" !";color:#d1242f}}
.selbar{{margin-left:auto;display:flex;gap:6px;align-items:baseline;color:var(--mu);font-size:13px}}.selbar code{{color:var(--ac);font:600 12px {pages.MONO}}}
#selcls{{font:12px {pages.MONO}}}header button#theme,.selbar button,.filter button{{font:inherit;font-size:12px;color:var(--fg);background:var(--hd);border:1px solid var(--ln);border-radius:6px;padding:2px 9px;cursor:pointer}}
main{{padding:14px 16px 40px;max-width:1500px;margin:0 auto}}section>.txs,section>.txw{{max-width:none}}
#tab-diagram iframe{{display:block;width:100%;height:calc(100vh - 92px);min-height:420px;border:1px solid var(--ln);border-radius:8px;background:var(--bg)}}
.filter{{background:var(--bg);border:1px solid var(--ln);border-left:3px solid var(--ac);border-radius:6px;padding:5px 10px;margin-bottom:10px;font-size:13px;color:var(--mu)}}.filter code{{color:var(--fg);font:600 12px {pages.MONO}}}
.note,.empty{{color:var(--mu);font-size:13px;margin:0 0 10px}}.empty{{background:var(--bg);border:1px dashed var(--ln);border-radius:8px;padding:16px}}.err{{color:#d1242f}}
.card{{margin-bottom:12px;cursor:pointer}}.card .paths{{font:12px {pages.MONO};color:var(--mu);margin:0 0 3px 2px}}.card.sel>div:last-child{{box-shadow:0 0 0 2px var(--ac)}}
#tab-summary tr[data-path]:not([data-path=""]),#tab-activations tr[data-path]{{cursor:pointer}}
main tr.sel>td{{background:var(--br)!important}}main tr.sel>td:first-child{{box-shadow:inset 3px 0 var(--ac)}}main tr.xo{{display:none}}
@media (max-width:700px){{header.top{{padding:6px 10px}}.selbar{{margin-left:0}}main{{padding:10px}}nav button{{padding:6px 8px}}}}"""

_JS = r"""(()=>{
const D=JSON.parse(document.getElementById("expex-data").textContent),$=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const norm=p=>p==null?null:String(p).replace(/\[\d+\]/g,"[*]");
const under=(p,s)=>p!==null&&(s===""||p===s||p.startsWith(s+".")||p.startsWith(s+"["));
const paths=new Set(D.modules.map(m=>m.path));
const module=p=>{p=norm(p);while(p!==null&&!paths.has(p)){const i=Math.max(p.lastIndexOf("."),p.lastIndexOf("["));p=i>0?p.slice(0,i):""}return p};
let sel=null,frame=null,ready=false,theme="auto";
const post=m=>{if(frame&&ready)frame.contentWindow.postMessage(m,"*")};

// tabs
const tabs=$$("[role=tab]");
function show(name){tabs.forEach(t=>t.setAttribute("aria-selected",String(t.dataset.tab===name)));
 $$("[role=tabpanel]").forEach(p=>p.hidden=p.id!=="tab-"+name);
 if(name==="diagram"){diagram();requestAnimationFrame(()=>post({expex:"select",path:sel===null?"":sel}))}
 try{history.replaceState(null,"","#"+name)}catch(e){}}
tabs.forEach(t=>t.onclick=()=>show(t.dataset.tab));
show(tabs.some(t=>location.hash==="#"+t.dataset.tab)?location.hash.slice(1):"summary");

// the diagram's page, gzipped: unpacked into its frame when the tab is first shown
async function diagram(){
 if(frame||!D.diagram)return;
 frame=document.createElement("iframe");frame.title="diagram";$("#tab-diagram").append(frame);
 frame.addEventListener("load",()=>{ready=true;post({expex:"theme",theme});if(sel!==null)post({expex:"select",path:sel})});
 try{const bytes=Uint8Array.from(atob(D.diagram),c=>c.charCodeAt(0));
  frame.srcdoc=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text()}
 catch(e){frame.srcdoc="<p style='font:14px sans-serif'>This browser cannot unpack the diagram ("+e+").</p>"}}
addEventListener("message",ev=>{if(!frame||ev.source!==frame.contentWindow||!ev.data)return;if(ev.data.expex==="select")select(module(ev.data.path),"diagram")});

// the watch rows' modules: a function's row belongs to the module of the call around it
const wrows=$$("#tab-activations tbody tr"),wby={};wrows.forEach(r=>wby[r.dataset.i]=r);
const wpath=r=>{for(let x=r;x;x=wby[x.dataset.p])if(x.dataset.path!==undefined)return norm(x.dataset.path);return null};
wrows.forEach(r=>r.dataset.module=wpath(r)??"");

function open(row){const by={};row.closest("tbody").querySelectorAll("tr").forEach(r=>by[r.dataset.i]=r);
 const up=[];for(let p=by[row.dataset.p];p;p=by[p.dataset.p])up.unshift(p);
 up.forEach(p=>{if(p.classList.contains("shut"))p.querySelector("button.tg")?.click()})}

function select(path,from){
 sel=path===undefined?null:path;const m=D.modules.find(x=>x.path===sel),all=sel===null||sel==="";
 $("#sel").textContent=sel===null?"none":sel===""?D.name:sel;$("#selcls").textContent=m&&sel!==""?m.label:"";$("#clear").hidden=sel===null;
 $$("[role=tabpanel] .filter").forEach(f=>{f.hidden=all||f.parentElement.id==="tab-summary"||f.parentElement.id==="tab-diagram";f.querySelector("code").textContent=sel});
 $$("#tab-summary tr[data-path]").forEach(r=>{const on=!all&&r.dataset.path===sel;r.classList.toggle("sel",on);if(on){open(r);if(from!=="summary")r.scrollIntoView({block:"nearest"})}});
 $$("#tab-cost .card").forEach(c=>{const ps=JSON.parse(c.dataset.paths);c.hidden=!all&&!ps.some(p=>under(p,sel));c.classList.toggle("sel",!all&&ps.includes(sel))});
 wrows.forEach(r=>{r.classList.toggle("xo",!all&&!(r.dataset.module&&under(r.dataset.module,sel)));r.classList.toggle("sel",!all&&r.dataset.path!==undefined&&norm(r.dataset.path)===sel);
  const t=r.querySelector("td.m"),x=t&&t.lastChild;if(x&&x.nodeType===3){r.label??=x.textContent;x.textContent=all?r.label:t.title}});  // filtered: full names, the parents are hidden
 const classes=new Set(D.modules.filter(x=>all||under(x.path,sel)).map(x=>x.cls));
 $$("#tab-profile details[data-name]").forEach(d=>d.hidden=!all&&!classes.has(d.dataset.name.split(".")[0]));
 if(from!=="diagram")post({expex:"select",path:sel===null?"":sel})}

$("#tab-summary").addEventListener("click",e=>{if(e.target.closest("button,th"))return;const r=e.target.closest("tr[data-path]");
 if(r&&r.dataset.path!=="")select(module(r.dataset.path)===sel?null:module(r.dataset.path),"summary")});
$("#tab-activations").addEventListener("click",e=>{if(e.target.closest("button,th"))return;const r=e.target.closest("tbody tr");
 if(r&&r.dataset.module){const p=module(r.dataset.module);select(p===sel?null:p,"activations")}});
$("#tab-cost").addEventListener("click",e=>{const c=e.target.closest(".card");if(c){const p=JSON.parse(c.dataset.paths)[0];select(p===sel?null:p,"cost")}});
$$(".filter .all").forEach(b=>b.onclick=()=>select(null));
$("#clear").onclick=()=>select(null);
addEventListener("keydown",e=>{if(e.key==="Escape")select(null)});

// light, dark, or the system's
function setTheme(t){theme=t;if(t==="auto")document.documentElement.removeAttribute("data-theme");else document.documentElement.dataset.theme=t;
 $("#theme").textContent=t[0].toUpperCase()+t.slice(1);try{localStorage.setItem("expex-theme",t)}catch(e){}post({expex:"theme",theme:t})}
try{theme=localStorage.getItem("expex-theme")||"auto"}catch(e){}
setTheme(theme);$("#theme").onclick=()=>setTheme(["auto","light","dark"][(["auto","light","dark"].indexOf(theme)+1)%3]);
window.expexSelect=select;
})();"""

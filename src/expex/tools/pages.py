"""The HTML the tools write: one page shell, one palette in light and dark, shared table parts.

``expex.summary``, ``debug.watch``, ``expex.diff`` and ``debug.profile`` render
to self-contained HTML (inline CSS and JS, nothing from a CDN). Notebook
output is a fragment whose styles are scoped to its own id (:func:`uid`), so
several can share a page; ``save(path)`` wraps it in :func:`document`. Each tool
keeps only what is its own: its markup, its extra styles and its script.

- :func:`tokens`: the palette as CSS custom properties on a selector, switched
  by ``prefers-color-scheme:dark``: ``--bg`` page, ``--fg`` text, ``--mu`` muted
  text, ``--ln`` lines, ``--hd`` headers and panels, ``--ac`` accent, ``--br``
  bars, ``--tip`` / ``--tipfg`` tooltips.
- :data:`TABLE_CSS` / :data:`TABLE_JS`: a table of a tree of rows (``data-i``,
  ``data-p`` for the parent), collapsible with ``button.tg`` and sortable by
  headers with ``data-k`` (rows sort within their parent by ``data-<k>``).
- :func:`data`: a value as JSON that is safe inside ``<script>``.

Light or dark follows ``prefers-color-scheme``; ``<html data-theme="light">`` or
``"dark"`` overrides it (the toggles of ``expex.inspect`` and the diagram page).
Fragments that ``expex.inspect`` puts in one page mark the modules they show
with ``data-path`` (``blocks[3].attn``), so that a click selects the module.
"""

import html
import itertools
import json
from typing import Any

_LIGHT = "--bg:#fff;--fg:#1f2328;--mu:#656d76;--ln:#d0d7de;--hd:#f6f8fa;--ac:#2f6fdb;--br:rgba(47,111,219,.16);--tip:#1f2328;--tipfg:#f6f8fa"
_DARK = "--bg:#0d1117;--fg:#e6edf3;--mu:#8d96a0;--ln:#30363d;--hd:#161b22;--ac:#58a6ff;--br:rgba(88,166,255,.22);--tip:#e6edf3;--tipfg:#0d1117"
MONO = "ui-monospace,SFMono-Regular,Menlo,monospace"
SANS = "system-ui,-apple-system,Segoe UI,sans-serif"
_ids = itertools.count()


def uid(prefix: str) -> str:
    """A new element id, so that each fragment on a page styles and scripts only itself."""
    return f"{prefix}{next(_ids)}"


def tokens(scope: str, light: str = "", dark: str = "") -> str:
    """The palette on ``scope`` (``:root`` or ``#id``), plus a tool's own ``light`` / ``dark`` properties.

    Dark follows ``prefers-color-scheme``, unless the page's ``<html data-theme="light|dark">``
    says otherwise (the toggle of ``expex.inspect`` and of the diagram page sets it).
    """
    on = lambda when: f"{when}{scope}" if scope == ":root" else f"{when} {scope}"  # noqa: E731
    return (f"{scope}{{{_LIGHT};{light}}}@media (prefers-color-scheme:dark){{{on(':root:not([data-theme=light])')}"
            f"{{{_DARK};{dark}}}}}{on(':root[data-theme=dark]')}{{{_DARK};{dark}}}")  # fmt: skip


#: ``color-scheme`` (form controls, scroll bars) following the same switch as :func:`tokens`.
SCHEME = ":root{color-scheme:light dark}:root[data-theme=light]{color-scheme:light}:root[data-theme=dark]{color-scheme:dark}"


def document(title: str, body: str, style: str = "") -> str:
    """A standalone page around ``body``; ``style`` is added to its stylesheet."""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
        f"content=\"width=device-width, initial-scale=1\"><title>{html.escape(title)}</title><style>{SCHEME}"
        f"body{{margin:0;padding:16px;background:Canvas;color:CanvasText}}{style}"
        f"</style></head><body>{body}</body></html>"
    )


def data(value: Any) -> str:
    """``value`` as JSON for a ``<script>``: ``</`` is escaped so the data cannot close the tag."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


#: Styles of a tree table inside ``#ID`` (replace ``#ID`` with the fragment's id).
TABLE_CSS = f"""#ID{{color:var(--fg);background:var(--bg);font:13px/1.45 {SANS};max-width:100%;container-type:inline-size;border:1px solid var(--ln);border-radius:8px;padding:10px 12px}}
#ID .hd{{margin-bottom:8px;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}}#ID .hd span{{color:var(--mu);font:12px {MONO}}}
#ID .sc{{overflow-x:auto}}#ID table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
#ID th,#ID td{{padding:3px 10px;border-bottom:1px solid var(--ln);text-align:left;white-space:nowrap;vertical-align:baseline}}
#ID th{{background:var(--hd);font-weight:600;color:var(--mu);font-size:12px;position:sticky;top:0}}#ID th[data-k]{{cursor:pointer;color:var(--ac)}}
#ID th[data-k]::after{{content:" ↕";opacity:.5}}#ID th[data-dir="1"]::after{{content:" ↓";opacity:1}}#ID th[data-dir="-1"]::after{{content:" ↑";opacity:1}}
#ID .n{{text-align:right}}#ID .m{{padding-left:calc(6px + var(--d)*16px);font-weight:500}}#ID .c,#ID .f{{font-family:{MONO};font-size:12px}}
#ID .f{{color:var(--mu);white-space:normal;min-width:12ch}}#ID .x{{color:var(--ac);margin-left:6px;font-weight:600}}
#ID .bar{{background:linear-gradient(to left,var(--br) calc(var(--s)*100%),transparent 0)}}#ID tbody tr:hover{{background:var(--hd)}}
#ID tfoot td{{font-weight:600;border-top:2px solid var(--ln);border-bottom:0}}#ID .tg{{display:inline-block;width:16px;margin-right:2px;border:0;background:none;color:var(--mu);cursor:pointer;padding:0;font:inherit}}
#ID tr.shut .tg{{transform:rotate(-90deg)}}#ID ul{{margin:6px 0 0;padding-left:18px;color:var(--mu);font-size:12px}}#ID ul:empty{{display:none}}"""

#: Collapsing and sorting for :data:`TABLE_CSS` (replace ``ID`` with the id as a JS string).
TABLE_JS = """(()=>{const root=document.getElementById(ID),body=root.querySelector("tbody"),rows=[...body.rows],by={},kids={};
rows.forEach(r=>{by[r.dataset.i]=r;(kids[r.dataset.p]??=[]).push(r)});let key=null,dir=1;
const vis=()=>rows.forEach(r=>{let p=r.dataset.p,h=false;while(p!==""){if(by[p].classList.contains("shut")){h=true;break}p=by[p].dataset.p}r.hidden=h});
const order=()=>{const out=[],walk=p=>{const c=[...(kids[p]||[])];if(key)c.sort((a,b)=>dir*(b.dataset[key]-a.dataset[key]));c.forEach(r=>{out.push(r);walk(r.dataset.i)})};walk("");out.forEach(r=>body.appendChild(r))};
root.addEventListener("click",e=>{const b=e.target.closest("button.tg");if(b){const r=b.closest("tr"),shut=r.classList.toggle("shut");b.setAttribute("aria-expanded",String(!shut));vis();return}
const h=e.target.closest("th[data-k]");if(!h)return;if(key===h.dataset.k){if(dir===1)dir=-1;else key=null}else{key=h.dataset.k;dir=1}
root.querySelectorAll("th[data-k]").forEach(t=>t.dataset.dir=t.dataset.k===key?dir:"");order()})})();"""


def table(scope: str, style: str, markup: str, light: str = "", dark: str = "") -> str:
    """A tree table fragment: ``markup`` (header and table) in ``<div id=scope>``, with its styles and script."""
    css = (tokens("#ID", light, dark) + TABLE_CSS + style).replace("#ID", "#" + scope)
    return (f'<div class="txs" id="{scope}"><style>{css}</style>{markup}'
            f'<script>{TABLE_JS.replace("ID", json.dumps(scope))}</script></div>')  # fmt: skip


def toggle(has_children: bool) -> str:
    """The collapse button of a tree row that has children, else a spacer."""
    if has_children:
        return '<button class="tg" aria-expanded="true" aria-label="collapse">▾</button>'
    return '<span class="tg"></span>'

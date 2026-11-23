"""Comparing two models or checkpoints field by field: ``tx.diff(a, b)``.

::

    d = tx.diff(model, tx.load(Model, "ckpt.npz"))   # or tx.diff(model, "ckpt.npz")
    print(d)                                         # 3 changed, 1 added, 120 unchanged ...
    d.save("diff.html")                              # an interactive report

Fields are named by the keys :func:`tx.save <typax._io.save>` writes
(``layers[2].mixer.x_proj``): a model is saved to memory and read back like a
file, so models and files compare the same way. Arrays are compared by shape,
dtype and values; the plain static fields that ``save`` keeps (numbers,
strings, ...) by equality. Other pytrees are named by ``jax.tree_util.keystr``.
"""

import html
import io
import itertools
import json
import math
import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._dims import describe
from ._io import _META, _decode, save
from ._spec import StructSpec, parse_owner

_STRUCTURAL = ("removed", "added", "reshaped", "static")
_ids = itertools.count()


@dataclass(frozen=True)
class Field:
    """One field of the diff. ``kind`` is ``same``, ``changed`` (values), ``reshaped``
    (shape or dtype), ``added``, ``removed`` or ``static``; ``a`` / ``b`` describe each
    side (``f32[36, 128]``, or a static value's repr; ``None`` when absent). For
    ``changed``: the max abs difference, ``‖a−b‖ / max(‖a‖, ‖b‖)`` and the fraction of elements
    outside the tolerance."""

    path: str
    kind: str
    a: str | None
    b: str | None
    max_abs: float = 0.0
    rel: float = 0.0
    frac: float = 0.0


class Diff:
    """The result of :func:`diff`. ``bool(d)`` is whether anything differs; ``d.changes``
    lists the differences in report order (structural first, then values by relative
    change); ``d.added``, ``d.removed``, ``d.reshaped``, ``d.static``, ``d.changed`` and
    ``d.unchanged`` list them by kind; ``d.fields`` has every field in model order."""

    added: list[Field]
    removed: list[Field]
    reshaped: list[Field]
    static: list[Field]
    changed: list[Field]

    def __init__(self, fields: list[Field], names: tuple[str, str]):
        self.fields, self.names = fields, names
        for kind in ("added", "removed", "reshaped", "static", "changed"):
            setattr(self, kind, [f for f in fields if f.kind == kind])
        self.changed.sort(key=lambda f: (-f.rel, -f.max_abs))
        self.unchanged = [f for f in fields if f.kind == "same"]
        self.changes = [f for f in fields if f.kind in _STRUCTURAL] + self.changed

    def __bool__(self) -> bool:
        return bool(self.changes)

    def summary(self) -> str:
        counts = [(len(self.changed), "changed"), (len(self.reshaped), "reshaped"), (len(self.added), "added"),
                  (len(self.removed), "removed"), (len(self.static), "static changed")]  # fmt: skip
        return ", ".join([f"{n} {w}" for n, w in counts if n] + [f"{len(self.unchanged)} unchanged"])

    def __repr__(self) -> str:
        return f"<Diff {self.names[0]} → {self.names[1]}: {self.summary()}>"

    def __str__(self) -> str:
        rows: list[list[str]] = [
            [f.kind, f.path, str(f.b if f.kind == "added" else f.a if f.kind == "removed" else f"{f.a} → {f.b}")]
            for f in self.changes if f.kind != "changed"
        ]  # fmt: skip
        if self.changed:
            rows += [[""]] * bool(rows) + [["", "", "", "max |a−b|", "relative", "changed"]]
            rows += [["changed", f.path, str(f.a), _num(f.max_abs), _num(f.rel), f"{f.frac:.1%}"] for f in self.changed]
        widths = [max((len(r[i]) for r in rows if len(r) > i + 1 or len(r) == 6), default=0) for i in range(6)]
        lines = ["  ".join(c.rjust(w) if i > 2 else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
                 for r in rows]  # fmt: skip
        return "\n".join([self.summary(), *([""] if lines else []), *lines])

    def _repr_html_(self) -> str:
        top = max([f.rel for f in self.changed if math.isfinite(f.rel)] or [1.0]) or 1.0
        uid = f"txdiff{next(_ids)}"
        legend = "".join(f'<span class="lg k-{k}"><i></i>{n} {w}</span>' for k, w, n in (
            ("changed", "changed values", len(self.changed)), ("reshaped", "shape / dtype", len(self.reshaped)),
            ("added", "added", len(self.added)), ("removed", "removed", len(self.removed)),
            ("static", "static", len(self.static)), ("same", "unchanged", len(self.unchanged))) if n)  # fmt: skip
        names = " → ".join(html.escape(n) for n in self.names)
        return (f'<div class="txdiff" id="{uid}"><style>{_CSS}</style><header><b>{names}</b>'
                f'<span class="sum">{html.escape(self.summary())}</span><div class="legend">{legend}</div>'
                f'<div class="ctl"><label><input type="checkbox" class="hide"> Hide unchanged</label>'
                f'<label>Order <select class="order"><option value="i">model order</option>'
                f'<option value="rel">largest change first</option></select></label></div></header>'
                f'<div class="kids">{_render(_tree(self.fields), top)}</div>'
                f'<script>{_JS.replace("UID", uid)}</script></div>')  # fmt: skip

    def save(self, path: Any) -> None:
        """Write the HTML report to ``path`` as a self-contained page."""
        page = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" '
                f'content="width=device-width,initial-scale=1"><title>Model diff</title>'
                f"<style>{_PAGE}</style></head><body>{self._repr_html_()}</body></html>")  # fmt: skip
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)


def diff(a: Any, b: Any, *, atol: float = 0.0, rtol: float = 0.0) -> Diff:
    """Compare two models, each a struct, another pytree or a file written by ``tx.save``.

    A field's values count as changed when any element is outside
    ``|a - b| <= atol + rtol * |b|`` (NaNs in the same place are equal).
    """
    (na, ea), (nb, eb) = _read(a), _read(b)
    order: dict[str, float] = {k: float(i) for i, k in enumerate(ea)}
    last = -1.0
    for k in eb:  # keys only in b go after the key that precedes them in b
        last = order[k] if k in order else order.setdefault(k, last + 1e-6)
    fields = []
    for k in sorted(order, key=order.__getitem__):
        x, y = ea.get(k), eb.get(k)
        arr_x, arr_y = isinstance(x, np.ndarray), isinstance(y, np.ndarray)
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            fields.append(_compare(k, x, y, atol, rtol))
        elif arr_x or arr_y:  # missing, or a None static on the other side
            fields.append(Field(k, "removed" if arr_x else "added", describe(x) if arr_x else None,
                                describe(y) if arr_y else None))  # fmt: skip
        elif k in ea and k in eb:  # one-sided statics come with their module's arrays
            fields.append(Field(k, "same" if x == y else "static", repr(x), repr(y)))
    return Diff(fields, (na, nb))


def _read(x: Any) -> tuple[str, dict[str, Any]]:
    """A name, and every array and plain static value by its ``tx.save`` key."""
    if not isinstance(x, (str, os.PathLike)):
        if not isinstance(parse_owner(type(x)), StructSpec):
            leaves = jax.tree_util.tree_flatten_with_path(x)[0]
            return type(x).__name__, {jax.tree_util.keystr(p).lstrip("."): np.asarray(v) for p, v in leaves}
        source: Any = io.BytesIO()
        save(x, source)
        source.seek(0)
    else:
        source = x
    with np.load(source, allow_pickle=False) as f:
        out: dict[str, Any] = {k: f[k] for k in f.files if k != _META}
        meta = json.loads(str(f[_META])) if _META in f.files else {}
    for k, name in meta.get("dtypes", {}).items():
        out[k] = out[k].view(jnp.dtype(name))
    out.update({k: _decode(v) for k, v in meta.get("static", {}).items()})
    name = meta.get("class", "?")
    return (f"{os.path.basename(x)} ({name})" if isinstance(x, (str, os.PathLike)) else name), out


def _compare(path: str, x: np.ndarray, y: np.ndarray, atol: float, rtol: float) -> Field:
    dx, dy = describe(x), describe(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return Field(path, "reshaped", dx, dy)
    t = np.complex128 if np.iscomplexobj(x) else np.float64
    u, v = x.astype(t), y.astype(t)
    with np.errstate(invalid="ignore", over="ignore"):
        d = np.abs(u - v)
        d = np.where(np.isnan(u) & np.isnan(v), 0.0, np.where(np.isnan(d), np.inf, d))
        far = ~np.isclose(u, v, rtol=rtol, atol=atol, equal_nan=True)
        if not far.any():
            return Field(path, "same", dx, dy)
        nd = float(np.sqrt((d**2).sum()))
        nu = max(float(np.linalg.norm(np.nan_to_num(x).ravel())) for x in (u, v))
    # relative to the larger side, so a zero-initialised bias that moved stays finite
    rel = nd / nu if nu else (0.0 if nd == 0 else math.inf)
    return Field(path, "changed", dx, dy, float(d.max()), rel, float(far.mean()))


def _num(x: float) -> str:
    return f"{x:.2e}" if x and math.isfinite(x) else str(x).replace(".0", "")


def _tree(fields: list[Field]) -> dict[str, Any]:
    """Fields nested by module: ``blocks[2].mlp.w1`` under ``blocks`` › ``[2]`` › ``mlp``."""
    root: dict[str, Any] = {}
    for f in fields:
        parts = f.path.replace("[", ".[").lstrip(".").split(".") if f.path else ["(root)"]
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = f
    return root


def _render(node: dict[str, Any], top: float) -> str:
    out = []
    for i, (name, v) in enumerate(node.items()):
        leaves = list(_leaves(v))
        ranks = (math.inf if f.kind in _STRUCTURAL else f.rel if f.kind == "changed" else -1 for f in leaves)
        rank = min(max(ranks), 1e300)  # structural first, then by relative change
        kind = next((k for k in (*_STRUCTURAL, "changed") if any(f.kind == k for f in leaves)), "same")
        attrs = f'class="{"g" if isinstance(v, dict) else "f"} k-{kind}" data-i="{i}" data-rel="{rank}"'
        if isinstance(v, dict):
            n = sum(f.kind != "same" for f in leaves)
            note = f"{n} of {len(leaves)} differ" if n else f"{len(leaves)} unchanged"
            out.append(f'<details open {attrs}><summary><i></i><b>{html.escape(name)}</b><span class="note">'
                       f'{note}</span></summary><div class="kids">{_render(v, top)}</div></details>')  # fmt: skip
            continue
        f, stats = v, ""
        desc = f.b if f.kind == "added" else f.a if f.kind in ("removed", "same", "changed") else f"{f.a} → {f.b}"
        if f.kind == "changed":
            w = 100 * min(f.rel / top, 1.0)
            stats = (f'<span class="st">rel {_num(f.rel)} · max {_num(f.max_abs)} · {f.frac:.0%} of values</span>'
                     f'<span class="bar"><span style="width:{w:.1f}%"></span></span>')  # fmt: skip
        tag = "unchanged" if f.kind == "same" else f.kind
        out.append(f'<div {attrs}><i></i><b>{html.escape(name)}</b><span class="tag">{tag}</span>'
                   f"<code>{html.escape(str(desc))}</code>{stats}</div>")  # fmt: skip
    return "".join(out)


def _leaves(v: Any) -> Any:
    if isinstance(v, Field):
        yield v
    else:
        for c in v.values():
            yield from _leaves(c)


_PAGE = "body{margin:0;background:#fbfbfa}@media (prefers-color-scheme:dark){body{background:#17181b}}"
_CSS = """
.txdiff{--bg:#fbfbfa;--fg:#1d1f23;--mute:#6b7079;--line:#e4e4e0;--same:#9aa0a8;--changed:#b45309;--reshaped:#7c3aed;
--added:#15803d;--removed:#c2410c;--static:#1d4ed8;--barbg:#ecebe6;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;padding:16px;max-width:980px;box-sizing:border-box}
@media (prefers-color-scheme:dark){.txdiff{--bg:#17181b;--fg:#e6e6e3;--mute:#9a9ea6;--line:#2c2e33;--same:#6b7079;
--changed:#f0a83c;--reshaped:#a78bfa;--added:#4ade80;--removed:#fb7f5a;--static:#60a5fa;--barbg:#26282d}}
.txdiff header{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;
padding-bottom:12px;border-bottom:1px solid var(--line)}
.txdiff header>b{font-size:16px}.txdiff .sum{color:var(--mute)}
.txdiff .legend,.txdiff .ctl{display:flex;flex-wrap:wrap;gap:4px 14px;width:100%}
.txdiff .lg{display:inline-flex;align-items:center;gap:6px;font-size:13px}.txdiff .ctl{color:var(--mute);font-size:13px}
.txdiff select,.txdiff input{font:inherit;accent-color:var(--changed)}
.txdiff select{background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}
.txdiff i{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--c);flex:none}
.txdiff .k-same{--c:var(--same)}.txdiff .k-changed{--c:var(--changed)}.txdiff .k-reshaped{--c:var(--reshaped)}
.txdiff .k-added{--c:var(--added)}.txdiff .k-removed{--c:var(--removed)}.txdiff .k-static{--c:var(--static)}
.txdiff .kids .kids{margin-left:9px;padding-left:12px;border-left:1px solid var(--line)}
.txdiff summary{cursor:pointer;padding:3px 0;display:flex;align-items:center;gap:8px;list-style:none}
.txdiff summary::-webkit-details-marker{display:none}.txdiff summary::before{content:"▸";color:var(--mute);width:10px}
.txdiff details[open]>summary::before{content:"▾"}.txdiff .note{color:var(--mute);font-size:12px}
.txdiff .f{display:flex;flex-wrap:wrap;align-items:center;gap:2px 8px;padding:3px 0 3px 18px}
.txdiff .f b,.txdiff summary b{font-weight:600}
.txdiff .k-same>b,.txdiff .k-same>summary b{font-weight:400;color:var(--mute)}
.txdiff .tag{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--c)}
.txdiff code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mute);overflow-wrap:anywhere}
.txdiff .k-reshaped>code,.txdiff .k-static>code,.txdiff .k-added>code,.txdiff .k-removed>code{color:var(--fg)}
.txdiff .st{margin-left:auto;font-size:12px;color:var(--mute);font-variant-numeric:tabular-nums}
.txdiff .bar{flex:1 1 80px;max-width:160px;min-width:60px;height:6px;border-radius:3px;background:var(--barbg)}
.txdiff .bar span{display:block;height:100%;border-radius:3px;background:var(--changed)}
.txdiff.hide-same .k-same{display:none}
"""
_JS = """(()=>{const r=document.getElementById("UID");
r.querySelector(".hide").onchange=e=>r.classList.toggle("hide-same",e.target.checked);
r.querySelector(".order").onchange=e=>{const by=e.target.value;r.querySelectorAll(".kids").forEach(k=>{
[...k.children].sort((a,b)=>by=="rel"?b.dataset.rel-a.dataset.rel||a.dataset.i-b.dataset.i:a.dataset.i-b.dataset.i)
.forEach(c=>k.appendChild(c))})}})();"""

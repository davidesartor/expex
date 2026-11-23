"""Line-by-line analytical profile: FLOPs, bytes and shapes per source line.

``profile(f, **sizes)`` traces ``f`` like ``expex.cost`` (symbolic sizes, no
data), counts each equation with the same rules, and attributes it to the
source line that created it::

    >>> print(profile(Transformer.__call__, T=1024, D=512, F=2048, Vocab=32000, blocks=12))

- Every function the trace passes through gets its own section (``f`` first,
  then the others by cost). An operation counts once, as *self* cost, on the
  innermost line that made it; the lines that called into it show it as
  *calls*. Self costs add up to ``expex.cost(f).flops``.
- Lines in a staged loop are multiplied by its trip count, and say so
  (``×L`` for a loop over a list of modules, ``×T`` for ``range(T)``). Like
  ``expex.cost``, a ``while`` body counts once and a ``cond`` its costliest branch.
- *bytes* is the size of every value a line's operations produce (times the
  trip count): data written, not peak memory (see ``expex.debug.memory``).
- ``str()`` is annotated source; ``_repr_html_()`` / ``save(path)`` give a
  self-contained page with a heat map (FLOPs or bytes) and the formulas on hover.
"""

import inspect
import linecache
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from . import pages
from .cost import Poly, _numel, cost, flops, si
from .trace import _HERE, resolve, sub_jaxprs

type Key = tuple[str, str, int]  # (file, qualified name, first line): one section


@dataclass
class Line:
    """One source line: its self cost, the cost of what it calls, and what it produces."""

    number: int
    text: str
    flops: Poly = field(default_factory=Poly)
    bytes: Poly = field(default_factory=Poly)
    calls: Poly = field(default_factory=Poly)  # FLOPs of the functions called from here
    loops: list[str] = field(default_factory=list)  # trip counts it is multiplied by
    shapes: list[str] = field(default_factory=list)  # values produced, last one last


@dataclass
class Section:
    """A function's source, line by line."""

    name: str
    file: str
    lines: list[Line]

    @property
    def flops(self) -> Poly:
        return sum((ln.flops for ln in self.lines), Poly())


@dataclass
class Profile:
    """Per-line costs of a function and the ones it calls. ``str()`` / ``save()`` show it."""

    name: str
    sizes: list[str]  # size names, display order
    given: dict[str, int]  # sizes given as keywords (a list's length as ``L``)
    flops: Poly  # total, as ``expex.cost`` (formulas here are in all sizes: see ``number``)
    sections: list[Section]
    other: Poly  # FLOPs not on any line of these functions (argument checks, ...)
    notes: list[str] = field(default_factory=list)

    @property
    def bytes(self) -> Poly:
        return sum((ln.bytes for s in self.sections for ln in s.lines), Poly())

    def number(self, p: Poly) -> Poly:
        """``p`` with the given sizes put in: a number once every size is given."""
        return p.subs(self.given)

    def share(self, p: Poly, of: Poly) -> float:
        """``p / of`` at the given sizes (every other size at 1000)."""
        return self.number(p).sample() / t if (t := self.number(of).sample()) else 0.0

    def _cell(self, p: Poly, unit: str) -> tuple[str, str]:
        return p.format(self.sizes), "" if (v := self.number(p).value) is None else si(v, unit)

    def _sizes(self) -> str:
        return ", ".join([n for n in self.sizes if n not in self.given] + [f"{k}={v}" for k, v in self.given.items()])

    def __str__(self) -> str:
        given = self._sizes()
        f, n = self._cell(self.flops, "FLOP")
        b, bn = self._cell(self.bytes, "B")
        out = [f"{self.name}   (sizes: {given or 'none'})", f"  flops  {f}" + (f"  = {n}" if n else ""),
               f"  bytes  {_clip(b)}" + (f"  = {bn}" if bn else "") + ("" if self.given else "   (shares: every size at 1000)")]  # fmt: skip
        cut = {id(s): _indent(s) for s in self.sections}
        width = min(max((len(ln.text.rstrip()) - cut[id(s)] for s in self.sections for ln in s.lines), default=20), 56)
        for s in self.sections:
            pct = 100 * self.share(s.flops, self.flops)
            out += ["", f"── {s.name}   {_short(s.file)}:{s.lines[0].number}   self {pct:.1f}%"]
            for ln in s.lines:
                src = ln.text.rstrip()[cut[id(s)] :]
                src = src if len(src) <= width else src[: width - 1] + "…"
                row = f"{ln.number:5d} | {src:<{width}}"
                if ln.flops.terms or ln.bytes.terms or ln.calls.terms:
                    share = self.share(ln.flops, self.flops)
                    f, n = self._cell(ln.flops, "FLOP") if ln.flops.terms else ("-", "")
                    loop = "×" + "·".join(ln.loops) + " " if ln.loops else ""
                    f = _clip(f)
                    num = f" {n:>11}" if self.given else ""
                    row += f"  {(ln.shapes or [''])[-1]:<14} {loop + f:<28}{num} {100 * share:5.1f}% {_bar(share):<10}"
                    if (cs := self.share(ln.calls, self.flops)) >= 5e-4:
                        row += f"  calls {100 * cs:.1f}%"
                    b, bn = self._cell(ln.bytes, "B")
                    row += f"  {bn or _clip(b) + ' B'}" if ln.bytes.terms else ""
                out.append(row.rstrip())
        if self.other.terms:
            out.append(f"\n(not on a source line: {self._cell(self.other, 'FLOP')[1] or self.other.format(self.sizes)})")
        return "\n".join(out + [f"note: {n}" for n in dict.fromkeys(self.notes)])

    __repr__ = __str__

    def _repr_html_(self) -> str:
        data: dict[str, Any] = {"title": self.name, "sizes": self._sizes(),
                "total": self._cell(self.flops, "FLOP"), "bytes": self._cell(self.bytes, "B"), "sections": []}  # fmt: skip
        for s in self.sections:
            rows = [{"n": ln.number, "src": ln.text.rstrip()[_indent(s) :], "f": self._cell(ln.flops, "FLOP"),
                     "b": self._cell(ln.bytes, "B"), "c": self._cell(ln.calls, "FLOP") if ln.calls.terms else None,
                     "fs": self.share(ln.flops, self.flops), "bs": self.share(ln.bytes, self.bytes),
                     "cs": self.share(ln.calls, self.flops), "loops": ln.loops, "shapes": ln.shapes[-4:],
                     "hot": bool(ln.flops.terms or ln.bytes.terms or ln.calls.terms)} for ln in s.lines]  # fmt: skip
            data["sections"].append({"name": s.name, "file": f"{_short(s.file)}:{s.lines[0].number}", "share": self.share(s.flops, self.flops), "bshare": self.share(sum((ln.bytes for ln in s.lines), Poly()), self.bytes), "rows": rows})  # fmt: skip
        style = pages.tokens(":root", _LIGHT, _DARK) + _CSS
        return pages.document(f"{self.name} profile", _BODY.replace("%DATA%", pages.data(data)), style)

    def save(self, path: str) -> str:
        """Write the HTML page to ``path``; returns the path."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._repr_html_())
        return path


def _indent(s: Section) -> int:
    return min((len(t) - len(t.lstrip()) for ln in s.lines if (t := ln.text.rstrip())), default=0)


def _clip(formula: str) -> str:
    return formula if len(formula) <= 24 else formula.split(" + ")[0].split(" - ")[0] + " + …"


def _bar(share: float) -> str:
    return "█" * round(10 * share) if share >= 0.01 else ""


def _short(file: str) -> str:
    rel = os.path.relpath(file)
    return os.path.basename(file) if rel.startswith("..") else rel


def _stack(eqn: Any) -> list[tuple[Key, int]] | None:
    """The user's frames that made ``eqn``, innermost first, as (section, line).

    expex's own frames are skipped, and so are the helper functions its
    rewriting creates (``__bn_body_1__``): their lines belong to the enclosing function.
    """
    from jax._src import source_info_util as su

    tb = eqn.source_info.traceback
    if tb is None:
        return None
    out: list[tuple[Key, int]] = []
    pending: int | None = None
    for code, lasti in zip(*tb.raw_frames()):
        if not su.is_user_filename(code.co_filename) or os.path.abspath(code.co_filename).startswith(_HERE):
            continue
        line = su.raw_frame_to_frame(code, lasti).start_line
        if code.co_name.startswith("__bn_"):
            pending = line if pending is None else pending
            continue
        key = (code.co_filename, code.co_qualname, code.co_firstlineno)
        line, pending = (pending if pending is not None else line), None
        if not out or out[-1][0] != key:
            out.append((key, line))
    return out or None


class _Walk:
    """One trace's per-line costs: ``lines[(section, line)]``."""

    def __init__(self, root: tuple[str, str], n: int, lists: bool):
        self.root, self.n, self.lists = root, n, lists
        self.lines: dict[tuple[Key, int], Line] = {}
        self.other = Poly()

    def line(self, at: tuple[Key, int]) -> Line:
        return self.lines.setdefault(at, Line(at[1], ""))

    def cut(self, stack: list[tuple[Key, int]] | None) -> list[tuple[Key, int]]:
        """``stack`` up to the profiled function; [] if it is not in it."""
        for i in range(len(stack or []) - 1, -1, -1):
            if stack[i][0][:2] == self.root:  # type: ignore[index]
                return stack[: i + 1]  # type: ignore[index]
        return []

    def run(self, jaxpr: Any, mult: Poly, loops: list[tuple[int, str]], outer: list[tuple[Key, int]]) -> None:
        for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
            stack = self.cut(st := _stack(eqn) or [])
            if not stack and outer:  # a body traced apart from its caller: continue the caller's stack
                stack = [*st, *outer[st[-1][0] == outer[0][0] :]] if st else outer
            name, p = eqn.primitive.name, eqn.params
            subs = [j for _, j in sub_jaxprs(p)]
            if name == "scan":
                n = p["length"]
                label = "L" if self.lists and isinstance(n, int) and n == self.n else Poly.of(n).format()
                for j in subs:
                    self.run(j, mult * Poly.of(n), loops + [(len(stack), label)], stack)
            elif name == "cond":
                costs = [flops(j, []) for j in subs]
                self.run(subs[costs.index(max(costs, key=Poly.sample))], mult, loops, stack)
            else:
                for j in subs:
                    self.run(j, mult, loops, stack)
            fl = Poly() if subs else flops(SimpleNamespace(eqns=[eqn]), []) * mult
            avals = [v.aval for v in eqn.outvars if hasattr(getattr(v, "aval", None), "shape")]
            by = sum((_numel(a.shape) * _itemsize(a.dtype) for a in avals), Poly()) * mult if not subs else Poly()
            if not stack:
                self.other += fl
                continue
            shapes = [_pretty(a.str_short(short_dtypes=True)) for a in avals if _itemsize(a.dtype)]
            for i, at in enumerate(stack):
                ln = self.line(at)
                if i == 0:
                    ln.flops += fl
                    ln.bytes += by
                else:
                    ln.calls += fl
                if fl.terms or by.terms or (subs and shapes) or i:
                    ln.shapes = [s for s in ln.shapes if s not in shapes[-1:]] + shapes[-1:]
                for k, label in loops:
                    if i <= len(stack) - k and label not in ln.loops and (fl.terms or by.terms or i):
                        ln.loops.append(label)


def _itemsize(dtype: Any) -> int:
    try:
        return np.dtype(dtype).itemsize
    except TypeError:
        return 0  # random keys and other extended dtypes


def _pretty(shape: str) -> str:
    return re.sub(r"floordiv\((\w+), (\d+)\)", r"\1//\2", shape)


def _source(key: Key) -> tuple[str, list[str], int]:
    file, qual, first = key
    lines = linecache.getlines(file)
    block = inspect.getblock(lines[first - 1 :]) if 0 < first <= len(lines) else []
    return qual, list(block), first


def profile(target: Any, /, **sizes: Any) -> Profile:
    """Cost of every source line of ``target`` and of the functions it calls.

    ``target`` and the keywords are as for ``expex.cost``: an ``@fn`` or a struct
    method (unbound: sizes stay symbolic; bound: the instance's sizes), static
    parameters (``heads=4``), list lengths (``blocks=12``) and sizes (``T=1024``).
    """
    total = cost(target, **sizes)
    s = resolve(target, sizes, "profile")
    raw = getattr(s.f.raw, "__func__", s.f.raw)
    root = (raw.__code__.co_filename, raw.__code__.co_qualname)
    walks = []
    for n in (1, 2):
        jaxpr, _, lists = s.trace(n)
        w = _Walk(root, n, bool(lists))
        w.run(jaxpr, Poly.of(1), [], [])
        walks.append(w)
        if not lists:
            break
    lines, other = walks[0].lines, walks[0].other
    if len(walks) == 2:  # linear in the list length: extrapolate from 1 and 2, as expex.cost does
        L, w2 = Poly.of("L") - 1, walks[1]
        for at, a in lines.items():
            w2.lines.setdefault(at, Line(a.number, "", loops=a.loops, shapes=a.shapes))
        for at, b in w2.lines.items():
            a = lines.get(at, Line(b.number, ""))
            b.flops, b.bytes = a.flops + L * (b.flops - a.flops), a.bytes + L * (b.bytes - a.bytes)
            b.calls = a.calls + L * (b.calls - a.calls)
        lines, other = w2.lines, other + L * (w2.other - other)
    sections: dict[Key, Section] = {}
    for key in dict.fromkeys(k for k, _ in lines):
        qual, src, first = _source(key)
        rows = [Line(first + i, t) for i, t in enumerate(src)] or [Line(0, "<source unavailable>")]
        sections[key] = Section(qual, key[0], rows)
    for (key, num), ln in lines.items():
        rows = sections[key].lines
        row = next((r for r in rows if r.number == num), None)
        if row is None:  # a line outside the function's block: append it
            rows.append(row := Line(num, linecache.getline(key[0], num)))
        row.flops, row.bytes, row.calls, row.loops, row.shapes = ln.flops, ln.bytes, ln.calls, ln.loops, ln.shapes
    main = [k for k in sections if k[:2] == root]
    rest = sorted((k for k in sections if k not in main), key=lambda k: -sections[k].flops.subs(total.given).sample())
    flops: Poly = other  # summed line by line: = expex.cost's total, before the sizes are put in
    for ln in lines.values():
        flops = flops + ln.flops
    return Profile(total.name, [*total.sizes, *total.given], total.given, flops,
                   [sections[k] for k in main + rest], other, total.notes)  # fmt: skip


_LIGHT = "--heat:255,110,40;--mem:40,120,255"
_DARK = "--heat:255,120,60;--mem:90,150,255"
_CSS = """*{box-sizing:border-box}body{margin:0;padding:0;background:var(--hd);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:20px 16px 48px}h1{font-size:18px;margin:0 0 4px;overflow-wrap:anywhere}
.meta{color:var(--mu);margin-bottom:14px;overflow-wrap:anywhere}.meta b{color:var(--fg)}
.bar{display:flex;gap:8px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
button{font:inherit;border:1px solid var(--ln);background:var(--bg);color:var(--fg);padding:4px 12px;border-radius:6px;cursor:pointer}
button[aria-pressed=true]{border-color:currentColor;font-weight:600}
details{background:var(--bg);border:1px solid var(--ln);border-radius:8px;margin-bottom:12px;overflow:hidden}
summary{padding:8px 12px;cursor:pointer;display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
summary .nm{font-weight:600;font-family:ui-monospace,monospace}summary .fl{color:var(--mu);font-size:12px}summary .pc{margin-left:auto;font-variant-numeric:tabular-nums}
.code{overflow-x:auto;border-top:1px solid var(--ln)}
.row{display:grid;grid-template-columns:3.2em minmax(28em,1fr) 17em;font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;min-width:min-content}
.row>*{padding:0 8px;white-space:pre}.no{color:var(--mu);text-align:right;user-select:none}
.src{background:rgba(var(--heat),var(--a,0))}.mem .src{background:rgba(var(--mem),var(--b,0))}
#s:not(.mem) .vb,.mem .vf{display:none}.cost b{font-weight:500}.cost{color:var(--mu);text-align:right;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis}.hot .cost{color:var(--fg)}
.loop{font-size:11px;border:1px solid var(--ln);border-radius:4px;padding:0 4px;margin-right:6px;color:var(--mu)}
#tip{position:fixed;pointer-events:none;background:var(--tip);color:var(--tipfg);padding:8px 10px;border-radius:6px;font:12px/1.5 ui-monospace,monospace;max-width:min(520px,calc(100vw - 24px));white-space:pre-wrap;display:none;z-index:9}
@media (max-width:700px){.row{grid-template-columns:2.6em minmax(24em,1fr) 10em}.cost .f{display:none}
.cost{position:sticky;right:0;background:var(--bg);box-shadow:-6px 0 8px -6px rgba(0,0,0,.25)}}"""
_BODY = """<main><h1 id="t"></h1><div class="meta" id="m"></div>
<div class="bar"><span>Heat by</span><button id="bf" aria-pressed="true">FLOPs</button><button id="bm" aria-pressed="false">bytes written</button></div>
<div id="s"></div></main><div id="tip"></div><script>
const D=%DATA%,$=id=>document.getElementById(id),pct=x=>(100*x).toFixed(x<0.001?2:1)+"%";
$("t").textContent=D.title;$("m").append("sizes: "+(D.sizes||"none")+" · total ",Object.assign(document.createElement("b"),{textContent:D.total[1]||D.total[0]})," · "+(D.bytes[1]||D.bytes[0])+" written · ↳ = in the functions called from the line");
const tip=$("tip"),root=$("s");let mf=Math.max(...D.sections.flatMap(s=>s.rows.map(r=>r.fs)),1e-9),mb=Math.max(...D.sections.flatMap(s=>s.rows.map(r=>r.bs)),1e-9);
D.sections.forEach((s,i)=>{const d=document.createElement("details");d.open=i<4;const sm=document.createElement("summary");
 sm.innerHTML='<span class="nm"></span><span class="fl"></span><span class="pc"></span>';sm.children[0].textContent=s.name;sm.children[1].textContent=s.file;sm.children[2].innerHTML='<span class="vf"></span><span class="vb"></span>';sm.children[2].children[0].textContent=pct(s.share)+" of FLOPs";sm.children[2].children[1].textContent=pct(s.bshare)+" of bytes";
 const c=document.createElement("div");c.className="code";d.append(sm,c);
 s.rows.forEach(r=>{const e=document.createElement("div");e.className="row"+(r.hot?" hot":"");
  e.style.setProperty("--a",Math.sqrt(r.fs/mf)*0.55);e.style.setProperty("--b",Math.sqrt(r.bs/mb)*0.55);
  const n=document.createElement("span");n.className="no";n.textContent=r.n;const t=document.createElement("span");t.className="src";t.textContent=r.src||" ";
  const k=document.createElement("span");k.className="cost";
  if(r.hot){if(r.loops.length){const l=document.createElement("span");l.className="loop";l.textContent="×"+r.loops.join("·");k.append(l)}
   const cell=(cls,num,txt)=>{const v=document.createElement("span");v.className=cls;const f=document.createElement("span");f.className="f";f.textContent=num?num+"  ":"";
    const q=document.createElement("b");q.textContent=txt;v.append(f,q);k.append(v)};
   cell("vf",r.fs?r.f[1]:"",r.fs>=5e-4?pct(r.fs):r.cs>=5e-4?"↳ "+pct(r.cs):"");cell("vb",r.bs?r.b[1]:"",r.bs>=5e-4?pct(r.bs):"");
   e.dataset.tip=[r.shapes.length?"produces  "+r.shapes.join(", "):"","FLOPs     "+r.f[0]+(r.f[1]?"  = "+r.f[1]:"")+"  ("+pct(r.fs)+")",r.c?"calls     "+r.c[0]+(r.c[1]?"  = "+r.c[1]:"")+"  ("+pct(r.cs)+")":"","bytes     "+r.b[0]+(r.b[1]?"  = "+r.b[1]:""),r.loops.length?"in a loop: ×"+r.loops.join(" × "):""].filter(Boolean).join("\\n")}
  e.append(n,t,k);c.append(e)});root.append(d)});
const show=(e,x,y)=>{tip.textContent=e.dataset.tip;tip.style.display="block";const w=tip.offsetWidth,h=tip.offsetHeight;tip.style.left=Math.max(8,Math.min(x+14,innerWidth-w-8))+"px";tip.style.top=(y+h+20>innerHeight?y-h-10:y+16)+"px"};
root.addEventListener("pointermove",ev=>{const e=ev.target.closest(".row[data-tip]");e?show(e,ev.clientX,ev.clientY):tip.style.display="none"});
root.addEventListener("pointerdown",ev=>{const e=ev.target.closest(".row[data-tip]");e&&show(e,ev.clientX,ev.clientY)});
root.addEventListener("pointerleave",()=>tip.style.display="none");
const mode=m=>{root.classList.toggle("mem",m);$("bf").setAttribute("aria-pressed",!m);$("bm").setAttribute("aria-pressed",m)};
$("bf").onclick=()=>mode(false);$("bm").onclick=()=>mode(true);
</script>"""

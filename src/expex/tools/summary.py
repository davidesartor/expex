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
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.random as jr

from ..annotations.dims import DConst, DExpr, DName, Env
from ..annotations.errors import coded, strip_see
from ..annotations.spec import ArraySpec, NoneSpec, SeqSpec, StructSpec, UnionSpec, fixed_sizes, join, parse_owner
from .. import function
from ..function import Fn, unwrap
from ..modules import _all_fields, build
from . import pages
from .cost import Poly, arrays, cost, si
from .trace import resolve


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
    fixed: dict[str, int] = field(default_factory=dict)  # named fixed sizes: Rows in i32[Rows:6, Cols:7]

    def _cells(self, r: Row) -> list[str]:
        fmt = lambda p: "" if p is None else p.format(self.order)
        flops = "" if r.flops_value is None else si(r.flops_value)
        return [r.label, f"× {r.count}" if r.count > 1 else "", f"{r.params:,}", fmt(r.formula),
                si(r.nbytes, "B"), flops, fmt(r.flops)]  # fmt: skip

    def __str__(self) -> str:
        head = ["module", "class", "", "params", "formula", "bytes", "flops", "flops formula"]
        body = [["  " * r.depth + r.path.rsplit(".", 1)[-1], *self._cells(r)] for r in self.rows]
        foot = ["total", *self._cells(self.total)]
        w = [max(len(x[i]) for x in [head, *body, foot]) for i in range(len(head))]
        right = {2, 3, 5, 6}
        line = lambda x: "  ".join(c.rjust(w[i]) if i in right else c.ljust(w[i]) for i, c in enumerate(x)).rstrip()  # fmt: skip
        rule = "─" * len(line(head))
        sizes = ", ".join([f"{k}={v}" for k, v in self.sizes.items()] + [f"{k} (= {v})" for k, v in self.fixed.items()])
        out = [f"{self.name}   ({sizes})", line(head), rule, *map(line, body), rule, line(foot)]
        return "\n".join(out + [f"note: {n}" for n in self.notes])

    __repr__ = __str__

    def _repr_html_(self) -> str:
        total = max(self.total.params, 1)

        def tr(i: Any, r: Row) -> str:
            c = self._cells(r)
            share = r.params * r.weight / total if i != "" else 0
            kids = any(x.parent == i for x in self.rows)
            name = html.escape(r.path.rsplit(".", 1)[-1] if i != "" else "total")
            tip = lambda s: f' title="{html.escape(s)}"' if s else ""
            ptip = f"{c[3]}" + (f"\n{r.weight} × {r.params:,} = {r.params * r.weight:,}" if r.weight > 1 else "") + (f"\n{share:.1%} of parameters" if share else "")  # fmt: skip
            return (
                f'<tr data-i="{i}" data-p="{"" if r.parent < 0 else r.parent}" data-path="{html.escape(r.path if i != "" else "")}" data-params="{r.params * r.weight}" '
                f'data-flops="{r.flops_value if r.flops_value is not None else -1}"><td class="m" '
                f'style="--d:{r.depth}" title="{html.escape(r.path)}">{pages.toggle(kids)}{name}</td><td class="c">{html.escape(c[0])}'
                f'<span class="x">{html.escape(c[1])}</span></td><td class="n bar" style="--s:{share:.4f}"{tip(ptip)}>'
                f'{c[2]}</td><td class="f">{html.escape(c[3])}</td><td class="n b">{c[4]}</td>'
                f'<td class="n"{tip(c[6])}>{c[5] or ("—" if not c[6] else "ƒ")}</td><td class="f">{html.escape(c[6])}</td></tr>'
            )  # fmt: skip

        sizes = " ".join(
            [f"<span>{html.escape(k)}={v}</span>" for k, v in self.sizes.items()]
            + [f"<span>{html.escape(k)} (= {v})</span>" for k, v in self.fixed.items()]
        )
        body = "".join(tr(i, r) for i, r in enumerate(self.rows))
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in self.notes)
        return pages.table(pages.uid("txs"), _CSS, (
            f'<div class="hd"><b>{html.escape(self.name)}</b> {sizes}</div><div class="sc"><table><thead><tr>'
            '<th>module</th><th>class</th><th data-k="params" class="n" title="parameters (click to sort)">params</th>'
            '<th class="f">formula</th><th class="n b">bytes</th><th data-k="flops" class="n" title="FLOPs of one call of __call__ (click to sort)">'
            'FLOPs</th><th class="f">formula</th></tr></thead>'
            f'<tbody>{body}</tbody><tfoot>{tr("", self.total)}</tfoot></table></div><ul>{notes}</ul>'
        ))  # fmt: skip

    def save(self, path: str | pathlib.Path) -> None:
        """Write the HTML table as a standalone page."""
        pathlib.Path(path).write_text(pages.document(f"{self.name} summary", self._repr_html_()), encoding="utf-8")


def summary(model: Any, config: Any = None, /, **sizes: int) -> Summary:
    """A table of a model's modules: parameters, bytes and FLOPs, as numbers and formulas.

    ``model`` is an ``@expex.struct`` instance, or a struct class with the sizes
    building it needs (``summary(Transformer, Vocab=50, D=32, F=64, blocks=4)``,
    built under ``jax.eval_shape``, so nothing is allocated). Other keywords
    (``T=128``) are sizes of the ``__call__`` methods, used for FLOP counts.
    ``config``: the same in a config object (``summary(Transformer, cfg)``);
    entries the summary does not use are ignored, and keywords override it.
    """
    if config is not None:
        from ..modules import model_config

        sizes = model_config(model, config, sizes, "expex.summary")
    if isinstance(model, type):
        spec = parse_owner(model)
        accepts = {p.__name__ for p in getattr(model, "__type_params__", ())} | _all_fields(spec)
        names: set[str] = set()
        spec.names(names)
        accepts |= {n for n in names if n.isidentifier()}
        built = {k: v for k, v in sizes.items() if k in accepts}
        sizes = {k: v for k, v in sizes.items() if k not in accepts}
        model = jax.eval_shape(lambda: build(model, jr.key(0), **built))
    cls = type(model)
    spec = parse_owner(cls)
    if not isinstance(spec, StructSpec):
        raise coded("EX705", TypeError(f"expex.summary: {cls.__name__} is not a @expex.struct with array fields"))
    env = Env()
    spec.match(model, cls.__name__, env)
    known = {k: int(v) for k, v in env.dims.items() if isinstance(v, int) and k.isidentifier()}
    rows: list[Row] = []
    lists: dict[str, int] = {}
    s = _Walk(rows, lists, known, sizes, [], seen=_call_sizes(model, sizes))
    s.fields(spec, model, "", 0, -1, 1)
    if lists:
        known["L"] = max(lists.values())
        s.notes.append(f"L = len({', '.join(lists)})")
    params = sum(r.params for r in rows if r.parent < 0 for _ in range(r.count))  # top-level rows
    if (hidden := sum(a.size for a in arrays(model)) - params) > 0:  # a layer from another library, say
        s.notes.append(f"{hidden:,} parameters in parts without array annotations are not counted")
    formula = _sum([None if r.formula is None else r.formula * (Poly.of("L") if r.count > 1 else 1) for r in rows if r.parent < 0])  # fmt: skip
    nbytes = sum(r.nbytes * r.count for r in rows if r.parent < 0)
    total = Row("total", 0, cls.__name__, 1, params, formula, nbytes)
    s.flops(total, cls, model, {})
    if s.fallback:
        s.notes.append(f"FLOP formulas of {', '.join(s.fallback)} are in the call's sizes only: static fields differ from the class defaults")  # fmt: skip
    used = set().union(*(r.flops.names() for r in [*rows, total] if r.flops is not None))
    if unknown := set(sizes) - used:
        raise coded("EX706", TypeError(f"expex.summary: unknown size {', '.join(sorted(unknown))} (the calls use {', '.join(sorted(used)) or 'none'})"))  # fmt: skip
    order = [p.__name__ for p in getattr(cls, "__type_params__", ())]
    calls = sorted((used | set(sizes)) - set(known) - {"L"} - set(order))  # T first, as expex.cost does
    order = list(dict.fromkeys([*calls, *order, *sorted(known.keys() - {"L"}), "L"]))
    shown = {k: v for k, v in sorted({**known, **sizes}.items(), key=lambda kv: order.index(kv[0]))}
    return Summary(cls.__name__, rows, total, shown, order, s.notes, fixed_sizes(spec))


@dataclass
class _Walk:
    rows: list[Row]
    lists: dict[str, int]
    known: dict[str, int]
    given: dict[str, int]
    notes: list[str]
    fallback: list[str] = field(default_factory=list)  # rows whose class formula does not apply
    seen: dict[type, dict[str, int]] = field(default_factory=dict)  # sizes of each class's call, in the model's call
    _own: dict[type, Poly | None] = field(default_factory=dict)

    def fields(self, spec: StructSpec, obj: Any, path: str, depth: int, parent: int, weight: int) -> None:
        for name, s in spec.fields.items():
            value, here = getattr(obj, name), join(path, name)
            if (s := _pick(s, value)) is None:
                continue
            n = 1
            if isinstance(s, SeqSpec) and isinstance(value, (list, tuple)):
                if not value:
                    continue
                s, value, n, here = s.elem, value[0], len(value), here + "[*]"
                self.lists[here.removesuffix("[*]")] = n
            leaves = arrays(value)
            row = Row(here, depth, str(s), n, sum(a.size for a in leaves), _formula(s, value),
                      sum(a.size * a.dtype.itemsize for a in leaves), parent=parent, weight=weight * n)  # fmt: skip
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
                self.notes.append(f"{cls.__name__}.__call__: no cost estimate ({type(e).__name__}: {strip_see(str(e))})")
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
        number = bound.subs({**self.seen.get(cls, {}), **self.given})
        row.flops_value = None if number.value is None else int(number.value)
        if number.value is None and (self.given or self.seen):
            self.notes.append(f"{row.path or 'total'}: no FLOP count without {', '.join(sorted(number.names()))}")


def _call_sizes(model: Any, sizes: dict[str, int]) -> dict[type, dict[str, int]]:
    """The sizes each module's ``__call__`` is called with when the model's is: ``S`` of a cache, say.

    The model's call is traced with the given sizes; a hook matches the
    arguments of every module call it makes. Empty if that call cannot be traced.
    """
    seen: dict[type, dict[str, int]] = {}

    def hook(fn: Any, args: Any, kwargs: Any, run: Any) -> Any:
        if fn._lazy:
            fn._resolve_owner()
        owner = getattr(fn, "_owner_cls", None)
        if owner is not None and getattr(fn.raw, "__name__", "") == "__call__":
            env = Env()
            try:
                bound = fn._sig.bind(*args, **kwargs).arguments
                for name, spec in fn._check_inputs:
                    if name in bound and name != "self":
                        spec.match(bound[name], name, env)
            except Exception:
                pass
            seen.setdefault(owner, {}).update({k: int(v) for k, v in env.dims.items() if isinstance(v, int)})
        return run(args, kwargs)

    if not isinstance(unwrap(getattr(type(model), "__call__", None)), Fn):
        return seen
    token = function.CALL_HOOK.set(hook)
    try:
        resolve(model.__call__, dict(sizes)).trace(preset={k: int(v) for k, v in sizes.items()})
    except Exception:
        return {}
    finally:
        function.CALL_HOOK.reset(token)
    return seen


def _pick(spec: Any, value: Any) -> Any:
    """The spec that describes ``value`` (the matching option of ``X | None``), or None to skip it."""
    if isinstance(spec, UnionSpec):
        spec = spec.option(value)
    return spec if spec.leafy and not isinstance(spec, NoneSpec) else None


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


_CSS = """@container (max-width:720px){#ID .f{display:none}#ID th,#ID td{padding:3px 6px}}@container (max-width:480px){#ID .b{display:none}#ID .c{white-space:normal}#ID .m{padding-left:calc(2px + var(--d)*10px)}}"""

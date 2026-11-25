"""``expex.debug.watch``: the activations and gradients of every typed call in one step.

::

    >>> w = watch(lambda m: m.loss.vectorize(batch).mean(), model)
    >>> print(w)                    # one row per call, in call order, flagged rows marked
    >>> w.rows[2].grad["rms"]       # d loss / d (that call's output), as statistics
    >>> w.save("watch.html")        # collapsible, sortable, RMS per layer as bars

``watch(loss_fn, *args, **kwargs)`` runs one forward and backward pass of
``loss_fn(*args, **kwargs)`` under ``jax.jit``. It returns a scalar loss, or
a tuple ``(loss, aux)`` whose ``aux`` is kept as ``Watch.aux``. Every
``@fn`` call made on the way (functions and struct methods) is a row:

- Named by module path when it is a method of a struct inside the first
  argument (``blocks[3].attn.__call__``), else by function name; a name that
  occurs more than once gets its call index (``Attention.causal#3``).
- Statistics of its output (mean, std, abs-max, RMS over the finite values,
  and the number of NaN and inf), and the same for the gradient of the loss
  with respect to that output. A lifted call (``.vectorize``, ``jax.vmap``) or a
  call inside a staged loop is one row with statistics over every element:
  ``×n`` counts them.
- Flagged when its output or gradient is not finite, when its gradient RMS is
  more than ``grad_ratio`` times above or below the median of the same module
  in the other layers, or when its output RMS grows by more than ``growth``
  from the previous layer.

How: the ``function.CALL_HOOK`` adds a zero scalar (the one differentiated input)
to each output and passes it through an identity with a custom VJP. Its
forward and backward each reduce the array to a few moments in the graph and
send them to the host with ``jax.debug.callback``: under ``jit``, once per
iteration of a ``scan``, and once per call under ``vmap`` / ``.vectorize`` (a
``custom_vmap`` rule merges the moments of the batch in the graph). As every
output depends on the zero, the backward pass reaches every call.

Layers: loops over lists of modules are unrolled while watching
(``config.lower_loops=False``), so ``blocks[0]``, ``blocks[1]`` ... are each
traced and reported, with their paths. Tracing and compiling then take time
in proportion to the depth, and loops over ranges and arrays are unrolled too.
``unroll=False`` keeps loops staged: a call inside a loop is then one row with
statistics over all iterations (named by function, since a layer sliced out
of a scan has no path).

JAX's caches are cleared before and after, so that functions compiled earlier
are traced with the hook, and later calls do not reuse those traces: they
compile again on their next call.
"""

import contextlib
import html
import math
import re
import statistics
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap

from .. import function
from ..annotations.dims import fmt_dim, short_dtype
from ..annotations.errors import coded
from . import pages

@dataclass
class Row:
    """One typed call. ``out`` / ``grad`` hold ``mean std absmax rms nan inf`` (grad None if never reached)."""

    name: str  #: ``blocks[3].attn.__call__``, or ``Attention.causal#3``
    depth: int  #: nesting: calls made inside the previous shallower row
    shape: str  #: the output, per call (per element of a ``.vectorize``)
    count: int  #: outputs the statistics cover: batch elements × calls × loop iterations
    out: dict[str, float]
    grad: dict[str, float] | None
    path: str | None = None  #: ``blocks[3].attn``: the module, when found in the first argument
    group: str = ""  #: identifies a module's rows across layers (list indices as ``[*]``)
    flags: list[str] = field(default_factory=list)


def _moments(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """(size, finite, nan, inf) and (mean, sum of squared deviations, abs-max) of the finite values."""
    x = x.astype(jnp.float32)
    fin = jnp.isfinite(x)
    n = fin.sum()
    mean = jnp.where(fin, x, 0).sum() / jnp.maximum(n, 1)
    m2 = jnp.where(fin, (x - mean) ** 2, 0).sum()
    top = jnp.max(jnp.where(fin, jnp.abs(x), 0), initial=0.0)
    counts = jnp.stack([jnp.asarray(x.size), n, jnp.isnan(x).sum(), jnp.isinf(x).sum()]).astype(jnp.int32)
    return counts, jnp.stack([mean, m2, top])


def _merge(c: jax.Array, f: jax.Array) -> tuple[jax.Array, jax.Array]:
    """The moments of a batch of parts (leading axis), as those of one."""
    n = c[:, 1].astype(jnp.float32)
    mean = (n * f[:, 0]).sum() / jnp.maximum(n.sum(), 1)
    m2 = (f[:, 1] + n * (f[:, 0] - mean) ** 2).sum()
    return c.sum(0), jnp.stack([mean, m2, f[:, 2].max()])


def _send(key: tuple["_Hook", int], kind: str, x: jax.Array) -> None:
    """Report ``x``'s moments to the host: once per call, or per loop iteration, not per vmapped element."""

    @custom_vmap
    def report(c: jax.Array, f: jax.Array) -> jax.Array:
        jax.debug.callback(partial(key[0].record, key[1], kind), c, f)
        return jnp.zeros((), jnp.int32)

    @report.def_vmap
    def _(size: int, batched: list[bool], c: jax.Array, f: jax.Array) -> tuple[jax.Array, bool]:
        c, f = (v if b else jnp.broadcast_to(v, (size, *v.shape)) for v, b in zip((c, f), batched))
        return report(*_merge(c, f)), False

    report(*_moments(x))


def _tap_fwd(key: tuple["_Hook", int], x: jax.Array) -> tuple[jax.Array, None]:
    _send(key, "out", x)
    return x, None


def _tap_bwd(key: tuple["_Hook", int], _: None, g: jax.Array) -> tuple[jax.Array]:
    _send(key, "grad", g)
    return (g,)


_tap = jax.custom_vjp(lambda key, x: _tap_fwd(key, x)[0], nondiff_argnums=(0,))
_tap.defvjp(_tap_fwd, _tap_bwd)


def _floating(x: Any) -> bool:
    return hasattr(x, "dtype") and hasattr(x, "shape") and jnp.issubdtype(x.dtype, jnp.floating)


def _brief(out: Any) -> str:
    ts = [short_dtype(x.dtype) + (f"[{', '.join(map(fmt_dim, x.shape))}]" if x.shape else "")
          for x in jax.tree.leaves(out) if _floating(x)]  # fmt: skip
    return ts[0] if len(ts) == 1 else f"({', '.join(ts)})" if ts else ""


class _Hook:
    """Installed as ``function.CALL_HOOK``: one row per call, its output tapped."""

    def __init__(self, prev: Callable[..., Any] | None):
        self.prev, self.zero, self.depth = prev, None, 0
        self.paths: dict[int, str] = {}  # id(leaf of the first argument) -> ".blocks[0].attn.wq"
        self.calls: list[dict[str, Any]] = []
        self.acc: dict[tuple[int, str], list[float]] = {}  # size, finite, mean, m2, absmax, nan, inf

    def record(self, row: int, kind: str, counts: Any, floats: Any) -> None:
        size, n, nan, inf = (float(c) for c in counts)
        mean, m2, top = (float(f) for f in floats)
        a = self.acc.setdefault((row, kind), [0.0] * 7)
        if tot := a[1] + n:  # Chan et al.: merge (n, mean, m2) of two parts
            d = mean - a[2]
            a[2], a[3] = a[2] + d * n / tot, a[3] + m2 + d * d * a[1] * n / tot
        a[0], a[1], a[4], a[5], a[6] = a[0] + size, tot, max(a[4], top), a[5] + nan, a[6] + inf

    def name(self, fn: Any, args: tuple[Any, ...]) -> tuple[str, str | None]:
        """``(blocks[0].attn.__call__, blocks[0].attn)``, or ``(Attention.causal, None)``."""
        short = fn.name.rsplit("<locals>.", 1)[-1]
        if next(iter(fn.params), None) != "self" or not args:
            return short, None
        leaves = jax.tree_util.tree_flatten_with_path(args[0])[0]
        full = self.paths.get(id(leaves[0][1])) if leaves else None
        rel = jax.tree_util.keystr(leaves[0][0]) if leaves else ""
        if full is None or not full.endswith(rel) or full == rel:
            return short, None  # the root, or a layer sliced out of a staged loop
        path = full[: len(full) - len(rel)].lstrip(".")
        return path + "." + short.rsplit(".", 1)[-1], path

    def __call__(self, fn: Any, args: Any, kwargs: Any, run: Any) -> Any:
        row = len(self.calls)
        name, path = self.name(fn, args)
        self.calls.append(dict(name=name, path=path, depth=self.depth, shape="", size=1))
        inner = run if self.prev is None else (lambda a, k: self.prev(fn, a, k, run))  # type: ignore[misc]
        self.depth += 1
        try:
            out = inner(args, kwargs)
        finally:
            self.depth -= 1
        leaves, tree = jax.tree.flatten(out)
        floats = [x for x in leaves if _floating(x)]
        self.calls[row].update(shape=_brief(out), size=max(1, sum(math.prod(x.shape) for x in floats)))
        z: Any = self.zero
        leaves = [_tap((self, row), x + z.astype(x.dtype)) if _floating(x) else x for x in leaves]
        return jax.tree.unflatten(tree, leaves)


def _stats(a: list[float]) -> dict[str, float]:
    n, mean = a[1], a[2] if a[1] else math.nan
    var = a[3] / n if n else math.nan
    return dict(mean=mean, std=math.sqrt(var), absmax=a[4], rms=math.sqrt(var + mean * mean), nan=a[5], inf=a[6])


def _rows(hook: _Hook, grad_ratio: float, growth: float) -> list[Row]:
    keep = [(i, c) for i, c in enumerate(hook.calls) if (i, "out") in hook.acc]  # traced, but never ran: dropped
    total = {n: sum(c["name"] == n for _, c in keep) for n in {c["name"] for _, c in keep}}
    seen: dict[str, int] = {}
    rows: list[Row] = []
    parents: list[tuple[int, str, int]] = []  # (depth, group, index) of the enclosing rows
    local: dict[tuple[int, str], int] = {}
    for i, c in keep:
        out, g = hook.acc[(i, "out")], hook.acc.get((i, "grad"))
        k = seen[c["name"]] = seen.get(c["name"], -1) + 1
        while parents and parents[-1][0] >= c["depth"]:
            parents.pop()
        up, at = parents[-1][1:] if parents else ("", -1)
        j = local[(at, c["name"])] = local.get((at, c["name"]), -1) + 1  # which call of it in the enclosing one
        group = f"{up}/{re.sub(r'\[\d+\]', '[*]', c['name'])}#{j}"
        parents.append((c["depth"], group, i))
        name = c["name"] + (f"#{k}" if total[c["name"]] > 1 else "")
        rows.append(Row(name, c["depth"], c["shape"], round(out[0] / c["size"]), _stats(out),
                        _stats(g) if g else None, c["path"], group))  # fmt: skip
    for r in rows:
        for what, s in (("output", r.out), ("gradient", r.grad)):
            if s and (s["nan"] or s["inf"]):
                r.flags.append(f"{what} not finite: {int(s['nan'])} NaN, {int(s['inf'])} inf")
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.group, []).append(r)
    for layers in (g for g in groups.values() if len(g) > 1):
        grads = [r.grad["rms"] for r in layers if r.grad]  # none when the loss does not reach these calls
        med = statistics.median(grads) if grads else 0.0
        for prev, r in zip([None, *layers], layers):
            if r.grad and med > 0 and r.grad["rms"] > grad_ratio * med:
                r.flags.append(f"gradient exploding: RMS {r.grad['rms'] / med:.2g}× the layers' median")
            elif r.grad and med > 0 and r.grad["rms"] * grad_ratio < med:
                r.flags.append(f"gradient vanishing: RMS 1/{med / max(r.grad['rms'], 1e-300):.2g} of the layers' median")
            if prev and prev.out["rms"] > 0 and r.out["rms"] > growth * prev.out["rms"]:
                r.flags.append(f"activations growing: RMS ×{r.out['rms'] / prev.out['rms']:.2g} from {prev.name}")
    return rows


def _num(x: float) -> str:
    return "—" if x != x else f"{x:.3g}"


def _bad(s: dict[str, float] | None) -> str:
    return f"{int(s['nan'])}/{int(s['inf'])}" if s and (s["nan"] or s["inf"]) else ""


#: (header, class, sort key) of the page's columns; the text table leaves out "layers"
_COLS = [("call", "m", ""), ("shape", "c", ""), ("×n", "n", "n"), ("mean", "n o", ""), ("std", "n o", ""),
         ("|max|", "n", "max"), ("rms", "n", "rms"), ("layers", "s", ""), ("nan/inf", "n", ""),
         ("mean", "n o g", ""), ("std", "n o", ""), ("|max|", "n", "gmax"), ("rms", "n", "grms"),
         ("layers", "s", ""), ("nan/inf", "n", ""), ("flags", "f", "f")]  # fmt: skip
_STATS = ("mean", "std", "absmax", "rms")


class Watch:
    """The rows of :func:`watch`. ``str()`` is a text table, ``save(path)`` / notebooks a page."""

    def __init__(self, name: str, loss: float | None, aux: Any, rows: list[Row], options: dict[str, Any]):
        self.name, self.loss, self.aux, self.options = name, loss, aux, options
        self.rows = rows  #: one :class:`Row` per call, in call order

    @property
    def flagged(self) -> list[Row]:
        return [r for r in self.rows if r.flags]

    def _cells(self, r: Row) -> list[str]:
        """The columns from shape to the gradient's nan/inf ("layers" empty)."""
        g = [_num(r.grad[k]) for k in _STATS] if r.grad else ["—"] * 4
        return [r.shape, f"{r.count:,}", *(_num(r.out[k]) for k in _STATS), "", _bad(r.out), *g, "", _bad(r.grad)]

    def _title(self) -> str:
        loss = "forward only" if self.loss is None else f"loss {self.loss:.6g}"  # None: built by expex.inspect from a call
        return (f"watch {self.name}: {loss}, {len(self.rows)} calls, {len(self.flagged)} flagged  "
                f"({', '.join(f'{k}={v}' for k, v in self.options.items())})")  # fmt: skip

    def __str__(self) -> str:
        keep = [i for i, c in enumerate(_COLS) if c[0] != "layers"]
        table = [["", *(_COLS[i][0] for i in keep)]] + [
            ["!" if r.flags else "", *([" " * 2 * r.depth + r.name, *self._cells(r), "; ".join(r.flags)][i] for i in keep)]
            for r in self.rows
        ]  # fmt: skip
        w = [max(len(t[i]) for t in table) for i in range(len(table[0]))]
        line = lambda a, b, t: t.center(sum(w[a:b]) + 2 * (b - a - 1), "─")  # noqa: E731
        out = [" " * (sum(w[:4]) + 8) + line(4, 9, "output") + "  " + line(9, 14, "d loss / d output")]
        for t in table:
            out.append("  ".join(c.ljust(w[i]) if i in (0, 1, 2, 14) else c.rjust(w[i]) for i, c in enumerate(t)).rstrip())
        return self._title() + "\n" + "\n".join(out)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        groups, parents, body = dict[str, list[Row]](), list[int](), []
        for r in self.rows:
            groups.setdefault(r.group, []).append(r)

        def spark(r: Row, kind: str) -> str:
            """The RMS of this module in every layer, log scale, this layer's bar highlighted."""
            g = groups[r.group]
            vals = [(x.out if kind == "out" else x.grad or {}).get("rms", math.nan) for x in g]
            logs = [math.log10(v) if v > 0 and math.isfinite(v) else None for v in vals]
            ok = [v for v in logs if v is not None]
            if len(g) < 2 or not ok:
                return ""
            lo, span = min(ok), (max(ok) - min(ok)) or 1.0
            hs = [0.0 if v is None else 2 + 16 * (v - lo) / span for v in logs]
            bars = "".join(f'<rect x="{j * 5}" y="{18 - h:.1f}" width="4" height="{h:.1f}"{' class="me"' if x is r else ''}/>'
                           for j, (x, h) in enumerate(zip(g, hs)))  # fmt: skip
            tip = html.escape("\n".join(f"{x.name}: {_num(v)}" for x, v in zip(g, vals)))
            return f'<svg width="{len(g) * 5}" height="18" role="img"><title>{tip}</title>{bars}</svg>'

        num = lambda s, k: s[k] if s and math.isfinite(s[k]) else -1  # noqa: E731
        for i, r in enumerate(self.rows):
            while parents and self.rows[parents[-1]].depth >= r.depth:
                parents.pop()
            kids = i + 1 < len(self.rows) and self.rows[i + 1].depth > r.depth
            data = dict(i=i, p=parents[-1] if parents else "", rms=num(r.out, "rms"), grms=num(r.grad, "rms"),
                        max=num(r.out, "absmax"), gmax=num(r.grad, "absmax"), n=r.count, f=len(r.flags))  # fmt: skip
            up = self.rows[parents[-1]].path if parents else None
            label = r.name.removeprefix(f"{up}.") if up and r.path else r.name  # split, under blocks[0].attn
            cells = [html.escape(c) for c in self._cells(r)]
            cells[6], cells[12] = spark(r, "out"), spark(r, "grad")
            body.append(
                "<tr " + " ".join(f'data-{k}="{v}"' for k, v in data.items())
                + (f' data-path="{html.escape(r.path)}"' if r.path is not None else "") + (' class="flag">' if r.flags else ">")
                + f'<td class="m" style="--d:{r.depth}" title="{html.escape(r.name)}">{pages.toggle(kids)}{html.escape(label)}</td>'
                + "".join(f'<td class="{c[1]}">{x}</td>' for c, x in zip(_COLS[1:3], cells[:2]))
                + '<td class="f">' + "".join(f'<span title="{html.escape(f)}">{f.split(":")[0]}</span>' for f in r.flags)
                + "</td>" + "".join(f'<td class="{c[1]}">{x}</td>' for c, x in zip(_COLS[3:], cells[2:]))
                + "</tr>"
            )  # fmt: skip
            parents.append(i)
        # flags next to the call, where a narrow output shows them without scrolling
        cols = [*_COLS[:3], _COLS[-1], *_COLS[3:-1]]
        head = "".join(f'<th class="{c}"' + (f' data-k="{k}"' if k else "") + f">{t}</th>" for t, c, k in cols)
        return pages.table(pages.uid("txw"), _CSS, light=_LIGHT, dark=_DARK, markup=(
            f'<div class="hd"><b>{html.escape(self._title())}</b></div><div class="sc"><table><thead><tr>'
            '<th colspan="4"></th><th colspan="6" class="gh w">output</th><th colspan="4" class="gh nw">output</th>'
            '<th colspan="6" class="gh g w">d loss / d output</th><th colspan="4" class="gh g nw">d loss / d output</th>'
            f"</tr><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
        ))

    def save(self, path: str | Path) -> Path:
        """Write the table as a standalone page (inline CSS and JS)."""
        path = Path(path)
        path.write_text(pages.document(f"watch {self.name}", self._repr_html_()), encoding="utf-8")
        return path


#: Flagged rows, light and dark.
_LIGHT, _DARK = "--fl:rgba(220,38,38,.11);--fx:#b42318", "--fl:rgba(248,81,73,.16);--fx:#ff7b72"
_CSS = """#ID td{vertical-align:top}#ID tr.flag td{background:var(--fl)}#ID tr.flag .m{box-shadow:inset 3px 0 var(--fx)}
#ID .m,#ID .c{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;white-space:nowrap}
#ID th,#ID td{padding:3px 7px}#ID .f{color:var(--fx);font-size:12px;white-space:normal}
#ID .f span{display:inline-block;white-space:nowrap;border:1px solid;border-radius:9px;padding:0 6px;margin:0 4px 2px 0}#ID .gh{text-align:center;border-bottom:0}
#ID .g{border-left:1px solid var(--ln)}#ID .s{padding:2px 6px 0}#ID svg{display:block}
#ID rect{fill:var(--mu);opacity:.45}#ID rect.me{fill:var(--ac);opacity:1}
#ID th.f{color:var(--mu)}#ID td.f{min-width:6ch}#ID .nw{display:none}
@container (max-width:1100px){#ID .o,#ID .w{display:none}#ID .nw{display:table-cell}#ID .f{display:table-cell}
#ID .m{position:sticky;left:0;z-index:1;background:var(--bg)}#ID tr.flag .m{background:linear-gradient(var(--fl),var(--fl)),var(--bg)}}"""


def watch(
    loss_fn: Callable[..., Any], /, *args: Any, grad_ratio: float = 100.0, growth: float = 10.0,
    unroll: bool = True, **kwargs: Any,
) -> Watch:  # fmt: skip
    """Run one forward and backward pass of ``loss_fn(*args, **kwargs)`` and report every typed call.

    ``loss_fn`` returns a scalar, or ``(loss, aux)``. ``grad_ratio`` and
    ``growth`` set when a row is flagged; ``unroll=False`` keeps loops over
    layers staged (one row for all layers). These three names are not passed
    on to ``loss_fn``. See the module docstring.
    """
    leaves, tree = jax.tree.flatten((args, kwargs))
    dyn = [i for i, x in enumerate(leaves) if isinstance(x, (jax.Array, np.ndarray))]
    hook = _Hook(function.CALL_HOOK.get())
    fname = getattr(loss_fn, "__qualname__", type(loss_fn).__name__).rsplit("<locals>.", 1)[-1]

    def step(zero: jax.Array, values: list[Any]) -> tuple[jax.Array, Any]:
        full = list(leaves)
        for i, v in zip(dyn, values):
            full[i] = v
        a, k = jax.tree.unflatten(tree, full)
        hook.zero = zero  # type: ignore[assignment]
        if a:
            hook.paths = {id(x): jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_flatten_with_path(a[0])[0]}
        out = loss_fn(*a, **k)
        loss, aux = out if isinstance(out, tuple) else (out, None)
        if jnp.shape(loss) != () or not jnp.issubdtype(jnp.result_type(loss), jnp.floating):
            raise coded("EX705", TypeError(f"watch: {fname} must return a float scalar loss, or (loss, aux); it returned {_brief(loss) or type(loss).__name__}"))  # fmt: skip
        return loss, aux

    jax.clear_caches()  # functions jitted earlier must be traced again, with the hook
    token = function.CALL_HOOK.set(hook)
    try:
        with function.config(lower_loops=False) if unroll else contextlib.nullcontext():
            grad_fn = jax.jit(jax.value_and_grad(step, has_aux=True))
            (loss, aux), _ = grad_fn(jnp.zeros((), jnp.float32), [leaves[i] for i in dyn])
            jax.effects_barrier()
    finally:
        function.CALL_HOOK.reset(token)
        jax.clear_caches()  # and later calls must not reuse those traces, taps included
    options = {"grad_ratio": grad_ratio, "growth": growth}
    return Watch(fname, float(loss), aux, _rows(hook, grad_ratio, growth), options)

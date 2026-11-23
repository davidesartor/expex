"""Symbolic cost estimates: FLOPs and parameters as formulas in the size names.

``cost(f)`` traces an ``@fn`` (or a struct method) the way the prover does, on
abstract inputs whose sizes are JAX symbolic dimensions, and walks the jaxpr::

    >>> print(cost(Transformer.__call__))
    Transformer.__call__   (sizes: T, D, F, Vocab, L)
      flops        8·T·D²·L + 4·T·D·F·L + 4·T²·D·L + 31·T²·L + ... + 2·T·D·Vocab + ...
      parameters   4·D²·L + 2·D·F·L + 4·D·L + 2·D·Vocab + 2·D
      largest      f32[4,T,T] (4·T² elements)
      note: L = len(self.blocks)

Keywords choose static parameters (``heads=4``), the length of lists of
modules (``L=12``, or by field name: ``blocks=12``) and sizes (``T=1024``); once every size is known the report
also gives numbers (``= 137 GFLOP``). A bound method (``model.__call__``)
takes its sizes, and its parameter count, from the instance.

What is counted, and what is approximate:

- ``dot_general``: 2 × (output elements) × (contracted size), exact.
  ``conv_general_dilated``: 2 × (output elements) × (kernel window × input features).
- Elementwise operations: one flop per output element, whatever the operation
  (``tanh`` and ``integer_pow`` count as much as ``add``). Reductions: one per
  input element. Data movement (reshape, transpose, broadcast, gather, slices,
  stacking, ...) and random-key bookkeeping are free. Small scalar index
  arithmetic is counted, which shows up as low-order terms.
- ``scan`` multiplies its body by its length, which may be symbolic.
  A ``while`` loop's trip count is unknown: one iteration is counted, and the
  report says so. A ``cond`` counts its most expensive branch.
- ``floordiv(D, k)`` in a shape is treated as ``D/k`` (exact when ``k`` divides
  ``D``, as a size constraint usually says).
- A list of modules of unknown length is traced with 1 and 2 elements and the
  cost extrapolated linearly to ``L`` elements; all lists share the same ``L``.
- Static parameters not given as keywords take the prover's first example
  value (``heads: int`` is 1); the report notes which.
- ``largest`` is the single largest value the trace creates, not peak memory.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from ..annotations.errors import coded
from ..annotations.spec import fn_fixed_sizes
from .trace import Target, resolve, source_of, sub_jaxprs

type Mono = tuple[tuple[str, int], ...]  # ((name, power), ...) sorted by name


class Poly:
    """A polynomial in size names with rational coefficients."""

    def __init__(self, terms: dict[Mono, Fraction] | None = None):
        self.terms = {m: c for m, c in (terms or {}).items() if c}

    @staticmethod
    def of(x: Any) -> Poly:
        """A ``Poly`` from a number, a name, a ``Poly`` or a JAX dimension expression."""
        if isinstance(x, Poly):
            return x
        if isinstance(x, str):
            return Poly({((x, 1),): Fraction(1)})
        if isinstance(x, (int, np.integer, Fraction)):
            return Poly({(): Fraction(x if isinstance(x, Fraction) else int(x))})
        out = Poly()
        for term, coeff in x._sorted_terms:  # JAX's _DimExpr: a sum of products
            out += math.prod((_factor(f) ** k for f, k in term._factors), start=Poly.of(coeff))
        return out

    def __add__(self, o: Any) -> Poly:
        out = dict(self.terms)
        for m, c in Poly.of(o).terms.items():
            out[m] = out.get(m, 0) + c
        return Poly(out)

    def __mul__(self, o: Any) -> Poly:
        out: dict[Mono, Fraction] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in Poly.of(o).terms.items():
                powers = dict(m1)
                for n, k in m2:
                    powers[n] = powers.get(n, 0) + k
                m = tuple(sorted(powers.items()))
                out[m] = out.get(m, 0) + c1 * c2
        return Poly(out)

    __radd__, __rmul__ = __add__, __mul__

    def __sub__(self, o: Any) -> Poly:
        return self + Poly.of(o) * -1

    def __pow__(self, k: int) -> Poly:
        return math.prod([self] * k, start=Poly.of(1))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, (Poly, int, str)) and self.terms == Poly.of(o).terms

    def subs(self, values: dict[str, Any]) -> Poly:
        """Substitute numbers (or polynomials) for some names."""
        return sum(
            (math.prod((Poly.of(values.get(n, n)) ** k for n, k in m), start=Poly.of(c))
             for m, c in self.terms.items()),
            Poly(),
        )  # fmt: skip

    @property
    def value(self) -> Fraction | None:
        """The number, if no names are left."""
        return self.terms.get((), Fraction(0)) if set(self.terms) <= {()} else None

    def names(self) -> set[str]:
        return {n for m in self.terms for n, _ in m}

    def sample(self, at: float = 1000.0) -> float:
        """The value with every name set to ``at``: used to rank costs."""
        return float(sum(c * at ** sum(k for _, k in m) for m, c in self.terms.items()))

    def format(self, order: list[str] | tuple[str, ...] = ()) -> str:
        """Dominant terms first (highest degree, then largest coefficient): ``2·T·D² + T``."""
        rank = {n: (i, n) for i, n in enumerate(order)}
        key = lambda n: rank.get(n, (len(rank), n))
        degree = lambda m: sum(k for _, k in m)
        out = ""
        for m, c in sorted(self.terms.items(), key=lambda t: (-degree(t[0]), -abs(t[1]), [key(n) for n, _ in t[0]])):  # fmt: skip
            names = "·".join(n + _sup(k) for n, k in sorted(m, key=lambda nk: key(nk[0])))
            num, den = abs(c).numerator, abs(c).denominator
            body = "·".join(x for x in (str(num) if num != 1 or not m else "", names) if x)
            out += (" - " if c < 0 else " + ") if out else ("-" if c < 0 else "")
            out += body + (f"/{den}" if den != 1 else "")
        return out or "0"

    __str__ = format

    def __repr__(self) -> str:
        return f"Poly({self.format()})"


def _sup(k: int) -> str:
    return "" if k == 1 else str(k).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def _factor(f: Any) -> Poly:
    if f.var is not None:
        return Poly.of(f.var)
    if f.operation == "floordiv" and (k := Poly.of(f.operands[1]).value) and k > 0:
        return Poly.of(f.operands[0]) * (1 / k)  # exact when k divides the operand
    return Poly.of(str(f))  # mod, max, ...: an opaque name


def _numel(shape: tuple[Any, ...]) -> Poly:
    return math.prod((Poly.of(d) for d in shape), start=Poly.of(1))


_FREE = {
    "reshape", "transpose", "broadcast_in_dim", "convert_element_type", "squeeze",
    "expand_dims", "slice", "dynamic_slice", "dynamic_update_slice", "gather",
    "concatenate", "pad", "iota", "copy", "copy_p", "dim_as_value", "rev",
    "sharding_constraint", "stop_gradient", "bitcast_convert_type", "split", "stack",
}  # fmt: skip
_REDUCE = {
    "reduce_sum", "reduce_max", "reduce_min", "reduce_prod", "reduce_and", "reduce_or",
    "argmax", "argmin", "cumsum", "cumprod", "cummax", "cummin", "cumlogsumexp",
}  # fmt: skip


def flops(jaxpr: Any, notes: list[str]) -> Poly:
    """FLOPs of a jaxpr as a polynomial in its symbolic sizes. See the module docstring."""
    total = Poly()
    for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        name, p = eqn.primitive.name, eqn.params
        inner = [flops(j, notes) for _, j in sub_jaxprs(p)]
        if name == "scan":
            total += Poly.of(p["length"]) * sum(inner, Poly())
        elif name == "cond":
            total += max(inner, key=Poly.sample)
        elif inner:  # while (one iteration), jit, custom_jvp_call, custom_vjp_call, remat, ...
            if name == "while":
                notes.append(f"while loop at {source_of(eqn)}: trip count unknown, one iteration counted")
            total += sum(inner, Poly())
        elif name == "dot_general":
            (contract, _), _ = p["dimension_numbers"]
            lhs = eqn.invars[0].aval.shape
            total += 2 * _numel(eqn.outvars[0].aval.shape) * _numel(tuple(lhs[i] for i in contract))
        elif name == "conv_general_dilated":
            rhs, spec = eqn.invars[1].aval.shape, p["dimension_numbers"].rhs_spec
            total += 2 * _numel(eqn.outvars[0].aval.shape) * _numel(tuple(rhs[i] for i in spec[1:]))
        elif name in _REDUCE:
            total += _numel(eqn.invars[0].aval.shape)
        elif name not in _FREE and not name.startswith("random_"):
            total += sum((_numel(v.aval.shape) for v in eqn.outvars), Poly())
    return total


@dataclass
class Cost:
    """A cost estimate. ``str()`` gives a readable report; the fields are ``Poly``s."""

    name: str
    sizes: list[str]  # the free size names, in display order
    flops: Poly
    parameters: Poly | None = None  # for methods: the elements of `self`'s arrays
    largest: tuple[str, Poly] | None = None  # (type of the largest value, its elements)
    notes: list[str] = field(default_factory=list)
    given: dict[str, int] = field(default_factory=dict)  # sizes given as keywords
    fixed: dict[str, int] = field(default_factory=dict)  # named fixed sizes: R in f32[R:6]

    def __str__(self) -> str:
        sizes = ", ".join(
            self.sizes + [f"{k}={v}" for k, v in self.given.items()] + [f"{k} (= {v})" for k, v in self.fixed.items()]
        )
        out = [f"{self.name}   (sizes: {sizes or 'none'})"]
        out.append(f"  flops        {self.flops.format(self.sizes)}{_num(self.flops, 'FLOP')}")
        if self.parameters is not None:
            out.append(f"  parameters   {self.parameters.format(self.sizes)}{_num(self.parameters, '')}")
        if self.largest is not None:
            out.append(f"  largest      {self.largest[0]} ({self.largest[1].format(self.sizes)} elements)")
        return "\n".join(out + [f"  note: {n}" for n in dict.fromkeys(self.notes)])

    __repr__ = __str__

    def _repr_html_(self) -> str:
        from .display import cost_html

        return cost_html(self)


def si(v: float | Fraction, unit: str = "") -> str:
    """``3.24 GFLOP``, ``25.2 M``, ``64 B``: three significant digits and an SI prefix."""
    x = float(v)
    scale, prefix = next((s, p) for s, p in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, "")) if abs(x) >= s or not p)  # fmt: skip
    return f"{x / scale:.3g} {prefix}{unit}".rstrip()


def _num(p: Poly, unit: str) -> str:
    return "" if (v := p.value) is None else f"   = {si(v, unit)}"


def arrays(tree: Any) -> list[Any]:
    """The arrays (or ``ShapeDtypeStruct``s) in a pytree, random keys left out."""
    leaves = jax.tree.leaves(tree, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct))
    return [x for x in leaves if hasattr(x, "shape") and hasattr(x, "dtype") and not jnp.issubdtype(x.dtype, jax.dtypes.prng_key)]  # fmt: skip


def _count(s: Target, n: int) -> tuple[Poly, Poly | None, tuple[str, Poly] | None, list[str], list[str]]:
    """Trace once with lists of length ``n``: (flops, parameters, largest value, notes, lists)."""
    f = s.f
    jaxpr, arguments, lists = s.trace(n)
    notes: list[str] = []
    total = flops(jaxpr, notes)
    first = next(iter(f.params))
    params = None
    if first in ("self", "cls") and getattr(f, "_owner_cls", None) is not None:
        params = sum((_numel(x.shape) for x in arrays(getattr(arguments[first], "obj", arguments[first]))), Poly())
    values = [v.aval for e in _user_eqns(jaxpr) for v in e.outvars]
    values = [a for a in values if hasattr(a, "shape") and not jnp.issubdtype(a.dtype, jax.dtypes.prng_key)]
    big = max(values, key=lambda a: _numel(a.shape).sample(), default=None)
    largest = None if big is None else (big.str_short(short_dtypes=True), _numel(big.shape))
    return total, params, largest, notes, lists


def _user_eqns(jaxpr: Any) -> Iterator[Any]:
    """The equations of a jaxpr and of its nested ones, but not inside JAX's own helpers.

    A private helper (``jit`` of ``_roll_dynamic``, say) may build values the
    program does not have at concrete sizes: with a symbolic shift, ``jnp.roll``
    concatenates the array with itself.
    """
    for eqn in getattr(jaxpr, "jaxpr", jaxpr).eqns:
        yield eqn
        if not str(eqn.params.get("name", "")).startswith("_"):
            for _, sub in sub_jaxprs(eqn.params):
                yield from _user_eqns(sub)


def cost(target: Any, config: Any = None, /, **sizes: Any) -> Cost:
    """Estimate the FLOPs (and, for a method, the parameter count) of an ``@fn``.

    ``target`` is an ``@fn``, an unbound struct method (``Model.__call__``: the
    parameters are a formula in the class's sizes) or a bound one
    (``model.__call__``: the instance's sizes and parameters). Keywords set
    static parameters (``heads=4``), the length ``L`` of lists of modules (also
    by the list's field name: ``blocks=12``), and sizes (``T=1024``). The counts are estimates: see the module docstring.
    ``config`` gives the same in a config object, as ``Model(key, cfg)`` takes it
    (entries the cost does not use are ignored; keywords override it).
    """
    if config is not None:
        from ..modules import call_config

        target, sizes = call_config(target, config, sizes, "expex.cost")
    given = dict(sizes)
    s = resolve(target, sizes, "estimate the cost of")
    f, sizes, static_kw, notes = s.f, s.sizes, s.static_kw, s.notes

    aliases: set[str] = set()
    fl, params, largest, more, lists = _count(s, 1)
    if lists:  # the cost is linear in the length of the lists: extrapolate from 1 and 2
        fl2, params2, _, _, _ = _count(s, 2)
        fl += (Poly.of("L") - 1) * (fl2 - fl)
        params = None if params is None or params2 is None else params + (Poly.of("L") - 1) * (params2 - params)  # fmt: skip
        notes.append(f"L = len({', '.join(dict.fromkeys(lists))})")
    aliases = {p.rsplit(".", 1)[-1] for p in lists}  # cost(f, blocks=12) for self.blocks
    values = {("L" if k in aliases else k): v for k, v in sizes.items() if k not in static_kw}
    known = fl.names() | (params.names() if params else set()) | (largest[1].names() if largest else set())
    for k in given:
        if k not in static_kw and k not in known | f.input_names | aliases | {"L"}:
            raise coded("EX706", TypeError(f"{f.name}: unknown size or parameter `{k}`"))
    owner = getattr(f, "_owner_cls", None)
    order = [t.__name__ for src in (f.raw, owner) for t in getattr(src, "__type_params__", ())]
    order = [n for n in dict.fromkeys([*order, *sorted(known - {"L"}), "L"]) if n in known and n not in values]  # fmt: skip
    return Cost(
        getattr(f.raw, "__qualname__", f.name),
        order,
        fl.subs(values),
        None if params is None else params.subs(values),
        None if largest is None else (largest[0], largest[1].subs(values)),
        notes + more,
        values,
        fn_fixed_sizes(f),
    )


# Kept for ``_diagram.py`` while it is being rewritten: the trace engine under its old names.
_setup = resolve


def _trace(s: Target, n: int, preset: dict[str, int] | None = None) -> tuple[Any, dict[str, Any], list[str]]:
    return s.trace(n, preset)

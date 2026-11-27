"""The oracle: run a generated program in plain Python and under expex, and compare.

Modes, each against the plain-Python ground truth (``truth``):

- ``eager``: the ``@expex.fn`` called on concrete arrays;
- ``jit``: ``jax.jit`` of it (``n`` static);
- ``batch``: called with an extra leading axis on every array argument, which
  expex maps over by itself, against a Python loop of the truth over the batch;
  ``batch2`` is the second such call, which runs compiled;
- ``vmap``: ``jax.vmap`` of it, the same comparison;
- ``grad``: where the result is a float scalar, ``jax.grad`` against a central
  finite difference of the truth along a random direction (skipped at kinks).

A mode passes when it agrees with the truth to float tolerance, or raises an
error with an expex code (``err.code``). Anything else (a different value, an
error without a code) is a failure.
"""

from __future__ import annotations

import itertools
import linecache
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import expex
from expex import f32, i32
from expex.annotations.errors import code_of

_COUNTER = itertools.count()


def __upd__(arr: Any, index: Any, op: str, value: Any) -> Any:
    """The truth's ``arr[index] op= value``: NumPy's in-place update, on a copy."""
    a = np.array(arr)
    idx = index()
    if isinstance(idx, tuple):
        idx = tuple(np.asarray(i) if isinstance(i, jax.Array) else i for i in idx)
    elif isinstance(idx, jax.Array):
        idx = np.asarray(idx)
    v = np.asarray(value)
    if op == "=":
        a[idx] = v
    elif op == "+":
        a[idx] += v.astype(a.dtype)
    elif op == "-":
        a[idx] -= v.astype(a.dtype)
    elif op == "*":
        a[idx] *= v.astype(a.dtype)
    return jnp.asarray(a)


def namespace() -> dict[str, Any]:
    return {"jnp": jnp, "jax": jax, "expex": expex, "f32": f32, "i32": i32, "__upd__": __upd__, "np": np}


def load(src: str, tag: str = "p") -> Any:
    """Compile ``src`` (one ``def f``) with its source registered, so that expex can read it."""
    fname = f"<fuzz-{tag}-{next(_COUNTER)}>"
    lines = src.splitlines(keepends=True)
    linecache.cache[fname] = (len(src), None, lines, fname)
    ns = namespace()
    exec(compile(src, fname, "exec"), ns)
    return ns["f"]


# ------------------------------------------------------------------ inputs


def inputs(rng: np.random.Generator, N: int, D: int, batch: tuple[int, ...] = ()) -> dict[str, Any]:
    def fl(*shape: int) -> Any:
        return jnp.asarray(rng.integers(-48, 49, size=batch + shape) / 16.0, jnp.float32)

    return {
        "x": fl(N),
        "y": fl(N),
        "m": fl(N, D),
        "s": fl(),
        "k": jnp.asarray(rng.integers(-2, 6, size=batch), jnp.int32),
        "n": int(rng.integers(0, 4)),
    }


def element(args: dict[str, Any], b: int) -> dict[str, Any]:
    return {k: (v if k == "n" else v[b]) for k, v in args.items()}


# ------------------------------------------------------------------ comparing


@dataclass
class Outcome:
    mode: str
    kind: str  # "ok", "coded", "mismatch", "error", "skip"
    detail: str = ""
    code: str | None = None

    @property
    def bad(self) -> bool:
        return self.kind in ("mismatch", "error")

    def signature(self) -> str:
        """What a shrunk program must keep failing with."""
        first = self.detail.splitlines()[0] if self.detail else ""
        if self.kind == "error":
            return f"{self.mode}:error:{first.split(':')[0]}"
        return f"{self.mode}:{self.kind}"


def same(truth: Any, got: Any, exact_dtype: bool = False) -> str | None:
    """None if ``got`` means the same as ``truth``; else why not."""
    try:
        lt, tt = jax.tree.flatten(truth)
        lg, tg = jax.tree.flatten(got)
    except Exception as e:  # pragma: no cover
        return f"cannot flatten: {e}"
    if len(lt) != len(lg):
        return f"structure: {tt} vs {tg}"
    if tt != tg:
        # a list vs a tuple, say
        return f"structure: {tt} vs {tg}"
    for a, b in zip(lt, lg):
        A, B = np.asarray(a), np.asarray(b)
        if A.shape != B.shape:
            return f"shape {A.shape} vs {B.shape}"
        ka, kb = A.dtype.kind, B.dtype.kind
        # A traced `if` gives a variable one dtype both branches can take: an int on one
        # path and a float on the other is a float (expex's documented reconciliation)
        promoted = not exact_dtype and {ka, kb} <= {"b", "i", "u", "f"} and "f" in (ka, kb)
        if ka != kb and not ({ka, kb} <= {"i", "u"}) and not promoted:
            return f"dtype kind {A.dtype} vs {B.dtype} (values {A!r} vs {B!r})"
        # (a Python float against a float32 array is fine: `max(x[0], 1.0)` is jnp.maximum by design)
        if exact_dtype and type(a) is not type(b) and {type(a), type(b)} <= {int, float, bool}:
            return f"type {type(a).__name__} vs {type(b).__name__} ({a!r} vs {b!r})"
        if not np.allclose(A.astype(np.float64), B.astype(np.float64), rtol=1e-5, atol=1e-5, equal_nan=True):
            return f"values {A.tolist()} vs {B.tolist()}"
    return None


#: JAX's own limits, which expex documents rather than works around: (mode, part of the message).
LIMITS = [
    ("grad", "Reverse-mode differentiation does not work for lax.while_loop"),
]


def _err(mode: str, e: BaseException) -> Outcome:
    code = code_of(e)
    msg = f"{type(e).__name__}: {e}"
    if code is not None:
        return Outcome(mode, "coded", msg, code)
    if any(m == mode and text in msg for m, text in LIMITS):
        return Outcome(mode, "limit", msg)
    tb = "".join(traceback.format_exception(e)[-6:])
    return Outcome(mode, "error", msg + "\n" + tb)


# ------------------------------------------------------------------ running


@dataclass
class Report:
    expex_src: str
    truth_src: str
    outcomes: list[Outcome] = field(default_factory=list)
    truth_failed: str | None = None

    @property
    def failures(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.bad]


def _call(f: Any, args: dict[str, Any]) -> Any:
    return f(**args)


def check(
    expex_src: str,
    truth_src: str,
    seed: int,
    modes: tuple[str, ...] = ("eager", "jit", "batch", "vmap", "grad"),
    sizes: list[tuple[int, int]] | None = None,
) -> Report:
    rep = Report(expex_src, truth_src)
    rng = np.random.default_rng(seed)
    try:
        truth_f = load(truth_src, "truth")
    except Exception as e:
        rep.truth_failed = f"compile: {e}"
        return rep
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = load(expex_src, "expex")
            fx = expex.fn(raw)
    except Exception as e:
        rep.outcomes.append(_err("define", e))
        return rep
    if not fx.converted and fx.convert_note not in ("no control flow to convert",):
        rep.outcomes.append(Outcome("define", "error", f"not converted: {fx.convert_note}"))
    if sizes is None:
        sizes = [(int(rng.integers(1, 5)), int(rng.integers(1, 4)))]
    for N, D in sizes:
        args = inputs(rng, N, D)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                want = _call(truth_f, args)
        except Exception as e:
            rep.truth_failed = f"{type(e).__name__}: {e}"
            return rep
        for mode in modes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep.outcomes.append(_mode(mode, fx, truth_f, args, want, rng, N, D))
    return rep


def _mode(mode: str, fx: Any, truth_f: Any, args: dict[str, Any], want: Any, rng: Any, N: int, D: int) -> Outcome:
    try:
        if mode == "eager":
            got = _call(fx, args)
            why = same(want, got, exact_dtype=True)
            return Outcome(mode, "mismatch", why) if why else Outcome(mode, "ok")
        if mode == "jit":
            got = jax.jit(fx, static_argnames=("n",))(**args)
            why = same(want, got)
            return Outcome(mode, "mismatch", why) if why else Outcome(mode, "ok")
        if mode in ("batch", "vmap"):
            B = 2
            bargs = inputs(rng, N, D, (B,))
            bargs["n"] = args["n"]
            wants = []
            for b in range(B):
                try:
                    wants.append(_call(truth_f, element(bargs, b)))
                except Exception:
                    return Outcome(mode, "skip", "truth raised on a batch element")
            try:
                want_b = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *wants)
            except Exception:
                return Outcome(mode, "skip", "the elements' results differ in structure or shape")
            if mode == "batch":
                got = _call(fx, bargs)
                why = same(want_b, got)
                if why:
                    return Outcome(mode, "mismatch", why)
                got2 = _call(fx, bargs)  # the second call runs compiled
                why = same(want_b, got2)
                return Outcome("batch2", "mismatch", why) if why else Outcome(mode, "ok")
            n = bargs.pop("n")
            got = jax.vmap(lambda a: fx(**a, n=n))(bargs)
            why = same(want_b, got)
            return Outcome(mode, "mismatch", why) if why else Outcome(mode, "ok")
        if mode == "grad":
            return _grad(fx, truth_f, args, want, rng)
        raise ValueError(mode)
    except Exception as e:
        return _err(mode, e)


def _scalar(out: Any) -> Any:
    if isinstance(out, tuple):
        out = out[0]
    return out


def _grad(fx: Any, truth_f: Any, args: dict[str, Any], want: Any, rng: Any) -> Outcome:
    w = _scalar(want)
    if np.ndim(w) != 0 or np.asarray(w).dtype.kind != "f" or not np.isfinite(np.asarray(w)):
        return Outcome("grad", "skip", "not a finite float scalar")
    names = list(f for f in ("x", "y", "m", "s"))
    base = {k: v for k, v in args.items() if k not in names}

    def loss(*vals: Any) -> Any:
        return _scalar(fx(**dict(zip(names, vals)), **base))

    g = jax.grad(loss, argnums=tuple(range(len(names))))(*[args[k] for k in names])
    direction = [np.asarray(rng.standard_normal(np.shape(args[k])), np.float32) for k in names]
    gv = float(sum(np.vdot(np.asarray(gi), di) for gi, di in zip(g, direction)))

    def at(t: float) -> float:
        a = dict(args)
        for k, d in zip(names, direction):
            a[k] = jnp.asarray(np.asarray(args[k], np.float64) + t * d, jnp.float32)
        return float(np.asarray(_scalar(truth_f(**a)), np.float64))

    def fd(h: float) -> float:
        return (at(h) - at(-h)) / (2 * h)

    try:
        d1, d2 = fd(1 / 64), fd(1 / 256)
        f0, h = at(0.0), 1 / 256
        right, left = (at(h) - f0) / h, (f0 - at(-h)) / h  # one-sided: differ at a kink (ties of max, abs at 0)
    except Exception:
        return Outcome("grad", "skip", "truth raised near the input")
    smooth = all(np.isfinite(v) for v in (d1, d2, right, left))
    if not smooth or abs(d1 - d2) > 1e-2 * (1 + abs(d1)) or abs(right - left) > 1e-2 * (1 + abs(right)):
        return Outcome("grad", "skip", "not smooth here")
    if not np.isfinite(gv):
        return Outcome("grad", "mismatch", f"gradient {gv} not finite, finite differences {d2}")
    if abs(gv - d2) > 2e-2 * (1 + abs(d2)):
        return Outcome("grad", "mismatch", f"directional derivative {gv} vs finite differences {d2} / {d1}")
    return Outcome("grad", "ok")

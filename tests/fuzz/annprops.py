"""Generated annotations: each one checks the value, or is refused when the function is defined.

``case(seed)`` draws an annotation (a dtype, some axes, maybe a wrapper: ``| None``,
``tuple[...]``, ``list[...]``, ``dict[str, ...]``, quotes), defines

    def f[T, D](a: <annotation>, t: f32[T], d: f32[D], n: int): ...

and calls it with a value that matches and with values that are wrong in one way
(an axis one longer, a missing axis, another dtype). The matching call must pass
and every wrong one must raise ``ShapeError``, unless defining the function
already raised a coded error. A wrong value that passes means the annotation was
silently dropped.

``prover_case(seed)`` draws a function whose result's annotation is deliberately
off for some sizes; ``expex.prove`` must not call it proved.
"""

from __future__ import annotations

import linecache
import random
import itertools
from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp
import numpy as np

import expex
from expex.annotations.errors import code_of

T, D, N_ARG, R = 3, 2, 2, 4

#: (text, its size at T=3, D=2, n=2), or None for a variadic of two axes (5, 6)
ITEMS: list[tuple[str, int | None]] = [
    ("T", T), ("D", D), ("T + 1", T + 1), ("T - 1", T - 1), ("2 * T", 2 * T), ("T * D", T * D),
    ("D + T", D + T), ("3", 3), ("1", 1), ("R:4", R), ("n", N_ARG), ("n + T", N_ARG + T),
    ("T // 2", T // 2), ("(T + 1) // 2", (T + 1) // 2), ("T ** 2", T**2), ("T % 2", T % 2),
    ("D * D - 1", D * D - 1), ("*B", None),
]  # fmt: skip

DTYPES = {  # name: (good dtype, a wrong one or None)
    "f32": (jnp.float32, jnp.int32), "i32": (jnp.int32, jnp.float32), "Float": (jnp.float32, jnp.int32),
    "Int": (jnp.int32, jnp.float32), "Num": (jnp.float32, jnp.bool_), "f16": (jnp.float16, jnp.float32),
    "Bool": (jnp.bool_, jnp.int32),
}  # fmt: skip

_COUNT = itertools.count()


def define(src: str, ns: dict[str, Any]) -> Any:
    fname = f"<annprop-{next(_COUNT)}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    exec(compile(src, fname, "exec"), ns)
    return ns["f"]


def namespace(drop: str | None = None) -> dict[str, Any]:
    ns: dict[str, Any] = {"jnp": jnp, "expex": expex}
    for name in ("f32", "i32", "f16", "Float", "Int", "Num", "Bool"):
        if name != drop:
            ns[name] = getattr(expex, name)
    return ns


@dataclass
class Case:
    src: str
    ann: str
    good: Any
    wrong: list[tuple[str, Any]]
    drop: str | None = None


def _array(rng: random.Random) -> tuple[str, Callable[[str], Any], list[tuple[str, Callable[[], Any]]]]:
    """An array annotation; a maker of a matching value; makers of wrong ones."""
    dt = rng.choice(list(DTYPES))
    good_dt, bad_dt = DTYPES[dt]
    k = rng.randint(0, 3)
    items = rng.sample(ITEMS, k)
    # at most one variadic
    seen = False
    chosen = []
    for it in items:
        if it[1] is None:
            if seen:
                continue
            seen = True
        chosen.append(it)
    text = f"{dt}[{', '.join(t for t, _ in chosen)}]" if chosen else dt
    shape: list[int] = []
    for _, v in chosen:
        shape += [5, 6] if v is None else [v]

    def val(shp: list[int], dtype: Any = good_dt) -> Any:
        return jnp.zeros(tuple(shp), dtype)

    wrong: list[tuple[str, Callable[[], Any]]] = []
    fixed = [i for i, (t, v) in enumerate(chosen) if v is not None]
    pos = 0
    offs = []
    for t, v in chosen:
        offs.append(pos)
        pos += 2 if v is None else 1
    for i in fixed:
        bad = list(shape)
        bad[offs[i]] += 1
        wrong.append((f"axis {offs[i]} ({chosen[i][0]}) one longer", lambda bad=bad: val(bad)))
    if fixed and not seen:
        wrong.append(("an axis missing", lambda: val(shape[1:])))
    if bad_dt is not None:
        wrong.append((f"dtype {jnp.dtype(bad_dt).name}", lambda: val(shape, bad_dt)))
    return text, (lambda _: val(shape)), wrong


def case(seed: int) -> Case:
    rng = random.Random(seed)
    text, good, wrong = _array(rng)
    wrap = rng.choice(["plain", "plain", "optional", "tuple", "list", "dict", "quoted", "union"])
    g = good("")
    if wrap == "optional":
        ann, gv, wv = f"{text} | None", g, [(w, f()) for w, f in wrong]
    elif wrap == "tuple":
        ann, gv, wv = f"tuple[{text}, {text}]", (g, g), [(w, (g, f())) for w, f in wrong]
    elif wrap == "list":
        ann, gv, wv = f"list[{text}]", [g, g], [(w, [g, f()]) for w, f in wrong]
    elif wrap == "dict":
        ann, gv, wv = f"dict[str, {text}]", {"a": g}, [(w, {"a": f()}) for w, f in wrong]
    elif wrap == "quoted":
        ann, gv, wv = f'"{text}"', g, [(w, f()) for w, f in wrong]
    else:  # a union with another array annotation of a different rank
        ann, gv, wv = text, g, [(w, f()) for w, f in wrong]
    drop = None
    if rng.random() < 0.15:
        drop = text.split("[")[0]
    src = f"def f[T, D](a: {ann}, t: f32[T], d: f32[D], n: int):\n    return a\n"
    if drop == "f32":
        src = src.replace("t: f32[T], d: f32[D]", "t: i32[T], d: i32[D]")
    return Case(src, ann, gv, wv, drop)


def run_case(c: Case) -> list[str]:
    """Problems found: empty if every annotation checked or was refused at definition."""
    try:
        f = expex.fn(define(c.src, namespace(c.drop)))
    except Exception as e:
        if code_of(e) is None:
            return [f"{c.ann}: defining raised without a code: {type(e).__name__}: {e}"]
        return []
    if c.drop is not None:
        return [f"{c.ann} with `{c.drop}` undefined: defined without an error"]
    tdt = jnp.int32 if c.drop == "f32" else jnp.float32
    t, d = jnp.zeros(T, tdt), jnp.zeros(D, tdt)
    out = []
    try:
        f(c.good, t, d, N_ARG)
    except Exception as e:
        if code_of(e) is None:
            out.append(f"{c.ann}: the matching value raised without a code: {type(e).__name__}: {e}")
        return out  # refused at the first call: the annotation is not dropped
    for what, v in c.wrong:
        try:
            f(v, t, d, N_ARG)
        except expex.ShapeError:
            continue
        except Exception as e:
            if code_of(e) is None:
                out.append(f"{c.ann}: {what}: raised without a code: {type(e).__name__}: {e}")
            continue
        out.append(f"{c.ann}: {what}: passed unchecked ({_desc(v)})")
    return out


def _desc(v: Any) -> str:
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_desc(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_desc(x)}" for k, x in v.items()) + "}"
    return f"{v.dtype}{list(v.shape)}"


# ------------------------------------------------------------------ the prover

#: (body, the result's size as a function of T)
BODIES: list[tuple[str, str, Callable[[int], int]]] = [
    ("x", "T", lambda t: t),
    ("x[1:]", "T - 1", lambda t: t - 1),
    ("jnp.concatenate([x, x])", "2 * T", lambda t: 2 * t),
    ("jnp.pad(x, (0, 1))", "T + 1", lambda t: t + 1),
    ("x[::2]", "(T + 1) // 2", lambda t: (t + 1) // 2),
    ("x[: T // 2]", "T // 2", lambda t: t // 2),
    ("jnp.zeros(T * T)", "T ** 2", lambda t: t * t),
    ("jnp.zeros(T // 2 * 2)", "T // 2 * 2", lambda t: t // 2 * 2),
    ("x[:-1] if T > 1 else x", "T - 1", lambda t: t - 1 if t > 1 else t),
    ("jnp.zeros(T % 2 + T)", "T + T % 2", lambda t: t + t % 2),
    ("jnp.zeros(max(T, 3))", "max(T, 3)", lambda t: max(t, 3)),
    ("jnp.zeros(T - 1 if T > 2 else T)", "T - 1", lambda t: t - 1 if t > 2 else t),
    ("x[1:] if 1 < T else x", "T - 1", lambda t: t - 1 if t > 1 else t),
]

DECLS = ["T", "T - 1", "T + 1", "2 * T", "(T + 1) // 2", "T // 2", "T * T", "T + T % 2", "T - T % 2", "3", "T + 0"]


def _decl_size(decl: str, t: int) -> int:
    return eval(decl, {"T": t})  # noqa: S307


def prover_case(seed: int) -> tuple[str, str, bool]:
    """A function, its source, and whether its declared result is right for every T >= 1 tried."""
    rng = random.Random(seed)
    body, _, size = rng.choice(BODIES)
    decl = rng.choice(DECLS)
    right = all(_decl_size(decl, t) == size(t) for t in range(1, 40))
    src = f"def f[T](x: f32[T]) -> f32[{decl}]:\n    return {body}\n"
    return src, decl, right


def run_prover_case(seed: int) -> str | None:
    src, decl, right = prover_case(seed)
    ns = namespace()
    try:
        f = expex.fn(define(src, ns))
    except Exception as e:
        return None if code_of(e) else f"defining raised without a code: {e}\n{src}"
    rep = expex.prove(f)
    if not right and rep.status == "proved":
        return f"proved, but the result is wrong for some T:\n{src}"
    if right and rep.status == "failed":
        return f"failed, but the result is right for every T tried:\n{src}{rep}"
    return None


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    bad = 0
    for s in range(n):
        for p in run_case(case(s)):
            bad += 1
            print(s, p)
    for s in range(n):
        p = run_prover_case(s)
        if p:
            bad += 1
            print("prover", s, p)
    print("problems:", bad)

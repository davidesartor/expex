"""What a shape error says: the message, a hint for the usual causes, and notes.

``error`` formats a mismatch found while checking a call; ``hint`` suggests
the usual fix (a missing ``vectorize``, permuted axes, a dtype conversion);
``annotate`` adds "in f(x: f32[3, 4])" notes to exceptions passing through an
``@fn``, and ``_stage_hint`` explains JAX's tracer errors.
"""

import linecache
import os
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from .dims import Env, describe
from .errors import Mismatch, ShapeError, StageError, coded, coded_note
from .spec import ArraySpec, KeySpec

if TYPE_CHECKING:
    from ..function import Fn


def error(f: Fn[Any, Any], m: Mismatch, role: str, env: Env, site: str = "") -> Exception:
    if role == "local":
        top = m.path.split(".")[0].split("[")[0]
        head = f"{f.name}(): local variable `{top}` does not match its annotation."
        declared = f"{m.path}: {m.spec}"
        got = "assigned"
    elif role == "argument":
        top = m.path.split(".")[0].split("[")[0]
        head = f"{f.name}(): argument `{top}` does not match its annotation."
        declared = f"{m.path}: {m.spec}"
        got = "received"
    else:
        head = f"{f.name}(): the return value does not match its annotation."
        sub = m.path[len("return"):]
        declared = f"-> {m.spec}" if not sub else f"return{sub}: {m.spec}"
        got = "returned"
    pad = " " * max(0, len(declared) - len(str(m.spec)))
    lines = [
        head,
        "",
        f"    declared  {declared}",
        f"    {got}  {pad}{describe(m.value)}",
        "",
        f"    {m.reason}",
    ]
    binds = env.summary()
    if binds and role != "argument":
        lines.append(f"    Bound so far: {binds}.")
    suggestion = hint(f, m, role, env)
    if suggestion:
        lines += ["", f"    Hint: {suggestion}"]
    if site:
        text = linecache.getline(*_split_site(site)).strip()
        lines += ["", f"    at {site}" + (f"\n    > {text}" if text else "")]
    lines += ["", f"    {f.signature}", f"    defined at {f.location}"]
    if getattr(m, "stage", False):
        return coded("EX202", StageError("\n".join(lines)))
    return coded(_CODES[role], ShapeError("\n".join(lines)))


_CODES = {"argument": "EX101", "return": "EX102", "local": "EX103"}


def hint(f: Fn[Any, Any], m: Mismatch, role: str, env: Env) -> str:
    """A suggestion for the most common causes of a mismatch, or ''."""
    spec = m.spec
    if not isinstance(spec, ArraySpec):
        return ""
    value: Any = m.value
    try:
        shape, dtype = tuple(value.shape), value.dtype
    except AttributeError:
        return ""
    if not spec.dtype.accepts(dtype):
        want, got = spec.dtype.dtype, np.dtype(dtype) if _is_np(dtype) else None
        if want is not None and got is not None and want.kind == got.kind:
            if got.itemsize > want.itemsize:
                src = "NumPy and Python default to 64 bits; " if got.itemsize == 8 else ""
                return f"{src}convert with jnp.asarray(x, jnp.{want.name})."
            return f"convert with x.astype(jnp.{want.name}), or declare the wider dtype."
        if want is not None and got is not None and want.kind == "f" and got.kind in "iub":
            return f"this is an integer array; convert with x.astype(jnp.{want.name})."
        return ""
    rank = spec.core_rank
    if rank is None:
        return ""
    if len(shape) > rank and role == "argument":
        extra = len(shape) - rank
        return (
            f"`{m.path}` has {extra} extra leading ax{'is' if extra == 1 else 'es'}. To map "
            f"over {'it' if extra == 1 else 'them'}, call {f.name}.vectorize(...)."
            + (
                " Functions annotated jaxtyping's way are not mapped by themselves: "
                "@expex.fn(vectorize=True) makes plain calls map too."
                if getattr(f, "_jaxtyping", False) and getattr(f, "auto_vectorize", None) is None
                else ""
            )
        )
    if len(shape) == rank and rank > 1:
        want_shape = []
        for d in spec.dims:
            v = d.value if hasattr(d, "value") else env.dims.get(getattr(d, "name", ""), None)
            if not isinstance(v, int):
                return ""
            want_shape.append(v)
        if tuple(want_shape) != shape and sorted(want_shape) == sorted(shape):
            perm = [want_shape.index(s) for s in shape]
            if len(set(perm)) == len(perm):
                names = ", ".join(str(spec.dims[i]) for i in perm)
                return f"the axes look permuted: this has the layout [{names}]."
    return ""


def annotate(f: Fn[Any, Any], e: Exception, bound: dict[str, Any], env: Env) -> None:
    try:
        _stage_hint(e, f)
        if isinstance(e, NameError):
            _size_hint(e, f, env)
        elif isinstance(e, TypeError) and not f.converted:
            _type_param_hint(e, f)
        args = ", ".join(
            f"{n}: {describe(v)}" for n, v in bound.items() if f.params[n].spec.leafy
        )
        binds = env.summary()
        note = f"expex: in {f.name}({args})" + (f"  [{binds}]" if binds else "")
        if note not in getattr(e, "__notes__", ()):
            e.add_note(note)
    except Exception:
        pass


def _type_param_hint(e: TypeError, f: Fn[Any, Any]) -> None:
    """A type parameter used as a size in a body that is not converted: there it is a ``TypeVar``."""
    import re
    import typing

    raw = f.raw
    code = getattr(raw, "__code__", None)
    if code is None or getattr(e, "_expex_hint", False):
        return
    tb, in_body = e.__traceback__, False
    while tb is not None and not in_body:
        in_body, tb = tb.tb_frame.f_code is code, tb.tb_next
    if not in_body:
        return
    params = []  # the type parameters the body reads: its own, and its class's
    for name, cell in zip(code.co_freevars, getattr(raw, "__closure__", None) or ()):
        try:
            if isinstance(cell.cell_contents, (typing.TypeVar, typing.TypeVarTuple)):
                params.append(name)
        except ValueError:  # an empty cell
            pass
    text = str(e)
    named = [n for n in params if re.search(rf"\b{re.escape(n)}\b", text)]
    if not (named or (params and "TypeVar" in text)):
        return
    name = (named or params)[0]
    setattr(e, "_expex_hint", True)
    why = f.convert_note or "it is not converted"
    coded_note(
        "EX406", e,
        f"expex: {name} is a size of {f.name}(), but the body cannot use it by name because "
        f"{why}: there, {name} is the type parameter itself, not its size.",
    )  # fmt: skip


def _size_hint(e: NameError, f: Fn[Any, Any], env: Env) -> None:
    """A size used by name outside the function it belongs to: in a helper defined elsewhere."""
    name = getattr(e, "name", None)
    if not name or getattr(e, "_expex_hint", False):
        return
    raw = f.raw
    sizes = {*env.dims, *env.vars, *f.defs}
    sizes |= {p.__name__ for p in getattr(raw, "__type_params__", ())}
    if name not in sizes:
        return
    tb = e.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return
    code = tb.tb_frame.f_code
    setattr(e, "_expex_hint", True)
    if code is getattr(f._impl, "__code__", None) or code is getattr(raw, "__code__", None):
        why = f.convert_note or "it is not converted"
        coded_note(
            "EX406", e,
            f"expex: {name} is a size of {f.name}(), declared in its annotations, but the body "
            f"cannot use it by name because {why}.",
        )  # fmt: skip
        return
    helper = getattr(code, "co_qualname", code.co_name)
    coded_note(
        "EX405", e,
        f"expex: {name} is a size of {f.name}(), but `{helper}` ({code.co_filename}:{tb.tb_lineno}) "
        f"is defined outside {f.name}(), so it does not see {f.name}()'s sizes.\n"
        f"    - define `{helper}` inside {f.name}(): a nested function sees the sizes of the call\n"
        f"    - or decorate it with @expex.fn and annotate its arguments with {name}",
    )  # fmt: skip


def _is_np(dtype: Any) -> bool:
    try:
        np.dtype(dtype)
    except TypeError:
        return False
    return True


def _split_site(site: str) -> tuple[str, int]:
    fname, _, line = site.rpartition(":")
    try:
        return fname, int(line)
    except ValueError:
        return site, 0


_PKG_DIRS = (os.path.dirname(os.path.abspath(jax.__file__)), os.path.dirname(os.path.abspath(__file__)))


def _user_frame(e: BaseException) -> tuple[str, int, str] | None:
    tb = e.__traceback__
    found = None
    while tb is not None:
        fname = tb.tb_frame.f_code.co_filename
        if not os.path.abspath(fname).startswith(_PKG_DIRS):
            found = (fname, tb.tb_lineno, linecache.getline(fname, tb.tb_lineno).strip())
        tb = tb.tb_next
    return found


def _stage_hint(e: Exception, f: Fn[Any, Any]) -> None:
    if not isinstance(e, jax.errors.JAXTypeError) or getattr(e, "_expex_hint", False):
        return
    E = jax.errors
    if isinstance(e, E.TracerBoolConversionError):
        need = "a concrete Python bool (for `if`, `while`, `and`/`or`, `assert`)"
        fixes = [
            "to choose between two values:        jnp.where(cond, a, b)",
            "to choose between two computations:  jax.lax.cond(cond, f, g, *operands)",
            "to loop until a condition:           jax.lax.while_loop(cond_fn, body_fn, init)",
        ]
    elif isinstance(e, E.TracerIntegerConversionError):
        need = "a concrete Python int (for `range`, a shape, or an index into a Python list)"
        fixes = [
            "if it is a size, read it from an array's .shape (shapes are always concrete)",
            "to loop a traced number of times:    jax.lax.fori_loop / jax.lax.scan",
        ]
    elif isinstance(e, E.TracerArrayConversionError):
        need = "a concrete NumPy array"
        fixes = ["use jax.numpy (jnp) instead of numpy (np) on traced values"]
    elif isinstance(e, E.NonConcreteBooleanIndexError):
        need = "a concrete boolean mask (the result's shape would depend on the data)"
        fixes = ["keep the shape fixed:  jnp.where(mask, x, fill)  or  x * mask"]
    elif isinstance(e, E.ConcretizationTypeError):
        need = "a concrete value"
        fixes = []
    else:
        return
    setattr(e, "_expex_hint", True)
    frame = _user_frame(e)
    where = f"`{frame[2]}`  ({frame[0]}:{frame[1]})" if frame and frame[2] else "this code"
    traced = [n for n, p in f.params.items() if isinstance(p.spec, (ArraySpec, KeySpec))]
    lines = [
        f"expex: {where} needs {need},",
        "    but the value is traced: inside jit/vmap/grad, array arguments have no "
        "values, only shapes.",
    ]
    lines += [f"    - {x}" for x in fixes]
    lines.append(
        "    - if the value is configuration rather than data, declare the parameter "
        "Static[...]"
        + (f" (array parameters of {f.name}: {', '.join(traced)})" if traced else "")
    )
    coded_note("EX201", e, "\n".join(lines))

"""Saving and loading typed modules: ``expex.save(model, path)``, ``expex.load(Model, path)``.

::

    expex.save(model, "mamba.npz")
    model = expex.load(Mamba, "mamba.npz", D=64)    # or expex.load(template, "mamba.npz")

``save`` writes every array of a struct to one ``.npz`` file under a readable
key (``layers[2].mixer.x_proj``), with the static fields that are plain data
(numbers, strings, ``None``, tuples and lists of those) and the lengths of
lists as JSON in the same file. ``load`` rebuilds the object from its class's
annotations, without calling ``__init__``, and checks every array against
its field's annotation, with sizes shared across the whole object.
"""

import json
from typing import Any

import jax.numpy as jnp
import numpy as np

from ._dims import Env, describe, resolve
from ._errors import Mismatch, ShapeError
from ._spec import (
    ArraySpec, DictSpec, NoneSpec, SeqSpec, Spec, StaticSpec, StructSpec, TupleSpec, UnionSpec,
    _field_defaults, parse_owner, rebuild,
)  # fmt: skip

_META = "__expex__"  # the npz entry holding the JSON metadata
_MISSING = object()


def save(obj: Any, path: Any) -> None:
    """Write the arrays and plain static fields of ``obj`` to the ``.npz`` file ``path``.

    Static fields that are not plain data (callables, enums, ...) are not
    written; loading uses the class default, or the template's value.
    """
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"class": type(obj).__name__, "static": {}, "lengths": {}, "keys": {}, "dtypes": {}}

    def walk(spec: Spec, value: Any, p: str) -> None:
        if isinstance(spec, ArraySpec):
            a = np.asarray(value)
            if a.dtype.type.__module__.startswith("ml_dtypes"):  # bfloat16, ...: npz loses them
                meta["dtypes"][p], a = a.dtype.name, a.view(f"u{a.dtype.itemsize}")
            arrays[p] = a
        elif isinstance(spec, StructSpec):
            for name, s in spec.fields.items():
                v, q = getattr(value, name), _join(p, name)
                if _is_static(s):
                    if _plain(v):
                        meta["static"][q] = _encode(v)
                else:
                    walk(s, v, q)
        elif isinstance(spec, (SeqSpec, TupleSpec)):
            meta["lengths"][p] = len(value)
            elems = spec.elems if isinstance(spec, TupleSpec) else [spec.elem] * len(value)
            for i, (s, v) in enumerate(zip(elems, value)):
                walk(s, v, f"{p}[{i}]")
        elif isinstance(spec, DictSpec):
            meta["keys"][p] = list(value)
            for k, v in value.items():
                walk(spec.value, v, f"{p}[{k!r}]")
        elif isinstance(spec, UnionSpec) and value is None:
            meta["static"][p] = None
        elif isinstance(spec, UnionSpec):
            walk(next(o for o in spec.options if not isinstance(o, NoneSpec)), value, p)
        else:
            raise TypeError(f"expex.save: cannot save `{p}` (annotated {spec}).")

    walk(_owner_spec(type(obj)), obj, "")
    arrays[_META] = np.array(json.dumps(meta))
    np.savez(path, **arrays)  # type: ignore[arg-type]


def load(cls_or_template: Any, path: Any, *, strict: bool = True, **sizes: int) -> Any:
    """Read a struct written by :func:`save`, checking it against its annotations.

    ``cls_or_template`` is the class (``Mamba``), or an existing object whose
    structure and static fields are used as they are. ``sizes`` fix sizes by
    name (``D=64``). Keys missing from the file are an error, and so are extra
    keys unless ``strict=False``. Arrays are returned as JAX arrays.
    """
    template = None if isinstance(cls_or_template, type) else cls_or_template
    cls = type(template) if template is not None else cls_or_template
    spec = _owner_spec(cls)
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files if k != _META}
        meta = json.loads(str(f[_META])) if _META in f.files else {}
    static, lengths, keys = (meta.get(k, {}) for k in ("static", "lengths", "keys"))
    for k, name in meta.get("dtypes", {}).items():
        if k in arrays:
            arrays[k] = arrays[k].view(jnp.dtype(name))
    used: set[str] = set()
    missing: list[str] = []

    def build(spec: Spec, t: Any, p: str) -> Any:
        if isinstance(spec, ArraySpec):
            if p not in arrays:
                missing.append(p)
            used.add(p)
            return arrays.get(p)
        if isinstance(spec, StructSpec):
            values = {}
            for name, s in spec.fields.items():
                q, tv = _join(p, name), getattr(t, name, _MISSING)
                values[name] = _static(spec.cls, name, q, tv, static) if _is_static(s) else build(s, tv, q)
            return rebuild(spec.cls, **{k: v for k, v in values.items() if v is not _MISSING})
        if isinstance(spec, (SeqSpec, TupleSpec)):
            n = len(t) if t is not _MISSING else lengths.get(p, len(getattr(spec, "elems", ())))
            elems = spec.elems if isinstance(spec, TupleSpec) else [spec.elem] * n
            ts = t if t is not _MISSING else [_MISSING] * n
            out = [build(s, v, f"{p}[{i}]") for i, (s, v) in enumerate(zip(elems, ts))]
            return tuple(out) if isinstance(spec, TupleSpec) or spec.kind is tuple else out
        if isinstance(spec, DictSpec):
            names = list(t) if t is not _MISSING else keys.get(p, [])
            return {k: build(spec.value, _MISSING if t is _MISSING else t[k], f"{p}[{k!r}]") for k in names}
        if isinstance(spec, UnionSpec):
            if (t is None) if t is not _MISSING else (p in static and static[p] is None):
                return None
            return build(next(o for o in spec.options if not isinstance(o, NoneSpec)), t, p)
        raise TypeError(f"expex.load: cannot load `{p}` (annotated {spec}).")

    obj = build(spec, template if template is not None else _MISSING, "")
    extra = sorted(set(arrays) - used)
    if missing or (extra and strict):
        lines = [f"    missing from the file: {', '.join(missing)}"] if missing else []
        lines += [f"    in the file but not in {cls.__name__}: {', '.join(extra)}"] if extra and strict else []
        hint = "\n\n    Pass strict=False to ignore the extra keys." if extra and strict and not missing else ""
        raise ShapeError(f"expex.load({cls.__name__}, {str(path)!r}): the keys do not match.\n\n" + "\n".join(lines) + hint)
    _check(spec, obj, sizes, f"expex.load({cls.__name__}, {str(path)!r})")
    return spec.walk(obj, "", lambda s, v, p: jnp.asarray(v))


def _check(spec: StructSpec, obj: Any, sizes: dict[str, int], where: str) -> None:
    """Match ``obj`` against ``spec`` with ``sizes`` bound, as a ShapeError naming the field."""
    known: set[str] = set()
    spec.names(known)
    for k in sorted(set(sizes) - known):
        raise TypeError(f"{where}: {spec.cls.__name__} has no size `{k}` (its sizes: {', '.join(sorted(known))}).")
    env = Env()
    for k, v in sizes.items():
        env.dims[k] = int(v)
        env.origin[k] = f"{k}={v} passed to expex.load"
    try:
        for name, s in spec.fields.items():  # field by field, so that paths read `layers[2].x`
            if s.leafy and not isinstance(s, StaticSpec):
                s.match(getattr(obj, name), name, env)
        spec.match(obj, spec.cls.__name__, env)  # the class's invariants
        resolve(env)
    except Mismatch as m:
        found = "" if isinstance(m.spec, StructSpec) else f", file has {describe(m.value)}"
        raise ShapeError(f"{where}:\n\n    {m.path}: declared {m.spec}{found}\n\n    {m.reason}") from None


def _owner_spec(cls: type) -> StructSpec:
    spec = parse_owner(cls)
    if not isinstance(spec, StructSpec):
        raise TypeError(f"{cls.__name__} has no array fields: nothing to save or load.")
    return spec


def _is_static(spec: Spec) -> bool:
    return not spec.leafy or isinstance(spec, StaticSpec)


def _static(cls: type, name: str, p: str, given: Any, saved: dict[str, Any]) -> Any:
    """A static field's value: the template's, the file's, else the class default."""
    if given is not _MISSING:
        return given
    if p in saved:
        return _decode(saved[p])
    if name in _field_defaults(cls):
        return _MISSING  # rebuild() skips it and fills in the default
    raise TypeError(
        f"expex.load: the static field `{p}` ({cls.__name__}.{name}) is not in the file (only "
        f"numbers, strings, None and tuples/lists of those are saved) and has no default. "
        f"Load into a template instead: expex.load(model, path)."
    )


def _join(p: str, name: str) -> str:
    return f"{p}.{name}" if p else name


def _plain(v: Any) -> bool:
    if v is None or type(v) in (bool, int, float, str):
        return True
    return type(v) in (list, tuple) and all(map(_plain, v))


def _encode(v: Any) -> Any:
    if isinstance(v, tuple):
        return {"tuple": [_encode(x) for x in v]}
    return [_encode(x) for x in v] if isinstance(v, list) else v


def _decode(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(_decode(x) for x in v["tuple"])
    return [_decode(x) for x in v] if isinstance(v, list) else v

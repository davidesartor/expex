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

Any pytree holding structs saves too, such as ``(model, opt_state)`` with an
optax state whose moments are models: structs by their annotations, other
arrays under their pytree path (``[1][0].count``). It loads into a template
of the same structure (``jax.eval_shape`` of one is enough). ``load(...,
mesh=, names=)`` puts each array straight from the file onto its devices, as
:func:`expex.device_put` would place it.
"""

import json
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .annotations.dims import Env, describe, resolve
from .annotations.errors import Mismatch, ShapeError, coded
from .annotations.spec import (
    ArraySpec, DictSpec, NoneSpec, SeqSpec, Spec, StaticSpec, StructSpec, TupleSpec, UnionSpec,
    _field_defaults, _struct_fields, fields, join, parse_owner, rebuild,
)  # fmt: skip

_META = "__expex__"  # the npz entry holding the JSON metadata
_MISSING = object()


def save(obj: Any, path: Any) -> None:
    """Write the arrays and plain static fields of ``obj`` to the ``.npz`` file ``path``.

    ``obj`` is a struct, or any pytree of structs and arrays (``(model,
    opt_state)``). Static fields that are not plain data (callables, enums,
    ...) are not written; loading uses the class default, or the template's value.
    """
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"class": type(obj).__name__, "static": {}, "lengths": {}, "keys": {}, "dtypes": {}}
    if _struct_spec(type(obj)) is None:  # a pytree holding structs, such as (model, opt_state)
        if not jax.tree.leaves(obj):
            _owner_spec(type(obj))  # nothing to save: the error of a struct without arrays
        meta["tree"] = True
    for p, spec, value in _parts(obj):
        if spec is None or isinstance(spec, ArraySpec):
            a = np.asarray(value)
            if a.dtype.type.__module__.startswith("ml_dtypes"):  # bfloat16, ...: npz loses them
                meta["dtypes"][p], a = a.dtype.name, a.view(f"u{a.dtype.itemsize}")
            arrays[p] = a
        elif _is_static(spec) or isinstance(spec, NoneSpec):
            if not spec.leafy and any(hasattr(x, "shape") for x in jax.tree.leaves(value)):
                raise coded("EX501", TypeError(
                    f"expex.save: `{p}` ({type(value).__name__}) holds arrays that no annotation describes, "
                    f"so they would not be saved. Annotate them, or save with Orbax (docs/interop.md)."
                ))  # a layer from another library, say
            if value is None or _plain(value):
                meta["static"][p] = _encode(value)
        elif isinstance(spec, (SeqSpec, TupleSpec)):
            meta["lengths"][p] = len(value)
        elif isinstance(spec, DictSpec):
            meta["keys"][p] = list(value)
        elif not isinstance(spec, StructSpec):
            raise coded("EX501", TypeError(f"expex.save: cannot save `{p}` (annotated {spec})."))
    arrays[_META] = np.array(json.dumps(meta))
    np.savez(path, **arrays)  # type: ignore[arg-type]


def load(
    cls_or_template: Any, path: Any, *, strict: bool = True, mesh: Any = None,
    names: dict[str, Any] | None = None, **sizes: int,
) -> Any:  # fmt: skip
    """Read a struct written by :func:`save`, checking it against its annotations.

    ``cls_or_template`` is the class (``Mamba``), or an existing object whose
    structure and static fields are used as they are. A pytree of structs
    (``(model, opt_state)``) loads into a template, which may be abstract
    (``jax.eval_shape``); its other arrays must keep their shapes and dtypes.
    ``sizes`` fix sizes by name (``D=64``). Keys missing from the file are an
    error, and so are extra keys unless ``strict=False``. Arrays are returned as
    JAX arrays; with ``mesh``, each is put on the mesh as
    ``expex.device_put(..., mesh, **names)`` would place it, without a copy on
    one device first.
    """
    template = None if isinstance(cls_or_template, type) else cls_or_template
    cls = type(template) if template is not None else cls_or_template
    if template is not None and _struct_spec(cls) is None and jax.tree.leaves(template):
        return _load_tree(template, path, strict, mesh, names, sizes)
    spec = _owner_spec(cls)
    arrays, meta = read(path)
    used: set[str] = set()
    missing: list[str] = []
    build = lambda spec, t, p: _load_parts(spec, t, p, arrays, used, missing, meta)
    obj = build(spec, template if template is not None else _MISSING, "")
    where = f"expex.load({cls.__name__}, {str(path)!r})"
    _keys_match(arrays, used, missing, strict, cls.__name__, where)
    _check(spec, obj, sizes, where)
    return _put(obj, mesh, names)


def _load_tree(template: Any, path: Any, strict: bool, mesh: Any, names: Any, sizes: dict[str, int]) -> Any:
    """:func:`load` into a pytree of structs and arrays: each struct as :func:`load`, other arrays by path."""
    arrays, meta = read(path)
    name = type(template).__name__
    where = f"expex.load({name}, {str(path)!r})"
    used: set[str] = set()
    missing: list[str] = []
    structs: list[tuple[StructSpec, Any, str]] = []
    is_struct = lambda x: _struct_spec(type(x)) is not None

    def one(kp: Any, leaf: Any) -> Any:
        p = jax.tree_util.keystr(kp)
        if is_struct(leaf):
            spec: Any = _struct_spec(type(leaf))
            obj = _load_parts(spec, leaf, p, arrays, used, missing, meta)
            structs.append((spec, obj, p))
            return obj
        used.add(p)
        if p not in arrays:
            missing.append(p)
            return leaf
        a = arrays[p]
        if (np.shape(leaf), jnp.dtype(getattr(leaf, "dtype", a.dtype))) != (a.shape, a.dtype):
            raise coded("EX503", ShapeError(
                f"{where}:\n\n    {p}: the template has {describe(leaf)}, file has {describe(a)}"
            ))
        return a

    obj = jax.tree_util.tree_map_with_path(one, template, is_leaf=is_struct)
    _keys_match(arrays, used, missing, strict, name, where)
    known: set[str] = set()
    for spec, _, _ in structs:
        spec.names(known)
    for k in sorted(set(sizes) - known):
        raise coded("EX504", TypeError(f"{where}: no struct in {name} has a size `{k}` (their sizes: {', '.join(sorted(known))})."))
    for spec, part, p in structs:
        own: set[str] = set()
        spec.names(own)
        _check(spec, part, {k: v for k, v in sizes.items() if k in own}, where, p)
    return _put(obj, mesh, names)


def _load_parts(
    spec: Spec, t: Any, p: str, arrays: dict[str, np.ndarray], used: set[str], missing: list[str],
    meta: dict[str, Any] | None = None,
) -> Any:  # fmt: skip
    """The part of a saved object at path ``p``, described by ``spec`` (``t``: its template, or _MISSING).

    Arrays stay NumPy arrays; the keys read are added to ``used``, the absent ones to ``missing``.
    """
    meta = meta if meta is not None else {}
    static, lengths, keys = (meta.get(k, {}) for k in ("static", "lengths", "keys"))
    build = lambda spec, t, p: _load_parts(spec, t, p, arrays, used, missing, meta)

    if isinstance(spec, ArraySpec):
        if p not in arrays:
            missing.append(p)
        used.add(p)
        return arrays.get(p)
    if isinstance(spec, StructSpec):
        values = {}
        for name, s in spec.fields.items():
            q, tv = join(p, name), getattr(t, name, _MISSING)
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
    raise coded("EX501", TypeError(f"expex.load: cannot load `{p}` (annotated {spec})."))


def read(source: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The arrays of a file written by :func:`save` (bfloat16 and co. restored), and its metadata."""
    with np.load(source, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files if k != _META}
        meta = json.loads(str(f[_META])) if _META in f.files else {}
    for k, name in meta.get("dtypes", {}).items():
        if k in arrays:
            arrays[k] = arrays[k].view(jnp.dtype(name))
    return arrays, meta


def _keys_match(arrays: dict[str, Any], used: set[str], missing: list[str], strict: bool, name: str, where: str) -> None:
    extra = sorted(set(arrays) - used)
    if missing or (extra and strict):
        lines = [f"    missing from the file: {', '.join(missing)}"] if missing else []
        lines += [f"    in the file but not in {name}: {', '.join(extra)}"] if extra and strict else []
        hint = "\n\n    Pass strict=False to ignore the extra keys." if extra and strict and not missing else ""
        raise coded("EX502", ShapeError(f"{where}: the keys do not match.\n\n" + "\n".join(lines) + hint))


def _put(obj: Any, mesh: Any, names: dict[str, Any] | None) -> Any:
    """The NumPy arrays of a loaded ``obj`` as JAX arrays: on ``mesh`` as ``names`` say, if given."""
    from .sharding import _check_names, _leaf, _structs

    if mesh is None:
        if names:
            raise coded("EX601", TypeError("expex.load(..., names=...) needs mesh=: the mesh whose axes they name."))
        visit = lambda s, v, p: jnp.asarray(v)
        other = jnp.asarray
    else:
        axes = _check_names(mesh, names or {})
        visit = lambda s, v, p: jax.device_put(v, _leaf(s, v, p, mesh, axes))
        repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        other = lambda v: jax.device_put(v, repl)
    obj = _structs(obj, None, visit)
    return jax.tree.map(lambda v: other(v) if isinstance(v, np.ndarray) else v, obj)


def _check(spec: StructSpec, obj: Any, sizes: dict[str, int], where: str, prefix: str = "") -> None:
    """Match ``obj`` against ``spec`` with ``sizes`` bound, as a ShapeError naming the field."""
    known: set[str] = set()
    spec.names(known)
    for k in sorted(set(sizes) - known):
        raise coded("EX504", TypeError(f"{where}: {spec.cls.__name__} has no size `{k}` (its sizes: {', '.join(sorted(known))})."))
    env = Env()
    for k, v in sizes.items():
        env.dims[k] = int(v)
        env.origin[k] = f"{k}={v} passed to expex.load"
    try:
        for name, s in spec.fields.items():  # field by field, so that paths read `layers[2].x`
            if s.leafy and not isinstance(s, StaticSpec):
                s.match(getattr(obj, name), join(prefix, name), env)
        spec.match(obj, prefix or spec.cls.__name__, env)  # the class's invariants
        resolve(env)
    except Mismatch as m:
        found = "" if isinstance(m.spec, StructSpec) else f", file has {describe(m.value)}"
        raise coded("EX503", ShapeError(f"{where}:\n\n    {m.path}: declared {m.spec}{found}\n\n    {m.reason}")) from None


def _parts(obj: Any) -> Iterator[tuple[str, Spec | None, Any]]:
    """:func:`fields` of a struct; of a pytree, those of each struct in it and ``(path, None, leaf)`` for other leaves."""
    if _struct_spec(type(obj)) is not None:
        yield from fields(_owner_spec(type(obj)), obj)
        return
    is_struct = lambda x: _struct_spec(type(x)) is not None
    for kp, leaf in jax.tree_util.tree_flatten_with_path(obj, is_leaf=is_struct)[0]:
        p = jax.tree_util.keystr(kp)
        if is_struct(leaf):
            yield from fields(_owner_spec(type(leaf)), leaf, p)
        else:
            yield p, None, leaf


def _struct_spec(cls: type) -> StructSpec | None:
    """The spec of a struct class with array fields, else None (plain containers, optax states)."""
    if _struct_fields(cls) is None:  # not a dataclass (a Flax NNX module): a pytree, saved by path
        return None
    spec = parse_owner(cls)
    return spec if isinstance(spec, StructSpec) else None


def _owner_spec(cls: type) -> StructSpec:
    spec = parse_owner(cls)
    if not isinstance(spec, StructSpec):
        raise coded("EX501", TypeError(f"{cls.__name__} has no array fields: nothing to save or load."))
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
    raise coded("EX505", TypeError(
        f"expex.load: the static field `{p}` ({cls.__name__}.{name}) is not in the file (only "
        f"numbers, strings, None and tuples/lists of those are saved) and has no default. "
        f"Load into a template instead: expex.load(model, path)."
    ))


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

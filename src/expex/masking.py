"""``expex.mask``: a pytree of booleans chosen by the annotations, for optax and Equinox.

::

    decay = expex.mask(model, lambda path, spec: spec.ndim >= 2)   # the matrices
    opt = optax.adamw(1e-3, weight_decay=0.1, mask=decay)

The annotation says what an array is, where its shape may not: in an ensemble
built by ``jax.vmap``, a bias is ``f32[D]`` with two axes.
"""

from typing import Any, Callable

import jax

from .annotations.spec import ArraySpec, StructSpec, _struct_fields, fields, parse_owner


def mask(tree: Any, where: Callable[[str, ArraySpec], bool]) -> Any:
    """A pytree of the structure of ``tree``: ``where(path, spec)`` for each annotated array.

    ``tree`` is a struct (an ``@expex.struct`` model, an Equinox module), or any
    pytree holding them (``(model, opt_state)``). ``path`` is the array's path
    in its struct, as errors and checkpoints name it (``blocks[2].w``); ``spec``
    its annotation, with ``ndim``, ``dtype`` and ``dims``. Arrays that no
    annotation describes (a layer from another library, a plain array next to
    the structs) are ``False``. The result fits wherever optax and Equinox take
    a mask: ``optax.adamw(..., mask=)``, ``optax.masked``, ``eqx.partition``.
    """

    def one(x: Any) -> Any:
        spec = _spec(type(x))
        parts = [] if spec is None else fields(spec, x)  # arrays matched by identity: nothing is rebuilt
        chosen = {id(v): bool(where(p, s)) for p, s, v in parts if isinstance(s, ArraySpec)}
        return jax.tree.map(lambda v: chosen.get(id(v), False), x)

    return jax.tree.map(one, tree, is_leaf=lambda x: _spec(type(x)) is not None)


_SPECS: dict[type, StructSpec | None] = {}


def _spec(cls: type) -> StructSpec | None:
    """The spec of a dataclass or NamedTuple that annotates arrays, else None."""
    if cls not in _SPECS:
        spec = parse_owner(cls) if _struct_fields(cls) is not None else None
        _SPECS[cls] = spec if isinstance(spec, StructSpec) else None
    return _SPECS[cls]

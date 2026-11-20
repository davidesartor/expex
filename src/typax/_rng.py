"""Implicit PRNG keys.

A *key context* holds a key and a counter. ``key()`` returns
``fold_in(context key, counter)`` and bumps the counter, so every draw gets a
distinct key without the user threading and splitting keys by hand.

Contexts nest: a ``@fn(random=True)`` function, each branch of a converted
``if`` and each iteration of a converted loop gets a child context derived
from its parent, so results are the same whether the code runs eagerly, under
``jit``, as a ``scan`` or as a ``vmap``.

The root of the chain is either an explicit ``with rng(key):`` block or the
global key (``seed(n)``). The global key is Python state, so it may only be
consumed outside of JAX transformations; inside ``jit`` it would be baked into
the compiled function as a constant.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Iterator

import jax

from ._errors import StageError

try:  # private, but there is no public equivalent
    from jax._src.core import trace_state_clean as _clean
except Exception:  # pragma: no cover

    def _clean() -> bool:
        return True


def tracing() -> bool:
    """True inside any JAX transformation (jit, vmap, grad, scan, ...)."""
    return not _clean()


class KeyCtx:
    """A key plus a counter. The key is derived lazily from ``(base, salt)``."""

    __slots__ = ("_base", "_salt", "_key", "n")

    def __init__(self, base: Any, salt: Any = None):
        self._base = base
        self._salt = salt
        self._key = base if salt is None else None
        self.n = 0

    @property
    def key(self) -> Any:
        if self._key is None:
            self._key = jax.random.fold_in(self._base, self._salt)
        return self._key

    def next(self) -> Any:
        k = jax.random.fold_in(self.key, self.n)
        self.n += 1
        return k


class _State(threading.local):
    def __init__(self) -> None:
        self.stack: list[KeyCtx] = []
        self.root: KeyCtx | None = None


_state = _State()


def seed(n: int | Any = 0) -> None:
    """Reset the global key. ``n`` is an int seed or a PRNG key."""
    _state.root = KeyCtx(_as_key(n))


def _as_key(k: Any) -> Any:
    if isinstance(k, int):
        return jax.random.key(k)
    return k


def current() -> KeyCtx | None:
    """The innermost active key context, if any."""
    return _state.stack[-1] if _state.stack else None


def fresh(what: str = "typax.key()") -> Any:
    """A new key from the active context, or from the global key at top level."""
    ctx = current()
    if ctx is not None:
        return ctx.next()
    if tracing():
        raise StageError(
            f"{what} needs a key, but it is being traced (jit/vmap/grad/scan) with no key "
            f"in scope.\n"
            f"    The global key lives in Python, so inside a compiled function it would be "
            f"frozen into a constant and every call would draw the same numbers.\n"
            f"    - let typax own the jit:   @fn(jit=True, random=True)\n"
            f"    - or pass a key explicitly:   with typax.rng(key): ...   (key being an "
            f"argument of the jitted function)"
        )
    if _state.root is None:
        _state.root = KeyCtx(jax.random.key(0))
    return _state.root.next()


def key() -> Any:
    """A fresh PRNG key. Each call returns a different one."""
    return fresh()


@contextlib.contextmanager
def rng(k: int | Any) -> Iterator[None]:
    """Use ``k`` (a key or an int seed) as the source of keys inside the block."""
    _state.stack.append(KeyCtx(_as_key(k)))
    try:
        yield
    finally:
        _state.stack.pop()


@contextlib.contextmanager
def child(base: Any, salt: Any = None) -> Iterator[None]:
    """Internal: run a block in a context derived from ``(base, salt)``."""
    if base is None:
        yield
        return
    _state.stack.append(KeyCtx(base, salt))
    try:
        yield
    finally:
        _state.stack.pop()


def sub() -> Any:
    """Internal: a key for a nested construct, or None when no context is active."""
    ctx = current()
    return None if ctx is None else ctx.next()

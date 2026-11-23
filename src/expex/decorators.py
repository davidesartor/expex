"""The decorators and their options: ``@fn``, ``@jit``, ``@vectorize``, ``@random``, ``@remat``.

``fn`` makes an :class:`~expex.function.Fn`; ``vectorize`` and ``random`` set
its flags (on a copy), so they stack in any order; ``jit`` and ``remat`` wrap
it in ``jax.jit`` / ``jax.checkpoint`` with the static arguments read from the
signature. ``jit(donate="auto")`` donates what each call site reassigns.
"""

import ast
import copy
import functools
import inspect
import linecache
import sys
from collections.abc import Iterator, Mapping
from typing import Any, Callable, overload

import jax

from . import keys
from .annotations.spec import StaticSpec
from .function import REGISTRY, BoundCompiled, Fn, P, Param, R, unwrap


@overload
def fn(f: Callable[P, R], /) -> Fn[P, R]: ...
@overload
def fn(
    *,
    jit: bool = False,
    vectorize: bool | None = None,
    random: bool = False,
    remat: Any = False,
    donate: str | tuple[str, ...] = (),
    examples: dict[str, Any] | None = None,
    prove: bool = True,
    convert: bool = True,
    **moved: Any,
) -> Callable[[Callable[P, R]], Fn[P, R]]: ...
def fn(
    f: Any = None,
    /,
    *,
    jit: bool = False,
    vectorize: bool | None = None,
    random: bool = False,
    remat: Any = False,
    donate: str | tuple[str, ...] = (),
    examples: dict[str, Any] | None = None,
    prove: bool = True,
    convert: bool = True,
    **moved: Any,
) -> Any:
    """Make a function's shape annotations real.

    ::

        @expex.fn
        def split_heads[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:
            assert D % heads == 0
            return x.reshape(T, heads, D // heads).transpose(1, 0, 2)

    Arguments and the result are checked on every call (once per trace under
    ``jax.jit``), the prover checks the shapes for all sizes, and plain
    ``if`` / ``while`` / ``for`` work on traced values. Sizes can be computed in
    the annotations from other sizes, int parameters and fields of ``self``.
    ``assert``s on sizes at the top of the body are constraints: checked before
    the body runs, and assumed by the prover.

    ``jit``
        Compile, as :func:`expex.jit`: static arguments read from the signature.
    ``vectorize``
        Whether a call whose arguments have extra leading axes (beyond their
        annotations) maps over them, as ``f.vectorize(...)`` does. On by default
        (``expex.config.vectorize``); ``False`` makes such a call a ``ShapeError``.
    ``random``
        An implicit PRNG key, ``expex.key()``, as :func:`expex.random`.
    ``remat``
        Gradient checkpointing, as :func:`expex.remat`: ``True``, or a policy
        (``"dots"``, ``"dots_no_batch"``, ``"everything"`` or a JAX policy).
    ``donate``
        With ``jit``: parameters whose buffers the call may reuse, e.g. the
        state of a training step. ``"auto"`` donates what each call site
        reassigns: ``state = step(state, batch)`` donates ``state``.
    ``examples``
        Values the prover should try for parameters it cannot guess,
        as ``{"name": [value, ...]}``.
    ``prove``
        ``False`` leaves the function out of ``expex.prove`` and ``pytest --expex``.
    ``convert``
        ``False`` keeps Python control flow as it is (no rewriting).

    Combine with :func:`vectorize`, :func:`random` and :func:`jit`, and with JAX's
    own transformations.
    """
    hints = {
        "checked": "use `with expex.config(checked=False):`",
        "lift": "the option is now called vectorize=True",
        "where": "write the constraint as an `assert` at the top of the body: assert T >= 2",
    }
    for k, v in moved.items():
        hint = hints.get(
            k, f"write the size in the annotation instead, e.g. -> f32[{v}] in place of {k}"
        )
        raise TypeError(f"@expex.fn got {k}={v!r}: {hint}.")

    def wrap(raw: Callable[..., Any]) -> Fn[Any, Any]:
        flags: dict[str, bool | None] = {}
        if isinstance(raw, Fn):
            flags = {"random": raw.random, "auto_vectorize": raw.auto_vectorize}
            REGISTRY.discard(raw)
            raw = raw.raw
        out = Fn(raw, prove=prove, convert=convert, examples=examples)
        vars(out).update(flags)
        if random:
            out.random = True
        if vectorize is not None:
            out.auto_vectorize = vectorize
        result: Any = out
        if remat:
            result = _remat(out, remat)
        if jit:
            how = donate if donate == "auto" else (donate,) if isinstance(donate, str) else tuple(donate)
            result = _jit(result, donate_argnames=how)
        elif donate:
            raise TypeError("@expex.fn(donate=...) only applies with jit=True")
        return result

    if f is not None:
        return wrap(f)
    return wrap


def _flagged(f: Any, flag: str) -> Any:
    if isinstance(f, (staticmethod, classmethod)):
        return type(f)(_flagged(f.__func__, flag))
    if not isinstance(f, Fn) and hasattr(f, "__wrapped__"):
        deco = "@expex.random" if flag == "random" else "@expex.vectorize"
        raise TypeError(
            f"{deco} must go under @jax.jit / @jax.checkpoint (inside it would be "
            f"compiled once and repeat): put it directly on the @expex.fn function, or use "
            f"@expex.jit, which handles random functions."
        )
    if isinstance(f, Fn):
        g = copy.copy(f)
        g.params = {k: Param(p.name, p.spec, p.kind, p.default) for k, p in f.params.items()}
        g._declared = set(f._declared)
        g.examples = dict(f.examples)
        REGISTRY.discard(f)
    else:
        g = fn(f)
    setattr(g, flag, True)
    REGISTRY.add(g)
    return g


def vectorize(f: Callable[P, R], /) -> Fn[P, R]:
    """Map plain calls over extra leading batch axes, as ``f.vectorize(...)`` does.

    ::

        @expex.vectorize
        def norm[D](x: f32[D]) -> f32: ...

        norm(jnp.ones((8, 3, 5)))   # f32[8, 3]

    Every ``@fn`` does this already unless ``expex.config.vectorize`` is off;
    this decorator (like ``@expex.fn(vectorize=True)``) keeps it on for this
    function regardless. Arguments without extra axes are shared by every
    element. A plain function is wrapped with :func:`fn` first.
    """
    return _flagged(f, "auto_vectorize")


def jit(f: Callable[P, R] | None = None, /, **options: Any) -> Any:
    """``jax.jit``, with the static arguments read from the signature.

    ::

        @expex.jit
        def tile[N, K](x: f32[N], n: int) -> f32[K]: ...    # n is static

    Parameters annotated with plain Python types (``int``, ``bool``, ``str``,
    tuples, callables, enums, ...) and ``Static[...]`` are static; arrays and
    ``float`` / ``complex`` numbers are traced, so a new learning rate does not
    recompile. A ``@expex.random`` function draws a fresh key outside the compiled
    code on every call, unless ``key=`` is given. Other options go to ``jax.jit``.
    A plain function is wrapped with :func:`fn` first.
    """
    if f is None:
        return lambda g: jit(g, **options)
    if isinstance(f, (staticmethod, classmethod)):
        return type(f)(jit(f.__func__, **options))
    if not isinstance(unwrap(f), Fn):
        f = fn(f)
    g: Any = unwrap(f)
    static = static_argnames(g)
    given = options.pop("static_argnames", ())
    static += [given] if isinstance(given, str) else list(given)
    static_names = tuple(dict.fromkeys(static))
    if options.get("donate_argnames") == "auto":
        options.pop("donate_argnames")
        compiled: Any = _AutoDonate(f, g, static_names, options)
    else:
        compiled = jax.jit(f, static_argnames=static_names, **options)
    vectorized = jax.jit(_vectorizer(g), static_argnames=static_names, **{k: v for k, v in options.items() if k != "donate_argnames"})
    if g.random:
        compiled, vectorized = _fresh_key(compiled, g), _fresh_key(vectorized, g)
    # the Fn's API on the compiled function: f.vectorize(...) compiled too, f.flow, f.info(), f.check()
    compiled.vectorize, compiled.flow, compiled.info, compiled.check = vectorized, _Flow(g), g.info, g.check
    return compiled


def _vectorizer(g: Fn[Any, Any]) -> Callable[..., Any]:
    """``g.vectorize``, as a function with ``g``'s signature (for ``jax.jit``)."""

    def vectorize(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        return g.vectorize(*args, **kwargs)

    vectorize.__signature__ = g._sig  # type: ignore[attr-defined]
    vectorize.__qualname__ = f"{g.name}.vectorize"
    return vectorize


def _fresh_key(compiled: Any, g: Fn[Any, Any]) -> Any:
    """A compiled random function that draws a fresh key outside the compiled code, unless given one."""

    @functools.wraps(compiled)
    def call(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        if "key" not in kwargs and "key" not in g.params:
            kwargs["key"] = keys.fresh(f"{g.name}()")
        return compiled(*args, **kwargs)

    return call


class _Flow(Mapping[int, str]):
    """``f.flow`` of a compiled ``f``: read from the ``Fn`` whenever it is used."""

    def __init__(self, g: Fn[Any, Any]):
        self.g = g

    def __getitem__(self, line: int) -> str:
        return self.g.flow[line]

    def __iter__(self) -> Iterator[int]:
        return iter(self.g.flow)

    def __len__(self) -> int:
        return len(self.g.flow)

    def __repr__(self) -> str:
        return repr(self.g.flow)


class CompiledMethod:
    """A compiled method of a struct: ``model.step(...)`` and ``model.step.vectorize(...)`` bound alike."""

    def __init__(self, compiled: Any):
        self.compiled = self.__wrapped__ = compiled
        self.__doc__ = compiled.__doc__

    def __get__(self, obj: Any, cls: Any = None) -> Any:
        return self.compiled if obj is None else BoundCompiled(self.compiled, obj)


#: ``jit``, for ``fn``, whose option of that name hides it.
_jit = jit


def static_argnames(g: Any) -> list[str]:
    """Parameters that must be static: plain Python types and Static[...], not float/complex."""
    return [
        n for n, p in g.params.items()
        if isinstance(p.spec, StaticSpec) and (p.spec.explicit or p.spec.inner not in (float, complex))
    ]  # fmt: skip


_POLICIES = {
    "nothing": jax.checkpoint_policies.nothing_saveable,
    "everything": jax.checkpoint_policies.everything_saveable,
    "dots": jax.checkpoint_policies.dots_saveable,
    "dots_no_batch": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
}


def remat(f: Any = None, /, *, policy: Any = None) -> Any:
    """Gradient checkpointing (``jax.checkpoint``) that knows the static arguments.

    In the backward pass, recompute this function's intermediates instead of
    storing them. ``policy`` chooses what is kept anyway: ``"dots"`` (matrix
    products), ``"dots_no_batch"``, ``"everything"``, or a
    ``jax.checkpoint_policies`` policy. Arguments with plain Python types stay
    Python values, as under :func:`jit`.
    """
    if f is None:
        return lambda g: remat(g, policy=policy)
    if isinstance(f, (staticmethod, classmethod)):
        return type(f)(remat(f.__func__, policy=policy))
    return _remat(f if isinstance(unwrap(f), Fn) else fn(f), True if policy is None else policy)


def _remat(f: Any, how: Any) -> Any:
    g: Any = unwrap(f)
    if isinstance(how, str) and how not in _POLICIES:
        raise ValueError(f"remat must be True, a JAX policy or one of {sorted(_POLICIES)}, not {how!r}")
    policy = None if how is True else _POLICIES.get(how, how) if isinstance(how, str) else how
    static = set(static_argnames(g))
    sig = g._sig

    @functools.wraps(f)
    def call(*args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        ba = sig.bind(*args, **kwargs)
        fixed = {n: v for n, v in ba.arguments.items() if n in static}
        dyn = {n: v for n, v in ba.arguments.items() if n not in static}

        def body(dyn: dict[str, Any]) -> Any:
            b = inspect.BoundArguments(sig, {n: fixed[n] if n in fixed else dyn[n] for n in ba.arguments})  # type: ignore[arg-type]
            return f(*b.args, **b.kwargs)

        return jax.checkpoint(body, policy=policy)(dyn)

    call.__signature__ = sig  # type: ignore[attr-defined]
    return call


class _Probe:
    def __call__(self, *args: Any, **kwargs: Any) -> int:
        return sys.getrefcount(args[0] if args else kwargs["a"])


def _held_once() -> tuple[int, int, int]:
    """Reference counts of an argument nothing else holds, seen from ``_AutoDonate.__call__``.

    Measured the same way (a call to an instance, the argument held by one
    local variable), since the count depends on how the interpreter passes
    arguments: (positional, positional when keywords are given too, keyword).
    """
    value, probe = object(), _Probe()
    return probe(value), probe(value, k=1), probe(a=value)


class _AutoDonate:
    """``jax.jit`` that donates the arguments a call site reassigns: ``state = step(state, x)``.

    The calling line is read once per call site (from the source, through the
    frame's instruction position). An argument passed as a name that the same
    statement assigns to is donated, unless something else also holds it (its
    reference count says so). Arrays inside it that are held elsewhere are not
    detected; reusing a donated array afterwards raises JAX's "deleted" error.
    """

    def __init__(self, f: Any, g: Any, static: tuple[str, ...], options: dict[str, Any]):
        self.f, self.static, self.options = f, static, options
        self.names = [n for n in g._sig.parameters]
        self.compiled: dict[tuple[str, ...], Any] = {}
        self.plain = self._variant(())
        self.sites: dict[tuple[Any, int], Any] = {}  # call site -> its plan (see _plan)
        self.base = _held_once()
        functools.update_wrapper(self, f)

    def _variant(self, donate: tuple[str, ...]) -> Any:
        if donate not in self.compiled:
            self.compiled[donate] = jax.jit(
                self.f, static_argnames=self.static, donate_argnames=donate, **self.options
            )
        return self.compiled[donate]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        __tracebackhide__ = True
        frame: Any = sys._getframe(1)
        while frame.f_globals.get("__name__", "").startswith("expex"):
            frame = frame.f_back  # through expex's own wrappers, to the user's call
        plan = self.sites.get((frame.f_code, frame.f_lasti), _UNSEEN)
        if plan is _UNSEEN:
            plan = self.sites[(frame.f_code, frame.f_lasti)] = self._plan(frame)
        if plan is None:  # this call site reassigns none of the arguments
            return self.plain(*args, **kwargs)
        positions, keywords, extra = plan
        plain, mixed, by_name = self.base
        base = (mixed if kwargs else plain) + extra
        donate = tuple(
            [n for i, n in positions if i < len(args) and sys.getrefcount(args[i]) <= base]
            + [k for k in keywords if k in kwargs and sys.getrefcount(kwargs[k]) <= by_name + extra]
        )
        return (self.compiled.get(donate) or self._variant(donate))(*args, **kwargs)

    def _plan(self, frame: Any) -> Any:
        """For a call site: the arguments it may donate (by position and keyword), or None."""
        positions, keywords = _reassigned(frame)
        named = tuple(
            (i, self.names[i]) for i in positions
            if i < len(self.names) and self.names[i] not in self.static
        )  # fmt: skip
        keywords = tuple(k for k in keywords if k in self.names and k not in self.static)
        if not named and not keywords:
            return None
        # module-level code reads its variables as globals, which holds one more reference
        return named, keywords, 0 if frame.f_code.co_flags & inspect.CO_OPTIMIZED else 1


_UNSEEN = object()
_SOURCES: dict[str, Any] = {}


def _reassigned(frame: Any) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Which arguments of the call running in ``frame`` its statement assigns back to."""
    code = frame.f_code
    try:
        line, _, col, _ = list(code.co_positions())[frame.f_lasti // 2]
        name = code.co_filename
        if name not in _SOURCES:
            _SOURCES[name] = ast.parse("".join(linecache.getlines(name)))
        tree = _SOURCES[name]
    except Exception:
        return (), ()
    for stmt in ast.walk(tree):
        if not isinstance(stmt, ast.Assign) or not (stmt.lineno <= (line or 0) <= (stmt.end_lineno or 0)):
            continue
        call = next(
            (n for n in ast.walk(stmt.value) if isinstance(n, ast.Call) and n.lineno == line and n.col_offset == col),
            None,
        )
        if call is None:
            continue
        targets = {n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        positions = tuple(i for i, a in enumerate(call.args) if isinstance(a, ast.Name) and a.id in targets)
        keywords = tuple(
            k.arg for k in call.keywords if k.arg and isinstance(k.value, ast.Name) and k.value.id in targets
        )
        return positions, keywords
    return (), ()


def random(f: Callable[P, R], /) -> Fn[P, R]:
    """Give a function an implicit PRNG key: ``expex.key()`` inside it is fresh on every call.

    Each call draws a new key from the global key (``expex.seed``) or from the
    enclosing ``with expex.rng(key):`` block. Pass ``key=`` to choose it, which is
    required under ``jax.jit`` (the global key is Python state and would be
    frozen into the compiled function)::

        @expex.random
        @expex.fn
        def noisy[N](x: f32[N]) -> f32[N]:
            return x + jr.normal(expex.key(), x.shape)

        noisy(x)                                    # a new draw every time
        jax.jit(noisy)(x, key=jr.key(0))            # compiled: pass the key

    Under ``f.vectorize(...)`` each element gets its own key.
    """
    return _flagged(f, "random")

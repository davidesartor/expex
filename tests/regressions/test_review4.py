"""Bugs found by the fourth review (composable decorators, plain jax.jit)."""

import gc

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import Key, f32, fn
from expex._fn import REGISTRY


@expex.random
@fn
def noise[D](x: f32[D]) -> f32[D]:
    return x + jr.normal(expex.key(), x.shape)


def test_random_above_a_jax_wrapper_is_refused():
    # inside jax.jit the key would be traced once and repeat on every call
    for wrap in (jax.jit, jax.checkpoint):
        with pytest.raises(TypeError, match="must go under @jax.jit"):
            expex.random(wrap(fn(lambda x: x)))


def test_rng_scope_opened_outside_jit_is_not_frozen_into_it():
    j = jax.jit(noise)
    with expex.rng(0), pytest.raises(expex.StageError, match="set outside this jit"):
        j(jnp.zeros(2))

    @jax.jit
    def inside(x, key):
        with expex.rng(key):  # the supported pattern: the scope's key is traced
            return noise(x)

    assert not jnp.allclose(inside(jnp.zeros(2), jr.key(0)), inside(jnp.zeros(2), jr.key(1)))


def test_stacked_decorators_register_once():
    @expex.lift
    @expex.random
    @fn
    def stacked[D](x: f32[D]) -> f32[D]:
        return x

    @fn(prove=False)
    @expex.lift
    def hidden[D](x: f32[D]) -> f32[D]:
        return x

    gc.collect()
    names = [f.name for f in REGISTRY]
    assert sum(n.endswith("stacked") for n in names) == 1
    assert all(f.prove is False for f in REGISTRY if f.name.endswith("hidden"))


def test_random_function_with_its_own_key_parameter_runs_under_jit():
    @expex.random
    @fn
    def own(x: f32[2], key: Key) -> f32[2]:
        return x + jr.normal(expex.key(), x.shape) + jr.normal(key, x.shape)

    a = jax.jit(own)(jnp.zeros(2), key=jr.key(0))
    assert not jnp.allclose(a, jax.jit(own)(jnp.zeros(2), key=jr.key(1)))
    assert jnp.allclose(own(jnp.zeros(2), jr.key(0)), a)  # deterministic in its key
    assert own.lift(jnp.zeros((3, 2)), jr.key(0)).shape == (3, 2)


def test_lift_takes_one_key_per_example():
    ks = jr.split(jr.key(0), 4)
    out = noise.lift(jnp.zeros((4, 2)), key=ks)
    assert jnp.allclose(out[1], noise(jnp.zeros(2), key=ks[1]))


def test_remat_policies_by_name():
    for policy in ("dots", "dots_no_batch", "nothing", "everything", True):
        f = fn(remat=policy)(lambda x: x * 2)
        assert jax.grad(lambda x: f(x).sum())(jnp.ones(3)).tolist() == [2.0, 2.0, 2.0]


def test_scalar_int_arrays_bind_sizes():
    @fn
    def split[D](x: f32[D], heads: int) -> f32:
        assert D % heads == 0
        return x.sum()

    with pytest.raises(expex.ShapeError, match="D % heads == 0"):
        split(jnp.ones(4), jnp.array(3))


def test_jitted_methods_bound_to_instances_work_with_tools():
    @expex.struct
    class S[D]:
        w: f32[D]
        heads: int = 2

        @jax.jit
        @fn
        def jitted(self, x: f32[D]) -> f32[self.heads]:
            assert D >= self.heads
            return x[: self.heads] * self.w[: self.heads]

        @jax.jit
        @fn
        def __call__(self, x: f32[D]) -> f32[D]:
            return x * self.w

    s = S(jnp.ones(3))
    assert expex.prove(s.jitted).ok
    assert "S.jitted" in str(expex.debug.shapes(s.jitted, jnp.ones(3)))
    assert s.lift(jnp.ones((5, 3))).shape == (5, 3)  # jitted __call__ still lifts


def test_struct_subclasses_keep_their_fields_and_lift():
    @expex.struct
    class Base[D]:
        w: f32[D]

        def __call__(self, x: f32[D]) -> f32[D]:
            return x * self.w

    @expex.struct
    class Child(Base):
        b: f32[3]  # type: ignore[valid-type]

    c = Child(jnp.ones(3), jnp.zeros(3))
    assert c.b.shape == (3,)
    assert c.lift(jnp.ones((4, 3))).shape == (4, 3)


def test_ellipsis_is_proved_at_several_ranks():
    @fn
    def first[D](x: f32[..., D]) -> f32[D]:
        return x[0]

    assert not expex.prove(first).ok


def test_string_sizes_in_struct_fields_are_reported():
    @expex.struct
    class Q[D]:
        w: f32[D]
        v: f32["D"]  # type: ignore  # noqa: F821

        def __call__(self) -> f32[D]:
            return self.w

    with pytest.raises(TypeError, match="Q.v.*type parameters"):
        Q(jnp.ones(3), jnp.ones(5))()


def test_traced_values_in_static_fields_are_refused():
    @expex.struct
    class M[D]:
        w: f32[D]
        k: int

    with pytest.raises(expex.StageError, match="M.k is a static field"):
        jax.jit(lambda k: M(jnp.ones(3), k))(2)


def test_number_parameters_check_kind_and_literal_and_enum_values():
    import enum
    from typing import Literal

    class Mode(enum.Enum):
        A = 1

    @fn
    def f(x: f32[3], n: int, z: complex = 1j, lit: Literal["a", "b"] = "a", m: Mode = Mode.A) -> f32[3]:
        return x

    f(jnp.ones(3), 1, 2)  # an int is a fine complex
    for kwargs, msg in [
        ({"n": jnp.bool_(True)}, "integer scalar"),
        ({"n": 1, "lit": "c"}, "one of"),
        ({"n": 1, "m": 1}, "expected Mode"),
    ]:
        with pytest.raises(expex.ShapeError, match=msg):
            f(jnp.ones(3), **kwargs)

    @fn
    def g(x: f32[3], n: int) -> f32[3]:
        return x * n

    with pytest.raises(expex.ShapeError, match="expected an integer scalar"):
        jax.jit(g)(jnp.ones(3), 2.0)  # a float, not "mark it static"


def test_constraint_on_an_output_size_is_checked_after_the_call():
    @fn
    def top[T, K](x: f32[T]) -> f32[K]:
        assert K <= T  # K only appears in the result
        return jnp.concatenate([x, x])

    with pytest.raises(expex.ShapeError, match=r"constraint `K <= T` is violated \(K=6, T=3\)"):
        top(jnp.ones(3))


def test_quoted_annotations_under_future_import_are_read():
    import typing

    class Poly[*B, K](typing.NamedTuple):
        c: "f32[*B, K]"

        @fn
        def pad_to(self, m: expex.Static[int]) -> "Poly[*B, m]":
            return Poly(jnp.pad(self.c, [(0, 0)] * (self.c.ndim - 1) + [(0, m - self.c.shape[-1])]))

    assert "-> Poly[*B, m]" in repr(Poly.pad_to)
    assert Poly(jnp.ones((2, 3))).pad_to(5).c.shape == (2, 5)

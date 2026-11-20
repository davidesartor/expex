"""Implicit PRNG keys (``random=True``)."""

import pytest

import jax
import jax.numpy as jnp

import typax as tx
from typax import Key, StageError, Static, f32, fn


@fn(random=True)
def noise[D](x: f32[D]) -> f32[D]:
    return x + jax.random.normal(tx.key(), x.shape)


def test_fresh_key_each_call_and_reproducible_with_seed():
    tx.seed(0)
    a, b = noise(jnp.zeros(3)), noise(jnp.zeros(3))
    assert not jnp.allclose(a, b)
    tx.seed(0)
    a2, b2 = noise(jnp.zeros(3)), noise(jnp.zeros(3))
    assert jnp.allclose(a, a2) and jnp.allclose(b, b2)


def test_several_draws_inside_one_call_differ():
    @fn(random=True)
    def two() -> tuple[f32, f32]:
        return jax.random.normal(tx.key()), jax.random.normal(tx.key())

    a, b = two()
    assert a != b


def test_explicit_key_scope():
    with tx.rng(42):
        a = noise(jnp.zeros(3))
    with tx.rng(jax.random.key(42)):
        b = noise(jnp.zeros(3))
    assert jnp.allclose(a, b)


def test_nested_random_functions_get_distinct_keys():
    @fn(random=True)
    def outer[D](x: f32[D]) -> tuple[f32[D], f32[D]]:
        return noise(x), noise(x)

    a, b = outer(jnp.zeros(3))
    assert not jnp.allclose(a, b)


def test_plain_jit_without_a_key_is_refused():
    with pytest.raises(StageError, match="frozen into a constant"):
        jax.jit(noise)(jnp.zeros(3))


def test_plain_jit_with_explicit_key_scope():
    @jax.jit
    def step(x, key):
        with tx.rng(key):
            return noise(x)

    a = step(jnp.zeros(3), jax.random.key(0))
    b = step(jnp.zeros(3), jax.random.key(1))
    assert not jnp.allclose(a, b)
    with tx.rng(jax.random.key(0)):
        assert jnp.allclose(a, noise(jnp.zeros(3)))  # same result eagerly


def test_owned_jit_gets_fresh_randomness_without_retracing():
    @fn(jit=True, random=True)
    def jnoise[D](x: f32[D]) -> f32[D]:
        return x + jax.random.normal(tx.key(), x.shape)

    outs = [jnoise(jnp.zeros(3)) for _ in range(4)]
    assert len({tuple(o.tolist()) for o in outs}) == 4
    assert len(jnoise.traces) == 1
    tx.seed(7)
    a = jnoise(jnp.zeros(3))
    tx.seed(7)
    assert jnp.allclose(a, jnoise(jnp.zeros(3)))


def test_lift_gives_each_example_its_own_key():
    out = noise.lift(jnp.zeros((5, 3)))
    assert out.shape == (5, 3)
    assert len({tuple(r.tolist()) for r in out}) == 5


def test_random_functions_are_provable():
    assert tx.prove(noise).status == "proved"

    @fn
    def caller[D](x: f32[D]) -> f32[D]:
        return noise(x)  # not itself random, but calls one

    assert tx.prove(caller).status == "proved"


def test_keys_agree_between_eager_and_staged_control_flow():
    @fn(random=True)
    def walk[T](steps: f32[T]) -> f32[T]:
        pos = 0.0
        out = []
        for s in steps:
            if s > 0:
                pos = pos + s * jax.random.normal(tx.key())
            else:
                pos = pos - jax.random.uniform(tx.key())
            out.append(pos)
        return jnp.asarray(out)

    steps = jnp.array([1.0, -1.0, 2.0, 0.5, -3.0])
    with tx.rng(3):
        eager = walk(steps)

    @jax.jit
    def run(steps, key):
        with tx.rng(key):
            return walk(steps)

    staged = run(steps, jax.random.key(3))
    assert jnp.allclose(eager, staged, atol=1e-6)
    assert "scan" in "".join(walk.flow.values())
    assert len(set(eager.tolist())) == 5


def test_independent_random_loop_uses_vmap_with_distinct_keys():
    @fn(random=True)
    def samples[T](x: f32[T]) -> f32[T]:
        out = []
        for v in x:
            out.append(v + jax.random.normal(tx.key()))
        return jnp.asarray(out)

    x = jnp.zeros(6)
    with tx.rng(1):
        eager = samples(x)
    staged = jax.jit(lambda x, k: _scoped(samples, k, x))(x, jax.random.key(1))
    assert jnp.allclose(eager, staged, atol=1e-6)
    assert len(set(staged.tolist())) == 6
    assert "vmap" in "".join(samples.flow.values())


def _scoped(f, key, *args):
    with tx.rng(key):
        return f(*args)

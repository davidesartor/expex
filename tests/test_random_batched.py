"""Implicit PRNG keys (``random=True``) and compile-once batching (``.batched``)."""

import pytest

import jax
import jax.numpy as jnp

import barnacle as bn
from barnacle import Key, StageError, Static, f32, fn


@fn(random=True)
def noise[D](x: f32[D]) -> f32[D]:
    return x + jax.random.normal(bn.key(), x.shape)


def test_fresh_key_each_call_and_reproducible_with_seed():
    bn.seed(0)
    a, b = noise(jnp.zeros(3)), noise(jnp.zeros(3))
    assert not jnp.allclose(a, b)
    bn.seed(0)
    a2, b2 = noise(jnp.zeros(3)), noise(jnp.zeros(3))
    assert jnp.allclose(a, a2) and jnp.allclose(b, b2)


def test_several_draws_inside_one_call_differ():
    @fn(random=True)
    def two() -> tuple[f32, f32]:
        return jax.random.normal(bn.key()), jax.random.normal(bn.key())

    a, b = two()
    assert a != b
    ks = bn.keys(4)
    assert ks.shape == (4,)


def test_explicit_key_scope():
    with bn.rng(42):
        a = noise(jnp.zeros(3))
    with bn.rng(jax.random.key(42)):
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
        with bn.rng(key):
            return noise(x)

    a = step(jnp.zeros(3), jax.random.key(0))
    b = step(jnp.zeros(3), jax.random.key(1))
    assert not jnp.allclose(a, b)
    with bn.rng(jax.random.key(0)):
        assert jnp.allclose(a, noise(jnp.zeros(3)))  # same result eagerly


def test_owned_jit_gets_fresh_randomness_without_retracing():
    @fn(jit=True, random=True)
    def jnoise[D](x: f32[D]) -> f32[D]:
        return x + jax.random.normal(bn.key(), x.shape)

    outs = [jnoise(jnp.zeros(3)) for _ in range(4)]
    assert len({tuple(o.tolist()) for o in outs}) == 4
    assert len(jnoise.traces) == 1
    bn.seed(7)
    a = jnoise(jnp.zeros(3))
    bn.seed(7)
    assert jnp.allclose(a, jnoise(jnp.zeros(3)))


def test_lift_gives_each_example_its_own_key():
    out = noise.lift(jnp.zeros((5, 3)))
    assert out.shape == (5, 3)
    assert len({tuple(r.tolist()) for r in out}) == 5


def test_random_functions_are_provable():
    assert bn.prove(noise).status == "proved"

    @fn
    def caller[D](x: f32[D]) -> f32[D]:
        return noise(x)  # not itself random, but calls one

    assert bn.prove(caller).status == "proved"


def test_keys_agree_between_eager_and_staged_control_flow():
    @fn(random=True)
    def walk[T](steps: f32[T]) -> f32[T]:
        pos = 0.0
        out = []
        for s in steps:
            if s > 0:
                pos = pos + s * jax.random.normal(bn.key())
            else:
                pos = pos - jax.random.uniform(bn.key())
            out.append(pos)
        return jnp.asarray(out)

    steps = jnp.array([1.0, -1.0, 2.0, 0.5, -3.0])
    with bn.rng(3):
        eager = walk(steps)

    @jax.jit
    def run(steps, key):
        with bn.rng(key):
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
            out.append(v + jax.random.normal(bn.key()))
        return jnp.asarray(out)

    x = jnp.zeros(6)
    with bn.rng(1):
        eager = samples(x)
    staged = jax.jit(lambda x, k: _scoped(samples, k, x))(x, jax.random.key(1))
    assert jnp.allclose(eager, staged, atol=1e-6)
    assert len(set(staged.tolist())) == 6
    assert "vmap" in "".join(samples.flow.values())


def _scoped(f, key, *args):
    with bn.rng(key):
        return f(*args)


# ------------------------------------------------------------------- batched


@fn
def score[T, D](x: f32[T, D], w: f32[D]) -> f32[T]:
    return jnp.tanh(x @ w)


def test_batched_matches_lift_for_any_batch_size_with_one_compile():
    run = score.batched(chunk=8)
    w = jnp.arange(3.0)
    for n in (1, 3, 8, 9, 16, 21):
        xs = jax.random.normal(jax.random.key(n), (n, 5, 3))
        assert jnp.allclose(run(xs, w), score.lift(xs, w), atol=1e-6), n
    assert run.traces == 1


def test_batched_sizes_bound_the_number_of_compiles():
    run = score.batched(sizes=(1, 4, 16))
    w = jnp.ones(3)
    for n in range(1, 40):
        xs = jnp.ones((n, 5, 3)) * n
        assert run(xs, w).shape == (n, 5)
    assert run.traces <= 3


def test_batched_layouts_and_broadcasting():
    run = score.batched(chunk=4)
    xs = jax.random.normal(jax.random.key(0), (6, 5, 3))
    ws = jax.random.normal(jax.random.key(1), (6, 3))
    assert jnp.allclose(run(xs, ws), score.lift(xs, ws), atol=1e-6)
    assert jnp.allclose(run(xs[0], ws), score.lift(xs[0], ws), atol=1e-6)
    # several batch axes
    x2 = jax.random.normal(jax.random.key(2), (2, 1, 5, 3))
    w2 = jax.random.normal(jax.random.key(3), (3, 3))
    out = run(x2, w2)
    assert out.shape == (2, 3, 5)
    assert jnp.allclose(out, score.lift(x2, w2), atol=1e-6)
    # no batch at all: a plain call
    assert run(xs[0], ws[0]).shape == (5,)


def test_batched_static_and_struct_arguments():
    @fn
    def f[D](x: f32[D], n: Static[int], opts: tuple[f32[D], f32] | None = None) -> f32[D]:
        y = x * n
        if opts is not None:
            y = y * opts[0] + opts[1]
        return y

    run = f.batched(chunk=4)
    xs = jnp.ones((6, 3))
    assert jnp.allclose(run(xs, 2), 2.0)
    assert jnp.allclose(run(xs, 3), 3.0)
    assert jnp.allclose(run(xs, 2, (jnp.ones(3) * 5, jnp.float32(1.0))), 11.0)
    assert run.traces == 3  # one per distinct static configuration
    run(jnp.ones((9, 3)), 2)
    assert run.traces == 3


def test_batched_random_and_keys():
    run = noise.batched(chunk=4)
    out = run(jnp.zeros((10, 3)))
    assert out.shape == (10, 3) and len({tuple(r.tolist()) for r in out}) == 10
    assert run.traces == 1

    @fn
    def explicit[D](x: f32[D], key: Key) -> f32[D]:
        return x + jax.random.normal(key, x.shape)

    run2 = explicit.batched(chunk=4)
    out2 = run2(jnp.zeros((6, 3)), jax.random.key(0))
    assert len({tuple(r.tolist()) for r in out2}) == 6


def test_batched_errors():
    with pytest.raises(TypeError, match="exactly one of"):
        score.batched()
    with pytest.raises(ValueError, match="positive ints"):
        score.batched(chunk=0)

    @fn
    def f[D](x: f32[D], cfg) -> f32[D]:
        return x

    with pytest.raises(TypeError, match="must be hashable"):
        f.batched(chunk=2)(jnp.ones((3, 2)), jnp.ones(2))

"""What mapped calls lower to: checked on the emitted program, not on timings.

A layer written for one element and mapped over several axes (a per-point
layer in a per-ray function, each mapped over its batch, or one call with two
extra leading axes) must give XLA dots with one free axis per operand: their
gradients then contract over one axis, and XLA:CPU does not transpose whole
activations to compute them (about twice the time of the flattened dot).
"""

import re

import jax
import jax.extend.core
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import expex
from expex import f32, fn


@expex.struct
class Dense[I, O]:
    kernel: f32[I, O] = jax.nn.initializers.glorot_uniform()
    bias: f32[O] = 0.0

    def __call__(self, x: f32[I]) -> f32[O]:
        return jax.nn.relu(x @ self.kernel + self.bias)


@fn
def per_ray[T](d: Dense, points: f32[T, 16]) -> f32:
    return d(d(points)).sum()  # per point: mapped over the T points


def plain(d: Dense, x: jax.Array) -> jax.Array:
    flat = x.reshape(-1, 16)
    return jax.nn.relu(jax.nn.relu(flat @ d.kernel + d.bias) @ d.kernel + d.bias).sum()


D = Dense(jr.key(0), I=16, O=16)
X = jr.normal(jr.key(1), (6, 5, 16))  # 6 rays of 5 points


def dots(f, *args) -> list:
    """Each dot_general of the program: (lhs shape, rhs shape, dimension numbers)."""
    jaxpr = jax.make_jaxpr(f)(*args)
    out = []

    def walk(j):
        for e in j.eqns:
            if e.primitive.name == "dot_general":
                out.append((e.invars[0].aval.shape, e.invars[1].aval.shape, e.params["dimension_numbers"]))
            for p in e.params.values():
                for sub in p if isinstance(p, (tuple, list)) else (p,):
                    if isinstance(sub, jax.extend.core.ClosedJaxpr):
                        walk(sub.jaxpr)
                    elif isinstance(sub, jax.extend.core.Jaxpr):
                        walk(sub)

    walk(jaxpr.jaxpr)
    return out


def compiled_ops(f, *args) -> list[str]:
    """The ops of the optimized XLA program, with their result types."""
    text = jax.jit(f).lower(*args).compile().as_text()
    return re.findall(r"= (\S+) (transpose|dot)\(", text)


def test_nested_mapped_calls_give_dots_with_one_free_axis():
    grad = jax.grad(lambda d, x: per_ray(d, x).sum())
    for lhs, rhs, ((lc, rc), (lb, rb)) in dots(grad, D, X):
        assert not lb and not rb
        assert len(lhs) == 2 and len(rhs) == 2, (lhs, rhs)  # [rays * points, 16], never [rays, points, 16]
    np.testing.assert_allclose(grad(D, X).kernel, jax.grad(plain)(D, X).kernel, rtol=1e-5)


def test_nested_mapped_gradient_has_no_transposes():
    ops = compiled_ops(jax.grad(lambda d, x: per_ray(d, x).sum()), D, X)
    assert [t for t, op in ops if op == "transpose"] == []
    assert [t for t, op in ops if op == "dot"]  # the dots are there, on [30, 16]


def test_one_call_with_two_extra_axes_maps_over_one():
    out = D(X)
    assert out.shape == (6, 5, 16)
    np.testing.assert_array_equal(out, jax.nn.relu(X @ D.kernel + D.bias))
    assert [(lhs, rhs) for lhs, rhs, _ in dots(lambda x: D(x), X)] == [((30, 16), (16, 16))]
    np.testing.assert_array_equal(D.vectorize(X), out)


def test_two_extra_axes_with_a_key_per_element():
    @expex.random
    @fn
    def noisy(x: f32[3]) -> f32[3]:
        return x + jr.normal(expex.key(), (3,))

    x = jnp.zeros((4, 2, 3))
    a, b = noisy(x, key=jr.key(7)), noisy(x, key=jr.key(7))
    np.testing.assert_array_equal(a, b)
    assert len({tuple(np.asarray(r)) for r in a.reshape(-1, 3)}) == 8  # each element its own key
    keys = jr.split(jr.key(7), (4, 2))  # the keys the elements get: one per element, in order
    np.testing.assert_array_equal(a[1, 0], noisy(x[1, 0], key=keys[1, 0]))


def test_arguments_with_some_of_the_axes_still_broadcast():
    @fn
    def add(x: f32[3], y: f32[3]) -> f32[3]:
        return x + y

    x, y = jnp.ones((4, 1, 3)), jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(add(x, y), x + y)


def test_rhs_mapped_dot_and_a_mapped_axis_not_next_to_the_free_ones():
    @fn
    def left(w: f32[4, 16], x: f32[5, 16]) -> f32[4, 5]:
        return w @ x.T  # the mapped operand on the right

    w, x = jr.normal(jr.key(2), (4, 16)), jr.normal(jr.key(3), (3, 5, 16))
    np.testing.assert_allclose(left(w, x), jnp.einsum("ad,bcd->bac", w, x), rtol=1e-5)

    @fn
    def inner(x: f32[16, 5], k: f32[16, 7]) -> f32[5, 7]:
        return x.T @ k  # contracts the leading axis: the free axis is not next to the mapped one

    xs = jr.normal(jr.key(4), (3, 16, 5))
    k = jr.normal(jr.key(5), (16, 7))
    np.testing.assert_allclose(inner(xs, k), jnp.einsum("bdc,de->bce", xs, k), rtol=1e-5)
    grads = jax.grad(lambda k: inner(xs, k).sum())(k)
    np.testing.assert_allclose(grads, jax.grad(lambda k: jnp.einsum("bdc,de->bce", xs, k).sum())(k), rtol=1e-5, atol=1e-5)


def test_plain_vmap_outside_expex_keeps_jax_rule():
    def f(x, w):
        return jax.lax.dot_general(x, w, (((0,), (0,)), ((), ())))

    w = jnp.ones((16, 16))
    nested = jax.vmap(jax.vmap(f, in_axes=(0, None)), in_axes=(0, None))
    assert [len(lhs) for lhs, _, _ in dots(nested, X, w)] == [3]  # JAX's own rule: [6, 5, 16]
    # (`x @ w` is a jitted jnp function, whose batched form JAX caches: it may be the one built in a mapped call)

"""Functions defined inside an ``@fn`` body see its sizes, as a value and as an axis."""

import jax
import jax.numpy as jnp
import pytest

import expex
from expex import f32, fn


@fn
def best[N, D](xs: f32[N, D]) -> f32[N]:
    def score(row):
        bias = jnp.zeros(D)  # D as a size
        return (row + bias).sum(D) / D  # as an axis, and as a value

    return jax.vmap(score)(xs)


def test_a_nested_function_sees_the_sizes():
    xs = jnp.arange(12.0).reshape(3, 4)
    assert jnp.allclose(best(xs), xs.mean(1))
    assert jnp.allclose(jax.jit(best)(xs), xs.mean(1))
    assert best.vectorize(jnp.ones((2, 3, 4))).shape == (2, 3)
    assert expex.prove(best).ok


def test_lambdas_and_comprehensions():
    @fn
    def lam[N, D](xs: f32[N, D]) -> f32[N]:
        f = lambda r: r.sum(D) * D
        return jax.vmap(f)(xs)

    @fn
    def comp[N, D](xs: f32[N, D]) -> f32[N]:
        return jnp.stack([xs[i].sum(D) + D for i in range(N)])

    xs = jnp.ones((3, 4))
    assert lam(xs).tolist() == [16.0] * 3 and expex.prove(lam).ok
    assert comp(xs).tolist() == [8.0] * 3 and expex.prove(comp).ok


def test_the_sizes_are_those_of_the_call_it_was_made_in():
    @fn
    def maker[D](x: f32[D]):
        return lambda y: y.sum(D) + D

    g = maker(jnp.ones(4))  # called after maker returned
    assert g(jnp.ones(4)) == 8.0

    @fn
    def apply[K](y: f32[K], h) -> f32[K]:
        return h(y)

    @fn
    def outer[N, D](xs: f32[N, D]) -> f32[N]:
        def scale(y):
            return y * D  # D of outer's call, though apply's call is running

        return apply(xs[:, 0], scale)

    assert outer(jnp.ones((3, 4))).tolist() == [4.0] * 3
    assert expex.prove(outer).ok


def test_its_own_variables_hide_the_enclosing_ones():
    @fn
    def f[N, D](xs: f32[N, D]) -> f32[N]:
        def score(xs):  # a row: not the annotated argument of the same name
            return xs.sum(D)

        return jax.vmap(score)(xs)

    @fn
    def g[N, D](xs: f32[N, D]) -> f32[N]:
        def score(row, D=2):  # its own D
            return row.sum() * D

        return jax.vmap(score)(xs)

    xs = jnp.ones((3, 4))
    assert f(xs).tolist() == [4.0] * 3
    assert g(xs).tolist() == [8.0] * 3


def test_declared_sizes_too():
    @fn
    def f(x: f32[R:6, C:7]) -> f32[R]:
        def row_total(r):
            return r.sum(C) + jnp.zeros(()) * R

        return jax.vmap(row_total)(x)

    assert f(jnp.ones((6, 7))).tolist() == [7.0] * 6
    assert expex.prove(f).ok


def test_a_nested_function_otherwise_stays_plain_python():
    @fn
    def f[D](x: f32[D], flag: bool) -> f32[D]:
        def pick(v):
            if flag and v.shape[0] == D:  # a plain `if` / `and`: not staged
                return v
            return -v

        return pick(x)

    src = f.converted_source
    assert "if flag and v.shape[0] == __expex_flow__.size_(D, __bn_call__):" in src
    assert f(jnp.ones(3), True).tolist() == [1.0] * 3

    @fn
    def g[D](x: f32[D]) -> f32[D]:
        def untouched(v):  # uses no size: left exactly as written
            return v + 1 if v is not None else v

        return untouched(x) if D > 0 else x

    assert "def untouched(v):\n        return v + 1 if v is not None else v" in g.converted_source


def plain_helper(row):
    return row.sum() * D  # noqa: F821  # D is a size of the caller, not a global


def test_a_module_level_helper_gets_a_hint():
    @fn
    def f[N, D](xs: f32[N, D]) -> f32[N]:
        return jax.vmap(plain_helper)(xs)

    with pytest.raises(NameError) as e:
        f(jnp.ones((3, 4)))
    notes = "\n".join(getattr(e.value, "__notes__", ()))
    assert "D is a size of" in notes and "`plain_helper`" in notes
    assert "define `plain_helper` inside" in notes and "@expex.fn" in notes


def test_a_body_left_unconverted_says_why():
    @fn(convert=False)
    def f(x: f32[R:6]) -> f32[R]:
        return x + jnp.zeros(R)

    with pytest.raises(NameError) as e:
        f(jnp.ones(6))
    assert "the body cannot use it by name because convert=False" in "\n".join(e.value.__notes__)

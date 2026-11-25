"""Bugs found while porting NeRF (examples/nerf/)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

import expex
from expex import f32, fn


@expex.struct
class Act[D]:
    w: f32[D]
    act: Callable = jax.nn.relu  # collections.abc.Callable, unsubscripted


def test_a_bare_collections_abc_callable_field_is_static():
    # it was traced as data, so jax.jit of a function taking the struct failed
    out = jax.jit(lambda m: m.act(m.w))(Act(-jnp.ones(3)))
    assert out.tolist() == [0.0, 0.0, 0.0]


@fn
def encode[X](x: f32[X], lo: int, hi: int) -> f32[X * (1 + hi - lo)]:
    assert lo <= hi
    return jnp.concatenate([x, *[x * 2.0**i for i in range(lo, hi)]])


def test_the_prover_skips_static_values_the_asserts_rule_out():
    # lo=2, hi=1 violates `lo <= hi`: that combination made the whole report "unchecked"
    report = expex.prove(encode)
    assert report.status == "proved", str(report)
    assert report.cases == 3


@expex.struct
class Lin[I, O]:
    w: f32[I, O]


@expex.struct
class Encoded[P, W]:
    first: Lin[P, W]
    skip: Lin[W + P, W]
    deg: int = 2

    def __call__(self, x: f32[3]) -> f32[W]:
        assert P == 3 * (1 + 2 * self.deg)  # P is fixed by a static field
        e = jnp.concatenate([x, *[f(2.0**i * x) for i in range(self.deg) for f in (jnp.sin, jnp.cos)]])
        return jnp.concatenate([e @ self.first.w, e]) @ self.skip.w


def test_a_size_an_assert_fixes_from_static_fields_is_proved():
    # the prover kept P symbolic (and tried random concrete sizes, none of them 15)
    model = Encoded(Lin(jnp.ones((15, 4))), Lin(jnp.ones((19, 4))))
    assert model(jnp.ones(3)).shape == (4,)
    report = expex.prove(Encoded.__call__)
    assert report.status == "proved", str(report)


@expex.struct
class Viewed[W, V]:
    w: f32[W, W]
    condition: Lin[W + V, W]  # V only in a sum
    deg: int = 2

    def __call__(self, h: f32[W]) -> f32[W]:
        assert V == 3 * self.deg
        return jnp.concatenate([h @ self.w, jnp.ones(V)]) @ self.condition.w

    def twice(self, h: f32[W]) -> f32[W]:
        return self(self(h))  # no argument has V on its own


def test_a_size_in_a_sum_is_solved_when_the_rest_is_symbolic():
    # `twice` passes self on: its `W + V` axis is `W + 6` with W symbolic, and
    # V could not be solved for ("cannot determine V from `W + V`")
    assert Viewed(jnp.ones((4, 4)), Lin(jnp.ones((10, 4)))).twice(jnp.ones(4)).shape == (4,)
    report = expex.prove(Viewed.twice)
    assert report.status == "proved", str(report)


@expex.struct
class Cat[W, V]:
    w: f32[W + V, W]  # V appears only inside a sum

    def __call__(self, x: f32[W], v: f32[V]) -> f32[W]:
        return jnp.concatenate([x, v]) @ self.w

    def first(self, x: f32[W]) -> f32[W]:
        return x @ self.w[:W]


@expex.struct
class Outer[W, V]:
    cat: Cat[W, V]

    def __call__(self, x: f32[W]) -> f32[W]:
        return self.cat.first(x)  # nothing passed in has V on its own


def test_the_tools_solve_a_size_used_only_in_a_sum():
    # the nested call's `W + V` (symbolic) could not be solved: "cannot determine V"
    for tool in (expex.cost, expex.debug.shapes, expex.diagram):
        tool(Outer.__call__)
    assert str(expex.cost(Outer.__call__).flops) == "2·W²"
    s = expex.summary(Outer(Cat(jnp.ones((7, 4)))))
    assert s.sizes == {"W": 4, "V": 3}  # was V=? (only in `w: f32[W + V, W]`)
    assert "Cat[W=4, V=3]" in str(s) and "no FLOP count" not in str(s)


def test_the_tools_on_nerf_use_the_sizes_its_asserts_fix():
    # V (27) and P (63) are fixed by `assert V == 3 * (1 + 2 * self.deg_view)`: traced
    # as symbols, `W + V` disagreed with the 27-wide view the body computes
    # ("axis 0 should be W + V = W + 27 ... but is W + V")
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples" / "nerf"))
    from nerf import NeRF

    for f in (NeRF.render, NeRF.__call__, NeRF.shade):
        c = expex.cost(f)
        assert "V" not in c.flops.names() and "P" not in c.flops.names(), c
    expex.diagram(NeRF.render)
    expex.debug.shapes(NeRF.shade)
    import jax.random as jr

    s = expex.summary(NeRF.create(jr.key(0), depth=4, width=8, width_condition=4, skip_layer=2))
    assert "MLP[P=63, V=27, W=8, C=4]" in str(s)

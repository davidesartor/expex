"""Library issues found while porting Llama 2 and Pgx (examples/llama/, examples/pgx/)."""

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import expex
from expex import Bool, f32, fn, i32


# 1. jit=True keeps the Fn API ------------------------------------------------------------


@fn(jit=True)
def double[N](x: f32[N]) -> f32[N]:
    for i in range(N):
        x = x + 0.0 * i
    return 2 * x


@expex.struct
class Scale[D]:
    w: f32[D]

    @fn(jit=True)
    def apply(self, x: f32[D]) -> f32[D]:
        return x * self.w


def test_jit_keeps_vectorize_flow_and_wrapped():
    assert double.vectorize(jnp.ones((4, 3))).shape == (4, 3)
    assert isinstance(double.__wrapped__, expex.Fn)
    assert "lax.scan" in str(dict(double.flow))
    assert "f32[N]" in double.info()


def test_jit_vectorize_is_compiled():
    double.vectorize(jnp.ones((2, 5)))
    assert double.vectorize._cache_size() >= 1  # a jax.jit function


def test_jitted_method_vectorizes_bound():
    m = Scale(jnp.arange(3.0))
    assert m.apply(jnp.ones(3)).tolist() == [0.0, 1.0, 2.0]
    assert m.apply.vectorize(jnp.ones((5, 3))).shape == (5, 3)
    assert expex.prove(m.apply).ok and expex.prove(Scale.apply).ok


# 2. a traced `if` traces each branch once ------------------------------------------------

_traced: list[int] = []


def _heavy(x):
    _traced.append(1)
    return jnp.sin(x).sum()


@fn
def branch(x: f32[3], p: f32) -> f32:
    if p > 0:
        y = _heavy(x)
    else:
        y = jnp.full((), 7)
    return y


@fn
def ternary(x: f32[3], p: f32) -> f32:
    return _heavy(x) if p > 0 else -1


@pytest.mark.parametrize("f", [branch, ternary])
def test_each_branch_traced_once(f):
    _traced.clear()
    jaxpr = jax.make_jaxpr(f)(jnp.ones(3), 1.0)
    assert len(_traced) == 1 and str(jaxpr).count("cond[") == 1
    assert float(jax.jit(f)(jnp.ones(3), -1.0)) in (7.0, -1.0)
    assert float(jax.jit(f)(jnp.ones(3), 1.0)) == pytest.approx(3 * float(jnp.sin(1.0)))


# 3. chained and tuple subscript targets --------------------------------------------------


@fn
def place(board: i32[8, 8]) -> i32[8, 8]:
    board[3, 4] = board[4, 3] = 1
    board[3, 3], board[4, 4] = 2, 3
    return board


def test_multi_target_subscript_assignment():
    for run in (place, jax.jit(place)):
        b = run(jnp.zeros((8, 8), jnp.int32))
        assert (int(b[3, 4]), int(b[4, 3]), int(b[3, 3]), int(b[4, 4])) == (1, 1, 2, 3)


# 4. debug.ops counts the program's own operations ----------------------------------------


@fn
def plain_if(x: f32[3], p: f32) -> f32:
    if p > 0:
        y = x.sum()
    else:
        y = x.max()
    return y


def test_ops_has_no_key_bookkeeping():
    report = expex.debug.ops(plain_if, jnp.ones(3), 1.0)
    assert not any(k.startswith("random") for k, _ in report.by_op)
    assert not any(w.startswith("<unknown>") for w, _, _ in report.by_line)


# 5. a size defined in the body, not in the signature -------------------------------------


@fn
def regroup[H, T, K](q: f32[H, T, K], groups: int) -> f32[H, T, K]:
    G = groups
    R = H // G
    assert H % groups == 0
    return expex.rearrange(q, to=(G, R, T, K)).reshape(H, T, K)


def test_body_size_used_as_a_name():
    assert regroup(jnp.ones((4, 3, 2)), 2).shape == (4, 3, 2)
    assert expex.prove(regroup).ok


# 6. einsum / rearrange results carry their axis names ------------------------------------


@fn
def masked[G, R, T, S, K](q: f32[G, R, T, K], k: f32[G, S, K], m: Bool[T, S]) -> f32[G, R, T, S]:
    return jax.nn.softmax(jnp.where(m, expex.einsum(q, k, to=(G, R, T, S)), -jnp.inf), axis=S)


@fn
def pooled[G, R, T, S, K](q: f32[G, R, T, K], k: f32[G, S, K]) -> f32[G, R, T]:
    scores = expex.einsum(q, k, to=(G, R, T, S))
    return scores.sum(S)


def test_einsum_result_names_when_sizes_are_equal():
    q, k = jr.normal(jr.key(0), (2, 3, 4, 5)), jr.normal(jr.key(1), (2, 4, 5))  # T == S
    m = jnp.tril(jnp.ones((4, 4), bool))
    out = masked(q, k, m)
    assert jnp.allclose(out.sum(-1), 1.0) and float(out[0, 0, 0, 1]) == 0.0
    assert jnp.allclose(pooled(q, k), jnp.einsum("grtk,gsk->grt", q, k), atol=1e-5)


# 7. an unrolled loop says so -------------------------------------------------------------


@fn
def grow[N](x: f32[N]) -> f32[N]:
    out = [x]
    for i in range(20):
        out.append(out[-1] + 1)
    return out[-1]


@fn
def grow_short[N](x: f32[N]) -> f32[N]:
    out = [x]
    for i in range(3):
        out.append(out[-1] + 1)
    return out[-1]


def test_unrolled_loop_warns_once():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        jax.jit(grow)(jnp.ones(2))
        jax.jit(grow)(jnp.ones(3))
        jax.jit(grow_short)(jnp.ones(2))
    msgs = [str(x.message) for x in w if "unrolled" in str(x.message)]
    assert len(msgs) == 1 and "20 copies" in msgs[0] and "test_port_issues.py" in msgs[0]


# 8. the prover uses divisibility ---------------------------------------------------------


@fn
def half_angles[T, K](x: f32[T, K]) -> f32[T, K]:
    assert K % 2 == 0
    angles = jnp.arange(T)[:, None] * jnp.sin(jnp.arange(0, K, 2) / K)[None, :]
    return x * jnp.tile(jnp.sin(angles), 2)


def test_divisibility_is_proved():
    assert expex.prove(half_angles).status == "proved"


# 9. summary and cost ---------------------------------------------------------------------


@expex.struct
class Attend[D]:
    w: f32[D, D]

    def __call__[T, S](self, x: f32[T, D], memory: f32[S, D]) -> f32[T, D]:
        return jax.nn.softmax((x @ self.w) @ memory.T, axis=-1) @ memory


@expex.struct
class Net[D]:
    attn: Attend[D]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        return self.attn(x, x)


def test_summary_binds_sizes_from_the_call():
    s = expex.summary(Net(Attend(jnp.eye(4))), T=8)
    row = next(r for r in s.rows if r.path == "attn")
    assert "S" in str(row.flops) and row.flops_value is not None
    assert not any("no FLOP count" in n for n in s.notes)


@fn
def rolled[T, K](x: f32[T, K]) -> f32[T, K]:
    return jnp.roll(x, K // 2, axis=1) + x


def test_cost_largest_ignores_symbolic_roll():
    assert "2*K" not in expex.cost(rolled).largest[0]

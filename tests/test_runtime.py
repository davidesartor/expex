"""Call-time checking of arguments and return values."""

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from typax import (
    Arr, Bool, Dim, Etc, Float, Int, Key, ShapeError, StageError, Static, config, f32, fn, i32,
)  # fmt: skip


@fn
def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
    return jnp.einsum("btd,bsd->bts", q, k)


def test_ok():
    assert attend(jnp.ones((2, 3, 4)), jnp.ones((2, 3, 4))).shape == (2, 3, 3)


def test_metadata_preserved():
    assert attend.__name__ == "attend"
    assert "attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]" in repr(attend)


def test_dim_conflict_names_both_arguments():
    with pytest.raises(ShapeError) as e:
        attend(jnp.ones((2, 3, 4)), jnp.ones((5, 3, 4)))
    msg = str(e.value)
    assert "argument `k`" in msg
    assert "B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5." in msg
    assert "declared  k: f32[B, T, D]" in msg
    assert "received     f32[5, 3, 4]" in msg


def test_rank_and_dtype():
    with pytest.raises(ShapeError, match="expected 3 axes, got 2"):
        attend(jnp.ones((2, 3, 4)), jnp.ones((3, 4)))
    with pytest.raises(ShapeError, match="expected dtype f32, got i32"):
        attend(jnp.ones((2, 3, 4)), jnp.ones((2, 3, 4), jnp.int32))
    with pytest.raises(ShapeError, match="expected an array, got str"):
        attend(jnp.ones((2, 3, 4)), "nope")  # type: ignore[arg-type]


def test_shape_error_is_a_type_error():
    assert issubclass(ShapeError, TypeError)


def test_return_checked():
    @fn
    def bad[B, T](x: f32[B, T]) -> f32[B, T]:
        return x[:, 1:]

    with pytest.raises(ShapeError) as e:
        bad(jnp.ones((2, 3)))
    assert "return value" in str(e.value)
    assert "returned     f32[2, 2]" in str(e.value)


def test_checked_under_transformations():
    @fn
    def bad[T](x: f32[T]) -> f32[T]:
        return x[1:]

    with pytest.raises(ShapeError):
        jax.jit(bad)(jnp.ones(3))
    with pytest.raises(ShapeError):
        jax.vmap(bad)(jnp.ones((2, 3)))

    @fn
    def good[T](x: f32[T]) -> f32:
        return x.sum()

    # vmap/grad see per-example shapes, which is what the annotation describes
    assert jax.vmap(good)(jnp.ones((2, 3))).shape == (2,)
    assert jax.grad(good)(jnp.ones(3)).shape == (3,)
    assert jax.jit(good)(jnp.ones(3)).shape == ()


def test_scalar_and_python_scalars():
    @fn
    def scale[N](x: f32[N], a: f32) -> f32[N]:
        return x * a

    scale(jnp.ones(3), 2.0)
    scale(jnp.ones(3), 2)
    scale(jnp.ones(3), jnp.float32(2))
    with pytest.raises(ShapeError, match="expected 0 axes, got 1"):
        scale(jnp.ones(3), jnp.ones(3))

    @fn
    def count(n: i32) -> i32:
        return jnp.asarray(n, jnp.int32)

    count(3)
    with pytest.raises(ShapeError, match="Python float"):
        count(3.0)  # type: ignore[arg-type]

    @fn
    def explicit(x: f32[()]) -> f32[()]:
        return x

    explicit(jnp.float32(1))


def test_dtype_categories():
    @fn
    def f[N](a: Float[N], b: Int[N], c: Bool[N], d: Arr[N]) -> Arr[N]:
        return a

    n = 3
    f(jnp.ones(n), jnp.ones(n, jnp.int32), jnp.ones(n, bool), jnp.ones(n, jnp.uint8))
    f(jnp.ones(n, jnp.float16), jnp.ones(n, jnp.uint8), jnp.ones(n, bool), jnp.ones(n, bool))
    with pytest.raises(ShapeError, match="expected dtype Float, got i32"):
        f(jnp.ones(n, jnp.int32), jnp.ones(n, jnp.int32), jnp.ones(n, bool), jnp.ones(n))
    with pytest.raises(ShapeError, match="expected dtype Int, got Bool"):
        f(jnp.ones(n), jnp.ones(n, bool), jnp.ones(n, bool), jnp.ones(n))


def test_numpy_arrays_accepted():
    @fn
    def f[N](x: f32[N]) -> f32[N]:
        return jnp.asarray(x)

    f(np.ones(3, np.float32))
    f(np.ones(3))  # float64 in NumPy, but JAX will make it float32
    with pytest.raises(ShapeError, match="expected dtype f32, got i64"):
        f(np.ones(3, np.int64))


class RGB(Dim, size=3): ...


class Hidden(Dim): ...


type Two = Literal[2]


def test_fixed_and_named_dims():
    @fn
    def gray[H, W](img: f32[H, W, RGB]) -> f32[H, W]:
        return img.mean(-1)

    gray(jnp.ones((4, 5, 3)))
    with pytest.raises(ShapeError, match="axis 2 should be RGB = 3, but is 4"):
        gray(jnp.ones((4, 5, 4)))

    @fn
    def lit(x: f32[Literal[3], Two], y: f32[Any, Hidden], z: f32[Hidden]) -> f32:
        return x.sum()

    lit(jnp.ones((3, 2)), jnp.ones((9, 5)), jnp.ones(5))
    with pytest.raises(ShapeError, match="axis 1 should be 2, but is 3"):
        lit(jnp.ones((3, 3)), jnp.ones((9, 5)), jnp.ones(5))
    with pytest.raises(ShapeError, match="Hidden is 5"):
        lit(jnp.ones((3, 2)), jnp.ones((9, 5)), jnp.ones(6))


def test_variadic():
    @fn
    def f[*Batch, T](x: f32[*Batch, T], y: f32[*Batch]) -> f32[*Batch]:
        return x.sum(-1) + y

    assert f(jnp.ones((2, 3, 4)), jnp.ones((2, 3))).shape == (2, 3)
    assert f(jnp.ones(4), jnp.float32(1)).shape == ()
    with pytest.raises(ShapeError, match=r"\*Batch is \(2, 3\).*but here it is \(2,\)"):
        f(jnp.ones((2, 3, 4)), jnp.ones((2,)))

    @fn
    def g[D](x: f32[Etc, D], y: f32[D, Etc]) -> f32[Etc]:
        return x

    g(jnp.ones((1, 2, 3)), jnp.ones(3))
    g(jnp.ones(3), jnp.ones((3, 7)))
    with pytest.raises(ShapeError, match="expected at least 1 axes, got 0"):
        g(jnp.float32(1), jnp.ones(3))


def test_two_variadics_rejected():
    with pytest.raises(TypeError, match="at most one variadic"):

        @fn
        def f[*A, *B](x: f32[*A, *B]) -> f32: ...


def test_derived_dims():
    @fn(U="T + 1")
    def pad[B, T, U](x: f32[B, T]) -> f32[B, U]:
        return jnp.pad(x, ((0, 0), (0, 1)))

    assert pad(jnp.ones((2, 3))).shape == (2, 4)

    @fn(U="T + 1")
    def wrong[B, T, U](x: f32[B, T]) -> f32[B, U]:
        return x

    with pytest.raises(ShapeError, match=r"axis 1 should be T \+ 1 = 4 \(with T=3\), but is 3"):
        wrong(jnp.ones((2, 3)))


def test_derived_dim_solved_from_input():
    @fn(U="2 * T")
    def half[T, U](x: f32[U]) -> f32[T]:
        return x[::2]

    assert half(jnp.ones(6)).shape == (3,)
    with pytest.raises(ShapeError, match="no size of T makes `2 \\* T` equal 7"):
        half(jnp.ones(7))


def test_derived_dim_must_be_in_signature():
    with pytest.raises(TypeError, match="`U` does not appear"):

        @fn(U="T + 1")
        def f[T](x: f32[T]) -> f32[T]: ...


def test_where():
    @fn(where="T >= 2")
    def diff[T](x: f32[T]) -> f32[T]:
        return x

    diff(jnp.ones(2))
    with pytest.raises(ShapeError, match=r"constraint `T >= 2` is violated \(T=1\)"):
        diff(jnp.ones(1))

    @fn(where=["K <= T"])
    def top[T, K](x: f32[T]) -> f32[K]:
        return jnp.concatenate([x, x])

    with pytest.raises(ShapeError, match="constraint `K <= T` is violated"):
        top(jnp.ones(3))


class Params[D](NamedTuple):
    w: f32[D, D]
    b: f32[D]


@dataclass
class Layer[I, O]:
    w: f32[O, I]
    name: str = "layer"


def test_structs():
    @fn
    def dense[B, D](p: Params[D], x: f32[B, D]) -> tuple[f32[B, D], f32]:
        return x @ p.w + p.b, jnp.sum(x)

    dense(Params(jnp.ones((3, 3)), jnp.ones(3)), jnp.ones((2, 3)))
    with pytest.raises(ShapeError) as e:
        dense(Params(jnp.ones((3, 3)), jnp.ones(4)), jnp.ones((2, 3)))
    assert "declared  p.b: f32[D]" in str(e.value)
    assert "D is 3 (from `p.w: f32[3, 3]`)" in str(e.value)

    @fn
    def apply[B, I, O](layer: Layer[I, O], x: f32[B, I]) -> f32[B, O]:
        return x @ layer.w.T

    assert apply(Layer(jnp.ones((5, 3))), jnp.ones((2, 3))).shape == (2, 5)
    with pytest.raises(ShapeError, match="I is 3"):
        apply(Layer(jnp.ones((5, 3))), jnp.ones((2, 4)))
    with pytest.raises(ShapeError, match="expected Layer"):
        apply("x", jnp.ones((2, 3)))  # type: ignore[arg-type]


def test_unparametrised_struct_has_private_dims():
    @fn
    def f[D](p: Params, x: f32[D]) -> f32[D]:
        return x

    f(Params(jnp.ones((3, 3)), jnp.ones(3)), jnp.ones(7))  # Params.D != D
    with pytest.raises(ShapeError, match="Params.D is 3"):
        f(Params(jnp.ones((3, 3)), jnp.ones(4)), jnp.ones(7))


def test_containers():
    @fn
    def f[D](xs: list[f32[D]], ys: tuple[f32[D], ...], d: dict[str, f32[D]]) -> tuple[f32[D], i32]:
        return xs[0], jnp.int32(len(xs))

    v = jnp.ones(3)
    f([v, v], (v,), {"a": v})
    with pytest.raises(ShapeError, match=r"declared  d\['b'\]: f32\[D\]"):
        f([v, v], (v,), {"a": v, "b": jnp.ones(4)})
    with pytest.raises(ShapeError, match="expected a list"):
        f((v,), (v,), {})  # type: ignore[arg-type]

    @fn
    def g[D](x: f32[D]) -> tuple[f32[D], f32[D]]:
        return (x,)  # type: ignore[return-value]

    with pytest.raises(ShapeError, match="expected a tuple of 2 elements, got 1"):
        g(v)


def test_optional_and_union():
    @fn
    def f[D](x: f32[D], m: f32[D] | None = None) -> f32[D]:
        return x if m is None else x * m

    v = jnp.ones(3)
    f(v)
    f(v, None)
    f(v, v)
    with pytest.raises(ShapeError, match="D is 3"):
        f(v, jnp.ones(4))

    @fn
    def g[D](x: f32[D] | i32[D]) -> f32[D] | i32[D]:
        return x

    g(v)
    g(jnp.ones(3, jnp.int32))
    with pytest.raises(ShapeError, match="matches none of the alternatives"):
        g(jnp.ones(3, bool))


def test_varargs():
    @fn
    def f[D](*xs: f32[D], **kw: f32[D]) -> f32[D]:
        return sum(xs) + sum(kw.values())  # type: ignore[return-value]

    v = jnp.ones(3)
    f(v, v, a=v)
    with pytest.raises(ShapeError, match=r"xs\[1\]"):
        f(v, jnp.ones(4))
    with pytest.raises(ShapeError, match="declared  a: f32"):
        f(v, a=jnp.ones(4))


def test_static():
    @fn
    def f[N](x: f32[N], n: int, mode: str = "a") -> f32[N]:
        return x * n

    f(jnp.ones(3), 2)
    f(jnp.ones(3), np.int64(2))
    with pytest.raises(ShapeError, match="expected int, got float"):
        f(jnp.ones(3), 2.0)  # type: ignore[arg-type]
    with pytest.raises(ShapeError, match="expected str, got int"):
        f(jnp.ones(3), 2, 3)  # type: ignore[arg-type]
    with pytest.raises(StageError, match="annotate it with a dtype instead"):
        jax.jit(lambda x, n: f(x, n))(jnp.ones(3), 2)
    with pytest.raises(ShapeError, match="to pass arrays, annotate it with a dtype"):
        f(jnp.ones(3), jnp.int32(2))

    @fn
    def g[N](x: f32[N], n: Static[int]) -> f32[N]:  # the explicit form still works
        return x * n

    assert g(jnp.ones(3), 2).tolist() == [2.0, 2.0, 2.0]


def test_key():
    @fn
    def noise[D](x: f32[D], key: Key) -> f32[D]:
        return x + jax.random.normal(key, x.shape)

    noise(jnp.zeros(3), jax.random.key(0))
    noise(jnp.zeros(3), jax.random.PRNGKey(0))
    with pytest.raises(ShapeError, match="expected a PRNG key"):
        noise(jnp.zeros(3), jnp.zeros(3))
    with pytest.raises(ShapeError, match="expected 0 axes, got 1"):
        noise(jnp.zeros(3), jax.random.split(jax.random.key(0), 4))

    @fn
    def many[N](keys: Key[N]) -> f32[N]:
        return jax.vmap(jax.random.normal)(keys)

    assert many(jax.random.split(jax.random.key(0), 4)).shape == (4,)


def test_methods():
    class M:
        def __init__(self, s: float):
            self.s = s

        @fn
        def call[D](self, x: f32[D]) -> f32[D]:
            return x * self.s

    m = M(2.0)
    assert m.call(jnp.ones(3)).shape == (3,)
    assert m.call.lift(jnp.ones((4, 3))).shape == (4, 3)
    with pytest.raises(ShapeError, match="expected 1 axis, got 2"):
        m.call(jnp.ones((3, 3)))


def test_unannotated_is_ignored():
    @fn
    def f(x, y: int = 2):
        return x

    assert f("anything") == "anything"


def test_wrong_arity_is_pythons_error():
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        attend(jnp.ones((2, 3, 4)))  # type: ignore[call-arg]


def test_disable_per_function_and_globally(monkeypatch):
    @fn(checked=False)
    def f[N](x: f32[N]) -> f32[N]:
        return x[1:]

    f(jnp.ones(3))

    @fn
    def g[N](x: f32[N]) -> f32[N]:
        return x[1:]

    monkeypatch.setattr(config, "checked", False)
    g(jnp.ones(3))


def test_bad_annotation_reported_at_definition():
    with pytest.raises(TypeError, match="bad annotation for `x`"):

        @fn
        def f(x: f32[3.5]) -> f32: ...  # type: ignore[valid-type]


def test_string_dims_at_runtime():
    # Not accepted by static checkers, but supported for those who do not use one.
    @fn
    def pad(x: f32["B", "T"]) -> f32["B", "T + 1"]:  # type: ignore  # noqa: F821
        return jnp.pad(x, ((0, 0), (0, 1)))

    assert pad(jnp.ones((2, 3))).shape == (2, 4)


def test_notes_on_exceptions():
    @fn
    def badadd[B, T](x: f32[B, T], y: f32[T, B]) -> f32[B, T]:
        return x + y

    with pytest.raises(TypeError) as e:
        badadd(jnp.ones((2, 3)), jnp.ones((3, 2)))
    notes = "\n".join(e.value.__notes__)
    assert "typax: in " in notes
    assert "badadd(x: f32[2, 3], y: f32[3, 2])  [B=2, T=3]" in notes


def test_nested_call_chain_in_notes():
    @fn
    def inner[N](a: f32[N], b: f32[N]) -> f32[N]:
        return a + b

    @fn
    def outer[B](x: f32[B]) -> f32[B]:
        return inner(x, x[:-1])

    with pytest.raises(ShapeError) as e:
        outer(jnp.ones(3))
    assert "inner(): argument `b`" in str(e.value)
    assert any("outer(x: f32[3])" in n for n in e.value.__notes__)


def test_stage_hint():
    @fn(convert=False)
    def relu[B](x: f32[B]) -> f32[B]:
        if x.sum() > 0:
            return x
        return -x

    relu(jnp.ones(3))  # fine eagerly
    with pytest.raises(jax.errors.TracerBoolConversionError) as e:
        jax.jit(relu)(jnp.ones(3))
    notes = "\n".join(e.value.__notes__)
    assert "`if x.sum() > 0:`" in notes
    assert "jnp.where(cond, a, b)" in notes
    assert "array parameters of" in notes and ": x)" in notes

    @fn(convert=False)
    def loop[B](x: f32[B], n: i32) -> f32[B]:
        for _ in range(n):  # type: ignore[call-overload]
            x = x * 2
        return x

    with pytest.raises(jax.errors.TracerIntegerConversionError) as e2:
        jax.jit(loop)(jnp.ones(3), 2)
    assert "jax.lax.fori_loop" in "\n".join(e2.value.__notes__)


def test_batch_annotation():
    from typax import Batch

    class Step(NamedTuple):
        obs: f32[Literal[4]]
        reward: f32
        key: Key

    @fn
    def total[T, N](traj: Batch[Step, T, N], last: Batch[f32[Literal[4]], N]) -> f32[N]:
        return traj.reward.sum(0) + last.sum(-1)

    keys = jax.random.split(jax.random.key(0), (6, 3))
    traj = Step(jnp.ones((6, 3, 4)), jnp.ones((6, 3)), keys)
    assert total(traj, jnp.ones((3, 4))).shape == (3,)
    assert "traj: Batch[Step, T, N]" in repr(total)
    with pytest.raises(ShapeError, match=r"declared  traj.reward: f32\[T, N\]"):
        total(Step(jnp.ones((6, 3, 4)), jnp.ones((6, 2)), keys), jnp.ones((3, 4)))
    from typax import prove

    assert prove(total).status == "proved"


def test_hints():
    @fn
    def f[B, T](x: f32[B, T], y: f32[B, T]) -> f32[B, T]:
        return x + y

    def hint(*args):
        with pytest.raises(ShapeError) as e:
            f(*args)
        return str(e.value)

    a = jnp.ones((2, 3))
    assert "Hint: `y` has 1 extra leading axis. To map over it, call" in hint(a, jnp.ones((5, 2, 3)))
    assert "Hint: the axes look permuted: this has the layout [T, B]." in hint(a, a.T)
    assert "Hint: convert with x.astype(jnp.float32), or declare the wider dtype" in hint(a, jnp.ones((2, 3), jnp.float16))
    assert "Hint: this is an integer array; convert with x.astype(jnp.float32)" in hint(a, jnp.ones((2, 3), jnp.int32))
    assert "Hint" not in hint(a, jnp.ones((2, 4)))


def test_info():
    @fn(U="T + 1", where="T >= 2", jit=True)
    def pad[B, T, U](x: f32[B, T]) -> f32[B, U]:
        return jnp.pad(x, ((0, 0), (0, 1)))

    pad(jnp.ones((2, 3)))
    pad(jnp.ones((2, 4)))
    text = pad.info()
    assert "pad[B, T, U](x: f32[B, T]) -> f32[B, T + 1]" in text
    assert "U = T + 1" in text and "where T >= 2" in text and "options: jit" in text
    assert "traced 2 time(s)" in text and "x: f32[2, 3]  ->  f32[2, 4]" in text
    assert "not converted (no control flow to convert)" in text


def test_show_and_scoped_config():
    import typax as tx

    class P(NamedTuple):
        w: Any
        b: Any

    tree = {"p": P(jnp.ones((3, 3)), jnp.zeros(3, jnp.int32)), "k": jax.random.key(0), "n": 2.0}
    assert tx.show(tree) == "{k: Key[()], n: Python float, p: P(w=f32[3, 3], b=i32[3])}"

    @fn
    def bad[T](x: f32[T]) -> f32[T]:
        return x[1:]

    with tx.config(checked=False):
        assert bad(jnp.ones(3)).shape == (2,)
    with pytest.raises(ShapeError):
        bad(jnp.ones(3))
    with pytest.raises(AttributeError):
        tx.config(nope=1)
    assert "checked=True" in repr(tx.config)

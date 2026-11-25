"""Bugs found by differential fuzzing (``tests/fuzz``) and the audit of silently unchecked annotations."""

import linecache
import itertools

import jax
import jax.numpy as jnp
import pytest

import expex
from expex import f32, i32

_N = itertools.count()


def define(src: str, **names):
    """Define ``f`` from source (expex reads the source it converts), in a namespace of ``names``."""
    fname = f"<fuzz-found-{next(_N)}>"
    linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
    ns = {"jnp": jnp, "expex": expex, **names}
    exec(compile(src, fname, "exec"), ns)
    return ns["f"]


# ------------------------------------------------------------ silently unchecked annotations


@pytest.mark.parametrize("ann", ["f32[T]", "f32", "f32[T] | None", "tuple[f32[T], f32[T]]", '"f32[T]"'])
def test_an_annotation_whose_dtype_is_not_imported_is_refused(ann):
    # was: the annotation could not be evaluated and was skipped, so `x` went unchecked
    src = f"def f[T](x: {ann}, y: i32[T]):\n    return y\n"
    with pytest.raises(TypeError) as e:
        expex.fn(define(src, i32=i32))
    assert e.value.code == "EX402"
    assert "f32, which is not defined" in str(e.value) and "from expex import f32" in str(e.value)


def test_a_method_and_a_struct_field_whose_dtype_is_not_imported_are_refused():
    src = "class C:\n    def f[T](self, x: f32[T], y: i32[T]):\n        return y\n"

    class C:
        pass

    C.f = expex.fn(define(src.replace("class C:\n    ", "").replace("\n        ", "\n    "), i32=i32))
    with pytest.raises(TypeError) as e:
        C().f(jnp.ones(3), jnp.ones(4, jnp.int32))  # a method's annotations are read at the first call
    assert e.value.code == "EX402"
    with pytest.raises(TypeError) as e:
        define("@expex.struct\nclass f[D]:\n    w: f32[D, D]\n")
    assert e.value.code == "EX402"


def test_a_bare_jaxtyping_name_with_nothing_imported_is_refused():
    # `Int` was read as jaxtyping's stand-in, which needs `Int[Array, "..."]`: unchecked
    with pytest.raises(TypeError) as e:
        expex.fn(define("def f(x: Int, y: i32):\n    return y\n", i32=i32))
    assert e.value.code == "EX402"


def test_an_undeclared_variadic_is_refused():
    # `f32[*B, T]` without `*B` among the type parameters: evaluating it failed, and it was skipped
    with pytest.raises(TypeError) as e:
        expex.fn(define("def f[T](x: f32[*B, T]):\n    return x\n", f32=f32))
    assert e.value.code == "EX402" and "B" in str(e.value)


def test_unsupported_size_arithmetic_in_quotes_is_refused():
    # `f32[T ** 2]` raises EX403; written in quotes it was read as a string, and skipped
    with pytest.raises(Exception) as e:
        expex.fn(define('def f[T](x: "f32[T ** 2]", t: f32[T]):\n    return x\n', f32=f32))
    assert e.value.code == "EX403"


# ------------------------------------------------------------ the prover


@pytest.mark.parametrize(
    "body",
    [
        "x[1:] if T == 6 else x",
        "x if T != 6 else x[1:]",
        "x[1:] if x.shape[0] in (4, 6) else x",
        "x[1:] if x.shape == (6,) else x",
    ],
)
def test_the_prover_does_not_take_equal_sizes_for_unequal(body):
    # JAX answers `T == 6` with False for a symbolic T (the expressions differ), so the
    # prover only ever traced one branch and proved a result that is wrong at T = 6
    f = expex.fn(define(f"def f[T](x: f32[T]) -> f32[T]:\n    return {body}\n", f32=f32))
    r = expex.prove(f)
    assert r.status == "failed", r
    assert "T=6" in str(r) or "T=4" in str(r)


def test_the_prover_decides_a_comparison_of_sizes_region_by_region():
    src = (
        "def f[T](x: f32[T]) -> f32[T - 1]:\n"
        "    if T > 1:\n"
        "        return x[1:]\n"
        "    return x[:0]\n"
    )
    assert expex.prove(expex.fn(define(src, f32=f32))).status == "proved"
    src = "def f[T](x: f32[T]) -> f32[T]:\n    if T == 1:\n        return x\n    return x[1:]\n"
    assert expex.prove(expex.fn(define(src, f32=f32))).status == "failed"


def test_a_converted_comparison_means_what_python_means():
    @expex.fn
    def f(x: f32, n: int):
        a = n == 3  # converted (it could meet a symbolic size): still a plain bool
        b = n in (1, 2)
        c = x == 2.0
        return a, b, c

    a, b, c = f(jnp.float32(2.0), 3)
    assert a is True and b is False and bool(c)


# ------------------------------------------------------------ staging


def test_a_strongly_typed_variable_changing_dtype_in_a_traced_while_is_refused():
    # was: JAX's raw "carry input and carry output must have equal types", without a code
    @expex.fn
    def f(x: f32, k: i32) -> f32:
        j = 0
        while j < 3 and x > 0:
            j += 1
            k = x
        return k + 0.0

    assert f(jnp.float32(1.5), jnp.int32(2)) == 1.5  # eagerly: Python's semantics
    with pytest.raises(expex.StageError) as e:
        jax.jit(f)(jnp.float32(1.5), jnp.int32(2))
    assert e.value.code == "EX212" and "`k`" in str(e.value)


def test_a_python_accumulator_may_become_a_float_in_a_traced_while():
    @expex.fn
    def f(x: f32):
        acc, j = 0, 0
        while j < 3 and x > 0:
            j += 1
            acc += x
        return acc

    assert jax.jit(f)(jnp.float32(1.5)) == 4.5


def test_an_in_place_update_of_an_array_a_nested_function_reads():
    # was: left as `v[1] -= 1.0`, which JAX arrays refuse, even eagerly (an error without a code)
    @expex.fn
    def f[N](x: f32[N]):
        v = jnp.ones(N)
        v[1] -= 1.0
        h = lambda a: v[1] + a  # noqa: E731
        v[0] += x[0]
        return h(2.0), v

    for run in (f, jax.jit(f)):
        r, v = run(jnp.full(3, 5.0))
        assert r == 2.0 and v.tolist() == [6.0, 0.0, 1.0]

    @expex.fn
    def g[N](x: f32[N]):
        v = jnp.ones(N)
        total = lambda: v.sum()  # noqa: E731
        for i in range(N):  # a loop that updates the captured array stays a Python loop
            v[i] += x[i]
        return total()

    assert g(jnp.arange(3.0)) == jax.jit(g)(jnp.arange(3.0)) == 6.0


def test_a_return_passed_on_from_an_inner_loop_does_not_read_an_unset_value():
    # was: `return` in an inner loop that never ran, inside a traced outer loop:
    # UnboundLocalError on '__bn_ret__' (the passed-on return read the unset value)
    @expex.fn
    def f[N](x: f32[N], n: int):
        for i in range(1, N):
            if x[0] > 0:
                break
            for j in range(n, 0, -1):
                return 1.0

    x = jnp.array([-1.0, 2.0, 3.0])
    jf = jax.jit(f, static_argnames="n")
    assert f(x, 0) is None and jf(x, 0) is None
    assert f(x, 2) == 1.0
    with pytest.raises(expex.StageError) as e:  # 1.0, or None if x[0] > 0: see the next test
        jf(x, 2)
    assert e.value.code == "EX213"


def test_falling_off_the_end_on_a_traced_path_is_refused():
    # was: None in Python, 0.0 under jax.jit (a zero stood in for the unset result)
    @expex.fn
    def f(x: f32):
        if x > 0:
            return x * 2

    @expex.fn
    def g(x: f32):
        for i in range(3):
            if x > i:
                return x * 2

    for h in (f, g):
        assert h(jnp.float32(-1.0)) is None and h(jnp.float32(5.0)) == 10.0
        with pytest.raises(expex.StageError) as e:
            jax.jit(h)(jnp.float32(-1.0))
        assert e.value.code == "EX213"


def test_functions_that_return_on_every_path_are_unaffected():
    @expex.fn
    def f(x: f32):
        for i in range(3):
            if x > i:
                return x * 2
        return -x

    @expex.fn
    def g(x: f32):
        if x > 0:
            return x
        raise ValueError("negative")

    @expex.fn
    def h(x: f32):
        i = 0
        while True:
            i += 1
            if x < i:
                return x + i

    for v in (-1.0, 0.5, 2.5):
        x = jnp.float32(v)
        assert f(x) == jax.jit(f)(x)
        assert h(x) == jax.jit(h)(x)
    assert jax.jit(g)(jnp.float32(2.0)) == 2.0


def test_rebinding_an_int_parameter_first_thing_is_an_ordinary_assignment():
    # was: `n = 0 - n` at the top of the body was read as a size definition (like
    # `H = heads`), and `n` then meant the argument, not the new value: range(n) ran
    # twice for n=2 instead of not at all
    @expex.fn
    def f[N, D](m: f32[N, D], n: int):
        n = 0 - n
        return sum([m[1, 0] for j in range(n)])

    @expex.fn
    def g[N](x: f32[N], n: int):
        n = N - n
        return x[n]

    m = jnp.arange(6.0).reshape(3, 2)
    assert f(m, 2) == 0 and jax.jit(f, static_argnames="n")(m, 2) == 0
    x = jnp.arange(4.0)
    assert g(x, 3) == 1.0 and jax.jit(g, static_argnames="n")(x, 3) == 1.0


def test_rebinding_a_type_parameter_first_thing_is_an_ordinary_assignment():
    # was: `N = 3` was read as a definition of the size N, which changed the
    # signature to f32[3] and refused a call that plain Python runs
    @expex.fn
    def f[N](x: f32[N]):
        N = 3
        return jnp.zeros(N) + x[0]

    assert f(jnp.ones(5)).tolist() == jax.jit(f)(jnp.ones(5)).tolist() == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "ann, code",
    [
        ("f32[B, Q]", "EX402"),  # a misspelt size
        ("i32[B, T]", "EX402"),  # a dtype not imported
        ("f32[*Z, T]", "EX402"),  # a variadic not declared
        ("f32[B, T ** 2]", "EX403"),  # not size arithmetic
        ("f32[B, T + 1]", "EX103"),  # size arithmetic: checked (was skipped)
        ('"f32[T, B]"', "EX103"),  # in quotes: checked (was skipped)
        ("f32[B, n + T]", "EX103"),  # an int parameter: checked (was solved for, so it always matched)
    ],
)
def test_an_annotated_local_variable_is_checked_or_refused(ann, code):
    # was: any annotation Python could not evaluate left the local variable unchecked
    src = f"def f[B, T](x: f32[B, T], n: int):\n    h: {ann} = x\n    return h\n"
    f = expex.fn(define(src, f32=f32))
    with pytest.raises(Exception) as e:
        f(jnp.ones((2, 3)), 1)
    assert getattr(e.value, "code", None) == code


def test_an_annotated_local_variable_that_matches_passes():
    src = "def f[B, T](x: f32[B, T], n: int):\n    h: f32[B, n + T] = jnp.pad(x, ((0, 0), (0, n)))\n    g: 'f32[B, T]' = x\n    return h\n"
    f = expex.fn(define(src, f32=f32))
    assert f(jnp.ones((2, 3)), 2).shape == (2, 5)
    assert jax.jit(f, static_argnames="n")(jnp.ones((2, 3)), 2).shape == (2, 5)


def test_an_annotated_local_in_a_method_sees_the_class_sizes():
    # the check of annotated locals added above first missed a class's sizes (`H` of
    # `class Net[H]`) and refused `h: f32[H]` with EX402 (found by proving examples/ppo)
    @expex.struct
    class Lin[H]:
        w: f32[H, H]

        def __call__(self, x: f32[H]) -> f32[H]:
            h: f32[H] = self.w @ x
            g: f32[H + 1] = jnp.pad(h, (0, 1))
            return g[:-1]

    m = Lin(jnp.eye(3))
    assert m(jnp.ones(3)).shape == (3,)
    assert expex.prove(Lin.__call__).status == "proved"


# ------------------------------------------------------------ builtins mean what Python means


def test_max_and_min_of_concrete_values_are_the_builtins():
    # fuzz seed 69542: `v5 += max(s, v5)` then `x[v5 % N]`. max_ used jnp.maximum even
    # eagerly, so max(x[0], 1) was a float array where Python gives the int 1, and the
    # later index failed ("Indexer must have integer or boolean type").
    src = (
        "def f[N](x: f32[N], k: i32) -> f32:\n"
        "    i = max(x[0], 1)\n"
        "    j = min(k, N - 1, 5)\n"
        "    return x[i] + x[j] + abs(-x[0]) + round(2.4) + int(x[1]) + float(x[0])\n"
    )
    f = expex.fn(define(src, f32=f32, i32=i32))
    x = jnp.array([0.5, 2.0, 3.0])
    # as plain Python: x[1] + x[2] + 0.5 + 2 + 2 + 0.5
    assert float(f(x, jnp.int32(7))) == 10.0
    g = expex.fn(define("def f(a, b, c):\n    return max(a, b), min(a, b, c), max([a, b]), min(a, c, key=abs)\n"))
    out = g(jnp.float32(0.5), 1, -2)
    assert out[0] == 1 and type(out[0]) is int  # the builtin returns the argument itself
    assert out[1] == -2 and type(out[1]) is int
    assert out[2] == 1 and float(out[3]) == 0.5


def test_max_and_min_of_traced_values_go_through_jnp():
    f = expex.fn(define("def f(x, y):\n    return max(x, 1.0, y), min(x, y)\n"))
    a, b = jax.jit(f)(jnp.float32(0.5), jnp.float32(2.0))
    assert (float(a), float(b)) == (2.0, 0.5)
    v = jax.vmap(f)(jnp.array([0.5, 3.0]), jnp.array([2.0, 0.0]))
    assert v[0].tolist() == [2.0, 3.0] and v[1].tolist() == [0.5, 0.0]
    # arrays of several elements, traced: element-wise (Python cannot compare them)
    m = jax.jit(f)(jnp.array([1.0, 1.0]), jnp.array([0.0, 5.0]))
    assert m[0].tolist() == [1.0, 5.0] and m[1].tolist() == [0.0, 1.0]
    # under jax.grad outside jit, values are concrete: Python's max picks an argument
    assert jax.grad(lambda x: f(x, 2.0)[0] * x)(3.0) == pytest.approx(6.0)

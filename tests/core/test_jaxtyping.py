"""Code annotated with jaxtyping, read by expex as it is; and `python -m expex migrate jaxtyping`.

Each module below is written once and loaded two ways: with jaxtyping imported
("installed": its classes) and with it imported only for type checkers
("text": expex reads the annotation's text). Both must mean the same.
"""

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import expex
from expex.annotations.errors import code_of

try:
    import jaxtyping  # noqa: F401

    HAVE_JAXTYPING = True
except ImportError:
    HAVE_JAXTYPING = False

NAMES = "Array, Bool, BFloat16, Float, Float32, Int, Integer, Key, PRNGKeyArray, PyTree, Shaped, UInt8"
HEADERS = {
    "installed": f"from jaxtyping import {NAMES}\nimport jaxtyping as jt\n",
    "text": f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from jaxtyping import {NAMES}\n    import jaxtyping as jt\n",
}
MODES = [
    "text",
    pytest.param("installed", marks=pytest.mark.skipif(not HAVE_JAXTYPING, reason="jaxtyping not installed")),
]

_n = 0


def load(tmp_path: pathlib.Path, src: str, mode: str = "text") -> object:
    """Write ``src`` after the mode's imports to a file (expex reads the source) and import it."""
    global _n
    _n += 1
    name = f"jt_{mode}_{_n}"
    path = tmp_path / f"{name}.py"
    path.write_text("import jax, jax.numpy as jnp, numpy as np\nimport expex\n" + HEADERS[mode] + textwrap.dedent(src))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def raises(code: str, f, *args):
    with pytest.raises(Exception) as e:
        f(*args)
    assert code_of(e.value) == code, e.value
    return str(e.value)


ones = jnp.ones

# ------------------------------------------------------------------ sizes


@pytest.mark.parametrize("mode", MODES)
def test_names_are_the_sizes(tmp_path, mode):
    m = load(tmp_path, '''
        @expex.fn
        def attend(q: Float[Array, "batch t d"], k: Float[Array, "batch s d"]) -> Float[Array, "batch t s"]:
            return jnp.einsum("btd,bsd->bts", q, k)
    ''', mode)
    f = m.attend
    assert f.signature == "attend(q: Float[batch, t, d], k: Float[batch, s, d]) -> Float[batch, t, s]"
    assert f(ones((2, 3, 4)), ones((2, 5, 4))).shape == (2, 3, 5)
    msg = raises("EX101", f, ones((2, 3, 4)), ones((5, 3, 4)))
    assert "batch is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5." in msg
    assert expex.prove(f).ok
    # extra leading axes are not mapped over by a plain call (jaxtyping annotations were
    # often never enforced): f.vectorize maps them, and a plain call says so
    assert f.vectorize(ones((7, 2, 3, 4)), ones((7, 2, 5, 4))).shape == (7, 2, 3, 5)
    msg = raises("EX101", f, ones((7, 2, 3, 4)), ones((7, 2, 5, 4)))
    assert "call attend.vectorize(...)" in msg and "@expex.fn(vectorize=True)" in msg
    assert jax.jit(f)(ones((2, 3, 4)), ones((2, 5, 4))).shape == (2, 3, 5)


@pytest.mark.parametrize("mode", MODES)
def test_no_automatic_batching_unless_asked(tmp_path, mode, monkeypatch):
    # jaxtyping annotations were often never enforced: mapping a call over extra axes
    # turned real mismatches into confusing errors later on (the dynamax port)
    m = load(tmp_path, '''
        @expex.fn
        def norm(x: Float[Array, "d"]) -> Float[Array, ""]:
            return jnp.sqrt((x * x).sum())

        @expex.fn(vectorize=True)
        def norm_mapped(x: Float[Array, "d"]) -> Float[Array, ""]:
            return jnp.sqrt((x * x).sum())

        @expex.fn
        def mixed[D](x: Float[Array, "d"], w: expex.f32[D]) -> Float[Array, ""]:
            return (x * w).sum()

        @expex.fn
        def own_style[D](x: expex.f32[D]) -> expex.f32:
            return jnp.sqrt((x * x).sum())

        @expex.struct
        class Scale:
            w: Float[Array, "d"]

            def __call__(self, x: Float[Array, "d"]) -> Float[Array, "d"]:
                return x * self.w
    ''', mode)
    batch = ones((5, 3))
    raises("EX101", m.norm, batch)  # as vectorize=False
    raises("EX101", m.mixed, batch, ones(3))  # one jaxtyping annotation is enough
    raises("EX101", m.Scale(w=ones(3)), batch)
    assert m.norm.vectorize(batch).shape == (5,)  # asked for explicitly
    assert m.norm_mapped(batch).shape == (5,)  # opted in on the function
    assert m.own_style(batch).shape == (5,)  # expex-style annotations: unchanged
    with expex.config(vectorize=True):  # set explicitly, globally
        assert m.norm(batch).shape == (5,) and m.Scale(w=ones(3))(batch).shape == (5, 3)
    raises("EX101", m.norm, batch)  # back to the default outside the block
    with expex.config(vectorize=False):
        raises("EX101", m.own_style, batch)
    monkeypatch.setattr(expex.function.Config, "_vectorize_set", False)
    monkeypatch.setattr(expex.config, "vectorize", True)  # assigned: explicit too
    assert m.norm(batch).shape == (5,)


@pytest.mark.parametrize("mode", MODES)
def test_variadic_ellipsis_literal_scalar(tmp_path, mode):
    m = load(tmp_path, '''
        @expex.fn
        def f(x: Float[Array, "*batch d"], y: Float[Array, "*batch 3"], z: Shaped[Array, "..."],
              s: Float[Array, ""], one: Float[Array, "b t=1"]) -> Float[Array, "*batch"]:
            return (x.sum(-1) + y.sum(-1)) * s
    ''', mode)
    f = m.f
    assert str(f.params["x"].spec) == "Float[*batch, d]"
    assert str(f.params["y"].spec) == "Float[*batch, 3]"
    assert str(f.params["z"].spec) == "Arr[...]"
    assert str(f.params["s"].spec) == "Float[()]"
    assert str(f.params["one"].spec) == "Float[b, 1]"
    args = [ones((2, 5, 4)), ones((2, 5, 3)), ones((9, 9), jnp.int32), jnp.float32(2), ones((6, 1))]
    assert f(*args).shape == (2, 5)
    assert f(*args[:3], 2.0, args[4]).shape == (2, 5)  # a Python float for a scalar, as for f32
    assert "*batch is (2, 5)" in raises("EX101", f, args[0], ones((2, 4, 3)), *args[2:])
    raises("EX101", f, args[0], ones((2, 5, 4)), *args[2:])  # 3 is fixed
    raises("EX101", f, *args[:4], ones((6, 2)))  # t=1 is 1
    assert expex.prove(f).ok


@pytest.mark.parametrize("mode", MODES)
def test_arithmetic(tmp_path, mode):
    m = load(tmp_path, '''
        @expex.fn
        def shift(x: Float[Array, "b t"]) -> Float[Array, "b t+1"]:
            return jnp.pad(x, ((0, 0), (1, 0)))

        @expex.fn
        def solve(x: Float[Array, "t+1"], y: Float[Array, "dim-1"], n: int, z: Float[Array, "{n}"]) -> Float[Array, "t"]:
            return x[1:]

        @expex.fn
        def wrong(x: Float[Array, "t"]) -> Float[Array, "t-1"]:
            return x

        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Lags:
            lags: int

            @expex.fn
            def stack(self, x: Float[Array, "t d"]) -> Float[Array, "t {self.lags}*d"]:
                return jnp.tile(x, (1, self.lags))
    ''', mode)
    assert m.shift.signature == "shift(x: Float[b, t]) -> Float[b, t + 1]"
    assert m.shift(ones((2, 3))).shape == (2, 4)
    assert expex.prove(m.shift).ok
    # a size used only in an expression is solved for; `{n}` is the int parameter n
    assert m.solve(ones(5), ones(2), 3, ones(3)).shape == (4,)
    raises("EX101", m.solve, ones(5), ones(2), 3, ones(4))
    assert expex.prove(m.solve).ok
    raises("EX102", m.wrong, ones(3))
    assert not expex.prove(m.wrong).ok
    # `{self.lags}`: the int field lags, as expex's f32[self.lags]
    assert m.Lags(3).stack(ones((2, 4))).shape == (2, 12)
    assert m.Lags.stack.signature.endswith("-> Float[t, lags * d]")


@pytest.mark.parametrize("mode", MODES)
def test_modifiers(tmp_path, mode):
    m = load(tmp_path, '''
        @expex.fn
        def f(a: Float[Array, "#b d"], b: Float[Array, "b d"], c: Float[Array, "#4"],
              u: Float[Array, "_ d"], v: Float[Array, "_name d"],
              p: Float[Array, "*#batch d"], q: Float[Array, "*#batch d"]) -> Float[Array, "b d"]:
            return a + b

        @expex.fn
        def g(b: Float[Array, "b d"], a: Float[Array, "#b d"]) -> Float[Array, "b d"]:
            return a + b
    ''', mode)
    f = m.f
    assert str(f.params["a"].spec) == "Float[#b, d]"
    assert str(f.params["u"].spec) == "Float[Any, d]"  # `_`: one unchecked axis
    assert str(f.params["v"].spec) == "Float[Any, d]"  # `_name` too: the name is for the reader
    assert str(f.params["p"].spec) == "Float[*#batch, d]"
    ok = [ones((1, 3)), ones((4, 3)), ones(1), ones((9, 3)), ones((8, 3)), ones((2, 1, 3)), ones((5, 3))]
    assert f(*ok).shape == (4, 3)  # #b is 1, #4 is 1, *#batch (2, 1) and (5,) broadcast
    assert f(ones((4, 3)), ones((4, 3)), ones(4), *ok[3:]).shape == (4, 3)
    assert "b is 2 (from `a: f32[2, 3]`)" in raises("EX101", f, ones((2, 3)), *ok[1:])  # not 1: binds b
    assert "neither 4 nor 1" in raises("EX101", m.g, ones((4, 3)), ones((2, 3)))
    raises("EX101", f, ok[0], ok[1], ones(3), *ok[3:])
    assert "does not broadcast" in raises("EX101", f, *ok[:5], ones((2, 3)), ones((4, 3)))
    assert expex.prove(f).ok


@pytest.mark.parametrize("mode", MODES)
def test_dtypes(tmp_path, mode):
    m = load(tmp_path, '''
        @expex.fn
        def f(a: Float32[Array, "n"], b: BFloat16[Array, "n"], c: Int[Array, "n"], d: UInt8[np.ndarray, "n"],
              e: Integer[Array, "n"], g: Bool[Array, "n"], h: Shaped[Array, "n"], k: Key[Array, "n"],
              key: PRNGKeyArray) -> Float[Array, "n"]:
            return a
    ''', mode)
    f = m.f
    assert [str(p.spec) for p in f.params.values()] == [
        "f32[n]", "bf16[n]", "Int[n]", "u8[n]", "Int[n]", "Bool[n]", "Arr[n]", "Key[n]", "Key",
    ]  # fmt: skip
    keys = jax.random.split(jax.random.key(0), 2)
    args = [ones(2), ones(2, jnp.bfloat16), ones(2, jnp.int32), np.ones(2, np.uint8), ones(2, jnp.uint32),
            ones(2, bool), ones(2, jnp.complex64), keys, jax.random.key(1)]  # fmt: skip
    assert f(*args).shape == (2,)

    def swap(i, v):
        return [v if j == i else a for j, a in enumerate(args)]

    raises("EX101", f, *swap(0, ones(2, jnp.float16)))  # Float32 is exactly f32
    raises("EX101", f, *swap(1, ones(2)))
    assert "expected dtype Int, got u32" in raises("EX101", f, *swap(2, ones(2, jnp.uint32)))  # jaxtyping's Int is signed
    raises("EX101", f, *swap(4, ones(2)))  # Integer: any integer
    raises("EX101", f, *swap(7, ones(2)))  # not keys
    assert f(*swap(8, jax.random.PRNGKey(0))).shape == (2,)  # a legacy uint32 key is a key too


@pytest.mark.parametrize("mode", MODES)
def test_structs_unions_and_the_module_name(tmp_path, mode):
    m = load(tmp_path, '''
        from typing import NamedTuple, Optional, Union

        class Opaque:  # matches anything: tried after the arrays, which bind their sizes
            pass

        class Params(NamedTuple):
            w: Union[Opaque, Float[Array, "e s"]]
            b: Optional[jt.Float[jt.Array, "e"]] = None

        @expex.fn
        def apply(p: Params, x: Float[Array, "s"]) -> Float[Array, "e"]:
            return p.w @ x + (0 if p.b is None else p.b)

        @expex.fn
        def tree(t: PyTree[Float[Array, "n"]], x: Float[Array, "n"]) -> Float[Array, "n"]:
            return x
    ''', mode)
    p = m.Params(ones((3, 4)), ones(3))
    assert m.apply(p, ones(4)).shape == (3,)
    assert "s is 4" in raises("EX101", m.apply, p, ones(5))
    raises("EX101", m.apply, m.Params(ones((3, 4)), ones(2)), ones(4))
    assert m.tree({"a": 1}, ones(2)).shape == (2,)  # PyTree[...]: not checked
    assert expex.prove(m.apply).ok


@pytest.mark.parametrize("mode", MODES)
def test_nested_modules_have_their_own_sizes(tmp_path, mode):
    """Layers of different widths in one model: each layer's `in` / `out` are its own, as jaxtyping has them."""
    m = load(tmp_path, '''
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Linear:
            weight: Float[Array, "out in"]
            bias: Float[Array, "out"]

            @expex.fn
            def __call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]:
                return self.weight @ x + self.bias

        @dataclasses.dataclass(frozen=True)
        class MLP:
            layers: tuple[Linear, ...]

            @expex.fn
            def __call__(self, x: Float[Array, "d"]) -> Float[Array, "k"]:
                for layer in self.layers:
                    x = layer(x)
                return x
    ''', mode)
    lin = m.Linear(ones((3, 4)), ones(3))
    assert lin(ones(4)).shape == (3,)
    assert "in is 4" in raises("EX101", lin, ones(5))  # `self` shares its sizes with the method
    mlp = m.MLP((m.Linear(ones((5, 4)), ones(5)), lin.__class__(ones((3, 5)), ones(3))))
    assert mlp(ones(4)).shape == (3,)
    bad = m.MLP((m.Linear(ones((5, 4)), ones(5)), m.Linear(ones((3, 5)), ones(4))))
    assert "self.layers[1].bias" in raises("EX101", bad, ones(4))  # still checked within each layer


@pytest.mark.parametrize("mode", MODES)
def test_unsupported(tmp_path, mode):
    for ann, code, says in [
        ('"?x"', "EX411", "PyTree"),
        ('"#t+1"', "EX411", "broadcast"),
        ('"{n+1}"', "EX411", "f-string"),
        ('"a b,c"', "EX401", "spaces"),
    ]:
        with pytest.raises(Exception) as e:
            load(tmp_path, f'''
                @expex.fn
                def bad(x: Float[Array, {ann}], n: int = 1):
                    return x
            ''', mode)
        if mode == "installed" and code == "EX401":
            assert says in str(e.value)  # jaxtyping itself refuses it, in its words
            continue
        assert code_of(e.value) == code, e.value
        assert says in str(e.value)


def test_expex_names_written_the_jaxtyping_way():
    """expex's own Float / Int / Key with an array type and a shape string mean the same."""

    @expex.fn
    def f(x: expex.Float[jax.Array, "b d"], i: expex.Int[jax.Array, "b"], k: expex.Key[jax.Array, ""]) -> expex.Float[jax.Array, "b"]:
        return x.sum(-1)

    assert f.signature == "f(x: Float[b, d], i: Int[b], k: Key) -> Float[b]"
    assert f(ones((2, 3)), jnp.zeros(2, jnp.uint8), jax.random.key(0)).shape == (2,)  # expex's Int: any integer
    raises("EX101", f, ones((2, 3)), jnp.zeros(3, jnp.int32), jax.random.key(0))


def test_never_imports_jaxtyping(tmp_path):
    """Without jaxtyping at all, its annotations (imported for type checkers only) are read from their text."""
    path = tmp_path / "nojt.py"
    path.write_text(textwrap.dedent('''
        import sys
        sys.modules["jaxtyping"] = None  # not installed
        from typing import TYPE_CHECKING
        import jax.numpy as jnp
        import expex
        if TYPE_CHECKING:
            from jaxtyping import Array, Float, Int

        @expex.fn
        def f(x: Float[Array, "b t"], i: Int[Array, "b"]) -> Float[Array, "b"]:
            return x.sum(-1)

        print(f.signature)
        try:
            f(jnp.ones((2, 3)), jnp.zeros(3, jnp.int32))
        except expex.ShapeError as e:
            print(e.code)
        print(expex.prove(f).ok)
    '''))
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["f(x:", "Float[b,", "t],", "i:", "Int[b])", "->", "Float[b]", "EX101", "True"]


# ------------------------------------------------------------------ with jaxtyping's own checks


def test_with_jaxtyped_and_beartype(tmp_path):
    """`@jaxtyped(typechecker=beartype)` under or over `@expex.fn`: both checks run."""
    pytest.importorskip("jaxtyping")
    pytest.importorskip("beartype")
    m = load(tmp_path, '''
        from beartype import beartype
        from jaxtyping import jaxtyped

        @expex.fn
        @jaxtyped(typechecker=beartype)
        def under(x: Float[Array, "n"], s: Float[Array, ""]) -> Float[Array, "n"]:
            return x * s

        @jaxtyped(typechecker=beartype)
        @expex.fn
        def over(x: Float[Array, "n"], s: Float[Array, ""]) -> Float[Array, "n"]:
            return x * s
    ''', "installed")
    from jaxtyping import TypeCheckError

    for f in (m.under, m.over):
        assert jnp.array_equal(f(ones(3), jnp.float32(2)), 2 * ones(3))
        with pytest.raises(TypeCheckError):
            f(ones(3), 2.0)  # jaxtyping's check: a Python float is not an Array (expex would take it)
    raises("EX101", m.under, ones(3, jnp.int32), jnp.float32(2))  # expex's check
    # expex on top maps over extra axes (when asked to), then jaxtyping checks each element
    assert m.under.vectorize(ones((4, 3)), jnp.float32(2)).shape == (4, 3)
    raises("EX101", m.under, ones((4, 3)), jnp.float32(2))  # not mapped by a plain call
    assert expex.prove(m.under).ok
    with pytest.raises(TypeCheckError):
        m.over(ones((4, 3)), jnp.float32(2))  # jaxtyping on top sees the batch first


# ------------------------------------------------------------------ migrate

MODEL = '''\
"""A model."""
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int, PRNGKeyArray

D = 64


class Linear:
    weight: Float[Array, "out in"]

    def __init__(self, key: PRNGKeyArray, d_in: int, d_out: int):
        self.weight = jax.random.normal(key, (d_out, d_in))

    def __call__(self, x: Float[Array, "in"]) -> Float[Array, "out"]:
        return self.weight @ x


def attend(q: Float[Array, "batch seq_len d"], k: Float[Array, "batch s d"]) -> Float[Array, "batch seq_len s"]:
    return jnp.einsum("btd,bsd->bts", q, k) / D


def shift(x: Float[Array, "*batch t"], i: Int[Array, ""]) -> Float[Array, "*batch t+1"]:
    y: Float[Array, "*batch 1"] = jnp.zeros(x.shape[:-1] + (1,))
    return jnp.concatenate([y, x], -1)
'''


def test_migrate(tmp_path, monkeypatch, capsys):
    from expex.cli import main

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "model.py"
    path.write_text(MODEL)
    assert main(["migrate", "jaxtyping", "model.py"]) == 0
    diff = capsys.readouterr().out
    assert diff.startswith("--- a/model.py\n+++ b/model.py\n")
    assert path.read_text() == MODEL  # a diff only
    assert "+def attend[Batch, SeqLen, d, S](q: Float[Batch, SeqLen, d], k: Float[Batch, S, d]) -> Float[Batch, SeqLen, S]:" in diff
    assert main(["migrate", "jaxtyping", "model.py", "--write"]) == 0
    assert "1 file rewritten" in capsys.readouterr().out
    new = path.read_text()
    for line in [
        "    pass\n",  # the TYPE_CHECKING block had only jaxtyping's imports
        "import expex\nfrom expex import Int, Float, Key\n",
        "class Linear[Out, In]:\n    weight: Float[Out, In]\n",
        "    def __init__(self, key: Key, d_in: int, d_out: int):\n",  # a constructor is not decorated
        "    @expex.fn\n    def __call__(self, x: Float[In]) -> Float[Out]:\n",
        # D is a global the body reads: the size d keeps its name
        "@expex.fn\ndef attend[Batch, SeqLen, d, S](",
        "@expex.fn\ndef shift[*Batch, T](x: Float[*Batch, T], i: Int) -> Float[*Batch, T + 1]:\n",
        "    y: Float[*Batch, 1] = ",
    ]:
        assert line in new, (line, new)
    assert "jaxtyping" not in new
    m = load_file(path)
    assert m.attend(ones((2, 3, 4)), ones((2, 5, 4))).shape == (2, 3, 5)
    assert m.shift(ones((2, 3)), jnp.int32(0)).shape == (2, 4)
    lin = m.Linear(jax.random.key(0), 4, 3)
    assert lin(ones(4)).shape == (3,)
    assert all(expex.prove(f).ok for f in (m.attend, m.shift, m.Linear.__call__))


def load_file(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_migrated", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_migrate_keeps_what_expex_cannot_write(tmp_path):
    from expex.tools.migrate import migrate

    path = tmp_path / "kept.py"
    path.write_text(textwrap.dedent('''\
        from typing import TYPE_CHECKING
        from jaxtyping import Array, Float, jaxtyped
        from beartype import beartype

        @jaxtyped(typechecker=beartype)
        def add(x: Float[Array, "b d"], y: Float[Array, "b d"]) -> Float[Array, "b d"]:
            return x + y

        def broadcast(x: Float[Array, "#b d"], y: Float[Array, "b d"]) -> Float[Array, "b d"]:
            return x + y

        def odd(x: Float[Array, "?b"]):
            return x

        Image = Float[Array, "h w 3"]
    '''))
    m = migrate(path, float_as="f32")
    new = m.new
    # `#b` stays jaxtyping's (expex reads it), so jaxtyping's Float stays and expex's is qualified
    assert "from jaxtyping import Array, Float\n" in new
    assert "@expex.fn\ndef add[B, D](x: expex.f32[B, D], y: expex.f32[B, D]) -> expex.f32[B, D]:" not in new
    assert "@expex.fn\ndef add[B, D](x: f32[B, D], y: f32[B, D]) -> f32[B, D]:" in new  # --float f32
    assert "jaxtyped" not in new.split("def add")[1].split("def broadcast")[0]
    assert "@expex.fn\ndef broadcast(x: Float[Array, \"#b d\"]" in new  # kept, and checked by expex as is
    assert "\ndef odd(" in new and "@expex.fn\ndef odd" not in new  # `?b`: expex cannot read it
    assert [k.split(":")[0] for k in m.kept] == ["broadcast", "odd", "Image"]
    assert "decorated with @expex.fn" in m.kept[0]
    assert m.diff().count("\n+") >= 3 and str(m).endswith(m.kept[-1])


def test_migrate_names_float_qualified_when_jaxtyping_float_stays(tmp_path):
    from expex.tools.migrate import migrate

    path = tmp_path / "q.py"
    path.write_text(textwrap.dedent('''\
        from jaxtyping import Array, Float

        def f(x: Float[Array, "n"]) -> Float[Array, "n"]:
            return x

        def g(x: Float[Array, "#n"]) -> Float[Array, "n"]:
            return x
    '''))
    new = migrate(path).new
    assert "def f[N](x: expex.Float[N]) -> expex.Float[N]:" in new
    assert "from jaxtyping import Array, Float\nimport expex\n" in new


def test_migrate_passes_sizes_through_classes(tmp_path):
    """jaxtyping's names mean one size throughout: a struct of structs, and the functions taking it, say so."""
    from expex.tools.migrate import migrate

    path = tmp_path / "params.py"
    path.write_text(textwrap.dedent('''\
        from typing import NamedTuple
        from jaxtyping import Array, Float

        class Initial(NamedTuple):
            mean: Float[Array, "state"]

        class Dynamics(NamedTuple):
            weights: Float[Array, "state state"]

        class Params(NamedTuple):
            initial: Initial
            dynamics: Dynamics
            history: list[Initial]  # each element may have its own sizes: left as it is

        def step(params: Params, x: Float[Array, "state"]) -> Float[Array, "state"]:
            return params.dynamics.weights @ x + params.initial.mean
    '''))
    new = migrate(path).new
    assert "class Initial[State](NamedTuple):\n    mean: Float[State]" in new
    assert "class Params[State](NamedTuple):\n    initial: Initial[State]\n    dynamics: Dynamics[State]\n    history: list[Initial]" in new
    assert "def step[State](params: Params[State], x: Float[State]) -> Float[State]:" in new
    path.write_text(new)
    m = load_file(path)
    p = m.Params(m.Initial(ones(3)), m.Dynamics(ones((3, 3))), [])
    assert m.step(p, ones(3)).shape == (3,)
    assert "State is 3" in raises("EX101", m.step, p, ones(4))  # the struct's size is the function's
    assert expex.prove(m.step).ok


def test_migrate_refuses_a_file_changed_since(tmp_path):
    from expex.tools.migrate import migrate

    path = tmp_path / "c.py"
    path.write_text('from jaxtyping import Array, Float\n\ndef f(x: Float[Array, "n"]):\n    return x\n')
    m = migrate(path)
    path.write_text(path.read_text() + "\n# edited\n")
    with pytest.raises(RuntimeError) as e:
        m.apply()
    assert code_of(e.value) == "EX707"


def test_migrate_command_line(tmp_path):
    path = tmp_path / "cli.py"
    path.write_text('from jaxtyping import Array, Float\n\ndef f(x: Float[Array, "n"]):\n    return x\n')
    r = subprocess.run([sys.executable, "-m", "expex", "migrate", "jaxtyping", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "+def f[N](x: Float[N]):" in r.stdout
    r = subprocess.run([sys.executable, "-m", "expex", "migrate", "nothing"], capture_output=True, text=True)
    assert r.returncode == 2 and "usage" in r.stderr

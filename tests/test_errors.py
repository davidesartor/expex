"""Every error has a code, every code a section of docs/errors.md, and every section an example raising it."""

import ast
import contextlib
import io
import pathlib
import re
import runpy

import jax.numpy as jnp
import pytest

import expex
from expex.annotations.errors import CODES, DOCS_URL, code_of, coded, see, strip_see

ROOT = pathlib.Path(__file__).parent.parent
SRC = ROOT / "src" / "expex"
PAGE = ROOT / "docs" / "errors.md"

# The exceptions a `raise` in expex must give a code (with `coded("EX...", Error(...))`).
CHECKED = {
    "ShapeError", "StageError", "TypeError", "ValueError", "AttributeError", "IndexError", "KeyError",
    "LookupError", "NameError", "RuntimeError", "DimError", "ExprError", "_CannotLower",
}  # fmt: skip

# Raises that never reach a user: Python protocols (``__getattr__``, ``__getitem__``),
# and internal signals caught where they are raised. (file, exception) -> why.
INTERNAL = {
    ("modules.py", "AttributeError"): "At.__getattr__ refuses dunders (Python's protocol)",
    ("_diagram.py", "AttributeError"): "a lazy attribute's __getattr__",
    ("_diagram.py", "RuntimeError"): "the profiler failed: caught, timing falls back to an estimate",
    ("annotations/sym.py", "AttributeError"): "a namespace's __getattr__",
    ("annotations/sym.py", "KeyError"): "a namespace lookup, caught",
    ("tools/inspect.py", "ValueError"): "a tab that cannot be drawn: caught, the tab shows it",
    ("staging/loops.py", "IndexError"): "rows.__getitem__ ends iteration (Python's protocol)",
    ("staging/loops.py", "_CannotLower"): "caught: the loop is unrolled instead",
    ("staging/flow.py", "NameError"): "an undefined variable's __bool__: the converter deletes those first",
}


def _sections() -> dict[str, str]:
    """code -> the text of its section of docs/errors.md."""
    text = PAGE.read_text()
    parts = re.split(r'<a id="(ex\d{3})"></a>', text)
    return {parts[i].upper(): parts[i + 1] for i in range(1, len(parts), 2)}


def _blocks(md: str, lang: str = "python") -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", md, flags=re.S)


def test_every_code_has_a_section():
    sections = _sections()
    assert list(sections) == list(CODES)  # in order, none missing, none extra
    for code, text in sections.items():
        assert f"### {code} · {CODES[code]}\n" in text, code
        assert "**Fix.**" in text and len(_blocks(text)) == 1, code
    index = PAGE.read_text().split("<a id=")[0]
    assert all(f"[{c}](#{c.lower()})" in index for c in CODES)


@pytest.mark.parametrize("code", list(CODES))
def test_section_example_raises_its_code(code, tmp_path, monkeypatch):
    """Each section's example, run alone after the page's imports, raises its error and prints it."""
    monkeypatch.chdir(tmp_path)  # some examples save files
    prelude = _blocks(PAGE.read_text().split("<a id=")[0])[0]
    section = _sections()[code]
    path = tmp_path / f"example_{code.lower()}.py"  # a file: @fn reads the source it converts
    path.write_text(prelude + "\n" + _blocks(section)[0])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(path))
    printed = out.getvalue()
    assert printed.split(None, 1)[0] == code, printed  # `print(e.code, e)`
    assert see(code) in printed
    (shown,) = re.findall(r"\n\n```\n(.*?)\n```", section, flags=re.S)  # the output block
    assert shown.split(None, 1)[0] == code
    assert shown.endswith(see(code)), "the page shows the message with its link"


def _src_files() -> list[pathlib.Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _name(node: ast.expr) -> str | None:
    return node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None


def test_every_raise_carries_a_code():
    """A `raise` of a user-facing exception in expex goes through `coded(...)` (or is internal)."""
    bare, used = [], set()
    for path in _src_files():
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text())
        wrapped = {
            id(n.args[1]) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _name(n.func) == "coded" and len(n.args) == 2
        }  # fmt: skip
        for n in ast.walk(tree):
            if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and id(n.exc) not in wrapped:
                name = _name(n.exc.func)
                if name == "coded":
                    continue
                if name in CHECKED:
                    if (rel, name) in INTERNAL:
                        used.add((rel, name))
                        continue
                    bare.append(f"{rel}:{n.lineno}: raise {name}(...)")
                elif name == "_no_field":  # builds an AttributeError
                    bare.append(f"{rel}:{n.lineno}: raise {name}(...)")
            # an expex error is only ever built with its code
            if isinstance(n, ast.Call) and _name(n.func) in ("ShapeError", "StageError") and id(n) not in wrapped:
                bare.append(f"{rel}:{n.lineno}: {_name(n.func)}(...) without a code")
    assert not bare, "errors without a code:\n" + "\n".join(bare)
    assert used == set(INTERNAL), f"stale INTERNAL entries: {set(INTERNAL) - used}"


def test_codes_in_the_source_are_documented():
    """Every code written in the source is in CODES, and every code in CODES is raised somewhere."""
    found = {m for p in _src_files() for m in re.findall(r'"(EX\d{3})"', p.read_text())}
    assert found - set(CODES) == set(), "codes missing from CODES and docs/errors.md"
    assert set(CODES) - found == set(), "codes no error uses"


@expex.fn
def scale[N](x: expex.f32[N], w: expex.f32[N]) -> expex.f32[N]:
    return x * w


def test_messages_end_with_their_link():
    with pytest.raises(expex.ShapeError) as e:
        scale(jnp.ones(3), jnp.ones(4))
    assert e.value.code == "EX101" and code_of(e.value) == "EX101"
    lines = str(e.value).splitlines()
    assert lines[0] == "scale(): argument `w` does not match its annotation."
    assert lines[-1] == f"  see {DOCS_URL}#ex101"
    assert sum("see https://" in line for line in lines) == 1


def test_coded_keeps_one_link():
    inner = coded("EX401", TypeError("bad dimension"))
    outer = coded("EX401", TypeError(f"f: bad annotation for `x`: {inner}"))
    assert str(outer) == f"f: bad annotation for `x`: bad dimension\n{see('EX401')}"
    assert strip_see(str(outer)) == "f: bad annotation for `x`: bad dimension"
    assert coded("EX101", coded("EX102", ValueError("m"))).code == "EX101"
    assert str(coded("EX102", coded("EX101", ValueError("m")))).count("see ") == 1
    assert expex.ShapeError("plain").code is None and code_of(ValueError("x")) is None

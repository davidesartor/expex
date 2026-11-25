"""The command line: `python -m expex [prove] ... [--watch]` and `python -m expex shapes`."""

import json
import os
import pathlib
import queue
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def env() -> dict[str, str]:
    path = os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": path, "PYTHONUNBUFFERED": "1"}


def write(path: pathlib.Path, src: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(src))


def expex(*args: str, cwd: pathlib.Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "expex", *args], cwd=cwd, env=env(), capture_output=True, text=True, timeout=300
    )


CORE = """\
    import expex
    from expex import f32


    @expex.fn
    def double[T](x: f32[T]) -> f32[T]:
        return x * 2
"""

USER = """\
    import expex
    from expex import f32

    from .core import double


    @expex.fn
    def quad[T](x: f32[T]) -> f32[T]:
        return double(double(x))
"""

OTHER = """\
    import expex
    from expex import f32


    @expex.fn
    def ident[T](x: f32[T]) -> f32[T]:
        return x
"""


@pytest.fixture
def package(tmp_path: pathlib.Path) -> pathlib.Path:
    write(tmp_path / "mypkg" / "__init__.py", "")
    write(tmp_path / "mypkg" / "core.py", CORE)
    write(tmp_path / "mypkg" / "user.py", USER)
    write(tmp_path / "mypkg" / "other.py", OTHER)
    return tmp_path


def test_prove_subcommand_and_the_old_form_agree(package):
    old = expex("mypkg/", cwd=package)
    new = expex("prove", "mypkg/", cwd=package)
    assert old.returncode == new.returncode == 0, old.stderr
    assert old.stdout == new.stdout
    assert "3 functions: 3 proved" in new.stdout


def test_a_folder_that_is_not_a_package_proves_each_file(tmp_path):
    # was: ValueError: Empty module name
    write(tmp_path / "scripts" / "core.py", CORE)
    write(tmp_path / "scripts" / "other.py", OTHER)
    write(tmp_path / "scripts" / "sub" / "other.py", OTHER.replace("ident", "same"))  # a second other.py
    write(tmp_path / "scripts" / "broken.py", "import not_a_module_anywhere\n")
    r = expex("scripts/", cwd=tmp_path)
    assert "3 functions: 3 proved" in r.stdout, r.stdout + r.stderr
    assert "ok    core.double" in r.stdout and "ok    sub_other.same" in r.stdout
    assert re.search(r"error: scripts/broken.py: could not import it: ModuleNotFoundError", r.stdout)
    assert r.returncode == 1  # a file that does not import is an error
    (tmp_path / "scripts" / "broken.py").unlink()
    assert expex("scripts", cwd=tmp_path).returncode == 0


def test_help_lists_the_subcommands(tmp_path):
    r = expex("--help", cwd=tmp_path)
    assert r.returncode == 0
    assert "--watch" in r.stdout and "shapes FILE [--json]" in r.stdout


class Watch:
    """`python -m expex prove pkg/ --watch` in a subprocess, read line by line."""

    def __init__(self, cwd: pathlib.Path, *args: str) -> None:
        cmd = [sys.executable, "-m", "expex", "prove", *args, "--watch", "--poll", "--interval", "0.05"]
        self.p = subprocess.Popen(
            cmd, cwd=cwd, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.lines: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.p.stdout is not None
        for line in self.p.stdout:
            self.lines.put(line)

    def until(self, prefix: str, timeout: float = 120) -> str:
        """The output up to and including the first line starting with ``prefix``."""
        out: list[str] = []
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                line = self.lines.get(timeout=0.2)
            except queue.Empty:
                if self.p.poll() is not None and self.lines.empty():
                    break
                continue
            out.append(line)
            if line.startswith(prefix):
                return "".join(out)
        self.p.kill()
        raise AssertionError(f"no line starting with {prefix!r} within {timeout} s:\n{''.join(out)}")

    def stop(self) -> tuple[int, str]:
        self.p.send_signal(signal.SIGINT)
        code = self.p.wait(timeout=30)
        rest = []
        while not self.lines.empty():
            rest.append(self.lines.get())
        return code, "".join(rest)


def edit(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    time.sleep(0.05)  # a new modification time, even on coarse clocks
    path.write_text(text.replace(old, new))


def test_watch_reproves_what_changed(package):
    w = Watch(package, "mypkg/")
    try:
        first = w.until("watching")
        assert "3 functions: 3 proved" in first
        assert "watching 4 files for changes" in first

        # break `double`: it and `quad`, which imports it, fail; `other` is not re-proved
        edit(package / "mypkg" / "core.py", "return x * 2", "return x[1:] * 2")
        out = w.until("[")
        assert "FAIL  mypkg.core.double" in out
        assert "axis 0 of `return` should be T, but is T - 1." in out
        assert "FAIL  mypkg.user.quad" in out
        assert "mypkg.other" not in out
        assert out.splitlines()[-1].endswith("2 failing")
        assert "mypkg.core: 2 re-proved in" in out

        # a change that keeps it broken in the same way prints only the summary
        edit(package / "mypkg" / "core.py", "return x[1:] * 2", "return x[1:] * 3")
        out = w.until("[")
        assert "FAIL" not in out and out.count("\n") == 1 and "2 failing" in out

        # fix it
        edit(package / "mypkg" / "core.py", "return x[1:] * 3", "return x * 2")
        out = w.until("[")
        assert "fixed mypkg.core.double" in out and "fixed mypkg.user.quad" in out
        assert out.splitlines()[-1].endswith("all clear: 3 functions")

        # a syntax error is reported, and so is its fix; a new module is picked up
        edit(package / "mypkg" / "other.py", "return x", "return x +")
        out = w.until("[")
        assert "ERR   mypkg.other (import)" in out and "SyntaxError" in out and "1 failing" in out
        edit(package / "mypkg" / "other.py", "return x +", "return x")
        out = w.until("[")
        assert "fixed mypkg.other (import)" in out and "all clear: 3 functions" in out
        write(package / "mypkg" / "extra.py", OTHER.replace("ident", "extra").replace("return x", "return x[:-1]"))
        out = w.until("[")
        assert "FAIL  mypkg.extra.extra" in out and "1 failing" in out
    finally:
        code, rest = w.stop()
    assert code == 0
    assert "Traceback" not in rest


def test_watch_survives_a_broken_start(tmp_path):
    write(tmp_path / "single.py", OTHER.replace("return x", "return x +"))
    w = Watch(tmp_path, "single.py")
    try:
        first = w.until("watching")
        assert "ERR   single.py (import)" in first and "SyntaxError" in first
        edit(tmp_path / "single.py", "return x +", "return x")
        out = w.until("[")
        assert "fixed single.py (import)" in out and out.splitlines()[-1].endswith("all clear: 1 function")
    finally:
        code, _ = w.stop()
    assert code == 0


SAMPLE = """\
    import jax.numpy as jnp

    import expex
    from expex import f32


    @expex.fn
    def attend[B, T, D](q: f32[B, T, D], k: f32[B, T, D]) -> f32[B, T, T]:
        scores = q @ k.swapaxes(-1, -2)
        top, idx = scores.max(-1), scores.argmax(-1)
        return scores - top[..., None]


    @expex.fn
    def shift[T, D](x: f32[T, D]) -> f32[T, D]:
        y = x[1:]
        return y


    print("the module's own output goes to stderr")
"""


def test_shapes_json(tmp_path):
    write(tmp_path / "shapes_sample.py", SAMPLE)
    r = expex("shapes", "shapes_sample.py", "--json", cwd=tmp_path)
    assert r.returncode == 1, r.stderr  # there is an error
    assert "the module's own output" in r.stderr
    file = str((tmp_path / "shapes_sample.py").resolve())

    def hint(line: int, column: int, end_column: int, text: str) -> dict:
        return {"file": file, "line": line, "column": column, "end_column": end_column, "text": text}

    assert json.loads(r.stdout) == {
        "hints": [
            hint(9, 4, 10, "f32[B, T, T]"),  # scores
            hint(10, 4, 7, "f32[B, T]"),  # top
            hint(10, 9, 12, "i32[B, T]"),  # idx
            hint(11, 11, 34, "-> f32[B, T, T]"),  # the returned expression
            hint(16, 4, 5, "f32[T - 1, D]"),  # up to the error, too
            hint(17, 11, 12, "-> f32[T - 1, D]"),
        ],
        "diagnostics": [
            {
                "file": file,
                "line": 17,
                "column": 4,
                "end_column": 12,
                "severity": "error",
                "message": "shapes_sample.shift: failed\n"
                "shift(): the return value does not match its annotation.\n"
                "\n"
                "    declared  -> f32[T, D]\n"
                "    returned     f32[T - 1, D]\n"
                "\n"
                "    axis 0 of `return` should be T, but is T - 1.\n"
                "\n"
                "    shift[T, D](x: f32[T, D]) -> f32[T, D]\n"
                f"    defined at {file}:14\n"
                "confirmed at sizes D=2, T=3.",
                "code": "EX102",
                "href": "https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex102",
            }
        ],
    }


def test_shapes_text_and_errors(tmp_path):
    write(tmp_path / "shapes_text.py", SAMPLE)
    r = expex("shapes", "shapes_text.py", cwd=tmp_path)
    assert r.returncode == 1
    assert re.search(r"\n    9 \|     scores = q @ k.swapaxes\(-1, -2\) +# scores: f32\[B, T, T\]\n", r.stdout)
    assert "top: f32[B, T], idx: i32[B, T]" in r.stdout
    assert "shapes_text.py:17: shapes_text.shift: failed" in r.stdout

    # an error inside the body is reported on its line; a syntax error on its own
    write(tmp_path / "shapes_call.py", SAMPLE.replace("y = x[1:]", "y = attend(x, x)[0]"))
    r = expex("shapes", "shapes_call.py", "--json", cwd=tmp_path)
    (d,) = json.loads(r.stdout)["diagnostics"]
    assert d["line"] == 16 and "argument `q` does not match" in d["message"]
    write(tmp_path / "shapes_bad.py", "def f(:\n    pass\n")
    r = expex("shapes", "shapes_bad.py", "--json", cwd=tmp_path)
    (d,) = json.loads(r.stdout)["diagnostics"]
    assert (d["line"], d["severity"]) == (1, "error") and d["message"].startswith("SyntaxError")


def test_docs_page_shows_real_output(tmp_path):
    """The `shapes` example on docs/tools.md is what the command prints."""
    blocks = re.findall(r"```(\w*)\n(.*?)```", (ROOT / "docs" / "tools.md").read_text(), flags=re.S)
    (src,) = [b for lang, b in blocks if lang == "python"]
    (text,) = [b for _, b in blocks if b.startswith("attention[B, T, D]")]
    (data,) = [b for _, b in blocks if b.startswith('{"hints"')]
    (tmp_path / "model.py").write_text(src)
    r = expex("shapes", "model.py", cwd=tmp_path)
    assert r.stdout == text
    r = expex("shapes", "model.py", "--json", cwd=tmp_path)
    assert r.stdout.replace(str((tmp_path / "model.py").resolve()), "/path/to/model.py") == data

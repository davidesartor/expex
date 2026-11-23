"""Static checkers must accept annotated code, and still catch real mistakes."""

import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent
USAGE = ROOT / "tests" / "typing" / "usage.py"
EXPECTED = {
    i for i, line in enumerate(USAGE.read_text().splitlines(), 1) if "# E:" in line
}


def error_lines(output: str) -> set[int]:
    return {
        int(m.group(1))
        for line in output.splitlines()
        if "error" in line and (m := re.search(r"usage\.py:(\d+)", line))
    }


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_pyright():
    r = subprocess.run(["pyright", str(USAGE)], capture_output=True, text=True, cwd=ROOT)
    assert error_lines(r.stdout) == EXPECTED, r.stdout
    r = subprocess.run(["pyright", "src"], capture_output=True, text=True, cwd=ROOT)
    assert "0 errors" in r.stdout, r.stdout


def test_mypy():
    pytest.importorskip("mypy")
    cmd = [sys.executable, "-m", "mypy", "--no-incremental"]
    r = subprocess.run([*cmd, str(USAGE)], capture_output=True, text=True, cwd=ROOT)
    assert error_lines(r.stdout) == EXPECTED, r.stdout
    r = subprocess.run([*cmd, "src/expex"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout

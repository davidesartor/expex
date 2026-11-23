"""The pytest plugin and the command line."""

import subprocess
import sys
import textwrap

MODULE = textwrap.dedent(
    """
    import jax.numpy as jnp
    from typax import fn, f32

    @fn
    def good[B, T](x: f32[B, T]) -> f32[T, B]:
        return x.T

    @fn
    def bad[B, T](x: f32[B, T]) -> f32[B, T]:
        return x[:, 1:]

    @fn
    def needs_example[B](x: f32[B], mode: str) -> f32[B]:
        return x if mode == "a" else -x

    @fn(prove=False)
    def opted_out[B](x: f32[B]) -> f32[B]:
        return x[1:]
    """
)


def test_plugin(pytester):
    pytester.makepyfile(shapes_mod=MODULE)
    pytester.syspathinsert()
    result = pytester.runpytest("--typax", "shapes_mod", "-rs")
    result.assert_outcomes(passed=1, failed=1, skipped=1)
    result.stdout.fnmatch_lines(
        [
            "*returned     f32?B, T - 1?*",
            "*Static parameter `mode` has no default*",
        ]
    )


def test_plugin_ini_and_off_by_default(pytester):
    pytester.makepyfile(shapes_mod=MODULE, test_x="def test_x(): pass")
    pytester.syspathinsert()
    pytester.runpytest().assert_outcomes(passed=1)
    pytester.makeini("[pytest]\ntypax = shapes_mod\n")
    pytester.runpytest().assert_outcomes(passed=2, failed=1, skipped=1)


def test_cli(tmp_path):
    (tmp_path / "shapes_cli.py").write_text(MODULE)
    r = subprocess.run(
        [sys.executable, "-m", "typax", "shapes_cli"],
        cwd=tmp_path, capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 1
    assert "ok    shapes_cli.good" in r.stdout
    assert "FAIL  shapes_cli.bad" in r.stdout
    assert "skip  shapes_cli.needs_example" in r.stdout
    assert "opted_out" not in r.stdout
    assert "3 functions: 1 proved, 1 unchecked, 1 failed" in r.stdout

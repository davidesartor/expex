"""`expex.prove` holds for every example model: a change to the staging or to how annotations
are read must not turn a proved example into an error (the ports' tests run them, not prove them)."""

import importlib.util
import pathlib
import sys

import pytest

import expex

ROOT = pathlib.Path(__file__).parent.parent
EXAMPLES = sorted(
    p for p in (ROOT / "examples").glob("**/*.py")
    if "notebooks" not in p.parts and "__pycache__" not in p.parts
    and p.stem not in ("compare", "run_original")
    and "multidevice" not in p.parts  # a script: it sets up its devices when imported
)  # fmt: skip


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: str(p.relative_to(ROOT / "examples")))
def test_every_example_proves(path):
    name = "example_" + "_".join(path.relative_to(ROOT / "examples").with_suffix("").parts)
    sys.path.insert(0, str(path.parent))  # an example's own helper modules
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except ImportError as e:  # an optional dependency of the example
            pytest.skip(f"{path.name}: {e}")
        reports = expex.prove(module)
    finally:
        sys.path.remove(str(path.parent))
    bad = [str(r) for r in reports if r.status in ("failed", "error")]
    assert not bad, "\n\n".join(bad)

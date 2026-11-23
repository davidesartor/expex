"""Every annotation checks its value or is refused when the function is defined; the prover never
proves a function whose result is wrong for some size. Generated cases: see ``annprops.py``."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import annprops  # noqa: E402


@pytest.mark.parametrize("chunk", range(4))
def test_no_annotation_is_silently_dropped(chunk):
    problems = []
    for seed in range(chunk * 60, (chunk + 1) * 60):
        problems += [f"seed {seed}: {p}" for p in annprops.run_case(annprops.case(seed))]
    assert not problems, "\n".join(problems)


def test_the_prover_never_proves_a_wrong_result():
    problems = [p for s in range(60) if (p := annprops.run_prover_case(s))]
    problems += [p for s in range(40) if (p := annprops.run_prover_case2(s))]
    assert not problems, "\n\n".join(problems)


@pytest.mark.parametrize(
    "ann",
    ["f32[T]", "f32", "f32[T] | None", "tuple[f32[T], f32[T]]", "list[f32[T]]", '"f32[T]"', "Float[T]", "Int"],
)
def test_an_undefined_dtype_is_refused_at_definition(ann):
    head = ann.strip('"').split("[")[0].split(" ")[0].removeprefix("tuple[").removeprefix("list[")
    head = head if head in annprops.DTYPES else "f32"
    src = f"def f[T](a: {ann}, t: i32[T]):\n    return a\n"
    with pytest.raises(TypeError) as e:
        annprops.expex.fn(annprops.define(src, annprops.namespace(drop=head)))
    assert e.value.code == "EX402" and head in str(e.value)


def test_an_undeclared_variadic_is_refused_at_definition():
    src = "def f[T](a: f32[*B, T]):\n    return a\n"
    with pytest.raises(TypeError) as e:
        annprops.expex.fn(annprops.define(src, annprops.namespace()))
    assert e.value.code == "EX402" and "B" in str(e.value)


def test_a_struct_field_with_an_undefined_dtype_is_refused():
    src = (
        "@expex.struct\n"
        "class f[D]:\n"
        "    w: f32[D, D]\n"
    )
    with pytest.raises(TypeError) as e:
        annprops.define(src, annprops.namespace(drop="f32"))
    assert e.value.code == "EX402"

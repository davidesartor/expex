"""pytest plugin: one test per ``@fn``, proving its shape annotations.

Enable with ``pytest --expex mypackage`` or in the config file::

    [tool.pytest.ini_options]
    expex = ["mypackage"]
"""

from typing import Any

import pytest


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("expex")
    group.addoption(
        "--expex",
        action="append",
        default=[],
        metavar="MODULE",
        help="prove the shape annotations of every @fn in this module/package (repeatable)",
    )
    parser.addini("expex", "modules whose @fn functions are proved", type="args", default=[])


class ProofFailed(Exception):
    pass


class ProofItem(pytest.Item):
    def __init__(self, *, fn: Any, **kw: Any) -> None:
        super().__init__(**kw)
        self.fn = fn

    def runtest(self) -> None:
        from ._check import ERROR, FAILED, SPOT, UNCHECKED, prove

        report = prove(self.fn)
        if report.status in (FAILED, ERROR):
            raise ProofFailed(str(report))
        if report.status == UNCHECKED:
            pytest.skip(report.detail)
        if report.status == SPOT:
            self.add_report_section("call", "expex", str(report))

    def repr_failure(self, excinfo: Any, style: Any = None) -> Any:
        if isinstance(excinfo.value, ProofFailed):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self) -> Any:
        code = getattr(self.fn.raw, "__code__", None)
        path = code.co_filename if code else "<unknown>"
        line = code.co_firstlineno - 1 if code else None
        return path, line, f"shape proof: {self.fn.signature}"


def pytest_collection_modifyitems(session: Any, config: Any, items: list[Any]) -> None:
    targets = list(config.getoption("expex")) + list(config.getini("expex"))
    if not targets:
        return
    from ._check import collect

    seen: set[int] = set()
    for t in targets:
        for f in collect(t):
            if id(f) in seen:
                continue
            seen.add(id(f))
            mod = getattr(f.raw, "__module__", "?")
            items.append(ProofItem.from_parent(session, name=f"expex::{mod}.{f.name}", fn=f))

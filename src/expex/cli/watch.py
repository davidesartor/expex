"""``python -m expex prove pkg/ --watch``: prove everything, then re-prove what changes.

One long-lived process: JAX and expex are imported once, and on each save only
the changed modules and the modules that import them (inside the watched
folders) are dropped from ``sys.modules`` and imported again, in dependency
order. Only the ``@fn`` functions defined in those modules are proved again,
and only the differences are printed: functions newly failing (with the usual
report), newly fixed, and one summary line.

Files are polled by modification time, or watched with ``watchfiles`` when it
is installed. A change starts a cycle once the files have been quiet for
``debounce`` seconds, so an editor's several writes make one cycle.
"""

import argparse
import ast
import importlib
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Callable, Iterable, TextIO

FAILING = ("failed", "error")


def _module_files(roots: list[types.ModuleType]) -> dict[str, Path]:
    """The imported modules that live in the watched folders, by name."""
    folders = [_folder(m) for m in roots]
    out: dict[str, Path] = {}
    for name, mod in list(sys.modules.items()):
        file = getattr(mod, "__file__", None)
        if not file or not file.endswith(".py"):
            continue
        path = Path(file).resolve()
        if any(path.is_relative_to(f) for f in folders):
            out[name] = path
    return out


def _folder(mod: types.ModuleType) -> Path:
    file = Path(mod.__file__ or ".").resolve()
    return file.parent


def _is_package(mod: types.ModuleType) -> bool:
    return hasattr(mod, "__path__")


def _imports(name: str, path: Path, known: set[str]) -> set[str]:
    """The modules in ``known`` that ``name`` imports, read from its syntax tree."""
    try:
        tree = ast.parse(path.read_text(), str(path))
    except (OSError, SyntaxError, ValueError):
        return set()
    is_pkg = path.name == "__init__.py"
    package = name if is_pkg else name.rpartition(".")[0]
    found: set[str] = set()

    def add(dotted: str) -> None:
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):  # `import a.b.c` imports a, a.b and a.b.c
            cand = ".".join(parts[:i])
            if cand in known:
                found.add(cand)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                mod = ".".join([*base, *([node.module] if node.module else [])])
            else:
                mod = node.module or ""
            if mod:
                add(mod)
            for alias in node.names:  # `from pkg import sub` imports pkg.sub
                if f"{mod}.{alias.name}" in known:
                    found.add(f"{mod}.{alias.name}")
    found.discard(name)
    return found


def _new_module_name(path: Path, files: dict[str, Path], roots: list[types.ModuleType]) -> str | None:
    """The module name of a new file inside a watched package, if it is one."""
    for root in roots:
        if not _is_package(root):
            continue
        folder = _folder(root)
        if path.is_relative_to(folder):
            rel = path.relative_to(folder).with_suffix("")
            parts = [p for p in rel.parts if p != "__init__"]
            return ".".join([root.__name__, *parts])
    return None


class Watcher:
    """The state of a watch session: the last report of every function."""

    def __init__(
        self,
        targets: list[str],
        out: TextIO = sys.stdout,
        verbose: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.targets = targets
        self.out = out
        self.verbose = verbose
        self.clock = clock
        self.roots: list[types.ModuleType] = []
        self.reports: dict[str, Any] = {}  # function name -> Report
        self.broken: dict[str, str] = {}  # module name -> import error
        self.files: dict[str, Path] = {}
        self.stamps: dict[Path, tuple[int, int]] = {}
        self.pending: dict[str, list[Path]] = {}  # targets that failed to import -> their files

    # -- output ---------------------------------------------------------------

    def say(self, text: str) -> None:
        print(text, file=self.out, flush=True)

    # -- the first run --------------------------------------------------------

    def start(self) -> int:
        from ..tools.prove import format_reports

        sys.path.insert(0, os.getcwd())
        reports = []
        errors: list[str] = []
        for t in self.targets:
            self.pending[t] = _target_files(t)
            reports += self._load(t, errors)
        for r in reports:
            self.reports[r.name] = r
        self.say("\n".join([*errors, format_reports(reports, self.verbose)]))
        self.files = _module_files(self.roots)
        self.stamps = self._scan()
        n = len(self._paths())
        self.say(f"watching {n} file{'s' * (n != 1)} for changes (Ctrl-C to stop)")
        return 1 if errors or any(r.status in FAILING for r in reports) else 0

    def _prove(self, f: Any) -> Any:
        from ..tools.prove import ERROR, Report, _summ, prove

        try:
            return prove(f)
        except Exception as e:  # a bug in a tool should not stop the watch
            return Report(f, ERROR, f"the prover raised:\n    {_summ(e)}")

    # -- watching -------------------------------------------------------------

    def _paths(self) -> set[Path]:
        paths = set(self.files.values())
        for files in self.pending.values():
            for f in files:
                if f.is_dir():
                    paths.update(p.resolve() for p in f.rglob("*.py"))
                else:
                    paths.add(f)
        for root in self.roots:
            if _is_package(root):
                paths.update(p.resolve() for p in _folder(root).rglob("*.py"))
        return paths

    def _scan(self) -> dict[Path, tuple[int, int]]:
        out = {}
        for p in self._paths():
            try:
                st = p.stat()
            except OSError:
                continue
            out[p] = (st.st_mtime_ns, st.st_size)
        return out

    def changed(self) -> set[Path]:
        """Files added, removed or modified since the last scan."""
        now = self._scan()
        diff = {p for p in now.keys() | self.stamps.keys() if now.get(p) != self.stamps.get(p)}
        self.stamps = now
        return diff

    def folders(self) -> list[Path]:
        found = {_folder(m) for m in self.roots}
        found.update(f if f.is_dir() else f.parent for fs in self.pending.values() for f in fs)
        return sorted(found)

    def _load(self, t: str, lines: list[str]) -> list[Any]:
        """Import target ``t`` and every module of it, and prove them; record import errors.

        A module that fails to import is reported in ``lines`` and watched: saving it
        imports it again. A target that fails stays in ``pending``.
        """
        import pkgutil

        from ..function import REGISTRY
        from ..tools.prove import import_target

        def failed(name: str, e: BaseException, path: Path | None) -> None:
            text = _import_error(e, path)
            if self.broken.get(name) != text:
                lines.append(f"ERR   {name} (import)\n{_indent(text)}")
            self.broken[name] = text

        try:
            mod = import_target(t)
        except Exception as e:  # keep watching: the next save may fix it
            failed(t, e, None)
            return []
        del self.pending[t]
        if self.broken.pop(t, None) is not None:
            lines.append(f"fixed {t} (import)")
        self.roots.append(mod)
        if _is_package(mod):
            for info in pkgutil.walk_packages(mod.__path__, mod.__name__ + ".", onerror=lambda n: None):
                if info.name.endswith("__main__") or info.name in sys.modules:
                    continue
                try:
                    importlib.import_module(info.name)
                except Exception as e:
                    failed(info.name, e, None)
        self.files = {**self.files, **_module_files(self.roots)}
        names = {n for n in sys.modules if n == mod.__name__ or n.startswith(mod.__name__ + ".")}
        return [self._prove(f) for f in _functions(names, REGISTRY)]

    def _retry(self, lines: list[str]) -> list[Any]:
        """Import the targets that failed to import before; their reports if they now do."""
        reports = []
        for t in list(self.pending):
            reports += self._load(t, lines)
        return reports

    # -- one cycle ------------------------------------------------------------

    def cycle(self, paths: Iterable[Path]) -> None:
        """Re-import the modules of ``paths`` and their dependents; prove them again."""
        from ..function import REGISTRY

        t0 = self.clock()
        lines: list[str] = []
        paths = set(paths)
        by_path = {p: n for n, p in self.files.items()}
        retried = self._retry(lines) if self.pending else []
        for r in retried:
            if r.status in FAILING:
                lines.append(_block(r))
            self.reports[r.name] = r
        changed: set[str] = set()
        for p in paths:
            name = by_path.get(p) or _new_module_name(p, self.files, self.roots)
            if name is None:
                continue
            changed.add(name)
            if not p.exists():
                self.files.pop(name, None)
                sys.modules.pop(name, None)
            elif name not in self.files:
                self.files[name] = p
        if not changed and not retried and not lines:
            return

        known = set(self.files)
        graph = {n: _imports(n, p, known) for n, p in self.files.items()}
        affected = set(changed)
        grew = True
        while grew:  # everything that imports something affected
            more = {n for n, deps in graph.items() if deps & affected} - affected
            affected |= more
            grew = bool(more)
        # parents of an affected module re-bind it as an attribute; keep them in sync
        order = self._order(affected & known, graph)

        for name in order:
            sys.modules.pop(name, None)
        failed_imports: set[str] = set()
        for name in order:
            if graph.get(name, set()) & failed_imports:
                failed_imports.add(name)  # blocked by a module that failed: reported there
                self.broken.pop(name, None)
                continue
            try:
                importlib.import_module(name)
            except BaseException as e:
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                failed_imports.add(name)
                sys.modules.pop(name, None)
                text = _import_error(e, self.files.get(name))
                if self.broken.get(name) != text:
                    lines.append(f"ERR   {name} (import)\n{_indent(text)}")
                self.broken[name] = text
                continue
            if name in self.broken:
                del self.broken[name]
                lines.append(f"fixed {name} (import)")
        for name in list(self.broken):
            if name not in self.files:
                del self.broken[name]

        # The functions of the re-imported modules, filtered from the registry as `collect` does.
        fresh = {n for n in affected - failed_imports if self._ours(n)}
        new = {}
        for f in _functions(fresh, REGISTRY):
            r = self._prove(f)
            new[r.name] = r
        gone = [n for n, r in self.reports.items() if r.fn.raw.__module__ in fresh and n not in new]

        for name, r in new.items():
            old = self.reports.get(name)
            if r.status in FAILING:
                if old is None or old.status != r.status or str(old) != str(r):
                    lines.append(_block(r))
            elif old is not None and old.status in FAILING:
                lines.append(f"fixed {name}")
            elif r.status not in ("proved",) and (old is None or old.status != r.status):
                lines.append(_block(r))  # newly spot-checked or unchecked
        for name in gone:
            if self.reports[name].status in FAILING:
                lines.append(f"gone  {name}")
            del self.reports[name]
        self.reports.update(new)
        self.files = {**self.files, **_module_files(self.roots)}

        failing = sorted(n for n, r in self.reports.items() if r.status in FAILING)
        n_bad = len(failing) + len(self.broken)
        took = self.clock() - t0
        stamp = time.strftime("%H:%M:%S")
        what = ", ".join(sorted(changed)) or ", ".join(self.targets)
        count = len(new) + len(retried)
        if n_bad:
            summary = f"{n_bad} failing"
        else:
            n = len(self.reports)
            summary = f"all clear: {n} function{'s' * (n != 1)}"
        lines.append(
            f"[{stamp}] {what}: {count} re-proved in {took:.2f} s, {summary}"
        )
        self.say("\n".join(lines))

    def _ours(self, name: str) -> bool:
        """Is ``name`` one of the targets, or inside one? (Others are only re-imported.)"""
        return any(name == r.__name__ or name.startswith(r.__name__ + ".") for r in self.roots)

    def _order(self, names: set[str], graph: dict[str, set[str]]) -> list[str]:
        """``names`` with the modules each imports first (and packages before their modules)."""
        out: list[str] = []
        seen: set[str] = set()

        def visit(n: str) -> None:
            if n in seen:
                return
            seen.add(n)
            parent = n.rpartition(".")[0]
            if parent in names:
                visit(parent)
            for d in sorted(graph.get(n, ())):
                if d in names:
                    visit(d)
            out.append(n)

        for n in sorted(names):
            visit(n)
        return out


def _functions(names: set[str], registry: Any) -> list[Any]:
    """The provable ``@fn`` functions defined in the (imported) modules ``names``."""
    out = []
    for f in list(registry):
        m = getattr(f.raw, "__module__", None)
        mod = sys.modules.get(m or "")
        if m in names and f.prove and mod is not None and getattr(f.raw, "__globals__", None) is vars(mod):
            out.append(f)
    out.sort(key=lambda f: (f.raw.__module__, getattr(getattr(f.raw, "__code__", None), "co_firstlineno", 0), f.name))
    return out


def _target_files(t: str) -> list[Path]:
    """The files to watch for a target that could not be imported."""
    if t.endswith(".py") or os.sep in t or os.path.isdir(t):
        return [Path(t).resolve()]
    import importlib.util

    try:
        spec = importlib.util.find_spec(t)
    except Exception:
        spec = None
    if spec is None or not spec.origin:
        raise SystemExit(f"expex: cannot find module {t!r}")
    origin = Path(spec.origin).resolve()
    return [origin.parent if spec.submodule_search_locations else origin]


def _indent(s: str, by: str = "        ") -> str:
    return "\n".join(by + line if line else line for line in s.splitlines())


def _block(r: Any) -> str:
    mark = {"proved": "ok  ", "spot-checked": "spot", "unchecked": "skip", "error": "ERR ", "failed": "FAIL"}
    body = "\n".join([r.detail, *r.notes]).strip()
    return f"{mark[r.status]}  {r.name}" + (f"\n{_indent(body)}" if body else "")


def _import_error(e: BaseException, path: Path | None) -> str:
    if isinstance(e, SyntaxError):
        where = f"{e.filename}:{e.lineno}" if e.filename else ""
        return f"SyntaxError: {e.msg}" + (f" at {where}" if where else "")
    frames = traceback.extract_tb(e.__traceback__)
    mine = [fr for fr in frames if path is not None and Path(fr.filename).resolve() == path]
    at = f" at {mine[-1].filename}:{mine[-1].lineno}" if mine else ""
    first = (str(e).strip().splitlines() or [""])[0]
    return f"{type(e).__name__}: {first}{at}"


def _wait_polling(w: Watcher, interval: float, debounce: float) -> set[Path]:
    while True:
        time.sleep(interval)
        paths = w.changed()
        if paths:
            while True:  # debounce: until the files are quiet
                time.sleep(debounce)
                more = w.changed()
                if not more:
                    return paths
                paths |= more


def _watchfiles_waiter(w: Watcher, debounce: float) -> Callable[[], set[Path]] | None:
    try:
        import watchfiles  # optional
    except ImportError:
        return None
    events = watchfiles.watch(*w.folders(), debounce=int(debounce * 1000), step=50)

    def wait() -> set[Path]:
        while True:
            next(events)
            paths = w.changed()
            if paths:
                return paths

    return wait


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="expex prove --watch",
        description="Prove every @fn, then prove again the modules that change.",
    )
    ap.add_argument("targets", nargs="+", help="module/package names, or paths to .py files or package folders")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--x64", action="store_true", help="enable 64-bit floats before importing")
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between polls (default 0.1)")
    ap.add_argument("--debounce", type=float, default=0.1, help="quiet seconds before a cycle (default 0.1)")
    ap.add_argument("--poll", action="store_true", help="poll even if watchfiles is installed")
    args = ap.parse_args(argv)
    try:
        if args.x64:
            import jax

            jax.config.update("jax_enable_x64", True)
        w = Watcher(args.targets, verbose=args.verbose)
        w.start()
        wait = None if args.poll else _watchfiles_waiter(w, args.debounce)
        while True:
            paths = wait() if wait else _wait_polling(w, args.interval, args.debounce)
            w.cycle(paths)
    except KeyboardInterrupt:
        print(file=sys.stdout, flush=True)
        return 0

"""The command line: ``python -m expex``.

::

    python -m expex pkg/                      # prove every @fn (as `prove`)
    python -m expex prove pkg/ [--watch]      # the same; --watch re-proves on save
    python -m expex shapes file.py [--json]   # the shape of every line, and failures
    python -m expex migrate jaxtyping file.py [--write]  # jaxtyping annotations in expex's style

A first argument that is not a subcommand is a target to prove, so
``python -m expex pkg/`` keeps working. (A module literally named ``prove``,
``shapes`` or ``migrate`` is proved with ``python -m expex prove prove``.)
"""

import sys

USAGE = """\
usage: python -m expex [prove] TARGET [TARGET ...] [-v] [--x64] [--watch]
       python -m expex shapes FILE [--json] [--x64]
       python -m expex migrate jaxtyping PATH [PATH ...] [--write] [--float f32]

  prove    prove the shape annotations of every @fn in modules, packages or files
           (the default); --watch keeps proving the files that change
  shapes   the shape of every assignment and return of each typed function in FILE,
           and its shape errors and proof failures (--json: for editors)
  migrate  rewrite jaxtyping annotations in expex's style: prints a diff, --write applies it
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0 if argv else 2
    if argv[0] == "shapes":
        from .shapes import main as shapes_main

        return shapes_main(argv[1:])
    if argv[0] == "migrate":
        from ..tools.migrate import main as migrate_main

        return migrate_main(argv[1:])
    if argv[0] == "prove":
        argv = argv[1:]
    if "-h" in argv or "--help" in argv:
        print(USAGE, end="")
        return 0
    if "--watch" in argv:
        from .watch import main as watch_main

        return watch_main([a for a in argv if a != "--watch"])
    from ..tools.prove import main as prove_main

    return prove_main(argv)

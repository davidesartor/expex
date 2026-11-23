"""The exceptions expex raises, and the codes that link each one to the guide.

Every error a user can hit has a short code (``EX101``) and ends with a line
pointing at the section of ``docs/errors.md`` that explains the fix. Codes are
attached when an error is built (:func:`coded`), never on a call that succeeds.
"""

import re
from typing import Any

#: Where the guide's error page lives; each code is a section of it (``#ex101``).
DOCS_URL = "https://github.com/davidesartor/expex/blob/main/docs/errors.md"

#: Every code, with the title of its section. Hundreds group them: 1 checked calls,
#: 2 traced values and control flow, 3 modules, 4 sizes and annotations,
#: 5 checkpoints, 6 sharding, 7 the prover and the tools, 8 decorators and settings.
CODES = {
    "EX101": "An argument does not match its annotation",
    "EX102": "The return value does not match its annotation",
    "EX103": "An annotated local variable does not match",
    "EX104": "A constraint on the sizes is violated",
    "EX105": "The batch axes of the arguments do not broadcast",
    "EX106": "A variadic shape cannot be vectorized",
    "EX201": "A traced value where Python needs a concrete one",
    "EX202": "A static parameter or a size is traced",
    "EX203": "A condition is an array, not a single True/False",
    "EX204": "The branches of a traced `if` disagree",
    "EX205": "A list grows inside traced control flow",
    "EX206": "A variable first assigned in a traced loop is used after it",
    "EX207": "A non-array value changes inside a traced loop",
    "EX208": "`range()` with a traced step",
    "EX209": "A Python object is updated inside traced control flow",
    "EX210": "Collected values of unknown number while proving",
    "EX211": "`expex.rows` / `expex.stack` with nothing to stack",
    "EX212": "A variable changes shape or dtype inside a traced loop",
    "EX213": "A traced condition decides whether the function returns a value",
    "EX221": "A key is needed while tracing, with no key in scope",
    "EX222": "The key in scope was set outside jit/vmap/grad",
    "EX223": "A random function under `jax.vmap` shares one key",
    "EX301": "A size is missing when building a module",
    "EX302": "A field is missing",
    "EX303": "Unknown or misplaced constructor arguments",
    "EX304": "A field cannot be built from its annotation",
    "EX305": "The module does not match its annotations",
    "EX306": "A struct is declared or used wrongly",
    "EX307": "`model.at` names a part that does not exist",
    "EX308": "`model.at` gets a value that does not fit",
    "EX401": "A malformed annotation",
    "EX402": "An annotation uses an undefined name",
    "EX403": "A size expression cannot be parsed",
    "EX404": "A size declared with `Name:size` conflicts",
    "EX405": "A size used in a helper defined outside the function",
    "EX406": "A size used by name in a body that is not converted",
    "EX407": "A size is not known in this call",
    "EX408": "An axis named by its size cannot be found",
    "EX409": "`expex.einsum` / `expex.rearrange` misused",
    "EX410": "An annotation used as a value",
    "EX411": "A jaxtyping annotation expex cannot read",
    "EX501": "A field cannot be saved or loaded",
    "EX502": "The checkpoint's keys do not match the class",
    "EX503": "A loaded array does not match its annotation",
    "EX504": "An unknown size given to `expex.load`",
    "EX505": "A static field is missing from the checkpoint",
    "EX601": "The mesh has no such axis",
    "EX602": "A mesh axis does not divide a size",
    "EX603": "`expex.shardings` needs a value",
    "EX701": "The prover cannot invent a value",
    "EX702": "Holds at the sizes tried, but not proved",
    "EX703": "No sizes satisfy the constraints",
    "EX704": "The function raised while traced on abstract inputs",
    "EX705": "A tool was given something it cannot inspect",
    "EX706": "An unknown size or parameter given to a tool",
    "EX707": "`expex.annotate` cannot use the names or the source",
    "EX708": "A diagram cannot be saved or explained",
    "EX801": "An invalid decorator option",
    "EX802": "`@expex.random` / `@expex.vectorize` above `@jax.jit`",
    "EX803": "`expex.config` has no such setting",
    "EX804": "`eps` / `tiny` / `huge` on a non-float array",
}

_SEE = re.compile(r"\n[ \t]*see \S+#ex\d{3}$", re.M)
_CODE = re.compile(r"EX\d{3}")


class ShapeError(TypeError):
    """A value does not match the shape/dtype annotation it was declared with."""

    #: The error's code (``EX101``), the section of ``docs/errors.md`` that explains it.
    code: str | None = None

    def _repr_html_(self) -> str:
        """In a notebook: the message, and ``e.diagram`` drawn below it (traced now, not when raised)."""
        from ..tools.display import error_html

        return error_html(self)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        from ..tools.display import error_text

        p.text(error_text(self))


class StageError(TypeError):
    """A value that must be known at trace time was traced (or vice versa)."""

    #: The error's code (``EX201``), the section of ``docs/errors.md`` that explains it.
    code: str | None = None


class Mismatch(Exception):
    """Internal: raised while matching a value against a spec.

    ``Fn`` turns it into a :class:`ShapeError` with the full context.
    """

    def __init__(self, path: str, spec: object, value: object, reason: str):
        super().__init__(reason)
        self.path = path
        self.spec = spec
        self.value = value
        self.reason = reason


def see(code: str) -> str:
    """The line that ends a message with code ``code``: where the guide explains the fix."""
    return f"  see {DOCS_URL}#{code.lower()}"


def strip_see(text: str) -> str:
    """``text`` without the "see ..." lines of the errors it quotes."""
    return _SEE.sub("", text)


def coded[E: BaseException](code: str, err: E) -> E:
    """Give ``err`` the code ``code``: as ``err.code``, and as a last line linking to the guide.

    A message that quotes another error loses that error's link: one link, at the end.
    """
    msg = strip_see(str(err.args[0])) if err.args else ""
    err.args = (f"{msg}\n{see(code)}" if msg else see(code).strip(), *err.args[1:])
    setattr(err, "code", code)
    return err


def coded_note(code: str, err: BaseException, note: str) -> None:
    """Add ``note`` to someone else's exception (JAX's, Python's), with the code's link."""
    err.add_note(f"{note}\n{see(code)}")
    if code_of(err) is None:
        setattr(err, "code", code)


def code_of(err: BaseException | None) -> str | None:
    """The expex code of ``err`` (``"EX101"``), or None."""
    code: Any = getattr(err, "code", None)
    return code if isinstance(code, str) and _CODE.fullmatch(code) else None

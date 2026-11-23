
class ShapeError(TypeError):
    """A value does not match the shape/dtype annotation it was declared with."""


class StageError(TypeError):
    """A value that must be known at trace time was traced (or vice versa)."""


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

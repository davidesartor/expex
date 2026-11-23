"""``expex.eps``, ``expex.tiny``, ``expex.huge``: dtype limits that take the dtype of what they meet.

::

    jnp.log(p + expex.tiny)          # the smallest normal number of p's dtype
    x / (norm + 1e3 * expex.eps)     # scaled, still lazy until it meets `norm`
    jnp.maximum(x, expex.tiny(x))    # as a function argument: give it the array

In an operation with an array (``+ - * /``, comparisons), the constant becomes
a scalar of that array's floating dtype, so bfloat16 gets bfloat16's epsilon
and float64 gets float64's. Multiplying by a Python number keeps it lazy. With
a Python number alone it takes JAX's default float dtype.
"""

import operator
from typing import Any

import jax.numpy as jnp
import numpy as np

from .annotations.errors import coded

_LIMITS = {"eps": "eps", "tiny": "smallest_normal", "huge": "max"}


class Limit:
    """A dtype limit, times a scale, resolved against the array it is used with."""

    __slots__ = ("kind", "scale")
    __array_ufunc__ = None  # NumPy arrays defer to us, as JAX arrays do

    def __init__(self, kind: str, scale: float = 1.0):
        self.kind, self.scale = kind, scale

    def __call__(self, like: Any) -> Any:
        """The value for ``like``'s dtype (an array, a dtype or a Python number)."""
        if isinstance(like, (np.dtype, type)):
            dtype = np.dtype(like)
        elif hasattr(like, "dtype"):
            dtype = like.dtype
        else:
            dtype = jnp.result_type(like)  # a Python number: JAX's default float
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise coded("EX804", TypeError(f"expex.{self.kind} needs a floating-point array, not one of dtype {dtype}"))
        real = jnp.finfo(dtype)
        return np.asarray(getattr(real, _LIMITS[self.kind]) * self.scale, dtype=real.dtype)

    def _with(self, other: Any, op: Any, swap: bool) -> Any:
        if isinstance(other, Limit):
            return NotImplemented
        number = isinstance(other, (int, float)) and not isinstance(other, bool)
        if number and (op is operator.mul or (op is operator.truediv and not swap)):
            return Limit(self.kind, op(self.scale, other))  # 1e3 * expex.eps: still lazy
        value = self(other)
        return op(other, value) if swap else op(value, other)

    def __add__(self, o: Any) -> Any:
        return self._with(o, operator.add, False)

    def __radd__(self, o: Any) -> Any:
        return self._with(o, operator.add, True)

    def __sub__(self, o: Any) -> Any:
        return self._with(o, operator.sub, False)

    def __rsub__(self, o: Any) -> Any:
        return self._with(o, operator.sub, True)

    def __mul__(self, o: Any) -> Any:
        return self._with(o, operator.mul, False)

    def __rmul__(self, o: Any) -> Any:
        return self._with(o, operator.mul, True)

    def __truediv__(self, o: Any) -> Any:
        return self._with(o, operator.truediv, False)

    def __rtruediv__(self, o: Any) -> Any:
        return self._with(o, operator.truediv, True)

    def __neg__(self) -> "Limit":
        return Limit(self.kind, -self.scale)

    def __lt__(self, o: Any) -> Any:
        return self._with(o, operator.lt, False)

    def __le__(self, o: Any) -> Any:
        return self._with(o, operator.le, False)

    def __gt__(self, o: Any) -> Any:
        return self._with(o, operator.gt, False)

    def __ge__(self, o: Any) -> Any:
        return self._with(o, operator.ge, False)

    def __repr__(self) -> str:
        return f"expex.{self.kind}" if self.scale == 1 else f"{self.scale:g} * expex.{self.kind}"


#: Machine epsilon: the gap between 1 and the next number, in the dtype it meets.
eps = Limit("eps")
#: The smallest positive normal number, in the dtype it meets.
tiny = Limit("tiny")
#: The largest finite number, in the dtype it meets.
huge = Limit("huge")

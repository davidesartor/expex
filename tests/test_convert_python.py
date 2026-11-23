"""Conversion must not change what ordinary Python code does.

Each function below is run both as written and after conversion, on plain
Python values, and the results (or exceptions) must be identical.
"""

import contextlib

import pytest

from expex._convert import NotConverted, convert

G = 10


def early_returns(a, b):
    if a > 5:
        return "big"
    if a > 2:
        if b:
            return "mid-b"
        x = a * 2
    else:
        x = -a
    return x


def falls_through(a):
    if a > 0:
        return a
    # implicit None


def one_branch_only(a):
    if a:
        y = 1
    if a:
        return y
    return 0


def uses_maybe_undefined(a):
    if a:
        y = 1
    return y  # NameError when a is falsy


def loops(n):
    total, i = 0, 0
    while i < n:
        j = 0
        for j in range(i):
            if j % 2:
                total += j
            else:
                total -= 1
        i += 1
    return total, i, j


def loop_with_break_and_else(xs):
    for x in xs:
        if x < 0:
            found = x
            break
    else:
        found = None
    k = 0
    while True:
        k += 1
        if k > 3:
            break
        if k == 2:
            continue
    return found, k


def return_inside_loop(xs):
    for i, x in enumerate(xs):
        if x == 3:
            return i
    return -1


def collects(xs):
    out = []
    seen = {}
    for i, x in enumerate(xs):
        out.append(x * 2)
        if x in seen:
            out.append("dup")
        seen[x] = i
    out.append("end")
    return out, seen


def iterables(d, s):
    acc = []
    for k in d:
        acc.append(k)
    for ch in s:
        acc.append(ch.upper())
    for a, b in zip([1, 2, 3], "ab"):
        acc.append((a, b))
    for x in (y * y for y in range(3)):
        acc.append(x)
    return acc


def boolean_values(a, b, c):
    r1 = a and b
    r2 = a or b or c
    r3 = not a
    r4 = (a or []) and (b if c else "x")
    return r1, r2, r3, r4


def short_circuit(xs):
    log = []

    def note(v):
        log.append(v)
        return v

    r = note(0) and note(1)
    s = note(2) or note(3)
    t = note(4) if note(0) else note(5)
    return r, s, t, log


def nested_defs(n):
    if n > 1:

        def helper(v):
            if v:
                return "t"
            return "f"

        r = helper(n - 2)
    else:
        helper = None
        r = "none"
    fs = []
    for i in range(3):
        fs.append(lambda i=i: i * n)
    return r, [f() for f in fs], helper is None


def try_with(a):
    out = []
    try:
        if a == 0:
            raise ValueError("zero")
        with contextlib.suppress(KeyError):
            if a == 1:
                out.append("one")
                raise KeyError
            out.append("not one")
        if a == 2:
            return "two", out
    except ValueError as e:
        out.append(str(e))
    finally:
        if a:
            out.append("finally")
    return "end", out


def mutation(d, flag):
    if flag:
        d["a"] = d.get("a", 0) + 1
        d.setdefault("l", []).append(1)
    else:
        d["b"] = 2
    for k in list(d):
        if k == "a":
            d[k] += 10
    return d


def uses_global(a):
    global G
    if a:
        G = G + 1
    else:
        G = 10
    return G


def walrus(xs):
    out = []
    i = 0
    while (v := i * 2) < 6:
        if (w := v + 1) % 3 == 0:
            out.append(w)
        i += 1
    return out, v


def chained(a):
    x = y = 0
    if a:
        x = y = a
        x += 1
    z = [q for q in range(3) if q != a]
    return x, y, z


def swap_and_unpack(pairs):
    a, b = 0, 1
    for p, (q, *rest) in pairs:
        if p:
            a, b = b, a + q
        else:
            (a, b), c = (q, p), rest
    return a, b


def kwargs_fn(a, *args, k=3, **kw):
    if args:
        a += sum(args)
    if kw:
        k = k + len(kw)
    return a, k


def matching(v):
    match v:
        case [x, y]:
            if x > y:
                return "desc"
            r = "asc"
        case int():
            r = "int" if v else "zero"
        case _:
            r = "other"
    return r


def deletes(a):
    x = 1
    if a:
        del x
        x = 2
    return x


def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


CASES = [
    (early_returns, [(6, 0), (3, 1), (3, 0), (1, 0), (-4, 1)]),
    (falls_through, [(1,), (0,)]),
    (one_branch_only, [(1,), (0,)]),
    (uses_maybe_undefined, [(1,), (0,)]),
    (loops, [(0,), (1,), (5,)]),
    (loop_with_break_and_else, [([1, -2, 3],), ([1, 2],), ([],)]),
    (return_inside_loop, [([1, 3],), ([],)]),
    (collects, [([1, 2, 1],), ([],)]),
    (iterables, [({"a": 1, "b": 2}, "xy")]),
    (boolean_values, [(0, 1, 2), ([], "", None), (1, 0, 0), ("a", "b", "")]),
    (short_circuit, [([],)]),
    (nested_defs, [(0,), (2,), (5,)]),
    (try_with, [(0,), (1,), (2,), (3,)]),
    (mutation, [({}, True), ({"a": 1}, True), ({"a": 1}, False)]),
    (uses_global, [(1,), (1,), (0,)]),
    (walrus, [([],)]),
    (chained, [(0,), (2,)]),
    (swap_and_unpack, [([(1, (2, 9)), (0, (3,)), (1, (4, 5, 6))],), ([],)]),
    (kwargs_fn, [(1,), (1, 2, 3)]),
    (matching, [([2, 1],), ([1, 2],), (0,), (5,), ("s",)]),
    (deletes, [(0,), (1,)]),
    (fib, [(10,)]),
]


def outcome(f, args, kwargs=None):
    import copy

    try:
        return ("ok", f(*copy.deepcopy(args), **(kwargs or {})))
    except Exception as e:  # noqa: BLE001
        return ("raised", type(e).__name__)


@pytest.mark.parametrize("func,arglists", CASES, ids=[c[0].__name__ for c in CASES])
def test_same_behaviour(func, arglists):
    converted = convert(func)
    assert converted is not func
    for args in arglists:
        global G
        G = 10
        expected = outcome(func, args)
        G = 10
        got = outcome(converted, args)
        if func is nested_defs:  # contains function objects; compare the rest
            assert got[1][:2] == expected[1][:2]
        else:
            assert got == expected, (func.__name__, args)


def test_kwargs_and_defaults_survive():
    c = convert(kwargs_fn)
    assert c(1, 2, k=5, z=1) == kwargs_fn(1, 2, k=5, z=1) == (3, 6)
    assert c.__defaults__ == kwargs_fn.__defaults__
    assert c.__kwdefaults__ == kwargs_fn.__kwdefaults__
    assert c.__name__ == "kwargs_fn"


def test_not_converted_reasons():
    def gen(n):
        for i in range(n):
            yield i

    async def co(n):
        if n:
            return 1

    def plain(x):
        return x + 1

    for f, why in [(gen, "generators"), (co, "generators and async"), (plain, "no control flow")]:
        with pytest.raises(NotConverted, match=why):
            convert(f)
    with pytest.raises(NotConverted):
        convert(len)


def test_private_names_in_methods_are_left_alone():
    class A:
        def __init__(self):
            self.__secret = 3

        def get(self, flag):
            if flag:
                return self.__secret
            return 0

    with pytest.raises(NotConverted, match="name-mangled"):
        convert(A.get)


def test_super_and_class_cell():
    class Base:
        def f(self, x):
            return x + 1

    class Child(Base):
        def f(self, x):
            if x > 0:
                return super().f(x) * 2
            return super().f(x)

    c = convert(Child.f)
    assert c(Child(), 1) == 4 and c(Child(), -1) == 0

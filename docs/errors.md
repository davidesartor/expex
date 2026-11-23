# Errors

Every error expex raises has a short code, `EX` and three digits, and its
message ends with a link to the section of this page that explains it:

```
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex101
```

The code is also `e.code` on the exception, `report.code` on a failed
[`expex.prove`](functions.md) report, and `code` (with `href`, the link) in the
diagnostics of [`python -m expex shapes --json`](tools.md), which the editor
extension shows next to each diagnostic. Some errors are JAX's or Python's own
(`TracerBoolConversionError`, `NameError`): expex adds a note to them, ending
with the link, and sets `e.code` too.

Each section says what triggers the error, shows a minimal example that
raises it and the message it prints (file paths shortened to `example.py`,
lines counted from the start of the example), and how to fix it. The examples
run as one program, after these imports:

```python
import jax, jax.numpy as jnp, jax.random as jr
import expex
from expex import f32, i32, Static
```

**1xx · Checked calls**

| Code | Error |
|---|---|
| [EX101](#ex101) | An argument does not match its annotation |
| [EX102](#ex102) | The return value does not match its annotation |
| [EX103](#ex103) | An annotated local variable does not match |
| [EX104](#ex104) | A constraint on the sizes is violated |
| [EX105](#ex105) | The batch axes of the arguments do not broadcast |
| [EX106](#ex106) | A variadic shape cannot be vectorized |

**2xx · Traced values, control flow and keys**

| Code | Error |
|---|---|
| [EX201](#ex201) | A traced value where Python needs a concrete one |
| [EX202](#ex202) | A static parameter or a size is traced |
| [EX203](#ex203) | A condition is an array, not a single True/False |
| [EX204](#ex204) | The branches of a traced `if` disagree |
| [EX205](#ex205) | A list grows inside traced control flow |
| [EX206](#ex206) | A variable first assigned in a traced loop is used after it |
| [EX207](#ex207) | A non-array value changes inside a traced loop |
| [EX208](#ex208) | `range()` with a traced step |
| [EX209](#ex209) | A Python object is updated inside traced control flow |
| [EX210](#ex210) | Collected values of unknown number while proving |
| [EX211](#ex211) | `expex.rows` / `expex.stack` with nothing to stack |
| [EX221](#ex221) | A key is needed while tracing, with no key in scope |
| [EX222](#ex222) | The key in scope was set outside jit/vmap/grad |
| [EX223](#ex223) | A random function under `jax.vmap` shares one key |

**3xx · Modules**

| Code | Error |
|---|---|
| [EX301](#ex301) | A size is missing when building a module |
| [EX302](#ex302) | A field is missing |
| [EX303](#ex303) | Unknown or misplaced constructor arguments |
| [EX304](#ex304) | A field cannot be built from its annotation |
| [EX305](#ex305) | The module does not match its annotations |
| [EX306](#ex306) | A struct is declared or used wrongly |
| [EX307](#ex307) | `model.at` names a part that does not exist |
| [EX308](#ex308) | `model.at` gets a value that does not fit |

**4xx · Sizes and annotations**

| Code | Error |
|---|---|
| [EX401](#ex401) | A malformed annotation |
| [EX402](#ex402) | An annotation uses an undefined name |
| [EX403](#ex403) | A size expression cannot be parsed |
| [EX404](#ex404) | A size declared with `Name:size` conflicts |
| [EX405](#ex405) | A size used in a helper defined outside the function |
| [EX406](#ex406) | A size used by name in a body that is not converted |
| [EX407](#ex407) | A size is not known in this call |
| [EX408](#ex408) | An axis named by its size cannot be found |
| [EX409](#ex409) | `expex.einsum` / `expex.rearrange` misused |
| [EX410](#ex410) | An annotation used as a value |
| [EX411](#ex411) | A jaxtyping annotation expex cannot read |

**5xx · Checkpoints**

| Code | Error |
|---|---|
| [EX501](#ex501) | A field cannot be saved or loaded |
| [EX502](#ex502) | The checkpoint's keys do not match the class |
| [EX503](#ex503) | A loaded array does not match its annotation |
| [EX504](#ex504) | An unknown size given to `expex.load` |
| [EX505](#ex505) | A static field is missing from the checkpoint |

**6xx · Sharding**

| Code | Error |
|---|---|
| [EX601](#ex601) | The mesh has no such axis |
| [EX602](#ex602) | A mesh axis does not divide a size |
| [EX603](#ex603) | `expex.shardings` needs a value |

**7xx · The prover and the tools**

| Code | Error |
|---|---|
| [EX701](#ex701) | The prover cannot invent a value |
| [EX702](#ex702) | Holds at the sizes tried, but not proved |
| [EX703](#ex703) | No sizes satisfy the constraints |
| [EX704](#ex704) | The function raised while traced on abstract inputs |
| [EX705](#ex705) | A tool was given something it cannot inspect |
| [EX706](#ex706) | An unknown size or parameter given to a tool |
| [EX707](#ex707) | `expex.annotate` cannot use the names or the source |
| [EX708](#ex708) | A diagram cannot be saved or explained |

**8xx · Decorators and settings**

| Code | Error |
|---|---|
| [EX801](#ex801) | An invalid decorator option |
| [EX802](#ex802) | `@expex.random` / `@expex.vectorize` above `@jax.jit` |
| [EX803](#ex803) | `expex.config` has no such setting |
| [EX804](#ex804) | `eps` / `tiny` / `huge` on a non-float array |

## 1xx · Checked calls

<a id="ex101"></a>

### EX101 · An argument does not match its annotation

A call's argument does not have the shape or dtype its annotation declares. The
message names the size that disagrees and where its value came from.

```python
@expex.fn
def scale[N](x: f32[N], w: f32[N]) -> f32[N]:
    return x * w

try:
    scale(jnp.ones(3), jnp.ones(4))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX101 scale(): argument `w` does not match its annotation.

    declared  w: f32[N]
    received     f32[4]

    N is 3 (from `x: f32[3]`), but axis 0 of `w` is 4.

    scale[N](x: f32[N], w: f32[N]) -> f32[N]
    defined at example.py:1
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex101
```

**Fix.** Pass arrays whose sizes agree, or fix the annotation if it is the annotation
that is wrong. The hint under the message covers the usual causes: an extra
leading axis (map over it with `f.vectorize(...)`), permuted axes, or a 64-bit
NumPy array where `f32` is declared (`jnp.asarray(x, jnp.float32)`).

<a id="ex102"></a>

### EX102 · The return value does not match its annotation

The value a function returns does not match its return annotation.

```python
@expex.fn
def shift[T](x: f32[T]) -> f32[T]:
    return x[1:]

try:
    shift(jnp.ones(5))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX102 shift(): the return value does not match its annotation.

    declared  -> f32[T]
    returned     f32[4]

    T is 5 (from `x: f32[5]`), but axis 0 of `return` is 4.
    Bound so far: T=5.

    shift[T](x: f32[T]) -> f32[T]
    defined at example.py:1
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex102
```

**Fix.** Either the body or the annotation is wrong: here the result has `T - 1` rows, so
declare `-> f32[T - 1]`, or pad the result back to `T`. `expex.prove(shift)`
finds these without running the function.

<a id="ex103"></a>

### EX103 · An annotated local variable does not match

A local variable annotated in the body (`y: f32[T] = ...`) is assigned a value
that does not match its annotation.

```python
@expex.fn
def double[T](x: f32[T]) -> f32[2 * T]:
    y: f32[T] = jnp.concatenate([x, x])
    return y

try:
    double(jnp.ones(3))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX103 double(): local variable `y` does not match its annotation.

    declared  y: f32[T]
    assigned     f32[6]

    T is 3 (from `x: f32[3]`), but axis 0 of `y` is 6.
    Bound so far: T=3.

    at example.py:3
    > y: f32[T] = jnp.concatenate([x, x])

    double[T](x: f32[T]) -> f32[2 * T]
    defined at example.py:1
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex103
```

**Fix.** The message gives the line (`at file:line`). Fix the annotation of the local
(`y: f32[2 * T]`) or the expression assigned to it.

<a id="ex104"></a>

### EX104 · A constraint on the sizes is violated

A constraint on the sizes (a leading `assert` such as `assert D % heads == 0`)
does not hold for the arguments of a call.

```python
@expex.fn
def split_heads[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]:
    assert D % heads == 0
    return x.reshape(T, heads, D // heads).transpose(1, 0, 2)

try:
    split_heads(jnp.ones((5, 8)), heads=3)
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX104 split_heads(): constraint `D % heads == 0` is violated (D=8, heads=3).

    split_heads[T, D](x: f32[T, D], heads: int) -> f32[heads, T, D // heads]
    defined at example.py:1
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex104
```

**Fix.** Call the function with sizes that satisfy the constraint (here a `heads` that
divides `D`). If the constraint is stricter than the body needs, relax it.

<a id="ex105"></a>

### EX105 · The batch axes of the arguments do not broadcast

A call is mapped over extra leading axes (automatically, or with
`f.vectorize(...)`), and the extra axes of the arguments do not broadcast
against each other.

```python
@expex.fn
def dot[N](a: f32[N], b: f32[N]) -> f32:
    return a @ b

try:
    dot.vectorize(jnp.ones((2, 3)), jnp.ones((4, 3)))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX105 dot.vectorize(): the batch axes of the arguments do not broadcast.

    a: f32[2, 3]  = batch [2] + f32[N]
    b: f32[4, 3]  = batch [4] + f32[N]

    dot[N](a: f32[N], b: f32[N]) -> f32[()]
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex105
```

**Fix.** Give the arguments batch axes that broadcast (equal, or 1): here `[2]` and
`[4]` do not. Add an axis to one of them (`a[:, None]`) to get every pair.

<a id="ex106"></a>

### EX106 · A variadic shape cannot be vectorized

An argument whose annotation has a variadic shape (`f32[*S]`, `f32[...]`) is
mapped over: its batch axes cannot be told apart from its own axes.

```python
@expex.fn
def total[*S](x: f32[*S]) -> f32:
    return x.sum()

try:
    total.vectorize(jnp.ones((2, 3)))
except TypeError as e:
    print(e.code, e)
```

```
EX106 total.vectorize(): `x: f32[*S]` has a variadic shape, so its batch axes cannot be told apart from its own axes.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex106
```

**Fix.** Map with `jax.vmap` yourself, which says which axis is mapped, or give the
argument a fixed number of axes in its annotation.

## 2xx · Traced values, control flow and keys

<a id="ex201"></a>

### EX201 · A traced value where Python needs a concrete one

Inside `jax.jit`, `jax.vmap` or `jax.grad` arrays have no values, only shapes,
and the code asks Python for one: an `if` in a plain helper function, `int(x)`,
`np.asarray(x)`, a boolean mask that changes a shape. JAX raises its own error
(here a `TracerBoolConversionError`); expex adds a note saying what needs the
value, with the code. The `if`, `while` and `for` of an `@expex.fn` body work on
traced values: this is about the code they call.

```python
def clip(v):  # a plain helper: its `if` is Python's
    if v > 1:
        return 1.0
    return v

@expex.fn(jit=True)
def squash(x: f32) -> f32:
    return clip(x)

try:
    squash(3.0)
except jax.errors.TracerBoolConversionError as e:
    print(e.code, e.__notes__[0])
```

```
EX201 expex: `if v > 1:`  (example.py:2) needs a concrete Python bool (for `if`, `while`, `and`/`or`, `assert`),
    but the value is traced: inside jit/vmap/grad, array arguments have no values, only shapes.
    - to choose between two values:        jnp.where(cond, a, b)
    - to choose between two computations:  jax.lax.cond(cond, f, g, *operands)
    - to loop until a condition:           jax.lax.while_loop(cond_fn, body_fn, init)
    - if the value is configuration rather than data, declare the parameter Static[...] (array parameters of squash: x)
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex201
```

**Fix.** Make the helper an `@expex.fn` (its body is converted too), define it inside
the function, or keep the computation in JAX: `jnp.where(cond, a, b)` to choose
a value, `jax.lax.cond` / `jax.lax.while_loop` for control flow, `x[i]` with a
traced `i` to index an array. If the value is configuration
rather than data, declare the parameter `Static[...]` (or `int`), so it stays
a Python value.

<a id="ex202"></a>

### EX202 · A static parameter or a size is traced

A parameter that must be known when the function is traced (a `Static[...]`
parameter, an `int` that sets a size) receives a traced value, usually because
the function is called under `jax.jit` without marking it static.

```python
@expex.fn
def tile[N](x: f32[N], times: int) -> f32[N * times]:
    return jnp.tile(x, times)

try:
    jax.jit(tile)(jnp.ones(3), 2)
except expex.StageError as e:
    print(e.code, e)
```

```
EX202 tile(): `times` sets a size, so it must be known when the function is traced, but it is traced. Under jax.jit, mark it static: jax.jit(f, static_argnames="times").
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex202
```

**Fix.** Mark the parameter static: `jax.jit(tile, static_argnames="times")`, or use
`expex.jit(tile)` / `@expex.fn(jit=True)`, which read the static parameters
from the signature.

<a id="ex203"></a>

### EX203 · A condition is an array, not a single True/False

The condition of an `if` or `while` is an array with several elements, not a
single True/False.

```python
@expex.fn
def relu[N](x: f32[N]) -> f32[N]:
    if x > 0:
        return x
    return 0 * x

try:
    relu(jnp.ones(3))
except expex.StageError as e:
    print(e.code, e)
```

```
EX203 the condition of the `if` at example.py:3 is an array of shape (3,), not a single True/False.
    - to choose per element:      write the function for one element (x: f32): a call
                                  on an array maps over it. Or jnp.where(cond, a, b)
    - to test all/any elements:   cond.all()  /  cond.any()
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex203
```

**Fix.** Write the function for one element (`x: f32`): called on an array, it is mapped
over it, and the `if` becomes a select per element. Or use
`jnp.where(x > 0, x, 0)`, or reduce the condition (`(x > 0).all()`).

<a id="ex204"></a>

### EX204 · The branches of a traced `if` disagree

A traced `if` runs both branches and selects the results, so every variable it
sets must have the same type, shape and dtype in both branches.

```python
@expex.fn(jit=True)
def head[N](x: f32[N]) -> f32:
    if x.sum() > 0:
        y = x
    else:
        y = x[:1]
    return y.sum()

try:
    head(jnp.ones(3))
except expex.StageError as e:
    print(e.code, e)
```

```
EX204 `y` has shape f32[3] in one branch of the traced `if` at example.py:3 and f32[1] in the other.
    Both branches must give it the same shape and dtype.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex204
```

**Fix.** Give the variable the same shape in both branches (pad or mask instead of
slicing a data-dependent amount), or compute the differing part after the `if`.

<a id="ex205"></a>

### EX205 · A list grows inside traced control flow

`list.append` runs under a traced condition (an `if` or `while` on traced
values, or a `for` whose number of iterations is traced), so the length of the
list would depend on the data.

```python
@expex.fn(jit=True)
def positives(x: f32) -> f32:
    ys = []
    if x > 0:
        ys.append(x)
    return jnp.asarray(ys).sum()

try:
    positives(1.0)
except expex.StageError as e:
    print(e.code, e)
```

```
EX205 `ys.append(...)` runs inside the traced `if` at example.py:4, so the list's length would depend on data.
    Append unconditionally and select the value instead: ys.append(jnp.where(cond, a, b)).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex205
```

**Fix.** Append unconditionally and select the value (`ys.append(jnp.where(x > 0, x, 0))`),
or keep a mask next to the values. For a traced number of iterations,
preallocate an array and fill it with `ys.at[i].set(...)`.

<a id="ex206"></a>

### EX206 · A variable first assigned in a traced loop is used after it

A variable that is not an array (a string, a Python object) is first assigned
inside a traced loop (a `while` on traced values, or a `for` whose number of
iterations is traced) and used after it: if the loop runs zero times it has no
value. (A number or an array first assigned in the loop starts as zeros.)

```python
@expex.fn(jit=True)
def grow(x: f32) -> f32:
    while x < 10:
        x = x * 2
        how = "doubled"
    print(how)
    return x

try:
    grow(1.0)
except expex.StageError as e:
    print(e.code, e)
```

```
EX206 `how` is first assigned inside the traced `while` at example.py:3 and used after it. Give it a value before the loop.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex206
```

**Fix.** Give the variable a value before the loop (`how = "doubled"`); if the loop
changes it, make it an array (see [EX207](#ex207)).

<a id="ex207"></a>

### EX207 · A non-array value changes inside a traced loop

A value that is not an array (a Python number, a string, a list) changes inside
a traced loop. Only arrays can change from one iteration of a compiled loop to
the next.

```python
@expex.fn(jit=True)
def halvings(x: f32) -> f32:
    steps = "none"
    while x > 1:
        x = x / 2
        steps = "some"
    return x

try:
    halvings(10.0)
except expex.StageError as e:
    print(e.code, e)
```

```
EX207 `steps` (str 'none') changes inside the traced `while` at example.py:4; only arrays can change in a traced loop.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex207
```

**Fix.** Make the value an array (`steps = jnp.int32(0)`, then `steps = steps + 1`), or
compute it outside the loop.

<a id="ex208"></a>

### EX208 · `range()` with a traced step

`range(start, stop, step)` is given a traced step in an `@expex.fn` body.

```python
@expex.fn(jit=True)
def every[N](x: f32[N], step: i32) -> f32:
    total = 0.0
    for i in range(0, N, step):
        total = total + x[i]
    return total

try:
    every(jnp.ones(6), jnp.int32(2))
except expex.StageError as e:
    print(e.code, e)
```

```
EX208 range() with a traced step is not supported; make the step a Python int.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex208
```

**Fix.** Make the step a Python int: a `Static[int]` (or `int`) parameter. A traced start
or stop is fine.

<a id="ex209"></a>

### EX209 · A Python object is updated inside traced control flow

A Python object (a dict, a list, an object's attribute) is updated in place
inside traced control flow. The update would happen once, while tracing, not
once per iteration or branch.

```python
@expex.fn(jit=True)
def count(x: f32) -> f32:
    seen = {"positive": 0}
    if x > 0:
        seen["positive"] += 1
    return x

try:
    count(1.0)
except expex.StageError as e:
    print(e.code, e)
```

```
EX209 item assignment on a dict inside traced control flow: the update would happen once while tracing, not once per iteration or branch. Use a JAX array (updated functionally) or move it out of the traced code
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex209
```

**Fix.** Keep the state in a JAX array and update it functionally
(`n = jnp.where(x > 0, n + 1, n)`), or move the update out of the traced code.

<a id="ex210"></a>

### EX210 · Collected values of unknown number while proving

The prover runs the body on symbolic sizes. There, a loop that collects values
(`ys.append(...)`) gives a list whose length is a symbol, and `jnp.stack(ys)`
needs to know it. The prover reports the function as failing with this code.

```python
@expex.fn
def running[T](x: f32[T]) -> f32[T]:
    s, ys = 0.0, []
    for t in range(T):
        s = s + x[t]
        ys.append(s)
    return jnp.stack(ys)

report = expex.prove(running)
print(report.code, report)
```

```
EX210 example.running: spot-checked
    holds at sizes T=2; T=3; T=1; T=11, but could not be proved for all sizes:
        TypeError: the values collected by this loop number T, a size not known while proving: use jnp.asarray(ys) (or expex.stack(ys) for pytrees), which takes the stacked array as it is, rather than jnp.stack(ys).
          see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex210
```

**Fix.** Use `jnp.asarray(ys)` (or `expex.stack(ys)` for pytrees), which takes the
stacked array as it is.

<a id="ex211"></a>

### EX211 · `expex.rows` / `expex.stack` with nothing to stack

`expex.stack` is given an empty list, or `expex.rows` a pytree with no arrays or
with arrays whose leading axes differ.

```python
try:
    expex.stack([])
except ValueError as e:
    print(e.code, e)
```

```
EX211 stack() of an empty list
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex211
```

**Fix.** Pass at least one element to `expex.stack`; give `expex.rows` arrays that share
their leading axis.

<a id="ex221"></a>

### EX221 · A key is needed while tracing, with no key in scope

Code that draws random numbers with `expex.key()` is traced (by `jax.jit`,
`jax.vmap`, `jax.grad`, `lax.scan`) with no key in scope. The global key lives
in Python: inside a compiled function it would be a constant, and every call
would draw the same numbers.

```python
@jax.jit
def noisy(x):
    return x + jr.normal(expex.key(), x.shape)

try:
    noisy(jnp.ones(3))
except expex.StageError as e:
    print(e.code, e)
```

```
EX221 expex.key() needs a key, but it is being traced (jit/vmap/grad/scan) with no key in scope.
    The global key lives in Python, so inside a compiled function it would be frozen into a constant and every call would draw the same numbers.
    - pass the key to a @expex.random function:   f(x, key=key)
    - or open a key scope inside the jitted function:   with expex.rng(key): ...
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex221
```

**Fix.** Make the function `@expex.fn(random=True)` (or `@expex.random`) and call it with
`key=`, or open a scope inside the traced function: `with expex.rng(key): ...`.
`expex.jit` handles random functions.

<a id="ex222"></a>

### EX222 · The key in scope was set outside jit/vmap/grad

A key scope (`with expex.rng(key):`) is open outside a `jax.jit` / `jax.vmap` /
`jax.grad`, and code inside the transformation draws from it: the key would be
a constant of the compiled function.

```python
@jax.jit
def noisy(x):
    return x + jr.normal(expex.key(), x.shape)

try:
    with expex.rng(jr.key(0)):
        noisy(jnp.ones(3))
except expex.StageError as e:
    print(e.code, e)
```

```
EX222 expex.key() needs a key, but the key in scope was set outside this jit/vmap/grad, so inside it would be a constant: a compiled function would draw the same numbers on every call.
    - pass the key to a @expex.random function:   f(x, key=key)
    - or open the scope inside the transformed function:   with expex.rng(key): ...
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex222
```

**Fix.** Pass the key into the transformed function (`f(x, key=key)` for an
`@expex.random` function), or open the scope inside it.

<a id="ex223"></a>

### EX223 · A random function under `jax.vmap` shares one key

A random `@expex.fn` is mapped with `jax.vmap` but its key is not: every element
of the batch would draw the same numbers.

```python
@expex.fn(random=True)
def drop[N](x: f32[N]) -> f32[N]:
    return x * jr.bernoulli(expex.key(), 0.5, x.shape)

try:
    jax.vmap(drop)(jnp.ones((4, 3)))
except expex.StageError as e:
    print(e.code, e)
```

```
EX223 drop() is being mapped with jax.vmap, but its random key is shared by the whole batch: every element would draw the same numbers.
    - use drop.vectorize(...), which gives each element its own key
    - or split a key yourself and pass the keys: drop(..., key=keys)
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex223
```

**Fix.** Call `drop.vectorize(xs)` (or just `drop(xs)`: extra leading axes are
mapped over), which gives each element its own key; or split a key yourself and
map over the keys too: `jax.vmap(drop)(xs, key=jr.split(key, 4))`.

## 3xx · Modules

<a id="ex301"></a>

### EX301 · A size is missing when building a module

A module is built from a key (`Linear(key, ...)`, `expex.build`) but a size its
annotations use is not given and cannot be found from anything else.

```python
@expex.struct
class Linear[I, O]:
    w: f32[I, O] = jax.nn.initializers.normal()
    b: f32[O] = 0.0

try:
    Linear(jr.key(0), I=3)
except TypeError as e:
    print(e.code, e)
```

```
EX301 Linear(key, ...) needs O: the size of `w` (f32[I, O]). Pass it by name, e.g. Linear(key, O=64, ...).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex301
```

**Fix.** Pass the size by name: `Linear(jr.key(0), I=3, O=4)`.

<a id="ex302"></a>

### EX302 · A field is missing

A field has no value: it has no default and was not passed, a list of modules
was given no length, or a hand-written `__init__` did not set it.

```python
@expex.struct
class Layer[D]:
    w: f32[D, D] = jax.nn.initializers.normal()

@expex.struct
class Tower[D]:
    layers: list[Layer[D]]

try:
    Tower(jr.key(0), D=4)
except TypeError as e:
    print(e.code, e)
```

```
EX302 Tower(key, ...) needs layers: the length of the list `layers` of Layer[D]. Pass it by name, e.g. Tower(key, layers=4, ...).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex302
```

**Fix.** Pass the field by name; for a list of modules, its length: `Tower(jr.key(0), D=4, layers=3)`.

<a id="ex303"></a>

### EX303 · Unknown or misplaced constructor arguments

A constructor got arguments it does not take: a keyword that is neither a size
nor a field, a field given twice, sizes passed without a key, or a key that is
not a PRNG key.

```python
@expex.struct
class Bias[D]:
    b: f32[D] = 0.0

try:
    expex.build(Bias, jr.key(0), D=3, scale=2.0)
except TypeError as e:
    print(e.code, e)
```

```
EX303 expex.build(Bias, key, ...) got scale=...: not a size (D) or a field (b) of Bias or of the modules in it.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex303
```

**Fix.** The message lists the sizes and fields the class takes. Sizes and the fields of
nested modules are taken only when building from a key: `Bias(jr.key(0), D=3)`;
given values, pass the fields: `Bias(jnp.zeros(3))`. (A keyword the generated
constructor has no parameter for at all, `Bias(jr.key(0), D=3, scale=2.0)`, is
Python's own `TypeError`, without a code.)

<a id="ex304"></a>

### EX304 · A field cannot be built from its annotation

Building from a key needs every array field to have an initializer, and this one
has none.

```python
@expex.struct
class Embed[V, D]:
    table: f32[V, D]
    scale: f32[D] = 1.0

try:
    Embed(jr.key(0), V=10, D=4)
except TypeError as e:
    print(e.code, e)
```

```
EX304 Embed(key, ...) cannot build `table`: f32[V, D] has no initializer. Give the field a default (a JAX initializer such as jax.nn.initializers.normal(), or a number), or pass it by name, e.g. Embed(key, table=..., ...).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex304
```

**Fix.** Give the field a default (a JAX initializer such as
`jax.nn.initializers.normal()`, or a number), or pass its value by name:
`Embed(jr.key(0), V=10, D=4, table=...)`.

<a id="ex305"></a>

### EX305 · The module does not match its annotations

A module is built from a key with a field given that does not match its
annotation, or a hand-written `__init__` sets arrays that do not match. (Built
from values, `Affine(w, b)`, a module is checked when it is passed to an
`@expex.fn`.)

```python
@expex.struct
class Affine[D]:
    w: f32[D, D] = jax.nn.initializers.normal()
    b: f32[D] = 0.0

try:
    Affine(jr.key(0), D=3, b=jnp.ones(4))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX305 Affine(key, ...): `b` is f32[4], but is declared f32[D].

    D is 3 (from D=3 passed to Affine(key, ...)), but axis 0 of `b` is 4.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex305
```

**Fix.** Pass arrays of the declared shapes: here `b` must be `f32[3]`, since `D = 3`.

<a id="ex306"></a>

### EX306 · A struct is declared or used wrongly

`@expex.struct` is used in a way it does not support: an option it does not
take, a method under `@jax.jit` but not checked, or a class that is not a
struct given to `expex.build`.

```python
try:
    expex.build(dict, jr.key(0))
except TypeError as e:
    print(e.code, e)
```

```
EX306 expex.build: 'dict' is not an @expex.struct class.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex306
```

**Fix.** Decorate the class with `@expex.struct`; put `@expex.fn` under `@jax.jit` on a
method; write invariants as an `assert` at the top of a method.

<a id="ex307"></a>

### EX307 · `model.at` names a part that does not exist

`model.at[...]` names a field, key or index the model does not have, or indexes
something that cannot be indexed.

```python
@expex.struct
class Scale[D]:
    w: f32[D] = 1.0

model = Scale(jr.key(0), D=3)
try:
    model.at.v.set(jnp.zeros(3))
except AttributeError as e:
    print(e.code, e)
```

```
EX307 Scale.at.v.set(...): the model (Scale) has no field `v`; its fields are w.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex307
```

**Fix.** Use a path that exists: the message lists the fields. `model.at.w.set(...)`.

<a id="ex308"></a>

### EX308 · `model.at` gets a value that does not fit

The new value given to `model.at[...]` does not fit the annotation of the part
it replaces, or the rest of the model.

```python
@expex.struct
class Gain[D]:
    w: f32[D] = 1.0
    b: f32[D] = 0.0

model = Gain(jr.key(0), D=3)
try:
    model.at.w.set(jnp.zeros(5))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX308 Gain.at.w.set(...): `w` is f32[5], but is declared f32[D].

    D is 3 (from `b: f32[3]`), but axis 0 of `w` is 5.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex308
```

**Fix.** Give a value of the declared shape and dtype: here `b` fixes `D = 3`. To change
a size of the model, build a new one.

## 4xx · Sizes and annotations

<a id="ex401"></a>

### EX401 · A malformed annotation

An annotation cannot be read: a dimension that is not a size (a string, a float,
`R:` without a size), two variadics in one shape, the wrong number of sizes for
a module.

```python
try:
    @expex.fn
    def first(x: f32["T"]) -> f32:
        return x[0]
except TypeError as e:
    print(e.code, e)
```

```
EX401 first: bad annotation for `x`: 'T': sizes are type parameters, not strings. Declare them in the signature (def f[T](...)) and compute with them directly: f32[T + 1] (annotations are evaluated lazily, so this works as written).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex401
```

**Fix.** Sizes are type parameters: `def first[T](x: f32[T])`. Compute with them directly
(`f32[T + 1]`); a fixed size is `Literal[3]`, `3`, or a named one `R:3`.

<a id="ex402"></a>

### EX402 · An annotation uses an undefined name

An annotation uses a name that is not defined: not a type parameter, not an
`int` parameter, not a field of `self`. Usually a misspelt size, or a size
missing from the type parameters.

```python
try:
    @expex.fn
    def last[T](x: f32[T]) -> f32[S]:
        return x[-1:]
except TypeError as e:
    print(e.code, e)
```

```
EX402 last: the annotation of the result (f32[S]) uses S, which is not defined. Sizes are type parameters (def f[T, D](...)), int parameters, or fields of self.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex402
```

**Fix.** Declare the size (`def last[T, S](...)`) or correct its name.

<a id="ex403"></a>

### EX403 · A size expression cannot be parsed

A size computed in the body's leading assignments (`H = D // heads`) or a
constraint uses something sizes do not support: `/` (sizes are integers), a
float, a call.

```python
try:
    @expex.fn
    def heads_first[T, D](x: f32[T, D], heads: int) -> f32[H, T, K]:
        H, K = heads, D / heads
        return x.reshape(T, H, K).transpose(1, 0, 2)
except ValueError as e:
    print(e.code, e)
```

```
EX403 use `//` instead of `/` in 'D / heads'
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex403
```

**Fix.** Sizes are integers: use `//` for division (`K = D // heads`, with
`assert D % heads == 0` to say it is exact), `%` for remainders, and integer
literals.

<a id="ex404"></a>

### EX404 · A size declared with `Name:size` conflicts

A size declared in an annotation with `Name:size` (`f32[R:6]`) conflicts with
another declaration: two different values, the same name as a type parameter or
a parameter, or no size after the colon.

```python
try:
    @expex.fn
    def board[R](x: f32[R:6, 7]) -> f32[R:6]:
        return x[:, 0]
except TypeError as e:
    print(e.code, e)
```

```
EX404 board: R:6 in `x: f32[R:6, 7]` declares R as a size fixed at 6, but R is also a type parameter, a size bound by the arguments. Use one or the other: drop R from the type parameters, or write R without `:6`.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex404
```

**Fix.** Use one declaration per size: drop `R` from the type parameters to fix it at 6,
or write `R` without `:6` to bind it from the arguments.

<a id="ex405"></a>

### EX405 · A size used in a helper defined outside the function

A function defined outside an `@expex.fn` uses one of its sizes by name. Sizes
are known inside the body of the function that declares them (and the functions
nested in it), not in helpers defined elsewhere. Python raises its `NameError`;
expex adds a note naming the helper, and the code.

```python
def flatten(x):
    return x.reshape(T * D)

@expex.fn
def flat[T, D](x: f32[T, D]) -> f32[T * D]:
    return flatten(x)

try:
    flat(jnp.ones((2, 3)))
except NameError as e:
    print(e.code, e.__notes__[0])
```

```
EX405 expex: T is a size of flat(), but `flatten` (example.py:2) is defined outside flat(), so it does not see flat()'s sizes.
    - define `flatten` inside flat(): a nested function sees the sizes of the call
    - or decorate it with @expex.fn and annotate its arguments with T
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex405
```

**Fix.** Define the helper inside the function, or make it an `@expex.fn` that declares
the size in its own annotations.

<a id="ex406"></a>

### EX406 · A size used by name in a body that is not converted

The body of an `@expex.fn` uses a size by name, but the body is not converted
(`convert=False`, or its source is not available), so the names are not bound.

```python
@expex.fn(convert=False)
def board(x: f32[R:6, 7]) -> f32[R]:
    return jnp.zeros(R)

try:
    board(jnp.ones((6, 7)))
except NameError as e:
    print(e.code, e.__notes__[0])
```

```
EX406 expex: R is a size of board(), declared in its annotations, but the body cannot use it by name because convert=False.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex406
```

**Fix.** Let the body be converted (drop `convert=False`), declare the size as a type
parameter (`def board[R](...)`: type parameters are always in scope), or read it
from a shape: `x.shape[0]`.

<a id="ex407"></a>

### EX407 · A size is not known in this call

A size is used by name where it has no value: outside an `@expex.fn` call, or
inside one whose arguments do not fix it.

```python
@expex.fn
def ones[T, N](x: f32[T]) -> f32[T]:
    return x + jnp.ones(N).sum()

try:
    ones(jnp.ones(3))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX407 ones(): the size N is not known in this call: only sizes that appear in the arguments, or are computed from them in an annotation, are known inside the body.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex407
```

**Fix.** Only the sizes that appear in the arguments, or are computed from them in an
annotation, are known in the body. Pass the size as an `int` parameter, or
derive it in an annotation.

<a id="ex408"></a>

### EX408 · An axis named by its size cannot be found

An axis is named by its size (`x.sum(T)`, `axis=T`), but the array has no axis
of that size, or several, so the name does not say which.

```python
@expex.fn
def row_sums[T](x: f32[T, T]) -> f32[T]:
    return x.sum(T)

try:
    row_sums(jnp.ones((3, 3)))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX408 T names more than one axis of `[T, T]`, so it cannot pick one. Give the axes different names, e.g. f32[T, S] with S for the second.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex408
```

**Fix.** Give the axes different names in the annotation (`f32[T, S]`), or annotate the
value so each axis has its own name; or use a position (`axis=1`).

<a id="ex409"></a>

### EX409 · `expex.einsum` / `expex.rearrange` misused

`expex.einsum` or `expex.rearrange` is used where it cannot work: outside an
`@expex.fn`, with a name twice in `to=`, with operands whose sizes disagree, or
with a rearrangement that splits and merges at once.

```python
@expex.fn
def gram[T, D](x: f32[T, D]) -> f32[T, T]:
    return expex.einsum(x, x, to=(T, T))

try:
    gram(jnp.ones((3, 2)))
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX409 example.py:3: T appears more than once in `to`. Two axes need two names: annotate a copy of the operand with another one, e.g. `h2: f32[T, E] = h` for `to=(D, E)`.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex409
```

**Fix.** Two axes need two names: annotate a copy of the operand with another name
(`y: f32[S, D] = x`) and write `to=(T, S)`.

<a id="ex410"></a>

### EX410 · An annotation used as a value

An annotation (`f32`, `Static[...]`, `Key`, `Batch[...]`) or a size is used as
a value: called to convert or construct.

```python
try:
    f32(jnp.ones(3))
except TypeError as e:
    print(e.code, e)
```

```
EX410 f32 is an annotation; use jnp.float32 to convert
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex410
```

**Fix.** Annotations describe values; to convert, use JAX: `jnp.asarray(x, jnp.float32)`
or `x.astype(jnp.float32)`; for a key, `jax.random.key(seed)`.

<a id="ex411"></a>

### EX411 · A jaxtyping annotation expex cannot read

expex reads [jaxtyping](porting.md#coming-from-jaxtyping) annotations as they
are, but a few of its forms have no expex equivalent: `?n` (a size that depends
on the position in a `PyTree`), `#` on an expression (`#t+1`: a computed size
that may also broadcast), and f-strings other than `{n}` (an int parameter) and
`{self.n}` (an int field).

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jaxtyping import Array, Float

try:
    @expex.fn
    def total(x: Float[Array, "?n"]) -> Float[Array, ""]:
        return x.sum()
except TypeError as e:
    print(e.code, e)
```

```
EX411 total: bad annotation for `x`: Float[Array, '?n']: `?n` depends on the position in a PyTree, and expex has no PyTree[...] annotation. Give the axis a plain name, or `_` to leave it unchecked.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex411
```

**Fix.** Give the axis a plain name (`"n"`), shared by every array that has it,
or `_` to leave it unchecked. For `#t+1`, drop the `#` (the size is then exact),
or name the axis. For an f-string, compute the size in the annotation:
`"{n}*d"` is `n * d`; for anything else, give the size a name.

## 5xx · Checkpoints

<a id="ex501"></a>

### EX501 · A field cannot be saved or loaded

`expex.save` or `expex.load` meets a field it cannot write or read (its
annotation is not an array, a struct, or a list, tuple or dict of them), a
field holding arrays that no annotation describes (a layer from another
library, such as `eqx.nn.Linear`: saving would leave it out), or a class with
no array fields at all.

```python
@expex.struct
class Settings:
    name: str = "base"

try:
    expex.save(Settings(), "settings.npz")
except TypeError as e:
    print(e.code, e)
```

```
EX501 Settings has no array fields: nothing to save or load.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex501
```

**Fix.** Save the module that holds the arrays; keep plain configuration in static
fields of it (numbers and strings are saved with it). For a layer whose arrays
are not annotated, annotate them, or save the model with
[Orbax](interop.md#orbax).

<a id="ex502"></a>

### EX502 · The checkpoint's keys do not match the class

The arrays in the file do not match the fields of the class being loaded: some
are missing, or there are extra ones (with `strict=True`, the default).

```python
@expex.struct
class Old[D]:
    w: f32[D] = 0.0

@expex.struct
class New[D]:
    w: f32[D] = 0.0
    b: f32[D] = 0.0

expex.save(Old(jr.key(0), D=3), "old.npz")
try:
    expex.load(New, "old.npz")
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX502 expex.load(New, 'old.npz'): the keys do not match.

    missing from the file: b
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex502
```

**Fix.** Load into the class the file was saved from, or convert the checkpoint. Extra
keys can be ignored with `strict=False`.

<a id="ex503"></a>

### EX503 · A loaded array does not match its annotation

An array read from the file does not match the annotation of its field, or the
sizes given to `expex.load`.

```python
@expex.struct
class Vec[D]:
    w: f32[D] = 0.0

expex.save(Vec(jr.key(0), D=3), "vec.npz")
try:
    expex.load(Vec, "vec.npz", D=4)
except expex.ShapeError as e:
    print(e.code, e)
```

```
EX503 expex.load(Vec, 'vec.npz'):

    w: declared f32[D], file has f32[3]

    D is 4 (from D=4 passed to expex.load), but axis 0 of `w` is 3.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex503
```

**Fix.** Pass the sizes the checkpoint was saved with, or load into the class it was
saved from.

<a id="ex504"></a>

### EX504 · An unknown size given to `expex.load`

`expex.load(cls, path, **sizes)` is given a size the class does not have.

```python
@expex.struct
class Row[D]:
    w: f32[D] = 0.0

expex.save(Row(jr.key(0), D=3), "row.npz")
try:
    expex.load(Row, "row.npz", N=3)
except TypeError as e:
    print(e.code, e)
```

```
EX504 expex.load(Row, 'row.npz'): Row has no size `N` (its sizes: D).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex504
```

**Fix.** Use the size names of the class (the message lists them).

<a id="ex505"></a>

### EX505 · A static field is missing from the checkpoint

A static field of the class is not in the file (only numbers, strings, None and
tuples or lists of those are saved) and has no default.

```python
@expex.struct
class Saved[D]:
    w: f32[D] = 0.0

@expex.struct
class Named[D]:
    label: str
    w: f32[D] = 0.0

expex.save(Saved(jr.key(0), D=3), "saved.npz")
try:
    expex.load(Named, "saved.npz")
except TypeError as e:
    print(e.code, e)
```

```
EX505 expex.load: the static field `label` (Named.label) is not in the file (only numbers, strings, None and tuples/lists of those are saved) and has no default. Load into a template instead: expex.load(model, path).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex505
```

**Fix.** Load into a template, which gives the static fields their values:
`expex.load(model, path)`; or give the field a default.

## 6xx · Sharding

<a id="ex601"></a>

### EX601 · The mesh has no such axis

A size is mapped to a mesh axis the mesh does not have.

```python
mesh = jax.make_mesh((1,), ("data",))

@expex.struct
class Proj[D]:
    w: f32[D, D] = jax.nn.initializers.normal()

try:
    expex.shardings(Proj(jr.key(0), D=4), mesh, D="model")
except ValueError as e:
    print(e.code, e)
```

```
EX601 D='model': the mesh has no axis 'model' (its axes are 'data').
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex601
```

**Fix.** Use the names of the mesh's axes (the message lists them), or build the mesh
with the axis.

<a id="ex602"></a>

### EX602 · A mesh axis does not divide a size

A size mapped to a mesh axis is not divisible by the number of devices on that
axis.

```python
mesh = jax.sharding.AbstractMesh((4,), ("model",))

@expex.struct
class Wide[D]:
    w: f32[D, D] = jax.nn.initializers.normal()

try:
    expex.shardings(Wide(jr.key(0), D=6), mesh, D="model")
except ValueError as e:
    print(e.code, e)
```

```
EX602 Wide.w: f32[D, D] has D = 6 on axis 0, which mesh axis 'model' cannot split evenly (4 devices do not divide 6).
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex602
```

**Fix.** Choose sizes that the mesh axis divides, or map the size to another axis (or to
none).

<a id="ex603"></a>

### EX603 · `expex.shardings` needs a value

`expex.shardings` is given a class rather than a value (a list of layers has no
length yet), or an annotation with a variadic shape and no value to give it a
rank.

```python
@expex.struct
class Blocks[D]:
    w: f32[D, D] = jax.nn.initializers.normal()

try:
    expex.shardings(Blocks, jax.make_mesh((1,), ("model",)), D="model")
except TypeError as e:
    print(e.code, e)
```

```
EX603 expex.shardings needs a value, not the class Blocks (lists of layers have no length yet). Pass jax.eval_shape(lambda: Blocks(...)) instead.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex603
```

**Fix.** Pass a value, or its shapes without allocating it:
`jax.eval_shape(lambda: Blocks(jr.key(0), D=64))`.

## 7xx · The prover and the tools

<a id="ex701"></a>

### EX701 · The prover cannot invent a value

The prover needs an example value for a parameter and cannot invent one: a
`Static` or plain parameter with no default, a method proved without an
instance. The report is `unchecked`.

```python
@expex.fn
def act[N](x: f32[N], kind: Static[str]) -> f32[N]:
    return jnp.tanh(x) if kind == "tanh" else x

report = expex.prove(act)
print(report.code, report)
```

```
EX701 example.act: unchecked
    Static parameter `kind` has no default, so the prover cannot invent a value for it. Add @fn(examples={"kind": [...]}).
    see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex701
```

**Fix.** Give the parameter values to try, `@expex.fn(examples={"kind": ["tanh", "id"]})`,
or a default; prove a method on an instance, `expex.prove(model.method)`.

<a id="ex702"></a>

### EX702 · Holds at the sizes tried, but not proved

The function holds at the concrete sizes the prover tried, but could not be
proved for all sizes: some step (often a `reshape`) depends on a property of
the sizes the prover cannot decide, such as Python control flow on a size. The
report is `spot-checked`. (When the reason has a code of its own, such as
[EX210](#ex210), the report has that code.)

```python
@expex.fn
def boost[T](x: f32[T]) -> f32[T]:
    if x.shape[0] > 4:  # Python control flow on a size
        return x * 2
    return x

report = expex.prove(boost)
print(report.code, report)
```

```
EX702 example.boost: spot-checked
    holds at sizes T=2; T=3; T=1; T=11, but could not be proved for all sizes:
        InconclusiveDimensionOperation: Symbolic dimension comparison 'T' > '4' is inconclusive.
        This error arises for comparison operations with shapes that
        are non-constant, and the result of the operation cannot be represented as
        a boolean value for all values of the symbolic dimensions involved.

        Please see https://docs.jax.dev/en/latest/501/shape-polymorphism.html#comparison-of-symbolic-dimensions-is-partially-supported
        ...
    see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex702
```

**Fix.** Let the prover decide the step: a leading `assert` on the sizes (`assert T > 4`)
is assumed by the prover and checked on every call; a choice made on data
(`jnp.where`) rather than in Python holds for every size. Or accept the spot
check: the function is right at every size tried.

<a id="ex703"></a>

### EX703 · No sizes satisfy the constraints

The prover could not find sizes satisfying the function's constraints (its
leading `assert`s), so it checked nothing. The report is `unchecked`.

```python
@expex.fn
def tiny_only[T](x: f32[T]) -> f32[T]:
    assert T > 10**12
    return x

report = expex.prove(tiny_only)
print(report.code, report)
```

```
EX703 example.tiny_only: unchecked
    could not find sizes satisfying the `where` constraints.
    see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex703
```

**Fix.** Check the constraints: they may contradict each other, or need sizes too large
to try. Give examples of the static parameters with `@expex.fn(examples=...)`.

<a id="ex704"></a>

### EX704 · The function raised while traced on abstract inputs

The function raised an error that is not a shape error while the prover traced
it on abstract inputs: usually JAX rejecting the shapes of an operation (a
matmul of mismatched sizes). The report is `error`, with JAX's message.

```python
@expex.fn
def pool[T, D](x: f32[T, D]) -> f32[D]:
    weights = jax.nn.softmax(x.sum(-1))
    return weights @ x[1:]

report = expex.prove(pool)
print(report.code, report)
```

```
EX704 example.pool: error
    the function raised while being traced on abstract inputs (D=2, T=3):
        TypeError: dot_general requires contracting dimensions to have the same shape, got (3,) and (2,).
        expex: in pool(x: f32[3, 2])  [T=3, D=2]
    see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex704
```

**Fix.** The message says at which sizes it failed and what JAX objected to: here
`weights` has `T` elements and `x[1:]` has `T - 1` rows.

<a id="ex705"></a>

### EX705 · A tool was given something it cannot inspect

A tool (`expex.prove`, `expex.cost`, `expex.summary`, `expex.debug.shapes`,
`expex.annotate`, `debug.watch`) is given something it cannot inspect: not an
`@expex.fn`, not a struct, or a function without source.

```python
try:
    expex.prove(42)
except TypeError as e:
    print(e.code, e)
```

```
EX705 cannot prove 42: expected an @fn function, a module or a name
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex705
```

**Fix.** Pass an `@expex.fn` function, a struct method (`Model.__call__`, or
`model.__call__` bound to an instance), a module, or the name of one.

<a id="ex706"></a>

### EX706 · An unknown size or parameter given to a tool

A tool is given a size or parameter by name that the function does not have.

```python
@expex.fn
def mix[D](x: f32[D], w: f32[D, D]) -> f32[D]:
    return w @ x

try:
    expex.cost(mix, N=4)
except TypeError as e:
    print(e.code, e)
```

```
EX706 mix: unknown size or parameter `N`
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex706
```

**Fix.** Use the names of the function's sizes and parameters (here `D`).

<a id="ex707"></a>

### EX707 · `expex.annotate` cannot use the names or the source

`expex.annotate` is given axis names it cannot use (not identifiers, more names
than axes, one name for two sizes), or asked to write a file it cannot.

```python
def mean_rows(x):
    return x.mean(0)

try:
    expex.annotate(mean_rows, jnp.ones((4, 3)), names={"x": ["T", "D", "E"]})
except ValueError as e:
    print(e.code, e)
```

```
EX707 names for `x`: 3 names for 2 axes (4, 3)
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex707
```

**Fix.** Give one valid name per axis, `names={"x": ["T", "D"]}` (`_` leaves an axis to
annotate); call `.write()` only on a function defined in a file.

<a id="ex708"></a>

### EX708 · A diagram cannot be saved or explained

A diagram is saved in a format or theme it does not support, or drawn to PNG
with no browser or matplotlib, or `diagram.explain` is given an error that has
no diagram.

```python
@expex.fn
def affine[D](x: f32[D], w: f32[D, D]) -> f32[D]:
    return w @ x

try:
    expex.diagram(affine).save("affine.pdf")
except ValueError as e:
    print(e.code, e)
```

```
EX708 cannot save a diagram as '.pdf': use .svg, .png or .html
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex708
```

**Fix.** Save as `.svg`, `.png` or `.html`, with `theme="light"` or `"dark"`.

## 8xx · Decorators and settings

<a id="ex801"></a>

### EX801 · An invalid decorator option

An `@expex.fn` option is invalid: one that moved (`where=`, `checked=`), an
option that needs another (`donate=` without `jit=True`), or an unknown `remat`
policy.

```python
try:
    @expex.fn(donate="x")
    def step[D](x: f32[D]) -> f32[D]:
        return x + 1
except TypeError as e:
    print(e.code, e)
```

```
EX801 @expex.fn(donate=...) only applies with jit=True
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex801
```

**Fix.** The message says what to write instead: here `@expex.fn(jit=True, donate="x")`.

<a id="ex802"></a>

### EX802 · `@expex.random` / `@expex.vectorize` above `@jax.jit`

`@expex.random` or `@expex.vectorize` is applied on top of `@jax.jit` (or
`@jax.checkpoint`): inside it, the key would be compiled in once.

```python
try:
    @expex.random
    @jax.jit
    @expex.fn
    def noise[N](x: f32[N]) -> f32[N]:
        return x + jr.normal(expex.key(), x.shape)
except TypeError as e:
    print(e.code, e)
```

```
EX802 @expex.random must go under @jax.jit / @jax.checkpoint (inside it would be compiled once and repeat): put it directly on the @expex.fn function, or use @expex.jit, which handles random functions.
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex802
```

**Fix.** Put it directly on the `@expex.fn` function, under `@jax.jit`, or use
`@expex.jit`, which handles random functions.

<a id="ex803"></a>

### EX803 · `expex.config` has no such setting

`expex.config(...)` is given a setting it does not have.

```python
try:
    expex.config(fast=True)
except AttributeError as e:
    print(e.code, e)
```

```
EX803 expex.config has no setting 'fast'
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex803
```

**Fix.** `print(expex.config)` lists the settings.

<a id="ex804"></a>

### EX804 · `eps` / `tiny` / `huge` on a non-float array

`expex.eps`, `expex.tiny` or `expex.huge` meets an array that is not
floating-point, so it has no such limit.

```python
try:
    jnp.arange(3) + expex.eps
except TypeError as e:
    print(e.code, e)
```

```
EX804 expex.eps needs a floating-point array, not one of dtype int32
  see https://github.com/davidesartor/expex/blob/main/docs/errors.md#ex804
```

**Fix.** Use them with floating-point arrays; convert integers first
(`x.astype(jnp.float32) + expex.eps`).

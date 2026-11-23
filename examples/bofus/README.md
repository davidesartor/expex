# bofus, rewritten with typax

A port of the library part of [bofus](https://github.com/davidesartor/bofus)
(branch `adapt_rho`, commit `af48011`): kernels, RBF-mixture RKHS geometry,
the Gaussian-process surrogate and the acquisition optimizer. It keeps the
public API, so the original project's own test-suite runs against it.

```
PYTHONPATH=examples/bofus pytest <bofus checkout>/tests     # the original's tests, on the port
python examples/bofus/parity.py reference <bofus checkout>/src ref.npz
python examples/bofus/parity.py check ref.npz               # same inputs, both implementations
python -m typax examples/bofus/bofus/kernels.py             # shape proofs
```

## What changed in the code

The two core objects are modules (`@tx.struct`) with methods:

```python
@tx.struct
class RBFMixture[M, D]:          # one mixture; a set of them is Batch[RBFMixture[M, D], N, O]
    l: Float[M, D]
    x: Float[M, D]
    a: Float[M]

    @tx.lift
    def __call__(self, x: Float[D], periodic: Periodic = None) -> Float: ...
    @tx.lift
    def inner[K](self, other: "RBFMixture[K, D]", l0=None, periodic=None) -> Float: ...
    @tx.lift
    def distance[K](self, other: "RBFMixture[K, D]", l0=None, periodic=None) -> Float: ...
    @tx.lift
    def pad_to(self, m: int) -> RBFMixture[m, D]: ...    # assert m >= M in the body

@tx.struct
class GaussianProcess[N, O, M, D]:
    ...
    f: Batch[RBFMixture[M, D], N, O]
    def __init__(self, key, f, y, *, l0=None, ...): ...     # fits the GP
    @fn
    def predict_joint[Q](self, f: Batch[RBFMixture[M, D], Q, O]) -> tuple[Float[Q], Float[Q, Q]]: ...
```

Methods are written for one mixture (or one pair) and map over any leading
axes of `self` and the arguments, so `f1.distance(f2)` on sets of shapes
`(n, 1, o)` and `(1, k, o)` gives an `(n, k, o)` distance table. The original
API stays as aliases (`rkhs.inner(f1, f2)`, `rkhs.distance(...)`,
`GaussianProcess.fit(...)`, `f._replace(...)`) so its tests run unchanged.

| Original | Port |
|---|---|
| `Float[Array, "*b m d"]` strings (jaxtyping), not checked at run time | `Float[M, D]` on a module, `Batch[RBFMixture[M, D], N, O]` for sets, checked on every call |
| `RBFMixture["*b m d"]` through a `__class_getitem__` that only documents | `@tx.struct class RBFMixture[M, D]`, checked field by field |
| `GaussianProcess.fit(...)` building a NamedTuple | `GaussianProcess(key, f, y, ...)`: a module whose constructor fits it; the result is checked against its annotations |
| Free functions over mixtures with `[..., None, :]` indexing for any leading axes | Methods written for one mixture, lifted over the rest |
| Sanitised double `where` for every piecewise function (`rbf`, the log-EI branches, `safe_sqrt`, the periodic series) | A plain `if` on one element, then `lift`; the gradient is NaN-free without sanitising inputs |
| `eqx.filter_jit` to keep functions and tuples out of the trace | `@tx.jit`: the annotations say what is static (`n: int`, `shape: tuple[int, ...]`, `profile: Profile`); a struct's non-array fields are static too |
| `jax.lax.map(f, xs, batch_size=k)` and `jax.vmap(solve)(starts)` | `for x in tx.rows(xs, chunk=k): out.append(f(x))` |
| `eqx.error_if(x, cond, msg)` | `assert not cond, msg` |
| `logits.at[i].set(jnp.inf)` | `logits[i] = jnp.inf` |
| `pad_to(m)` returns "some mixture" | `-> RBFMixture[m, D]` with `assert m >= M`, checked |

The port does not depend on equinox or jaxtyping.

## Results

- 42 `jnp.where` calls in the original, 13 in the port (the ones left apply
  masks to data).
- The original's 74 tests (kernels, utils, rkhs, gp, acquisition) pass against
  the port without changes to the tests.
- `parity.py` compares 164 arrays of values and gradients on identical inputs
  and keys, in float64. Largest relative difference per module: utils 0
  (identical), rkhs 1e-15, gp 3e-15 (four L-BFGS fits included), kernels
  6e-13, acquisition 6e-7 (the L-BFGS acquisition optimizer, whose stopping
  tolerance is 1e-5). NaN and infinity patterns are identical.
- Speed, same machine, CPU, float64 (first call includes compilation; "then"
  is the median of repeated calls). Run-to-run noise is around 10-20%.

  | | original, first | original, then | port, first | port, then |
  |---|---|---|---|---|
  | `GaussianProcess.fit` (24 points, 4 restarts) | 2.1 s | 4.5 ms | 2.1 s | 5.4 ms |
  | `predict_joint` | 0.41 s | 0.53 ms | 0.44 s | 0.51 ms |
  | gradient of `predict_joint` | 1.4 s | 5.0 ms | 1.2 s | 5.1 ms |
  | `optimize_expected_improvement`, q = 1 | 6.2 s | 102 ms | 4.5 s | 83 ms |
  | `optimize_expected_improvement`, q = 2 | 7.5 s | 165 ms | 6.0 s | 173 ms |

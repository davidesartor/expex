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

| Original | Port |
|---|---|
| `Float[Array, "*b m d"]` strings (jaxtyping), not checked at run time | `Float[*B, M, D]` with real dimensions, checked on every call |
| `RBFMixture["*b m d"]` through a `__class_getitem__` that only documents | `class RBFMixture[*B, M, D](NamedTuple)`, checked field by field |
| Sanitised double `where` for every piecewise function (`rbf`, the log-EI branches, `safe_sqrt`, the periodic series) | A plain `if` on one element, then `lift`; the gradient is NaN-free without sanitising inputs |
| Functions written for arbitrary leading axes with `[..., None, :]` indexing | Functions written for one pair of mixtures, `@fn(lift=True)` broadcasts the rest |
| `eqx.filter_jit` to keep functions and tuples out of the trace | `@fn(jit=True)`; `Static[...]` fields and non-array values are static automatically |
| `jax.lax.map(f, xs, batch_size=k)` and `jax.vmap(solve)(starts)` | `for x in tx.rows(xs, chunk=k): out.append(f(x))` |
| `eqx.error_if(x, cond, msg)` | `assert not cond, msg` |
| `logits.at[i].set(jnp.inf)` | `logits[i] = jnp.inf` |
| `pad_to(m)` returns "some mixture" | `-> RBFMixture[*B, K, D]` with `K="m"`, checked |

The port does not depend on equinox or jaxtyping.

## Results

- 42 `jnp.where` calls in the original, 13 in the port (the ones left apply
  masks to data); 789 lines against 741.
- The original's 74 tests (kernels, utils, rkhs, gp, acquisition) pass against
  the port without changes to the tests.
- `parity.py` compares 164 arrays of values and gradients on identical inputs
  and keys, in float64. Largest relative difference per module: utils 0
  (identical), rkhs 1e-15, gp 3e-15 (four L-BFGS fits included), kernels
  6e-13, acquisition 4e-7 (the L-BFGS acquisition optimizer, whose stopping
  tolerance is 1e-5). NaN and infinity patterns are identical.
- The whole `parity.py` workload, which is mostly compilation, takes 61 s
  with the original and 74 s with the port on the same machine.

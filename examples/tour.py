"""A tour of typax. Run with:  python examples/tour.py"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import typax as tx
from typax import Static, debug, f32, fn, i32


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# 1. Shapes in the signature -------------------------------------------------


class Params[D, H](NamedTuple):
    w_in: f32[H, D]
    w_rec: f32[H, H]
    w_out: f32[D, H]


@fn
def cell[D, H](p: Params[D, H], h: f32[H], x: f32[D]) -> f32[H]:
    return jnp.tanh(p.w_in @ x + p.w_rec @ h)


@fn
def rnn[T, D, H](p: Params[D, H], xs: f32[T, D]) -> f32[T, D]:
    """Plain Python control flow: the loop becomes a lax.scan under jit."""
    h = jnp.zeros(p.w_rec.shape[0])
    ys = []
    for x in xs:
        h = cell(p, h, x)
        ys.append(p.w_out @ h)
    return jnp.asarray(ys)


@fn
def loss[T, D, H](p: Params[D, H], xs: f32[T, D], ys: f32[T, D]) -> f32:
    return jnp.mean((rnn(p, xs) - ys) ** 2)


def init(key, d: int, h: int) -> Params:
    k1, k2, k3 = jax.random.split(key, 3)
    s = 0.3
    return Params(
        s * jax.random.normal(k1, (h, d)),
        s * jax.random.normal(k2, (h, h)),
        s * jax.random.normal(k3, (d, h)),
    )


if __name__ == "__main__":
    p = init(jax.random.key(0), d=3, h=8)
    xs = jax.random.normal(jax.random.key(1), (20, 3))

    section("1. A call that breaks the contract")
    try:
        rnn(p, xs[:, :2])
    except tx.ShapeError as e:
        print(e)

    section("2. Proving the annotations for all sizes (no data needed)")
    for f in (cell, rnn, loss):
        print(tx.prove(f))

    @fn
    def off_by_one[T, D](xs: f32[T, D]) -> f32[T, D]:
        return xs[1:] - xs[:-1]

    print(tx.prove(off_by_one))

    section("3. Plain Python control flow, staged automatically")
    step = jax.jit(jax.value_and_grad(loss))
    value, grads = step(p, xs, xs)
    print(f"loss = {value:.4f}; the loop ran as: {rnn.flow}")
    print("traced program:", debug.summary(loss, p, xs, xs).ops, "operations")
    plain = fn(convert=False)(rnn.raw)
    print("without conversion:", debug.summary(plain, p, xs).ops, "operations (unrolled)")

    @fn
    def collatz(n: i32) -> i32:
        steps = 0
        while n != 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            steps += 1
        return steps

    print("collatz, vectorised over traced ints:", jax.vmap(collatz)(jnp.array([6, 7, 27])))

    section("4. Vectorising from the signature")
    batch = jax.random.normal(jax.random.key(2), (32, 20, 3))
    print("rnn.lift on a batch:", rnn.lift(p, batch).shape)
    run = rnn.batched(chunk=16)
    for n in (5, 16, 40):
        run(p, batch[:1].repeat(n, 0))
    print(f"rnn.batched(chunk=16) served batches of 5, 16 and 40 with {run.traces} compile")

    section("5. Implicit random keys")

    @fn(random=True)
    def dropout[D](x: f32[D], rate: Static[float] = 0.5) -> f32[D]:
        keep = jax.random.bernoulli(tx.key(), 1 - rate, x.shape)
        return jnp.where(keep, x / (1 - rate), 0.0)

    tx.seed(0)
    print(dropout(jnp.ones(6)), dropout(jnp.ones(6)), sep="\n")
    try:
        jax.jit(dropout)(jnp.ones(6))
    except tx.StageError as e:
        print("\nunder a plain jit:", str(e).splitlines()[0])

    section("6. Finding a NaN")

    @fn
    def entropy[N](logits: f32[N]) -> f32:
        p = jax.nn.softmax(logits)
        return -(p * jnp.log(p)).sum()

    print(debug.find_nan(entropy, jnp.array([0.0, 200.0, -200.0])))

    section("7. Where the time goes")
    print(debug.timing(loss, p, xs, xs))

    section("8. Retraces, explained")

    @fn(jit=True)
    def scale[N](x: f32[N], by: f32) -> f32[N]:
        return x * by

    scale(jnp.ones(3), 2.0)
    scale(jnp.ones(4), 2.0)
    scale(jnp.ones(4), jnp.float32(2.0))
    print(scale.explain_retrace(1))
    print(scale.explain_retrace(2))

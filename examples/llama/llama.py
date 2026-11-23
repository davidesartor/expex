"""Llama 2 (Touvron et al., 2023) written with expex.

A port of github.com/ayaka14732/llama-2-jax (pure JAX), checked against it in
tests/ports/test_llama.py. The model is the same; what changes is how it is
written:

- every module states its sizes (`Attention[D, G, R, K]`: width D, G groups
  of R query heads sharing a key/value head, head size K), so shapes are
  checked on every call and proved for all sizes: `python -m expex examples/llama/llama.py`
- heads are contracted by axis name (`expex.einsum(q, cache.k, to=(G, R, T, S))`),
  with no einsum strings and no reshape/transpose chains
- per-token layers (norms, MLP) are written for one token and `.vectorize`d
- one code path for training and generation: a forward pass is a pass over an
  empty KV cache; generation is a plain loop carrying the layers' caches, and
  the loop over layers is one `lax.scan`
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Int, f32, fn

normal = jax.nn.initializers.normal(0.02)  # Llama's initializer_range


@expex.struct
class RMSNorm[D]:
    weight: f32[D] = 1.0
    eps: float = 1e-6

    def __call__(self, x: f32[D]) -> f32[D]:
        return x / jnp.sqrt(jnp.mean(x * x) + self.eps) * self.weight


@fn
def rope[T, K](x: f32[T, K], at: Int[T]) -> f32[T, K]:
    """Rotary embedding: each pair (x[i], x[i + K/2]) rotated by angle `at / 10000^(2i/K)`."""
    assert K % 2 == 0
    i = jnp.arange(K) % (K // 2)
    angle = at[:, None] * (1.0 / 10000 ** (2 * i / K))
    rotated = jnp.where(jnp.arange(K) < K // 2, -1.0, 1.0) * jnp.roll(x, K // 2, axis=K)  # (-x2, x1)
    return x * jnp.cos(angle) + rotated * jnp.sin(angle)


@expex.struct
class Cache[G, S, K]:
    """Keys and values for S positions, per key/value head."""

    k: f32[G, S, K]
    v: f32[G, S, K]

    def write[T](self, k: f32[G, T, K], v: f32[G, T, K], start: Int) -> Cache[G, S, K]:
        """With positions start, ..., start + T - 1 filled in."""
        assert T <= S
        return Cache(
            jax.lax.dynamic_update_slice_in_dim(self.k, k, start, axis=1),
            jax.lax.dynamic_update_slice_in_dim(self.v, v, start, axis=1),
        )


@expex.struct
class Attention[D, G, R, K]:
    """Grouped-query attention: G groups of R query heads, each group sharing one key/value head."""

    wq: f32[D, G, R, K] = normal
    wk: f32[D, G, K] = normal
    wv: f32[D, G, K] = normal
    wo: f32[G, R, K, D] = normal

    def __call__[T, S](self, x: f32[T, D], at: Int[T], cache: Cache[G, S, K]) -> tuple[f32[T, D], Cache[G, S, K]]:
        """T tokens at positions `at`, each attending to the cached positions up to its own."""
        assert K % 2 == 0 and T <= S  # K: for the rotary embedding
        q: f32[G, R, T, K] = rope.vectorize(expex.einsum(x, self.wq, to=(G, R, T, K)), at)
        k = rope.vectorize(expex.einsum(x, self.wk, to=(G, T, K)), at)
        cache = cache.write(k, expex.einsum(x, self.wv, to=(G, T, K)), at[0])
        visible = jnp.arange(S) <= at[:, None]  # causal, and only positions written so far
        scores: f32[G, R, T, S] = jnp.where(visible, expex.einsum(q, cache.k, to=(G, R, T, S)) / jnp.sqrt(K), -jnp.inf)
        weights: f32[G, R, T, S] = jax.nn.softmax(scores, axis=S)
        out: f32[G, R, T, K] = expex.einsum(weights, cache.v, to=(G, R, T, K))
        return expex.einsum(out, self.wo, to=(T, D)), cache


@expex.struct
class MLP[D, F]:
    """SwiGLU."""

    gate: f32[D, F] = normal
    up: f32[D, F] = normal
    down: f32[F, D] = normal

    def __call__(self, x: f32[D]) -> f32[D]:
        return (jax.nn.silu(x @ self.gate) * (x @ self.up)) @ self.down


@expex.struct
class Block[D, F, G, R, K]:
    attn_norm: RMSNorm[D]
    attn: Attention[D, G, R, K]
    mlp_norm: RMSNorm[D]
    mlp: MLP[D, F]

    def __call__[T, S](self, x: f32[T, D], at: Int[T], cache: Cache[G, S, K]) -> tuple[f32[T, D], Cache[G, S, K]]:
        assert T <= S
        y, cache = self.attn(self.attn_norm.vectorize(x), at, cache)
        x = x + y
        return x + self.mlp.vectorize(self.mlp_norm.vectorize(x)), cache


@expex.struct
class Llama[V, D, F, G, R, K]:
    """A language model: vocabulary V, width D, MLP width F, G key/value heads each shared by R query heads."""

    embed: f32[V, D] = normal
    layers: list[Block[D, F, G, R, K]]
    norm: RMSNorm[D]
    unembed: f32[D, V] = normal

    def __call__[T](self, tokens: Int[T]) -> f32[T, V]:
        """Logits at every position: the causal forward pass, over an empty cache."""
        h, _ = self.extend(tokens, 0, self.empty(T))
        return h @ self.unembed

    def empty(self, S: int) -> list[Cache[G, S, K]]:
        return [Cache(jnp.zeros((G, S, K)), jnp.zeros((G, S, K))) for _ in self.layers]

    def extend[T, S](
        self, tokens: Int[T], start: Int, caches: list[Cache[G, S, K]]
    ) -> tuple[f32[T, D], list[Cache[G, S, K]]]:
        """Final hidden states of T tokens at positions start, ..., start + T - 1, with every layer's cache."""
        assert T <= S
        at = start + jnp.arange(T)
        x = self.embed[tokens]
        new = []
        for layer, cache in zip(self.layers, caches):  # identical blocks: one lax.scan
            x, cache = layer(x, at, cache)
            new.append(cache)
        return self.norm.vectorize(x), new

    def loss[T](self, tokens: Int[T]) -> f32:
        assert T >= 2
        logp = jax.nn.log_softmax(self(tokens[:-1]))
        return -jnp.take_along_axis(logp, tokens[1:, None], axis=1).mean()

    @expex.fn(jit=True, vectorize=True)
    def generate[T](self, prompt: Int[T], n: int) -> Int[T + n]:
        """Greedy decoding: the prompt in one pass, then one token at a time with the caches."""
        assert n >= 1
        h, caches = self.extend(prompt, 0, self.empty(T + n - 1))
        token = (h[-1] @ self.unembed).argmax()  # logits at the last position only
        out = []
        for i in range(n - 1):  # a lax.scan carrying the caches
            out.append(token)
            h, caches = self.extend(token[None], T + i, caches)
            token = (h[0] @ self.unembed).argmax()
        out.append(token)
        return jnp.concatenate([prompt, jnp.asarray(out)])


if __name__ == "__main__":
    model = Llama.init(jr.key(0), V=100, D=64, F=172, G=2, R=4, K=8, layers=3)  # 8 query heads, 2 kv heads
    tokens = jr.randint(jr.key(1), (16, 24), 0, 100)

    @expex.fn(jit=True, donate="auto")
    def step[B, T](model: Llama, batch: Int[B, T]) -> tuple[Llama, f32]:
        loss, grads = jax.value_and_grad(lambda m: m.loss.vectorize(batch).mean())(model)
        return jax.tree.map(lambda p, g: p - 0.5 * g, model, grads), loss

    for i in range(5):
        model, loss = step(model, tokens)
        print(f"step {i}: loss {float(loss):.4f}")
    print("generated", model.generate(tokens[0, :8], 8))
    print()
    print(expex.prove(__import__("__main__")))

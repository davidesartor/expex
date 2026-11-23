"""The Vision Transformer (Dosovitskiy et al., 2020) written with expex.

A port of `vit_jax/models_vit.py` from github.com/google-research/vision_transformer
(Flax), checked against it in tests/ports/test_vit.py. The model is the same;
what changes is how it is written:

- every module states its sizes (`Attention[D, H, K]`: width D, H heads of
  size K), so shapes are checked on every call and proved for all sizes:
  `python -m expex examples/vit/vit.py`
- an image is cut into patches by naming the axes it is split into
  (`expex.rearrange(image, to=(GH, P, Wpx, C))`), and heads are contracted by
  axis name (`expex.einsum(q, k, to=(H, T, S))`): no einops strings, no
  reshape/transpose chains
- the image size is in the model's signature (`f32[GH * P, GW * Q, C]`), with
  the position embeddings laid out on the patch grid
- per-token layers (norms, MLP) are written for one token, and the model for
  one image: called on a sequence or a batch, they map over it
- the loop over layers is a plain `for` (identical blocks: one `lax.scan`)

The ViT models of the paper: a class token, no ResNet stem, no
`representation_size` layer, and no dropout (the released checkpoints were
fine-tuned without it).
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Int, f32

# as the original: Glorot-uniform kernels, the heads' axes counted as one (Flax's DenseGeneral)
dense = jax.nn.initializers.xavier_uniform()
to_heads = jax.nn.initializers.xavier_uniform(in_axis=0, out_axis=(1, 2))  # [D, H, K]
from_heads = jax.nn.initializers.xavier_uniform(in_axis=(0, 1), out_axis=2)  # [H, K, D]
conv = jax.nn.initializers.lecun_normal(in_axis=(0, 1, 2), out_axis=3)  # [P, Q, C, D]


@expex.struct
class LayerNorm[D]:
    scale: f32[D] = 1.0
    bias: f32[D] = 0.0
    eps: float = 1e-6

    def __call__(self, x: f32[D]) -> f32[D]:
        return (x - x.mean()) * jax.lax.rsqrt(x.var() + self.eps) * self.scale + self.bias


@expex.struct
class PatchEmbed[P, Q, C, D]:
    """Each P x Q patch of C channels mapped to width D (the original's strided convolution)."""

    kernel: f32[P, Q, C, D] = conv
    bias: f32[D] = 0.0

    def __call__[Hpx, Wpx](self, image: f32[Hpx, Wpx, C]) -> f32[GH, GW, D]:
        GH, GW = Hpx // P, Wpx // Q  # the grid of patches
        assert Hpx % P == 0 and Wpx % Q == 0
        rows = expex.rearrange(image, to=(GH, P, Wpx, C))  # height split into rows of patches
        patches = expex.rearrange(rows, to=(GH, GW, P, Q, C))  # width split into columns
        return expex.einsum(patches, self.kernel, to=(GH, GW, D)) + self.bias


@expex.struct
class Attention[D, H, K]:
    """Self-attention with H heads of size K."""

    wq: f32[D, H, K] = to_heads
    bq: f32[H, K] = 0.0
    wk: f32[D, H, K] = to_heads
    bk: f32[H, K] = 0.0
    wv: f32[D, H, K] = to_heads
    bv: f32[H, K] = 0.0
    wo: f32[H, K, D] = from_heads
    bo: f32[D] = 0.0

    def __call__[T, S](self, x: f32[T, D]) -> f32[T, D]:
        keys: f32[S, D] = x  # the same tokens, attended to
        q: f32[T, H, K] = (expex.einsum(x, self.wq, to=(T, H, K)) + self.bq) / jnp.sqrt(K)
        k: f32[S, H, K] = expex.einsum(keys, self.wk, to=(S, H, K)) + self.bk
        v: f32[S, H, K] = expex.einsum(keys, self.wv, to=(S, H, K)) + self.bv
        weights: f32[H, T, S] = jax.nn.softmax(expex.einsum(q, k, to=(H, T, S)), axis=S)
        out = expex.einsum(weights, v, to=(T, H, K))
        return expex.einsum(out, self.wo, to=(T, D)) + self.bo


@expex.struct
class MLP[D, F]:
    w1: f32[D, F] = dense
    b1: f32[F] = jax.nn.initializers.normal(1e-6)
    w2: f32[F, D] = dense
    b2: f32[D] = jax.nn.initializers.normal(1e-6)

    def __call__(self, x: f32[D]) -> f32[D]:
        return jax.nn.gelu(x @ self.w1 + self.b1) @ self.w2 + self.b2


@expex.struct
class Block[D, H, K, F]:
    attn_norm: LayerNorm[D]
    attn: Attention[D, H, K]
    mlp_norm: LayerNorm[D]
    mlp: MLP[D, F]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


@expex.struct
class Encoder[D, H, K, F]:
    layers: list[Block[D, H, K, F]]
    norm: LayerNorm[D]

    def __call__[T](self, x: f32[T, D]) -> f32[T, D]:
        for layer in self.layers:  # identical blocks: one lax.scan
            x = layer(x)
        return self.norm(x)


@expex.struct
class ViT[GH, GW, P, Q, C, D, H, K, F, Classes]:
    """Images of GH x GW patches of P x Q pixels and C channels, width D, H heads of size K, MLP width F."""

    embed: PatchEmbed[P, Q, C, D]
    cls: f32[D] = 0.0
    cls_pos: f32[D] = jax.nn.initializers.normal(0.02)  # the class token's position embedding
    pos: f32[GH, GW, D] = jax.nn.initializers.normal(0.02)  # the patches', on their grid
    encoder: Encoder[D, H, K, F]
    head: f32[D, Classes] = 0.0
    head_bias: f32[Classes] = 0.0

    def __call__(self, image: f32[GH * P, GW * Q, C]) -> f32[Classes]:
        N = GH * GW
        grid: f32[GH, GW, D] = self.embed(image) + self.pos
        x = jnp.concatenate([(self.cls + self.cls_pos)[None], expex.rearrange(grid, to=(N, D))])  # row by row
        return self.encoder(x)[0] @ self.head + self.head_bias  # read out at the class token

    def loss(self, image: f32[GH * P, GW * Q, C], label: Int) -> f32:
        return -jax.nn.log_softmax(self(image))[label]


if __name__ == "__main__":
    # ViT-Ti (width 192, 3 heads, MLP width 768) with 4 of its 12 layers, on 32 x 32 images in 4 x 4 patches
    model = ViT(jr.key(0), GH=8, GW=8, P=4, Q=4, C=3, D=192, H=3, K=64, F=768, Classes=10, layers=4)
    images = jr.normal(jr.key(1), (16, 32, 32, 3))
    labels = jr.randint(jr.key(2), (16,), 0, 10)

    @jax.jit  # the model's call checks the images against its sizes
    def step(model: ViT, images: jax.Array, labels: jax.Array) -> tuple[ViT, jax.Array]:
        loss, grads = jax.value_and_grad(lambda m: m.loss(images, labels).mean())(model)
        return jax.tree.map(lambda p, g: p - 0.05 * g, model, grads), loss

    for i in range(5):
        model, loss = step(model, images, labels)
        print(f"step {i}: loss {float(loss):.4f}")
    print("predicted", model(images).argmax(-1))
    print()
    print(expex.prove(__import__("__main__")))

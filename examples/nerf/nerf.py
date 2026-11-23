"""NeRF (Mildenhall et al., 2020) written with expex.

A port of JaxNeRF (github.com/google-research/google-research, jaxnerf/, Flax),
checked against it in tests/ports/test_nerf.py. The model is the same; what
changes is how it is written:

- everything is written for one ray: S samples along it, F more drawn from its
  coarse weights, and `.vectorize`d over rays, so there is no batch axis to
  carry and no `[..., None, :]`: the compositing is `weights @ rgb`, the
  CDF a `jnp.cumsum`, and finding each draw's interval a `jnp.searchsorted`
- the MLP is written for one point and states its sizes (`MLP[P, V, W, C]`:
  encoded point and view direction, width, width of the colour branch), so
  shapes are checked on every call and proved for all sizes:
  `python -m expex examples/nerf/nerf.py`
- the skip connection is in the structure (`before_skip`, `skip`,
  `after_skip`), not an `i % skip_layer` test in a loop

Random numbers are drawn for the whole batch at once, as the original does,
so the port's samples are the original's for the same keys.
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Batch, Dim, Key, f32, fn


class XYZ(Dim, size=3): ...


class RGB(Dim, size=3): ...


class Ray(NamedTuple):
    origin: f32[XYZ]
    direction: f32[XYZ]
    viewdir: f32[XYZ]  # the direction, normalised


class Pixel(NamedTuple):
    rgb: f32[RGB]
    disp: f32  # inverse depth
    acc: f32  # opacity: 1 - the light that reaches the background


@fn
def camera_ray(pose: f32[XYZ, 4], focal: float, x: f32, y: f32, w: int, h: int) -> Ray:
    """The ray through pixel (x, y) of a w x h pinhole camera; `pose` is camera-to-world, looking down -z."""
    d = pose[:, :3] @ jnp.array([(x - w * 0.5) / focal, -(y - h * 0.5) / focal, -1.0])
    return Ray(pose[:, 3], d, d / jnp.linalg.norm(d))


@fn
def posenc[X](x: f32[X], min_deg: int, max_deg: int) -> f32[X * (1 + 2 * (max_deg - min_deg))]:
    """x, then sin and cos of x scaled by 2^min_deg, ..., 2^(max_deg - 1)."""
    assert min_deg <= max_deg
    scaled = jnp.outer(jnp.array([2.0**i for i in range(min_deg, max_deg)]), x).ravel()
    return jnp.concatenate([x, jnp.sin(jnp.concatenate([scaled, scaled + 0.5 * jnp.pi]))])  # cos x = sin(x + pi/2)


@expex.struct
class Dense[I, O]:
    kernel: f32[I, O] = jax.nn.initializers.glorot_uniform()
    bias: f32[O] = 0.0

    def __call__(self, x: f32[I]) -> f32[O]:
        return x @ self.kernel + self.bias


@expex.struct
class MLP[P, V, W, C]:
    """Density at a point (P features), and colour from the point and the view direction (V features).

    Width W; the point is fed in again after `skip`, and the colour branch is C wide.
    """

    first: Dense[P, W]
    before_skip: list[Dense[W, W]]
    skip: Dense[W + P, W]
    after_skip: list[Dense[W, W]]
    sigma: Dense[W, Literal[1]]
    bottleneck: Dense[W, W]
    condition: Dense[W + V, C]
    rgb: Dense[C, RGB]
    act: Callable = jax.nn.relu

    def __call__(self, x: f32[P], view: f32[V]) -> tuple[f32[RGB], f32]:
        """Raw colour and density."""
        h = self.act(self.first(x))
        for layer in self.before_skip:  # identical layers: one lax.scan
            h = self.act(layer(h))
        h = self.act(self.skip(jnp.concatenate([h, x])))
        for layer in self.after_skip:
            h = self.act(layer(h))
        h_view = self.act(self.condition(jnp.concatenate([self.bottleneck(h), view])))
        return self.rgb(h_view), self.sigma(h)[0]


@fn
def stratified[S](near: float, far: float, u: f32[S], randomized: bool, lindisp: bool) -> f32[S]:
    """S depths from near to far, evenly spaced (in disparity if lindisp); moved within their bins by u if randomized."""
    assert S >= 2  # also tells the prover which branch jnp.linspace takes
    t = jnp.linspace(0.0, 1.0, S)
    z = 1.0 / (1.0 / near * (1.0 - t) + 1.0 / far * t) if lindisp else near * (1.0 - t) + far * t
    if randomized:
        mids = 0.5 * (z[1:] + z[:-1])
        lower, upper = jnp.concatenate([z[:1], mids]), jnp.concatenate([mids, z[-1:]])
        z = lower + (upper - lower) * u
    return z


@fn
def composite[S](
    rgb: f32[S, RGB], sigma: f32[S], z: f32[S], direction: f32[XYZ], white_bkgd: bool
) -> tuple[Pixel, f32[S]]:
    """Alpha compositing along the ray: the pixel, and each sample's weight in it."""
    eps = 1e-10
    dists = jnp.append(z[1:] - z[:-1], 1e10) * jnp.linalg.norm(direction)
    alpha = 1.0 - jnp.exp(-sigma * dists)
    transmittance = jnp.concatenate([jnp.ones(1), jnp.cumprod(1.0 - alpha[:-1] + eps)])
    weights = alpha * transmittance
    depth, acc = weights @ z, weights.sum()
    disp = acc / depth
    disp = jnp.where((disp > 0) & (disp < 1 / eps) & (acc > eps), disp, 1 / eps)
    color = weights @ rgb + (1.0 - acc if white_bkgd else 0.0)
    return Pixel(color, disp, acc), weights


@fn
def sample_pdf[N, F](bins: f32[N + 1], weights: f32[N], u: f32[F]) -> f32[F]:
    """F depths from the piecewise-constant density with `weights` on the N bins: the inverse CDF at u."""
    assert N >= 1
    total = weights.sum()
    padding = jnp.maximum(0.0, 1e-5 - total)  # no NaNs when every weight is (nearly) zero
    pdf = (weights + padding / N) / (total + padding)
    cdf = jnp.concatenate([jnp.zeros(1), jnp.minimum(1.0, jnp.cumsum(pdf[:-1])), jnp.ones(1)])
    above = jnp.searchsorted(cdf, u, side="right")  # the bin of u is [cdf[above - 1], cdf[above])
    lo, hi = jnp.maximum(above - 1, 0), jnp.minimum(above, N)
    t = jnp.clip(jnp.nan_to_num((u - cdf[lo]) / (cdf[hi] - cdf[lo])), 0.0, 1.0)
    return jax.lax.stop_gradient(bins[lo] + t * (bins[hi] - bins[lo]))


@expex.struct
class NeRF[P, V, W, C]:
    """The coarse and fine MLPs, and how a ray is rendered with them. Defaults are the original's."""

    coarse: MLP[P, V, W, C]
    fine: MLP[P, V, W, C]
    coarse_samples: int = 64
    fine_samples: int = 128
    near: float = 2.0
    far: float = 6.0
    min_deg_point: int = 0
    max_deg_point: int = 10
    deg_view: int = 4
    lindisp: bool = False
    white_bkgd: bool = True
    rgb_act: Callable = jax.nn.sigmoid
    sigma_act: Callable = jax.nn.relu

    @classmethod
    def create(
        cls, key: Key, depth: int = 8, width: int = 256, width_condition: int = 128, skip_layer: int = 4, **settings
    ) -> NeRF:
        """The sizes from the original's settings: `depth` layers, the point fed in again after layer `skip_layer`."""
        assert skip_layer + 2 <= depth < 2 * skip_layer + 1  # exactly one skip connection, as the original's
        degrees = {k: settings.get(k, getattr(cls, k)) for k in ("min_deg_point", "max_deg_point", "deg_view")}
        P = 3 * (1 + 2 * (degrees["max_deg_point"] - degrees["min_deg_point"]))
        V = 3 * (1 + 2 * degrees["deg_view"])
        layers = dict(before_skip=skip_layer, after_skip=depth - skip_layer - 2)
        return cls(key, P=P, V=V, W=width, C=width_condition, **layers, **settings)

    def __call__[B](
        self, rays: Batch[Ray, B], rng_0: Key, rng_1: Key, randomized: bool
    ) -> tuple[Batch[Pixel, B], Batch[Pixel, B]]:
        """Coarse and fine pixels of B rays. Random numbers are drawn as the original draws them: per batch."""
        S, F = self.coarse_samples, self.fine_samples
        jitter = jr.uniform(jr.split(rng_0)[0], (B, S))
        if randomized:
            u = jr.uniform(jr.split(rng_1)[0], (B, F))
        else:  # evenly spaced over [0, 1), as uniform draws are
            u = jnp.broadcast_to(jnp.linspace(0.0, 1.0 - jnp.finfo(jnp.float32).eps, F), (B, F))
        return self.render.vectorize(rays, jitter, u, randomized)

    def render[S, F](self, ray: Ray, jitter: f32[S], u: f32[F], randomized: bool) -> tuple[Pixel, Pixel]:
        """One ray: S stratified samples through the coarse MLP, then S + F, F of them where its weights are, through the fine one."""
        assert S >= 3
        view = posenc(ray.viewdir, 0, self.deg_view)
        z = stratified(self.near, self.far, jitter, randomized, self.lindisp)
        coarse, weights = self.shade(self.coarse, ray, z, view)
        z = jnp.sort(jnp.concatenate([z, sample_pdf(0.5 * (z[1:] + z[:-1]), weights[1:-1], u)]))
        fine, _ = self.shade(self.fine, ray, z, view)
        return coarse, fine

    def shade[T](self, mlp: MLP[P, V, W, C], ray: Ray, z: f32[T], view: f32[V]) -> tuple[Pixel, f32[T]]:
        """The pixel from the samples at depths z, and their weights."""
        assert P == 3 * (1 + 2 * (self.max_deg_point - self.min_deg_point)) and V == 3 * (1 + 2 * self.deg_view)
        points = ray.origin + jnp.outer(z, ray.direction)
        rgb, sigma = mlp.vectorize(posenc.vectorize(points, self.min_deg_point, self.max_deg_point), view)
        return composite(self.rgb_act(rgb), self.sigma_act(sigma), z, ray.direction, self.white_bkgd)

    def loss[B](self, rays: Batch[Ray, B], pixels: f32[B, RGB], rng_0: Key, rng_1: Key) -> f32:
        """The training loss of the original's train.py: squared error of the fine and of the coarse colours."""
        coarse, fine = self(rays, rng_0, rng_1, randomized=True)
        return ((fine.rgb - pixels) ** 2).mean() + ((coarse.rgb - pixels) ** 2).mean()


if __name__ == "__main__":
    model = NeRF.create(jr.key(0), depth=8, width=64, width_condition=32)
    origins = jr.normal(jr.key(1), (16, 3)) * 0.5 + jnp.array([0.0, 0.0, 4.0])
    directions = jr.normal(jr.key(2), (16, 3)) * 0.2 + jnp.array([0.0, 0.0, -1.0])
    rays = Ray(origins, directions, directions / jnp.linalg.norm(directions, axis=-1, keepdims=True))
    render = jax.jit(NeRF.__call__, static_argnums=4)
    coarse, fine = render(model, rays, jr.key(3), jr.key(4), True)
    print("coarse rgb", coarse.rgb.shape, " fine rgb", fine.rgb.shape, " fine acc", fine.acc.round(3))
    print()
    print(expex.prove(__import__("__main__")))

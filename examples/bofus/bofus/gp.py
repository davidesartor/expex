from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_solve, cholesky
from vlse import optim

import typax as tx
from typax import Batch, Float, Key, fn

from . import kernels, rkhs, utils
from .rkhs import Periodic, RBFMixture

type Two = Literal[2]


@fn(jit=True)
def factorize[N](
    K: Float[N, N], y: Float[N]
) -> tuple[Float[N, N], Float[N], Float[N], Float, Float]:
    """Cholesky factor and solves of K, with the trend b and the REML scale nu.
    A nan observation marks padding, masked to an identity block of the factor.
    """
    # mask the covariance matrix and observations
    mask = jnp.isfinite(y)
    dof = jnp.maximum(mask.sum() - 1, 1)
    K = utils.mask_covariance(K, mask)
    y = jnp.where(mask, y, 0.0)

    # one solve for K^-1 @ 1 and K^-1 @ y, the trend b weights the observations by Ki_1
    K_chol = cholesky(K, lower=True)
    Ki_1, Ki_y = cho_solve((K_chol, True), jnp.stack([1.0 * mask, y], axis=-1)).T
    b = jnp.average(y, weights=Ki_1)

    # REML scale nu, one dof spent on b
    Ki_r = Ki_y - b * Ki_1
    nu = jnp.sum(jnp.where(mask, (y - b) * Ki_r, 0.0)) / dof

    # constant observations give nu = 0, floor it at their float resolution
    resolution = jnp.finfo(y.dtype).eps * jnp.max(jnp.abs(y))
    nu = jnp.maximum(nu, jnp.maximum(resolution**2, jnp.finfo(y.dtype).tiny))
    return K_chol, Ki_1, Ki_r, b, nu


@fn(jit=True)
def loglikelihood[N](K: Float[N, N], y: Float[N], nu: Float | None = None) -> Float:
    """REML likelihood: trend integrated out under a flat prior, scale nu concentrated unless given."""
    K_chol, Ki_1, _, _, nu_hat = factorize(K, y)
    dof = jnp.maximum(jnp.isfinite(y).sum() - 1, 1)
    logdetK = 2.0 * jnp.sum(jnp.log(jnp.diag(K_chol))) + jnp.log(Ki_1.sum())
    if nu is None:
        return -0.5 * (dof * jnp.log(nu_hat) + logdetK)
    return -0.5 * (dof * jnp.log(nu) + logdetK + dof * nu_hat / nu)


def _expand(tree, axis):
    return jax.tree.map(lambda z: jnp.expand_dims(z, axis), tree)


def _take(tree, index):
    return jax.tree.map(lambda z: z[index], tree)


@fn(jit=True)
def pairwise_squared_distance[N1, N2, O, M, D](
    l0: Float[D] | None,
    f1: Batch[RBFMixture[M, D], N1, O],
    f2: Batch[RBFMixture[M, D], N2, O] | None = None,
    periodic: Periodic = None,
) -> Float[N1, N2, O]:
    """Squared distances between two sets of functions, per output.
    Distances live in the rkhs of ambient lengthscale l0, in L2 on the box if l0 is None.
    """
    other = f1 if f2 is None else f2
    # distance() is written for one pair of mixtures: the (n1, 1, o) and (1, n2, o)
    # leading axes broadcast
    d2 = _expand(f1, 1).distance(_expand(other, 0), l0, periodic) ** 2

    # floating point arithmetic is not guaranteed to cancel out
    if f2 is None:
        I = jnp.eye(len(d2), dtype=bool)[..., None]
        d2 = jnp.where(I, 0.0, d2)
    return d2


@fn(jit=True)
def _fit[N, O, M, D](
    key: Key,
    f: Batch[RBFMixture[M, D], N, O],
    y: Float[N],
    y_std: Float[N] | None = None,
    profile: kernels.Profile = kernels.matern52,
    l0: Float[D] | None = None,
    periodic: Periodic = None,
    rho_range: Float[Two] = jnp.array([1e-2, 1e2]),
    nugget_range: Float[Two] = jnp.array([1e-4, 1e0]),
    nu_range: Float[Two] = jnp.array([1e-2, 1e2]),
    restarts: int = 1,
):
    """MAP fit of (rho, g) on the REML likelihood, nu integrated out unless y_std is given."""
    n, o, _, _ = f.l.shape

    # squared distances per output, constant throughout the optimisation
    mask = jnp.isfinite(y)
    d2 = pairwise_squared_distance(l0, f, periodic=periodic)

    # center rho to average inverse distances
    valid = (mask[:, None] & mask[None, :])[..., None] & jnp.isfinite(d2)
    assert jnp.any(valid, axis=(0, 1)).all(), "no valid distances"
    rho_scale = 1 / jnp.mean(d2, axis=(0, 1), where=valid)

    # known measurement variances, zeroed on padding so nan never reaches the gradients
    noise = None if y_std is None else jnp.where(mask, y_std, 0.0) ** 2

    def covariance(rho, g, nu):
        K = profile(utils.safe_sqrt(d2 @ rho)) + g * jnp.eye(n)
        return K if nu is None else K + jnp.diag(noise / nu)

    # center the nu search on the noiseless closed form at the box center
    g_center = jnp.sqrt(jnp.prod(nugget_range))
    nu_scale = factorize(covariance(rho_scale, g_center, None), y)[-1]

    def unpack(log_params):
        log_rho, log_g, *log_nu = log_params
        nu = nu_scale * jnp.exp(log_nu[0]) if log_nu else None
        return rho_scale * jnp.exp(log_rho), jnp.exp(log_g), nu

    # log-normal priors on (rho, g), box = ±3 std; nu stays flat in log, as in the noiseless branch
    ranges = (rho_range, nugget_range) + (() if y_std is None else (nu_range,))
    lo, hi = zip(*(jnp.log(r) for r in ranges))
    prior_mean = tuple((a + b) / 2 for a, b in zip(lo[:2], hi[:2]))
    prior_std = tuple((b - a) / 6 for a, b in zip(lo[:2], hi[:2]))

    def map_loss(log_params):
        rho, g, nu = unpack(log_params)
        z = jax.tree.map(lambda u, m, s: (u - m) / s, log_params[:2], prior_mean, prior_std)
        neg_log_prior = 0.5 * sum(jnp.sum(zi**2) for zi in z)
        return -loglikelihood(covariance(rho, g, nu), y, nu) + neg_log_prior

    # L-BFGS-B from uniform restarts over the log-space box
    shapes = ((restarts, o), (restarts,), (restarts,))
    keys = jr.split(key, len(ranges))
    starts = tuple(
        jr.uniform(k, s, minval=a, maxval=b) for k, s, a, b in zip(keys, shapes, lo, hi)
    )
    runs = []
    for start in tx.rows(starts):
        runs.append(optim.minimise(map_loss, start, bounds=(lo, hi)))
    results = tx.stack(runs)

    # extract the optimal parameters
    assert jnp.any(jnp.isfinite(results.f)), "fit ended with non finite loss"
    rho, g, nu = unpack(_take(results.x, jnp.nanargmin(results.f)))

    # condition on the observations, caching the solves predict needs
    K_chol, Ki_1, Ki_r, b, nu_hat = factorize(covariance(rho, g, nu), y)
    # integrating nu out leaves a student-t with n - 1 dof
    dof = None if nu is not None else jnp.maximum(mask.sum() - 1, 1).astype(y.dtype)
    nu = nu_hat if nu is None else nu
    return rho, g, nu, dof, b, K_chol, Ki_1, Ki_r


@tx.struct
class GaussianProcess[N, O, M, D]:
    """GP over functions, compared through their ambient-space or L2 distances.

    ``GaussianProcess(key, f, y)`` fits it: a MAP fit of the lengthscales and
    nugget on the REML likelihood, the scale integrated out unless ``y_std``
    gives the measurement noise. A nan observation marks a padded entry.
    """

    l0: Float[D] | None  # ambient lengthscale, None for L2

    # fitted parameters
    rho: Float[O]  # weight output dimensions
    g: Float  # noise nugget
    nu: Float  # covariance scale
    dof: Float | None  # student-t degrees of freedom
    b: Float  # mean trend

    # observations, nan values mark padded entries
    f: Batch[RBFMixture[M, D], N, O]
    y: Float[N]

    # cached for posterior evaluation, K is the covariance before scaling by nu
    K_chol: Float[N, N]
    Ki_1: Float[N]  # K^-1 @ 1
    Ki_r: Float[N]  # K^-1 @ (y - b)

    profile: kernels.Profile = kernels.matern52  # kernel profile, e.g. RBF or Matern
    periodic: Periodic = None  # dimensions that wrap; None for none

    def __init__(
        self,
        key: Key,
        f: Batch[RBFMixture[M, D], N, O],
        y: Float[N],
        *,
        y_std: Float[N] | None = None,
        profile: kernels.Profile = kernels.matern52,
        l0: Float[D] | None = None,
        periodic: Periodic = None,
        rho_range=jnp.array([1e-2, 1e2]),
        nugget_range=jnp.array([1e-4, 1e0]),
        nu_range=jnp.array([1e-2, 1e2]),
        restarts: int = 1,
    ):
        fitted = _fit(
            key, f, y, y_std, profile, l0, periodic, rho_range, nugget_range, nu_range, restarts
        )
        self.rho, self.g, self.nu, self.dof, self.b, self.K_chol, self.Ki_1, self.Ki_r = fitted
        self.profile, self.l0, self.periodic, self.f, self.y = profile, l0, periodic, f, y

    @fn(jit=True)
    def predict_joint[Q](self, f: Batch[RBFMixture[M, D], Q, O]) -> tuple[Float[Q], Float[Q, Q]]:
        """Posterior location and scale at f, student-t with self.dof dof (gaussian if None)."""
        # cancellation only approximates the exact zeros of coincident functions
        Kqq = self.profile(
            utils.safe_sqrt(pairwise_squared_distance(self.l0, f, periodic=self.periodic) @ self.rho)
        )
        Knq = self.profile(
            utils.safe_sqrt(
                pairwise_squared_distance(self.l0, self.f, f, periodic=self.periodic) @ self.rho
            )
        )

        # a nan observation marks padding, zero its cross covariance
        Knq = Knq * jnp.isfinite(self.y)[:, None]

        # posterior mean and unit-scale covariance
        mean = self.b + Knq.T @ self.Ki_r
        cov = Kqq - Knq.T @ cho_solve((self.K_chol, True), Knq)

        # correction from estimating the trend on the same data
        Kbq = 1 - Knq.T @ self.Ki_1
        cov = cov + jnp.outer(Kbq, Kbq) / self.Ki_1.sum()
        return mean, self.nu * cov

    @fn(jit=True)
    def predict_marginal[Q](self, f: Batch[RBFMixture[M, D], Q, O]) -> tuple[Float[Q], Float[Q]]:
        """Posterior location and squared scale of each function on its own."""
        mean, cov = self.predict_joint.lift(_expand(f, 1))
        return mean.squeeze(-1), cov.squeeze((-2, -1))


GaussianProcess.fit = staticmethod(GaussianProcess)  # the original API's constructor name

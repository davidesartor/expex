import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import vlse.optim

import typax as tx
from typax import Float, Int, Key, Static, fn

from . import gp, rkhs, utils
from .rkhs import RBFMixture


@fn(jit=True)
def upper_confidence_bound(mu, sigma, beta):
    return -mu + jnp.sqrt(beta) * sigma


@jax.custom_jvp
def student_t_cdf(z, dof):
    """Student-t cdf, differentiable through z only: betainc's own jvp is singular at z = 0."""
    lower_tail = 0.5 * jsp.special.betainc(dof / 2, 0.5, dof / (dof + z**2))
    return jnp.where(z < 0, lower_tail, 1.0 - lower_tail)


@student_t_cdf.defjvp
def _student_t_cdf_jvp(primals, tangents):
    z, dof = primals
    return student_t_cdf(z, dof), jsp.stats.t.pdf(z, dof) * tangents[0]


@fn(lift=True)
def log_student_t_expected_improvement(mu: Float, sigma: Float, y_best: Float, dof: Float) -> Float:
    """Stable log EI under a student-t, log1mexp in the tail is safe since the ratio stays < (dof-1)/dof."""
    z = (y_best - mu) / sigma
    log_A_pdf = jnp.log((dof + z**2) / (dof - 1)) + jsp.stats.t.logpdf(z, dof)
    if z > -1:  # direct
        log_h = jnp.log(z * student_t_cdf(z, dof) + jnp.exp(log_A_pdf))
    else:  # A * pdf - |z| * cdf in log space
        log_h = log_A_pdf + jax.nn.log1mexp(
            log_A_pdf - jnp.log(-z) - jnp.log(student_t_cdf(z, dof))
        )
    return jnp.log(sigma) + log_h


@fn(lift=True)
def log_expected_improvement(mu: Float, sigma: Float, y_best: Float) -> Float:
    """Stable log EI: log(sigma) + log(pdf(z) + z * cdf(z)) (Ament et al. 2023).
    https://arxiv.org/abs/2310.20708
    """
    z = (y_best - mu) / sigma
    z_asymptotic = -jnp.finfo(z.dtype).eps ** -0.25
    half_log_2pi = jnp.log(2 * jnp.pi) / 2
    if z > -1:  # direct evaluation
        log_h = jnp.log(jsp.stats.norm.pdf(z) + z * jsp.stats.norm.cdf(z))
    elif z < z_asymptotic:  # asymptotic expansion
        log_h = -(z**2) / 2 - half_log_2pi - 2 * jnp.log(-z)
    else:  # stable log1mexp trick
        log_h = (
            -(z**2) / 2
            - half_log_2pi
            + jax.nn.log1mexp(-jnp.log(-z) - jsp.stats.norm.logsf(-z) - z**2 / 2 - half_log_2pi)
        )
    return jnp.log(sigma) + log_h


@fn(lift=True)
def _log_softplus(z: Float) -> Float:
    """log softplus(z), finite gradients for large |z|."""
    if z > 30.0:
        return jnp.log(z)
    if z < -30.0:
        return z
    return jnp.log(jax.nn.softplus(z))


@fn(lift=True)
def q_log_expected_improvement[Q, S](
    mu: Float[Q],
    cov: Float[Q, Q],
    y_best: Float,
    eps: Float[S, Q],
    tau_relu: Float = 1e-6,
    tau_max: Float = 1e-2,
) -> Float:
    """Stable Monte Carlo batch log EI with fat-tailed relu and smooth max (Ament et al. 2023).
    https://arxiv.org/abs/2310.20708
    """
    L = jnp.linalg.cholesky(cov)
    ys = mu + jnp.einsum("sq,rq->sr", eps, L)

    # log softplus(y_best - y) at temperature tau_relu
    z = (y_best - ys) / tau_relu

    # cauchy tail keeps gradients alive for samples far worse than the incumbent
    log_cauchy = jnp.log(0.1) - jnp.log1p(z**2)
    log_imp = jnp.log(tau_relu) + jnp.logaddexp(_log_softplus(z), log_cauchy)

    # smooth max over the batch at temperature tau_max, then log mean over the samples
    log_max = tau_max * jax.nn.logsumexp(log_imp / tau_max, axis=-1)
    return jax.nn.logsumexp(log_max, axis=-1) - jnp.log(len(eps))


@fn(jit=True, K="n")
def boltzmann_select[N, K](
    key: Key, values: Float[N], n: Static[int], eta: Static[float] = 2.0
) -> Int[K]:
    """Sample n distinct indices with weights exp(eta * standardized values), keeping the argmax."""
    finite = jnp.isfinite(values)
    v = jnp.where(finite, values, jnp.nan)
    z = (v - jnp.nanmean(v)) / jnp.maximum(jnp.nanstd(v), jnp.finfo(v.dtype).eps)
    logits = jnp.where(finite, eta * z, -jnp.inf)
    logits[jnp.nanargmax(v)] = jnp.inf

    # gumbel top-k is sampling without replacement
    _, idx = jax.lax.top_k(logits + jr.gumbel(key, values.shape), n)
    return idx


def _concat(*trees):
    return jax.tree.map(lambda *zs: jnp.concatenate(zs), *trees)


def _perturb(key, z, lo, hi, sigma):
    """Gaussian perturbation relative to the range, clipped to the box."""
    return (z + sigma * (hi - lo) * jr.normal(key, z.shape)).clip(lo, hi)


@fn(jit=True, Q="batch_size")
def optimize_expected_improvement[N, O, M, D, Q](
    key: Key,
    surrogate: gp.GaussianProcess[N, O, M, D],
    l_range: tuple[Float, Float],
    a_range: tuple[Float, Float],
    batch_size: Static[int] = 1,
    multi_starts: Static[int] = 32,
    n_probes: Static[int] = 1024,
    probe_chunk: Static[int] = 1,
    solve_chunk: Static[int] = 1,
    gtol: Static[float] = 1e-5,
    ftol: Static[float] = 1e-5,
    n_mc: Static[int] = 512,
    n_best: Static[int] = 4,
    sigma_around_best: Static[float] = 1e-2,
    eta: Static[float] = 2.0,
) -> RBFMixture[Q, O, M, D]:
    """Maximize Monte Carlo batch EI over RBF mixtures, screening probe batches then L-BFGS-B."""
    _, o, m, d = surrogate.f.l.shape
    x_range = (jnp.asarray(-rkhs.X_MAX), jnp.asarray(rkhs.X_MAX))
    mask = jnp.isfinite(surrogate.y)

    log_l_range = (jnp.log(l_range[0]), jnp.log(l_range[1]))
    bounds = tuple(zip(log_l_range, x_range, a_range))

    # incumbent as the best posterior mean over the observations, consistent with the nugget
    mu_obs, _ = surrogate.predict_marginal(surrogate.f)
    y_best = jnp.where(mask, mu_obs, jnp.inf).min()

    # half the probe batches draw from the prior, half perturb the n_best observed points
    key_cover, key_center, key_fresh, key_perturb, key_select, key_mc = jr.split(key, 6)
    n_cover = n_probes // 2
    n_around = n_probes - n_cover
    cover = RBFMixture.sample(key_cover, (n_cover, batch_size, o, m, d), l_range, a_range)
    order = jnp.argsort(jnp.where(mask, surrogate.y, jnp.inf))
    n_top = jnp.minimum(n_best, mask.sum())
    center = order[jr.randint(key_center, (n_around, batch_size), 0, n_top)]
    around = gp._take(surrogate.f, center)

    # inert padded atoms get fresh lengthscales and centers, so the optimizer can grow them
    fresh = RBFMixture.sample(key_fresh, around.l.shape, l_range, a_range)
    inert = (around.a == 0.0)[..., None]
    around.l = jnp.where(inert, fresh.l, around.l)
    around.x = jnp.where(inert, fresh.x, around.x)

    # gaussian perturbation of (log_l, x, a) relative to each range, clipped to the box
    perturbed = []
    for k, z, (lo, hi) in zip(
        jr.split(key_perturb, 3), (jnp.log(around.l), around.x, around.a), zip(*bounds)
    ):
        perturbed.append(_perturb(k, z, lo, hi, sigma_around_best))
    probes = _concat((jnp.log(cover.l), cover.x, cover.a), tuple(perturbed))

    # stratified normal draws, fixed so the Monte Carlo objective is deterministic
    tiny = jnp.finfo(surrogate.y.dtype).eps
    u = utils.latin_hypercube_sample(key_mc, (n_mc, batch_size))
    eps = jsp.special.ndtri(u.clip(tiny, 1.0 - tiny))

    # student-t draws share one chi-square scale per sample across the batch
    if surrogate.dof is not None:
        chi2 = jr.chisquare(jr.fold_in(key_mc, 1), surrogate.dof, (n_mc,))
        eps = eps * jnp.sqrt(surrogate.dof / chi2)[:, None]
    tau_relu = 1e-6 * jnp.sqrt(surrogate.nu)

    # analytic log EI for a single point, MC batch log EI else
    def loss(log_l_x_a):
        log_l, x, a = log_l_x_a
        mu, cov = surrogate.predict_joint(RBFMixture(jnp.exp(log_l), x, a))
        if batch_size == 1:
            mu, sigma = mu.squeeze(), cov.squeeze() ** 0.5
            if surrogate.dof is None:
                return -log_expected_improvement(mu, sigma, y_best)
            return -log_student_t_expected_improvement(mu, sigma, y_best, surrogate.dof)
        cov = cov + 1e-6 * jnp.trace(cov) / batch_size * jnp.eye(batch_size)
        return -q_log_expected_improvement(mu, cov, y_best, eps, tau_relu)

    # screen the probe batches by the joint acquisition in chunks, Boltzmann sample the starts
    scores = []
    for probe in tx.rows(probes, chunk=probe_chunk):
        scores.append(-loss(probe))
    starts = gp._take(probes, boltzmann_select(key_select, jnp.asarray(scores), multi_starts, eta))

    # box constrained L-BFGS-B from each start, in sequence: together, every start would
    # run for as long as the slowest
    runs = []
    for start in tx.rows(starts, chunk=solve_chunk):
        runs.append(vlse.optim.minimise(loss, start, bounds=bounds, tol=gtol, ftol=ftol))
    results = tx.stack(runs)

    # return the batch with the best Monte Carlo EI
    assert jnp.any(jnp.isfinite(results.f)), "all restarts ended with non finite loss"
    log_l, x, a = gp._take(results.x, jnp.nanargmin(results.f))
    return RBFMixture(jnp.exp(log_l), x, a)

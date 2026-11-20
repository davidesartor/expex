"""Numerical parity between the original bofus and the typax port.

    # outputs of the original (needs a checkout of davidesartor/bofus@adapt_rho):
    python parity.py reference <checkout>/src reference.npz
    # compare the port in this directory against them:
    python parity.py check reference.npz

Every case feeds identical inputs and PRNG keys to the same public function
and records values and, where meaningful, gradients.
"""

import importlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

L_RANGE = (jnp.array(0.04), jnp.array(4.0))
A_RANGE = (jnp.array(-3.0), jnp.array(3.0))
PERIODIC = (True, False)


def cases(src: str, quick: bool = False) -> dict[str, np.ndarray]:
    """Outputs of the bofus package found in ``src``. ``quick`` skips the slow optimisations."""
    sys.path.insert(0, src)
    for name in [m for m in sys.modules if m == "bofus" or m.startswith("bofus.")]:
        del sys.modules[name]
    kernels = importlib.import_module("bofus.kernels")
    utils = importlib.import_module("bofus.utils")
    rkhs = importlib.import_module("bofus.rkhs")
    gp = importlib.import_module("bofus.gp")
    acq = importlib.import_module("bofus.acquisition")
    out: dict[str, np.ndarray] = {}

    def put(name, value):
        for i, leaf in enumerate(jax.tree.leaves(value)):
            out[f"{name}/{i}"] = np.asarray(leaf)

    # ---- kernels
    d = jnp.array([0.0, 1e-8, 0.3, 1.0, 4.0, 30.0, jnp.inf])
    profiles = {
        "rbf": kernels.rbf, "matern12": kernels.matern12, "matern32": kernels.matern32,
        "matern52": kernels.matern52, "matern(3)": kernels.matern(3), "matern(40)": kernels.matern(40),
    }  # fmt: skip
    for name, profile in profiles.items():
        put(f"kernels/{name}", profile(d))
        put(f"kernels/{name}/grad", jax.vmap(jax.grad(profile))(d))
    x1 = jr.normal(jr.key(0), (5, 3))
    x2 = jnp.concatenate([x1[:2], jr.normal(jr.key(1), (4, 3))])  # includes coincident points
    l = jnp.array([0.5, 1.0, 2.0])
    dist = lambda l, a, b: kernels.euclidean_distance(l, a[:, None], b[None])  # noqa: E731
    put("kernels/distance", dist(l, x1, x2))
    put("kernels/distance/grad", jax.grad(lambda l, a: dist(l, a, x2).sum(), (0, 1))(l, x1))
    put("kernels/matern52(distance)/grad", jax.grad(lambda l: kernels.matern52(dist(l, x1, x1)).sum())(l))

    # ---- utils
    put("utils/lhs", utils.latin_hypercube_sample(jr.key(2), (16, 3, 2)))
    put("utils/safe_sqrt/grad", jax.vmap(jax.grad(utils.safe_sqrt))(jnp.array([4.0, 0.0, -1.0])))

    # ---- rkhs
    f1 = rkhs.RBFMixture.sample(jr.key(3), (4, 5, 2), L_RANGE, A_RANGE)
    f2 = rkhs.RBFMixture.sample(jr.key(4), (4, 3, 2), L_RANGE, A_RANGE)
    put("rkhs/sample", f1)
    put("rkhs/pad_to", f1.pad_to(7))
    pts = jr.uniform(jr.key(5), (6, 2), minval=-1, maxval=1)
    put("rkhs/call", jax.vmap(f1)(pts))
    put("rkhs/call/periodic", jax.vmap(lambda p: f1(p, PERIODIC))(pts))
    l0 = jnp.array([0.03, 0.05])
    wide = f1._replace(l=f1.l + 0.05)
    wide2 = f2._replace(l=f2.l + 0.05)
    for tag, args in {
        "ambient": (l0, None), "ambient+periodic": (l0, PERIODIC),
        "l2": (None, None), "l2+periodic": (None, PERIODIC),
    }.items():  # fmt: skip
        put(f"rkhs/inner/{tag}", rkhs.inner(wide, wide2, *args))
        put(f"rkhs/distance/{tag}", rkhs.distance(wide, wide2, *args))
        grads = jax.grad(lambda a, b: rkhs.distance(a, b, *args).sum(), (0, 1))(wide, wide2)
        put(f"rkhs/distance/{tag}/grad", grads)
    put("rkhs/inner/outside-the-rkhs", rkhs.inner(f1, f2, jnp.array([1.0, 1.0])))
    put("rkhs/distance/self", rkhs.distance(wide, wide, l0))
    put("rkhs/distance/self/grad", jax.grad(lambda a: rkhs.distance(a, a, l0).sum())(wide))

    # ---- gp
    n, o, m, dd = 9, 2, 3, 2
    fs = rkhs.RBFMixture.sample(jr.key(6), (n, o, m, dd), L_RANGE, A_RANGE)
    fs = fs._replace(l=fs.l + 0.05)
    ys = jnp.sin(3 * fs.a.sum((-1, -2))) + 0.1 * fs.x.sum((-1, -2, -3))
    ys_padded = ys.at[-2:].set(jnp.nan)
    put("gp/pairwise", gp.pairwise_squared_distance(l0, fs))
    put("gp/pairwise/cross", gp.pairwise_squared_distance(None, fs, jax.tree.map(lambda z: z[:4], fs), PERIODIC))
    K = kernels.matern52(utils.safe_sqrt(gp.pairwise_squared_distance(l0, fs) @ jnp.array([0.7, 1.3])))
    K = K + 1e-3 * jnp.eye(n)
    put("gp/factorize", gp.factorize(K, ys_padded))
    put("gp/loglikelihood", gp.loglikelihood(K, ys_padded))
    put("gp/loglikelihood/nu", gp.loglikelihood(K, ys, jnp.array(0.8)))
    put("gp/loglikelihood/grad", jax.grad(lambda K: gp.loglikelihood(K, ys_padded))(K))
    queries = rkhs.RBFMixture.sample(jr.key(7), (5, o, m, dd), L_RANGE, A_RANGE)
    queries = queries._replace(l=queries.l + 0.05)
    fits = {
        "default": dict(l0=l0),
        "padded+restarts": dict(l0=l0, restarts=3),
        "l2+periodic+rbf": dict(periodic=PERIODIC, profile=kernels.rbf),
        "known-noise": dict(l0=l0, y_std=jnp.full(n, 0.05)),
    }
    surrogates = {}
    for tag, kw in fits.items():
        if quick and tag != "default":
            continue
        y = ys_padded if "padded" in tag else ys
        s = surrogates[tag] = gp.GaussianProcess.fit(jr.key(8), fs, y, **kw)
        put(f"gp/fit/{tag}/params", (s.rho, s.g, s.nu, s.b, s.K_chol, s.Ki_1, s.Ki_r))
        put(f"gp/fit/{tag}/dof", jnp.asarray(-1.0 if s.dof is None else s.dof))
        put(f"gp/fit/{tag}/predict_joint", s.predict_joint(queries))
        put(f"gp/fit/{tag}/predict_marginal", s.predict_marginal(queries))
        grads = jax.grad(lambda q: s.predict_joint(q)[1].sum())(queries)
        put(f"gp/fit/{tag}/predict_joint/grad", grads)

    # ---- acquisition
    mu = jnp.array([-30.0, -3.0, -1.0, -0.2, 0.0, 0.7, 5.0, 60.0])[:, None]
    sigma = jnp.array([1e-3, 0.5, 2.0])
    put("acq/ucb", acq.upper_confidence_bound(mu, sigma, jnp.array(2.0)))
    put("acq/log_ei", acq.log_expected_improvement(mu, sigma, 0.1))
    ei_grad = jax.grad(lambda m, s: acq.log_expected_improvement(m, s, 0.1).sum(), (0, 1))
    put("acq/log_ei/grad", ei_grad(mu, sigma))
    put("acq/log_t_ei", acq.log_student_t_expected_improvement(mu, sigma, 0.1, 5.0))
    t_grad = jax.grad(lambda m, s: acq.log_student_t_expected_improvement(m, s, 0.1, 5.0).sum(), (0, 1))
    put("acq/log_t_ei/grad", t_grad(mu, sigma))
    A = jr.normal(jr.key(9), (2, 3, 3))
    cov = A @ jnp.swapaxes(A, -1, -2) + 0.1 * jnp.eye(3)
    mus = jr.normal(jr.key(10), (2, 3))
    eps = jr.normal(jr.key(11), (64, 3))
    put("acq/q_log_ei", acq.q_log_expected_improvement(mus, cov, jnp.array([0.2, -0.5]), eps))
    q_grad = jax.grad(lambda m, c: acq.q_log_expected_improvement(m, c, 0.2, eps).sum(), (0, 1))
    put("acq/q_log_ei/grad", q_grad(mus, cov))
    values = jnp.array([0.3, jnp.nan, 2.0, -1.0, jnp.inf, 0.9, 1.1, -0.4])
    put("acq/boltzmann", acq.boltzmann_select(jr.key(12), values, 3))
    for tag, kw in {} if quick else {
        "student-t q=1": dict(surrogate="default"),
        "gaussian q=1": dict(surrogate="known-noise"),
        "student-t q=2": dict(surrogate="default", batch_size=2, n_mc=64),
        "periodic q=1 chunked": dict(surrogate="l2+periodic+rbf", probe_chunk=8, solve_chunk=2),
    }.items():
        s = surrogates[kw.pop("surrogate")]
        best = acq.optimize_expected_improvement(
            jr.key(13), s, L_RANGE, A_RANGE, multi_starts=4, n_probes=32, **kw
        )
        put(f"acq/optimize/{tag}", best)
        put(f"acq/optimize/{tag}/posterior", s.predict_joint(best))
    return out


def compare(got: dict[str, np.ndarray], ref) -> list[tuple[str, float, bool]]:
    rows = []
    for name in sorted(ref):
        if name not in got:
            continue
        a, b = got[name], ref[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append((name, float("inf"), False))
            continue
        same_special = np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(
            np.isinf(a), np.isinf(b)
        )
        fin = np.isfinite(a) & np.isfinite(b)
        if a.dtype.kind in "iub":
            err, ok = float(np.max(np.abs(a.astype(float) - b), initial=0.0)), np.array_equal(a, b)
        else:
            scale = np.maximum(1.0, np.abs(b[fin]))
            err = float(np.max(np.abs(a[fin] - b[fin]) / scale, initial=0.0))
            ok = same_special and np.array_equal(np.sign(a[~fin]), np.sign(b[~fin]))
        rows.append((name, err, bool(ok)))
    return rows


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "reference":
        np.savez_compressed(sys.argv[3], **cases(sys.argv[2]))
        print("wrote", sys.argv[3])
    else:
        ref = np.load(sys.argv[2])
        got = cases(__file__.rsplit("/", 1)[0])
        rows = compare(got, ref)
        groups: dict[str, list[tuple[str, float, bool]]] = {}
        for row in rows:
            groups.setdefault(row[0].split("/")[0], []).append(row)
        worst_all = 0.0
        for g, rs in groups.items():
            worst = max(rs, key=lambda r: r[1])
            worst_all = max(worst_all, worst[1])
            bad = [r[0] for r in rs if not r[2]]
            print(f"{g:8s} {len(rs):3d} arrays   worst relative difference {worst[1]:.2e}  ({worst[0]})"
                  + (f"   NaN/inf/shape mismatch: {bad}" if bad else ""))
        print(f"\nlargest relative difference overall: {worst_all:.2e}")

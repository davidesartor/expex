"""PPO on CartPole, ported from PureJaxRL (github.com/luchris429/purejaxrl, ppo.py).

The original is written with nested `jax.lax.scan`s over closures and
`jax.vmap` with hand-written `in_axes`. This port keeps the algorithm, the
random-number stream and the arithmetic identical, but writes it as:

* per-environment, per-example functions with shapes in their signatures,
  vectorised with `.lift`;
* plain Python `for` loops and `if` statements, staged by `@fn`.

`examples/compare_ppo.py` checks that it reproduces the original's numbers.

    python examples/ppo.py                  # train and print returns
    python -m typax examples/ppo.py         # prove the shapes
"""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import optax

import typax as tx
from typax import Batch, Bool, Key, f32, fn, i32

# ------------------------------------------------------------------ config


class Config(NamedTuple):
    lr: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128
    num_updates: int = 976
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True


def make_optimizer(cfg: Config) -> optax.GradientTransformation:
    def schedule(count):
        frac = 1.0 - (count // (cfg.num_minibatches * cfg.update_epochs)) / cfg.num_updates
        return cfg.lr * frac

    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(learning_rate=schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )


# ------------------------------------------------------------- environment

GRAVITY, MASSPOLE, TOTAL_MASS, LENGTH = 9.8, 0.1, 1.1, 0.5
POLEMASS_LENGTH, FORCE_MAG, TAU = 0.05, 10.0, 0.02
THETA_LIMIT, X_LIMIT, MAX_STEPS = 12 * 2 * jnp.pi / 360, 2.4, 500

type Obs = Literal[4]
type Actions = Literal[2]


class Cart(NamedTuple):
    x: f32
    x_dot: f32
    theta: f32
    theta_dot: f32
    time: i32


class EnvState(NamedTuple):
    """A cart plus the episode statistics that PureJaxRL's LogWrapper keeps."""

    cart: Cart
    episode_return: f32
    episode_length: i32
    returned_return: f32
    returned_length: i32
    timestep: i32


class Info(NamedTuple):
    returned_return: f32
    returned_length: i32
    timestep: i32
    returned_episode: Bool


@fn
def observe(c: Cart) -> f32[Obs]:
    return jnp.array([c.x, c.x_dot, c.theta, c.theta_dot])


@fn
def terminal(c: Cart) -> Bool:
    out_of_track = (c.x < -X_LIMIT) or (c.x > X_LIMIT)
    fallen = (c.theta < -THETA_LIMIT) or (c.theta > THETA_LIMIT)
    return out_of_track or fallen or c.time >= MAX_STEPS


@fn
def new_cart(key: Key) -> Cart:
    s = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
    return Cart(s[0], s[1], s[2], s[3], jnp.int32(0))


@fn
def reset(key: Key) -> tuple[f32[Obs], EnvState]:
    cart = new_cart(key)
    zero_f, zero_i = jnp.float32(0), jnp.int32(0)
    return observe(cart), EnvState(cart, zero_f, zero_i, zero_f, zero_i, zero_i)


@fn
def physics(c: Cart, action: i32) -> Cart:
    force = FORCE_MAG * action - FORCE_MAG * (1 - action)
    cos, sin = jnp.cos(c.theta), jnp.sin(c.theta)
    temp = (force + POLEMASS_LENGTH * c.theta_dot**2 * sin) / TOTAL_MASS
    theta_acc = (GRAVITY * sin - cos * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos / TOTAL_MASS
    return Cart(
        c.x + TAU * c.x_dot,
        c.x_dot + TAU * x_acc,
        c.theta + TAU * c.theta_dot,
        c.theta_dot + TAU * theta_acc,
        c.time + 1,
    )


@fn
def step(
    key: Key, s: EnvState, action: i32
) -> tuple[f32[Obs], EnvState, f32, Bool, Info]:
    """One environment step, starting a new episode when the current one ends."""
    _, reset_key = jax.random.split(key)
    reward = 1.0 - terminal(s.cart)  # based on the state before the step
    cart = physics(s.cart, action)
    done = terminal(cart)

    episode_return = s.episode_return + reward
    episode_length = s.episode_length + 1
    if done:
        cart = new_cart(reset_key)
        s = EnvState(cart, jnp.float32(0), jnp.int32(0), episode_return, episode_length, s.timestep + 1)
    else:
        s = EnvState(cart, episode_return, episode_length, s.returned_return, s.returned_length, s.timestep + 1)
    info = Info(s.returned_return, s.returned_length, s.timestep, done)
    return observe(cart), s, jnp.float32(reward), done, info


# ------------------------------------------------------------------ network


class Layer[I, O](NamedTuple):
    w: f32[I, O]
    b: f32[O]


class Net[H](NamedTuple):
    actor: tuple[Layer[Obs, H], Layer[H, H], Layer[H, Actions]]
    critic: tuple[Layer[Obs, H], Layer[H, H], Layer[H, Literal[1]]]


@fn
def mlp[I, H, O](layers: tuple[Layer[I, H], Layer[H, H], Layer[H, O]], x: f32[I]) -> f32[O]:
    a, b, c = layers
    h: f32[H] = jnp.tanh(x @ a.w + a.b)
    h = jnp.tanh(h @ b.w + b.b)
    return h @ c.w + c.b


@fn
def policy[H](net: Net[H], obs: f32[Obs]) -> tuple[f32[Actions], f32]:
    """Normalised action log-probabilities and the state value."""
    return jax.nn.log_softmax(mlp(net.actor, obs)), mlp(net.critic, obs)[0]


def init_net(key, hidden: int = 64) -> Net:
    ortho = jax.nn.initializers.orthogonal
    keys = iter(jax.random.split(key, 6))

    def layer(i, o, scale):
        return Layer(ortho(scale)(next(keys), (i, o)), jnp.zeros(o))

    s = jnp.sqrt(2)
    return Net(
        (layer(4, hidden, s), layer(hidden, hidden, s), layer(hidden, 2, 0.01)),
        (layer(4, hidden, s), layer(hidden, hidden, s), layer(hidden, 1, 1.0)),
    )


# ---------------------------------------------------------------- algorithm


class Transition(NamedTuple):
    done: Bool
    action: i32
    value: f32
    reward: f32
    log_prob: f32
    obs: f32[Obs]
    info: Info


@fn(examples={"gamma": [0.99], "lam": [0.95]}, where="T >= 1")
def gae[T](
    reward: f32[T], value: f32[T], done: Bool[T], last_value: f32,
    gamma: float, lam: float,
) -> f32[T]:  # fmt: skip
    """Generalised advantage estimates for one environment's trajectory."""
    advantage, next_value = jnp.float32(0), last_value
    out = []
    for t in reversed(range(reward.shape[0])):
        delta = reward[t] + gamma * next_value * (1 - done[t]) - value[t]
        advantage = delta + gamma * lam * (1 - done[t]) * advantage
        next_value = value[t]
        out.append(advantage)
    return jnp.asarray(out)[::-1]


@fn
def ppo_loss[M, H](
    net: Net[H], batch: Batch[Transition, M], advantage: f32[M], target: f32[M], cfg: Config
) -> tuple[f32, tuple[f32, f32, f32]]:
    logp_all, value = policy.lift(net, batch.obs)
    log_prob: f32[M] = jnp.sum(logp_all * jax.nn.one_hot(batch.action, 2), axis=-1)

    clipped = batch.value + (value - batch.value).clip(-cfg.clip_eps, cfg.clip_eps)
    value_loss = 0.5 * jnp.maximum(jnp.square(value - target), jnp.square(clipped - target)).mean()

    ratio = jnp.exp(log_prob - batch.log_prob)
    adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    actor_loss = -jnp.minimum(
        ratio * adv, jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    ).mean()
    entropy = -jnp.sum(jnp.exp(logp_all) * logp_all, axis=-1).mean()

    total = actor_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
    return total, (value_loss, actor_loss, entropy)


def flatten_time(tree, n):
    return jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), tree)


def take(tree, index):
    return jax.tree.map(lambda x: jnp.take(x, index, axis=0), tree)


def split_rows(tree, parts):
    return jax.tree.map(lambda x: x.reshape((parts, -1) + x.shape[1:]), tree)


def time_major(x):
    return jnp.swapaxes(x, 0, 1)


@fn(examples={"cfg": [Config(num_envs=3, num_steps=8, num_updates=2, num_minibatches=2)]})
def train[H, T, N](key: Key, cfg: Config, net: Net[H] | None = None):
    """Train PPO. Returns the final network and per-step episode statistics."""
    optimizer = make_optimizer(cfg)
    key, init_key = jax.random.split(key)
    if net is None:
        net = init_net(init_key)
    opt_state = optimizer.init(net)

    key, reset_key = jax.random.split(key)
    obs, env = reset.lift(jax.random.split(reset_key, cfg.num_envs))
    _, key = jax.random.split(key)

    metrics = []
    for _update in range(cfg.num_updates):
        # Collect a trajectory from every environment.
        trajectory = []
        for _t in range(cfg.num_steps):
            key, act_key = jax.random.split(key)
            logp, value = policy.lift(net, obs)
            action = jax.random.categorical(act_key, logp, shape=(1, cfg.num_envs))[0]
            log_prob = jnp.sum(logp * jax.nn.one_hot(action, 2), axis=-1)

            key, step_key = jax.random.split(key)
            step_keys = jax.random.split(step_key, cfg.num_envs)
            next_obs, env, reward, done, info = step.lift(step_keys, env, action)
            trajectory.append(Transition(done, action, value, reward, log_prob, obs, info))
            obs = next_obs
        traj: Batch[Transition, T, N] = tx.stack(trajectory)

        # Advantages, per environment.
        _, last_value = policy.lift(net, obs)
        advantages = time_major(
            gae.lift(
                time_major(traj.reward), time_major(traj.value), time_major(traj.done),
                last_value, cfg.gamma, cfg.gae_lambda,
            )
        )  # fmt: skip
        targets = advantages + traj.value

        # Several epochs of minibatch updates.
        batch_size = cfg.num_steps * cfg.num_envs
        for _epoch in range(cfg.update_epochs):
            key, perm_key = jax.random.split(key)
            order = jax.random.permutation(perm_key, batch_size)
            data = take(flatten_time((traj, advantages, targets), batch_size), order)
            for batch, adv, target in tx.rows(split_rows(data, cfg.num_minibatches)):
                grads, _aux = jax.grad(ppo_loss, has_aux=True)(net, batch, adv, target, cfg)
                updates, opt_state = optimizer.update(grads, opt_state, net)
                net = optax.apply_updates(net, updates)

        metrics.append(traj.info)
    return net, tx.stack(metrics)


if __name__ == "__main__":
    cfg = Config(num_updates=200)
    net, info = jax.jit(lambda k: train(k, cfg))(jax.random.key(30))
    finished = info.returned_episode
    returns = jnp.where(finished, info.returned_return, jnp.nan)
    for lo in range(0, cfg.num_updates, 40):
        chunk = returns[lo : lo + 40]
        print(f"updates {lo:3d}-{lo + 39:3d}: mean episode return {jnp.nanmean(chunk):6.1f}")
    print("\nhow the loops in train() ran:")
    for line, how in train.flow.items():
        print(f"  line {line}: {how}")

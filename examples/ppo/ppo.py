"""PPO on CartPole, ported from PureJaxRL (github.com/luchris429/purejaxrl, ppo.py).

The original is written with nested `jax.lax.scan`s over closures and
`jax.vmap` with hand-written `in_axes`. This port keeps the algorithm, the
random-number stream and the arithmetic identical, but writes it as:

* modules (`@expex.struct`) with shapes in their signatures, written for one
  environment or one example and vectorised with `.lift`;
* plain Python `for` loops and `if` statements, staged by `@fn`.

`compare.py` (next to this file) checks that it reproduces the original's numbers.

    python examples/ppo/ppo.py              # train and print returns
    python -m expex examples/ppo/ppo.py     # prove the shapes
"""

from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import expex
from expex import Batch, Bool, Key, f32, fn, i32

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
SQRT2 = 2**0.5

type Obs = Literal[4]
type Actions = Literal[2]


@expex.struct
class Cart:
    x: f32
    x_dot: f32
    theta: f32
    theta_dot: f32
    time: i32

    @staticmethod
    def new(key: Key) -> "Cart":
        s = jr.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
        return Cart(s[0], s[1], s[2], s[3], jnp.int32(0))

    def observe(self) -> f32[Obs]:
        return jnp.array([self.x, self.x_dot, self.theta, self.theta_dot])

    def terminal(self) -> Bool:
        out_of_track = (self.x < -X_LIMIT) or (self.x > X_LIMIT)
        fallen = (self.theta < -THETA_LIMIT) or (self.theta > THETA_LIMIT)
        return out_of_track or fallen or self.time >= MAX_STEPS

    def push(self, action: i32) -> "Cart":
        force = FORCE_MAG * action - FORCE_MAG * (1 - action)
        cos, sin = jnp.cos(self.theta), jnp.sin(self.theta)
        temp = (force + POLEMASS_LENGTH * self.theta_dot**2 * sin) / TOTAL_MASS
        theta_acc = (GRAVITY * sin - cos * temp) / (
            LENGTH * (4.0 / 3.0 - MASSPOLE * cos**2 / TOTAL_MASS)
        )
        x_acc = temp - POLEMASS_LENGTH * theta_acc * cos / TOTAL_MASS
        return Cart(
            self.x + TAU * self.x_dot,
            self.x_dot + TAU * x_acc,
            self.theta + TAU * self.theta_dot,
            self.theta_dot + TAU * theta_acc,
            self.time + 1,
        )


@expex.struct
class Info:
    returned_return: f32
    returned_length: i32
    timestep: i32
    returned_episode: Bool


@expex.struct
class Env:
    """A cart plus the episode statistics that PureJaxRL's LogWrapper keeps."""

    cart: Cart
    episode_return: f32
    episode_length: i32
    returned_return: f32
    returned_length: i32
    timestep: i32

    @staticmethod
    def reset(key: Key) -> tuple[f32[Obs], "Env"]:
        cart = Cart.new(key)
        zero_f, zero_i = jnp.float32(0), jnp.int32(0)
        return cart.observe(), Env(cart, zero_f, zero_i, zero_f, zero_i, zero_i)

    def step(self, key: Key, action: i32) -> tuple[f32[Obs], "Env", f32, Bool, Info]:
        """One step, starting a new episode when the current one ends."""
        _, reset_key = jr.split(key)
        reward = 1.0 - self.cart.terminal()  # based on the state before the step
        cart = self.cart.push(action)
        done = cart.terminal()

        episode_return = self.episode_return + reward
        episode_length = self.episode_length + 1
        if done:
            cart = Cart.new(reset_key)
            env = Env(cart, jnp.float32(0), jnp.int32(0), episode_return, episode_length, self.timestep + 1)
        else:
            env = Env(cart, episode_return, episode_length, self.returned_return, self.returned_length, self.timestep + 1)
        info = Info(env.returned_return, env.returned_length, env.timestep, done)
        return cart.observe(), env, jnp.float32(reward), done, info


# ------------------------------------------------------------------ network


@expex.struct
class Layer[I, O]:
    w: f32[I, O]
    b: f32[O]

    def __init__(self, key: Key, i: int, o: int, scale: float = SQRT2):
        self.w = jax.nn.initializers.orthogonal(scale)(key, (i, o))
        self.b = jnp.zeros(o)

    def __call__(self, x: f32[I]) -> f32[O]:
        return x @ self.w + self.b


@expex.struct
class MLP[I, H, O]:
    layers: tuple[Layer[I, H], Layer[H, H], Layer[H, O]]

    def __init__(self, key: Key, i: int, hidden: int, o: int, out_scale: float):
        k1, k2, k3 = jr.split(key, 3)
        self.layers = (Layer(k1, i, hidden), Layer(k2, hidden, hidden), Layer(k3, hidden, o, out_scale))

    def __call__(self, x: f32[I]) -> f32[O]:
        a, b, c = self.layers
        h: f32[H] = jnp.tanh(a(x))
        return c(jnp.tanh(b(h)))


@expex.struct
class Transition:
    done: Bool
    action: i32
    value: f32
    reward: f32
    log_prob: f32
    obs: f32[Obs]
    info: Info


@expex.struct
class Net[H]:
    actor: MLP[Obs, H, Actions]
    critic: MLP[Obs, H, Literal[1]]

    def __init__(self, key: Key, hidden: int = 64):
        ka, kc = jr.split(key)
        self.actor = MLP(ka, 4, hidden, 2, out_scale=0.01)
        self.critic = MLP(kc, 4, hidden, 1, out_scale=1.0)

    def __call__(self, obs: f32[Obs]) -> tuple[f32[Actions], f32]:
        """Normalised action log-probabilities and the state value."""
        return jax.nn.log_softmax(self.actor(obs)), self.critic(obs)[0]

    def loss[M](
        self, batch: Batch[Transition, M], advantage: f32[M], target: f32[M], cfg: Config
    ) -> tuple[f32, tuple[f32, f32, f32]]:
        """The PPO objective on a minibatch: clipped policy and value losses, entropy bonus."""
        logp_all, value = self.lift(batch.obs)
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


# ---------------------------------------------------------------- algorithm


@fn
def gae[T](
    reward: f32[T], value: f32[T], done: Bool[T], last_value: f32,
    gamma: float, lam: float,
) -> f32[T]:  # fmt: skip
    """Generalised advantage estimates for one environment's trajectory."""
    assert T >= 1
    advantage, next_value = jnp.float32(0), last_value
    out = []
    for t in reversed(range(reward.shape[0])):
        delta = reward[t] + gamma * next_value * (1 - done[t]) - value[t]
        advantage = delta + gamma * lam * (1 - done[t]) * advantage
        next_value = value[t]
        out.append(advantage)
    return jnp.asarray(out)[::-1]


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
    key, init_key = jr.split(key)
    if net is None:
        net = Net(init_key)
    opt_state = optimizer.init(net)

    key, reset_key = jr.split(key)
    obs, env = Env.reset.lift(jr.split(reset_key, cfg.num_envs))
    _, key = jr.split(key)

    metrics = []
    for _update in range(cfg.num_updates):
        # Collect a trajectory from every environment.
        trajectory = []
        for _t in range(cfg.num_steps):
            key, act_key = jr.split(key)
            logp, value = net.lift(obs)
            action = jr.categorical(act_key, logp, shape=(1, cfg.num_envs))[0]
            log_prob = jnp.sum(logp * jax.nn.one_hot(action, 2), axis=-1)

            key, step_key = jr.split(key)
            step_keys = jr.split(step_key, cfg.num_envs)
            next_obs, env, reward, done, info = env.step.lift(step_keys, action)
            trajectory.append(Transition(done, action, value, reward, log_prob, obs, info))
            obs = next_obs
        traj: Batch[Transition, T, N] = expex.stack(trajectory)

        # Advantages, per environment.
        _, last_value = net.lift(obs)
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
            key, perm_key = jr.split(key)
            order = jr.permutation(perm_key, batch_size)
            data = take(flatten_time((traj, advantages, targets), batch_size), order)
            for batch, adv, target in expex.rows(split_rows(data, cfg.num_minibatches)):
                grads, _aux = jax.grad(Net.loss, has_aux=True)(net, batch, adv, target, cfg)
                updates, opt_state = optimizer.update(grads, opt_state, net)
                net = optax.apply_updates(net, updates)

        metrics.append(traj.info)
    return net, expex.stack(metrics)


if __name__ == "__main__":
    cfg = Config(num_updates=200)
    net, info = jax.jit(lambda k: train(k, cfg))(jr.key(30))
    finished = info.returned_episode
    returns = jnp.where(finished, info.returned_return, jnp.nan)
    for lo in range(0, cfg.num_updates, 40):
        chunk = returns[lo : lo + 40]
        print(f"updates {lo:3d}-{lo + 39:3d}: mean episode return {jnp.nanmean(chunk):6.1f}")
    print("\nhow the loops in train() ran:")
    for line, how in train.flow.items():
        print(f"  line {line}: {how}")

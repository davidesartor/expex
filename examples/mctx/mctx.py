"""MuZero and Gumbel MuZero tree search, ported from mctx (github.com/google-deepmind/mctx).

mctx searches a batch of trees at once. Its tree arrays carry a batch axis,
the search is `lax.fori_loop` over simulations around a `jax.vmap`ped
`lax.while_loop` that walks down to a leaf and another that backs values up
to the root, and choosing between the root and interior selection rules is a
`lax.cond`. This port writes the same search for ONE tree, with plain Python,
and batches it with `.vectorize`:

* `Tree[N, A, E]` holds N nodes with A actions and embeddings of width E;
* `simulate` is `while True:` with a `return`, `backward` is
  `while node != ROOT:`, the root rule is `if depth == 0:`, and
  `search` is `for sim in range(S):` (staged as `lax.while_loop`s, a
  `lax.cond` and a `lax.scan`);
* updates are `visits[parent] += 1`, not `x.at[b, i].set(...)` under `vmap`;
* sizes by name: `jr.uniform(key, (A,))`, `Tree[S + 1, A, E]`.

Trees, action weights and actions are identical to mctx's for the same keys
(tests/ports/test_mctx.py). Differences in the API, all for one tree:

* the recurrent function is written for one state:
  `recurrent_fn(params, key, action: i32, embedding: f32[E]) -> (Step[A], f32[E])`,
  and the root is one `Root[A, E]` (mctx's `RecurrentFnOutput` and `RootFnOutput`
  without the batch axis); a batch of roots is `Batch[Root[A, E], B]`;
* an embedding is one vector per node (mctx allows any pytree);
* `extra_data` is the root's Gumbel noise, a field of the tree (zeros for MuZero);
* each tree's recurrent function gets its own key (`jr.split(key, B)` of
  mctx's key for the batch, which a batched model would use as a whole).

    python examples/mctx/mctx.py            # search a small MDP, print one tree
    python -m expex examples/mctx/mctx.py   # prove the shapes
"""

import functools
import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

import expex
from expex import Batch, Bool, Key, f32, i32

ROOT, UNVISITED, NO_PARENT = 0, -1, -1


@expex.struct
class Root[A, E]:
    """The representation network's output for the state at the root (mctx's `RootFnOutput`)."""

    prior_logits: f32[A]
    value: f32
    embedding: f32[E]


@expex.struct
class Step[A]:
    """The recurrent function's output for one transition (mctx's `RecurrentFnOutput`)."""

    reward: f32
    discount: f32  # between `reward` and `value`
    prior_logits: f32[A]
    value: f32  # of the state reached


type Model = Callable[..., tuple[Step, Any]]  # (params, key, action: i32, embedding: f32[E]) -> (Step[A], f32[E])
type Select = Callable[..., i32]  # (key, tree, node: i32[, depth: i32]) -> action


@expex.struct
class Summary[A]:
    """Statistics of the root (mctx's `SearchSummary`)."""

    visit_counts: f32[A]
    visit_probs: f32[A]
    value: f32
    qvalues: f32[A]


@expex.struct
class Tree[N, A, E]:
    """A search tree of N nodes: node 0 is the root, node i is the one first reached by simulation i."""

    node_visits: i32[N]
    raw_values: f32[N]  # the network's value
    node_values: f32[N]  # the search's value
    parents: i32[N]
    action_from_parent: i32[N]
    children_index: i32[N, A]  # UNVISITED, or the child's node index
    children_prior_logits: f32[N, A]
    children_visits: i32[N, A]
    children_rewards: f32[N, A]
    children_discounts: f32[N, A]
    children_values: f32[N, A]
    embeddings: f32[N, E]
    root_invalid_actions: Bool[A]
    root_gumbel: f32[A]  # mctx's extra_data for Gumbel MuZero

    def qvalues(self, node: i32) -> f32[A]:
        return self.children_rewards[node] + self.children_discounts[node] * self.children_values[node]

    def summary(self) -> Summary[A]:
        counts = self.children_visits[ROOT].astype(jnp.float32)
        total = counts.sum()
        probs = counts / total if total > 0 else jnp.full(A, 1 / A)
        return Summary(counts, probs, self.node_values[ROOT], self.qvalues(ROOT))

    def simulate(self, key: Key, select_root: Select, select_interior: Select, max_depth: int) -> tuple[i32, i32]:
        """Descend from the root to an unexpanded action, or to `max_depth`: (node, action)."""
        node, depth = ROOT, 0
        while True:
            key, pick = jr.split(key)
            if depth == 0:
                action = select_root(pick, self, node)
            else:
                action = select_interior(pick, self, node, depth)
            child = self.children_index[node, action]
            depth += 1
            if child == UNVISITED or depth >= max_depth:
                return node, action
            node = child

    def expand(self, params: Any, key: Key, model: Model, parent: i32, action: i32, child: i32) -> Tree[N, A, E]:
        """Evaluate `action` from `parent` with the model, and store the result as node `child`."""
        step, embedding = model(params, key, action, self.embeddings[parent])
        tree = self.fill(child, step.prior_logits, step.value, embedding)
        index, rewards, discounts = tree.children_index, tree.children_rewards, tree.children_discounts
        parents, actions = tree.parents, tree.action_from_parent
        index[parent, action] = child
        rewards[parent, action] = step.reward
        discounts[parent, action] = step.discount
        parents[child] = parent
        actions[child] = action
        return tree.replace(
            children_index=index, children_rewards=rewards, children_discounts=discounts,
            parents=parents, action_from_parent=actions,
        )  # fmt: skip

    def fill(self, node: i32, prior_logits: f32[A], value: f32, embedding: f32[E]) -> Tree[N, A, E]:
        """Node `node`'s network outputs. With `max_depth`, a leaf can be expanded again."""
        logits, raw, values, visits, embeddings = (
            self.children_prior_logits, self.raw_values, self.node_values, self.node_visits, self.embeddings
        )  # fmt: skip
        logits[node] = prior_logits
        raw[node] = value
        values[node] = value
        visits[node] += 1
        embeddings[node] = embedding
        return self.replace(
            children_prior_logits=logits, raw_values=raw, node_values=values, node_visits=visits, embeddings=embeddings
        )

    def backward(self, leaf: i32) -> Tree[N, A, E]:
        """Back the leaf's value up to the root, updating the statistics on the way."""
        visits, values = self.node_visits, self.node_values
        child_visits, child_values = self.children_visits, self.children_values
        value, node = values[leaf], leaf
        while node != ROOT:
            parent, action = self.parents[node], self.action_from_parent[node]
            value = self.children_rewards[parent, action] + self.children_discounts[parent, action] * value
            child_values[parent, action] = values[node]
            child_visits[parent, action] += 1
            values[parent] = (values[parent] * visits[parent] + value) / (visits[parent] + 1.0)
            visits[parent] += 1
            node = parent
        return self.replace(
            node_visits=visits, node_values=values, children_visits=child_visits, children_values=child_values
        )


@expex.fn
def new_tree[A, E](root: Root[A, E], nodes: int, invalid: Bool[A], gumbel: f32[A]) -> Tree[nodes, A, E]:
    """An empty tree of `nodes` nodes with the root filled in."""
    tree = Tree(
        node_visits=jnp.zeros(nodes, jnp.int32),
        raw_values=jnp.zeros(nodes),
        node_values=jnp.zeros(nodes),
        parents=jnp.full(nodes, NO_PARENT),
        action_from_parent=jnp.full(nodes, NO_PARENT),
        children_index=jnp.full((nodes, A), UNVISITED),
        children_prior_logits=jnp.zeros((nodes, A)),
        children_visits=jnp.zeros((nodes, A), jnp.int32),
        children_rewards=jnp.zeros((nodes, A)),
        children_discounts=jnp.zeros((nodes, A)),
        children_values=jnp.zeros((nodes, A)),
        embeddings=jnp.zeros((nodes, E)),
        root_invalid_actions=invalid,
        root_gumbel=gumbel,
    )
    return tree.fill(ROOT, root.prior_logits, root.value, root.embedding)


@expex.fn
def search[S, A, E](
    params: Any,
    select_keys: Key[S],
    model_keys: Key[S],
    root: Root[A, E],
    model: Model,
    select_root: Select,
    select_interior: Select,
    invalid: Bool[A],
    gumbel: f32[A],
    max_depth: int | None = None,
) -> Tree[S + 1, A, E]:
    """S simulations, each adding one node (mctx's `search`, for one tree)."""
    tree = new_tree(root, S + 1, invalid, gumbel)
    for sim in range(S):
        parent, action = tree.simulate(select_keys[sim], select_root, select_interior, max_depth or S)
        child = tree.children_index[parent, action]
        if child == UNVISITED:  # else max_depth was reached: evaluate the leaf again
            child = sim + 1
        tree = tree.expand(params, model_keys[sim], model, parent, action, child)
        tree = tree.backward(child)
    return tree


@expex.fn
def search_keys(key: Key, simulations: int) -> tuple[Key[simulations], Key[simulations]]:
    """mctx's keys: three per simulation, (next, select, model). `.vectorize` splits each one per tree."""
    select, model = [], []
    for _ in range(simulations):
        key, s, m = jr.split(key, 3)
        select.append(s)
        model.append(m)
    return jnp.stack(select), jnp.stack(model)


# ------------------------------------------------------------------ q-value transforms


@expex.fn
def qtransform_by_parent_and_siblings[N, A, E](tree: Tree[N, A, E], node: i32, *, epsilon: float = 1e-8) -> f32[A]:
    """Q-values scaled to [0, 1] by the min and max over the node's value and its children's; 0 if unvisited."""
    q, visited, value = tree.qvalues(node), tree.children_visits[node] > 0, tree.node_values[node]
    safe = jnp.where(visited, q, value)
    lo, hi = jnp.minimum(value, safe.min()), jnp.maximum(value, safe.max())
    return (jnp.where(visited, q, lo) - lo) / jnp.maximum(hi - lo, epsilon)


@expex.fn
def qtransform_completed_by_mix_value[N, A, E](
    tree: Tree[N, A, E],
    node: i32,
    *,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: float = 1e-8,
) -> f32[A]:
    """Q-values with unvisited ones completed by the mixed value, scaled for Gumbel MuZero."""
    q, visits, value = tree.qvalues(node), tree.children_visits[node], tree.raw_values[node]
    if use_mixed_value:
        value = mixed_value(value, q, visits, jax.nn.softmax(tree.children_prior_logits[node]))
    completed = jnp.where(visits > 0, q, value)
    if rescale_values:
        completed = (completed - completed.min()) / jnp.maximum(completed.max() - completed.min(), epsilon)
    return (maxvisit_init + visits.max()) * value_scale * completed


@expex.fn
def mixed_value[A](raw_value: f32, q: f32[A], visits: i32[A], prior: f32[A]) -> f32:
    """The raw value mixed with the prior-weighted Q-values of the visited actions."""
    total, visited = visits.sum(), visits > 0
    prior = jnp.maximum(prior, expex.tiny(prior))  # no NaN if a visited action has zero prior
    visited_prior = jnp.where(visited, prior, 0.0).sum()
    weighted_q = jnp.where(visited, prior * q / jnp.where(visited, visited_prior, 1.0), 0.0).sum()
    return (raw_value + total * weighted_q) / (total + 1)


# ------------------------------------------------------------------ action selection


@expex.fn
def muzero_action_selection[N, A, E](
    key: Key,
    tree: Tree[N, A, E],
    node: i32,
    depth: i32,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: Callable[..., f32[A]] = qtransform_by_parent_and_siblings,
) -> i32:
    """The PUCT rule (MuZero, Appendix B), with a tiny noise to break ties."""
    visits, node_visits = tree.children_visits[node], tree.node_visits[node]
    pb_c = pb_c_init + jnp.log((node_visits + pb_c_base + 1.0) / pb_c_base)
    prior = jax.nn.softmax(tree.children_prior_logits[node])
    policy_score = jnp.sqrt(node_visits) * pb_c * prior / (visits + 1)
    noise = 1e-7 * jr.uniform(key, (A,))
    return masked_argmax(qtransform(tree, node) + policy_score + noise, tree.root_invalid_actions & (depth == 0))


@expex.fn
def gumbel_muzero_root_action_selection[N, A, E](
    key: Key,
    tree: Tree[N, A, E],
    node: i32,
    *,
    num_simulations: int,
    max_num_considered_actions: int,
    qtransform: Callable[..., f32[A]] = qtransform_completed_by_mix_value,
) -> i32:
    """Sequential Halving with Gumbel: the best of the actions with the visit count due now."""
    visits = tree.children_visits[node]
    table = jnp.array(considered_visits_table(max_num_considered_actions, num_simulations))
    considered = jnp.minimum(max_num_considered_actions, (~tree.root_invalid_actions).sum())
    score = score_considered(
        table[considered, visits.sum()], tree.root_gumbel, tree.children_prior_logits[node], qtransform(tree, node), visits
    )
    return masked_argmax(score, tree.root_invalid_actions)


@expex.fn
def gumbel_muzero_interior_action_selection[N, A, E](
    key: Key,
    tree: Tree[N, A, E],
    node: i32,
    depth: i32,
    *,
    qtransform: Callable[..., f32[A]] = qtransform_completed_by_mix_value,
) -> i32:
    """The action whose visit share lags most behind softmax(prior_logits + completed Q-values)."""
    visits = tree.children_visits[node]
    probs = jax.nn.softmax(tree.children_prior_logits[node] + qtransform(tree, node))
    return jnp.argmax(probs - visits / (1 + visits.sum())).astype(jnp.int32)


@expex.fn
def masked_argmax[A](x: f32[A], invalid: Bool[A]) -> i32:
    """The valid action with the highest `x`; action 0 if none is valid."""
    return jnp.argmax(jnp.where(invalid, -jnp.inf, x)).astype(jnp.int32)


@expex.fn
def score_considered[A](considered_visit: i32, gumbel: f32[A], logits: f32[A], q: f32[A], visits: i32[A]) -> f32[A]:
    """Gumbel + logits + Q for the actions with `considered_visit` visits, -inf for the others."""
    penalty = jnp.where(visits == considered_visit, 0.0, -jnp.inf)
    return jnp.maximum(-1e9, gumbel + logits - logits.max() + q) + penalty


def considered_visits_table(max_considered: int, simulations: int) -> tuple[tuple[int, ...], ...]:
    """For m = 0..max_considered actions: the visit count Sequential Halving considers at each simulation."""
    return tuple(considered_visits(m, simulations) for m in range(max_considered + 1))


def considered_visits(considered: int, simulations: int) -> tuple[int, ...]:
    if considered <= 1:
        return tuple(range(simulations))
    log2max = math.ceil(math.log2(considered))
    sequence: list[int] = []
    visits = [0] * considered
    while len(sequence) < simulations:
        for _ in range(max(1, int(simulations / (log2max * considered)))):
            sequence += visits[:considered]
            visits[:considered] = [v + 1 for v in visits[:considered]]
        considered = max(2, considered // 2)
    return tuple(sequence[:simulations])


# ------------------------------------------------------------------ policies


@expex.struct
class Policy[N, A, E]:
    """mctx's `PolicyOutput`: the action to take, the training target and the tree."""

    action: i32
    action_weights: f32[A]
    search_tree: Tree[N, A, E]


@expex.vectorize
def mask_invalid[A](logits: f32[A], invalid: Bool[A]) -> f32[A]:
    """Invalid actions at the lowest finite logit, so a softmax gives them no mass and no NaN."""
    logits = logits - logits.max()
    return jnp.where(invalid, -expex.huge(logits), logits)


@expex.vectorize
def logits_of[A](probs: f32[A]) -> f32[A]:
    return jnp.log(jnp.maximum(probs, expex.tiny(probs)))


@expex.vectorize
def with_temperature[A](logits: f32[A], temperature: f32) -> f32[A]:
    """`logits / temperature`, also for temperature 0."""
    return (logits - logits.max()) / jnp.maximum(expex.tiny(logits), temperature)


@expex.fn
def muzero_policy[B, A, E](
    params: Any,
    key: Key,
    root: Batch[Root[A, E], B],
    recurrent_fn: Model,
    num_simulations: int,
    invalid_actions: Bool[B, A] | None = None,
    max_depth: int | None = None,
    *,
    qtransform: Callable[..., f32[A]] = qtransform_by_parent_and_siblings,
    dirichlet_fraction: float = 0.25,
    dirichlet_alpha: float = 0.3,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    temperature: float = 1.0,
) -> Batch[Policy[num_simulations + 1, A, E], B]:
    """MuZero: Dirichlet noise at the root, PUCT, and an action drawn by visit counts."""
    key, noise_key, search_key = jr.split(key, 3)
    noise = jr.dirichlet(noise_key, jnp.full(A, dirichlet_alpha), (B,))
    probs = (1 - dirichlet_fraction) * jax.nn.softmax(root.prior_logits) + dirichlet_fraction * noise
    logits = logits_of(probs)
    if invalid_actions is not None:
        logits = mask_invalid(logits, invalid_actions)
    else:
        invalid_actions = jnp.zeros((B, A), bool)
    select = functools.partial(muzero_action_selection, pb_c_init=pb_c_init, pb_c_base=pb_c_base, qtransform=qtransform)
    trees = search.vectorize(
        params, *search_keys(search_key, num_simulations), root.replace(prior_logits=logits), recurrent_fn,
        functools.partial(select, depth=0), select, invalid_actions, jnp.zeros((B, A)), max_depth,
    )  # fmt: skip
    weights = trees.summary.vectorize().visit_probs
    action = jr.categorical(key, with_temperature(logits_of(weights), temperature))
    return Policy(action.astype(jnp.int32), weights, trees)


@expex.fn
def gumbel_muzero_policy[B, A, E](
    params: Any,
    key: Key,
    root: Batch[Root[A, E], B],
    recurrent_fn: Model,
    num_simulations: int,
    invalid_actions: Bool[B, A] | None = None,
    max_depth: int | None = None,
    *,
    qtransform: Callable[..., f32[A]] = qtransform_completed_by_mix_value,
    max_num_considered_actions: int = 16,
    gumbel_scale: float = 1.0,
) -> Batch[Policy[num_simulations + 1, A, E], B]:
    """Gumbel MuZero: Sequential Halving with Gumbel at the root, deterministic selection below."""
    logits = root.prior_logits
    if invalid_actions is not None:
        logits = mask_invalid(logits, invalid_actions)
    else:
        invalid_actions = jnp.zeros((B, A), bool)
    key, gumbel_key = jr.split(key)
    gumbel = gumbel_scale * jr.gumbel(gumbel_key, (B, A))
    select_root = functools.partial(
        gumbel_muzero_root_action_selection,
        num_simulations=num_simulations, max_num_considered_actions=max_num_considered_actions, qtransform=qtransform,
    )  # fmt: skip
    select = functools.partial(gumbel_muzero_interior_action_selection, qtransform=qtransform)
    trees = search.vectorize(
        params, *search_keys(key, num_simulations), root.replace(prior_logits=logits), recurrent_fn,
        select_root, select, invalid_actions, gumbel, max_depth,
    )  # fmt: skip
    return gumbel_decision.vectorize(trees, qtransform)


@expex.fn
def gumbel_decision[N, A, E](tree: Tree[N, A, E], qtransform: Callable[..., f32[A]]) -> Policy[N, A, E]:
    """The best of the most visited root actions, and softmax(prior_logits + completed Q) as the target."""
    visits, logits = tree.children_visits[ROOT], tree.children_prior_logits[ROOT]
    q = qtransform(tree, ROOT)
    score = score_considered(visits.max(), tree.root_gumbel, logits, q, visits)
    weights = jax.nn.softmax(mask_invalid(logits + q, tree.root_invalid_actions))
    return Policy(masked_argmax(score, tree.root_invalid_actions), weights, tree)


# ------------------------------------------------------------------ a demo


@expex.struct
class MDP[S, A]:
    """A deterministic tabular MDP; states are one-hot vectors of width S."""

    transitions: i32[S, A]
    rewards: f32[S, A]
    discounts: f32[S, A]
    logits: f32[S, A]  # a stand-in for a policy network
    values: f32[S]  # and for a value network

    def root(self, state: i32) -> Root[A, S]:
        return Root(self.logits[state], self.values[state], jax.nn.one_hot(state, S))

    def step(self, key: Key, action: i32, embedding: f32[S]) -> tuple[Step[A], f32[S]]:
        state = embedding.argmax()
        following = self.transitions[state, action]
        step = Step(self.rewards[state, action], self.discounts[state, action], self.logits[following], self.values[following])
        return step, jax.nn.one_hot(following, S)


def recurrent_fn(mdp: MDP, key: Key, action: i32, embedding: f32) -> tuple[Step, Any]:
    """mctx's interface: the model's parameters (here, the MDP) come first."""
    return mdp.step(key, action, embedding)


@expex.fn
def random_mdp(key: Key, states: int, actions: int) -> MDP[states, actions]:
    k = jr.split(key, 4)
    return MDP(
        transitions=jr.randint(k[0], (states, actions), 0, states),
        rewards=jr.normal(k[1], (states, actions)),
        discounts=jnp.full((states, actions), 0.95),
        logits=jr.normal(k[2], (states, actions)),
        values=jr.normal(k[3], (states,)),
    )


if __name__ == "__main__":
    mdp = random_mdp(jr.key(0), 12, 4)
    roots = mdp.root.vectorize(jnp.arange(8))
    for policy in (muzero_policy, gumbel_muzero_policy):
        out = jax.jit(policy, static_argnums=(3, 4))(mdp, jr.key(1), roots, recurrent_fn, 32)
        print(f"{policy.__name__}: actions {out.action}, root visits of tree 0 {out.search_tree.children_visits[0, ROOT]}")
    print("how simulate() ran:", Tree.simulate.flow)
    print("how backward() ran:", Tree.backward.flow)
    print("how search() ran:", search.flow)

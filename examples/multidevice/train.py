"""The transformer of `examples/transformer.py`, trained on a 4 x 2 mesh: data x model.

    python examples/multidevice/train.py          # everything below, on 8 simulated CPU devices

Every sharding comes from two axis names, `B="data"` and `D="model"`: the
batch is split four ways, the model width two ways, and the optimizer state,
an optax Adam state whose moments are models, follows the parameters. The
run:

1. prints the mesh and the per-device shapes of parameters and activations;
2. trains a character-level model on this file's own source, with a jitted
   step (`expex.shard(..., donate="auto")`) and optax, saving at step 10;
3. resumes from the checkpoint in a new process (`expex.load(..., mesh=)`),
   checks every restored sharding and that the losses continue exactly;
4. trains the same run on 1 device in a third process: the losses agree to
   float tolerance;
5. times a step on 1 and 8 devices, and costs the model per device.

The same script runs one configuration alone, as the processes of steps 3
and 4 (and tests/ports/test_multidevice.py) do:

    python examples/multidevice/train.py --devices 1 --steps 20 --json
    python examples/multidevice/train.py --devices 8 --steps 20 --resume ckpt.npz --json
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--devices", type=int, help="run this configuration alone, on 1 or 8 devices")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--save-at", type=int, default=10, help="write a checkpoint after this many steps")
parser.add_argument("--ckpt", help="where --save-at writes the checkpoint")
parser.add_argument("--resume", help="a checkpoint to continue from")
parser.add_argument("--json", action="store_true", help="print the results as one line of JSON")
args = parser.parse_args()

# The device count is fixed when JAX starts its CPU backend: set it before importing JAX.
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.devices or 8}"

from typing import Any  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax.sharding import AxisType, PartitionSpec as P  # noqa: E402

sys.path.insert(0, str(HERE.parent))
import expex  # noqa: E402
from expex import Int, f32  # noqa: E402
from transformer import Transformer, Vocab  # noqa: E402

CONFIG = dict(Vocab=128, D=64, F=256, blocks=2, heads=4)  # 128: the ASCII characters
BATCH, SEQ = 32, 32  # sequences per step, characters per sequence
NAMES = dict(B="data", D="model")  # the batch over "data", the model width over "model"
TEXT = np.frombuffer(pathlib.Path(__file__).read_bytes(), np.uint8).astype(np.int32)  # tiny real text
OPT = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 3e-3, 5, 200), weight_decay=0.01),
)


def make_mesh(devices: int) -> jax.sharding.Mesh:
    shape = (4, 2) if devices == 8 else (1, 1)
    return jax.make_mesh(shape, ("data", "model"), axis_types=(AxisType.Auto,) * 2, devices=jax.devices()[:devices])


def batch(step: int) -> jax.Array:
    """Batch ``step``: BATCH windows of SEQ + 1 characters, the same on any number of devices."""
    starts = np.random.default_rng(step).integers(0, len(TEXT) - SEQ - 1, BATCH)
    return jnp.asarray(np.stack([TEXT[s : s + SEQ + 1] for s in starts]))


def _tokens[B, T](tokens: Int[B, T]) -> None: ...


TOKENS = _tokens.__annotations__["tokens"]  # Int[B, T]: what places a batch over "data"


@expex.fn
def train_step[B, T](model: Transformer, opt_state, tokens: Int[B, T]) -> tuple[Transformer, Any, f32]:
    # loss is written for one sequence: called on B of them, it maps over the rows
    loss, grads = jax.value_and_grad(lambda m: m.loss(tokens).mean())(model)
    updates, opt_state = OPT.update(grads, opt_state, model)
    return optax.apply_updates(model, updates), opt_state, loss


@expex.fn
def hidden[B, T, D](model: Transformer, tokens: Int[B, T]) -> f32[B, T, D]:
    """The activations after the blocks: an [B, T, D] array to look at."""
    x = model.embed[tokens]
    for block in model.blocks:
        x = block(x)
    return x


@expex.fn
def logits[B, T](model: Transformer, tokens: Int[B, T]) -> f32[B, T, Vocab]:
    return model(tokens)


def init() -> dict[str, Any]:
    model = Transformer(jr.key(0), CONFIG)
    return {"model": model, "opt": OPT.init(model)}


def train(state: dict[str, Any], mesh, start: int, stop: int, save_at: int | None = None, ckpt: str | None = None):
    """Steps ``start`` to ``stop`` from ``state`` (which is emptied, so that its buffers can be donated).

    Returns the losses, the time of each step, how many steps reused their
    inputs' buffers, and the final state.
    """
    step = expex.shard(train_step, mesh, donate="auto", **NAMES)
    model, opt_state = state.pop("model"), state.pop("opt")
    losses, times, donated = [], [], 0
    for i in range(start, stop):
        tokens, before = batch(i), model.embed
        t0 = time.perf_counter()
        model, opt_state, loss = step(model, opt_state, tokens)  # reassigned, so donated
        loss.block_until_ready()
        times.append(time.perf_counter() - t0)
        losses.append(float(loss))
        donated += before.is_deleted()
        if ckpt and i + 1 == save_at:
            expex.save({"model": model, "opt": opt_state, "step": jnp.int32(i + 1)}, ckpt)
    return losses, times, donated, {"model": model, "opt": opt_state}


def check_shardings(state: Any, mesh) -> int:
    """Assert that every array of ``state`` is placed as ``expex.shardings`` says; the number checked."""
    want = expex.shardings(state, mesh, **NAMES)
    pairs = list(zip(jax.tree.leaves(state), jax.tree.leaves(want)))
    for x, s in pairs:
        assert x.sharding.is_equivalent_to(s, x.ndim), (x.shape, x.sharding, s)
    return len(pairs)


def run_alone() -> dict[str, Any]:
    """One configuration: ``--devices`` devices, from the start or from ``--resume``."""
    mesh = make_mesh(args.devices)
    if args.resume:
        template = jax.eval_shape(init) | {"step": jax.ShapeDtypeStruct((), jnp.int32)}
        loaded = expex.load(template, args.resume, mesh=mesh, names=NAMES)  # each array onto its devices
        start, state = int(loaded.pop("step")), loaded
    else:
        start, state = 0, expex.device_put(init(), mesh, **NAMES)
    checked = check_shardings(state, mesh)
    losses, times, donated, state = train(state, mesh, start, args.steps, args.save_at, args.ckpt)
    check_shardings(state, mesh)
    return {"devices": args.devices, "start": start, "losses": losses, "times": times, "checked": checked, "donated": donated}


def subprocess_run(*flags: str) -> dict[str, Any]:
    out = subprocess.run(
        [sys.executable, __file__, "--json", *flags], capture_output=True, text=True, check=True
    ).stdout  # fmt: skip
    return json.loads(out.strip().splitlines()[-1])


def per_device(x: jax.Array) -> tuple[int, ...]:
    return x.sharding.shard_shape(x.shape)


def show(name: str, x: jax.Array, want: tuple[int, ...]) -> None:
    shards = per_device(x)
    print(f"  {name:<26} {str(x.shape):<14} {str(x.sharding.spec):<26} {str(shards):<13}")
    assert shards == want, (name, shards, want)


def main() -> None:
    mesh = make_mesh(8)
    print(f"mesh: {dict(mesh.shape)} over {len(mesh.devices.flat)} {jax.devices()[0].platform} devices")
    print(f"names: {NAMES}\n")

    # 1. Shardings, read off the annotations: no PartitionSpec written anywhere.
    state = expex.device_put(init(), mesh, **NAMES)
    model, tokens = state["model"], batch(0)
    D, F, V = CONFIG["D"], CONFIG["F"], CONFIG["Vocab"]
    print(f"  {'array':<26} {'shape':<14} {'spec':<26} {'per device':<13}")
    show("embed  f32[Vocab, D]", model.embed, (V, D // 2))
    show("blocks[0].attn.wq [D,D]", model.blocks[0].attn.wq, (D // 2, D))
    show("blocks[0].mlp.w1 [D,F]", model.blocks[0].mlp.w1, (D // 2, F))
    show("blocks[0].mlp.w2 [F,D]", model.blocks[0].mlp.w2, (F, D // 2))
    show("norm.scale  f32[D]", model.norm.scale, (D // 2,))
    adam = state["opt"][1][0]  # chain(clip, adamw): adamw's ScaleByAdamState
    show("opt mu.blocks[0].attn.wq", adam.mu.blocks[0].attn.wq, (D // 2, D))
    show("opt count (replicated)", adam.count, ())
    tokens = expex.device_put(tokens, mesh, annotation=TOKENS, **NAMES)
    show("tokens  Int[B, T]", tokens, (BATCH // 4, SEQ + 1))
    h = expex.shard(hidden, mesh, **NAMES)(model, tokens)
    show("hidden  f32[B, T, D]", h, (BATCH // 4, SEQ + 1, D // 2))
    out = expex.shard(logits, mesh, **NAMES)(model, tokens)
    show("logits  f32[B, T, Vocab]", out, (BATCH // 4, SEQ + 1, V))
    try:
        print("\nblocks[0].attn.wq, split by rows over 'model' (2 x 4 devices hold each half):")
        jax.debug.visualize_array_sharding(model.blocks[0].attn.wq)
        print("tokens [B, T], split by rows over 'data':")
        jax.debug.visualize_array_sharding(tokens)
    except ImportError:  # it draws with `rich`
        pass

    # 2. Train on 8 devices, writing a checkpoint at --save-at.
    steps, k = args.steps, args.save_at
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "ckpt.npz")
        del model, adam
        losses, times, donated, state = train(state, mesh, 0, steps, k, ckpt)
        assert donated == steps, "every step should reuse the model's and optimizer's buffers"
        n = check_shardings(state, mesh)
        print(f"\ntrained {steps} steps on 8 devices, each reusing its inputs' buffers (donate='auto'): {donated}/{steps}")
        print(f"after training, {n} arrays (parameters and optimizer state) are where expex.shardings says")
        print(f"checkpoint at step {k}: {os.path.getsize(ckpt) / 1e6:.2f} MB, keys like",
              ", ".join(sorted(np.load(ckpt).files)[:3]))  # fmt: skip

        # 3. Resume in a new process; 4. the same run on one device.
        resumed = subprocess_run("--devices", "8", "--steps", str(steps), "--resume", ckpt)
    single = subprocess_run("--devices", "1", "--steps", str(steps))

    print(f"\n{'step':>4}  {'8 devices':>10}  {'resumed (new process)':>22}  {'1 device':>10}  {'|8 - 1|':>8}")
    for i in range(steps):
        r = resumed["losses"][i - k] if i >= k else None
        mark = "" if r is None else ("  exact" if r == losses[i] else "  DIFFERENT")
        rs = f"{r:.6f}{mark}" if r is not None else ""
        print(f"{i:>4}  {losses[i]:>10.6f}  {rs:>22}  {single['losses'][i]:>10.6f}  {abs(losses[i] - single['losses'][i]):>8.1e}")
    print(f"resumed at step {resumed['start']}: {resumed['checked']} arrays loaded onto the mesh with their shardings")
    assert resumed["start"] == k and resumed["losses"] == losses[k:], "resume should continue exactly"
    gap = max(abs(a - b) / abs(b) for a, b in zip(losses, single["losses"]))
    print(f"8 devices vs 1: largest relative difference {gap:.1e}")
    assert gap < 1e-4, gap

    # 5. Speed and cost.
    t8, t1 = np.median(times[1:]) * 1e3, np.median(single["times"][1:]) * 1e3
    print(f"\nstep time (median, compile excluded): 1 device {t1:.1f} ms, 8 simulated devices {t8:.1f} ms")
    print(f"  ({os.cpu_count()} CPU cores: the 8 'devices' share them, so this is no speedup; the point is that it runs)")

    # expex.cost and expex.summary count the whole model (they read annotations, not shardings);
    # per device: the bytes each device holds, and the FLOPs of XLA's partitioned program.
    print()
    print(expex.summary(state["model"], T=SEQ))
    cost = expex.cost(Transformer.loss, CONFIG, T=SEQ + 1)
    print()
    print(cost)
    total = sum(x.nbytes for x in jax.tree.leaves(state))
    held = sum(x.addressable_shards[0].data.nbytes for x in jax.tree.leaves(state))
    print(f"\nparameters + optimizer state: {total / 1e6:.2f} MB in all, {held / 1e6:.2f} MB on each device"
          " ('model' halves it, 'data' replicates it)")  # fmt: skip
    tokens = expex.device_put(batch(0), mesh, annotation=TOKENS, **NAMES)
    compiled = jax.jit(train_step).lower(state["model"], state["opt"], tokens).compile()
    xla = compiled.cost_analysis()["flops"]
    estimate = 3 * BATCH * float(cost.flops) / 8  # forward + backward ~ 3 forwards, over 8 devices
    print(f"one training step per device: {estimate / 1e6:.1f} MFLOP from expex.cost "
          f"(3 x {BATCH} sequences / 8 devices), {xla / 1e6:.1f} MFLOP in XLA's partitioned program")  # fmt: skip


if __name__ == "__main__":
    if args.devices is None:
        main()
    else:
        result = run_alone()
        if args.json:
            print(json.dumps(result))
        else:
            for i, loss in enumerate(result["losses"], result["start"]):
                print(f"step {i}: loss {loss:.6f}")

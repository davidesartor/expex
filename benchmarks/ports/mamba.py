"""mamba-jax (Equinox) vs examples/mamba: forward, gradient and sampling, same parameters.

The original's parameters come from its own constructor; the port gets them
through examples/mamba/compare.py's `from_reference`, and the original's import
stubs come from examples/mamba/run_original.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_example, use_original

NAME = "mamba"
ORIGINALS = ["mamba-jax"]
CLAIM = {"forward": "4.5x", "gradient": "3x"}  # README: 2048 tokens, CPU (width 128, 4 layers, vocab 1000)

CONFIGS = {
    "quick": dict(dim=64, layers=2, vocab=256, T=256, prompt=8, new=16),
    "readme": dict(dim=128, layers=4, vocab=1000, T=2048, prompt=16, new=64),
    # mamba-130m's shape: width 768, 24 layers, GPT-NeoX vocabulary
    "realistic": dict(dim=768, layers=24, vocab=50280, T=2048, prompt=128, new=256),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["readme"], "gpu": ["readme", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    use_original("mamba-jax")
    use_example("mamba")
    ro = load_module("mamba_run_original", EXAMPLES / "mamba" / "run_original.py")  # stubs, MambaLLM, KernelType
    compare = load_module("mamba_compare", EXAMPLES / "mamba" / "compare.py")
    import equinox as eqx

    orig = ro.MambaLLM(
        dim=cfg["dim"], num_layers=cfg["layers"], vocab_size=cfg["vocab"], state_dim=16, kernel_size=4, expand=2,
        kernel_mode=ro.KernelType.XLA, key=jax.random.PRNGKey(0),
    )  # fmt: skip
    ref = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(orig)[0]:
        if isinstance(leaf, jax.Array):
            ref["param" + jax.tree_util.keystr(path)] = np.asarray(leaf)
    port = compare.from_reference(ref)
    params, static = eqx.partition(orig, eqx.is_array)
    tokens = jax.random.randint(jax.random.PRNGKey(1), (cfg["T"],), 0, cfg["vocab"])
    prompt, new, key = tokens[: cfg["prompt"]], cfg["new"], jax.random.PRNGKey(2)

    def loss_original(p, t):  # as in run_original.py
        lp = jax.nn.log_softmax(eqx.combine(p, static)(t[:-1]))
        return -jnp.take_along_axis(lp, t[1:, None], axis=1).mean()

    def max_diff(a, b):
        return {"max abs diff": float(jnp.abs(a - b).max())}

    def loss_diff(a, b):
        return {"loss abs diff": abs(float(a[0]) - float(b[0]))}

    def tokens_differ(a, b):
        return {"tokens differing": int((np.asarray(a) != np.asarray(b)).sum()), "of": int(new)}

    T = cfg["T"]
    return [
        Measure(
            "forward",
            Fn(jax.jit(lambda p, t: eqx.combine(p, static)(t)), (params, tokens)),
            Fn(jax.jit(lambda m, t: m(t)), (port, tokens)),
            items=T, unit="tokens", check=max_diff,
        ),
        Measure(
            "gradient",
            Fn(jax.jit(jax.value_and_grad(loss_original)), (params, tokens)),
            Fn(jax.jit(jax.value_and_grad(lambda m, t: m.loss(t))), (port, tokens)),
            items=T - 1, unit="tokens", check=loss_diff,
        ),
        Measure(
            "generate",
            Fn(jax.jit(lambda p, t, k: eqx.combine(p, static).generate(t, gen_len=new, key=k)), (params, prompt, key)),
            Fn(jax.jit(lambda m, t, k: m.generate(t, new, k)), (port, prompt, key)),
            items=new, unit="new tokens", check=tokens_differ,
            note=f"prompt {cfg['prompt']}, {new} sampled tokens, batch 1",
        ),
    ]

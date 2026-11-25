"""llama-2-jax vs examples/llama: forward, gradient and greedy generation with the KV cache, same parameters.

The original's forward pass and its stand-in tokenizer come from
examples/llama/run_original.py; the port gets the original's parameters through
examples/llama/compare.py's `from_reference`. Generation is the original's own
`lib.generation.generate` (two jitted stages and a little host work, as a user
calls it) against the port's `generate`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from harness import EXAMPLES, Fn, Measure, load_module, use_example, use_original

NAME = "llama"
ORIGINALS = ["llama-2-jax"]
# README: 4 layers, width 256, 8 query and 2 key/value heads, batch 4, 256 tokens, CPU
CLAIM = {"forward": "1.08x (1.14x)", "gradient": "1.61x (1.71x)", "generate": "1.48x (1.58x)"}  # README, from results/cpu-2026-11-25-v0.1.md: best (median)

_SMALL = dict(V=256, D=64, F=172, G=2, R=2, K=16, L=2, B=2, T=64, prompt=8, max_len=24)
CONFIGS = {
    "quick": _SMALL,
    # the README's: the vocabulary was not recorded; 32,000 is Llama 2's
    "readme": dict(V=32000, D=256, F=688, G=2, R=4, K=32, L=4, B=4, T=256, prompt=64, max_len=128),
    "realistic": dict(V=32000, D=1024, F=2816, G=4, R=4, K=64, L=8, B=4, T=2048, prompt=128, max_len=384),
}
DEFAULTS = {"quick": ["quick"], "cpu": ["readme"], "gpu": ["readme", "realistic"]}


def setup(cfg: dict) -> list[Measure]:
    use_original("llama-2-jax")
    use_example("llama")
    ro = load_module("llama_run_original", EXAMPLES / "llama" / "run_original.py")  # stubs, forward, Tokens, greedy
    compare = load_module("llama_compare", EXAMPLES / "llama" / "compare.py")
    import lib.generation

    V, D, F, G, R, K, L = (cfg[k] for k in "VDFGRKL")
    B, T, P, max_len = cfg["B"], cfg["T"], cfg["prompt"], cfg["max_len"]
    config = ro.ModelConfig(
        d_ff=F, d_k=K, d_model=D, d_v=K, n_heads_kv=G, n_layers=L, n_rep_kv=R, rms_norm_eps=1e-6,
        token_id_bos=1, token_id_eos=2, token_id_pad=0, vocab_size=V, dropout_rate=None, return_kv_cache=False,
    )  # fmt: skip
    ro.CONFIG = config  # run_original.forward reads it
    lib.generation.model_config_llama2_7B = config  # the original's generation hard-codes the 7B config

    params = ro.init_llama(key=jax.random.key(0), model_config=config)
    flat = {f"param.{jax.tree_util.keystr(p, simple=True, separator='.')}": np.asarray(x)
            for p, x in jax.tree_util.tree_flatten_with_path(params)[0]}  # fmt: skip
    model = compare.from_reference(flat)
    tokens = jax.random.randint(jax.random.key(1), (B, T), 3, V).astype(jnp.uint16)  # the original's token type
    tokens32 = tokens.astype(jnp.int32)

    def loss_original(p, t):  # as in run_original.py
        lp = jax.nn.log_softmax(ro.forward(p, t[:, :-1]))
        return -jnp.take_along_axis(lp, t[:, 1:, None].astype(jnp.int32), axis=2).mean()

    new = max_len - P + 1  # the original fills the left padding, plus one token
    prompts = np.asarray(tokens[:, :P]).tolist()

    def generate_original(p):
        return lib.generation.generate(prompts, ro.Tokens(), p, ro.greedy, max_len=max_len, key=jax.random.key(2))

    def max_diff(a, b):
        return {"max abs diff": float(jnp.abs(a - b).max())}

    def loss_diff(a, b):
        return {"loss abs diff": abs(float(a[0]) - float(b[0]))}

    def tokens_differ(a, b):
        got_a, got_b = np.asarray(a)[:, P - 1 :], np.asarray(b)[:, P:]  # the original returns the last max_len
        return {"tokens differing": int((got_a != got_b).sum()), "of": int(got_a.size)}

    return [
        Measure(
            "forward",
            Fn(jax.jit(ro.forward), (params, tokens)),
            Fn(jax.jit(lambda m, t: m.vectorize(t)), (model, tokens32)),
            items=B * T, unit="tokens", check=max_diff, note=f"batch {B} x {T}",
        ),
        Measure(
            "gradient",
            Fn(jax.jit(jax.value_and_grad(loss_original)), (params, tokens)),
            Fn(jax.jit(jax.value_and_grad(lambda m, t: m.loss.vectorize(t).mean())), (model, tokens32)),
            items=B * (T - 1), unit="tokens", check=loss_diff,
        ),
        Measure(
            "generate",
            Fn(generate_original, (params,)),
            Fn(jax.jit(lambda m, p: m.generate(p, new)), (model, tokens32[:, :P])),
            items=B * new, unit="new tokens", check=tokens_differ,
            note=f"batch {B}, prompt {P}, {new} greedy tokens",
        ),
    ]

"""Run the original llama-2-jax (github.com/ayaka14732/llama-2-jax) and save reference outputs.

    git clone https://github.com/ayaka14732/llama-2-jax /tmp/llama-2-jax    # 6bb1905, January 2024
    pip install einops
    PYTHONPATH=/tmp/llama-2-jax python examples/llama/run_original.py tests/data/llama_reference.npz

Saves the parameters of a small randomly initialised model, its logits on two
token sequences, the gradients of a loss, and the tokens its own `generate`
picks greedily with the KV cache, for tests/ports/test_llama.py.
"""

import sys
import types

import jax
import jax.numpy as jnp
import numpy as np

jax.tree_map = jax.tree.map  # the original (January 2024) uses the old name, removed since

# lib.generation imports the tokenizer class at import time; generation itself
# only calls the tokenizer object it is given (a stand-in below).
_transformers = types.ModuleType("transformers")
_transformers.LlamaTokenizer = object  # type: ignore[attr-defined]
sys.modules.setdefault("transformers", _transformers)

import lib.generation
from lib.llama import ModelConfig, forward_llama, init_llama, make_rotary_values

CONFIG = ModelConfig(
    d_ff=48, d_k=8, d_model=32, d_v=8, n_heads_kv=2, n_layers=2, n_rep_kv=2, rms_norm_eps=1e-6,
    token_id_bos=1, token_id_eos=2, token_id_pad=0, vocab_size=64, dropout_rate=None, return_kv_cache=False,
)  # fmt: skip
PROMPT, MAX_LEN = 6, 16  # generation: prompts of 6 tokens, left-padded to 16: 11 new tokens


def forward(params, tokens):
    """The original's forward pass on unpadded sequences: a causal mask, positions 0..T-1."""
    B, T = tokens.shape
    mask = jnp.tril(jnp.ones((B, 1, 1, T, T), bool))
    rotary = make_rotary_values(None, B, T, model_config=CONFIG)
    logits, _ = forward_llama(params, tokens, mask, rotary_values=rotary, model_config=CONFIG)
    return logits


class Tokens:
    """A stand-in tokenizer for `lib.generation.generate`: the 'sentences' are token ids already."""

    def __call__(self, rows, *, padding, truncation, max_length, return_tensors):
        pad = max_length - len(rows[0])
        ids = jnp.asarray([[CONFIG.token_id_pad] * pad + list(r) for r in rows])
        return types.SimpleNamespace(input_ids=ids, attention_mask=jnp.broadcast_to(jnp.arange(max_length) >= pad, ids.shape))

    def batch_decode(self, seq, skip_special_tokens):
        return seq


def greedy(logits, **_):
    return jnp.argmax(logits, axis=-1).astype(jnp.uint16)  # token ids are uint16 there


def main(out: str) -> None:
    params = init_llama(key=jax.random.key(0), model_config=CONFIG)
    tokens = jax.random.randint(jax.random.key(1), (2, 12), 0, CONFIG.vocab_size).astype(jnp.uint16)
    logits = forward(params, tokens)

    def loss(p):
        lp = jax.nn.log_softmax(forward(p, tokens[:, :-1]))
        return -jnp.take_along_axis(lp, tokens[:, 1:, None].astype(jnp.int32), axis=2).mean()

    value, grads = jax.jit(jax.value_and_grad(loss))(params)

    # its own generation loop, with the model's config in place of the 7B one it hard-codes
    lib.generation.model_config_llama2_7B = CONFIG
    prompts = np.asarray(tokens[:, :PROMPT]).tolist()
    seq = lib.generation.generate(prompts, Tokens(), params, greedy, max_len=MAX_LEN, key=jax.random.key(2))
    generated = np.asarray(seq)[:, PROMPT - 1 :]  # seq is the last MAX_LEN tokens: prompt[1:] + new

    arrays = {}
    for prefix, tree in (("param", params), ("grad", grads)):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            arrays[f"{prefix}.{jax.tree_util.keystr(path, simple=True, separator='.')}"] = np.asarray(leaf)
    np.savez(
        out, tokens=np.asarray(tokens), logits=np.asarray(logits), loss=np.asarray(value),
        generated=generated, **arrays,
    )  # fmt: skip
    print(f"saved {len(arrays)} arrays to {out}; loss {float(value):.6f}; generated {generated.shape}")


if __name__ == "__main__":
    main(sys.argv[1])

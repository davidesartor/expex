# Notebooks

One notebook per ported project, plus a quickstart. Each installs expex, shows the port's
code and its annotated signatures, runs it at a small size (well under a minute on a CPU),
uses expex's tools on it (`prove`, `summary`, `cost`, `diagram`, `inspect`, a deliberate
shape error, `debug.watch`), checks it against the original's saved outputs in
[`tests/data/`](../../tests/data/), and ends with a short benchmark.

| Notebook | What it shows | |
|---|---|---|
| [quickstart](quickstart.ipynb) | The README's examples: checked shapes, loops and branches, modules, proofs, cost, `.at` edits, summary, diagrams (plain, timed, and the one a `ShapeError` carries), `inspect`, `find_nan`, `watch`, `annotate`; compiled-call speed against `jax.jit` and plain JAX | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/quickstart.ipynb) |
| [llama](llama.ipynb) | Llama 2 (llama-2-jax): training steps, KV-cache generation, a 7B-sized cost, a violated `T <= S` constraint, a planted exploding layer found by `watch`; logits to 7e-7, tokens exact | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/llama.ipynb) |
| [mamba](mamba.ipynb) | Mamba (mamba-jax): the selective scan as a `for` loop staged as `lax.scan`, the mixer's diagram, a checked `.at` edit; logits to 6e-7, sampled tokens exact | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/mamba.ipynb) |
| [vit](vit.ipynb) | The Vision Transformer: patches by axis names, the image size in the signature (a wrong image size refused), ViT-B/16's cost; logits to 5e-7 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/vit.ipynb) |
| [nerf](nerf.ipynb) | NeRF (JaxNeRF): rendering written for one ray and mapped over a camera, hierarchical sampling, the MLP's diagram; pixels to 2e-7, loss exact | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/nerf.ipynb) |
| [ppo](ppo.ipynb) | PPO on CartPole (PureJaxRL): 100 updates in one jitted call, every loop staged as `lax.scan`; the original's 3-update run reproduced exactly | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/ppo.ipynb) |
| [pgx](pgx.ipynb) | Connect Four and Othello (Pgx): game rules in plain Python (`while`, `if`, `board[r, c] = ...`), 1000 random games; every state identical to Pgx's | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/pgx.ipynb) |
| [mctx](mctx.ipynb) | MuZero and Gumbel MuZero search (mctx), written for one tree and mapped over a batch; every tree bit-identical to mctx's | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/mctx.ipynb) |
| [multidevice](multidevice.ipynb) | The transformer trained on a 4 x 2 mesh of 8 simulated CPU devices, shardings read off the annotations; losses match a 1-device run | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davidesartor/expex/blob/main/examples/notebooks/multidevice.ipynb) |

## Python 3.14

expex needs Python 3.14 or newer. Colab's hosted runtimes ship an older Python (3.12 in the
2026.01 runtime), so on a hosted runtime the setup cell stops with a message. Check
*Runtime → Change runtime type*: if it offers a runtime version with Python 3.14, pick it.
Otherwise, use one of these:

### Running in Colab

Colab can run a notebook on a Jupyter server on your own machine, keeping its interface:

```
# on your machine, with Python 3.14
python3.14 -m venv ~/expex-env && source ~/expex-env/bin/activate
pip install jupyter "git+https://github.com/davidesartor/expex@main"
jupyter notebook --ServerApp.allow_origin='https://colab.research.google.com' --port=8888 --ServerApp.port_retries=0
```

Then open a notebook with its badge above, and choose *Connect → Connect to a local runtime*,
pasting the URL (with its token) that Jupyter printed.

### Running locally

```
git clone -b main https://github.com/davidesartor/expex && cd expex
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e . jupyter optax
jupyter notebook examples/notebooks/
```

Opened from a checkout, the setup cell uses it (it looks for `src/expex` above the working
directory, or `$EXPEX_REPO`) instead of installing from GitHub.

## How the notebooks are made

The `.py` files here are the source of truth: cells separated by `# %%` lines, in Jupytext's
"percent" format. [`build.py`](build.py) writes the `.ipynb` files from them and runs them:

```
python examples/notebooks/build.py --execute            # all of them, with outputs (needs nbclient, ipykernel, playwright)
python examples/notebooks/build.py --execute llama      # one
python examples/notebooks/build.py --check              # are the .ipynb files up to date?
```

The committed outputs come from a run on a CPU. GitHub does not render inline pages with
scripts, so in those outputs the interactive diagram and `expex.inspect` pages are PNG images
(drawn by a headless browser) and the other expex displays their text. Running a notebook gives
the interactive versions. `tests/test_notebooks.py` runs every notebook's code as a script and
checks that each `.ipynb` matches its source.

The install cells and the badges point to the branch `main`: change `BRANCH` in
`build.py` (and the links above) when it moves.

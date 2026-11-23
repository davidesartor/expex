"""examples/multidevice/train.py: the transformer trained on a 4 x 2 data x model mesh of 8 CPU devices.

The device count is fixed when JAX starts, so each run is its own process
(the script sets ``--xla_force_host_platform_device_count`` itself): 8 devices
writing a checkpoint, 1 device, and 8 devices resuming from the checkpoint.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("optax")  # the example trains with optax (a dev dependency)

ROOT = pathlib.Path(__file__).parent.parent.parent
SCRIPT = ROOT / "examples" / "multidevice" / "train.py"
STEPS, SAVE_AT = 6, 3


def start(*flags: str) -> subprocess.Popen[str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    env.pop("XLA_FLAGS", None)
    cmd = [sys.executable, str(SCRIPT), "--json", "--steps", str(STEPS), *flags]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def result(p: subprocess.Popen[str]) -> dict:
    out, err = p.communicate(timeout=300)
    assert p.returncode == 0, out + err
    return json.loads(out.strip().splitlines()[-1])


def test_one_device_matches_eight_and_resume_is_exact(tmp_path):
    ckpt = str(tmp_path / "ckpt.npz")
    eight = start("--devices", "8", "--save-at", str(SAVE_AT), "--ckpt", ckpt)
    one = start("--devices", "1")
    eight, one = result(eight), result(one)
    resumed = result(start("--devices", "8", "--resume", ckpt))

    assert len(eight["losses"]) == len(one["losses"]) == STEPS
    for a, b in zip(eight["losses"], one["losses"]):
        assert abs(a - b) <= 1e-5 * abs(b), (eight["losses"], one["losses"])
    assert eight["losses"][-1] < eight["losses"][0]  # it learns

    assert resumed["start"] == SAVE_AT
    assert resumed["losses"] == eight["losses"][SAVE_AT:]  # bit for bit
    # every parameter and optimizer array is placed as expex.shardings says, after the load and at the end
    assert resumed["checked"] == eight["checked"] > 50
    # each step reused its inputs' buffers (donate="auto")
    assert eight["donated"] == STEPS and resumed["donated"] == STEPS - SAVE_AT

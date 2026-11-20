"""The bofus port in examples/bofus matches the original implementation.

``tests/data/bofus_reference.npz`` was produced by the unmodified original
(davidesartor/bofus@adapt_rho) with ``examples/bofus/parity.py reference``.
This test runs the quick subset; the script runs everything, including the
acquisition optimizer.
"""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("vlse")
ROOT = pathlib.Path(__file__).parent.parent
PORT = ROOT / "examples" / "bofus"


def test_values_and_gradients_match_the_original():
    import subprocess

    # float64 must be enabled before JAX is used, so this runs in its own process
    code = (
        "import sys, warnings; warnings.simplefilter('ignore'); sys.path.insert(0, %r)\n"
        "import parity, numpy as np\n"
        "ref = np.load(%r)\n"
        "rows = parity.compare(parity.cases(%r, quick=True), ref)\n"
        "bad = [r for r in rows if not r[2] or r[1] > 1e-9]\n"
        "print('ROWS', len(rows)); print('BAD', bad)\n"
    ) % (str(PORT), str(ROOT / "tests" / "data" / "bofus_reference.npz"), str(PORT))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert "BAD []" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
    n = int(r.stdout.split("ROWS")[1].split()[0])
    assert n >= 90


def test_port_has_no_hand_sanitised_branches_or_other_frameworks():
    source = "".join(p.read_text() for p in (PORT / "bofus").glob("*.py"))
    assert "equinox" not in source and "jaxtyping" not in source
    # the original has 42 `jnp.where` calls; the ones left here select data (masks)
    assert source.count("jnp.where(") <= 13

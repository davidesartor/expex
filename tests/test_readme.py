"""Every Python block in the README must run."""

import pathlib
import re
import runpy

ROOT = pathlib.Path(__file__).parent.parent


def test_readme_blocks_run(tmp_path, capsys):
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    assert len(blocks) >= 10
    # written to a real file so that @fn can read the source it converts
    path = tmp_path / "readme_blocks.py"
    path.write_text("\n\n".join(blocks))
    ns = runpy.run_path(str(path))
    out = capsys.readouterr().out
    assert "B is 2 (from `q: f32[2, 3, 4]`), but axis 0 of `k` is 5." in out
    assert "axis 0 of `return` should be T, but is T - 1." in out
    assert "[  8  16 111]" in out
    assert "for -> lax.scan (carries h)" in out
    assert "NaN first appears at" in out
    assert "x: f32[3]  ->  f32[4]" in out
    assert ns["run"].traces == 1

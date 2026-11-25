"""mkdocs hooks for the docs site (see mkdocs.yml).

- The front page is the repository's README.md, the guide's index is docs/README.md
  (served as guide.md), and CHANGELOG.md is a page too, so that each is written once.
- Links are written for GitHub, relative to the file they are in. Those to a page or an
  image of the site are pointed at it; those to anything else in the repository (examples/,
  benchmarks/, ...) at the file on GitHub. Code is left alone.
- A fence like ```python optax (a block the tests skip without optax) is shown as ```python.
"""

import posixpath
import re
from pathlib import Path

from mkdocs.structure.files import File

ROOT = Path(__file__).parent
GITHUB = "https://github.com/davidesartor/expex/blob/main/"

# site page -> the repository file it comes from
SOURCES = {"index.md": "README.md", "guide.md": "docs/README.md", "changelog.md": "CHANGELOG.md"}
PAGES = {source: page for page, source in SOURCES.items()}

LINK = re.compile(r"(\]\(|(?:src|srcset|href)=\")([^)\"\s]+)")
CODE = re.compile(r"(^```.*?^```|`[^`\n]*`)", re.S | re.M)
FENCE = re.compile(r"^```python [\w ]+$", re.M)


def on_files(files, config):
    readme = files.get_file_from_path("README.md")
    if readme is not None:
        files.remove(readme)
    for page, source in SOURCES.items():
        files.append(File.generated(config, page, content=(ROOT / source).read_text(encoding="utf-8")))
    return files


def _target(link: str, source: str) -> str:
    if re.match(r"^[a-z]+:|^#|^/", link):
        return link
    path, _, anchor = link.partition("#")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if resolved in PAGES:
        site = PAGES[resolved]
    elif resolved.startswith("docs/") and not resolved.endswith("/"):
        site = resolved.removeprefix("docs/")
    else:
        site = GITHUB + resolved + ("/" if path.endswith("/") else "")
    return site + ("#" + anchor if anchor else "")


def on_page_markdown(markdown, page, config, files):
    source = SOURCES.get(page.file.src_uri, "docs/" + page.file.src_uri)
    parts = CODE.split(FENCE.sub("```python", markdown))
    for i in range(0, len(parts), 2):  # the text between code spans and blocks
        parts[i] = LINK.sub(lambda m: m.group(1) + _target(m.group(2), source), parts[i])
    return "".join(parts)

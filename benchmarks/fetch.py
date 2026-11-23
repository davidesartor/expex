"""Clone the original projects pinned in originals.toml into benchmarks/.originals/.

    python benchmarks/fetch.py                 # all of them
    python benchmarks/fetch.py --only mamba,llama   # those the named ports need (or names from the toml)
    python benchmarks/fetch.py --list          # what is pinned, and whether it is fetched

Each clone is a shallow fetch of exactly the pinned commit (GitHub serves
commits by hash), checked out detached; a clone already at its commit is left
alone. Big repositories with `sparse = [...]` check out only those directories.
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tomllib

HERE = pathlib.Path(__file__).resolve().parent
ORIGINALS = HERE / ".originals"


def pins() -> dict[str, dict]:
    with open(HERE / "originals.toml", "rb") as f:
        return tomllib.load(f)


def location(name: str) -> pathlib.Path:
    """Where the original `name` is cloned."""
    return ORIGINALS / name


def import_path(name: str) -> pathlib.Path:
    """What goes on sys.path to import the original `name`."""
    pin = pins()[name]
    return location(name) / pin.get("path", "")


def head(path: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def git(*args: str, cwd: pathlib.Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch(name: str, pin: dict) -> None:
    dest = location(name)
    if head(dest) == pin["commit"]:
        print(f"{name}: at {pin['commit'][:9]} already")
        return
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print(f"{name}: fetching {pin['url']} @ {pin['commit'][:9]}")
    git("init", "-q", cwd=dest)
    git("remote", "add", "origin", pin["url"], cwd=dest)
    fetch_args = ["fetch", "-q", "--depth", "1"]
    if pin.get("sparse"):
        fetch_args.append("--filter=blob:none")
        git("sparse-checkout", "set", *pin["sparse"], cwd=dest)
    git(*fetch_args, "origin", pin["commit"], cwd=dest)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=dest)
    got = head(dest)
    if got != pin["commit"]:
        sys.exit(f"{name}: checked out {got}, expected {pin['commit']}")


def selected(only: str | None) -> dict[str, dict]:
    all_pins = pins()
    if not only:
        return all_pins
    wanted = set(only.split(","))
    out = {n: p for n, p in all_pins.items() if n in wanted or wanted & set(p.get("ports", []))}
    unknown = wanted - set(out) - {port for p in out.values() for port in p.get("ports", [])}
    if unknown:
        sys.exit(f"unknown originals or ports: {', '.join(sorted(unknown))}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="comma-separated port names (ppo, mamba, ...) or original names")
    parser.add_argument("--list", action="store_true", help="show the pins and what is fetched")
    args = parser.parse_args()
    chosen = selected(args.only)
    if args.list:
        for name, pin in chosen.items():
            got = head(location(name))
            state = "fetched" if got == pin["commit"] else ("other commit" if got else "missing")
            print(f"{name:20s} {pin['commit'][:9]}  {state:12s} ports: {', '.join(pin.get('ports', []))}  {pin['url']}")
        return
    for name, pin in chosen.items():
        fetch(name, pin)


if __name__ == "__main__":
    main()

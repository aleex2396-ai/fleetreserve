"""List files that can change how pytest, ruff or uv behave, other than the two known ones.

Walks the project folder (or the folder given as the one argument) and finds every file whose
name pytest, ruff or uv read as settings, that pytest loads as plugin code (`conftest.py`), or
that ruff reads as an ignore list (`.ignore`). Only `pyproject.toml` and `tests/conftest.py` at
exactly those paths are allowed. Each other one is printed, one path per line, and the exit
status is 1; with none, one line is printed and the exit status is 0. If the exclude file exists
and cannot be read, the exit status is 2.

Not looked at: the top-level folders listed in TOP_LEVEL_SKIPPED (version control, the
environment, data, run output, earlier material), the top-level folders that the checkout's own
`.git/info/exclude` lists by name (read by `read_local_private` in `repo_rules.py`, the same
reading the repository rules use), and tool cache folders at any depth. A folder of the same name
deeper down is still walked, because pytest and ruff read it.
"""

import importlib.util
import os
import sys
from pathlib import Path

SETTINGS_NAMES = {
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "tox.ini",
    "setup.cfg",
    "ruff.toml",
    ".ruff.toml",
    "uv.toml",
    "pyproject.toml",
    ".ignore",
}
ALLOWED = {"pyproject.toml", "tests/conftest.py"}
TOP_LEVEL_SKIPPED = {".git", ".venv", "data", "runs.noindex", "reference"}
CACHE_FOLDERS = {".ruff_cache", ".pytest_cache", "__pycache__"}


class CouldNotRun(Exception):
    pass


def local_folders(root: Path) -> set[str]:
    """Top-level folder names the checkout's exclude file lists (loaded from the sibling file)."""
    path = Path(__file__).resolve().with_name("repo_rules.py")
    spec = importlib.util.spec_from_file_location("repo_rules", path)
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    try:
        local = rules.read_local_private(root)
    except rules.CouldNotRun as error:
        raise CouldNotRun(str(error)) from error
    return set(local.anything | local.folders)


def stray_files(root: Path) -> list[str]:
    found = []
    top_skipped = TOP_LEVEL_SKIPPED | local_folders(root)
    for folder, subfolders, files in os.walk(root):
        skipped = CACHE_FOLDERS | (top_skipped if Path(folder) == root else set())
        subfolders[:] = sorted(name for name in subfolders if name not in skipped)
        for name in sorted(files):
            if name in SETTINGS_NAMES:
                relative = (Path(folder) / name).relative_to(root).as_posix()
                if relative not in ALLOWED:
                    found.append(relative)
    return found


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1]
    try:
        stray = stray_files(root)
    except CouldNotRun as error:
        print(f"COULD NOT RUN: {error}")
        return 2
    if not stray:
        print("no settings files other than pyproject.toml and tests/conftest.py")
        return 0
    print("\n".join(stray))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

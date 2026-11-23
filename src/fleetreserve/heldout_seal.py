"""Where sealed held-out files live, and the bytes-only operations allowed on them before the tag.

Held-out files are kept unparsed under `data/heldout/<set>/<original name>` (plan section 3,
"Seal rule"). This module names that folder, lists the names of the files there and hashes a
file's bytes. Hashing (SHA-256 and MD5 over the raw bytes) and reading the byte size are the
only operations it performs on a file.

It never decodes, parses, prints or summarises the content of any file. Before the tag
`frozen-plan` the only code that parses a sealed file is the Norway split, run once to move the
held-out sites' rows into the sealed folder (plan section 3, "Norway"); after the tag, the
held-out loader alone. The download tool and the hash list take the folder from here.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

FOLDER = "data/heldout"
IGNORED_NAMES = frozenset({".DS_Store"})  # folder metadata written by the operating system
_BLOCK = 1 << 20


@dataclass(frozen=True)
class Digest:
    size: int
    sha256: str
    md5: str


def folder(root: Path) -> Path:
    """The sealed folder under a project top folder."""
    return Path(root) / FOLDER


def sealed_path(root: Path, set_name: str, file_name: str) -> Path:
    """Where a held-out file of one data set is kept: `data/heldout/<set>/<file name>`."""
    for part in (set_name, file_name):
        if not part or "/" in part or "\\" in part or part in (".", ".."):
            raise ValueError(f"not a plain name: {part!r}")
    return folder(root) / set_name / file_name


def sealed_files(root: Path) -> list[Path]:
    """Every file under the sealed folder, sorted; names only, nothing is opened."""
    top = folder(root)
    if not top.is_dir():
        return []
    return sorted(p for p in top.rglob("*") if p.is_file() and p.name not in IGNORED_NAMES)


def file_digest(path: Path) -> Digest:
    """Byte size, SHA-256 and MD5 of a file's raw bytes. The bytes are never decoded."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return Digest(size, sha256.hexdigest(), md5.hexdigest())

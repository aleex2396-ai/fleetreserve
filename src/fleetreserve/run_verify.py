"""Check that a results file still matches the manifest of the run that wrote it.

Command line:
    python -m fleetreserve.run_verify <results file> [more files...]

In code, table and figure steps call `require_verified(path)`: it returns the run's manifest or
raises `NotVerified`. Given `read` (a function of a path giving that file's bytes, raising
`OSError`), it reads the manifest and the file through it, each once, instead of opening them
itself: the WP12 and WP14 checks pass a reader that refuses a symbolic link, a "..", a "." or an
absolute part on the path. A reader that needs the file's content calls `read_verified(path)`
(with the same `read`, when given): the manifest and the file's bytes, read once, once those
bytes have the SHA-256 the manifest lists; without a `read` the file, every folder from it up to
its run folder and the run's manifest must not be symbolic links, and are opened through a path
that refuses one (a link is never followed, `CannotCheck`). A file that is not there is refused
as "the file is missing" before any manifest is looked for: `read_verified` looks at it without
following a link (`lstat`, as its read; a link at the file counts as there and is then refused
by name; any other `OSError` of that look, a link loop in a folder above the file for one, is
`CannotCheck`, never raised raw) or, given a `read`, only through that reader, whose read is
then the one read of its bytes (a refusal of the reader other than a missing file is raised
where those bytes are hashed, after the manifest's checks, as before). After it parses those bytes,
`still_verified(path, data)` hashes the file again and raises `NotVerified` when it no longer
holds them, so a file changed between its check and its parse is refused, never
parsed in place of the file checked.

A file is verified only if all of these hold:
- its run folder (the nearest parent folder holding `manifest.json`) has a readable manifest
  with the fields commit, command, start, end, exit_code and outputs;
- the manifest does not say the run hit its time limit (`timed_out` true); a manifest without
  that field, written before the limit existed, counts as not timed out;
- the run's exit code is 0 and the manifest does not say the run was interrupted;
- the file's path inside the run folder is listed in the manifest's outputs, with the SHA-256
  the file has now;
- the recorded commit exists in the git repository.

Exit status: 0 if every file is verified; 1 if any file has a manifest but fails a condition;
2 if a file or its manifest is missing or unreadable, or no file was given. 1 wins over 2.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from fleetreserve.run_record import MANIFEST_NAME, sha256_file

_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class NotVerified(Exception):
    """A condition for trust failed (exit status 1)."""

    exit_status = 1


class CannotCheck(NotVerified):
    """The file or its manifest is missing or unreadable, so no check was possible (status 2)."""

    exit_status = 2


MISSING = "the file is missing"
_NOT_THERE = (FileNotFoundError, NotADirectoryError, IsADirectoryError)


def plain_name(name: object) -> bool:
    """`name` is a plain output name below a run folder, as the recorder lists one: "/"-separated
    parts, none "..", "." or empty (so no absolute start). A caller verifying the outputs a
    manifest lists refuses any other before building a path from it, since `require_verified`
    normalises its path (`os.path.abspath`) before reading."""
    return isinstance(name, str) and all(part not in ("", ".", "..") for part in name.split("/"))


def _find_run_folder(file: Path) -> Path:
    for parent in file.parents:
        if (parent / MANIFEST_NAME).is_file():
            return parent
    raise CannotCheck(f"no {MANIFEST_NAME} in any parent folder of the file")


def _read_manifest(run_folder: Path, read: Callable[[Path], bytes] | None = None) -> dict:
    path = run_folder / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text() if read is None else read(path))
    except (OSError, ValueError) as problem:
        raise CannotCheck(f"{MANIFEST_NAME} cannot be read: {problem}") from problem
    if not isinstance(manifest, dict):
        raise CannotCheck(f"{MANIFEST_NAME} is not a JSON object")
    lacking = [
        field
        for field in ("commit", "command", "start", "end", "exit_code", "outputs")
        if field not in manifest
    ]
    if lacking:
        raise CannotCheck(f"{MANIFEST_NAME} lacks: {', '.join(lacking)}")
    outputs = manifest["outputs"]
    code = manifest["exit_code"]
    well_formed = (
        isinstance(manifest["commit"], str)
        and isinstance(manifest["command"], list)
        and isinstance(manifest["start"], str)
        and isinstance(manifest["end"], str)
        and isinstance(code, int)
        and not isinstance(code, bool)
        and isinstance(manifest.get("timed_out", False), bool)
        and isinstance(outputs, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in outputs.items())
    )
    if not well_formed:
        raise CannotCheck(f"{MANIFEST_NAME} has a field of the wrong type")
    return manifest


def _check_commit(run_folder: Path, commit: str) -> None:
    if not _COMMIT.fullmatch(commit):
        raise NotVerified("the recorded commit is not a full hexadecimal SHA")
    try:
        found = subprocess.run(
            ["git", "-C", str(run_folder), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        )
    except OSError as problem:
        raise CannotCheck(f"git could not be started: {problem}") from problem
    if found.returncode != 0:
        raise NotVerified(f"commit {commit[:7]} does not exist in the repository")


def require_verified(
    path: str | os.PathLike[str], read: Callable[[Path], bytes] | None = None
) -> dict:
    """Return the manifest of the run that wrote `path`, or raise `NotVerified` (module text)."""

    def digest(file: Path) -> str:
        return sha256_file(file) if read is None else hashlib.sha256(read(file)).hexdigest()

    return _verified(path, digest, read)


def still_verified(path: str | os.PathLike[str], data: bytes) -> None:
    """Raises `NotVerified` when `path` no longer holds `data`, the bytes `read_verified` read and
    the caller parsed (module text)."""
    try:
        current = sha256_file(Path(os.path.abspath(path)))
    except OSError as problem:
        raise CannotCheck(f"the file cannot be read again after its check: {problem}") from problem
    if current != hashlib.sha256(data).hexdigest():
        raise NotVerified(
            "the file changed after it was read and checked (its SHA-256 is no longer that of "
            "the bytes parsed); nothing parsed from it is used"
        )


def _verified(
    path: str | os.PathLike[str],
    digest,
    read: Callable[[Path], bytes] | None = None,
    there: Callable[[Path], bool] = Path.is_file,
) -> dict:
    """The manifest of the run that wrote `path` (read through `read` when given), the file hashed
    by `digest`, refused as missing unless `there`; raises `NotVerified`."""
    file = Path(os.path.abspath(path))
    if not there(file):
        raise CannotCheck(MISSING)
    run_folder = _find_run_folder(file)
    manifest = _read_manifest(run_folder, read)
    if manifest.get("timed_out", False):
        raise NotVerified("the run hit its time limit and was stopped")
    if manifest["exit_code"] != 0:
        raise NotVerified(f"the run ended with exit code {manifest['exit_code']}, not 0")
    if "interrupted" in manifest:
        raise NotVerified(f"the run was interrupted by {manifest['interrupted']}")
    relative = file.relative_to(run_folder).as_posix()
    if relative not in manifest["outputs"]:
        raise NotVerified(f"{relative} is not listed in the outputs of the manifest")
    try:
        current = digest(file)
    except OSError as problem:
        raise CannotCheck(f"the file cannot be read: {problem}") from problem
    if current != manifest["outputs"][relative]:
        raise NotVerified("the file has changed since the run (its hash differs)")
    _check_commit(run_folder, manifest["commit"])
    return manifest


def _no_link(name, flags: int) -> int:
    """An opener that refuses a symbolic link (`OSError`)."""
    return os.open(name, flags | os.O_NOFOLLOW)


def _read_no_link(file: Path) -> bytes:
    """The bytes of `file`, opened so that a symbolic link at it raises `OSError`."""
    with open(file, "rb", opener=_no_link) as handle:
        return handle.read()


def _there_no_link(file: Path) -> bool:
    """Whether `file` is there to read, looked at as `_read_no_link` opens it, never following a
    link at it (`lstat`): a regular file, or a symbolic link (refused by name afterwards); not a
    missing name, a folder or another kind of file. Any other `OSError` of the look (a link loop
    in a folder above, no permission) is `CannotCheck`, status 2, never raised raw."""
    try:
        mode = os.lstat(file).st_mode
    except _NOT_THERE:
        return False
    except OSError as problem:  # a link loop, no permission, an input/output error: no look made
        raise CannotCheck(f"{file} could not be looked at: {problem.strerror}") from None
    return stat.S_ISREG(mode) or stat.S_ISLNK(mode)


def read_verified(
    path: str | os.PathLike[str], read: Callable[[Path], bytes] | None = None
) -> tuple[dict, bytes]:
    """(the manifest of the run that wrote `path`, the file's bytes), or raise `NotVerified`
    (module text): the manifest and the bytes are read once each, through `read` when given (the
    links reader of the checks, which refuses the paths it refuses), else through a path that
    refuses a symbolic link; the SHA-256 compared with the manifest's is that of the bytes
    returned."""
    file = Path(os.path.abspath(path))
    found = {}
    if read is None:
        if not _there_no_link(file):
            raise CannotCheck(MISSING)
        run_folder = _find_run_folder(file)
        for place in (file, *file.parents[: file.parents.index(run_folder) + 1]):
            if place.is_symlink():
                raise CannotCheck(f"{place.name} is a symbolic link, never followed")
        if (run_folder / MANIFEST_NAME).is_symlink():
            raise CannotCheck(f"{MANIFEST_NAME} is a symbolic link, never followed")
        read, there = _read_no_link, _there_no_link
    else:
        try:
            found["data"] = read(file)
        except _NOT_THERE:
            raise CannotCheck(MISSING) from None
        except OSError as problem:
            found["refused"] = problem

        def there(_: Path) -> bool:
            return True

    def digest(opened: Path) -> str:
        if "refused" in found:
            raise found["refused"]
        if "data" not in found:
            found["data"] = read(opened)
        return hashlib.sha256(found["data"]).hexdigest()

    return _verified(file, digest, read, there), found["data"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fleetreserve.run_verify",
        description="Check results files against the manifests of the runs that wrote them.",
    )
    parser.add_argument("files", nargs="*", help="results files to check")
    files = parser.parse_args(argv).files
    if not files:
        parser.error("give at least one results file")
    statuses = set()
    for file in files:
        try:
            manifest = require_verified(file)
        except NotVerified as problem:
            print(f"NOT VERIFIED: {file}: {problem}")
            statuses.add(problem.exit_status)
        else:
            print(f"verified: {file} (commit {manifest['commit'][:7]}, {manifest['start']})")
    if 1 in statuses:
        return 1
    return 2 if 2 in statuses else 0


if __name__ == "__main__":
    sys.exit(main())

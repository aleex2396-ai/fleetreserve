"""Check the files git tracks against the project's repository rules.

Run as `python tools/repo_rules.py [root]`. The root defaults to the folder above `tools/` and
must be the top of a git repository. The files checked are those in git's index: the files git
tracks plus the files staged for commit, as they are staged. Files that are not in git are not
looked at.

Each violation prints one line, `<path>:<line or ->: <rule name>: <what was found>`. Exit status:
1 if any violation, 0 (with one summary line) if none, 2 if git is missing or the root is not the
top of a git repository.

The rules defend against accidents by an honest contributor: a data file added by mistake, a path
pasted from a terminal, a solver called directly out of habit, a second module opening held-out
files. They do not defend against deliberate evasion, and a rule that cannot be made reliable
without false alarms on honest code is left out. No rule can be switched off from the command
line or the environment.

Rules, in the order they run:
  tracked-private  private folders, local-only files and credential files stay out of git. The
                   local-only names are the built-in ones plus the plain names and folder
                   patterns in the checkout's own `.git/info/exclude` file (see read_local_private).
  large-file       no file over LARGE_FILE_BYTES.
  absolute-path    no text file holds a Unix absolute path of a user or a machine (Windows
                   drive paths are not checked: they look like maths notation).
  secret-text      no text file holds a string shaped like a live key.
  solver-door      under src/ and tools/, only SOLVER_DOOR_FILES name a linear-programme solver.
  heldout-seal     under src/ and tools/, only HELDOUT_FILES name the held-out data.

A binary file (a NUL byte in the first 8192 bytes) is skipped by the text rules and still gets
the path and size rules. Any other file is text; bytes that are not valid UTF-8 are replaced
before the text rules read it.
"""

import fnmatch
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LARGE_FILE_BYTES = 1_000_000
BINARY_SNIFF_BYTES = 8192

# tracked-private
PRIVATE_TOP_FOLDERS = ("data", "runs.noindex", "reference", ".venv")
PRIVATE_FOLDERS_ANYWHERE = (".venv", "runs.noindex", ".key")
PRIVATE_FILE_NAMES = ("STATUS.md",)
# Matched without regard to letter case; ".env.*" covers ".env.local" and the like.
CREDENTIAL_NAME_PATTERNS = (".env", ".env.*", "*.token", "*api_key*", "*anthropic_key*")

# absolute-path: the first folder of a path that belongs to a user or a machine.
ABSOLUTE_ROOTS = ("Users", "home", "private", "tmp", "var", "Volumes", "opt", "etc", "dev", "usr")
ALLOWED_ABSOLUTE_PATHS = ("/usr/bin/env",)

# solver-door
SOLVER_FOLDERS = ("src/", "tools/")
SOLVER_DOOR_FILES = ["src/fleetreserve/lp.py", "tools/repo_rules.py"]
SOLVER_NAMES = ("linprog", "milp", "highspy", "cvxpy", "pulp", "gurobipy")
SOLVER_PRIVATE_MODULE = "scipy.optimize._highs"

# heldout-seal
HELDOUT_FOLDERS = ("src/", "tools/")
HELDOUT_FILES = [
    "src/fleetreserve/heldout_loader.py",
    "src/fleetreserve/heldout_seal.py",
    # calls the held-out loader's gated routes only; opens no file
    "src/fleetreserve/heldout_truth_build.py",
    "src/fleetreserve/sealed_route.py",  # calls the held-out loader's checks only; opens no file
    "tools/data_hashes.py",  # hashes the bytes of sealed files, never parses them
    "tools/fetch_data.py",  # moves verified downloaded bytes into the sealed folder
    "tools/norway_split.py",  # the plan's one split of the Norway session file
    "tools/repo_rules.py",
]


@dataclass(frozen=True)
class TrackedFile:
    path: str
    size: int
    text: str | None  # None for a binary file, and for a file too large to read


Finding = tuple[int | None, str]  # (line number or None, what was found)


@dataclass(frozen=True)
class LocalPrivate:
    """Names this checkout keeps out of git, read from its own exclude file.

    `anything` holds names that stand for a file or a folder; `folders` holds names that stand
    for a folder only.
    """

    anything: frozenset[str] = frozenset()
    folders: frozenset[str] = frozenset()


NO_LOCAL = LocalPrivate()


@dataclass(frozen=True)
class Rule:
    name: str
    check: Callable[..., list[Finding]]
    needs_local: bool = False  # the check also takes the checkout's LocalPrivate


def lines_matching(text: str, pattern: re.Pattern[str], describe: Callable[[str], str]):
    found: list[Finding] = []
    for number, line in enumerate(text.split("\n"), start=1):
        for match in pattern.finditer(line):
            found.append((number, describe(match.group(0))))
    return found


# --- tracked-private -----------------------------------------------------------------------------


def check_tracked_private(file: TrackedFile, local: LocalPrivate = NO_LOCAL) -> list[Finding]:
    parts = file.path.split("/")
    name = parts[-1]
    if parts[0] in PRIVATE_TOP_FOLDERS and len(parts) > 1:
        return [(None, f"tracked path is under {parts[0]}/, which stays out of git")]
    for folder in parts[:-1]:
        if folder in PRIVATE_FOLDERS_ANYWHERE:
            return [(None, f"tracked path is under a {folder}/ folder, which stays out of git")]
    if name in PRIVATE_FILE_NAMES:
        return [(None, f"{name} is a local-only file and stays out of git")]
    for folder in parts[:-1]:
        if folder in local.anything or folder in local.folders:
            return [
                (None, f"tracked path is under {folder}/, which this checkout keeps out of git")
            ]
    if name in local.anything:
        return [(None, f"{name} is listed in this checkout's exclude file and stays out of git")]
    for pattern in CREDENTIAL_NAME_PATTERNS:
        if fnmatch.fnmatchcase(name.lower(), pattern):
            return [(None, f"file name matches the credential pattern {pattern}")]
    return []


# --- large-file ----------------------------------------------------------------------------------


def check_large_file(file: TrackedFile) -> list[Finding]:
    if file.size > LARGE_FILE_BYTES:
        return [(None, f"{file.size} bytes, over the limit of {LARGE_FILE_BYTES}")]
    return []


# --- absolute-path -------------------------------------------------------------------------------

# A path token starts at a "/" that is not the middle of a longer word or path: not after a
# letter, digit, "_", "/", ".", "~", "-", "$", or a closing bracket. That keeps "and/or", "1/2",
# "https://host/home/x", "src/tmp/x", "./tmp/x", "a//tmp/b" and "${DIR}/var/x" from matching. A
# "/" after "file://" starts a path even though it follows a "/".
# The first folder must be one of ABSOLUTE_ROOTS and be followed by a "/" (so "/tmpfile",
# "/homepage" and a bare "/tmp" do not match).
# Windows drive paths are not checked: a drive path cannot be told apart from maths notation
# with a colon and a backslash.
_START = r"(?:(?<![\w/.~$\-)\]}])|(?<=file://))"
_UNIX_PATH = re.compile(_START + "/(?:" + "|".join(ABSOLUTE_ROOTS) + r")/[^\s\"'`<>()\[\]{}|,;]*")


def describe_path(token: str) -> str:
    return f"absolute path {token}"


def check_absolute_path(file: TrackedFile) -> list[Finding]:
    if file.text is None:
        return []
    found: list[Finding] = []
    for number, line in enumerate(file.text.split("\n"), start=1):
        for match in _UNIX_PATH.finditer(line):
            token = match.group(0).rstrip(".:!?")
            if token not in ALLOWED_ABSOLUTE_PATHS:
                found.append((number, describe_path(token)))
    return found


# --- secret-text ---------------------------------------------------------------------------------

# The message never repeats the string found.
_SECRET_PATTERNS = (
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"), "a string shaped like an sk-ant- key"),
    (
        re.compile(r"(?<![A-Za-z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
        "a string shaped like an AWS key id",
    ),
    (re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"), "the start of a private key block"),
)


def check_secret_text(file: TrackedFile) -> list[Finding]:
    if file.text is None:
        return []
    found: list[Finding] = []
    for pattern, what in _SECRET_PATTERNS:
        found.extend(lines_matching(file.text, pattern, lambda _token, what=what: what))
    return found


# --- solver-door ---------------------------------------------------------------------------------

_SOLVER = re.compile(
    r"\b(?:" + "|".join(SOLVER_NAMES) + r")\b|\b" + re.escape(SOLVER_PRIVATE_MODULE) + r"\b"
)


def check_solver_door(file: TrackedFile) -> list[Finding]:
    if file.text is None or not file.path.endswith(".py"):
        return []
    if not file.path.startswith(SOLVER_FOLDERS) or file.path in SOLVER_DOOR_FILES:
        return []
    return lines_matching(
        file.text, _SOLVER, lambda name: f"{name} named outside the one solve function"
    )


# --- heldout-seal --------------------------------------------------------------------------------

# "heldout" in any letter case is the name of the held-out data folder and of the two modules
# that may touch it, so it is flagged wherever it appears. "held_out" and "held-out" are also
# ordinary words for a held-out fold in model fitting, so they are flagged only when used in a
# path: touching a "/" or a "\" on either side, and not the end of a longer word. The one module
# name `heldout_limits` (the loaders' removal limits, no data folder) is not flagged.
_HELDOUT = re.compile(
    r"heldout(?!_limits\b)|(?<=[/\\])held[_-]out|(?<![A-Za-z0-9])held[_-]out(?=[/\\])", re.I
)


def check_heldout_seal(file: TrackedFile) -> list[Finding]:
    if file.text is None:
        return []
    if not file.path.startswith(HELDOUT_FOLDERS) or file.path in HELDOUT_FILES:
        return []
    return lines_matching(
        file.text, _HELDOUT, lambda name: f"{name} named outside the held-out loader and seal"
    )


RULES = [
    Rule("tracked-private", check_tracked_private, needs_local=True),
    Rule("large-file", check_large_file),
    Rule("absolute-path", check_absolute_path),
    Rule("secret-text", check_secret_text),
    Rule("solver-door", check_solver_door),
    Rule("heldout-seal", check_heldout_seal),
]


# --- reading git ---------------------------------------------------------------------------------


class CouldNotRun(Exception):
    pass


def git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args],
            input=stdin,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (FileNotFoundError, NotADirectoryError, subprocess.TimeoutExpired) as error:
        raise CouldNotRun(f"cannot run git: {error}") from error
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace").strip().splitlines()
        raise CouldNotRun(
            f"git {' '.join(args[:2])} failed: {detail[0] if detail else 'no message'}"
        )
    return done.stdout


def index_entries(root: Path) -> list[tuple[str, str, str]]:
    """Return (mode, object id, path) for every path in git's index, once each, sorted."""
    top = git(root, "rev-parse", "--show-toplevel").decode(errors="replace").strip()
    if Path(top).resolve() != root.resolve():
        raise CouldNotRun(f"{root} is not the top of a git repository")
    entries: dict[str, tuple[str, str, str]] = {}
    for record in git(root, "ls-files", "--cached", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        info, _, raw_path = record.partition(b"\t")
        mode, sha, _stage = info.decode().split(" ")
        path = raw_path.decode("utf-8", errors="surrogateescape")
        entries[path] = (mode, sha, path)
    return [entries[path] for path in sorted(entries)]


def read_files(root: Path) -> list[TrackedFile]:
    entries = index_entries(root)
    readable = {sha for mode, sha, _ in entries if mode != "160000"}  # not a folder of another repo
    if not readable:
        return []
    ordered = sorted(readable)
    listing = git(root, "cat-file", "--batch-check", stdin=("\n".join(ordered) + "\n").encode())
    sizes: dict[str, int] = {}
    for line in listing.decode().splitlines():
        sha, kind, *rest = line.split(" ")
        if kind == "missing":
            raise CouldNotRun(f"git has no object {sha}")
        sizes[sha] = int(rest[0])
    small = [sha for sha in ordered if sizes[sha] <= LARGE_FILE_BYTES]
    contents: dict[str, bytes] = {}
    if small:
        data = git(root, "cat-file", "--batch", stdin=("\n".join(small) + "\n").encode())
        position = 0
        for sha in small:
            end = data.index(b"\n", position)
            size = int(data[position:end].split(b" ")[2])
            contents[sha] = data[end + 1 : end + 1 + size]
            position = end + 1 + size + 1
    files = []
    for mode, sha, path in entries:
        if mode == "160000":
            files.append(TrackedFile(path, 0, None))
            continue
        files.append(TrackedFile(path, sizes[sha], decode_text(contents.get(sha))))
    return files


def parse_exclude(text: str) -> LocalPrivate:
    """Read the plain names and folder patterns of an exclude file.

    Read: `name` (a file or folder of that name at any depth), `name/` (a folder), and either
    form after `**/`. Not read, so they add nothing: a pattern with a wildcard (`*`, `?`, `[`),
    with a backslash, or with a "/" anywhere else (a rooted `/name` or a path such as `a/b`),
    a `!` pattern, and a comment or blank line. Trailing spaces are dropped.
    """
    anything: set[str] = set()
    folders: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        line = line.removeprefix("**/")
        folder_only = line.endswith("/")
        name = line.removesuffix("/")
        if not name or any(mark in name for mark in "/*?[\\"):
            continue
        (folders if folder_only else anything).add(name)
    return LocalPrivate(frozenset(anything), frozenset(folders))


def read_local_private(root: Path) -> LocalPrivate:
    """The names in `<git common folder>/info/exclude` of the repository whose top is `root`.

    The common folder is read, so a linked working copy sees the exclude file of the main one.
    A folder that is not the top of a git repository, or a missing file, gives no names. A file
    that exists and cannot be read raises CouldNotRun. Git's global ignore file and the
    `.gitignore` files are not read: they are shared or per-user, not this checkout's own list.
    """
    try:
        index_top = git(root, "rev-parse", "--show-toplevel").decode(errors="replace").strip()
        if Path(index_top).resolve() != root.resolve():
            return LocalPrivate()
        common = git(root, "rev-parse", "--git-common-dir").decode(errors="replace").strip()
    except CouldNotRun:
        return LocalPrivate()
    path = root / common / "info" / "exclude"
    try:
        return parse_exclude(path.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return LocalPrivate()
    except OSError as error:
        raise CouldNotRun(f"cannot read {path}: {error}") from error


def decode_text(content: bytes | None) -> str | None:
    if content is None or b"\0" in content[:BINARY_SNIFF_BYTES]:
        return None
    return content.decode("utf-8", errors="replace")


# --- running -------------------------------------------------------------------------------------


def find_violations(files: list[TrackedFile], local: LocalPrivate = NO_LOCAL) -> list[str]:
    found: list[tuple[str, int, str, str]] = []
    for file in files:
        for rule in RULES:
            findings = rule.check(file, local) if rule.needs_local else rule.check(file)
            for line, what in findings:
                found.append((file.path, line or 0, rule.name, what))
    found.sort(key=lambda item: (item[0].encode("utf-8", "surrogateescape"), item[1]))
    return [
        f"{path}:{line or '-'}: {rule}: {what}".replace("\n", " ")
        for path, line, rule, what in found
    ]


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print("COULD NOT RUN: usage: repo_rules.py [root]")
        return 2
    root = Path(argv[1]) if len(argv) == 2 else Path(__file__).resolve().parents[1]
    try:
        files = read_files(root)
        local = read_local_private(root)
    except CouldNotRun as error:
        print(f"COULD NOT RUN: {error}")
        return 2
    problems = find_violations(files, local)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"{len(files)} tracked files follow the {len(RULES)} repository rules")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

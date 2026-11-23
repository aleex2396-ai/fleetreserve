"""Made-up sealed files in a made-up repository, for the tests of the gated opener and the loaders.

`sealed_repo` builds a small git repository in a temporary folder: a `.gitignore` that keeps
`data/` out of git (as the project's own does), made-up sealed files under
`data/heldout/<set>/<file>`, a hash list `data-hashes.json` in the layout `tools/data_hashes.py`
writes, a copy of the running package's modules under `src/fleetreserve/` (so that the opener's
check of the running code passes; `package=False` leaves it out), one commit and, unless asked
not to, the tag `frozen-plan` on it. Its root must lie in the system's temporary folder, so no
test built on it can reach a checkout's sealed folder.

`git` runs git as the tests' own tool: without any variable whose name starts with `GIT_` and
without global or system settings, so a variable that a test sets for the opener, or one set by
the program that started the tests, never changes how the made-up repositories are built.

`COMMAND_TEXT`, `write_command` and `run_python` run the opener in a new Python process, as
the sealed run's commands are run: `write_command` writes a made-up command module (or script)
that opens one sealed file of a root given on its command line and prints "opened <bytes>" or
"refused: <message>"; it refuses to run with a root outside the system's temporary folder.
`run_python` runs a Python command with the given folder first on its module search path,
without writing byte-code files (so no `__pycache__` folder makes a made-up tree unclean) and
without any `GIT_` variable.

A parsing fix after the tag (the opener's reading H6): `freeze` commits a made-up plan holding
section 14 (`write_plan`) and tags it; `correction` commits a correction of one set's loader (a
line added to the loader, or to a made-up test file of it, `loader_test_file`); `entry` commits
section 14 naming commits by their full ids; `log_refusal` writes a line of the openings log and
then one of the refusal log by hand (the refusal line alone with `opened=False`), for tests whose
made-up sealed files no parser refuses. `approve` writes by hand, as the owner's approval of a
fix commit is written, one line of the approvals record (the opener's reading H7).
`gate_claiming_a_fix` makes the sealed routes' gate return a fix commit at the tag, for tests of
what a command records at a fix.

The a.s.r. layout here is built only from the data set's README: the columns it lists, in the
order it lists them, times written YYYY-MM-DDThh:mm:ss in local time, energy in kWh. The field
separator and the form of a missing value are not stated there; the writer uses "," and an empty
field by default, and can write ";" with a decimal comma. No real data is read or copied.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from git_run import run_git

import fleetreserve

ASR_HEADER = (
    "EV_id_x",
    "start_datetime",
    "end_datetime",
    "total_energy",
    "rail",
    "evse_id",
    "channel",
    "capacity_kwh",
    "commute_km_range_min",
    "commute_km_range_max",
    "EV_brand_selfreported",
    "EV_model_selfreported",
    "capacity_kWh_selfreported",
    "ownership",
)
ASR_FILE = ("asr", "202410DatasetEVOfficeParking_v0.csv")
NORWAY_FILE = ("norway", "Dataset1_charging_reports-held-out.csv")
SEALED = "data/heldout"
TAG = "frozen-plan"
PACKAGE = "src/fleetreserve"


def is_temporary(path: Path) -> bool:
    """True when `path` lies inside the system's temporary folder."""
    top = Path(tempfile.gettempdir()).resolve()
    return top in Path(path).resolve().parents


def tool_environment() -> dict[str, str]:
    """The environment of the tests' own git calls: no `GIT_` variable, no global or system
    settings."""
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    return environment


def git(root: Path, *args: str) -> str:
    done = run_git(
        [
            "git",
            "-c",
            "init.defaultBranch=main",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=tool_environment(),
    )
    return done.stdout.strip()


def commit(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "--no-verify", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def sealed_repo(
    tmp_path: Path,
    files: dict[tuple[str, str], bytes],
    *,
    listed: dict[tuple[str, str], bytes] | None = None,
    tag: bool = True,
    role: str = "held-out",
    package: bool = True,
) -> Path:
    """A repository with `files` sealed and the hashes of `listed` (default: `files`) committed.

    Every listed entry has the role `role`; "held-out", the default, is the role
    `tools/data_hashes.py` gives every file of the sealed folder. The repository is
    `tmp_path / "repo"`; a missing `tmp_path` is made.
    """
    root = tmp_path / "repo"
    if not is_temporary(root):
        raise AssertionError(f"a made-up repository must lie in the temporary folder: {root}")
    root.mkdir(parents=True)
    git(root, "init", "-q")
    (root / ".gitignore").write_text("data/\n", encoding="utf-8")
    for (set_name, file_name), content in files.items():
        path = root / SEALED / set_name / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    entries = [
        {
            "path": f"{SEALED}/{set_name}/{file_name}",
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "role": role,
        }
        for (set_name, file_name), content in sorted((files if listed is None else listed).items())
    ]
    lines = ",\n".join(json.dumps(entry) for entry in entries)
    (root / "data-hashes.json").write_text('{"files": [\n' + lines + "\n]}\n", encoding="utf-8")
    if package:
        copy_package(root)
    commit(root, "made-up sealed files and hash list")
    if tag:
        git(root, "tag", TAG)
    return root


def copy_package(root: Path, package: str = PACKAGE) -> None:
    """Copy every module file of the running package `fleetreserve` to `<package>/` under
    `root` (default `src/fleetreserve/`)."""
    source = Path(fleetreserve.__file__).resolve().parent
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        copy = root / package / path.relative_to(source)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy)


COMMAND_TEXT = '''"""A made-up command that opens one sealed file of a made-up root."""

import sys
import tempfile
from pathlib import Path

from fleetreserve import heldout_loader

root = Path(sys.argv[1])
if Path(tempfile.gettempdir()).resolve() not in root.resolve().parents:
    raise SystemExit("a made-up root in the temporary folder only")
try:
    content = heldout_loader.open_sealed(root, sys.argv[2], sys.argv[3])
except heldout_loader.SealRefused as refused:
    print(f"refused: {refused}")
    raise SystemExit(3) from None
print(f"opened {len(content)}")
'''


def write_command(path: Path, extra: str = "") -> Path:
    """The made-up command at `path` (its folder made), with `extra` appended to its text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(COMMAND_TEXT + extra, encoding="utf-8")
    return path


def run_python(
    args: list[str], *, path: Path, cwd: Path, extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """`python <args>` in `cwd` with `path` first on the module search path and the variables
    `extra` set."""
    environment = tool_environment()
    environment["PYTHONPATH"] = str(path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(extra or {})
    return subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


WORK = "Made-up work packages.\n"
LOADERS = {"asr": "load_asr", "norway": "load_heldout_norway", "korea": "load_korea"}
REFUSALS = "data/heldout-refusals.log"
OPENINGS = "data/heldout-openings.log"
OPENING_FIELDS = ("time_utc", "set", "file", "sha256", "commit", "tag_commit", "command")
REFUSAL_FIELDS = ("time_utc", "set", "file", "sha256", "commit", "tag_commit", "layout")


def write_plan(root: Path, section: str = "None yet.\n", work: str = WORK) -> None:
    """A made-up `docs/plan.md` with sections 13, 14 (holding `section`) and 15."""
    path = root / "docs" / "plan.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        f"# A made-up plan\n\n## 13. Work packages\n\n{work}\n## 14. Changes after the tag\n\n"
        f"{section}\n## 15. References\n\nMade-up references.\n",
        encoding="utf-8",
    )


def freeze(root: Path) -> str:
    """The made-up plan committed and tagged `frozen-plan`; the tag's commit."""
    write_plan(root)
    tagged = commit(root, "a made-up plan")
    git(root, "tag", TAG)
    return tagged


def loader_test_file(root: Path, set_name: str, name: str = "fix") -> Path:
    """A made-up test file of the set's loader, as the opener's reading H6(b) names one."""
    return root / "tests" / "wp2" / f"test_{LOADERS[set_name]}_{name}.py"


def append_line(path: Path, text: str = "# a made-up parsing fix\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def correction(root: Path, set_name: str = "asr", *, loader: bool = False, name: str = "fix"):
    """A loader correction of one set: a line added to its loader, or to a test file of it; the
    commit."""
    if loader:
        append_line(root / PACKAGE / f"{LOADERS[set_name]}.py")
    else:
        append_line(loader_test_file(root, set_name, name))
    return commit(root, f"a made-up correction of the {set_name} loader")


def entry(root: Path, *named: str) -> str:
    """The entry in section 14 naming each commit of `named` by its full id; the commit."""
    write_plan(root, "".join(f"- {one}: a made-up parsing fix.\n" for one in named))
    return commit(root, "a made-up entry in plan section 14")


def log_refusal(
    root: Path,
    names: tuple[str, str],
    content: bytes,
    at: str,
    tag: str | None = None,
    *,
    opened: bool = True,
    time: str = "2026-01-01T00:00:00.000000Z",
) -> None:
    """Lines written here, not by a route: the layout of the sealed file `names` (holding
    `content`) refused at an opening at the commit `at`, the tag's, or a fix commit of the tag's
    commit `tag`: the opening's line in the openings log (as `open_sealed` writes it, with
    `fix_of_tag` at a fix), unless `opened` is false, then the refusal's line, both at `time`."""
    digest = hashlib.sha256(content).hexdigest()
    tagged = at if tag is None else tag
    if opened:
        fields = (time, *names, digest, at, tagged, ["made-up command"])
        opening = dict(zip(OPENING_FIELDS, fields, strict=True))
        if tagged != at:
            opening["fix_of_tag"] = tagged
        append_line(root / OPENINGS, json.dumps(opening) + "\n")
    values = (time, *names, digest, at, tagged)
    line = dict(zip(REFUSAL_FIELDS, (*values, "a made-up layout refusal"), strict=True))
    append_line(root / REFUSALS, json.dumps(line) + "\n")


APPROVALS = "data/heldout-fix-approvals.log"
APPROVAL_FIELDS = ("time_utc", "fix_commit", "words")


def approve(
    root: Path,
    fix: str,
    words: str = "yes, open the sealed files at this made-up fix",
    time: str = "2026-01-01T00:00Z",
) -> None:
    """One line of the approvals record, written here by hand as a person writes it, not by a
    function of the package: the fix commit `fix` approved in `words` at the UTC time `time`."""
    line = dict(zip(APPROVAL_FIELDS, (time, fix, words), strict=True))
    append_line(root / APPROVALS, json.dumps(line) + "\n")


CLAIMED_FIX = "e" * 40


def gate_claiming_a_fix(monkeypatch, fix: str = CLAIMED_FIX) -> None:
    """The sealed routes' gate (`sealed_route.require_gate`) making its real checks, at the tag of
    a made-up repository, and returning the tag's commit with `fix` as the fix commit, as it does
    at a logged parsing fix: for the tests of what a command records at a fix, which need no
    second repository (the real gate at a fix: `test_heldout_route_scoring`, item 9)."""
    from fleetreserve import sealed_route

    real = sealed_route.require_gate

    def claimed(root=sealed_route.TOP):
        return sealed_route.GateCommit(real(root), fix)

    monkeypatch.setattr(sealed_route, "require_gate", claimed)


def unreadable(path: Path) -> None:
    """Take every permission off a file, so any attempt to read its bytes raises."""
    path.chmod(0)
    try:
        path.open("rb").close()
    except PermissionError:
        return
    raise AssertionError("the file is still readable: the no-content-read proof would be void")


def asr_row(ev, start, end, energy, *, rail="B", evse="EVSE-1", channel="1", **extra) -> dict:
    row = {
        "EV_id_x": ev,
        "start_datetime": start,
        "end_datetime": end,
        "total_energy": energy,
        "rail": rail,
        "evse_id": evse,
        "channel": channel,
    }
    row.update(extra)
    return row


def asr_csv(rows: list[dict], header=ASR_HEADER, sep: str = ",") -> bytes:
    """The rows in the README's layout; None is written as an empty field."""
    lines = [sep.join(header)]
    for row in rows:
        cells = []
        for name in header:
            value = row.get(name)
            text = "" if value is None else str(value)
            if sep == ";" and isinstance(value, float):
                text = text.replace(".", ",")
            cells.append(text)
        lines.append(sep.join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def record_run(root: Path, label: str, source: Path, command: list[str], stamp: int = 1) -> Path:
    """A recorded run of `command` at the repository's checked-out commit: the folder
    `runs.noindex/<UTC start>-<label>` of `root` holding a copy of every file of `source` and a
    manifest in the recorder's form (exit 0, each file listed with its SHA-256). For the tests of
    what `run_evidence` reads of a run written by a real command, in a made-up repository."""
    folder = root / "runs.noindex" / f"20260101T{stamp:06d}Z-{label}"
    folder.mkdir(parents=True)
    outputs = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            shutil.copyfile(path, folder / path.name)
            outputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "commit": git(root, "rev-parse", "HEAD"),
        "command": command,
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T00:00:01Z",
        "exit_code": 0,
        "outputs": outputs,
        "long": False,
        "max_seconds": None,
        "timed_out": False,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return folder

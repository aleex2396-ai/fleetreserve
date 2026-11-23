"""Made-up recorded runs for the tests of `tools/wp9_check.py`.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real. The throwaway repository commits the project's
`gate/matrix.json`, the entries of the project's `passmarks/wp9-t8-runs.json` whose matrix is
`gate/matrix.json` (the development runs, `DEVELOPMENT_LIST_BYTES`; a test that reads the
training-site runs commits them and their matrices itself) and a stand-in package `fleetreserve`
whose modules `truth_build`, `gate_run`,
`soft_scale_check`, `size_resolve_whatif`, `dial_choice`, `compare_run` and `other` each only
write the made-up files they are handed (a JSON object of file name to text in the environment
variable `FAKE_FILES`, and every file of the folder `FAKE_FROM` names, for files too large for an
environment variable) and exit with `FAKE_EXIT`, so the recorded commands have the form of real
runs, for example `python -m fleetreserve.gate_run --matrix gate/matrix.json`.
"""

import hashlib
import importlib
import itertools
import json
import os
import sys
from pathlib import Path

from git_run import run_git

from fleetreserve.run_record import record_run

TOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOP / "tools"))
check = importlib.import_module("wp9_check")

MATRIX_BYTES = (TOP / "gate" / "matrix.json").read_bytes()
MATRIX_SHA = hashlib.sha256(MATRIX_BYTES).hexdigest()
GATE_UNITS = list(json.loads(MATRIX_BYTES)["gate_units"])
T8_LIST = "passmarks/wp9-t8-runs.json"
T8_LIST_BYTES = (TOP / T8_LIST).read_bytes()
_T8_LISTED = json.loads(T8_LIST_BYTES)
DEVELOPMENT_LIST_BYTES = json.dumps(
    {**_T8_LISTED, "runs": [r for r in _T8_LISTED["runs"] if r["matrix"] == "gate/matrix.json"]},
    indent=1,
).encode()
TRUTH_LABEL = "wp2-truth"

FAKE = """import json
import os
import shutil
import sys

out = os.environ["FLEETRESERVE_RUN_DIR"]
for name, text in json.loads(os.environ.get("FAKE_FILES", "{}")).items():
    with open(os.path.join(out, name), "w") as handle:
        handle.write(text)
source = os.environ.get("FAKE_FROM")
if source:
    for name in sorted(os.listdir(source)):
        shutil.copy(os.path.join(source, name), os.path.join(out, name))
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""
MODULES = (
    "truth_build",
    "gate_run",
    "soft_scale_check",
    "size_resolve_whatif",
    "dial_choice",
    "compare_run",
    "other",
)


def isolate_git(monkeypatch) -> None:
    """Run git with no user or system settings and none of the caller's GIT_ variables."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> None:
    run_git(["git", "-C", str(repo), *args], capture_output=True, check=True)


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "gate").mkdir(parents=True)
    (root / "fleetreserve").mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n__pycache__/\n")
    (root / "tracked.txt").write_text("one\n")
    (root / "gate" / "matrix.json").write_bytes(MATRIX_BYTES)
    (root / "passmarks").mkdir()
    (root / T8_LIST).write_bytes(DEVELOPMENT_LIST_BYTES)
    (root / "fleetreserve" / "__init__.py").write_text("")
    for module in MODULES:
        (root / "fleetreserve" / f"{module}.py").write_text(FAKE)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


def commit_change(repo: Path) -> None:
    (repo / "tracked.txt").write_text(f"{(repo / 'tracked.txt').read_text()}more\n")
    git(repo, "commit", "-q", "-am", "next")


def commit_files(repo: Path, files: dict) -> None:
    """Write `files` (path to text) in the repository and commit them."""
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    git(repo, "add", *files)
    git(repo, "commit", "-q", "-m", "files")


def head(repo: Path) -> str:
    done = run_git(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def docs_commit(repo: Path) -> str:
    """Commit a change under `docs/` only (on the keep-list of `run_evidence`); the commit before
    it."""
    earlier = head(repo)
    commit_files(repo, {"docs/notes.md": f"after {earlier}\n"})
    return earlier


def earlier_note(commit: str) -> str:
    """The words `run_evidence.Evidence.note` gives for a run accepted from `commit`."""
    return (
        f" (accepted from the earlier commit {commit[:12]}: every committed path that differs "
        "between it and the checked-out commit is on the keep-list of run_evidence; uncommitted "
        "files were not compared)"
    )


MADE = itertools.count(1)


def record(
    repo: Path,
    monkeypatch,
    label: str,
    module: str,
    args,
    files: dict,
    exit_code: int = 0,
    copied: Path | None = None,
) -> Path:
    """A recorded run of `python -m fleetreserve.<module> <args>` that writes `files` (name to
    text) and every file of the folder `copied` (when given) and exits with `exit_code`; its
    folder is renamed to a start later than every earlier one of the tests (two runs in a second
    would share a name)."""
    monkeypatch.setenv("FAKE_FILES", json.dumps(files))
    monkeypatch.setenv("FAKE_EXIT", str(exit_code))
    if copied is None:
        monkeypatch.delenv("FAKE_FROM", raising=False)
    else:
        monkeypatch.setenv("FAKE_FROM", str(copied))
    runs = repo / "runs.noindex"
    before = set(runs.glob("*")) if runs.exists() else set()
    command = [sys.executable, "-m", f"fleetreserve.{module}", *args]
    assert record_run(label, command, start_dir=repo) == exit_code
    (folder,) = set(runs.glob("*")) - before
    n = next(MADE)
    return folder.rename(runs / f"20000101T{n // 3600:02d}{n // 60 % 60:02d}{n % 60:02d}Z-{label}")


def truth(repo: Path, monkeypatch) -> str:
    """A recorded main truth build at the repository's commit; its folder's name."""
    files = {"truth-summary.json": json.dumps({"setting": "main", "units": {}})}
    return record(repo, monkeypatch, TRUTH_LABEL, "truth_build", [], files).name


def inputs(truth_run) -> dict:
    return {"setting": "main", "truth_run": truth_run, "table_sha256": {}}


def run_item(repo: Path, item: str, capsys) -> tuple[int, list[str]]:
    capsys.readouterr()  # what the run recorder printed
    code = check.main([item, "--top", str(repo)])
    return code, capsys.readouterr().out.splitlines()

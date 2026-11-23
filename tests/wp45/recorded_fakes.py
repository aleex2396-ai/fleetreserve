"""Made-up recorded runs for the tests of the WP4 and WP5 check tools.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real. The command of each run only writes a given results file and
ends with a given exit code (or sleeps past its time limit).
"""

import json
import os
import sys
from pathlib import Path

from git_run import run_git

from fleetreserve.run_record import record_run

WRITE = (
    "import os, sys; "
    "open(os.path.join(os.environ['FLEETRESERVE_RUN_DIR'], sys.argv[1]), 'w')"
    ".write(sys.argv[2]); sys.exit(int(sys.argv[3]))"
)
SLEEP_AFTER_WRITE = WRITE.replace("sys.exit(int(sys.argv[3]))", "import time; time.sleep(60)")


def clean_git_environment(monkeypatch) -> None:
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout


def make_repo(root: Path) -> Path:
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n")
    (root / "tracked.txt").write_text("one\n")
    git(root, "add", ".gitignore", "tracked.txt")
    git(root, "commit", "-q", "-m", "first")
    return root


def new_commit(repo: Path) -> None:
    (repo / "tracked.txt").write_text((repo / "tracked.txt").read_text() + "more\n")
    git(repo, "commit", "-q", "-am", "next")


def docs_commit(repo: Path) -> str:
    """Commit a change under `docs/` only (on the keep-list of `run_evidence`) and return the
    commit before it."""
    earlier = git(repo, "rev-parse", "HEAD").strip()
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "notes.md").write_text(f"later than {earlier}\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    return earlier


def earlier_note(commit: str) -> str:
    """The start of `Evidence.note` for a run read from the earlier commit `commit`."""
    return f" (accepted from the earlier commit {commit[:12]}: "


def make_run(
    repo: Path,
    label: str,
    name: str,
    document,
    *,
    exit_code: int = 0,
    time_out: bool = False,
    stamp: str | None = None,
) -> Path:
    """Record a run whose command writes `document` (JSON, or text as given) to `name`."""
    text = document if isinstance(document, str) else json.dumps(document)
    runs = repo / "runs.noindex"
    before = set(runs.glob("*")) if runs.exists() else set()
    if time_out:
        record_run(
            label,
            [sys.executable, "-c", SLEEP_AFTER_WRITE, name, text, "0"],
            start_dir=repo,
            max_seconds=1,
            grace_seconds=0.2,
        )
    else:
        status = record_run(
            label, [sys.executable, "-c", WRITE, name, text, str(exit_code)], start_dir=repo
        )
        assert status == exit_code
    (folder,) = set(runs.glob("*")) - before
    if stamp is not None:
        target = folder.parent / f"{stamp}-{label}"
        folder.rename(target)
        folder = target
    return folder

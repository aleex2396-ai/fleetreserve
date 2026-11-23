"""`tools/rehearsal_check.py` on a rehearsal drive written at an earlier commit (reading R1).

Each log is written by the real driver in a throwaway repository, as in
`tests/wp12/test_rehearsal_check.py`, whose helpers are used here; then the repository gets a later
commit and the check reads the log written before it.

1. A later commit that changes only paths of `run_evidence.UNREAD_BY_RUNS` (a file under `docs/`):
   passed (0), the log's line followed by one naming the earlier commit and saying every path
   changed since is on the keep-list, the last line naming the drive's commit and carrying the
   same words.
2. A later commit that changes a path off the keep-list (under `src/`, the run list itself, another
   file under `run-lists/`): stale, could not run (2), naming the path.
3. The drive's commit no ancestor of the checked-out one (the drive's commit amended with a change
   under `docs/`): stale, could not run (2).
4. A step whose command names a place of the keep-list (`docs/`), then a later commit under
   `docs/` alone: stale, could not run (2), naming the word (`run_evidence` reading E1).
"""

import os
from pathlib import Path

import pytest
from drive_fake import LIST_PATH, entry, git, make_repo, no_start_check, step
from test_rehearsal_check import (
    NO_DIFFERENCE_END,
    drive,
    read_log,
    run_check,
    the_log,
    two_passing,
)

from fleetreserve.run_evidence import earlier_note


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """No GIT_* variable and no global or system git settings, as in the WP12 check's tests."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    no_start_check(monkeypatch)  # the stand-ins' starts (drive_fake, module text)
    return make_repo(tmp_path)


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def change(repo: Path, path: str, text: str = "a later line\n", *, amend: bool = False) -> None:
    """Write `text` to `path` and commit it, as a new commit or by amending the last one."""
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text)
    git(repo, "add", path)
    if amend:
        git(repo, "commit", "-q", "--amend", "--no-edit")
    else:
        git(repo, "commit", "-q", "-m", f"a later change of {path}")


def stale_start(repo: Path, written: str) -> str:
    return (
        f"COULD NOT RUN: {the_log(repo).name} is stale: written at commit {written[:12]}, not at "
        f"the checked-out commit {head(repo)[:12]}, and "
    )


def test_a_drive_before_a_docs_only_commit_passes_and_names_its_commit(repo, capsys):
    assert drive(repo, two_passing()) == 0
    written = head(repo)
    change(repo, "docs/notes.md")
    code, lines = run_check(repo, capsys)
    assert code == 0, lines
    name, note = the_log(repo).name, earlier_note(written)
    assert lines[2] == f"log {name}, list {LIST_PATH}, commit {written[:12]}"
    assert lines[3] == f"  {name}{note}"
    assert "every committed path that differs between it and the checked-out commit" in note
    assert lines[4].startswith("  wp2-truth: run runs.noindex/")
    assert lines[5].startswith("  rehearsal-learn: run runs.noindex/")
    seconds = read_log(the_log(repo))[-1]["seconds"]
    assert lines[6] == (
        f"passed: the rehearsal of the run list {LIST_PATH} ran to its end at commit "
        f"{written[:12]}: 2 entries, each exited 0 with no timeout and verifies; total duration "
        f"{seconds:.0f} s ({seconds / 3600:.2f} h); {name}{note}{NO_DIFFERENCE_END}"
    )
    assert len(lines) == 7


@pytest.mark.parametrize(
    "path",
    ["src/fleetreserve/later.py", LIST_PATH, "run-lists/another.json"],
    ids=["under-src", "the-run-list", "under-run-lists"],
)
def test_a_drive_before_a_change_off_the_keep_list_is_stale(repo, capsys, path):
    assert drive(repo, two_passing()) == 0
    written = head(repo)
    change(repo, "docs/notes.md")
    text = (repo / path).read_text() + " \n" if (repo / path).exists() else "{}\n"
    change(repo, path, text)
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1] == f"{stale_start(repo, written)}{path} differs between them", lines


def test_a_drive_at_a_commit_that_is_no_ancestor_is_stale(repo, capsys):
    assert drive(repo, two_passing()) == 0
    written = head(repo)
    change(repo, "docs/notes.md", amend=True)
    assert head(repo) != written
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1] == (
        f"{stale_start(repo, written)}its recorded commit {written[:12]} is not an ancestor of the "
        "checked-out commit"
    ), lines


def test_a_drive_whose_step_names_the_keep_list_is_stale_after_a_docs_only_commit(repo, capsys):
    assert drive(repo, [entry("wp2-truth", step("truth_build", "--about", "docs/notes.md"))]) == 0
    written = head(repo)
    change(repo, "docs/notes.md")
    code, lines = run_check(repo, capsys)
    assert code == 2, lines
    assert lines[-1] == (
        f"{stale_start(repo, written)}its recorded command names docs/ (the word "
        "'docs/notes.md'; reading E1)"
    ), lines

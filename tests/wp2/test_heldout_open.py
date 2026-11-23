"""The gated opener of sealed files, in made-up repositories with made-up sealed files.

Refusals: a set or file name that is not plain; a root that is not a repository's top folder; no
tag; HEAD on a later or an earlier commit than the tag; a modified tracked file; an untracked
file; a tracked file marked assume-unchanged or skip-worktree and then modified; a file not in
the committed hash list, or listed with another role; a listed file that is absent; a sealed file
or set folder that is a symbolic link leading outside the sealed folder; a hash that differs.
Before the hash check the sealed file (or the target of the link) has no permissions at all, so
a read of its bytes before the check under test would fail, and the test, which expects that
check's own message, would fail with it. At the hash check the bytes are read and hashed; the
test then shows that no bytes came back, the parser was never called and nothing was logged.

Errors met on the way: a sealed file that cannot be read at the hash check, and git that cannot
be run or whose output cannot be decoded, each end in `SealRefused` with the error as its cause.

Failures to log: a log file that cannot be written and a log path that is a folder each end in
`SealRefused` with the first error as its cause, no bytes returned and no parser called.

Openings: the bytes come back and the openings log gains one line per opening with every field;
a symbolic link that stays inside the sealed folder is followed.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime

import pytest
from fake_heldout import ASR_FILE, commit, git, is_temporary, sealed_repo, unreadable

from fleetreserve import heldout_loader, load_asr
from fleetreserve.heldout_loader import SealRefused
from fleetreserve.sessions import TOP

CONTENT = b"made-up sealed bytes\n"


def _sealed_file(root):
    return root / "data" / "heldout" / ASR_FILE[0] / ASR_FILE[1]


def _refused(root, match):
    unreadable(_sealed_file(root))
    with pytest.raises(SealRefused, match=match):
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert heldout_loader.logged_openings(root) == []


def test_refuses_without_the_tag(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    _refused(root, "no tag frozen-plan")


def test_refuses_on_a_commit_after_the_tag(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "later.txt").write_text("a later commit\n", encoding="utf-8")
    commit(root, "a later commit")
    _refused(root, "after the tag")


def test_refuses_on_a_commit_before_the_tag(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, tag=False)
    (root / "later.txt").write_text("the tagged commit\n", encoding="utf-8")
    commit(root, "the tagged commit")
    git(root, "tag", "frozen-plan")
    git(root, "checkout", "-q", "--detach", "HEAD~1")
    _refused(root, "does not descend from the tag")


def test_refuses_with_a_modified_tracked_file(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("# a change not committed\n")
    _refused(root, "uncommitted changes")


def test_refuses_with_an_untracked_file(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    (root / "stray.txt").write_text("not in git\n", encoding="utf-8")
    _refused(root, "uncommitted changes")


def test_refuses_a_file_not_in_the_committed_list(tmp_path):
    other = ("asr", "another.csv")
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT, other: CONTENT}, listed={ASR_FILE: CONTENT})
    unreadable(root / "data" / "heldout" / other[0] / other[1])
    with pytest.raises(SealRefused, match="not listed as held-out"):
        heldout_loader.open_sealed(root, *other)
    assert heldout_loader.logged_openings(root) == []


def test_refuses_a_file_listed_with_another_role(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, role="document")
    _refused(root, "not listed as held-out")


def test_refuses_a_listed_file_that_is_absent(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    _sealed_file(root).unlink()
    with pytest.raises(SealRefused, match="listed but not present"):
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert heldout_loader.logged_openings(root) == []


def test_refuses_a_root_that_is_not_the_top_of_a_repository(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    unreadable(_sealed_file(root))
    with pytest.raises(SealRefused, match="not the top folder of a git repository"):
        heldout_loader.open_sealed(root / "data", *ASR_FILE)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_refuses_a_tracked_file_hidden_from_git_status(tmp_path, flag):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    git(root, "update-index", flag, ".gitignore")
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("# a change git status does not show\n")
    assert git(root, "status", "--porcelain") == ""
    _refused(root, "marked assume-unchanged or skip-worktree: .gitignore")


@pytest.mark.parametrize("name", [("..", "x.csv"), ("asr", "../x.csv"), ("asr", "a/b"), ("", "x")])
def test_a_name_that_is_not_plain_is_refused_and_returns_nothing(tmp_path, name):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    unreadable(_sealed_file(root))
    result = None
    with pytest.raises(SealRefused, match="not a plain name") as refused:
        result = heldout_loader.open_sealed(root, *name)
    assert isinstance(refused.value.__cause__, ValueError)
    assert result is None
    assert heldout_loader.logged_openings(root) == []


def _outside(tmp_path):
    folder = tmp_path / "outside"
    folder.mkdir()
    return folder


def test_refuses_a_sealed_file_that_links_outside_the_sealed_folder(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    target = _outside(tmp_path) / ASR_FILE[1]
    target.write_bytes(CONTENT)
    _sealed_file(root).unlink()
    _sealed_file(root).symlink_to(target)
    unreadable(target)
    with pytest.raises(SealRefused, match="leads outside the sealed folder"):
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert heldout_loader.logged_openings(root) == []


def test_refuses_a_set_folder_that_links_outside_the_sealed_folder(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    moved = _outside(tmp_path) / ASR_FILE[0]
    _sealed_file(root).parent.rename(moved)
    _sealed_file(root).parent.symlink_to(moved, target_is_directory=True)
    unreadable(moved / ASR_FILE[1])
    with pytest.raises(SealRefused, match="leads outside the sealed folder"):
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert heldout_loader.logged_openings(root) == []


def test_a_link_that_stays_inside_the_sealed_folder_opens(tmp_path):
    other = ("asr-copy", ASR_FILE[1])
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT, other: CONTENT})
    _sealed_file(root).unlink()
    _sealed_file(root).symlink_to(root / "data" / "heldout" / other[0] / other[1])
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT


def test_a_file_that_cannot_be_read_is_refused_and_returns_nothing(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    unreadable(_sealed_file(root))
    result = None
    with pytest.raises(SealRefused, match="could not be read") as refused:
        result = heldout_loader.open_sealed(root, *ASR_FILE)
    assert isinstance(refused.value.__cause__, PermissionError)
    assert result is None
    assert heldout_loader.logged_openings(root) == []


@pytest.mark.parametrize(
    "error", [OSError("no git"), UnicodeDecodeError("utf-8", b"\xff", 0, 1, "made up")]
)
def test_git_that_cannot_be_run_is_a_refusal(tmp_path, monkeypatch, error):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    unreadable(_sealed_file(root))

    calls = []

    def fails(argv, **kwargs):
        calls.append(argv)
        raise error

    monkeypatch.setattr(heldout_loader.subprocess, "run", fails)
    with pytest.raises(SealRefused, match="could not be run") as refused:
        heldout_loader.open_sealed(root, *ASR_FILE)
    assert refused.value.__cause__ is error
    # The message names git and the call's own arguments, after the fixed settings.
    assert len(calls) == 1
    asked = calls[0][3 + len(heldout_loader.GIT_SETTINGS) :]
    assert asked and calls[0][: 3 + len(heldout_loader.GIT_SETTINGS)] == [
        "git",
        "-C",
        str(root),
        *heldout_loader.GIT_SETTINGS,
    ]
    assert str(refused.value) == f"git {' '.join(asked)} could not be run: {error}"


def _log_cannot_be_written(root, cause, monkeypatch):
    calls = []
    monkeypatch.setattr(load_asr, "from_bytes", lambda content: calls.append(content))
    result = None
    with pytest.raises(SealRefused, match="openings log could not be written") as refused:
        result = heldout_loader.asr(root)
    assert isinstance(refused.value.__cause__, cause)
    assert result is None and calls == []


def test_a_log_that_cannot_be_written_returns_nothing(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    log = heldout_loader.log_path(root)
    log.write_text("", encoding="utf-8")
    log.chmod(0o444)
    _log_cannot_be_written(root, PermissionError, monkeypatch)
    assert log.read_text(encoding="utf-8") == ""


def test_a_log_path_that_is_a_folder_returns_nothing(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.log_path(root).mkdir()
    _log_cannot_be_written(root, IsADirectoryError, monkeypatch)


def test_refuses_a_differing_hash_without_parsing(tmp_path, monkeypatch):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT}, listed={ASR_FILE: b"other bytes\n"})
    calls = []
    monkeypatch.setattr(load_asr, "from_bytes", lambda content: calls.append(content))
    result = None
    with pytest.raises(SealRefused, match="SHA-256 differs"):
        result = heldout_loader.asr(root)
    assert result is None and calls == []
    assert heldout_loader.logged_openings(root) == []
    assert not heldout_loader.log_path(root).exists()


def test_an_opening_returns_the_bytes_and_logs_one_line_with_every_field(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    head = git(root, "rev-parse", "HEAD")
    before = datetime.now(UTC)
    assert heldout_loader.open_sealed(root, *ASR_FILE) == CONTENT
    after = datetime.now(UTC)
    lines = heldout_loader.log_path(root).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    line = json.loads(lines[0])
    assert set(line) == {"time_utc", "set", "file", "sha256", "commit", "tag_commit", "command"}
    assert line["time_utc"].endswith("Z")
    when = datetime.strptime(line["time_utc"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert before <= when <= after
    assert (line["set"], line["file"]) == ASR_FILE
    assert line["sha256"] == hashlib.sha256(CONTENT).hexdigest()
    assert line["commit"] == head and line["tag_commit"] == head
    assert line["command"] == sys.argv
    assert heldout_loader.logged_openings(root) == [line]


def test_two_openings_give_two_lines_and_the_tree_stays_clean(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.open_sealed(root, *ASR_FILE)
    heldout_loader.open_sealed(root, *ASR_FILE)
    openings = heldout_loader.logged_openings(root)
    assert len(openings) == 2
    assert [o["file"] for o in openings] == [ASR_FILE[1], ASR_FILE[1]]
    assert git(root, "status", "--porcelain") == ""


def test_the_log_lies_beside_the_sealed_folder_not_in_it(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    path = heldout_loader.log_path(root)
    assert path.parent == root / "data"
    assert root / "data" / "heldout" not in path.parents


def test_an_absent_or_empty_log_reads_as_no_opening(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert heldout_loader.logged_openings(root) == []
    heldout_loader.log_path(root).write_text("", encoding="utf-8")
    assert heldout_loader.logged_openings(root) == []


def test_a_damaged_log_line_is_an_error(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    heldout_loader.log_path(root).write_text("not a log line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        heldout_loader.logged_openings(root)


def test_made_up_repositories_lie_in_the_temporary_folder(tmp_path):
    root = sealed_repo(tmp_path, {ASR_FILE: CONTENT})
    assert is_temporary(root) and is_temporary(_sealed_file(root))
    assert not is_temporary(TOP)
    with pytest.raises(AssertionError, match="temporary folder"):
        sealed_repo(TOP / "not-made", {ASR_FILE: CONTENT})
    assert not (TOP / "not-made").exists()

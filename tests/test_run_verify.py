"""The run verifier accepts a results file only when its run's manifest still vouches for it.

Runs are made with the recorder in a throwaway git repository inside pytest's temporary folder,
with git's own configuration switched off.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve import run_verify
from fleetreserve.run_record import record_run
from fleetreserve.run_verify import (
    CannotCheck,
    NotVerified,
    main,
    read_verified,
    require_verified,
)

REQUIRED_FIELDS = ["commit", "command", "start", "end", "exit_code", "outputs"]


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
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


WRITE_RESULT = (
    "import os; folder = os.environ['FLEETRESERVE_RUN_DIR']; "
    "os.makedirs(os.path.join(folder, 'sub')); "
    "open(os.path.join(folder, 'result.txt'), 'w').write('42\\n'); "
    "open(os.path.join(folder, 'sub', 'deep.txt'), 'w').write('deep\\n')"
)


def record(repo: Path, label: str = "run", extra: str = "") -> Path:
    """Record a run that writes result.txt; return its run folder."""
    command = [sys.executable, "-c", WRITE_RESULT + extra]
    before = set((repo / "runs.noindex").glob("*")) if (repo / "runs.noindex").exists() else set()
    record_run(label, command, start_dir=repo)
    (folder,) = set((repo / "runs.noindex").glob("*")) - before
    return folder


def manifest_of(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text())


def rewrite_manifest(folder: Path, change) -> None:
    manifest = manifest_of(folder)
    change(manifest)
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_untouched_results_file_is_verified(repo):
    folder = record(repo)
    manifest = require_verified(folder / "result.txt")
    assert manifest == manifest_of(folder)
    assert manifest["exit_code"] == 0


def test_file_in_a_subfolder_of_the_run_is_verified(repo):
    folder = record(repo)
    assert require_verified(folder / "sub" / "deep.txt")["commit"] == manifest_of(folder)["commit"]


def test_the_log_file_is_verified_too(repo):
    folder = record(repo)
    require_verified(folder / "log.txt")


def test_relative_path_with_dots_is_verified(repo, monkeypatch):
    folder = record(repo)
    monkeypatch.chdir(folder)
    require_verified(Path("sub") / ".." / "result.txt")


def test_changed_byte_is_not_verified(repo):
    folder = record(repo)
    with (folder / "result.txt").open("ab") as handle:
        handle.write(b"0")
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "hash" in str(raised.value)


def test_same_length_change_is_not_verified(repo):
    folder = record(repo)
    (folder / "result.txt").write_text("43\n")
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1


def test_run_with_nonzero_exit_code_is_not_verified(repo):
    folder = record(repo, extra="; import sys; sys.exit(4)")
    assert manifest_of(folder)["exit_code"] == 4
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "exit code 4" in str(raised.value)


def test_file_added_after_the_run_is_not_verified(repo):
    folder = record(repo)
    (folder / "added.txt").write_text("late\n")
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "added.txt")
    assert raised.value.exit_status == 1
    assert "not listed" in str(raised.value)


def test_manifest_file_itself_is_not_a_result(repo):
    folder = record(repo)
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "manifest.json")
    assert raised.value.exit_status == 1


def test_interrupted_run_is_not_verified_even_with_exit_code_0(repo):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.update(interrupted="SIGTERM"))
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "SIGTERM" in str(raised.value)


def test_unknown_commit_is_not_verified(repo):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.update(commit="0" * 40))
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "0000000" in str(raised.value)


@pytest.mark.parametrize("commit", ["HEAD", "--help", "main", "abc", ""])
def test_commit_that_is_not_a_full_hex_sha_is_not_verified(repo, commit):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.update(commit=commit))
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1


def test_run_folder_copied_outside_the_repository_is_not_verified(repo, tmp_path):
    folder = record(repo)
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(folder, elsewhere)
    with pytest.raises(NotVerified) as raised:
        require_verified(elsewhere / "result.txt")
    assert raised.value.exit_status == 1


def test_file_with_no_manifest_cannot_be_checked(tmp_path):
    (tmp_path / "result.txt").write_text("42\n")
    with pytest.raises(NotVerified) as raised:
        require_verified(tmp_path / "result.txt")
    assert raised.value.exit_status == 2
    assert "manifest" in str(raised.value)


def test_results_file_copied_out_of_its_run_cannot_be_checked(repo, tmp_path):
    folder = record(repo)
    copy = tmp_path / "copy.txt"
    shutil.copy(folder / "result.txt", copy)
    with pytest.raises(NotVerified) as raised:
        require_verified(copy)
    assert raised.value.exit_status == 2


def test_missing_file_cannot_be_checked(repo):
    folder = record(repo)
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "no_such_file.txt")
    assert raised.value.exit_status == 2


def test_deleted_results_file_cannot_be_checked(repo):
    folder = record(repo)
    (folder / "result.txt").unlink()
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 2


@pytest.mark.parametrize("text", ["", "not json", "{", "[]", "null", '"x"'])
def test_manifest_that_is_not_a_json_object_cannot_be_checked(repo, text):
    folder = record(repo)
    (folder / "manifest.json").write_text(text)
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 2


def test_manifest_with_bytes_that_are_not_text_cannot_be_checked(repo):
    folder = record(repo)
    (folder / "manifest.json").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 2


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_manifest_lacking_a_field_cannot_be_checked(repo, field):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.pop(field))
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 2
    assert field in str(raised.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exit_code", "0"),
        ("exit_code", True),
        ("exit_code", None),
        ("outputs", []),
        ("outputs", {"result.txt": 5}),
        ("commit", 7),
        ("command", "sh"),
    ],
)
def test_manifest_with_a_wrongly_typed_field_cannot_be_checked(repo, field, value):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.update({field: value}))
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 2


def test_require_verified_returns_the_manifest_when_passing_and_raises_when_not(repo):
    folder = record(repo)
    assert require_verified(str(folder / "result.txt"))["exit_code"] == 0
    (folder / "result.txt").write_text("tampered\n")
    with pytest.raises(NotVerified):
        require_verified(str(folder / "result.txt"))


def test_main_prints_one_line_per_file_and_exits_0_when_all_are_verified(repo, capsys):
    folder = record(repo)
    capsys.readouterr()  # drop the recorder's own printed lines
    files = [str(folder / "result.txt"), str(folder / "sub" / "deep.txt")]
    assert main(files) == 0
    manifest = manifest_of(folder)
    sha7 = manifest["commit"][:7]
    assert capsys.readouterr().out.splitlines() == [
        f"verified: {file} (commit {sha7}, {manifest['start']})" for file in files
    ]


def test_main_prints_the_reason_and_exits_1_for_a_changed_file(repo, capsys):
    folder = record(repo)
    capsys.readouterr()  # drop the recorder's own printed lines
    (folder / "result.txt").write_text("tampered\n")
    assert main([str(folder / "result.txt")]) == 1
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"NOT VERIFIED: {folder / 'result.txt'}: ")
    assert "hash" in line


def test_main_exits_2_for_a_missing_file(repo, capsys):
    missing = repo / "nothing.txt"
    assert main([str(missing)]) == 2
    assert capsys.readouterr().out.startswith(f"NOT VERIFIED: {missing}: ")


def test_main_exits_2_without_arguments():
    with pytest.raises(SystemExit) as raised:
        main([])
    assert raised.value.code == 2


def test_main_exit_1_wins_over_2(repo, capsys):
    folder = record(repo)
    capsys.readouterr()  # drop the recorder's own printed lines
    good = folder / "result.txt"
    (folder / "sub" / "deep.txt").write_text("tampered\n")
    changed = folder / "sub" / "deep.txt"
    missing = repo / "nothing.txt"
    assert main([str(good), str(missing)]) == 2
    assert main([str(good), str(changed)]) == 1
    assert main([str(missing), str(changed), str(good)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert sum(line.startswith("verified: ") for line in lines) == 3
    assert sum(line.startswith("NOT VERIFIED: ") for line in lines) == 4


def test_command_line_exit_statuses(repo):
    folder = record(repo)

    def run(*files: str) -> int:
        done = subprocess.run(
            [sys.executable, "-m", "fleetreserve.run_verify", *files],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return done.returncode

    assert run(str(folder / "result.txt")) == 0
    assert run() == 2
    assert run(str(repo / "nothing.txt")) == 2
    (folder / "result.txt").write_text("tampered\n")
    assert run(str(folder / "result.txt")) == 1


# ---- runs that hit their time limit ------------------------------------------------------------


def record_limited(repo: Path, extra: str, label: str = "limited") -> Path:
    """Record a run that writes result.txt, then runs `extra`, under a limit of 2 seconds."""
    command = [sys.executable, "-c", WRITE_RESULT + extra]
    record_run(label, command, max_seconds=2, grace_seconds=10, start_dir=repo)
    (folder,) = (repo / "runs.noindex").glob(f"*-{label}")
    return folder


SLEEP_LONG = "; import time; time.sleep(30)"
TRAP_AND_EXIT_0 = (
    "; import signal, sys, time; "
    "signal.signal(signal.SIGTERM, lambda number, frame: sys.exit(0)); time.sleep(30)"
)


def test_run_that_hit_its_time_limit_is_not_verified(repo):
    folder = record_limited(repo, SLEEP_LONG)
    manifest = manifest_of(folder)
    assert manifest["timed_out"] is True
    assert manifest["exit_code"] != 0
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "time limit" in str(raised.value)


def test_timed_out_run_that_exited_0_is_not_verified(repo):
    folder = record_limited(repo, TRAP_AND_EXIT_0)
    manifest = manifest_of(folder)
    assert manifest["exit_code"] == 0
    assert manifest["timed_out"] is True
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "time limit" in str(raised.value)


def test_manifest_saying_timed_out_is_not_verified_whatever_its_exit_code(repo):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.update(timed_out=True))
    assert manifest_of(folder)["exit_code"] == 0
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 1
    assert "time limit" in str(raised.value)


def test_main_prints_the_reason_and_exits_1_for_a_timed_out_run(repo, capsys):
    folder = record_limited(repo, SLEEP_LONG)
    capsys.readouterr()  # drop the recorder's own printed lines
    assert main([str(folder / "result.txt")]) == 1
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"NOT VERIFIED: {folder / 'result.txt'}: ")
    assert "time limit" in line


def test_run_that_finished_inside_its_limit_is_verified(repo):
    command = [sys.executable, "-c", WRITE_RESULT]
    record_run("quick", command, max_seconds=60, start_dir=repo)
    (folder,) = (repo / "runs.noindex").glob("*-quick")
    manifest = require_verified(folder / "result.txt")
    assert manifest["timed_out"] is False
    assert manifest["max_seconds"] == 60


def test_manifest_written_before_the_time_limit_existed_is_still_verified(repo):
    folder = record(repo)

    def make_old(manifest: dict) -> None:
        manifest.pop("timed_out")
        manifest.pop("max_seconds")

    rewrite_manifest(folder, make_old)
    assert "timed_out" not in manifest_of(folder)
    assert require_verified(folder / "result.txt")["exit_code"] == 0
    assert main([str(folder / "result.txt")]) == 0


@pytest.mark.parametrize("value", ["true", 1, None, [], "false"])
def test_manifest_with_a_wrongly_typed_timed_out_cannot_be_checked(repo, value):
    folder = record(repo)
    rewrite_manifest(folder, lambda manifest: manifest.update(timed_out=value))
    with pytest.raises(NotVerified) as raised:
        require_verified(folder / "result.txt")
    assert raised.value.exit_status == 2


# --- read_verified: one read, its bytes hashed; no link at a folder on the default path ----------


def changed_after_the_hash(monkeypatch, file: Path) -> bytes:
    """`file`'s bytes now; the file is given other bytes in the commit check, which runs after
    the file's hash is compared with the manifest's."""
    before = file.read_bytes()
    real = run_verify._check_commit

    def then_changed(run_folder, commit):
        file.write_bytes(b"changed after its check\n")
        real(run_folder, commit)

    monkeypatch.setattr(run_verify, "_check_commit", then_changed)
    return before


def test_read_verified_gives_the_bytes_it_hashed_not_a_later_read(repo, monkeypatch):
    """The default reader: a change after the file's one read never reaches the caller."""
    folder = record(repo)
    before = changed_after_the_hash(monkeypatch, folder / "result.txt")
    manifest, data = read_verified(folder / "result.txt")
    assert data == before == b"42\n"
    assert manifest == manifest_of(folder)
    assert (folder / "result.txt").read_bytes() == b"changed after its check\n"


def test_read_verified_reads_the_file_once_through_the_reader_given(repo, monkeypatch):
    """The file first, then the manifest, each once: with a reader given, the reader is the only
    look at the file, and a file it does not find is refused as missing before any manifest is
    looked for (`tests/test_read_verified_order.py`), so its one read comes first."""
    folder = record(repo)
    before = changed_after_the_hash(monkeypatch, folder / "result.txt")
    reads = []

    def reader(path: Path) -> bytes:
        reads.append(Path(path).name)
        return Path(path).read_bytes()

    _, data = read_verified(folder / "result.txt", reader)
    assert data == before
    assert reads == ["result.txt", "manifest.json"]


def test_read_verified_refuses_a_link_at_a_folder_between_the_run_folder_and_the_file(repo):
    """`sub` moved out of the run folder, inside the repository, and a symbolic link to it left in
    its place: the file has the same bytes and the manifest still lists it, but the default
    reader never follows the link."""
    folder = record(repo)
    kept = repo / "kept-sub"
    shutil.move(folder / "sub", kept)
    (folder / "sub").symlink_to(kept)
    with pytest.raises(CannotCheck, match="^sub is a symbolic link, never followed$"):
        read_verified(folder / "sub" / "deep.txt")


def test_read_verified_refuses_a_run_folder_that_is_a_link(repo):
    """The run folder moved, inside the repository, and a symbolic link to it left in its place:
    never followed by the default reader."""
    folder = record(repo)
    kept = repo / "kept-run"
    shutil.move(folder, kept)
    folder.symlink_to(kept)
    assert require_verified(kept / "result.txt")["exit_code"] == 0  # the moved folder verifies
    with pytest.raises(CannotCheck, match=f"^{folder.name} is a symbolic link, never followed$"):
        read_verified(folder / "result.txt")

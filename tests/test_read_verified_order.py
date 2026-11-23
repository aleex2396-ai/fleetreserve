"""`run_verify.read_verified` refuses a file that is not there as "the file is missing" before it
looks for any manifest, through its own reader (no `read`) and through a given reader alike; a
file that is there with no manifest in any parent folder is refused as having none.

The look at the file follows no link: without a `read` it is an `lstat`, the rule of the read
itself (`O_NOFOLLOW` on the file), so a link at the file counts as there and is then refused by
name, never followed to a missing or a sealed target; with a `read` the file is looked at only
through that reader, which is the one read of its bytes (a reader's own refusal is raised where the
bytes are hashed, as before). Run folders and manifests are written by hand; the recorded commit is
not looked up (`_check_commit` is replaced).
"""

import errno
import hashlib
import json
from pathlib import Path

import pytest
from sealed_watch import watch

from fleetreserve import plain_read, run_verify
from fleetreserve.run_verify import CannotCheck, read_verified

MISSING = "the file is missing"
NO_MANIFEST = "no manifest.json in any parent folder of the file"
READERS = ["own reader", "links reader"]


@pytest.fixture(autouse=True)
def unchecked_commit(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def no_manifest_above(file: Path) -> bool:
    return not [place for place in file.parents if (place / "manifest.json").exists()]


def write_run(folder: Path, files: dict[str, bytes]) -> Path:
    """A run folder holding `files` and a manifest listing each with its SHA-256."""
    folder.mkdir(parents=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    outputs = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    manifest = {
        "commit": "a" * 40,
        "command": ["made-up"],
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T00:00:01Z",
        "exit_code": 0,
        "outputs": outputs,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def reader_of(kind: str, top: Path):
    return None if kind == "own reader" else (lambda path: plain_read.read(top, path))


def refusal(file: Path, read) -> str:
    with pytest.raises(CannotCheck) as raised:
        read_verified(file, read)
    return str(raised.value)


@pytest.mark.parametrize("kind", READERS)
def test_a_file_not_there_is_missing_before_any_manifest_is_looked_for(tmp_path, kind):
    file = tmp_path / "20260101T000000Z-made-up" / "result.json"  # no such folder
    assert no_manifest_above(file)
    assert refusal(file, reader_of(kind, tmp_path)) == MISSING


@pytest.mark.parametrize("kind", READERS)
def test_a_file_there_with_no_manifest_anywhere_is_refused_as_having_none(tmp_path, kind):
    file = tmp_path / "result.json"
    file.write_text("{}")
    assert no_manifest_above(file)
    assert refusal(file, reader_of(kind, tmp_path)) == NO_MANIFEST


@pytest.mark.parametrize("kind", READERS)
def test_a_file_removed_from_its_run_folder_is_missing(tmp_path, kind):
    run = write_run(tmp_path / "run", {"result.json": b"{}"})
    (run / "result.json").unlink()
    assert refusal(run / "result.json", reader_of(kind, tmp_path)) == MISSING


@pytest.mark.parametrize("kind", READERS)
def test_a_file_there_in_its_run_folder_is_read_once_and_verified(tmp_path, kind):
    run = write_run(tmp_path / "run", {"result.json": b'{"a": 1}'})
    read = reader_of(kind, tmp_path)
    calls = []

    def counted(path):
        calls.append(Path(path))
        return read(path)

    manifest, data = read_verified(run / "result.json", None if read is None else counted)
    assert (manifest["exit_code"], data) == (0, b'{"a": 1}')
    if read is not None:
        assert sorted(calls) == [run / "manifest.json", run / "result.json"]


def test_a_dangling_link_at_the_file_is_not_followed_to_call_it_missing(tmp_path):
    """Own reader: a link at the file whose target does not exist is there (an `lstat` sees the
    link); with no manifest anywhere it is refused as having none, and in a run folder by name as
    a link. Following the link would call it missing."""
    file = tmp_path / "result.json"
    file.symlink_to(tmp_path / "nowhere.json")
    assert no_manifest_above(file)
    assert refusal(file, None) == NO_MANIFEST
    run = write_run(tmp_path / "run", {"result.json": b"{}"})
    (run / "result.json").unlink()
    (run / "result.json").symlink_to(tmp_path / "nowhere.json")
    assert refusal(run / "result.json", None) == "result.json is a symbolic link, never followed"


def test_a_link_at_the_file_into_a_sealed_folder_is_never_opened(tmp_path):
    """Own reader: a link at the file to a made-up sealed file is looked at, never opened, and
    with no manifest anywhere is refused as having none."""
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "result.json").write_text("{}")
    file = tmp_path / "result.json"
    file.symlink_to(sealed / "result.json")
    with watch(sealed) as seen:
        problem = refusal(file, None)
    assert (problem, seen) == (NO_MANIFEST, [])


def test_through_a_reader_a_file_the_reader_cannot_find_is_missing(tmp_path):
    """With a `read`, the file is looked at only through it: a file on disk, with no manifest
    anywhere, that the reader finds missing is missing, before any manifest is looked for."""
    file = tmp_path / "result.json"
    file.write_text("{}")
    assert no_manifest_above(file)
    calls = []

    def finds_nothing(path):
        calls.append(Path(path))
        raise FileNotFoundError(2, "No such file or directory", str(path))

    assert refusal(file, finds_nothing) == MISSING
    assert calls == [file]


def test_through_a_reader_a_file_only_the_reader_holds_is_there_and_read_once(tmp_path):
    """With a `read`, a file not on disk that the reader gives is there: its bytes are those the
    reader gave, read once, and the manifest is read once through it."""
    held = b'{"held": true}'
    run = write_run(tmp_path / "run", {"result.json": held})
    (run / "result.json").unlink()
    calls = []

    def holds(path):
        calls.append(Path(path))
        return held if Path(path) == run / "result.json" else plain_read.read(tmp_path, path)

    manifest, data = read_verified(run / "result.json", holds)
    assert (manifest["exit_code"], data) == (0, held)
    assert sorted(calls) == [run / "manifest.json", run / "result.json"]


def test_through_a_reader_a_dangling_link_at_the_file_is_refused_by_the_reader(tmp_path):
    """With the links reader, a link at the file whose target does not exist is never followed
    to call it missing: the reader's refusal of the link is raised where the bytes are hashed."""
    run = write_run(tmp_path / "run", {"result.json": b"{}"})
    (run / "result.json").unlink()
    (run / "result.json").symlink_to(tmp_path / "nowhere.json")
    assert refusal(run / "result.json", reader_of("links reader", tmp_path)) == (
        "the file cannot be read: run/result.json is a symbolic link"
    )


def test_through_a_reader_its_refusal_keeps_its_place_after_the_manifest_checks(tmp_path):
    """With the links reader, a link at the file in a run that hit its time limit is refused for
    the time limit first, as when the file was read after the manifest checks."""
    run = write_run(tmp_path / "run", {"result.json": b"{}"})
    (run / "result.json").unlink()
    (run / "result.json").symlink_to(tmp_path / "nowhere.json")
    manifest = json.loads((run / "manifest.json").read_text())
    (run / "manifest.json").write_text(json.dumps({**manifest, "timed_out": True}))
    with pytest.raises(run_verify.NotVerified) as raised:
        read_verified(run / "result.json", reader_of("links reader", tmp_path))
    assert str(raised.value) == "the run hit its time limit and was stopped"


def refused_with_exit_two(file: Path) -> None:
    with pytest.raises(CannotCheck) as raised:
        read_verified(file)
    assert raised.value.exit_status == 2
    assert str(raised.value) != MISSING


def test_a_loop_of_folder_links_in_the_run_is_cannot_check_never_a_raw_oserror(tmp_path):
    folder = write_run(tmp_path / "20260101T000000Z-made-up", {"ok.json": b"{}"})
    (folder / "a").symlink_to(folder / "b")
    (folder / "b").symlink_to(folder / "a")
    refused_with_exit_two(folder / "a" / "x.json")


def test_a_folder_the_look_may_not_enter_is_cannot_check_never_a_raw_oserror(tmp_path):
    folder = write_run(tmp_path / "20260101T000000Z-made-up", {"ok.json": b"{}"})
    (folder / "sub").mkdir()
    (folder / "sub" / "x.json").write_bytes(b"{}")
    (folder / "sub").chmod(0)
    try:
        refused_with_exit_two(folder / "sub" / "x.json")
    finally:
        (folder / "sub").chmod(0o755)


def test_any_other_oserror_of_the_look_is_cannot_check(tmp_path, monkeypatch):
    folder = write_run(tmp_path / "20260101T000000Z-made-up", {"x.json": b"{}"})
    file = folder / "x.json"
    real = run_verify.os.lstat

    def failing(path, *args, **kwargs):
        if Path(path) == file:
            raise OSError(errno.EIO, "made-up input/output error")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(run_verify.os, "lstat", failing)
    refused_with_exit_two(file)

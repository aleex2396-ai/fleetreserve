"""`wp16_drive`: which drive WP16 reads, and how it reads a run's files (made-up drive logs).

Each test is red when the reading is broken: a drive is taken only when its end line's hash
recomputes; the one complete drive is taken; with none, the newest drive that ran to its end is
read and its failed and not-run steps are named; a stopped drive, two complete drives and a later
drive that ran a step are refused; a file is read only when its run's manifest has the SHA-256 the
drive log records and lists the file with the SHA-256 it has now.
"""

import pytest
from wp16_made_up import COMMIT, MadeUp, sha

from fleetreserve.wp16_drive import Missed, NoDrive, Source, find_drive, parse_log

A = {"x.json": {"k": 1}}
B = {"y.json": {"k": 2}}


def test_the_complete_drive_is_found(tmp_path):
    MadeUp(tmp_path).drive([("a", A), ("b", B)])
    drive = find_drive(tmp_path)
    assert drive.complete
    assert list(drive.steps) == ["a", "b"]
    assert drive.commit == COMMIT
    assert drive.problems() == []


def test_an_incomplete_drive_is_read_and_its_failures_are_named(tmp_path):
    MadeUp(tmp_path).drive([("a", A), ("b", "fail"), ("c", "skip")])
    drive = find_drive(tmp_path)
    assert not drive.complete
    assert drive.problems() == ["failed: b (exit status 1)", "not run: c (an earlier step)"]
    source = Source(tmp_path, drive)
    assert source.json("a", "x.json") == {"k": 1}
    with pytest.raises(Missed, match=r"failed: b \(exit status 1\)"):
        source.output("b", "y.json")
    with pytest.raises(Missed, match="not run: c"):
        source.output("c", "z.json")
    assert source.why_not("a") is None
    assert source.why_not("nope").startswith("no entry writes it")


def test_a_stopped_drive_is_refused(tmp_path):
    MadeUp(tmp_path).drive([("a", A), ("b", "skip")], stop=True)
    with pytest.raises(NoDrive, match="stopped"):
        find_drive(tmp_path)


def test_a_complete_drive_after_a_stopped_one_is_the_run(tmp_path):
    made = MadeUp(tmp_path)
    made.drive([("a", A), ("b", "skip")], stamp="20261123T020000Z", stop=True)
    made.drive([("a", A), ("b", B)], stamp="20261123T120000Z")
    assert find_drive(tmp_path).name == "20261123T120000Z-sealed-run.jsonl"


def test_two_complete_drives_are_refused(tmp_path):
    made = MadeUp(tmp_path)
    made.drive([("a", A)], stamp="20261123T020000Z")
    made.drive([("a", A)], stamp="20261123T120000Z")
    with pytest.raises(NoDrive, match="2 complete"):
        find_drive(tmp_path)


def test_a_later_drive_that_ran_a_step_is_refused(tmp_path):
    made = MadeUp(tmp_path)
    made.drive([("a", A)], stamp="20261123T020000Z")
    made.drive([("a", "fail")], stamp="20261123T120000Z")
    with pytest.raises(NoDrive, match="after the complete drive"):
        find_drive(tmp_path)


def test_no_drive_log_is_refused(tmp_path):
    with pytest.raises(NoDrive, match="no sealed-run drive log"):
        find_drive(tmp_path)


def test_a_log_whose_hash_does_not_recompute_is_not_taken(tmp_path):
    path = MadeUp(tmp_path).drive([("a", A)])
    data = path.read_bytes()
    assert b"made-up" in data
    with pytest.raises(NoDrive, match="does not recompute"):
        parse_log(path.name, data.replace(b"made-up", b"made-uq", 1))
    path.write_bytes(data.replace(b"made-up", b"made-uq", 1))
    with pytest.raises(NoDrive, match="no sealed-run drive ran a step"):
        find_drive(tmp_path)


def test_a_log_that_is_cut_off_is_not_taken(tmp_path):
    path = MadeUp(tmp_path).drive([("a", A)])
    with pytest.raises(NoDrive, match="cut off"):
        parse_log(path.name, path.read_bytes()[:-1])


def test_an_output_that_changed_is_not_read(tmp_path):
    MadeUp(tmp_path).drive([("a", A)])
    source = Source(tmp_path, find_drive(tmp_path))
    (folder,) = (tmp_path / "runs.noindex").glob("*-a")
    (folder / "x.json").write_text('{"k": 2}')
    with pytest.raises(Missed, match="not the file its run wrote"):
        source.output("a", "x.json")


def test_a_manifest_the_log_does_not_record_is_not_read(tmp_path):
    MadeUp(tmp_path).drive([("a", A)])
    source = Source(tmp_path, find_drive(tmp_path))
    (folder,) = (tmp_path / "runs.noindex").glob("*-a")
    (folder / "manifest.json").write_text("{}")
    with pytest.raises(Missed, match="manifest is not the one the log records"):
        source.output("a", "x.json")


def test_a_file_the_manifest_does_not_list_is_not_read(tmp_path):
    MadeUp(tmp_path).drive([("a", A)])
    source = Source(tmp_path, find_drive(tmp_path))
    with pytest.raises(Missed, match="does not list it"):
        source.output("a", "other.json")


def test_every_file_read_is_recorded_with_its_hash(tmp_path):
    made = MadeUp(tmp_path)
    data = made.commit_file("run-lists/run-list.json", {"units": {}})
    made.drive([("a", A)])
    source = Source(tmp_path, find_drive(tmp_path))
    source.output("a", "x.json")
    source.committed("run-lists/run-list.json")
    (folder,) = (tmp_path / "runs.noindex").glob("*-a")
    key = f"runs.noindex/{folder.name}/x.json"
    assert source.hashes[key] == sha((folder / "x.json").read_bytes())
    assert source.hashes["run-lists/run-list.json"] == sha(data)


def test_a_link_on_the_path_is_never_followed(tmp_path):
    MadeUp(tmp_path).drive([("a", A)])
    source = Source(tmp_path, find_drive(tmp_path))
    (folder,) = (tmp_path / "runs.noindex").glob("*-a")
    real = folder / "real.json"
    real.write_text('{"k": 1}')
    (folder / "x.json").unlink()
    (folder / "x.json").symlink_to(real)
    with pytest.raises(Missed, match="symbolic link"):
        source.output("a", "x.json")

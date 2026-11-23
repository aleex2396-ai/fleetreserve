"""`tools/wp15_check.py` on a complete sealed-run drive that WP14's pass mark fails although every
part of WP15 reads (decision 99 (5): WP15 is never passed while WP14 fails). WP15 runs WP14's own
check on the same drive (its output collected) and, when that does not exit 0, exits 1 naming each
line of it that failed, begun "WP14:". One test for each of WP14's three marks:

1. run: a learn step other than those of parts (a) to (c) exited 0 with a failed configuration in
   its `learn.json` (on the held-out route a configuration that raised an error on one unit leaves
   its learn step's exit status at 0, `learn_run`), named with step, unit, configuration, error;
2. logged parsing fixes: an earlier drive's start line names another tag commit;
3. logs hashed: an earlier drive's manifest is not the one its log records.

The drives are those of `test_wp15_check` (the real driver through the real gate, in a throwaway
repository frozen at a made-up tag). The earlier drive of tests 2 and 3 ended on a failed step, so
the complete drive after it is the run WP15 reads, and only WP14 reads the earlier one.
"""

import os
from pathlib import Path

import test_wp15_check as base
from test_wp15_check import isolated_git  # noqa: F401  (autouse fixture)

check = base.check
EARLIER = "learn-earlier_gbm"
UNIT = "korea-work"
ERROR = "RuntimeError: made to fail"
CONFIG = base.CONFIGURATIONS[1]
NAMED = f"learn step {EARLIER}: configuration {CONFIG} failed on unit {UNIT}: {ERROR}"


def with_earlier_learn(tmp_path, monkeypatch, *, failed: bool) -> Path:
    """The made-up repository whose run list has one more learn step, `learn-earlier_gbm`, with a
    `learn.json` that lists a failed configuration when `failed`."""
    found = [*base.entries(), base.learn_entry(label=EARLIER)]
    root = base.made(tmp_path, monkeypatch, found)
    document = base.learn_json(base.nine(), (root / base.MAIN).read_bytes())
    if failed:
        document["units"][UNIT]["configurations"][CONFIG]["rungs"] = {"reason": f"failed: {ERROR}"}
        document["failed_configurations"] = [f"{CONFIG} | {UNIT}: {ERROR}"]
    base.put(root, EARLIER, "learn.json", document)
    base.drive(0)  # every step exits 0: the drive is complete
    return root


def wp14_says(root: Path, capsys) -> tuple[int, list[str]]:
    """WP14's own check on the same drive: its exit status and its lines."""
    folder = root / "runs.noindex" / "recorder-run"
    os.environ[check.wp14.RUN_DIR_VARIABLE] = str(folder)
    folder.mkdir(parents=True, exist_ok=True)
    capsys.readouterr()
    try:
        status = check.wp14.main(["--root", str(root)])
    finally:
        del os.environ[check.wp14.RUN_DIR_VARIABLE]
    return status, capsys.readouterr().out.splitlines()


def test_1_a_failed_configuration_of_another_learn_step_fails_wp15_as_it_fails_wp14(
    tmp_path, monkeypatch, capsys
):
    root = with_earlier_learn(tmp_path, monkeypatch, failed=True)
    status14, said14 = wp14_says(root, capsys)
    assert status14 == 1, said14
    assert any(NAMED in x for x in said14), said14
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    (named,) = [x for x in lines if NAMED in x]
    assert named.startswith("FAILED: WP14: "), named
    assert lines[-1].startswith("FAILED: WP15, 3 of 3 parts reported"), lines
    assert not any(x.startswith("passed: ") for x in lines), lines


def test_the_parts_are_still_printed_when_another_learn_step_failed(tmp_path, monkeypatch, capsys):
    root = with_earlier_learn(tmp_path, monkeypatch, failed=True)
    _, lines = base.run(root, capsys)
    for part in check.PARTS:
        assert f"{part}: reported" in lines, lines


def test_a_drive_whose_other_learn_step_lists_no_failure_passes_as_before(
    tmp_path, monkeypatch, capsys
):
    root = with_earlier_learn(tmp_path, monkeypatch, failed=False)
    status, lines = base.run(root, capsys)
    assert status == 0, lines
    assert lines[-1].startswith("passed: WP15, 3 of 3 parts reported"), lines
    assert not any("learn step" in x for x in lines), lines


def earlier_then_complete(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """A drive that ended on a failed step, then the complete drive; (the root, the earlier log)."""
    root = base.made(tmp_path, monkeypatch)
    base.put(root, check.learn_label(0.2), "exit", "1\n")
    base.put(root, "sealed-truth", "no-opening", "1\n")  # the earlier drive opened nothing
    base.drive(1)
    (root / "data" / "made-up" / check.learn_label(0.2) / "exit").unlink()
    (root / "data" / "made-up" / "sealed-truth" / "no-opening").unlink()
    base.drive(0)
    first, _ = base.logs(root)
    return root, first


def test_2_an_earlier_drive_with_another_tag_commit_fails_wp15_as_it_fails_wp14(
    tmp_path, monkeypatch, capsys
):
    root, first = earlier_then_complete(tmp_path, monkeypatch)
    base.rewritten(first, lambda records: records[0].update(tag_commit="0" * 40))
    status14, said14 = wp14_says(root, capsys)
    assert status14 == 1, said14
    assert "logged parsing fixes: failed" in said14, said14
    assert "one logged run: passed" in said14 and "logs hashed: passed" in said14, said14
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith(f"FAILED: WP14: {first.name} was driven with the tag's commit ") for x in lines
    ), lines
    assert lines[-1].startswith("FAILED: WP15, 3 of 3 parts reported"), lines


def test_3_an_earlier_drives_manifest_changed_fails_wp15_as_it_fails_wp14(
    tmp_path, monkeypatch, capsys
):
    root, first = earlier_then_complete(tmp_path, monkeypatch)
    entries = [r for r in base.drive_fake.read_log(first) if "manifest_sha256" in r]
    manifest = root / entries[-1]["run"] / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    status14, said14 = wp14_says(root, capsys)
    assert status14 == 1, said14
    assert "logs hashed: failed" in said14, said14
    assert "one logged run: passed" in said14, said14
    assert "logged parsing fixes: passed" in said14, said14
    status, lines = base.run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith("FAILED: WP14: ") and "is not the one the log records" in x for x in lines
    ), lines
    assert lines[-1].startswith("FAILED: WP15, 3 of 3 parts reported"), lines


def test_the_check_leaves_no_run_folder_and_no_variable_behind(tmp_path, monkeypatch, capsys):
    root = with_earlier_learn(tmp_path, monkeypatch, failed=True)
    before = sorted(p.name for p in (root / "runs.noindex").iterdir())
    base.run(root, capsys)
    assert sorted(p.name for p in (root / "runs.noindex").iterdir()) == before
    assert check.RUN_DIR_VARIABLE not in os.environ

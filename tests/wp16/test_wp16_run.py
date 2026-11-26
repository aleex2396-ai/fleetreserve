"""`python -m fleetreserve.wp16_run` on made-up top folders: files, manifest, banner, exit status.

Red when an output is not named in the manifest with its hash, when a summary read is not named,
when an incomplete drive is written as a passed run, when a failed step is filled from elsewhere or
when a run that cannot find its drive exits 0. The figures are drawn by `test_wp16_figures.py`;
here `render` is replaced by a stand-in that writes a placeholder, so no drawing is needed.
"""

import json

import pytest
from wp16_made_up import (
    COMMIT,
    NAME_MAIN,
    UNITS,
    MadeUp,
    learn_document,
    matrix_same,
    run_list_document,
    sha,
)

from fleetreserve import wp16_figures, wp16_run


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)

    def render(spec, folder):
        name = f"{spec['kind']}.png"
        (folder / name).write_bytes(b"placeholder")
        return name

    monkeypatch.setattr(wp16_figures, "render", render)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)


def made_up_top(tmp_path, learn):
    """A top folder with the run list, a matrix and a drive: the truth build (u3 has no window)
    and the learn step (`learn` is its outcome: files, or "fail")."""
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    truth = {"abstained": {"u3": {"reason": "no window"}}, "units": {"u1": {}, "u2": {}}}
    gate = {
        "configurations": {
            NAME_MAIN: {
                "units": {
                    u: {"fixed_every_request": {"test_mean_offer_kwh_per_day": 0.25}}
                    for u in ("u1", "u2")
                }
            }
        }
    }
    files = {
        "learn.json": learn_document(
            sha(data), {"u1": (1.0, 2.0), "u2": (1.0, 3.0)}, {u: NAME_MAIN for u in UNITS}
        ),
        "learn-gate.json": gate,
    }
    steps = [("sealed-truth", {"truth-summary.json": truth})]
    steps.append(("learn-lgbm_tail", "fail" if learn == "fail" else files))
    if learn == "fail":
        steps.append(("intervals-lgbm_tail", "skip"))
    made.drive(steps)


def test_a_complete_drive_gives_every_file_and_a_manifest_that_names_them(tmp_path):
    made_up_top(tmp_path, "pass")
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["commit"] == COMMIT
    assert manifest["drive"]["complete"] is True and manifest["drive"]["steps_not_passed"] == []
    assert manifest["command"][:3] == ["python", "-m", "fleetreserve.wp16_run"]
    assert manifest["made_apart"]["rows"] == []  # one headline configuration: nothing made apart
    assert manifest["made_apart"]["sentence"].startswith("made apart at a later commit")
    for name in ("o1-summary.csv", "o1-summary.md", "o4-availability-shortfall.csv", "outcomes.md"):
        assert manifest["outputs"][name] == sha((out / name).read_bytes())
    assert "o9.png" in manifest["outputs"]
    read = manifest["summaries_read"]
    assert any(path.endswith("learn.json") for path in read)
    assert "run-lists/run-list.json" in read and "gate/sealed-matrix/lgbm_tail.json" in read
    units = (out / "units.csv").read_text().splitlines()
    assert "u3,none,no window" in units


def test_an_incomplete_drive_is_never_written_as_a_passed_run(tmp_path):
    made_up_top(tmp_path, "fail")
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 1
    page = (out / "outcomes.md").read_text()
    assert page.count("NOT COMPLETE") >= 5  # every table opens with it
    assert "not run: intervals-lgbm_tail (an earlier step)" in page
    summary = (out / "o1-summary.csv").read_text()
    assert "failed: learn-lgbm_tail (exit status 1)" in summary  # the slot says so, no figure
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert manifest["drive"]["complete"] is False
    assert len(manifest["drive"]["steps_not_passed"]) == 2


def test_no_drive_is_could_not_run(tmp_path, capsys):
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_a_stopped_drive_is_could_not_run(tmp_path):
    MadeUp(tmp_path).drive([("sealed-truth", {"truth-summary.json": {}}), ("x", "skip")], stop=True)
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(tmp_path / "out")]) == 2


def test_no_output_folder_is_could_not_run(tmp_path, capsys):
    made_up_top(tmp_path, "pass")
    assert wp16_run.main(["--root", str(tmp_path)]) == 2
    assert "give --out" in capsys.readouterr().out


def test_the_output_folder_can_come_from_the_run_recorder(tmp_path, monkeypatch):
    made_up_top(tmp_path, "pass")
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path / "run"))
    assert wp16_run.main(["--root", str(tmp_path)]) == 0
    assert (tmp_path / "run" / "outcomes.md").is_file()

"""`drift_run` on a held-out unit the held-out truth build abstained (module text of
`test_heldout_abstain`; the made-up repository, settings file and builds are those of
`test_heldout_route_drift`).

The run list's drift step names one unit, OSL_T. A unit abstained by the build has no tables, so
the run goes on with the units that have them, or with none: it exits 0 and prints and writes one
line per abstained unit; a run on OSL_T and TRO_R with a Korean portfolio that abstained scores the
two Norwegian sites.
"""

import json
import shutil

import test_heldout_route_drift as drift
from test_heldout_route_scoring import _record

from fleetreserve import drift_path, drift_run

repo = drift.repo
builds = drift.builds
setup = drift.setup

ABSTAINED = "korea-visits"


def _line(s: dict) -> str:
    reason = s["summary"]["abstained"][ABSTAINED]["reason"]
    return (
        f"{ABSTAINED}: abstained in the held-out truth build {s['held_run'].name}: {reason}; "
        "not scored, counted as not passing"
    )


def test_a_run_on_norway_with_an_abstained_portfolio_scores_the_norwegian_sites(
    repo, setup, monkeypatch, capsys
):
    s = setup
    monkeypatch.setattr(drift_path, "N_CALIBRATION", 5)
    arguments = drift._arguments(s, (*drift.UNITS, ABSTAINED), held=drift._held(run=s["held_run"]))
    assert drift_run.main(arguments, root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[3] == _line(s)  # after the route's three lines
    document = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())
    assert sorted(document["units"]) == sorted(drift.UNITS)
    held = document["sealed_route"]["held_out_inputs"]
    assert list(held["abstained"]) == [ABSTAINED]


def test_a_run_whose_only_unit_abstained_goes_on_with_no_unit(repo, setup, monkeypatch, capsys):
    s = setup
    monkeypatch.setattr(drift_path, "N_CALIBRATION", 5)
    arguments = drift._arguments(s, (ABSTAINED,), held=drift._held(run=s["held_run"]))
    assert drift_run.main(arguments, root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert _line(s) in shown
    document = json.loads((s["out"] / drift_run.RESULTS_NAME).read_text())
    assert document["units"] == {} and document["run"]["units"] == []
    reason = f"abstained ({s['summary']['abstained'][ABSTAINED]['reason']})"
    assert document["measure_not_made"] == {ABSTAINED: reason}
    assert (
        f"The drift measure could not be made for {ABSTAINED}: {reason}; counted as not passing."
        in shown
    )
    assert "NOTHING WAS MEASURED: no unit of this run has tables." in shown
    assert list(document["sealed_route"]["held_out_inputs"]["abstained"]) == [ABSTAINED]


def test_a_recorded_build_without_the_abstention_still_could_not_run(repo, setup, tmp_path, capsys):
    s = setup
    run = s["held_run"].parent / "20261117T000000Z-copy"
    shutil.copytree(s["held_run"], run)  # inside the made-up repository: git checks its commit
    summary = json.loads((run / "truth-summary.json").read_text())
    del summary["abstained"][ABSTAINED]
    (run / "truth-summary.json").write_text(json.dumps(summary))
    _record(run, s["tag"])
    arguments = drift._arguments(s, (ABSTAINED,), held=drift._held(run=run))
    assert drift_run.main(arguments, root=repo) == 2
    assert f"COULD NOT RUN: {ABSTAINED} is not in the recorded truth build" in (
        capsys.readouterr().out
    )

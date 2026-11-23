"""A configuration that raises an error on one unit does not fail a whole learn step on the sealed
route: the unit is written as `failed: <error>` under that configuration, counts as not passing
(never as abstained), the other units and configurations are kept, and the step exits 0.

The made-up repository, its tag and the builds are those of `test_heldout_route_scoring`. The run
is `learn_run` on OSL_T and korea-home (both have tables in the made-up held-out build) with the
stand-in model of that test; a stand-in for one stage of the gate raises for korea-home under the
empirical configuration `NAMES[0]` (its oracle partner, `NAMES[1]`, runs).

1. A failure in the preparation of the days, in the calibration and scoring of a job, and in the
   ceilings of a unit (which fails the unit under every configuration): `learn_run` exits 0;
   `failed_configurations` names each configuration and unit with the error; OSL_T's entry is
   what a run without the failure writes; the failed unit's configuration holds
   `failed: <error>` as its reason, in its offer, its E' figures and the gate's decisions, never
   "abstained", and the unit does not pass either rule; the summary shows it.
2. Without a failure the gate's results are the same with and without the sealed route's flag
   (`gate_run.run_gate(held_out=...)`).
3. A failure of the ceilings off the route still ends the run (the error is raised).
"""

import json

import pytest
import test_heldout_route_scoring as scoring
from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import small_gate_units, small_matrix

from fleetreserve import gate_calibrate, gate_config, gate_run, learn_run, run_verify
from fleetreserve.days_from_truth import parse_unit_days

builds = scoring.builds
setup = scoring.setup

SCORED = scoring.SCORED
OTHER = "korea-home"
NAMES = scoring.NAMES
ERROR = "RuntimeError: made to fail"
TEXT = f"failed: {ERROR}"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    from fake_heldout import sealed_repo
    from heldout_truth_fake import sealed_files

    folder = tmp_path_factory.mktemp("failed-route")
    return sealed_repo(folder, sealed_files(folder / "made"))


def _learn(s: dict, repo, capsys) -> tuple[int, dict, dict, str]:
    """`learn_run` on OSL_T and korea-home on the route: (exit status, learn.json, learn-gate.json,
    the summary printed)."""
    matrix = scoring._matrix(s["tmp"], [SCORED, OTHER])
    arguments = scoring._arguments(
        "learn_run", matrix, s["tables"], s["run"], s["held_tables"], s["held_run"]
    )
    status = learn_run.main(arguments, root=repo)
    printed = capsys.readouterr().out
    out = s["out"]
    results = json.loads((out / "learn.json").read_text())
    gate = json.loads((out / "learn-gate.json").read_text())
    return status, results, gate, printed


def _fail_prepare(monkeypatch) -> None:
    real = gate_run._prepare_task

    def task(job):
        unit, _, configuration = job
        if unit == OTHER and configuration.name == NAMES[0]:
            raise RuntimeError("made to fail")
        return real(job)

    monkeypatch.setattr(gate_run, "_prepare_task", task)


def _fail_calibration(monkeypatch) -> None:
    real, seen = gate_calibrate.run_job, []

    def run_job(prepared, *arguments, **options):
        if prepared[0].unit == OTHER and not seen:
            seen.append(1)  # the first job of the unit: the empirical configuration's
            raise RuntimeError("made to fail")
        return real(prepared, *arguments, **options)

    monkeypatch.setattr(gate_calibrate, "run_job", run_job)


def _fail_ceilings(monkeypatch, unit: str = OTHER) -> None:
    real = gate_run._ceiling_task

    def task(job):
        if job[0] == unit:
            raise RuntimeError("made to fail")
        return real(job)

    monkeypatch.setattr(gate_run, "_ceiling_task", task)


def _failed_under(*names: str) -> list[str]:
    return sorted(f"{name} | {OTHER}: {ERROR}" for name in names)


def _check_the_failure(status, results, gate, printed, names) -> None:
    assert status == 0, printed
    assert results["failed_configurations"] == _failed_under(*names)
    assert gate["failed_configurations"] == results["failed_configurations"]
    assert list(results["units"]) == [SCORED, OTHER]
    assert results["units"][SCORED]["configurations"][NAMES[0]]["rungs"]["fixed"]["test"]
    failed = results["units"][OTHER]["configurations"][NAMES[0]]
    assert failed["rungs"] == {"reason": TEXT}
    assert results["units"][OTHER]["e_prime"][NAMES[0]] == {"reason": TEXT}
    entry = gate["configurations"][NAMES[0]]["units"][OTHER]
    assert entry == {"failed": TEXT, "reason": TEXT}
    assert "abstained" not in json.dumps([results["units"][OTHER], entry, gate["gate"]])
    for pair in gate["gate"]["by_pair"].values():  # not passing, with the reason
        for decision in pair["decisions"]:
            row = decision["units"][OTHER]
            assert (row["passes_share"], row["passes_ratio"], row["reason"]) == (False, False, TEXT)
            assert decision["units"][SCORED].get("reason") != TEXT
    assert TEXT in printed and "abstained" not in printed
    assert f"FAILED CONFIGURATIONS: {'; '.join(_failed_under(*names))}" in printed


def _unit_without_failure(s: dict, repo, capsys) -> dict:
    """OSL_T's entry of a run with no failure."""
    status, results, _, printed = _learn(s, repo, capsys)
    assert status == 0 and results["failed_configurations"] == [], printed
    return results["units"][SCORED]


@pytest.mark.parametrize(
    ("fail", "names"),
    [
        (_fail_prepare, (NAMES[0],)),
        (_fail_calibration, (NAMES[0],)),
        (_fail_ceilings, NAMES),
    ],
    ids=["preparation", "calibration", "ceilings"],
)
def test_a_failure_on_one_unit_is_written_as_failed_and_the_step_exits_0(
    repo, setup, capsys, monkeypatch, fail, names
):
    s = setup
    clean = _unit_without_failure(s, repo, capsys)
    fail(monkeypatch)
    status, results, gate, printed = _learn(s, repo, capsys)
    _check_the_failure(status, results, gate, printed, names)
    assert results["units"][SCORED] == clean  # the other unit's results are those without it
    if names == NAMES:  # a failed ceiling fails the unit under every configuration
        entry = gate["configurations"][NAMES[1]]["units"][OTHER]
        assert entry == {"failed": TEXT, "reason": TEXT}
    else:
        partner = results["units"][OTHER]["configurations"][NAMES[0]]["oracle_partner"]
        assert partner["rungs"]["fixed"]["test"]  # the oracle partner ran on the unit


def _small_units(folder, monkeypatch):
    """(matrix, units) of the made-up development gate units (`learn_fake`), read as `gate_run`
    reads them."""
    truth, tables = write_build(folder, small_gate_units(), service_hours=2)
    path = folder / "matrix.json"
    path.write_text(json.dumps(small_matrix()))
    matrix, _ = gate_config.read(path)
    monkeypatch.setattr(run_verify, "_check_commit", lambda found, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    _, table_bytes = gate_run.read_inputs(truth, tables, matrix.gate_units)
    units = {
        name: parse_unit_days(
            table_bytes,
            name,
            rho=matrix.rho,
            no_fit=matrix.no_fit,
            small_fleet=matrix.small_fleet,
        )
        for name in matrix.gate_units
    }
    return matrix, units


def test_without_a_failure_the_flag_changes_nothing(tmp_path, monkeypatch):
    matrix, units = _small_units(tmp_path, monkeypatch)
    kept = {}
    for held_out in (False, True):
        (document, rows, breaks), _ = gate_run.run_gate(
            matrix, units, workers=1, only=None, trial=False, held_out=held_out
        )
        kept[held_out] = (json.dumps(document, sort_keys=True), rows, breaks)
    assert kept[True] == kept[False]
    assert json.loads(kept[True][0])["failed_configurations"] == []


def test_off_the_route_a_failed_ceiling_still_ends_the_run(tmp_path, monkeypatch):
    matrix, units = _small_units(tmp_path, monkeypatch)
    _fail_ceilings(monkeypatch, GATE_UNITS[1])
    with pytest.raises(RuntimeError, match="made to fail"):
        gate_run.run_gate(matrix, units, workers=1, only=None, trial=False)

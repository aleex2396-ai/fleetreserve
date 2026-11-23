"""The readers of a learn step's results on the sealed route show a unit whose configuration raised
an error as `failed: <error>`, counted as not passing, never as abstained.

The run is `learn_run` through the route (`test_heldout_route_ledger`'s stand-in model that gives
boxes, its two configurations `NAMES`) on OSL_T and korea-home, the empirical configuration failing
on korea-home in the preparation of its days; it exits 0 (`test_heldout_learn_failed`). Then:

1. `ledger_run` reads it (exit 0): korea-home holds the reason `failed: <error>` under the failed
   configuration and its numbers under the other; OSL_T's entry is that of a run without the
   failure.
2. `day_intervals` reads it with the run's resamples (exit 0): the failed unit is left out of the
   pooling with the reason `failed: <error>`, without the mark of an abstained unit, and its
   table row says so.
"""

import json
import sys
from pathlib import Path

import pytest
import test_heldout_route_ledger as ledger_tests
import test_heldout_route_scoring as scoring

from fleetreserve import day_intervals, gate_run, ledger_run, run_verify

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp14"))
from ledger_fake import LEARN_MODULE, record  # noqa: E402

builds = scoring.builds
repo = ledger_tests.repo
SCORED = scoring.SCORED
OTHER = "korea-home"
NAMES = ledger_tests.NAMES
ERROR = "RuntimeError: made to fail"
TEXT = f"failed: {ERROR}"


def _fail_prepare(monkeypatch) -> None:
    real = gate_run._prepare_task

    def task(job):
        unit, _, configuration = job
        if unit == OTHER and configuration.name == NAMES[0]:
            raise RuntimeError("made to fail")
        return real(job)

    monkeypatch.setattr(gate_run, "_prepare_task", task)


@pytest.fixture
def made(repo, builds, tmp_path, monkeypatch):
    """The recorded run with one failed configuration, and the same without it: (matrix, runs)."""
    matrix = ledger_tests._matrix(tmp_path, (SCORED, OTHER))
    runs = repo / "data" / "derived" / "runs"
    clean = tmp_path / "learn-clean"
    ledger_tests._learn(repo, builds["tag"], matrix, builds, clean)
    clean_run = record(runs, clean, LEARN_MODULE, "learn-clean", commit=builds["tag"])
    _fail_prepare(monkeypatch)
    source = tmp_path / "learn-failed"
    ledger_tests._learn(repo, builds["tag"], matrix, builds, source)  # asserts exit 0
    failed_run = record(runs, source, LEARN_MODULE, "learn-failed", commit=builds["tag"])
    return {"matrix": matrix, "run": failed_run, "clean": clean_run, "tmp": tmp_path}


def _read(repo, builds, monkeypatch, run, matrix, out: Path) -> dict:
    arguments = ledger_tests._arguments(builds, {"run": run, "matrix": matrix})
    with monkeypatch.context() as patched:
        ledger_tests._at_head(patched, builds["tag"])
        status = ledger_tests._ledger(arguments, repo, out, patched)
    assert status == 0
    return json.loads((out / ledger_run.RESULTS_NAME).read_text())


def test_ledger_run_shows_the_failed_unit_as_failed(repo, builds, made, monkeypatch):
    tmp = made["tmp"]
    failed = _read(repo, builds, monkeypatch, made["run"], made["matrix"], tmp / "ledger-failed")
    clean = _read(repo, builds, monkeypatch, made["clean"], made["matrix"], tmp / "ledger-clean")
    unit = failed["units"][OTHER]["configurations"]
    assert unit[NAMES[0]] == {"reason": TEXT}
    assert "reason" not in unit.get(NAMES[1], {})
    assert "abstained" not in json.dumps(failed["units"][OTHER])
    assert failed["units"][SCORED] == clean["units"][SCORED]


def test_day_intervals_leaves_the_failed_unit_out_with_its_reason(
    repo, builds, tmp_path, monkeypatch, capsys
):
    matrix = ledger_tests._matrix(tmp_path, (SCORED, OTHER))
    runs = repo / "data" / "derived" / "runs"
    _fail_prepare(monkeypatch)
    source = tmp_path / "learn-failed"
    options = ["--interval-seed", "3", "--interval-resamples", "10"]
    _learn_with(repo, builds, matrix, source, options)
    run = record(runs, source, LEARN_MODULE, "learn-failed-intervals", commit=builds["tag"])
    out = tmp_path / "intervals-out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    with monkeypatch.context() as patched:
        patched.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patched.setattr(day_intervals, "head_commit", lambda top: builds["tag"])
        status = day_intervals.main(["--run", str(run), "--seed", "3", "--resamples", "10"])
    assert status == 0, capsys.readouterr().out
    found = json.loads((out / day_intervals.RESULTS_NAME).read_text())
    block = found["pooled"]["configurations"][NAMES[0]]
    assert {"unit": OTHER, "reason": TEXT} in block["left_out"]
    assert not [u for u in block["left_out"] if u.get("abstained")]
    summary = (out / day_intervals.SUMMARY_NAME).read_text()
    assert f"{OTHER} ({TEXT})" in summary and "abstained" not in summary


def _learn_with(repo, builds, matrix, source, options) -> None:
    """`ledger_tests._learn` with the run's interval options."""
    import types

    from fleetreserve import learn_run

    source.mkdir()
    with pytest.MonkeyPatch.context() as patched:
        ledger_tests._at_head(patched, builds["tag"])
        module = types.ModuleType(f"fleetreserve.models.{ledger_tests.MODEL}")
        module.make_model = lambda m: ledger_tests.BoxModel(ledger_tests.MODEL, m.rho)
        patched.setitem(sys.modules, module.__name__, module)
        patched.setenv("FLEETRESERVE_RUN_DIR", str(source))
        arguments = ["--model", ledger_tests.MODEL, "--matrix", str(matrix)]
        arguments += ["--tables", str(builds["tables"]), "--truth-run", str(builds["run"])]
        arguments += ["--held-out-tables", str(builds["held_tables"])]
        arguments += ["--held-out-truth-run", str(builds["held_run"]), *options]
        assert learn_run.main(arguments, root=repo) == 0

"""`soft_scale_check` on training-site days: the committed matrices `gate/t8-training-a.json` and
`gate/t8-training-b.json`, rotation what-ifs whose gate units are training units (`gate_config`),
on made-up units and a made-up verified truth build.

The units: every training unit of `truth_build.TRAIN_UNITS` with the exercising cars of
`test_soft_scale_check` (three cars a day, three service hours), 12, 14, ..., 26 counting days in
the order of `TRAIN_UNITS` (one site each, S0 to S7), and the development gate units of
`test_soft_scale_check`. Each committed matrix scores four of them: `a` caltech, office001, ASK and
BAR (12, 16, 20 and 24 days); `b` jpl, OSL_S, TRO and BOD (14, 18, 22 and 26 days).

1. With `lgbm_tail_rotation` (fitted on the other four training units) the check runs on each
   matrix's four units, 6 days of each, and passes (exit 0); every output says that it is a
   rotation what-if: the key `rotation_whatif` of the results (the four units scored and left out
   of the fit, the other four fitted on, the evidence sentence), the label and the definitions,
   and the first two lines of the summary, which is what was printed.
2. A run of the development gate units after it, in the same process, carries no trace of the
   label in any output.
3. With `lgbm_tail`, fitted on every training unit and so on the checked ones, the check could not
   run (exit 2) and writes nothing; so does a matrix with the same training gate units without
   `rotation_whatif` (refused by `gate_config`).
"""

import json

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import pytest
import torch
from gate_fake import COMMIT, TOP, matrix_document, write_build
from test_soft_scale_check import NAMES, SERVICE_HOURS, exercising_cars, exercising_units

from fleetreserve import gate_config, gate_run, rotation_whatif, run_verify, soft_scale_check
from fleetreserve.truth_build import TRAIN_UNITS

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

MATRICES = {"a": TOP / "gate" / "t8-training-a.json", "b": TOP / "gate" / "t8-training-b.json"}
SCORED = {"a": ("caltech", "office001", "ASK", "BAR"), "b": ("jpl", "OSL_S", "TRO", "BOD")}
CONFIGURATION = "main-k0.3-empirical"
DAYS = 6


@pytest.fixture
def made_up(tmp_path, monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    training = {u: exercising_cars(12 + 2 * k, f"S{k}") for k, u in enumerate(TRAIN_UNITS)}
    run, tables = write_build(tmp_path, {**exercising_units(), **training}, SERVICE_HOURS)
    return {"tmp": tmp_path, "inputs": ["--tables", str(tables), "--truth-run", str(run)]}


def _run(
    made_up, monkeypatch, matrix, model, name, capsys, configuration=CONFIGURATION
) -> tuple[int, str, dict]:
    out = made_up["tmp"] / name
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    code = soft_scale_check.main(
        ["--model", model, "--configuration", configuration, "--days", str(DAYS)]
        + ["--matrix", str(matrix), *made_up["inputs"]]
    )
    files = {path.name: path.read_text() for path in sorted(out.iterdir())}
    return code, capsys.readouterr().out, files


@pytest.mark.parametrize("half", ("a", "b"))
def test_the_committed_training_matrices_are_checked_and_labelled(
    made_up, monkeypatch, capsys, half
):
    code, printed, files = _run(
        made_up, monkeypatch, MATRICES[half], "lgbm_tail_rotation", "rotation", capsys
    )
    assert code == 0, printed
    results = json.loads(files[soft_scale_check.RESULTS_NAME])
    assert sorted(results["units"]) == sorted(SCORED[half])  # the file's keys are sorted
    assert all(unit["checked_days"] == DAYS for unit in results["units"].values())
    counting = {name: unit["counting_days"] for name, unit in results["units"].items()}
    assert counting == {u: 12 + 2 * TRAIN_UNITS.index(u) for u in SCORED[half]}
    found = results["rotation_whatif"]
    fitted = [u for u in TRAIN_UNITS if u not in SCORED[half]]
    assert found["what"] == rotation_whatif.WHAT
    assert found["scored_training_units"] == list(SCORED[half])
    assert found["left_out_of_the_fit"] == list(SCORED[half])
    assert found["fit_units"] == fitted
    assert found["evidence"] == rotation_whatif.evidence_sentence(SCORED[half], fitted)
    line = rotation_whatif.label(gate_config.load(MATRICES[half]), fitted)
    assert results["label"] == f"{line}. T8 check (plan section 7)"
    assert results["definitions"]["rotation_whatif"] == found["evidence"]
    summary = files[soft_scale_check.SUMMARY_NAME]
    assert summary == printed
    assert summary.splitlines()[:2] == [rotation_whatif.WHAT, found["evidence"]]
    assert summary.splitlines()[2].startswith("T8 check: model lgbm_tail_rotation")
    # 2. a run of the development units afterwards carries no trace of the label
    development = made_up["tmp"] / "development.json"
    development.write_text(json.dumps(matrix_document(NAMES)))
    code, printed, files = _run(
        made_up, monkeypatch, development, "empirical", "development", capsys, NAMES[0]
    )
    assert code == 0, printed
    for content in (printed, *files.values()):
        assert "rotation" not in content and "what-if" not in content.lower()


def test_a_model_fitted_on_a_checked_unit_or_a_matrix_without_the_field_could_not_run(
    made_up, monkeypatch, capsys
):
    code, printed, files = _run(made_up, monkeypatch, MATRICES["a"], "lgbm_tail", "plain", capsys)
    assert code == 2 and printed.startswith("COULD NOT RUN") and "scores" in printed, printed
    assert files == {}
    document = json.loads(MATRICES["a"].read_text())
    del document["rotation_whatif"]
    unflagged = made_up["tmp"] / "unflagged.json"
    unflagged.write_text(json.dumps(document))
    code, printed, files = _run(
        made_up, monkeypatch, unflagged, "lgbm_tail_rotation", "unflagged", capsys
    )
    assert code == 2 and "rotation_whatif" in printed and files == {}

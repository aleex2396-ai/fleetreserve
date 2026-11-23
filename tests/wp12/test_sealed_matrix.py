"""The run matrices of the run from the tag and of its rehearsal (`sealed_matrix`), written from
made-up kappa-choice results made by `kappa_choice.joint` itself (the hand case of
`tests/wp6gate/test_kappa_choice.py`), one per model of the full method.

1. Each model's matrix is at that model's own kappa and dials of reading R2: in the hand case R2
   gives kappa 0.1, workplaces dial 0.05 and Norway and Korea dial 0.05; a second model's results
   are given R2 kappa 0.3 with dials 0.01 and 0.05, and its matrix holds those. The sealed matrix
   scores the nine sealed units, the rehearsal's the gate matrix's development units; both are
   valid matrices (`gate_config`) with one dial family (`dial_choice.find_family`) whose dials
   are the R2 dials, and `compare_run` takes each by `--dial`. The field `sealed_choice` records
   reading R2, the model, the pair, the units per site type, the headline configuration per site
   type and the source file with its SHA-256.
2. The main model also gets the forecast-energy comparison at its own kappa and dials, energy
   bounds empirical, and the sensitivity matrices at rho 0.1 and 0.2 (plan WP15), at its own
   kappa and dials, each recording its rho; no other model does. `check` refuses a recorded rho
   that is not the matrix's.
3. Could not run (exit 2, nothing written): a model of the full method missing, a model given
   twice, a model whose R2 is not defined. Refused (exit 1): a results file whose run did not
   end with exit 0.
4. `gate_config` reads `sealed_choice` only to refuse a reading other than R2; `check` refuses a
   headline that is not its site type's configuration.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from fleetreserve import (
    compare_run,
    dial_choice,
    gate_config,
    kappa_choice,
    run_verify,
    sealed_matrix,
    sealed_route,
)

_WP6GATE = Path(__file__).resolve().parents[1] / "wp6gate"
if str(_WP6GATE) not in sys.path:
    sys.path.insert(0, str(_WP6GATE))
import test_kappa_choice as made_up_choice  # noqa: E402

MODELS = sealed_matrix.FULL_METHOD_MODELS
W, OTHER = "workplaces", "Norway and Korea"


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def joint(model: str, r2=None) -> dict:
    document = copy.deepcopy(kappa_choice.joint(made_up_choice.runs()))
    document["run"]["model"] = model
    if r2 is not None:
        kappa, dials = r2
        document["readings"]["R2"]["kappa"] = kappa
        for site_type, dial in dials.items():
            document["readings"]["R2"]["dials"][site_type] = (
                None if dial is None else {"dial": dial, "unit": "u", "mean_offer": 1.0}
            )
    return document


def record(folder: Path, document: dict, code: int = 0) -> Path:
    folder.mkdir(parents=True)
    path = folder / kappa_choice.RESULTS_NAME
    path.write_text(json.dumps(document))
    outputs = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"commit": "c" * 40, "command": ["python", "-m", "fleetreserve.kappa_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": code, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return path


def files(tmp_path: Path, documents: dict, code: int = 0) -> list[str]:
    return [
        str(record(tmp_path / "runs" / f"20261120T000000Z-kappa-{m}", d, code))
        for m, d in documents.items()
    ]


def honest() -> dict:
    found = {m: joint(m) for m in MODELS}
    found["earlier_gbm"] = joint("earlier_gbm", (0.3, {W: 0.01, OTHER: 0.05}))
    return found


def read(out: Path, folder: str, name: str) -> dict:
    return json.loads((out / folder / f"{name}.json").read_text())


def test_each_model_gets_its_own_r2_pair_in_a_sealed_and_a_rehearsal_matrix(tmp_path, capsys):
    out = tmp_path / "gate"
    paths = files(tmp_path, honest())
    assert sealed_matrix.main([*paths, "--out", str(out)]) == 0, capsys.readouterr().out
    hand = kappa_choice.joint(made_up_choice.runs())["readings"]["R2"]
    assert (hand["kappa"], hand["dials"][W]["dial"], hand["dials"][OTHER]["dial"]) == (
        0.1,
        0.05,
        0.05,
    )
    expected = {m: (0.1, {W: 0.05, OTHER: 0.05}) for m in MODELS} | {
        "earlier_gbm": (0.3, {W: 0.01, OTHER: 0.05})
    }
    template = json.loads(sealed_matrix.TEMPLATE.read_text())
    for model, (kappa, dials) in expected.items():
        for folder, units in (
            ("sealed-matrix", list(sealed_route.HELD_OUT_UNITS)),
            ("rehearsal-matrix", template["gate_units"]),
        ):
            document = read(out, folder, model)
            matrix = gate_config.parse(document)
            assert list(matrix.gate_units) == units
            assert {c.kappa for c in matrix.configurations} == {kappa}
            _, family = dial_choice.find_family(matrix)
            assert sorted(c.driver_risk for c in family) == sorted(set(dials.values()))
            for dial in dials.values():
                assert compare_run.dial_configurations(matrix, dial)["forecast"].kappa == kappa
            choice = document["sealed_choice"]
            assert choice["reading"] == "R2" and choice["model"] == model
            assert choice["kappa"] == kappa and choice["dials"] == dials
            assert sealed_matrix.check(document) == []
            path = tmp_path / "runs" / f"20261120T000000Z-kappa-{model}" / "kappa-choice.json"
            assert choice["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            readings = {c.departure_seen for c in matrix.configurations}
            assert readings == {dial_choice.MAIN_READING, "hour-start"}
            assert matrix.split_seed == template["split_seed"]
    nine = read(out, "sealed-matrix", "lgbm_tail")["sealed_choice"]["units"]
    assert nine[W] == ["asr-before-march-2023", "asr-from-march-2023"]
    assert len(nine[OTHER]) == 7


def test_only_the_main_model_gets_the_forecast_energy_comparison(tmp_path, capsys):
    out = tmp_path / "gate"
    assert sealed_matrix.main([*files(tmp_path, honest()), "--out", str(out)]) == 0
    names = sorted(p.name for p in (out / "sealed-matrix").iterdir())
    assert names == sorted(
        [f"{m}.json" for m in MODELS]
        + ["lgbm_tail-forecast-energy.json", "lgbm_tail-rho0.1.json", "lgbm_tail-rho0.2.json"]
    )
    document = read(out, "sealed-matrix", "lgbm_tail-forecast-energy")
    matrix = gate_config.parse(document)
    assert {c.energy_bounds for c in matrix.configurations} == {"empirical"}
    assert {c.kappa for c in matrix.configurations} == {0.1}
    assert {c.departure_seen for c in matrix.configurations} == {dial_choice.MAIN_READING}
    assert document["sealed_choice"]["comparison"] == sealed_matrix.COMPARISON_TEXT
    assert sealed_matrix.check(document) == []
    assert (out / "rehearsal-matrix" / "lgbm_tail-forecast-energy.json").is_file()


def test_the_main_model_gets_its_matrices_at_rho_0_1_and_0_2(tmp_path):
    out = tmp_path / "gate"
    assert sealed_matrix.main([*files(tmp_path, honest()), "--out", str(out)]) == 0
    main = read(out, "sealed-matrix", "lgbm_tail")
    assert main["rho"] == 0.0 and main["sealed_choice"]["rho"] == 0.0
    for rho in (0.1, 0.2):
        for folder in ("sealed-matrix", "rehearsal-matrix"):
            document = read(out, folder, f"lgbm_tail-rho{rho}")
            matrix = gate_config.parse(document)
            assert matrix.rho == rho and document["sealed_choice"]["rho"] == rho
            base = read(out, folder, "lgbm_tail")
            assert document["configurations"] == base["configurations"]
            assert document["gate_units"] == base["gate_units"]
            assert sealed_matrix.check(document) == []
            kept = {k for k in document if k not in ("rho", "about", "sealed_choice")}
            assert {k: document[k] for k in kept} == {k: base[k] for k in kept}
            assert {k: v for k, v in document["sealed_choice"].items() if k != "rho"} == {
                k: v for k, v in base["sealed_choice"].items() if k != "rho"
            }
    wrong = read(out, "sealed-matrix", "lgbm_tail-rho0.1")
    wrong["rho"] = 0.2
    assert sealed_matrix.check(wrong) == ["sealed_choice.rho is not the matrix's rho"]


@pytest.mark.parametrize(
    "change",
    ["missing", "twice", "not defined"],
)
def test_could_not_run_writes_nothing(tmp_path, capsys, change):
    documents = honest()
    if change == "missing":
        documents.pop("earlier_population")
    elif change == "twice":
        documents["again"] = joint("lgbm_tail")
    else:
        documents["hazard_tuned_inputs"] = joint(
            "hazard_tuned_inputs", (0.3, {W: 0.05, OTHER: None})
        )
    out = tmp_path / "gate"
    assert sealed_matrix.main([*files(tmp_path, documents), "--out", str(out)]) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN: ")
    assert not out.exists()


def test_a_run_that_did_not_end_with_exit_0_is_refused(tmp_path, capsys):
    out = tmp_path / "gate"
    assert sealed_matrix.main([*files(tmp_path, honest(), code=1), "--out", str(out)]) == 1
    assert capsys.readouterr().out.startswith("REFUSED: ")
    assert not out.exists()


def test_gate_config_refuses_another_reading_and_check_a_wrong_headline(tmp_path):
    out = tmp_path / "gate"
    assert sealed_matrix.main([*files(tmp_path, honest()), "--out", str(out)]) == 0
    document = read(out, "sealed-matrix", "lgbm_tail")
    other = copy.deepcopy(document)
    other["sealed_choice"]["reading"] = "R1"
    with pytest.raises(gate_config.MatrixError, match="sealed_choice"):
        gate_config.parse(other)
    plain = copy.deepcopy(document)
    del plain["sealed_choice"]
    gate_config.parse(plain)
    wrong = copy.deepcopy(document)
    wrong["sealed_choice"]["headline"][W] = "main-k0.3-none-empirical"
    assert sealed_matrix.check(wrong) == [
        f"the headline of {W} is not its configuration at its dial"
    ]

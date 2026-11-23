"""`dial_choice --kappa <level>` on made-up gate units built with `gate_fake` (the empirical
model), and the joint table `kappa_choice` builds from its six runs, one per level of plan
section 2's grid (0.05, 0.1, 0.2, 0.3, 0.4, 0.5; the matrix of the tests lists all six).

The units (`units`) are `gate_fake.gate_units` with the second car leaving 2 hours after the
decision time on every third day (every third and fourth day at the pooled unit's two sites): about
one car in nine, so the levels 0.05 and 0.1 give an earlier departure bound than 0.2 and 0.3, and
the driver-risk rows bind at kappa 0.3 (a dial's figures differ from those without rows there),
while every day still offers at 0.05.

1. At the family's own level, 0.3, the option changes nothing: the results, the summary and the
   forecast table are byte-identical (SHA-256) to those of the run without the option; the run
   also writes the certificate timeouts per configuration, which the run without it does not.
2. At 0.1 the configurations measured are the matrix's configuration without rows at 0.1 and the
   family's dials with only kappa changed (names with `-k0.1-`, the oracle partners likewise);
   the results and the summary name kappa 0.1 and say how the family was made.
3. The model predicts at the levels up to the run's kappa only: the forecast table holds 0.05 and
   0.1 at kappa 0.1, and 0.05 alone at 0.05; at 0.4 and 0.5, levels of the grid since the grid has
   six, `--kappa` runs and writes the grid's file names (no what-if file), the configurations are
   the family's with `-k0.4-` and `-k0.5-`, and the table holds the levels up to the run's.
4. At 0.05 the driver-risk rows have the one level 0.05 at the presence bound and hold on the
   whole forecast set (dial_choice reading K1): every dial's figures are those of the
   configuration without rows, on units where days offer; at 0.3 they are not.
5. A certificate timeout is counted per configuration, and the counts add up to the run's total.
6. Exit 2 for a level outside the plan's grid, a matrix without the configuration without rows at
   the level, and a dial whose name does not hold its level.
7. The six runs, each with a made-up manifest, give `kappa_choice` its joint table: exit 0, and
   each pair's figures and re-solve timeouts are those of the run at the pair's kappa.
"""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    PLAN_LEVELS,
    gate_cars,
    gate_units,
    matrix_document_six,
    write_build,
)

from fleetreserve import dial_choice, gate_config, gate_day, gate_run, kappa_choice, run_verify
from fleetreserve.forecast_model import read_table
from fleetreserve.gate_config import MatrixError

PLAINS = tuple(
    f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")
)
DIALS = tuple(
    f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")
)
NAMES = PLAINS + DIALS
SPLITS = 6
LEVELS = PLAN_LEVELS  # the six levels of plan section 2
CALIBRATABLE = GATE_UNITS[1:]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def units() -> dict:
    """The made-up gate units (module text)."""
    found = gate_units()
    found[GATE_UNITS[1]] = gate_cars(60, early_every=3, early_out=12.0)
    found[GATE_UNITS[2]] = gate_cars(30, "B", early_every=3, early_out=12.0) + gate_cars(
        30, "C", early_every=4, early_out=12.0
    )
    return found


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    return run, tables, matrix


def _argv(build, *extra, matrix=None) -> list[str]:
    run, tables, committed = build
    arguments = ["--model", "empirical", "--matrix", str(matrix or committed)]
    arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
    return [*arguments, *extra]


def _run(build, out: Path, *extra, patch=None) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out.mkdir(parents=True)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return dial_choice.main(_argv(build, *extra))


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    """level (None: no option) -> (exit status, run folder)."""
    folder = tmp_path_factory.mktemp("runs")
    found = {None: (_run(build, folder / "plain"), folder / "plain")}
    for kappa in LEVELS:
        out = folder / f"k{kappa:g}"
        found[kappa] = (_run(build, out, "--kappa", str(kappa)), out)
    return found


def _document(out: Path) -> dict:
    return json.loads((out / dial_choice.RESULTS_NAME).read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. the family's own level -------------------------------------------------------------------


def test_at_the_familys_own_level_the_option_changes_nothing(runs):
    (code, plain), (code_k, at_k) = runs[None], runs[0.3]
    assert code == code_k == 0
    for name in (dial_choice.RESULTS_NAME, dial_choice.SUMMARY_NAME, dial_choice.TABLE_NAME):
        assert _sha(plain / name) == _sha(at_k / name), name
    assert (at_k / dial_choice.CERTIFICATE_TIMEOUTS_NAME).is_file()
    assert not (plain / dial_choice.CERTIFICATE_TIMEOUTS_NAME).exists()
    assert "kappa_option" not in _document(at_k)["definitions"]


# --- 2. the family at another level --------------------------------------------------------------


def test_the_family_at_another_level_changes_only_kappa():
    matrix = gate_config.parse(matrix_document_six(NAMES))
    plain, dials = dial_choice.find_family(matrix)
    measured, (plain_at, dials_at) = dial_choice.family_at(matrix, (plain, dials), 0.1)
    assert plain_at == matrix.configuration("main-rounded-hour-k0.1-empirical")
    assert [d.name for d in dials_at] == [
        "main-rounded-hour-k0.1-eps0.03-empirical",
        "main-rounded-hour-k0.1-eps0.1-empirical",
    ]
    assert plain_at.kappa == 0.1 and {d.kappa for d in dials_at} == {0.1}
    for dial, dial_at in zip(dials, dials_at, strict=True):
        assert measured.configuration(dial_at.oracle_partner).kappa == 0.1
        back = dataclasses.replace(
            dial_at, kappa=0.3, name=dial.name, oracle_partner=dial.oracle_partner
        )
        assert back == dial
        partner, partner_at = (
            matrix.configuration(dial.oracle_partner),
            measured.configuration(dial_at.oracle_partner),
        )
        assert partner_at.name == partner.name.replace("-k0.3-", "-k0.1-")
        assert dataclasses.replace(partner_at, kappa=0.3, name=partner.name) == partner
    assert dial_choice.family_at(matrix, (plain, dials), 0.3) == (matrix, (plain, dials))


def test_a_run_at_kappa_0_1_measures_the_family_at_0_1(runs):
    code, out = runs[0.1]
    assert code == 0
    document = _document(out)
    assert document["run"]["kappa"] == 0.1 and document["run"]["dials"] == [0.03, 0.1]
    names = [
        "main-rounded-hour-k0.1-empirical",
        "main-rounded-hour-k0.1-eps0.03-empirical",
        "main-rounded-hour-k0.1-eps0.1-empirical",
    ]
    for unit in CALIBRATABLE:
        configurations = document["units"][unit]["configurations"]
        assert list(configurations) == names
        partners = [c["gate_reading"]["oracle_partner"] for c in configurations.values()]
        assert partners == [n.replace("-empirical", "-oracle") for n in names]
    assert document["definitions"]["dial_configurations"] == {"0.03": names[1], "0.1": names[2]}
    assert document["definitions"]["configuration_without_rows"] == names[0]
    option = document["definitions"]["kappa_option"]
    assert "kappa 0.3" in option and "kappa 0.1" in option
    summary = (out / dial_choice.SUMMARY_NAME).read_text()
    assert "kappa 0.1." in summary.splitlines()[0]
    assert option in summary


# --- 3. the levels the model predicts at ---------------------------------------------------------


def test_the_model_predicts_at_the_levels_up_to_the_runs_kappa(runs):
    for kappa, levels in (
        (0.05, {0.05}),
        (0.1, {0.05, 0.1}),
        (0.3, {0.05, 0.1, 0.2, 0.3}),
        (0.4, {0.05, 0.1, 0.2, 0.3, 0.4}),
        (0.5, set(LEVELS)),
    ):
        table = read_table(runs[kappa][1] / dial_choice.TABLE_NAME)
        assert set(table["kappa"]) == levels, kappa
        assert _document(runs[kappa][1])["forecast_table"]["rows"] == len(table)


@pytest.mark.parametrize("kappa", [0.4, 0.5])
def test_the_levels_0_4_and_0_5_are_grid_levels_of_kappa(runs, kappa):
    code, out = runs[kappa]
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(dial_choice.PLAN_NAMES.values())
    document = _document(out)
    assert "what_if" not in document and document["label"] == f"dial choice: {dial_choice.LABEL}"
    assert document["run"]["kappa"] == kappa and document["run"]["dials"] == [0.03, 0.1]
    names = [
        f"main-rounded-hour-k{kappa:g}-empirical",
        f"main-rounded-hour-k{kappa:g}-eps0.03-empirical",
        f"main-rounded-hour-k{kappa:g}-eps0.1-empirical",
    ]
    for unit in CALIBRATABLE:
        assert list(document["units"][unit]["configurations"]) == names
    assert f"kappa {kappa:g}" in document["definitions"]["kappa_option"]


# --- 4. kappa 0.05 -------------------------------------------------------------------------------


def test_at_kappa_0_05_every_dial_gives_the_figures_of_the_set_without_rows(runs):
    code, out = runs[0.05]
    assert code == 0
    document = _document(out)
    for unit in CALIBRATABLE:
        plain, *dials = document["units"][unit]["configurations"].values()
        assert plain["driver_risk"] is None and len(dials) == 2
        assert plain["offering_days"] > 0 and plain["tiers"]["A"]["splits_that_offer"] > 0
        for dial in dials:
            assert dial["offering_days"] == plain["offering_days"]
            assert dial["tiers"] == plain["tiers"], (unit, dial["driver_risk"])
    # at 0.3 the rows bind on these days, so the comparison above can fail
    document = _document(runs[0.3][1])
    for unit in CALIBRATABLE:
        plain, *dials = document["units"][unit]["configurations"].values()
        assert any(d["tiers"] != plain["tiers"] for d in dials), unit


# --- 5. certificate timeouts per configuration ---------------------------------------------------


def test_a_certificate_timeout_is_counted_per_configuration(build, runs, tmp_path):
    first_date = sorted({c.date for c in units()[GATE_UNITS[1]]})[0]
    original = gate_day.prepare_day

    def patch(monkeypatch):
        def prepare(truth, forecaster, configuration, settings):
            found = original(truth, forecaster, configuration, settings)
            if truth.record.date == first_date and not found.abstains:
                return dataclasses.replace(
                    found,
                    abstain_reason="no certificate (timeout)",
                    certificate_outcome="timeout",
                    value=0.0,
                )
            return found

        monkeypatch.setattr(gate_day, "prepare_day", prepare)

    out = tmp_path / "timeouts"
    assert _run(build, out, "--kappa", "0.1", patch=patch) == 0
    counts = json.loads((out / dial_choice.CERTIFICATE_TIMEOUTS_NAME).read_text())
    assert counts["kappa"] == 0.1
    by_unit = counts["certificate_timeouts"]
    assert set(by_unit) == set(CALIBRATABLE)
    names = list(_document(out)["units"][GATE_UNITS[1]]["configurations"])
    assert list(by_unit[GATE_UNITS[1]]) == names
    assert all(n >= 1 for unit in CALIBRATABLE for n in by_unit[unit].values())
    total = sum(n for unit in CALIBRATABLE for n in by_unit[unit].values())
    assert total == _document(out)["timeouts"]["certificates"]
    unpatched = json.loads((runs[0.1][1] / dial_choice.CERTIFICATE_TIMEOUTS_NAME).read_text())
    assert {
        n for unit in CALIBRATABLE for n in unpatched["certificate_timeouts"][unit].values()
    } == {0}


# --- 6. refusals ---------------------------------------------------------------------------------


def test_a_level_outside_the_grid_could_not_run(build, capsys):
    assert dial_choice.main(_argv(build, "--kappa", "0.15")) == 2
    assert "--kappa must be one of" in capsys.readouterr().out


def test_a_matrix_without_the_plain_configuration_at_the_level_could_not_run(
    build, tmp_path, capsys
):
    names = tuple(n for n in NAMES if "-k0.1-" not in n)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(names)))
    assert dial_choice.main(_argv(build, "--kappa", "0.1", matrix=matrix)) == 2
    assert "without driver-risk rows at kappa 0.1" in capsys.readouterr().out


def test_a_dial_name_without_its_level_could_not_run(build, tmp_path, capsys):
    document = matrix_document_six(NAMES)
    renames = {n: n.replace("-k0.3-eps", "-eps") for n in DIALS}
    for c in document["configurations"]:
        c["name"] = renames.get(c["name"], c["name"])
        c["oracle_partner"] = renames.get(c["oracle_partner"], c["oracle_partner"])
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(document))
    parsed = gate_config.parse(document)
    family = dial_choice.find_family(parsed)
    with pytest.raises(MatrixError, match="does not hold its level"):
        dial_choice.family_at(parsed, family, 0.1)
    assert dial_choice.main(_argv(build, "--kappa", "0.1", matrix=matrix)) == 2
    assert "does not hold its level" in capsys.readouterr().out


# --- 7. the joint table of the six runs ---------------------------------------------------------


def _recorded(source: Path, folder: Path) -> Path:
    """A copy of the run folder `source` with a made-up manifest, as `run_record` writes it."""
    shutil.copytree(source, folder)
    outputs = {p.name: _sha(p) for p in sorted(folder.iterdir())}
    manifest = {"commit": COMMIT, "command": ["python", "-m", "fleetreserve.dial_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": 0, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder / dial_choice.RESULTS_NAME


def test_the_six_runs_give_the_joint_table(runs, tmp_path, monkeypatch):
    order = (0.2, 0.5, 0.3, 0.05, 0.4, 0.1)
    files = [_recorded(runs[k][1], tmp_path / "runs" / f"k{k:g}") for k in order]
    out = tmp_path / "joint"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert kappa_choice.main([str(f) for f in files]) == 0
    joint = json.loads((out / kappa_choice.RESULTS_NAME).read_text())
    assert [r["kappa"] for r in joint["runs"]] == list(LEVELS)
    for kappa in LEVELS:
        document = _document(runs[kappa][1])
        for unit in CALIBRATABLE:
            entry = joint["units"][unit]
            for found in document["units"][unit]["configurations"].values():
                key = kappa_choice.key(found["driver_risk"])
                for tier in ("A", "B"):
                    cell = entry["pairs"][tier][f"{kappa:g}"][key]
                    assert cell["defined"] is True
                    for field in kappa_choice.CELL_FIELDS:
                        assert cell[field] == found["tiers"][tier][field], (kappa, unit, key)
                resolves = found["evaluations"]["re_solves_of_the_one_way_rung"]
                timed = entry["timeouts"][f"{kappa:g}"][key]
                assert timed == {"certificates": 0, "re_solves": resolves.get("timeout", 0)}
        assert joint["units"][GATE_UNITS[0]]["pairs"]["A"][f"{kappa:g}"]["none"]["defined"] is False
    text = (out / kappa_choice.SUMMARY_NAME).read_text()
    assert kappa_choice.LABEL in text and "not defined" in text

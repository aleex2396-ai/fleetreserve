"""The dial-choice command (`dial_choice`) on the made-up gate units of `gate_fake`.

1. The dial family and the configuration without rows are found by their fields, not by their
   names; a matrix without the family, with a family that differs in another field, with a dial
   twice or without the plain partner is refused.
2. The seed of the matrix (`split_seed`) gives the gate run's own split. For that split the
   on-grid scale of tier A is at most the scale the gate run bisected (one-way rung) and at least
   the largest grid value not above it; the same for tier B against the gate's `tier_b`. Cases
   include scales strictly between 0 and 1, so the bounds are not met by 0 and 1 alone.
3. No (day, scale) is evaluated twice for one configuration, and every offering day is evaluated
   at every scale of the grid and at no other.
4. A full run ends with exit 0 and writes every output; the unit that cannot be split gets no
   numbers; both tiers are given for every configuration and the dial chosen is a dial, never the
   configuration without rows. Two runs (one and two workers) give the same results and summary;
   only the timing file may differ.
5. The summary carries the label "measured, not fixed" on every unit and tier line, shows the
   tiers as two separate blocks, and its numbers are those of the results of that tier.
6. A day whose certificate times out offers nothing and is counted; a missing ceiling is counted.
7. Exit status 2 for a matrix without the dial family, a stale truth build, a changed table, a bad
   argument and an unknown model.
"""

import copy
import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
from gate_fake import (
    COMMIT,
    COMMITTED,
    GATE_UNITS,
    gate_cars,
    gate_units,
    matrix_document,
    write_build,
    write_tables,
)

from fleetreserve import (
    dial_choice,
    dial_splits,
    dial_summary,
    gate_calibrate,
    gate_config,
    gate_day,
    gate_run,
    run_verify,
)
from fleetreserve.day_split import split_days
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_config import MatrixError

PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
HIGH = ("main-rounded-hour-k0.3-eps0.1-empirical", "main-rounded-hour-k0.3-eps0.1-oracle")
NAMES = PLAIN + LOW + HIGH
SPLITS = 12


def matrix_of(names=NAMES):
    return gate_config.parse(matrix_document(names))


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def command(build, out: Path, monkeypatch, *extra, model="empirical") -> int:
    run, tables, matrix = build
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    return dial_choice.main(["--model", model, *arguments, "--splits", str(SPLITS), *extra])


def _module_run(build, folder: Path, *extra, patch=None):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out = folder / "out"
        return command(build, out, monkeypatch, *extra), out


def _document(out: Path) -> dict:
    return json.loads((out / dial_choice.RESULTS_NAME).read_text())


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    return _module_run(build, tmp_path_factory.mktemp("full"), "--workers", "1")


# --- 1. the family -------------------------------------------------------------------------------


def test_the_family_is_found_by_fields_not_by_names():
    document = matrix_document(NAMES)
    renames = {c["name"]: f"x{k}" for k, c in enumerate(document["configurations"])}
    for c in document["configurations"]:
        c["oracle_partner"] = renames.get(c["oracle_partner"])
        c["name"] = renames[c["name"]]
    plain, dials = dial_choice.find_family(gate_config.parse(document))
    assert plain.driver_risk is None and plain.departure == "empirical"
    assert plain.departure_seen == "rounded-hour"
    assert [c.driver_risk for c in dials] == [0.03, 0.1]
    assert {c.kappa for c in (plain, *dials)} == {0.3}


def test_the_committed_matrix_has_the_whole_dial_family():
    committed = gate_config.parse(copy.deepcopy(COMMITTED))
    plain, dials = dial_choice.find_family(committed)
    assert plain.name == "main-rounded-hour-k0.3-empirical"
    assert [c.driver_risk for c in dials] == [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1]
    assert all(c.family == "alternate" and c.departure_seen == "rounded-hour" for c in dials)


def _entry(document, name):
    return next(c for c in document["configurations"] if c["name"] == name)


def test_a_matrix_without_a_usable_family_is_refused():
    with pytest.raises(MatrixError, match="no dial family"):
        dial_choice.find_family(matrix_of(PLAIN))
    document = matrix_document(PLAIN + LOW + HIGH)  # a family that differs in another field
    _entry(document, HIGH[1])["energy_bounds"] = "empirical"
    _entry(document, HIGH[0])["energy_bounds"] = "empirical"
    with pytest.raises(MatrixError, match="differ in a field other than the dial"):
        dial_choice.find_family(gate_config.parse(document))
    with pytest.raises(MatrixError, match="exactly one configuration without driver-risk rows"):
        dial_choice.find_family(matrix_of(LOW))
    document = matrix_document(PLAIN + LOW + HIGH)
    for name in HIGH:
        _entry(document, name)["driver_risk"] = 0.03
    with pytest.raises(MatrixError, match="same dial"):
        dial_choice.find_family(gate_config.parse(document))


def test_the_hour_start_reading_is_not_the_family():
    names = (*NAMES, "main-k0.3-empirical", "main-k0.3-oracle")
    plain, dials = dial_choice.find_family(matrix_of(names))
    assert plain.name == PLAIN[0] and len(dials) == 2


# --- 2. the gate run's split and its bisected scale ---------------------------------------------


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """(gate result, losses, split) per made-up unit and configuration, one cache per case."""
    folder = tmp_path_factory.mktemp("cases")
    matrix = matrix_of()
    found = {}
    for key, (every, out) in {
        "plain": (3, 11.3),
        "early": (6, 12.0),
        "rare": (12, 11.3),
    }.items():
        write_tables(folder, key, gate_cars(60, early_every=every, early_out=out))
        days = load_unit_days(folder, key, rho=0.0, no_fit="cap", small_fleet="keep")
        plan = gate_run.plan_unit(key, days, matrix)
        ceilings = [gate_day.day_ceilings(t.day) for t in days.days]
        for name in (PLAIN[0], LOW[0]):
            c = matrix.configuration(name)
            forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
            prepared = [
                gate_day.prepare_day(t, forecaster, c, gate_run.settings(matrix)) for t in days.days
            ]

            def batch(pairs):
                return [gate_day.evaluate(p, b).compact() for p, b in pairs]

            result, *_ = gate_calibrate.run_job(prepared, ceilings, plan.roles, batch, tier_b=True)
            losses, _ = dial_choice.measure(prepared, [c.best_causal for c in ceilings], batch)
            (split,) = dial_choice.split_indices(key, days, matrix, [matrix.split_seed])
            found[(key, name)] = (result["rungs"]["one-way"], losses, split, plan, days)
    return found


def test_the_seed_of_the_matrix_gives_the_gate_runs_split(cases):
    matrix = matrix_of()
    assert matrix.split_seed == 1
    for (key, _), (_, _, (seed, calibration, test), plan, days) in cases.items():
        assert seed == matrix.split_seed
        assert calibration == [d for d, r in enumerate(plan.roles) if r == "calibration"]
        assert test == [d for d, r in enumerate(plan.roles) if r == "test"]
        keys = [f"{t.record.date} {t.record.site}" for t in days.days]
        direct = split_days(key, matrix.split_seed, keys, n_calibration=50, min_days=51)
        assert [keys[d] for d in calibration] == list(direct.calibration)
        assert [keys[d] for d in test] == list(direct.test)


def test_other_seeds_give_other_splits(cases):
    matrix = matrix_of()
    _, _, _, plan, days = cases[("plain", PLAIN[0])]
    found = dial_choice.split_indices("plain", days, matrix, [1, 2, 3])
    assert len({tuple(cal) for _, cal, _ in found}) == 3
    assert all(len(cal) == 50 and len(cal) + len(test) == 60 for _, cal, test in found)


def _gate_floor(gate_scale: float) -> float:
    held = [g for g in dial_splits.GRID if g <= gate_scale]
    return max(held, default=0.0)


@pytest.mark.parametrize("tier", ["A", "B"])
def test_the_on_grid_scale_is_bracketed_by_the_gate_runs_scale(cases, tier):
    between = 0
    for key, (rung, losses, (seed, calibration, test), _, _) in cases.items():
        found = dial_splits.score_split(losses, seed, calibration, test, tier)
        gate = float(rung["beta_hat"] if tier == "A" else rung["tier_b"]["beta_hat"])
        assert _gate_floor(gate) <= found.scale <= gate + 1e-12, key
        between += 0 < found.scale < 1
    assert between >= 1  # a scale strictly inside the grid was met (tier B: 0.3 against 0.333)


def test_the_cases_meet_both_tiers_with_a_scale_strictly_inside(cases):
    scales = {
        (tier, key): dial_splits.score_split(losses, 1, cal, test, tier).scale
        for (key, name), (_, losses, (_, cal, test), _, _) in cases.items()
        for tier in ("A", "B")
    }
    assert any(0 < s < 1 for (tier, _), s in scales.items() if tier == "A")
    assert any(0 < s < 1 for (tier, _), s in scales.items() if tier == "B")
    assert any(s == 0 for (tier, _), s in scales.items() if tier == "A")


# --- 3. no pair evaluated twice ------------------------------------------------------------------


def test_no_day_scale_pair_is_evaluated_twice(cases):
    key = ("plain", LOW[0])
    _, _, _, plan, days = cases[key]
    matrix = matrix_of()
    c = matrix.configuration(LOW[0])
    forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
    prepared = [
        gate_day.prepare_day(t, forecaster, c, gate_run.settings(matrix)) for t in days.days
    ]
    seen = []

    def batch(pairs):
        seen.extend((p.date, b) for p, b in pairs)
        return [gate_day.evaluate(p, b).compact() for p, b in pairs]

    losses, cache = dial_choice.measure(prepared, [None] * len(prepared), batch)
    assert len(seen) == len(set(seen))
    assert max(cache.calls.values()) == 1
    offering = [p.date for p in prepared if not p.abstains]
    assert len(offering) > 0
    assert set(seen) == {(date, g) for date in offering for g in dial_splits.GRID}
    assert len(cache.calls) == len(offering) * len(dial_splits.GRID)
    assert np.isnan(losses.ceiling).all()  # a missing ceiling stays missing


# --- 4. a full run -------------------------------------------------------------------------------


def test_a_full_run_writes_every_output(full):
    code, out = full
    assert code == 0
    for name in (
        dial_choice.RESULTS_NAME,
        dial_choice.SUMMARY_NAME,
        dial_choice.TABLE_NAME,
        dial_choice.TIMING_NAME,
    ):
        assert (out / name).is_file()
    document = _document(out)
    assert document["failed_configurations"] == []
    assert set(document["timeouts"].values()) == {0}
    assert document["run"]["scales"] == list(dial_splits.GRID)
    assert document["run"]["dials"] == [0.03, 0.1] and document["run"]["splits"] == SPLITS
    assert document["model"] == "empirical" and document["inputs"]["setting"] == "main"
    assert "definitions" in document and document["forecast_table"]["rows"] > 0
    small, usable, pooled = (document["units"][u] for u in GATE_UNITS)
    assert small["status"].startswith("not calibratable") and small["configurations"] == {}
    assert small["split_of_the_gate_run"] == {}
    assert set(document["dial_choice"]) == set(GATE_UNITS[1:])
    for entry in (usable, pooled):
        assert list(entry["configurations"]) == [PLAIN[0], LOW[0], HIGH[0]]
        for found in entry["configurations"].values():
            assert set(found["tiers"]) == {"A", "B"}
            assert found["evaluations"]["pairs_evaluated_more_than_once"] == 0
            assert found["tiers"]["A"]["splits"] == SPLITS
            assert "driver_loss_over_limit" in found["tiers"]["A"]
            assert "driver_loss_over_limit" not in found["tiers"]["B"]


def test_the_chosen_dial_is_a_dial_with_the_largest_mean_offer(full):
    _, out = full
    document = _document(out)
    for unit, tiers in document["dial_choice"].items():
        configurations = document["units"][unit]["configurations"]
        for tier, chosen in tiers.items():
            offers = {
                c["driver_risk"]: c["tiers"][tier]["mean_offer_kwh_per_test_day"]
                for c in configurations.values()
                if c["driver_risk"] is not None
            }
            assert chosen["dial"] == min(d for d in offers if offers[d] == max(offers.values()))
            assert chosen["dial"] in (0.03, 0.1) and chosen["label"] == dial_summary.LABEL
            assert set(chosen["mean_offers_by_dial"]) == {"0.03", "0.1"}


def test_the_dial_pick_never_takes_the_configuration_without_rows():
    entry = {
        "configurations": {
            "plain": {"driver_risk": None, "tiers": {"A": {"mean_offer_kwh_per_test_day": 99.0}}},
            "low": {"driver_risk": 0.03, "tiers": {"A": {"mean_offer_kwh_per_test_day": 1.0}}},
            "high": {"driver_risk": 0.1, "tiers": {"A": {"mean_offer_kwh_per_test_day": 1.0}}},
        }
    }
    chosen = dial_choice._choice(entry, "A")
    assert chosen["dial"] == 0.03 and chosen["mean_offer_kwh_per_test_day"] == 1.0
    assert set(chosen["mean_offers_by_dial"]) == {"0.03", "0.1"}


def test_the_same_command_twice_gives_the_same_results(build, full, tmp_path):
    code_one, one = full
    code_two, two = _module_run(build, tmp_path, "--workers", "2")
    assert code_one == code_two == 0
    for name in (dial_choice.RESULTS_NAME, dial_choice.SUMMARY_NAME, dial_choice.TABLE_NAME):
        assert (one / name).read_bytes() == (two / name).read_bytes()
    assert "total_seconds" in json.loads((one / dial_choice.TIMING_NAME).read_text())


# --- 5. the summary ------------------------------------------------------------------------------


def test_the_summary_labels_every_line_and_keeps_the_tiers_apart(full):
    _, out = full
    document = _document(out)
    text = (out / dial_choice.SUMMARY_NAME).read_text()
    lines = text.splitlines()
    chosen = [line for line in lines if "has the largest mean offer" in line]
    assert len(chosen) == 2 * 2  # two units, two tiers
    assert all(
        "measured, not fixed: the plan fixes the dial at the main model" in x for x in chosen
    )
    assert text.count("tier A (driver-protected)\n") == 2
    assert text.count("tier B (deadline contract)\n") == 2
    assert not re.search(r"combined|overall|both tiers|across tiers", text)
    assert "never added or averaged" in text
    # every table row shows the offer of its own tier
    names = {"rows": PLAIN[0], "0.03": LOW[0], "0.1": HIGH[0]}
    unit, tier, rows = None, None, 0
    for line in lines:
        if line.startswith(tuple(GATE_UNITS)):
            unit = line.split(" ")[0]
        elif line.startswith(" tier "):
            tier = line.split()[1]
        elif unit in document["dial_choice"] and line.startswith(("  no rows", "  eps ")):
            cells = line.split()
            value = document["units"][unit]["configurations"][names[cells[1]]]["tiers"][tier]
            assert cells[2] == f"{value['mean_offer_kwh_per_test_day']:.2f}"
            rows += 1
    assert rows == 2 * 2 * 3


def test_the_results_hold_no_key_that_mixes_the_tiers(full):
    _, out = full
    document = _document(out)
    for entry in document["units"].values():
        for found in entry["configurations"].values():
            assert set(found["tiers"]) == {"A", "B"}
    assert all(set(tiers) == {"A", "B"} for tiers in document["dial_choice"].values())
    assert document["definitions"]["tiers"]["pooling"].endswith("never pooled or averaged")


# --- 6. timeouts ---------------------------------------------------------------------------------


def test_a_certificate_timeout_abstains_and_is_counted(build, full, tmp_path):
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

    base = _document(full[1])
    units = gate_units()
    first_date = sorted({c.date for c in units[GATE_UNITS[1]]})[0]
    code, out = _module_run(build, tmp_path, patch=patch)
    document = _document(out)
    assert code == 0
    usable = GATE_UNITS[1]
    for name in (PLAIN[0], LOW[0], HIGH[0]):
        before = base["units"][usable]["configurations"][name]["offering_days"]
        after = document["units"][usable]["configurations"][name]["offering_days"]
        assert after < before  # the day no longer offers
    assert document["timeouts"]["certificates"] >= 3
    assert "certificates" in (out / dial_choice.SUMMARY_NAME).read_text()


def test_a_missing_ceiling_is_counted(build, tmp_path):
    original = gate_day.day_ceilings

    def patch(monkeypatch):
        def ceilings(day):
            found = original(day)
            return dataclasses.replace(found, best_causal=None, best_causal_outcome="timeout")

        monkeypatch.setattr(gate_day, "day_ceilings", ceilings)

    code, out = _module_run(build, tmp_path, patch=patch)
    document = _document(out)
    assert code == 0 and document["timeouts"]["best_causal_ceilings"] == 120
    for found in document["units"][GATE_UNITS[1]]["configurations"].values():
        for tier in found["tiers"].values():
            assert tier["splits_with_ceiling_share"] == 0
            assert tier["mean_share_of_best_causal_ceiling"] is None


# --- 7. refusals ---------------------------------------------------------------------------------


def _argv(build, matrix=None, **extra):
    run, tables, committed = build
    arguments = ["--model", "empirical", "--matrix", str(matrix or committed)]
    arguments += ["--tables", str(tables), "--truth-run", str(run)]
    for key, value in extra.items():
        arguments += [f"--{key.replace('_', '-')}", str(value)]
    return arguments


def test_a_matrix_without_the_dial_family_could_not_run(build, tmp_path, capsys):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(matrix_document(PLAIN)))
    assert dial_choice.main(_argv(build, bare)) == 2
    assert "no dial family" in capsys.readouterr().out


def test_a_stale_truth_build_could_not_run(build, monkeypatch, capsys):
    monkeypatch.setattr(gate_run, "_head_commit", lambda: "b" * 40)
    assert dial_choice.main(_argv(build)) == 2
    assert "not at the checked-out commit" in capsys.readouterr().out


def test_a_changed_table_could_not_run(build, tmp_path, capsys):
    run, tables, matrix = build
    other = tmp_path / "tables"
    other.mkdir()
    for path in tables.iterdir():
        (other / path.name).write_bytes(path.read_bytes())
    path = other / f"{GATE_UNITS[1]}-cars.csv"
    path.write_text(path.read_text() + "\n")
    assert dial_choice.main(_argv((run, other, matrix))) == 2
    assert "differs" in capsys.readouterr().out


def test_a_truth_build_of_the_other_setting_could_not_run(build, tmp_path, capsys):
    from setting_fake import held_copy

    run, tables, matrix = build
    held_run, held_tables = held_copy(run, tables, tmp_path)
    assert dial_choice.main(_argv((run, held_tables, matrix))) == 2
    assert "of the held setting" in capsys.readouterr().out


def test_a_bad_argument_or_model_could_not_run(build, capsys):
    assert dial_choice.main(_argv(build, workers=0)) == 2
    assert dial_choice.main(_argv(build, splits=0)) == 2
    unknown = _argv(build)
    unknown[1] = "no_such_model"
    assert dial_choice.main(unknown) == 2
    assert capsys.readouterr().out.count("COULD NOT RUN") == 3

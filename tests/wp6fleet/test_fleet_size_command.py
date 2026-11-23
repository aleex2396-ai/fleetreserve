"""The fleet-size what-if command (`fleet_size_whatif`) on made-up gate units with fleets of 3 to 6
cars (`fleet_fake`).

1. A full run ends with exit 0, writes the four outputs, and its results carry the what-if text
   and no timing, verdict or recommendation.
2. A minimum of 0 gives `dial_choice`'s own figures, number for number, for the same matrix, model
   and splits; the default configurations are the ones `dial_choice` evaluates.
3. A larger minimum changes only what the definition says: the per-day rows do not depend on the
   minimum; the figures of a minimum are those of `dial_choice.score_splits` on the per-day rows
   with every day of fewer cars made a day that does not offer (rebuilt here from the rows); a
   minimum above the largest fleet gives an offer of 0.
4. The table by count adds up to the unit totals and to what the per-day rows say.
5. The steered-car count is the independent count on days where a forecaster leaves chosen cars
   without a forecast, under both rules for an uncertifiable car.
6. The per-day losses are those of the gate's own evaluation.
6b. The gate-reading figures of a minimum are rebuilt from the per-day rows of the configuration
   and of its oracle partner, each restricted by its own count of steered cars; with a
   calibration-days rule that halves small units the minimum 0 is still `dial_choice`.
7. One, two and three workers write identical files.
8. A failed configuration is recorded and the exit is 1.
9. Exit status 2 for a stale truth build, a changed table, a minimum that is negative or repeated,
   an unknown model, configuration or unit, a bad worker or split count; no minimum given is an
   argument error.
"""

import csv
import dataclasses
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from fleet_fake import varied_units
from gate_fake import COMMIT, GATE_UNITS, matrix_document, write_build

from fleetreserve import (
    dial_choice,
    dial_splits,
    fleet_size_whatif,
    fleet_size_whatif_summary,
    gate_calibrate,
    gate_config,
    gate_day,
    gate_run,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster

PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
HIGH = ("main-rounded-hour-k0.3-eps0.1-empirical", "main-rounded-hour-k0.3-eps0.1-oracle")
NAMES = PLAIN + LOW + HIGH
CONFIGURATIONS = (PLAIN[0], LOW[0], HIGH[0])
SPLITS = 12
MINIMUMS = ("0", "4", "5", "9")  # the made-up fleets have 3 to 6 cars
OUTPUTS = (
    fleet_size_whatif.ROWS_NAME,
    fleet_size_whatif.RESULTS_NAME,
    fleet_size_whatif.SUMMARY_NAME,
    fleet_size_whatif.TABLE_NAME,
)


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, varied_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _arguments(build, minimums):
    run, tables, matrix = build
    return [
        "--matrix",
        str(matrix),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        "--splits",
        str(SPLITS),
        "--min-cars",
        *minimums,
    ]


def command(build, out: Path, monkeypatch, *extra, minimums=MINIMUMS, model="empirical") -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return fleet_size_whatif.main(["--model", model, *_arguments(build, minimums), *extra])


def _module_run(build, folder: Path, *extra, patch=None, minimums=MINIMUMS):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out = folder / "out"
        return command(build, out, monkeypatch, *extra, minimums=minimums), out


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    return _module_run(build, tmp_path_factory.mktemp("full"), "--workers", "1")


def _document(out: Path) -> dict:
    return json.loads((out / fleet_size_whatif.RESULTS_NAME).read_text())


def _rows(out: Path) -> list[dict]:
    with (out / fleet_size_whatif.ROWS_NAME).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _configurations(document, unit):
    return document["units"][unit]["configurations"]


# --- 1. a full run -------------------------------------------------------------------------------


def test_a_full_run_writes_every_output(full):
    code, out = full
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUTS)
    document = _document(out)
    assert document["label"] == fleet_size_whatif_summary.LABEL
    assert document["failed_configurations"] == [] and set(document["timeouts"].values()) == {0}
    assert document["run"]["min_cars"] == [int(n) for n in MINIMUMS]
    assert document["run"]["configurations"] == list(CONFIGURATIONS)
    assert document["run"]["scales"] == list(dial_splits.GRID)
    assert document["model"] == "empirical" and document["inputs"]["setting"] == "main"
    assert document["forecast_table"]["rows"] > 0
    small, usable, pooled = (document["units"][u] for u in GATE_UNITS)
    assert small["status"].startswith("not calibratable") and small["configurations"] == {}
    for entry in (usable, pooled):
        assert list(entry["configurations"]) == list(CONFIGURATIONS)
        for found in entry["configurations"].values():
            assert [m["min_cars"] for m in found["minimums"]] == [int(n) for n in MINIMUMS]
            for item in found["minimums"]:
                assert set(item["tiers"]) == {"A", "B"}
                assert item["tiers"]["A"]["splits"] == SPLITS


def test_the_results_say_it_is_a_what_if_and_hold_no_clock_and_no_verdict(full):
    code, out = full
    document = _document(out)
    text = document["definitions"]["what_if"]
    assert "no minimum number of steered cars" in text and "not a rule" in text
    assert "recommendation" in text  # only to say there is none
    assert "len(PreparedDay.steered)" in document["definitions"]["steered_cars"]

    def keys(value):
        if isinstance(value, dict):
            for key, inner in value.items():
                yield key
                yield from keys(inner)
        elif isinstance(value, list):
            for inner in value:
                yield from keys(inner)

    assert not [k for k in keys(document) if "second" in k or "time" in k.replace("timeouts", "")]
    shown = (out / fleet_size_whatif.SUMMARY_NAME).read_text()
    for word in ("NO GO", "GO:", "verdict", "recommend:", "we recommend", "should"):
        assert word not in shown
    assert "never added or averaged" in shown


def test_the_summary_is_the_text_on_standard_output_and_shows_the_results(
    build, tmp_path, monkeypatch, capsys
):
    assert command(build, tmp_path / "out", monkeypatch) == 0
    shown = capsys.readouterr().out
    out = tmp_path / "out"
    assert shown == (out / fleet_size_whatif.SUMMARY_NAME).read_text()
    document = _document(out)
    assert shown.count("tier A (driver-protected)\n") == 2 * len(CONFIGURATIONS)
    assert shown.count("tier B (deadline contract)\n") == 2 * len(CONFIGURATIONS)
    lines = shown.splitlines()
    for unit in GATE_UNITS[1:]:
        for name, found in _configurations(document, unit).items():
            first = next(i for i, line in enumerate(lines) if line.startswith(f"{unit} ("))
            head = f" {name} " + _kind(found)
            start = next(i for i in range(first, len(lines)) if lines[i] == head)
            block = lines[start : start + 40]
            offers = [
                line.split()[2] for line in block if re.match(r"  min \d+ ", line)
            ]  # tier A rows then tier B rows
            wanted = [
                f"{m['tiers'][tier]['mean_offer_kwh_per_test_day']:.2f}"
                for tier in ("A", "B")
                for m in found["minimums"]
            ]
            assert offers[: len(wanted)] == wanted


def _kind(found) -> str:
    dial = found["driver_risk"]
    return "(no driver-risk rows)" if dial is None else f"(eps {dial:g})"


# --- 2. a minimum of 0 is dial_choice ------------------------------------------------------------


@pytest.fixture(scope="module")
def dial(build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("dial")
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out = folder / "out"
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        run, tables, matrix = build
        argv = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
        argv += ["--truth-run", str(run), "--splits", str(SPLITS)]
        assert dial_choice.main(argv) == 0
    return json.loads((out / dial_choice.RESULTS_NAME).read_text())


def test_a_minimum_of_zero_gives_dial_choices_own_figures(full, dial):
    document = _document(full[1])
    assert document["run"]["configurations"] == list(dial["units"][GATE_UNITS[1]]["configurations"])
    checked = 0
    for unit in GATE_UNITS[1:]:
        theirs = dial["units"][unit]["configurations"]
        mine = _configurations(document, unit)
        assert list(mine) == list(theirs)
        for name, found in mine.items():
            zero = found["minimums"][0]
            assert zero["min_cars"] == 0 and zero["counting_days_below"] == 0
            assert zero["offering_days_removed"] == 0
            assert zero["offering_days"] == theirs[name]["offering_days"] == found["offering_days"]
            assert zero["tiers"] == theirs[name]["tiers"]  # every number, tier A and tier B
            assert zero["gate_reading"] == theirs[name]["gate_reading"]  # and the gate reading
            assert zero["gate_reading"]["ceilings"]  # which is not empty
            assert found["evaluations"] == theirs[name]["evaluations"]
            checked += 1
    assert checked == 2 * len(CONFIGURATIONS)
    for unit in GATE_UNITS:
        for field in ("counting_days", "calibration_days_used", "test_days", "status"):
            assert dial["units"][unit][field] == document["units"][unit][field], (unit, field)


# --- 3. a larger minimum -------------------------------------------------------------------------


def _unit_rows(out: Path, unit: str, name: str, role: str = "configuration") -> list[dict]:
    return [
        r for r in _rows(out) if (r["unit"], r["configuration"], r["role"]) == (unit, name, role)
    ]


def _losses_from_rows(rows: list[dict], counts, minimum: int, ceiling) -> dial_splits.UnitLosses:
    """The unit's losses rebuilt from the per-day rows, with every day of fewer than `minimum` cars
    made a day that does not offer, by a plain loop."""
    scales = [f"{g:g}" for g in dial_splits.GRID]
    value = np.zeros(len(rows))
    offering = np.zeros(len(rows), dtype=bool)
    slot = np.zeros((len(rows), len(scales)))
    driver = np.zeros((len(rows), len(scales)))
    for d, row in enumerate(rows):
        if counts[d] < minimum or not int(row["offers"]):
            continue
        value[d], offering[d] = float(row["offer_at_scale_1_kwh"]), True
        slot[d] = [float(row[f"slot_loss_at_{s}"]) for s in scales]
        driver[d] = [float(row[f"driver_loss_at_{s}"]) for s in scales]
    return dial_splits.UnitLosses(value, offering, slot, driver, ceiling)


@pytest.fixture(scope="module")
def ceilings(build):
    """Best-causal ceilings per unit, made with the gate's own function."""
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    found = {}
    for unit in GATE_UNITS[1:]:
        days = load_unit_days(
            tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        found[unit] = (
            days,
            np.array(
                [
                    np.nan if c.best_causal is None else c.best_causal
                    for c in (gate_day.day_ceilings(t.day) for t in days.days)
                ]
            ),
        )
    return found


def test_the_per_day_rows_do_not_depend_on_the_minimum(build, full, tmp_path):
    code, out = _module_run(build, tmp_path, minimums=("0",))
    assert code == 0
    mine = (out / fleet_size_whatif.ROWS_NAME).read_bytes()
    assert mine == (full[1] / fleet_size_whatif.ROWS_NAME).read_bytes()
    only = _document(out)
    document = _document(full[1])
    for unit in GATE_UNITS[1:]:
        for name, found in _configurations(only, unit).items():
            assert found["by_count"] == _configurations(document, unit)[name]["by_count"]
            assert found["minimums"][0] == _configurations(document, unit)[name]["minimums"][0]


def test_the_figures_of_a_minimum_are_the_dial_choice_figures_on_the_reduced_days(
    build, full, ceilings
):
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    document = _document(full[1])
    changed = 0
    for unit in GATE_UNITS[1:]:
        days, ceiling = ceilings[unit]
        indices = dial_choice.split_indices(unit, days, matrix, range(1, SPLITS + 1))
        for name, found in _configurations(document, unit).items():
            rows = _unit_rows(full[1], unit, name)
            counts = [int(r["steered_cars"]) for r in rows]
            for item in found["minimums"]:
                losses = _losses_from_rows(rows, counts, item["min_cars"], ceiling)
                expected = dial_choice.score_splits(losses, indices, dial_splits.TIERS)
                assert item["tiers"] == expected, (unit, name, item["min_cars"])
                below = [c < item["min_cars"] for c in counts]
                assert item["counting_days_below"] == sum(below)
                assert item["offering_days_removed"] == sum(
                    b and r["offers"] == "1" for b, r in zip(below, rows, strict=True)
                )
                assert item["offering_days"] == int(losses.offering.sum())
                changed += item["min_cars"] > 0 and item["offering_days_removed"] > 0
    assert changed >= 4  # the minimums do remove offering days here


def test_a_minimum_above_the_largest_fleet_offers_nothing(full):
    document = _document(full[1])
    for unit in GATE_UNITS[1:]:
        for found in _configurations(document, unit).values():
            top = max(r["steered_cars"] for r in found["by_count"])
            assert top == 6
            huge = found["minimums"][-1]
            assert huge["min_cars"] == 9 and huge["offering_days"] == 0
            for tier in huge["tiers"].values():
                assert tier["mean_offer_kwh_per_test_day"] == 0.0
                assert tier["splits_that_offer"] == 0 and tier["mean_scale"] == 0.0
            zero = found["minimums"][0]
            assert zero["offering_days"] == found["offering_days"] > 0


def test_a_minimum_changes_only_the_days_below_it(full):
    """Between two minimums the offering days that are lost are exactly those with the counts in
    between."""
    document = _document(full[1])
    for unit in GATE_UNITS[1:]:
        for name, found in _configurations(document, unit).items():
            rows = _unit_rows(full[1], unit, name)
            for item in found["minimums"]:
                lost = sum(
                    r["offers"] == "1" and int(r["steered_cars"]) < item["min_cars"] for r in rows
                )
                assert item["offering_days"] == found["offering_days"] - lost


# --- 4. the table by count -----------------------------------------------------------------------


def test_the_table_by_count_adds_up_and_matches_the_rows(full):
    document = _document(full[1])
    scales = [f"{g:g}" for g in dial_splits.GRID]
    distinct = set()
    for unit in GATE_UNITS[1:]:
        for name, found in _configurations(document, unit).items():
            rows = _unit_rows(full[1], unit, name)
            assert len(rows) == document["units"][unit]["counting_days"]
            table = found["by_count"]
            counts = [r["steered_cars"] for r in table]
            assert counts == sorted(set(counts))  # one row per exact count, increasing
            distinct.add(tuple(counts))
            by = defaultdict(list)
            for row in rows:
                by[int(row["steered_cars"])].append(row)
            assert sorted(by) == counts
            for row in table:
                mine = by[row["steered_cars"]]
                assert row["counting_days"] == len(mine)
                assert row["offering_days"] == sum(r["offers"] == "1" for r in mine)
                assert row["offer_at_scale_1_kwh_sum"] == pytest.approx(
                    sum(float(r["offer_at_scale_1_kwh"]) for r in mine)
                )
                for k, scale in enumerate(scales):
                    cell = row["scales"][k]
                    slot = [float(r[f"slot_loss_at_{scale}"]) for r in mine]
                    driver = [float(r[f"driver_loss_at_{scale}"]) for r in mine]
                    assert cell["slot_loss_sum"] == pytest.approx(sum(slot))
                    assert cell["driver_loss_sum"] == pytest.approx(sum(driver))
                    assert cell["days_with_a_slot_loss"] == sum(x > 0 for x in slot)
                    assert cell["days_with_a_driver_loss"] == sum(x > 0 for x in driver)
            assert sum(r["counting_days"] for r in table) == len(rows)
            assert sum(r["offering_days"] for r in table) == found["offering_days"]
            assert sum(r["offer_at_scale_1_kwh_sum"] for r in table) == pytest.approx(
                sum(float(r["offer_at_scale_1_kwh"]) for r in rows)
            )
    assert all(len(c) >= 3 for c in distinct)  # the units have several counts


# --- 5. the steered-car count --------------------------------------------------------------------


class Blanking(EmpiricalForecaster):
    """Leaves the first two cars of a day without a forecast on days of the month divisible by 3."""

    def forecast_car(self, truth, i, kappa, mode, departure):
        if int(truth.record.date[-2:]) % 3 == 0 and i < 2:
            return None
        return super().forecast_car(truth, i, kappa, mode, departure)


def blanking(unit_days, matrix):
    return Blanking(unit_days, reference=matrix.reference, min_cell=matrix.min_cell)


def _hand_run(build, configuration, minimums=(0,)):
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    unit = GATE_UNITS[1]
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    document, rows = fleet_size_whatif.run(
        matrix, {unit: days}, blanking, (configuration,), (unit,), minimums, splits=2, workers=1
    )
    return days, [r for r in rows if r["role"] == "configuration"], document


def test_the_count_is_the_cars_the_plan_steers(build):
    matrix = gate_config.load(build[2])
    leave = matrix.configuration(PLAIN[0])
    assert leave.uncertifiable_car == "leave-unsteered"
    days, rows, document = _hand_run(build, leave, (0, 4, 5))
    assert len(rows) == len(days.days) > 0
    blanked_days, by_hand = 0, []
    for truth, row in zip(days.days, rows, strict=True):
        cars = len(truth.day.cars)
        blank = min(2, cars) if int(truth.record.date[-2:]) % 3 == 0 else 0
        assert row["cars"] == cars and row["date"] == truth.record.date
        assert row["steered_cars"] == cars - blank  # counted by hand: the cars with a forecast
        blanked_days += blank > 0
        by_hand.append(cars - blank)
    assert blanked_days >= 10
    # the minimum and the table use the count of steered cars, not the size of the fleet
    found = document["units"][GATE_UNITS[1]]["configurations"][leave.name]
    assert {r["steered_cars"]: r["counting_days"] for r in found["by_count"]} == {
        n: by_hand.count(n) for n in set(by_hand)
    }
    for item in found["minimums"]:
        n = item["min_cars"]
        assert item["counting_days_below"] == sum(h < n for h in by_hand)
        assert item["offering_days_removed"] == sum(
            h < n and row["offers"] for h, row in zip(by_hand, rows, strict=True)
        )
    fleets = [row["cars"] for row in rows]
    assert [sum(h < 4 for h in by_hand), sum(c < 4 for c in fleets)] != [0, 0]
    assert sum(h < 4 for h in by_hand) != sum(c < 4 for c in fleets)  # the two counts differ


def test_a_day_that_abstains_for_an_uncertifiable_car_has_no_steered_car(build):
    matrix = gate_config.load(build[2])
    strict = dataclasses.replace(matrix.configuration(PLAIN[0]), uncertifiable_car="day-abstains")
    days, rows, _ = _hand_run(build, strict)
    for truth, row in zip(days.days, rows, strict=True):
        blanked = int(truth.record.date[-2:]) % 3 == 0
        assert row["steered_cars"] == (0 if blanked else len(truth.day.cars))
        if blanked:
            assert row["offers"] == 0 and "day-abstains" in row["reason_for_not_offering"]


# --- 6. the losses are the gate's -----------------------------------------------------------------


def test_the_per_day_losses_are_those_of_the_gate_s_evaluation(build, full):
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    configuration = matrix.configuration(LOW[0])
    unit = GATE_UNITS[2]
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    forecaster = EmpiricalForecaster(days, reference=matrix.reference, min_cell=matrix.min_cell)
    rows = _unit_rows(full[1], unit, LOW[0])
    nonzero = 0
    for truth, row in zip(days.days, rows, strict=True):
        prepared = gate_day.prepare_day(truth, forecaster, configuration, gate_run.settings(matrix))
        assert row["steered_cars"] == str(len(prepared.steered))
        assert row["offers"] == str(int(not prepared.abstains))
        assert float(row["offer_at_scale_1_kwh"]) == prepared.value
        for scale in (0.5, 1.0):
            found = gate_day.evaluate(prepared, scale).rungs["one-way"]
            assert float(row[f"slot_loss_at_{scale:g}"]) == found.slot_loss
            assert float(row[f"driver_loss_at_{scale:g}"]) == found.driver_loss
            nonzero += found.slot_loss > 0
    assert nonzero >= 1  # some day loses something, so this compares more than zeros


def test_the_rows_have_the_documented_columns_and_one_row_per_day(full):
    rows = _rows(full[1])
    assert tuple(rows[0]) == fleet_size_whatif.COLUMNS
    document = _document(full[1])
    days = sum(document["units"][u]["counting_days"] for u in GATE_UNITS[1:])
    assert len(rows) == 2 * len(CONFIGURATIONS) * days  # each configuration and its oracle partner
    partners = {c: c.replace("empirical", "oracle") for c in CONFIGURATIONS}
    for row in rows:
        want = (
            row["configuration"]
            if row["role"] == "configuration"
            else partners[row["configuration"]]
        )
        assert row["evaluated"] == want


# --- 6b. the gate reading and the calibration-days rule -------------------------------------------


class OracleKeepsItsCars(Blanking):
    """As `Blanking`, but the oracle-departure forecasts keep every car, so the oracle partner of a
    configuration steers more cars than the configuration on the blanked days."""

    def forecast_car(self, truth, i, kappa, mode, departure="empirical"):
        if departure == "oracle":
            return EmpiricalForecaster.forecast_car(self, truth, i, kappa, mode, departure)
        return super().forecast_car(truth, i, kappa, mode, departure)


def keeping(unit_days, matrix):
    return OracleKeepsItsCars(unit_days, reference=matrix.reference, min_cell=matrix.min_cell)


def test_the_gate_reading_restricts_the_oracle_partner_by_its_own_count(build):
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    configuration = matrix.configuration(PLAIN[0])
    unit = GATE_UNITS[1]
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    minimums = (0, 4, 5, 6)
    document, rows = fleet_size_whatif.run(
        matrix, {unit: days}, keeping, (configuration,), (unit,), minimums, splits=6, workers=1
    )
    found = document["units"][unit]["configurations"][configuration.name]
    own = [r for r in rows if r["role"] == "configuration"]
    partner = [r for r in rows if r["role"] == "oracle partner"]
    assert {r["evaluated"] for r in partner} == {PLAIN[1]} and len(partner) == len(own)
    own_counts = [int(r["steered_cars"]) for r in own]
    oracle_counts = [int(r["steered_cars"]) for r in partner]
    blanked = [int(t.record.date[-2:]) % 3 == 0 for t in days.days]
    assert oracle_counts == [len(t.day.cars) for t in days.days]  # the partner keeps every car
    assert own_counts == [
        len(t.day.cars) - (min(2, len(t.day.cars)) if b else 0)
        for t, b in zip(days.days, blanked, strict=True)
    ]
    assert found["oracle_count_differs_on_days"] == sum(blanked) > 5
    ceilings = {
        name: np.array(
            [
                np.nan if v is None else v
                for v in (
                    gate_calibrate.ceiling_values(gate_day.day_ceilings(t.day), name)
                    for t in days.days
                )
            ]
        )
        for name in matrix.ceilings
    }
    indices = dial_choice.split_indices(unit, days, matrix, range(1, 7))
    best = np.array(
        [
            np.nan if c is None else c
            for c in (gate_day.day_ceilings(t.day).best_causal for t in days.days)
        ]
    )
    differing = 0
    for item in found["minimums"]:
        n = item["min_cars"]
        mine = _losses_from_rows(own, own_counts, n, best)
        theirs = _losses_from_rows(partner, oracle_counts, n, best)
        expected = dial_splits.summarise_gate(
            [dial_splits.gate_split(mine, theirs, ceilings, s, c, t) for s, c, t in indices],
            matrix.ceilings,
        )
        expected["oracle_partner"] = PLAIN[1]
        assert item["gate_reading"] == expected, n
        # the configuration's own count applied to the partner would give another reading
        wrong = _losses_from_rows(partner, own_counts, n, best)
        other = dial_splits.summarise_gate(
            [dial_splits.gate_split(mine, wrong, ceilings, s, c, t) for s, c, t in indices],
            matrix.ceilings,
        )
        differing += (
            other["mean_oracle_offer_kwh_per_test_day"]
            != expected["mean_oracle_offer_kwh_per_test_day"]
        )
    assert differing >= 1
    assert found["minimums"][-1]["gate_reading"]["mean_offer_kwh_per_test_day"] == 0.0


def test_a_configuration_without_an_oracle_partner_has_no_gate_reading(build):
    _, tables, matrix_file = build
    matrix = gate_config.load(matrix_file)
    oracle = matrix.configuration(PLAIN[1])
    assert oracle.oracle_partner is None
    unit = GATE_UNITS[1]
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    document, rows = fleet_size_whatif.run(
        matrix, {unit: days}, blanking, (oracle,), (unit,), (0, 5), splits=2, workers=1
    )
    found = document["units"][unit]["configurations"][oracle.name]
    assert found["oracle_partner"] is None and "oracle_failed" not in found
    assert [m["gate_reading"] for m in found["minimums"]] == [
        {"not_computed": "the configuration has no oracle partner"}
    ] * 2
    assert {r["role"] for r in rows} == {"configuration"}
    assert document["failed_configurations"] == []


def test_a_failing_oracle_partner_is_recorded_and_the_other_numbers_stay(build, tmp_path):
    real = fleet_size_whatif._measure_oracle

    def failing(*arguments):
        raise RuntimeError("partner made to fail")

    code, out = _module_run(
        build,
        tmp_path,
        "--configuration",
        LOW[0],
        patch=lambda m: m.setattr(fleet_size_whatif, "_measure_oracle", failing),
    )
    assert real is not failing and code == 1
    document = _document(out)
    assert len(document["failed_configurations"]) == 2
    assert all(
        f"{LOW[1]}: RuntimeError: partner made to fail" in f
        for f in document["failed_configurations"]
    )
    for unit in GATE_UNITS[1:]:
        found = _configurations(document, unit)[LOW[0]]
        assert [m["tiers"]["A"]["splits"] for m in found["minimums"]] == [SPLITS] * len(MINIMUMS)
        assert all(
            m["gate_reading"]["failed"].startswith("RuntimeError") for m in found["minimums"]
        )
    assert {r["role"] for r in _rows(out)} == {"configuration"}


def test_the_summary_shows_the_gate_reading_of_each_minimum(full):
    code, out = full
    document = _document(out)
    lines = (out / fleet_size_whatif.SUMMARY_NAME).read_text().splitlines()
    run = document["run"]
    checked = 0
    for unit in GATE_UNITS[1:]:
        start = next(i for i, line in enumerate(lines) if line.startswith(f"{unit} ("))
        for name, found in _configurations(document, unit).items():
            head = next(
                i for i in range(start, len(lines)) if lines[i] == f" {name} {_kind(found)}"
            )
            tail = next(i for i in range(head, len(lines)) if "Gate reading, tier A" in lines[i])
            for ceiling in run["ceilings"]:
                block = next(
                    i for i in range(tail, len(lines)) if lines[i] == f"  ceiling {ceiling}"
                )
                rows = [line.split() for line in lines[block + 2 : block + 2 + len(MINIMUMS)]]
                for cells, item in zip(rows, found["minimums"], strict=True):
                    reading = item["gate_reading"]
                    assert cells[:2] == ["min", str(item["min_cars"])]
                    assert cells[2] == f"{reading['mean_offer_kwh_per_test_day']:.2f}"
                    assert cells[5] == f"{reading['mean_oracle_offer_kwh_per_test_day']:.2f}"
                    checked += 1
    assert checked == 2 * len(CONFIGURATIONS) * len(run["ceilings"]) * len(MINIMUMS)
    assert "Calibration days per unit: n_calibration 50" in "\n".join(lines)
    assert (
        f"{document['units'][GATE_UNITS[1]]['calibration_days_used']} calibration days"
        in "\n".join(lines)
    )


@pytest.fixture(scope="module")
def half_rule(build, tmp_path_factory):
    """The usable units under a matrix whose rule halves units of fewer than 100 days: 60 days
    split into 30 calibration days and 30 test days."""
    folder = tmp_path_factory.mktemp("half")
    run, tables, _ = build
    matrix = folder / "matrix.json"
    document = matrix_document(NAMES)
    document["calibration_half_below"] = 100
    matrix.write_text(json.dumps(document))
    outs = {}
    for label, module, extra in (
        ("whatif", fleet_size_whatif, ["--min-cars", "0", "5"]),
        ("dial", dial_choice, []),
    ):
        out = folder / label
        out.mkdir()
        with pytest.MonkeyPatch.context() as monkeypatch:
            _verified(monkeypatch)
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
            argv = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
            argv += ["--truth-run", str(run), "--splits", "4", *extra]
            assert module.main(argv) == 0
        outs[label] = out
    return outs


def test_under_the_half_the_days_rule_a_minimum_of_zero_is_still_dial_choice(half_rule):
    mine = json.loads((half_rule["whatif"] / fleet_size_whatif.RESULTS_NAME).read_text())
    theirs = json.loads((half_rule["dial"] / dial_choice.RESULTS_NAME).read_text())
    assert mine["run"]["calibration_half_below"] == theirs["run"]["calibration_half_below"] == 100
    for unit in GATE_UNITS:
        assert mine["units"][unit]["status"] == "calibratable"
        days = mine["units"][unit]["counting_days"]
        assert mine["units"][unit]["calibration_days_used"] == days // 2
        assert mine["units"][unit]["test_days"] == days - days // 2
        for name, found in _configurations(mine, unit).items():
            zero = found["minimums"][0]
            assert zero["tiers"] == theirs["units"][unit]["configurations"][name]["tiers"]
            assert (
                zero["gate_reading"]
                == theirs["units"][unit]["configurations"][name]["gate_reading"]
            )
    assert mine["units"][GATE_UNITS[0]]["counting_days"] == 10  # 5 and 5: split, then abstains
    assert [
        m["offering_days"]
        for m in _configurations(mine, GATE_UNITS[0])[CONFIGURATIONS[0]]["minimums"]
    ] == [10, 4]  # fleets of 3, 4, 5, 6, 3, 4, 5, 6, 3, 4 cars: four days have 5 or more


# --- 7. workers ----------------------------------------------------------------------------------


def test_one_and_three_workers_write_the_same_bytes(build, full, tmp_path):
    code, out = _module_run(build, tmp_path, "--workers", "3")
    assert code == 0
    for name in OUTPUTS:
        assert (out / name).read_bytes() == (full[1] / name).read_bytes(), name


# --- 8. selections and a failed configuration ----------------------------------------------------


def test_configurations_and_units_can_be_chosen(build, full, tmp_path):
    code, out = _module_run(
        build, tmp_path, "--configuration", LOW[0], "--unit", GATE_UNITS[2], minimums=("0", "5")
    )
    assert code == 0
    document = _document(out)
    assert list(document["units"]) == [GATE_UNITS[2]]
    assert list(_configurations(document, GATE_UNITS[2])) == [LOW[0]]
    assert document["run"]["min_cars"] == [0, 5]
    mine = _configurations(document, GATE_UNITS[2])[LOW[0]]
    theirs = _configurations(_document(full[1]), GATE_UNITS[2])[LOW[0]]
    assert mine["by_count"] == theirs["by_count"]
    assert [m for m in mine["minimums"]] == [theirs["minimums"][0], theirs["minimums"][2]]
    kept = [r for r in _rows(full[1]) if (r["unit"], r["configuration"]) == (GATE_UNITS[2], LOW[0])]
    assert _rows(out) == kept and kept
    assert {r["role"] for r in kept} == {"configuration", "oracle partner"}


def test_a_configuration_that_fails_is_recorded_and_the_exit_is_one(build, tmp_path):
    real = fleet_size_whatif.measure_configuration

    def failing(runner, name, n_days, configuration, *rest):
        if configuration.name == LOW[0]:
            raise RuntimeError("made to fail")
        return real(runner, name, n_days, configuration, *rest)

    code, out = _module_run(
        build,
        tmp_path,
        patch=lambda m: m.setattr(fleet_size_whatif, "measure_configuration", failing),
    )
    assert code == 1
    document = _document(out)
    assert len(document["failed_configurations"]) == 2
    assert all("RuntimeError: made to fail" in f for f in document["failed_configurations"])
    assert "by_count" in _configurations(document, GATE_UNITS[1])[PLAIN[0]]
    assert "failed" in _configurations(document, GATE_UNITS[1])[LOW[0]]
    assert "FAILED: " in (out / fleet_size_whatif.SUMMARY_NAME).read_text()


# --- 9. could not run ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, minimums, kwargs",
    [
        ((), ("-1",), {}),
        ((), ("2", "2"), {}),
        (("--workers", "0"), MINIMUMS, {}),
        (("--splits", "0"), MINIMUMS, {}),
        (("--unit", "no-such-unit"), MINIMUMS, {}),
        (("--unit", GATE_UNITS[1], "--unit", GATE_UNITS[1]), MINIMUMS, {}),
        (("--configuration", "no-such-configuration"), MINIMUMS, {}),
        (("--configuration", LOW[0], "--configuration", LOW[0]), MINIMUMS, {}),
        ((), MINIMUMS, {"model": "no_such_model"}),
    ],
)
def test_a_bad_argument_could_not_run(
    build, tmp_path, monkeypatch, capsys, extra, minimums, kwargs
):
    assert command(build, tmp_path / "out", monkeypatch, *extra, minimums=minimums, **kwargs) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


def test_a_missing_minimum_is_an_argument_error(build):
    run, tables, matrix = build
    arguments = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
    with pytest.raises(SystemExit) as stopped:
        fleet_size_whatif.main([*arguments, "--truth-run", str(run)])
    assert stopped.value.code == 2


def test_a_matrix_without_the_dial_family_could_not_run_unless_configurations_are_named(
    build, tmp_path, monkeypatch, capsys
):
    run, tables, _ = build
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(matrix_document(PLAIN)))
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    arguments = ["--model", "empirical", "--matrix", str(bare), "--tables", str(tables)]
    arguments += ["--truth-run", str(run), "--min-cars", "0"]
    assert fleet_size_whatif.main(arguments) == 2
    assert "no dial family" in capsys.readouterr().out


def test_a_stale_truth_build_could_not_run(build, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate_run, "_head_commit", lambda: "b" * 40)
    assert command(build, tmp_path / "out", monkeypatch) == 2
    assert "not at the checked-out commit" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


def test_a_changed_table_could_not_run(tmp_path, monkeypatch, capsys):
    run, tables = write_build(tmp_path, varied_units(20))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    table = tables / f"{GATE_UNITS[1]}-cars.csv"
    table.write_text(table.read_text() + "\n")
    assert command((run, tables, matrix), tmp_path / "out", monkeypatch) == 2
    assert "differs from the table" in capsys.readouterr().out

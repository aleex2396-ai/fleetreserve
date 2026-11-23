"""The share of days in E' that the learn run reports (plan section 5, T2's event E').

Two hand-made units, three cars a day (true departure hours 4, 5 and 6, rates 2, 2.5 and 3 kW,
rooms of 1 to 2.5 kWh), one service hour. Four configurations at kappa 0.1, each with its oracle
partner: `main` (known energy bounds, so E'_i is `d_hat <= d`), `floored` (the forecast
departure is rounded down), `absent` (a day with an uncertifiable car abstains) and `energy` (the
forecast rate, cap and need are the table's). The model forecasts a departure of 3.5 hours for
every car except on named days; its rate, cap and need are the cars' own except on `fast` days.

Unit one (25 days, 5 test days; hand count). Days 1 and 2 forecast 5.5 hours: the first two cars
(true hours 4 and 5) are outside E'_i and the third (hour 6) is inside; floored, 5.5 becomes 5 and
the second car is inside as well. Day 3 gives no forecast (no steered car, the day has no box).
Days 4 and 5 forecast 3.5.
  main, energy, absent: 5 test days, 4 certified, days in E' 2 (days 4 and 5); steered cars
    3 + 3 + 3 + 3 = 12, in E'_i 1 + 1 + 3 + 3 = 8.
  floored: days in E' 2; cars in E'_i 2 + 2 + 3 + 3 = 10 of 12.

Unit two (two sites, 24 site-days, 4 test days). Day 1: the first car forecasts 0.4 hours, so its
forecast set is empty (it needs more than the 0.8 kWh it can draw): not steered, not judged. Day
2: the forecast rate is 1 kW above the cars' own (outside E'_i by the rate alone). Day 3: every car
forecasts 1.0 hours, so the set is not empty but the box has value 0 (the gate abstains on it:
`gate_day.ZERO_VALUE`). Day 4: plain.
  main, floored (0.4 floors to 0): 4 test days, 3 certified (day 3 is not), days in E' 3; steered
    cars 2 + 3 + 3 = 8, in E'_i 8.
  absent: day 1 abstains (an uncertifiable car), so 2 certified, days in E' 2, cars 6 of 6.
  energy: days in E' 2 (days 1 and 4), cars 8, in E'_i 2 + 0 + 3 = 5.

The 5 days of the first made-up unit cannot be split, so it has no test day: its shares are null,
never 0 and never 1. The definition written into the results names the denominators.
"""

import csv
import json
import math
import sys
import types

import pandas as pd
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_cars, write_build
from learn_fake import small_matrix

from fleetreserve import gate_day, gate_run, learn_run, run_verify
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import KEY_COLUMNS, TABLE_COLUMNS
from fleetreserve.gate_config import parse

EARLY, LATE, SHORT, ZERO = 3.5, 5.5, 0.4, 1.0
CONFIGURATION = "main-k0.1-empirical"
PARTNER = "main-k0.1-oracle"
FLOORED, ABSENT, ENERGY = "floored-k0.1-empirical", "absent-k0.1-empirical", "energy-k0.1-empirical"
ONE, TWO = GATE_UNITS[1], GATE_UNITS[2]


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


class _ByDay:
    """A model that forecasts `EARLY` hours, `LATE` on the days in `late`, nothing on `blank`,
    `ZERO` on `zero`, `SHORT` for each day's first car on `short`; rate, cap and need are the
    car's own (`truth`: session -> (rate, room)), the rate one kW higher on `fast`. Days are
    (unit, site, date)."""

    name = "by_day"
    fit_units = ()

    def __init__(self, truth, **days):
        self.truth = truth
        self.days = days

    def fit(self, frame):
        pass

    def _in(self, name, key):
        return key in self.days.get(name, set())

    def predict(self, frame, kappas):
        rows = []
        for unit, site, date, session in frame[list(KEY_COLUMNS)].itertuples(index=False):
            key = (unit, site, date)
            rate, room = self.truth[session]
            d_hat = EARLY
            if self._in("late", key):
                d_hat = LATE
            if self._in("zero", key):
                d_hat = ZERO
            if self._in("short", key) and session.endswith("-0"):
                d_hat = SHORT
            if self._in("blank", key):
                d_hat = math.nan
            for kappa in kappas:
                row = dict.fromkeys(TABLE_COLUMNS, math.nan)
                row.update(unit=unit, site=site, date=date, session_id=session)
                row.update(kappa=float(kappa), rho=0.0, d_hat=d_hat, e_cap=room, e_need=room)
                row.update(r_hat=rate + (1.0 if self._in("fast", key) else 0.0))
                rows.append(row)
        return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def _units() -> dict:
    return {
        GATE_UNITS[0]: gate_cars(5),
        ONE: gate_cars(25, early_every=0),
        TWO: gate_cars(12, "B", early_every=0) + gate_cars(12, "C", early_every=0),
    }


def _matrix_document() -> dict:
    document = small_matrix()
    base = {c["name"]: c for c in document["configurations"]}
    for tag, change in (
        ("floored", {"departure_floor": "floored"}),
        ("absent", {"uncertifiable_car": "day-abstains"}),
        ("energy", {"energy_bounds": "empirical"}),
    ):
        for kind, source in (("empirical", CONFIGURATION), ("oracle", PARTNER)):
            entry = {**base[source], **change, "name": f"{tag}-k0.1-{kind}"}
            if kind == "empirical":
                entry["oracle_partner"] = f"{tag}-k0.1-oracle"
            document["configurations"].append(entry)
    return document


def _test_days(tables, matrix, unit) -> list[tuple]:
    days = load_unit_days(tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet="keep")
    plan = gate_run.plan_unit(unit, days, matrix)
    keys = [k.split() for k, role in zip(plan.keys, plan.roles, strict=True) if role == "test"]
    return [(unit, site, date) for date, site in keys]


class Built:
    def __init__(self, folder):
        self.run, self.tables = write_build(folder, _units())
        self.matrix_file = folder / "matrix.json"
        self.matrix_file.write_text(json.dumps(_matrix_document()))
        matrix = parse(json.loads(self.matrix_file.read_text()))
        one, two = _test_days(self.tables, matrix, ONE), _test_days(self.tables, matrix, TWO)
        assert len(one) == 5 and len(two) == 4
        self.days = {
            "late": set(one[:2]),
            "blank": {one[2]},
            "short": {two[0]},
            "fast": {two[1]},
            "zero": {two[2]},
        }
        self.truth = {
            car.session: (car.rate, car.room) for cars in _units().values() for car in cars
        }
        self.two = two

    def main(self, out, monkeypatch, *extra, matrix_file=None) -> int:
        out.mkdir()
        module = types.ModuleType("fleetreserve.models.by_day")
        module.make_model = lambda matrix: _ByDay(self.truth, **self.days)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--matrix", str(matrix_file or self.matrix_file)]
        arguments += ["--tables", str(self.tables), "--truth-run", str(self.run)]
        return learn_run.main([*arguments, "--model", "by_day", *extra])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return Built(tmp_path_factory.mktemp("e_prime"))


@pytest.fixture(scope="module")
def run(built, tmp_path_factory):
    out = tmp_path_factory.mktemp("e_prime_out") / "out"
    seen = {}
    real = learn_run.e_prime_figures

    def spy(*arguments):
        seen["arguments"] = arguments
        return real(*arguments)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setattr(learn_run, "e_prime_figures", spy)
        assert built.main(out, monkeypatch) == 0
    results = json.loads((out / learn_run.RESULTS_NAME).read_text())
    return out, results, seen["arguments"]


def _found(run, unit, configuration):
    return run[1]["units"][unit]["e_prime"][configuration]


def _counts(found) -> tuple:
    return (
        found["test_days"],
        found["certified_test_days"],
        found["days_in_e_prime"],
        found["steered_cars_on_certified_test_days"],
        found["steered_cars_in_e_prime_i"],
    )


def test_two_of_five_days_are_in_the_event(run):
    found = _found(run, ONE, CONFIGURATION)
    assert _counts(found) == (5, 4, 2, 12, 8)
    assert found["share_of_test_days_in_e_prime"] == pytest.approx(0.4)
    assert found["share_of_certified_test_days_in_e_prime"] == pytest.approx(0.5)
    assert found["share_of_steered_cars_in_e_prime_i"] == pytest.approx(8 / 12)


def test_the_oracle_partner_is_in_the_event_on_every_day(run):
    found = _found(run, ONE, PARTNER)
    assert found["test_days"] == found["certified_test_days"] == 5  # no forecast is read
    assert found["days_in_e_prime"] == 5
    assert found["share_of_steered_cars_in_e_prime_i"] == 1.0


def test_a_unit_with_no_scored_day_has_no_share(run):
    found = _found(run, GATE_UNITS[0], CONFIGURATION)
    assert found["test_days"] == 0 and found["certified_test_days"] == 0
    assert found["days_in_e_prime"] == 0
    for key in (
        "share_of_test_days_in_e_prime",
        "share_of_certified_test_days_in_e_prime",
        "share_of_steered_cars_in_e_prime_i",
    ):
        assert key in found and found[key] is None


def test_the_definition_names_the_denominators(run):
    text = run[1]["e_prime_definition"]
    for words in (
        "counting test days",
        "certified test days",
        "steered cars",
        "E'_i",
        "null",
        "zero_value_kwh",
        "t2_day",
    ):
        assert words in text
    assert "d_hat <= d" in text


def test_the_summary_has_a_line_per_unit_and_configuration(run):
    lines = learn_run.summary_lines(run[1])
    mine = [line for line in lines if "days in E'" in line]
    assert len(mine) == len(GATE_UNITS) * 8
    text = "\n".join(mine)
    assert f"{CONFIGURATION}: 2 of 5 counting test days in E' (40.0%), " in text
    assert "2 of 4 certified test days in E' (50.0%); 8 of 12 steered cars in E'_i (66.7%)" in text
    assert "no counting test day" in text


def test_the_summary_line_prints_both_shares_beside_each_other(run):
    lines = learn_run.summary_lines(run[1])
    start = next(i for i, x in enumerate(lines) if x.startswith(f"{TWO} ("))
    line = next(
        x for x in lines[start:] if x.startswith(f"  {ENERGY}: ") and "counting test days" in x
    )
    # unit two, energy bounds: 2 of 4 counting test days, 2 of 3 certified test days
    assert (
        "2 of 4 counting test days in E' (50.0%), 2 of 3 certified test days in E' (66.7%)" in line
    )


def test_the_other_outputs_do_not_hold_the_key(run):
    out = run[0]
    for name in (learn_run.GATE_NAME, learn_run.DAYS_NAME):
        assert "e_prime" not in (out / name).read_text()


def test_a_floored_departure_is_judged_as_the_gate_plans_it(run):
    assert _counts(_found(run, ONE, FLOORED)) == (5, 4, 2, 12, 10)  # 5.5 floors to 5: car two in
    assert _counts(_found(run, ONE, CONFIGURATION)) == (5, 4, 2, 12, 8)


def test_the_energy_bounds_count_as_well_as_the_departure(run):
    found = _found(run, TWO, ENERGY)  # day 2's forecast rate is above the true rate
    assert _counts(found) == (4, 3, 2, 8, 5)
    assert _counts(_found(run, TWO, CONFIGURATION)) == (4, 3, 3, 8, 8)


def test_a_car_with_an_empty_forecast_set_is_not_steered(run):
    found = _found(run, TWO, CONFIGURATION)  # day 1's first car: two steered cars, not three
    assert found["steered_cars_on_certified_test_days"] == 8
    assert _counts(_found(run, TWO, FLOORED)) == (4, 3, 3, 8, 8)


def test_a_day_abstains_when_the_rule_says_an_uncertifiable_car_does(run):
    found = _found(run, TWO, ABSENT)  # day 1 abstains with its empty-set car
    assert _counts(found) == (4, 2, 2, 6, 6)
    assert _counts(_found(run, ONE, ABSENT)) == (5, 4, 2, 12, 8)


def test_a_day_of_zero_value_is_not_certified(run, built):
    found = _found(run, TWO, CONFIGURATION)
    assert found["test_days"] - found["certified_test_days"] == 1
    _, site, date = built.two[2]
    with (run[0] / learn_run.DAYS_NAME).open() as handle:
        (row,) = [
            r
            for r in csv.DictReader(handle)
            if (r["unit"], r["configuration"], r["site"], r["date"])
            == (TWO, CONFIGURATION, site, date)
        ]
    assert row["abstain_reason"] == gate_day.ZERO_VALUE and int(row["steered_cars"]) == 3


def test_a_configuration_that_failed_has_a_reason_and_no_counts(run):
    matrix, units, table, rows = run[2]
    kept = [r for r in rows if r["configuration"] != PARTNER]
    found = learn_run.e_prime_figures(matrix, units, table, kept)
    assert set(found[ONE][PARTNER]) == {"reason"}
    assert _counts(found[ONE][CONFIGURATION]) == (5, 4, 2, 12, 8)
    summary = learn_run._e_prime_lines({PARTNER: found[ONE][PARTNER]})
    assert summary == [
        f"  {PARTNER}: no scored day, no share of days in E' (the configuration "
        "failed, no day was scored)"
    ]


def test_a_disagreeing_count_of_steered_cars_stops_the_run_with_a_true_message(run):
    matrix, units, table, rows = run[2]
    changed = [dict(r) for r in rows]
    target = next(
        r
        for r in changed
        if r["unit"] == ONE
        and r["configuration"] == CONFIGURATION
        and r["role"] == "test"
        and not r["abstain_reason"]
    )
    target["steered_cars"] = target["steered_cars"] + 1
    with pytest.raises(learn_run.EPrimeDisagrees) as stopped:
        learn_run.e_prime_figures(matrix, units, table, changed)
    message = str(stopped.value)
    for words in (target["date"], f"site {target['site']}", f"unit {ONE}", CONFIGURATION):
        assert words in message
    assert "disagree with the gate's count" in message


def test_the_command_stops_and_writes_nothing_when_the_counts_disagree(
    built, tmp_path, monkeypatch, capsys
):
    pair = tmp_path / "pair.json"
    pair.write_text(json.dumps(small_matrix()))
    real = learn_run._steered_in_e_prime
    monkeypatch.setattr(learn_run, "_steered_in_e_prime", lambda *a: (real(*a)[0] + 1, real(*a)[1]))
    out = tmp_path / "out"
    assert built.main(out, monkeypatch, matrix_file=pair) == 1
    shown = capsys.readouterr().out
    assert "FAILED: the steered cars found again for E'" in shown
    assert (
        "disagree with the gate's count" in shown and "the model could not be scored" not in shown
    )
    assert list(out.iterdir()) == []  # the results are written after the scoring: none exists

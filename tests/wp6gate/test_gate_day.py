"""The per-day work of the gate run (`gate_day`), on hand-checkable days.

1. Both ceilings on the one-car decision-hour example: one car, departure 2, rate 1 kW, room
   1 kWh, rho 0, one service hour. In hindsight the decision hour can be chosen per request path
   (draw 1 kWh in hour 0 when the request is low, in hour 1 when it is high), so b(S) - p(S) =
   1 kWh; a real-time rule must fix hour 0 at τ, and the car's 1 kWh then fixes hour 1: 0 kWh.
2. Uncertifiable cars: a car whose forecast set is empty (forecast departure 0.5 h at 2 kW cannot
   hold its 3 kWh need) and a car with no reference sample are left out of the steered fleet under
   "leave-unsteered" (and the other cars are certified), and make the whole day abstain under
   "day-abstains".
3. Forecast departure 2.5 h: "as-planned" keeps 2.5 in the forecast set and the survival bounds
   (3.7 h at hours 1 and 2, then none, so 3.7 is kept); "floored" gives 2 and 3.
4. Oracle departure: the departure bound of every hour is the car's true departure hour.
5. `evaluate` returns, per rung, what `day_replay.replay_day` gives (day losses the largest over
   the paths), and its T5 check finds exactly the breaks an independent check of the replay finds,
   on the recorded counterexample day of `tests/wp6/test_t5_ledger.py` over all 16 corner paths.
6. A break injected through the replay seam (one car's two-way energy lowered on one path) is found
   and written out with the car, the rung and both energies.
7. `departure_seen`: the configuration's reading is kept by `prepare_day` and reaches
   `replay_day` (a stand-in records it).
8. The two readings give different replays on the made-up unit's early day (`gate_fake`): the
   second car leaves within service hour 1 while its forecast still has it plugged in, so the
   fixed plan plans power for it in hour 1 and fails there on every audited path. Under
   "hour-start" the one-way rung sees the car plugged in at the start of hour 1, copies the fixed
   plan and fails in hour 1 on every path without a re-solve; under "rounded-hour" it sees the
   car gone, re-solves hour 1 on every path, and at least one path no longer fails.
"""

import dataclasses
import datetime

import numpy as np
import pytest
from gate_fake import gate_cars, write_tables
from made_up_offer_days import departure_rule
from test_t5_ledger import COUNTEREXAMPLE

from fleetreserve import gate_day, products
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.day_replay import replay_day
from fleetreserve.days_from_truth import DayRecord, FleetCar, TruthDay, load_unit_days
from fleetreserve.forecast_empirical import EmpiricalForecaster
from fleetreserve.gate_config import Configuration
from fleetreserve.products import corner_paths
from fleetreserve.rungs import DEPARTURE_SEEN, RUNGS

DATE = datetime.date(2020, 1, 1)
SETTINGS = gate_day.DaySettings(0.0, 5.0, 1e-6, 1e-6, 1e-6)
MAIN = dict(
    name="c",
    family="main",
    kappa=0.1,
    departure="empirical",
    energy_bounds="known",
    uncertifiable_car="leave-unsteered",
    shape_rule="largest-smallest-width",
    departure_floor="as-planned",
    driver_loss_cars="steered-only",
    departure_seen="rounded-hour",
    oracle_partner="o",
    driver_risk=None,
)


def configuration(**change):
    return Configuration(**{**MAIN, **change})


def truth_day(day: Day) -> TruthDay:
    record = DayRecord(
        day.unit,
        "A",
        day.date.isoformat(),
        "t",
        len(day.cars),
        len(day.cars),
        0,
        0.0,
        0,
        0.0,
        0,
        0.0,
        False,
        False,
        0,
        0,
    )
    fleet = tuple(
        FleetCar(i, f"s{i}", "A", 8, False, float(c.departure) + 0.5)
        for i, c in enumerate(day.cars)
    )
    return TruthDay(day, record, fleet)


class Stub:
    """A forecaster with given forecasts (None: no reference sample) and survival bounds."""

    def __init__(self, forecasts, survival=None):
        self.forecasts = forecasts
        self.survival = survival or {}

    def forecast_car(self, truth, i, kappa, mode, departure):
        found = self.forecasts[i]
        if found is not None and departure == "oracle":
            found = dataclasses.replace(found, departure=float(truth.day.cars[i].departure))
        return found

    def survival_bound(self, truth, i, hour, kappa):
        return self.survival.get((i, hour))


def test_both_ceilings_on_the_decision_hour_example():
    day = Day("u", DATE, 1, 0, 0.0, (TrueCar(2, 1.0, 1.0),))
    found = gate_day.day_ceilings(day)
    assert found.hindsight == pytest.approx(1.0, abs=1e-9)
    assert found.best_causal_outcome == "optimal"
    assert found.best_causal == pytest.approx(0.0, abs=1e-7)


def test_a_day_without_cars_has_ceilings_zero():
    found = gate_day.day_ceilings(Day("u", DATE, 2, 0, 0.0, ()))
    assert (found.hindsight, found.best_causal) == (0.0, 0.0)


CARS = (TrueCar(4, 2.0, 2.0), TrueCar(4, 2.0, 3.0), TrueCar(5, 3.0, 2.0), TrueCar(4, 2.0, 1.0))
FORECASTS = [
    ForecastCar(4.0, 2.0, 2.0, 2.0),
    ForecastCar(0.5, 2.0, 3.0, 3.0),  # empty: at most 1 kWh before 0.5 h
    ForecastCar(5.0, 3.0, 2.0, 2.0),
    None,  # no reference sample
]


def test_uncertifiable_cars_are_left_unsteered_or_make_the_day_abstain():
    truth = truth_day(Day("u", DATE, 2, 3, 0.0, CARS))
    left = gate_day.prepare_day(truth, Stub(FORECASTS), configuration(), SETTINGS)
    assert left.unsteered == ((1, gate_day.EMPTY_SET), (3, gate_day.NO_REFERENCE))
    assert left.steered == (0, 2)
    assert left.abstain_reason is None and left.certificate_outcome == "optimal"
    assert left.day.cars == (CARS[0], CARS[2]) and left.value > 0
    assert left.n_cars == 4
    abstains = gate_day.prepare_day(
        truth, Stub(FORECASTS), configuration(uncertifiable_car="day-abstains"), SETTINGS
    )
    assert abstains.abstain_reason == gate_day.DAY_ABSTAINS
    assert abstains.steered == () and abstains.value == 0.0 and abstains.certificate is None
    evaluation = gate_day.evaluate(abstains, 1.0)
    assert evaluation.abstained and evaluation.paths_checked == 0
    assert all(r.slot_loss == r.driver_loss == 0.0 for r in evaluation.rungs.values())


def test_all_cars_certifiable_gives_the_same_day_under_both_rules():
    forecasts = [FORECASTS[0], FORECASTS[2]]
    truth = truth_day(Day("u", DATE, 2, 3, 0.0, (CARS[0], CARS[2])))
    a = gate_day.prepare_day(truth, Stub(forecasts), configuration(), SETTINGS)
    b = gate_day.prepare_day(
        truth, Stub(forecasts), configuration(uncertifiable_car="day-abstains"), SETTINGS
    )
    assert a.unsteered == b.unsteered == () and a.value == pytest.approx(b.value)


def test_floored_against_as_planned_departure():
    cars = (TrueCar(4, 2.0, 2.0), TrueCar(5, 3.0, 2.0))
    forecasts = [ForecastCar(2.5, 2.0, 2.0, 2.0), ForecastCar(5.0, 3.0, 2.0, 2.0)]
    survival = {(0, 1): 3.7, (0, 2): 3.7, (1, 1): 5.2, (1, 2): 5.4}
    truth = truth_day(Day("u", DATE, 3, 2, 0.0, cars))
    planned = gate_day.prepare_day(truth, Stub(forecasts, survival), configuration(), SETTINGS)
    floored = gate_day.prepare_day(
        truth, Stub(forecasts, survival), configuration(departure_floor="floored"), SETTINGS
    )
    assert planned.day.forecasts[0].departure == 2.5
    assert floored.day.forecasts[0].departure == 2.0
    np.testing.assert_array_equal(planned.departure_bounds[0], [2.5, 3.7, 3.7, 3.7])
    np.testing.assert_array_equal(floored.departure_bounds[0], [2.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(planned.departure_bounds[1], [5.0, 5.2, 5.4, 5.4])
    np.testing.assert_array_equal(floored.departure_bounds[1], [5.0, 5.0, 5.0, 5.0])


def test_the_oracle_variant_passes_the_true_departure():
    cars = (TrueCar(4, 2.0, 2.0), TrueCar(5, 3.0, 2.0))
    forecasts = [ForecastCar(2.5, 2.0, 2.0, 2.0), ForecastCar(5.0, 3.0, 2.0, 2.0)]
    truth = truth_day(Day("u", DATE, 3, 2, 0.0, cars))
    survival = {(0, 1): 3.7, (1, 1): 5.2}
    oracle = gate_day.prepare_day(
        truth,
        Stub(forecasts, survival),
        configuration(departure="oracle", oracle_partner=None),
        SETTINGS,
    )
    np.testing.assert_array_equal(oracle.departure_bounds, [[4.0] * 4, [5.0] * 4])
    assert [f.departure for f in oracle.day.forecasts] == [4.0, 5.0]


def _counterexample(paths, seen="rounded-hour"):
    day = COUNTEREXAMPLE
    certificate = certify(
        day_forecast_rows(day), products.block(), "causal", shape_rule=ANY_OPTIMAL
    )
    rule = departure_rule(day, seen)
    bounds = np.array([[rule(i, t) for t in range(day.n_service + 1)] for i in range(5)])
    return gate_day.PreparedDay(
        day.unit,
        "A",
        day.date.isoformat(),
        5,
        (),
        (0, 1, 2, 3, 4),
        None,
        "optimal",
        float(certificate.value),
        day,
        certificate,
        bounds,
        paths,
        5.0,
        1e-6,
        0.0,
        seen,
    ), rule


def _independent_breaks(result, tolerance=1e-6):
    found = set()
    for previous, name in zip(RUNGS, RUNGS[1:], strict=False):
        for k, (now, before) in enumerate(
            zip(result.by_rung[name], result.by_rung[previous], strict=True)
        ):
            a, b = now.first_failure, before.first_failure
            if a is not None and (b is None or a < b):
                found.add((k, "a", name, None))
            for car in np.flatnonzero(now.final < before.final - tolerance):
                found.add((k, "b", name, int(car)))
    return found


@pytest.mark.parametrize("seen", ["hour-start", "rounded-hour"])
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_evaluate_is_the_replay_and_its_t5_check_agrees(beta, seen):
    prepared, rule = _counterexample(corner_paths(COUNTEREXAMPLE.n_service), seen)
    evaluation = gate_day.evaluate(prepared, beta)
    result = replay_day(
        COUNTEREXAMPLE,
        prepared.certificate,
        "two-way",
        beta,
        prepared.paths,
        departure_bound=rule,
        departure_seen=seen,
    )
    for name in RUNGS:
        runs = result.by_rung[name]
        got = evaluation.rungs[name]
        assert got.slot_loss == max(r.slot_loss for r in runs)
        assert got.driver_loss == max(r.driver_loss for r in runs)
        assert got.first_failures == tuple(r.first_failure for r in runs)
        np.testing.assert_array_equal(got.finals, [r.final for r in runs])
        assert sum(got.resolves.values()) == sum(len(r.resolves) for r in runs)
    assert evaluation.paths_checked == 16
    found = {(b.path, b.kind, b.rung, b.car) for b in evaluation.t5_breaks}
    assert found == _independent_breaks(result)
    for b in evaluation.t5_breaks:
        if b.kind == "b":
            assert b.value < b.predecessor_value - 1e-6


def test_an_injected_break_is_found(monkeypatch):
    prepared, _ = _counterexample(corner_paths(COUNTEREXAMPLE.n_service)[:3])
    clean = gate_day.evaluate(prepared, 0.2)
    assert clean.t5_breaks == ()

    def lowered(*args, **kwargs):
        result = replay_day(*args, **kwargs)
        path = result.by_rung["two-way"][1]
        final = path.final.copy()
        final[3] = result.by_rung["one-way"][1].final[3] - 0.25
        changed = list(result.by_rung["two-way"])
        changed[1] = dataclasses.replace(path, final=final)
        return dataclasses.replace(result, by_rung={**result.by_rung, "two-way": tuple(changed)})

    monkeypatch.setattr(gate_day, "replay_day", lowered)
    found = gate_day.evaluate(prepared, 0.2).t5_breaks
    assert len(found) == 1
    (b,) = found
    assert (b.path, b.kind, b.rung, b.predecessor, b.car) == (1, "b", "two-way", "one-way", 3)
    assert b.predecessor_value - b.value == pytest.approx(0.25)


def test_departure_seen_is_passed_to_a_replay_that_takes_it(monkeypatch):
    seen = []

    def stand_in(
        day, certificate, rung, beta, paths, *, departure_bound, budget, time_limit, departure_seen
    ):
        seen.append((rung, beta, departure_seen))
        return "replayed"

    monkeypatch.setattr(gate_day, "replay_day", stand_in)
    prepared, _ = _counterexample(corner_paths(COUNTEREXAMPLE.n_service)[:2])
    for reading in ("hour-start", "rounded-hour"):
        assert gate_day.replay(dataclasses.replace(prepared, departure_seen=reading), 0.5) == (
            "replayed"
        )
    assert seen == [("two-way", 0.5, "hour-start"), ("two-way", 0.5, "rounded-hour")]

    truth = truth_day(Day("u", DATE, 2, 3, 0.0, (CARS[0], CARS[2])))
    found = gate_day.prepare_day(
        truth,
        Stub([FORECASTS[0], FORECASTS[2]]),
        configuration(departure_seen="hour-start"),
        SETTINGS,
    )
    assert found.departure_seen == "hour-start"


def test_the_two_readings_replay_a_car_leaving_within_the_hour_differently(tmp_path):
    write_tables(tmp_path, "u", gate_cars(60))
    days = load_unit_days(tmp_path, "u", rho=0.0, no_fit="cap", small_fleet="keep")
    forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
    truth = days.days[0]
    assert truth.day.n_service == 1 and truth.day.cars[1].departure == 1
    found = {}
    for seen in DEPARTURE_SEEN:
        prepared = gate_day.prepare_day(
            truth, forecaster, configuration(departure_seen=seen), SETTINGS
        )
        assert prepared.steered == (0, 1, 2) and prepared.departure_bounds[1, 1] > 1
        found[seen] = gate_day.evaluate(prepared, 0.5).rungs
    n_paths = len(found["hour-start"]["fixed"].first_failures)
    assert n_paths == 8
    for rungs in found.values():
        assert rungs["fixed"].first_failures == (1,) * n_paths
    hour_start, rounded = found["hour-start"]["one-way"], found["rounded-hour"]["one-way"]
    assert hour_start.first_failures == (1,) * n_paths and hour_start.resolves == {}
    assert sum(rounded.resolves.values()) == n_paths and rounded.resolves.get("optimal", 0) > 0
    assert None in rounded.first_failures

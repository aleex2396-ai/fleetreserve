"""A car that cannot pause on a small made-up day (`gate_day.prepare_day` with `no_pause`).

The day: three cars, two service hours, three recovery hours, every forecast known and
certifiable (departures 4, 4 and 5 hours, rates 2, 2 and 3 kW, rooms 2, 3 and 2 kWh). Car 1 is
the car that cannot pause.

1. It is left unsteered with the reason `NO_PAUSE`; the steered cars are 0 and 2; the forecaster
   is never asked about it, also with the dial set to 0.02, where the day without the draw asks
   about car 1 more than once (its departure scenarios).
2. It is in no programme and no box: the steered sub-day, the rows certified on, the certified
   value, the box (`m` and `W` of the policy) and the departure bounds equal those of the same
   day prepared without car 1 at all, and they do not change when car 1's true rate and room are
   changed (rate 50 kW, room 40 kWh).
3. It loses nothing: at the scales 0.2, 0.6 and 1, the rungs' slot and driver losses and final
   energies, and the losses and solver counts of every controller of `compare_methods.METHODS`
   (`compare_day.evaluate`), equal those of the day without car 1, and the final energies hold
   the two steered cars only. The controller `max-entropy-feedback` in particular, at the scales
   0.6, 1, 1.5 and 2: its losses and deviation counts are those of the day without car 1, with
   deviating hours at 1.5 and 2; every view it plans holds the two steered cars, every believed
   future the view's plugged cars, and its box is `m -+ beta w` of the drawn day at that scale;
   with car 1 steered its counts at 0.6 and 1 differ.
4. It is not an uncertifiable car: under "day-abstains" the day still offers; with a car without
   a forecast beside it, the day abstains as before.
5. A drawn car has the reason `NO_PAUSE` even when it has no forecast (a stand-in that gives none).
6. A day whose every car is drawn has no steered car and abstains ("no steered car"), offers 0
   and loses 0, and no car is forecast.
7. With `no_pause` empty the prepared day equals the one prepared without the argument.
8. A position that is not a car of the day is refused; a NumPy whole number is a position. The
   reason's text is "cannot pause (no-pause sensitivity)".
"""

import dataclasses
import datetime

import numpy as np
import pytest

from fleetreserve import compare_day, compare_methods, gate_day, mef_controller
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.days_from_truth import DayRecord, FleetCar, TruthDay
from fleetreserve.gate_config import Configuration

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
CARS = (TrueCar(4, 2.0, 2.0), TrueCar(4, 2.0, 3.0), TrueCar(5, 3.0, 2.0))
FORECASTS = (
    ForecastCar(4.0, 2.0, 2.0, 2.0),
    ForecastCar(4.0, 2.0, 3.0, 3.0),
    ForecastCar(3.6, 3.0, 2.0, 2.0),
)
SCALES = (0.2, 0.6, 1.0)
CONTROLLERS = tuple(m for m in compare_methods.METHODS if m.kind == "controller")


def configuration(**change):
    return Configuration(**{**MAIN, **change})


def truth_of(cars, sessions) -> TruthDay:
    day = Day("u", DATE, 2, 3, 0.0, tuple(cars))
    record = DayRecord(
        "u",
        "A",
        DATE.isoformat(),
        "t",
        len(cars),
        len(cars),
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
        FleetCar(i, s, "A", 8, False, float(c.departure) + 0.5)
        for i, (c, s) in enumerate(zip(cars, sessions, strict=True))
    )
    return TruthDay(day, record, fleet)


class Stub:
    """Forecasts by session; records the sessions it was asked about."""

    def __init__(self, by_session):
        self.by_session = by_session
        self.asked = []

    def forecast_car(self, truth, i, kappa, mode, departure):
        session = truth.fleet[i].session_id
        self.asked.append(session)
        return self.by_session[session]

    def survival_bound(self, truth, i, hour, kappa):
        return 3.9 if hour >= 1 else None


FULL = truth_of(CARS, ("s0", "s1", "s2"))
WITHOUT = truth_of((CARS[0], CARS[2]), ("s0", "s2"))
BY_SESSION = {"s0": FORECASTS[0], "s1": FORECASTS[1], "s2": FORECASTS[2]}


def prepare(truth, no_pause=(), forecasts=BY_SESSION, **change):
    stub = Stub(forecasts)
    found = gate_day.prepare_day(truth, stub, configuration(**change), SETTINGS, no_pause=no_pause)
    return found, stub


def _same_offer(a, b):
    assert a.abstain_reason is None and b.abstain_reason is None
    assert a.day.cars == b.day.cars and a.day.forecasts == b.day.forecasts
    assert a.day.n_recovery == b.day.n_recovery
    assert a.value == b.value > 0
    assert np.array_equal(a.certificate.policy.m, b.certificate.policy.m)
    assert np.array_equal(a.certificate.policy.w, b.certificate.policy.w)
    assert np.array_equal(a.departure_bounds, b.departure_bounds)
    assert a.rows.n_cars == b.rows.n_cars == 2
    for field in ("car", "coefficients", "bounds", "kinds"):
        assert np.array_equal(getattr(a.rows, field), getattr(b.rows, field)), field


def test_the_car_is_left_unsteered_and_never_forecast():
    found, stub = prepare(FULL, no_pause=(1,))
    assert found.unsteered == ((1, gate_day.NO_PAUSE),)
    assert found.steered == (0, 2) and found.n_cars == 3
    assert "s1" not in stub.asked and set(stub.asked) == {"s0", "s2"}


def test_the_car_is_not_forecast_when_the_dial_is_set():
    found, stub = prepare(FULL, no_pause=(1,), driver_risk=0.02)
    plain, asked = prepare(FULL, driver_risk=0.02)
    assert found.unsteered == ((1, gate_day.NO_PAUSE),) and found.steered == (0, 2)
    assert "s1" not in stub.asked and {"s0", "s2"} <= set(stub.asked)
    assert asked.asked.count("s1") > 1  # without the draw, car 1 is asked at several levels


def test_the_car_is_in_no_programme_and_no_box():
    drawn, _ = prepare(FULL, no_pause=(1,))
    alone, _ = prepare(WITHOUT)
    _same_offer(drawn, alone)
    bigger = truth_of((CARS[0], TrueCar(4, 50.0, 40.0), CARS[2]), ("s0", "s1", "s2"))
    changed, _ = prepare(bigger, no_pause=(1,))
    _same_offer(changed, alone)


def test_the_car_loses_nothing():
    drawn, _ = prepare(FULL, no_pause=(1,))
    alone, _ = prepare(WITHOUT)
    for beta in SCALES:
        a, b = gate_day.evaluate(drawn, beta), gate_day.evaluate(alone, beta)
        for rung in a.rungs:
            assert a.rungs[rung].slot_loss == b.rungs[rung].slot_loss
            assert a.rungs[rung].driver_loss == b.rungs[rung].driver_loss
            assert a.rungs[rung].finals.shape[1] == 2
            assert np.array_equal(a.rungs[rung].finals, b.rungs[rung].finals)
        c, d = (
            compare_day.evaluate(drawn, beta, CONTROLLERS),
            compare_day.evaluate(alone, beta, CONTROLLERS),
        )
        assert c == d


MEF = "max-entropy-feedback"


def test_the_feasibility_guard_sees_the_steered_cars_only(monkeypatch):
    """`max-entropy-feedback` (among `CONTROLLERS`, so also in the check above) at the scales 0.6
    to 2: its losses and deviation counts with car 1 drawn equal those of the day without car 1,
    some hours deviate there (at 1.5 and 2), every hour's view holds the two steered cars only,
    its believed future the cars of the view still plugged in, and its box is the drawn day's
    `m -+ beta w` at the scale evaluated. With car 1 steered instead, the counts at 0.6 and 1
    differ, so a drawn car reaching the check would show."""
    assert MEF in {m.name for m in CONTROLLERS}
    (method,) = compare_methods.select([MEF])
    seen = []

    class Seeing(mef_controller.MefController):
        def future(self, view, powers):
            found = super().future(view, powers)
            seen.append((view.n_cars, found.high.shape[0] == view.plugged.sum(), self.lo, self.hi))
            return found

    drawn, _ = prepare(FULL, no_pause=(1,))
    alone, _ = prepare(WITHOUT)
    full, _ = prepare(FULL)
    policy = drawn.certificate.policy
    deviating = {}
    for beta in (0.6, 1.0, 1.5, 2.0):
        with monkeypatch.context() as patch:
            patch.setattr(compare_methods, "MefController", Seeing)
            a = compare_day.evaluate(drawn, beta, [method])[MEF]
        assert seen and all(n == 2 and plugged for n, plugged, _, _ in seen)
        low, high = policy.m - beta * policy.w, policy.m + beta * policy.w
        assert all(np.array_equal(lo, low) and np.array_equal(hi, high) for *_, lo, hi in seen)
        seen.clear()
        b = compare_day.evaluate(alone, beta, [method])[MEF]
        assert a == b and a.deviation_counts is not None
        deviating[beta] = a.deviation_counts["deviating_hours"]
        c = compare_day.evaluate(full, beta, [method])[MEF]
        if beta < 1.5:
            assert c.deviation_counts != a.deviation_counts
    assert deviating[0.6] == deviating[1.0] == 0 and deviating[1.5] > 0 and deviating[2.0] > 0


def test_a_drawn_car_never_makes_the_day_abstain():
    drawn, _ = prepare(FULL, no_pause=(1,), uncertifiable_car="day-abstains")
    assert drawn.abstain_reason is None and drawn.value > 0
    without_forecast = dict(BY_SESSION, s2=None)
    both, _ = prepare(
        FULL, no_pause=(1,), forecasts=without_forecast, uncertifiable_car="day-abstains"
    )
    assert both.abstain_reason == gate_day.DAY_ABSTAINS
    assert both.unsteered == ((1, gate_day.NO_PAUSE), (2, gate_day.NO_REFERENCE))


def test_a_drawn_car_without_forecast_has_the_no_pause_reason():
    found, _ = prepare(FULL, no_pause=(1,), forecasts=dict(BY_SESSION, s1=None))
    assert found.unsteered == ((1, gate_day.NO_PAUSE),)


def test_a_day_whose_every_car_is_drawn_abstains():
    found, stub = prepare(FULL, no_pause=(0, 1, 2))
    assert found.abstain_reason == gate_day.NO_STEERED
    assert found.unsteered == tuple((i, gate_day.NO_PAUSE) for i in range(3))
    assert found.steered == () and found.value == 0.0 and stub.asked == []
    evaluation = gate_day.evaluate(found, 1.0)
    assert evaluation.abstained
    assert all(r.slot_loss == r.driver_loss == 0.0 for r in evaluation.rungs.values())


def test_no_car_drawn_is_the_day_without_the_argument():
    stub = Stub(BY_SESSION)
    plain = gate_day.prepare_day(FULL, stub, configuration(), SETTINGS)
    empty, _ = prepare(FULL, no_pause=())
    fields = [f.name for f in dataclasses.fields(plain) if f.name != "certificate_seconds"]
    for name in fields:
        a, b = getattr(plain, name), getattr(empty, name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), name
        elif name == "certificate":
            assert a.value == b.value and np.array_equal(a.policy.m, b.policy.m)
        elif name == "rows":
            assert np.array_equal(a.coefficients, b.coefficients)
            assert np.array_equal(a.bounds, b.bounds) and np.array_equal(a.car, b.car)
        elif name == "day":
            assert a.cars == b.cars and a.forecasts == b.forecasts
        else:
            assert a == b, name


def test_a_numpy_position_is_a_car():
    found, _ = prepare(FULL, no_pause=(np.int64(1),))
    assert found.unsteered == ((1, gate_day.NO_PAUSE),)
    assert gate_day.NO_PAUSE == "cannot pause (no-pause sensitivity)"


@pytest.mark.parametrize("bad", [(3,), (-1,), (True,), (1.0,)])
def test_a_position_that_is_not_a_car_is_refused(bad):
    with pytest.raises(ValueError, match="not a car of the day"):
        prepare(FULL, no_pause=bad)

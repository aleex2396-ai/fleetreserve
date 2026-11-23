"""Driver-risk rows (`driver_risk`, plan sections 2, 5, 6): the rows, the scenarios, the dial.

1. Field null: the day is the day without the rows: forecasts carry no scenarios, the forecaster
   is asked only at the configuration's kappa, and the certificate rows, the certified box and the
   replay losses equal those built from the unchanged pieces (`carsets.day_forecast_rows`,
   `certificate.certify`, `gate_day.replay`).
2. The rows equal the definition: on random points of a made-up car's forecast set, every row
   holds exactly when `sum_k pi_k max{need_k - E_k(u), 0} <= epsilon * need`, with `E_k` and
   `need_k` computed here straight from the plan's text (the most `E_k` can be from a linear
   programme), on cars with a level dropped for having no shortfall and with equal bounds merged.
3. The certified value never falls when the dial grows, falls below the value without rows at the
   dial 0, and is the value without rows at a dial large enough.
4. Scenario construction: running maximum, cap at the presence bound, a missing level read as the
   presence bound, merged equal bounds, the floored reading, the levels and masses of a smaller
   kappa.
5. The oracle departure: every scenario is the true departure, every row holds on the whole set,
   and the value equals the value without rows.
6. The matrix: the field is required, null or one of the plan's dials, an oracle partner agrees in
   it, it is part of the work key; the committed matrix holds the family of eight dials; results
   carry the field.
"""

import copy
import dataclasses

import numpy as np
import pytest
from gate_fake import COMMITTED, TOP
from scipy.optimize import linprog
from test_gate_day import DATE, SETTINGS, configuration, truth_day

from fleetreserve import (
    carsets,
    driver_risk,
    gate_config,
    gate_day,
    gate_summary,
    learn_eval,
    learn_run,
    products,
)
from fleetreserve.certificate import certify
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.forecast_empirical import KAPPAS

PLAN_DIALS = (0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
PLAN_LEVEL_MASSES = (0.05, 0.05, 0.1, 0.1)


class LevelStub:
    """A forecaster whose departure depends on the level: `by_level[i][kappa]` (None: no
    forecast at that level); rate, cap and need are the car's own at every level. It records the
    levels it was asked for."""

    def __init__(self, by_level, shape):
        self.by_level, self.shape, self.asked = by_level, shape, []

    def forecast_car(self, truth, i, kappa, mode, departure):
        self.asked.append((i, kappa))
        d = self.by_level[i].get(kappa)
        if d is None:
            return None
        if departure == "oracle":
            d = float(truth.day.cars[i].departure)
        return ForecastCar(d, *self.shape[i])

    def survival_bound(self, truth, i, hour, kappa):
        return None


# Two cars on a day of two service hours and three recovery hours.
CARS = (TrueCar(4, 2.0, 3.0), TrueCar(5, 3.0, 3.0))
SHAPE = ((2.0, 3.0, 3.0), (3.0, 3.0, 3.0))  # (rate, cap, need) of each car's forecast
LEVELS = [
    {0.05: 1.5, 0.1: 2.2, 0.2: 3.0, 0.3: 4.0},
    {0.05: 2.5, 0.1: 2.5, 0.2: 3.5, 0.3: 5.0},
]


def same_scenarios(found, expected):
    np.testing.assert_allclose(np.array(found), np.array(expected), atol=1e-12)


def stub(by_level=None):
    return LevelStub(LEVELS if by_level is None else by_level, SHAPE)


def day():
    return truth_day(Day("u", DATE, 2, 3, 0.0, CARS))


def prepare(forecaster, **change):
    change = {"kappa": 0.3, **change}
    return gate_day.prepare_day(day(), forecaster, configuration(**change), SETTINGS)


# --- 1. field null --------------------------------------------------------------------------


def test_a_null_field_gives_the_day_without_rows():
    forecaster = stub()
    prepared = prepare(forecaster, driver_risk=None)
    assert all(f.scenarios == () for f in prepared.day.forecasts)
    assert {kappa for _, kappa in forecaster.asked} == {0.3}
    rows = carsets.day_forecast_rows(prepared.day)
    by_hand = certify(
        rows, products.hourly(), "causal", shape_rule=gate_day.shape_rule(configuration(), SETTINGS)
    )
    assert prepared.value == by_hand.value
    assert prepared.certificate.scale == by_hand.scale
    np.testing.assert_array_equal(prepared.certificate.policy.m, by_hand.policy.m)
    np.testing.assert_array_equal(prepared.certificate.policy.w, by_hand.policy.w)
    np.testing.assert_array_equal(prepared.certificate.policy.g, by_hand.policy.g)
    np.testing.assert_array_equal(prepared.certificate.policy.G, by_hand.policy.G)
    again = prepare(stub(), driver_risk=None)
    one = gate_day.evaluate(prepared, 0.5)
    two = gate_day.evaluate(again, 0.5)
    for rung in one.rungs:
        assert one.rungs[rung].slot_loss == two.rungs[rung].slot_loss
        assert one.rungs[rung].driver_loss == two.rungs[rung].driver_loss
        np.testing.assert_array_equal(one.rungs[rung].finals, two.rungs[rung].finals)


# --- 2. the rows equal the definition ---------------------------------------------------------


def _share(departure, hour):
    return min(max(departure - hour, 0.0), 1.0)


def _brute_shortfall(car, n_service, n_recovery, u):
    """`sum_k pi_k max{need_k - E_k(u), 0}` from the plan's text, over the car's scenarios, with
    each `need_k` from a linear programme on the forecast set at the presence bound."""
    base = carsets.forecast_rows(dataclasses.replace(car, scenarios=()), n_service, n_recovery)
    total = 0.0
    for bound, mass in car.scenarios:
        weights = np.array(
            [
                1.0
                if _share(car.departure, t) <= 0
                else min(1.0, _share(bound, t) / _share(car.departure, t))
                for t in range(n_service + 1)
            ]
        )
        supply = car.rate * sum(
            _share(bound, t) for t in range(n_service + 1, n_service + 1 + n_recovery)
        )
        best = linprog(-weights, A_ub=base.coefficients, b_ub=base.bounds, bounds=(None, None))
        assert best.status == 0
        need_k = min(car.need, -best.fun + supply)
        total += mass * max(need_k - (weights @ u + supply), 0.0)
    return total


def _points(rows, n, count, seed):
    """Points of the set `rows`: some uniform, some filled hour by hour in a random order, so that
    both little and much shortfall occur."""
    rng = np.random.default_rng(seed)
    high = rows.bounds[: n + 1]  # the power limits (rows 0 to n are the "power-high" rows)
    out = []
    while len(out) < count:
        if len(out) % 2:
            u = rng.uniform(0, 3.0, n + 1)
        else:
            u = np.zeros(n + 1)
            left = rng.uniform(1.0, 3.0)
            for t in rng.permutation(n + 1):
                u[t] = min(high[t] * rng.uniform(0.5, 1.0), left)
                left -= u[t]
        if np.all(rows.coefficients @ u <= rows.bounds + 1e-12):
            out.append(u)
    return out


SCENARIO_CARS = [
    # four distinct bounds, all with a shortfall
    ForecastCar(4.0, 2.0, 3.0, 3.0, ((1.5, 0.05), (2.2, 0.05), (3.0, 0.1), (4.0, 0.1))),
    # equal bounds to merge, and the last bound at the presence bound
    ForecastCar(3.5, 2.5, 3.0, 2.5, ((2.0, 0.05), (2.0, 0.05), (3.5, 0.1), (3.5, 0.1))),
    # a level whose recovery supply alone meets its need (no shortfall): dropped
    ForecastCar(5.0, 3.0, 3.0, 3.0, ((2.5, 0.05), (4.5, 0.05), (4.5, 0.1), (5.0, 0.1))),
]


def _holds(car, epsilon, u, n_service=2, n_recovery=3):
    coefficients, bounds = driver_risk.shortfall_rows(car, n_service, n_recovery, epsilon)
    assert 1 <= bounds.size <= 15
    return bool(np.all(coefficients @ u <= bounds + 1e-9))


@pytest.mark.parametrize("car", SCENARIO_CARS)
def test_the_rows_hold_exactly_when_the_expected_shortfall_is_within_the_dial(car):
    n_service, n_recovery = 2, 3
    base = carsets.forecast_rows(dataclasses.replace(car, scenarios=()), n_service, n_recovery)
    points = _points(base, n_service, 300, seed=7)
    left = np.array([_brute_shortfall(car, n_service, n_recovery, u) for u in points])
    ratio = left / car.need
    assert np.ptp(ratio) > 0.01  # the sample spans a range of shortfalls
    # the plan's dials and dials cut from the sample, so each has points on both sides of it
    dials = [*PLAN_DIALS, *np.quantile(ratio, [0.25, 0.5, 0.75])]
    for epsilon in dials:
        holds = np.array([_holds(car, epsilon, u) for u in points])
        np.testing.assert_array_equal(holds, ratio <= epsilon + 1e-9)
    for epsilon in dials[-3:]:
        assert 20 < sum(ratio <= epsilon) < len(points) - 20
    # at the boundary of one point: just above its shortfall the rows hold there, just below not
    for k in np.flatnonzero(ratio > 1e-4)[:10]:
        assert _holds(car, ratio[k] + 1e-6, points[k])
        assert not _holds(car, ratio[k] - 1e-6, points[k])


def test_the_car_rows_are_the_forecast_rows_and_the_new_need_rows():
    car = SCENARIO_CARS[0]
    plain = carsets.forecast_rows(car, 2, 3)
    rows = driver_risk.car_rows(car, 2, 3, 0.02)
    assert rows.n_service == 2
    np.testing.assert_array_equal(rows.coefficients[: plain.bounds.size], plain.coefficients)
    np.testing.assert_array_equal(rows.bounds[: plain.bounds.size], plain.bounds)
    assert rows.kinds[: plain.bounds.size] == plain.kinds
    assert rows.kinds[plain.bounds.size :] == ("need",) * 15
    assert plain.bounds.size + 15 == rows.bounds.size


def test_equal_bounds_merge_and_a_level_without_shortfall_is_dropped():
    merged = SCENARIO_CARS[1]
    assert driver_risk.shortfall_rows(merged, 2, 3, 0.0)[1].size == 3  # two bounds: 2.0 and 3.5
    one = dataclasses.replace(merged, scenarios=((2.0, 0.1), (3.5, 0.2)))
    for found, expected in zip(
        driver_risk.shortfall_rows(merged, 2, 3, 0.02),
        driver_risk.shortfall_rows(one, 2, 3, 0.02),
        strict=True,
    ):
        np.testing.assert_allclose(found, expected)
    # at 4.5 and at 5.0 the recovery hours alone supply the whole need: both levels are dropped
    dropped = SCENARIO_CARS[2]
    only_first = dataclasses.replace(dropped, scenarios=dropped.scenarios[:1])
    for found, expected in zip(
        driver_risk.shortfall_rows(dropped, 2, 3, 0.02),
        driver_risk.shortfall_rows(only_first, 2, 3, 0.02),
        strict=True,
    ):
        np.testing.assert_allclose(found, expected)
    assert driver_risk.shortfall_rows(dropped, 2, 3, 0.02)[1].size == 1


def test_a_car_without_scenarios_is_refused_by_the_rows_and_has_none_by_default():
    with pytest.raises(ValueError, match="no departure scenarios"):
        driver_risk.car_rows(ForecastCar(4.0, 2.0, 3.0, 3.0), 2, 3, 0.0)
    assert ForecastCar(4.0, 2.0, 3.0, 3.0).scenarios == ()
    with pytest.raises(ValueError, match="epsilon"):
        driver_risk.shortfall_rows(SCENARIO_CARS[0], 2, 3, -0.1)


def test_scenarios_on_a_forecast_car_are_checked():
    with pytest.raises(ValueError, match="at most the forecast departure"):
        ForecastCar(4.0, 2.0, 3.0, 3.0, ((2.0, 0.1), (4.5, 0.1)))
    with pytest.raises(ValueError, match="not decrease"):
        ForecastCar(4.0, 2.0, 3.0, 3.0, ((3.0, 0.1), (2.0, 0.1)))
    with pytest.raises(ValueError, match="positive"):
        ForecastCar(4.0, 2.0, 3.0, 3.0, ((3.0, 0.0),))


# --- 3. the dial -----------------------------------------------------------------------------


def test_the_certified_value_grows_with_the_dial_and_reaches_the_value_without_rows():
    without = prepare(stub(), driver_risk=None).value
    values = [prepare(stub(), driver_risk=eps).value for eps in (*PLAN_DIALS, 0.3, 1.0)]
    assert all(b >= a - 1e-7 for a, b in zip(values, values[1:], strict=False))  # never falls
    assert values[0] < values[len(PLAN_DIALS) - 1]  # and does grow over the plan's dials
    assert values[0] < without - 1e-3  # the rows at the dial 0 cost something on this day
    assert values[-1] == pytest.approx(without, abs=1e-7)
    assert values[-2] == pytest.approx(without, abs=1e-7)


# --- 4. scenario construction ----------------------------------------------------------------


def test_running_maximum_and_cap_at_the_presence_bound():
    found = driver_risk.departure_scenarios(4.0, [3.0, 2.0, 5.0, 4.0], PLAN_LEVEL_MASSES)
    assert found == ((3.0, 0.05), (3.0, 0.05), (4.0, 0.1), (4.0, 0.1))


def test_a_missing_level_is_read_as_the_presence_bound():
    found = driver_risk.departure_scenarios(4.0, [2.0, None, 3.0, 4.0], PLAN_LEVEL_MASSES)
    assert found == ((2.0, 0.05), (4.0, 0.05), (4.0, 0.1), (4.0, 0.1))


def test_levels_and_masses_follow_the_configurations_kappa():
    assert driver_risk.LEVELS == KAPPAS == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    assert driver_risk.levels_up_to(0.3) == (0.05, 0.1, 0.2, 0.3)
    assert driver_risk.levels_up_to(0.1) == (0.05, 0.1)
    assert driver_risk.level_masses((0.05, 0.1, 0.2, 0.3)) == pytest.approx(PLAN_LEVEL_MASSES)
    assert driver_risk.level_masses((0.05, 0.1)) == pytest.approx((0.05, 0.05))
    with pytest.raises(ValueError, match="kappa"):
        driver_risk.levels_up_to(0.15)


def test_the_levels_0_4_and_0_5_belong_to_the_grid_and_the_what_if_levels_are_above_them():
    assert driver_risk.levels_up_to(0.4) == (0.05, 0.1, 0.2, 0.3, 0.4)
    assert driver_risk.levels_up_to(0.5) == (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
    assert driver_risk.WHAT_IF_LEVELS == (0.6, 0.7)
    assert driver_risk.levels_up_to(0.6) == (*driver_risk.LEVELS, 0.6)
    assert driver_risk.levels_up_to(0.7) == (*driver_risk.LEVELS, 0.6, 0.7)
    masses = driver_risk.level_masses(driver_risk.levels_up_to(0.5))
    assert masses == pytest.approx((0.05, 0.05, 0.1, 0.1, 0.1, 0.1))


def test_the_rows_at_kappa_0_5_use_the_levels_up_to_0_5():
    by_level = [
        {**LEVELS[0], 0.4: 4.0, 0.5: 4.0},
        {**LEVELS[1], 0.4: 5.0, 0.5: 5.0},
    ]
    forecaster = stub(by_level)
    prepared = prepare(forecaster, driver_risk=0.02, kappa=0.5)
    assert sorted({kappa for _, kappa in forecaster.asked}) == [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    same_scenarios(
        prepared.day.forecasts[0].scenarios,
        ((1.5, 0.05), (2.2, 0.05), (3.0, 0.1), (4.0, 0.1), (4.0, 0.1), (4.0, 0.1)),
    )
    same_scenarios(
        prepared.day.forecasts[1].scenarios,
        ((2.5, 0.05), (2.5, 0.05), (3.5, 0.1), (5.0, 0.1), (5.0, 0.1), (5.0, 0.1)),
    )
    assert prepared.day.forecasts[0].departure == 4.0
    at_0_4 = stub(by_level)
    prepare(at_0_4, driver_risk=0.02, kappa=0.4)
    assert sorted({kappa for _, kappa in at_0_4.asked}) == [0.05, 0.1, 0.2, 0.3, 0.4]


def test_the_forecasts_carry_scenarios_built_from_the_forecasters_levels():
    forecaster = stub()
    prepared = prepare(forecaster, driver_risk=0.02)
    same_scenarios(
        prepared.day.forecasts[0].scenarios, ((1.5, 0.05), (2.2, 0.05), (3.0, 0.1), (4.0, 0.1))
    )
    same_scenarios(
        prepared.day.forecasts[1].scenarios, ((2.5, 0.05), (2.5, 0.05), (3.5, 0.1), (5.0, 0.1))
    )
    assert prepared.day.forecasts[0].departure == 4.0
    assert sorted({kappa for _, kappa in forecaster.asked}) == [0.05, 0.1, 0.2, 0.3]
    smaller = prepare(stub(), driver_risk=0.02, kappa=0.1)
    assert smaller.day.forecasts[0].departure == 2.2
    same_scenarios(smaller.day.forecasts[0].scenarios, ((1.5, 0.05), (2.2, 0.05)))


def test_a_level_the_forecaster_does_not_give_is_the_presence_bound():
    by_level = [dict(LEVELS[0]), dict(LEVELS[1])]
    by_level[0][0.1] = None
    prepared = prepare(stub(by_level), driver_risk=0.02)
    same_scenarios(
        prepared.day.forecasts[0].scenarios, ((1.5, 0.05), (4.0, 0.05), (4.0, 0.1), (4.0, 0.1))
    )


def test_the_floored_reading_floors_the_scenarios_too():
    by_level = [
        {0.05: 1.7, 0.1: 2.4, 0.2: 3.6, 0.3: 4.3},
        {0.05: 2.5, 0.1: 2.5, 0.2: 3.5, 0.3: 5.0},
    ]
    planned = prepare(stub(by_level), driver_risk=0.02)
    same_scenarios(
        planned.day.forecasts[0].scenarios, ((1.7, 0.05), (2.4, 0.05), (3.6, 0.1), (4.3, 0.1))
    )
    floored = prepare(stub(by_level), driver_risk=0.02, departure_floor="floored")
    assert floored.day.forecasts[0].departure == 4.0
    same_scenarios(
        floored.day.forecasts[0].scenarios, ((1.0, 0.05), (2.0, 0.05), (3.0, 0.1), (4.0, 0.1))
    )
    same_scenarios(
        floored.day.forecasts[1].scenarios, ((2.0, 0.05), (2.0, 0.05), (3.0, 0.1), (5.0, 0.1))
    )


# --- 5. the oracle departure -----------------------------------------------------------------


def test_with_the_oracle_departure_the_rows_hold_everywhere_and_the_value_is_unchanged():
    forecaster = stub()
    oracle = {"departure": "oracle", "oracle_partner": None, "kappa": 0.3}
    with_rows = prepare(forecaster, driver_risk=0.0, **oracle)
    without = prepare(stub(), driver_risk=None, **oracle)
    for i, car in enumerate(with_rows.day.forecasts):
        true = CARS[i].departure
        assert car.departure == float(true)
        same_scenarios(car.scenarios, tuple((float(true), mass) for mass in PLAN_LEVEL_MASSES))
        base = carsets.forecast_rows(
            dataclasses.replace(car, scenarios=()), 2, with_rows.day.n_recovery
        )
        coefficients, bounds = driver_risk.shortfall_rows(car, 2, with_rows.day.n_recovery, 0.0)
        for u in _points(base, 2, 300, seed=3):
            assert np.all(coefficients @ u <= bounds + 1e-9)
    assert with_rows.value == pytest.approx(without.value, abs=1e-7)


# --- 6. the matrix ---------------------------------------------------------------------------


def _document():
    return copy.deepcopy(COMMITTED)


def _refused(document, text):
    with pytest.raises(gate_config.MatrixError, match=text):
        gate_config.parse(document)


def _pair(document):
    """The index of a configuration with driver_risk null and its partner."""
    names = [c["name"] for c in document["configurations"]]
    return names.index("main-rounded-hour-k0.3-empirical"), names.index(
        "main-rounded-hour-k0.3-oracle"
    )


def test_the_field_is_required():
    document = _document()
    del document["configurations"][0]["driver_risk"]
    _refused(document, r"missing fields \['driver_risk'\]")


@pytest.mark.parametrize("value", [0.04, -0.01, 0.2, "0.1", True, False, [0.1], 1])
def test_a_value_outside_the_plans_dials_is_refused(value):
    document = _document()
    k, partner = _pair(document)
    for index in (k, partner):
        document["configurations"][index]["driver_risk"] = value
    _refused(document, "driver_risk must be null or one of")


@pytest.mark.parametrize("value", [None, *PLAN_DIALS])
def test_null_and_every_dial_of_the_plan_is_read(value):
    document = _document()
    k, partner = _pair(document)
    for index in (k, partner):
        document["configurations"][index]["driver_risk"] = value
    matrix = gate_config.parse(document)
    found = matrix.configuration("main-rounded-hour-k0.3-empirical").driver_risk
    assert found == value and (value is None or isinstance(found, float))
    assert gate_config.DIALS == driver_risk.DIALS == tuple(map(float, PLAN_DIALS))


def test_an_oracle_partner_must_agree_in_the_field():
    document = _document()
    k, partner = _pair(document)
    document["configurations"][k]["driver_risk"] = 0.02
    _refused(document, r"differs in \['driver_risk'\]")
    document["configurations"][partner]["driver_risk"] = 0.05
    _refused(document, r"differs in \['driver_risk'\]")
    document["configurations"][partner]["driver_risk"] = 0.02
    gate_config.parse(document)


def test_the_field_is_part_of_the_work_key():
    fields = {**dict(configuration().__dict__)}
    a = gate_config.Configuration(**{**fields, "driver_risk": None})
    b = gate_config.Configuration(**{**fields, "driver_risk": 0.0})
    c = gate_config.Configuration(**{**fields, "driver_risk": 0.02})
    assert len({a.work_key(), b.work_key(), c.work_key()}) == 3


def test_the_committed_matrix_holds_the_family_of_dials():
    matrix = gate_config.load(gate_config_path())
    base = matrix.configuration("main-rounded-hour-k0.3-empirical")
    base_oracle = matrix.configuration("main-rounded-hour-k0.3-oracle")
    assert base.driver_risk is None and base_oracle.driver_risk is None
    assert [c.name for c in matrix.configurations if c.driver_risk is None] == [
        c["name"] for c in COMMITTED["configurations"] if c["driver_risk"] is None
    ]
    family = [c for c in matrix.configurations if c.driver_risk is not None]
    assert sorted(c.driver_risk for c in family if c.departure == "empirical") == sorted(
        map(float, PLAN_DIALS)
    )
    for c in family:
        assert c.family == "alternate"
        eps = f"{c.driver_risk:g}"
        reference = base if c.departure == "empirical" else base_oracle
        assert c.name == f"main-rounded-hour-k0.3-eps{eps}-{c.departure}"
        same = [
            f.name
            for f in dataclasses.fields(c)
            if f.name not in ("name", "family", "oracle_partner", "driver_risk")
        ]
        assert all(getattr(c, f) == getattr(reference, f) for f in same)
        if c.departure == "empirical":
            partner = matrix.configuration(c.oracle_partner)
            assert partner.driver_risk == c.driver_risk and partner.departure == "oracle"
    assert len(family) == 2 * len(PLAN_DIALS)


def gate_config_path():
    return TOP / "gate" / "matrix.json"


def test_the_summary_and_the_pairs_name_the_dial():
    matrix = gate_config.load(gate_config_path())
    tables = gate_summary.gate_tables(matrix, {})
    pairs = tables["by_pair"]
    assert pairs["main-rounded-hour-k0.3-empirical"]["driver_risk"] is None
    assert pairs["main-rounded-hour-k0.3-eps0.02-empirical"]["driver_risk"] == 0.02
    document = {
        "trial": False,
        "partial": False,
        "t5": {"breaks_a": 0, "breaks_b": 0, "day_paths_checked": 7},
        "timeouts": {},
        "failed_configurations": [],
        "calibration_rule": {"n_calibration": 100, "calibration_half_below": 200},
        "units": {},
        "configurations": {},
        "gate": tables,
    }
    lines = gate_summary.summary_lines(document, matrix)
    start = next(i for i, x in enumerate(lines) if x.startswith("Alternates"))
    risk = [x for x in lines[start:] if "main-rounded-hour-k0.3-eps0.02-empirical (" in x]
    plain = [x for x in lines[start:] if "alt-floored-k0.1-empirical (" in x]
    assert risk and all("driver-risk rows at epsilon 0.02" in x for x in risk)
    assert plain and all("driver-risk" not in x for x in plain)


def _entry(offer):
    test = {"mean_offer_kwh_per_day": offer, "offering_days": 4, "days": 5, "failing_days": 0}
    rungs = {rung: {"beta_hat": 0.5, "reason": "", "test": test} for rung in learn_eval.OFFER_RUNGS}
    return {"units": {"u": {"rungs": rungs}}}


def test_the_learning_run_reports_the_dial_of_each_configuration():
    matrix = gate_config.load(gate_config_path())
    names = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-eps0.05-empirical")
    document = {"configurations": {n: _entry(1.0) for n in names}}
    document["configurations"]["main-rounded-hour-k0.3-oracle"] = _entry(2.0)
    document["configurations"]["main-rounded-hour-k0.3-eps0.05-oracle"] = _entry(2.0)
    found = learn_eval.offers(document, matrix, "u")
    assert set(found) == set(names)
    assert found[names[0]]["driver_risk"] is None and found[names[1]]["driver_risk"] == 0.05
    unit = {
        "status": "calibratable",
        "counting_days": 9,
        "calibration_days_used": 4,
        "test_days": 5,
        "kappas": {},
        "configurations": found,
    }
    rule = {"n_calibration": 100, "calibration_half_below": 200}
    results = {
        "model": "m",
        "units": {"u": unit},
        "calibration_rule": rule,
        "failed_configurations": [],
    }
    text = "\n".join(learn_run.summary_lines(results))
    assert f"  {names[0]}:" in text
    assert f"  {names[1]} [driver-risk rows at epsilon 0.05]:" in text

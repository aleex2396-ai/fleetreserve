"""The closed-form known-car ceiling against the programme it replaces
(`gate_day.day_ceilings(day).best_causal`), on seeded made-up fleets and on two hand cases."""

import datetime

import numpy as np
import pytest

from fleetreserve import known_car_ceiling
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.gate_day import day_ceilings

SEEDS = range(600)
AGREE_KWH = 1e-6
MAX_CARS = 30
ROOM_LABELS = ("zero", "r", "r_s", "r_c", "r_s_plus_c", "r_d", "uniform_to_r_d", "uniform_to_3r")


def make_day(cars: list[TrueCar], n_service: int) -> Day:
    return Day(
        unit="made-up",
        date=datetime.date(2020, 1, 1),
        n_service=n_service,
        n_recovery=covering_recovery_hours(cars, n_service),
        rho=0.0,
        cars=tuple(cars),
    )


def _room(rng, label: str, rate: float, departure: int, n_service: int) -> float:
    """A room of the named kind, never above `rate * departure` (the day type's limit)."""
    s = min(departure - 1, n_service)
    c = max(departure - 1 - n_service, 0)
    room = {
        "zero": 0.0,
        "r": rate,
        "r_s": rate * s,
        "r_c": rate * c,
        "r_s_plus_c": rate * (s + c),
        "r_d": rate * departure,
        "uniform_to_r_d": rate * departure * float(rng.uniform(0.0, 1.0)),
        "uniform_to_3r": rate * float(rng.uniform(0.0, 3.0)),
    }[label]
    return min(room, rate * departure)


def _car(rng, n_service: int, kinds: dict) -> TrueCar:
    rate = float(rng.uniform(1.0, 22.0))
    departure = int(rng.integers(1, n_service + 9))
    label = ROOM_LABELS[int(rng.integers(0, len(ROOM_LABELS)))]
    kinds[label] = kinds.get(label, 0) + 1
    kinds["leaves_hour_1"] = kinds.get("leaves_hour_1", 0) + (departure == 1)
    kinds["in_window"] = kinds.get("in_window", 0) + (1 < departure <= n_service)
    kinds["past_s"] = kinds.get("past_s", 0) + (departure > n_service)
    return TrueCar(departure, rate, _room(rng, label, rate, departure, n_service))


def random_fleet(seed: int) -> tuple[Day, dict]:
    """A fleet on 1 to 8 service hours, with rates of 1 to 22 kW and rooms up to rate times
    departure hour, many of them exactly on the break points 0, r, r s, r c, r (s + c), r d.
    Seeds divisible by 12 give a single car, those with remainder 1 a fleet of identical cars,
    the others up to 30 different cars. `kinds` counts the cars of each kind."""
    rng = np.random.default_rng(seed)
    n_service = int(rng.integers(1, 9))
    kinds = {"single": int(seed % 12 == 0), "identical": int(seed % 12 == 1)}
    if seed % 12 == 0:
        cars = [_car(rng, n_service, kinds)]
    elif seed % 12 == 1:
        cars = [_car(rng, n_service, kinds)] * int(rng.integers(2, MAX_CARS + 1))
    else:
        cars = [_car(rng, n_service, kinds) for _ in range(int(rng.integers(1, MAX_CARS + 1)))]
    return make_day(cars, n_service), kinds


def test_closed_form_equals_the_programme_on_made_up_fleets():
    differing, largest, solved = [], 0.0, 0
    for seed in SEEDS:
        day, _ = random_fleet(seed)
        reference = day_ceilings(day)
        assert reference.best_causal_outcome == "optimal", (seed, reference)
        solved += 1
        gap = abs(known_car_ceiling.known_car_ceiling(day) - reference.best_causal)
        largest = max(largest, gap)
        if gap > AGREE_KWH:
            differing.append((seed, gap))
    assert solved == len(SEEDS) >= 300
    assert not differing, f"fleets that disagree (seed, difference in kWh): {differing}"
    assert largest <= AGREE_KWH


def test_the_fleets_cover_the_cases_asked_for():
    total = {}
    sizes, services = set(), set()
    for seed in SEEDS:
        day, kinds = random_fleet(seed)
        sizes.add(len(day.cars))
        services.add(day.n_service)
        for name, count in kinds.items():
            total[name] = total.get(name, 0) + count
    assert min(sizes) == 1 and max(sizes) >= 25 and services == set(range(1, 9))
    kinds = [*ROOM_LABELS, "leaves_hour_1", "in_window", "past_s"]
    assert all(total[name] >= 100 for name in kinds), total
    assert total["single"] >= 40 and total["identical"] >= 40, total


def test_hand_case_a_car_that_stays_past_the_service_hours():
    # One service hour (n = 1), a car at 6 kW with 10 kWh of room that leaves in hour 4 (present
    # in hours 0 to 3): it is present for s = 1 service hour and c = 2 recovery hours. It can draw
    # at most 6 kWh in the service hour, and the 12 kWh of the recovery hours cover the whole
    # room, so it may draw nothing in hour 0 and nothing at all in the service hour: its
    # flexibility over the service hour is the 6 kWh the hour can carry (largest energy in S, 6,
    # minus smallest, 0). With x = 0 the bracket is min(6, 10) - max(10 - 12, 0) = 6.
    car = TrueCar(departure=4, rate=6.0, room=10.0)
    assert known_car_ceiling.car_value(6.0, 10.0, 4, 1) == pytest.approx(6.0, abs=1e-12)
    day = make_day([car], n_service=1)
    assert known_car_ceiling.known_car_ceiling(day) == pytest.approx(6.0, abs=1e-12)
    assert day_ceilings(day).best_causal == pytest.approx(6.0, abs=1e-9)


def test_hand_case_a_car_that_leaves_inside_the_service_hours():
    # Three service hours, a car at 6 kW with 6 kWh of room that leaves in hour 2 (present in
    # hours 0 and 1): s = 1, c = 0. All 6 kWh must be drawn in hours 0 and 1 and there is no
    # recovery hour, so the energy drawn in the service hour is 6 - x for the x drawn in hour 0.
    # Whatever x is, the energy the car must draw is fixed once x is: the bracket is
    # min(6, 6 - x) - max(6 - x, 0) = 0, so the car gives no flexibility.
    car = TrueCar(departure=2, rate=6.0, room=6.0)
    assert known_car_ceiling.car_value(6.0, 6.0, 2, 3) == pytest.approx(0.0, abs=1e-12)
    day = make_day([car], n_service=3)
    assert known_car_ceiling.known_car_ceiling(day) == pytest.approx(0.0, abs=1e-12)
    assert day_ceilings(day).best_causal == pytest.approx(0.0, abs=1e-9)


def test_arrays_and_day_agree_and_an_empty_day_is_zero():
    day, _ = random_fleet(7)
    from_arrays = known_car_ceiling.known_car_ceiling_arrays(
        [c.rate for c in day.cars],
        [c.room for c in day.cars],
        [c.departure for c in day.cars],
        day.n_service,
    )
    assert from_arrays == known_car_ceiling.known_car_ceiling(day)
    assert known_car_ceiling.known_car_ceiling(make_day([], n_service=3)) == 0.0


def test_a_day_with_rho_above_zero_is_refused():
    day = Day(
        unit="made-up",
        date=datetime.date(2020, 1, 1),
        n_service=2,
        n_recovery=0,
        rho=0.1,
        cars=(TrueCar(2, 6.0, 3.0),),
    )
    with pytest.raises(ValueError, match="rho"):
        known_car_ceiling.known_car_ceiling(day)

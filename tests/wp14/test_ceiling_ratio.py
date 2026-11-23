"""Outcome O8's ceiling ratio V (`ceiling_ratio`) on hand-worked made-up days, and against a
programme over every corner path written in this file.

The hand day (two service hours, 1 and 2; two recovery hours, 3 and 4; rho 0, so every car needs
its whole room), two cars:

- G: departure hour 1 (it draws in hour 0 only), rate 1 kW, room 0.5 kWh. It is in no service
  hour, one-way or two-way, so it adds nothing to either ceiling.
- A: departure hour 5 (hours 0 to 4), rate 4 kW, room 4 kWh.

One-way, A can take its 4 kWh in hours 1 and 2 (b(S) = min(2 x 4, 4) = 4) or in hour 0
(p(S) = max(4 - 3 x 4, 0) = 0): the one-way ceiling is 4. Two-way at budget B, A may give back
min(B, 4) in hour 0 and then take up to 4 + min(B, 4) in hours 1 and 2 (at most 8 by its rate),
so b(S) = 4 + min(B, 4); or take 4 in hour 0, give back up to 4 + B (at most 8) in hours 1 and 2
and take it back in hours 3 and 4, so p(S) = -min(4 + B, 8). At B = 5: b(S) = 8, p(S) = -8, the
two-way ceiling 16 and V = 16 / 4 = 4. At B = 2: b(S) = 6, p(S) = -6, the two-way ceiling 12 and
V = 3. Read with the one-way sets in place of the two-way ones the two-way ceiling would be 4.

The day of an empty car (one car, departure hour 5, rate 4 kW, room 0 kWh, the same hours): one-way
it can draw nothing, so its one-way ceiling is 0. Two-way at budget B it may give back min(B, 4) in
hour 0 and take it back in hours 1 and 2 (b(S) = min(B, 4)), or give back B (at most 8) in hours 1
and 2 and take it back in hours 3 and 4 (p(S) = -min(B, 8)): 4 + 5 = 9 at B = 5, 2 + 2 = 4 at
B = 2. V is not defined on such a day alone (its one-way ceiling is 0).

The ratio over days (reading C1, a ratio of sums): days with (one-way, two-way) ceilings (4, 16),
(1, 1) and (0, 9) give the sums 5 and 26 and V = 26 / 5 = 5.2; the third day is counted and named
as a day whose one-way ceiling is 0, with its two-way 9 kWh (reading C3). The mean of the daily
ratios is not defined there (9 / 0), and over the other two days it would be (4 + 1) / 2 = 2.5.

The programme over every corner path (`corner_path_ceiling`, independent of `exact`): the
largest sum over the service hours of hi - lo such that, for every one of the 2^|S| paths at lo
or hi in each service hour, some plan of each car that knows the whole path meets it, with the
true sets written out row by row: power in [0, r] one-way or [-r, r] two-way before the departure
hour and 0 after; the energy stored since the decision time at most the room after every hour,
and at least -B (two-way; 0 one-way, from the power); at least the protected energy at the end.
It is checked on the two days above and on 12 seeded made-up days (`made_up_days`) at 5 kWh.
"""

import datetime
import itertools
import random
import re

import numpy as np
import pytest
from made_up_days import made_up_sample

from fleetreserve import ceiling_ratio
from fleetreserve.day import Day, TrueCar
from fleetreserve.lp import BOX_SECONDS, solve_lp

DATE = datetime.date(2020, 1, 1)
HAND_DAY = Day("hand", DATE, 2, 2, 0.0, (TrueCar(1, 1.0, 0.5), TrueCar(5, 4.0, 4.0)))
EMPTY_CAR_DAY = Day("hand", DATE, 2, 2, 0.0, (TrueCar(5, 4.0, 0.0),))
# (day, budget) -> (one-way ceiling, two-way ceiling), by hand (module text).
BY_HAND = {
    ("hand", 5.0): (4.0, 16.0),
    ("hand", 2.0): (4.0, 12.0),
    ("empty car", 5.0): (0.0, 9.0),
    ("empty car", 2.0): (0.0, 4.0),
}
DAYS = {"hand": HAND_DAY, "empty car": EMPTY_CAR_DAY}
SEED = 23
N_SEEDED = 12
TOLERANCE = 1e-6  # relative to max(1, value)


def _close(a, b) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(b))


def corner_path_ceiling(day: Day, budget: float | None) -> float:
    """The hourly ceiling by the programme over every corner path (module text); one-way when
    `budget` is None."""
    n, hours, cars = day.n_service, day.n_hours, day.cars
    paths = list(itertools.product((-1, 1), repeat=n))
    size = 2 * n + len(paths) * len(cars) * hours

    def u(p: int, i: int, h: int) -> int:
        return 2 * n + (p * len(cars) + i) * hours + h

    bounds = [(None, None)] * (2 * n)
    for _ in paths:
        for car in cars:
            low = 0.0 if budget is None else -car.rate
            bounds += [(low, car.rate) if h < car.departure else (0.0, 0.0) for h in range(hours)]
    upper_rows, upper = [], []
    equal_rows, equal = [], []

    def row(entries):
        line = np.zeros(size)
        for k, value in entries:
            line[k] += value
        return line

    for t in range(n):  # lo_t <= hi_t
        upper_rows.append(row([(t, 1.0), (n + t, -1.0)]))
        upper.append(0.0)
    for p, path in enumerate(paths):
        for t, z in enumerate(path, start=1):
            box = t - 1 if z < 0 else n + t - 1
            equal_rows.append(row([(u(p, i, t), 1.0) for i in range(len(cars))] + [(box, -1.0)]))
            equal.append(0.0)
        for i, car in enumerate(cars):
            for h in range(hours):
                stored = [(u(p, i, s), 1.0) for s in range(h + 1)]
                upper_rows.append(row(stored))
                upper.append(car.room)
                if budget is not None:
                    upper_rows.append(row([(k, -v) for k, v in stored]))
                    upper.append(budget)
            upper_rows.append(row([(u(p, i, s), -1.0) for s in range(hours)]))
            upper.append(-car.need(day.rho))
    cost = row([(n + t, -1.0) for t in range(n)] + [(t, 1.0) for t in range(n)])
    result = solve_lp(
        cost,
        A_ub=np.array(upper_rows),
        b_ub=np.array(upper),
        A_eq=np.array(equal_rows),
        b_eq=np.array(equal),
        bounds=bounds,
        time_limit=BOX_SECONDS,
    )
    assert result.outcome == "optimal", result.message
    return -float(result.objective)


@pytest.mark.parametrize("case", sorted(BY_HAND))
def test_the_ceilings_are_the_hand_values(case):
    name, budget = case
    found = ceiling_ratio.day_ceilings(DAYS[name], budget)
    assert found == pytest.approx(BY_HAND[case], abs=1e-9)


def test_the_two_way_ceiling_is_read_at_the_budget_given():
    at_five = ceiling_ratio.day_ceilings(HAND_DAY, 5.0)[1]
    at_two = ceiling_ratio.day_ceilings(HAND_DAY, 2.0)[1]
    assert (at_five, at_two) == pytest.approx((16.0, 12.0), abs=1e-9)
    assert at_five / ceiling_ratio.day_ceilings(HAND_DAY, 5.0)[0] == pytest.approx(4.0)


@pytest.mark.parametrize("case", sorted(BY_HAND))
def test_the_hand_days_agree_with_the_corner_path_programme(case):
    name, budget = case
    one, two = BY_HAND[case]
    assert corner_path_ceiling(DAYS[name], None) == pytest.approx(one, abs=1e-6)
    assert corner_path_ceiling(DAYS[name], budget) == pytest.approx(two, abs=1e-6)


def test_seeded_made_up_days_agree_with_the_corner_path_programme_at_5_kwh():
    days, _ = made_up_sample(random.Random(SEED), N_SEEDED)
    assert len(days) == N_SEEDED
    wrong, above = [], 0
    for day in days:
        one, two = ceiling_ratio.day_ceilings(day, 5.0)
        programme = (corner_path_ceiling(day, None), corner_path_ceiling(day, 5.0))
        if not (_close(one, programme[0]) and _close(two, programme[1])):
            wrong.append((day, (one, two), programme))
        above += two > one + TOLERANCE
    assert not wrong, wrong
    assert above >= N_SEEDED // 2  # the two-way sets give more on most days: not a copy


def test_a_day_without_cars_has_both_ceilings_0():
    assert ceiling_ratio.day_ceilings(Day("hand", DATE, 2, 0, 0.0, ()), 5.0) == (0.0, 0.0)


def test_v_is_a_ratio_of_sums_and_days_with_a_one_way_ceiling_of_0_are_counted():
    days = [("A day 1", 4.0, 16.0), ("A day 2", 1.0, 1.0), ("A day 3", 0.0, 9.0)]
    found = ceiling_ratio.ratio(days)
    assert found.test_days == 3
    assert (found.one_way_sum_kwh, found.two_way_sum_kwh) == (5.0, 26.0)
    assert found.v == pytest.approx(5.2)
    assert found.one_way_zero_days == ("A day 3",)
    assert found.two_way_sum_on_one_way_zero_days_kwh == 9.0


def test_v_is_not_defined_when_the_one_way_sum_is_0():
    found = ceiling_ratio.ratio([("A day 1", 0.0, 9.0), ("A day 2", 0.0, 0.0)])
    assert found.v is None and found.one_way_zero_days == ("A day 1", "A day 2")
    assert found.two_way_sum_kwh == found.two_way_sum_on_one_way_zero_days_kwh == 9.0


def test_the_offer_share_is_a_ratio_of_sums():
    assert ceiling_ratio.offer_share([2.0, 1.0], [16.0, 4.0]) == pytest.approx(3.0 / 20.0)
    assert ceiling_ratio.offer_share([0.0], [0.0]) is None
    with pytest.raises(ValueError):
        ceiling_ratio.offer_share([1.0], [1.0, 2.0])


def test_the_readings_listed_are_those_of_the_module_text():
    written = re.findall(r"^  (C\d+)\. ", ceiling_ratio.__doc__, flags=re.MULTILINE)
    assert written == list(ceiling_ratio.READINGS) == [f"C{k}" for k in range(1, 10)]


def test_the_label_says_b5_only_at_5_kwh():
    assert ceiling_ratio.budget_label(5.0) == "B=5" and ceiling_ratio.budget_note(5.0) == ""
    assert ceiling_ratio.budget_label(2.0) == "B=2"
    assert ceiling_ratio.budget_label(0.5) == "B=0.5"
    assert ceiling_ratio.budget_label(5.0000001) == "B=5.0000001"
    for budget in (0.0, 0.5, 2.0, 4.999, 5.0000001, 10.0):
        assert ceiling_ratio.budget_note(budget) == ", not outcome O8's B=5"

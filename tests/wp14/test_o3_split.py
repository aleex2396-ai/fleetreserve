"""The general split of outcome O3 (`o3_split.general_split`) and the split without decision-hour
gain beside it (`certificate.certify(..., "full")`).

1. The hand day of `test_o3_hand` (rho 0, two service hours; car A: departure 2, rate 1 kW, room
   1 kWh; car B: departure 6, rate 1 kW, room 3 kWh, its set the cube [0, 1]^3 on hours 0 to 2).
   With z in [-1, 1]^2 the request's position in the box, the rule worked by hand there gives A
   (0.5 - 0.5 z1, 0.5 + 0.5 z1, 0) and B (0.5 + 0.5 z1, 0.5 + 0.5 z1, 0.5 + 0.5 z2) on hours 0 to
   2: the hour-0 gains -0.5 and +0.5 sum to zero, the fleet gives 1 + z1 in hour 1 and
   0.5 + 0.5 z2 in hour 2, so the hourly box has widths 1 and 0.5 and value 3, the ceiling; the
   test checks that rule against both cars' rows at the four corners of z. For the block product
   B gives 0.5 in hour 1 instead: widths 0.5 and 0.5, value 2, the block ceiling. Without
   decision-hour gain A's hour-1 power is one number and B adds at most 0.5 per hour: 2 and 2.
2. A day of the review of ac84a00 (its follow-up 1): rho 0.1, three service hours, cars as
   (departure, rate kW, room kWh) (3, 1, 2.206), (4, 1, 1.126), (3, 3.3, 7.199), (1, 3.3, 2.534).
   The review's own programme gave, for the block product, 1.0531 kWh with full gains (the split
   without decision-hour gain) and 0.3378 with causal ones; `o3_day` must report the former, so
   that swapping full gains for causal ones turns this test red. The general split with
   decision-hour gains gives 1.3065 on both products (this code, not by hand), at least the
   split without them.
"""

import datetime
import itertools

import numpy as np
import pytest

from fleetreserve import products
from fleetreserve.carsets import true_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.o3_day import DayTask, TimeLimits, compute_day
from fleetreserve.o3_split import general_split

TOLERANCE = 1e-6
REVIEW_TOLERANCE = 1e-4  # the review's figures have four decimals


def _day(n, rho, cars) -> Day:
    cars = tuple(TrueCar(*car) for car in cars)
    return Day("made-up", datetime.date(2020, 1, 1), n, covering_recovery_hours(cars, n), rho, cars)


HAND = _day(2, 0.0, [(2, 1.0, 1.0), (6, 1.0, 3.0)])
REVIEW = _day(3, 0.1, [(3, 1.0, 2.206), (4, 1.0, 1.126), (3, 3.3, 7.199), (1, 3.3, 2.534)])


def _hand_rule_worst(rows) -> float:
    """The largest row excess of the hand rule over both cars' rows at the four corners of z."""
    worst = -np.inf
    for z1, z2 in itertools.product((-1.0, 1.0), repeat=2):
        powers = {
            0: np.array([0.5 - 0.5 * z1, 0.5 + 0.5 * z1, 0.0]),
            1: np.array([0.5 + 0.5 * z1, 0.5 + 0.5 * z1, 0.5 + 0.5 * z2]),
        }
        assert powers[0][0] + powers[1][0] == pytest.approx(1.0)  # the fleet's hour 0
        assert powers[0][1] + powers[1][1] == pytest.approx(1.0 + z1)
        assert powers[0][2] + powers[1][2] == pytest.approx(0.5 + 0.5 * z2)
        for car, u in powers.items():
            mine = rows.car == car
            worst = max(worst, float(np.max(rows.coefficients[mine] @ u - rows.bounds[mine])))
    return worst


@pytest.mark.parametrize(("product", "split", "fixed"), [("hourly", 3.0, 2.0), ("block", 2.0, 2.0)])
def test_the_hand_day(product, split, fixed):
    rows, kind = true_rows(HAND), products.Product(product)
    assert _hand_rule_worst(rows) <= 1e-12  # the hand rule is feasible, so 3 is reached
    found = general_split(rows, kind)
    assert found.outcome == "optimal"
    assert found.value == pytest.approx(split, abs=TOLERANCE)
    without = certify(rows, kind, "full", shape_rule=ANY_OPTIMAL)
    assert without.value == pytest.approx(fixed, abs=TOLERANCE)


def test_the_review_day_reports_full_gains_not_causal_ones():
    result = compute_day(DayTask("made-up", "A", "1", REVIEW, "day-average", TimeLimits()))
    block, hourly = result.values["block"], result.values["hourly"]
    assert block["general_split_no_hour0_gain"] == pytest.approx(1.0531, abs=REVIEW_TOLERANCE)
    causal = certify(true_rows(REVIEW), products.Product("block"), "causal", shape_rule=ANY_OPTIMAL)
    assert causal.value == pytest.approx(0.3378, abs=REVIEW_TOLERANCE)
    assert block["general_split"] == pytest.approx(1.3065, abs=REVIEW_TOLERANCE)
    assert hourly["general_split"] == pytest.approx(1.3065, abs=REVIEW_TOLERANCE)
    assert block["general_split_no_hour0_gain"] <= block["general_split"] + TOLERANCE

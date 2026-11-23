"""LP (27)'s largest box found reaches boxes held by optimal solutions that an independent search
found (`o3_lp27`, steps 2e and 5).

An independent search over LP (27)'s optimal face (own programmes, U0 given by its vertices: 40
seeded face vertices within 1e-7 of the optimal trace, then alternating ascent from the 8 best),
run on two seeded sets of 60 made-up days of two or three service hours, found the boxes of
`WITNESSES`: (rho, cars as (departure, rate kW, room kWh), product, the box found, rounded down
to four places, in kWh, the ceiling in kWh). The maps of the first two were found again at the
exact optimal trace and checked against every car's rows over U0. The largest found must be at
least each witness, to `TOLERANCE` relative to max(1, ceiling); the second and the fifth are the
ceiling itself, so nothing larger exists there.

What each pins (values of earlier forms of the search, on the same day):

- the first three: the search with 4 draws, 3 ascent starts and 4 rounds found 7.0349, 12.6625
  and 6.7504;
- the fourth: the search without the mixtures of step 2e found 3.6256;
- the fifth (one car): the block ascents without starts from the hourly ascents' maps found
  8.2344, while the hourly product reached the ceiling.
"""

import datetime

import pytest

from fleetreserve import o3_lp27
from fleetreserve.base_sets import day_average
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.o3_day import PRODUCTS, car_order

N_SERVICE = 3
WITNESSES = {
    "two-cars-hourly": (0.0, ((6, 1.0, 2.473), (5, 3.3, 10.975)), "hourly", 7.8040, 7.998),
    "at-ceiling": (0.0, ((3, 6.6, 9.768), (4, 6.6, 15.311)), "hourly", 13.2, 13.2),
    "block": (
        0.1,
        ((4, 1.0, 2.289), (3, 6.6, 7.326), (6, 6.6, 36.159)),
        "block",
        7.2142,
        15.6118,
    ),
    "mixtures": (
        0.0,
        ((5, 2.0, 5.473), (1, 6.6, 5.199), (5, 1.0, 3.936)),
        "block",
        4.0417,
        5.064,
    ),
    "other-product": (0.2, ((4, 6.6, 11.387),), "block", 8.8773, 8.8774),
}
TOLERANCE = 1e-5


@pytest.mark.parametrize("name", WITNESSES)
def test_the_largest_found_reaches_the_independent_witness(name):
    rho, cars, product, value, ceiling = WITNESSES[name]
    cars = tuple(TrueCar(*car) for car in cars)
    recovery = covering_recovery_hours(cars, N_SERVICE)
    day = Day("made-up", datetime.date(2020, 1, 1), N_SERVICE, recovery, rho, cars)
    rows = true_rows(car_order(day))
    found = o3_lp27.search(rows, day_average(rows), PRODUCTS, 300.0, 30.0)
    assert found.outcomes == dict.fromkeys(PRODUCTS, "optimal")
    assert found.high[product] >= value - TOLERANCE * max(1.0, ceiling), found.high
    assert found.high[product] <= ceiling + TOLERANCE * max(1.0, ceiling)

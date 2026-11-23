"""Outcome O3 on a day worked by hand (`o3_fake.hand_cars`), by `o3_day` and by the command.

The day: rho 0, two service hours (hours 1 and 2 after the decision hour 0), two cars in the fleet.

- Car A: departure hour 2, rate 1 kW, room 1 kWh, no recovery hour: it draws exactly 1 kWh over
  hours 0 and 1, and nothing in hour 2.
- Car B: departure hour 6, rate 1 kW, room 3 kWh; its three recovery hours (3 to 5) can supply
  3 kWh, so its need row is slack and its set on hours 0 to 2 is the cube [0, 1]^3.

Ceiling (V_hind, hour 0 free per path). b(S) = A min(1 x 1, 1) + B min(2 x 1, 3) = 3 and
p(S) = A max(1 - 1, 0) + B max(3 - 4, 0) = 0, so the hourly ceiling is 3: on a path with a high
hour-1 request A gives its 1 kWh in hour 1, on a low one in hour 0. The service hours' set is
{0 <= y1 <= 2, 0 <= y2 <= 1}; the block box is y1 = 1.5 +- 0.5, y2 = 0.5 +- 0.5, value 2.

Day-average base set (`base_sets.day_average`): A's limits (1, 1, 0) with energy 1, B's (1, 1, 1)
with energy 3, so power (1, 1, 0.5) and energy 2.

LP (24). u0 = (1, 1, 0) lies in the base set and u0 = 0 too; the fleet draws at least 1 kWh (A)
and at most 3 kWh (A 1, B 2) in hours 0 and 1, so p0 + p1 >= 1 and p0 + p1 + 2 alpha <= 3: alpha
<= 1, and alpha = 1 holds with A fixed and B given u0 (B's powers u0 are within [0, 1]). Box
(closed form): hourly alpha min(1 + 0.5, 2) = 1.5; block 2 lam = alpha min(min(1, 0.5), 2 / 2) =
0.5, value 2 lam x 2 = 1.

LP (27) (one programme per car, Remark 5). B: its set is the cube and each base-set hour reaches
its upper limit (1, 1, 0.5) with the others at 0, so B's largest trace is diag(1, 1, 2), 4, and
only it (an off-diagonal entry costs its own diagonal). A: A1 = a + c . u0 with A0 = 1 - A1 and
A2 = 0; A1 within [0, 1] gives c1 - c0 <= 1, so A's trace is 1, at c = (-a, 1 - a, 0) for any a
in [0, 1]. Trace 5. Every optimal solution is one such a with B's diagonal map. For every
such a, hour 1 of the certified set runs from 0 (u0 = (1, 0, .)) to 2 (u0 = (0, 1, .)) and hour 2
from 0 to 1 (u0_2 = 0.5, B's 2 u0_2), each corner within the base set's energy: hourly box 3,
block box 2. So the smallest and the largest box over LP (27)'s optimal solutions are both 3
(hourly) and 2 (block). LP (27)'s car maps move A's hour-0 power with the hour-1 request.

General split (T2's programme with every non-causal gain, decision-hour gains summing to zero over
the cars). With the requests y1 = m1 + z1 w1, y2 = m2 + z2 w2, z in [-1, 1]^2: A gives
0.5 - 0.5 z1 in hour 0, 0.5 + 0.5 z1 in hour 1, 0 in hour 2 (its 1 kWh over hours 0 and 1); B
gives 0.5 + 0.5 z1 in hour 0, 0.5 + 0.5 z1 in hour 1 and 0.5 + 0.5 z2 in hour 2 (within its cube).
The hour-0 gains are -0.5 and +0.5, summing to zero; the fleet gives 1 in hour 0, 1 + z1 in hour
1 (m1 = 1, w1 = 1) and 0.5 + 0.5 z2 in hour 2 (m2 = 0.5, w2 = 0.5): hourly value 2 (1 + 0.5) = 3,
the ceiling, so 3. Block (w1 = w2): B gives 0.5 in hour 1 instead, the fleet 1 + 0.5 z1 and
0.5 + 0.5 z2: 2, the block ceiling, so 2. B's hour-0 power follows the hour-1 request, which a
causal rule cannot do.

General split, decision hour without gain (the same programme with each car's hour-0 power one
number): A's hour-1 power is then fixed and it adds no width; B adds width 0.5 at most in each
service hour (its cube): hourly 2, block 2 (lam 0.5).

LP (27)'s bound (`o3_lp27_bound`): the largest found is the ceiling in both products, so the
bound is the ceiling, 3 and 2, closed.

So hourly (ceiling, LP (24), LP (27) smallest, largest found and bound, general split, without
hour-0 gain) = (3, 1.5, 3, 3, 3, 3, 2), shares 0.5, 1, 1, 1, 1, 2/3; block (2, 1, 2, 2, 2, 2, 2),
shares 0.5, 1, 1, 1, 1, 1.
"""

import csv
import datetime
import json

import pytest
from o3_fake import build, hand_cars, trust_commit

from fleetreserve import o3_run
from fleetreserve.day import Day, TrueCar, covering_recovery_hours
from fleetreserve.o3_day import DayTask, TimeLimits, compute_day

HOURLY = {
    "ceiling": 3.0,
    "lp24": 1.5,
    "lp27_smallest_found": 3.0,
    "lp27_largest_found": 3.0,
    "lp27_largest_bound": 3.0,
    "general_split": 3.0,
    "general_split_no_hour0_gain": 2.0,
}
BLOCK = {
    "ceiling": 2.0,
    "lp24": 1.0,
    "lp27_smallest_found": 2.0,
    "lp27_largest_found": 2.0,
    "lp27_largest_bound": 2.0,
    "general_split": 2.0,
    "general_split_no_hour0_gain": 2.0,
}
SHARES = {
    "hourly": {
        "lp24": 0.5,
        "lp27_smallest_found": 1.0,
        "lp27_largest_found": 1.0,
        "lp27_largest_bound": 1.0,
        "general_split": 1.0,
        "general_split_no_hour0_gain": 2.0 / 3.0,
    },
    "block": {
        "lp24": 0.5,
        "lp27_smallest_found": 1.0,
        "lp27_largest_found": 1.0,
        "lp27_largest_bound": 1.0,
        "general_split": 1.0,
        "general_split_no_hour0_gain": 1.0,
    },
}
ALPHA, TRACE = 1.0, 5.0
TOLERANCE = 1e-6
UNIT = "OSL_1"
DATES = ("2020-01-01", "2020-01-02", "2020-01-03")


def _hand_day() -> Day:
    cars = (TrueCar(2, 1.0, 1.0), TrueCar(6, 1.0, 3.0))
    return Day(UNIT, datetime.date(2020, 1, 1), 2, covering_recovery_hours(cars, 2), 0.0, cars)


def test_the_day_values_match_the_hand_values():
    task = DayTask(UNIT, "A", DATES[0], _hand_day(), "day-average", TimeLimits())
    result = compute_day(task)
    assert result.error is None
    for product, expected in (("hourly", HOURLY), ("block", BLOCK)):
        for key, value in expected.items():
            assert result.values[product][key] == pytest.approx(value, abs=TOLERANCE), (
                product,
                key,
            )
        for method, share in SHARES[product].items():
            assert result.share(product, method) == pytest.approx(share, abs=TOLERANCE)
    assert result.outcomes["hourly"] == {
        "ceiling": "exact",
        "lp24": "optimal",
        "lp27_smallest_found": "optimal",
        "lp27_largest_found": "optimal",
        "lp27_largest_bound": "closed",
        "general_split": "optimal",
        "general_split_no_hour0_gain": "optimal",
    }
    block = dict(result.outcomes["block"])
    assert block.pop("lp27_largest_bound") == "closed"
    assert set(block.values()) == {"optimal"}
    assert result.alpha_24 == pytest.approx(ALPHA, abs=TOLERANCE)
    assert result.trace_27 == pytest.approx(TRACE, abs=TOLERANCE)


@pytest.fixture(scope="module")
def hand_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("o3-hand")
    with pytest.MonkeyPatch.context() as patch:
        trust_commit(patch)
        cars = [car for date in DATES for car in hand_cars("A", date)]
        run, tables = build(root, {UNIT: cars}, service_hours=2)
        out = root / "out"
        out.mkdir()
        patch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = o3_run.main(
            [
                "--units",
                UNIT,
                "--days",
                "3",
                "--seed",
                "1",
                "--base-set",
                "day-average",
                "--tables",
                str(tables),
                "--truth-run",
                str(run),
            ]
        )
    return code, out


def test_the_command_reads_the_hand_day_from_the_tables(hand_run):
    code, out = hand_run
    assert code == 0
    with (out / o3_run.DAYS_NAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["date"], r["product"]) for r in rows] == [
        (date, product) for date in DATES for product in ("hourly", "block")
    ]
    for row in rows:
        assert row["cars"] == "2" and row["service_hours"] == "2"
        expected = HOURLY if row["product"] == "hourly" else BLOCK
        for key, value in expected.items():
            assert float(row[f"{key}_kwh"]) == pytest.approx(value, abs=TOLERANCE), (row, key)
        for method, share in SHARES[row["product"]].items():
            assert float(row[f"{method}_share"]) == pytest.approx(share, abs=TOLERANCE)
        assert float(row["lp24_alpha"]) == pytest.approx(ALPHA, abs=TOLERANCE)
        assert float(row["lp27_trace"]) == pytest.approx(TRACE, abs=TOLERANCE)


def test_the_unit_summary_of_the_hand_days(hand_run):
    _, out = hand_run
    document = json.loads((out / o3_run.RESULTS_NAME).read_text())
    assert document["sample"][UNIT]["sampled_days"] == 3
    assert document["missing_values"] == {}
    assert document["failures"] == []
    for product, expected in (("hourly", HOURLY), ("block", BLOCK)):
        entry = document["units"][UNIT][product]
        for key, value in expected.items():
            assert entry[key]["kwh"]["mean"] == pytest.approx(value, abs=TOLERANCE)
            assert entry[key]["kwh"]["median"] == pytest.approx(value, abs=TOLERANCE)
            assert entry[key]["days_with_value"] == 3
        for method, share in SHARES[product].items():
            assert entry[method]["share"]["mean"] == pytest.approx(share, abs=TOLERANCE)
            assert entry[method]["share"]["days"] == 3
            assert entry[method]["ratio_of_means"] == pytest.approx(share, abs=TOLERANCE)
    text = (out / o3_run.SUMMARY_NAME).read_text()
    assert "MISSING VALUES (never read as zero): 0 (none)" in text
    assert text.rstrip().endswith("all checks held")

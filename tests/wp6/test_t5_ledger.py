"""Plan section 5, T5 (fallback-monotone ledger) on sampled made-up day-paths and a written-out day.

T5: "for any beta and path: (a) no rung fails before its predecessor; (b) each car ends with >= its
fixed-plan energy under one-way re-solve, and >= its one-way energy under each two-way rung".
Checked for the one-way rung against the fixed plan and the two-way rung (B = 5 kWh) against the
one-way rung, at scales `BETAS`, on all `2^n` corner paths of each certified day, under each reading
of when a departure is seen (`rungs.py`, information model). Energies are compared within
`TOLERANCE` kWh.

Sample 1 (`RATE_SAFE`): forecasts conservative, with an early departure, or with less room than
forecast; no forecast rate above the true rate. T5(a), T5(b), and the order of slot loss and of
driver loss (each rung's no larger than its predecessor's) hold on every day-path. The one-way
rung leaves the fixed plan on `left` day-paths and moves the first failure later (or removes it)
on `later` of them, each pinned per reading: under "hour-start" none on this sample, since a rung
sees a car's departure only at the start of the hour after it; under "rounded-hour" it sees the
departure at the start of the departure hour itself.

Sample 2 (`ALL_KINDS`): also forecasts whose rate is above the true rate. T5(a), T5(b) and the
order of driver loss hold on every day-path.

Sample 3 (`RATE_OVER`): about two cars in three with a forecast rate above the true rate, the rest
leaving before their forecast. T5(a), T5(b) and the order of driver loss hold on every day-path.

The day `COUNTEREXAMPLE`, written out here: 4 service hours, 2 recovery hours, rho 0.1, block
product certified on the forecast sets (causal, solver's vertex). Cars as (true departure,
rate kW, room kWh | forecast departure, rate, cap, need):

    1: 7, 1.0, 4.16056 | 6.82882, 1.40071, 3.87334, 3.75575   (rate above the true rate)
    2: 1, 2.0, 0.99118 | 3.13660, 1.75469, 0.97733, 0.89785   (leaves before its forecast)
    3: 1, 2.0, 0.27375 | 0.76212, 3.07609, 0.26211, 0.24700
    4: 2, 2.0, 0.70049 | 1.69458, 2.66342, 0.69426, 0.63182
    5: 7, 2.0, 6.21014 | 6.70539, 2.53578, 6.16406, 5.60988

On every corner path at scales 0.5, 0.6 and 1.0, T5(a) and T5(b) hold under both readings
(`COUNTEREXAMPLE_BREAKS`). On the path (-1, +1, -1, -1) at scale 0.6, under "hour-start", the
one-way rung re-solves in hour 2 and plans car 1
above its true rate 1 kW; car 1 is capped. The re-solve in hour 3 is infeasible, so the fixed plan
stays in hour 3 (plan section 5: "if infeasible or timed out, the predecessor's plan stays"); the
rung re-solves again in hour 4, and car 1 ends at no less than under the fixed plan. An
alternative reading, in which a rung keeps its own earlier plan after a failed re-solve, breaks
T5(b) on this day: car 1 then ends below its fixed-plan energy on 2 of the 48 day-paths. Under
"rounded-hour" the rung sees cars 2 and 3 gone at the start of hour 1 and re-solves then; the
re-solve in hour 3 is again infeasible, and car 1 again ends at no less than under the fixed plan.
"""

import functools
import random
from dataclasses import dataclass

import numpy as np
import pytest
from made_up_days import DATE
from made_up_offer_days import departure_rule, offer_day

from fleetreserve import products
from fleetreserve.carsets import day_forecast_rows
from fleetreserve.certificate import ANY_OPTIMAL, certify
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.products import corner_paths
from fleetreserve.rungs import DEPARTURE_SEEN, RUNGS, run_rungs

BETAS = (0.5, 1.0, 2.0)
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Sample:
    seed: int
    n_days: int
    kinds: tuple[str, ...]
    certified: int
    day_paths: int
    left: tuple  # (reading, count) pairs, by reading of when a departure is seen
    later: tuple


def _by_reading(hour_start, rounded_hour):
    return (("hour-start", hour_start), ("rounded-hour", rounded_hour))


RATE_SAFE = Sample(
    17,
    20,
    ("conservative", "early departure", "less room"),
    19,
    516,
    _by_reading(159, 192),
    _by_reading(0, 59),
)
ALL_KINDS = Sample(
    7,
    20,
    ("conservative", "early departure", "less room", "lower rate"),
    20,
    492,
    _by_reading(189, 218),
    _by_reading(12, 64),
)
RATE_OVER = Sample(
    41,
    40,
    ("lower rate", "lower rate", "early departure"),
    40,
    1056,
    _by_reading(155, 233),
    _by_reading(0, 62),
)


def _later_or_equal(first, before):
    """`first` (a rung's first failure) is not earlier than `before` (its predecessor's)."""
    return first is None or (before is not None and first >= before)


def _check_t5a(runs):
    for previous, name in zip(RUNGS, RUNGS[1:], strict=False):
        before, now = runs[previous], runs[name]
        assert _later_or_equal(now.run.first_failure, before.run.first_failure)
        assert now.own_from is None or (
            before.run.first_failure is not None and now.own_from >= before.run.first_failure
        )
        assert now.run.slot_loss <= before.run.slot_loss


def _t5b_losers(runs):
    """Per rung after the fixed plan, the cars that end below their predecessor's energy."""
    return {
        name: np.flatnonzero(runs[name].run.final < runs[previous].run.final - TOLERANCE)
        for previous, name in zip(RUNGS, RUNGS[1:], strict=False)
    }


@functools.cache
def _counts(sample: Sample, seen: str):
    rng = random.Random(sample.seed)
    certified = day_paths = left = later = 0
    broken = driver_worse = 0
    for k in range(sample.n_days):
        day, certificate, _ = offer_day(rng, k, sample.kinds)
        if certificate.outcome != "optimal":
            continue
        certified += 1
        for beta in BETAS:
            for path in corner_paths(day.n_service):
                bound = departure_rule(day, seen)
                runs = run_rungs(
                    day, certificate.policy, beta, path, departure_bound=bound, departure_seen=seen
                )
                day_paths += 1
                _check_t5a(runs)
                broken += any(losers.size for losers in _t5b_losers(runs).values())
                driver_worse += any(
                    runs[name].run.driver_loss > runs[previous].run.driver_loss + TOLERANCE
                    for previous, name in zip(RUNGS, RUNGS[1:], strict=False)
                )
                left += runs["one-way"].own_from is not None
                later += runs["one-way"].run.first_failure != runs["fixed"].run.first_failure
    return (certified, day_paths, left, later), broken, driver_worse


def _expected(sample: Sample, seen: str):
    return (sample.certified, sample.day_paths, dict(sample.left)[seen], dict(sample.later)[seen])


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_t5_and_the_loss_order_hold_on_every_day_path_when_no_rate_is_over_forecast(seen):
    found, broken, driver_worse = _counts(RATE_SAFE, seen)
    print(seen, "rate-safe sample (certified, day-paths, left, later):", found)
    assert found == _expected(RATE_SAFE, seen)
    assert broken == 0
    assert driver_worse == 0


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_t5a_holds_on_every_day_path_with_every_kind_of_forecast(seen):
    found, broken, driver_worse = _counts(ALL_KINDS, seen)
    print(
        seen, "all-kinds sample (certified, day-paths, left, later):", found, broken, driver_worse
    )
    assert found == _expected(ALL_KINDS, seen)
    assert (broken, driver_worse) == (0, 0)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_t5_holds_on_every_day_path_when_most_rates_are_over_forecast(seen):
    found, broken, driver_worse = _counts(RATE_OVER, seen)
    print(
        seen, "rate-over sample (certified, day-paths, left, later):", found, broken, driver_worse
    )
    assert found == _expected(RATE_OVER, seen)
    assert (broken, driver_worse) == (0, 0)


COUNTEREXAMPLE = Day(
    "counterexample",
    DATE,
    4,
    2,
    0.1,
    (
        TrueCar(7, 1.0, 4.16056079890434),
        TrueCar(1, 2.0, 0.9911778242957345),
        TrueCar(1, 2.0, 0.27375482321301664),
        TrueCar(2, 2.0, 0.7004896276331872),
        TrueCar(7, 2.0, 6.210142201833124),
    ),
    (
        ForecastCar(6.828815989216846, 1.4007099464057966, 3.873340847334622, 3.755752870427336),
        ForecastCar(3.136596033642985, 1.754685556040155, 0.9773328579534232, 0.8978544341312722),
        ForecastCar(0.7621215061562905, 3.076089450410528, 0.2621146375804065, 0.2469983927753359),
        ForecastCar(1.6945772151760856, 2.6634178848758903, 0.6942632763747193, 0.6318212395764249),
        ForecastCar(6.705388338772137, 2.5357822467039814, 6.164056210873256, 5.6098820352654295),
    ),
    0.1,
)
COUNTEREXAMPLE_BREAKS = {0.5: (0, 0), 0.6: (0, 0), 1.0: (0, 0)}  # (one-way, two-way)
COUNTEREXAMPLE_PATH = (-1.0, 1.0, -1.0, -1.0)
COUNTEREXAMPLE_RESOLVES = _by_reading(
    [(2, "optimal"), (3, "infeasible"), (4, "optimal")],
    [(1, "optimal"), (3, "infeasible"), (4, "optimal")],
)


@pytest.mark.parametrize("seen", DEPARTURE_SEEN)
def test_t5_holds_on_the_written_out_day_with_a_rate_above_the_truth(seen):
    day = COUNTEREXAMPLE
    certificate = certify(
        day_forecast_rows(day), products.block(), "causal", shape_rule=ANY_OPTIMAL
    )
    assert certificate.outcome == "optimal"
    found = {}
    for beta in COUNTEREXAMPLE_BREAKS:
        counts = [0, 0]
        for path in corner_paths(day.n_service):
            bound = departure_rule(day, seen)
            runs = run_rungs(
                day, certificate.policy, beta, path, departure_bound=bound, departure_seen=seen
            )
            _check_t5a(runs)
            for k, losers in enumerate(_t5b_losers(runs).values()):
                counts[k] += losers.size > 0
            if beta == 0.6 and tuple(path) == COUNTEREXAMPLE_PATH:
                fixed, one = runs["fixed"].run, runs["one-way"].run
                np.testing.assert_allclose(fixed.final[0], 4.1377, atol=1e-4)
                assert one.final[0] >= fixed.final[0] - TOLERANCE
                resolves = [(r.hour, r.outcome) for r in runs["one-way"].resolves]
                print(seen, "re-solves on the path:", resolves)
                assert resolves == dict(COUNTEREXAMPLE_RESOLVES)[seen]
                if seen == "hour-start":
                    assert one.planned[0, 2] > day.cars[0].rate + 0.3  # planned above the truth
                    np.testing.assert_allclose(one.draws[0, 2], day.cars[0].rate, atol=1e-12)
                    assert runs["one-way"].plans[3] is runs["fixed"].plans[3]
        found[beta] = tuple(counts)
    print(seen, "T5(b) breaks (one-way, two-way) by scale:", found)
    assert found == COUNTEREXAMPLE_BREAKS

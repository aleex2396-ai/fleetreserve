"""The validity limit of the fixed plan's closed forms (`policy.VALIDITY_LIMIT`) and their
stated errors against replay, V1 to V8 of `fixed_plan_loss`, module text, "The validity limit and
the stated errors", in the exact rational arithmetic of `validity_errors` (this folder): no check
of a stated error allows a margin.

Hand days:

- `Z1`: one car, two service hours, leaving in hour 3 (no recovery hour), rate 1, room the double
  `1 + 5e-8`, planning that room in hour 1 and 0 in hour 2 with no gain. Its power-high row of
  hour 1 fails at the centre by that double less 1, within `ROW_TOLERANCE`, so it reads as holding
  and both driver-loss bounds are 0 at 1, at the limit and at 1e12. Replay draws 1 in hour 1,
  missing the request by the row's excess (the code's miss too), and ends short by it: its driver
  loss is either bound plus `D / P` exactly, at each of those scales (V1, V3).
- `EARLY` in the hand unit of `test_every_scale_calibrate.py` (19 copies of `corner_fake.SINGLE`
  and `EARLY` as calibration days; `EARLY` and two copies of `SINGLE` as test days): the earlier
  bound's tier A scale 0.5 (`gate_calibrate.EARLIER`; under the every-scale bound tier A
  abstains), no validity mark. At 0.5 the slot form fails `EARLY`'s hour 2 (car A plans 0.5 kWh
  in the hour after it has left); today's bound is 0 there and replay's driver loss 1/8 on every
  corner, `D / P` (that row's excess 0.5 over the protected 4 kWh), with no row read as holding
  exceeded, so above today's bound plus V7's error over the rows of `D` read as holding. The
  every-scale bound is 1/8, with `D_C` 0 (V3, V7).
- `fleet(k, rate, gain)`: one service hour, `k` cars leaving in hour 2 with rate `rate` and room
  twice it, each planning its rate with the gain `gain` on the request, the box summed in floating
  point. Every row's gains are at most `ZERO_GAIN`, so no hour fails in the closed form; `K_1 = k`
  (the power-high rows). At the limit, at the doubles on either side of it and at 99,999, with
  rates 3.3, 1.0, 7.4 and 0.7 and gains `ZERO_GAIN` and the double below it: the code's miss is
  within `E_1 + B_1 + R_1`, exact replay's within `E_1 + B_1`; with 1,000 cars, rate 3.3 and gains
  `ZERO_GAIN` at the limit, exact replay meets the request (misses it by 9.99999993e-5) and the
  code fails the hour (misses it by 1.0000000002e-4); at the next double `E_1` passes
  `FAILURE_KWH`; with 999 cars no hour fails (V2, V4 to V6).
- one car (rate 2, room 3.5, leaving in hour 3) planning 1.1 in its one service hour with the gain
  0.3: the code reads its power-high row's threshold as exactly 3, and at 3 the double 1.1 plus
  three times the double 0.3 passes the rate by 5.6e-17, within `delta_k` (V1); exact replay misses
  the request by that excess.
- `gap_day(centre, gain)`: one car (rate 2, room 3, leaving in hour 3) planning 1 in hour 1 with no
  gain, the box `1 + centre` with the gain `gain` on its request. No row is exceeded and exact
  replay misses the request by exactly `B_1` on the worst corner (V8): with a gap of 2e-4 in the
  centre the code's replay fails the hour at 1 and at the limit; with a gap of 1.1e-9 in the gains
  at the limit and not at 1.
- V8's two gap sizes at their edges, with no row exceeded: at scale 1, gaps in the centre of the
  doubles on either side of `FAILURE_KWH`; at the limit, gaps in the summed gains of the doubles
  on either side of `FAILURE_KWH / VALIDITY_LIMIT` and, on two cars, the gap of `1e-9 - 3e-26`
  summed exactly, above 1e-9 and below `FAILURE_KWH / VALIDITY_LIMIT`. Exact replay misses the
  request by `B_1` on the worst corner and fails the hour exactly where `B_1` is above
  `FAILURE_KWH`; the code's miss is within `B_1 + R_1`.

The search (`search`): seeded made-up days (`made_up_day`) of 1 to 3 service hours and 1 to 3
cars, centres at 0, at a car's rate, inside, a slack of 1e-4 from either end, or past either end by
up to `ROW_TOLERANCE`; gains zero, ordinary U(-1, 1), tiny (either sign, 10^U(-15, -12)), just
below `ZERO_GAIN` (U(0.9, 1) times it), just above it (U(1, 1.1) times it), or exactly
`ZERO_GAIN` or either double beside it; off-diagonal gains balanced by the last car. A planted day
puts on its first car a power-high row at its bound with a gain of U(0.5, 1) times `ZERO_GAIN`,
`ZERO_GAIN` or the double above it, offset in a later hour so that its need-row gains cancel, and
gives that car the room its plan exactly fills. Per day and scale (15 log-spaced from 1e-2 to the
limit and the double below it; the double above it, 1.01e5, 2e5, 1e6, 1e8, 1e10 and 1e12), on
every corner and two inner paths:

1. V2 and V3, at every scale;
2. V4, at every scale;
3. V1, at every scale (`validity_errors.row_bound`);
4. up to the limit, V5 for each row with an excess, `E_t` within V6's bound and V6's consequence
   (where `K_t LAMBDA + delta_t + B_t + R_t` is at most `FAILURE_KWH` in every hour before the
   closed form's first failing hour, the code's replay fails no hour before it), and V7; and where
   an hour fails in the closed form, today's bound plus V7's error over the rows of `D` read as
   holding is below replay's driver loss on some days;
5. just above the limit, at 1.01e5, a row read as holding is exceeded by more than V5's bound, and
   the code's miss exceeds `K_t LAMBDA + delta_t + B_t + R_t` before the first failing hour, on
   some days (the driver bounds above the limit: day `H3` of `test_every_scale_readings.py`);
6. the every-scale bound passes 1 by at most the excesses of the power-low rows of the hours
   outside `C` over the protected energy (`fixed_plan_loss`, module text, reading R3);
7. the route's copy of today's bound (`gate_calibrate.driver_terms`) within the rounding that
   `gate_calibrate`'s module text states (driver loss).

The search draws only from numpy's generator seeded with its seed, so its tallies depend on its
arguments alone.
"""

import datetime
import math
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest
from corner_fake import SINGLE, hand
from test_rungs_hand import EARLY, EARLY_POLICY
from validity_errors import (
    LAMBDA,
    U,
    balance_gaps,
    driver_error,
    driver_limit_error,
    driver_rows,
    exact,
    exact_bounds,
    exact_replay,
    excesses,
    forms,
    hour_errors,
    limit_bound,
    limit_errors,
    protected_energy,
    replayed,
    roundings,
    row_bound,
    tiny_count,
)

from fleetreserve import gate_calibrate
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.fixed_plan_loss import (
    CLOSING_KINDS,
    closed_form_slot_loss,
    compare_with_replay,
    drawn_policy,
    driver_loss_bound,
    every_scale_driver_bound,
    first_failing_hour,
    hour_thresholds,
)
from fleetreserve.policy import ROW_TOLERANCE, VALIDITY_LIMIT, ZERO_GAIN, AffinePolicy
from fleetreserve.products import corner_paths
from fleetreserve.replay import FAILURE_KWH

BELOW = math.nextafter(VALIDITY_LIMIT, 0.0)
ABOVE = math.nextafter(VALIDITY_LIMIT, math.inf)
INSIDE = tuple(float(b) for b in np.logspace(-2, 5, 15)) + (BELOW,)
OUTSIDE = (ABOVE, 1.01e5, 2e5, 1e6, 1e8, 1e10, 1e12)
FK = exact(FAILURE_KWH)
SEED = 20261117
DAYS = 30  # per kind of day (planted or not) in the test; a larger run sets its own number


def _day(n, cars, rho=0.0) -> Day:
    forecasts = tuple(ForecastCar(float(c.departure), c.rate, c.room, c.room) for c in cars)
    return Day("made-up", datetime.date(2020, 1, 1), n, 2, rho, tuple(cars), forecasts, 0.1)


def _policy(g, G) -> AffinePolicy:
    g, G = np.asarray(g, dtype=float), np.asarray(G, dtype=float)
    w = [G[:, t, t - 1].sum() for t in range(1, g.shape[1])]
    return AffinePolicy(g[:, 1:].sum(axis=0), w, g, G)


# --- hand days ----------------------------------------------------------------------------------


def test_z1_a_zero_gain_row_failing_within_row_tolerance_costs_exactly_its_stated_error():
    room = 1.0 + 5e-8
    day = _day(2, [TrueCar(3, 1.0, room)])
    policy = _policy([[0.0, room, 0.0]], np.zeros((1, 3, 2)))
    rows = true_rows(day)
    f = forms(policy, rows)
    over = exact(room) - 1
    assert 0 < over <= exact(ROW_TOLERANCE)
    for beta in (1.0, VALIDITY_LIMIT, 1e12):
        ex = excesses(f, rows, beta)
        assert hour_thresholds(policy, rows).tolist() == [math.inf, math.inf]
        assert driver_loss_bound(policy, rows, day, beta) == 0.0
        assert every_scale_driver_bound(policy, rows, day, beta) == 0.0
        assert exact_bounds(f, policy, rows, day, beta) == (0, 0)
        assert hour_errors(rows, ex) == [over, 0]
        assert driver_error(f, rows, beta, False, ex) == over
        assert driver_error(f, rows, beta, True, ex) == over
        assert tiny_count(f, rows, 1, ex) == 2
        for z in corner_paths(2):
            misses, driver = exact_replay(day, policy, beta, z)
            assert misses == [over, 0]
            assert driver == over / exact(room)
        for misses, first, _ in replayed(day, policy, beta, corner_paths(2)):
            assert misses == [over, 0]
            assert first is None


@pytest.fixture(scope="module")
def early_unit():
    prepared = [SINGLE] * 19 + [hand(EARLY, EARLY_POLICY)] + [SINGLE, SINGLE]
    return prepared, list(range(20)), [19, 20, 21]


def test_early_where_the_slot_form_fails_an_hour_todays_bound_is_not_within_a_small_error(
    early_unit,
):
    prepared, calibration, test = early_unit
    found = gate_calibrate.every_request(prepared, calibration, test, tier_b=True, earlier=True)
    earlier = found[gate_calibrate.EARLIER]
    beta = earlier["beta_used"]
    assert beta == pytest.approx(0.5) and gate_calibrate.VALIDITY not in earlier
    assert found["abstain"] and found["beta_used"] == 0.0
    early = prepared[19]
    day, policy = early.day, early.certificate.policy
    rows = true_rows(day)
    f = forms(policy, rows)
    ex = excesses(f, rows, beta)
    thresholds = hour_thresholds(policy, rows)
    assert first_failing_hour(thresholds, beta) == 2
    assert closed_form_slot_loss(thresholds, beta) == 0.5
    assert driver_loss_bound(policy, rows, day, beta) == 0.0
    assert gate_calibrate.ramp_loss(gate_calibrate.driver_terms(early), beta) == 0.0
    assert every_scale_driver_bound(policy, rows, day, beta) == 0.125
    today, every = exact_bounds(f, policy, rows, day, beta)
    total = protected_energy(day)
    assert (today, every, total) == (0, Fraction(1, 8), 4)
    assert driver_error(f, rows, beta, False, ex) == Fraction(1, 2)
    assert driver_error(f, rows, beta, True, ex) == 0
    holding = [k for k in range(len(ex)) if rows.kinds[k] != "need" and f.read[k] >= beta]
    assert holding and all(ex[k] == 0 for k in holding)
    small = sum(
        (limit_bound(f, k, beta, ex) or 0 for k in driver_rows(f, rows, beta, False)), Fraction(0)
    )
    assert small < Fraction(1, 10**12)
    for z in corner_paths(2):
        _, driver = exact_replay(day, policy, beta, z)
        assert driver == Fraction(1, 8) == today + driver_error(f, rows, beta, False, ex) / total
        assert driver > today + small / total
        assert driver <= every
    (compared,) = compare_with_replay(day, policy, rows, [beta])
    assert (compared.earlier_driver_bound, compared.replay_driver) == (0.0, 0.125)
    assert compared.driver_bound == 0.125
    assert found["test_offering_days"] == 3


def fleet(k: int, rate: float, gain: float):
    cars = tuple(TrueCar(2, rate, 2 * rate) for _ in range(k))
    day = Day("made-up", datetime.date(2020, 1, 1), 1, 0, 0.0, cars)
    g = np.zeros((k, 2))
    g[:, 1] = rate
    G = np.zeros((k, 2, 1))
    G[:, 1, 0] = gain
    return day, AffinePolicy(g[:, 1:].sum(axis=0), G[:, 1, :].sum(axis=0), g, G)


@pytest.fixture(scope="module")
def fleets():
    out = {}
    for k in (1000, 999):
        for rate in (3.3, 1.0, 7.4, 0.7):
            for gain in (ZERO_GAIN, math.nextafter(ZERO_GAIN, 0.0)):
                day, policy = fleet(k, rate, gain)
                rows = true_rows(day)
                f = forms(policy, rows)
                for beta in (99999.0, BELOW, VALIDITY_LIMIT, ABOVE):
                    ex = excesses(f, rows, beta)
                    errors, gaps = hour_errors(rows, ex), balance_gaps(policy, beta)
                    found = {
                        "t_min": first_failing_hour(hour_thresholds(policy, rows), beta),
                        "K": tiny_count(f, rows, 1, ex),
                        "E": errors[0],
                        "B": gaps[0],
                        "R": roundings(day, policy, beta, errors, gaps)[0],
                        "limit": limit_errors(f, rows, beta, ex)[0],
                        "exact": [exact_replay(day, policy, beta, [z])[0][0] for z in (1.0, -1.0)],
                        "code": replayed(day, policy, beta, [np.array([1.0]), np.array([-1.0])]),
                    }
                    out[(k, rate, gain, beta)] = found
    return out


def test_k_t_of_1000_rows_at_the_limit_and_its_neighbours_within_the_stated_errors(fleets):
    failed = 0
    for (k, _, gain, beta), found in fleets.items():
        assert found["t_min"] is None and found["K"] == k
        assert found["E"] == k * exact(beta) * exact(gain)
        for exact_miss, (code_miss, first, _) in zip(found["exact"], found["code"], strict=True):
            assert exact_miss <= found["E"] + found["B"]
            assert code_miss[0] <= found["E"] + found["B"] + found["R"]
            failed += int(k == 1000 and beta <= VALIDITY_LIMIT and first is not None)
        if beta <= VALIDITY_LIMIT:
            assert found["E"] <= k * LAMBDA < FK
            assert found["E"] <= found["limit"]
        if k == 999:
            assert all(first is None for _, first, _ in found["code"])
            assert found["limit"] + found["B"] + found["R"] < FK
    assert failed > 0
    boundary = fleets[(1000, 3.3, ZERO_GAIN, VALIDITY_LIMIT)]
    (code_miss, first, _), _ = boundary["code"]
    assert first == 1 and code_miss[0] > FK
    assert float(code_miss[0]) == pytest.approx(1.0000000002e-4, rel=1e-10)
    assert boundary["exact"][0] < FK
    assert float(boundary["exact"][0]) == pytest.approx(9.99999993e-5, rel=1e-8)
    assert fleets[(1000, 3.3, ZERO_GAIN, ABOVE)]["E"] > FK
    assert 0 < FK - 999 * LAMBDA - Fraction(1, 10**7) < Fraction(1, 10**20)
    assert 0 < FK - 1000 * LAMBDA < Fraction(1, 10**20)
    assert 0 < LAMBDA - exact(ROW_TOLERANCE) < Fraction(3, 10**24)
    assert 0 < exact(ABOVE) * exact(ZERO_GAIN) - LAMBDA < Fraction(2, 10**23)


def test_a_row_read_as_holding_at_its_own_threshold_is_exceeded_within_its_rounding():
    day = _day(1, [TrueCar(3, 2.0, 3.5)])
    policy = _policy([[0.0, 1.1]], [[[0.0], [0.3]]])
    rows = true_rows(day)
    f = forms(policy, rows)
    beta = 3.0
    assert hour_thresholds(policy, rows).tolist() == [beta]
    ex = excesses(f, rows, beta)
    (k,) = [j for j in range(len(ex)) if rows.kinds[j] == "power-high" and rows.last_hours[j] == 1]
    assert f.read[k] == beta and not f.tiny[k]
    assert ex[k] == exact(1.1) + 3 * exact(0.3) - 2 > 0
    assert float(ex[k]) == pytest.approx(5.551115123125783e-17, rel=1e-12)
    assert ex[k] <= row_bound(f, k, beta)
    errors, gaps = hour_errors(rows, ex), balance_gaps(policy, beta)
    assert errors == [ex[k]] and gaps == [0]
    assert exact_replay(day, policy, beta, [1.0])[0] == [ex[k]]
    ((misses, first, _),) = replayed(day, policy, beta, [np.array([1.0])])
    assert first is None
    assert misses[0] <= errors[0] + roundings(day, policy, beta, errors, gaps)[0]


def gap_day(centre: float, gain: float):
    day = _day(1, [TrueCar(3, 2.0, 3.0)])
    return day, AffinePolicy([1.0 + centre], [gain], [[0.0, 1.0]], np.zeros((1, 2, 1)))


def test_a_balance_gap_is_the_worst_corners_miss_where_no_row_is_exceeded():
    for centre, gain, fails_at in ((2e-4, 0.0, (1.0, VALIDITY_LIMIT)), (0.0, 1.1e-9, (1e5,))):
        day, policy = gap_day(centre, gain)
        rows = true_rows(day)
        f = forms(policy, rows)
        for beta in (1.0, VALIDITY_LIMIT):
            assert hour_errors(rows, excesses(f, rows, beta)) == [0]
            (gap,) = balance_gaps(policy, beta)
            assert gap == abs(exact(1.0 + centre) - 1) + exact(beta) * exact(gain)
            assert max(exact_replay(day, policy, beta, z)[0][0] for z in ([1.0], [-1.0])) == gap
            failed = any(first for _, first, _ in replayed(day, policy, beta, [[1.0], [-1.0]]))
            assert failed == (beta in fails_at)


def _gap_edge(day, policy, beta):
    """On a day of one service hour with no row exceeded: `B_1` and exact replay's larger miss over
    the two corners; the code's misses checked within `B_1 + R_1`."""
    rows = true_rows(day)
    errors = hour_errors(rows, excesses(forms(policy, rows), rows, beta))
    assert errors == [0]
    (gap,) = balance_gaps(policy, beta)
    worst = max(exact_replay(day, policy, beta, z)[0][0] for z in ([1.0], [-1.0]))
    (rounding,) = roundings(day, policy, beta, errors, [gap])
    for misses, _, _ in replayed(day, policy, beta, [[1.0], [-1.0]]):
        assert misses[0] <= gap + rounding
    return gap, worst


def test_at_scale_1_a_centre_gap_fails_the_hour_in_exact_replay_only_above_failure_kwh():
    below = 1.0 + FAILURE_KWH
    while exact(below) - 1 > FK:
        below = math.nextafter(below, 0.0)
    above = math.nextafter(below, 2.0)
    assert exact(below) - 1 <= FK < exact(above) - 1
    day, _ = gap_day(0.0, 0.0)
    for m, fails in ((below, False), (above, True)):
        policy = AffinePolicy([m], [0.0], [[0.0, 1.0]], np.zeros((1, 2, 1)))
        gap, worst = _gap_edge(day, policy, 1.0)
        assert gap == exact(m) - 1 == worst
        assert (worst > FK) == fails


def test_at_the_limit_a_gain_gap_fails_the_hour_in_exact_replay_only_above_its_size():
    edge = FK / exact(VALIDITY_LIMIT)
    below, above = math.nextafter(1e-9, 0.0), 1e-9
    assert math.nextafter(below, 1.0) == above and exact(below) < edge < exact(above)
    summed = exact(1e-9) + exact(-3e-26)
    assert Fraction(1, 10**9) < summed < edge
    day, _ = gap_day(0.0, 0.0)
    two = _day(1, [TrueCar(3, 2.0, 3.0), TrueCar(3, 2.0, 3.0)])
    G = np.zeros((2, 2, 1))
    G[0, 1, 0], G[1, 1, 0] = 1e-9, -3e-26
    cases = (
        (day, AffinePolicy([1.0], [below], [[0.0, 1.0]], np.zeros((1, 2, 1))), exact(below), False),
        (two, AffinePolicy([2.0], [0.0], [[0.0, 1.0], [0.0, 1.0]], G), summed, False),
        (day, AffinePolicy([1.0], [above], [[0.0, 1.0]], np.zeros((1, 2, 1))), exact(above), True),
    )
    for case_day, policy, gains_gap, fails in cases:
        gap, worst = _gap_edge(case_day, policy, VALIDITY_LIMIT)
        assert gap == exact(VALIDITY_LIMIT) * gains_gap == worst
        assert (worst > FK) == fails == (gains_gap > edge)


# --- the search ---------------------------------------------------------------------------------


def _gain(rng, ordinary: bool) -> float:
    u = rng.random()
    if u < 0.4:
        return 0.0
    if ordinary and u < 0.55:
        return float(rng.uniform(-1.0, 1.0))
    sign = float(rng.choice([-1.0, 1.0]))
    if u < 0.8:
        return sign * float(10 ** rng.uniform(-15, -12))
    if u < 0.88:
        return sign * ZERO_GAIN * float(rng.uniform(0.9, 1.0))
    if u < 0.95:
        return sign * ZERO_GAIN * float(rng.uniform(1.0, 1.1))
    edge = [math.nextafter(ZERO_GAIN, 0.0), ZERO_GAIN, math.nextafter(ZERO_GAIN, 1.0)]
    return sign * edge[int(rng.integers(0, 3))]


def _centre(rng, cap: float, planted: bool) -> float:
    if planted:
        return float(rng.uniform(0.2, 0.8)) * cap
    near = float(rng.uniform(0.0, ROW_TOLERANCE))
    return float(rng.choice([0.0, cap, rng.uniform(0.0, cap), cap - 1e-4, 1e-4, cap + near, -near]))


def _room(rng, g_i, supply: float, rho: float, rate: float, departure: int, tight: bool) -> float:
    if tight:
        return (g_i[1:].sum() + supply) / (1.0 - rho)
    u = rng.random()
    if u < 0.1:
        return 0.0
    if u < 0.4:
        hours = int(rng.integers(0, g_i.size))
        return min(max(g_i[0], 0.0) + g_i[1 : hours + 1].sum() + supply, 15.0)
    return float(rng.uniform(0.01, min(rate * departure / (1.0 - rho), 15.0)))


def _planted_gain(rng) -> float:
    u = rng.random()
    size = ZERO_GAIN * float(rng.uniform(0.5, 1.0))
    if u > 0.8:
        size = ZERO_GAIN if u < 0.9 else math.nextafter(ZERO_GAIN, 1.0)
    return float(rng.choice([-1.0, 1.0])) * size


def made_up_day(rng, planted: bool):
    """One made-up day and its policy (module text). A planted day keeps every car present
    through the window, its centres inside (0.2 to 0.8 of the rate), its gains zero or at most
    the double above `ZERO_GAIN` and the rooms of the other cars 0.5 to 3 kWh above what they
    plan."""
    n = int(rng.integers(2 if planted else 1, 4))
    k = int(rng.integers(2 if planted else 1, 4))
    rho = float(rng.choice([0.0, 0.1]))
    departures = [int(rng.integers(n + 1 if planted else 1, n + 4)) for _ in range(k)]
    rates = rng.uniform(0.5, 7.0, k)
    g = np.zeros((k, n + 1))
    G = np.zeros((k, n + 1, n))
    for i in range(k):
        g[i, 0] = float(rng.choice([0.0, rng.uniform(0.0, rates[i]), rates[i] + 0.5, -0.3]))
        for t in range(1, n + 1):
            g[i, t] = _centre(rng, rates[i] if t < departures[i] else 0.0, planted)
            for s in range(t):
                G[i, t, s] = _gain(rng, not planted)
    if planted:
        t1 = int(rng.integers(1, n))
        t2 = int(rng.integers(t1 + 1, n + 1))
        x = _planted_gain(rng)
        g[0, 0] = 0.0
        G[0] = 0.0
        G[0, t1, t1 - 1] = x
        G[0, t2, t1 - 1] = -x
        g[0, t1] = rates[0]
        g[0, t2] = min(max(g[0, t2], 0.5), rates[0])
    for t in range(1, n + 1):
        for s in range(n):
            if s != t - 1:
                G[-1, t, s] = -G[:-1, t, s].sum() if k > 1 else 0.0
    rooms = []
    for i in range(k):
        supply = rates[i] * max(departures[i] - 1 - n, 0)
        if planted and i > 0:
            room = max(g[i, 0], 0.0) + g[i, 1:].sum() + supply + float(rng.uniform(0.5, 3.0))
        else:
            room = _room(rng, g[i], supply, rho, rates[i], departures[i], planted and i == 0)
        top = rates[i] * departures[i] * (1.0 - 1e-12) / (1.0 - rho)  # the true set's largest
        rooms.append(min(max(float(room), 0.0), top))
    cars = [TrueCar(departures[i], float(rates[i]), rooms[i]) for i in range(k)]
    return _day(n, cars, rho), _policy(g, G)


def _route_check(day, policy, rows, terms, beta, tally):
    """Item 7 of the module text, at `beta`, with the route's terms `terms` of the day."""
    c_route, s_route, total = terms
    need = rows.kinds == "need"
    centre, spread = drawn_policy(policy, rows)._forms(rows)
    centre, s, b = centre[need], spread[need], rows.bounds[need]
    for i in range(s.size):
        allowed = U * (abs(exact(centre[i]) + exact(s[i])) + exact(s[i])) * (1 + U)
        tally["route gains"] += 1
        tally["route gain beyond rounding"] += int(abs(exact(s_route[i]) - exact(s[i])) > allowed)
    if total <= 0:
        return
    beta_ = exact(beta)
    route = today = allowed = Fraction(0)
    for i in range(s.size):
        route += max(Fraction(0), exact(c_route[i]) + beta_ * exact(s_route[i]))
        today += max(Fraction(0), exact(centre[i]) + beta_ * exact(s[i]) - exact(b[i]))
        rounding = U * (abs(exact(centre[i]) + exact(s[i])) + exact(s[i])) * (1 + U)
        allowed += U * abs(exact(centre[i]) - exact(b[i])) + beta_ * rounding
    tally["route bounds"] += 1
    tally["route bound beyond rounding"] += int(abs(route - today) > allowed)


def _one_scale(day, policy, rows, f, terms, beta, paths, tally):
    inside = beta <= VALIDITY_LIMIT
    where = "inside" if inside else "outside"
    ex = excesses(f, rows, beta)
    t_min = first_failing_hour(hour_thresholds(policy, rows), beta)
    before = range((t_min - 1) if t_min is not None else day.n_service)
    errors, gaps = hour_errors(rows, ex), balance_gaps(policy, beta)
    rounding = roundings(day, policy, beta, errors, gaps)
    certified = limit_errors(f, rows, beta, ex)
    total = protected_energy(day)
    for k in np.flatnonzero(np.isin(rows.kinds, CLOSING_KINDS) & (rows.last_hours >= 1)):
        bound = row_bound(f, k, beta)
        if bound is not None:
            tally[f"rows read as holding {where}"] += 1
            tally[f"rows beyond their bound {where}"] += int(ex[k] > bound)
            limit = limit_bound(f, k, beta, ex)
            tally[f"rows beyond the limit's bound at {beta!r}"] += int(ex[k] > limit)
    for t in before:
        tally["hours before the first failing hour"] += 1
        tally["E beyond K LAMBDA + rho inside"] += int(inside and errors[t] > certified[t])
    today, every = exact_bounds(f, policy, rows, day, beta)
    d, d_c = driver_error(f, rows, beta, False, ex), driver_error(f, rows, beta, True, ex)
    small_c = driver_limit_error(f, rows, beta, True, ex)
    small = sum(
        (limit_bound(f, k, beta, ex) or 0 for k in driver_rows(f, rows, beta, False)), Fraction(0)
    )
    if total > 0:
        low = driver_error(f, rows, beta, True, ex, kind="power-low")
        tally[f"R3 above 1 beyond its power-low excesses {where}"] += int(every > 1 + low / total)
        tally[f"R3 above 1 {where}"] += int(every > 1)
    safe = all(certified[t] + gaps[t] + rounding[t] <= FK for t in before)
    code = replayed(day, policy, beta, paths)
    for z, (code_misses, first, _) in zip(paths, code, strict=True):
        misses, driver = exact_replay(day, policy, beta, z)
        tally[f"paths {where}"] += 1
        tally[f"exact misses beyond E + B {where}"] += int(
            any(misses[t] > errors[t] + gaps[t] for t in range(day.n_service))
        )
        tally[f"code misses beyond E + B + R {where}"] += int(
            any(code_misses[t] > errors[t] + gaps[t] + rounding[t] for t in range(day.n_service))
        )
        beyond = any(code_misses[t] > certified[t] + gaps[t] + rounding[t] for t in before)
        tally[f"code misses beyond the limit's bound at {beta!r}"] += int(beyond)
        if inside and safe:
            tally["failing-hour checks inside"] += 1
            tally["replay earlier inside"] += int(first is not None and first <= len(before))
        if total <= 0:
            continue
        tally[f"driver comparisons {where}"] += 1
        tally[f"R2 beyond bound + D / P {where}"] += int(driver > today + d / total)
        tally[f"R3 beyond bound + D_C / P {where}"] += int(driver > every + d_c / total)
        tally[f"R3 beyond the limit's bound at {beta!r}"] += int(driver > every + small_c / total)
        if t_min is None:
            tally[f"R2 no failing hour {where}"] += 1
            limit = driver_limit_error(f, rows, beta, False, ex)
            tally[f"R2 beyond the limit's bound at {beta!r}"] += int(driver > today + limit / total)
        else:
            tally[f"R2 with a failing hour {where}"] += 1
            tally[f"R2 beyond the small error with a failing hour {where}"] += int(
                driver > today + small / total
            )
    if inside:
        _route_check(day, policy, rows, terms, beta, tally)


def search(days: int, seed: int) -> Counter:
    """The tallies of the search (module text) over `days` planted and `days` other days."""
    rng = np.random.default_rng(seed)
    tally: Counter = Counter()
    for planted in (True, False):
        for _ in range(days):
            day, policy = made_up_day(rng, planted)
            rows = true_rows(day)
            f = forms(policy, rows)
            terms = gate_calibrate.driver_terms(hand(day, policy))
            paths = list(corner_paths(day.n_service)) + [
                rng.uniform(-1.0, 1.0, day.n_service) for _ in range(2)
            ]
            tally["days"] += 1
            tally["planted days"] += int(planted)
            for beta in INSIDE + OUTSIDE:
                _one_scale(day, policy, rows, f, terms, beta, paths, tally)
    return tally


@pytest.fixture(scope="module")
def tally():
    return search(DAYS, SEED)


def test_in_exact_arithmetic_the_stated_errors_hold_at_every_scale(tally):
    for where in ("inside", "outside"):
        assert tally[f"paths {where}"] > 1000 and tally[f"driver comparisons {where}"] > 1000
        assert tally[f"exact misses beyond E + B {where}"] == 0
        assert tally[f"R2 beyond bound + D / P {where}"] == 0
        assert tally[f"R3 beyond bound + D_C / P {where}"] == 0


def test_the_codes_replay_misses_by_at_most_the_stated_errors_and_its_rounding(tally):
    for where in ("inside", "outside"):
        assert tally[f"code misses beyond E + B + R {where}"] == 0


def test_every_row_read_as_holding_is_exceeded_by_at_most_its_bound(tally):
    for where in ("inside", "outside"):
        assert tally[f"rows read as holding {where}"] > 1000
        assert tally[f"rows beyond their bound {where}"] == 0


def test_up_to_the_limit_the_limits_bounds_hold_and_replay_fails_no_earlier(tally):
    assert tally["hours before the first failing hour"] > 1000
    assert tally["E beyond K LAMBDA + rho inside"] == 0
    assert tally["failing-hour checks inside"] > 1000
    assert tally["replay earlier inside"] == 0
    for beta in INSIDE:
        assert tally[f"rows beyond the limit's bound at {beta!r}"] == 0
        assert tally[f"code misses beyond the limit's bound at {beta!r}"] == 0
        assert tally[f"R3 beyond the limit's bound at {beta!r}"] == 0
        assert tally[f"R2 beyond the limit's bound at {beta!r}"] == 0
    assert tally["R2 no failing hour inside"] > 1000
    assert tally["R2 with a failing hour inside"] > 1000
    assert tally["R2 beyond the small error with a failing hour inside"] > 0


def test_just_above_the_limit_the_limits_bounds_fail_on_some_days(tally):
    assert tally[f"rows beyond the limit's bound at {OUTSIDE[1]!r}"] > 0
    assert tally[f"code misses beyond the limit's bound at {OUTSIDE[1]!r}"] > 0


def test_the_every_scale_bound_passes_1_by_at_most_its_power_low_excesses(tally):
    for where in ("inside", "outside"):
        assert tally[f"R3 above 1 beyond its power-low excesses {where}"] == 0


def test_the_routes_copy_of_todays_bound_differs_only_by_the_stated_rounding(tally):
    assert tally["route gains"] > 100
    assert tally["route bounds"] > 100
    assert tally["route gain beyond rounding"] == 0
    assert tally["route bound beyond rounding"] == 0

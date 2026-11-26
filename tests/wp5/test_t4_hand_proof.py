"""Plan section 4, T4: the claims of two hand proofs (Theorem B with Corollary C, and Theorem H),
checked on the made-up days of `test_hourly_equality` and on hand-made days. None of the claims
had been run before this file.

Words. Hour 0 is the decision hour: each car's power in it is one number fixed before the first
request. The service hours S are hours 1 to n; the recovery hours are the hours after S. A car's
limits are a power range in every hour and a range for the energy stored since the decision time
after every hour (`Limits` below). `b_i(c)` and `p_i(c)` are the most and the least energy car i can
draw over S once it has drawn `c` in hour 0; `w_i(c) = b_i(c) - p_i(c)` is the width it keeps.
`V_commit` is the sum over cars of the largest `w_i(c)` over the hour-0 power `c`. `b(S)` and `p(S)`
are the most and the least the fleet can draw over S with hour 0 free; `b(S) - p(S)` is the
hindsight ceiling. The box of a car is `[w, x]`: every service-hour power vector that takes `w_t`
or `x_t` in each hour is a corner.

Which test checks which claim.

- Theorem B (diagonal = affine causal = tree = `V_commit`, and `V_commit <= b - p`):
  `test_theorem_b_causal_values_equal_v_commit_from_the_cars_own_limits` and
  `test_theorem_b_v_commit_is_at_most_the_hindsight_ceiling`. The first also exercises Lemma 4
  (the recovery hours are free in the routine below, but charge at full power until full or gone
  in the repository's rows).
- Lemma 4 on its own: `test_lemma_4_free_and_full_power_recovery_give_the_same_sets_per_car`
  computes, for every car, the feasible hour-0 powers and the most and least over S under both
  recovery rules with the routine below, and requires them equal (for cars in the capped class
  they are the same expression, which its docstring says).
  `test_lemma_4_a_car_with_a_forced_draw_in_recovery_is_outside_the_capped_class` adds a car
  outside the class, where the two rules must differ.
- Corollary C (no day below the ceiling meets its conditions):
  `test_corollary_c_equality_holds_on_every_day_meeting_its_conditions`.
- Theorem H (a box of width `b(S) - p(S)` lies in the fleet's set, so the hindsight value is
  `b(S) - p(S)`): `test_theorem_h_the_box_of_full_width_lies_in_the_set`. It builds each car's box
  as the proof does and decides whether every corner can be served by another routine.
  `test_servable_refuses_a_vector_only_because_of_a_recovery_hour` checks that routine on two
  cars where only a recovery hour refuses a vector.
  `test_repository_ceiling_equals_an_independent_b_minus_p` is NOT Theorem H: it only compares
  the repository's ceiling, which is `b(S) - p(S)` by construction, with an independent `b - p`.
- Hand days, with the numbers written as literals so that the repository, the routine that
  computes `V_commit`, `b` and `p`, the box builder, or the routine that decides whether a car
  can serve a vector can each be seen to be wrong: Example 5.4 (the smallest failing day;
  V_commit 0, b - p 1), 5.5 (the same car with a recovery hour; 1 and 1), 5.6 (a two-way car; 1.5
  and 2), 7.7 (a two-way car with two service hours; 1 and 2) and 7.9 (that car beside a one-way
  car; 1 and 3). Example 7.8 (a forced draw and a need) is not included.

How the values are computed here, without calling the code under test, in exact fractions from
each car's own limits. `_profile`: a backward pass over the stored-energy intervals gives the
hour-0 powers that can be completed; for a given hour-0 power `c` the most and the least drawn
over S are `min(Hn, Kh - c)` and `max(Ln, Kl - c)`, two piecewise linear functions with one bend
each. `_car_box`: the proof's construction (highest path for `x`, then the lowest path below it
from the largest feasible hour-0 power for `w`). `_servable`: forward interval steps only; it turns
each stored-energy limit of hours 0 to n into a bound on the hour-0 power, then follows the
reachable stored energies through the recovery hours. It shares no step with `_car_box`.

Tolerance: `test_hourly_equality.TOLERANCE`, relative to `max(1, value)`, between the repository's
linear-programme values and the exact fractions turned into floats. Routine against routine
comparisons are exact.

Not covered by the made-up days: a site cap, driver-risk rows, forecast sets, several decision
hours, rolling-window or ramp limits. The days here have none of them, so the claims are checked
inside the class the hand proofs cover and nowhere else.
"""

import functools
import random
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from made_up_days import DATE, RHOS
from test_hourly_equality import TOLERANCE, _cases, _close, _hourly_values

from fleetreserve import exact
from fleetreserve.carsets import CarRows, stack, true_rows
from fleetreserve.day import Day, StoredEnergyCar, TrueCar, covering_recovery_hours

SUPPLIED_SEED = 29
N_SUPPLIED = 30  # one-way days built to meet the conditions of Corollary C

BIG = Fraction(10**9)  # stands for "no limit" in a stored-energy floor or cap


@dataclass(frozen=True)
class Limits:
    """One car as the hand proofs describe it, over the hours 0 to H (H the last recovery hour).

    `low[t] <= power[t] <= high[t]` and `floor[t] <= stored energy after hour t <= cap[t]`, all
    exact fractions; `-BIG` and `BIG` stand for no limit. Hour 0 is the decision hour.
    """

    low: tuple
    high: tuple
    floor: tuple
    cap: tuple

    @property
    def last_hour(self):
        return len(self.low) - 1


def _true_limits(car, day, budget):
    """The plan's true car as `Limits`, written from its departure, rate, room and protected
    energy (the need `(1 - rho) * room` sits on the last hour the car is present)."""
    hours = day.n_hours
    rate, room = Fraction(car.rate), Fraction(car.room)
    need = Fraction((1 - day.rho) * car.room)
    present = [t < car.departure for t in range(hours)]
    low_when_present = Fraction(0) if budget is None else -rate
    floor_default = -BIG if budget is None else -Fraction(budget)
    floor = [floor_default] * hours
    floor[car.departure - 1] = max(floor_default, need)
    return Limits(
        tuple(low_when_present if p else Fraction(0) for p in present),
        tuple(rate if p else Fraction(0) for p in present),
        tuple(floor),
        (room,) * hours,
    )


def _stored_limits(low, high, stored_low, stored_high, need, hours):
    """A hand-made car with the same limits in every hour and a need on the last hour."""
    low, high, stored_low, stored_high, need = (
        Fraction(v) for v in (low, high, stored_low, stored_high, need)
    )
    floor = [stored_low] * hours
    floor[-1] = max(stored_low, need)
    return Limits((low,) * hours, (high,) * hours, tuple(floor), (stored_high,) * hours)


def _completion(limits, n_service, rule):
    """The stored energies after hour n from which the later hours can be finished, as
    `(low, high)`.

    Rule "free": any completion within the limits, by a backward pass over the recovery hours.
    Rule "full power": the plan's rule, charge at full power until full or gone. For a car whose
    recovery hours have lowest power <= 0 <= highest power and one cap, the stored energy x after
    hour n becomes `min(x + highest powers so far, cap)`, which must reach every later floor.
    """
    n = n_service
    last = limits.last_hour
    if rule == "free":
        low, high = limits.floor[last], limits.cap[last]
        assert low <= high, "empty car"
        for t in range(last, n, -1):
            low = max(limits.floor[t - 1], low - limits.high[t])
            high = min(limits.cap[t - 1], high - limits.low[t])
            assert low <= high, "empty car"
        return low, high
    assert rule == "full power", rule
    low, high = limits.floor[n], limits.cap[n]
    assert low <= high, "empty car"
    for t in range(n + 1, last + 1):
        assert limits.low[t] <= 0 <= limits.high[t] and limits.cap[t] == high, "not a capped car"
        assert limits.floor[t] <= high, "empty car"
        low = max(low, limits.floor[t] - sum(limits.high[n + 1 : t + 1]))
    return low, high


def _profile(limits, n_service, rule="free"):
    """The car's hour-0 interval and the constants of its slice functions.

    By a backward pass over the stored-energy intervals from hour n down to hour 0, the stored
    energies after hour t from which the later hours can still be completed form an interval. The
    hour-0 power `c` is possible iff it lies in the interval for hour 0 and in the hour-0 power
    limits. From a start `c` the highest stored energy after hour n is `min(c + high[1..n],
    B_s + high[s+1..n] for s = 1..n, top of the completion interval)`, the lowest is the same
    with the lowest powers and floors; so the most the car can draw over the service hours is
    `min(Hn, Kh - c)` and the least is `max(Ln, Kl - c)`.
    """
    n = n_service
    low, high = _completion(limits, n, rule)
    bottom_end, top_end = low, high
    for t in range(n, 0, -1):
        low = max(limits.floor[t - 1], low - limits.high[t])
        high = min(limits.cap[t - 1], high - limits.low[t])
        assert low <= high, "empty car"
    c_low = max(low, limits.low[0])
    c_high = min(high, limits.high[0])
    assert c_low <= c_high, "empty car"
    most = sum(limits.high[1 : n + 1])
    least = sum(limits.low[1 : n + 1])
    top = min(
        [top_end] + [limits.cap[s] + sum(limits.high[s + 1 : n + 1]) for s in range(1, n + 1)]
    )
    bottom = max(
        [bottom_end] + [limits.floor[s] + sum(limits.low[s + 1 : n + 1]) for s in range(1, n + 1)]
    )
    return c_low, c_high, most, least, top, bottom


def _curves(limits, n_service, rule="free"):
    """The hour-0 powers worth looking at (the ends of the interval and the two bends that fall
    inside it) and the functions `b_at(c)`, `p_at(c)`: the most and the least drawn over S."""
    c_low, c_high, most, least, top, bottom = _profile(limits, n_service, rule)
    points = [c for c in (c_low, c_high, top - most, bottom - least) if c_low <= c <= c_high]

    def b_at(c):
        return min(most, top - c)

    def p_at(c):
        return max(least, bottom - c)

    return points, b_at, p_at


def _car_extremes(limits, n_service, rule="free"):
    """`(widest, b, p)` of one car: the largest, over the hour-0 power c, of most minus least
    energy drawn over the service hours after drawing c; the most over the hour-0 powers; the
    least over them. Each function of c is piecewise linear with one bend, so the maxima and
    minima are met at an end of the interval or at a bend."""
    points, b_at, p_at = _curves(limits, n_service, rule)
    return (
        max(b_at(c) - p_at(c) for c in points),
        max(b_at(c) for c in points),
        min(p_at(c) for c in points),
    )


def _car_box(limits, n_service):
    """The box `(w, x)` of the hand proof for Theorem H, one car, in exact fractions.

    `x` is a point of the set with the largest total: the highest stored-energy path from the
    hour-0 power that allows the largest total, never above the highest level from which hour n
    can still be reached. `w <= x` is the lowest stored-energy path (the proof's lowest path,
    where the upper power limits are replaced by `x`) from the largest feasible hour-0 power,
    which reaches the smallest total.
    """
    n = n_service
    points, b_at, _ = _curves(limits, n)
    start = max(points, key=b_at)
    floor = list(limits.floor[: n + 1])
    cap = list(limits.cap[: n + 1])
    floor[n], cap[n] = _completion(limits, n, "free")
    level_high = list(cap)  # highest stored energy after hour t from which hour n can be reached
    for t in range(n, 0, -1):
        level_high[t - 1] = min(cap[t - 1], level_high[t] - limits.low[t])
    stored, x = start, []
    for t in range(1, n + 1):
        after = min(stored + limits.high[t], level_high[t])
        x.append(after - stored)
        stored = after
    lowest = [None] * (n + 1)  # the floors made consistent with the powers `x` that can follow
    lowest[n] = floor[n]
    for t in range(n - 1, 0, -1):
        lowest[t] = max(floor[t], lowest[t + 1] - x[t])
    start_w = min(
        limits.high[0], cap[0], *(cap[t] - sum(limits.low[1 : t + 1]) for t in range(1, n + 1))
    )
    stored, w = start_w, []
    for t in range(1, n + 1):
        after = max(stored + limits.low[t], lowest[t])
        w.append(after - stored)
        stored = after
    return tuple(w), tuple(x)


def _start_interval(limits, n_service, u):
    """The hour-0 powers that let the car draw `u` in the service hours, within every limit of
    hours 0 to n, as `(low, high)`; None when there is none. Written without the backward pass:
    each stored-energy limit is turned into a bound on the hour-0 power."""
    n = n_service
    if any(not limits.low[t] <= u[t - 1] <= limits.high[t] for t in range(1, n + 1)):
        return None
    low, high = limits.low[0], limits.high[0]
    drawn = Fraction(0)
    for t in range(n + 1):
        if t > 0:
            drawn += u[t - 1]
        low = max(low, limits.floor[t] - drawn)
        high = min(high, limits.cap[t] - drawn)
    return (low, high) if low <= high else None


def _servable(limits, n_service, u):
    """True when the car can draw `u` in the service hours for SOME hour-0 power and SOME
    completion of the recovery hours: the hour-0 powers allowed by hours 0 to n, then the stored
    energies reachable hour by hour through the recovery hours (forward interval steps)."""
    start = _start_interval(limits, n_service, u)
    if start is None:
        return False
    low = start[0] + sum(u)
    high = start[1] + sum(u)
    for t in range(n_service + 1, limits.last_hour + 1):
        low = max(limits.floor[t], low + limits.low[t])
        high = min(limits.cap[t], high + limits.high[t])
        if low > high:
            return False
    return True


def _corners(w, x):
    """Every vector that takes `w_t` or `x_t` in each service hour."""
    return product(*zip(w, x, strict=True))


def _fleet_extremes(cars, n_service):
    """`(V_commit, b(S), p(S))` of independent cars: sums of each car's `_car_extremes`."""
    parts = [_car_extremes(car, n_service) for car in cars]
    return tuple(sum(part[k] for part in parts) for k in range(3))


def _corollary_failures(limits, n_service):
    """Which conditions of the corollary the car misses (an empty list: it meets them all).

    Conditions: its lowest power is 0 in every hour (one-way, no forced draw); no positive
    stored-energy requirement in hours 0 to n; one cap, the same at hour n and every recovery hour;
    recovery supply: charging at full power from stored energy 0 after hour n meets every floor.
    """
    last = limits.last_hour
    n = n_service
    failures = []
    if any(v != 0 for v in limits.low):
        failures.append("lowest power is not 0 in every hour")
    if any(limits.floor[t] > 0 for t in range(n + 1)):
        failures.append("a positive stored-energy requirement inside the service hours")
    if len({limits.cap[t] for t in range(n, last + 1)}) != 1:
        failures.append("the cap differs between hour n and the recovery hours")
    stored = Fraction(0)
    for t in range(n + 1, last + 1):
        stored = min(stored + limits.high[t], limits.cap[t])
        if stored < limits.floor[t]:
            failures.append("the recovery hours alone cannot meet the need")
            break
    return failures


@dataclass(frozen=True, eq=False)
class Case:
    """One day with the repository's values and the values computed here from the limits."""

    name: str
    kind: str  # "one-way" or "two-way"
    n_service: int
    cars: tuple  # of Limits
    values: dict  # repository: diagonal, causal, tree, ceiling
    commit: float
    b: float
    p: float


def _day_case(day, budget, values):
    cars = tuple(_true_limits(car, day, budget) for car in day.cars)
    commit, b, p = _fleet_extremes(cars, day.n_service)
    kind = "one-way" if budget is None else "two-way"
    name = f"{kind} budget {budget} rho {day.rho} n_service {day.n_service} cars {day.cars}"
    return Case(name, kind, day.n_service, cars, values, float(commit), float(b), float(p))


def _supplied_day(rng, k):
    """A one-way day whose every car leaves after the service hours (at least one whole recovery
    hour before departure) and whose protected energy is at most 99 percent of what its recovery
    hours alone can supply, or whose room is 0: the conditions of Corollary C."""
    n_service = rng.randint(2, 4)
    rho = RHOS[k % 3]
    cars = []
    for _ in range(rng.randint(1, 4)):
        rate = float(rng.choice([1, 2, 3]))
        departure = rng.randint(n_service + 2, n_service + 3)
        supply = rate * (departure - 1 - n_service)
        room = 0.0 if rng.random() < 1 / 6 else rng.uniform(0.2, 0.99) * supply / (1 - rho)
        cars.append(TrueCar(departure, rate, room))
    n_recovery = covering_recovery_hours(cars, n_service)
    return Day("made-up", DATE, n_service, n_recovery, rho, cars)


@functools.cache
def _all_cases():
    """The 160 cases of `test_hourly_equality` (80 days, one-way and two-way), then the
    supplied days (one-way)."""
    cases = [_day_case(day, budget, values) for day, budget, values in _cases()]
    rng = random.Random(SUPPLIED_SEED)
    for k in range(N_SUPPLIED):
        day = _supplied_day(rng, k)
        values = _hourly_values(true_rows(day, None))
        values["ceiling"] = exact.hourly_ceiling(exact.day_functions(day, None))
        cases.append(_day_case(day, None, values))
    return cases


def _below(value, ceiling):
    return value < ceiling - TOLERANCE * max(1.0, abs(ceiling))


def test_theorem_b_causal_values_equal_v_commit_from_the_cars_own_limits():
    """If it failed, a causal rule's value differs from the sum over cars of the best width left
    after the hour-0 power, so the proof's description of the causal values is wrong for the
    repository's rows (or the plan's fixed recovery rule differs from a free completion)."""
    cases = _all_cases()
    kinds = {case.kind for case in cases}
    outside = [case.name for case in cases if not any(car.high[0] > 0 for car in case.cars)]
    wrong = [
        (case.name, key, case.values[key], case.commit)
        for case in cases
        for key in ("diagonal", "causal", "tree")
        if not _close(case.values[key], case.commit)
    ]
    separated = sum(_below(case.commit, case.b - case.p) for case in cases)
    counts = f"{len(cases)} days, kinds {sorted(kinds)}, {separated} with V_commit below b - p"
    assert kinds == {"one-way", "two-way"}, counts
    assert not outside, f"days with no hour-0 power in any car: {outside}"
    assert separated >= 1, f"no day tells V_commit from b - p: {counts}"
    assert not wrong, f"{len(wrong)} differences; {counts}: {wrong}"


def test_theorem_b_v_commit_is_at_most_the_hindsight_ceiling():
    """If it failed, a value reached with the hour-0 power fixed in advance is above the value
    reached with hour 0 free, which no rule can do."""
    cases = _all_cases()
    over = [
        (case.name, case.commit, case.b - case.p, case.values["ceiling"])
        for case in cases
        if case.commit > case.b - case.p + TOLERANCE * max(1.0, case.b - case.p)
        or case.commit > case.values["ceiling"] + TOLERANCE * max(1.0, case.values["ceiling"])
    ]
    assert len(cases) >= 1
    assert not over, over


def test_corollary_c_equality_holds_on_every_day_meeting_its_conditions():
    """If it failed, either a day meeting the conditions has a diagonal value below `b - p`
    (the corollary is false), or the two ceilings (the repository's and the one computed here)
    differ. The days below the ceiling must be none of those meeting the conditions."""
    cases = _all_cases()
    met, below, below_and_met, unequal = [], [], [], []
    for case in cases:
        failures = [f for car in case.cars for f in _corollary_failures(car, case.n_service)]
        value = case.values["diagonal"]
        ceiling = case.values["ceiling"]
        is_below = _below(value, ceiling)
        if is_below:
            below.append(case)
        if failures:
            continue
        met.append(case.name)
        if is_below:
            below_and_met.append(case.name)
        if not (_close(value, ceiling) and _close(value, case.b - case.p)):
            unequal.append((case.name, value, ceiling, case.b - case.p))
    below_one_way = [case for case in below if case.kind == "one-way"]
    counts = (
        f"{len(cases)} days; {len(met)} meet the conditions; {len(below)} are below the "
        f"ceiling ({len(below_one_way)} of them one-way); {len(below_and_met)} of those below "
        f"meet the conditions (the claim: none)"
    )
    print(counts)
    assert met, f"no day meets the conditions: {counts}"
    assert below_one_way, f"no one-way day is below the ceiling; the claim is untested: {counts}"
    assert not below_and_met, f"{counts}: {below_and_met}"
    assert not unequal, f"{counts}: {unequal}"


def test_repository_ceiling_equals_an_independent_b_minus_p():
    """Not Theorem H's content (that is the box test below). If it failed, the repository's
    ceiling `exact.hourly_ceiling`, which is by construction b(S) - p(S), differs from the most
    minus the least the fleet can draw over S with hour 0 free, computed here per car."""
    cases = _all_cases()
    kinds = {case.kind for case in cases}
    wrong = [
        (case.name, case.values["ceiling"], case.b - case.p)
        for case in cases
        if not _close(case.values["ceiling"], case.b - case.p)
    ]
    assert kinds == {"one-way", "two-way"}, kinds
    assert not wrong, f"{len(wrong)} of {len(cases)} days differ: {wrong}"


def test_lemma_4_free_and_full_power_recovery_give_the_same_sets_per_car():
    """If it failed, the plan's rule of charging at full power until full or gone leaves a car
    with other hour-0 powers, or another most or least over S, than any completion of the
    recovery hours would: then the repository's rows (which use the rule) and a free completion
    (which the proofs use) describe different sets. Both sides are computed by this file's
    routine.

    What the cases here can and cannot catch. Every car of these days is in the capped class (no
    forced draw, one cap from hour n on). For such a car the two completions are the same
    expression: the lowest stored energy after hour n is `max(floor[t] - highest powers from
    hour n + 1 to t)` over t from n to the last hour under either rule, and the highest is the
    one cap under either rule (backward, `min(cap, cap - lowest power)` is the cap because the
    lowest power is at most 0). So the equality holds by construction. The test can go red only
    if (a) a car leaves the class (the full-power branch then stops with "not a capped car"),
    (b) one branch is edited so that it no longer reduces to the same expression, or (c) the
    made-up days stop having a recovery hour or stop having a need met late (the two guards
    below). It cannot catch two swapped branches, the same mistake in both, or a gap between this
    routine and the repository's rows. A car outside the class, where the two completions must
    differ, is in
    `test_lemma_4_a_car_with_a_forced_draw_in_recovery_is_outside_the_capped_class`."""
    differ = []
    with_recovery = binding = 0
    for case in _all_cases():
        n = case.n_service
        for car in case.cars:
            free = (_profile(car, n, "free")[:2], _car_extremes(car, n, "free"))
            full = (_profile(car, n, "full power")[:2], _car_extremes(car, n, "full power"))
            if free != full:
                differ.append((case.name, free, full))
            with_recovery += car.last_hour > n
            binding += _completion(car, n, "full power")[0] > car.floor[n]
    counts = f"{with_recovery} cars with recovery hours, {binding} where the need is met late"
    assert with_recovery >= 1 and binding >= 1, counts
    assert not differ, f"{counts}: {differ}"


def _full_power_powers(limits, n_service, stored):
    """The plan's rule on the recovery hours from the stored energy `stored` after hour n: in each
    recovery hour draw `min(highest power, cap - stored)`. The powers are returned as a list."""
    powers = []
    for t in range(n_service + 1, limits.last_hour + 1):
        power = min(limits.high[t], limits.cap[t] - stored)
        powers.append(power)
        stored += power
    return powers


def test_lemma_4_a_car_with_a_forced_draw_in_recovery_is_outside_the_capped_class():
    """Written by hand, not run. If it failed, the full-power branch of `_completion` accepts a car
    that leaves the capped class, or the free branch no longer gives the completion interval of
    such a car, or the two branches are swapped (a swap leaves the test above green).

    The car (the proof's example for K1 failing, section 4): hours 0 to 3, n = 1, so hours 2 and 3
    are recovery hours. Power is 0 in hours 0 and 1. In hours 2 and 3 the power lies in [1, 2]
    (a forced draw: lowest power 1 > 0). The stored energy is at most 2 after every hour and has
    no floor (-BIG).
    Free completion, by hand, backward from hour 3: top of the interval 2 (the cap). Hour 3:
    top = min(cap, 2 - lowest power 1) = 1, bottom = -BIG. Hour 2: top = min(2, 1 - 1) = 0, bottom
    = -BIG. So the stored energy after hour 1 from which hours 2 and 3 can be finished is
    `(-BIG, 0)`: two draws of at least 1 each, in all at most 2. From stored energy 0 the
    completion (1, 1) exists (powers 1 and 1 lie in [1, 2]; stored energy 1 then 2, at most 2).
    Full power from stored energy 0: hour 2 draws min(2, 2 - 0) = 2, stored energy 2; hour 3
    draws min(2, 2 - 2) = 0, which is below the lowest power 1, so the rule's completion (2, 0)
    leaves the limits. The two completions differ: the free one has top 0, while the expression
    of the full-power branch (top = the cap of hour n) would say 2; and the powers (1, 1)
    against (2, 0).
    """
    car = Limits(
        tuple(Fraction(v) for v in (0, 0, 1, 1)),
        tuple(Fraction(v) for v in (0, 0, 2, 2)),
        (-BIG,) * 4,
        tuple(Fraction(v) for v in (2, 2, 2, 2)),
    )
    n = 1
    free = _completion(car, n, "free")
    assert free == (-BIG, 0), free
    assert free[1] != car.cap[n]  # the full-power expression would give the cap, 2
    assert (car.low[2], car.low[3], car.high[2], car.high[3]) == (1, 1, 2, 2)
    # the completion (1, 1) from stored energy 0 is within the limits
    assert all(car.low[t] <= 1 <= car.high[t] for t in (2, 3)) and car.cap[3] >= 0 + 1 + 1
    # the rule's own completion from stored energy 0 is (2, 0), and its second power is too low
    powers = _full_power_powers(car, n, Fraction(0))
    assert powers == [2, 0], powers
    assert powers[1] < car.low[3]
    with pytest.raises(AssertionError, match="not a capped car"):
        _completion(car, n, "full power")


MAX_SERVICE_HOURS = 4  # made-up days have 2 to 4 service hours, so a box has at most 16 corners


def test_theorem_h_the_box_of_full_width_lies_in_the_set():
    """Theorem H's content. If it failed, the proof's box for some car is not inside the car's
    set: a request in the box can then not be served by that car when the hour-0 power may
    depend on the request path, or the box is narrower than b - p.

    For every car of every day the box `[w, x]` is built (`_car_box`: `x` a point with the
    largest total, `w` the proof's lowest path below it). The test asserts `w <= x`, total of
    `x` equal to b and total of `w` equal to p (exact), and that EVERY corner of the box (each
    service hour at `w_t` or `x_t`; at most 2**4 = 16 corners) can be served for some hour-0
    power and some recovery completion, decided by `_servable`, which does not use how the box
    was built. A vector with a total above b must be refused (so `_servable` is not always yes).
    The fleet box is the sum of the car boxes; its width must equal the repository's ceiling.
    """
    problems = []
    checked = {"cars": 0, "corners": 0, "wider than any single hour-0 power": 0, "wide": 0}
    kinds = set()
    for case in _all_cases():
        n = case.n_service
        assert 1 <= n <= MAX_SERVICE_HOURS, case.name
        kinds.add(case.kind)
        width = Fraction(0)
        for k, car in enumerate(case.cars):
            widest, b, p = _car_extremes(car, n)
            w, x = _car_box(car, n)
            where = (case.name, k)
            if any(lo > hi for lo, hi in zip(w, x, strict=True)):
                problems.append((where, "w is above x", w, x))
            if (sum(x), sum(w)) != (b, p):
                problems.append((where, "totals are not b and p", sum(x), b, sum(w), p))
            refused = [u for u in _corners(w, x) if not _servable(car, n, u)]
            if refused:
                problems.append((where, "corners the car cannot serve", refused[:3], w, x))
            too_much = (x[0] + 1, *x[1:])
            if _servable(car, n, too_much):
                problems.append((where, "serves a total above b", too_much))
            width += sum(x) - sum(w)
            checked["cars"] += 1
            checked["corners"] += 2**n
            checked["wide"] += sum(x) > sum(w)
            checked["wider than any single hour-0 power"] += widest < b - p
        if not _close(case.values["ceiling"], float(width)):
            problems.append((case.name, "fleet box width", float(width), case.values["ceiling"]))
    assert kinds == {"one-way", "two-way"}, kinds
    assert all(checked.values()), checked
    assert not problems, f"{checked}: {problems}"


def _recovery_car(low, high, floor, cap):
    """A car with hours 0 to 2 (n = 1, hour 2 the recovery hour), from lists of numbers."""
    return Limits(*(tuple(Fraction(v) for v in seq) for seq in (low, high, floor, cap)))


HALF = Fraction(1, 2)
# Two cars with one service hour (n = 1) and one recovery hour, each with the hour-0 power fixed
# at 0. In both, a vector is refused only by the recovery hour: hours 0 and 1 alone accept it.
#
# "short of the need": powers 0 to 1 in hour 1, 0 to 1/2 in hour 2; stored energy at most 1; the
# stored energy after hour 2 must be at least 1 (floor 1). With stored energy s after hour 1, hour
# 2 raises it by at most 1/2, so s >= 1/2 is needed; the cap 1 gives s <= 1. With the hour-0 power
# 0, s = u (the hour-1 power). By hand: u = 0 gives (bottom, top) after hour 2 of
# (max(1, 0 + 0), min(1, 0 + 1/2)) = (1, 1/2), empty: refused. u = 1/4: (1, 3/4), empty: refused.
# u = 1/2: (1, 1): served. u = 1: (1, min(1, 1 + 1/2)) = (1, 1): served.
# "forced draw": powers 0 to 1 in hour 1, 1/2 to 1 in hour 2 (a forced draw), stored energy at
# most 1, no floor. s <= 1 - 1/2 = 1/2 is needed after hour 1. By hand: u = 3/4 gives (bottom, top)
# after hour 2 of (3/4 + 1/2, min(1, 3/4 + 1)) = (5/4, 1), empty: refused. u = 1: (3/2, 1): refused.
# u = 1/2: (1, 1): served. u = 1/4: (3/4, 1): served. u = 0: (1/2, 1): served.
RECOVERY_CARS = {
    "short of the need": (
        _recovery_car((0, 0, 0), (0, 1, HALF), (-BIG, -BIG, 1), (1, 1, 1)),
        [(Fraction(0),), (Fraction(1, 4),)],
        [(HALF,), (Fraction(1),)],
        (HALF, Fraction(1), HALF),  # (widest, b, p): the car can draw 1/2 to 1 over S
    ),
    "forced draw": (
        _recovery_car((0, 0, HALF), (0, 1, 1), (-BIG, -BIG, -BIG), (1, 1, 1)),
        [(Fraction(3, 4),), (Fraction(1),)],
        [(Fraction(0),), (Fraction(1, 4),), (HALF,)],
        (HALF, HALF, Fraction(0)),  # the car can draw 0 to 1/2 over S
    ),
}


@pytest.mark.parametrize("name", sorted(RECOVERY_CARS))
def test_servable_refuses_a_vector_only_because_of_a_recovery_hour(name):
    """Written by hand, not run. If it failed, `_servable` accepts a vector that no recovery
    completion can finish (it ignores the recovery hours, or their floor, cap or power limits), or
    refuses one that can be finished. Each refused vector is first shown to pass hours 0 to n (its
    start interval is not empty), so that only the recovery hour can refuse it; the vectors on the
    other side of the boundary are served. The derivations are in the comment above the cars. The
    car's own most and least over S (computed by `_car_extremes`, a different routine) give the same
    boundary: a vector is served exactly when its single power lies between the least and the most.
    """
    car, refused, served, extremes = RECOVERY_CARS[name]
    n = 1
    assert car.last_hour > n
    assert _car_extremes(car, n) == extremes, extremes
    _, b, p = extremes
    for u in refused:
        assert _start_interval(car, n, u) is not None, u
        assert not _servable(car, n, u), u
        assert not p <= u[0] <= b, u
    for u in served:
        assert _servable(car, n, u), u
        assert p <= u[0] <= b, u


# Hand days from the proofs' own examples. Each has literal expected values.


@dataclass(frozen=True, eq=False)
class Hand:
    name: str
    n_service: int
    cars: tuple  # of Limits
    rows: object  # carsets.FleetRows
    functions: object  # exact.SetFunctions
    commit: float  # V_commit as the proofs state it
    b_minus_p: float  # b(S) - p(S), which is also the hindsight value V_hind
    meets: bool  # whether Corollary C's conditions hold
    boxes: tuple  # per car, the box (w, x) as the proof text gives it
    outside: tuple  # per car, one vector of service-hour powers the car cannot serve


# Example numbers are those of the proofs: 5.4, 5.5, 5.6 (the Theorem B and Corollary C examples),
# 7.7 and 7.9 (the Theorem H examples with two service hours, one car and two cars).
HAND_NAMES = ("5.4", "5.5", "5.6", "7.7", "7.9")


def _hand_rows(spec, n_service):
    """The rows of a hand-made car over hours 0 to n, written out here."""
    low, high, stored_low, stored_high, need = spec
    hours = n_service + 1
    eye = np.eye(hours)
    prefix = np.tril(np.ones((hours, hours)))
    return CarRows(
        n_service,
        np.vstack([eye, -eye, prefix, -prefix, -np.ones((1, hours))]),
        np.concatenate(
            [
                np.full(hours, high),
                np.full(hours, -low),
                np.full(hours, stored_high),
                np.full(hours, -stored_low),
                [-need],
            ]
        ),
        ("power-high",) * hours
        + ("power-low",) * hours
        + ("stored-high",) * hours
        + ("stored-low",) * hours
        + ("need",),
    )


def _hand_true(name, day, expected):
    cars = tuple(_true_limits(car, day, None) for car in day.cars)
    functions = exact.day_functions(day, None)
    return Hand(name, day.n_service, cars, true_rows(day), functions, *expected)


def _hand_stored(name, n_service, specs, expected):
    """Cars given as (low, high, stored low, stored high, need), the same in every hour from
    0 to n, with no recovery hour."""
    hours = n_service + 1
    cars = tuple(_stored_limits(*spec, hours) for spec in specs)
    rows = stack([_hand_rows(spec, n_service) for spec in specs])
    functions = exact.subset_functions([StoredEnergyCar(hours, *s) for s in specs], n_service)
    return Hand(name, n_service, cars, rows, functions, *expected)


@functools.cache
def _hand(name):
    """Each day with (V_commit, b - p, meets Corollary C, the proof's box per car, one vector
    outside the set per car). The outside vectors: (1.5,) is above the power limit 1; (1, 1) is
    within the power limits but draws a total of 2, above the most the car can draw."""
    if name == "5.4":  # one car, power 0 to 1 in hours 0 and 1, total exactly 1, no recovery
        day = Day("made-up", DATE, 1, 0, 0.0, (TrueCar(2, 1.0, 1.0),))
        return _hand_true(name, day, (0.0, 1.0, False, (((0.0,), (1.0,)),), ((1.5,),)))
    if name == "5.5":  # the same car with one recovery hour that can supply the need
        day = Day("made-up", DATE, 1, 1, 0.0, (TrueCar(3, 1.0, 1.0),))
        return _hand_true(name, day, (1.0, 1.0, True, (((0.0,), (1.0,)),), ((1.5,),)))
    if name == "5.6":  # two-way, stored energy -1 to 0.5, hours 0 and 1
        expected = (1.5, 2.0, False, (((-1.0,), (1.0,)),), ((1.5,),))
        return _hand_stored(name, 1, [(-1.0, 1.0, -1.0, 0.5, -1.0)], expected)
    two_way = (-1.0, 1.0, 0.0, 1.0, 0.0)  # power -1 to 1, stored energy 0 to 1, hours 0 to 2
    car_a = (((-1.0, 0.0), (1.0, 0.0)),)  # the proof's box for that car
    if name == "7.7":
        return _hand_stored(name, 2, [two_way], (1.0, 2.0, False, car_a, ((1.0, 1.0),)))
    if name == "7.9":  # that car beside a one-way car that must end with exactly 1 stored
        boxes = car_a + (((0.0, 0.0), (1.0, 0.0)),)
        expected = (1.0, 3.0, False, boxes, ((1.0, 1.0), (1.0, 1.0)))
        return _hand_stored(name, 2, [two_way, (0.0, 1.0, 0.0, 1.0, 1.0)], expected)
    raise ValueError(name)


@pytest.mark.parametrize("name", HAND_NAMES)
def test_hand_days_v_commit_literals(name):
    """If it failed, either the routine that computes V_commit from the limits or the
    repository's diagonal, affine or tree value is wrong on a day whose answer is known."""
    hand = _hand(name)
    commit, _, _ = _fleet_extremes(hand.cars, hand.n_service)
    assert float(commit) == pytest.approx(hand.commit, abs=1e-12)
    values = _hourly_values(hand.rows)
    for key in ("diagonal", "causal", "tree"):
        assert values[key] == pytest.approx(hand.commit, abs=TOLERANCE), (name, key)


@pytest.mark.parametrize("name", HAND_NAMES)
def test_hand_days_hindsight_literals(name):
    """If it failed, either the routine that computes b(S) - p(S) from the limits or the
    repository's exact ceiling is wrong on a day whose answer is known."""
    hand = _hand(name)
    _, b, p = _fleet_extremes(hand.cars, hand.n_service)
    assert float(b - p) == pytest.approx(hand.b_minus_p, abs=1e-12)
    assert exact.hourly_ceiling(hand.functions) == pytest.approx(hand.b_minus_p, abs=TOLERANCE)


@pytest.mark.parametrize("name", HAND_NAMES)
def test_hand_days_corollary_conditions_literals(name):
    """If it failed, the conditions of Corollary C are classified wrongly on a day whose class is
    known: the day with a recovery hour (5.5) meets them (V_commit = b - p); the others do not
    (V_commit < b - p)."""
    hand = _hand(name)
    failures = [f for car in hand.cars for f in _corollary_failures(car, hand.n_service)]
    assert (not failures) == hand.meets, failures
    if hand.meets:
        assert hand.commit == hand.b_minus_p
    else:
        assert hand.commit < hand.b_minus_p


@pytest.mark.parametrize("name", HAND_NAMES)
def test_hand_days_boxes_literals(name):
    """If it failed, the box construction does not give the box the proof writes down, or the
    routine that decides whether a car can serve a vector accepts a corner of the proof's box
    it should accept, or accepts a vector outside the set that it should refuse (so it cannot
    pass by always answering yes)."""
    hand = _hand(name)
    n = hand.n_service
    for car, (z, x), outside in zip(hand.cars, hand.boxes, hand.outside, strict=True):
        z = tuple(Fraction(v) for v in z)
        x = tuple(Fraction(v) for v in x)
        assert _car_box(car, n) == (z, x)
        assert all(_servable(car, n, u) for u in _corners(z, x))
        assert not _servable(car, n, tuple(Fraction(v) for v in outside)), outside


def test_hand_day_7_7_hour_0_power_depends_on_the_request():
    """If it failed, the hour-0 powers that serve the two ends of the proof's box in Example 7.7
    are not the single powers 0 (request 1 in hour 1) and 1 (request -1), so the proof's claim
    that one fixed hour-0 power cannot serve the whole box is not what the limits give."""
    hand = _hand("7.7")
    (car,) = hand.cars
    top = tuple(Fraction(v) for v in (1, 0))
    bottom = tuple(Fraction(v) for v in (-1, 0))
    assert _start_interval(car, 2, top) == (0, 0)
    assert _start_interval(car, 2, bottom) == (1, 1)

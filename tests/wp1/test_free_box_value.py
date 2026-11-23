"""Plan section 2 and section 4 (T4): the free-box value is the fleet's total flexibility.

Under the definition "largest box value, with lo <= hi free in every service hour", on lossless
toy fleets of 1 to 4 cars and 2 to 4 service hours:

1. V_hind = b(S) - p(S), the fleet's upper minus lower set function over all service hours S.
2. V_diag = V_hind on every fleet, whether or not the T4 proposition's hypotheses hold.
3. A gap between the diagonal rule and hindsight appears only when the box's shape (centre and
   widths) is fixed and only its scale is free, and then only on fleets that break the hypotheses.

The T4 hypotheses are: one-way cars, no car leaving inside the service window, and for every car
recovery hours times rate at least the required energy. A fleet breaks them when some car leaves
inside the window, or some car with positive required energy has rate times recovery hours below
that energy. A change to the definitions of the values has to keep these statements true, or say
which one it gives up.
"""

import random

from toy_values import (
    diagonal_scale,
    diagonal_value,
    hindsight_scale,
    hindsight_value,
)

from fleetreserve.gpoly import Car, fleet_b, fleet_p

N_BREAKING = 300
N_TWO_WAY = 100
N_MEETING = 100
MAX_ATTEMPTS = 20000  # hard cap on generated fleets in the fixed-shape test
VALUE_TOLERANCE = 1e-7
POSITIVE_SCALE = 1e-6  # a hindsight scale at or below this is not counted as positive
GAP_TOLERANCE = 1e-6  # relative: the diagonal scale is strictly below hindsight by more than this


def _breaks_hypotheses(cars, service):
    """True when a car leaves inside the window, or its recovery cannot supply its energy."""
    for car in cars:
        if max(car.hours) < max(service):
            return True
        recovery = len(car.hours - set(service))
        if car.e_low > 0 and car.high * recovery < car.e_low:
            return True
    return False


def _one_way_car(rng, n_hours):
    """A one-way car present from hour 1 to a departure anywhere from hour 1 to n_hours + 3."""
    rate = rng.choice([1.0, 2.0, 3.0])
    departure = rng.randint(1, n_hours + 3)
    hours = frozenset(range(1, departure + 1))
    e_high = rng.uniform(0.2, 1.3) * rate * len(hours)
    e_low = e_high if rng.random() < 0.5 else rng.uniform(0.0, e_high)
    return Car(hours, 0.0, rate, min(e_low, rate * len(hours)), e_high)


def _breaking_fleet(rng):
    """A one-way fleet that breaks at least one hypothesis; returns the cars and service hours."""
    while True:
        n_hours = rng.randint(2, 4)
        service = list(range(1, n_hours + 1))
        cars = [_one_way_car(rng, n_hours) for _ in range(rng.randint(1, 4))]
        if _breaks_hypotheses(cars, service):
            return cars, service


def _meeting_fleet(rng):
    """A one-way fleet meeting every hypothesis, built as in the T4 proposition test."""
    n_hours = rng.randint(2, 4)
    service = list(range(1, n_hours + 1))
    cars = []
    for _ in range(rng.randint(1, 4)):
        rate = rng.choice([1.0, 2.0, 3.0])
        room = rng.uniform(0.2, 1.3) * rate * n_hours
        recovery = -(-room // rate) + rng.randint(0, 1)  # ceil(room / rate), possibly one more
        hours = frozenset(range(1, n_hours + int(recovery) + 1))
        cars.append(Car(hours, 0.0, rate, room, room))
    return cars, service


def _two_way_fleet(rng):
    """A fleet of two-way cars (charge or discharge), departures anywhere in or after the window."""
    n_hours = rng.randint(2, 4)
    service = list(range(1, n_hours + 1))
    cars = []
    for _ in range(rng.randint(1, 4)):
        low = -rng.choice([1.0, 2.0])
        high = rng.choice([1.0, 2.0, 3.0])
        hours = frozenset(range(1, rng.randint(1, n_hours + 3) + 1))
        first, second = sorted(rng.uniform(low * len(hours), high * len(hours)) for _ in range(2))
        if rng.random() < 0.3:
            second = first
        cars.append(Car(hours, low, high, first, second))
    return cars, service


def _free_box_differences(cars, service):
    """The two differences from total flexibility: hindsight and diagonal, as absolute values."""
    flexibility = fleet_b(cars, service) - fleet_p(cars, service)
    return (
        abs(hindsight_value(cars, service) - flexibility),
        abs(diagonal_value(cars, service) - flexibility),
    )


def _check_free_box(label, fleets):
    worst_hind = worst_diag = 0.0
    failures = []
    for cars, service in fleets:
        d_hind, d_diag = _free_box_differences(cars, service)
        worst_hind = max(worst_hind, d_hind)
        worst_diag = max(worst_diag, d_diag)
        if max(d_hind, d_diag) > VALUE_TOLERANCE:
            failures.append((cars, service, d_hind, d_diag))
    print(
        f"{label}: {len(fleets) - len(failures)} of {len(fleets)} fleets equal; largest "
        f"|V_hind - (b - p)| {worst_hind:.2e}, largest |V_diag - (b - p)| {worst_diag:.2e}"
    )
    assert not failures, f"{len(failures)} fleets differ; all of them: {failures}"


def test_free_box_hypotheses_broken_one_way():
    rng = random.Random(0)
    fleets = [_breaking_fleet(rng) for _ in range(N_BREAKING)]
    assert all(_breaks_hypotheses(cars, service) for cars, service in fleets)
    assert all(all(car.low == 0.0 for car in cars) for cars, _ in fleets)  # one-way
    _check_free_box("free box, hypotheses broken, one-way", fleets)


def test_free_box_two_way():
    rng = random.Random(1)
    fleets = [_two_way_fleet(rng) for _ in range(N_TWO_WAY)]
    assert all(any(car.low < 0 for car in cars) for cars, _ in fleets)
    _check_free_box("free box, two-way", fleets)


def _point_in_fleet_set(rng, cars, service):
    """A random point of the fleet set on the service hours (a feasible plan of every car)."""
    centre = dict.fromkeys(service, 0.0)
    for car in cars:
        hours = sorted(car.hours)
        remaining = rng.uniform(car.e_low, min(car.e_high, car.high * len(hours)))
        rng.shuffle(hours)
        for left, t in zip(range(len(hours) - 1, -1, -1), hours, strict=True):
            lo = max(car.low, remaining - car.high * left)
            hi = min(car.high, remaining - car.low * left)
            power = rng.uniform(lo, hi) if hi > lo else lo
            remaining -= power
            if t in centre:
                centre[t] += power
    return centre


def _shape_cases(rng, make_fleet, count):
    """Generate fleets with a random fixed shape until `count` have a positive hindsight scale.

    Returns (cars, service, centre, widths, hindsight scale, diagonal scale) for each fleet.
    """
    cases = []
    for _ in range(MAX_ATTEMPTS):
        cars, service = make_fleet(rng)
        centre = _point_in_fleet_set(rng, cars, service)
        widths = {t: rng.uniform(0.1, 1.0) for t in service}
        scale = hindsight_scale(cars, service, centre, widths)
        if scale > POSITIVE_SCALE:
            cases.append(
                (
                    cars,
                    service,
                    centre,
                    widths,
                    scale,
                    diagonal_scale(cars, service, centre, widths),
                )
            )
            if len(cases) == count:
                return cases
    raise AssertionError(f"only {len(cases)} of {count} positive-scale fleets in {MAX_ATTEMPTS}")


def test_fixed_shape():
    rng = random.Random(2)
    breaking = _shape_cases(rng, _breaking_fleet, N_BREAKING)
    above = [case for case in breaking if case[5] > case[4] + VALUE_TOLERANCE]
    gaps = [case for case in breaking if case[4] - case[5] > GAP_TOLERANCE * case[4]]
    largest = max((case[4] - case[5]) / case[4] for case in breaking)
    print(
        f"fixed shape, hypotheses broken: {len(gaps)} of {len(breaking)} fleets have the diagonal "
        f"scale strictly below hindsight; largest relative gap {largest:.4f}"
    )
    assert not above, f"{len(above)} fleets have a diagonal scale above hindsight: {above}"
    assert len(gaps) >= 1
    assert len(gaps) < len(breaking) / 2

    meeting = _shape_cases(rng, _meeting_fleet, N_MEETING)
    unequal = [case for case in meeting if abs(case[4] - case[5]) > GAP_TOLERANCE * case[4]]
    worst = max(abs(case[4] - case[5]) / case[4] for case in meeting)
    print(
        f"fixed shape, hypotheses met: {len(meeting) - len(unequal)} of {len(meeting)} fleets "
        f"have equal scales; largest relative difference {worst:.2e}"
    )
    assert not unequal, f"{len(unequal)} fleets with the hypotheses have unequal scales: {unequal}"

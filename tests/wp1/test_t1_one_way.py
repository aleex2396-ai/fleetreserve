"""Plan section 4, T1, one-way: the fast box test agrees with the vertex programmes.

Random one-way fleets (1 to 4 cars, 6 hours, each car present in one run of hours, rate 1, 2 or
3 kW, energy limits inside what the rate allows) and random boxes near the fleet's scale. On each
instance the fast test (`box_inside`, every subset of hours) must give the same answer as the
vertex programmes (`vertex_truth.py`). Required: 400 of 400, with both answers present (each at
least 5% of the sample).
"""

import random

from vertex_truth import box_inside_by_vertices

from fleetreserve.gpoly import Car, box_inside

N_INSTANCES = 400
N_HOURS = 6
MIN_SHARE = 0.05  # each answer (inside, outside) must make up at least this share of the sample


def _instance(rng):
    cars = []
    for _ in range(rng.randint(1, 4)):
        start = rng.randint(0, N_HOURS - 1)
        end = rng.randint(start + 1, N_HOURS)
        rate = rng.choice([1, 2, 3])
        e_high = rng.uniform(0.3, 1.0) * rate * (end - start)
        e_low = rng.uniform(0, 1) * e_high
        cars.append(Car(frozenset(range(start, end)), 0, rate, e_low, e_high))
    present = [sum(c.high for c in cars if t in c.hours) for t in range(N_HOURS)]
    hi = [rng.uniform(0, 0.6 * present[t]) for t in range(N_HOURS)]
    lo = [rng.uniform(0, h) for h in hi]
    return cars, lo, hi


def test_box_check_agrees_with_vertex_programmes_on_400_of_400():
    rng = random.Random(0)
    agree = inside = 0
    disagreements = []
    for _ in range(N_INSTANCES):
        cars, lo, hi = _instance(rng)
        truth = box_inside_by_vertices(lo, hi, cars, N_HOURS)
        fast = box_inside(lo, hi, cars, range(N_HOURS))
        inside += truth
        if truth == fast:
            agree += 1
        else:
            disagreements.append((truth, fast, cars, lo, hi))
    print(
        f"one-way: agree {agree} of {N_INSTANCES}; inside {inside}, outside {N_INSTANCES - inside}"
    )
    assert not disagreements, (
        f"agree {agree} of {N_INSTANCES}; first disagreements: {disagreements[:3]}"
    )
    assert agree == N_INSTANCES
    assert MIN_SHARE * N_INSTANCES <= inside <= (1 - MIN_SHARE) * N_INSTANCES, (
        f"sample too one-sided: {inside} inside, {N_INSTANCES - inside} outside"
    )

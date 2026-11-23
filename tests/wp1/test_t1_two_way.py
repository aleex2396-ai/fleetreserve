"""Plan section 4, T1, two-way: the fast box test agrees with the vertex programmes.

Random two-way fleets (1 to 3 cars, 6 hours, power between minus and plus the rate, net energy
limits that may be negative) and random boxes, some with negative values. On each instance the
fast test (`box_inside`, every subset of hours, with the shift of plan section 9) must give the
same answer as the vertex programmes (`vertex_truth.py`). Required: 300 of 300, with both
answers present (each at least 5% of the sample).
"""

import random

from vertex_truth import box_inside_by_vertices

from fleetreserve.gpoly import Car, box_inside

N_INSTANCES = 300
N_HOURS = 6
MIN_SHARE = 0.05  # each answer (inside, outside) must make up at least this share of the sample


def _instance(rng):
    cars = []
    for _ in range(rng.randint(1, 3)):
        start = rng.randint(0, N_HOURS - 1)
        end = rng.randint(start + 1, N_HOURS)
        rate = rng.choice([1, 2])
        e_high = rng.uniform(0, 1) * rate * (end - start)
        e_low = rng.uniform(-0.5, 1) * e_high
        cars.append(Car(frozenset(range(start, end)), -rate, rate, e_low, e_high))
    present = [sum(c.high for c in cars if t in c.hours) for t in range(N_HOURS)]
    hi = [rng.uniform(-0.5, 0.7) * present[t] for t in range(N_HOURS)]
    lo = [h - rng.uniform(0, 0.6) * present[t] for t, h in enumerate(hi)]
    return cars, lo, hi


def test_box_check_agrees_with_vertex_programmes_on_300_of_300():
    rng = random.Random(1)
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
        f"two-way: agree {agree} of {N_INSTANCES}; inside {inside}, outside {N_INSTANCES - inside}"
    )
    assert not disagreements, (
        f"agree {agree} of {N_INSTANCES}; first disagreements: {disagreements[:3]}"
    )
    assert agree == N_INSTANCES
    assert MIN_SHARE * N_INSTANCES <= inside <= (1 - MIN_SHARE) * N_INSTANCES, (
        f"sample too one-sided: {inside} inside, {N_INSTANCES - inside} outside"
    )

"""Plan section 4, T1, floors and free recovery hours: the box test agrees with the vertex LPs.

Random fleets of 1 to 4 cars, one-way and two-way mixed, over 7 hours of which the first 4 are
the service hours S (hours 4 to 6 are the recovery hours). Every car has a positive lower energy
(the protected energy of plan section 1), and a car's run of hours may start in S and reach
into the recovery hours, or lie wholly in the recovery hours. The box is on S only.

The fast test is `box_inside(lo, hi, cars, S)`: its set functions count a car's hours outside S
as available. The ground truth (`vertex_truth.py`) takes every vertex of the box and asks a
linear programme whether the cars can split it, with the recovery-hour powers free within each
car's own limits. Required: 600 of 600, with between a quarter and three quarters of the boxes
inside.

Each box is drawn around a point of the fleet set (a feasible split, mixed from random powers;
see `vertex_truth.py`), with half-widths of a random share, up to `MAX_SHARE`, of the fleet's
power range in that hour, so that small boxes tend to be inside and large ones outside.
"""

import random

from vertex_truth import N_HOURS, box_inside_by_vertices, centre_and_spread, random_car

from fleetreserve.gpoly import box_inside

N_INSTANCES = 600
SERVICE = range(4)  # hours 0 to 3; hours 4 to 6 are recovery hours
MAX_SHARE = 0.5  # half-width of a box side, as a share of the hour's total power range
MIN_INSIDE, MAX_INSIDE = 0.25, 0.75  # share of the sample that must be inside


def _instance(rng):
    cars = [random_car(rng, N_HOURS) for _ in range(rng.randint(1, 4))]
    centre, spread = centre_and_spread(rng, cars, SERVICE)
    scale = rng.uniform(0, MAX_SHARE)
    lo = [centre[t] - scale * rng.uniform(0, 1) * spread[t] for t in SERVICE]
    hi = [centre[t] + scale * rng.uniform(0, 1) * spread[t] for t in SERVICE]
    return cars, lo, hi


def test_box_check_with_floors_and_free_recovery_hours_agrees_on_600_of_600():
    rng = random.Random(2)
    agree = inside = 0
    disagreements = []
    for _ in range(N_INSTANCES):
        cars, lo, hi = _instance(rng)
        truth = box_inside_by_vertices(lo, hi, cars, N_HOURS, SERVICE)
        fast = box_inside(lo, hi, cars, SERVICE)
        inside += truth
        if truth == fast:
            agree += 1
        else:
            disagreements.append((truth, fast, cars, lo, hi))
    print(
        f"floors and recovery: agree {agree} of {N_INSTANCES}; "
        f"inside {inside}, outside {N_INSTANCES - inside}"
    )
    assert not disagreements, (
        f"agree {agree} of {N_INSTANCES}; first disagreements: {disagreements[:3]}"
    )
    assert agree == N_INSTANCES
    assert MIN_INSIDE * N_INSTANCES <= inside <= MAX_INSIDE * N_INSTANCES, (
        f"sample not balanced: {inside} inside, {N_INSTANCES - inside} outside"
    )

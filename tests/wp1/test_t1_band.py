"""Plan section 4, T1(c), band scale: the closed form agrees with linear programmes.

Random fleets of 1 to 4 cars (one- and two-way, positive floors, runs of hours that may reach
two recovery hours after the service hours S), 3 to 5 service hours, a centre inside the fleet
set and random widths (up, down), some of them zero. The band is `{centre + d}` with
`-lam * down <= d <= lam * up` hour by hour and `sum(d) = 0` over S.

The closed form is `band_scale`. The ground truth is independent of it: list the band's
vertices at `lam = 1` (choose one hour to be free, put every other hour at one of its two limits,
let the free hour take the balance, keep it if within its limits), then for each vertex `v` one
linear programme gives the largest `lam` with `centre + lam * v` splittable among the cars
(`vertex_truth.largest_scale_along`). The band's vertices scale linearly with `lam` and the fleet
set is convex and holds the centre, so the largest scale is the smallest of these.

Required: 400 of 400 within a relative tolerance of 1e-6 (absolute 1e-9), with the binding term
of the numerator's `min` being `b(A)` in at least a tenth and `c(S) - p(S - A)` in at least a tenth
of the instances. The binding term is read from the closed form taken with that one term only:
the instance's scale is the smaller of the two one-term scales.
"""

import itertools
import math
import random

from vertex_truth import centre_and_spread, largest_scale_along, random_car

from fleetreserve.gpoly import band_scale, fleet_b, fleet_p

N_INSTANCES = 400
N_EXTRA = 2  # recovery hours after the service hours
REL_TOL, ABS_TOL = 1e-6, 1e-9
MIN_BINDING_SHARE = 0.1  # each term of the numerator must bind in at least this share


def _instance(rng):
    n_service = rng.randint(3, 5)
    n_hours = n_service + N_EXTRA
    service = list(range(n_service))
    while True:
        cars = [random_car(rng, n_hours) for _ in range(rng.randint(1, 4))]
        centre, spread = centre_and_spread(rng, cars, service)
        active = [t for t in service if spread[t] > 0]  # hours where some car can draw
        if len(active) >= 2:
            break
    up = {t: 0.0 for t in service}
    down = {t: 0.0 for t in service}
    for t in active:
        # The first two active hours keep both widths positive, so the band is more than a point.
        keep = t in active[:2]
        up[t] = 0.0 if not keep and rng.random() < 0.15 else rng.uniform(0.1, 1.0) * spread[t]
        down[t] = 0.0 if not keep and rng.random() < 0.15 else rng.uniform(0.1, 1.0) * spread[t]
    return cars, n_hours, service, centre, up, down


def _band_vertices(service, up, down):
    """Directions `v` of the band's vertices at scale 1 (some repeats; none missing)."""
    found = set()
    for free in service:
        others = [t for t in service if t != free]
        for limits in itertools.product((0, 1), repeat=len(others)):
            d = {t: (up[t] if side else -down[t]) for t, side in zip(others, limits, strict=True)}
            d[free] = -sum(d.values())
            if -down[free] - 1e-12 <= d[free] <= up[free] + 1e-12:
                found.add(tuple(round(d[t], 12) for t in service))
    return [v for v in found if any(v)]


def _truth(cars, n_hours, service, centre, up, down):
    start = [centre[t] for t in service]
    return min(
        largest_scale_along(start, v, cars, n_hours, service)
        for v in _band_vertices(service, up, down)
    )


def _one_term_scales(cars, service, centre, up, down):
    """The closed form with only `b(A)`, and with only `c(S) - p(S - A)`, in the numerator."""
    everything = frozenset(service)
    total = sum(centre[t] for t in service)
    by_b = by_p = math.inf
    for size in range(1, len(service)):
        for chosen in itertools.combinations(service, size):
            a, rest = frozenset(chosen), everything - frozenset(chosen)
            width = min(sum(up[t] for t in a), sum(down[t] for t in rest))
            if width > 0:
                c_a = sum(centre[t] for t in a)
                by_b = min(by_b, (fleet_b(cars, a) - c_a) / width)
                by_p = min(by_p, (total - fleet_p(cars, rest) - c_a) / width)
    return by_b, by_p


def test_band_scale_agrees_with_vertex_programmes_on_400_of_400():
    rng = random.Random(3)
    agree = b_binds = p_binds = tie = one_term_mismatch = 0
    disagreements = []
    for _ in range(N_INSTANCES):
        cars, n_hours, service, centre, up, down = _instance(rng)
        truth = _truth(cars, n_hours, service, centre, up, down)
        fast = band_scale(centre, up, down, cars, service)
        by_b, by_p = _one_term_scales(cars, service, centre, up, down)
        if not math.isclose(min(by_b, by_p), fast, rel_tol=1e-12, abs_tol=1e-12):
            one_term_mismatch += 1
        if math.isclose(by_b, by_p, rel_tol=1e-9, abs_tol=1e-12):
            tie += 1
        elif by_b < by_p:
            b_binds += 1
        else:
            p_binds += 1
        if math.isclose(truth, fast, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            agree += 1
        else:
            disagreements.append((truth, fast, cars, centre, up, down))
    print(
        f"band: agree {agree} of {N_INSTANCES}; b(A) binds {b_binds}, "
        f"c(S) - p(S - A) binds {p_binds}, tie {tie}"
    )
    assert not disagreements, (
        f"agree {agree} of {N_INSTANCES}; first disagreements: {disagreements[:3]}"
    )
    assert agree == N_INSTANCES
    assert one_term_mismatch == 0, "the two one-term scales do not give the closed form's minimum"
    assert b_binds >= MIN_BINDING_SHARE * N_INSTANCES, f"b(A) binds only {b_binds}"
    assert p_binds >= MIN_BINDING_SHARE * N_INSTANCES, f"c(S) - p(S - A) binds only {p_binds}"

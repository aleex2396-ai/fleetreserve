"""T8's check (plan section 7): `soft_scale_check`'s measuring rule handed deliberately wrong
gradients, each judged against the true derivative from exact arithmetic, on the hard fleets of
the review of 2078c32 (its section D; the fleets of its `fleets.py` and `ref_rule.py`).

The fleets (`_fleet`), each seeded from its scenario's name and a seed: `generic:3` and
`generic:30` (3 or 30 cars with fractional departures, a share of tight, empty and cap-binding
cars), `whole:30` (departures on whole hours), `start:30` (departures at the start of a service
hour, a fifth of the cars empty), `kink1:30` (one car moved so that a kink of the smallest term
lies at, inside or just outside the first step), `kink2:30` (two cars moved so that kinks lie on
both sides of the point within the first step: the review's corner C1), `spread:30` (direction
coordinates spread over seven orders of magnitude) and `big:30` (energies and widths times 1e4);
1 to 4 service hours, 0 to 2 recovery hours. `SEEDS` (12) seeds per scenario, plus the six fleets
of `FOUND`: four `kink2:30` seeds on which the review found most decisions against the guarantee
(corner C1), and the two on which a kink too small to show left the soft minimum's allowance too
small (corner C2: `spread:30` seed 39, `big:30` seed 7); 102 fleets, 10,780 planted comparisons
in all, about 25 s on one core. Per fleet, field (departure, cap, need,
rate) and function (the exact scale and the soft minimum at `SIGMA`), one seeded unit direction
`v` and one measurement (`soft_scale_check.measure`, which never reads the gradient).

The referee (`_truth`), independent of `soft_scale`: every term in rational arithmetic
(`fractions.Fraction`) from T1's set functions; each term's one-sided slopes along `v` from a
rational step of 1e-40 (each term is piecewise linear along a line in one field, and a float
point lies further than that from any kink but one at the point itself); the exact scale's
derivative where one term is smallest and its slopes agree; the soft minimum's, the sum of the
slopes under the weights `softmax(-T / sigma)` in 60-digit decimal arithmetic, where every term's
slopes agree; else none. The per-car part of a derivative is computed the same way.

The gradients handed to `decide`, where the derivative `g` exists and is not 0: `g (1 +- x)`
("all") and `g +- x c_j` with `c_j` the largest car part ("one"), for `x` in `PLANTED` (100% to
0.01%). Asserted on every one, as the measuring rule guarantees (module text, part 3: the
interval holds the derivative at every step that decides): none with a true relative error above
the target (1e-3) passes; none with one of at most 0.499 times the target fails; and one of at
most the target fails only at a resolvable step (where one not shown within the target fails),
never because the interval excludes it. True errors within `EDGE` (1e-12, relatively) of the
target are left out of the first and third: the planted 0.1% lands there, and the rule's
comparisons in double precision decide such a tie either way (the review's N3). The sample is
not empty in any scenario. The counts are printed (`pytest -s`). Before the fixes of findings 2
and 3 of that review, 104 of the 10,780 planted comparisons went against this, all on the six
fleets of `FOUND` (gradients within the target failed: 76 on `kink2`, 28 on the C2 fleets); the
review's infinitely wrong gradient that passed is a hand case in `test_soft_scale_measure.py`.
"""

import math
import random
import zlib
from collections import Counter
from decimal import Decimal, getcontext
from fractions import Fraction

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import torch

from fleetreserve import soft_scale
from fleetreserve import soft_scale_check as check

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SCENARIOS = (
    "generic:3",
    "generic:30",
    "whole:30",
    "start:30",
    "kink1:30",
    "kink2:30",
    "spread:30",
    "big:30",
)
SEEDS = 12
# fleets on which the review found the allowance too small: its corner C1 (kinks of opposite
# convexity on both sides of the point) and its corner C2 (a kink too small to show)
FOUND = (
    ("kink2:30", 33),
    ("kink2:30", 55),
    ("kink2:30", 65),
    ("kink2:30", 86),
    ("spread:30", 39),
    ("big:30", 7),
)
PLANTED = (1.0, 0.1, 0.01, 0.002, 0.001, 0.0005, 0.0001)
EDGE = Decimal("1e-12")  # true errors this close to the target, relatively, are not judged
DELTA = Fraction(1, 10**40)
FIELDS = ("departure", "rate", "cap", "need")


def _share(d, t):
    return min(max(d - t, 0.0), 1.0)


def _car(rng, n, n_rec, *, whole=False, at_start=False):
    last = 1 + n + n_rec
    if at_start:
        d = float(rng.randint(1, n))
    elif whole:
        d = float(rng.randint(1, last))
    else:
        d = rng.uniform(0.3, last)
    r = rng.uniform(1.0, 7.0)
    kind = rng.choices(["drawn", "tight", "empty", "capbind"], [6, 1, 1, 1])[0]
    if kind == "empty":
        return dict(departure=d, rate=r, cap=0.0, need=0.0)
    cap = rng.uniform(0.0, 1.2) * r * d
    if kind == "capbind":
        cap = r * rng.randint(0, max(1, int(d)))
    supply = r * sum(_share(d, t) for t in range(n + 1, n + 1 + n_rec))
    most = min(cap + supply, r * d)
    need = most if kind == "tight" else rng.uniform(0.0, 1.0) * most
    while need > 0 and (need > cap + supply or need > r * d):
        need = math.nextafter(need, 0.0)
    return dict(departure=d, rate=r, cap=cap, need=need)


def _plan(car, n, n_rec, rng):
    d, r = car["departure"], car["rate"]
    high = [r * _share(d, t) for t in range(n + 1)]
    reach = sum(high)
    supply = r * sum(_share(d, t) for t in range(n + 1, n + 1 + n_rec))
    least = max(car["need"] - supply, 0.0)
    most = min(car["cap"], reach)
    if reach <= 0 or most <= 0:
        return [0.0] * n
    alpha = rng.uniform(least, most) / reach * 0.999 + 0.0005 * (least + most) / reach
    alpha = min(max(alpha, least / reach), most / reach)
    return [h * alpha for h in high[1:]]


def _fleet(scenario: str, seed: int):
    """(rng, cars, centre, up, down, n) of one made-up fleet (module text)."""
    rng = random.Random(seed * 7919 + zlib.crc32(scenario.encode()))
    name, size = scenario.split(":")
    n = rng.randint(1, 4)
    options = {"whole": {"whole": True}, "start": {"at_start": True}}.get(name, {})
    n_rec = rng.randint(0, 2)
    cars = [_car(rng, n, n_rec, **options) for _ in range(int(size))]
    centre = [0.0] * n
    for car in cars:
        centre = [a + b for a, b in zip(centre, _plan(car, n, n_rec, rng), strict=True)]
    while True:
        up = [0.0 if rng.random() < 0.2 else rng.random() for _ in range(n)]
        down = [0.0 if rng.random() < 0.2 else rng.random() for _ in range(n)]
        if any(up) or any(down):
            break
    scale = max(1.0, sum(centre) / max(n, 1)) * rng.choice([0.05, 0.3, 1.0])
    up, down = [w * scale for w in up], [w * scale for w in down]
    if name == "start":
        for car in cars[: max(1, len(cars) // 5)]:
            car["cap"] = car["need"] = 0.0
    if name == "big":
        for car in cars:
            for key in ("rate", "cap", "need"):
                car[key] *= 1e4
        centre, up, down = ([x * 1e4 for x in values] for values in (centre, up, down))
    return rng, cars, centre, up, down, n


def _direction(rng, size: int, scenario: str) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(size)]
    if scenario.startswith("spread"):
        v = [x * 10 ** rng.uniform(-7, 0) for x in v]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _bounds(cars) -> soft_scale.Bounds:
    def column(key):
        return torch.tensor([c[key] for c in cars], dtype=torch.float64)

    return soft_scale.Bounds(column("departure"), column("rate"), column("cap"), column("need"))


def _shape(centre, up, down):
    return tuple(torch.tensor(x, dtype=torch.float64) for x in (centre, up, down))


def _place_kinks(rng, cars, shape_, n, field, v, kind, scenario):
    """Move one car (`kink1`) or two (`kink2`, one each side of the point) so that a kink of the
    smallest term lies near the point along `v`."""
    k = int(torch.argmin(soft_scale.terms(_bounds(cars), *shape_)))
    mask, upper = k % (2**n), k < 2**n
    h = check.STEPS[kind]
    count = 2 if scenario.startswith("kink2") else 1
    choices = [0.0, 0.3, -0.3, 0.9, -0.9, 0.999, -0.999, 1.001, -1.001, 0.5, -0.5, 1e-3, -1e-3]
    order = list(range(len(cars)))
    rng.shuffle(order)
    used = 0
    for j in order:
        if used == count:
            break
        if abs(v[j]) < 1e-3:
            continue
        car = cars[j]
        d, r = car["departure"], car["rate"]
        s = sum(_share(d, t) for t in range(1, n + 1) if mask >> (t - 1) & 1)
        t0 = rng.choice(choices) * h
        if count == 2:
            t0 = (0.6 if used == 0 else -0.6) * h * rng.uniform(0.5, 1.0)
        if field == "cap" and upper:
            car["cap"] = r * s - t0 * v[j]
        elif field == "need" and not upper:
            car["need"] = max(r * (d - s) - t0 * v[j], 0.0)
        elif field == "departure":
            car["departure"] = rng.randint(1, n + 1) - t0 * v[j]
        elif field == "rate" and upper and s > 0:
            car["rate"] = car["cap"] / s - t0 * v[j]
            if car["rate"] <= 0:
                continue
        else:
            continue
        car["cap"] = max(car["cap"], 0.0)
        most = min(car["cap"], car["rate"] * car["departure"]) + 0.0
        if car["need"] > most:
            car["need"] = max(most * 0.999, 0.0)
        used += 1


def _car_parts(car, n):
    """`(p(A), b(A))` of one car for every subset, exact (T1's set functions)."""
    d, r, cap, need = (car[k] for k in ("departure", "rate", "cap", "need"))
    shares = [min(max(d - t, Fraction(0)), Fraction(1)) for t in range(1, n + 1)]
    lows, ups = [], []
    for mask in range(2**n):
        s = sum((shares[k] for k in range(n) if mask >> k & 1), Fraction(0))
        ups.append(min(r * s, cap))
        lows.append(max(need - r * (d - s), Fraction(0)))
    return lows, ups


def _terms(cars, sums, n):
    """Every term in `soft_scale.terms`' order (None when left out), exact."""
    c, w_up, w_down = sums
    lows, ups = [Fraction(0)] * 2**n, [Fraction(0)] * 2**n
    for car in cars:
        low, up = _car_parts(car, n)
        lows = [a + b for a, b in zip(lows, low, strict=True)]
        ups = [a + b for a, b in zip(ups, up, strict=True)]
    above = [(ups[m] - c[m]) / w_up[m] if w_up[m] > 0 else None for m in range(2**n)]
    below = [(c[m] - lows[m]) / w_down[m] if w_down[m] > 0 else None for m in range(2**n)]
    return above + below


def _moved(cars, field, v, t):
    return [dict(car, **{field: car[field] + t * x}) for car, x in zip(cars, v, strict=True)]


def _slopes(cars, sums, n, field, v, only=None):
    """(terms at 0, right slopes, left slopes), exact, of the fleet or of one car's part."""
    if only is not None:
        cars, v = [cars[only]], [v[only]]
    base = _terms(cars, sums, n)
    ahead = _terms(_moved(cars, field, v, DELTA), sums, n)
    behind = _terms(_moved(cars, field, v, -DELTA), sums, n)
    right = [None if a is None else (b - a) / DELTA for a, b in zip(base, ahead, strict=True)]
    left = [None if a is None else (a - b) / DELTA for a, b in zip(base, behind, strict=True)]
    return base, right, left


def _decimal(q: Fraction) -> Decimal:
    return Decimal(q.numerator) / Decimal(q.denominator)


def _weights(values, finite) -> dict:
    sigma = Decimal(check.SIGMA)
    low = min(_decimal(values[k]) for k in finite)
    raw = {k: (-(_decimal(values[k]) - low) / sigma).exp() for k in finite}
    total = sum(raw.values())
    return {k: w / total for k, w in raw.items()}


def _truth(kind, cars, sums, n, field, v):
    """(the derivative along `v`, the largest car part of it), both None where it does not
    exist (module text)."""
    base, right, left = _slopes(cars, sums, n, field, v)
    finite = [k for k, t in enumerate(base) if t is not None]
    if kind == "exact":
        least = min(base[k] for k in finite)
        at = [k for k in finite if base[k] == least]
        if len(at) > 1 or right[at[0]] != left[at[0]]:
            return None, None

        def combine(slopes):
            return _decimal(slopes[at[0]])

    else:
        if any(right[k] != left[k] for k in finite):
            return None, None
        weights = _weights(base, finite)

        def combine(slopes):
            return sum((weights[k] * _decimal(slopes[k]) for k in finite), Decimal(0))

    g = combine(right)
    parts = [combine(_slopes(cars, sums, n, field, v, only=j)[1]) for j in range(len(cars))]
    return g, max(parts, key=abs)


def _comparisons(scenario: str, seed: int):
    """[(kind, field, true relative error, outcome)] of every planted gradient on one fleet."""
    rng, cars, centre, up, down, n = _fleet(scenario, seed)
    found = []
    for field in FIELDS:
        v = _direction(rng, len(cars), scenario)
        for kind in check.KINDS:
            moved = [dict(car) for car in cars]
            shape_ = _shape(centre, up, down)
            if scenario.startswith("kink"):
                _place_kinks(rng, moved, shape_, n, field, v, kind, scenario)
            bounds = _bounds(moved)
            if kind == "exact" and check.smallest_terms(bounds, shape_) > 1:
                continue
            exact = [{k: Fraction(x) for k, x in car.items()} for car in moved]
            sums = [
                [
                    sum((Fraction(w[k]) for k in range(n) if m >> k & 1), Fraction(0))
                    for m in range(2**n)
                ]
                for w in (centre, up, down)
            ]
            g, part = _truth(kind, exact, sums, n, field, [Fraction(x) for x in v])
            if g is None or g == 0:
                continue
            steps = check.measure(kind, bounds, field, torch.tensor(v, dtype=torch.float64), shape_)
            for x in PLANTED:
                for sign in (1, -1):
                    for handed in (g * (1 + sign * Decimal(x)), g + sign * Decimal(x) * part):
                        a = float(handed)
                        error = abs(Decimal(a) - g) / abs(g)
                        outcome, index = check.decide(a, steps)
                        resolvable = index is not None and steps[index]["resolvable"]
                        found.append((kind, field, error, outcome, resolvable))
    return found


def test_planted_gradients_are_judged_by_the_true_derivative():
    getcontext().prec = 60
    tally = Counter()
    wrong = []
    jobs = [(s, seed) for s in SCENARIOS for seed in range(SEEDS)] + list(FOUND)
    limit = Decimal(check.LIMIT)
    for scenario, seed in jobs:
        for kind, field, error, outcome, resolvable in _comparisons(scenario, seed):
            edge = abs(error - limit) <= EDGE * limit
            above = error > limit
            side = "at the edge" if edge else "above" if above else "within"
            tally[(scenario, side, outcome)] += 1
            where = (scenario, seed, kind, field, float(error), outcome, resolvable)
            if outcome == "passed" and above and not edge:
                wrong.append(where)
            if outcome == "failed" and error <= Decimal("0.499") * limit:
                wrong.append(where)
            if outcome == "failed" and not above and not edge and not resolvable:
                wrong.append(where)
    for key in sorted(tally):
        print(key, tally[key])
    print(f"{len(jobs)} fleets, {sum(tally.values())} planted comparisons")
    assert not wrong, wrong[:10]
    for scenario in SCENARIOS:
        decided = sum(
            tally[(scenario, side, outcome)]
            for side in ("above", "within")
            for outcome in ("passed", "failed")
        )
        assert decided > 0, scenario

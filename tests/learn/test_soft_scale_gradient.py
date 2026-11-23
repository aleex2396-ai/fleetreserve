"""Plan section 7, T8's check on made-up days: "autograd matches finite differences on 100 days to
1e-3 relative error, departure and energy directions".

Fixed before the first run: seed `SEED`, 100 days of `soft_scale_fake.forecast_day(generic=True)`
(1 to 4 service hours, fractional departures, no car whose limits sit on a kink by construction),
double precision throughout. What is compared, per day and per direction: the derivative along one
seeded random unit direction over the day's cars in one field of the bounds (departure; the energy
cap and the protected need, the two energy directions; and the rate), from autograd and from
finite differences, for the soft minimum at `soft_scale_check.SIGMA` and for the hard minimum
(`soft_scale.exact_scale`), by `soft_scale_check.compare` (its measuring rule: module text).

Pass: no comparison fails by the rule; in every direction at least 5 of the 100 soft-minimum
comparisons are resolvable, so the check is not passed by derivatives it cannot resolve (on many
made-up days the smallest terms are lower terms with `p = 0`, which do not move with the cap or
the need). An earlier form of this test measured with a rounding floor that a review found too
large to catch a wrong gradient; the seed and days are unchanged.

At the kinks (the shares at a whole-hour departure, the `max` with 0 in `p`, the `min` with the
cap in `b`), autograd gives the mean of the two one-sided derivatives, exactly, where one kink is
met; where two coincide it does not (`soft_scale`'s module text and the last tests here).
"""

import random

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import pytest
import torch
from soft_scale_fake import forecast_day, point_of, shape

from fleetreserve import soft_scale, soft_scale_check

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SEED = 2611
N_DAYS = 100
FIELDS = ("departure", "cap", "need", "rate")


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def test_autograd_matches_central_differences_on_100_days():
    rng = random.Random(SEED)
    generator = torch.Generator().manual_seed(SEED)
    outcomes, failures = {}, []
    for day_index in range(N_DAYS):
        day = forecast_day(rng, generic=True)
        bounds = soft_scale.forecast_bounds(day)
        shape_ = (_t(point_of(day, rng)), *(_t(w) for w in shape(rng, day.n_service)))
        for field in FIELDS:
            direction = torch.randn(bounds.rate.shape, generator=generator, dtype=torch.float64)
            direction = direction / direction.norm()
            for kind in soft_scale_check.KINDS:
                found = soft_scale_check.compare(kind, bounds, field, direction, shape_)
                key = (kind, field)
                counts = outcomes.setdefault(key, {"resolvable": 0})
                counts[found["outcome"]] = counts.get(found["outcome"], 0) + 1
                counts["resolvable"] += found["resolvable"]
                if found["outcome"] == "failed":
                    failures.append((day_index, kind, field, found))
    print(f"outcomes per function and field, of {N_DAYS}: {outcomes}")
    assert not failures, failures[:3]
    for field in FIELDS:
        assert outcomes[("soft", field)]["resolvable"] >= 5, (field, outcomes)


def test_a_whole_hour_departure_gets_the_mean_of_its_two_one_sided_derivatives():
    """At a whole-hour departure the shares of the hour before and the hour after it have a kink;
    the derivative of each is the mean of its derivative from below and from above (1/2), which a
    central difference also gives."""
    departure = _t([2.0]).requires_grad_(True)
    found = soft_scale.shares(departure, 3)  # hours 1 to 3: shares 1, 0, 0
    assert found.detach().tolist() == [[1.0, 0.0, 0.0]]
    for hour, expected in enumerate((0.5, 0.5, 0.0)):
        (gradient,) = torch.autograd.grad(found[0, hour], departure, retain_graph=True)
        step = 1e-3
        ahead = soft_scale.shares(departure.detach() + step, 3)[0, hour]
        behind = soft_scale.shares(departure.detach() - step, 3)[0, hour]
        assert float(gradient) == expected
        assert abs(float((ahead - behind) / (2 * step)) - expected) <= 1e-9


def _gradient_of(value, *leaves):
    return [float(g) for g in torch.autograd.grad(value, leaves)]


def test_the_max_with_0_gets_the_mean_of_its_two_one_sided_derivatives():
    """`p({1})` of one car at its kink: departure 1.5 (share 0.5 of hour 1), rate 1, need 1, so
    `need - rate (d - s) = 1 - 1 * 1 = 0`. From above the derivative in the need is 1, from below
    0; in the rate -1 and 0."""
    need, rate = _t([1.0]).requires_grad_(True), _t([1.0]).requires_grad_(True)
    bounds = soft_scale.Bounds(_t([1.5]), rate, _t([5.0]), need)
    lower, _ = soft_scale.set_functions(bounds, 1)
    assert float(lower[1].detach()) == 0.0
    assert _gradient_of(lower[1], need, rate) == [0.5, -0.5]


def test_the_min_gets_the_mean_of_its_two_one_sided_derivatives():
    """`b({1})` of one car at its kink: departure 1.5, rate 2, cap 1, so `rate * s = 1 = cap`.
    One-sided derivatives: in the cap 1 and 0, in the rate `s = 0.5` and 0."""
    cap, rate = _t([1.0]).requires_grad_(True), _t([2.0]).requires_grad_(True)
    bounds = soft_scale.Bounds(_t([1.5]), rate, cap, _t([0.0]))
    _, upper = soft_scale.set_functions(bounds, 1)
    assert float(upper[1].detach()) == 1.0
    assert _gradient_of(upper[1], cap, rate) == [0.5, 0.25]


def _one_sided(departure: float, rate: float, cap: float, step: float = 1e-6):
    """(from below, from above) of `b({1})` of one car in its departure, one service hour."""

    def upper(d):
        bounds = soft_scale.Bounds(_t([d]), _t([rate]), _t([cap]), _t([0.0]))
        return float(soft_scale.set_functions(bounds, 1)[1][1])

    middle = upper(departure)
    return (middle - upper(departure - step)) / step, (upper(departure + step) - middle) / step


def test_where_two_kinks_coincide_the_gradient_is_the_product_of_the_halves():
    """`b({1}) = min(rate f_1(d), cap)` at departure 2 (the share of hour 1 reaches 1, a kink) with
    rate 2 and cap 2 (`rate * s = cap`, a kink of the `min` at the same point). From below the
    share rises and `rate * s` is below the cap: 2; from above both are flat: 0; their mean is 1.
    Each tie halves what passes through it, so autograd gives 2 * 1/2 * 1/2 = 0.5."""
    departure = _t([2.0]).requires_grad_(True)
    bounds = soft_scale.Bounds(departure, _t([2.0]), _t([2.0]), _t([0.0]))
    _, upper = soft_scale.set_functions(bounds, 1)
    assert _gradient_of(upper[1], departure) == [0.5]
    below, above = _one_sided(2.0, 2.0, 2.0)
    assert below == pytest.approx(2.0, rel=1e-6) and above == 0.0
    # one kink only (the cap far away): the mean of the one-sided derivatives, 1
    departure = _t([2.0]).requires_grad_(True)
    bounds = soft_scale.Bounds(departure, _t([2.0]), _t([5.0]), _t([0.0]))
    assert _gradient_of(soft_scale.set_functions(bounds, 1)[1][1], departure) == [1.0]


def test_an_empty_car_at_the_start_of_an_hour_has_derivative_0_and_gradient_one_half():
    """`b({1}) = min(2 f_1(d), 0) = 0` for every departure (cap 0): the derivative is 0 from both
    sides. At departure 1 the share's `max` with 0 ties and so does the cap's `min`, and autograd
    gives 2 * 1/2 * 1/2 = 0.5: where kinks coincide the gradient need not be the derivative even
    where one exists (`soft_scale`'s module text)."""
    departure = _t([1.0]).requires_grad_(True)
    bounds = soft_scale.Bounds(departure, _t([2.0]), _t([0.0]), _t([0.0]))
    _, upper = soft_scale.set_functions(bounds, 1)
    assert _gradient_of(upper[1], departure) == [0.5]
    assert _one_sided(1.0, 2.0, 0.0) == (0.0, 0.0)


def test_at_a_tie_of_smallest_terms_the_exact_scale_shares_the_gradient_evenly():
    """Two equal smallest terms (one service hour, both widths 1, centre 0.75, one car with `b` the
    cap 1.5 and `p = 0`: both terms 0.75): each gets half of the gradient, so the cap's derivative
    (1 in the upper term, 0 in the lower) is 1/2."""
    cap = _t([1.5]).requires_grad_(True)
    bounds = soft_scale.Bounds(_t([3.0]), _t([2.0]), cap, _t([0.0]))
    scale = soft_scale.exact_scale(bounds, _t([0.75]), _t([1.0]), _t([1.0]))
    assert float(scale.detach()) == 0.75
    assert _gradient_of(scale, cap) == [0.5]

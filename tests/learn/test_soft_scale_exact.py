"""Plan section 7, T8: `soft_scale`'s exact scale against `exact` and against linear programmes.

1. Whole-hour departures (true one-way cars of `made_up_days`, 2 to 4 service hours, rho cycling
   through 0, 0.1, 0.2): `soft_scale.set_functions` of `true_bounds` equal `exact.day_functions`,
   and `exact_scale` equals `exact.box_scale` (centre a point of the set from
   `fleet_lps.random_point`, widths as in `tests/wp4/test_t1_exact.py`). A centre moved outside
   the set (0.5 kWh above every hour's most, or below every hour's least) makes
   `exact.box_scale` raise, `centre_inside` False and `exact_scale` negative.
2. Fractional last hours (`soft_scale_fake.forecast_day`): each car's `p_i(A)` and `b_i(A)` from
   `soft_scale` equal the least and most energy a programme on the code base's own forecast rows
   (`carsets.forecast_rows`) draws in `A`, for every subset `A` of the service hours.
3. Fractional last hours, the fleet: `exact_scale` equals the smallest largest step over the box's
   vertex directions, one programme per vertex over every car's rows (`vertex_scale`), with an
   upper face binding in at least a tenth and a lower face in at least a tenth of the shapes.
4. A batch of days of different fleet sizes, cars padded with zero limits, gives each day's own
   values.
5. No finite term (every width zero): the scale and the soft minimum are infinite, the count of
   terms is 0, and the gradient is zero, not NaN.
6. Three service hours, for every non-empty subset, a width of 1 on its hours only (up, then
   down) gives `exact.box_scale`'s scale on whole hours.
7. `centre_inside` at its edge: a centre half of `soft_scale.TOLERANCE` (1e-9) beyond `b` or `p`
   is inside, twice it is outside.
8. The guards: `subset_bits` takes 1 to `MAX_SERVICE_HOURS` (8) service hours, each row the
   binary digits of its subset, and refuses 0 and 9; `soft_scale` refuses a sigma of 0 or below;
   `forecast_bounds` refuses a day without forecasts.

The departures of 2 and 3 are drawn from a continuous range; the test of 2 checks that every one
of its departures falls inside an hour.
"""

import dataclasses
import itertools
import math
import random

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pytest
import torch
from fleet_lps import random_point
from made_up_days import made_up_sample
from soft_scale_fake import car_extreme, car_rows, forecast_day, point_of, shape, vertex_scale

from fleetreserve import soft_scale
from fleetreserve.day import MAX_SERVICE_HOURS
from fleetreserve.exact import box_scale, day_functions

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net

SEED = 81
N_WHOLE = 150
N_FRACTIONAL = 120
N_CARS = 60
TOLERANCE = 1e-9


def _t(values):
    return torch.tensor(values, dtype=torch.float64)


def _scale(bounds, centre, up, down):
    return float(soft_scale.exact_scale(bounds, _t(centre), _t(up), _t(down)))


def test_whole_hour_departures_give_the_set_functions_of_exact():
    days, _ = made_up_sample(random.Random(SEED), N_WHOLE)
    for day in days:
        lower, upper = soft_scale.set_functions(soft_scale.true_bounds(day), day.n_service)
        functions = day_functions(day)
        assert np.allclose(lower.numpy(), functions.lower, rtol=0, atol=TOLERANCE)
        assert np.allclose(upper.numpy(), functions.upper, rtol=0, atol=TOLERANCE)


def test_whole_hour_departures_give_exact_box_scale():
    rng = random.Random(SEED + 1)
    days, _ = made_up_sample(random.Random(SEED + 1), N_WHOLE)
    worst = 0.0
    for day in days:
        n = day.n_service
        centre = random_point(rng, list(day.one_way_cars()), n)
        up, down = shape(rng, n)
        bounds = soft_scale.true_bounds(day)
        truth = box_scale(day_functions(day), centre, up, down)
        fast = _scale(bounds, centre, up, down)
        assert bool(soft_scale.centre_inside(bounds, _t(centre)))
        assert math.isfinite(truth)
        gap = abs(max(fast, 0.0) - truth) / max(1.0, truth)
        worst = max(worst, gap)
        assert gap <= TOLERANCE, (day, centre, up, down, truth, fast)
    print(f"whole hours: {N_WHOLE} of {N_WHOLE} agree with exact.box_scale; worst {worst:.1e}")


def test_a_centre_outside_the_set_is_flagged_and_gives_a_negative_scale():
    rng = random.Random(SEED + 2)
    days, _ = made_up_sample(random.Random(SEED + 2), 40)
    for day in days:
        n = day.n_service
        functions = day_functions(day)
        above = [functions.upper[1 << k] + 0.5 for k in range(n)]  # above every hour's most
        below = [functions.lower[1 << k] - 0.5 for k in range(n)]  # below every hour's least
        bounds = soft_scale.true_bounds(day)
        for centre, side in ((above, 0), (below, 1)):
            widths = list(shape(rng, n))
            widths[side][0] = widths[side][0] or 0.5  # that face of hour 1 has a width
            with pytest.raises(ValueError, match="outside"):
                box_scale(functions, centre, *widths)
            assert not bool(soft_scale.centre_inside(bounds, _t(centre)))
            assert _scale(bounds, centre, *widths) < 0


def _car_bounds(forecast):
    return soft_scale.Bounds(
        _t([forecast.departure]), _t([forecast.rate]), _t([forecast.cap]), _t([forecast.need])
    )


def test_fractional_last_hour_car_functions_match_the_programmes():
    rng = random.Random(SEED + 3)
    checked = fractional = cars = 0
    for _ in range(N_CARS):
        day = forecast_day(rng)
        n = day.n_service
        for forecast, rows in zip(day.forecasts, car_rows(day), strict=True):
            lower, upper = soft_scale.set_functions(_car_bounds(forecast), n)
            cars += 1
            fractional += forecast.departure != math.floor(forecast.departure)
            for mask in range(2**n):
                subset = [k + 1 for k in range(n) if mask >> k & 1]
                least = car_extreme(rows, subset, -1.0)
                most = car_extreme(rows, subset, 1.0)
                assert abs(float(lower[mask]) - least) <= 1e-7, (forecast, n, subset)
                assert abs(float(upper[mask]) - most) <= 1e-7, (forecast, n, subset)
                checked += 1
    print(f"fractional last hour: {checked} car subsets agree; {fractional} fractional departures")
    assert fractional == cars  # every drawn departure ends inside an hour


def _binding(bounds, centre, up, down):
    zero = [0.0] * len(centre)
    up_only = _scale(bounds, centre, up, zero) if any(up) else math.inf
    down_only = _scale(bounds, centre, zero, down) if any(down) else math.inf
    return "upper" if up_only <= down_only else "lower"


def test_fractional_last_hour_fleet_scale_matches_the_vertex_programmes():
    rng = random.Random(SEED + 4)
    binding = {"upper": 0, "lower": 0}
    worst = 0.0
    for _ in range(N_FRACTIONAL):
        day = forecast_day(rng)
        n = day.n_service
        centre = point_of(day, rng)
        up, down = shape(rng, n)
        bounds = soft_scale.forecast_bounds(day)
        assert bool(soft_scale.centre_inside(bounds, _t(centre)))
        truth = vertex_scale(car_rows(day), centre, up, down)
        fast = _scale(bounds, centre, up, down)
        gap = abs(fast - truth) / max(1.0, abs(truth))
        worst = max(worst, gap)
        assert gap <= 1e-6, (day, centre, up, down, truth, fast)
        binding[_binding(bounds, centre, up, down)] += 1
    print(f"fractional fleets: {N_FRACTIONAL} agree; worst {worst:.1e}; binding {binding}")
    assert min(binding.values()) >= 0.1 * N_FRACTIONAL


def _pad(bounds_list):
    width = max(b.rate.numel() for b in bounds_list)

    def stack(name):
        rows = []
        for b in bounds_list:
            value = getattr(b, name)
            rows.append(torch.cat([value, torch.zeros(width - value.numel(), dtype=value.dtype)]))
        return torch.stack(rows)

    return soft_scale.Bounds(stack("departure"), stack("rate"), stack("cap"), stack("need"))


def test_a_padded_batch_gives_each_days_own_values():
    rng = random.Random(SEED + 5)
    days = [forecast_day(rng, n_service=3) for _ in range(12)]
    shapes = [shape(rng, 3) for _ in days]
    centres = [point_of(day, rng) for day in days]
    singles = [soft_scale.forecast_bounds(day) for day in days]
    batch = _pad(singles)
    c = _t(centres)
    up = _t([s[0] for s in shapes])
    down = _t([s[1] for s in shapes])
    hard = soft_scale.exact_scale(batch, c, up, down)
    soft = soft_scale.soft_scale(batch, c, up, down, 0.05)
    assert len({b.rate.numel() for b in singles}) > 1  # the padding is exercised
    assert len({round(float(v), 9) for v in hard}) > len(days) // 2  # the days differ
    for k, single in enumerate(singles):
        one = soft_scale.exact_scale(single, c[k], up[k], down[k])
        assert float(hard[k]) == pytest.approx(float(one), abs=1e-12)
        one_soft = soft_scale.soft_scale(single, c[k], up[k], down[k], 0.05)
        assert float(soft[k]) == pytest.approx(float(one_soft), abs=1e-12)


def test_no_finite_term_gives_infinity_and_a_zero_gradient():
    rng = random.Random(SEED + 6)
    day = forecast_day(rng, n_service=2)
    bounds = soft_scale.forecast_bounds(day)
    need = bounds.need.clone().requires_grad_(True)
    bounds = soft_scale.Bounds(bounds.departure, bounds.rate, bounds.cap, need)
    centre = _t(point_of(day, rng))
    zero = torch.zeros(2, dtype=torch.float64)
    assert int(soft_scale.term_count(zero, zero)) == 0
    hard = soft_scale.exact_scale(bounds, centre, zero, zero)
    soft = soft_scale.soft_scale(bounds, centre, zero, zero, 0.1)
    assert math.isinf(float(hard.detach())) and float(hard.detach()) > 0
    assert math.isinf(float(soft.detach())) and float(soft.detach()) > 0
    (gradient,) = torch.autograd.grad(soft, need)
    assert torch.all(gradient == 0)


def test_shapes_with_widths_on_one_subset_agree_with_exact():
    """Three service hours; for every non-empty subset `A`, the shape with width 1 on the hours of
    `A` (up only, then down only) gives `exact.box_scale`'s scale on whole hours."""
    rng = random.Random(SEED + 7)
    days, _ = made_up_sample(random.Random(SEED + 7), 60, service_range=(3, 3))
    for day in days:
        centre = random_point(rng, list(day.one_way_cars()), 3)
        for hours in itertools.chain.from_iterable(
            itertools.combinations(range(3), r) for r in (1, 2, 3)
        ):
            up = [1.0 if k in hours else 0.0 for k in range(3)]
            for widths in ((up, [0.0] * 3), ([0.0] * 3, up)):
                truth = box_scale(day_functions(day), centre, *widths)
                fast = _scale(soft_scale.true_bounds(day), centre, *widths)
                assert abs(max(fast, 0.0) - truth) <= TOLERANCE * max(1.0, truth)


def test_the_centre_is_inside_within_the_tolerance_and_outside_beyond_it():
    """One service hour, one car (departure 2, rate 1, cap 1.5, need 0): `b({1}) = 1`, `p({1}) =
    0`. A centre half the tolerance (1e-9) above 1 or below 0 is inside; twice it is outside."""
    assert soft_scale.TOLERANCE == 1e-9
    bounds = soft_scale.Bounds(_t([2.0]), _t([1.0]), _t([1.5]), _t([0.0]))
    for centre, inside in (
        (1.0 + 0.5e-9, True),
        (1.0 + 2e-9, False),
        (-0.5e-9, True),
        (-2e-9, False),
    ):
        assert bool(soft_scale.centre_inside(bounds, _t([centre]))) == inside, centre


def test_the_guards_on_service_hours_sigma_and_forecasts():
    for n in (1, MAX_SERVICE_HOURS):
        bits = soft_scale.subset_bits(n)
        assert bits.shape == (2**n, n)
        for mask in range(2**n):
            assert [int(b) for b in bits[mask]] == [(mask >> k) & 1 for k in range(n)]
    for n in (0, MAX_SERVICE_HOURS + 1):
        with pytest.raises(ValueError, match="n_service must be 1 to 8"):
            soft_scale.subset_bits(n)
    bounds = soft_scale.Bounds(_t([2.0]), _t([1.0]), _t([1.5]), _t([0.0]))
    for sigma in (0.0, -0.01):
        with pytest.raises(ValueError, match="sigma must be positive"):
            soft_scale.soft_scale(bounds, _t([0.5]), _t([1.0]), _t([1.0]), sigma)
    days, _ = made_up_sample(random.Random(SEED + 8), 1)
    day = dataclasses.replace(days[0], forecasts=None)
    with pytest.raises(ValueError, match="the day has no forecasts"):
        soft_scale.forecast_bounds(day)

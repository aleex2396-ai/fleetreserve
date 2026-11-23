"""The label of a day for sizing (`size_label`): the exact scale of the certified box on the true
cars of the steered sub-day.

1. A hand-made day: one car, true departure hour 3, rate 2 kW, room 1 kWh, one service hour and
   one recovery hour. Its set in service hour 1 is [0, 1] kW (it can draw its 1 kWh in hour 0 or
   hour 2 instead). With the
   certificate's box replaced by centre 0.4 and half-width 0.2, the scale is
   min((1 - 0.4) / 0.2, (0.4 - 0) / 0.2) = 2 (clipped 1); half-width 0.5 gives 0.8; centre 1.5 is
   outside the set and gives 0.
2. On random small fleets (`size_fake`) with random centres and widths, the label equals the
   largest half-width scale at which every corner of the box is met by some plan of the cars, from
   a linear programme over per-car powers in every hour (`fleet_lps`), which does not use the set
   functions; a centre that no plan meets gives 0. The labels found take many values.
3. The fixed plan's own scale: the closed-form slot loss is 0 at it and above 0 just above it;
   a negative one is a loss at every scale above 0.
4. A day that does not offer has no label.
"""

import dataclasses
from itertools import product

import numpy as np
import pytest
from fleet_lps import _fleet, _hour_terms
from size_fake import START, prepared_days
from test_gate_day import SETTINGS, Stub, configuration, truth_day
from toy_values import _Model

from fleetreserve import gate_day, size_label
from fleetreserve.carsets import true_rows
from fleetreserve.day import Day, ForecastCar, TrueCar
from fleetreserve.fixed_plan_loss import closed_form_slot_loss, hour_thresholds
from fleetreserve.lp import LpNotOptimal
from fleetreserve.policy import AffinePolicy


def with_box(prepared, centre, width):
    policy = prepared.certificate.policy
    changed = AffinePolicy(np.array(centre), np.array(width), policy.g, policy.G)
    certificate = dataclasses.replace(prepared.certificate, policy=changed)
    return dataclasses.replace(prepared, certificate=certificate)


@pytest.fixture(scope="module")
def one_car():
    day = Day("u", START, 1, 1, 0.0, (TrueCar(3, 2.0, 1.0),))
    forecasts = [ForecastCar(3.0, 2.0, 1.0, 1.0)]
    prepared = gate_day.prepare_day(truth_day(day), Stub(forecasts), configuration(), SETTINGS)
    assert not prepared.abstains
    return prepared


@pytest.mark.parametrize(
    "centre, width, scale, clipped, inside",
    [(0.4, 0.2, 2.0, 1.0, True), (0.4, 0.5, 0.8, 0.8, True), (1.5, 0.2, 0.0, 0.0, False)],
)
def test_the_hand_made_day(one_car, centre, width, scale, clipped, inside):
    found = size_label.day_label(with_box(one_car, [centre], [width]))
    assert found.scale == pytest.approx(scale, abs=1e-9)
    assert found.clipped == pytest.approx(clipped, abs=1e-9)
    assert found.centre_inside is inside


def scale_by_corners(cars, n, centre, width) -> float:
    """The largest lam with every corner of centre +- lam width met by a plan of the cars."""
    m = _Model()
    lam = m.var(0.0, None)
    for path in product((-1, 1), repeat=n):
        plans = _fleet(m, cars)
        for k, z in enumerate(path):
            m.equal(_hour_terms(plans, k + 1) + [(lam, -float(z) * width[k])], centre[k])
    try:
        return m.maximise([(lam, 1.0)])
    except LpNotOptimal:  # no plan meets the centre itself
        return 0.0


def test_the_label_is_the_largest_scale_whose_corners_the_true_cars_meet():
    rng = np.random.default_rng(5)
    days = [p for p in prepared_days(11, 40) if not p.abstains]
    found = []
    for prepared in days[:25]:
        n = prepared.day.n_service
        centre = rng.uniform(0.0, 4.0, n)
        width = rng.uniform(0.1, 1.5, n)
        label = size_label.day_label(with_box(prepared, centre, width))
        cars = prepared.day.one_way_cars()
        expected = scale_by_corners(cars, n, centre, width)
        assert label.scale == pytest.approx(expected, abs=1e-6)
        assert label.clipped == pytest.approx(min(expected, 1.0), abs=1e-6)
        found.append(round(label.scale, 6))
    assert len(set(found)) >= 10
    assert 0.0 in found and any(s > 1 for s in found) and any(0 < s < 1 for s in found)


def test_the_fixed_plan_scale_is_where_the_closed_form_slot_loss_leaves_0():
    seen = set()
    for prepared in prepared_days(12, 40, early=0.2):
        if prepared.abstains:
            continue
        found = size_label.day_label(prepared).fixed_plan_scale
        thresholds = hour_thresholds(prepared.certificate.policy, true_rows(prepared.day))
        if np.isfinite(found) and found > 0:
            assert closed_form_slot_loss(thresholds, found) == 0.0
            assert closed_form_slot_loss(thresholds, found * (1 + 1e-9)) > 0.0
            seen.add("positive")
        elif found < 0:
            assert closed_form_slot_loss(thresholds, 1e-12) > 0.0
            seen.add("negative")
    assert seen == {"positive", "negative"}


def test_a_day_that_does_not_offer_has_no_label():
    day = Day("u", START, 1, 0, 0.0, (TrueCar(1, 2.0, 0.5),))
    prepared = gate_day.prepare_day(truth_day(day), Stub([None]), configuration(), SETTINGS)
    assert prepared.abstains
    with pytest.raises(ValueError, match="no label"):
        size_label.day_label(prepared)

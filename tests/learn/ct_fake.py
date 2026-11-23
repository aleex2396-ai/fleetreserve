"""Made-up labelled days with a certified value, for the tests of conformal training (`size_ct`).

`planted(rng, units, days, rule)`: as `size_net_fake.planted` (the car frame in the layout of
`size_net_inputs.car_frame`, one row per day), with each row's `size_ct.VALUE` beside its label;
`rule(n_cars, school_holiday, rng)` gives the day's (clipped label, value).

- `unequal_rule`, T7's case, where the marginal loss per unit value differs between day types:
  school-holiday days (about 30%) have labels spread over [0.1, 0.5] and value 30; the other days
  with 8 cars or more have labels in [0.85, 1] and value 10; those with fewer, labels in
  [0.53, 0.57] and value 5. A constant scale is held down by the holiday days; a scale per type
  that loses 10% of each type's days spends the loss allowance where it buys little value; the
  best sizing spends it on the holiday days, whose spread labels give the most value per day
  lost.
- `equal_rule`: the label uniform on [0.2, 1] and the value uniform on [5, 20], each drawn apart
  from the day's cars and holiday, so no day type differs from another.

`split_figures(labels, values, raw, arm, tier, splits)`: the mean offer per counting test day and
the mean test slot loss over `splits` random splits of the days (100 calibration, the rest test),
each day losing every slot above its label (so the losses `size_ct` trains on), sized and
calibrated by the existing route (`size_scale.sizes`, `size_resolve.relative`,
`size_resolve.calibrate`, `size_resolve.score`), as `tests/learn/test_size_net.py` scores the
net; plus the per-split offers.

`exact_figures(labels, values, raw, arm, splits)`: the mean offer per counting test day over the
same splits with the exact scalar of the calibration days (`conformal_scalar.exact_scalar`, at the
slot alpha) capped at 1 in place of the grid's: what the sizes give without the grid's steps.
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import size_net_fake
from truth_tables_fake import date_of

from fleetreserve import (
    conformal_scalar,
    crc,
    size_ct,
    size_gbm,
    size_net_inputs,
    size_resolve,
    size_scale,
)
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.learn_features import SITE_TYPES


def unequal_rule(n_cars: int, school_holiday: float, rng) -> tuple[float, float]:
    if school_holiday:
        return float(rng.uniform(0.1, 0.5)), 30.0
    if n_cars >= 8:
        return float(rng.uniform(0.85, 1.0)), 10.0
    return float(rng.uniform(0.53, 0.57)), 5.0


def equal_rule(n_cars: int, school_holiday: float, rng) -> tuple[float, float]:
    return float(rng.uniform(0.2, 1.0)), float(rng.uniform(5.0, 20.0))


def planted(rng, units=("caltech", "jpl", "OSL_S", "ASK"), days=200, rule=unequal_rule):
    """(car frame, labelled rows with values) of the module text."""
    cars, rows = [], []
    for unit in units:
        for k in range(days):
            n = int(rng.integers(3, 13))
            holiday = float(rng.random() < 0.3)
            date = date_of(k)
            day = {c: size_net_fake._value(c, rng) for c in size_net_fake.DAY_LEVEL}
            day.update(
                fleet_at_tau=n,
                public_holiday=False,
                site_type=SITE_TYPES[unit],
                school_holiday=holiday,
            )
            for j in range(n):
                car = {c: size_net_fake._value(c, rng) for c in size_net_inputs.COLUMNS}
                car.update(day)
                car.update(
                    unit=unit,
                    site=unit,
                    date=date,
                    session_id=f"{unit}-{k}-{j}",
                    arrived_earlier_day=bool(rng.random() < 0.1),
                    user_known=bool(rng.random() < 0.8),
                )
                cars.append(car)
            label, value = rule(n, holiday, rng)
            rows.append(
                {
                    "unit": unit,
                    "site": unit,
                    "date": date,
                    size_gbm.LABEL: label,
                    size_ct.VALUE: value,
                }
            )
    frame = pd.DataFrame(cars)[[*KEY_COLUMNS, *size_net_inputs.COLUMNS]]
    return frame, pd.DataFrame(rows)


def split_figures(labels, values, raw, arm: str, tier: str, splits: int = 40):
    """(mean offer, mean test slot loss, per-split offers) of the module text."""
    labels = np.asarray(labels, dtype=float)
    values = np.asarray(values, dtype=float)
    offering = np.ones(labels.size, dtype=bool)
    r = size_resolve.relative(size_scale.sizes(raw, offering, arm))
    losses = {
        (d, s): (1.0 if s > labels[d] + 1e-12 else 0.0, 0.0) for d, s in size_resolve.wanted(r)
    }
    offers, slots = [], []
    for split in range(splits):
        order = np.random.default_rng([7, split]).permutation(labels.size)
        calibration, test = sorted(order[:100]), sorted(order[100:])
        scalar = size_resolve.calibrate(losses, r, calibration, tier)
        scored = size_resolve.score(losses, r, values, scalar.used, test)
        offers.append(scored.mean_offer_kwh)
        slot = size_scale.mean_or_none(scored.slot_losses)
        if slot is not None:
            slots.append(slot)
    mean_slot = float(np.mean(slots)) if slots else math.nan
    return float(np.mean(offers)), mean_slot, np.array(offers)


def exact_figures(labels, values, raw, arm: str, splits: int = 40) -> float:
    """The mean offer of the module text with the exact scalar in place of the grid's."""
    labels = np.asarray(labels, dtype=float)
    values = np.asarray(values, dtype=float)
    r = size_resolve.relative(size_scale.sizes(raw, np.ones(labels.size, dtype=bool), arm))
    offers = []
    for split in range(splits):
        order = np.random.default_rng([7, split]).permutation(labels.size)
        calibration, test = np.sort(order[:100]), np.sort(order[100:])
        on = ~np.isnan(r[calibration])
        sizes = np.where(on, r[calibration], 1.0)
        scalar = conformal_scalar.exact_scalar(sizes, labels[calibration], on, crc.ALPHA_SLOT)
        scalar = min(scalar, 1.0)
        offers.append(float((scalar * np.nan_to_num(r[test]) * values[test]).mean()))
    return float(np.mean(offers))

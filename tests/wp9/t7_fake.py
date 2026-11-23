"""Made-up units for the tests of T7's measurement (`t7_unit`): per-day `size_scale.DayTerms` with
known thresholds, and seeded splits.

`made_up_unit(seed, n, planted)`: `n` offering days on consecutive dates from Monday 6 January
2020, no public holiday. Each day has one service hour, or two on one day in four. A day's first
threshold is drawn from [-0.1, 0.3] with probability `LOW` (it loses at small scales), else from
[0.35, 0.75] when its weekday is `planted` (many days lose just above scale 0.35) and from
[0.35, 3.0] otherwise; a second hour's threshold is the first plus [0, 1]. Values are drawn from
[5, 15] kWh. Each day has two driver ramps `c + beta s` with c in [-1.5, -0.2] and s in
[0.5, 1.5], over a protected energy of `protected` kWh. With `planted=None` every day is drawn the
same way, so the day types are exchangeable.

`step_unit(monday, tuesday)`: 140 days of two types whose losses are steps at three thresholds, for
readings worked by hand (its own text).

`random_splits(n, count, n_calibration, seed)`: `count` splits with seeds 1 to `count`, each a
seeded random choice of `n_calibration` calibration days, the rest test.
"""

import datetime

import numpy as np
from size_fake import ramp_terms

from fleetreserve import t7_stat

START = datetime.date(2020, 1, 6)  # a Monday
LOW = 0.08
STEPS = (1.07, 1.15, 3.0)
NO_DRIVER = (np.zeros(0), np.zeros(0), 0.0)


def made_up_unit(seed: int, n: int = 140, planted: str | None = None, protected: float = 10.0):
    rng = np.random.default_rng(seed)
    terms, types = [], []
    for k in range(n):
        date = (START + datetime.timedelta(days=k)).isoformat()
        kind = t7_stat.raw_type(date, False)
        if rng.random() < LOW:
            first = rng.uniform(-0.1, 0.3)
        elif kind == planted:
            first = rng.uniform(0.35, 0.75)
        else:
            first = rng.uniform(0.35, 3.0)
        thresholds = [first]
        if rng.random() < 0.25:
            thresholds.append(first + rng.uniform(0.0, 1.0))
        driver = (rng.uniform(-1.5, -0.2, 2), rng.uniform(0.5, 1.5, 2), protected)
        terms.append(ramp_terms(float(rng.uniform(5.0, 15.0)), np.array(thresholds), driver))
        types.append(kind)
    return terms, types


def step_unit(monday, tuesday):
    """140 days, Monday and Tuesday in turn, each with one service hour, value 2 kWh and no driver
    ramp; `monday` and `tuesday` give how many of each type's 70 days have the threshold 1.07, 1.15
    and 3.0. No threshold is below 1.07, so no day loses at a scale up to 1.07: every split's
    scalar used is 1 (capped), and the windows run 0.95 to 1.05, 0.9 to 1.1 and 0.8 to 1.2."""
    thresholds = {
        name: [x for x, n in zip(STEPS, counts, strict=True) for _ in range(n)]
        for name, counts in (("Monday", monday), ("Tuesday", tuesday))
    }
    terms, types = [], []
    for k in range(70):
        for name in ("Monday", "Tuesday"):
            terms.append(ramp_terms(2.0, np.array([thresholds[name][k]]), NO_DRIVER))
            types.append(name)
    return terms, types


def random_splits(n: int, count: int = 4, n_calibration: int = 70, seed: int = 0):
    out = []
    for s in range(1, count + 1):
        order = np.random.default_rng([seed, s]).permutation(n)
        out.append(
            (s, sorted(order[:n_calibration].tolist()), sorted(order[n_calibration:].tolist()))
        )
    return out

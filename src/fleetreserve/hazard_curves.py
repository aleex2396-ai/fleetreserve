"""Survival curves of the hazard model and the departure columns read from them (plan sections 1,
7; `forecast_model`'s rule for discrete survival curves).

Grid (`bin_starts`): the starts of the bins with a hazard, then the start of the last open bin. A
car leaves in the bin that holds its departure (hours from τ); in the open bin it leaves at the
last grid point or later.

Curve (`survival_curves`): from the hazards `h_j` of the bins, `S_j = prod_{k <= j} (1 - h_k)` is
the share still plugged in at the end of bin `j`; the last entry, for the open bin, is 0. Read as
`forecast_model` reads a discrete curve, the mass of bin `j` sits at the bin's start `t_j`
(`S_j = P(D > t_j)` for the departure `D` moved down to its bin's start), so every bound is at or
below the model's own quantile of the departure; the bound is the smallest grid point whose
distribution function exceeds the level (`forecast_model.survival_quantile`). A bound below
`fine_until_h` is a multiple of `fine_step_h` (a quarter hour by default), so a car's bound may
fall inside the last hour of a service window ("fractional last hour").

Presence at a whole hour `h` (a grid point under the default grid; `HazardSettings.check_grid`
guarantees it for the whole hours up to `fine_until_h`, see `hazard_settings`): the share of cars
still plugged in at the start of the hour is `P(D >= h)`, which is `S` at the bin before the one
that starts at `h`. `forecast_model.survival_quantile(times, S, level, after=a)` takes the last
time `t <= a` for that share and the times `t > a` for the bound; with `a = h - PRESENCE_SHIFT`
(1e-6 hours, far below the steps of the default grid) the share is `P(D >= h)` and the bound is
read over the bins from `h` on. So a bound given presence is at least `h` and may equal `h` (a car
present at `h` may leave within the bin that starts at `h`). With `a = h` itself the share would
be taken after the bin that starts at `h`, so the cars that leave within that bin would not count
as present and the bound could lie above their departure.

`lower_quantiles` is `survival_quantile` for many curves at once (a test checks them equal on
random curves, with and without `after`); `departure_table` fills `d_hat`, `d_median` and
`survival_h1` to `survival_h8` for every level. A level for which the curve cannot give a bound
(nothing left at the hour, or the curve ends before the level is reached) gives NaN, as in
`forecast_model`.

Hours the data do not support: `departure_table` can also leave `survival_h<t>` NaN. Given the
list `supported` (per service hour, whether enough fit rows reached it; `models.hazard` gives the
list of the set's own site type) an hour that is not supported is NaN for every car, and given
`presence_floor` a car whose own probability of being present at the start of the hour
(`presence_probability`) is below the floor is NaN at that hour. NaN in a survival column makes
the gate keep the bound of the hour before (`forecast_model`); the rules exist because a number
read from bins that no fit car reached would be an extrapolation. `d_hat` and `d_median` are
never changed by this.
"""

import math
from collections.abc import Sequence

import numpy as np

from fleetreserve.forecast_model import MEDIAN, PROBABILITY_TOLERANCE, SURVIVAL_COLUMNS
from fleetreserve.hazard_settings import HazardSettings

PRESENCE_SHIFT = 1e-6


def bin_starts(settings: HazardSettings) -> np.ndarray:
    """The start of every bin with a hazard, then the start of the open bin (module text)."""
    fine = np.arange(round(settings.fine_until_h / settings.fine_step_h)) * settings.fine_step_h
    n_coarse = round((settings.grid_end_h - settings.fine_until_h) / settings.coarse_step_h)
    coarse = settings.fine_until_h + np.arange(n_coarse + 1) * settings.coarse_step_h
    return np.concatenate([fine, coarse])


def n_hazard_bins(settings: HazardSettings) -> int:
    return len(bin_starts(settings)) - 1


def bin_widths(settings: HazardSettings) -> np.ndarray:
    return np.diff(bin_starts(settings))


def bin_of(departure: np.ndarray, settings: HazardSettings) -> np.ndarray:
    """The bin that holds each departure (hours from τ): a number from 0 to the number of bins
    with a hazard, the last one being the open bin. A departure below 0 is in bin 0."""
    starts = bin_starts(settings)
    found = np.searchsorted(starts, np.asarray(departure, float), side="right") - 1
    return np.clip(found, 0, len(starts) - 1)


def survival_curves(hazards: np.ndarray) -> np.ndarray:
    """`S` at the end of every bin, along the last axis, with a last entry of zero for the open bin:
    shape (..., bins with a hazard + 1), float64."""
    h = np.asarray(hazards, dtype=np.float64)
    survival = np.cumprod(1.0 - h, axis=-1)
    return np.concatenate([survival, np.zeros(survival.shape[:-1] + (1,))], axis=-1)


def lower_quantiles(
    times: np.ndarray, survival: np.ndarray, level: float, after: float | None = None
) -> np.ndarray:
    """`forecast_model.survival_quantile` for every row of `survival` (module text)."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    s = np.atleast_2d(survival)
    t = np.asarray(times, dtype=float)
    if after is None:
        alive = np.ones(s.shape[0])
        later = np.ones(t.size, dtype=bool)
    else:
        before = np.flatnonzero(t <= after)
        alive = np.ones(s.shape[0]) if before.size == 0 else s[:, before[-1]]
        later = t > after
    beyond = (alive[:, None] - s) > level * alive[:, None] + PROBABILITY_TOLERANCE
    beyond &= later[None, :]
    found = beyond.any(axis=1) & (alive > 0)
    first = beyond.argmax(axis=1)
    return np.where(found, t[first], math.nan)


def presence_probability(times: np.ndarray, survival: np.ndarray, hour: float) -> np.ndarray:
    """Per row of `survival`, the probability of being present at the start of `hour`, a grid point
    (module text): `S` at the last time before it."""
    s = np.atleast_2d(survival)
    before = np.flatnonzero(np.asarray(times, dtype=float) <= hour - PRESENCE_SHIFT)
    return np.ones(s.shape[0]) if before.size == 0 else s[:, before[-1]]


def departure_table(
    times: np.ndarray,
    survival: np.ndarray,
    kappas: Sequence[float],
    supported: Sequence[bool] | None = None,
    presence_floor: float = 0.0,
) -> dict[float, dict[str, np.ndarray]]:
    """Per level, the arrays `d_hat`, `d_median` and `survival_h1` to `survival_h8` over the rows
    of `survival` (module text); `supported` and `presence_floor` as the module text says."""
    median = lower_quantiles(times, survival, MEDIAN)
    out = {}
    for kappa in kappas:
        row = {"d_hat": lower_quantiles(times, survival, kappa), "d_median": median}
        for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
            row[column] = lower_quantiles(times, survival, kappa, after=hour - PRESENCE_SHIFT)
            if supported is not None and not supported[hour - 1]:
                row[column] = np.full_like(row[column], math.nan)
            low = presence_probability(times, survival, hour) < presence_floor
            row[column] = np.where(low, math.nan, row[column])
        out[float(kappa)] = row
    return out

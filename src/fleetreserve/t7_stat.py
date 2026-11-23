"""Day types, marginal losses per unit value and the permutation test of T7's measurement (plan
section 7, T7: "a constant multiplier is optimal if marginal expected loss per unit value is equal
across contexts. Tested per day type on training sites"). Pure functions on arrays; the command is
`t7_test`, the per-unit measurement `t7_unit`.

Day types (a reading; the plan does not define them). `raw_type(date, holiday)`: the weekday of the
date ("Monday" to "Sunday"), or `HOLIDAY` when the day is a public holiday, whatever its weekday.
`pool_types(raw)`: every type with fewer than `MIN_TYPE_DAYS` (10) days is renamed `OTHER`; the
pooled `OTHER` itself can still hold fewer than 10 days. `tested_types(types)`: the types with at
least 10 days after pooling, in the order of `TYPE_ORDER`.

Losses at a scale b (`day_losses`): 0 for every day at b <= 0 (plan section 6, T6: both losses are
0 at beta <= 0); above 0, from the day's `size_scale.DayTerms` at size 1, the fixed plan's
closed-form slot loss (`fixed_plan_loss.closed_form_slot_loss`: an hour fails when its threshold is
strictly below b) under the key "slot", and under the key "driver" the driver term: the
every-scale driver-loss bound as the every-request route computes it
(`gate_calibrate.every_scale_loss` of the day's `driver` and `hours`, capped at 1;
`fixed_plan_loss.every_scale_driver_bound`), the driver bound wherever one is used by the owner's
decision of 16 November 2026 (`gate_calibrate`, module text). The driver term counts each car's
decision-hour draw and bounds the replayed driver loss at every scale, with the errors of V3 and V7
of `fixed_plan_loss`'s module text. Every figure under "driver" in this measurement is about this
term, not about the replayed driver loss.

Window (`window(b, h)`): from `max(b - h, 0)` to `b + h`; its width is their difference, so `2 h`
when `b >= h` and `b + h` (one-sided) below that. `loss_differences(terms, b, h)`: per day and loss
the loss at the upper end minus the loss at the lower end.

Per type g (`type_readings`): its days, its mean certified value at scale 1, its mean loss at each
end, its slope (mean loss change over the width), its marginal loss per unit value, the slope
over the mean value, which equals the type's summed loss change over (width times its summed
value), and `changing`, the number of its days whose loss at the upper end differs from that at
the lower end. The value of an offer at scale b is b times the value at scale 1, so the value is
linear in b and its derivative is the value at scale 1. The slot loss is a step function of the
scale and the driver term does not fall as the scale rises (a ramp where no power row breaks, with
upward jumps where one does), so only the days whose loss changes inside the window carry
information about a type's marginal loss: every other day adds 0 to the loss change and its value
to the sum of values.

The statistic (`spread_statistic`): the largest minus the smallest marginal loss per unit value
over the types given (codes 0 to k - 1); it is never below 0. `permutation_p`: the share of
`rounds` seeded random permutations of the type labels among the days (each day keeps its loss
change and value) whose statistic is at least the observed one, counted as (1 + hits) / (1 +
rounds), so never 0. A permuted statistic counts as at least the observed one when it is at or
above `observed - TIE * observed` (rounding: the same sums in another order can differ in their
last digits); so an observed statistic of 0 gives p = 1.

The statistic sees each day only through its pair of loss change and value. The null the p-value
tests is that the type labels are exchangeable among the days: which type a day carries tells
nothing about its pair. Under that null the chance of p below 0.05 is at most 0.05. T7's condition
(equal expected marginal losses per unit value) is weaker: day types that meet it but whose days
differ in value, or in the spread of their loss changes (which a different loss level can bring:
both losses are at most 1, so a day whose loss is already above 0 at the window's lower end can
change by at most 1 minus that loss), are outside the null, and for them that chance is not
bounded by 0.05. So p below 0.05 can come from such a difference and not from unequal marginal
losses, and it does not say which; a loss level that leaves the pairs as they are does not enter
the statistic. A p-value of 0.05 or more is not a finding that the marginal losses are equal: a
type's marginal loss is read only from those of its days whose loss changes inside the window,
which can be few, so with a few hundred offering days or fewer the test can have little power and
often miss a real difference in marginal loss per unit value.

The random permutations come from `numpy.random.default_rng(seed)`, drawn in blocks of `BLOCK`;
sums per type are taken by `numpy.bincount`, which adds in day order, so the result does not
depend on the machine's linear algebra library.

`holm(p)`: Holm's step-down adjusted p-values (sorted ascending, the i-th of m times (m - i + 1),
running maximum, capped at 1); ties keep the order of the names given.
"""

import datetime
import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fleetreserve.fixed_plan_loss import closed_form_slot_loss
from fleetreserve.gate_calibrate import every_scale_loss

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
HOLIDAY = "public holiday"
OTHER = "other"
TYPE_ORDER = (*WEEKDAYS, HOLIDAY, OTHER)
MIN_TYPE_DAYS = 10
MAIN_WINDOW = 0.1
WINDOWS = (0.1, 0.05, 0.2)  # the main window first
LOSSES = ("slot", "driver")
ROUNDS = 10_000
LEVEL = 0.05
TIE = 1e-9
BLOCK = 500


def raw_type(date: str, holiday: bool) -> str:
    """The day type before pooling (module text)."""
    if holiday:
        return HOLIDAY
    return WEEKDAYS[datetime.date.fromisoformat(date).weekday()]


def pool_types(raw: Sequence[str], min_days: int = MIN_TYPE_DAYS) -> list[str]:
    """`raw` with every type of fewer than `min_days` days renamed `OTHER`."""
    counts = Counter(raw)
    return [t if counts[t] >= min_days else OTHER for t in raw]


def type_counts(types: Sequence[str]) -> dict[str, int]:
    """Days per type, in the order of `TYPE_ORDER`."""
    counts = Counter(types)
    return {t: counts[t] for t in TYPE_ORDER if counts[t]}


def tested_types(types: Sequence[str], min_days: int = MIN_TYPE_DAYS) -> tuple[str, ...]:
    """The types with at least `min_days` days, in the order of `TYPE_ORDER`."""
    return tuple(t for t, n in type_counts(types).items() if n >= min_days)


def day_losses(terms: Sequence, scale: float) -> dict[str, np.ndarray]:
    """Per loss, the loss of each day (`size_scale.DayTerms`) at `scale` (module text)."""
    if scale <= 0:
        zeros = np.zeros(len(terms))
        return {"slot": zeros, "driver": zeros.copy()}
    return {
        "slot": np.array([closed_form_slot_loss(t.thresholds, scale) for t in terms], dtype=float),
        "driver": np.array(
            [every_scale_loss((t.driver, t.hours), scale) for t in terms], dtype=float
        ),
    }


def window(scale: float, h: float) -> tuple[float, float]:
    """The lower and upper ends of the window of half-width `h` around `scale`."""
    return max(scale - h, 0.0), scale + h


@dataclass(frozen=True)
class Differences:
    """The days' losses at both ends of one window."""

    scale: float
    h: float
    low: float
    high: float
    at_low: dict  # loss -> per-day losses at `low`
    at_high: dict

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def differences(self) -> dict:
        return {loss: self.at_high[loss] - self.at_low[loss] for loss in LOSSES}


def loss_differences(terms: Sequence, scale: float, h: float) -> Differences:
    low, high = window(scale, h)
    return Differences(scale, h, low, high, day_losses(terms, low), day_losses(terms, high))


@dataclass(frozen=True)
class TypeReading:
    days: int
    mean_value: float
    mean_loss_low: float
    mean_loss_high: float
    slope: float
    marginal: float  # slope over mean value
    changing: int  # days whose loss differs between the window's two ends


def type_readings(
    found: Differences, loss: str, values, types: Sequence[str], names: Sequence[str] | None = None
) -> dict:
    """Per type in `names` (default: the types present, those of `TYPE_ORDER` in its order, any
    other name after them in the order of its first day) its `TypeReading` over its days, None for
    a type with no day."""
    values = np.asarray(values, dtype=float)
    types = np.asarray(types, dtype=object)
    if names is None:
        present = list(dict.fromkeys(types.tolist()))
        names = sorted(present, key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else 99)
    out = {}
    for name in names:
        mask = types == name
        if not mask.any():
            out[name] = None
            continue
        low = float(found.at_low[loss][mask].mean())
        high = float(found.at_high[loss][mask].mean())
        mean_value = float(values[mask].mean())
        slope = (high - low) / found.width
        changing = int(np.sum(found.at_high[loss][mask] != found.at_low[loss][mask]))
        out[name] = TypeReading(
            int(mask.sum()), mean_value, low, high, slope, slope / mean_value, changing
        )
    return out


def _per_type_sums(rows, codes, k: int) -> np.ndarray:
    """Sums per type of each row of `rows` (rows x days), by `numpy.bincount` in day order."""
    rows = np.atleast_2d(rows)
    flat = (codes[None, :] + k * np.arange(rows.shape[0])[:, None]).ravel()
    return np.bincount(flat, weights=rows.ravel(), minlength=rows.shape[0] * k).reshape(-1, k)


def _check(diff, values, codes, k: int):
    diff = np.asarray(diff, dtype=float)
    values = np.asarray(values, dtype=float)
    codes = np.asarray(codes, dtype=np.int64)
    if not (diff.shape == values.shape == codes.shape) or diff.ndim != 1:
        raise ValueError("loss changes, values and type codes must be arrays of the same length")
    if np.any(values <= 0):
        raise ValueError("every day's value must be above 0")
    if codes.size and (codes.min() < 0 or codes.max() >= k):
        raise ValueError(f"type codes must be 0 to {k - 1}")
    if any(not np.any(codes == g) for g in range(k)):
        raise ValueError("every type must have a day")
    return diff, values, codes


def marginals(diff, values, codes, k: int, width: float) -> np.ndarray:
    """Per type the marginal loss per unit value (module text)."""
    diff, values, codes = _check(diff, values, codes, k)
    return _ratios(diff[None, :], values[None, :], codes, k, width)[0]


def _ratios(diff_rows, value_rows, codes, k, width):
    return _per_type_sums(diff_rows, codes, k) / (width * _per_type_sums(value_rows, codes, k))


def spread_statistic(diff, values, codes, k: int, width: float) -> float:
    found = marginals(diff, values, codes, k, width)
    return float(found.max() - found.min())


def permutation_p(
    diff, values, codes, k: int, width: float, rounds: int = ROUNDS, seed: int = 0
) -> tuple[float, float]:
    """(observed statistic, permutation p-value) (module text)."""
    diff, values, codes = _check(diff, values, codes, k)
    observed = spread_statistic(diff, values, codes, k, width)
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < rounds:
        size = min(BLOCK, rounds - done)
        order = rng.permuted(np.tile(np.arange(codes.size), (size, 1)), axis=1)
        found = _ratios(diff[order], values[order], codes, k, width)
        hits += int(np.sum(found.max(axis=1) - found.min(axis=1) >= observed - TIE * observed))
        done += size
    return observed, (1 + hits) / (1 + rounds)


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm's adjusted p-values (module text), by name."""
    names = sorted(pvalues, key=lambda n: pvalues[n])
    m = len(names)
    out, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def permutation_seed(unit: str) -> int:
    """The seed of a unit's permutations: the first 8 bytes of SHA-256 of "t7 <unit>" read as a
    big-endian whole number: each unit name has its own fixed seed."""
    return int.from_bytes(hashlib.sha256(f"t7 {unit}".encode()).digest()[:8], "big")

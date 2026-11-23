"""T7's measurement on one unit: its constant scales, per-type marginal losses per unit value and
the verdict (plan section 7, T7; section 6 for the calibration). The readings below are this
project's (the plan leaves them open); `t7_stat` has the functions, `t7_test` the command.

Input: the unit's counting days in day order, each as its `size_scale.DayTerms` (None for a day
that does not offer), the raw day type of each offering day (`t7_stat.raw_type`), and the unit's
seeded splits (seed, calibration day indices, test day indices). An offering day's certified value
is above 0 in the command's input: a day offers only when its certificate's value is above the
matrix's `zero_value_kwh` (`gate_day.prepare_day`), which `gate_config` refuses below 0. Here an
offering day of value 0 or below raises `ValueError` before any reading, so it fails its unit
(`t7_test`) and enters no type mean.

1. Policy and losses: the fixed plan with its closed-form slot loss and its driver term, as
   `size_whatif` computes them (`size_scale`); no re-solve. The driver term is the every-scale
   driver-loss bound of the every-request route (`gate_calibrate.every_scale_terms`,
   `fixed_plan_loss.every_scale_driver_bound`), the driver bound wherever one is used by the
   owner's decision of 16 November 2026, with each car's decision-hour draw; it bounds the
   replayed driver loss at every scale, with the errors of V3 and V7 of `fixed_plan_loss`'s module
   text. Below, "loss" covers the slot loss and the driver term, and "driver" names the driver
   term: tier A's driver scalar, the binding and deciding "driver" loss and every driver p-value
   and verdict are about this term, not about the replayed driver loss.
2. Day types: pooled over the unit's offering days by `t7_stat.pool_types`; the types tested are
   those with at least 10 offering days (`t7_stat.tested_types`). The days of a pooled "other"
   with fewer than 10 days are reported and left out of the statistic and its permutations.
3. Constant scale: per split and tier, `size_scale.calibrate` with size 1 on every offering day
   (the `constant` sizing of `size_whatif`, the same function on the same inputs), so the scalar
   used is the calibrated one capped at 1. Tier A: the smaller of the slot and driver scalars;
   tier B: the slot scalar.
4. Internal check: on every split and tier whose scalar used is above 0, the per-day losses this
   module reads at that scalar on the offering test days equal those of `size_scale.score`
   exactly; otherwise `InternalCheckFailed` is raised (the command then exits 1).
5. Per-split readings: on each split, at that split's scalar used, over its offering test days,
   per type, loss and window the marginal loss per unit value (`t7_stat.type_readings`); per type
   the mean, p5 and p95 over the splits on which the type has an offering test day (numpy linear
   interpolation), and their number.
6. Pooled reading, per tier: at the pooled scalar (`pooled_scalar`; a reading), the lower median
   of the scalars used that are above 0: with n such splits, sorted from the smallest, the
   ((n + 1) // 2)-th, so always a scalar some split used; 0 when no split's scalar used is above 0.
   Over all the unit's offering days, per type, loss and window, the type readings and
   `days_changing`, the number of the type's days whose loss differs between the window's two
   ends; over the tested types the statistic (largest minus smallest marginal loss per unit value)
   and its p-value from `t7_stat.ROUNDS` (10,000) permutations seeded by
   `t7_stat.permutation_seed(unit)` (the first 8 bytes of SHA-256 of "t7 <unit>"): one seed per
   unit, so the same permutations for every tier, loss and window of a unit and other
   permutations for another unit.
7. The deciding loss (a reading of T7's single loss under two tiers): tier B's scalar is the slot
   loss's alone, so tier B is decided on the slot loss. Tier A's scalar is the smaller of the two
   losses' scalars, so the binding loss of a split is the driver term when tier A's calibrated
   scalar is below tier B's (the slot scalar), else the slot loss; tier A is decided on the loss
   that binds on more splits, the slot loss on a tie. The other loss's figures are reported beside
   it and do not decide.
8. Verdict per tier, the first that applies: no split ("not testable: the unit is not split"); a
   scalar used of 0 on more than half of the splits ("not testable: no offer"; counted over all
   the splits, those of scalar 0 included, which the pooled scalar leaves out); fewer than 2
   tested types ("not testable: one day type"); every tested type's slope of the deciding loss
   exactly 0 at every window ("not testable: no day changes loss inside the window"); p below 0.05
   at the main window 0.1 on the deciding loss ("UNEQUAL"); else "not shown unequal". The
   p-values at windows 0.05 and 0.2 do not change the verdict. What a verdict says (`t7_stat`'s
   text): the p-value's null is that the type labels are exchangeable among the days, which is
   stronger than T7's condition, so day types with equal expected marginal losses per unit value
   whose days differ in value or in the spread of their loss changes can give "UNEQUAL"; "not
   shown unequal" is not a finding that the marginal losses are equal, and the test can have
   little power; only the days whose loss changes inside the window carry information
   (`days_changing`).
9. Also counted per tier: the splits whose scalar used is 0, those whose scalar used is above 0
   (`splits_that_offer`), and those whose scalar used is below the calibrated one (capped at 1).
   Where the cap binds, the scalar is set by the cap and not by the loss limit, so T7's
   condition, stated at E[loss] = alpha, is not the one that sets the scalar there (a reading);
   the verdict does not read the counts.
"""

import math
from collections import Counter

import numpy as np

from fleetreserve import gate_calibrate, size_scale, t7_stat

UNEQUAL = "UNEQUAL"
NOT_SHOWN = "not shown unequal"
NOT_SPLIT = "not testable: the unit is not split"
NO_OFFER = "not testable: no offer"
ONE_TYPE = "not testable: one day type"
FLAT = "not testable: no day changes loss inside the window"
POINTS = (0.05, 0.95)
LOSS_NAMES = {"slot": "slot losses", "driver": "driver terms"}


class InternalCheckFailed(Exception):
    """Step 4 of the module text failed."""


def _number(value):
    return gate_calibrate.number(value)


def _key(h: float) -> str:
    return f"{h:g}"


def constant_sizes(terms) -> np.ndarray:
    offering = np.array([t is not None for t in terms], dtype=bool)
    return size_scale.sizes(np.where(offering, 1.0, math.nan), offering, "scale")


def check_scores(unit, seed, terms, q, scalars, test) -> None:
    """Step 4 of the module text."""
    offering = [d for d in test if terms[d] is not None]
    for tier, scalar in scalars.items():
        if scalar.used <= 0:
            continue
        found = size_scale.score(terms, q, scalar.used, test)
        mine = t7_stat.day_losses([terms[d] for d in offering], scalar.used)
        for loss, theirs in (("slot", found.slot_losses), ("driver", found.driver_losses)):
            if not np.array_equal(mine[loss], np.array(theirs, dtype=float)):
                raise InternalCheckFailed(
                    f"unit {unit}, split {seed}, tier {tier}: the {LOSS_NAMES[loss]} at the scalar "
                    f"{scalar.used!r} differ from size_scale.score's"
                )


def _reading(r: t7_stat.TypeReading | None) -> dict | None:
    if r is None:
        return None
    return {
        "days": r.days,
        "mean_value_kwh": r.mean_value,
        "mean_loss_low": r.mean_loss_low,
        "mean_loss_high": r.mean_loss_high,
        "slope": r.slope,
        "marginal_per_kwh": r.marginal,
        "days_changing": r.changing,
    }


def _spread(values) -> dict:
    if not values:
        return {"splits": 0, "mean": None, "p5": None, "p95": None}
    low, high = np.quantile(values, POINTS)
    return {
        "splits": len(values),
        "mean": float(np.mean(values)),
        "p5": float(low),
        "p95": float(high),
    }


def per_split(terms, types, names, scales, tests) -> dict:
    """Step 5: loss -> window -> type -> spread over splits of the marginal per unit value."""
    found = {
        loss: {_key(h): {n: [] for n in names} for h in t7_stat.WINDOWS} for loss in t7_stat.LOSSES
    }
    for scale, test in zip(scales, tests, strict=True):
        days = [d for d in test if terms[d] is not None]
        values = np.array([terms[d].value for d in days], dtype=float)
        labels = [types[d] for d in days]
        for h in t7_stat.WINDOWS:
            differences = t7_stat.loss_differences([terms[d] for d in days], scale, h)
            for loss in t7_stat.LOSSES:
                readings = t7_stat.type_readings(differences, loss, values, labels, names)
                for name, r in readings.items():
                    if r is not None:
                        found[loss][_key(h)][name].append(r.marginal)
    return {
        loss: {w: {n: _spread(v) for n, v in by_type.items()} for w, by_type in by_window.items()}
        for loss, by_window in found.items()
    }


def pooled(unit, terms, types, names, tested, scale, rounds) -> dict:
    """Step 6: loss -> window -> readings, statistic and p-value."""
    days = [d for d, t in enumerate(terms) if t is not None]
    values = np.array([terms[d].value for d in days], dtype=float)
    labels = [types[d] for d in days]
    inside = np.array([label in tested for label in labels], dtype=bool)
    codes = np.array([tested.index(label) for label in labels if label in tested], dtype=np.int64)
    out = {loss: {} for loss in t7_stat.LOSSES}
    for h in t7_stat.WINDOWS:
        differences = t7_stat.loss_differences([terms[d] for d in days], scale, h)
        for loss in t7_stat.LOSSES:
            readings = t7_stat.type_readings(differences, loss, values, labels, names)
            entry = {
                "low": differences.low,
                "high": differences.high,
                "width": differences.width,
                "types": {n: _reading(r) for n, r in readings.items()},
                "statistic": None,
                "p_value": None,
            }
            if len(tested) >= 2:
                statistic, p = t7_stat.permutation_p(
                    differences.differences[loss][inside],
                    values[inside],
                    codes,
                    len(tested),
                    differences.width,
                    rounds,
                    t7_stat.permutation_seed(unit),
                )
                entry["statistic"], entry["p_value"] = statistic, p
            out[loss][_key(h)] = entry
    return out


def _flat(reading: dict, tested) -> bool:
    return all(
        reading[_key(h)]["types"][t]["slope"] == 0.0 for h in t7_stat.WINDOWS for t in tested
    )


def verdict(splits: int, zero: int, tested, reading: dict | None, loss: str) -> str:
    """Step 8 of the module text."""
    if splits == 0:
        return NOT_SPLIT
    if zero > splits / 2:
        return NO_OFFER
    if len(tested) < 2:
        return ONE_TYPE
    if _flat(reading[loss], tested):
        return FLAT
    return (
        UNEQUAL
        if reading[loss][_key(t7_stat.MAIN_WINDOW)]["p_value"] < t7_stat.LEVEL
        else NOT_SHOWN
    )


def pooled_scalar(used) -> float:
    """Step 6: the lower median of the scalars used above 0; 0 when there is none."""
    offering = sorted(u for u in used if u > 0)
    return float(offering[(len(offering) - 1) // 2]) if offering else 0.0


def tier_reading(unit, tier, terms, types, names, tested, rows, rounds) -> dict:
    """Steps 5 to 9 for one tier; `rows`: (seed, scalars by tier, test indices) per split."""
    used = [scalars[tier].used for _, scalars, _ in rows]
    zero = sum(u <= 0 for u in used)
    capped = sum(scalars[tier].used < scalars[tier].calibrated for _, scalars, _ in rows)
    binding = Counter(
        "driver" if tier == "A" and scalars["A"].calibrated < scalars["B"].calibrated else "slot"
        for _, scalars, _ in rows
    )
    deciding = "driver" if binding["driver"] > binding["slot"] else "slot"
    out = {
        "splits": len(rows),
        "splits_scalar_zero": zero,
        "splits_that_offer": len(rows) - zero,
        "splits_scalar_capped": capped,
        "binding_loss_splits": {"slot": binding["slot"], "driver": binding["driver"]},
        "deciding_loss": deciding,
        "pooled_scalar": None,
        "pooled": None,
        "per_split": None,
    }
    reading = None
    if rows:
        scale = pooled_scalar(used)
        reading = pooled(unit, terms, types, names, tested, scale, rounds)
        out["pooled_scalar"] = scale
        out["pooled"] = reading
        out["per_split"] = per_split(terms, types, names, used, [test for _, _, test in rows])
    out["verdict"] = verdict(len(rows), zero, tested, reading, deciding)
    return out


def measure_unit(unit, terms, raw_types, splits, rounds: int = t7_stat.ROUNDS) -> dict:
    """The module text's steps for one unit. `raw_types[d]`: the raw type of offering day d."""
    offering = [d for d, t in enumerate(terms) if t is not None]
    for d in offering:
        if not terms[d].value > 0:
            raise ValueError(f"offering day {d} has value {terms[d].value!r}, not above 0")
    pooled_types = t7_stat.pool_types([raw_types[d] for d in offering])
    types = dict(zip(offering, pooled_types, strict=True))
    names = tuple(t7_stat.type_counts(pooled_types))
    tested = t7_stat.tested_types(pooled_types)
    q = constant_sizes(terms)
    rows = []
    for seed, calibration, test in splits:
        scalars = size_scale.calibrate(terms, q, calibration)
        check_scores(unit, seed, terms, q, scalars, test)
        rows.append((seed, scalars, test))
    return {
        "offering_days": len(offering),
        "raw_day_types": t7_stat.type_counts([raw_types[d] for d in offering]),
        "day_types": t7_stat.type_counts(pooled_types),
        "tested_types": list(tested),
        "splits": [
            {
                "seed": seed,
                **{
                    tier: {
                        "calibrated": _number(s.calibrated),
                        "used": s.used,
                        "reason": s.reason,
                    }
                    for tier, s in scalars.items()
                },
            }
            for seed, scalars, _ in rows
        ],
        "tiers": {
            tier: tier_reading(unit, tier, terms, types, names, tested, rows, rounds)
            for tier in size_scale.TIERS
        },
    }

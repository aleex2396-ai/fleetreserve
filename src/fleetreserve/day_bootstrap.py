"""The percentile bootstrap over days of one calibrated job (plan section 11: "intervals: 95%
bootstrap over days"), as pure functions: no files, no solver; the only randomness is `draws`.

Input: one record of `day_losses` (a configuration group on one unit, its one split), parsed by
`Job.from_record`, and per rung and tier the scale the run chose. Decided by the project owner on
16 Nov 2026: the offer's calibration days and test days are resampled and the scale is recomputed
in every resample by the run's own rule; the losses are resampled over the test days only, at the
run's scale; 2,000 resamples; the seed is an argument with no default; the interval is the 2.5%
and 97.5% percentiles. Readings of what that leaves open (each with its alternative in brackets):

  I1. Groups. A resample draws as many calibration days as the run has, with replacement, from
      the run's calibration days, and as many test days, with replacement, from its test days;
      the two counts stay as in the run. [Alternative: one pooled resample of all the unit's
      days, the roles kept by day.]
  I2. Every calibration day is drawn, offering or not. The number of offering calibration days
      then varies from resample to resample; a resample with too few of them abstains, as the
      run's rule does (`crc.enough_days`: 19 for tier A, 9 for tier B), and offers 0. A run
      whose own calibration has too few offering days evaluates no scale, so its per-day file
      may hold none. A resample draws as many calibration days as the run has, so with fewer
      calibration days in all than a tier needs, the same rule abstains in every resample of
      that tier; with more, a resample that draws enough offering days (a day drawn more than
      once) is calibrated from the pairs the scoring run evaluated for it.
  I3. The scale of a resample is `crc.calibrate_search` itself, the run's calibration, called on
      the resampled offering calibration days (each as often as it is drawn) with each day's raw
      losses (tier A: slot loss at `crc.ALPHA_SLOT` and driver loss at `crc.ALPHA_DRIVER`; tier
      B: the slot loss alone, the driver alpha 1, as `gate_calibrate` calibrates tier B). The
      search asks for the scales its own path needs, and the scoring run has evaluated every
      (day, scale) pair the resamples of its seed ask for (`day_resample`, the per-day file's
      `resampled`), so every scale is the rule's own, exactly. Only where an evaluation failed
      (an error, recorded with the pair) is a pair's loss unknown; the search is then run twice,
      once with that pair's losses 1 (`low`) and once with them 0 (`high`). The search's result
      cannot rise when losses rise, so `low` <= the rule's scale <= `high`; such a resample is
      counted and labelled (`resamples_with_failed_pairs`). A pair neither evaluated nor failed
      means the file does not hold this resampling; it is counted
      (`resamples_with_pairs_not_evaluated`), bounded the same way, and fails the command. On
      the run's own days the search asks only pairs the run evaluated, so the run's scale comes
      back exactly. [Alternative: replay inside the intervals command, from the run's inputs.]
  I4. The offer of a resample is its scale times the mean certified value of its drawn test days
      (0 on a day that does not offer), kWh per counting test day; per car, the scale times the
      sum of the certified values over the sum of the cars of the drawn test days; as a share of
      a ceiling, the scale times the sum of the certified values over the sum of that ceiling
      (the ratio of sums, `gate_stat` reading G1), not computed when a test day's ceiling is
      missing in the run (a timeout), and left out for a resample whose ceilings sum to 0
      (counted). Its interval is [the 2.5% percentile of the `low` offers, the 97.5% percentile
      of the `high` offers]: the rule's own interval when every resample is exact (`low` =
      `high`), an outer bound otherwise; `lower_end_between` and `upper_end_between` give how far
      each end can move. [Alternative: no interval unless every resample is exact.]
  I5. The losses of a resample are the means of the stored raw test-day losses at the run's scale
      over its drawn test days that offer, as the run's own test means. A resample that draws no
      offering test day has no loss: it is left out of the loss intervals and counted. At a run
      scale of 0 nothing is offered and no loss interval exists. [Alternative: such a resample
      counted as loss 0.]
  I6. Availability is 1 minus the slot loss and shortfall is the driver loss (plan section 0 and
      O4: "mean slot availability >= 90%, mean driver shortfall <= 5%"), both over the offering
      test days. [Alternative: availability as the share of all service hours of the offering
      days that are met, which differs from 1 minus the mean slot loss when the days of a unit
      have different numbers of service hours, as the pooled unit's sites may.]
  I7. Percentiles are numpy's `quantile` with method "linear" (Hyndman and Fan's type 7: the
      position (B - 1) p between order statistics, linear between them), no bias correction.
      [Alternative: "inverted_cdf", the ceil(B p)-th smallest value.]
  I8. Draws (`draws`): numpy's default generator (PCG64) seeded with the seed and a number taken
      from the SHA-256 of the unit name; the calibration positions are drawn first, then the test
      positions; each resample's positions are sorted. One draw per unit is used for every
      configuration, rung and tier, as one split serves every configuration. The draws repeat
      under the numpy version pinned in `uv.lock`. [Alternative: one hash per drawn day, which no
      library version can change, at a higher cost.]
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from fleetreserve import crc
from fleetreserve.dial_splits import tier_alphas
from fleetreserve.gate_config import CEILINGS

RESAMPLES = 2000  # the project owner's decision of 16 Nov 2026
LEVELS = (0.025, 0.975)
QUANTILE_METHOD = "linear"  # reading I7


@dataclass(frozen=True)
class Scale:
    """The scale of one resample (reading I3): `low` <= the rule's scale <= `high`, equal when
    `exact`. `too_few`: too few offering calibration days, so the rule evaluates nothing and
    abstains (exactly). `failed_pair`: the search met a pair whose evaluation failed in the run;
    `missing_pair`: it met a pair the file does not hold at all."""

    low: float
    high: float
    exact: bool
    too_few: bool
    failed_pair: bool = False
    missing_pair: bool = False


def scale_of(slot, driver, scales, alpha_slot: float, alpha_driver: float, failed=None) -> Scale:
    """`crc.calibrate_search` on days whose raw losses at `scales` are the rows of `slot` and
    `driver` (shape (days, len(scales)); NaN where a pair is not known). An unknown pair, or a
    scale not in `scales`, is taken as losing everything for `low` and nothing for `high`
    (reading I3); `failed` (same shape, True where the run's evaluation of the pair failed) tells
    a failed pair from a missing one."""
    slot = np.asarray(slot, dtype=float)
    driver = np.asarray(driver, dtype=float)
    n = slot.shape[0]
    where = {float(b): k for k, b in enumerate(scales)}
    if slot.shape != (n, len(where)) or driver.shape != slot.shape:
        raise ValueError("slot and driver losses must have one column per scale")
    failed = np.zeros(slot.shape, dtype=bool) if failed is None else np.asarray(failed, dtype=bool)
    unknown = np.isnan(slot) | np.isnan(driver)

    def search(fill: float) -> tuple[crc.SearchCalibration, bool, bool]:
        met = {"failed": False, "missing": False}

        def loss_at(beta):
            k = where.get(float(beta))
            if k is None:
                met["missing"] = True
                return np.full(n, fill), np.full(n, fill)
            gone = unknown[:, k]
            if not gone.any():
                return slot[:, k], driver[:, k]
            met["failed"] = met["failed"] or bool((gone & failed[:, k]).any())
            met["missing"] = met["missing"] or bool((gone & ~failed[:, k]).any())
            return np.where(gone, fill, slot[:, k]), np.where(gone, fill, driver[:, k])

        found = crc.calibrate_search(loss_at, n, alpha_slot, alpha_driver)
        return found, met["failed"], met["missing"]

    low, failed_low, missing_low = search(1.0)
    too_few = not low.evaluations and low.abstain
    if not (failed_low or missing_low):
        return Scale(low.beta, low.beta, True, too_few)
    high, failed_high, missing_high = search(0.0)
    return Scale(
        low.beta,
        high.beta,
        False,
        too_few,
        failed_low or failed_high,
        missing_low or missing_high,
    )


def draws(seed: int, unit: str, n_calibration: int, n_test: int, resamples: int):
    """Positions of the drawn calibration days and test days, arrays of shape (resamples,
    n_calibration) and (resamples, n_test), each row sorted (reading I8)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"the seed must be a non-negative integer, got {seed!r}")
    if resamples < 1:
        raise ValueError("at least one resample is needed")
    key = int.from_bytes(hashlib.sha256(unit.encode("utf-8")).digest()[:8], "big")
    generator = np.random.default_rng([seed, key])
    calibration = generator.integers(0, n_calibration, size=(resamples, n_calibration))
    test = generator.integers(0, n_test, size=(resamples, n_test))
    return np.sort(calibration, axis=1), np.sort(test, axis=1)


# --- one job --------------------------------------------------------------------------------------


def _floats(values) -> np.ndarray:
    return np.array([math.nan if v is None else float(v) for v in values], dtype=float)


def _pairs(section, key: str, row_of: dict, unit: str) -> tuple[np.ndarray, list]:
    """(loss rows, scales) of the `key` pairs ("pairs" or "failed") of a record's `resampled`."""
    if section is None:
        return np.zeros(0, dtype=int), []
    days = [int(d) for d in section[key]["day"]]
    scales = [float(b) for b in section[key]["scale"]]
    if len(days) != len(scales) or any(d not in row_of for d in days):
        raise ValueError(f"{unit}: a resampled pair is not on an offering calibration day")
    if len(set(zip(days, scales, strict=True))) != len(days):
        raise ValueError(f"{unit}: a resampled pair is listed twice")
    return np.array([row_of[d] for d in days], dtype=int), scales


@dataclass(frozen=True)
class Job:
    """One record of `day_losses`, as arrays. Calibration rows follow `offering_calibration`."""

    unit: str
    configurations: tuple
    calibration_days: np.ndarray  # positions of the calibration days, in day order
    test_days: np.ndarray  # positions of the test days, in day order
    row_of_calibration: np.ndarray  # per calibration day: its loss row, -1 when it does not offer
    scales: tuple  # the run's scales and the resamples' (`resampled`), ascending
    slot: dict  # rung -> (offering calibration days, scales), NaN where a pair is not known
    driver: dict
    offers: np.ndarray  # per day
    value: np.ndarray  # per day, kWh at scale 1
    cars: np.ndarray  # per day
    ceilings: dict  # ceiling name -> per day, NaN where missing
    tests: dict  # rung -> tier -> {"scale", "slot", "driver"} (arrays over test days or None)
    failed: np.ndarray = None  # (offering calibration days, scales): the pair's evaluation failed
    resampled: dict | None = None  # seed, resamples and tiers of the resamples' pairs, if any

    @classmethod
    def from_record(cls, record: dict, rungs) -> "Job":
        days = record["days"]
        roles = list(days["role"])
        n = len(roles)
        calibration = np.array([d for d, r in enumerate(roles) if r == "calibration"], dtype=int)
        test = np.array([d for d, r in enumerate(roles) if r == "test"], dtype=int)
        offers = np.array(days["offers"], dtype=bool)
        value = np.array(days["certified_value_kwh"], dtype=float)
        cars = np.array(days["cars"], dtype=float)
        ceilings = {name: _floats(days["ceiling_kwh"][name]) for name in CEILINGS}
        lengths = {len(offers), len(value), len(cars), *(len(c) for c in ceilings.values())}
        if lengths != {n} or set(roles) - {"calibration", "test"}:
            raise ValueError(f"{record['unit']}: the day lists do not match the roles")
        offering = [int(d) for d in record["calibration"]["days"]]
        if offering != [int(d) for d in calibration if offers[d]]:
            raise ValueError(f"{record['unit']}: the offering calibration days do not match")
        row_of = {d: k for k, d in enumerate(offering)}
        rows = np.array([row_of.get(int(d), -1) for d in calibration], dtype=int)
        run_scales = tuple(float(b) for b in record["calibration"]["scales"])
        if list(run_scales) != sorted(set(run_scales)):
            raise ValueError(f"{record['unit']}: the scales are not ascending and distinct")
        section = record.get("resampled")
        pairs = _pairs(section, "pairs", row_of, record["unit"])
        gone = _pairs(section, "failed", row_of, record["unit"])
        scales = tuple(sorted({*run_scales, *pairs[1], *gone[1]}))
        column = {b: k for k, b in enumerate(scales)}
        own = [column[b] for b in run_scales]
        at = (pairs[0], np.array([column[b] for b in pairs[1]], dtype=int))
        slot, driver = {}, {}
        for rung in rungs:
            for name, store in (("slot_loss", slot), ("driver_loss", driver)):
                table = np.array(record["calibration"][name][rung], dtype=float).reshape(
                    len(offering), len(run_scales)
                )
                # A job whose run evaluated no scale (too few offering calibration days) has an
                # empty table: nothing to check, and every resample abstains by the rule (I2).
                inside = table.size == 0 or (
                    np.isfinite(table).all() and table.min() >= 0 and table.max() <= 1
                )
                if not inside:
                    raise ValueError(f"{record['unit']}: a calibration loss is not in [0, 1]")
                full = np.full((len(offering), len(scales)), math.nan)
                full[:, own] = table
                if section is not None:
                    values = np.array(section["pairs"][name][rung], dtype=float)
                    if values.shape != pairs[0].shape or not np.isnan(full[at]).all():
                        raise ValueError(f"{record['unit']}: the resamples' pairs do not match")
                    if values.size and not (values.min() >= 0 and values.max() <= 1):
                        raise ValueError(f"{record['unit']}: a resampled loss is not in [0, 1]")
                    full[at] = values
                store[rung] = full
        failed = np.zeros((len(offering), len(scales)), dtype=bool)
        failed[gone[0], np.array([column[b] for b in gone[1]], dtype=int)] = True
        if any(not np.isnan(table[failed]).all() for table in (*slot.values(), *driver.values())):
            raise ValueError(f"{record['unit']}: a failed pair also has losses")
        resampled = None
        if section is not None:
            resampled = {
                "seed": int(section["seed"]),
                "resamples": int(section["resamples"]),
                "tiers": tuple(section["tiers"]),
            }
        if [int(d) for d in record["test"]["days"]] != [int(d) for d in test]:
            raise ValueError(f"{record['unit']}: the test days do not match the roles")
        tests = {}
        for rung in rungs:
            tests[rung] = {}
            for tier, found in record["test"]["rungs"][rung].items():
                scale = float(found["scale"])
                entry = {"scale": scale, "slot": None, "driver": None}
                if scale > 0:
                    entry["slot"] = np.array(found["slot_loss"], dtype=float).reshape(len(test))
                    entry["driver"] = np.array(found["driver_loss"], dtype=float).reshape(len(test))
                tests[rung][tier] = entry
        return cls(
            record["unit"],
            tuple(record["configurations"]),
            calibration,
            test,
            rows,
            scales,
            slot,
            driver,
            offers,
            value,
            cars,
            ceilings,
            tests,
            failed,
            resampled,
        )

    def scale(self, rung: str, tier: str, positions) -> Scale:
        """The scale of the calibration days at `positions` (of `calibration_days`), reading I3."""
        rows = self.row_of_calibration[np.asarray(positions, dtype=int)]
        rows = rows[rows >= 0]
        alpha_slot, alpha_driver = tier_alphas(tier)
        return scale_of(
            self.slot[rung][rows],
            self.driver[rung][rows],
            self.scales,
            alpha_slot,
            alpha_driver,
            self.failed[rows],
        )

    def run_scale(self, rung: str, tier: str) -> Scale:
        """The scale recomputed on the run's own calibration days, each once."""
        return self.scale(rung, tier, np.arange(len(self.calibration_days)))


# --- percentiles ----------------------------------------------------------------------------------


def quantiles(values) -> list[float]:
    """The two percentiles of `LEVELS` (reading I7)."""
    found = np.quantile(np.asarray(values, dtype=float), LEVELS, method=QUANTILE_METHOD)
    return [float(found[0]), float(found[1])]


def bracketed(point, low, high) -> dict:
    """A quantity whose resample values lie between `low` and `high` (reading I4). Values that
    are not numbers (left out by I4) are dropped and counted."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    kept = ~(np.isnan(low) | np.isnan(high))
    out = {"point": point, "resamples_left_out": int((~kept).sum())}
    if not kept.any():
        return {**out, "interval": None, "lower_end_between": None, "upper_end_between": None}
    lows, highs = quantiles(low[kept]), quantiles(high[kept])
    return {
        **out,
        "interval": [lows[0], highs[1]],
        "lower_end_between": [lows[0], highs[0]],
        "upper_end_between": [lows[1], highs[1]],
    }


def plain(point, values) -> dict:
    """A quantity known exactly in every resample; values that are not numbers (reading I5) are
    left out and counted."""
    values = np.asarray(values, dtype=float)
    kept = ~np.isnan(values)
    out = {"point": point, "resamples_left_out": int((~kept).sum())}
    return {**out, "interval": quantiles(values[kept]) if kept.any() else None}


def _ratio(top, bottom) -> np.ndarray:
    out = np.full(np.shape(top), math.nan)
    usable = bottom > 0
    np.divide(top, bottom, out=out, where=usable)
    return out


@dataclass(frozen=True)
class Resampled:
    """One job's rung and tier over every resample (readings I1 to I4): the scale bounds and the
    sums over each resample's drawn test days. `ceilings`: ceiling name -> sums, None where a test
    day's ceiling is missing in the run."""

    found: list
    low: np.ndarray
    high: np.ndarray
    values: np.ndarray  # certified value at scale 1, kWh
    cars: np.ndarray
    count: int  # test days drawn per resample
    ceilings: dict


def resample_offer(job: Job, rung: str, tier: str, calibration, test) -> Resampled:
    """The scales and the drawn test days' sums of one rung and tier, for every resample (the one
    place the calibration search is called per resample; `offer_intervals` and the pooled figures
    both read it)."""
    found = [job.scale(rung, tier, row) for row in calibration]
    days = job.test_days[np.asarray(test, dtype=int)]
    ceilings = {}
    for name in CEILINGS:
        missing = np.isnan(job.ceilings[name][job.test_days]).any()
        ceilings[name] = None if missing else job.ceilings[name][days].sum(axis=1)
    return Resampled(
        found,
        np.array([s.low for s in found]),
        np.array([s.high for s in found]),
        job.value[days].sum(axis=1),
        job.cars[days].sum(axis=1),
        days.shape[1],
        ceilings,
    )


def offer_intervals(
    job: Job, rung: str, tier: str, points: dict, calibration, test, drawn: Resampled | None = None
) -> dict:
    """The scale, abstentions and offer intervals of one rung and tier (readings I1 to I4, I7).
    `calibration`, `test`: drawn positions, one row per resample. `points`: the run's own values
    (`point_values`). `drawn`: `resample_offer` of the same arguments, when it is at hand."""
    if drawn is None:
        drawn = resample_offer(job, rung, tier, calibration, test)
    found, low, high = drawn.found, drawn.low, drawn.high
    values, cars, count = drawn.values, drawn.cars, drawn.count
    shares = {}
    for name in CEILINGS:
        ceiling = drawn.ceilings[name]
        if ceiling is None:
            shares[name] = {
                "point": points["share_of_ceiling"][name],
                "interval": None,
                "reason": "a test day's ceiling is missing in the run",
            }
            continue
        shares[name] = bracketed(
            points["share_of_ceiling"][name],
            _ratio(low * values, ceiling),
            _ratio(high * values, ceiling),
        )
    resamples = len(found)
    return {
        "scale": {
            "point": points["scale"],
            "resamples": resamples,
            "resamples_exact": sum(s.exact for s in found),
            "resamples_with_failed_pairs": sum(s.failed_pair for s in found),
            "resamples_with_pairs_not_evaluated": sum(s.missing_pair for s in found),
            "interval": [quantiles(low)[0], quantiles(high)[1]],
        },
        "abstaining": {
            "resamples_at_scale_zero": int((high <= 0).sum()),
            "resamples_possibly_at_scale_zero": int((low <= 0).sum()),
            "resamples_too_few_offering_calibration_days": sum(s.too_few for s in found),
            "share_of_resamples_abstaining": float((high <= 0).sum() / resamples),
            "share_of_resamples_possibly_abstaining": float((low <= 0).sum() / resamples),
        },
        "offer_kwh_per_counting_test_day": bracketed(
            points["offer_kwh_per_counting_test_day"], low * values / count, high * values / count
        ),
        "offer_kwh_per_car_per_day": bracketed(
            points["offer_kwh_per_car_per_day"],
            _ratio(low * values, cars),
            _ratio(high * values, cars),
        ),
        "share_of_ceiling": shares,
    }


def loss_intervals(job: Job, rung: str, tier: str, points: dict, test) -> dict:
    """Availability, shortfall, slot and driver loss at the run's scale over the drawn test days
    that offer (readings I5, I6)."""
    found = job.tests[rung][tier]
    names = ("availability", "shortfall", "slot_loss", "driver_loss")
    if found["slot"] is None:
        reason = "the run's scale is 0: nothing is offered, so there is no loss"
        return {name: {"point": None, "interval": None, "reason": reason} for name in names}
    positions = np.asarray(test, dtype=int)
    offering = job.offers[job.test_days][positions]
    counts = offering.sum(axis=1)
    slot = _ratio((found["slot"][positions] * offering).sum(axis=1), counts)
    driver = _ratio((found["driver"][positions] * offering).sum(axis=1), counts)
    return {
        "availability": plain(points["availability"], 1.0 - slot),
        "shortfall": plain(points["shortfall"], driver),
        "slot_loss": plain(points["slot_loss"], slot),
        "driver_loss": plain(points["driver_loss"], driver),
    }


# --- pooled over the units of a configuration ----------------------------------------------------

FIGURES = ("sum_of_unit_offers", "total_over_total_days")  # readings I13, I14


def _figures(offered, cars, ceilings, counts) -> dict:
    """Both pooled figures (readings I13, I14) of an offer, per car and as a share of each ceiling
    from the units' offered kWh, cars and ceilings over their test days (`ceilings`: name -> one
    entry per unit, None where a unit lacks one). The same arithmetic serves the run's own values
    (floats) and a bound of every resample (arrays); a ratio with a denominator of 0 is NaN."""
    per_day = sum(o / c for o, c in zip(offered, counts, strict=True))
    total = sum(offered)
    days = sum(counts)
    car_days = sum(k / c for k, c in zip(cars, counts, strict=True))
    out = {
        FIGURES[0]: {
            "offer_kwh_per_counting_test_day": np.asarray(per_day, dtype=float),
            "offer_kwh_per_car_per_day": _ratio(per_day, car_days),
            "share_of_ceiling": {},
        },
        FIGURES[1]: {
            "offer_kwh_per_counting_test_day": np.asarray(total / days, dtype=float),
            "offer_kwh_per_car_per_day": _ratio(total, sum(cars)),
            "share_of_ceiling": {},
        },
    }
    for name, found in ceilings.items():
        if any(c is None for c in found):
            continue
        out[FIGURES[0]]["share_of_ceiling"][name] = _ratio(
            per_day, sum(c / n for c, n in zip(found, counts, strict=True))
        )
        out[FIGURES[1]]["share_of_ceiling"][name] = _ratio(total, sum(found))
    return out


def pooled_intervals(jobs, points, drawn) -> dict:
    """The two pooled figures of one rung and tier over the units of a configuration, with their
    intervals (readings I13 to I16). `jobs`: one `Job` per unit; `points`: each unit's own values
    (`point_values`, same order); `drawn`: each unit's `resample_offer` (same order, one draw per
    unit and resample number, so resample r combines resample r of every unit)."""
    counts = [d.count for d in drawn]
    names = {name: [d.ceilings[name] for d in drawn] for name in CEILINGS}
    own_ceilings = {}
    for name in CEILINGS:
        test = [j.ceilings[name][j.test_days] for j in jobs]
        own_ceilings[name] = [None if np.isnan(t).any() else float(t.sum()) for t in test]
    own = _figures(
        [p["offer_kwh_per_counting_test_day"] * c for p, c in zip(points, counts, strict=True)],
        [float(j.cars[j.test_days].sum()) for j in jobs],
        own_ceilings,
        counts,
    )
    bounds = {}
    for side in ("low", "high"):
        bounds[side] = _figures(
            [getattr(d, side) * d.values for d in drawn], [d.cars for d in drawn], names, counts
        )
    out = {
        "resamples": len(drawn[0].found),
        "resamples_every_scale_exact": int(
            np.all([[s.exact for s in d.found] for d in drawn], axis=0).sum()
        ),
    }
    for figure in FIGURES:
        shown = {}
        for key in ("offer_kwh_per_counting_test_day", "offer_kwh_per_car_per_day"):
            point = float(own[figure][key])
            shown[key] = bracketed(
                None if np.isnan(point) else point,
                bounds["low"][figure][key],
                bounds["high"][figure][key],
            )
        shown["share_of_ceiling"] = {}
        for name in CEILINGS:
            if name not in own[figure]["share_of_ceiling"]:
                shown["share_of_ceiling"][name] = {
                    "point": None,
                    "interval": None,
                    "reason": "a test day's ceiling is missing in the run",
                }
                continue
            point = float(own[figure]["share_of_ceiling"][name])
            shown["share_of_ceiling"][name] = bracketed(
                None if np.isnan(point) else point,
                bounds["low"][figure]["share_of_ceiling"][name],
                bounds["high"][figure]["share_of_ceiling"][name],
            )
        out[figure] = shown
    return out


def point_values(entry: dict) -> dict:
    """The run's own values of one rung and tier (`gate_calibrate`'s test summary `entry`, with
    the scale under "scale")."""
    test = entry["test"]
    slot = test["mean_slot_loss_offering_days"]
    driver = test["mean_driver_loss_offering_days"]
    shares = test["share_of_ceiling"]
    return {
        "scale": float(entry["beta_hat"]),
        "offer_kwh_per_counting_test_day": test["mean_offer_kwh_per_day"],
        "offer_kwh_per_car_per_day": test["offer_kwh_per_car_per_day"],
        "share_of_ceiling": {
            name: None if shares[name] is None else shares[name]["ratio_of_sums"]
            for name in CEILINGS
        },
        "availability": None if slot is None else 1.0 - slot,
        "shortfall": driver,
        "slot_loss": slot,
        "driver_loss": driver,
    }

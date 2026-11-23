"""The arithmetic of the ceiling report: summaries and checks of `ceiling_values.DayResult`s.

Block-product sample (the sampled days). V_caus is the corner tree's value, V_aff the causal
affine certificate's, V_diag the diagonal one's, V_hind `exact.block_ceiling`.

- The ratio V_caus / V_hind is formed on days where both values are present and V_hind is above
  `ZERO` kWh. A day with V_hind = 0 (reading, marked) cannot form a ratio: it is counted and left
  out of the mean, median and minimum. A day with a missing value (a solve that ended in a
  timeout or any other outcome but optimal) is counted separately and left out too; a missing
  value is never read as zero.
- "Below" and "differs" are judged with the relative tolerance `REL_TOL`, taken against
  max(1, the larger reference value), in kWh.
- T4 check: on every sampled day meeting T4's hypotheses (one-way, no in-window departure,
  enough recovery supply; an empty fleet does not count), V_hind = V_caus within `REL_TOL`.
  A day with a missing value cannot be checked and is counted as such. A violation fails the
  check.
- Order check (added to the plan's checks): V_diag <= V_aff <= V_caus <= V_hind on every sampled
  day, V_caus (tree) = V_causal hourly on every sampled day, and the hourly best causal value
  <= the hourly ceiling on every day, each within `REL_TOL`. The plan states these orders; a
  violation fails the check.
- Not decisions: V_aff < V_caus is the plan's open conjecture (T4); the days where it happens
  are listed. Whether the plan's rule applies is not decided here.

Outcome O2 on every day (`o2_summary`, plan section 11: "block product, M days with |S|<=8:
V_caus/V_hind mean X%, median Q%; P% on days with an in-window departure; V_aff=V_caus on K.
Pass: V_caus=V_hind on every one-way day meeting T4's hypotheses"), over one unit's counting
days, the M days being those whose `sampled` is set (`ceiling_report.every_day_tasks`):
- the ratio as above over the M days (mean, median, minimum; who is left out is counted);
- the M days with an in-window departure: their number, their share of the M days and the ratio
  over them; the ratio over the others beside it (reading P below);
- K: the M days where V_aff and V_caus are both present and do not differ (`differs`), out of
  the days where both are present, the days with a missing value counted;
- the T4 check over the M days and O2's pass from it (`o2_pass`): "failed" when V_hind differs
  from V_caus on any M day meeting T4's hypotheses; else "could not run" when any M day could
  not be built (`error` set, its text empty or not), whatever the unit's other days; else "no day
  to check" when no M day meets them, the unit with no M day (or no counting day) included,
  decided from those day counts alone (reading N); else "could not run" when such a day has a
  missing value (a timeout included); else "passed". Never a pass on nothing checked;
- the counting days outside M (more than 8 service hours) and how many of them meet T4's
  hypotheses: counted, not checked;
- the command's own faults on the unit's days (`o2_with_faults`, `ceiling_report` reading O2-6,
  decision 85 (1)): every broken order and every day not built counted in `faults`; O2's pass
  changed only by those on M days, a block order broken there making it "failed" and an M day
  not built as above. An hourly order broken on any day, a day outside M not built: counted,
  never in the pass. [Any broken order "failed", else any day not built, in M or not, "could
  not run".]
Readings (alternatives in brackets): P. "P% on days with an in-window departure" is read as the
ratio V_caus/V_hind over the M days with an in-window departure; their share of the M days is
given beside it. [P as that share alone.] K is a count of days, given with the days it is out
of. [K as a share.] N. One verdict over several units (`pass_over`; the owner's decision of 19
November 2026, option (b)): it is judged on the units that have an M
day meeting T4's hypotheses; a unit whose pass is "no day to check" never counts as passed or
failed and is listed by name beside the verdict (`ceiling_report`); with every unit given "no
day to check", the verdict over the units is "no day to check" too, never "could not run" (the
owner's words below); with no unit given at all (an empty input, not a unit with no day) it is
"could not run". A unit with no M day at all (no counting day, or every counting
day outside M) has no day meeting T4's hypotheses to check, so it is "no day to check" too (the
owner's words: "a unit with none is listed as 'no day to check' beside it, never turning the
verdict into 'could not run'"). A unit with an M day that could not be built (whatever its other
days) is "could not run", not "no day to check" nor "passed", since what that day would show is
not known; so then is the verdict over the units, unless one failed. [A unit with no M day as
"could not run".] A day outside M that meets T4's hypotheses does
not make a unit checkable (reading O2-3 of `ceiling_report`: counted, not checked). [Such a day
making the unit "could not run".]

Hourly ceilings (every counting day of a unit): the sums, means, ratio of sums, mean and median
of the per-day ratios over days with a positive hindsight ceiling and a causal value, the share
of days where the two differ, and the share with a zero causal value but a positive ceiling.
Shares and means are over the days with both values; days with a missing causal value are counted.
Each is given for all days, for days with an in-window departure and for days without.
"""

import statistics

from fleetreserve.ceiling_values import DayResult

REL_TOL = 1e-6
ZERO = 1e-7  # kWh: a value at or below this is zero
THRESHOLD = 0.99  # plan 13 WP2, owner's rule

GROUPS = ("all days", "in-window departure", "no in-window departure")


def _r(value, digits: int = 6):
    return None if value is None else round(float(value), digits)


def label(result: DayResult) -> str:
    return f"{result.unit} {result.site} {result.date}"


def differs(a: float, b: float) -> bool:
    return abs(a - b) > REL_TOL * max(1.0, abs(a), abs(b))


def below(a: float, b: float) -> bool:
    """True when `a` is below `b` by more than the tolerance."""
    return a < b - REL_TOL * max(1.0, abs(a), abs(b))


def _spread(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "mean": _r(statistics.fmean(values)),
        "median": _r(statistics.median(values)),
        "min": _r(min(values)),
    }


def missing_counts(results: list[DayResult], key: str) -> dict:
    """Days asked for `key` whose value is missing: timeouts and every other outcome."""
    asked = [r for r in results if r.outcomes[key] != "not solved"]
    gone = [r for r in asked if r.values[key] is None]
    timeout = sum(r.outcomes[key] == "timeout" for r in gone)
    return {"asked": len(asked), "timeout": timeout, "other": len(gone) - timeout}


def _both(results: list[DayResult], a: str, b: str):
    return [r for r in results if r.values[a] is not None and r.values[b] is not None]


def ratio_block(results: list[DayResult]) -> dict:
    """V_caus / V_hind over the given days (see the module text for who is left out)."""
    both = _both(results, "block_caus", "block_hind")
    positive = [r for r in both if r.values["block_hind"] > ZERO]
    ratios = [r.values["block_caus"] / r.values["block_hind"] for r in positive]
    return {
        "days": len(results),
        "days_in_ratio": len(ratios),
        "days_with_zero_hindsight": len(both) - len(positive),
        "days_with_a_missing_value": len(results) - len(both),
        **(_spread(ratios) or {"mean": None, "median": None, "min": None}),
    }


def _pairs_below(results, lower: str, upper: str) -> dict:
    both = _both(results, lower, upper)
    hits = [r for r in both if below(r.values[lower], r.values[upper])]
    return {
        "of_days": len(both),
        "days": len(hits),
        "list": [
            {
                "day": label(r),
                lower: _r(r.values[lower]),
                upper: _r(r.values[upper]),
                "cars": r.cars,
                "service_hours": r.n_service,
                "in_window_departure": r.in_window,
                "t4_hypotheses": r.t4_hypotheses,
            }
            for r in hits
        ],
    }


def t4_check(results: list[DayResult]) -> dict:
    """V_hind = V_caus on the sampled days meeting T4's hypotheses."""
    days = [r for r in results if r.t4_hypotheses]
    both = _both(days, "block_hind", "block_caus")
    broken = [r for r in both if differs(r.values["block_hind"], r.values["block_caus"])]
    return {
        "days_meeting_hypotheses": len(days),
        "days_checked": len(both),
        "days_not_checked_missing_value": len(days) - len(both),
        "violations": [
            {
                "day": label(r),
                "block_hind": _r(r.values["block_hind"]),
                "block_caus": _r(r.values["block_caus"]),
            }
            for r in broken
        ],
        "held": not broken,
    }


BLOCK, HOURLY = "block", "hourly"  # the product an order is of (`order_breaks`)


def order_breaks(results: list[DayResult]) -> list[tuple[str, str]]:
    """The orders the plan states, broken beyond the tolerance: (product, line) each, the product
    `BLOCK` (V_diag <= V_aff <= V_caus <= V_hind, on sampled days) or `HOURLY` (the hourly tree =
    the best causal value, on sampled days; the best causal value <= the hourly ceiling)."""
    found = []
    for r in results:
        v = r.values
        if r.sampled:
            for lower, upper in (
                ("block_diag", "block_aff"),
                ("block_aff", "block_caus"),
                ("block_caus", "block_hind"),
            ):
                if None not in (v[lower], v[upper]) and below(v[upper], v[lower]):
                    found.append(
                        (BLOCK, f"{label(r)}: {lower} {v[lower]:.6g} above {upper} {v[upper]:.6g}")
                    )
            if None not in (v["hourly_tree"], v["hourly_causal"]) and differs(
                v["hourly_tree"], v["hourly_causal"]
            ):
                found.append(
                    (
                        HOURLY,
                        f"{label(r)}: hourly tree {v['hourly_tree']:.6g} differs from best causal "
                        f"{v['hourly_causal']:.6g}",
                    )
                )
        if None not in (v["hourly_causal"], v["hourly_hind"]) and below(
            v["hourly_hind"], v["hourly_causal"]
        ):
            found.append(
                (
                    HOURLY,
                    f"{label(r)}: hourly best causal {v['hourly_causal']:.6g} above the ceiling "
                    f"{v['hourly_hind']:.6g}",
                )
            )
    return found


def order_violations(results: list[DayResult]) -> list[str]:
    """The orders the plan states, broken beyond the tolerance, one line each, both products."""
    return [line for _, line in order_breaks(results)]


def block_summary(sampled: list[DayResult]) -> dict:
    groups = {
        "in-window departure": [r for r in sampled if r.in_window],
        "no in-window departure": [r for r in sampled if not r.in_window],
        "meets T4's hypotheses": [r for r in sampled if r.t4_hypotheses],
    }
    keys = ("block_hind", "block_caus", "block_aff", "block_diag", "hourly_tree")
    overall = ratio_block(sampled)
    mean = overall["mean"]
    statement = (
        f"mean V_caus/V_hind = {mean:.6f}; the owner's rule threshold is {THRESHOLD}"
        if mean is not None
        else f"mean V_caus/V_hind = none (no day forms a ratio); the owner's rule threshold is "
        f"{THRESHOLD}"
    )
    return {
        "days": len(sampled),
        "empty_fleet_days": sum(r.cars == 0 for r in sampled),
        "ratio_caus_over_hind": overall,
        "ratio_by_group": {name: ratio_block(rs) for name, rs in groups.items()},
        "caus_below_hind": _pairs_below(sampled, "block_caus", "block_hind"),
        "aff_below_caus": _pairs_below(sampled, "block_aff", "block_caus"),
        "diag_below_aff": {
            k: v for k, v in _pairs_below(sampled, "block_diag", "block_aff").items() if k != "list"
        },
        "missing": {key: missing_counts(sampled, key) for key in keys},
        "t4_check": t4_check(sampled),
        "statement": statement,
    }


PASSED, FAILED, NOT_RUN = "passed", "failed", "could not run"  # O2's pass (module text)
NO_DAY = "no day to check"  # a unit's O2 pass when no M day of it meets T4's hypotheses


def o2_pass(check: dict, days_not_built: int) -> str:
    """O2's pass from a `t4_check` of a unit's M days and the number of them that could not be
    built (module text, reading N): day counts alone decide it."""
    if check["violations"]:
        return FAILED
    if days_not_built:
        return NOT_RUN
    if check["days_meeting_hypotheses"] == 0:
        return NO_DAY
    if check["days_checked"] == 0 or check["days_not_checked_missing_value"]:
        return NOT_RUN
    return PASSED


def pass_over(verdicts) -> str:
    """One verdict over several units' O2 passes (reading N): failed if any failed; else no day
    to check when every pass given is "no day to check" (at least one given); else passed when
    every judged unit passed; else could not run (none given included)."""
    verdicts = list(verdicts)
    if FAILED in verdicts:
        return FAILED
    judged = [v for v in verdicts if v != NO_DAY]
    if verdicts and not judged:
        return NO_DAY
    return PASSED if judged and all(v == PASSED for v in judged) else NOT_RUN


def no_day_units(passes: dict) -> list[str]:
    """The names, in the order given, of the units whose O2 pass is "no day to check"."""
    return [name for name, verdict in passes.items() if verdict == NO_DAY]


def aff_equals_caus(results: list[DayResult]) -> dict:
    """K: the days where V_aff and V_caus are both present and do not differ (module text)."""
    both = _both(results, "block_aff", "block_caus")
    equal = [r for r in both if not differs(r.values["block_aff"], r.values["block_caus"])]
    return {
        "days": len(equal),
        "of_days": len(both),
        "days_with_a_missing_value": len(results) - len(both),
    }


def o2_summary(results: list[DayResult]) -> dict:
    """Outcome O2 over one unit's counting days, its M days those with `sampled` (module text)."""
    m_days = [r for r in results if r.sampled]
    outside = [r for r in results if not r.sampled]
    inside = [r for r in m_days if r.in_window]
    others = [r for r in m_days if not r.in_window]
    check = t4_check(m_days)
    not_built = sum(r.error is not None for r in m_days)
    keys = ("block_hind", "block_caus", "block_aff", "block_diag", "hourly_tree")
    return {
        "counting_days": len(results),
        "days_in_m": len(m_days),
        "days_in_m_not_built": not_built,
        "days_outside_m": len(outside),
        "hypotheses_days_outside_m": sum(r.t4_hypotheses for r in outside),
        "empty_fleet_days": sum(r.cars == 0 for r in m_days),
        "ratio_caus_over_hind": ratio_block(m_days),
        "in_window_departure": {
            "days": len(inside),
            "share_of_m": _r(len(inside) / len(m_days)) if m_days else None,
            "ratio_caus_over_hind": ratio_block(inside),
        },
        "no_in_window_departure": {
            "days": len(others),
            "ratio_caus_over_hind": ratio_block(others),
        },
        "aff_equals_caus": aff_equals_caus(m_days),
        "missing": {key: missing_counts(m_days, key) for key in keys},
        "t4_check": check,
        "o2_pass": o2_pass(check, not_built),
    }


def _group_hourly(results: list[DayResult]) -> dict:
    both = _both(results, "hourly_hind", "hourly_causal")
    hind = [r.values["hourly_hind"] for r in both]
    causal = [r.values["hourly_causal"] for r in both]
    positive = [r for r in both if r.values["hourly_hind"] > ZERO]
    ratios = [r.values["hourly_causal"] / r.values["hourly_hind"] for r in positive]
    differing = sum(differs(h, c) for h, c in zip(hind, causal, strict=True))
    zero_causal = sum(r.values["hourly_causal"] <= ZERO for r in positive)

    def share(part):
        return _r(part / len(both)) if both else None

    return {
        "days": len(results),
        "days_with_both_values": len(both),
        "days_with_a_missing_causal_value": len(results) - len(both),
        "sum_hindsight_kwh": _r(sum(hind)),
        "sum_best_causal_kwh": _r(sum(causal)),
        "mean_hindsight_kwh_per_day": _r(statistics.fmean(hind)) if both else None,
        "mean_best_causal_kwh_per_day": _r(statistics.fmean(causal)) if both else None,
        "ratio_of_sums": _r(sum(causal) / sum(hind)) if sum(hind) > ZERO else None,
        "days_with_positive_hindsight": len(positive),
        "per_day_ratio_mean": _r(statistics.fmean(ratios)) if ratios else None,
        "per_day_ratio_median": _r(statistics.median(ratios)) if ratios else None,
        "share_days_differing": share(differing),
        "share_days_zero_causal_positive_hindsight": share(zero_causal),
    }


def hourly_summary(results: list[DayResult]) -> dict:
    """One unit's hourly ceilings, for all days and split by in-window departure."""
    parts = {
        GROUPS[0]: results,
        GROUPS[1]: [r for r in results if r.in_window],
        GROUPS[2]: [r for r in results if not r.in_window],
    }
    return {name: _group_hourly(rs) for name, rs in parts.items()}


def _not_built(everything: list[DayResult]) -> list[str]:
    """One line per day whose `error` is set, its text empty or not (as `o2_summary` counts)."""
    return [
        f"{label(r)}: the day could not be built: {r.error}"
        for r in everything
        if r.error is not None
    ]


def faults(everything: list[DayResult]) -> list[str]:
    """The lines of `failures` that are not the T4 equality: days that could not be built and
    the orders (with `--every-day` the T4 equality is O2's pass, `ceiling_report` reading O2-4)."""
    return _not_built(everything) + order_violations(everything)


def failures(sampled: list[DayResult], everything: list[DayResult]) -> list[str]:
    """Lines for every broken check: the T4 equality, the orders, days that could not be built."""
    lines = _not_built(everything)
    check = t4_check(sampled)
    lines += [
        f"{v['day']}: V_hind {v['block_hind']} differs from V_caus {v['block_caus']} on a day "
        "meeting T4's hypotheses"
        for v in check["violations"]
    ]
    lines += order_violations(everything)
    return lines


def o2_with_faults(results: list[DayResult]) -> dict:
    """`o2_summary` of one unit's counting days with the command's own faults on them counted
    (`ceiling_report` reading O2-6, decision 85 (1)): `faults` gives the number of broken orders
    (`order_violations`, both products, every day) and of days that could not be built, M days
    or not, and apart the block orders broken on M days. O2's pass is changed only by faults on
    M days: a block order broken on an M day makes it "failed"; an M day not built already makes
    a pass other than "failed" "could not run" (`o2_pass`). An hourly order broken on any day and
    a day outside M not built are counted, never in the pass."""
    found = o2_summary(results)
    breaks, not_built = order_violations(results), _not_built(results)
    m_days = [r for r in results if r.sampled]
    block_on_m = [line for product, line in order_breaks(m_days) if product == BLOCK]
    found["faults"] = {
        "order_breaks": len(breaks),
        "days_not_built": len(not_built),
        "block_order_breaks_on_m_days": len(block_on_m),
    }
    if block_on_m:
        found["o2_pass"] = FAILED
    return found

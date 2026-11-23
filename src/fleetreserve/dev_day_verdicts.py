"""The verdicts of the WP5 development-day checks, from the per-day results (plan sections 4, 5, 9
and 13 WP5). Each check gives `passed`, `failed` or `could not run`, the lines that explain it,
and how many days it covered out of how many it should have.

Coverage rule (missing never passes silently). A day a check should cover is covered when every
value the check needs is present. Otherwise it is missing: a timeout when a needed solve ended in
a timeout, else "other" (any other outcome, or a day whose sets could not be built). A check with
a failure line has failed; else a check that covered fewer than all its days could not run (a
timeout is missing, never a pass, plan section 13); else it passed.

Tolerances: "below" and "differs" are those of `ceiling_summary` (relative 1e-6, against
max(1, the larger value), in kWh).

- `order_check`: on every counting day, per product, every pair of the four values in the order
  diagonal <= causal affine <= tree <= hindsight (`dev_day_values`) that are both present. A day
  whose sets could not be built is a failure line.
- `t4_log`: the block product on every counting day: whether the day meets T4's hypotheses and
  has an in-window departure, the gaps V_hind - V_caus (tree) and V_caus - V_aff; summary counts
  in three groups (days meeting the hypotheses; other days without an in-window departure; other
  days with one). V_hind = V_caus on every day meeting the hypotheses is checked (plan T4); a
  day with V_caus above V_aff is a recorded finding about the plan's open conjecture and is
  listed, never a failure.
- `t2_check`: T2's check on the sampled days (`t2_day`). Covered: a certified day, or a day with
  no certificate by the configured rule (no steered car, or the day abstains). Reading, marked:
  the check could not run when no day was certified or no day is in E', because a check that
  looked at nothing is no evidence. The shares of days in E' and of steered cars in E'_i are
  given with every certified count.
- `two_way_check`: the two-way corner tests on the sampled days (`two_way_day`): replay and the
  enlarged box per product, and the order diagonal <= causal <= tree <= ceiling.
"""

import statistics
from itertools import combinations

from fleetreserve.ceiling_summary import below, differs
from fleetreserve.dev_day_values import BLOCK_ORDER, HOURLY_ORDER, OrderValues
from fleetreserve.t2_day import CERTIFIED, T2Result
from fleetreserve.two_way_day import PRODUCTS, VALUE_NAMES, TwoWayResult

PASSED = "passed"
FAILED = "failed"
COULD_NOT_RUN = "could not run"
T4_GROUPS = (
    "meets T4's hypotheses",
    "other days, no in-window departure",
    "other days, in-window departure",
)


def label(result) -> str:
    return f"{result.unit} {result.site} {result.date}"


def _r(value, digits: int = 6):
    return None if value is None else round(float(value), digits) + 0.0  # -0.0 becomes 0.0


def verdict(lines: list[str], of: int, covered: int, timeouts: int, other: int, extra=None):
    """The verdict dict of a check; `extra` is a reason it could not run despite full cover."""
    if lines:
        result = FAILED
    elif of == 0 or covered < of or extra:
        result = COULD_NOT_RUN
    else:
        result = PASSED
    reasons = []
    if covered < of:
        reasons.append(
            f"covered {covered} of {of} days: {timeouts} with a timeout, {other} with another "
            "missing value or a day that could not be built"
        )
    if of == 0:
        reasons.append("no day to check")
    if extra:
        reasons.append(extra)
    return {
        "verdict": result,
        "of_days": of,
        "covered_days": covered,
        "timeout_days": timeouts,
        "other_missing_days": other,
        "failure_lines": lines,
        "could_not_run_reasons": reasons if result == COULD_NOT_RUN else [],
    }


def _missing_kind(values: dict, outcomes: dict, keys) -> str | None:
    """None when no value of `keys` is missing, else "timeout" (a needed solve timed out) or
    "other"."""
    gone = [k for k in keys if values[k] is None]
    if not gone:
        return None
    return "timeout" if any(outcomes[k] == "timeout" for k in gone) else "other"


def _day_missing(r: OrderValues, keys) -> str | None:
    if r.error is not None:
        return "other"
    return _missing_kind(r.values, r.outcomes, keys)


def _order_lines(name: str, values: dict, order) -> list[str]:
    lines = []
    for lower, upper in combinations(order, 2):
        a, b = values[lower], values[upper]
        if a is not None and b is not None and below(b, a):
            lines.append(f"{name}: {lower} {a:.6g} above {upper} {b:.6g}")
    return lines


def order_check(results: list[OrderValues]) -> dict:
    lines, timeouts, other = [], 0, 0
    for r in results:
        if r.error is not None:
            lines.append(f"{label(r)}: the day could not be built: {r.error}")
        lines += _order_lines(label(r), r.values, HOURLY_ORDER)
        lines += _order_lines(label(r), r.values, BLOCK_ORDER)
        kind = _day_missing(r, HOURLY_ORDER + BLOCK_ORDER)
        timeouts += kind == "timeout"
        other += kind == "other"
    return verdict(lines, len(results), len(results) - timeouts - other, timeouts, other)


def t4_rows(results: list[OrderValues]) -> list[dict]:
    rows = []
    for r in results:
        v = r.values
        hind, caus, aff = v["block_hind"], v["block_caus"], v["block_aff"]
        rows.append(
            {
                "day": label(r),
                "meets_hypotheses": r.t4_hypotheses,
                "in_window_departure": r.in_window,
                "block_hind": _r(hind),
                "block_caus": _r(caus),
                "block_aff": _r(aff),
                "gap_hind_minus_tree": None if None in (hind, caus) else _r(hind - caus),
                "gap_tree_minus_affine": None if None in (caus, aff) else _r(caus - aff),
            }
        )
    return rows


def _group(results: list[OrderValues]) -> dict:
    full = [r for r in results if None not in (r.values[k] for k in ("block_hind", "block_caus"))]
    pairs = [r for r in results if None not in (r.values[k] for k in ("block_caus", "block_aff"))]
    tree_gap = [r.values["block_hind"] - r.values["block_caus"] for r in full]
    aff_gap = [r.values["block_caus"] - r.values["block_aff"] for r in pairs]
    return {
        "days": len(results),
        "days_with_hind_and_tree": len(full),
        "days_tree_below_hind": sum(
            below(r.values["block_caus"], r.values["block_hind"]) for r in full
        ),
        "mean_gap_hind_minus_tree_kwh": _r(statistics.fmean(tree_gap)) if tree_gap else None,
        "days_with_tree_and_affine": len(pairs),
        "days_affine_below_tree": sum(
            below(r.values["block_aff"], r.values["block_caus"]) for r in pairs
        ),
        "mean_gap_tree_minus_affine_kwh": _r(statistics.fmean(aff_gap)) if aff_gap else None,
    }


def t4_log(results: list[OrderValues]) -> dict:
    groups = {
        T4_GROUPS[0]: [r for r in results if r.t4_hypotheses],
        T4_GROUPS[1]: [r for r in results if not r.t4_hypotheses and not r.in_window],
        T4_GROUPS[2]: [r for r in results if not r.t4_hypotheses and r.in_window],
    }
    keys = ("block_hind", "block_caus", "block_aff")
    lines, findings, timeouts, other = [], [], 0, 0
    for r in results:
        kind = _day_missing(r, keys)
        timeouts += kind == "timeout"
        other += kind == "other"
        v = r.values
        both = None not in (v["block_hind"], v["block_caus"])
        if r.t4_hypotheses and both and differs(v["block_hind"], v["block_caus"]):
            lines.append(
                f"{label(r)}: V_hind {v['block_hind']:.6g} differs from V_caus "
                f"{v['block_caus']:.6g} on a day meeting T4's hypotheses"
            )
        if None not in (v["block_caus"], v["block_aff"]) and below(v["block_aff"], v["block_caus"]):
            findings.append(
                f"{label(r)}: V_caus (tree) {v['block_caus']:.6g} above V_aff "
                f"{v['block_aff']:.6g} (in-window departure: {r.in_window}; meets T4's "
                f"hypotheses: {r.t4_hypotheses})"
            )
    hypothesis_days = groups[T4_GROUPS[0]]
    checked = [
        r for r in hypothesis_days if None not in (r.values["block_hind"], r.values["block_caus"])
    ]
    found = verdict(lines, len(results), len(results) - timeouts - other, timeouts, other)
    found |= {
        "groups": {name: _group(rs) for name, rs in groups.items()},
        "hypothesis_days": len(hypothesis_days),
        "hypothesis_days_checked": len(checked),
        "findings_tree_above_affine": findings,
    }
    return found


def t2_check(results: list[T2Result], of: int, unbuilt: int) -> dict:
    """`unbuilt`: sampled days whose inputs could not be built (counted as other missing)."""
    certified = [r for r in results if r.status == CERTIFIED]
    by_rule = [r for r in results if r.status.startswith("no certificate")]
    timeouts = sum(r.status == "timeout" for r in results)
    other = len(results) - len(certified) - len(by_rule) - timeouts + unbuilt
    lines = [line for r in results for line in r.problems]
    in_e_prime = sum(r.day_in_e_prime for r in certified)
    steered = sum(r.steered for r in certified)
    steered_e = sum(r.steered_in_e_prime for r in certified)
    extra = None
    if not certified:
        extra = "no sampled day was certified: nothing was checked (reading)"
    elif not in_e_prime:
        extra = "no certified day is in E': 'no violation where E' holds' looked at no day"
    found = verdict(lines, of, len(certified) + len(by_rule), timeouts, other, extra)

    def share(part, whole):
        return _r(part / whole) if whole else None

    found |= {
        "certified_days": len(certified),
        "days_without_certificate_by_rule": {
            status: sum(r.status == status for r in by_rule)
            for status in sorted({r.status for r in by_rule})
        },
        "days_in_e_prime": in_e_prime,
        "share_of_sampled_days_in_e_prime": share(in_e_prime, of),
        "share_of_certified_days_in_e_prime": share(in_e_prime, len(certified)),
        "steered_cars_on_certified_days": steered,
        "share_of_steered_cars_in_e_prime_i": share(steered_e, steered),
        "cars_on_sampled_days": sum(r.cars for r in results),
        "cars_left_unsteered_no_forecast": sum(r.cars_without_forecast for r in results),
        "cars_left_unsteered_empty_forecast_set": sum(r.cars_with_empty_set for r in results),
        "forecast_rows_checked": sum(r.rows_checked for r in certified),
        "replay_mismatches": sum(r.replay_mismatches for r in certified),
        "cars_checked_in_e_prime_i": sum(r.cars_checked_in_e_prime for r in certified),
        "true_row_violations_in_e_prime_i": sum(
            r.true_row_violations_in_e_prime for r in certified
        ),
        "cars_outside_e_prime_i_breaking_a_true_row": sum(
            r.cars_outside_e_prime_breaking_a_true_row for r in certified
        ),
        "paths_with_request_missed": sum(r.paths_with_request_missed for r in certified),
    }
    return found


def two_way_check(results: list[TwoWayResult], of: int, unbuilt: int) -> dict:
    """`unbuilt`: sampled days whose inputs could not be built (counted as other missing)."""
    lines, timeouts, other, matched = [], 0, unbuilt, 0
    for r in results:
        if r.error is not None:
            other += 1
            continue
        kinds = []
        for name in PRODUCTS:
            test = r.tests[name]
            values = {f"{name}_{k}": test.values[k] for k in VALUE_NAMES}
            outcomes = {f"{name}_{k}": test.outcomes[k] for k in VALUE_NAMES}
            lines += _order_lines(label(r), values, list(values))
            if test.matched is False:
                lines.append(
                    f"{label(r)}: {name} tree does not match: replay failed on "
                    f"{test.replay_failures} of {test.replay_paths} corner paths (least slack "
                    f"{test.least_slack}, largest request miss {test.largest_request_miss}), "
                    f"scale with the box's shape fixed {test.enlarged_scale}"
                )
            kind = _missing_kind(values, outcomes, list(values))
            if kind is None and test.matched is None:
                kind = "timeout" if test.enlarged_outcome == "timeout" else "other"
            kinds.append(kind)
        if "timeout" in kinds:
            timeouts += 1
        elif "other" in kinds:
            other += 1
        matched += all(r.tests[name].matched is True for name in PRODUCTS)
    found = verdict(lines, of, of - timeouts - other, timeouts, other)
    found["days_both_products_matched"] = matched
    return found

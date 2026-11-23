"""The results document and the summary text of `t7_test`.

`results(units, failed, run)` builds the document from the per-unit entries of `t7_test` (each
holding `t7_unit.measure_unit`'s output) and adds Holm's adjusted p-values (`add_holm`);
`summary_lines(document)` gives the text, every figure read from the document.

Holm (`add_holm`): per tier, loss and window, over the units whose verdict in that tier is
"UNEQUAL" or "not shown unequal", the units' p-values adjusted by `t7_stat.holm`, written as
`holm_p` beside each p-value; a unit with another verdict gets `holm_p` None where it has a pooled
reading. They are reported only: the verdicts read the unadjusted p-value.

The summary: three heading lines and a fourth that says how to read the verdicts (`NULL`,
`LEVEL_OR_VALUE`, `NOT_EQUAL`, `STEPS`, the same texts as in the definitions `verdict` and
`pooled`); per unit one line of counts and day types, and per tier a verdict line with the pooled
scalar, one line for the slot loss and one for the driver term (`t7_unit`, step 1: the every-scale
driver-loss bound of the every-request route, not the replayed driver loss) with its p-values and
Holm's at the three windows and its statistic at the main window, and one line per day type with,
per loss at the main window, its pooled marginal loss per unit value beside the number of its days
whose loss changes ("no day changes" in place of the marginal when there is none) and its
per-split mean, p5 and p95; the failed units, if any; a closing line under the verdicts
(`LEVEL_OR_VALUE`, `NOT_EQUAL`, `STEPS`). Numbers are written with 4 significant digits ("none"
where there is no number).
"""

from fleetreserve import size_scale, t7_stat, t7_unit
from fleetreserve.learn_features import OTHER_SITE, WORKPLACE

WHAT = "T7 TEST ON TRAINING UNITS: A MEASUREMENT WITH A VERDICT RULE, NOT A PASS MARK"
TYPE_NAMES = {WORKPLACE: "workplace", OTHER_SITE: "other"}
TESTABLE = (t7_unit.UNEQUAL, t7_unit.NOT_SHOWN)
NAMES = {"slot": "slot loss", "driver": "driver term"}
NULL = (
    "the p-value sees each day only through its loss change and its value; its null is that the "
    "day-type labels are exchangeable among the offering days of the tested types (the labels are "
    "shuffled among the days, each day keeping its loss change and its value), under which the "
    f"chance of a p-value below {t7_stat.LEVEL:g} is at most {t7_stat.LEVEL:g}; T7's condition of "
    "equal expected marginal losses per unit value is weaker"
)
LEVEL_OR_VALUE = (
    "day types with equal expected marginal losses per unit value whose days differ in value, or "
    "in the spread of their loss changes (which a different loss level can bring), are outside "
    f"that null, so '{t7_unit.UNEQUAL}' can come from such a difference in place of unequal "
    "marginal losses, and does not say which"
)
NOT_EQUAL = (
    f"'{t7_unit.NOT_SHOWN}' is not a finding that the marginal losses per unit value are equal: "
    "a type's marginal loss is read only from those of its days whose loss changes inside the "
    "window, which can be few, so with a few hundred offering days or fewer the test can have "
    "little power and often miss a real difference"
)
STEPS = (
    "the slot loss is a step in the scale and the driver term does not fall as the scale rises (a "
    "ramp where no power row breaks, with upward jumps where one does), so only the days whose "
    "loss changes inside the window carry information on a type's marginal loss; days_changing "
    "counts them per type, loss and window, and each day-type line of the summary, at window "
    f"{t7_stat.MAIN_WINDOW:g}, writes 'no day changes' for a loss on which the type has none"
)


def _key(h: float) -> str:
    return f"{h:g}"


def add_holm(units: dict) -> None:
    """Holm's adjusted p-values (module text), written into `units` in place."""
    for tier in size_scale.TIERS:
        names = [u for u, e in units.items() if e["tiers"][tier]["verdict"] in TESTABLE]
        for loss in t7_stat.LOSSES:
            for h in t7_stat.WINDOWS:
                p = {u: units[u]["tiers"][tier]["pooled"][loss][_key(h)]["p_value"] for u in names}
                adjusted = t7_stat.holm(p)
                for u, entry in units.items():
                    pooled = entry["tiers"][tier]["pooled"]
                    if pooled is not None:
                        pooled[loss][_key(h)]["holm_p"] = adjusted.get(u)


def results(units: dict, failed: dict, run: dict) -> dict:
    """The results document (module text)."""
    add_holm(units)
    return {
        "label": WHAT,
        "definitions": definitions(),
        "run": run,
        "units": units,
        "failed_units": sorted(f"{u}: {why}" for u, why in failed.items()),
    }


def definitions() -> dict:
    return {
        "what": (
            f"{WHAT}: training units are scored, each with a forecast model not fitted on it; "
            "the figures are development evidence, not held-out evidence"
        ),
        "policy": (
            "the fixed plan with its closed-form slot loss and its driver term, as size_whatif "
            "computes them; no re-solve"
        ),
        "driver_term": (
            "what every 'driver' figure here is about: the every-scale driver-loss bound of the "
            "every-request route (gate_calibrate.every_scale_loss, "
            "fixed_plan_loss.every_scale_driver_bound), capped at 1, with each car's "
            "decision-hour draw; it bounds the replayed driver loss at every scale, up to the "
            "errors stated in V1 to V8 of fixed_plan_loss's module text, so the driver p-values "
            "and verdicts are about this bound, not about the replayed driver loss"
        ),
        "day_type": (
            "the weekday of the day, or public holiday (the public_holiday column of size_whatif) "
            f"whatever its weekday; a type with fewer than {t7_stat.MIN_TYPE_DAYS} offering days "
            f"is pooled into '{t7_stat.OTHER}'; the tested types are those with at least "
            f"{t7_stat.MIN_TYPE_DAYS} offering days after pooling"
        ),
        "scalar": (
            "per split and tier, size_scale.calibrate with size 1 on every offering day (the "
            "constant sizing of size_whatif), capped at 1; tier A the smaller of the slot and "
            "driver scalars, tier B the slot scalar"
        ),
        "marginal_per_kwh": (
            "a type's mean loss at the upper end of the window minus its mean loss at the lower "
            "end, over the window's width (the slope), over the type's mean certified value at "
            "scale 1 in kWh; the window runs from max(b - h, 0) to b + h, where a loss is 0 at "
            "scale 0"
        ),
        "per_split": (
            "on each split at its scalar, over its offering test days; mean, p5 and p95 over the "
            "splits on which the type has an offering test day (numpy linear interpolation)"
        ),
        "pooled": (
            "at pooled_scalar, the lower median of the scalars used above 0 (with n such splits, "
            "sorted from the smallest, the ((n + 1) // 2)-th, so a scalar some split used), or 0 "
            "when no split's scalar used is above 0, over all the unit's offering days; "
            "statistic: largest minus smallest marginal_per_kwh over the tested types; p-value: "
            f"(1 + hits) / (1 + {t7_stat.ROUNDS}) over {t7_stat.ROUNDS} seeded permutations of "
            "the type labels among the days of the tested types, a permutation a hit when its "
            f"statistic is at or above the observed one less {t7_stat.TIE:g} times it; seeded per "
            "unit by the first 8 bytes of SHA-256 of 't7 <unit>', so the same permutations for "
            f"every tier, loss and window of a unit and others for another unit; {NULL}; "
            f"{LEVEL_OR_VALUE}; {NOT_EQUAL}; {STEPS}"
        ),
        "days_changing": (
            "per type, loss and window of the pooled reading, the number of the type's offering "
            "days whose loss at the window's upper end differs from that at its lower end; with 0 "
            "the type's marginal_per_kwh is exactly 0, and at window "
            f"{t7_stat.MAIN_WINDOW:g} the summary writes 'no day changes' in its place"
        ),
        "deciding_loss": (
            "tier B: the slot loss; tier A: the loss that binds on more splits (the driver term "
            "on a split where tier A's calibrated scalar is below the slot scalar), the slot "
            "loss on a tie"
        ),
        "verdict": (
            f"'{t7_unit.UNEQUAL}' when the deciding loss's p-value at window "
            f"{t7_stat.MAIN_WINDOW:g} is below {t7_stat.LEVEL:g}, else '{t7_unit.NOT_SHOWN}'; "
            f"before that, the first that applies: '{t7_unit.NOT_SPLIT}', '{t7_unit.NO_OFFER}' "
            "(scalar 0 on more than half of all the splits), "
            f"'{t7_unit.ONE_TYPE}' (fewer than 2 tested types), '{t7_unit.FLAT}' (every tested "
            "type's slope of the deciding loss exactly 0 at every window); the other windows' "
            f"p-values and Holm's adjusted p-values do not change it; {NULL}; {LEVEL_OR_VALUE}; "
            f"{NOT_EQUAL}; {STEPS}"
        ),
        "holm_p": (
            "Holm's adjusted p-value per tier, loss and window across the units whose verdict in "
            f"that tier is '{t7_unit.UNEQUAL}' or '{t7_unit.NOT_SHOWN}'"
        ),
    }


def _f(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return f"{value:.4g}"


def _unit_line(name: str, e: dict) -> str:
    types = ", ".join(f"{t} {n}" for t, n in e["day_types"].items()) or "none"
    tested = ", ".join(e["tested_types"]) or "none"
    return (
        f"{name} ({e['site_type']}, {e['status']}): {e['counting_days']} counting days, "
        f"{e['offering_days']} offering, {sum(e['days_not_offering_by_reason'].values())} not "
        f"offering, {e['certificate_timeouts']} certificate timeouts; day types of the offering "
        f"days: {types}; tested: {tested}"
    )


def _tier_lines(tier: str, t: dict) -> list[str]:
    binds = t["binding_loss_splits"][t["deciding_loss"]]
    offer = t["splits_that_offer"]
    where = f"lower median of the {offer} splits that offer" if offer else "no split offers"
    lines = [
        f"  tier {tier}: {t['verdict']}; deciding {NAMES[t['deciding_loss']]} (binds on {binds} of "
        f"{t['splits']} splits); pooled at scalar {_f(t['pooled_scalar'])} ({where}); scalar 0 "
        f"on {t['splits_scalar_zero']} splits, capped at 1 on {t['splits_scalar_capped']}"
    ]
    pooled = t["pooled"]
    if pooled is None:
        return lines
    for loss in t7_stat.LOSSES:
        parts = []
        for h in t7_stat.WINDOWS:
            entry = pooled[loss][_key(h)]
            parts.append(
                f"p {_f(entry['p_value'])} at window {h:g} (Holm {_f(entry.get('holm_p'))})"
            )
        main = pooled[loss][_key(t7_stat.MAIN_WINDOW)]
        lines.append(
            f"    {NAMES[loss]}: " + ", ".join(parts) + f"; statistic {_f(main['statistic'])} per "
            f"kWh at window {t7_stat.MAIN_WINDOW:g}"
        )
    main = _key(t7_stat.MAIN_WINDOW)
    for name, reading in pooled["slot"][main]["types"].items():
        parts = []
        for loss in t7_stat.LOSSES:
            r = pooled[loss][main]["types"][name]
            s = t["per_split"][loss][main][name]
            if r["days_changing"] == 0:
                at = f"no day changes pooled (0 of {r['days']} days)"
            else:
                at = (
                    f"{_f(r['marginal_per_kwh'])} pooled ({r['days_changing']} of {r['days']} "
                    "days change)"
                )
            parts.append(
                f"{NAMES[loss]} {at}, {_f(s['mean'])} over {s['splits']} splits (p5 "
                f"{_f(s['p5'])}, p95 {_f(s['p95'])})"
            )
        lines.append(
            f"    {name} ({reading['days']} days, mean value {_f(reading['mean_value_kwh'])} "
            "kWh): " + "; ".join(parts)
        )
    return lines


def summary_lines(document: dict) -> list[str]:
    run = document.get("run", {})
    lines = [
        document["label"],
        document["definitions"]["what"],
        f"model {run.get('model')}, configuration {run.get('matrix_configuration')} (dial "
        f"{run.get('dial')}, kappa {run.get('kappa')}), {run.get('splits')} splits, "
        f"{run.get('rounds')} permutations; fixed plan, every-request slot loss and driver term "
        "(the every-scale driver-loss bound, not the replayed driver loss); marginal losses per "
        "unit value are loss per unit of scale per kWh of certified value, per type at window "
        f"{t7_stat.MAIN_WINDOW:g}.",
        "HOW TO READ THE VERDICTS: " + "; ".join((NULL, LEVEL_OR_VALUE, NOT_EQUAL, STEPS)) + ".",
    ]
    for name, entry in document["units"].items():
        lines.append(_unit_line(name, entry))
        for tier in size_scale.TIERS:
            lines += _tier_lines(tier, entry["tiers"][tier])
    if document["failed_units"]:
        lines.append("FAILED UNITS: " + "; ".join(document["failed_units"]))
    lines.append("UNDER THE VERDICTS: " + "; ".join((LEVEL_OR_VALUE, NOT_EQUAL, STEPS)) + ".")
    return lines

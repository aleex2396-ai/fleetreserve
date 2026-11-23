"""The results document and the summary text of `size_whatif`.

`results(...)` builds the document from the per-unit data of `size_whatif`;
`summary_lines(document)` gives the text, every figure read from the document. Per unit, and
pooled per site type (`learn_features.SITE_TYPES`: "workplace" and "other") over the units of that
type:

- `labels`: counts of counting days, offering days, days that do not offer and certificates that
  timed out; over the labelled days, the share whose label is 0, the share whose label is at least
  1, the share whose centre is outside the true set, and the points p10, p50, p90 of the unclipped
  label (numpy's default linear interpolation);
- `pinball`: per level, the mean pinball loss (`learn_eval.pinball_loss`) of the booster's
  predictions and of the constant predictor against the clipped labels, both from the fits that left
  the row's unit out;
- `tiers`, per tier and sizing: over the splits, the mean offer per counting test day (a split at
  scalar 0 counting 0); the mean of the per-split mean test slot loss and driver-loss bound over the
  splits where they are defined; the share of the offering days the sizing abstains on; and the
  paired difference from `constant` on the same splits (mean, p5 and p95 of the per-split
  difference in kWh per counting test day, and each over the mean offer of `constant`, None when
  that is 0). A pooled split adds the test days of every unit of the type at that seed: its offer
  is the mean over all those counting test days and its losses the mean over all their offering
  test days. The mean scalar is given per unit only.
"""

import numpy as np

from fleetreserve import gate_calibrate, size_gbm, size_scale
from fleetreserve.learn_eval import pinball_loss
from fleetreserve.learn_features import OTHER_SITE, WORKPLACE

WHAT = "SIZING WHAT-IF ON TRAINING UNITS, NOT A RUN OF THE PLAN"
TYPE_NAMES = {WORKPLACE: "workplace", OTHER_SITE: "other"}
ORACLE_NOTE = "uses the day's own outcome: a reference, not a method"
POINTS = (0.05, 0.95)
KINDS = ("gbm", "constant")


def _number(value):
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else gate_calibrate.number(value)


def _mean(values):
    return _number(np.mean(values)) if len(values) else None


def label_figures(rows: list, counting: int, not_offering: int, timeouts: int) -> dict:
    scales = np.array([row["label_scale"] for row in rows], dtype=float)
    labelled = len(rows)
    points = np.quantile(scales, (0.1, 0.5, 0.9)) if labelled else (None,) * 3
    return {
        "counting_days": counting,
        "offering_days": labelled,
        "days_not_offering": not_offering,
        "certificate_timeouts": timeouts,
        "share_zero": _number(np.mean(scales == 0)) if labelled else None,
        "share_at_or_above_one": _number(np.mean(scales >= 1)) if labelled else None,
        "share_centre_outside": (
            _number(np.mean([not row["centre_inside"] for row in rows])) if labelled else None
        ),
        "p10": _number(points[0]),
        "p50": _number(points[1]),
        "p90": _number(points[2]),
    }


def pinball_figures(labels, predictions: dict, kinds=KINDS) -> dict:
    """Per level, the mean pinball loss of each predictor of `kinds` (None where it has no
    prediction)."""
    out = {}
    for level in size_gbm.LEVELS:
        found = predictions.get(float(level))
        if found is None or not len(labels):
            out[f"{level:g}"] = dict.fromkeys(kinds)
            continue
        out[f"{level:g}"] = {
            kind: _number(pinball_loss(labels, found[kind], float(level)))
            for kind in kinds
            if kind in found
        }
        out[f"{level:g}"].update({kind: None for kind in kinds if kind not in found})
    return out


def _split_offer(scores) -> float:
    offers = [o for s in scores for o in s.offers_kwh]
    return float(np.mean(offers)) if offers else 0.0


def _split_loss(scores, field: str):
    losses = [x for s in scores for x in getattr(s, field)]
    return float(np.mean(losses)) if losses else None


def tier_figures(per_seed: dict, constant: dict, scalars=None) -> dict:
    """`per_seed`, `constant`: seed -> list of `size_scale.TestScore` (one per unit)."""
    seeds = sorted(per_seed)
    offers = np.array([_split_offer(per_seed[s]) for s in seeds])
    base = np.array([_split_offer(constant[s]) for s in seeds])
    slot = [x for x in (_split_loss(per_seed[s], "slot_losses") for s in seeds) if x is not None]
    driver = [
        x for x in (_split_loss(per_seed[s], "driver_losses") for s in seeds) if x is not None
    ]
    difference = offers - base
    base_mean = float(base.mean()) if len(base) else 0.0
    low, high = np.quantile(difference, POINTS) if len(seeds) else (None, None)
    out = {
        "splits": len(seeds),
        "mean_offer_kwh_per_counting_test_day": _mean(offers),
        "splits_with_test_losses": len(slot),
        "mean_test_slot_loss": _mean(slot),
        "mean_test_driver_loss_bound": _mean(driver),
        "difference_from_constant": {
            "mean_kwh": _mean(difference),
            "p5_kwh": _number(low),
            "p95_kwh": _number(high),
            "mean_share": _mean(difference / base_mean) if base_mean > 0 else None,
            "p5_share": _number(low / base_mean) if base_mean > 0 else None,
            "p95_share": _number(high / base_mean) if base_mean > 0 else None,
        },
    }
    if scalars is not None:
        used = [s.used for s in scalars]
        out["mean_scalar_used"] = _mean(used)
        out["splits_that_offer"] = sum(1 for u in used if u > 0)
        out["splits_scalar_capped"] = sum(1 for s in scalars if s.used < s.calibrated)
    return out


def results(data: dict, frame, predictions: dict, q_all: dict, measured: dict, failed: dict):
    """The results document (module text)."""
    units, groups = {}, {}
    for name, d in data.items():
        rows = d.rows
        labels = np.array([row[size_gbm.LABEL] for row in rows], dtype=float)
        offering = np.zeros(d.counting_days, dtype=bool)
        offering[[row["day"] for row in rows]] = True
        tiers = {}
        for tier in size_scale.TIERS:
            constant = {seed: [score] for seed, _, score in measured[name]["constant"][tier]}
            tiers[tier] = {}
            for sizing, found in measured[name].items():
                per_seed = {seed: [score] for seed, _, score in found[tier]}
                figures = tier_figures(per_seed, constant, [s for _, s, _ in found[tier]])
                figures["share_of_offering_days_abstained"] = (
                    _number(np.isnan(q_all[name][sizing][offering]).mean())
                    if offering.any()
                    else None
                )
                tiers[tier][sizing] = figures
        units[name] = {
            "site_type": TYPE_NAMES.get(d.site_type),
            "status": d.status,
            "fit_units": list(d.fit_units),
            "days_not_offering_by_reason": d.not_offering,
            "labels": label_figures(
                rows, d.counting_days, sum(d.not_offering.values()), d.certificate_timeouts
            ),
            "pinball": pinball_figures(labels, predictions.get(name, {})),
            "tiers": tiers,
        }
        if d.site_type is not None:
            groups.setdefault(TYPE_NAMES[d.site_type], []).append(name)
    pooled = {
        kind: _pooled(names, data, predictions, q_all, measured) for kind, names in groups.items()
    }
    return {
        "label": WHAT,
        "definitions": definitions(),
        "units": units,
        "pooled_by_site_type": dict(sorted(pooled.items())),
        "failed_units": sorted(f"{u}: {why}" for u, why in failed.items()),
    }


def _pooled(names, data, predictions, q_all, measured) -> dict:
    rows = [row for n in names for row in data[n].rows]
    labels = np.array([row[size_gbm.LABEL] for row in rows], dtype=float)
    joined = {}
    for level in size_gbm.LEVELS:
        parts = [
            predictions[n][float(level)] for n in names if float(level) in predictions.get(n, {})
        ]
        if len(parts) == len(names):
            joined[float(level)] = {
                kind: np.concatenate([p[kind] for p in parts]) for kind in ("gbm", "constant")
            }
    counting = sum(data[n].counting_days for n in names)
    not_offering = sum(sum(data[n].not_offering.values()) for n in names)
    timeouts = sum(data[n].certificate_timeouts for n in names)
    scored = [n for n in names if data[n].splits]
    tiers = {}
    for tier in size_scale.TIERS:
        tiers[tier] = {}
        sizings = [s for s in measured[scored[0]]] if scored else []
        constant = _by_seed(scored, measured, "constant", tier)
        for sizing in sizings:
            if not all(sizing in measured[n] for n in scored):
                continue
            figures = tier_figures(_by_seed(scored, measured, sizing, tier), constant)
            kept = [~np.isnan(q_all[n]["constant"]) for n in scored]
            offered = sum(int(k.sum()) for k in kept)
            abstained = sum(
                int(np.isnan(q_all[n][sizing][k]).sum()) for n, k in zip(scored, kept, strict=True)
            )
            figures["share_of_offering_days_abstained"] = (
                _number(abstained / offered) if offered else None
            )
            tiers[tier][sizing] = figures
    return {
        "units": list(names),
        "units_scored": scored,
        "labels": label_figures(rows, counting, not_offering, timeouts),
        "pinball": pinball_figures(labels, joined),
        "tiers": tiers,
    }


def _by_seed(names, measured, sizing, tier) -> dict:
    out = {}
    for n in names:
        for seed, _, score in measured[n][sizing][tier]:
            out.setdefault(seed, []).append(score)
    return out


def definitions() -> dict:
    return {
        "what": (
            f"{WHAT}: training units are scored, each with a forecast model not fitted on it; the "
            "figures are development evidence, not held-out evidence"
        ),
        "label": (
            "the exact scale of the certified box on the true cars of the steered sub-day "
            "(size_label; 0 when the centre is outside the true set); label_clipped is it "
            "clipped to [0, 1]"
        ),
        "sizings": (
            "constant: size 1 on every offering day; gbm-<level>-<arm>: the booster's prediction "
            "at that level, fitted on the other units' labelled days, whose context rows and "
            "certificates come from forecast models fitted on their fit_units, which can hold "
            "the scored unit: so these sizes can depend on the scored unit's outcomes, and their "
            "gain is not shown to be that of a sizing fitted without them; oracle-<arm>: the "
            "day's own "
            f"unclipped label, which {ORACLE_NOTE}; arm scale: size clipped to "
            f"[{size_scale.FLOOR}, 1]; arm abstain: a day with a raw size below "
            f"{size_scale.FLOOR} does not offer, the others get the raw size capped at 1"
        ),
        "scalar": (
            "per split and tier, the fixed plan's every-request calibration of "
            "gate_calibrate.every_request on the calibration days that offer under the sizing, "
            "with each day's hour thresholds and power-row thresholds divided by its size and "
            "the gains of its every-scale driver-loss bound multiplied by it, so that its scale "
            "is the scalar times its size; the scalar used is capped at 1 over the largest size "
            "of the unit's offering days, so no day's scale exceeds 1; tier A the smaller of the "
            "slot and driver scalars, tier B the slot scalar"
        ),
        "offer": (
            "per split, the mean over the counting test days of scalar times size times certified "
            "value (0 on a day that does not offer); the mean is over splits"
        ),
        "test_losses": (
            "per split, the closed-form slot loss and the driver-loss bound at the day's scale, "
            "averaged over the test days that offer under the sizing; the mean is over the splits "
            "where they are defined"
        ),
        "difference_from_constant": (
            "per split, the sizing's offer minus constant's on the same split; mean, p5 and p95 "
            "over splits (numpy linear interpolation), in kWh per counting test day and over "
            "constant's mean offer"
        ),
    }


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def _tier_lines(tiers: dict) -> list[str]:
    lines = []
    for tier, found in tiers.items():
        lines.append(f"  tier {tier}:")
        for sizing, f in found.items():
            note = f" ({ORACLE_NOTE})" if sizing.startswith("oracle") else ""
            diff = f["difference_from_constant"]
            lines.append(
                f"    {sizing}{note}: offer {_f(f['mean_offer_kwh_per_counting_test_day'])} kWh "
                f"per counting test day, slot loss {_f(f['mean_test_slot_loss'])}, driver loss "
                f"bound {_f(f['mean_test_driver_loss_bound'])}, abstains on "
                f"{_f(f['share_of_offering_days_abstained'])} of offering days; against constant "
                f"{_f(diff['mean_kwh'])} kWh (p5 {_f(diff['p5_kwh'])}, p95 {_f(diff['p95_kwh'])}), "
                f"share {_f(diff['mean_share'])} (p5 {_f(diff['p5_share'])}, p95 "
                f"{_f(diff['p95_share'])})"
            )
    return lines


def _head_lines(name: str, entry: dict) -> list[str]:
    lab = entry["labels"]
    lines = [
        f"{name}: {lab['counting_days']} counting days, {lab['offering_days']} labelled, "
        f"{lab['days_not_offering']} not offering, {lab['certificate_timeouts']} certificate "
        f"timeouts; label 0 on {_f(lab['share_zero'])}, at least 1 on "
        f"{_f(lab['share_at_or_above_one'])}, centre outside on {_f(lab['share_centre_outside'])}; "
        f"label p10 {_f(lab['p10'])}, p50 {_f(lab['p50'])}, p90 {_f(lab['p90'])}"
    ]
    for level, p in entry["pinball"].items():
        net = f", net {_f(p['net'], 4)}" if "net" in p else ""
        booster = f"booster {_f(p['gbm'], 4)}, constant {_f(p['constant'], 4)}"
        lines.append(f"  pinball at {level}: {booster}{net}")
    return lines


def summary_lines(document: dict) -> list[str]:
    run = document.get("run", {})
    lines = [
        document["label"],
        document["definitions"]["what"],
        f"model {run.get('model')}, configuration {run.get('matrix_configuration')} (dial "
        f"{run.get('dial')}, kappa {run.get('kappa')}), {run.get('splits')} splits; fixed plan, "
        "every-request losses; oracle lines use the day's own outcome.",
    ]
    for name, entry in document["units"].items():
        lines += _head_lines(f"{name} ({entry['site_type']}, {entry['status']})", entry)
        lines += _tier_lines(entry["tiers"])
    for kind, entry in document["pooled_by_site_type"].items():
        lines += _head_lines(f"pooled {kind} ({', '.join(entry['units_scored'])} scored)", entry)
        lines += _tier_lines(entry["tiers"])
    if document["failed_units"]:
        lines.append("FAILED UNITS: " + "; ".join(document["failed_units"]))
    return lines

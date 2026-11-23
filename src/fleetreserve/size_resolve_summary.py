"""The results document and the summary text of `size_resolve_whatif`.

`results(...)` builds the document; `summary_lines(document)` gives the text, every figure read
from the document. Per unit, and pooled per site type ("workplace" and "other") over the scored
units of that type, as `size_whatif_summary` does for the fixed plan, with the same label and
pinball figures (`size_whatif_summary.label_figures`, `pinball_figures`; the pinball loss of the
day-size net beside the booster's and the constant's, `KINDS`) and, per tier and sizing, the
figures of `size_whatif_summary.tier_figures` read on the one-way re-solve:

- over the splits, the mean offer per counting test day (a split at scalar 0 counting 0) and the
  paired difference from `constant` on the same splits: its mean, p5 and p95 over the splits in
  kWh per counting test day, and each over `constant`'s mean offer (`mean_share`, `p5_share`,
  `p95_share`; None when that offer is 0). A pooled split adds the test days of every unit of the
  type at that seed;
- the mean, over the splits where they are defined, of the split's mean test slot loss and mean
  test driver loss (`mean_test_driver_loss`: the replayed driver loss of the audited paths, where
  the fixed-plan what-if has the driver-loss bound);
- per unit the mean scalar used and the number of splits at a scalar above 0;
- `mean_share_of_offering_test_days_abstained`: per split, one minus the number of test days that
  offer under the sizing over the number that offer under `constant` (the split left out when
  that is 0), the mean over the splits;
- per unit, `evaluations` (the (day, scale) pairs evaluated and the one-way rung's re-solves by
  outcome, a timeout among them) and `day_type_scales` (per tier and type, over the splits where
  the type has a day, the number of such splits and the mean of the multiplier's scale).

The summary also says, when the document holds the day-size net's fit records (`net_fits`,
written by `size_resolve_whatif`), how many networks were fitted and how many were kept at their
starting constant (`net_fit_line`), and, when it holds those of conformal training (`ct_fits`),
how many of its networks were fitted and how many were kept at the net they start from
(`ct_fit_line`).

Shuffled controls (`CONTROL`): a sizing `<name>` whose control `shuffled-<name>` is measured
apart from the sizings, once per draw (`size_resolve_whatif.control_sizings`), gets, beside its
difference from `constant`, `difference_from_shuffled` (`control_figures`): the number of draws,
the control's mean offer over draws and splits, then two spreads kept apart. Over splits: per
split the mean over draws of the sizing's offer minus the draw's on the same split and tier, its
mean, p5 and p95 over the splits, in kWh and over the control's mean offer. Over draws
(`over_draws`): per draw the sizing's mean offer over splits minus the draw's, its p5 and p95 over
the draws, in kWh and over the same mean offer; a sizing whose sizes carry nothing about the days
sits inside that spread except by chance. The same per-draw gains, draw by draw, with their
smallest and largest and the number of draws at or above the sizing (`by_draw`,
`by_draw_figures`), let a reader tell whether the sizing beats every draw. Each difference is
taken draw by draw before any mean, so a sizing whose every draw is itself (the same size on every
day) differs by exactly 0. A pool reads a control only when every unit of it has one. The summary
gives these (`by_draw` apart) on the line after the sizing's, lists the units whose controls
failed (`failed_controls`), and, when the document counts the pairs the controls added
(`control_evaluations`), says what a control is and why (`definitions()["shuffled_control"]`) and
gives those counts per unit (`control_lines`).

Development units (`results(..., development=...)`, from `size_resolve_whatif
--development-units`): each gets its entry under `units` as a training unit does, in no pool; the
document's label is `DEVELOPMENT_LABEL` and its definitions gain `development_units`
(`DEVELOPMENT_TEXT`). The summary gives the training units and the pools as without them (the
label line apart), then a part for the development units: what they are, the finished fits'
counts (`development_fit_line`), the pairs their controls added, and each unit's lines, where each
control line is followed by the sizing's gain over each draw with their smallest, largest and the
draws at or above the sizing (`by_draw_line`; the training units' lines leave `by_draw` out, as
before, so their text stays as it was).
`development=None` (no development units asked for) leaves the document and the summary as before.

No verdict is formed: the figures are reported as they are, and the summary says so.
"""

import numpy as np

from fleetreserve import size_gbm, size_net, size_scale, size_whatif_summary
from fleetreserve.size_whatif_summary import (
    ORACLE_NOTE,
    POINTS,
    TYPE_NAMES,
    _f,
    _head_lines,
    _mean,
    _split_offer,
)

WHAT = size_whatif_summary.WHAT
DEVELOPMENT_LABEL = (
    f"{WHAT}; DEVELOPMENT UNITS SCORED TOO, WITH SIZINGS FITTED ON THE TRAINING UNITS ONLY"
)
DEVELOPMENT_TEXT = (
    "the development units of run.development_units are scored with finished sizings: each "
    "development unit's forecast model is built with it as its only gate unit and fitted on "
    "training units only, and the boosters, the constant, the day-size net and conformal "
    "training are each fitted once on every labelled day of the run's training units and read "
    "at the development unit's days (development_fits); nothing is fitted or tuned on a "
    "development unit's days but, as on every unit, each split's scalar and the day-type "
    "multiplier on that split's calibration days; the development units are in no pool, and "
    "the training units' figures are those of the same run without them"
)
ROUTE = "one-way re-solve, audited-set losses, scalar on the gate's grid"
KINDS = (*size_whatif_summary.KINDS, size_net.KIND)
CONTROL = "shuffled"


def _number(value):
    return size_whatif_summary._number(value)


def _abstained(per_seed: dict, constant: dict):
    shares = []
    for seed in sorted(per_seed):
        base = sum(s.offering_days for s in constant[seed])
        if base > 0:
            shares.append(1.0 - sum(s.offering_days for s in per_seed[seed]) / base)
    return _number(np.mean(shares)) if shares else None


def tier_figures(per_seed: dict, constant: dict, scalars=None) -> dict:
    """`size_whatif_summary.tier_figures` on re-solve scores (module text). `per_seed`,
    `constant`: seed -> list of `size_scale.TestScore` (one per unit)."""
    out = size_whatif_summary.tier_figures(per_seed, constant, scalars)
    out["mean_test_driver_loss"] = out.pop("mean_test_driver_loss_bound")
    out.pop("splits_scalar_capped", None)  # the scalar is a grid point: never capped
    out["mean_share_of_offering_test_days_abstained"] = _abstained(per_seed, constant)
    return out


SPREAD = ("p5_kwh", "p95_kwh", "p5_share", "p95_share")
BY_DRAW = ("gain_kwh", "smallest_gain_kwh", "largest_gain_kwh", "draws_at_or_above")


def control_figures(per_seed: dict, draws: list) -> dict:
    """`difference_from_shuffled` (module text). `per_seed`: seed -> the sizing's
    `size_scale.TestScore` (one per unit); `draws`: per draw of its control, the same."""
    seeds = sorted(per_seed)
    out = {"draws": len(draws), "control_mean_offer_kwh_per_counting_test_day": None}
    if not seeds:  # a unit that is not split
        empty = dict.fromkeys(SPREAD)
        return {
            **out,
            "mean_kwh": None,
            "mean_share": None,
            **empty,
            "over_draws": dict(empty),
            "by_draw": by_draw_figures(None),
        }
    offers = np.array([_split_offer(per_seed[s]) for s in seeds])
    by_draw = np.array([[_split_offer(d[s]) for s in seeds] for d in draws])
    gaps = offers - by_draw  # (draws, splits): the sizing's offer minus the draw's
    base = float(by_draw.mean())
    out["control_mean_offer_kwh_per_counting_test_day"] = _number(base)
    difference = gaps.mean(axis=0)  # per split, over the draws
    gains = gaps.mean(axis=1)  # per draw, over the splits

    def spread(values) -> dict:
        low, high = np.quantile(values, POINTS)
        shares = (low / base, high / base) if base > 0 else (None, None)
        return dict(zip(SPREAD, map(_number, (low, high, *shares)), strict=True))

    return {
        **out,
        "mean_kwh": _mean(difference),
        "mean_share": _mean(difference / base) if base > 0 else None,
        **spread(difference),  # over splits
        "over_draws": spread(gains),
        "by_draw": by_draw_figures(gains),
    }


def by_draw_figures(gains) -> dict:
    """`by_draw` of `difference_from_shuffled` (module text); `gains`: per draw, in the order of
    the draws, the sizing's mean offer over the splits minus the draw's, or None for a unit that is
    not split."""
    if gains is None:
        return dict.fromkeys(BY_DRAW)
    return {
        "gain_kwh": [_number(g) for g in gains],
        "smallest_gain_kwh": _number(gains.min()),
        "largest_gain_kwh": _number(gains.max()),
        "draws_at_or_above": int((gains <= 0).sum()),
    }


def _draws(controls: dict, names: list, control: str, tier: str) -> list:
    """Per draw of `control`, seed -> its test scores (one per unit of `names`)."""
    out = []
    for k in range(len(controls[names[0]][control])):
        by_seed = {}
        for n in names:
            for seed, _, score in controls[n][control][k][tier]:
                by_seed.setdefault(seed, []).append(score)
        out.append(by_seed)
    return out


def _tiers(measured: dict, names: list, controls: dict | None = None) -> dict:
    """Per tier and sizing the figures over the units `names` (one unit: with its scalars).
    `controls`: unit -> `shuffled-<sizing>` -> per draw, tier -> list of (seed, `Scalar`,
    `TestScore`)."""
    controls = controls or {}
    tiers = {}
    for tier in size_scale.TIERS:
        constant = _by_seed(measured, names, "constant", tier)
        tiers[tier] = {}
        for sizing in measured[names[0]]:
            if not all(sizing in measured[n] for n in names):
                continue  # a sizing some unit lacks (`size_whatif.sizings`) is not pooled
            scalars = None
            if len(names) == 1:
                scalars = [s for _, s, _ in measured[names[0]][sizing][tier]]
            per_seed = _by_seed(measured, names, sizing, tier)
            tiers[tier][sizing] = tier_figures(per_seed, constant, scalars)
            control = f"{CONTROL}-{sizing}"
            if all(control in controls.get(n, {}) for n in names):
                draws = _draws(controls, names, control, tier)
                tiers[tier][sizing]["difference_from_shuffled"] = control_figures(per_seed, draws)
    return tiers


def _by_seed(measured, names, sizing, tier) -> dict:
    out = {}
    for n in names:
        for seed, _, score in measured[n][sizing][tier]:
            out.setdefault(seed, []).append(score)
    return out


def _type_scales(found: dict) -> dict:
    return {
        tier: {
            name: {"splits": len(values), "mean_scale": _number(np.mean(values))}
            for name, values in sorted(by_type.items())
        }
        for tier, by_type in found.items()
    }


def _unit_entry(name, d, predictions, measured, evaluations, type_scales, controls) -> dict:
    labels = np.array([row["label_clipped"] for row in d.rows], dtype=float)
    return {
        "site_type": TYPE_NAMES.get(d.site_type),
        "status": d.status,
        "fit_units": list(d.fit_units),
        "days_not_offering_by_reason": d.not_offering,
        "labels": size_whatif_summary.label_figures(
            d.rows, d.counting_days, sum(d.not_offering.values()), d.certificate_timeouts
        ),
        "pinball": size_whatif_summary.pinball_figures(labels, predictions.get(name, {}), KINDS),
        "evaluations": evaluations.get(name, {}),
        "day_type_scales": _type_scales(type_scales.get(name, {})),
        "tiers": _tiers(measured, [name], controls),
    }


def results(
    data,
    predictions,
    measured,
    evaluations,
    type_scales,
    failed,
    controls=None,
    failed_controls=(),
    development=None,
) -> dict:
    """The results document (module text). `data`: unit -> `size_whatif.UnitData` of the units
    measured; `measured`: unit -> sizing -> tier -> list of (seed, `Scalar`, `TestScore`);
    `controls`: as `_tiers` reads them; `failed_controls`: unit -> why its controls failed;
    `development`: the same as `data` for the development units measured, or None when none were
    asked for."""
    units, groups = {}, {}
    for name, d in data.items():
        units[name] = _unit_entry(
            name, d, predictions, measured, evaluations, type_scales, controls
        )
        if d.site_type is not None:
            groups.setdefault(TYPE_NAMES[d.site_type], []).append(name)
    for name, d in (development or {}).items():
        units[name] = _unit_entry(
            name, d, predictions, measured, evaluations, type_scales, controls
        )
    pooled = {}
    for kind, names in groups.items():
        rows = [row for n in names for row in data[n].rows]
        scored = [n for n in names if data[n].splits]
        pooled[kind] = {
            "units": list(names),
            "units_scored": scored,
            "labels": size_whatif_summary.label_figures(
                rows,
                sum(data[n].counting_days for n in names),
                sum(sum(data[n].not_offering.values()) for n in names),
                sum(data[n].certificate_timeouts for n in names),
            ),
            "pinball": size_whatif_summary.pinball_figures(
                np.array([row["label_clipped"] for row in rows], dtype=float),
                _joined(names, predictions),
                KINDS,
            ),
            "tiers": _tiers(measured, scored, controls) if scored else {},
        }
    document = {
        "label": WHAT,
        "definitions": definitions(),
        "units": units,
        "pooled_by_site_type": dict(sorted(pooled.items())),
        "failed_units": sorted(f"{u}: {why}" for u, why in failed.items()),
        "failed_controls": sorted(f"{u}: {why}" for u, why in dict(failed_controls).items()),
    }
    if development is not None:
        document["label"] = DEVELOPMENT_LABEL
        document["definitions"]["development_units"] = DEVELOPMENT_TEXT
    return document


def _joined(names, predictions) -> dict:
    joined = {}
    for level in map(float, size_gbm.LEVELS):
        parts = [predictions[n][level] for n in names if level in predictions.get(n, {})]
        if len(parts) == len(names):
            joined[level] = {
                kind: np.concatenate([p[kind] for p in parts])
                for kind in KINDS
                if all(kind in p for p in parts)
            }
    return joined


def definitions() -> dict:
    return {
        "what": (
            f"{WHAT}: training units are scored, each with a forecast model not fitted on it; the "
            "figures are development evidence, not held-out evidence; no verdict is formed"
        ),
        "route": (
            "the one-way re-solve (dial_choice.RUNG) replayed by gate_day.evaluate on each day's "
            "audited paths, its day-level slot and driver losses the largest over the paths, as "
            "dial_choice and compare_run read them"
        ),
        "sizings": (
            "constant: size 1 on every offering day; gbm-<level>-<arm>: the booster's prediction "
            "at that level, fitted on the other units' labelled days (size_whatif: these sizes "
            "can depend on the scored unit's outcomes); net-<level>-<arm>: the day-size quantile "
            "net's prediction at that level (size_net: a deep-sets network on the cars plugged "
            "in at the decision time, each with the departure model's inputs, size_net_inputs), "
            "fitted on the same days as the booster, with the same caveat; ct-<arm>: the "
            "day-size net at level 0.1 trained further, under the arm's rule, for the offer "
            "calibrated at alpha 0.10 on part of each batch of days and applied to the rest "
            "(size_ct: conformal training), fitted on the same days with the same caveat; "
            "oracle-<arm>: the day's own unclipped "
            f"box-scale label, which {ORACLE_NOTE}; oracle-resolve-<arm>: the largest grid scale "
            "up to which the day's re-solve has slot loss 0 at every grid scale (no request missed "
            f"on any audited path), which {ORACLE_NOTE}; day-type-<arm>: per split and tier the "
            "scale of the day's type (t7_stat types pooled over the split's offering "
            "calibration days), the largest grid scale at which the type's mean calibration "
            "losses are within the tier's alphas, fitted on the calibration days the scalar is "
            f"then calibrated on; arm scale: size clipped to [{size_scale.FLOOR}, 1]; arm "
            f"abstain: a day with a raw size below {size_scale.FLOOR} does not offer, the others "
            "get the raw size capped at 1"
        ),
        "scalar": (
            "per split and tier, the day's scale is the scalar times its size over the largest "
            "size of the unit's days that offer under the sizing; the scalar is the largest point "
            "of the gate's grid (crc.GRID up to 1) at which the calibration bound of crc holds on "
            "the split's calibration days that offer under the sizing, each day's losses at its "
            "own scales made a running maximum over the grid (dial_splits.on_grid_scale); tier A "
            "slot loss at alpha 0.10 and driver loss at alpha 0.05, tier B the slot loss alone"
        ),
        "offer": (
            "per split, the mean over the counting test days of scalar times relative size times "
            "certified value (0 on a day that does not offer); the mean is over splits"
        ),
        "test_losses": (
            "per split, the day-level slot and driver losses of the one-way re-solve at the "
            "day's scale, averaged over the test days that offer under the sizing; the mean is "
            "over the splits where they are defined"
        ),
        "difference_from_constant": (
            "per split, the sizing's offer minus constant's on the same split and tier; mean, p5 "
            "and p95 over splits (numpy linear interpolation), in kWh per counting test day and "
            "over constant's mean offer"
        ),
        "guarantee": (
            "T6's bound covers sizes fixed before the calibration days are seen; the oracle "
            "sizings read the day's own outcome and day-type its calibration days, so their test "
            "losses are reported beside their offers"
        ),
        "shuffled_control": (
            f"{CONTROL}-<sizing>, for net-<level>-<arm>, ct-<arm> and day-type-<arm>: the same "
            "relative sizes moved across the unit's labelled days that offer under constant by a "
            "fixed permutation of those days, one per unit and draw "
            "(size_resolve_whatif.control_order), the same on every split and tier, then "
            "calibrated and scored by the same code, over several draws; a learned sizing is read "
            "against it as well as against constant because on the gate's grid any spread of "
            "sizes, even one unrelated to the days, can gain over constant: constant's scalar is "
            "rounded down to the grid point below the largest scale its bound allows, the grid's "
            "steps are wide at small scales (0.1 to 0.2 is a doubling), and sizes spread below 1 "
            "raise the scalar to where that rounding costs less; difference_from_shuffled: per "
            "split the mean over draws of the sizing's offer minus the draw's on the same split "
            "and tier, mean, p5 and p95 over splits in kWh per counting test day and over the "
            "control's mean offer (over draws and splits); over_draws: per draw the sizing's mean "
            "offer over splits minus the draw's, p5 and p95 over the draws, in kWh and over the "
            "same mean offer; a sizing whose sizes carry nothing about the days sits inside the "
            "spread over draws except by chance, so a gain beyond chance needs its p5 above 0"
        ),
        "shuffled_control_by_draw": (
            "by_draw, beside over_draws in difference_from_shuffled: gain_kwh, per draw in the "
            "order of the draws, the sizing's mean offer over splits minus the draw's (the values "
            "over_draws spreads); their smallest and largest; draws_at_or_above, the number of "
            "draws whose mean offer over splits is at or above the sizing's (a gain at or below "
            "0); a sizing whose sizes carry nothing about the days beats every one of n draws "
            "(draws_at_or_above 0) with a chance of at most 1/(n+1); None on a unit that is not "
            "split"
        ),
    }


def _tier_lines(tiers: dict, by_draw: bool = False) -> list[str]:
    """The lines of a unit's or a pool's tiers; with `by_draw`, each control line followed by
    `by_draw_line` (module text)."""
    lines = []
    for tier, found in tiers.items():
        lines.append(f"  tier {tier}:")
        for sizing, f in found.items():
            note = f" ({ORACLE_NOTE})" if sizing.startswith("oracle") else ""
            diff = f["difference_from_constant"]
            lines.append(
                f"    {sizing}{note}: offer {_f(f['mean_offer_kwh_per_counting_test_day'])} kWh "
                f"per counting test day, slot loss {_f(f['mean_test_slot_loss'])}, driver loss "
                f"{_f(f['mean_test_driver_loss'])}, abstains on "
                f"{_f(f['mean_share_of_offering_test_days_abstained'])} of offering test days; "
                f"against constant {_f(diff['mean_kwh'])} kWh (p5 {_f(diff['p5_kwh'])}, p95 "
                f"{_f(diff['p95_kwh'])}), share {_f(diff['mean_share'])} (p5 "
                f"{_f(diff['p5_share'])}, p95 {_f(diff['p95_share'])})"
            )
            if "difference_from_shuffled" in f:
                lines.append(_control_line(sizing, f["difference_from_shuffled"]))
                if by_draw:
                    lines.append(by_draw_line(f["difference_from_shuffled"]["by_draw"]))
    return lines


def by_draw_line(found: dict) -> str:
    """The gains by draw (`by_draw_figures`) as a summary line."""
    if found["gain_kwh"] is None:
        return "        gains by draw: none (the unit is not split)"
    return (
        "        gains by draw, kWh (the sizing's offer minus each draw's): "
        f"{', '.join(_f(gain) for gain in found['gain_kwh'])}; smallest "
        f"{_f(found['smallest_gain_kwh'])}, largest {_f(found['largest_gain_kwh'])}; draws at or "
        f"above the sizing {found['draws_at_or_above']}"
    )


def _control_line(sizing: str, diff: dict) -> str:
    draws = diff["over_draws"]
    return (
        f"      against its shuffled control {CONTROL}-{sizing}, mean of {diff['draws']} draws "
        f"(offer {_f(diff['control_mean_offer_kwh_per_counting_test_day'])} kWh): "
        f"{_f(diff['mean_kwh'])} kWh; spread over splits p5 {_f(diff['p5_kwh'])}, p95 "
        f"{_f(diff['p95_kwh'])}; spread over draws p5 {_f(draws['p5_kwh'])}, p95 "
        f"{_f(draws['p95_kwh'])}; share of the control's offer {_f(diff['mean_share'])} (over "
        f"splits p5 {_f(diff['p5_share'])}, p95 {_f(diff['p95_share'])}; over draws p5 "
        f"{_f(draws['p5_share'])}, p95 {_f(draws['p95_share'])})"
    )


def control_lines(counts: dict) -> list[str]:
    """What a shuffled control is, then per unit the (day, scale) pairs and re-solves its controls
    added to the evaluations; `counts`: unit -> those counts (`size_resolve_whatif`)."""
    lines = [f"shuffled controls: {definitions()['shuffled_control']}"]
    for unit, found in counts.items():
        lines.append(
            f"{unit}: the shuffled controls added {found['day_scale_pairs']} (day, scale) pairs; "
            f"one-way re-solves {found['re_solves_of_the_one_way_rung']}"
        )
    return lines


def net_fit_line(fits: dict) -> str:
    """How many of the day-size net's networks were fitted and how many were kept at epoch 0, the
    starting constant (`size_net`); `fits`: unit -> level -> `size_net.Fitted.record`."""
    records = [r for by_level in fits.values() for r in by_level.values()]
    kept = sum(r["epoch_kept"] == 0 for r in records)
    return (
        f"day-size net: {len(records)} networks fitted, {kept} kept at epoch 0 (the starting "
        "constant)"
    )


def ct_fit_line(fits: dict) -> str:
    """How many of conformal training's networks were fitted and how many were kept at epoch 0,
    the day-size net they start from (`size_ct`); `fits`: unit -> arm -> `size_ct.Fitted.record`."""
    records = [r for by_arm in fits.values() for r in by_arm.values()]
    kept = sum(r["epoch_kept"] == 0 for r in records)
    return (
        f"conformal training: {len(records)} networks fitted, {kept} kept at epoch 0 (the "
        "day-size net it starts from)"
    )


def development_fit_line(fits: dict) -> str:
    """How many networks the development units' finished sizings fitted and how many were kept
    at epoch 0; `fits`: `development_fits` of `size_resolve_whatif` (`net`: level -> record, `ct`:
    arm -> record)."""
    net, ct = list(fits.get("net", {}).values()), list(fits.get("ct", {}).values())
    return (
        "development units' sizings, fitted once on the training units' labelled days: day-size "
        f"net {len(net)} networks, {sum(r['epoch_kept'] == 0 for r in net)} kept at epoch 0 (the "
        f"starting constant); conformal training {len(ct)} networks, "
        f"{sum(r['epoch_kept'] == 0 for r in ct)} kept at epoch 0 (the day-size net it starts "
        "from)"
    )


def _unit_lines(name: str, entry: dict, by_draw: bool = False) -> list[str]:
    lines = _head_lines(f"{name} ({entry['site_type']}, {entry['status']})", entry)
    counts = entry["evaluations"]
    if counts:
        lines.append(
            f"  evaluations: {counts['day_scale_pairs']} (day, scale) pairs; one-way "
            f"re-solves {counts['re_solves_of_the_one_way_rung']}"
        )
    return lines + _tier_lines(entry["tiers"], by_draw)


def summary_lines(document: dict) -> list[str]:
    run = document.get("run", {})
    development = list(run.get("development_units") or [])
    lines = [
        document["label"],
        document["definitions"]["what"],
        f"model {run.get('model')}, configuration {run.get('matrix_configuration')} (dial "
        f"{run.get('dial')}, kappa {run.get('kappa')}), {run.get('splits')} splits; {ROUTE}; "
        "oracle lines use the day's own outcome.",
    ]
    if "net_fits" in document:
        lines.append(net_fit_line(document["net_fits"]))
    if "ct_fits" in document:
        lines.append(ct_fit_line(document["ct_fits"]))
    counted = document.get("control_evaluations")
    if counted is not None:
        lines += control_lines({u: c for u, c in counted.items() if u not in development})
    for name, entry in document["units"].items():
        if name not in development:
            lines += _unit_lines(name, entry)
    for kind, entry in document["pooled_by_site_type"].items():
        lines += _head_lines(f"pooled {kind} ({', '.join(entry['units_scored'])} scored)", entry)
        lines += _tier_lines(entry["tiers"])
    if development:
        lines.append(f"development units: {document['definitions']['development_units']}")
        if "development_fits" in document:
            lines.append(development_fit_line(document["development_fits"]))
        for name in development:
            if counted is not None and name in counted:
                lines += control_lines({name: counted[name]})[1:]
            if name in document["units"]:
                lines += _unit_lines(name, document["units"][name], by_draw=True)
    if document["failed_units"]:
        lines.append("FAILED UNITS: " + "; ".join(document["failed_units"]))
    if document.get("failed_controls"):
        found = "; ".join(document["failed_controls"])
        lines.append(f"FAILED CONTROLS (the unit's other figures kept): {found}")
    return lines

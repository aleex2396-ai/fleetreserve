"""The figures and the summary text of the centre-scale what-if (`centre_scale_whatif`).

Pure functions: numpy only, no files, no solver.

Kinds. `"unmixed"` is the gate's own policy; each room factor `c` of `centre_scale_day` has its own
mixed kind (`kind_name`): `"mixed"` for `c = 1` (the mix as first built, so a run of factor 1 alone
writes the keys it wrote before room factors existed) and `"mixed_room_<c>"` for every other
factor, for example `"mixed_room_4"`.

Per split (`split_record`), from the gate-format results of each kind and rung (what
`gate_calibrate.calibrate_rung`, `score_test` and `score_tier_b` give, tier B under the key
`tier_b`), per tier (`tier_figures`): the scale, its reason when it is 0, the offer per counting
test day, the share of each ceiling (ratio of sums), the numbers of offering and failing test days
and the mean slot and driver loss over the offering test days (None at scale 0); tier A adds the
scales of the slot loss alone and of the driver loss alone and the calibration's evaluations
(scale, mean slot and driver loss after the running maximum, whether the bound held); tier B adds
the stay-to-bound check. Beside them, per rung, tier and mixed kind, the paired difference of that
kind minus the unmixed one of the offer and of the scale on that split (key `<kind>_minus_unmixed`).

Over splits (`over_splits`): per rung, tier and kind, the mean, 5th and 95th percentile (numpy's
linear interpolation) of each figure over the splits where it is defined (a scale and an offer are
defined on every split, 0 when the scale is 0; a loss only where the split offers and has an
offering test day), with that number of splits; the same for the paired differences.

The summary (`summary_lines`) gives first the label, the run, the days whose front-loaded plan is
outside the set, the identity checks and the timeouts; then the floor of the mix (the losses at the
smallest grid scale, mixed beside unmixed); then per unit, dial, tier and rung the calibrated
figures, mixed beside unmixed. With one split each figure is that split's; with more, the mean and
its 5th and 95th percentile. These parts show room factor 1. When the run has other room factors
(`run.room_factors`), the timeouts of each follow the timeouts line, and the summary ends with one
table (`_room_lines`): per unit, dial, rung, tier and room factor, then the unmixed policy, the
floor, the mean calibrated scale, offer and losses, and the offer minus the unmixed one, paired by
split, with its 5th and 95th percentile. It forms no verdict.
"""

import numpy as np

from fleetreserve import crc
from fleetreserve.gate_config import TIER_LABELS

MIXED = "mixed"  # the kind of room factor 1
UNMIXED = "unmixed"
ROOM = "mixed_room_"  # the kind of any other room factor: ROOM + the factor
DIFFERENCE = "_minus_unmixed"  # a mixed kind's paired difference: kind + DIFFERENCE
TIERS = ("A", "B")
POINTS = (0.05, 0.95)
LISTED = 20  # days outside the set named in the summary; the results name every one
LABEL = (
    "WHAT-IF, not a run of the plan: the scale also moves the planned charging toward the "
    "front-loaded plan (centre (1 - beta) x front-loaded plan + beta x certified plan, widths "
    "beta x certified widths, beta up to 1); it reports numbers and decides nothing"
)
FIGURES = (
    "scale",
    "offer_kwh_per_day",
    "mean_slot_loss_offering_days",
    "mean_driver_loss_offering_days",
    "failing_test_days",
)


def tier_figures(result: dict, tier: str) -> dict:
    """The figures of one tier from one kind's gate-format results of one rung (module text)."""
    scored = result if tier == "A" else result["tier_b"]
    test = scored["test"]
    out = {
        "scale": float(scored["beta_hat"]),
        "reason": scored["reason"],
        "offering_test_days": test["offering_days"],
        "failing_test_days": test["failing_days"],
        "offer_kwh_per_day": test["mean_offer_kwh_per_day"],
        "share_of_ceiling": {
            name: None if share is None else share["ratio_of_sums"]
            for name, share in test["share_of_ceiling"].items()
        },
        "mean_slot_loss_offering_days": test["mean_slot_loss_offering_days"],
        "mean_driver_loss_offering_days": test["mean_driver_loss_offering_days"],
    }
    if tier == "A":
        out["slot_only_scale"] = float(result["slot_only"]["beta_hat"])
        out["driver_only_scale"] = float(result["driver_only"]["beta_hat"])
        out["evaluations"] = result["evaluations"]
    else:
        out["stay_to_bound"] = scored["stay_to_bound"]
    return out


def kind_name(factor: float) -> str:
    """The kind of room factor `factor` (module text)."""
    return MIXED if factor == 1.0 else f"{ROOM}{factor:g}"


def factor_of(kind: str) -> float:
    """The room factor of a mixed kind (the inverse of `kind_name`)."""
    return 1.0 if kind == MIXED else float(kind.removeprefix(ROOM))


def mixed_kinds(entry) -> list[str]:
    """The mixed kinds among the keys of `entry` (a split's or the over-splits' figures of one rung
    or tier), in the order of their factors."""
    found = (k for k in entry if k == MIXED or (k.startswith(ROOM) and not k.endswith(DIFFERENCE)))
    return sorted(found, key=factor_of)


def room_factors(document: dict) -> list[float]:
    """The room factors of a results document: `run.room_factors`, absent when 1 is the only one."""
    return [float(c) for c in document["run"].get("room_factors", [1.0])]


def _difference(mixed, unmixed):
    return None if mixed is None or unmixed is None else mixed - unmixed


def split_record(seed: int, found: dict) -> dict:
    """One split's figures: `found[kind][rung]` the gate-format results of `"unmixed"` and of each
    mixed kind (module text)."""
    rungs = list(found[UNMIXED])
    mixed = mixed_kinds(found)
    record = {"seed": seed, "rungs": {}}
    for rung in rungs:
        kinds = {
            kind: {tier: tier_figures(found[kind][rung], tier) for tier in TIERS}
            for kind in (*mixed, UNMIXED)
        }
        for kind in mixed:
            kinds[kind + DIFFERENCE] = {
                tier: {
                    field: _difference(kinds[kind][tier][field], kinds[UNMIXED][tier][field])
                    for field in ("offer_kwh_per_day", "scale")
                }
                for tier in TIERS
            }
        record["rungs"][rung] = kinds
    return record


def stats(values) -> dict:
    """Mean, 5th and 95th percentile over the values that are not None, with their number."""
    kept = [float(v) for v in values if v is not None]
    if not kept:
        return {"splits": 0, "mean": None, "p5": None, "p95": None}
    low, high = (float(q) for q in np.quantile(kept, POINTS))
    return {"splits": len(kept), "mean": float(np.mean(kept)), "p5": low, "p95": high}


def over_splits(records: list[dict], ceilings) -> dict:
    """Per rung, tier and kind the figures over splits, and the paired differences (module
    text)."""
    if not records:
        return {}
    out = {}
    for rung in records[0]["rungs"]:
        out[rung] = {}
        mixed = mixed_kinds(records[0]["rungs"][rung])
        for tier in TIERS:
            entry = {}
            for kind in (*mixed, UNMIXED):
                found = [r["rungs"][rung][kind][tier] for r in records]
                entry[kind] = {field: stats([f[field] for f in found]) for field in FIGURES}
                entry[kind]["share_of_ceiling"] = {
                    name: stats([f["share_of_ceiling"][name] for f in found]) for name in ceilings
                }
            for kind in mixed:
                entry[kind + DIFFERENCE] = {
                    field: stats(
                        [r["rungs"][rung][kind + DIFFERENCE][tier][field] for r in records]
                    )
                    for field in ("offer_kwh_per_day", "scale")
                }
            out[rung][tier] = entry
    return out


def floor_figures(mixed, unmixed, offering: int, scale: float, others=None) -> dict:
    """The floor of one rung: per kind the mean slot and driver loss over the offering days at
    `scale`; `mixed` (room factor 1), `unmixed` and each value of `others` (kind -> losses, the
    other room factors): (slot losses, driver losses)."""
    out = {"scale": scale, "offering_days": offering}
    kinds = {MIXED: mixed, UNMIXED: unmixed, **(others or {})}
    for kind, (slot, driver) in kinds.items():
        out[kind] = {
            "mean_slot_loss": float(np.mean(slot)) if len(slot) else None,
            "mean_driver_loss": float(np.mean(driver)) if len(driver) else None,
        }
    return out


# --- the text ------------------------------------------------------------------------------------


def _number(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _figure(entry: dict, digits: int) -> str:
    """One figure over splits: the value with one split, else the mean [p5, p95]."""
    if entry["splits"] == 0:
        return "-"
    if entry["splits"] == 1:
        return _number(entry["mean"], digits)
    return (
        f"{_number(entry['mean'], digits)} [{_number(entry['p5'], digits)}, "
        f"{_number(entry['p95'], digits)}]"
    )


def _pair(entry: dict, field: str, digits: int) -> str:
    return f"{_figure(entry['mixed'][field], digits)} / {_figure(entry['unmixed'][field], digits)}"


def dial_label(configuration: dict) -> str:
    dial = configuration["driver_risk"]
    return "no driver-risk rows" if dial is None else f"dial {dial:g}"


def _outside_lines(document: dict) -> list[str]:
    named = [
        f"{unit} | {name} | {day['day']} (slack {day['smallest_slack_kwh']:.3g})"
        for unit, entry in document["units"].items()
        for name, found in entry["configurations"].items()
        for day in found.get("front_loaded_outside", [])
    ]
    head = (
        f"Front-loaded plan outside the set: {len(named)} days "
        "(each replayed unmixed at every scale)"
    )
    if not named:
        return [head]
    more = len(named) - LISTED
    tail = [f"  and {more} more (every one is named in the results)"] if more > 0 else []
    return [head + ":", *(f"  {line}" for line in named[:LISTED]), *tail]


def _identity_lines(document: dict) -> list[str]:
    identity = document["identity"]
    return [
        (
            "Identity: the unmixed figures of the gate run's split "
            f"(seed {identity['seed']}) equal "
            f"the gate's (gate_calibrate.run_job) on {identity['gate_checked']} unit and "
            "configuration pairs; at scale 1 the mixed policy's arrays are the certified "
            f"policy's on {identity['policies_checked']} offering days."
        ),
        (
            "At scale 1 (every offering day replayed there under every room factor and unmixed) "
            "the mixed replay equals the unmixed one on "
            f"{identity['scale_one_compared'] - identity['scale_one_differ_with_timeout']} of "
            f"{identity['scale_one_compared']} day and room factor pairs; "
            f"{identity['scale_one_differ_with_timeout']} differ, each with a re-solve that "
            "timed out."
        ),
    ]


def _room_timeout_lines(document: dict) -> list[str]:
    """The re-solve timeouts of the room factors other than 1 (none when 1 is the only one)."""
    others = [c for c in room_factors(document) if c != 1.0]
    if not others:
        return []
    timeouts = document["timeouts"]
    counts = [f"room {c:g} {timeouts.get(f're_solves_{kind_name(c)}', 0)}" for c in others]
    return [f"Timeouts of the other room factors: re-solves {', '.join(counts)}."]


def _mean(entry: dict, digits: int) -> str:
    return "-" if entry["splits"] == 0 else _number(entry["mean"], digits)


def _room_lines(document: dict) -> list[str]:
    """The table of every room factor beside the unmixed policy (module text); none when 1 is the
    only factor."""
    factors = room_factors(document)
    if factors == [1.0]:
        return []
    run = document["run"]
    shown = "the split's value" if run["splits"] == 1 else "mean [p5, p95]"
    head = (
        f"Room factors {', '.join(f'{c:g}' for c in factors)}: at scale beta the planned charging "
        "is (1 - w) x the front-loaded plan + w x the certified plan with w = min(1, room x "
        "beta), widths beta x the certified widths (room 1 is the mix above; w = 1, the unmixed "
        "policy, wherever room x beta >= 1). Per unit, dial, rung, tier and room factor, then the "
        f"unmixed policy: the floor at scale {run['floor_scale']:g} (mean slot and driver loss "
        "over the offering days, the same under both tiers); over the splits, the mean "
        "calibrated scale (mixed capped at 1, unmixed at 2), offer in kWh per counting test day "
        "and slot and driver loss over the offering test days; the offer minus the unmixed one, "
        f"paired by split ({shown})."
    )
    rows = [
        (
            "unit",
            "dial",
            "rung",
            "tier",
            "room",
            "floor slot",
            "floor driver",
            "scale",
            "offer",
            "slot",
            "driver",
            "offer minus unmixed",
        )
    ]
    kinds = [*(kind_name(c) for c in factors), UNMIXED]
    for unit, entry in document["units"].items():
        for found in entry["configurations"].values():
            if "over_splits" not in found:
                continue
            for rung, tiers in found["over_splits"].items():
                floor = found["floor"][rung]
                for tier in TIERS:
                    e = tiers[tier]
                    for kind in kinds:
                        mixed = kind != UNMIXED
                        rows.append(
                            (
                                unit,
                                dial_label(found),
                                rung,
                                tier,
                                f"{factor_of(kind):g}" if mixed else UNMIXED,
                                _number(floor[kind]["mean_slot_loss"]),
                                _number(floor[kind]["mean_driver_loss"]),
                                _mean(e[kind]["scale"], 3),
                                _mean(e[kind]["offer_kwh_per_day"], 2),
                                _mean(e[kind]["mean_slot_loss_offering_days"], 3),
                                _mean(e[kind]["mean_driver_loss_offering_days"], 3),
                                _figure(e[kind + DIFFERENCE]["offer_kwh_per_day"], 2)
                                if mixed
                                else "-",
                            )
                        )
    widths = [max(len(row[k]) for row in rows) for k in range(len(rows[0]))]
    table = [
        " | ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip()
        for row in rows
    ]
    return [head, *table]


def _floor_lines(document: dict) -> list[str]:
    scale = document["run"]["floor_scale"]
    lines = [
        (
            f"Floor of the mix: the losses at scale {scale:g} (the smallest grid scale), mean over "
            "the offering days, mixed / unmixed; the same under both tiers. Limits: slot "
            f"{crc.ALPHA_SLOT:g}, driver {crc.ALPHA_DRIVER:g} (tier A)."
        )
    ]
    for unit, entry in document["units"].items():
        for found in entry["configurations"].values():
            if "floor" not in found:
                continue
            for rung, floor in found["floor"].items():
                mixed, unmixed = floor["mixed"], floor["unmixed"]
                lines.append(
                    f"  {unit} | {dial_label(found)} | {rung}: {floor['offering_days']} days; "
                    f"slot {_number(mixed['mean_slot_loss'])} / "
                    f"{_number(unmixed['mean_slot_loss'])}; driver "
                    f"{_number(mixed['mean_driver_loss'])} / {_number(unmixed['mean_driver_loss'])}"
                )
    return lines


def _figure_lines(document: dict) -> list[str]:
    run = document["run"]
    ceiling = run["gate_ceiling"]
    splits = run["splits"]
    shown = "the split's value" if splits == 1 else f"{splits} splits: mean [p5, p95]"
    lines = [
        (
            f"Calibrated figures per unit, dial, tier and rung, mixed / unmixed ({shown}): scale; "
            "offer in kWh per counting test day; offer mixed minus unmixed, paired by split; share "
            f"of the {ceiling} ceiling; mean slot and driver loss over the offering test days; "
            "failing test days."
        )
    ]
    for tier in TIERS:
        lines.append(TIER_LABELS[tier])
        for unit, entry in document["units"].items():
            for found in entry["configurations"].values():
                if "over_splits" not in found:
                    continue
                for rung, tiers in found["over_splits"].items():
                    e = tiers[tier]
                    share = (
                        f"{_figure(e['mixed']['share_of_ceiling'][ceiling], 3)} / "
                        f"{_figure(e['unmixed']['share_of_ceiling'][ceiling], 3)}"
                    )
                    lines.append(
                        f"  {unit} | {dial_label(found)} | {rung}: scale {_pair(e, 'scale', 3)}; "
                        f"offer {_pair(e, 'offer_kwh_per_day', 2)}; mixed minus unmixed "
                        f"{_figure(e['mixed_minus_unmixed']['offer_kwh_per_day'], 2)}; share "
                        f"{share}; slot {_pair(e, 'mean_slot_loss_offering_days', 3)}; driver "
                        f"{_pair(e, 'mean_driver_loss_offering_days', 3)}; failing "
                        f"{_pair(e, 'failing_test_days', 1)}"
                    )
    return lines


def summary_lines(document: dict) -> list[str]:
    """The summary text of a results document (module text)."""
    run = document["run"]
    timeouts = document["timeouts"]
    failed = document["failed_configurations"]
    lines = [
        LABEL,
        (
            f"Model {document['model']}; matrix {document['matrix']['file']}; {run['splits']} "
            f"split(s), seeds {run['seeds'][0]} to {run['seeds'][-1]} (seed {run['split_seed']} "
            f"is the gate run's split); configurations: {', '.join(run['configurations'])}."
        ),
        (
            "Scales: mixed on the grid up to 1 (the mix is not defined above 1); unmixed on the "
            "gate's grid up to 2, as the gate calibrates."
        ),
        *_outside_lines(document),
        *_identity_lines(document),
        (
            f"Timeouts: certificates {timeouts.get('certificates', 0)}; re-solves mixed "
            f"{timeouts.get('re_solves_mixed', 0)}, unmixed "
            f"{timeouts.get('re_solves_unmixed', 0)}; "
            f"best-causal ceilings {timeouts.get('best_causal_ceilings', 0)}."
        ),
        *_room_timeout_lines(document),
        "Failed configurations: " + ("none" if not failed else "; ".join(failed)),
    ]
    for unit, entry in document["units"].items():
        if entry["status"] != "calibratable":
            lines.append(f"{unit}: {entry['status']}")
    return [*lines, *_floor_lines(document), *_figure_lines(document), *_room_lines(document)]

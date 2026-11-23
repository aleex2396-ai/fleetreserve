"""The plain-text summary of a cold-start results document (`cold_start`).

The label first (a run on development units is a rehearsal, not outcome O9), then the models,
the configuration and the words of A_k and Y_k; per unit its counting and scored days, then one
table per tier: one row per model and k with the offering local days, the scale and why it is 0,
the offer per counting scored day, A_k, Y_k against the hindsight ceiling and against the
best-causal one, and the failing scored days. The tiers are two separate blocks; nothing is added
or averaged across them, across units or across k. Timeouts and failures close the summary.
"""

from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.sealed_route import abstention_lines

COLUMNS = (
    ("model", 22),
    ("k", 3),
    ("offering local", 15),
    ("scale", 6),
    ("offer kWh/day", 14),
    ("A_k", 6),
    ("Y_k hindsight", 14),
    ("Y_k best-causal", 16),
    ("offering scored", 16),
    ("failing", 8),
    ("scale 0 because", 0),
)


def _number(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _row(values) -> str:
    """One table row; a row with fewer values than columns fills the first columns only."""
    cells = [
        str(v).ljust(width) if width else str(v)
        for v, (_, width) in zip(values, COLUMNS, strict=False)
    ]
    return "    " + " ".join(cells).rstrip()


def _share(found: dict, name: str):
    return found["offer_share"].get(name, {}).get("share_of_ceiling")


def summary_lines(document: dict) -> list[str]:
    run, models = document["run"], document["models"]
    hindsight, best = run["ceilings"][0], run["ceilings"][1]
    lines = [
        document["label"],
        (
            f"cold start (plan section 11, O9): {models['cold_started']} against "
            f"{models['reference']}; configuration {run['configuration']} (dial {run['dial']:g}, "
            f"kappa {run['kappa']:g}); the {run['rung']} rung; k = "
            f"{', '.join(str(k) for k in run['local_days'])} local days in time order"
        ),
        (
            f"scored days: the counting days after the first {run['largest_k']}, the same at "
            "every k; offer: kWh per counting scored day; A_k = 1 - mean slot loss over the "
            "offering scored days; Y_k = offer / mean ceiling per counting scored day ("
            f"{hindsight} is the plan's ceiling, {best} beside it)"
        ),
        (
            "a scored day's user histories hold the local sessions that had left by the latest "
            "decision time of the k local days (C2a); the truth of the local days is the truth "
            "build's, set from all of a unit's days (C14); readings C1 to C14 are in the "
            "cold_start module text"
        ),
    ]
    lines += abstention_lines(document.get("held_out_inputs"))  # a sealed unit left out
    for unit, entry in document["units"].items():
        lines.append("")
        scored = entry["scored_days"]
        if not scored:
            lines.append(f"{unit}: {entry['status']}")
            continue
        lines.append(
            f"{unit}: {entry['counting_days']} counting days; {len(scored)} scored days "
            f"({scored[0]} to {scored[-1]})"
        )
        for tier in run["tiers"]:
            lines.append(f"  {TIER_LABELS[tier]}")
            lines.append(_row(name for name, _ in COLUMNS))
            for model in (models["cold_started"], models["reference"]):
                by_k = document["figures"].get(model, {}).get(unit, {})
                for k in run["local_days"]:
                    found = by_k.get(str(k), {}).get(tier)
                    if found is None:
                        lines.append(_row((model, k, "not scored: see failed")))
                        continue
                    lines.append(
                        _row(
                            (
                                model,
                                k,
                                found["offering_local_days"],
                                _number(found["scale"], 2),
                                _number(found["offer_kwh_per_counting_scored_day"]),
                                _number(found["availability"]),
                                _number(_share(found, hindsight)),
                                _number(_share(found, best)),
                                found["offering_scored_days"],
                                "-"
                                if found["failing_scored_days"] is None
                                else found["failing_scored_days"],
                                found["scale_zero_reason"] or "",
                            )
                        )
                    )
    lines.append("")
    timeouts = document["timeouts"]
    lines.append(
        "timeouts: "
        + (", ".join(f"{name} {n}" for name, n in timeouts.items()) if timeouts else "none")
    )
    lines.append("failed: " + ("; ".join(document["failed"]) if document["failed"] else "none"))
    return lines

"""Counts, shares and quantiles over the rows of the re-solve diagnosis, and its text summary.

Pure arithmetic on the rows `resolve_diagnosis_day.rows` gives (no files, no solver). A
measurement only: nothing here names a choice or a recommendation.

`summarise(rows)`, for any set of rows (one unit at one scale, or at all scales):

- `re_solves`, `by_outcome`: the re-solves attempted and their outcomes;
  `day_paths_with_a_re_solve`;
- `infeasible`, over the infeasible re-solves (shares are of their number):
  - `recheck`: the answers of the same rows with objective 0;
  - `by_cause`: `resolve_diagnosis_lp.cause`, the whole set of removals that make a re-solve
    feasible, as one label; the labels partition the infeasible re-solves;
  - `single_group`: per group, the re-solves made feasible by removing that group alone (a
    re-solve counts under every such group, so the shares can add up to more than 1);
  - `departure_relaxed`: the answers with every present car taken to stay;
  - `box_scale`, `need_slack_kwh`, `request_miss_kwh`: per distance, its programme's outcomes,
    and over the re-solves where it ended optimal (`count`) the mean and the quantiles
    `QUANTILES` (linear interpolation); for the request miss the quantiles are of its size and
    `short`, `over` count its sign (delivery below, above the request);
  - `after_early_leaver`: the re-solves with at least one steered car that had truly left before
    its forecast departure by that hour (hindsight, for measurement only), and the quantiles of
    the energy the fixed plan still planned for those cars, over those re-solves;
  - `by_hour`: the infeasible re-solves per service hour;
  - `relaxed_timeouts`, `relaxed_errors`: relaxed programmes that ended in a timeout or an
    error (after the retry), and `re_solves_with_a_relaxed_timeout`;
    `solver_first_errors`, `solver_retried_answered`: relaxed programmes whose first attempt
    ended in a solver error, and of those the retries (simplex method) that ended optimal or
    infeasible.

`DEFINITIONS` are the lines of text that say what the box scale and the early-leaver count mean;
the text summary prints them under its heading and the results document carries them.
"""

from collections import Counter

import numpy as np

from fleetreserve.resolve_diagnosis_lp import GROUPS

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
LABEL = (
    "re-solve diagnosis: a measurement of the one-way rung's re-solves; it changes no rung and no "
    "gate result"
)
NONE_INFEASIBLE = "no infeasible re-solve"
DEFINITIONS = {
    "box_scale": (
        "largest box scale s: the largest s in [0, 1] for which the re-solve is feasible when "
        "the remaining box is its centre plus s times its half-widths, the predecessor plan's "
        "gains on the future requests are also s times as large, and the request of the current "
        "hour (already known) is as it is; s = 1 is the re-solve itself, so for an infeasible "
        "re-solve the value is below 1; 'none' (outcome infeasible) means not even s = 0 is "
        "feasible; the feasible values of s form an interval from 0 up to the value reported"
    ),
    "early_leaver": (
        "after an early leaver: the infeasible re-solves for which, by the start of the hour, "
        "at least one steered car had truly left (departure hour d at most the hour) before the "
        "forecast departure d_hat made at the decision time (d below d_hat); d_hat is neither "
        "the bound the re-solve used nor the declared time; every steered car of the day counts, "
        "including cars the rung had already seen gone; the count is cumulative along a path, so "
        "a car is counted again at each later infeasible re-solve of that path"
    ),
    "solver_errors": (
        "a relaxed programme whose first attempt (interior point) ended in a solver error is "
        "tried once more with the simplex method; a timeout is never retried and never read as "
        "infeasible"
    ),
}


def _counts(values, total: int) -> dict:
    counted = Counter(values)
    return {
        key: {"count": counted[key], "share": counted[key] / total if total else None}
        for key in sorted(counted)
    }


def _points(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {"count": 0, "mean": None, "quantiles": None}
    quantiles = np.quantile(values, QUANTILES)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "quantiles": {
            f"p{round(100 * q)}": float(v) for q, v in zip(QUANTILES, quantiles, strict=True)
        },
    }


def _distance(rows, column: str, outcome: str, size: bool = False) -> dict:
    values = [float(r[column]) for r in rows if r[column] != ""]
    out = {"by_outcome": dict(sorted(Counter(r[outcome] for r in rows).items()))}
    out.update(_points(np.abs(values) if size else values))
    if size:
        out["short"] = sum(v > 0 for v in values)
        out["over"] = sum(v < 0 for v in values)
    return out


def summarise(rows: list[dict]) -> dict:
    """The summary of a set of rows (module text)."""
    infeasible = [r for r in rows if r["outcome"] == "infeasible"]
    n = len(infeasible)
    early = [r for r in infeasible if r["early_gone_cars"] > 0]
    paths = {(r["site"], r["date"], r["scale"], r["path_index"]) for r in rows}
    return {
        "re_solves": len(rows),
        "by_outcome": dict(sorted(Counter(r["outcome"] for r in rows).items())),
        "day_paths_with_a_re_solve": len(paths),
        "infeasible": {
            "count": n,
            "recheck": _counts((r["recheck"] for r in infeasible), n),
            "by_cause": _counts((r["cause"] for r in infeasible), n),
            "single_group": {
                group: {
                    "count": (count := sum(group in r["single"].split("|") for r in infeasible)),
                    "share": count / n if n else None,
                }
                for group in GROUPS
            },
            "departure_relaxed": _counts((r["departure_relaxed"] for r in infeasible), n),
            "box_scale": _distance(infeasible, "box_scale", "box_scale_outcome"),
            "need_slack_kwh": _distance(infeasible, "need_slack_kwh", "need_slack_outcome"),
            "request_miss_kwh": _distance(
                infeasible, "request_miss_kwh", "request_miss_outcome", size=True
            ),
            "after_early_leaver": {
                "count": len(early),
                "share": len(early) / n if n else None,
                "expected_kwh": _points(float(r["early_gone_expected_kwh"]) for r in early),
            },
            "by_hour": {
                str(hour): count
                for hour, count in sorted(Counter(r["hour"] for r in infeasible).items())
            },
            "relaxed_timeouts": sum(r["relaxed_timeouts"] for r in infeasible),
            "relaxed_errors": sum(r["relaxed_errors"] for r in infeasible),
            "solver_first_errors": sum(r["solver_first_errors"] for r in infeasible),
            "solver_retried_answered": sum(r["solver_retried_answered"] for r in infeasible),
            "re_solves_with_a_relaxed_timeout": sum(r["relaxed_timeouts"] > 0 for r in infeasible),
        },
    }


def _share(entry) -> str:
    return f"{entry['count']} ({entry['share']:.1%})"


def _quantiles(found: dict, unit: str) -> str:
    if not found["count"]:
        return "no value"
    points = ", ".join(f"{name} {value:.4g}" for name, value in found["quantiles"].items())
    return f"{found['count']} values; mean {found['mean']:.4g}{unit}; {points}"


def _outcomes(counts: dict) -> str:
    return ", ".join(f"{name} {count}" for name, count in counts.items()) or "none"


def _block(found: dict) -> list[str]:
    lines = [f"  re-solves: {found['re_solves']} ({_outcomes(found['by_outcome'])})"]
    bad = found["infeasible"]
    if bad["count"] == 0:
        lines.append(f"  {NONE_INFEASIBLE}")
        return lines
    lines.append(f"  infeasible re-solves: {bad['count']}; by service hour: {bad['by_hour']}")
    lines.append(
        "  the same rows with objective 0: "
        + ", ".join(f"{k} {_share(v)}" for k, v in bad["recheck"].items())
    )
    lines.append("  feasible after removing one group alone (a re-solve can count more than once):")
    for group, entry in bad["single_group"].items():
        lines.append(f"    {group}: {_share(entry)}")
    lines.append("  by the whole set of removals that make it feasible:")
    ordered = sorted(bad["by_cause"].items(), key=lambda item: (-item[1]["count"], item[0]))
    for label, entry in ordered:
        lines.append(f"    {label}: {_share(entry)}")
    lines.append(
        "  with every present car taken to stay: "
        + ", ".join(f"{k} {_share(v)}" for k, v in bad["departure_relaxed"].items())
    )
    for key, title, unit in (
        ("box_scale", "largest box scale s (none = outcome infeasible)", ""),
        ("need_slack_kwh", "smallest total slack on the needed energy", " kWh"),
        ("request_miss_kwh", "smallest miss of the hour's request (size)", " kWh"),
    ):
        entry = bad[key]
        lines.append(f"  {title}: {_quantiles(entry, unit)}")
        lines.append(f"    its programme's outcomes: {_outcomes(entry['by_outcome'])}")
    miss = bad["request_miss_kwh"]
    lines.append(f"    delivery below the request: {miss['short']}; above: {miss['over']}")
    after = bad["after_early_leaver"]
    lines.append(
        "  after at least one steered car truly left before its forecast departure (hindsight): "
        + _share(after)
    )
    lines.append(
        f"    energy the fixed plan still planned for those cars: "
        f"{_quantiles(after['expected_kwh'], ' kWh')}"
    )
    lines.append(
        f"  relaxed programmes that timed out: {bad['relaxed_timeouts']} "
        f"(in {bad['re_solves_with_a_relaxed_timeout']} re-solves); "
        f"that ended in an error: {bad['relaxed_errors']}; "
        f"first attempt in a solver error: {bad['solver_first_errors']}, of which the simplex "
        f"retry answered: {bad['solver_retried_answered']}"
    )
    return lines


def summary_lines(document: dict) -> list[str]:
    """The text summary of the results document `resolve_diagnosis.run` builds."""
    run = document["run"]
    lines = [
        document["label"],
        f"model {document.get('model', 'not named')}; configuration {run['configuration']}; "
        f"rung {run['rung']}; scales " + ", ".join(run["scales"]),
        "definitions:",
        *(f"  {text}" for text in DEFINITIONS.values()),
        "",
    ]
    for name, entry in document["units"].items():
        lines.append(f"unit {name}")
        if "failed" in entry:
            lines += [f"  FAILED: {entry['failed']}", ""]
            continue
        lines.append(
            f"  counting days {entry['counting_days']}, of which offering "
            f"{entry['offering_days']}; {entry['paths_per_day']} audited paths a day"
        )
        for scale, found in entry["scales"].items():
            lines.append(f" scale {scale}")
            lines += _block(found)
        if len(entry["scales"]) > 1:
            lines.append(" all scales together")
            lines += _block(entry["all_scales"])
        lines.append("")
    counts = document["timeouts"]
    lines.append(
        f"timeouts: certificates {counts['certificates']}, re-solves {counts['re_solves']}, "
        f"relaxed programmes {counts['relaxed_programmes']}"
    )
    return lines

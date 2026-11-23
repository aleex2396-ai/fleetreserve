"""Choose the workplace window by the known-car ceiling on the training workplaces (plan section 2).

Command line, from the project's top folder:
    python -m fleetreserve.window_choice [--raw-root data/raw] [--out <folder>]

Candidates: decision time τ in 7:00, 8:00, ..., 12:00 local and |S| in 2, 3, 4, 6 service hours,
S starting one hour after τ (24 pairs). For each pair and each of the two training workplaces
Caltech and JPL, the command builds the main truth cars exactly as `truth_build` builds them
(`truth_build.acn_site_tables`: the same loader, rates, energy-before-τ rule and counting rule of
at least 3 cars plugged in at τ), turns each counting day into a `Day` as the ceiling report
does (`days_from_truth.unit_days`, ρ = 0, room capped, small fleets kept), and takes the known-car
ceiling of each day (`known_car_ceiling`, the closed form of the best causal value of the hourly
product on true sets). The score of a pair is the sum of those ceilings over the counting days of
both sites.

The ceiling is read as the known-car causal ceiling, with the power of the decision hour fixed at
the decision time (the gate's best-causal ceiling); it is not b(S) - p(S) with a free decision
hour. Cars that leave within the decision hour are left out but their day still counts; a room
that does not fit is capped at rate times departure hour; counting days whose fleet falls below
3 cars are kept. `READINGS` states these in words, and every output carries them.

The chosen pair has the largest score. Pairs whose scores differ by at most `TIE_KWH` are tied,
and a tie goes to the earlier τ and then to the shorter S; the output says whether a tie
decided the choice and lists the tied pairs.

Only `data/raw/acn` is read. No offer, forecast or calibration is computed, and no linear
programme is solved.

Outputs, in `--out` (default: the run folder when `FLEETRESERVE_RUN_DIR` is set, which the run
recorder sets, otherwise `data/derived/window-choice`; never in git): `window-choice.json` (every
number) and `window-choice.txt` (the summary, also printed). The same inputs give the same bytes.

Exit status: 0 the choice was made; 1 a file does not have its published layout, or a day could
not be built; 2 a raw data file is missing or the output folder cannot be written.
"""

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fleetreserve import load_acn
from fleetreserve.days_from_truth import NO_FIT_CAP, SMALL_KEEP, unit_days
from fleetreserve.known_car_ceiling import known_car_ceiling
from fleetreserve.sessions import RAW_ROOT, TOP, LayoutError
from fleetreserve.truth_build import acn_site_tables, csv_bytes
from fleetreserve.windows import Window

SITES = ("caltech", "jpl")
TAU_HOURS = (7, 8, 9, 10, 11, 12)
SERVICE_HOURS = (2, 3, 4, 6)
RHO = 0.0
TIE_KWH = 1e-9
RESULTS_NAME = "window-choice.json"
TEXT_NAME = "window-choice.txt"
DEFAULT_OUT = TOP / "data" / "derived" / "window-choice"
READINGS = {
    "ceiling": (
        "The ceiling is the known-car causal ceiling: the best real-time value of the hourly "
        "product on the true sets of all cars, with the power of the decision hour fixed at the "
        "decision time (the gate's best-causal ceiling, certificate on true rows). It is not "
        "b(S) - p(S) with a free decision hour (the hindsight ceiling), which can be larger."
    ),
    "decision_hour_departures": (
        "A car that leaves within the decision hour (departure hour 0) is left out of the fleet; "
        "the day still counts, because the counting rule of at least 3 cars is applied at the "
        "decision time."
    ),
    "room_cap": (
        "A room that does not fit (protected energy above rate times departure hour) is capped "
        "at rate times departure hour; the energy removed is reported per site and pair."
    ),
    "small_fleets": (
        "A counting day whose fleet falls below 3 cars once cars are left out is kept, and "
        "counts in the sums."
    ),
}
RULE = (
    "largest sum over the counting days of Caltech and JPL of the known-car ceiling "
    "(hourly product, one-way, rho 0, main truth); pairs within "
    f"{TIE_KWH:g} kWh are tied and go to the earlier tau, then the shorter S"
)

Pair = tuple[int, int]  # (τ clock hour, number of service hours)


@dataclass(frozen=True)
class Choice:
    pair: Pair
    total_kwh: float
    tied: tuple[Pair, ...]  # every pair within TIE_KWH of the best, in the order of the rule

    @property
    def decided_by_tie_rule(self) -> bool:
        return len(self.tied) > 1


def candidates() -> list[Pair]:
    return [(tau, hours) for tau in TAU_HOURS for hours in SERVICE_HOURS]


def window_for(pair: Pair) -> Window:
    tau, hours = pair
    return Window(
        tau_hour=tau, service_hours=hours, rule=f"candidate: tau {tau:02d}:00, |S| {hours}"
    )


def choose(totals: dict[Pair, float]) -> Choice:
    """The pair with the largest total; ties go to the earlier τ, then the shorter S."""
    if not totals:
        raise ValueError("no pair to choose from")
    best = max(totals.values())
    tied = tuple(sorted(pair for pair, total in totals.items() if best - total <= TIE_KWH))
    return Choice(pair=tied[0], total_kwh=totals[tied[0]], tied=tied)


def _as_text(frame: pd.DataFrame) -> pd.DataFrame:
    """The table as `truth_build` writes it and `days_from_truth` reads it."""
    return pd.read_csv(io.BytesIO(csv_bytes(frame)), dtype=str, keep_default_na=False)


def site_score(sessions: pd.DataFrame, site: str, pair: Pair) -> dict:
    """One site's counting days, cars and summed known-car ceiling for one pair."""
    cars, days = acn_site_tables(sessions, site, window_for(pair))
    built = unit_days(
        _as_text(cars),
        _as_text(days),
        unit=site,
        rho=RHO,
        no_fit=NO_FIT_CAP,
        small_fleet=SMALL_KEEP,
    )
    values = [known_car_ceiling(item.day) for item in built.days]
    counting = len(built.days)
    total = float(sum(values))
    return {
        "counting_days": counting,
        "cars_at_tau": int(sum(item.record.cars_at_tau for item in built.days)),
        "cars_in_fleet": int(sum(item.record.cars_in_fleet for item in built.days)),
        "cars_left_out_departing_in_decision_hour": int(
            sum(item.record.depart_zero_cars for item in built.days)
        ),
        "cars_with_room_capped": int(sum(item.record.no_fit_cars for item in built.days)),
        "kwh_removed_by_room_cap": round(
            float(sum(item.record.no_fit_energy_removed_kwh for item in built.days)), 6
        ),
        "days_with_fleet_below_3_kept": int(
            sum(item.record.fewer_than_three for item in built.days)
        ),
        "ceiling_kwh_sum": round(total, 6),
        "ceiling_kwh_mean_per_counting_day": round(total / counting, 6) if counting else None,
        "_exact_sum": total,
    }


def run(raw_root: Path) -> dict:
    sessions = {site: load_acn.load(site, raw_root)[0] for site in SITES}
    rows, totals = [], {}
    for pair in candidates():
        tau, hours = pair
        per_site = {site: site_score(sessions[site], site, pair) for site in SITES}
        total = sum(entry["_exact_sum"] for entry in per_site.values())
        totals[pair] = total
        rows.append(
            {
                "tau": f"{tau:02d}:00",
                "tau_hour": tau,
                "service_hours": hours,
                "service": f"{tau + 1:02d}:00 to {tau + 1 + hours:02d}:00",
                "sites": {
                    site: {k: v for k, v in entry.items() if k != "_exact_sum"}
                    for site, entry in per_site.items()
                },
                "total_ceiling_kwh": round(total, 6),
            }
        )
    choice = choose(totals)
    tau, hours = choice.pair
    return {
        "rule": RULE,
        "readings": READINGS,
        "sites": list(SITES),
        "rho": RHO,
        "pairs": rows,
        "chosen": {
            "tau": f"{tau:02d}:00",
            "tau_hour": tau,
            "service_hours": hours,
            "service": f"{tau + 1:02d}:00 to {tau + 1 + hours:02d}:00",
            "total_ceiling_kwh": round(choice.total_kwh, 6),
            "decided_by_tie_rule": choice.decided_by_tie_rule,
            "tied_pairs": [
                {"tau_hour": t, "service_hours": h, "total_ceiling_kwh": round(totals[(t, h)], 6)}
                for t, h in choice.tied
            ],
        },
    }


def report_lines(document: dict) -> list[str]:
    chosen = document["chosen"]
    lines = [
        "WINDOW CHOICE (known-car ceiling, Caltech and JPL, hourly product, rho 0)",
        f"rule: {document['rule']}",
        "",
        "READINGS",
        *[f"- {text}" for text in document["readings"].values()],
        "",
        f"CHOSEN: tau {chosen['tau']}, S {chosen['service']} ({chosen['service_hours']} hours), "
        f"summed ceiling {chosen['total_ceiling_kwh']:.3f} kWh",
    ]
    if chosen["decided_by_tie_rule"]:
        tied = ", ".join(
            f"tau {t['tau_hour']:02d}:00 with {t['service_hours']} hours"
            for t in chosen["tied_pairs"]
        )
        lines.append(f"TIE: the tie rule decided the choice; tied pairs: {tied}")
    else:
        lines.append("TIE: none; the largest sum is reached by this pair alone")
    lines += [
        "",
        "ALL PAIRS (counting days, cars in the fleet, summed ceiling in kWh, mean per counting "
        "day, kWh removed by the room cap)",
    ]
    for row in document["pairs"]:
        parts = []
        for site, entry in row["sites"].items():
            mean = entry["ceiling_kwh_mean_per_counting_day"]
            mean_text = "none" if mean is None else f"{mean:.3f}"
            parts.append(
                f"{site} {entry['counting_days']} days, {entry['cars_in_fleet']} cars, "
                f"{entry['ceiling_kwh_sum']:.3f} (mean {mean_text}), room cap removed "
                f"{entry['kwh_removed_by_room_cap']:.3f} kWh"
            )
        lines.append(
            f"- tau {row['tau']}, {row['service_hours']} hours: total "
            f"{row['total_ceiling_kwh']:.3f}; " + "; ".join(parts)
        )
    return lines


def default_out() -> Path:
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    return Path(run_dir) if run_dir else DEFAULT_OUT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else default_out()
    try:
        document = run(args.raw_root)
    except FileNotFoundError as problem:
        print(f"COULD NOT RUN: a raw data file is missing: {problem}")
        return 2
    except (LayoutError, ValueError) as problem:
        print(f"FAILED: a file or a day could not be read: {problem}")
        return 1
    text = "\n".join(report_lines(document)) + "\n"
    try:
        out.mkdir(parents=True, exist_ok=True)
        (out / RESULTS_NAME).write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        (out / TEXT_NAME).write_text(text)
    except OSError as problem:
        print(f"COULD NOT RUN: cannot write the results: {problem}")
        return 2
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

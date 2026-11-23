"""Split the one-way rung's offer and losses by the number of steered cars of a day, and recompute
the 200-split offers of `dial_choice` as if days with fewer than N steered cars did not offer
(plan section 6).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.fleet_size_whatif --model <name> --min-cars <N> [<N> ...]
        [--matrix <file>] [--splits N] [--workers N] [--configuration <name> ...]
        [--unit <name> ...] [--tables <folder>] [--truth-run <folder>]

A measurement and a what-if. The plan has no minimum number of steered cars for a day to offer;
this command adds none, changes no rung, no gate, no calibration rule and no existing result, and
writes no choice and no recommendation (`definitions` in the results says so).

Everything but the minimum is `dial_choice`, whose functions it calls: the model is checked, fitted
and turned into the forecaster by `dial_choice.make_factory`; the truth tables are checked as
`gate_run.check_inputs` checks them (a verified truth build at the commit checked out now, of the
tables' setting, with the tables' hashes); the seeded splits (seeds 1 to `--splits`, each made by
`day_split` with the matrix's calibration-days rule: `n_calibration` days, or half of the unit's
days under `calibration_half_below`; `dial_choice.split_indices`), the grid of scales
(`dial_splits.GRID`), the two tiers and the calibration and scoring of each split are those of
`dial_choice`. `--matrix`: a run matrix of the gate's format (default: the committed one); its gate
units are the units measured (`--unit`, repeatable, keeps some of them). `--configuration`,
repeatable: the configurations by name; by default the ones `dial_choice` evaluates
(`dial_choice.find_family`: the configuration without driver-risk rows and each dial). `--min-cars`:
one or more whole numbers, 0 or more, all different; the forecast table is built at the largest
kappa of the configurations.

Steered cars. The number of steered cars of a counting day is `len(PreparedDay.steered)` of
`gate_day.prepare_day`: the cars of the fleet in the steered sub-day, that is, those the fixed plan
steers (a car with a forecast whose forecast set is not empty), none when the configuration's
rule is day-abstains and a car is uncertifiable. The field keeps its cars also on a day that
abstains for another reason (a zero value, no certificate), so the count belongs to every counting
day. Cars outside the box (no reference sample, an empty forecast set) are not counted.

Per unit that can be split and configuration, every day is prepared (`gate_day.prepare_day`) and
every day that offers is evaluated once at each scale of the grid (`dial_choice.measure`, one
cache), so no (day, scale) is evaluated twice and each is reused for every N. The one-way rung's
day-level slot and driver losses are read. Then:
- the per-day rows (`ROWS_NAME`): the count, whether the day offers, the offer at scale 1 and the
  two losses at every scale;
- the table by exact count of steered cars (`fleet_size_whatif_core.by_count`), no class edges;
- for each N, the figures `dial_choice` reports for the unit, configuration and tier over the
  seeded splits, and its gate-reading figures (tier A, over all splits: mean offer, mean ceilings
  and shares, the oracle partner's mean offer and its ratio to the offer), with every day of fewer
  than N steered cars (calibration and test days alike) treated as a day that does not offer
  (`fleet_size_whatif_core.restrict`). N = 0 gives `dial_choice`'s own figures.

The oracle partner of a configuration (its `oracle_partner` field) is prepared and measured like
the configuration itself, as `dial_choice.gate_reading` does it, and its days are restricted by
its own count of steered cars; the per-day rows carry the partner's days too (`role` column) and
the results count the days on which the two counts differ. A configuration without an oracle
partner has no gate-reading figures. `dial_choice.gate_reading` itself is not called, because it
measures the partner again for every call; the same functions it uses are (`dial_choice.measure`,
`dial_splits.gate_split`, `dial_splits.summarise_gate`).

Timeouts are recorded outcomes, counted as `dial_choice` counts them: a day whose certificate
times out abstains; a re-solve that times out leaves the rung on its predecessor's plan; a missing
best-causal ceiling leaves the ceiling share of the splits that test on that day out.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output. They hold no wall-clock time and do not depend on `--workers`:
- `fleet-size-whatif-days.csv`: one row per unit, configuration, role (the configuration, then its
  oracle partner) and counting day;
- `fleet-size-whatif.json`: the table by count and the figures per N, with `definitions`;
- `fleet-size-whatif-summary.txt`: the text summary;
- `fleet-size-whatif-forecasts.csv`: the model's forecast table (its hash is in the results).

Exit status: 0 every configuration ran; 1 the model was refused or failed, or a configuration
failed; 2 could not run (a bad matrix, argument, model, configuration, unit or minimum, a table
missing, no verified truth build at this commit, a table differing from it).
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from fleetreserve import dial_choice, dial_splits, gate_run, learn_run
from fleetreserve import fleet_size_whatif_core as core
from fleetreserve import fleet_size_whatif_summary as summary
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.gate_calibrate import ceiling_values
from fleetreserve.gate_config import Configuration, Matrix, MatrixError, load
from fleetreserve.gate_stat import RATIO_THRESHOLD, SHARE_THRESHOLD
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

MATRIX = dial_choice.MATRIX
ROLES = ("configuration", "oracle partner")
ROWS_NAME = "fleet-size-whatif-days.csv"
RESULTS_NAME = "fleet-size-whatif.json"
SUMMARY_NAME = "fleet-size-whatif-summary.txt"
TABLE_NAME = "fleet-size-whatif-forecasts.csv"
SPLITS = dial_choice.SPLITS
RUNG = dial_choice.RUNG
SCALE_LABELS = tuple(f"{g:g}" for g in dial_splits.GRID)
COLUMNS = (
    "unit",
    "configuration",
    "role",
    "evaluated",
    "site",
    "date",
    "cars",
    "steered_cars",
    "offers",
    "reason_for_not_offering",
    "offer_at_scale_1_kwh",
    *(f"slot_loss_at_{s}" for s in SCALE_LABELS),
    *(f"driver_loss_at_{s}" for s in SCALE_LABELS),
)


def definitions(matrix: Matrix, splits: int) -> dict:
    return {
        "what_if": (
            "the plan has no minimum number of steered cars for a day to offer; this is a "
            "measurement of what such a minimum would have done to the figures, not a rule, a "
            "choice or a recommendation, and it changes no rung, gate or calibration rule and "
            "no existing result"
        ),
        "steered_cars": (
            "len(PreparedDay.steered) of gate_day.prepare_day: the cars of the fleet in the "
            "steered sub-day, that is, those the fixed plan steers (a car with a forecast whose "
            "forecast set is not empty; none when the configuration's rule is day-abstains and a "
            "car is uncertifiable); the field keeps its cars on a day that abstains for another "
            "reason (zero value, no certificate), so every counting day has a count; cars "
            "outside the box are not counted"
        ),
        "minimum": (
            "for a minimum N, every day of the unit with fewer than N steered cars, calibration "
            "and test days alike, is treated exactly as a day that does not offer is treated "
            "today: it offers 0 at every scale, has no loss and is in no mean and no calibration "
            "count; its best-causal ceiling is kept; N = 0 is dial_choice's own figures"
        ),
        "scales": (
            "dial_splits.GRID (crc.GRID up to and including 1); each (day, scale) is evaluated "
            "once and reused for every N"
        ),
        "splits": (
            f"{splits} seeded splits per unit (seeds 1 to {splits}), each made by day_split "
            f"with the calibration-days rule (n_calibration {matrix.n_calibration}, "
            f"calibration_half_below {matrix.calibration_half_below}), as dial_choice makes them"
        ),
        "tiers": (
            "A: slot loss at alpha 0.10 and driver loss at alpha 0.05; B: the slot loss alone; "
            "never pooled; the figures of a tier are those of dial_choice (see its definitions)"
        ),
        "by_count": (
            "one row per exact number of steered cars, in increasing order, over all counting "
            "days of the unit: counting days, offering days, the sum of the certified value at "
            "scale 1 in kWh, and per scale the sums of the one-way rung's slot loss and driver "
            "loss and the numbers of days with a slot loss and with a driver loss above 0; the "
            "rows add up to the unit's totals"
        ),
        "minimums": (
            "per N: the counting days below N, the offering days that no longer offer, and the "
            "figures dial_choice reports per unit, configuration and tier over the seeded "
            "splits (mean offer per test day, share of splits that offer, mean scale, mean test "
            "slot and driver loss, p10, p50, p90), and its gate-reading figures"
        ),
        "gate_reading": (
            "as dial_choice's gate_reading (see its definitions): tier A, over all the splits, "
            "the mean calibrated offer per counting test day, per ceiling the mean ceiling and "
            "the share of it, the oracle partner's mean offer and its ratio to the mean offer; "
            "for a minimum N the configuration's days are restricted by its own count of steered "
            "cars and the oracle partner's days by the partner's own count (the two can differ; "
            "the number of counting days on which they do is in oracle_count_differs_on_days); "
            "the plan's thresholds are listed in run.thresholds and are not applied"
        ),
        "timeouts": (
            "a day whose certificate times out abstains; a re-solve that times out leaves the "
            "rung on its predecessor's plan and the day enters with the losses of that replay; "
            "a missing ceiling leaves the ceiling share of the splits that test on that day out"
        ),
    }


def _number(value) -> str:
    return repr(float(value))


def day_rows(name: str, configuration: str, role: str, evaluated: str, prepared, losses):
    """The per-day rows of one unit (`COLUMNS`). `configuration`: the configuration measured;
    `role`: one of `ROLES`; `evaluated`: the configuration whose days these are (the oracle
    partner's name on the partner's rows)."""
    rows = []
    for d, p in enumerate(prepared):
        row = {
            "unit": name,
            "configuration": configuration,
            "role": role,
            "evaluated": evaluated,
            "site": p.site,
            "date": p.date,
            "cars": p.n_cars,
            "steered_cars": len(p.steered),
            "offers": int(losses.offering[d]),
            "reason_for_not_offering": p.abstain_reason or "",
            "offer_at_scale_1_kwh": _number(losses.value[d]),
        }
        for k, label in enumerate(SCALE_LABELS):
            row[f"slot_loss_at_{label}"] = _number(losses.slot[d, k])
            row[f"driver_loss_at_{label}"] = _number(losses.driver[d, k])
        rows.append(row)
    return rows


def _measure_oracle(runner, name, matrix, configuration, n_days, best_causal, timeouts):
    """(prepared days, losses) of the oracle partner of `configuration`, prepared and measured like
    the configuration's own days (`dial_choice.gate_reading`)."""
    partner = matrix.configuration(configuration.oracle_partner)
    prepared = runner.map(gate_run._prepare_task, [(name, i, partner) for i in range(n_days)])
    losses, cache = dial_choice.measure(
        prepared, best_causal, lambda pairs: runner.map(gate_run._evaluate_task, pairs)
    )
    timeouts["oracle_certificates"] += sum(p.certificate_outcome == "timeout" for p in prepared)
    timeouts["oracle_re_solves"] += dial_choice._evaluations(cache)[
        "re_solves_of_the_one_way_rung"
    ].get("timeout", 0)
    return partner, prepared, losses


def measure_configuration(
    runner,
    name: str,
    n_days: int,
    configuration: Configuration,
    matrix: Matrix,
    best_causal,
    ceilings: dict,
    indices,
    min_cars,
    timeouts: Counter,
):
    """(entry of the results document, per-day rows) of one unit under one configuration (module
    text); the timeouts are added to `timeouts`. `ceilings`: ceiling name -> per-day array."""
    prepared = runner.map(gate_run._prepare_task, [(name, i, configuration) for i in range(n_days)])
    losses, cache = dial_choice.measure(
        prepared, best_causal, lambda pairs: runner.map(gate_run._evaluate_task, pairs)
    )
    found = dial_choice._evaluations(cache)
    timeouts["certificates"] += sum(p.certificate_outcome == "timeout" for p in prepared)
    timeouts["re_solves"] += found["re_solves_of_the_one_way_rung"].get("timeout", 0)
    counts = [len(p.steered) for p in prepared]
    rows = day_rows(name, configuration.name, ROLES[0], configuration.name, prepared, losses)
    entry = {
        "driver_risk": configuration.driver_risk,
        "offering_days": int(losses.offering.sum()),
        "evaluations": found,
        "by_count": core.by_count(losses, counts),
        "oracle_partner": configuration.oracle_partner,
    }
    oracle = oracle_counts = None
    if configuration.oracle_partner is not None:
        try:
            partner, oracle_days, oracle = _measure_oracle(
                runner, name, matrix, configuration, n_days, best_causal, timeouts
            )
        except Exception as problem:  # recorded, the other numbers stay
            entry["oracle_failed"] = f"{type(problem).__name__}: {problem}"
        else:
            oracle_counts = [len(p.steered) for p in oracle_days]
            rows += day_rows(name, configuration.name, ROLES[1], partner.name, oracle_days, oracle)
            entry["oracle_count_differs_on_days"] = core.differing(counts, oracle_counts)
    minimums = []
    for n in min_cars:
        below = core.below(losses, counts, n)
        restricted = core.restrict(losses, counts, n)
        item = {
            "min_cars": n,
            "counting_days_below": below["counting_days"],
            "offering_days_removed": below["offering_days"],
            "offering_days": int(restricted.offering.sum()),
            "tiers": dial_choice.score_splits(restricted, indices, dial_splits.TIERS),
        }
        if oracle_counts is not None:
            item["gate_reading"] = core.gate_reading(
                losses, counts, oracle, oracle_counts, n, ceilings, indices, matrix.ceilings
            )
            item["gate_reading"]["oracle_partner"] = configuration.oracle_partner
        elif "oracle_failed" in entry:
            item["gate_reading"] = {
                "failed": entry["oracle_failed"],
                "oracle_partner": configuration.oracle_partner,
            }
        else:
            item["gate_reading"] = {"not_computed": "the configuration has no oracle partner"}
        minimums.append(item)
    entry["minimums"] = minimums
    return entry, rows


def run(
    matrix: Matrix,
    units: dict,
    factory,
    configurations,
    names,
    min_cars,
    *,
    splits: int,
    workers: int,
):
    """Everything but the input checks and the writing of files: (results document, per-day rows).
    `units`: unit name -> `UnitDays` for the matrix's gate units; `factory`: as
    `gate_run.run_gate` takes it; `names`: the units measured."""
    seeds = list(range(1, splits + 1))
    runner = gate_run.Runner(workers, units, matrix, factory)
    entries, rows, timeouts, failed = {}, [], Counter(), {}
    try:
        for name in names:
            days = units[name]
            plan = gate_run.plan_unit(name, days, matrix)
            entry = {
                "counting_days": len(plan.keys),
                "calibration_days_used": plan.calibration_days,
                "test_days": plan.test_days,
                "status": plan.status,
                "configurations": {},
            }
            entries[name] = entry
            if plan.status != "calibratable":
                continue
            found = runner.map(gate_run._ceiling_task, [(name, i) for i in range(len(plan.keys))])
            timeouts["best_causal_ceilings"] += sum(
                c.best_causal_outcome == "timeout" for c in found
            )
            best_causal = [c.best_causal for c in found]
            ceilings = {
                ceiling: np.array(
                    [
                        np.nan if v is None else v
                        for v in (ceiling_values(c, ceiling) for c in found)
                    ],
                    dtype=float,
                )
                for ceiling in matrix.ceilings
            }
            indices = dial_choice.split_indices(name, days, matrix, seeds)
            for configuration in configurations:
                try:
                    done, day_found = measure_configuration(
                        runner,
                        name,
                        len(plan.keys),
                        configuration,
                        matrix,
                        best_causal,
                        ceilings,
                        indices,
                        min_cars,
                        timeouts,
                    )
                except Exception as problem:  # recorded, the other configurations go on
                    failed[(name, configuration.name)] = f"{type(problem).__name__}: {problem}"
                    entry["configurations"][configuration.name] = {
                        "driver_risk": configuration.driver_risk,
                        "failed": failed[(name, configuration.name)],
                    }
                    continue
                if "oracle_failed" in done:
                    failed[(name, configuration.oracle_partner)] = done["oracle_failed"]
                entry["configurations"][configuration.name] = done
                rows += day_found
    finally:
        runner.close()
    document = {
        "label": summary.LABEL,
        "definitions": definitions(matrix, splits),
        "run": {
            "splits": splits,
            "seeds": [seeds[0], seeds[-1]],
            "scales": list(dial_splits.GRID),
            "n_calibration": matrix.n_calibration,
            "calibration_half_below": matrix.calibration_half_below,
            "tiers": list(dial_splits.TIERS),
            "rung": RUNG,
            "min_cars": list(min_cars),
            "configurations": [c.name for c in configurations],
            "units": list(names),
            "ceilings": list(matrix.ceilings),
            "gate_ceiling": matrix.ceilings[0],
            "thresholds": {
                "share_of_ceiling": SHARE_THRESHOLD,
                "oracle_to_offer_ratio": RATIO_THRESHOLD,
            },
        },
        "units": entries,
        "timeouts": dict(sorted(timeouts.items())),
        "failed_configurations": sorted(f"{a} | {b}: {why}" for (a, b), why in failed.items()),
    }
    return document, rows


def rows_csv(rows: list[dict]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--min-cars", type=int, nargs="+", required=True)
    parser.add_argument("--matrix", type=Path, default=MATRIX)
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--configuration", action="append", default=None)
    parser.add_argument("--unit", action="append", default=None)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        if args.splits < 1:
            raise MatrixError("--splits must be at least 1")
        if any(n < 0 for n in args.min_cars) or len(set(args.min_cars)) != len(args.min_cars):
            raise MatrixError("--min-cars must be different whole numbers, 0 or more")
        matrix = load(args.matrix)
        if args.configuration:
            if len(set(args.configuration)) != len(args.configuration):
                raise MatrixError("a --configuration is given twice")
            configurations = tuple(matrix.configuration(n) for n in args.configuration)
        else:
            plain, dials = dial_choice.find_family(matrix)
            configurations = (plain, *dials)
        names = tuple(args.unit) if args.unit else matrix.gate_units
        outside = [name for name in names if name not in matrix.gate_units]
        if outside or len(set(names)) != len(names):
            raise MatrixError(
                f"--unit must name distinct gate units of the matrix {list(matrix.gate_units)}, "
                f"got {list(names)}"
            )
        model = learn_run.load_model(args.model, matrix)
    except MatrixError as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    fit_units = tuple(model.fit_units)
    try:
        learn_run.check_model(args.model, model, fit_units)
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    needed = tuple(dict.fromkeys((*matrix.gate_units, *fit_units)))
    try:
        inputs = gate_run.check_inputs(args.truth_run, args.tables, needed)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    try:
        loaded = {
            name: load_unit_days(
                args.tables,
                name,
                rho=matrix.rho,
                no_fit=matrix.no_fit,
                small_fleet=matrix.small_fleet,
            )
            for name in needed
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    units = {name: loaded[name] for name in matrix.gate_units}
    fit_frames = {name: loaded[name].cars for name in fit_units}
    partners = [matrix.configuration(c.oracle_partner) for c in configurations if c.oracle_partner]
    kappa = max(c.kappa for c in (*configurations, *partners))
    try:
        factory, table = dial_choice.make_factory(
            args.model, model, matrix, units, fit_frames, kappa
        )
    except learn_run.Refused as problem:
        print(f"REFUSED: {problem}")
        return 1
    except ValueError as problem:
        print(f"FAILED: the model could not be scored: {problem}")
        return 1
    document, rows = run(
        matrix,
        units,
        factory,
        configurations,
        names,
        tuple(args.min_cars),
        splits=args.splits,
        workers=args.workers,
    )
    document["model"] = model.name
    document["fit_units"] = list(fit_units)
    document["inputs"] = inputs
    document["matrix"] = {
        "file": args.matrix.name,
        "sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
    }
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        write_table(table, folder / TABLE_NAME)
        document["forecast_table"] = {
            "file": TABLE_NAME,
            "rows": len(table),
            "sha256": hashlib.sha256((folder / TABLE_NAME).read_bytes()).hexdigest(),
        }
    text = "\n".join(summary.summary_lines(document)) + "\n"
    print(text, end="")
    if run_dir:
        (folder / ROWS_NAME).write_text(rows_csv(rows))
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
    return 1 if document["failed_configurations"] else 0


if __name__ == "__main__":
    sys.exit(main())

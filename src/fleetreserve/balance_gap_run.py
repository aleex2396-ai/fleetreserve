"""The balance gap of every offering day's certificate in one recorded gate or learn run (plan
sections 5 (T2) and 6 (T6); `fixed_plan_loss`, module text, V8).

Command line, from the project's top folder:
    python -m fleetreserve.balance_gap_run --gate-run <run folder> [--tables <folder>]
        [--truth-run <folder>] [--workers N]

What it measures. Replay can miss the request by a certificate's balance gap, the amount by which
its cars do not share the box out exactly (V2, V4 and V8 of `fixed_plan_loss`'s module text); no
code enforces it. This command forms again the certificate of every offering day of one recorded
run, as the run formed it, and measures its gaps per service hour (`balance_gap`: the centre gap
`c_t`, the gain gap `gamma_t` and `B_t(beta) = c_t + beta gamma_t`, exactly). It decides nothing:
it reports the gaps, their counts against three levels and the hours in which the gap alone can
fail the hour.

The run (`--gate-run`): one recorded run of `python -m fleetreserve.gate_run` (`gate.json`,
`gate-days.csv`) or of `python -m fleetreserve.learn_run` (`learn-gate.json`,
`learn-gate-days.csv` and its forecast table `learn-forecasts.csv`), known by its recorded command
(`ledger_run.scoring_module`). In this order, a refusal at any step could not run (exit 2):

1. the results file is verified (the run ended with exit 0, was neither stopped nor timed out, the
   file is the one it wrote) and the run was recorded at the commit checked out now
   (`run_evidence.read_evidence`; else stale);
2. held-out units: a run whose results name a unit outside `unit_list.KNOWN_UNITS` (the training
   and development units), or that was made on the sealed-data route (its results hold the key
   `gate_summary.ROUTE_KEY`), is refused here, before the matrix, the run's day table, its forecast
   table or any truth table is opened. Held-out units are read only through the sealed-data path at
   the tag, and this command has no route to them (reading M9);
3. the matrix: the file the recorded command names with `--matrix` (reading M6), passed through
   `gate_run.check_files` and read once (`gate_config.read`); the run must not be a trial and must
   have run on that file (`ledger_run.check_document`: the SHA-256 it recorded), the matrix's gate
   units must be the run's units and each of the run's configurations must be in the matrix;
4. the day table is verified (`run_verify.require_verified`), and for a learn run the forecast
   table too, read by `forecast_model.read_table`;
5. the truth tables of the gate units (`--tables`, default the main build's folder, and
   `--truth-run`, default the latest recorded build at this commit) are checked and read once as
   `gate_run` reads them (`gate_run.read_inputs`), each with the SHA-256 the run recorded, of the
   setting it recorded (`ledger_run.check_tables`);
6. the days are built from those bytes with the matrix's readings
   (`days_from_truth.parse_unit_days`; a day that cannot be built: exit 1) and split as the run
   split them (`gate_run.plan_unit`); every configuration's day rows must be those days, in that
   order, with those roles (`ledger_run.match_days`), and each scale of reading M3 the run
   recorded must be a finite number of at least 0.

Forming again. The run's configurations are grouped by work key in the matrix's order, as
`gate_run` groups them. For each group and gate unit, every day that a configuration of the group
recorded as offering (no abstain reason, certificate outcome "optimal") is prepared again by
`gate_run._prepare_task` with the group's first configuration, the run's forecaster (a gate run:
`gate_run.empirical_forecaster`; a learn run: `forecast_model.TableFactory` of its recorded table,
reading M8) and the matrix's day settings, in `--workers` processes (`gate_run.Runner`). Every
certificate solve keeps the gate's time limit (`lp.BOX_SECONDS` per stage). Per configuration of
the group and day formed again (reading M4):

- the certificate outcome "timeout": counted and listed with its seconds; no gap is written for it;
- else, when its abstain reason, certificate outcome and certified value (as `repr` writes it) are
  not all those the run recorded: the day is refused, listed with both, and no gap is written;
- else, when the run recorded it as offering: measured, one row per service hour.

An error raised while forming again the days of a group and unit is recorded with that group and
unit (exit 1), and the others go on.

What is measured: the gaps of the certificate's policy (`balance_gap.hour_gaps`; the policy with
the decision hour's draws has the same gaps, `balance_gap`, module text), and `B_t` at scale 1, at
the validity limit and at each fixed-plan scale the run calibrated for that unit and configuration
(reading M3).

Readings, each with its alternative in brackets:

  M1. The gaps are V8's terms of `B_t`, computed exactly on the doubles the certificate holds, each
      written as the nearest double, every comparison exact (`balance_gap`, module text).
      [Alternative: sums in floating point, whose rounding can be above the levels counted.]
  M2. "V8's premise held up to scale `beta`" means `B_t(beta) <= FAILURE_KWH` in every measured
      hour: the gap alone fails no hour (V8: "a `B_t` above `FAILURE_KWH` can fail the hour");
      `B_t` does not fall as `beta` rises, so scale 1 and the validity limit decide "up to" each.
      The hours shared out exactly (`B_t` 0 at every scale) are counted beside it. It is needed,
      not enough: V4's condition `B_t <= FAILURE_KWH - E_t - R_t` needs `E_t` and `R_t`, which are
      not computed here, and under V6 the room left for the other terms at the validity limit can
      be about 1e-7 kWh or less; the summary says so wherever the premise held. The flags
      `held_up_to_scale_1` and `held_up_to_validity_limit` of a unit are true only when every
      offering day of the unit was measured (counted once per configuration) and no measured hour
      can fail by the gap alone; null when an offering day was not measured (timed out, refused or
      not formed again) and no measured hour can fail, and also when no hour was measured; false
      when a measured hour can fail, whatever the unmeasured days would show: that hour is a
      measured outcome. The counts of offering days, days measured, days timed out and days not
      measured sit beside the flags. [Alternatives: V4's condition, which needs `E_t` and `R_t` on
      the true rows at each scale; a flag that is null whenever any day is unmeasured, a failing
      hour included.]
  M3. The run's scales are the fixed plan's, each in a column of its own: tier A and tier B of the
      audited-set route (`rungs.fixed.beta_hat`, `rungs.fixed.tier_b.beta_hat`) and, of the
      every-request route, tier A and tier B (`beta_used` of `fixed_every_request` and of its
      `tier_b`); empty where the run has none (a unit not split, a tier not scored). Since commit
      1063724 tier A of the every-request route is under the every-scale bound, so it has no
      second column for it (the earlier version of this command had one, which since that commit
      would have held the same figure; nothing read it). "The present shape" is a positive test:
      the unit's `fixed_every_request` record holds `earlier_driver_bound`, which the present
      writer always writes beside `beta_used`, and does not hold `every_scale_driver_bound`, even
      as null. Any other record (a run made before 1063724, or not written by this code) could not
      run (exit 2, a line naming the key and the commit, nothing written), never read with an
      empty or a mislabelled column. The re-solve rungs' scales are not used: their plans change
      at a re-solve. [Alternatives: the re-solve rungs' scales too; an older run read from its own
      key, in a column labelled by the bound it was made under.]
  M4. A day formed again counts as the run's only when its abstain reason, certificate outcome and
      certified value are, bit for bit, those the run recorded; any other difference refuses it
      (exit 1). An equal value does not prove an equal policy: a programme can have one optimal
      value at two policies. A certificate that times out is listed and counted, not refused: the
      day is not measured, and the exit status is that of M10. [Alternatives: a tolerance on the
      value; a timeout counted as a refusal.]
  M5. Every offering day of the run is measured, of every role (calibration, test, a unit not
      split); each certificate is formed once per group of configurations that share work, and
      its gaps are written for each configuration of the group, so the hours of a unit are counted
      once per configuration. [Alternative: test days only.]
  M6. The matrix is the file the recorded command names with `--matrix` (a learn run without it:
      `learn/matrix.json`, `learn_run`'s default), relative to the top folder, where the run was
      started. [Alternative: an option `--matrix`, as `ledger_run` takes.]
  M7. The truth tables are those of `--tables` and `--truth-run`, checked against the run's
      record of them, as `ledger_run` reads them. [Alternative: the folders the recorded command
      names.]
  M8. A learn run's forecasts are its recorded table; the model is not fitted again.
      [Alternative: fit the model again, which needs its dependency group and repeats its fit.]
  M9. Held-out units are refused (no route), not read through a route of this command.
      [Alternative: the route of `sealed_route`, at the tag only.]
  M10. A run with a certificate that timed out exits 2, "could not run in full": the measured part
      is written and labelled (the label leads the summary and the csv; the results hold
      `measured_in_full` false and the label), and the flags of M2 are never true beside the
      unmeasured days. A refused day or an error in forming days again exits 1, also beside a
      timeout: a failure is reported before an incomplete run. [Alternative: exit 0 with the
      count of days not measured beside it, as before; a reader that checks the exit status alone
      would then take a run with unmeasured days as complete.]
  M11. A fixed-plan scale of 0 that the run recorded is read like any other: the count of hours
      that can fail by the gap alone is `B_t(0) = c_t > FAILURE_KWH`, the centre gap alone, though
      the gate does not replay at scale 0 (`gate_calibrate`, module text: nothing is offered). The
      results name the configurations that have a scale of 0 (`scales_at_0`), the definitions and
      the summary say what the count is there, and no number is changed or left out for it.
      [Alternative: no count at a scale of 0.]

Outputs, written into `FLEETRESERVE_RUN_DIR` when it is set (the summary also on standard output);
a partial run's label (`--only`) leads each, and, when a certificate timed out, the label of M10
after it:
- `balance-gaps.csv`: one row per unit, configuration, measured day and service hour: the centre
  gap, the gain gap, and `B_t` at scale 1, at the validity limit and at each scale of M3
  (`b_<name>_kwh`, empty where the run has none);
- `balance-gap.json`: per unit and configuration the days (offering in the run, measured, timed
  out, refused, not formed again), the run's abstaining days by reason, its scales (and the names
  of those at 0, `scales_at_0`), the hours, the
  hours shared out exactly, the largest centre and gain gaps, the hours whose gain gap is above each
  level and, per scale, the largest `B_t` and the hours that can fail by the gap alone; per unit
  the premise of M2 at scale 1 and at the limit, with the counts of offering days, days measured,
  days timed out and days not measured, and its sentence; whether the run was measured in full
  (`run.measured_in_full`: every offering day measured) and, when a certificate timed out, its
  label (`run.incomplete_label`); the timeouts, the refused days and the failures; the
  definitions, the levels (the nearest doubles, and exactly as fractions) and the readings;
- `balance-gap-summary.txt`: the same in words, one sentence per unit on the premise;
- `balance-gap-timing.json`: seconds, kept apart.

Exit status: 0 every offering day was measured; 1 a day was refused, forming again a group's days
raised an error, or a day could not be built from the tables (also when a certificate timed out); 2
could not run in full (M10: a certificate timed out, the measured part written and labelled), or
could not run (a bad argument, a run that is not verified, stale, a trial or not a scoring run, a
held-out unit or the sealed-data route, a matrix other than the run's, no verified truth build at
this commit, a table other than the run's, the run's days not those of the tables, a scale that is
not a finite number of at least 0). A run recorded with exit 2 is not verified for the readers of
`run_verify` and `run_evidence`: the outputs of a run with a timeout are read as files.
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fleetreserve import balance_gap, gate_run, ledger_run, sealed_route
from fleetreserve.balance_gap import AT_LIMIT, GAIN_LEVELS, SCALE_1
from fleetreserve.days_from_truth import parse_unit_days
from fleetreserve.forecast_model import TableFactory, read_table
from fleetreserve.gate_config import MatrixError, read
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.learn_run import TABLE_NAME
from fleetreserve.policy import ROW_TOLERANCE, VALIDITY_LIMIT, ZERO_GAIN
from fleetreserve.replay import FAILURE_KWH
from fleetreserve.run_evidence import CannotCheck, read_evidence
from fleetreserve.run_verify import NotVerified, require_verified
from fleetreserve.sessions import TOP
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun, _head_commit
from fleetreserve.unit_list import KNOWN_UNITS

LEARN = "fleetreserve.learn_run"
LEARN_MATRIX = Path("learn") / "matrix.json"  # `learn_run`'s default, from the top folder
CSV_NAME = "balance-gaps.csv"
RESULTS_NAME = "balance-gap.json"
SUMMARY_NAME = "balance-gap-summary.txt"
TIMING_NAME = "balance-gap-timing.json"
OPTIMAL = "optimal"
TIMEOUT = "timeout"
# The fixed plan's scales a run records per unit and configuration (reading M3), by column name.
SCALE_FIELDS = {
    "fixed_tier_a": ("rungs", "fixed", "beta_hat"),
    "fixed_tier_b": ("rungs", "fixed", "tier_b", "beta_hat"),
    "every_request_tier_a": ("fixed_every_request", "beta_used"),
    "every_request_tier_b": ("fixed_every_request", "tier_b", "beta_used"),
}
# The record of the every-request route (reading M3): the present writer (`gate_calibrate`) always
# writes the key MARKER beside `beta_used`, and no longer writes the key of the runs made before
# 1063724, OLDER, under which they held the every-scale figures beside a tier A of the earlier
# bound.
MARKER = "earlier_driver_bound"
OLDER = "every_scale_driver_bound"
COLUMNS = (
    "unit",
    "configuration",
    "site",
    "date",
    "role",
    "hour",
    "centre_gap_kwh",
    "gain_gap_kwh",
    *(f"b_{name}_kwh" for name in (SCALE_1, AT_LIMIT, *SCALE_FIELDS)),
)
NO_ROUTE = (
    f"held-out units are read only through the sealed-data path {sealed_route.AT_THE_TAG}, and "
    "this command has no route to them; nothing else was read"
)
DEFINITIONS = {
    "centre_gap_kwh": "c_t = |sum_i g_it - m_t| over the certificate's cars, service hour t",
    "gain_gap_kwh": (
        "gamma_t = sum_s |sum_i G_its - [s = t] w_t|, the request's own gain W_t = w_t e_t"
    ),
    "b_kwh": (
        "B_t(beta) = c_t + beta gamma_t, the term B_t of fixed_plan_loss's module text, at the "
        "scale its column names"
    ),
    "can_fail_alone": "B_t(beta) > FAILURE_KWH (V8: a B_t above FAILURE_KWH can fail the hour)",
    "arithmetic": (
        "exact rational arithmetic on the doubles the certificate holds; each number written is "
        "the nearest double; every comparison is exact"
    ),
    "not_measured": "E_t and R_t, so V4's condition B_t <= FAILURE_KWH - E_t - R_t is not checked",
    "premise_held": (
        "needed, not enough: B_t <= FAILURE_KWH in every measured hour leaves E_t and R_t "
        "unmeasured, and under V6 the room the other terms have at the validity limit can be "
        "about 1e-7 kWh or less"
    ),
    "scale_0": (
        "a fixed-plan scale of 0 recorded by the run is read like any other: the count of hours "
        "that can fail by the gap alone is B_t(0) = c_t > FAILURE_KWH, the centre gap alone, "
        "though the gate does not replay at scale 0; no number is changed or left out for it"
    ),
}
READINGS = {
    "M1": "the gaps are V8's terms of B_t, computed exactly",
    "M2": (
        "V8's premise held up to scale beta: B_t(beta) <= FAILURE_KWH in every measured hour, "
        "needed, not enough (V4's condition with E_t and R_t is not computed); the flags are "
        "true only when every offering day was measured, null when one was not and no measured "
        "hour fails, false when a measured hour fails"
    ),
    "M3": "the run's fixed-plan scales, each in a column of its own; the re-solve rungs' not used",
    "M4": (
        "a day formed again counts only when its reason, outcome and value are those the run "
        "recorded, bit for bit; a timeout is listed and counted, not refused (exit status: M10)"
    ),
    "M5": "every offering day of every role; gaps written for each configuration of a work group",
    "M6": "the matrix the recorded command names with --matrix",
    "M7": "the truth tables of --tables and --truth-run, checked against the run's record",
    "M8": "a learn run's forecasts are its recorded table",
    "M9": "held-out units are refused (no route)",
    "M10": (
        "a run with a certificate that timed out exits 2 (could not run in full), its measured "
        "part written and labelled; a refused day or an error exits 1, also beside a timeout; "
        "alternative: exit 0 with the count of days not measured"
    ),
    "M11": (
        "a recorded scale of 0 is read like any other: the count is the centre gap alone, though "
        "the gate does not replay at scale 0; named in scales_at_0, nothing changed"
    ),
}


@dataclass(frozen=True)
class Formed:
    """One day formed again (module text): what is compared with the run, and its gaps."""

    outcome: str
    reason: str
    value: float
    gaps: tuple | None  # `balance_gap.HourGap` per service hour; None when the day abstains
    seconds: float  # the certificate's solve seconds, timing only


def _measure_task(job) -> Formed:
    """`gate_run._prepare_task` of `job` and the gaps of its certificate when the day offers."""
    p = gate_run._prepare_task(job)
    gaps = None if p.abstain_reason is not None else balance_gap.hour_gaps(p.certificate.policy)
    reason = p.abstain_reason or ""
    return Formed(p.certificate_outcome, reason, float(p.value), gaps, p.certificate_seconds)


def offering(row: dict) -> bool:
    """A day row the run recorded as offering."""
    return row["abstain_reason"] == "" and row["certificate_outcome"] == OPTIMAL


def recorded(row: dict) -> dict:
    return {
        "abstain_reason": row["abstain_reason"],
        "certificate_outcome": row["certificate_outcome"],
        "certified_value_kwh": row["certified_value_kwh"],
    }


def formed_record(found: Formed) -> dict:
    return {
        "abstain_reason": found.reason,
        "certificate_outcome": found.outcome,
        "certified_value_kwh": repr(found.value),
    }


def _at(document, *keys):
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


# --- the run -------------------------------------------------------------------------------------


@dataclass
class Run:
    module: str
    manifest: dict
    document: dict
    matrix: object
    matrix_file: Path
    rows: list
    factory: object


def matrix_file(module: str, command: list, root: Path) -> Path:
    """The matrix the recorded command names (reading M6), from `root`."""
    named = None
    for k, part in enumerate(command):
        if part == "--matrix" and k + 1 < len(command):
            named = command[k + 1]
        elif part.startswith("--matrix="):
            named = part.split("=", 1)[1]
    if named is None:
        if module != LEARN:
            raise CannotCheck("the recorded command names no --matrix")
        named = os.fspath(LEARN_MATRIX)
    path = gate_run.argument_path(named)
    return path if path.is_absolute() else root / path


def check_known_units(document: dict, folder: Path) -> None:
    """Refuses a run that names a held-out unit or was made on the sealed-data route (module text,
    step 2); raises `CannotCheck`."""
    units = document.get("units")
    configurations = document.get("configurations")
    if not isinstance(units, dict) or not isinstance(configurations, dict):
        raise CannotCheck(f"the results of {folder.name} hold no units or no configurations")
    named = set(units)
    for entry in configurations.values():
        found = entry.get("units") if isinstance(entry, dict) else None
        if not isinstance(found, dict):
            raise CannotCheck(f"a configuration of {folder.name} holds no units")
        named |= set(found)
    unknown = sorted(u for u in named if u not in KNOWN_UNITS)
    if unknown:
        raise CannotCheck(
            f"{folder.name} scores {', '.join(unknown)}, not a training or development unit of "
            f"the truth build ({', '.join(KNOWN_UNITS)}); {NO_ROUTE}"
        )
    if ROUTE_KEY in document:
        raise CannotCheck(f"{folder.name} was made on the sealed-data route; {NO_ROUTE}")


def read_run(folder: Path, root: Path, head: str) -> Run:
    """Steps 1 to 4 of the module text; raises `CannotCheck`, `CouldNotRun` or `MatrixError`."""
    try:
        manifest = json.loads((folder / "manifest.json").read_text())
    except (OSError, ValueError) as problem:
        raise CannotCheck(f"the manifest of {folder.name} cannot be read: {problem}") from problem
    command = manifest.get("command") if isinstance(manifest, dict) else None
    module = ledger_run.scoring_module(command)
    if module is None:
        raise CannotCheck(
            f"{folder.name} is not a recorded run of {' or '.join(ledger_run.SCORING)} (its "
            "command)"
        )
    results_name, days_name = ledger_run.SCORING[module]
    evidence = read_evidence(folder, results_name, head)
    if not evidence.verified:
        raise CannotCheck(
            f"{folder.name} ended with exit code {evidence.manifest.get('exit_code')}, so it is "
            "not verified"
        )
    document = evidence.data
    check_known_units(document, folder)
    path = matrix_file(module, command, root)
    gate_run.check_files([path])
    matrix, _ = read(path)
    ledger_run.check_document(document, path, folder)
    if sorted(matrix.gate_units) != sorted(document["units"]):
        raise CannotCheck(f"the gate units of {path.name} are not the units of {folder.name}")
    names = {c.name for c in matrix.configurations}
    stray = sorted(set(document["configurations"]) - names)
    if stray:
        raise CannotCheck(f"{folder.name} ran configurations not in {path.name}: {stray}")
    tables = [days_name, TABLE_NAME] if module == LEARN else [days_name]
    try:
        for name in tables:
            require_verified(folder / name)
    except NotVerified as problem:
        raise CannotCheck(f"a table of {folder.name} is not verified: {problem}") from problem
    factory = gate_run.empirical_forecaster
    if module == LEARN:
        try:
            factory = TableFactory(read_table(folder / TABLE_NAME))
        except ValueError as problem:
            raise CannotCheck(f"{TABLE_NAME} of {folder.name}: {problem}") from problem
    rows = ledger_run.read_day_rows(folder / days_name)
    return Run(module, manifest, document, matrix, path, rows, factory)


def run_scales(entry, where: str) -> dict:
    """The scales of reading M3 the run recorded in one unit's entry; raises `CannotCheck`."""
    if isinstance(entry, dict) and "fixed_every_request" in entry:
        record = entry["fixed_every_request"]
        if not isinstance(record, dict) or MARKER not in record or OLDER in record:
            raise CannotCheck(
                f"{where}: the every-request record is not of the present shape (it must hold "
                f"`{MARKER}` and not `{OLDER}`, even as null): a run made before commit 1063724, "
                "whose tier A is under the earlier driver-loss bound, or not of this code's "
                "writer; this command reads runs of the present shape only (M3)"
            )
    out = {}
    for name, keys in SCALE_FIELDS.items():
        value = _at(entry, *keys)
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        if value is not None and not (ok and math.isfinite(value) and value >= 0):
            raise CannotCheck(f"{name} of {where} is not a finite number of at least 0: {value!r}")
        out[name] = None if value is None else float(value)
    return out


def work_groups(matrix, document) -> list[list]:
    """The run's configurations grouped by work key, in the matrix's order, as `gate_run`."""
    groups: dict = {}
    for c in matrix.configurations:
        if c.name in document["configurations"]:
            groups.setdefault(c.work_key(), []).append(c)
    return list(groups.values())


# --- forming again and measuring ------------------------------------------------------------------


def form_again(runner, groups, plans, grouped) -> tuple[dict, dict, dict]:
    """Per (group head, unit): the days formed again by index, the failures, the timing."""
    formed, failed, timing = {}, {}, {}
    for group in groups:
        head = group[0]
        for unit in plans:
            indices = sorted(
                {
                    d
                    for c in group
                    for d, row in enumerate(grouped.get((c.name, unit), []))
                    if offering(row)
                }
            )
            if not indices:
                continue
            begun = time.perf_counter()
            key = (head.name, unit)
            try:
                found = runner.map(_measure_task, [(unit, d, head) for d in indices])
            except Exception as problem:  # recorded, the other groups and units go on
                failed[key] = f"{type(problem).__name__}: {problem}"
                continue
            formed[key] = dict(zip(indices, found, strict=True))
            timing[f"{head.name} | {unit}"] = {
                "days": len(indices),
                "seconds": time.perf_counter() - begun,
                "certificate_seconds": sum(f.seconds for f in found),
            }
    return formed, failed, timing


def _text(value) -> str:
    return "" if value is None else repr(float(value))


def _gap_rows(unit, name, row, gaps, scales) -> list[dict]:
    out = []
    for g in gaps:
        found = {
            "unit": unit,
            "configuration": name,
            "site": row["site"],
            "date": row["date"],
            "role": row["role"],
            "hour": g.hour,
            "centre_gap_kwh": _text(g.centre),
            "gain_gap_kwh": _text(g.gain),
        }
        every = {**balance_gap.FIXED_SCALES, **scales}
        for scale, beta in every.items():
            found[f"b_{scale}_kwh"] = "" if beta is None else _text(g.miss(beta))
        out.append(found)
    return out


def _day(unit: str, row: dict) -> dict:
    return {"unit": unit, "site": row["site"], "date": row["date"], "role": row["role"]}


def list_timeouts(groups, plans, grouped, formed) -> list[dict]:
    """Each certificate that timed out, once, with the configurations that offered the day."""
    out = []
    for group in groups:
        for unit in plans:
            for d, found in formed.get((group[0].name, unit), {}).items():
                if found.outcome != TIMEOUT:
                    continue
                offered = [
                    (c.name, rows[d])
                    for c in group
                    if (rows := grouped.get((c.name, unit), [])) and offering(rows[d])
                ]
                names = [name for name, _ in offered]
                timed_out = {"configurations": names, "seconds": found.seconds}
                out.append({**_day(unit, offered[0][1]), **timed_out})
    return out


def measure_configuration(unit, name, day_rows, days, scales) -> tuple[list, list, Counter, list]:
    """One configuration's day rows of one unit against the days formed again (reading M4): the
    csv rows, the gaps of the measured hours, the counts and the refused days."""
    rows, gaps, counts, refused = [], [], Counter(), []
    for d, row in enumerate(day_rows):
        found = days.get(d)
        counts["offering"] += offering(row)
        if found is None or found.outcome == TIMEOUT:
            if offering(row):
                counts["not_formed" if found is None else "timed_out"] += 1
            continue
        if formed_record(found) != recorded(row):
            counts["refused"] += 1
            refused.append(
                {
                    **_day(unit, row),
                    "configuration": name,
                    "recorded": recorded(row),
                    "re_formed": formed_record(found),
                }
            )
        elif offering(row):
            counts["measured"] += 1
            gaps.extend(found.gaps)
            rows += _gap_rows(unit, name, row, found.gaps, scales)
    return rows, gaps, counts, refused


def measure(plans, groups, grouped, formed, failed, scales):
    """The csv rows, the per-unit entries and the refused days (module text)."""
    rows, units, refused = [], {}, []
    for unit, plan in plans.items():
        entries = {}
        for group in groups:
            days = formed.get((group[0].name, unit), {})
            for c in group:
                day_rows = grouped.get((c.name, unit), [])
                found = measure_configuration(unit, c.name, day_rows, days, scales[(c.name, unit)])
                rows += found[0]
                refused += found[3]
                counts = found[2]
                abstaining = Counter(
                    row["abstain_reason"] or row["certificate_outcome"]
                    for row in day_rows
                    if not offering(row)
                )
                entries[c.name] = {
                    "offering_days_in_run": counts["offering"],
                    "days_measured": counts["measured"],
                    "days_timed_out": counts["timed_out"],
                    "days_refused": counts["refused"],
                    "days_not_formed_again": counts["not_formed"],
                    "failed": failed.get((group[0].name, unit)),
                    "abstaining_days_in_run": dict(sorted(abstaining.items())),
                    "scales": scales[(c.name, unit)],
                    "scales_at_0": sorted(
                        n for n, beta in scales[(c.name, unit)].items() if beta == 0.0
                    ),
                    **balance_gap.tally(found[1], scales[(c.name, unit)]),
                }
        units[unit] = {
            "status": plan.status,
            "configurations": entries,
            "premise": premise(unit, entries),
        }
    return rows, units, refused


NEEDED_NOT_ENOUGH = (
    "a premise that held is needed, not enough: E_t and R_t are not measured, and under V6 the "
    "room the other terms have at the validity limit can be about 1e-7 kWh or less (V6, V8)"
)


def premise(unit: str, entries: dict) -> dict:
    """V8's premise over every measured hour of the unit (reading M2), with its sentence."""
    hours = sum(e["hours"] for e in entries.values())
    at_1 = sum(e["at_scales"][SCALE_1]["hours_can_fail_alone"] for e in entries.values())
    at_limit = sum(e["at_scales"][AT_LIMIT]["hours_can_fail_alone"] for e in entries.values())
    offering_days = sum(e["offering_days_in_run"] for e in entries.values())
    measured_days = sum(e["days_measured"] for e in entries.values())
    timed_out = sum(e["days_timed_out"] for e in entries.values())
    missing = offering_days - measured_days

    def flag(failing: int) -> bool | None:
        """True only when every offering day was measured and no measured hour can fail; false
        when a measured hour can fail; else null (reading M2)."""
        if hours == 0 or (failing == 0 and missing):
            return None
        return failing == 0

    word = "day" if missing == 1 else "days"
    unmeasured = f"{missing} offering {word} not measured (timed out, refused or not formed again)"
    if hours == 0:
        sentence = (
            f"{unit}: no hour was measured ({unmeasured}), so whether V8's premise held is not "
            "known."
        )
    else:

        def part(failing: int) -> str:
            if failing == 0:
                return f"held in every one of the {hours} hours measured"
            return f"did not hold in {failing} of the {hours} hours measured"

        sentence = (
            f"{unit}: up to scale 1, V8's premise (the balance gap alone cannot fail the hour: B_t "
            f"at most FAILURE_KWH) {part(at_1)}, counted once per configuration; up to the "
            f"validity limit it {part(at_limit)}"
        )
        if at_1 == 0 or at_limit == 0:
            sentence += f"; {NEEDED_NOT_ENOUGH}"
        if missing:
            sentence += (
                f"; {unmeasured}, so a premise that held is not known to hold for them and its "
                "flag is null"
            )
        sentence += "."
    return {
        "hours_measured": hours,
        "hours_can_fail_alone_at_scale_1": at_1,
        "hours_can_fail_alone_at_validity_limit": at_limit,
        "held_up_to_scale_1": flag(at_1),
        "held_up_to_validity_limit": flag(at_limit),
        "offering_days": offering_days,
        "days_measured": measured_days,
        "days_timed_out": timed_out,
        "offering_days_not_measured": missing,
        "sentence": sentence,
    }


# --- the outputs ---------------------------------------------------------------------------------


SCALE_WORDS = {SCALE_1: "scale 1", AT_LIMIT: "the validity limit"}


def _number(value) -> str:
    return "none" if value is None else f"{value:.3g}"


def summary_lines(results: dict) -> list[str]:
    lines = []
    if results["run"]["partial"]:
        lines.append(str(results["run"]["label"]))
    if results["run"]["incomplete_label"]:
        lines.append(results["run"]["incomplete_label"])
    refused, failed, timeouts = results["refused_days"], results["failed"], results["timeouts"]
    if refused or failed:
        word = "day" if len(refused) == 1 else "days"
        lines.append(
            f"FAILED: {len(refused)} {word} refused, {len(failed)} forming(s) again raised an "
            "error (listed below)."
        )
    measured = sum(
        e["days_measured"] for u in results["units"].values() for e in u["configurations"].values()
    )
    lines.append(
        f"Balance gaps (fixed_plan_loss, V8) of the certificates of {results['run']['folder']} "
        f"({results['run']['module']}), formed again at this commit: {measured} configuration-days "
        f"measured, {len(timeouts)} certificates timed out, {len(refused)} days refused."
    )
    lines.append(
        "c_t = |sum_i g_it - m_t|, gamma_t = sum_s |sum_i G_its - [s = t] w_t|, "
        "B_t(beta) = c_t + beta gamma_t, all exact; an hour can fail by the gap alone at beta "
        f"when B_t(beta) > FAILURE_KWH ({FAILURE_KWH:g} kWh). Gain-gap levels: ZERO_GAIN "
        f"{ZERO_GAIN:g}, FAILURE_KWH / VALIDITY_LIMIT {FAILURE_KWH / VALIDITY_LIMIT:g} "
        f"(exactly, 1e-9 plus 4.8e-26), ROW_TOLERANCE {ROW_TOLERANCE:g} kWh."
    )
    for unit, found in results["units"].items():
        lines.append(f"{unit} ({found['status']}):")
        for name, e in found["configurations"].items():
            above = e["gain_gap_above"]
            text = (
                f"  {name}: {e['days_measured']} of {e['offering_days_in_run']} offering days "
                f"measured ({e['days_timed_out']} timed out, {e['days_refused']} refused, "
                f"{e['days_not_formed_again']} not formed again); {e['hours']} hours, "
                f"{e['hours_shared_exactly']} shared out exactly; largest centre gap "
                f"{_number(e['largest_centre_gap_kwh'])} kWh, largest gain gap "
                f"{_number(e['largest_gain_gap_kwh'])} kWh; gain gaps above ZERO_GAIN "
                f"{above['zero_gain']}, above FAILURE_KWH / VALIDITY_LIMIT "
                f"{above['failure_over_limit']}, above ROW_TOLERANCE {above['row_tolerance']}; "
                "hours that can fail by the gap alone: "
            )
            text += ", ".join(
                f"{at['hours_can_fail_alone']} at {SCALE_WORDS.get(scale, scale)} "
                f"({_number(at['scale'])})"
                for scale, at in e["at_scales"].items()
            )
            if e["scales_at_0"]:
                text += (
                    f"; at a run scale of 0 ({', '.join(e['scales_at_0'])}) the gate does not "
                    "replay at scale 0, and the count there is c_t > FAILURE_KWH, the centre gap "
                    "alone"
                )
            if e["failed"]:
                text += f"; FORMING AGAIN FAILED: {e['failed']}"
            lines.append(text + ".")
        lines.append(f"  {found['premise']['sentence']}")
    lines.append(
        "Not measured here: E_t and R_t, so V4's condition B_t <= FAILURE_KWH - E_t - R_t is not "
        "checked (fixed_plan_loss, module text, V4 and V8)."
    )
    if timeouts:
        lines.append(
            f"Timeouts ({len(timeouts)}): "
            + "; ".join(
                f"{t['unit']} {t['site']} {t['date']} under {', '.join(t['configurations'])} "
                f"({t['seconds']:.1f} s)"
                for t in timeouts
            )
            + "."
        )
    if refused:
        lines.append(
            f"Refused days ({len(refused)}): "
            + "; ".join(
                f"{r['unit']} {r['site']} {r['date']} under {r['configuration']}: recorded "
                f"{_record_text(r['recorded'])}, formed again {_record_text(r['re_formed'])}"
                for r in refused
            )
            + "."
        )
    if failed:
        lines.append(f"Failed ({len(failed)}): " + "; ".join(failed) + ".")
    return lines


def _record_text(found: dict) -> str:
    reason = found["abstain_reason"] or "offering"
    return f"{reason}, {found['certificate_outcome']}, {found['certified_value_kwh']} kWh"


def _csv(rows, top: str | None) -> str:
    handle = io.StringIO()
    if top is not None:
        handle.write(f"# {top}\n")
    writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def heading(results: dict) -> str | None:
    """The label that leads the csv and the timing file: the partial run's, then that of M10."""
    parts = []
    if results["run"]["partial"]:
        parts.append(str(results["run"]["label"]))
    if results["run"]["incomplete_label"]:
        parts.append(results["run"]["incomplete_label"])
    return " | ".join(parts) if parts else None


def incomplete_label(timeouts: list, units: dict) -> str | None:
    """The label of a run with a certificate that timed out (reading M10), else None."""
    if not timeouts:
        return None
    days = sum(
        e["days_timed_out"] for found in units.values() for e in found["configurations"].values()
    )
    word = "certificate" if len(timeouts) == 1 else "certificates"
    day_word, verb = ("day", "has") if days == 1 else ("days", "have")
    return (
        f"NOT MEASURED IN FULL: {len(timeouts)} {word} timed out, so {days} offering {day_word} "
        f"(counted once per configuration) {verb} no gap; the rows and counts are the measured "
        "part only, and V8's premise is not known for the days left out (exit 2 unless a day was "
        "refused or forming again raised an error: exit 1)"
    )


def write_outputs(folder: Path, results, rows, text, timing) -> None:
    top = heading(results)
    (folder / CSV_NAME).write_text(_csv(rows, top))
    (folder / RESULTS_NAME).write_text(_json(results))
    (folder / SUMMARY_NAME).write_text(text)
    (folder / TIMING_NAME).write_text(_json(dict(timing, label=top)))


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate-run", type=Path, required=True)
    parser.add_argument("--tables", type=Path, default=OUT)
    parser.add_argument("--truth-run", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        gate_run.argument_paths(args, ("gate_run", "tables", "truth_run"))
        if args.workers < 1:
            raise MatrixError("--workers must be at least 1")
        run = read_run(args.gate_run, root, _head_commit())
        units = run.matrix.gate_units
        inputs, table_bytes = gate_run.read_inputs(args.truth_run, args.tables, units)
        ledger_run.check_tables(run.document, inputs, units, args.gate_run)
    except (MatrixError, CannotCheck, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    m = run.matrix
    try:
        days = {
            name: parse_unit_days(
                table_bytes, name, rho=m.rho, no_fit=m.no_fit, small_fleet=m.small_fleet
            )
            for name in units
        }
    except ValueError as problem:
        print(f"FAILED: a day could not be built from the tables: {problem}")
        return 1
    plans = {name: gate_run.plan_unit(name, days[name], m) for name in units}
    grouped = ledger_run.group_rows(run.rows)
    try:
        ledger_run.match_days(run.document, grouped, plans, args.gate_run)
        scales = {
            (name, unit): run_scales(entry["units"].get(unit), f"{unit} under {name}")
            for name, entry in run.document["configurations"].items()
            for unit in plans
        }
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except (KeyError, TypeError) as problem:  # a field read is not in the run's files
        print(f"COULD NOT RUN: the run's files lack what this command reads: {problem!r}")
        return 2
    groups = work_groups(m, run.document)
    runner = gate_run.Runner(args.workers, days, m, run.factory)
    try:
        formed, failed, job_timing = form_again(runner, groups, plans, grouped)
    finally:
        runner.close()
    rows, unit_results, refused = measure(plans, groups, grouped, formed, failed, scales)
    timeouts = list_timeouts(groups, plans, grouped, formed)
    failures = sorted(f"{head} | {unit}: {why}" for (head, unit), why in failed.items())
    code = 1 if refused or failures else (2 if timeouts else 0)
    not_measured = sum(
        e["offering_days_in_run"] - e["days_measured"]
        for found in unit_results.values()
        for e in found["configurations"].values()
    )
    results = {
        "run": {
            "folder": args.gate_run.name,
            "module": run.module,
            "command": run.manifest.get("command"),
            "label": run.document.get("label"),
            "partial": bool(run.document.get("partial")),
            "measured_in_full": not_measured == 0,
            "incomplete_label": incomplete_label(timeouts, unit_results),
            "setting": _at(run.document, "inputs", "setting"),
            "truth_run": _at(run.document, "inputs", "truth_run"),
            "matrix_file": run.matrix_file.name,
            "matrix_sha256": _at(run.document, "matrix", "sha256"),
        },
        "definitions": DEFINITIONS,
        "readings": READINGS,
        "levels_kwh": {
            **{name: float(level) for name, level in GAIN_LEVELS.items()},
            "failure": FAILURE_KWH,
            "validity_limit_scale": VALIDITY_LIMIT,
        },
        "levels_exact": {name: str(level) for name, level in GAIN_LEVELS.items()},
        "scale_columns": list(SCALE_FIELDS),
        "units": unit_results,
        "timeouts": timeouts,
        "refused_days": refused,
        "failed": failures,
        "exit_status": code,
    }
    text = "\n".join(summary_lines(results)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        timing = {
            "workers": args.workers,
            "jobs": job_timing,
            "total_seconds": time.perf_counter() - started,
        }
        write_outputs(Path(run_dir), results, rows, text, timing)
    return code


if __name__ == "__main__":
    sys.exit(main())

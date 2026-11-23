"""Build the truth tables of the held-out units from the sealed files (WP2; first step of WP14).

Command line, from the project's top folder, at the tag `frozen-plan`:
    python -m fleetreserve.heldout_truth_build [--setting main|held] [--out <folder>]
        [--matrix <file>]

It reaches the three sealed sets only through the gated routes of `heldout_loader` (`ROUTES`),
each called once, in the order `SETS` (Norway, a.s.r., Korea): `norway_sites(root)`, `asr(root)`
and `korea(root)`, each of which opens its file through the loader's gated opener and logs the
opening. This module itself opens, lists, reads, stats and hashes no file of the sealed folder,
and names no path in it (a test reads its source for that). It then builds every held-out
unit's truth tables with the functions the training and development units use: `truth.build`
(with its own rate rule), the window rule of `windows`, the counts of `truth_readings`, the
checks of `data_checks`, and `truth_build`'s own
summary, table writer and text helpers (`truth_build.Build`, `summarize`, `write`, `csv_bytes`
through `write`, `refuse_other_setting`, `summary_text`, `setting_line`, `_window_entry`). The
one rule written here is the search for a held-out unit's window (below), which plan section 2
asks for and the training build has no use for. `root` is the project's top folder
(`sessions.TOP`, the folder that holds `src/fleetreserve/`).

Units (plan sections 1 and 3), all with the role "held-out":
- a.s.r.: `asr-before-march-2023` and `asr-from-march-2023` (`load_asr.UNITS`), each day of the
  car park in the unit `load_asr.unit_of` gives its date. A workplace: its window is
  `windows.WORKPLACE`, the pair the recorded WP2 run fixed, handed to both units as
  `truth_build.build_nrel` hands it to the two NREL units (plan 2: "used unchanged at NREL and
  a.s.r."). As for NREL, the rate history is taken over every a.s.r. session, both units
  together (reading T2 below: `truth.build` is given every a.s.r. session and the unit's days);
  the sessions and dropped rows summarised per unit are those whose
  plug-in date (`load_asr.periods`) or local date (`load_asr.unit_of`) lies in the unit.
- Norway: `OSL_T` and `TRO_R` (`load_heldout_norway.SITES`), one unit per site.
- Korea: the five portfolios of `load_korea.UNITS`, one unit per portfolio; `load_korea` writes
  each portfolio's name as the sessions' site.

Windows of the Norway and Korea units (plan section 2: "τ one hour before the median plugged-in
count peaks, S the next hours (<= 8) with median >= half the peak, over calibration days
(held-out units)"). The rule on a profile is `windows.window_from_profile`; the profile is
`windows.occupancy_profile` of the unit's sessions over all its calendar days
(`windows.site_days`), of which only the rows of the unit's calibration days are given to the
rule. A held-out unit's calibration days are those the calibration protocol draws (plan section
6; `gate_run.plan_unit`): of its counting days at τ (`truth.build`'s `counts`, at least
`truth.COUNTING_CARS` cars plugged in at τ), keyed "<date> <site>", the
`day_split.calibration_count` of them (the matrix's `n_calibration`, or half of the counting days
rounded down under its `calibration_half_below`) drawn by `day_split.split_days` with the unit's
name and the matrix's `split_seed`; none when `day_split.is_split` says the unit cannot be split.
The matrix is `--matrix` (default `gate/matrix.json` under `root`); the summary records the path
as given, the default as `gate/matrix.json`.

Reading W1 (the plan leaves the order open; plan section 2 rules on a cycle). The calibration
days depend on τ (a day counts by its cars at τ) and τ is set on the calibration days, so the
window is sought as a fixed point of the map τ -> the rule's τ on the calibration days at τ, by
`search_window`: it starts from the
calibration draw above made over all the unit's calendar days (no counting rule, the same seed,
rank and count rule), applies the rule, takes the calibration days at the τ given, applies the
rule again, and stops at the first τ it meets twice. Each step either stops or meets a τ not met
before, and τ has 24 values, so the search stops by the 25th application of the rule.
  W1(a) The search settles when the τ met twice is the τ whose calibration days the rule was just
  given (a fixed point). The window is then the rule's result on exactly those calibration
  days, which are the days the protocol later calibrates on at that window.
  W1(b) The search cycles when the τ met twice is another τ met earlier (for example τ 14:00,
  then 10:00, then 14:00). Plan section 2 (the owner's decision of 15 November 2026): the first
  τ met twice stands, with the calibration days drawn at that τ, and the cycle is recorded as a
  finding. The window's rule text is then `CYCLE_RULE`; its entry adds `search_cycle` (true)
  and `cycle_rule_tau_hour`, the rule's own τ on those calibration days, which differs from τ:
  that difference is the cycle, and τ is not replaced by it. A settled entry has neither key.
  W1(c) S of a cycled window (plan section 2, the owner's decisions of 15 November 2026): the S
  of the window the rule last gave at τ. That is the window of the search's last step, the step
  that met τ again, which the rule gave on the calibration days at the τ before it (the
  second-to-last of `search_taus`). Its τ is τ, so S starts at τ + 1, and it always holds the
  peak hour, whose median is the peak and so at least half of it: S is never empty, and a
  search that cycles always gives a window. S comes from days other than those the unit is
  calibrated on (the calibration days at τ); the entry's `median_by_hour`, `peak_hour`,
  `peak_median`, `days` and earliest-peak τ are those of the rule on the calibration days at
  τ, whose τ is `cycle_rule_tau_hour`.
Three cases about a unit that gets no window (the whole rule, stated once):
  (1) The unit abstains, exit 0, when its window search has no calibration day at a τ it reaches
      (fewer than 2 counting days; or at the start draw), or
      when every row of the unit read in the held-out files is dropped by the cleaning rules
      (rows were read and none is kept: a result about the data; its reason is
      `all_dropped_reason`, "every row dropped by cleaning: <n> rows, by reason <reason>: <n>,
      ..."). The plan is silent on such a unit; the rule applied is the code's own,
      `sealed_route.COUNTS_AS` and `sealed_route.abstained_gate_data`: an abstained unit is
      recorded and counts as not passing. The summary records it under `abstained` with its
      reason and its sessions and rows removed by reason, and, when its window search ran, the
      search's `search_taus` and `days_read`; the report lists it among the findings.
  (2) The command exits 2, a reading fault and not a result, when the held-out files hold no row
      at all of a named Norwegian site or Korean portfolio or of an a.s.r. period (`NO_SESSION`);
      see "Exit status" below.
  (3) Every other unit has a window and its tables.
What the search reads, per unit: the counting status at each τ it reaches of every day, and the
whole occupancy profile of the days of each set it gives the rule. The summary records, for every
unit whose search ran, the τ of each step (`search_taus`; when a window is found, the last one is
the τ met twice) and the number of days read (`days_read`); for a unit with a window also, as
`test_days_read`, how many of them are test days at the final window (counting days at its τ that
are not calibration days). An abstaining unit has no window, so none of its days is a test day,
and its entry has no `test_days_read`. A test day's profile can be read only on a path through
another τ, or by the start draw when the half rule gives the start more days than the final
count; when
`test_days_read` is above 0 the report lists it among the findings. When every calendar day
counts at the final τ, the start draw is the final calibration draw and no test day is read.
[Alternatives to the search: (a) the window set once on the start draw over calendar days, not
searched further; (b) every τ of the day tried and the fixed points compared; (c) the calibration
days drawn once over calendar days and kept, so that a held-out unit would calibrate on fewer
than the plan's 100 counting days.]
A unit with no row at all in the held-out files is not such an abstention: held-out files that
hold no row, kept or dropped, of a named Norwegian site or Korean portfolio or of an a.s.r.
period, are a fault in reading the file, not a result about the unit (the a.s.r. file spans both
periods, so a period with no row means the dates were read wrongly). The command then prints
"COULD NOT RUN: <unit>: no session in the held-out files: a reading fault, not a result; nothing
written" for each such unit and exits 2 before any table is written (the unit's entry in
`abstained` is still made by the build's functions, `NO_SESSION`; only the command refuses it). A
unit whose rows were read and all dropped by cleaning is not this fault (case (1) above).
The loaders refuse such files first: a named Norwegian site, a.s.r. period or Korean portfolio
with no row, or whose rows other than Korean fast-charger rows are all removed by the loader's
removal rules, is a `LayoutError` of the loader (`load_heldout_norway`, `load_asr`, `load_korea`),
so the route logs the refusal and the command exits 1 (the parsing-fix route of plan section 13).
No table that went through a loader reaches the exit 2 above, nor case (1) for a Norwegian site or
an a.s.r. period; both stay as guards for tables that did not come from a loader. A Korean
portfolio of fast-charger rows only is not refused: it abstains by case (1), exit 0.
The a.s.r. periods (`build_asr`), by what each period holds. Rows all dropped by cleaning: the
period abstains (case (1), exit 0). No row at all, kept or dropped, whatever the other period
holds: the reading fault above, exit 2. Kept sessions: the period is built; a period with kept
sessions always has a counting day (the days run from the first to the last plug-in date), so
the guard that would give it the reason "no day of the unit" is not reached.

Midnight. The window rule reads the 24 clock hours as a circle, so a peak late in the evening
gives an evening τ and service hours that run past midnight (`windows`). A day is the local date
of its τ: its cars are those plugged in at τ on that date, and its hours are counted from τ in
real time (`truth`), so service hour t of a day with τ 22:00 is the hour starting at 22:00 + t
hours, on the next date after midnight. `truth_build._window_entry` writes S modulo 24 hours
(for example "23:00 to 07:00"). A searched window's rule text is `SEARCH_RULE` (`CYCLE_RULE`
after a cycle, W1(b)), not the label "Norway: ..." of `window_from_profile`, which would misname
a Korean unit.

Settings (plan 3): `--setting main` (the default) or `held` (no charging before τ), exactly as in
`truth_build`; windows and counting days are those of the main setting in both (the search uses
`truth.MAIN`). Default output folders, under the ignored `data/`, never `data/derived/truth` or
`truth-held`: `data/derived/truth-heldout` (main) and `data/derived/truth-heldout-held` (held,
with `truth_build.SETTING_FILE`).

Outputs, in the output folder: `<unit>-cars.csv` and `<unit>-days.csv` per unit with a window
(columns `truth.CAR_COLUMNS`, `truth.DAY_COLUMNS`; `truth_build.write`). When
`FLEETRESERVE_RUN_DIR` is set (the run recorder sets it), also `truth-summary.json` and
`truth-report.txt` there, shaped as `truth_build`'s: per unit sessions, rows removed by reason,
days, counting days, shares, fallback shares, reading counts and table hashes; the windows; the
abstained units; the findings; the checks; `calibration_rule` (seed, counts, matrix file); and
`korea_fast_charger_rows_removed`, the number of Korean rows removed as fast-charger sessions
(plan section 3: "the number left out is logged at the opening"; per portfolio also under each
unit's `rows_removed`), and the key `sealed_route` (`gate_summary.ROUTE_KEY`; reading T4).
The same inputs give the same bytes.

Order of work, so that an opening is never spent on a run that cannot finish and a failed run
writes nothing:
1. Before any opening (`preflight`): the arguments, the matrix, and the output folder and the run
   folder are checked. Refused, with exit 2 and nothing opened or written: an output or run
   folder that lies inside the sealed folder (`heldout_seal.FOLDER` under `root`, compared by
   the written paths alone, first, so that nothing there is looked at; a path that reaches it
   through a symbolic link is not seen by this check, which is left to the honest caller of the
   opener's scope); an output folder that is `data/derived/truth` or `data/derived/truth-held`
   under `root`; one that holds the other setting's tables
   (`truth_build.refuse_other_setting`); one that is not a folder or cannot be made or written;
   and, for the output and the run folder, a place inside `root` that git does not ignore
   (`git check-ignore`, run as the opener runs git: `heldout_loader.git_environment` and
   `heldout_loader.GIT_SETTINGS`), since a file written there would make the tree unclean and
   every later opening refuse. A run folder that is not an existing writable folder is refused
   too. A place outside `root` is allowed.
2. The three sets are opened and parsed, in the order `SETS`, before anything is written.
3. The units are built in memory; 4. the tables are written; 5. the summary and the report.

Exit status, and what a failure of one set does to the others:
- 0: the tables of every unit with a window are written (a window search that cycles included,
  W1(b)); a unit that abstains (case (1) above) leaves the exit at 0.
- 1: a sealed file does not have its published layout (`LayoutError` of its parser). The
  message names the set; the sets after it in `SETS` are not opened; nothing is written for any
  set; the openings already made stay logged (the failing set's own opening included, since the
  opener logs before its bytes are parsed) and the message gives their number. The route that
  called the parser has logged the refusal and the parser's message in the opener's refusal log
  (`heldout_loader`, reading H6), the record a later parsing fix of that set needs.
- 2: could not run: an argument, the matrix or a folder is refused (step 1, nothing opened), or
  the opener refused a set (`SealRefused`: not at the tag, an unclean tree, changed code, the
  wrong working copy, a wrong hash, and the rest of `heldout_loader`'s checks; the message names
  the set, nothing is written, and the openings of the sets before it stay logged), or the tables
  cannot be written, or a Norwegian site, Korean portfolio or a.s.r. period has no row at all in
  the held-out files, kept or dropped (a reading fault, nothing is written; the openings of all
  three sets are logged); a unit whose rows are all dropped by cleaning is not this: it
  abstains and the exit is 0.
Any other error is raised unchanged, so Python exits 1 with its trace; nothing is reported as
success.

Readings, each with its alternative in brackets:
  T1. W1 above (the held-out window's calibration days).
  T2. a.s.r.: the rate history ("user's highest observed average", plan 3) is taken over every
      a.s.r. session, before and from March 2023 together, as `truth_build.build_nrel` takes
      NREL's over both its units. [Alternative: over the unit's own sessions only.]
  T3. A held-out unit's data checks are those of `data_checks` that need only the session table
      and the dropped rows (`dst_shift`, `units`, `midnight`, `drop_shares`,
      `weekday_plug_in_hours`), with the rows read counted as the rows kept plus the rows
      dropped, per site (a.s.r.: the whole car park; Korea: per portfolio). The checks that read
      the raw file (`gap_and_doubled`, `stated_duration`) are not made: the gated routes return
      no raw table. No count is compared with a stated count. [Alternative: none without a
      change to the routes.]
  T4. The summary holds the key `sealed_route` with the record `ROUTE_RECORD`, as every results
      file of a run through the sealed-data gate does, so that `run_evidence` never reads this
      build from an earlier commit (its point 4, reading E3). The key is written whenever the
      summary is, that is, after the three gated routes passed. [Alternative: no key here, and
      `run_evidence` taking a summary with `calibration_rule` and held-out units for one; that
      reads a field that is a side effect of the build, not a record of the gate.]
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from fleetreserve import (
    data_checks,
    gate_config,
    heldout_loader,
    heldout_seal,
    load_asr,
    load_heldout_norway,
    load_korea,
    load_norway,
    truth,
    truth_build,
    truth_readings,
)
from fleetreserve.day_split import calibration_count, is_split, split_days
from fleetreserve.gate_summary import ROUTE_KEY
from fleetreserve.sessions import PHYSICAL_LIMIT_KW, TOP, LayoutError
from fleetreserve.windows import (
    WORKPLACE,
    Window,
    occupancy_profile,
    site_days,
    window_from_profile,
)

ROLE = "held-out"
# What the summary records of the gate (reading T4): the key every result of a run through the
# sealed-data gate holds, whose presence `run_evidence` reads (point 4, reading E3).
ROUTE_RECORD = {
    "kind": "held-out",
    "label": "built from the sealed sets through the gated routes of the held-out loader",
}
# The three gated routes, in the order they are called; the only way this module sees sealed data.
ROUTES = (
    ("norway", heldout_loader.norway_sites),
    ("asr", heldout_loader.asr),
    ("korea", heldout_loader.korea),
)
SETS = tuple(name for name, _ in ROUTES)
NORWAY_UNITS = load_heldout_norway.SITES
ASR_UNITS = load_asr.UNITS
KOREA_UNITS = load_korea.UNITS
UNITS = (*ASR_UNITS, *NORWAY_UNITS, *KOREA_UNITS)
OUT_NAMES = {truth.MAIN: "truth-heldout", truth.HELD: "truth-heldout-held"}
MATRIX = Path("gate") / "matrix.json"
FAST_REASON = load_korea.REASONS[0]
# The rule text of a searched window, in place of the label "Norway: ..." of
# `window_from_profile`, which would misname a Korean unit.
SEARCH_RULE = (
    "plan 2, Norway and Korea: τ one hour before the median peak, S while median >= half the "
    "peak (at most 8 hours), on the unit's calibration days (reading W1)"
)
# The rule text of a window whose search cycles (plan 2: the first τ met twice stands, with S the
# window the rule last gave at that τ).
CYCLE_RULE = (
    "plan 2, Norway and Korea, a search that cycles: τ the first τ the search met twice, with the "
    "unit's calibration days at τ; S that of the window the rule (S while median >= half the "
    "peak, at most 8 hours) last gave at τ, on the calibration days at the τ before it (reading "
    "W1)"
)
# The report's heading of `findings`: every kind of line it lists. A removal share above the
# threshold of `data_checks` is listed whatever its reason, the Korean fast-charger rows, which
# plan section 3 itself removes, included; that line is a size, not a contradiction.
FINDINGS_HEADING = (
    "FINDINGS (checks that contradict an assumption of the plan; any reason that removes more "
    f"than {data_checks.DROP_FINDING_SHARE:.0%} of a site's rows, the plan's own removal of "
    "fast-charger sessions included; units abstaining; window searches that cycle or read test "
    "days)"
)
SITE_SETS = {
    "norway": (load_norway.ZONE, PHYSICAL_LIMIT_KW["norway"]),
    "korea": (load_korea.ZONE, load_korea.LIMIT_KW),
}


class CouldNotRun(Exception):
    """A check before the openings, or the opener, refused the run (exit 2)."""


class SetLayoutError(Exception):
    """A sealed set's parser raised `LayoutError` (exit 1)."""

    def __init__(self, set_name: str, problem: LayoutError, openings: int):
        super().__init__(f"the {set_name} file does not have its published layout: {problem}")
        self.openings = openings


class SplitRule(NamedTuple):
    """The calibration protocol's draw (plan 6): seed, count, half rule; from the run matrix."""

    seed: int
    n_calibration: int
    half_below: int | None
    matrix: str

    @classmethod
    def of(cls, matrix: gate_config.Matrix, path: str) -> "SplitRule":
        return cls(matrix.split_seed, matrix.n_calibration, matrix.calibration_half_below, path)


def default_out(root: Path, setting: str) -> Path:
    return Path(root) / "data" / "derived" / OUT_NAMES[setting]


# --- the calibration days and the window of a Norway or Korea unit --------------------------------


def calibration_dates(unit: str, site: str, dates, rule: SplitRule) -> tuple:
    """The calibration days among `dates` as the protocol draws them (`gate_run.plan_unit`):
    keys "<date> <site>", `calibration_count` of them by `split_days`; none when not split."""
    ordered = sorted(dates)
    keys = [f"{day} {site}" for day in ordered]
    count = calibration_count(len(keys), rule.n_calibration, rule.half_below)
    if not is_split(len(keys), count):
        return ()
    chosen = set(split_days(unit, rule.seed, keys, n_calibration=count, min_days=count + 1)[0])
    return tuple(day for day, key in zip(ordered, keys, strict=True) if key in chosen)


@dataclass
class WindowSearch:
    """The outcome of `search_window` for one unit."""

    window: Window | None
    detail: dict
    calibration: tuple = ()
    reason: str = ""  # why the unit abstains; empty when it has a window
    read: set = field(default_factory=set)


def search_window(part: pd.DataFrame, *, unit: str, zone: str, limit_kw: float, rule: SplitRule):
    """The window of a held-out Norway or Korea unit on its calibration days (reading W1)."""
    site = str(part["site"].iloc[0])
    profile = occupancy_profile(part["plug_in"], part["plug_out"], zone)
    days = list(profile.index)
    chosen = calibration_dates(unit, site, days, rule)
    start_days, taus, read = len(chosen), [], set(chosen)
    produced_by = None  # the τ whose calibration days `chosen` are; None for the start draw
    counting: dict[int, list] = {}
    drawn: dict[int, tuple] = {}  # τ -> its calibration days
    ruled: dict[int, tuple] = {}  # τ -> the rule's (window, detail) on its calibration days
    # Each pass either stops or adds a τ not met before; τ has 24 values, so the search stops by
    # the 25th application of the rule.
    while True:
        if not chosen:
            where = "the start draw" if produced_by is None else f"τ {produced_by:02d}:00"
            reason = f"no calibration day at {where}"
            if produced_by is not None:
                reason += f" ({len(counting[produced_by])} counting days)"
            return WindowSearch(None, {"search_taus": taus}, (), reason, read)
        window, detail = window_from_profile(profile.loc[list(chosen)])
        if produced_by is not None:
            ruled[produced_by] = (window, detail)
        taus.append(window.tau_hour)
        if window.tau_hour in counting:  # met before: settled (the τ just used) or a cycle
            break
        _, table = truth.build(
            part,
            window,
            zone,
            unit=unit,
            days=days,
            limit_kw=limit_kw,
            bound_by_charge_fields=False,
            setting=truth.MAIN,
        )
        counting[window.tau_hour] = [
            d for d, c in zip(table["date"], table["counts"], strict=True) if c
        ]
        produced_by = window.tau_hour
        chosen = calibration_dates(unit, site, counting[produced_by], rule)
        drawn[produced_by] = chosen
        read |= set(chosen)
    tau = taus[-1]  # the first τ met twice: the fixed point when the search settles
    cycle = tau != produced_by
    window_days = "calibration days of the unit at its window (plan 2; reading W1)"
    if cycle:
        # `window` is the rule's at this last step, the last to give τ: its S stands (W1(c))
        chosen = drawn[tau]
        found, detail = ruled[tau]
        window = replace(window, rule=CYCLE_RULE)
        detail = {**detail, "search_cycle": True, "cycle_rule_tau_hour": found.tau_hour}
        window_days = (
            "calibration days of the unit at the first τ its search met twice; S from the window "
            "the rule last gave at that τ, on the calibration days at the τ before it (plan 2, a "
            "search that cycles; reading W1)"
        )
    else:
        window = replace(window, rule=SEARCH_RULE)
    test_days = set(counting[tau]) - set(chosen)
    detail.update(
        {
            "window_days": window_days,
            "calibration_seed": rule.seed,
            "calibration_days": len(chosen),
            "calibration_dates": [str(day) for day in chosen],
            "counting_days_at_tau": len(counting[tau]),
            "search_taus": taus,
            "start_days": start_days,
            "days_read": len(read),
            "test_days_read": len(read & test_days),
        }
    )
    return WindowSearch(window, detail, chosen, "", read)


# --- building the units ---------------------------------------------------------------------------


class HeldoutBuild(truth_build.Build):
    """`truth_build.Build` with the held-out units' abstentions and Korean fast-charger count."""

    def __init__(self, setting: str, rule: SplitRule, fast_charger_rows: int) -> None:
        super().__init__(setting)
        self.rule = rule
        self.abstained: dict[str, dict] = {}
        self.fast_charger_rows = fast_charger_rows


def _removed(dropped: pd.DataFrame) -> dict:
    counts = dropped["reason"].value_counts().sort_index()
    return {str(reason): int(n) for reason, n in counts.items()}


def _site_checks(checks: dict, site: str, sessions, dropped, zone: str, limit: float) -> None:
    rows = len(sessions) + len(dropped)
    checks["rows_read"][site] = rows
    checks["drop_shares"][site] = data_checks.drop_shares(rows, dropped)
    if len(sessions):
        checks["dst_shift"][site] = data_checks.dst_shift(sessions, zone)
        checks["units"][site] = data_checks.units(sessions, limit)
        checks["midnight"][site] = data_checks.midnight(sessions)
        checks["weekday_plug_in_hours"][site] = data_checks.weekday_plug_in_hours(sessions)


def _empty_checks() -> dict:
    names = ("rows_read", "drop_shares", "dst_shift", "units", "midnight", "weekday_plug_in_hours")
    return {name: {} for name in names}


NO_SESSION = "no session"  # a unit with no row at all in the held-out files: on the command a fault
NO_SESSION_FAULT = "no session in the held-out files: a reading fault, not a result"
ALL_DROPPED = "every row dropped by cleaning"  # rows read, none kept: a result about the data


def all_dropped_reason(dropped) -> str:
    """`every row dropped by cleaning: <n> rows, by reason <reason>: <n>, ...`, the reasons in the
    order of their names (`_removed`)."""
    removed = _removed(dropped)
    by_reason = ", ".join(f"{reason}: {n}" for reason, n in removed.items())
    return f"{ALL_DROPPED}: {sum(removed.values())} rows, by reason {by_reason}"


def _abstain_without_session(result: HeldoutBuild, unit: str, sessions, dropped) -> None:
    """A unit with no kept session: every row dropped by cleaning (rows were read, so a result
    about the data: its reason is `all_dropped_reason`), or no row at all (`NO_SESSION`: the
    command refuses it as a reading fault)."""
    reason = all_dropped_reason(dropped) if len(dropped) else NO_SESSION
    _abstain(result, unit, reason, sessions, dropped)


def _abstain(result: HeldoutBuild, unit: str, reason: str, sessions, dropped, found=None) -> None:
    """Record an abstaining unit; with `found`, the window search it ran: its τ of each step and
    its number of days read (no window, so no test days)."""
    result.abstained[unit] = {
        "reason": reason,
        "sessions_loaded": len(sessions),
        "rows_removed": _removed(dropped),
    }
    if found is not None:
        result.abstained[unit]["search_taus"] = found.detail["search_taus"]
        result.abstained[unit]["days_read"] = len(found.read)


def build_asr(result: HeldoutBuild, sessions: pd.DataFrame, dropped: pd.DataFrame) -> None:
    """The two a.s.r. units at the workplace window (readings T2, T3)."""
    limit = load_asr.LIMIT_KW
    checks = _empty_checks()
    _site_checks(checks, load_asr.SITE, sessions, dropped, load_asr.ZONE, limit)
    result.checks["asr"] = checks
    unit_of_drop = dropped["local_date"].map(load_asr.unit_of)
    if not len(sessions):
        for unit in ASR_UNITS:
            _abstain_without_session(
                result, unit, sessions, dropped[(unit_of_drop == unit).to_numpy()]
            )
        return
    unit_of_session = load_asr.periods(sessions)["unit"]
    all_days = site_days(sessions["plug_in"])
    for unit in ASR_UNITS:
        part = sessions[(unit_of_session == unit).to_numpy()]
        gone = dropped[(unit_of_drop == unit).to_numpy()]
        days = [day for day in all_days if load_asr.unit_of(day) == unit]
        if not days:
            if not len(part):  # no kept session: every row dropped by cleaning, or no row at all
                _abstain_without_session(result, unit, part, gone)
            else:  # kept sessions, but no counting day of the unit
                _abstain(result, unit, "no day of the unit", part, gone)
            continue
        cars, day_table = truth.build(
            sessions,  # every a.s.r. session, so the rate history spans both units (T2)
            WORKPLACE,
            load_asr.ZONE,
            unit=unit,
            days=days,
            limit_kw=limit,
            bound_by_charge_fields=False,
            setting=result.setting,
        )
        readings = truth_readings.count_changes(
            cars, sessions, limit_kw=limit, bound_by_charge_fields=False, setting=result.setting
        )
        result.add_unit(unit, ROLE, cars, day_table, part, gone, readings)
        result.windows[unit] = truth_build._window_entry(WORKPLACE, load_asr.ZONE)


def build_sites(
    result: HeldoutBuild, set_name: str, units, sessions: pd.DataFrame, dropped: pd.DataFrame
) -> None:
    """One unit per site (Norway: per site; Korea: per portfolio), each at its searched window."""
    zone, limit = SITE_SETS[set_name]
    checks = _empty_checks()
    for unit in units:
        part = sessions[(sessions["site"] == unit).to_numpy()]
        gone = dropped[(dropped["site"] == unit).to_numpy()]
        _site_checks(checks, unit, part, gone, zone, limit)
        if not len(part):
            _abstain_without_session(result, unit, part, gone)
            continue
        found = search_window(part, unit=unit, zone=zone, limit_kw=limit, rule=result.rule)
        if found.window is None:
            _abstain(result, unit, found.reason, part, gone, found)
            continue
        window, detail = found.window, found.detail
        days = site_days(part["plug_in"])
        cars, day_table = truth.build(
            part,
            window,
            zone,
            unit=unit,
            days=days,
            limit_kw=limit,
            bound_by_charge_fields=False,
            setting=result.setting,
        )
        alternative = Window(detail["alternative_tau_hour_earliest_peak"], 8, "alternative")
        _, other_days = truth.build(
            part,
            alternative,
            zone,
            unit="alternative",
            days=days,
            limit_kw=limit,
            bound_by_charge_fields=False,
            setting=result.setting,
        )
        detail["alternative_counting_days"] = int(other_days["counts"].sum())
        result.windows[unit] = truth_build._window_entry(window, zone, detail)
        readings = truth_readings.count_changes(
            cars, part, limit_kw=limit, bound_by_charge_fields=False, setting=result.setting
        )
        result.add_unit(unit, ROLE, cars, day_table, part, gone, readings)
    result.checks[set_name] = checks


def build_heldout(loaded: dict, setting: str, rule: SplitRule) -> HeldoutBuild:
    """Every held-out unit's tables in `setting`, from the three sets' (sessions, dropped)."""
    fast = int((loaded["korea"][1]["reason"] == FAST_REASON).sum())
    result = HeldoutBuild(setting, rule, fast)
    build_asr(result, *loaded["asr"])
    build_sites(result, "norway", NORWAY_UNITS, *loaded["norway"])
    build_sites(result, "korea", KOREA_UNITS, *loaded["korea"])
    return result


# --- opening --------------------------------------------------------------------------------------


def load_sets(root: Path) -> dict:
    """The three sets through the gated routes, in the order `SETS`, each opened once."""
    loaded = {}
    for number, (set_name, route) in enumerate(ROUTES):
        try:
            loaded[set_name] = route(root)
        except heldout_loader.SealRefused as problem:
            raise CouldNotRun(
                f"the {set_name} file was refused by the opener: {problem}; openings logged by "
                f"this run: {number}"
            ) from problem
        except LayoutError as problem:
            raise SetLayoutError(set_name, problem, number + 1) from problem
    return loaded


# --- checks before the first opening --------------------------------------------------------------


def _inside(path: Path, top: Path) -> bool:
    return path == top or top in path.parents


def _written(path) -> Path:
    """`path` made absolute from its text and the current folder alone, with no call to the
    file system: symbolic links are not followed."""
    return Path(os.path.normpath(os.path.abspath(path)))


def _ignored(root: Path, path: Path) -> bool:
    """Whether git, run as the opener runs it, ignores `path` (inside `root`)."""
    relative = path.relative_to(root).as_posix()
    done = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            *heldout_loader.GIT_SETTINGS,
            "check-ignore",
            "-q",
            "--",
            relative,
        ],
        capture_output=True,
        text=True,
        check=False,
        env=heldout_loader.git_environment(),
    )
    return done.returncode == 0


def _writable(folder: Path) -> bool:
    """Whether `folder` (absolute, so the climb ends at the root folder at the latest) is, or
    could be made as, a writable folder."""
    place = folder
    while not place.exists():
        if place.is_symlink():
            return False
        place = place.parent
    return place.is_dir() and os.access(place, os.W_OK | os.X_OK)


def preflight(root: Path, out: Path, setting: str, run_dir: str | None) -> None:
    """Raise `CouldNotRun` for anything that would stop the run after an opening (step 1)."""
    sealed = _written(Path(root) / heldout_seal.FOLDER)
    for given, what in ((out, "output folder"), (run_dir, "run folder")):
        if given and _inside(_written(given), sealed):
            raise CouldNotRun(f"the {what} {given} lies inside the sealed folder")
    top = Path(root).resolve()
    target = Path(out).resolve()
    for name in ("truth", "truth-held"):
        if target == (top / "data" / "derived" / name).resolve():
            raise CouldNotRun(f"{out} is the training and development build's folder")
    try:
        truth_build.refuse_other_setting(target, setting)
    except OSError as problem:
        raise CouldNotRun(str(problem)) from problem
    if target.exists() and not target.is_dir():
        raise CouldNotRun(f"{out} is not a folder")
    if not _writable(_written(out)):  # as written: a link to nothing below it cannot be made
        raise CouldNotRun(f"{out} cannot be made or written")
    if _inside(target, top) and not _ignored(top, target):
        raise CouldNotRun(f"{out} lies inside the repository and git does not ignore it")
    if run_dir:
        folder = Path(run_dir).resolve()
        if not (folder.is_dir() and os.access(folder, os.W_OK | os.X_OK)):
            raise CouldNotRun(f"the run folder {run_dir} is not a writable folder")
        if _inside(folder, top) and not _ignored(top, folder):
            raise CouldNotRun(f"the run folder {run_dir} lies inside the repository, not ignored")


# --- report and summary ---------------------------------------------------------------------------


def findings(result: HeldoutBuild) -> list[str]:
    out = []
    for source, checks in result.checks.items():
        out.extend(data_checks.findings(source, checks))
    for unit, entry in result.abstained.items():
        out.append(f"{unit}: no tables, the unit abstains: {entry['reason']}")
    for unit, entry in result.windows.items():
        if entry.get("search_cycle"):
            out.append(
                f"{unit}: the window search cycles through τ {entry['search_taus']}: τ "
                f"{entry['tau']}, the first τ met twice, stands with its calibration days, on "
                f"which the rule's own τ is {entry['cycle_rule_tau_hour']:02d}:00 (plan 2; "
                "reading W1)"
            )
        if entry.get("test_days_read", 0) > 0:
            out.append(
                f"{unit}: the window search read {entry['test_days_read']} test days of its "
                "final window (reading W1)"
            )
    return out


def report_lines(result: HeldoutBuild, written: dict[str, dict]) -> list[str]:
    lines = []
    if result.setting != truth.MAIN:
        lines += [truth_build.setting_line(result.setting), ""]
    lines.append(FINDINGS_HEADING)
    lines += [f"- {text}" for text in findings(result)] or ["- none"]
    lines.append("")
    lines.append(
        f"KOREA FAST CHARGERS: {result.fast_charger_rows} rows removed as "
        f"'{FAST_REASON}' at the opening (plan 3)"
    )
    lines.append("")
    lines.append("WINDOWS")
    if result.setting != truth.MAIN:
        lines.append(truth_build.WINDOWS_AS_MAIN)
    for unit, entry in result.windows.items():
        extra = ""
        if "peak_median" in entry:
            cycle = ""
            if entry.get("search_cycle"):
                cycle = (
                    f", a cycle: the rule's own τ on these days "
                    f"{entry['cycle_rule_tau_hour']:02d}:00; S from the window the rule last gave "
                    f"at τ {entry['tau']}, on the calibration days at τ "
                    f"{entry['search_taus'][-2]:02d}:00"
                )
            extra = (
                f"; median peak {entry['peak_median']} cars at {entry['peak_hour']:02d}:00 over "
                f"{entry['days']} calibration days (seed {entry['calibration_seed']}; search τ "
                f"{entry['search_taus']}{cycle}; days read {entry['days_read']}, test days read "
                f"{entry['test_days_read']}); earliest-peak reading τ "
                f"{entry['alternative_tau_hour_earliest_peak']:02d}:00 would give "
                f"{entry['alternative_counting_days']} counting days"
            )
        lines.append(
            f"- {unit}: τ {entry['tau']}, S {entry['service']} ({entry['service_hours']} hours)"
            f"{extra}"
        )
    lines.append("")
    lines.append("UNITS")
    for unit, entry in result.summary.items():
        spread = entry["cars_per_counting_day"] or {}
        lines.append(
            f"- {unit} ({entry['role']}): sessions {entry['sessions_loaded']}, removed "
            f"{entry['rows_removed'] or 'none'}, days {entry['days']}, counting days "
            f"{entry['counting_days']}, cars per counting day median {spread.get('median')} "
            f"(p10 {spread.get('p10')}, p90 {spread.get('p90')}), in-window departure "
            f"{entry['share_in_window_departure']}, zero room {entry['share_zero_room']}, "
            f"room not fitting {entry['share_room_not_fitting_rate_times_depart_hour']}, "
            f"depart hour 0 {entry['share_depart_hour_zero']}"
        )
        lines.append(f"  fallback shares {entry['fallback_shares']}")
        lines.append(f"  cars changed by other readings {entry['reading_changes']}")
    for unit, entry in result.abstained.items():
        lines.append(
            f"- {unit} ({ROLE}): abstains, {entry['reason']}; sessions "
            f"{entry['sessions_loaded']}, removed {entry['rows_removed'] or 'none'}"
        )
    lines.append("")
    lines.append("READINGS COUNTED PER UNIT (cars on counting days whose value would change)")
    for key, text in truth_readings.READINGS.items():
        if result.setting == truth.HELD and key in truth_readings.NOT_APPLICABLE_HELD:
            text += " (not applicable in the held setting: no energy is drawn before τ)"
        lines.append(f"- {key}: {text}")
    lines.append("")
    lines.append("CHECKS")
    for source, checks in result.checks.items():
        for name, value in checks.items():
            lines.append(f"- {source} {name}: {json.dumps(value, sort_keys=True, default=str)}")
    lines.append("")
    lines.append("TABLES WRITTEN")
    for unit, hashes in written.items():
        lines.append(f"- {unit}: {hashes}")
    return lines


def summary_document(result: HeldoutBuild, written: dict[str, dict]) -> dict:
    units = {unit: {**entry, "sha256": written[unit]} for unit, entry in result.summary.items()}
    document = {
        "units": units,
        "windows": result.windows,
        "abstained": result.abstained,
        "findings": findings(result),
        "checks": result.checks,
        "calibration_rule": result.rule._asdict(),
        "korea_fast_charger_rows_removed": result.fast_charger_rows,
    }
    if result.setting != truth.MAIN:
        document["setting"] = result.setting
    return document


# --- the command ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None, root: Path = TOP) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--setting", choices=truth.SETTINGS, default=truth.MAIN)
    parser.add_argument("--out", type=Path, default=None, help="default: by setting (OUT_NAMES)")
    parser.add_argument("--matrix", type=Path, default=None, help="default: gate/matrix.json")
    args = parser.parse_args(argv)
    root = Path(root)
    out = args.out if args.out is not None else default_out(root, args.setting)
    given = args.matrix if args.matrix is not None else MATRIX  # recorded as given
    matrix = given if args.matrix is not None else root / MATRIX
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    try:
        rule = SplitRule.of(gate_config.load(matrix), str(given))
        preflight(root, out, args.setting, run_dir)
    except gate_config.MatrixError as problem:
        print(f"COULD NOT RUN: the run matrix is refused: {problem}; nothing opened")
        return 2
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}; nothing opened")
        return 2
    try:
        loaded = load_sets(root)
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}; nothing written")
        return 2
    except SetLayoutError as problem:
        print(
            f"FAILED: {problem}; nothing written for any set; openings logged by this run: "
            f"{problem.openings}"
        )
        return 1
    result = build_heldout(loaded, args.setting, rule)
    faults = [unit for unit, entry in result.abstained.items() if entry["reason"] == NO_SESSION]
    if faults:  # a reading fault, not a result: no table is written (module text, no session)
        print(
            "COULD NOT RUN: "
            + "; ".join(f"{unit}: {NO_SESSION_FAULT}" for unit in faults)
            + "; nothing written"
        )
        return 2
    try:
        written = truth_build.write(result, Path(out))
    except OSError as problem:
        print(f"COULD NOT RUN: cannot write the tables: {problem}")
        return 2
    text = "\n".join(report_lines(result, written)) + "\n"
    print(text, end="")
    if run_dir:
        folder = Path(run_dir)
        document = summary_document(result, written)
        document[ROUTE_KEY] = ROUTE_RECORD
        (folder / truth_build.SUMMARY_NAME).write_text(truth_build.summary_text(document))
        (folder / truth_build.REPORT_NAME).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

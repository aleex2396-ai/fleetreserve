"""The closed-form half of the WP11 pass-mark item, read from the recorded corner search (plan
section 13, WP11: "Corner search; run matrix frozen. Pass: closed-form worst case reproduced on 20
days").

It covers the closed-form half only: "run matrix frozen" is not checked here, so its exit 0 says
that the fixed plan's closed-form worst case was reproduced on 20 days, never that WP11 is
finished. The first line it prints says so.

Run as `python tools/corner_closed_form_check.py [--top <folder>]`.

The run read is the newest folder in `runs.noindex/` named `<UTC start>-wp11-corner-search`
(`run_evidence.newest_run`; an older run is never used in its place), recorded after a truth
build at the same commit as
    python -m fleetreserve.run_record --name wp11-corner-search --long --max-seconds <n> --
        uv run --frozen --group learn python -m fleetreserve.corner_run --model <model>
        --dial <dial> --matrix gate/matrix.json --days <n> --workers <n>
(`--group learn` for a model that needs the learn packages, `lgbm_tail` among them). Neither the
model nor the dial is fixed by the item: the closed form and the corners are of each searched
day's certified box at the scales of the grid, whichever forecast made the day. That run decides
nothing (could not run) unless:
- its results `corner.json` are verified (the run ended with exit 0, was neither stopped nor timed
  out, and the file is the one it wrote) or intact (it ended by itself with exit 1 and the file
  is the one it wrote: from such a run only a disagreement is read, never a pass;
  `run_evidence`), and it was recorded at the commit checked out now or is read from an earlier
  commit by the rule of `run_evidence` (module text there; else it is stale);
- its recorded command runs `python -m fleetreserve.corner_run`;
- it read the committed matrix: the SHA-256 `corner.json` records for its matrix equals that of
  `gate/matrix.json` at the checked-out commit;
- it read the main truth setting, from the truth build its `inputs.truth_run` names: a folder
  `runs.noindex/<UTC start>-wp2-truth` whose `truth-summary.json` is verified and was recorded at
  the checked-out commit or is read from an earlier commit by the same rule (`corner_run`
  requires a build at its own commit when it starts; it is read again here so that the pass
  rests on the build's own record, not only on the search's word);
- it checked with the committed tolerance and scales: `run.slot_tolerance` equals
  `corner_search.SLOT_TOLERANCE` and `run.check_scales` equals `dial_splits.GRID`;
- each unit and variant's pass-mark entry adds up: its checked days are the first of its searched
  days, every checked day is checked at every scale, and the counts of pairs agreeing strictly,
  agreeing, differing by reason and disagreeing are those of the same pairs.

What is compared. `corner_run` checks the first `CHECK_DAYS` searched days of each unit and
variant (reading K3 of `corner_run`): per pair (day, scale of the grid) the fixed plan's closed
form on the steered cars' true rows (plan section 6, T6: the first failing hour of the hour
thresholds, and the slot loss of `gate_calibrate.slot_steps`) against the worst case over every
corner of the box, each corner replayed (`corner_search.closed_form_check`).

The tolerance, and why. A pair agrees (a) when the first failing hours are equal and the slot
losses differ by at most `corner_search.SLOT_TOLERANCE` = 1e-9, or (b) when
`fixed_plan_loss.compare_with_replay` labels the difference "within tolerance". (a): each slot
loss is (|S| - t + 1) / |S| for a whole first failing hour t (`replay.slot_loss`, and the levels
of `gate_calibrate.slot_steps`), so two slot losses of different hours of one day are at least
1 / |S| apart (1/24 or more), and the same fraction reached by two computations differs by
rounding only, a few times 1e-16; 1e-9 lies between the two, so it accepts rounding and no real
difference. (b): at the closed form's failing hour every corner's draws miss the plan by at most
`replay.FAILURE_KWH` = 1e-4 kWh in total, the amount plan section 1 does not count as a failure
("delivery misses y_t by > 1e-4 kWh"): replay meets the request there by the plan's own failure
rule, so no tolerance is added to the plan's. Every other difference is a disagreement, "decision-
hour cap" among them (the case worked by hand in tests/wp11/test_corner_decision_hour.py).

Readings (alternatives in brackets):

  C1. A day is a unit's counting day: it is reproduced when every pair of it agrees, under every
      variant that checked it, and a day checked under both variants (`forecast`,
      `oracle-departure`) counts once. [Alternative: each (unit, variant, day) counts apart; the
      same day then counts twice, under two departure readings of the same cars.]
  C2. "On 20 days" is read as 20 reproduced days in all, over the units, with no pair of any
      checked day disagreeing. [Alternative: 20 reproduced days in each unit and variant that
      offers, which `CHECK_DAYS` = 20 per unit and variant allows; a unit with fewer offering
      days could then never pass.]

Every line this tool writes that names a run read from an earlier commit (the search or its
truth build) carries `Evidence.note` after the run's name, whatever the line then says (passed,
disagreed, could not run). Not written by this tool, so without it: Python's own text of an
unexpected error, printed after "the check stopped on", which can name a file in a run's folder.

Output: the line on the closed-form half; a line with the commit and the matrix; one line per
unit and variant checked: its days, pairs, pairs agreeing strictly, within tolerance and
disagreeing, and for a disagreement its reasons and the first pairs (day, date, scale, the two
first failing hours, reason); a last line: `passed:`, `DISAGREED:`, `NO RUN:` or `COULD NOT RUN:`
with the reason.

Exit status: 0 passed (at least `PASS_DAYS` reproduced days, no pair disagreeing, from a verified
run); 1 a pair disagrees (this wins over 2, as in `check.sh`); 2 could not run (no run, a run that
cannot decide by the list above, fewer than `PASS_DAYS` reproduced days with none disagreeing, a
run that ended with exit 1 and records no disagreement: a failed variant, a unit and variant with
no pair checked, or no unit that can be split). An unexpected error is could not run too, never 1
(Python's own exit status for an uncaught error is 1, which `passmark.sh` reads as failed), also
when the tool's own imports fail.
"""

import argparse
import hashlib
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    from fleetreserve.corner_run import RESULTS_NAME
    from fleetreserve.corner_search import SLOT_TOLERANCE, WITHIN_TOLERANCE
    from fleetreserve.dial_splits import GRID
    from fleetreserve.run_evidence import CannotCheck, head_commit, newest_run, read_evidence
    from fleetreserve.run_record import RUNS_FOLDER
    from fleetreserve.truth import MAIN
    from fleetreserve.truth_build import SUMMARY_NAME as TRUTH_SUMMARY
    from fleetreserve.truth_verify import RUN_LABEL as TRUTH_LABEL
except Exception as problem:  # the check cannot start: could not run, never Python's exit 1
    print(f"COULD NOT RUN: the check cannot start: {type(problem).__name__}: {problem}")
    sys.exit(2)

LABEL = "wp11-corner-search"
MATRIX = "gate/matrix.json"
CORNER_RUN = "fleetreserve.corner_run"
PASS_DAYS = 20  # plan section 13, WP11: "reproduced on 20 days" (reading C2)
SHOWN = 3  # the disagreeing pairs named per unit and variant
HALF = (
    "WP11, closed-form half only: the fixed plan's closed-form worst case against the corner "
    "search's replayed worst case; 'run matrix frozen' is not checked here, so exit 0 is not all "
    "of WP11"
)
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2


def committed_matrix_sha256(top: Path) -> str:
    """SHA-256 of `gate/matrix.json` as committed at the checked-out commit."""
    done = subprocess.run(
        ["git", "-C", str(top), "show", f"HEAD:{MATRIX}"], capture_output=True, check=False
    )
    if done.returncode != 0:
        raise CannotCheck(f"{MATRIX} is not committed at the checked-out commit")
    return hashlib.sha256(done.stdout).hexdigest()


def runs_corner_run(command) -> bool:
    """True for a recorded command that runs `python -m fleetreserve.corner_run`."""
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        return False
    return any(command[k - 1 : k + 1] == ["-m", CORNER_RUN] for k in range(1, len(command)))


def _at(document, *keys):
    """`document[keys[0]][keys[1]]...`, or None where a key is absent or a level is no mapping."""
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def check_truth_build(top: Path, name, head: str) -> None:
    """The truth build `name` is a main build of `runs.noindex/`, verified and recorded at `head`
    or read from an earlier commit by the rule of `run_evidence`."""
    pattern = r"\d{8}T\d{6}Z-" + re.escape(TRUTH_LABEL)
    if not isinstance(name, str) or not re.fullmatch(pattern, name):
        raise CannotCheck(
            f"the search names no main truth build of {RUNS_FOLDER}/ (inputs.truth_run {name!r})"
        )
    evidence = read_evidence(top / RUNS_FOLDER / name, TRUTH_SUMMARY, head)
    if not evidence.verified:
        raise CannotCheck(f"the truth build {name}{evidence.note} is not verified")


def read_run(run: Path, top: Path, head: str, committed: str):
    """The evidence of `corner.json`, once the run is shown to decide (module text); raises
    `CannotCheck` otherwise."""
    evidence = read_evidence(run, RESULTS_NAME, head)
    name = f"{run.name}{evidence.note}"
    if not runs_corner_run(evidence.manifest.get("command")):
        raise CannotCheck(f"the recorded command of {name} does not run {CORNER_RUN}")
    results = evidence.data
    used = _at(results, "matrix", "sha256")
    if not isinstance(used, str):
        raise CannotCheck(
            f"{RESULTS_NAME} of {name} records no SHA-256 of its matrix, so {MATRIX} differs"
        )
    if used != committed:
        raise CannotCheck(
            f"{name} ran on another matrix: SHA-256 {used[:12]}, the committed {MATRIX} has "
            f"{committed[:12]}"
        )
    setting = _at(results, "inputs", "setting")
    if setting != MAIN:
        raise CannotCheck(f"{name} read the truth setting {setting!r}, not {MAIN!r}")
    check_truth_build(top, _at(results, "inputs", "truth_run"), head)
    tolerance = _at(results, "run", "slot_tolerance")
    if tolerance != SLOT_TOLERANCE:
        raise CannotCheck(
            f"{name} checked with the slot tolerance {tolerance!r}, not "
            f"corner_search.SLOT_TOLERANCE {SLOT_TOLERANCE!r}"
        )
    scales = _at(results, "run", "check_scales")
    if scales != list(GRID):
        raise CannotCheck(f"{name} checked at the check scales {scales!r}, not {list(GRID)!r}")
    return evidence


@dataclass(frozen=True)
class Checked:
    """The pass-mark check of one unit and variant, as `corner.json` records it."""

    unit: str
    variant: str
    days: tuple[int, ...]
    pairs: int
    strict: int
    within: int
    disagreements: tuple[dict, ...]

    def line(self) -> str:
        agreed = self.pairs - len(self.disagreements)
        text = (
            f"  {self.unit} | {self.variant}: {len(self.days)} days, {self.pairs} pairs: "
            f"{self.strict} agree strictly, {agreed - self.strict} within tolerance, "
            f"{len(self.disagreements)} disagree"
        )
        if not self.disagreements:
            return text
        named = "; ".join(
            f"day {d['day']} ({d['date']}) scale {d['scale']:g}: closed-form hour "
            f"{d['closed_form_first_failure']}, corners hour {d['corner_first_failure']}, "
            f"{d['reason']}"
            for d in self.disagreements[:SHOWN]
        )
        more = len(self.disagreements) - SHOWN
        return f"{text} ({_reasons(self.disagreements)}); {named}" + (
            f"; and {more} more" if more > 0 else ""
        )


def _reasons(disagreements) -> str:
    counts = Counter(d["reason"] for d in disagreements)
    return ", ".join(f"{reason} {n}" for reason, n in sorted(counts.items()))


def _checked(unit: str, variant: str, found: dict, scales: list, where: str) -> Checked:
    """One unit and variant's pass-mark entry, once it adds up (module text)."""
    p = found["pass_mark"]
    days = tuple(p["days"])
    wrong = tuple(p["disagreements"])
    reasons = p["differences_by_reason"]
    pairs, strict, agreed = p["pairs"], p["pairs_agreed_strict"], p["pairs_agreed"]
    within = reasons.get(WITHIN_TOLERANCE, 0)
    adds_up = (
        p["days_checked"] == len(days)
        and list(days) == list(found["searched_days"][: len(days)])
        and p["scales"] == (scales if days else [])
        and pairs == len(days) * len(scales)
        and agreed == pairs - len(wrong)
        and sum(reasons.values()) == pairs - strict
        and agreed - strict == within
        and all(d["day"] in days and d["reason"] != WITHIN_TOLERANCE for d in wrong)
    )
    if not adds_up:
        raise CannotCheck(f"{where}: the pass-mark counts do not add up")
    return Checked(unit, variant, days, pairs, strict, within, wrong)


def checked_entries(results: dict, name: str) -> tuple[list[Checked], list[str]]:
    """The unit and variant checks of `corner.json` and the units and variants without one;
    `name` is how a line names the run."""
    found, unchecked = [], []
    for unit, entry in results["units"].items():
        where = f"{name}, {unit}"
        try:
            variants = entry["variants"].items()
        except (KeyError, TypeError, AttributeError) as problem:
            raise CannotCheck(f"{where}: {RESULTS_NAME} is malformed ({problem!r})") from problem
        for variant, value in variants:
            where = f"{name}, {unit} | {variant}"
            if "pass_mark" not in value:
                unchecked.append(f"{unit} | {variant}")
                continue
            try:
                found.append(_checked(unit, variant, value, list(GRID), where))
            except (KeyError, TypeError, AttributeError) as problem:
                raise CannotCheck(
                    f"{where}: {RESULTS_NAME} is malformed ({problem!r})"
                ) from problem
    return found, unchecked


def reproduced_days(entries: list[Checked]) -> set[tuple[str, int]]:
    """The (unit, day) pairs checked with no disagreeing pair under any variant (reading C1)."""
    checked = {(e.unit, d) for e in entries for d in e.days}
    wrong = {(e.unit, d["day"]) for e in entries for d in e.disagreements}
    return checked - wrong


def decide(evidence, run: Path) -> tuple[int, list[str]]:
    """(exit status, lines) of a run that can decide (module text)."""
    results = evidence.data
    entries, unchecked = checked_entries(results, f"{run.name}{evidence.note}")
    lines = [e.line() for e in entries]
    wrong = [(e.unit, d) for e in entries for d in e.disagreements]
    if wrong:
        days = {(unit, d["day"]) for unit, d in wrong}
        lines.append(
            f"DISAGREED: {len(wrong)} pair{'s' if len(wrong) > 1 else ''} on {len(days)} "
            f"day{'s' if len(days) > 1 else ''} disagree ({_reasons([d for _, d in wrong])}); "
            f"{run.name}{evidence.note}"
        )
        return FAILED, lines
    problems = []
    failed = results.get("failed")
    if failed:
        problems.append(f"failed: {'; '.join(str(f) for f in failed)}")
    elif unchecked:
        problems.append(f"not checked: {', '.join(unchecked)}")
    empty = [f"{e.unit} | {e.variant}" for e in entries if e.pairs == 0]
    if empty:
        problems.append(f"no pair checked: {', '.join(empty)}")
    if not entries:
        problems.append("no unit and variant checked")
    if not evidence.verified or problems:
        code = evidence.manifest.get("exit_code")
        why = "; ".join(problems) or "no reason recorded"
        lines.append(
            f"COULD NOT RUN: {run.name}{evidence.note} ended with exit code {code} and records no "
            f"disagreement ({why}), so it cannot decide the item"
        )
        return COULD_NOT_RUN, lines
    days = reproduced_days(entries)
    if len(days) < PASS_DAYS:
        lines.append(
            f"COULD NOT RUN: the closed-form worst case is reproduced on {len(days)} days, fewer "
            f"than the {PASS_DAYS} the pass mark needs, and no pair disagrees"
        )
        return COULD_NOT_RUN, lines
    units = len({unit for unit, _ in days})
    lines.append(
        f"passed: the closed-form worst case is reproduced on {len(days)} days of {units} units "
        f"(at least {PASS_DAYS} needed), {sum(e.pairs for e in entries)} pairs, none disagreeing; "
        f"{run.name}{evidence.note} (model {results.get('model')}, dial {results.get('dial')})"
    )
    return PASSED, lines


def check(top: Path) -> int:
    print(HALF)
    head = head_commit(top)
    committed = committed_matrix_sha256(top)
    print(
        f"corner search checked at commit {head[:12]} (a run accepted from an earlier commit says "
        f"so on its line), on the committed {MATRIX} ({committed[:12]})"
    )
    try:
        run = newest_run(top, LABEL)
    except CannotCheck as problem:
        print(f"NO RUN: {problem}")
        return COULD_NOT_RUN
    evidence = read_run(run, top, head, committed)
    code, lines = decide(evidence, run)
    for line in lines:
        print(line)
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/corner_closed_form_check.py",
        description="Decide the closed-form half of the WP11 pass-mark item from the recorded "
        "corner search.",
    )
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        return check(options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return COULD_NOT_RUN
    except Exception as problem:  # an unexpected error (module text)
        print(f"COULD NOT RUN: the check stopped on {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main())

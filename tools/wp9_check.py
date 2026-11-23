"""WP9's pass mark, read from recorded runs (plan section 13, WP9: "Calibration, conformal training,
T7 test, η. Pass: loss 0 at abstain; development loss within its 200-split spread; T8's check (if
failed, the owner may cut 3: conformal training ...); conformal training kept only if it raises
the development offer").

Run as `python tools/wp9_check.py <item> [--top <folder>]`. Each item is one clause of that line,
its own line of `passmarks/wp9.sh` with its own exit status:
  abstain    "loss 0 at abstain"
  spread     "development loss within its 200-split spread"
  t8         "T8's check"
  conformal  "conformal training kept only if it raises the development offer"

Every run read is the newest folder `runs.noindex/<UTC start>-<label>` of its label
(`run_evidence.newest_run`; an older run is never used in its place). It decides nothing (could
not run) unless its results file is verified (the run ended with exit 0, was neither stopped nor
timed out, and the file is the one it wrote) or, where an item says so, intact (it ended by itself
with exit 1 or 2 and the file is the one it wrote: from such a run only a failure is read, never
a pass, so a failure the run recorded beside a part that could not run is not hidden); it was
recorded at the commit checked out now or is read from an earlier commit by the rule of
`run_evidence` (module text there; else it is stale; item t8 adds a condition, reading W3f); its
recorded command runs the item's command; it read the committed matrix (the SHA-256 its results
record for the matrix equals that of the matrix file at the checked-out commit); and it read the
main truth setting from the truth build its `inputs.truth_run` names, a folder
`runs.noindex/<UTC start>-wp2-truth` whose `truth-summary.json` is verified, and recorded at the
checked-out commit or read from an earlier commit by the same rule. Every line this tool writes
that names a run read from an earlier commit (the item's run, a dial-choice run or a truth build)
carries `Evidence.note` after the run's name, whatever the line then says (passed, failed, could
not run). Not written by this tool, so without it: Python's own text of an unexpected error,
printed after "the check stopped on", which can name a file in a run's folder.

abstain. Plan section 1: "Abstain: offer nothing; cars charge as logged ... β<=0 abstains";
section 6, T6: the losses "are 0 at β<=0 ... A day that does not offer has no loss and is in
neither mean". The run read is the gate run, label `wp6-gate`,
    python -m fleetreserve.run_record --name wp6-gate --long --max-seconds <n> --
        uv run --frozen python -m fleetreserve.gate_run --matrix gate/matrix.json --workers <n>
(after a truth build at the same commit): its `gate.json` and `gate-days.csv`. A trial
(`--limit-days`) or a partial run (`--only`) cannot decide.

  W1. An abstaining test entry is one (configuration, unit, test day, rung, tier) of
      `gate-days.csv` on a unit that is split where the unit's scale for that rung and tier in
      `gate.json` (`rungs.<rung>.beta_hat`; tier B `rungs.<rung>.tier_b.beta_hat`) is 0, or where
      the day does not offer (`abstain_reason` not empty). Each must show offer 0, slot loss 0
      and driver loss 0 exactly (tier A's columns and tier B's `_tier_b_` columns), and each tier
      needs at least one such entry, or the clause is not shown in it. [Alternative: the
      per-split rows of a recorded comparison run, `compare-splits.csv`, whose splits at scale 0
      write no loss at all (an empty field), so that a loss of exactly 0 is not shown there.]
  W1b. The rows of a unit that is not split (role "not calibratable") are not read and are
      counted: no scale is chosen there and the run writes no offer or loss for them, so no
      recorded quantity shows a loss of 0. [Alternative: read them as abstaining with loss 0,
      which the record does not show.]
  What it shows: the recorded offers and losses at abstain are 0, as `gate_calibrate.score_test`
  and `gate_day.evaluate` write them. No day is replayed at scale 0 (`crc`, reading R8: scale 0 is
  never evaluated; its zero loss is taken as given, as T6 states it).
  Exit: 0 every abstaining entry at 0 in both tiers; 1 an entry with an offer or a loss that is
  not 0 (read from a verified or an intact run); 2 otherwise.

spread. Section 6: "200 seeded development splits give the expected spread (fixed plan) and are
the splits on which the dial ε is fixed and the WP6 gate is read (one-way re-solve)". The reading
is the owner's decision of 16 November 2026, which replaces the two readings W2a and W2b written
here before it (W2a: one split inside the 5th to 95th percentile of the 200; W2b: the 200-split
mean at most the level): "development loss" is the mean test loss over the 200 development
splits, and it is "within its 200-split spread" when it is at most the level plus two standard
errors, the standard error being the standard deviation of the per-split mean test loss across
the splits divided by the square root of their number. The losses are the slot loss at level
0.10 and, in tier A, the driver loss at level 0.05 (`dial_splits.tier_alphas`; tier B has no
driver level). The runs read are those the WP10 item reads, through its own readers
(`tools/compare_shared_check.py`: R5, R6 and "Runs read" there): the dial-choice run of its
`GATE_MODEL` lgbm_tail (label `wp6-dial-choice-lgbm_tail`) gives each gate unit its tier A dial
at the unit `DIAL_SOURCES` names, and for each distinct dial the comparison run
    python -m fleetreserve.run_record --name wp10-compare-lgbm_tail-eps<dial> --long
        --max-seconds <n> -- uv run --frozen --group learn python -m fleetreserve.compare_run
        --model lgbm_tail --dial <dial> --matrix gate/matrix.json --splits 200 --workers <n>
is read for the gate units of that dial: verified, with 200 splits, the seeds 1 to 200, both
tiers and both variants, no draw, and its split table (`compare-splits.csv`, `compare_splits`
module text) verified and with the header of `compare_splits.columns` for the run's ceilings.

  W2c. The losses read are those of the fixed plan without re-solving (`fixed-plan`), from the
      words "give the expected spread (fixed plan)"; the one-way re-solve (`re-solve`, with which
      the gate is read) is printed beside it as shown only and decides nothing. [Alternative:
      the re-solve decides, as the rung of the gate and of the dial; or both decide.]
  W2d. The model lgbm_tail (the gate's), its forecast variant (not the oracle departures), every
      gate unit of the committed matrix, each at its tier A dial read per site type from the
      recorded dial-choice run (as the WP10 item's reading P6), tier B at the same dial.
      [Alternative: also the main hazard model, which the WP10 item compares too; also the
      oracle-departure variant.]
  W2e. A split's loss is its `mean_test_slot_loss` or `mean_test_driver_loss`: the mean over its
      offering test days at its scale (`dial_splits.score_split`). A split offers when its scale
      is above 0 and it has an offering test day. A split that offers nothing has no loss and is
      in neither the mean nor the count (section 6, T6: "A day that does not offer has no loss
      and is in neither mean"), so n is the number of splits that offer, the mean is over them,
      the standard deviation is the sample one (divisor n - 1) and the standard error is it
      divided by the square root of n. A unit, tier and loss where no split offers is not a test
      (there is no loss); with one split that offers it has no standard error and could not run.
      [Alternative: every one of the 200 splits counted, those that offer nothing at loss 0, and
      the square root of 200 always; the divisor n for the standard deviation; a unit where no
      split offers could not run.]
  W2f. The mean is compared with the level plus two standard errors exactly, with no tolerance.
      Each of tier A slot loss, tier A driver loss and tier B slot loss needs at least one unit
      with a loss, else the clause is not tested there (could not run). A run cannot decide when
      its split table lacks a seed of 1 to 200 of a unit, tier and method read (or has one
      twice), has a split that offers with no loss, or a loss on a split that offers nothing.
  Worked by hand (the made-up unit of the item's tests): of 200 splits, 10 at scale 0 and 95 each
  at slot losses 0.05 and 0.16, so n = 190, mean 0.105, standard error 0.055 / sqrt(189) =
  0.004001, level plus two standard errors 0.108001: within. At 0.06 and 0.17 the mean is 0.115,
  the standard error the same: above, failed.
  Exit: 0 every test within, with a test in every tier and loss; 1 a test above (this wins over
  2, also beside a run that cannot decide); 2 otherwise.

t8. Section 7, T8: "Check: autograd matches finite differences on 100 days to 1e-3 relative error,
departure and energy directions; a comparison that shows a kink at both step sizes is recorded as a
kink, neither a pass nor a failure, and with whole-hour departures the departure direction is
reported as not applicable ...; one MPS batch timed; Λ's tracking of the certified causal value
logged on development days." The runs read are those listed in `T8_LIST` (reading W3e), every
one required, each recorded as
    python -m fleetreserve.run_record --name <label> --long --max-seconds <n> --
        uv run --frozen --group learn python -m fleetreserve.soft_scale_check --model <model>
        --configuration <configuration> [--matrix <its matrix>] [--days <its days per unit>]
with the listed model, configuration and matrix, the listed days per unit (the command's default
`DAYS_PER_UNIT` unless the entry names more, reading W3e; fewer: a trial; more: it cannot decide
either) and every gate unit of its matrix (fewer: a partial run). Its
`soft-scale-check.json` is read verified, or intact with exit 1 or 2 (only a failed comparison is
read from it). The failed deciding comparisons and the fields not exercised are counted again
from the per-day comparisons, with the run's own deciding fields; counts that differ from the
run's own cannot decide.

  W3a. The runs are combined over distinct days, a day being its (site, date): a day checked in
      more than one run (the same days under two configurations, or a site's day in its own unit
      and in a pooled one) counts once. Per direction (function soft or exact, field departure,
      cap or need), a day's outcome is the strongest over the runs where that field is deciding:
      failed, then passed at a resolvable step, passed at a step that is not resolvable, kink,
      unresolved, tied. So where two runs disagree on a day and direction, a failure in either
      is a failure (and fails the item), and otherwise a pass in either is a pass. The day is
      credited to the unit of the first check (in the order of `T8_LIST`) that gave it.
      [Alternative: every check counts apart, which counts the same day twice.]
  W3b. "Autograd matches finite differences on 100 days ..., departure and energy directions" is
      read as: in each of the six directions (soft and exact, each in departure, cap and need) a
      passed comparison on at least `T8_DAYS` distinct days, and no deciding comparison failed in
      any run. A comparison passed at a step that is not resolvable counts: the command passes
      one there only when its whole interval is within the limit (`soft_scale_check`, rule part
      3). [Alternative: only a pass at a resolvable step counts.] Kink, unresolved and tied
      comparisons count toward no direction's 100. The plan settles that fewer days with a pass do
      not satisfy it: T8 asks 100 distinct days with a passed comparison per direction (the
      owner's decision of 18 November 2026). Below 100 in a direction the item answers 2 and
      prints each direction's counts. [Alternative, not taken: 100 distinct days checked and
      every direction passed at least once, under which one passed comparison per direction and
      every other one unresolved would pass.]
  W3c. A field that is not deciding in a run (the departure under whole-hour forecast departures,
      written by the command in `not_applicable`) is not read from that run, a failure there
      included. In that direction the run's days are not applicable, unless another listed run
      decides the direction on the same day, and count toward no direction's 100. A direction is
      decided by the listed runs where it is deciding: one where it is not applicable never
      blocks it, so the fractional run's departure on 100 days is enough beside a whole-hour run;
      a direction that is deciding in no listed run has no passed day, and the item answers 2.
  W3d. "One MPS batch timed": at least one run whose `batch_timing.mps` holds its seconds (on a
      machine without MPS the command writes "not available": could not run). "Tracking logged
      on development days": at least one run with a development unit
      (`truth_build.DEVELOPMENT_UNITS`) whose tracking covers at least one day.
  W3e. The runs combined are those listed in `T8_LIST` as committed at the checked-out commit
      (an entry written but not committed is not read): per entry the run label and the model,
      configuration and committed matrix its run must have, and, where the entry names it
      (`T8_DAYS_FIELD`), its days per unit (the command's `--days`), a whole number of at least
      `DAYS_PER_UNIT` (absent: `DAYS_PER_UNIT`), so that a run on a few units can check more of
      their days than the command's default while a trial still cannot decide. A label must have
      the form the run recorder allows (letters, digits, '-' and '_') and begin with `T8_PREFIX`;
      a label listed twice, an entry without its four fields or with a field it does not know,
      days per unit that are not such a number, or a list that cannot be read cannot decide. A
      further run (a check on training-site days, say, on a matrix whose gate units are those
      sites) is added by an entry in that list, with no change to this file; it is verified as
      the others are (above) and its days join theirs by (site, date), W3a. Adding the entry is
      a commit; a run recorded before it cannot decide (reading W3f). A matrix whose gate units
      are training units is a rotation what-if (`gate_config`), checked with a model not fitted
      on them (`lgbm_tail_rotation`); its derivative comparisons measure no forecast and are read
      as any other run's; its tracking is not of a development unit (W3d). [Alternative: every
      run under a label with the prefix found in `runs.noindex/`, which would let an unlisted run
      decide and a forgotten one go unnoticed. For the days: every run at `DAYS_PER_UNIT`, so
      that a matrix of k units adds at most k times `DAYS_PER_UNIT` days.]
  W3f. A listed run recorded at an earlier commit than the checked-out one decides only when
      `T8_LIST` is the same at its commit as at the checked-out one (`read_evidence` with
      `also_compare`: git lists no difference for it between the two commits), although
      `passmarks/` is on the keep-list of `run_evidence`. So a run recorded before its entry was
      listed cannot decide the item, and any later change of the list makes every listed run from
      an earlier commit stale, to be recorded again. [Alternative: the list read like the rest of
      the keep-list, so that a run seen first could be listed afterwards and decide.]
  Exit: 0 every requirement met on verified runs; 1 a deciding comparison failed (this wins over
  2); 2 otherwise.

conformal. A decision rule, not a pass or a failure of the work package: per arm of conformal
training (`ct-scale`, `ct-abstain`) the item decides "kept" or "not kept", exit 0 for a decision
either way, and answers 2 when it cannot decide. Keeping or not keeping is recorded here; only the
owner cuts. The run read: label `SIZING_LABEL`,
    python -m fleetreserve.run_record --name wp9-size-resolve-whatif-ct-lgbm_tail_rotation --long
        --max-seconds <n> -- uv run --frozen --group learn python -m
        fleetreserve.size_resolve_whatif --model lgbm_tail_rotation --matrix gate/matrix.json
        --dial 0.02 --splits 200 --workers <n> --control-draws 19 --development-units
        nrel-before-may-2019 nrel-from-may-2019 norway-development-pooled
(the development units named being the gate units of the committed matrix), verified only (one
that ended with exit 1 or 2, a failed unit or control or a part that could not run, cannot
decide: the item is a decision, not a pass, so it has no failure to read), with `SPLITS` splits,
the development splits of section 6 (fewer: a trial), of the model and at the dial of reading
W4e.

  W4a. "The development offer" is the offer on the development units, read as WP6 and WP8 read
      it: per gate unit of the committed matrix, scored by the sizing comparison as a
      development unit (its run's `development_units`, `DEVELOPMENT`) with the finished sizings,
      fitted on training units only (the owner's decision of 16 November 2026, which settles the
      question this reading left open before; `size_resolve_whatif`, "Development units"). A
      run that scores none of them cannot decide (the development offer is not measured); one
      that scores only some is partial. The training units the comparison also scores, each with
      sizings fitted on the other training units, and any other development unit it scores, are
      printed per unit and tier as "shown only" and decide nothing. [Alternative, not taken by
      that decision: the training-unit rotation read as the development offer.]
  W4b. Per development unit and tier, never pooled, and per arm: ct raises the offer when its
      mean offer per counting test day beats every one of the run's n draws of its shuffled
      control (`shuffled-ct-<arm>`). A sizing whose sizes carry nothing about the days is, given
      its sizes, one more random arrangement of them (`size_resolve_whatif`, reading SC4), so it
      beats all n draws on one unit and tier with a chance of at most 1/(n+1); "kept" needs that
      on every unit and tier where an offer is made, so such a sizing is kept with a chance of at
      most 1/(n+1), at most 5% from n = `DRAWS_NEEDED` = 19; a run with fewer draws cannot decide.
      It is read from the gains by draw (`difference_from_shuffled.by_draw`: per draw ct's mean
      offer over the splits minus the draw's): ct beats every draw when no draw is at or above
      it, `draws_at_or_above` 0, a count the item makes again from `gain_kwh` (one gain per
      draw); a record without them (before they were written) cannot be told, and counts that
      differ from the record's own cannot decide. [Alternative: the 5th percentile over draws
      above 0, met by any gain above the draws' interpolated 95th percentile: by rank, a chance of
      up to (n - j)/(n + 1) for a sizing that carries nothing, j = floor(0.95 (n - 1)), which is
      above 5% for every n: 18% at 10 draws, 10% at 19.]
  W4c. Against what: ct must also offer more than the constant multiplier (`constant`, the offer
      without any sizing: "raises the development offer" read literally) and more than the
      day-size net it starts from (`net-<START_LEVEL>-<arm>`; `size_ct`, reading C9: conformal
      training fine-tunes that net, so a gain the net already has is not conformal training's),
      each by its mean offer per counting test day on the same splits. [Alternative: the
      constant alone, or the shuffled control alone; a gain over the constant alone proves
      nothing on the gate's grid (`size_resolve_summary`, definition "shuffled_control").]
  W4d. A unit and tier is not a test when it is not split or when nothing offers there (ct, its
      control and the constant all at 0): there is no offer to raise. On a test, ct raises when
      its gains over the constant and over the net are above 0 and no draw is at or above it
      (W4b); otherwise it does not raise. An arm is not kept when it does not raise on some test;
      kept when it raises on every test; otherwise (a record without gains by draw), or with no
      test, it cannot be decided. [Alternative: kept per unit and tier where it raises, which
      would share the chance over the units and tiers: up to k times 1/(n+1) for k of them.]
  W4e. The clause is about conformal training on the model the WP6 gate is read with, lgbm_tail,
      whose training-unit rotation `SIZING_MODEL` the comparison takes (as its label says), at
      `SIZING_DIAL` 0.02, taken as lgbm_tail's tier A dial at every site type: a run whose
      recorded command or results name another model or dial cannot decide. [Alternative: the
      dial read per site type from lgbm_tail's recorded dial-choice run, as
      `tools/compare_shared_check.py` reads it (reading P6 there); the command takes one dial
      for all units, so a run could match only where the site types' dials agree.]
  W4f. "Finished" sizings are those fitted on the labelled days of every training unit
      (`truth_build.TRAIN_UNITS`): the run's `units`, the training units it labels, must be all
      of them, else its development units' sizings are fitted on fewer and it cannot decide.
      [Alternative: any run whose development sizings are fitted on training units only, however
      many of them.]
  Exit: 0 every arm decided; 2 otherwise. Never 1.

An unexpected error is could not run (2), never 1 (Python's own exit status for an uncaught
error, which `passmark.sh` reads as failed), also when the tool's own imports fail.

The tool runs without the learn packages, so the names it reads from `soft_scale_check`,
`size_resolve_whatif`, `size_resolve_summary`, `size_ct`, `size_net` and `size_scale` are written
out here and pinned to their sources by `tests/learn/test_wp9_check_names.py`. The spread item
imports the WP10 item's readers (`tools/compare_shared_check.py`, which need no learn package)
only when it runs, so the other items do not depend on them.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    from fleetreserve.crc import ALPHA_SLOT
    from fleetreserve.gate_run import DAY_COLUMNS, DAYS_NAME
    from fleetreserve.gate_run import RESULTS_NAME as GATE_RESULTS
    from fleetreserve.run_evidence import CannotCheck, head_commit, newest_run, read_evidence
    from fleetreserve.run_record import _LABEL as RECORDER_LABEL
    from fleetreserve.run_record import RUNS_FOLDER, sha256_file
    from fleetreserve.run_verify import NotVerified, require_verified
    from fleetreserve.rungs import RUNGS
    from fleetreserve.truth import MAIN
    from fleetreserve.truth_build import DEVELOPMENT_UNITS, TRAIN_UNITS
    from fleetreserve.truth_build import SUMMARY_NAME as TRUTH_SUMMARY
    from fleetreserve.truth_verify import RUN_LABEL as TRUTH_LABEL
except Exception as problem:  # the check cannot start: could not run, never Python's exit 1
    print(f"COULD NOT RUN: the check cannot start: {type(problem).__name__}: {problem}")
    sys.exit(2)

ITEMS = ("abstain", "spread", "t8", "conformal")
PASSED, FAILED, COULD_NOT_RUN = 0, 1, 2
MATRIX = "gate/matrix.json"
SHOWN = 3  # the failures named in a line
TIERS = ("A", "B")  # size_scale.TIERS; the gate's tiers have the same names

GATE_LABEL = "wp6-gate"
GATE_MODULE = "fleetreserve.gate_run"

SOFT_MODULE = "fleetreserve.soft_scale_check"
SOFT_RESULTS = "soft-scale-check.json"
KINDS = ("soft", "exact")
DECIDING = ("departure", "cap", "need")
DAYS_PER_UNIT = 100  # the command's default days per unit
T8_DAYS = 100  # plan section 7, T8: "on 100 days"
T8_LIST = "passmarks/wp9-t8-runs.json"  # the runs the item combines (reading W3e)
T8_PREFIX = "wp9-soft-scale-check-"
T8_FIELDS = ("label", "model", "configuration", "matrix")
T8_DAYS_FIELD = "days_per_unit"  # an entry's optional days per unit (reading W3e)


@dataclass(frozen=True)
class T8Run:
    """One recorded run of T8's check that the item requires (an entry of `T8_LIST`)."""

    label: str
    model: str
    configuration: str
    matrix: str
    days_per_unit: int = DAYS_PER_UNIT


T8_LABEL = f"{T8_PREFIX}<letters, digits, '-' or '_'>"
STRENGTH = ("failed", "resolvable", "passed", "kink", "unresolved", "tied")  # reading W3a
PASSES = ("resolvable", "passed")  # passed at a resolvable step or at one that is not (W3b)
T8_PLAN_READING = (
    f"  T8 asks {T8_DAYS} distinct days with a passed comparison per direction (plan section 7; "
    "the owner's decision of 18 November 2026): fewer days with a passed comparison in a "
    "direction do not satisfy it, and kink, unresolved and tied comparisons count toward none "
    f"of the {T8_DAYS} (reading W3b)"
)

SIZING_LABEL = "wp9-size-resolve-whatif-ct-lgbm_tail_rotation"
SIZING_MODULE = "fleetreserve.size_resolve_whatif"
SIZING_RESULTS = "size-resolve-whatif.json"
CONTROL = "shuffled"
CT = "ct"
NET = "net"
ARMS = ("scale", "abstain")
START_LEVEL = ALPHA_SLOT  # the net's level conformal training starts from
P5 = "p5_kwh"
BY_DRAW = "by_draw"  # in difference_from_shuffled: the gains by draw (reading W4b)
GAINS, SMALLEST, AT_OR_ABOVE = "gain_kwh", "smallest_gain_kwh", "draws_at_or_above"
SIZING_MODEL = "lgbm_tail_rotation"  # reading W4e
SIZING_DIAL = 0.02  # reading W4e
SPLITS = 200  # plan section 6: the 200 seeded development splits
DRAWS_NEEDED = 19  # the fewest draws n with 1/(n+1) at most 5% (reading W4b)
DEVELOPMENT = "development_units"  # in the run's settings: the development units scored (W4a)

SPREAD_VARIANT = "forecast"  # reading W2d
SPREAD_DECIDING = "fixed-plan"  # section 6: "the expected spread (fixed plan)" (reading W2c)
SPREAD_SHOWN = "re-solve"  # shown only (reading W2c)
SPREAD_METHODS = (SPREAD_DECIDING, SPREAD_SHOWN)
SPREAD_LOSSES = (("A", "slot"), ("A", "driver"), ("B", "slot"))  # (tier, loss) with a level
SPREAD_READ = "tier A slot and driver loss, tier B slot loss"


# --- shared --------------------------------------------------------------------------------------


def committed(top: Path, path: str) -> bytes:
    """The bytes of `path` as committed at the checked-out commit."""
    done = subprocess.run(
        ["git", "-C", str(top), "show", f"HEAD:{path}"], capture_output=True, check=False
    )
    if done.returncode != 0:
        raise CannotCheck(f"{path} is not committed at the checked-out commit")
    return done.stdout


def gate_units(data: bytes, path: str) -> list[str]:
    """The gate units of a committed matrix file."""
    try:
        units = json.loads(data)["gate_units"]
    except (ValueError, KeyError, TypeError) as problem:
        raise CannotCheck(f"{path} names no gate units ({problem!r})") from problem
    if not isinstance(units, list) or not all(isinstance(u, str) for u in units):
        raise CannotCheck(f"{path} names no gate units")
    return units


def runs_module(command, module: str) -> bool:
    """True for a recorded command that runs `python -m <module>`."""
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
        return False
    return any(command[k - 1 : k + 1] == ["-m", module] for k in range(1, len(command)))


def flag(command: list, name: str):
    """The value after `name` in a recorded command, or None."""
    for k in range(len(command) - 1):
        if command[k] == name:
            return command[k + 1]
    return None


def _at(document, *keys):
    """`document[keys[0]][keys[1]]...`, or None where a key is absent or a level is no mapping."""
    for key in keys:
        if not isinstance(document, dict) or key not in document:
            return None
        document = document[key]
    return document


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise CannotCheck(f"{where}: {value!r} is not a number")
    try:
        return float(value)
    except ValueError as problem:
        raise CannotCheck(f"{where}: {value!r} is not a number") from problem


def check_truth_build(top: Path, name, head: str) -> None:
    """The truth build `name` is a main build of `runs.noindex/`, verified, and recorded at
    `head` or read from an earlier commit by the rule of `run_evidence`."""
    pattern = r"\d{8}T\d{6}Z-" + re.escape(TRUTH_LABEL)
    if not isinstance(name, str) or not re.fullmatch(pattern, name):
        raise CannotCheck(
            f"the run names no main truth build of {RUNS_FOLDER}/ (inputs.truth_run {name!r})"
        )
    evidence = read_evidence(top / RUNS_FOLDER / name, TRUTH_SUMMARY, head)
    if not evidence.verified:
        raise CannotCheck(f"the truth build {name}{evidence.note} is not verified")


def check_run(top: Path, evidence, head: str, module: str, matrix: str, matrix_sha: str) -> None:
    """The run's command, matrix, truth setting and truth build (module text)."""
    name, results = noted_name(evidence), evidence.data
    if not runs_module(evidence.manifest.get("command"), module):
        raise CannotCheck(f"the recorded command of {name} does not run {module}")
    used = _at(results, "matrix", "sha256")
    if not isinstance(used, str):
        raise CannotCheck(f"the results of {name} record no SHA-256 of its matrix")
    if used != matrix_sha:
        raise CannotCheck(
            f"{name} ran on another matrix: SHA-256 {used[:12]}, the committed {matrix} has "
            f"{matrix_sha[:12]}"
        )
    setting = _at(results, "inputs", "setting")
    if setting != MAIN:
        raise CannotCheck(f"{name} read the truth setting {setting!r}, not {MAIN!r}")
    check_truth_build(top, _at(results, "inputs", "truth_run"), head)


def noted_name(evidence) -> str:
    """The run's folder name as a line names it: with `Evidence.note` after it when the run is
    read from an earlier commit (module text)."""
    return f"{evidence.run.name}{evidence.note}"


def check_file(evidence, name: str) -> Path:
    """`name` of the run, as the run wrote it: verified with a verified run, listed with its
    SHA-256 in the manifest with an intact one."""
    path = evidence.run / name
    if evidence.verified:
        try:
            require_verified(path)
        except NotVerified as problem:
            raise CannotCheck(
                f"{name} of {noted_name(evidence)} is not verified: {problem}"
            ) from None
        return path
    outputs = evidence.manifest.get("outputs")
    found = sha256_file(path) if path.is_file() else None
    if not isinstance(outputs, dict) or found is None or outputs.get(name) != found:
        raise CannotCheck(f"{name} of {noted_name(evidence)} is not the file the run wrote")
    return path


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# --- abstain -------------------------------------------------------------------------------------


@dataclass
class TierCount:
    """The abstaining test entries of one tier (reading W1)."""

    at_zero: int = 0
    not_offering: int = 0
    pairs: frozenset = frozenset()
    wrong: tuple = ()

    @property
    def entries(self) -> int:
        return self.at_zero + self.not_offering

    def line(self, tier: str) -> str:
        units = len({unit for _, unit in self.pairs})
        configurations = len({c for c, _ in self.pairs})
        text = (
            f"  tier {tier}: {self.entries} abstaining test entr"
            f"{'y' if self.entries == 1 else 'ies'} ({self.at_zero} at scale 0, "
            f"{self.not_offering} on days that do not offer) in {_plural(units, 'unit', 'units')} "
            f"under {_plural(configurations, 'configuration', 'configurations')}, "
        )
        if not self.wrong:
            return text + "every one with offer 0, slot loss 0 and driver loss 0"
        more = len(self.wrong) - SHOWN
        named = "; ".join(self.wrong[:SHOWN]) + (f"; and {more} more" if more > 0 else "")
        return text + f"{len(self.wrong)} not 0: {named}"


def _scale(results: dict, configuration: str, unit: str, rung: str, tier: str):
    """The unit's scale for the rung and tier in `gate.json`, or None (not split, not scored)."""
    entry = _at(results, "configurations", configuration, "units", unit)
    if not isinstance(entry, dict) or "rungs" not in entry:
        return None
    keys = ("rungs", rung, "beta_hat") if tier == "A" else ("rungs", rung, "tier_b", "beta_hat")
    found = _at(entry, *keys)
    if found is None:
        return None
    return _number(found, f"{configuration} | {unit} | {rung} tier {tier} scale")


def abstaining_entries(results: dict, rows: list[dict]) -> tuple[dict, int]:
    """Per tier its `TierCount`, and the number of rows of units that are not split."""
    found = {tier: {"at_zero": 0, "not_offering": 0, "pairs": set(), "wrong": []} for tier in TIERS}
    not_split = 0
    for row in rows:
        if row["role"] == "not calibratable":
            not_split += 1
            continue
        if row["role"] != "test":
            continue
        configuration, unit = row["configuration"], row["unit"]
        for tier in TIERS:
            infix = "_tier_b" if tier == "B" else ""
            for rung in RUNGS:
                fields = [f"{rung}{infix}_{f}" for f in ("offer_kwh", "slot_loss", "driver_loss")]
                texts = [row[f] for f in fields]
                if all(t == "" for t in texts):
                    continue  # not scored in this tier
                where = f"{configuration} | {unit} | {row['date']} {rung} tier {tier}"
                values = [_number(t, where) for t in texts]
                scale = _scale(results, configuration, unit, rung, tier)
                if scale is None:
                    raise CannotCheck(f"{where}: a scored row with no scale in {GATE_RESULTS}")
                if row["abstain_reason"]:
                    found[tier]["not_offering"] += 1
                elif scale == 0:
                    found[tier]["at_zero"] += 1
                else:
                    continue
                found[tier]["pairs"].add((configuration, unit))
                if any(v != 0 for v in values):
                    offer, slot, driver = values
                    found[tier]["wrong"].append(
                        f"{configuration} | {unit} | {row['date']} {rung}: offer {offer:g}, slot "
                        f"loss {slot:g}, driver loss {driver:g}"
                    )
    counts = {
        tier: TierCount(f["at_zero"], f["not_offering"], frozenset(f["pairs"]), tuple(f["wrong"]))
        for tier, f in found.items()
    }
    return counts, not_split


def read_days(path: Path, name: str) -> list[dict]:
    """The rows of `gate-days.csv` (a first line starting with "#" is the trial's label); `name`
    is how a line names the run."""
    text = "".join(line for line in path.read_text().splitlines(True) if not line.startswith("#"))
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(DAY_COLUMNS):
        raise CannotCheck(f"{DAYS_NAME} of {name} has other columns than gate_run.DAY_COLUMNS")
    return list(reader)


def abstain(top: Path) -> int:
    print(
        "WP9, loss 0 at abstain: the abstaining test entries (scale 0, or a day that does not "
        "offer) of the recorded gate run, both tiers (reading W1)"
    )
    head = head_commit(top)
    matrix_sha = hashlib.sha256(committed(top, MATRIX)).hexdigest()
    print(
        f"gate run checked at commit {head[:12]} (a run accepted from an earlier commit says so on "
        f"its line), on the committed {MATRIX} ({matrix_sha[:12]})"
    )
    try:
        run = newest_run(top, GATE_LABEL)
    except CannotCheck as problem:
        print(f"NO RUN: {problem}")
        return COULD_NOT_RUN
    evidence = read_evidence(run, GATE_RESULTS, head)
    check_run(top, evidence, head, GATE_MODULE, MATRIX, matrix_sha)
    results = evidence.data
    if results.get("trial") is not False or results.get("partial") is not False:
        raise CannotCheck(
            f"{noted_name(evidence)} is a trial (--limit-days) or a partial (--only) run"
        )
    rows = read_days(check_file(evidence, DAYS_NAME), noted_name(evidence))
    counts, not_split = abstaining_entries(results, rows)
    for tier in TIERS:
        print(counts[tier].line(tier))
    print(
        f"  not read: {_plural(not_split, 'row', 'rows')} of units that are not split (no scale is "
        "chosen and no loss written)"
    )
    wrong = {tier: len(counts[tier].wrong) for tier in TIERS}
    if sum(wrong.values()):
        total = sum(wrong.values())
        print(
            f"FAILED: {total} abstaining test entr{'y' if total == 1 else 'ies'} with an offer or "
            f"a loss that is not 0 (tier A {wrong['A']}, tier B {wrong['B']}); {run.name}"
            f"{evidence.note}"
        )
        return FAILED
    if not evidence.verified:
        print(
            f"COULD NOT RUN: {run.name}{evidence.note} ended with exit code "
            f"{evidence.manifest.get('exit_code')} and shows no abstaining entry with an offer or "
            "a loss, so it cannot decide the item"
        )
        return COULD_NOT_RUN
    empty = [tier for tier in TIERS if counts[tier].entries == 0]
    if empty:
        print(
            f"COULD NOT RUN: no abstaining test entry in tier {' and tier '.join(empty)}, so loss "
            f"0 at abstain is not shown there; {run.name}{evidence.note}"
        )
        return COULD_NOT_RUN
    print(
        "passed: offer 0 and slot and driver loss exactly 0 on every abstaining test entry, "
        f"{counts['A'].entries} in tier A and {counts['B'].entries} in tier B; {run.name}"
        f"{evidence.note}"
    )
    return PASSED


# --- spread --------------------------------------------------------------------------------------


def _named(names: list) -> str:
    more = len(names) - SHOWN
    return ", ".join(names[:SHOWN]) + (f" and {more} more" if more > 0 else "")


def _wp10_readers():
    """The WP10 item's readers of the dial-choice and comparison runs (module text, spread),
    imported when the item runs: a failure to import them is could not run."""
    try:
        import compare_shared_check
    except Exception as problem:
        raise CannotCheck(
            "the readers of tools/compare_shared_check.py cannot be imported: "
            f"{type(problem).__name__}: {problem}"
        ) from problem
    return compare_shared_check


@dataclass(frozen=True)
class SpreadTest:
    """One unit, tier, loss and method of a comparison run: the losses of its splits that offer
    (reading W2e), the number of splits and the level."""

    unit: str
    tier: str
    loss: str
    method: str
    splits: int
    losses: tuple
    level: float

    @property
    def name(self) -> str:
        return f"{self.unit} tier {self.tier} {self.loss} loss"

    @property
    def figures(self) -> tuple:
        """(mean, standard error, level plus two standard errors), None where not defined."""
        n = len(self.losses)
        mean = statistics.fmean(self.losses) if n else None
        error = statistics.stdev(self.losses) / math.sqrt(n) if n > 1 else None
        return mean, error, None if error is None else self.level + 2.0 * error

    @property
    def outcome(self) -> str:
        mean, _, bound = self.figures
        if mean is None:
            return "not a test"
        if bound is None:
            return "no standard error"
        return "within" if mean <= bound else "above"

    def line(self) -> str:
        shown = "" if self.method == SPREAD_DECIDING else " (shown only, reading W2c)"
        text = f"  {self.name}, {self.method}{shown}: "
        mean, error, bound = self.figures
        if mean is None:
            return text + f"not a test (none of its {self.splits} splits offers)"
        n = len(self.losses)
        text += (
            f"mean {mean:.6f} over {_plural(n, 'split', 'splits')} with a loss "
            f"({self.splits - n} offer nothing), "
        )
        if error is None:
            return text + "standard error not defined (one split with a loss)"
        return text + (
            f"standard error {error:.6f}, level {self.level:g}, level plus two standard errors "
            f"{bound:.6f}: {self.outcome}"
        )


def _split_losses(rows: list, place: str, losses: list, problems: list) -> dict:
    """Loss -> the losses of the rows' splits that offer, in seed order (reading W2e); adds the
    rows' problems (reading W2f)."""
    found = {loss: [] for loss in losses}
    for row in sorted(rows, key=lambda r: int(r["seed"])):
        where = f"{place}, seed {row['seed']}"
        try:
            scale, offering = float(row["scale"]), int(row["offering_test_days"])
        except ValueError:
            problems.append(f"{where}: no number for its scale or its offering test days")
            continue
        offers = scale > 0 and offering > 0
        for loss in losses:
            text = row[f"mean_test_{loss}_loss"]
            if offers and text == "":
                problems.append(
                    f"{where}: the split offers (scale {scale:g}, {offering} offering test days) "
                    f"and its {loss} loss is missing"
                )
            elif not offers and text != "":
                problems.append(
                    f"{where}: the split offers nothing (scale {scale:g}, {offering} offering "
                    f"test days) and a {loss} loss is written"
                )
            elif offers:
                try:
                    value = float(text)
                except ValueError:
                    value = math.nan
                if not math.isfinite(value):
                    problems.append(f"{where}: its {loss} loss {text!r} is not a finite number")
                else:
                    found[loss].append(value)
    return found


def spread_tests(read, units: list, splits: int, columns, tier_alphas) -> tuple[list, list]:
    """The tests of `units` in the comparison run's split table (`SpreadTest`, every method of
    `SPREAD_METHODS`) and the table's problems (reading W2f)."""
    name = read.names["splits"]
    ceilings = read.document["run"]["ceilings"]
    header = list(columns(ceilings))
    reader = csv.reader(io.StringIO((read.run / name).read_text()))
    if next(reader, None) != header:
        return [], [f"{name} has not the header of compare_splits.columns for {ceilings}"]
    problems, groups = [], defaultdict(list)
    for number, cells in enumerate(reader, start=2):
        if len(cells) != len(header):
            problems.append(f"line {number} of {name} has {len(cells)} cells, not {len(header)}")
            continue
        row = dict(zip(header, cells, strict=True))
        if row["variant"] == SPREAD_VARIANT and row["method"] in SPREAD_METHODS:
            groups[(row["unit"], row["tier"], row["method"])].append(row)
    seeds = Counter(str(seed) for seed in range(1, splits + 1))
    tests = []
    for unit in units:
        for tier in TIERS:
            losses = [loss for t, loss in SPREAD_LOSSES if t == tier]
            levels = dict(zip(("slot", "driver"), tier_alphas(tier), strict=True))
            for method in SPREAD_METHODS:
                place = f"{unit} tier {tier}, {method}"
                rows = groups.get((unit, tier, method), [])
                if Counter(row["seed"] for row in rows) != seeds:
                    problems.append(
                        f"{place}: {len(rows)} splits in the split table, not the seeds 1 to "
                        f"{splits}"
                    )
                    continue
                found = _split_losses(rows, place, losses, problems)
                for loss in losses:
                    tests.append(
                        SpreadTest(
                            unit, tier, loss, method, splits, tuple(found[loss]), levels[loss]
                        )
                    )
    order = {key: k for k, key in enumerate(SPREAD_LOSSES)}
    tests.sort(
        key=lambda t: (units.index(t.unit), order[(t.tier, t.loss)], SPREAD_METHODS.index(t.method))
    )
    return tests, problems


def spread(top: Path) -> int:
    print(
        "WP9, development loss within its 200-split spread: per gate unit, the mean test loss over "
        f"the {SPLITS} development splits at most the level plus two standard errors, "
        f"{SPREAD_READ} (readings W2c to W2f)"
    )
    shared = _wp10_readers()
    from fleetreserve.compare_splits import columns
    from fleetreserve.dial_splits import tier_alphas

    head = head_commit(top)
    data = committed(top, MATRIX)
    matrix_sha = hashlib.sha256(data).hexdigest()
    units = gate_units(data, MATRIX)
    print(
        f"comparison runs checked at commit {head[:12]} (a run accepted from an earlier commit "
        f"says so on its line), on the committed {MATRIX} ({matrix_sha[:12]})"
    )
    outside = [unit for unit in units if unit not in shared.DIAL_SOURCES]
    if outside:
        raise CannotCheck(
            f"the gate units {', '.join(outside)} have no site type in "
            "compare_shared_check.DIAL_SOURCES, so their dial is not known"
        )
    ctx = shared.Context(top, head, matrix_sha, tuple(units), shared.Setup(splits=SPLITS), ())
    model = shared.GATE_MODEL
    try:
        dial_run, dials, dial_note = shared.read_dials(ctx, model)
    except shared.NoRun as problem:
        print(f"NO RUN: dials of {model}: {problem}")
        return COULD_NOT_RUN
    except CannotCheck as problem:
        print(f"COULD NOT RUN: dials of {model}: {problem}")
        return COULD_NOT_RUN
    print(
        f"  dials of {model}: {dial_run.name}{dial_note}: "
        f"{', '.join(f'{unit} {dials[unit]:g}' for unit in units)} (tier A, reading W2d)"
    )
    by_dial = defaultdict(list)
    for unit in units:
        by_dial[dials[unit]].append(unit)
    tests, reasons, read_runs = [], [], []
    for dial, its_units in by_dial.items():
        where = f"{model} at dial {dial:g}"
        try:
            read = shared.read_comparison(ctx, model, dial, None)
        except shared.NoRun as problem:
            print(f"  NO RUN: {where}: {problem}")
            reasons.append(f"NO RUN: {where}: {problem}")
            continue
        except CannotCheck as problem:
            print(f"  COULD NOT RUN: {where}: {problem}")
            reasons.append(f"{where}: {problem}")
            continue
        found, problems = [], list(read.problems)
        if not problems:
            found, problems = spread_tests(read, its_units, SPLITS, columns, tier_alphas)
        if problems:
            more = len(problems) - SHOWN
            listed = "; ".join(problems[:SHOWN]) + (f"; and {more} more" if more > 0 else "")
            print(f"  COULD NOT RUN: {where}: {read.run.name}{read.note}: {listed}")
            reasons.append(f"{where}: {read.run.name}{read.note}: {listed}")
            continue
        print(f"  {where}: {read.run.name}{read.note}, read for {', '.join(its_units)}")
        read_runs.append(f"{read.run.name}{read.note}")
        for test in found:
            print(test.line())
        tests.extend(found)
    return _spread_decide(tests, reasons, ", ".join(read_runs))


def _spread_decide(tests: list, reasons: list, runs: str) -> int:
    deciding = [test for test in tests if test.method == SPREAD_DECIDING]
    decided = [test for test in deciding if test.outcome in ("within", "above")]
    counted = _plural(len(decided), "unit, tier and loss", "units, tiers and losses")
    above = [test.name for test in deciding if test.outcome == "above"]
    if above:
        print(
            "FAILED: the mean test loss over the development splits is above the level plus two "
            f"standard errors on {len(above)} of {counted} with a loss ({_named(above)}; "
            f"{SPREAD_DECIDING}); {runs}"
        )
        return FAILED
    every_run_read = not reasons
    lone = [test.name for test in deciding if test.outcome == "no standard error"]
    if lone:
        reasons.append(
            "no standard error on "
            f"{_plural(len(lone), 'unit, tier and loss', 'units, tiers and losses')} with one "
            f"split with a loss ({_named(lone)}; {SPREAD_DECIDING})"
        )
    if every_run_read:  # each tier and loss needs a unit with a loss (reading W2f)
        missing = [
            f"tier {tier} {loss} loss"
            for tier, loss in SPREAD_LOSSES
            if not any(t.tier == tier and t.loss == loss and t.losses for t in deciding)
        ]
        if missing:
            reasons.append(
                f"no unit with a loss in {' or in '.join(missing)}, so the clause is not tested "
                "there"
            )
    if reasons:
        print(f"COULD NOT RUN: {'; '.join(reasons)}")
        return COULD_NOT_RUN
    not_tests = sum(test.outcome == "not a test" for test in deciding)
    print(
        "passed: the mean test loss over the development splits is at most the level plus two "
        f"standard errors on {len(decided)} of {counted} with a loss ({SPREAD_DECIDING}; "
        f"{SPREAD_READ}), {not_tests} not a test; {runs}"
    )
    return PASSED


# --- t8 ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DayCheck:
    """One checked day of one run: per deciding direction (kind, field) its outcome (W3a)."""

    run: str
    unit: str
    day: tuple  # (site, date)
    outcomes: dict


@dataclass(frozen=True)
class T8Read:
    """A listed run that can be read: its checks and what it records beside them."""

    evidence: object
    deciding: tuple
    checks: tuple
    failures: tuple
    mps: object
    tracked: tuple  # development units with tracking on at least one day

    def line(self) -> str:
        units = Counter(c.unit for c in self.checks)
        data = self.evidence.data
        return (
            f"  {self.evidence.run.name}{self.evidence.note}: model {data.get('model')}, "
            f"configuration {data.get('configuration')}, deciding {', '.join(self.deciding)}; "
            f"{len(self.checks)} days checked ({', '.join(f'{u} {n}' for u, n in units.items())})"
        )


def _outcome(comparison: dict) -> str:
    outcome = comparison["outcome"]
    if outcome == "passed" and comparison.get("resolvable"):
        return "resolvable"
    if outcome not in STRENGTH:
        raise CannotCheck(f"a comparison has the outcome {outcome!r}")
    return outcome


def read_t8_list(data: bytes) -> tuple[T8Run, ...]:
    """The runs listed in the bytes of `T8_LIST` (reading W3e); raises `CannotCheck`."""
    try:
        entries = json.loads(data)["runs"]
    except (ValueError, KeyError, TypeError) as problem:
        raise CannotCheck(f"{T8_LIST} lists no runs ({problem!r})") from problem
    if not isinstance(entries, list) or not entries:
        raise CannotCheck(f"{T8_LIST} lists no runs")
    runs = []
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not set(T8_FIELDS) <= set(entry) <= {*T8_FIELDS, T8_DAYS_FIELD}
            or not all(isinstance(entry[f], str) and entry[f] for f in T8_FIELDS)
        ):
            raise CannotCheck(
                f"{T8_LIST}: an entry is not a label, model, configuration and matrix (and, "
                f"optionally, its {T8_DAYS_FIELD}): {entry!r}"
            )
        label = entry["label"]
        named = label.startswith(T8_PREFIX) and label != T8_PREFIX
        if not (named and RECORDER_LABEL.fullmatch(label)):
            raise CannotCheck(f"{T8_LIST}: {label!r} is not a run label of the form {T8_LABEL}")
        days = entry.get(T8_DAYS_FIELD, DAYS_PER_UNIT)
        if isinstance(days, bool) or not isinstance(days, int) or days < DAYS_PER_UNIT:
            raise CannotCheck(
                f"{T8_LIST}: {label}'s {T8_DAYS_FIELD} {days!r} is not a whole number of at least "
                f"{DAYS_PER_UNIT}"
            )
        runs.append(T8Run(**{f: entry[f] for f in T8_FIELDS}, days_per_unit=days))
    labels = [run.label for run in runs]
    twice = sorted({label for label in labels if labels.count(label) > 1})
    if twice:
        raise CannotCheck(f"{T8_LIST} lists {', '.join(twice)} more than once")
    return tuple(runs)


def read_t8(top: Path, spec: T8Run, head: str, shas: dict) -> T8Read:
    """A listed run of T8's check, once it is shown to decide (module text); raises
    `CannotCheck` otherwise."""
    run = newest_run(top, spec.label)
    evidence = read_evidence(run, SOFT_RESULTS, head, also_compare=(T8_LIST,))  # reading W3f
    if spec.matrix not in shas:
        data = committed(top, spec.matrix)
        shas[spec.matrix] = (hashlib.sha256(data).hexdigest(), gate_units(data, spec.matrix))
    matrix_sha, units = shas[spec.matrix]
    results, name = evidence.data, noted_name(evidence)
    command = evidence.manifest.get("command")
    check_run(top, evidence, head, SOFT_MODULE, spec.matrix, matrix_sha)
    if results.get("model") != spec.model or flag(command, "--model") != spec.model:
        raise CannotCheck(f"{name} is of the model {results.get('model')!r}, not {spec.model!r}")
    configuration = results.get("configuration")
    if configuration != spec.configuration or flag(command, "--configuration") != configuration:
        raise CannotCheck(
            f"{name} checked the configuration {configuration!r}, not {spec.configuration!r}"
        )
    try:
        return _t8_read(evidence, units, spec.matrix, spec.days_per_unit)
    except (KeyError, TypeError, AttributeError) as problem:
        raise CannotCheck(f"{name}: {SOFT_RESULTS} is malformed ({problem!r})") from problem


def _t8_read(evidence, gate: list, matrix: str, days: int) -> T8Read:
    run, results, name = evidence.run, evidence.data, noted_name(evidence)
    settings = results["settings"]
    found = settings["days_per_unit"]
    if found != days:
        if isinstance(found, int | float) and not isinstance(found, bool) and found < days:
            raise CannotCheck(f"{name} is a trial: {found} days per unit, not {days}")
        raise CannotCheck(f"{name} checked {found!r} days per unit, not the {days} its entry lists")
    checked = list(results["units"])
    if set(checked) != set(gate):
        raise CannotCheck(
            f"{name} is a partial run: it checked {', '.join(checked)}, not every gate unit "
            f"of {matrix} ({', '.join(gate)})"
        )
    deciding = tuple(settings["deciding_fields"])
    if not set(deciding) <= set(DECIDING):
        raise CannotCheck(f"{name} decides in the fields {deciding}, not among {DECIDING}")
    checks, failures, resolved, tracked = [], [], set(), []
    for unit in checked:
        entry = results["units"][unit]
        for day in entry["days"]:
            outcomes = {}
            for kind in KINDS:
                for field in deciding:
                    comparison = day["comparisons"][kind][field]
                    outcomes[(kind, field)] = _outcome(comparison)
                    if comparison.get("resolvable"):  # as the command counts it, failed included
                        resolved.add((kind, field))
                    if outcomes[(kind, field)] == "failed":
                        failures.append(
                            f"{name} | {unit} | {day['site']} {day['date']} | {kind} {field}"
                        )
            checks.append(DayCheck(run.name, unit, (day["site"], day["date"]), outcomes))
        if unit in DEVELOPMENT_UNITS and (entry["tracking"] or {}).get("days", 0) >= 1:
            tracked.append(unit)
    missing = [
        f"{k} {f}" for k in KINDS for f in DECIDING if f in deciding and (k, f) not in resolved
    ]
    if (
        len(failures) != results["failed_comparisons"]
        or missing != list(results["not_exercised"])
        or len(checks) != results["checked_days"]
    ):
        raise CannotCheck(
            f"{name}: the counts do not add up (failed comparisons {len(failures)} against "
            f"{results['failed_comparisons']}, not exercised {missing} against "
            f"{results['not_exercised']}, days {len(checks)} against {results['checked_days']})"
        )
    mps = _at(results, "batch_timing", "mps")
    return T8Read(evidence, deciding, tuple(checks), tuple(failures), mps, tuple(tracked))


def _seconds(value) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value >= 0


def combine_days(reads: list[T8Read]) -> dict:
    """Per direction (kind, field): day -> (outcome, unit), the strongest outcome over the runs
    where the field is deciding and the unit of its first check (reading W3a)."""
    combined = {(k, f): {} for k in KINDS for f in DECIDING}
    for read in reads:
        for check in read.checks:
            for direction, outcome in check.outcomes.items():
                held = combined[direction].get(check.day)
                if held is None or STRENGTH.index(outcome) < STRENGTH.index(held[0]):
                    combined[direction][check.day] = (outcome, check.unit)
    return combined


def _by_unit(found: dict, outcomes: tuple, order: list) -> str:
    counts = Counter(unit for o, unit in found.values() if o in outcomes)
    text = ", ".join(f"{u} {counts[u]}" for u in order if counts[u])
    return text or "none"


def passed_days(found: dict) -> int:
    """The days of one direction with a passed comparison (reading W3b)."""
    return sum(o in PASSES for o, _ in found.values())


def direction_line(direction: tuple, found: dict, days: int, order: list) -> str:
    """One direction's days by outcome; of the `days` checked, those where it is deciding in no
    run that checked them are not applicable (reading W3c)."""
    counts = Counter(o for o, _ in found.values())
    kind, field = direction
    return (
        f"  {kind} {field}: {_plural(passed_days(found), 'day', 'days')} passed "
        f"({counts['resolvable']} at a resolvable step; {_by_unit(found, PASSES, order)}), "
        f"{counts['kink']} kink, {counts['unresolved']} unresolved "
        f"({_by_unit(found, ('unresolved',), order)}), {counts['tied']} tied, {counts['failed']} "
        f"failed, {days - len(found)} not applicable"
    )


def t8(top: Path) -> int:
    print(
        "WP9, T8's check: the recorded soft-scale checks, combined over distinct days (site, date) "
        "(readings W3a to W3e)"
    )
    head = head_commit(top)
    listed = read_t8_list(committed(top, T8_LIST))
    print(
        f"soft-scale checks read at commit {head[:12]} (a run accepted from an earlier commit says "
        f"so on its line): {', '.join(s.label for s in listed)}"
    )
    reads, missing, problems, shas = [], [], [], {}
    for spec in listed:
        try:
            reads.append(read_t8(top, spec, head, shas))
        except CannotCheck as problem:
            text = str(problem)
            (missing if text.startswith("no run named") else problems).append(text)
    for read in reads:
        print(read.line())
    failures = [f for read in reads for f in read.failures]
    if failures:
        more = len(failures) - SHOWN
        print(
            f"FAILED: {_plural(len(failures), 'deciding comparison', 'deciding comparisons')} "
            f"failed: {'; '.join(failures[:SHOWN])}" + (f"; and {more} more" if more > 0 else "")
        )
        return FAILED
    for read in reads:
        if not read.evidence.verified:
            exercised = ", ".join(read.evidence.data.get("not_exercised") or []) or "none"
            problems.append(
                f"{read.evidence.run.name}{read.evidence.note} ended with exit code "
                f"{read.evidence.manifest.get('exit_code')} and shows no failed comparison (not "
                f"exercised: {exercised}), so it cannot decide the item"
            )
    if missing or problems:
        word = "NO RUN" if not problems else "COULD NOT RUN"
        print(f"{word}: {'; '.join(missing + problems)}")
        return COULD_NOT_RUN
    return _t8_decide(reads)


def _t8_decide(reads: list[T8Read]) -> int:
    combined = combine_days(reads)
    order = list(dict.fromkeys(c.unit for read in reads for c in read.checks))
    days = {c.day for read in reads for c in read.checks}
    checks = sum(len(read.checks) for read in reads)
    print(
        f"  days checked: {len(days)} distinct days in {checks} checks (a day checked in more than "
        "one run counts once)"
    )
    for direction, found in combined.items():
        print(direction_line(direction, found, len(days), order))
    short = [
        f"{k} {f} {passed_days(found)}" + ("" if found else " (deciding in no run)")
        for (k, f), found in combined.items()
        if passed_days(found) < T8_DAYS
    ]
    print(
        f"  a passed comparison on at least {T8_DAYS} distinct days in every direction (reading "
        "W3b): " + (f"not met ({', '.join(short)})" if short else "met")
    )
    if short:
        print(T8_PLAN_READING)
    timed = [r for r in reads if isinstance(r.mps, dict) and _seconds(r.mps.get("seconds"))]
    for read in timed[:1]:
        print(
            f"  MPS batch timed: {read.evidence.run.name}{read.evidence.note} "
            f"({read.mps['seconds']:g} s for "
            f"{read.mps.get('batch_days')} days)"
        )
    tracked = list(dict.fromkeys(u for r in reads for u in r.tracked))
    if tracked:
        print(f"  tracking logged on development units: {', '.join(tracked)}")
    problems = []
    if short:
        problems.append(f"fewer than {T8_DAYS} days with a passed comparison in {', '.join(short)}")
    if not timed:
        problems.append("no run timed an MPS batch (MPS not available)")
    if not tracked:
        problems.append("no run logged the tracking of the certified value on a development unit")
    if problems:
        print(f"COULD NOT RUN: {'; '.join(problems)}")
        return COULD_NOT_RUN
    print(
        f"passed: T8's check: a passed comparison on at least {T8_DAYS} distinct days in each of "
        f"the {len(combined)} directions (soft and exact; departure, cap and need), no deciding "
        "comparison failed, an MPS batch timed and the tracking logged on development units; "
        + ", ".join(f"{read.evidence.run.name}{read.evidence.note}" for read in reads)
    )
    return PASSED


# --- conformal -----------------------------------------------------------------------------------

NOT_A_TEST, LOWER, UNTOLD, RAISES = "not a test", "does not raise", "cannot tell", "raises"


def at_or_above(control: dict, draws: int, where: str):
    """(the draws at or above ct, its smallest gain) from `by_draw` (reading W4b), or None when
    the record holds no gains by draw; raises `CannotCheck` when they do not add up."""
    found = control.get(BY_DRAW)
    if not isinstance(found, dict) or found.get(GAINS) is None:
        return None
    gains = [_number(g, where) for g in found[GAINS]]
    if len(gains) != draws:
        raise CannotCheck(f"{where}: {len(gains)} gains by draw, the run {draws} draws")
    counted, recorded = sum(g <= 0 for g in gains), found.get(AT_OR_ABOVE)
    if counted != recorded or _number(found.get(SMALLEST), where) != min(gains):
        raise CannotCheck(
            f"{where}: the gains by draw do not add up ({counted} at or above it against "
            f"{recorded!r}, smallest {min(gains)!r} against {found.get(SMALLEST)!r})"
        )
    return counted, min(gains)


def unit_and_tier(entry: dict, unit: str, tier: str, arm: str, draws: int, run: str):
    """(outcome, text) of one arm on one unit and tier (readings W4b to W4d)."""
    figures = entry["tiers"][tier]
    net_name = f"{NET}-{START_LEVEL:g}-{arm}"
    ct, constant, net = figures[f"{CT}-{arm}"], figures["constant"], figures[net_name]
    if ct["splits"] == 0:
        return NOT_A_TEST, "not a test (not split)"
    control = ct["difference_from_shuffled"]
    if control["draws"] != draws:
        raise CannotCheck(
            f"{run}, {unit} tier {tier}, {CT}-{arm}: its shuffled control has {control['draws']} "
            f"draws, the run {draws}"
        )
    where = f"{run}, {unit} tier {tier}, {CT}-{arm}"
    offer = _number(ct["mean_offer_kwh_per_counting_test_day"], where)
    base = _number(constant["mean_offer_kwh_per_counting_test_day"], where)
    shuffled = _number(control["control_mean_offer_kwh_per_counting_test_day"], where)
    if offer == 0 and base == 0 and shuffled == 0:
        return NOT_A_TEST, "not a test (nothing offers)"
    over_constant = _number(ct["difference_from_constant"]["mean_kwh"], where)
    over_net = offer - _number(net["mean_offer_kwh_per_counting_test_day"], where)
    over_control = _number(control["mean_kwh"], where)
    low = _number(control["over_draws"][P5], where)
    beaten = at_or_above(control, draws, where)
    reasons = []
    if over_constant <= 0:
        reasons.append("not above the constant")
    if over_net <= 0:
        reasons.append(f"not above {net_name}")
    text = (
        f"offer {offer:.3f} kWh per counting test day; over constant {over_constant:+.3f}, over "
        f"{net_name} {over_net:+.3f}, over its shuffled control: mean {over_control:+.3f}, p5 over "
        f"{draws} draws {low:+.3f}"
    )
    if beaten is None and low <= 0:  # the 5th percentile is never below the smallest gain
        reasons.append("some draw not beaten")
    if beaten is not None:
        above, smallest = beaten
        text += f", smallest gain {smallest:+.3f}, {above} of {draws} draws at or above it"
        if above:
            reasons.append(f"{above} of {draws} draws at or above it")
    if reasons:
        return LOWER, text + f": does not raise ({', '.join(reasons)})"
    if beaten is None:
        return UNTOLD, text + ": cannot tell (the record holds no gains by draw)"
    return RAISES, text + ": raises"


def _tag(unit: str, deciding: list) -> str:
    if unit in deciding:
        return ""
    return " (training unit, shown only)" if unit in TRAIN_UNITS else " (shown only)"


def conformal(top: Path) -> int:
    print(
        "WP9, conformal training kept only if it raises the development offer: a decision per arm "
        "from the recorded sizing comparison, per development unit and tier (readings W4a to W4f)"
    )
    head = head_commit(top)
    data = committed(top, MATRIX)
    matrix_sha, development = hashlib.sha256(data).hexdigest(), gate_units(data, MATRIX)
    try:
        run = newest_run(top, SIZING_LABEL)
    except CannotCheck as problem:
        print(f"NO RUN: {problem}")
        return COULD_NOT_RUN
    evidence = read_evidence(run, SIZING_RESULTS, head)
    results, name = evidence.data, noted_name(evidence)
    if not evidence.verified:
        failed = [*(results.get("failed_units") or []), *(results.get("failed_controls") or [])]
        raise CannotCheck(
            f"{name} ended with exit code {evidence.manifest.get('exit_code')} (failed: "
            f"{'; '.join(map(str, failed)) or 'none recorded'}), so it cannot decide the item"
        )
    check_run(top, evidence, head, SIZING_MODULE, MATRIX, matrix_sha)
    check_model_and_dial(name, evidence.manifest.get("command"), results.get("run"))
    try:
        return _conformal_decide(results, name, head, development)
    except (KeyError, TypeError, AttributeError) as problem:
        raise CannotCheck(f"{name}: {SIZING_RESULTS} is malformed ({problem!r})") from problem


def check_model_and_dial(name: str, command: list, settings) -> None:
    """The run is of `SIZING_MODEL` at `SIZING_DIAL`, by its command and its results (W4e)."""
    settings = settings if isinstance(settings, dict) else {}
    models = {flag(command, "--model"), settings.get("model")}
    if models != {SIZING_MODEL}:
        found = ", ".join(sorted(map(repr, models)))
        raise CannotCheck(f"{name} is of the model {found}, not {SIZING_MODEL!r} (reading W4e)")
    dials = [flag(command, "--dial"), settings.get("dial")]
    try:
        same = all(float(dial) == SIZING_DIAL for dial in dials)
    except (TypeError, ValueError):
        same = False
    if not same:
        raise CannotCheck(
            f"{name} is at the dial {dials[0]!r} (results {dials[1]!r}), not {SIZING_DIAL:g} "
            "(reading W4e)"
        )


def _conformal_decide(results: dict, name: str, head: str, development: list) -> int:
    """The item's lines and exit status; `name` is how a line names the run (`noted_name`)."""
    settings = results["run"]
    splits, draws, units = settings["splits"], settings["control_draws"], list(settings["units"])
    scored = list(settings.get(DEVELOPMENT) or [])
    deciding = [u for u in scored if u in development]
    if splits != SPLITS:
        raise CannotCheck(f"{name} is a trial: {splits} splits, not the plan's {SPLITS}")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise CannotCheck(f"{name} records no number of control draws ({draws!r})")
    print(
        f"sizing comparison {name}, checked at commit {head[:12]}: model "
        f"{settings.get('model')}, dial {settings.get('dial')}, {splits} splits, {draws} control "
        f"draws, on the committed {MATRIX}"
    )
    print(
        "  rule (readings W4b, W4c): kept when, on every development unit and tier where an offer "
        f"is made, ct beats the constant, the net it starts from and every one of its {draws} "
        "shuffled draws (no draw at or above it in the gains by draw); a sizing whose sizes carry "
        "nothing is kept with a chance of at most "
        f"1/{draws + 1} = {100 / (draws + 1):.1f}%"
    )
    outcomes = {arm: [] for arm in ARMS}
    for unit in [*units, *scored]:
        if unit not in results["units"]:
            raise CannotCheck(f"{name} lists the unit {unit} but has no figures for it")
        for tier in TIERS:
            for arm in ARMS:
                outcome, text = unit_and_tier(results["units"][unit], unit, tier, arm, draws, name)
                print(f"  {unit} tier {tier}, {CT}-{arm}{_tag(unit, deciding)}: {text}")
                if unit in deciding:
                    outcomes[arm].append((unit, tier, outcome))
    reasons = []
    if draws < DRAWS_NEEDED:
        reasons.append(
            f"too few control draws to decide: {name} has {draws}, at least {DRAWS_NEEDED} needed "
            "(a sizing whose sizes carry nothing is kept with a chance of at most 1/(draws + 1), "
            f"5% at {DRAWS_NEEDED})"
        )
    if set(units) != set(TRAIN_UNITS):
        reasons.append(
            f"{name} labels the training units {', '.join(units)}, not every one "
            f"({', '.join(TRAIN_UNITS)}; reading W4f), so the development units' sizings are not "
            "the finished ones fitted on every training unit"
        )
    measured = [u for u in development if u in deciding]
    if not measured:
        reasons.append(
            f"the development offer is not measured: {name} scores "
            f"{', '.join(scored) or 'no development unit'}, none of the gate's "
            f"({', '.join(development)}; reading W4a); the comparison scores them with "
            "--development-units"
        )
    elif len(measured) < len(development):
        reasons.append(
            f"{name} is a partial run: it scores the development units {', '.join(measured)}, not "
            f"every one of the gate ({', '.join(development)}; reading W4a)"
        )
    if reasons:
        print(f"COULD NOT RUN: {'; '.join(reasons)}")
        return COULD_NOT_RUN
    verdicts, untold, empty = {}, [], []
    for arm in ARMS:
        tests = [(u, t, o) for u, t, o in outcomes[arm] if o != NOT_A_TEST]
        others = len(outcomes[arm]) - len(tests)
        lower = [f"{u} tier {t}" for u, t, o in tests if o == LOWER]
        raised = [o for _, _, o in tests if o == RAISES]
        if tests and len(raised) == len(tests):
            verdicts[arm] = "kept"
            print(
                f"  {CT}-{arm}: kept: raises the offer on {len(tests)} of {len(tests)} units and "
                f"tiers with an offer; {others} not a test"
            )
        elif lower:
            verdicts[arm] = "not kept"
            print(
                f"  {CT}-{arm}: not kept: does not raise the offer on {len(lower)} of {len(tests)} "
                f"units and tiers with an offer ({', '.join(lower)}); {others} not a test"
            )
        elif tests:
            untold.append(f"{CT}-{arm}")
            told = len(raised)
            print(
                f"  {CT}-{arm}: cannot tell: raises on {told} and cannot tell on "
                f"{len(tests) - told} of {len(tests)} units and tiers with an offer; {others} "
                "not a test"
            )
        else:
            empty.append(f"{CT}-{arm}")
            print(f"  {CT}-{arm}: no development unit and tier with an offer; {others} not a test")
    if untold or empty:
        problems = []
        if untold:
            problems.append(
                f"{', '.join(untold)} cannot be told from {name}: its record holds no gains by "
                "draw, so beating every draw cannot be read"
            )
        if empty:
            problems.append(f"{', '.join(empty)} has no development unit and tier with an offer")
        print(f"COULD NOT RUN: {'; '.join(problems)}")
        return COULD_NOT_RUN
    decided = ", ".join(f"{CT}-{arm} {verdicts[arm]}" for arm in ARMS)
    print(
        f"decided: {decided} (reading W4b on the development units); keeping is recorded, only "
        f"the owner cuts; {name}"
    )
    return PASSED


# --- the command ---------------------------------------------------------------------------------

CHECKS = {"abstain": abstain, "spread": spread, "t8": t8, "conformal": conformal}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python tools/wp9_check.py",
        description="Decide one item of the WP9 pass mark from the recorded runs.",
    )
    parser.add_argument("item", choices=ITEMS, help="the clause of the WP9 pass mark")
    parser.add_argument(
        "--top",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="the repository top folder, where runs.noindex/ is (default: this repository)",
    )
    options = parser.parse_args(argv)
    try:
        return CHECKS[options.item](options.top)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return COULD_NOT_RUN
    except Exception as problem:  # an unexpected error (module text)
        print(f"COULD NOT RUN: the check stopped on {type(problem).__name__}: {problem}")
        return COULD_NOT_RUN


if __name__ == "__main__":
    sys.exit(main())

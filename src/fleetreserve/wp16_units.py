"""The units of the run from the tag and the configurations the slots read (plan section 11).

Read from committed files and from the run's own summaries, never from the data:
- `nine(source)`: the nine held-out units, from the run list (`run-lists/run-list.json`,
  `units.sealed-run.nine`);
- `abstained(source)`: the units the held-out truth build gave no window, with its reason, from
  `truth-summary.json` of the step `sealed-truth` (`abstained`); raises `Missed` when that file
  cannot be read, so that no unit is shown as having a window on no evidence;
- `run_list(source)` and `interval_settings(source)`: the run list, once it is the one the drive ran
  (its SHA-256 is the drive's `list_sha256`), and the seed and resamples it fixes for intervals;
- `option_value(command, option)`: the value after an option in a step's command as run;
- `matrix_of(source, step, sha256)`: the committed matrix a learn step named with `--matrix`, once
  its bytes have the SHA-256 the run's `learn.json` recorded;
- `headline_by_unit(matrix)`: each unit's headline configuration, `sealed_choice.headline` of its
  site type (`sealed_choice.units`), the configuration whose figures O1 and O4 read.
"""

import hashlib
import json

from fleetreserve.wp16_drive import LIST_FILE, Missed, Source

TRUTH_STEP = "sealed-truth"
TRUTH_SUMMARY = "truth-summary.json"


def run_list(source: Source) -> dict:
    """The committed run list, once its SHA-256 is the one the drive's start line records (the
    drive ran this list); raises `Missed`."""
    data = source.committed(LIST_FILE)
    if hashlib.sha256(data).hexdigest() != source.drive.list_sha256:
        raise Missed(f"unreadable: {LIST_FILE} is not the run list the drive ran (its SHA-256)")
    try:
        document = json.loads(data)
    except ValueError as problem:
        raise Missed(f"unreadable: {LIST_FILE}: not JSON: {problem}") from problem
    if not isinstance(document, dict):
        raise Missed(f"unreadable: {LIST_FILE}: not a JSON object")
    return document


def interval_settings(source: Source) -> tuple[int, int]:
    """(the seed, the number of resamples) the run list fixes for the intervals."""
    try:
        fixed = run_list(source)["fixed"]
        return fixed["seeds"]["intervals"], fixed["counts"]["resamples"]
    except (KeyError, TypeError) as problem:
        raise Missed(
            f"unreadable: {LIST_FILE}: no fixed seeds.intervals and counts.resamples"
        ) from problem


def nine(source: Source) -> list[str]:
    document = run_list(source)
    try:
        units = document["units"]["sealed-run"]["nine"]
    except (KeyError, TypeError) as problem:
        raise Missed(f"unreadable: {LIST_FILE}: no units.sealed-run.nine") from problem
    if not isinstance(units, list) or not all(isinstance(u, str) for u in units):
        raise Missed(f"unreadable: {LIST_FILE}: units.sealed-run.nine is not a list of names")
    return list(units)


def abstained(source: Source) -> dict[str, str]:
    summary = source.json(TRUTH_STEP, TRUTH_SUMMARY)  # `Missed` when it cannot be read
    found = summary.get("abstained")
    out = {}
    for unit, entry in (found if isinstance(found, dict) else {}).items():
        reason = entry.get("reason") if isinstance(entry, dict) else None
        out[unit] = reason if isinstance(reason, str) else "no reason recorded"
    return out


def option_value(command, option: str) -> str | None:
    if not isinstance(command, list) or option not in command:
        return None
    at = command.index(option)
    return command[at + 1] if at + 1 < len(command) else None


def matrix_of(source: Source, label: str, sha256: str | None) -> dict:
    """The matrix of the learn step `label` (module text); raises `Missed`."""
    step = source.drive.steps.get(label)
    path = option_value(step.command if step else None, "--matrix")
    if path is None:
        raise Missed(f"unreadable: {label}: its command names no --matrix")
    data = source.committed(path)
    if not isinstance(sha256, str):
        raise Missed(f"unreadable: {label}: its learn.json records no SHA-256 of the matrix")
    if hashlib.sha256(data).hexdigest() != sha256:
        raise Missed(f"unreadable: {path} is not the matrix the run {label} recorded")
    try:
        document = json.loads(data)
    except ValueError as problem:
        raise Missed(f"unreadable: {path}: not JSON: {problem}") from problem
    if not isinstance(document, dict):
        raise Missed(f"unreadable: {path}: not a JSON object")
    return document


def headline_by_unit(matrix: dict) -> dict[str, str]:
    choice = matrix.get("sealed_choice")
    if not isinstance(choice, dict):
        raise Missed("unreadable: the matrix holds no sealed_choice")
    try:
        return {
            unit: choice["headline"][kind]
            for kind, units in choice["units"].items()
            for unit in units
        }
    except (KeyError, TypeError, AttributeError) as problem:
        raise Missed("unreadable: the matrix has no sealed_choice headline") from problem


def beside_name(matrix: dict, name: str) -> str | None:
    """The configuration of the `beside` family (departures seen at the start of the next hour)
    with the headline `name`'s kappa, dial and departure kind, or None."""
    configurations = matrix.get("configurations")
    mine = next((c for c in configurations or [] if c.get("name") == name), None)
    if mine is None:
        return None
    for c in configurations:
        if (
            c.get("family") == "beside"
            and c.get("kappa") == mine.get("kappa")
            and c.get("driver_risk") == mine.get("driver_risk")
            and c.get("departure") == mine.get("departure")
        ):
            return c.get("name")
    return None

"""Two models' calibrated offers compared split by split on the same seeded splits (plan sections
6 and 7).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.model_pairs --first <dial-choice.json ...> [--first-pair LEVEL:DIAL]
        --second <dial-choice.json ...> [--second-pair LEVEL:DIAL]
        [--bootstrap-seed S] [--resamples N]

`dial_choice` measures one model at one caution level (kappa) and gives, per unit, configuration
(a dial) and tier, the calibrated offer of each of its seeded splits (seeds 1 to 200, the file
`dial_choice.SPLIT_OFFERS_NAME` beside its results). The splits are seeded by `day_split`, so split
s of one model and split s of another hold the same calibration and test days. This command takes
two sets of recorded `dial_choice` results files, one set per model, and for each gate unit and
tier prints, for the pair (kappa, dial) of each model: the offers of both on every seed, their
difference per seed (first minus second), the mean difference, a 95% interval for it, and the
plain count of splits where the first offers more, the same, or less. Nothing is pooled across
tiers or units, and the units of a site type are never added.

A set (reading P1). The files of one model, of one kind: either runs of the plan's grid
(`dial_choice --kappa`, at most one per level) or what-if runs of ONE fit (`kappa_choice`
reading K10: at most one per level, all with one `what_if.fit` record or all without), never both
in one set. Each results file is read and verified as `kappa_choice` reads it
(`kappa_choice.read_run`: manifest, label, certificate timeouts). Its per-split file is verified
the same way (listed in the manifest with the hash it has now), must name the results file beside
it with that file's SHA-256, must carry the run's identity unchanged, and must hold, for every
cell it lists, offers whose mean is the results' mean offer. A run made before the per-split file
existed has none and is refused, saying so. The runs of a set agree in every field
`kappa_choice.check_agreement` lists (commit, matrix, truth inputs, splits, seeds, scales, tiers,
rung, dials, units) except the level.

The pair of a set (reading P2). `--first-pair` and `--second-pair` give `LEVEL:DIAL` (the dial a
number, or `none` for the configuration without driver-risk rows), the same pair for every unit
and tier; the set must hold a run at that level. Without the option the set's own best pair is
read by `kappa_choice` reading R1, per site type (`kappa_choice.SITE_TYPE_UNITS`), tier A: for a
set of the plan's grid it needs the six runs of the grid and is exactly the R1 of `kappa_choice`;
for a set of what-if runs it is R1 over the levels the set holds (printed). Only the unit of each
site type is compared then, at the pair of its site type, in both tiers. A pair that has no figures
(unit not split, configuration failed or not measured) is "not defined", never an offer of 0.

The two sets (reading P3). They must agree in matrix and its hash, truth inputs, the split seeds
and their number, scales, tiers, rung, dials and per unit status and counting days; they differ in
model (and fit units). They must agree in commit too, unless both sets' runs are readable under
`run_evidence.read_evidence` at the checked-out commit (a commit that differs only in places the
keep-list names); otherwise the command refuses. The label of a what-if set stays on every line
that names it, with its seed or seed list and levels fitted.

The interval (reading P4). A percentile bootstrap over splits: the per-seed differences are
resampled with replacement `--resamples` times (default 2000) with a generator seeded by
`--bootstrap-seed` (default 1), the mean of each resample taken, and the 2.5th and 97.5th
percentiles are the interval. Each cell is resampled from a generator of its own made from the
seed, so the interval of one cell does not depend on which other cells are printed. Two offers
within `dial_splits.TIE` kWh count as the same in the plain count.

The caution printed in the head of the summary: when each model's pair is picked as its best on
these same splits, the comparison is optimistic for both, and splits of one unit share days, so
the interval is over splits, not over independent days.

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output:
- `model-pairs.json` (`RESULTS_NAME`): the sets (files, SHA-256, commit, run folder, kind, fit),
  the pair per unit and set, and per unit and tier the figures above with the per-seed offers;
- `model-pairs-summary.txt` (`SUMMARY_NAME`).

Exit status: 0 written; 1 a file was refused by its run's manifest (`run_verify.NotVerified`); 2
could not run (a file, manifest or per-split file missing or unreadable, a set that mixes kinds,
fits or repeats a level, sets that differ, a pair the set cannot give, or no cell to compare).
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

from fleetreserve import (
    dial_choice,
    dial_splits,
    dial_summary,
    kappa_choice,
    run_evidence,
)
from fleetreserve.gate_config import TIER_LABELS
from fleetreserve.kappa_choice import CouldNotRun
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified

RESULTS_NAME = "model-pairs.json"
SUMMARY_NAME = "model-pairs-summary.txt"
RESAMPLES = 2000
BOOTSTRAP_SEED = 1
INTERVAL = (2.5, 97.5)
CAUTION = (
    "When each model's pair is picked as its best on these same splits, the comparison is "
    "optimistic for both, and splits of one unit share days, so the interval is over splits, not "
    "over independent days."
)
SAME_ACROSS_SETS = ("matrix", "inputs")
SAME_RUN_ACROSS_SETS = ("splits", "seeds", "scales", "tiers", "rung", "dials")
IDENTITY = ("label", "model", "fit_units", "matrix", "inputs", "run")
NONE_WORD = "none"


class Member(NamedTuple):
    """One run of a set: its results (`kappa_choice.Run`), its per-split file, where they lie."""

    run: kappa_choice.Run
    split: dict
    split_source: dict
    folder: Path


class Setting(NamedTuple):
    """One set: its runs, kind ("grid" or "what-if") and the fit of its what-if runs."""

    members: list[Member]
    kind: str
    fit: dict | None


# --- reading a set -------------------------------------------------------------------------------


def _verified(path: Path) -> None:
    try:
        require_verified(path)
    except NotVerified as problem:
        raise type(problem)(f"{path.parent.name}/{path.name}: {problem}") from problem


def read_member(path) -> Member:
    """One results file with its per-split file (module text). Raises `NotVerified` or
    `CouldNotRun`."""
    run = kappa_choice.read_run(path)
    folder = Path(os.path.abspath(path)).parent
    names = dial_choice.WHAT_IF_NAMES if run.what_if else dial_choice.PLAN_NAMES
    file = folder / names["split_offers"]
    where = f"{folder.name}/{file.name}"
    if not file.is_file():
        raise CouldNotRun(
            f"{folder.name} holds no {file.name}: the run was made before dial_choice wrote the "
            "per-split offers and has to be made again"
        )
    _verified(file)
    content = file.read_bytes()
    try:
        split = json.loads(content)
        if not isinstance(split, dict):
            raise ValueError("it is not a JSON object")
        if split["results"] != {"file": names["results"], "sha256": run.source["sha256"]}:
            raise ValueError("it does not name the results file beside it with its SHA-256")
        for key in IDENTITY:
            if split.get(key) != run.document.get(key):
                raise ValueError(f"its {key} is not the results file's")
        if ("what_if" in split) != run.what_if or split.get("what_if") != run.document.get(
            "what_if"
        ):
            raise ValueError("its what-if record is not the results file's")
        _check_units(split, run)
    except (ValueError, KeyError, TypeError) as problem:
        raise CouldNotRun(f"{where} cannot be used: {problem}") from problem
    source = {"split_offers": file.name, "split_offers_sha256": hashlib.sha256(content).hexdigest()}
    return Member(run, split, source, folder)


def _check_units(split: dict, run: kappa_choice.Run) -> None:
    """Raises `ValueError` unless the per-split file agrees with its results file cell by cell."""
    units, seeds = run.document["units"], run.document["run"]["seeds"]
    wanted = list(range(seeds[0], seeds[1] + 1)) if seeds else []
    if set(split["units"]) != set(units):
        raise ValueError("its units are not the results file's")
    for unit, entry in split["units"].items():
        found = units[unit]
        if (entry["status"], entry["counting_days"]) != (found["status"], found["counting_days"]):
            raise ValueError(f"unit {unit}: its status or counting days are not the results'")
        if set(entry["configurations"]) != set(found["configurations"]):
            raise ValueError(f"unit {unit}: its configurations are not the results'")
        for name, configuration in entry["configurations"].items():
            theirs = found["configurations"][name]
            if "failed" in configuration or "failed" in theirs:
                if configuration != {k: theirs[k] for k in ("driver_risk", "failed")}:
                    raise ValueError(f"unit {unit}, {name}: a failure is not recorded in both")
                continue
            if configuration["driver_risk"] != theirs["driver_risk"]:
                raise ValueError(f"unit {unit}, {name}: its dial is not the results'")
            for tier, rows in configuration["tiers"].items():
                if sorted(r["seed"] for r in rows) != wanted:
                    raise ValueError(f"unit {unit}, {name}, tier {tier}: its seeds are not {seeds}")
                mean = float(np.mean([r["offer_kwh"] for r in sorted(rows, key=_seed)]))
                recorded = theirs["tiers"][tier]["mean_offer_kwh_per_test_day"]
                if not math.isclose(mean, recorded, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(
                        f"unit {unit}, {name}, tier {tier}: the mean of its offers, {mean!r}, is "
                        f"not the results' mean offer, {recorded!r}"
                    )


def _seed(row: dict) -> int:
    return row["seed"]


def make_setting(members: list[Member], need_grid: bool) -> Setting:
    """The set of `members` (module text, reading P1); `need_grid` asks for the six runs of the
    plan's grid. Raises `CouldNotRun`."""
    runs = [m.run for m in members]
    kinds = {r.what_if for r in runs}
    if len(kinds) != 1:
        raise CouldNotRun("a set holds runs of the plan's grid or what-if runs, never both")
    what_if = kinds.pop()
    levels = [r.document["run"]["kappa"] for r in runs]
    if what_if:
        groups = {kappa_choice.fit_group(r) for r in runs}
        if len(groups) != 1:
            raise CouldNotRun(
                "a set holds what-if runs of one fit only (`kappa_choice` reading K10)"
            )
        fit = kappa_choice.fit_of(runs[0])
    else:
        fit = None
    twice = sorted({k for k in levels if levels.count(k) > 1})
    if twice:
        raise CouldNotRun(f"a set holds two runs at kappa {', '.join(f'{k:g}' for k in twice)}")
    if need_grid and not what_if:
        kappa_choice.check_runs(runs)
    kappa_choice.check_agreement(runs)
    return Setting(sorted(members, key=_level), "what-if" if what_if else "grid", fit)


def _level(member: Member) -> float:
    return member.run.document["run"]["kappa"]


def tag(setting: Setting) -> str:
    """What kind of set it is, printed on every line that names the set (reading P1): for a
    what-if set its label and fit."""
    if setting.kind == "grid":
        return "the plan's grid"
    label = setting.members[0].run.document["what_if"]["label"]
    fit = "no fit option" if setting.fit is None else kappa_choice.fit_text(setting.fit)
    return f"what-if: {label}; {fit}"


def describe(setting: Setting) -> str:
    """The set in words."""
    levels = ", ".join(f"{_level(m):g}" for m in setting.members)
    model = setting.members[0].run.document["model"]
    return f"model {model} ({tag(setting)}; levels {levels})"


# --- the pair ------------------------------------------------------------------------------------


def parse_pair(text: str) -> tuple[float, float | None]:
    """`LEVEL:DIAL` as (level, dial); the dial `none` is None. Raises `CouldNotRun`."""
    try:
        level, dial = text.split(":")
        return float(level), None if dial == NONE_WORD else float(dial)
    except ValueError as problem:
        raise CouldNotRun(
            f"a pair is written LEVEL:DIAL (the dial a number or none): {text!r}"
        ) from problem


def best_pairs(setting: Setting) -> dict:
    """Per unit of each site type, the set's best pair by `kappa_choice` reading R1 (module text,
    reading P2): unit -> (level, dial) or None, with the site type."""
    runs = [m.run for m in setting.members]
    dials = list(runs[0].document["run"]["dials"])
    tiers = list(runs[0].document["run"]["tiers"])
    if setting.kind == "grid":
        found = kappa_choice.joint(runs)["readings"]["R1"]
    else:
        units = {
            u: kappa_choice._unit(u, [m.run for m in setting.members], [None, *dials], tiers)
            for u in runs[0].document["units"]
        }
        levels = tuple(_level(m) for m in setting.members)
        found = kappa_choice.readings(units, dials, levels)["R1"]
    out = {}
    for site_type, unit in kappa_choice.SITE_TYPE_UNITS.items():
        pair = found[site_type]
        out[unit] = {
            "site_type": site_type,
            "pair": None if pair is None else (pair["kappa"], pair["dial"]),
        }
    return out


def cell_of(setting: Setting, unit: str, pair: tuple, tier: str) -> tuple[dict | None, str]:
    """The per-split rows (seed -> offer) of a pair in one unit and tier, or None and why not."""
    level, dial = pair
    members = [m for m in setting.members if _level(m) == level]
    if not members:
        return None, f"the set holds no run at kappa {level:g}"
    entry = members[0].split["units"].get(unit)
    if entry is None:
        return None, f"the unit {unit} is not in the run"
    if entry["status"] != "calibratable":
        return None, f"the unit cannot be split ({entry['status']})"
    found = [c for c in entry["configurations"].values() if c["driver_risk"] == dial]
    if not found:
        return None, f"no configuration at dial {kappa_choice.key(dial)}"
    if "failed" in found[0]:
        return None, f"the configuration failed: {found[0]['failed']}"
    rows = found[0]["tiers"].get(tier)
    if rows is None:
        return None, f"no figures in tier {tier}"
    return {r["seed"]: r["offer_kwh"] for r in rows}, ""


# --- the comparison ------------------------------------------------------------------------------


def interval(differences: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    """The percentile bootstrap interval of the mean of `differences` (module text, reading P4)."""
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, len(differences), size=(resamples, len(differences)))
    means = differences[picks].mean(axis=1)
    low, high = (float(q) for q in np.percentile(means, INTERVAL))
    return low, high


def compare(first: dict, second: dict, resamples: int, seed: int) -> dict:
    """The comparison of two cells (seed -> offer), paired by seed. Raises `CouldNotRun` when the
    seeds differ."""
    if set(first) != set(second):
        raise CouldNotRun("the two cells do not hold the same split seeds")
    seeds = sorted(first)
    a = np.array([first[s] for s in seeds], dtype=float)
    b = np.array([second[s] for s in seeds], dtype=float)
    d = a - b
    low, high = interval(d, resamples, seed)
    return {
        "splits": len(seeds),
        "mean_offer_first": float(a.mean()),
        "mean_offer_second": float(b.mean()),
        "mean_difference": float(d.mean()),
        "interval_95": [low, high],
        "first_offers_more": int((d > dial_splits.TIE).sum()),
        "equal": int((np.abs(d) <= dial_splits.TIE).sum()),
        "first_offers_less": int((d < -dial_splits.TIE).sum()),
        "per_split": [
            {"seed": s, "first": float(x), "second": float(y), "difference": float(z)}
            for s, x, y, z in zip(seeds, a, b, d, strict=True)
        ],
    }


def check_sets(one: Setting, two: Setting, head: str) -> None:
    """Raises `CouldNotRun` unless the two sets agree (module text, reading P3). `head`: the
    checked-out commit, against which `run_evidence` reads runs of another commit."""
    runs = [one.members[0].run, two.members[0].run]
    for field in SAME_ACROSS_SETS:
        values = [r.document.get(field) for r in runs]
        if values[0] != values[1]:
            raise CouldNotRun(f"the sets differ in {field}: {values[0]!r} and {values[1]!r}")
    for field in SAME_RUN_ACROSS_SETS:
        values = [r.document["run"].get(field) for r in runs]
        if values[0] != values[1]:
            raise CouldNotRun(f"the sets differ in {field}: {values[0]!r} and {values[1]!r}")
    units = [
        {u: (e.get("status"), e.get("counting_days")) for u, e in r.document["units"].items()}
        for r in runs
    ]
    if units[0] != units[1]:
        raise CouldNotRun("the sets differ in their units, their status or counting days")
    if runs[0].source["commit"] != runs[1].source["commit"]:
        for member in (*one.members, *two.members):
            _readable(member, head)


def _readable(member: Member, head: str) -> None:
    """Raises `CouldNotRun` unless the member's run is readable under `run_evidence` (P3)."""
    names = dial_choice.WHAT_IF_NAMES if member.run.what_if else dial_choice.PLAN_NAMES
    try:
        run_evidence.read_evidence(member.folder, names["results"], head)
    except run_evidence.CannotCheck as problem:
        raise CouldNotRun(
            f"the sets differ in commit and {member.folder.name} is not readable under "
            f"run_evidence: {problem}"
        ) from problem


def pairs_for(setting: Setting, option: str | None) -> dict:
    """Per unit, the pair of the set (module text, reading P2): unit -> {"pair": (level, dial) or
    None, "why": text, "chosen": "given" or the reading and site type}."""
    first = setting.members[0].run.document
    if option is not None:
        pair = parse_pair(option)
        if not any(_level(m) == pair[0] for m in setting.members):
            raise CouldNotRun(f"the set holds no run at kappa {pair[0]:g}: {describe(setting)}")
        return {
            unit: {"pair": pair, "chosen": "given on the command line"}
            for unit, entry in first["units"].items()
            if entry["status"] == "calibratable"
        }
    return {
        unit: {**found, "chosen": f"R1 of {found['site_type']}"}
        for unit, found in best_pairs(setting).items()
    }


def comparison(first: Setting, second: Setting, options: tuple, resamples: int, seed: int) -> dict:
    """Per unit and tier the paired comparison of the two sets (module text)."""
    chosen = [pairs_for(first, options[0]), pairs_for(second, options[1])]
    tiers = list(first.members[0].run.document["run"]["tiers"])
    units = {}
    for unit in (u for u in chosen[0] if u in chosen[1]):
        found = {}
        for tier in tiers:
            cells, why = [], []
            for setting, picks in zip((first, second), chosen, strict=True):
                pick = picks[unit]
                if pick["pair"] is None:
                    cells.append(None)
                    why.append("no pair: nothing to choose from")
                    continue
                cell, reason = cell_of(setting, unit, pick["pair"], tier)
                cells.append(cell)
                why.append(reason)
            if None in cells:
                found[tier] = {"defined": False, "why": [w for w in why if w]}
            else:
                found[tier] = {"defined": True, **compare(*cells, resamples, seed)}
        units[unit] = {
            "pairs": [_pair_record(picks[unit]) for picks in chosen],
            "tiers": found,
        }
    return units


def _pair_record(pick: dict) -> dict:
    """The pair of one unit as the results file holds it."""
    if pick["pair"] is None:
        return {"kappa": None, "dial": None, "chosen": pick["chosen"]}
    level, dial = pick["pair"]
    return {"kappa": level, "dial": dial, "chosen": pick["chosen"]}


# --- the command ---------------------------------------------------------------------------------


def _set_record(setting: Setting) -> dict:
    return {
        "kind": setting.kind,
        "description": describe(setting),
        "tag": tag(setting),
        "fit": setting.fit,
        "runs": [{"kappa": _level(m), **m.run.source, **m.split_source} for m in setting.members],
    }


def summary_lines(document: dict) -> list[str]:
    """The plain-text summary of a document of `main` (module text)."""
    lines = [
        "Two models compared split by split on the same seeded splits (model_pairs).",
        CAUTION,
        f"First : {document['sets']['first']['description']}",
        f"Second: {document['sets']['second']['description']}",
        f"Offers in kWh per counting test day; differences are first minus second; the interval "
        f"is a percentile bootstrap over the splits ({document['run']['resamples']} resamples, "
        f"seed {document['run']['bootstrap_seed']}); counts are splits where the first offers "
        "more / the same / less.",
    ]
    for unit, entry in document["units"].items():
        lines.append("")
        lines.append(f"{unit}")
        for number, name in enumerate(("first", "second")):
            record = entry["pairs"][number]
            if record["kappa"] is None:
                text = "no pair"
            else:
                text = f"kappa {record['kappa']:g}, dial {kappa_choice.key(record['dial'])}"
            lines.append(
                f"  {name} pair: {text} ({record['chosen']}; {document['sets'][name]['tag']})"
            )
        for tier, found in entry["tiers"].items():
            lines.append(f" {TIER_LABELS[tier]}")
            if not found["defined"]:
                lines.append(f"  not defined: {'; '.join(found['why'])}")
                continue
            low, high = found["interval_95"]
            lines.append(
                f"  first {found['mean_offer_first']:.3f}  second {found['mean_offer_second']:.3f}"
                f"  difference {found['mean_difference']:+.3f} (95% interval {low:+.3f} to "
                f"{high:+.3f}) over {found['splits']} splits; "
                f"{found['first_offers_more']} / {found['equal']} / {found['first_offers_less']}"
            )
    return lines


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _read_set(paths: list[Path], pair: str | None) -> Setting:
    return make_setting([read_member(p) for p in paths], need_grid=pair is None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", nargs="+", type=Path, required=True)
    parser.add_argument("--second", nargs="+", type=Path, required=True)
    parser.add_argument("--first-pair", default=None)
    parser.add_argument("--second-pair", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args(argv)
    try:
        if args.resamples < 1:
            raise CouldNotRun("--resamples must be at least 1")
        first = _read_set(args.first, args.first_pair)
        second = _read_set(args.second, args.second_pair)
        head = run_evidence.head_commit(dial_choice.TOP)
        check_sets(first, second, head)
        units = comparison(
            first, second, (args.first_pair, args.second_pair), args.resamples, args.bootstrap_seed
        )
        if not any(t["defined"] for u in units.values() for t in u["tiers"].values()):
            raise CouldNotRun("no unit and tier has a pair in both sets to compare")
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except NotVerified as problem:
        print(f"REFUSED: {problem}")
        return 1
    except (CouldNotRun, run_evidence.CannotCheck) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    one = first.members[0].run
    document = {
        "label": f"model pairs: {dial_summary.LABEL}",
        "caution": CAUTION,
        "run": {
            "commits": sorted({m.run.source["commit"] for s in (first, second) for m in s.members}),
            "matrix": one.document.get("matrix"),
            "inputs": one.document.get("inputs"),
            "splits": one.document["run"]["splits"],
            "seeds": one.document["run"]["seeds"],
            "resamples": args.resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "tiers": one.document["run"]["tiers"],
        },
        "sets": {"first": _set_record(first), "second": _set_record(second)},
        "units": units,
    }
    text = "\n".join(summary_lines(document)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

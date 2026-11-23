"""Kappa and the dial read together from six dial-choice runs, one per level (plan sections 2, 6
and 7).

Command line, from the project's top folder (usually under `fleetreserve.run_record`):
    python -m fleetreserve.kappa_choice <run folder>/dial-choice.json [five more such files]
        [<run folder>/dial-choice-what-if.json ...]

Plan section 6 fixes the dial, per site type, "as the value of Section 2 with the largest mean
calibrated Tier A re-solve offer over those 200 development splits at the main model and κ", and
section 7 fixes kappa as the "largest calibrated development offer", kappa in section 2's grid
0.05, 0.1, 0.2, 0.3, 0.4, 0.5 (`forecast_empirical.KAPPAS`). `dial_choice --kappa <level>` measures
the dial family at one level with the dial's definitions, splits and calibration; this command
reads one such run per level and prints, per gate unit and tier, one table with four lines per
kappa and a column per dial, the configuration without rows ("none") first: per pair the mean
offer, the share of splits that offer and the mean slot and driver losses over the splits that
offer; then per unit the timeouts per pair. It names, per site type, the pair each reading of the
joint rule gives (K5), labelled "measured, not fixed". It writes no chosen value into code or
settings: the plan fixes both settings at the main model before the tag, and which reading holds
is the owner's decision. Plan section 7 gives the neural model and each of the two earlier
forecasts (`earlier_gbm`, `earlier_population`) its own kappa and dial by the same rule: this
command reads the six runs of one model, whichever it is (`dial_choice`, module text, "The
earlier forecasts"), and refuses runs of two models together (`SAME_FIELDS`).

Runs read. Each file named is a results file of `dial_choice` (its label: a rotation what-if is
not read) verified against its run's manifest (`run_verify.require_verified`: exit code 0, not
stopped at a time limit, not interrupted, listed with the hash it has now, its commit known) and
lying directly in its run folder; the run's certificate-timeouts file
(`dial_choice.CERTIFICATE_TIMEOUTS_NAME`, written with `--kappa`) must be there, verified the same
way and of the same kappa. The runs must hold one level each of plan section 2's grid, six in all,
and agree in all else they measured: the commit of their manifests, the model and its fit units,
the matrix file and its hash, the truth inputs, the splits, seeds, scales, tiers, rung and dials,
and per gate unit its status and counting days.

Per pair (kappa, dial) and tier the figures are those of the run at that kappa, its configuration
with that dial (`CELL_FIELDS`, as `dial_splits.summarise` gives them). Timeouts per pair: the
days whose certificate timed out (from the certificate-timeouts file; such a day abstains) and the
one-way re-solves that timed out (from the configuration's evaluations; the rung keeps its
predecessor's plan); each run's own totals, ceilings and oracle partners included, are printed
too.

Readings (K1 and K2 are `dial_choice`'s):
  K3. Site types (`SITE_TYPE_UNITS`): workplaces read at NREL from May 2019 and Norway and Korea
      at the pooled Norwegian development sites, plan section 6's words and the units
      `DIAL_SOURCES` of `tools/compare_shared_check.py` names. NREL before May 2019 is measured
      and printed and enters no reading. [Alternative: both NREL units for workplaces.]
  K4. The configuration without rows is listed and is never a pair of either reading, as in
      `dial_choice`, where it is listed and never the dial. [Alternative: "none" as a value of the
      grid, which plan section 2 lists among the dial's values.]
  K5. The joint rule, two readings, both printed, neither chosen:
      R1: per site type, kappa and the dial together: the pair with the largest mean tier A offer
          at the site type's unit;
      R2: one kappa for the model: the level with the largest tier A offer summed over the two
          site types' units, each at its own best dial at that level (`dial_splits.pick_dial`);
          then per site type the dial with the largest offer at that kappa.
      `differ_at` lists the site types where the two give different pairs.
  K6. Ties: two offers within `dial_splits.TIE` kWh are equal; a tie goes to the smaller kappa,
      then to the smaller dial (the dial's own tie rule, the smaller value being the more careful
      one: an earlier departure bound, a smaller shortfall). [Alternative: to the larger kappa.]
  K7. Not defined: a pair with no figures in its run (the unit is not split, or the configuration
      failed or was not measured) is "not defined": printed so, never as 0, and in no reading. A
      sum of R2 at a level where a site type's unit has no dial with a mean offer is not defined,
      and so is a reading with nothing to choose from. A pair whose splits all abstain is defined:
      it offers 0, and its losses, over no offering split, are "-". [Alternative: a pair without
      figures read as an offer of 0, which would let a failed run choose.]
  K8. The readings use tier A, plan section 6's "Tier A re-solve offer"; tier B is printed only.
  K9. What-if runs (`dial_choice --what-if-kappa`, files `dial_choice.WHAT_IF_NAMES`) may be named
      besides the six runs of the grid. Each is read and verified as a run of the grid is (its
      own label and its what-if certificate-timeouts file), at most one per level, and must agree
      with the grid's runs in all else they measured (commit included, so the grid's six runs
      are recorded at the what-if runs' commit). They count as no run of the grid: the six runs
      of the grid are still needed, every figure and reading above is that of those six alone,
      and the summary prints it unchanged. After it, under the key `what_if` and in a block of the
      summary labelled "what-if, level not on the plan's grid": per unit and tier the table and
      timeouts of the what-if runs, and R1 and R2 taken over all levels given, the grid's levels
      from the grid's runs and the levels off the grid from the what-if runs; a what-if run at a
      grid level (the comparison at the same fit, `dial_choice` reading W3) is printed and is in
      neither reading. [Alternative: the readings over the what-if runs alone, which needs a
      what-if run at every grid level.]
  K10. What-if runs with a fit option (`dial_choice --what-if-fit-seed`, `--what-if-fit-levels`;
      the key `what_if.fit` of their results holds the seed and the levels the network was fitted
      at) are told apart by that fit. A run without a fit option is a fit of its own (its seed is
      not in its file). At most one run per level within one fit (so seeds 1, 2 and 3 at 0.3 are
      three runs of three fits, not three runs at one level), and every line that names a what-if
      run prints its seed and levels fitted. Runs of one fit make one block as in K9, with a line
      saying that its grid levels come from the plan's six runs; runs of different fits make one
      block each under the key `separate_fits`, and nothing is read across two blocks: the
      summary says so. [Alternative: one table of all fits side by side per level; not done, a
      reader would compare cells that are not like for like, and the comparison of two fits is
      the reader's: the figures stay in their blocks.] An average over seeds (`dial_choice
      --what-if-fit-seeds`: the key `seeds` of `what_if.fit`, the sorted seed list, in place of
      `seed`) is a fit of its own, told apart by its seed list and levels fitted: never read with
      a single-seed fit (seed 1 alone is not the average over 1, 2 and 3) nor with another list
      (1, 2, 3 and 0, 1, 2, 3 are two fits); its lines print "seeds 1, 2, 3 averaged".

Outputs, in `FLEETRESERVE_RUN_DIR` when it is set (the run recorder sets it), the summary also on
standard output:
- `kappa-choice.json` (`RESULTS_NAME`): every pair's figures and timeouts, the readings, and the
  runs read (results file, its SHA-256, run folder, commit), with `definitions`;
- `kappa-choice-summary.txt` (`SUMMARY_NAME`, `kappa_summary`).
With what-if runs, the key `what_if` and the summary's what-if block are added (K9; with fits,
K10); without them both files are as before.

Exit status: 0 written; 1 a file was refused by its run's manifest (`run_verify.NotVerified`: the
run did not end with exit 0, or the file changed); 2 could not run (a file or its manifest is
missing or unreadable, a file is not a results file of `dial_choice`, a run has no certificate
timeouts, the runs are not one per level of the grid, two what-if runs of one fit measure one
level, or they differ).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

from fleetreserve import dial_choice, dial_splits, dial_summary, driver_risk, kappa_summary
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified

RESULTS_NAME = "kappa-choice.json"
SUMMARY_NAME = "kappa-choice-summary.txt"
LEVELS = driver_risk.LEVELS  # plan section 2's grid of kappa
SITE_TYPE_UNITS = {
    "workplaces": "nrel-from-may-2019",
    "Norway and Korea": "norway-development-pooled",
}  # reading K3
TIER = "A"  # reading K8
NO_ROWS = "none"
CELL_FIELDS = (
    "mean_offer_kwh_per_test_day",
    "share_of_splits_that_offer",
    "splits_that_offer",
    "mean_test_slot_loss",
    "mean_test_driver_loss",
)
LABEL = (
    "measured, not fixed: the plan fixes kappa and the dial at the main model before the tag, "
    "and which reading of the joint rule holds is the owner's decision"
)
WHAT_IF_LABEL = (
    f"{dial_choice.OFF_GRID}: development evidence only, not the plan's rule, which is taken over "
    "the grid's runs alone"
)
WHAT_IF_DEFINITIONS = {
    "what_if": (
        "the what-if runs of dial_choice --what-if-kappa, read apart from the six runs of the "
        "grid (reading K9): their pairs per unit and tier, their timeouts, and R1 and R2 over all "
        "levels given, the grid's levels from the grid's runs and the levels off the grid from "
        "the what-if runs; a what-if run at a grid level is printed and in neither reading"
    ),
}
FIT_DEFINITIONS = {
    "fits": (
        "what-if runs of different fits (`dial_choice --what-if-fit-seed`, `--what-if-fit-seeds` "
        "and `--what-if-fit-levels`: the seed or the list of seeds averaged and the levels the "
        "network was fitted at, reading K10) "
        "are never read together: each fit has its own table and its own readings; a run without "
        "a fit option is a fit of its own, whose seed is not stated in its file"
    ),
}
SAME_FIELDS = ("model", "fit_units", "matrix", "inputs")
SAME_RUN_FIELDS = ("splits", "seeds", "scales", "tiers", "rung", "dials")
DEFINITIONS = {
    "pairs": (
        "per gate unit, tier, kappa and dial, the figures of the dial-choice run at that kappa "
        "for its configuration with that dial: the mean calibrated one-way re-solve offer per "
        "counting test day over the splits (a split with no safe scale counting as 0), the share "
        "and number of splits that offer, and the mean test slot and driver losses over the "
        "splits that offer (null when none offers)"
    ),
    "none": (
        "the configuration without driver-risk rows at that kappa, listed and never a pair of "
        "either reading (reading K4)"
    ),
    "not_defined": (
        "a pair with no figures in its run (the unit is not split, or the configuration failed "
        "or was not measured): printed 'not defined', never as 0, and in no reading (reading K7)"
    ),
    "kappa_0.05": (
        "At kappa 0.05 the driver-risk rows have the one level 0.05 at the presence bound and "
        "hold on the whole forecast set (dial_choice reading K1): its pairs are measured like "
        "the others, and their sets are the set without rows"
    ),
    "R1": (
        "Reading R1, kappa and the dial together per site type: the pair with the largest mean "
        "tier A offer at the site type's unit"
    ),
    "R2": (
        "Reading R2, one kappa for the model: the level with the largest tier A offer summed over "
        "the two site types' units, each at its own best dial there; then per site type the dial "
        "with the largest offer at that kappa"
    ),
    "ties": (
        f"offers within {dial_splits.TIE} kWh are equal; a tie goes to the smaller kappa, then "
        "to the smaller dial (reading K6)"
    ),
    "site_types": (
        "workplaces from NREL from May 2019, Norway and Korea from the pooled Norwegian "
        "development sites (plan section 6; reading K3)"
    ),
    "tiers": (
        "the readings use tier A (plan section 6); tier B is printed only; the tiers are never "
        "added or averaged"
    ),
    "timeouts": (
        "per pair, the days whose certificate timed out (each abstains) and the one-way "
        "re-solves that timed out (the rung keeps its predecessor's plan); run_timeouts holds "
        "each run's own totals"
    ),
}


class CouldNotRun(Exception):
    """The joint table cannot be made from the files named; the message says why."""


class Run(NamedTuple):
    """One dial-choice run: its results, its certificate-timeouts file, and where they came from
    (`results`, `sha256`, `run_folder`, `commit`)."""

    document: dict
    certificate_timeouts: dict
    source: dict
    what_if: bool = False


def key(dial: float | None) -> str:
    """The column key of a dial: its value, or `NO_ROWS` for the configuration without rows."""
    return NO_ROWS if dial is None else f"{dial:g}"


# --- reading the runs ----------------------------------------------------------------------------


def _verified(path: Path) -> dict:
    try:
        return require_verified(path)
    except NotVerified as problem:
        raise type(problem)(f"{path.parent.name}/{path.name}: {problem}") from problem


def read_run(path) -> Run:
    """One run's files (module text, runs read). Raises `NotVerified` (`CannotCheck` included)
    or `CouldNotRun`."""
    file = Path(os.path.abspath(path))
    manifest = _verified(file)
    where = f"{file.parent.name}/{file.name}"
    if not (file.parent / MANIFEST_NAME).is_file():
        raise CouldNotRun(f"{where} does not lie directly in its run folder")
    content = file.read_bytes()
    try:
        document = json.loads(content)
        what_if = "what_if" in document
        expected = document["what_if"]["label"] if what_if else dial_summary.LABEL
        if document["label"] != f"dial choice: {expected}":
            raise ValueError("its label is not that of a run of the plan or of a what-if run")
        kappa = document["run"]["kappa"]
        if what_if and kappa != document["what_if"]["kappa"]:
            raise ValueError("its what-if level is not its kappa")
        if not isinstance(document["units"], dict):
            raise ValueError("its units are not a mapping")
    except (ValueError, KeyError, TypeError) as problem:
        raise CouldNotRun(
            f"{where} is not a results file of dial_choice ({problem}); a rotation what-if is "
            "not read"
        ) from problem
    names = dial_choice.WHAT_IF_NAMES if what_if else dial_choice.PLAN_NAMES
    extra = file.parent / names["certificate_timeouts"]
    _verified(extra)
    try:
        counts = json.loads(extra.read_text())
        if counts["kappa"] != kappa or not isinstance(counts["certificate_timeouts"], dict):
            raise ValueError(f"its kappa {counts['kappa']!r} is not the run's {kappa!r}")
    except (ValueError, KeyError, TypeError) as problem:
        raise CouldNotRun(f"{file.parent.name}/{extra.name} cannot be used: {problem}") from problem
    source = {
        "results": file.name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "run_folder": file.parent.name,
        "commit": manifest["commit"],
    }
    return Run(document, counts, source, what_if)


def check_runs(runs: list[Run]) -> None:
    """Raises `CouldNotRun` unless the runs are one per level and agree (module text)."""
    levels = sorted(r.document["run"]["kappa"] for r in runs)
    if levels != list(LEVELS):
        raise CouldNotRun(
            f"the runs measure kappa {levels}; the joint table needs one run at each level of "
            f"plan section 2, {list(LEVELS)}"
        )
    check_agreement(runs)


def fit_of(run: Run) -> dict | None:
    """The fit of a what-if run (`what_if.fit` of its results: the seed and the levels the network
    was fitted at, reading K10), or None for a run without a fit option."""
    return run.document["what_if"].get("fit")


def fit_text(fit: dict) -> str:
    """A fit in words, as every line that names one prints it."""
    levels = ", ".join(f"{k:g}" for k in fit["levels_fitted"])
    return f"{dial_summary.seed_words(fit)}, levels fitted {levels}"


def fit_group(run: Run) -> tuple:
    """The key that tells what-if runs of one fit from those of another (reading K10): runs
    without a fit option first, then single-seed fits by seed and levels fitted, then averages
    over seeds (`--what-if-fit-seeds`, the key `seeds`) by seed list and levels fitted."""
    fit = fit_of(run)
    if fit is None:
        return (0,)
    if "seeds" in fit:
        return (2, tuple(fit["seeds"]), tuple(fit["levels_fitted"]))
    return (1, fit["seed"], tuple(fit["levels_fitted"]))


def check_what_if(runs: list[Run], what_if: list[Run]) -> None:
    """Raises `CouldNotRun` unless the what-if runs of each fit measure one level each and all
    agree with the runs of the grid `runs` (readings K9 and K10)."""
    for group in sorted({fit_group(r) for r in what_if}):
        levels = [r.document["run"]["kappa"] for r in what_if if fit_group(r) == group]
        twice = sorted({k for k in levels if levels.count(k) > 1})
        if twice:
            same = "" if group == (0,) else " of one fit"
            raise CouldNotRun(
                f"two what-if runs{same} measure kappa {', '.join(f'{k:g}' for k in twice)}"
            )
    check_agreement([*runs, *what_if])


def check_agreement(runs: list[Run]) -> None:
    """Raises `CouldNotRun` unless the runs agree in all else they measured (module text)."""

    def same(field: str, values: list) -> None:
        if any(v != values[0] for v in values):
            shown = "; ".join(
                f"kappa {r.document['run']['kappa']:g}{' (what-if)' if r.what_if else ''}: {v!r}"
                for r, v in zip(runs, values, strict=True)
            )
            raise CouldNotRun(f"the runs differ in {field}: {shown}")

    same("commit", [r.source["commit"] for r in runs])
    for field in SAME_FIELDS:
        same(field, [r.document.get(field) for r in runs])
    for field in SAME_RUN_FIELDS:
        same(field, [r.document["run"].get(field) for r in runs])
    same(
        "units",
        [
            {u: (e.get("status"), e.get("counting_days")) for u, e in r.document["units"].items()}
            for r in runs
        ],
    )


# --- the joint table -----------------------------------------------------------------------------


def _cell(entry: dict, found, tier: str) -> dict:
    """One pair's figures in one tier (module text), or why it is not defined."""
    if entry["status"] != "calibratable":
        return {"defined": False, "why": f"the unit cannot be split ({entry['status']})"}
    if found is None:
        return {"defined": False, "why": "the configuration was not measured"}
    _, configuration = found
    if "failed" in configuration:
        return {"defined": False, "why": f"the configuration failed: {configuration['failed']}"}
    figures = configuration.get("tiers", {}).get(tier)
    if figures is None:
        return {"defined": False, "why": f"the configuration has no figures in tier {tier}"}
    return {"defined": True, **{field: figures[field] for field in CELL_FIELDS}}


def _timeouts(entry: dict, found, counts: dict) -> dict | None:
    """One pair's timeouts (module text); None when the pair is not defined."""
    if entry["status"] != "calibratable" or found is None or "failed" in found[1]:
        return None
    name, configuration = found
    resolves = configuration["evaluations"]["re_solves_of_the_one_way_rung"]
    return {"certificates": counts.get(name), "re_solves": resolves.get("timeout", 0)}


def _unit(unit: str, ordered: list[Run], columns: list, tiers: list) -> dict:
    first = ordered[0].document["units"][unit]
    out = {
        "status": first["status"],
        "counting_days": first["counting_days"],
        "pairs": {tier: {} for tier in tiers},
        "timeouts": {},
    }
    for run in ordered:
        level = f"{run.document['run']['kappa']:g}"
        entry = run.document["units"][unit]
        by_dial = {c["driver_risk"]: (name, c) for name, c in entry["configurations"].items()}
        counts = run.certificate_timeouts["certificate_timeouts"].get(unit, {})
        for tier in tiers:
            out["pairs"][tier][level] = {
                key(d): _cell(entry, by_dial.get(d), tier) for d in columns
            }
        out["timeouts"][level] = {key(d): _timeouts(entry, by_dial.get(d), counts) for d in columns}
    return out


def _largest(offers: dict):
    """The key with the largest offer, ties to the smallest key (reading K6); None if empty."""
    best = None
    for found in sorted(offers):
        if best is None or offers[found] > offers[best] + dial_splits.TIE:
            best = found
    return best


def _offers(entry: dict | None, dials: list, levels=LEVELS) -> dict:
    """(kappa, dial) -> mean tier A offer of the unit's defined pairs with a dial (K4, K7)."""
    if entry is None:
        return {}
    found = {}
    for kappa in levels:
        row = entry["pairs"][TIER][f"{kappa:g}"]
        for dial in dials:
            cell = row[key(dial)]
            if cell["defined"] and cell["mean_offer_kwh_per_test_day"] is not None:
                found[(kappa, dial)] = cell["mean_offer_kwh_per_test_day"]
    return found


def readings(units: dict, dials: list, levels=LEVELS, label: str = LABEL) -> dict:
    """What readings R1 and R2 give per site type (reading K5), over `levels`."""
    offers = {t: _offers(units.get(u), dials, levels) for t, u in SITE_TYPE_UNITS.items()}
    r1 = {}
    for site_type, unit in SITE_TYPE_UNITS.items():
        pair = _largest(offers[site_type])
        r1[site_type] = None
        if pair is not None:
            r1[site_type] = {
                "unit": unit,
                "kappa": pair[0],
                "dial": pair[1],
                "mean_offer": offers[site_type][pair],
            }
    sums = {}
    for kappa in levels:
        best = {
            t: dial_splits.pick_dial({d: o for (k, d), o in found.items() if k == kappa})
            for t, found in offers.items()
        }
        defined = all(d is not None for d in best.values())
        sums[kappa] = sum(offers[t][(kappa, d)] for t, d in best.items()) if defined else None
    chosen = _largest({k: s for k, s in sums.items() if s is not None})
    r2_dials = {t: None for t in SITE_TYPE_UNITS}
    if chosen is not None:
        for site_type, unit in SITE_TYPE_UNITS.items():
            at = {d: o for (k, d), o in offers[site_type].items() if k == chosen}
            dial = dial_splits.pick_dial(at)
            r2_dials[site_type] = {"unit": unit, "dial": dial, "mean_offer": at[dial]}
    differ = []
    for site_type in SITE_TYPE_UNITS:
        one = None if r1[site_type] is None else (r1[site_type]["kappa"], r1[site_type]["dial"])
        two = None if chosen is None else (chosen, r2_dials[site_type]["dial"])
        if one != two:
            differ.append(site_type)
    return {
        "R1": r1,
        "R2": {
            "kappa": chosen,
            "sums_by_kappa": {f"{k:g}": s for k, s in sums.items()},
            "dials": r2_dials,
        },
        "differ_at": differ,
        "label": label,
    }


def joint(runs: list[Run]) -> dict:
    """The joint document of runs that `check_runs` accepts (module text)."""
    ordered = sorted(runs, key=lambda r: r.document["run"]["kappa"])
    first = ordered[0].document
    dials = list(first["run"]["dials"])
    tiers = list(first["run"]["tiers"])
    columns = [None, *dials]
    units = {unit: _unit(unit, ordered, columns, tiers) for unit in first["units"]}
    return {
        "label": f"kappa choice: {LABEL}",
        "definitions": DEFINITIONS,
        "run": {
            "model": first.get("model"),
            "fit_units": first.get("fit_units"),
            "matrix": first.get("matrix"),
            "inputs": first.get("inputs"),
            "commit": ordered[0].source["commit"],
            "splits": first["run"]["splits"],
            "seeds": first["run"]["seeds"],
            "scales": first["run"]["scales"],
            "tiers": tiers,
            "rung": first["run"]["rung"],
            "kappas": [r.document["run"]["kappa"] for r in ordered],
            "dials": dials,
            "site_types": dict(SITE_TYPE_UNITS),
        },
        "runs": [{"kappa": r.document["run"]["kappa"], **r.source} for r in ordered],
        "units": units,
        "run_timeouts": {
            f"{r.document['run']['kappa']:g}": r.document["timeouts"] for r in ordered
        },
        "readings": readings(units, dials),
    }


def what_if_block(document: dict, what_if: list[Run]) -> dict:
    """The key `what_if` of the joint document `document` of the grid's runs, from the what-if
    runs (reading K9). What-if runs of one fit (reading K10) make one block; runs of different
    fits make one block each, under `separate_fits`, and no figure is read across two blocks."""
    groups: dict[tuple, list[Run]] = {}
    for found in what_if:
        groups.setdefault(fit_group(found), []).append(found)
    blocks = [_what_if_group(document, runs) for _, runs in sorted(groups.items())]
    if len(blocks) == 1:
        return blocks[0]
    return {
        "label": WHAT_IF_LABEL,
        "definitions": {**WHAT_IF_DEFINITIONS, **FIT_DEFINITIONS},
        "separate_fits": blocks,
    }


def _what_if_group(document: dict, what_if: list[Run]) -> dict:
    """The block of the what-if runs `what_if` of one fit (`what_if_block`)."""
    ordered = sorted(what_if, key=lambda r: r.document["run"]["kappa"])
    fit = fit_of(ordered[0])
    dials = list(document["run"]["dials"])
    tiers = list(document["run"]["tiers"])
    columns = [None, *dials]
    units = {u: _unit(u, ordered, columns, tiers) for u in ordered[0].document["units"]}
    off_grid = [
        r.document["run"]["kappa"] for r in ordered if r.document["run"]["kappa"] not in LEVELS
    ]
    levels = sorted({*LEVELS, *off_grid})
    merged = {}
    for unit, entry in document["units"].items():
        pairs = dict(entry["pairs"][TIER])
        pairs |= {f"{k:g}": units[unit]["pairs"][TIER][f"{k:g}"] for k in off_grid}
        merged[unit] = {"pairs": {TIER: pairs}}
    over_all = readings(merged, dials, levels, WHAT_IF_LABEL)
    block = {
        "label": WHAT_IF_LABEL,
        "definitions": WHAT_IF_DEFINITIONS
        if fit is None
        else {**WHAT_IF_DEFINITIONS, **FIT_DEFINITIONS},
        "kappas": [r.document["run"]["kappa"] for r in ordered],
        "runs": [
            {
                "kappa": r.document["run"]["kappa"],
                "what": r.document["what_if"]["what"],
                **({} if fit is None else {"fit": fit_of(r)}),
                **r.source,
            }
            for r in ordered
        ],
        "units": units,
        "run_timeouts": {
            f"{r.document['run']['kappa']:g}": r.document["timeouts"] for r in ordered
        },
        "readings_over_all_levels": {"levels": levels, **over_all},
    }
    if fit is not None:
        block["fit"] = fit
        block["readings_over_all_levels"]["mixes"] = (
            "the grid's levels are read from the plan's six runs (fitted at the plan's six "
            f"levels), the levels off the grid from these what-if runs ({fit_text(fit)})"
        )
    return block


# --- the command ---------------------------------------------------------------------------------


def _json(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        found = [read_run(path) for path in args.results]
        runs = [r for r in found if not r.what_if]
        what_if = [r for r in found if r.what_if]
        check_runs(runs)
        if what_if:
            check_what_if(runs, what_if)
    except CannotCheck as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except NotVerified as problem:
        print(f"REFUSED: {problem}")
        return 1
    except CouldNotRun as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    document = joint(runs)
    if what_if:
        document["what_if"] = what_if_block(document, what_if)
    text = "\n".join(kappa_summary.summary_lines(document)) + "\n"
    print(text, end="")
    run_dir = os.environ.get("FLEETRESERVE_RUN_DIR")
    if run_dir:
        folder = Path(run_dir)
        (folder / RESULTS_NAME).write_text(_json(document))
        (folder / SUMMARY_NAME).write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

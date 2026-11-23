"""The settings file of the drift guard's step size eta: written from the results of a recorded
`fleetreserve.drift_run` on training units, by a rule named on the command line (plan section 6,
T10: "η set in WP9").

Command line, from the project's top folder:
    python -m fleetreserve.drift_settings <run folder>/drift-run.json --rule <rule>
        --start <calibrated|zero> --grouping <all-units|site-type> [--out learn/drift-eta.json]

With beta_1 = 0 the bound of T10, alpha + (1 - alpha)/t, does not depend on eta, so the bound
alone sets no eta. Plan section 6, T10, states the rule: "β_1=0 (set here), η set in WP9: per
tier and per site type ..., from the candidate step sizes of the chronological run on the
training units from β_1=0, taken in increasing order: the largest one below the first that
fails, or the largest one tried when none fails, a step size failing when its mean losses do not
end within their levels on every unit of the type (slot loss; in Tier A also driver loss)", and
"Where no candidate step size is valid (the smallest already fails), the drift guard's run offers
nothing for that tier and site type" (the owner's decisions of 18 November 2026, which replace
the rule of 16 November 2026, this module's E10). `drift_run` reads that as this module's rule
`largest-below-first-failure` (reading E25) with the start `zero` and the grouping `site-type`,
and accepts no other in the file it reads on the sealed route (its reading E17). This module
names four rules (readings E10 to E12 and E25) and writes a file only for the rule, start and
grouping named on the command line.

Grouping (`GROUPINGS`, named on the command line, with no default):
- `all-units`: the figures and the rule over every unit of the run, one step size per tier (the
  file described below);
- `site-type`: read per site type as the dial is (plan section 6: "workplaces: from NREL from May
  2019; Norway and Korea: from the pooled Norwegian development sites"), here workplaces from the
  run's workplace training units and Norway and Korea (the type "other") from its Norwegian ones:
  per tier and site type, the figures and the rule on that type's units apart (reading E14).

Figures (`figures`), per tier and step size, from the results' path summaries of the start named,
over the units of the run, or of one site type (reading E13):
- `within_alpha`: on every unit with at least one guarded step, the final running mean of each
  loss the tier holds down is at most its alpha (tier A: slot 0.10 and driver 0.05; tier B: slot
  0.10), the summaries' `within_alpha`;
- `offer`: the mean offer per guarded counting day, summed over the units (kWh per day);
- `abstaining_share`: the guard's abstaining days over the offering guarded days, pooled over the
  units.
The tiers are read apart and never pooled: the file holds one step size per tier (with
`site-type`, one per tier and site type). `figures` reads a step size only where a unit has a
guarded step and skips a unit with no summary or no guarded step there; the rule E25 reads
`grid_figures` instead (readings E26 to E28), the same three figures with, per step size, the
units it fails on (`failing`).

Rules (`RULES`), each giving a step size of the run's grid or none (E25: or `OFFERS_NOTHING`); a
difference of at most `TIE` is a tie:
  E10. `largest-offer-within-alpha`: among the step sizes within alpha, the largest offer; a tie
       goes to the smaller step size. It asks the run's own days to show what T10 promises only
       in the long run. [Alternative: E11.]
  E11. `largest-offer`: the largest offer, with no loss condition: T10 bounds the average loss of
       every step size by alpha + (beta_1/eta + 1 - alpha)/t on any sequence, so the condition
       adds nothing in the long run; a tie goes to the smaller step size. [Alternative: E10.]
  E12. `fewest-abstaining-within-alpha`: among the step sizes within alpha, the smallest
       abstaining share (an offer that is there on more days); a tie goes to the larger offer,
       then to the smaller step size. [Alternative: E10.]
  E13. Across units: the offers are summed, so a unit counts by its kWh, and the loss condition
       must hold on every unit. [Alternatives: each unit's offer as a share of its own best step
       size, averaged; the loss condition on the pooled mean.]
  E14. A unit's site type (`site-type`) is `learn_features.SITE_TYPES` of its name, a training
       unit being one site under its own name (`truth_build`), and the types are named as in
       `t7_summary.TYPE_NAMES` ("workplace", "other"). `SITE_TYPES` is the code's one mapping of
       sites to site types (plan section 2's split, "workplaces (ACN, NREL, a.s.r.)" against
       Norway and Korea, the split section 6 fixes the dial by); the dial's own command,
       `dial_choice`, measures per unit and names no type, and the gate units' dial sources
       (`DIAL_SOURCES` of `tools/compare_shared_check.py`) name no training unit. Per type, the
       file lists under `units` the type's units whose days reached the guard, the units its
       figures are read from, and under `units_left_out` the others, each with the status its
       run recorded (too few counting days). [Alternative: a mapping of units to the dial's site
       types written here, beside `DIAL_SOURCES`.]
  E15. Readers: `step_size` gives a scored unit's step size from a file of either form: the
       tier's one value for every site type, or the value of the unit's site type, which the
       reader takes from the unit's sites (`learn_features.SITE_TYPES`, as `size_whatif.site_type`
       reads it). `drift_run` reads the file on the sealed route, as the tag's commit holds it,
       and takes each tier's step size for the units' site type through `step_size`.
       [Alternative: no reading function until a command reads the file.]
  E25. `largest-below-first-failure`, the plan's rule of 18 November 2026: the step sizes of
       `grid_figures`, taken in increasing order; the first that fails (not within alpha on
       every unit judged, E27 and E28) ends the scan, and the rule gives the step size before
       it, or the largest when none fails. A step size that passes again after the first failure
       is never chosen, and the offer plays no part. When the smallest already fails it gives
       `OFFERS_NOTHING` ("offers nothing"), a value of its own that the file records for that
       tier (and site type) in place of a step size: the drift guard's run then offers nothing
       there (`step_size` returns it; `drift_run` reading E29). With no unit judged it gives
       none, as the other rules do with no figures. [Alternative: E10, the rule of 16 November
       2026, which the plan's text replaces.]
  E26. A step size of the run's grid (`run.etas`, the candidate step sizes of the chronological
       run) at which no unit judged has a guarded step, which `figures` drops, is kept by
       `grid_figures` and fails: no mean loss there is shown to end within its level, and
       dropping it would let the scan step over it to a larger step size. On a results file
       `drift_run` writes this changes no choice: a guarded day is a step at every step size or
       at none (`drift_path` reading E3: it is a step when it offers, whatever eta), and every
       unit that ran has a summary at every step size of the grid, so a unit with a guarded step
       at one step size has one at all. [Alternative: drop it, as `figures` does, and take the
       scan on to the next step size.]
  E27. A unit judged (E28) that at one step size has no summary or no guarded step, which
       `figures` skips there, makes that step size fail: the plan asks for mean losses within
       their levels "on every unit of the type", and that unit's are not shown there; skipped,
       it would let a step size pass on the other units and hide the first failure. As in E26,
       a results file `drift_run` writes has no such unit. [Alternatives: skip the unit at that
       step size, as `figures` does; or refuse the file (could not run).]
  E28. The units judged are the units read (with `site-type`, the type's `units`) that have a
       guarded step at some step size of the grid. The others count at no step size: a unit
       left out (`units_left_out`, kept in the file with its status) has no more counting days
       than the 50 that calibrate, so T10 takes no step on it, and a unit whose guarded days
       never offer takes none either; neither has a mean loss at any step size, the same at
       every step size, so neither can make one step size fail before another. The plan's
       chronological run is "first 50 days calibrate, then T10", and its levels bound T10's
       mean losses. The file names the units judged under `units_judged`. [Alternative: such a
       unit fails every step size, and its type offers nothing.]

It writes `ETA` (`learn/drift-eta.json`, or `--out`). With `all-units`: `rule` and its text,
`start`, `eta` (tier -> step size), the figures it was chosen from (tier -> step size -> figures),
the units, model and dial of the run, and `results`: the results file's name and SHA-256, its run
folder's name and the commit read from that run's manifest. With `site-type`, also `grouping` and
its text, and per tier one step size per site type: `eta` (tier -> site type -> step size),
`figures` (tier -> site type -> step size -> figures) and `site_types` (site type -> the units
read and the units left out). With the rule E25, a value of `eta` is a step size or "offers
nothing", the figures cover every step size of the run's grid, each with `units_failing` (unit
-> why the step size fails on it), and `units_judged` gives per tier (and site type) the units
the rule judged (E28). It writes nothing, and says why, unless all of these hold:
- the results file is verified against its run's manifest (`run_verify.require_verified`: exit
  code 0, no time limit hit, not interrupted, the file listed with the hash it has now, its
  commit exists) and lies directly in its run folder;
- it is a results file of `drift_run` (its label, and the keys read here);
- every unit of the run is a training unit (`truth_build.TRAIN_UNITS`): step sizes are chosen on
  training units, never on a development or held-out unit;
- the start named is one the run measured, and in each tier the rule finds a step size (with
  `site-type`: in each tier and site type, which also needs every unit to have a site type and
  each type a unit that has a guarded step); for the rule E25, "offers nothing" is a finding
  and is written.

Exit status: 0 the file was written; 1 it was refused (with `all-units`, also when the rule finds
no step size in a tier, as before the grouping existed); 2 could not run (the results file or its
manifest is missing or unreadable; an unknown rule, start or grouping, or none named, from the
command line; with `site-type`, a unit with no site type, or a site type with no unit that has a
guarded step or where the rule finds no step size: never a default, and never a value borrowed
from the other type or from all units).
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

from fleetreserve import drift_path, drift_run
from fleetreserve.learn_features import SITE_TYPES
from fleetreserve.run_record import MANIFEST_NAME
from fleetreserve.run_verify import CannotCheck, NotVerified, require_verified
from fleetreserve.sessions import TOP
from fleetreserve.t7_summary import TYPE_NAMES
from fleetreserve.truth_build import TRAIN_UNITS

ETA = TOP / "learn" / "drift-eta.json"
TIE = 1e-9
OFFERS_NOTHING = "offers nothing"  # rule E25's value where the smallest step size fails
FIRST_FAILURE = "largest-below-first-failure"  # the rule of reading E25
OUTSIDE = "a guarded loss's final mean is above its level"  # why a step size fails (E25)
NO_STEP = "no guarded step at this step size"  # why a step size fails on a unit (E26, E27)
ALL_UNITS, SITE_TYPE = "all-units", "site-type"
GROUPINGS = (ALL_UNITS, SITE_TYPE)
RAN = "run"  # `drift_path.UnitRun.status` of a unit whose days reached the guard
ABOUT = (
    "The step size eta of the drift guard (plan section 6, T10), per tier, written by "
    "fleetreserve.drift_settings from the results of a recorded fleetreserve.drift_run on "
    "training units, by the rule named here. Not edited by hand."
)
ABOUT_BY_TYPE = (
    "The step size eta of the drift guard (plan section 6, T10), per tier and site type, written "
    "by fleetreserve.drift_settings from the results of a recorded fleetreserve.drift_run on "
    "training units, by the rule named here applied to each site type's units apart. Not edited "
    "by hand."
)
GROUPING_TEXT = (
    "per site type, as the dial is read (plan section 6): workplaces from the workplace training "
    "units of the run, Norway and Korea (the type other) from its Norwegian training units; the "
    "figures and the rule on each type's units apart; a unit's type is learn_features.SITE_TYPES "
    "of its name (reading E14)"
)


class Refused(Exception):
    """The settings file must not be written; the message says why."""


class CouldNotRun(Exception):
    """Per site type, a step size cannot be read (module text); the command could not run."""


class Figures(NamedTuple):
    """One tier and step size (module text)."""

    within_alpha: bool
    offer: float
    abstaining_share: float | None
    failing: tuple = ()  # `grid_figures` only: (unit, why) of each unit it fails on


def _best(candidates: dict, better) -> float | None:
    """The step size of `candidates` (eta -> Figures) that no later one beats by `better(new,
    old)`, scanning in increasing eta, so a tie keeps the smaller step size."""
    best = None
    for eta in sorted(candidates):
        if best is None or better(candidates[eta], candidates[best]):
            best = eta
    return best


def _more_offer(new: Figures, old: Figures) -> bool:
    return new.offer > old.offer + TIE


def _fewer_abstaining(new: Figures, old: Figures) -> bool:
    share_new = 1.0 if new.abstaining_share is None else new.abstaining_share
    share_old = 1.0 if old.abstaining_share is None else old.abstaining_share
    if share_new < share_old - TIE:
        return True
    return abs(share_new - share_old) <= TIE and _more_offer(new, old)


def largest_offer_within_alpha(figures: dict) -> float | None:
    return _best({e: f for e, f in figures.items() if f.within_alpha}, _more_offer)


def largest_offer(figures: dict) -> float | None:
    return _best(figures, _more_offer)


def fewest_abstaining_within_alpha(figures: dict) -> float | None:
    return _best({e: f for e, f in figures.items() if f.within_alpha}, _fewer_abstaining)


def largest_below_first_failure(figures: dict) -> float | str | None:
    """Reading E25: in increasing step size, the one before the first that fails, the largest
    when none fails, `OFFERS_NOTHING` when the smallest fails; none with no figures."""
    if not figures:
        return None
    chosen = OFFERS_NOTHING
    for eta in sorted(figures):
        if not figures[eta].within_alpha:
            break
        chosen = eta
    return chosen


RULES = {
    "largest-offer-within-alpha": largest_offer_within_alpha,
    "largest-offer": largest_offer,
    "fewest-abstaining-within-alpha": fewest_abstaining_within_alpha,
    FIRST_FAILURE: largest_below_first_failure,
}
RULE_TEXTS = {
    "largest-offer-within-alpha": (
        "among the step sizes whose final mean of every guarded loss is within its alpha on "
        "every training unit, the largest offer per counting day summed over the units; a tie "
        "goes to the smaller step size (reading E10)"
    ),
    "largest-offer": (
        "the largest offer per counting day summed over the training units, with no loss "
        "condition; a tie goes to the smaller step size (reading E11)"
    ),
    "fewest-abstaining-within-alpha": (
        "among the step sizes within alpha on every training unit, the smallest share of "
        "offering days the guard abstains, pooled over the units; a tie goes to the larger "
        "offer, then to the smaller step size (reading E12)"
    ),
    FIRST_FAILURE: (
        "from the candidate step sizes of the chronological run on the training units from "
        "beta_1 = 0, taken in increasing order: the largest one below the first that fails, or "
        "the largest one tried when none fails, a step size failing when its mean losses do not "
        "end within their levels on every unit of the type (slot loss; in tier A also driver "
        "loss); where the smallest already fails, the drift guard's run offers nothing for that "
        "tier and site type (plan section 6, T10; readings E25 to E28)"
    ),
}


def figures(document: dict, tier: str, start: str, units=None) -> dict:
    """Step size -> `Figures` of one tier and start, over the run's units, or over `units` alone
    when given (module text)."""
    out = {}
    for eta in document["run"]["etas"]:
        key = drift_run.eta_key(float(eta))
        within, offer, abstaining, offering, any_steps = True, 0.0, 0, 0, False
        for unit, entry in document["units"].items():
            if units is not None and unit not in units:
                continue
            summary = entry["paths"].get(tier, {}).get(start, {}).get(key)
            if summary is None:
                continue
            if summary["mean_offer_kwh_per_counting_day"] is not None:
                offer += summary["mean_offer_kwh_per_counting_day"]
            if summary["offering_days"]:
                any_steps = True
                offering += summary["offering_days"]
                abstaining += summary["guard_abstaining_days"]
                guarded = [e for e in summary["losses"].values() if e["guarded"]]
                within = within and all(e["within_alpha"] for e in guarded)
        if any_steps:
            out[float(eta)] = Figures(within, offer, abstaining / offering if offering else None)
    return out


def judged_units(document: dict, tier: str, start: str, units=None) -> list:
    """The units the rule E25 judges (reading E28): those read (all, or `units`) with a guarded
    step at some step size of the run's grid, in the run's order."""
    keys = [drift_run.eta_key(float(eta)) for eta in document["run"]["etas"]]
    judged = []
    for unit, entry in document["units"].items():
        if units is not None and unit not in units:
            continue
        by_eta = entry["paths"].get(tier, {}).get(start, {})
        if any((by_eta.get(key) or {}).get("offering_days") for key in keys):
            judged.append(unit)
    return judged


def grid_figures(document: dict, tier: str, start: str, units=None) -> dict:
    """Step size -> `Figures` of one tier and start at every step size of the run's grid, judged
    on `judged_units` (readings E26 to E28): a step size fails on a unit judged whose guarded
    final mean is above its level there, or that has no summary or no guarded step there; {}
    when no unit is judged."""
    judged = judged_units(document, tier, start, units)
    if not judged:
        return {}
    out = {}
    for eta in document["run"]["etas"]:
        key = drift_run.eta_key(float(eta))
        failing, offer, abstaining, offering = {}, 0.0, 0, 0
        for unit in judged:
            summary = document["units"][unit]["paths"].get(tier, {}).get(start, {}).get(key)
            if summary is None or not summary["offering_days"]:
                failing[unit] = NO_STEP
                continue
            if summary["mean_offer_kwh_per_counting_day"] is not None:
                offer += summary["mean_offer_kwh_per_counting_day"]
            offering += summary["offering_days"]
            abstaining += summary["guard_abstaining_days"]
            if not all(e["within_alpha"] for e in summary["losses"].values() if e["guarded"]):
                failing[unit] = OUTSIDE
        share = abstaining / offering if offering else None
        out[float(eta)] = Figures(not failing, offer, share, tuple(failing.items()))
    return out


def _table(found: dict) -> dict:
    """Step size -> the figures, as the settings file holds them."""
    return {
        drift_run.eta_key(e): {
            "within_alpha_on_every_unit": f.within_alpha,
            "offer_kwh_per_counting_day_summed_over_units": f.offer,
            "share_guard_abstaining_of_offering_days": f.abstaining_share,
        }
        for e, f in sorted(found.items())
    }


def _grid_table(found: dict) -> dict:
    """`_table` of `grid_figures`, with each step size's `units_failing` (rule E25)."""
    table = _table(found)
    for eta, f in found.items():
        table[drift_run.eta_key(eta)]["units_failing"] = dict(f.failing)
    return table


def _choose(document: dict, rule: str, tier: str, start: str, units=None):
    """(the rule's value, its figures as the file holds them, the units judged or None) of one
    tier, over the run's units or `units`; the value None when the rule finds no step size."""
    if rule == FIRST_FAILURE:
        found = grid_figures(document, tier, start, units)
        judged = judged_units(document, tier, start, units)
        return RULES[rule](found), _grid_table(found), judged
    found = figures(document, tier, start, units)
    return RULES[rule](found), _table(found), None


def eta_text(value) -> str:
    """A value of a settings file's `eta`: a step size, or "offers nothing" (rule E25)."""
    return OFFERS_NOTHING if value == OFFERS_NOTHING else f"{value:g}"


def site_types(document: dict) -> dict:
    """Site type name -> `units` (the type's units whose days reached the guard) and
    `units_left_out` (unit -> the status its run recorded), in the run's order (reading E14);
    raises `CouldNotRun` for a unit with no site type."""
    entries = document["units"]
    unknown = [u for u in entries if u not in SITE_TYPES]
    if unknown:
        raise CouldNotRun(
            f"the units {unknown} have no site type in learn_features.SITE_TYPES, so their step "
            "size per site type is not known"
        )
    grouped = {}
    for kind, name in TYPE_NAMES.items():
        status = {u: entry["status"] for u, entry in entries.items() if SITE_TYPES[u] == kind}
        grouped[name] = {
            "units": [u for u, s in status.items() if s == RAN],
            "units_left_out": {u: s for u, s in status.items() if s != RAN},
        }
    return grouped


def _by_site_type(document: dict, rule: str, start: str) -> dict:
    """The fields of a `site-type` file that differ from an `all-units` one (module text); raises
    `CouldNotRun`."""
    grouped = site_types(document)
    etas = list(document["run"]["etas"])
    chosen, table, judged = {}, {}, {}
    for tier in drift_path.TIERS:
        chosen[tier], table[tier], judged[tier] = {}, {}, {}
        for name, group in grouped.items():
            eta, found, judged[tier][name] = _choose(document, rule, tier, start, group["units"])
            if not found:
                raise CouldNotRun(
                    f"site type {name} in tier {tier} has no unit that has a guarded step (start "
                    f"{start}): units read {group['units']}, left out "
                    f"{sorted(group['units_left_out'])}"
                )
            if eta is None:
                raise CouldNotRun(
                    f"the rule {rule} finds no step size for site type {name} in tier {tier} "
                    f"among {etas} (start {start})"
                )
            chosen[tier][name] = eta
            table[tier][name] = found
    by_type = {
        "about": ABOUT_BY_TYPE,
        "grouping": SITE_TYPE,
        "grouping_text": GROUPING_TEXT,
        "eta": chosen,
        "figures": table,
        "site_types": grouped,
    }
    return {**by_type, "units_judged": judged} if rule == FIRST_FAILURE else by_type


def eta_document(results_path: Path, rule: str, start: str, grouping: str) -> dict:
    """The content of the settings file (module text). Raises `NotVerified`, `Refused` or, with
    `site-type`, `CouldNotRun`."""
    if grouping not in GROUPINGS:
        raise ValueError(f"the grouping must be one of {GROUPINGS}, got {grouping!r}")
    file = Path(os.path.abspath(results_path))
    manifest = require_verified(file)
    if not (file.parent / MANIFEST_NAME).is_file():
        raise Refused(f"{file.name} does not lie directly in its run folder")
    content = file.read_bytes()
    try:
        document = json.loads(content)
        if document["label"] != drift_run.LABEL:
            raise ValueError("its label is not that of drift_run")
        run, units = document["run"], document["units"]
        named = list(run["units"])
        starts, etas = list(run["starts"]), list(run["etas"])
        if not all(isinstance(entry["paths"], dict) for entry in units.values()):
            raise ValueError("a unit's paths are not a mapping")
    except (ValueError, KeyError, TypeError) as problem:
        raise Refused(f"{file.name} is not a results file of drift_run: {problem}") from problem
    outside = [u for u in dict.fromkeys([*named, *units]) if u not in TRAIN_UNITS]
    if outside:
        raise Refused(
            f"{file.name} holds {outside}, which are not training units; step sizes are chosen "
            f"on training units only ({', '.join(TRAIN_UNITS)})"
        )
    if document.get("failed_units"):
        raise Refused(f"{file.name} records failed units: {document['failed_units']}")
    if start not in starts:
        raise Refused(f"the run measured the starts {starts}, not {start!r}")
    common = {
        "about": ABOUT,
        "rule": rule,
        "rule_text": RULE_TEXTS[rule],
        "start": start,
        "units": named,
        "model": run.get("model"),
        "dial": run.get("dial"),
        "results": {
            "file": file.name,
            "sha256": hashlib.sha256(content).hexdigest(),
            "run_folder": file.parent.name,
            "commit": manifest["commit"],
        },
    }
    if grouping == SITE_TYPE:
        return {**common, **_by_site_type(document, rule, start)}
    chosen, table, judged = {}, {}, {}
    for tier in drift_path.TIERS:
        eta, table[tier], judged[tier] = _choose(document, rule, tier, start)
        if eta is None:
            raise Refused(
                f"the rule {rule} finds no step size in tier {tier} among {etas} (start {start})"
            )
        chosen[tier] = eta
    written = {**common, "eta": chosen, "figures": table}
    return {**written, "units_judged": judged} if rule == FIRST_FAILURE else written


def step_size(settings: dict, tier: str, site_type: int) -> float | str:
    """Tier `tier`'s step size for a scored unit of site type `site_type` (a value of
    `learn_features.SITE_TYPES`), from a settings file's content of either form (reading E15): a
    file without `grouping` holds one value per tier, the same for every site type; or
    `OFFERS_NOTHING` where the file records it (reading E25)."""
    grouping = settings.get("grouping", ALL_UNITS)
    if grouping == ALL_UNITS:
        value = settings["eta"][tier]
    elif grouping == SITE_TYPE:
        value = settings["eta"][tier][TYPE_NAMES[site_type]]
    else:
        raise ValueError(f"a settings file of the unknown grouping {grouping!r}")
    return OFFERS_NOTHING if value == OFFERS_NOTHING else float(value)


def _chosen_text(document: dict) -> str:
    if document.get("grouping") == SITE_TYPE:
        return "; ".join(
            f"tier {t} "
            + ", ".join(f"{name} {eta_text(eta)}" for name, eta in document["eta"][t].items())
            for t in drift_path.TIERS
        )
    return ", ".join(f"tier {t} {eta_text(document['eta'][t])}" for t in drift_path.TIERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--rule", required=True, choices=sorted(RULES))
    parser.add_argument("--start", required=True, choices=drift_path.STARTS)
    parser.add_argument("--grouping", required=True, choices=GROUPINGS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = ETA if args.out is None else args.out
    try:
        document = eta_document(args.results, args.rule, args.start, args.grouping)
    except (CannotCheck, CouldNotRun) as problem:
        print(f"COULD NOT RUN: {problem}")
        return 2
    except (NotVerified, Refused) as problem:
        print(f"REFUSED, nothing written: {problem}")
        return 1
    out.write_text(json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n")
    how = args.rule if args.grouping == ALL_UNITS else f"{args.rule} per site type"
    print(
        f"Wrote {out.name}: {_chosen_text(document)}, by {how} (start {args.start}), from "
        f"{document['results']['run_folder']} at commit {document['results']['commit'][:7]}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

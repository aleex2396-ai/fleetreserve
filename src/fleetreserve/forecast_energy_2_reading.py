"""The pairing and the reading rule of the forecast trial `energy-2` (`forecast_energy_2`; a trial
made after the sealed run, `after_sealed_run`), fixed before any result (the `reading` block and
`bars` of `trials-after-sealed-run/forecast-energy-2.json`). The first energy trial's rule
(`forecast_energy_reading`) is applied once to a new comparison; its functions are imported and are
not changed.

Five arms are measured as `dial_choice` measures a model, on the same units, dial family, rung and
seeded splits, with the same fitted model and forecast table; they differ in the energy bounds only:

1. `known`: the car's true top charging speed, storable energy and needed energy;
2. `empirical`: those three bounds as quantiles at the run's level (0.2);
3. `cautious`: the top charging speed and the storable energy at level 0.05, the needed energy as
   in arm 2 (the first energy trial's value; re-run here so that every arm is paired split by
   split; read by no rule of this trial);
4. `cautious-need`: the needed energy alone at level 0.05 (the 0.95 quantile), the other two
   bounds as in arm 2 (information only);
5. `cautious-all`: all three bounds at level 0.05 (the judged arm).

Pairing (`compare`). Per unit that can be split and per tier, each arm is read at its own dial (the
dial of `dial_splits.pick_dial`: the largest mean offer of the arm over the splits, a tie to the
smaller dial). Two cells are paired split by split, by `forecast_mix_pairs.paired_cell` (seeds
checked equal), arm 2 as the main side: the JUDGED comparison with arm 5 as the trial side, and the
INFORMATION comparison with arm 4 as the trial side. The difference is the trial arm minus arm 2 of
the calibrated offer per counting test day, a split with no safe scale counting as an offer of 0.
Arm 1's own-dial offers are paired with arm 2's in a further cell, which checks that the splits
are the same. The mean offers of all five arms are given, each also as a share of arm 1's mean
offer; for each comparison also the difference as a share of arm 1's mean offer and as a share of
the gap between arm 1 and arm 2 (none unless that gap is above 0), and the test losses of the trial
arm. The two tiers are separate blocks and are never pooled or averaged.

What is judged: tier A, the judged comparison only (arm 5 minus arm 2), per judged unit (a unit with
a bar). With G the mean paired difference over the splits, the unit's bar is its share times arm 1's
mean offer:

- "shows a gain" if G is above the bar and arm 5's mean test slot loss is at most the slot limit
  (0.10) and its mean test driver loss at most the driver limit (0.05) (the test losses of the
  tier A summary at arm 5's own dial);
- "shows a loss" if G is below minus the bar;
- otherwise "no measurable difference": a share equal to a bar is inside it; also a share above
  the bar with a loss limit exceeded or undefined, where the output says so (`note`), and a unit
  with no figure.

A unit without a bar is printed and not judged. A judged unit whose arm 1 mean offer is 0 is also
printed and NOT JUDGED (no bar can be defined): it gets no word and the one-line reading does not
read it (the first trial's reader, which is not changed, gives such a unit "no measurable
difference"). The one-line reading is from the words of the judged units: "all three energy bounds
at level 0.05 help" when at least one unit shows a gain and none shows a loss; "... hurt" when at
least one shows a loss and none shows a gain; otherwise "no measurable difference; a mixed result
is reported as mixed"; when no unit has a word at all (every judged unit not judged), the line says
that no unit was judged. The information comparison (arm 4 minus arm 2) is read by no rule: it has
no word, and its figures are not an input of `read_units` or `one_line`. Tier B is numbers only,
information. The bars are the neural model's seed spread at another level, an analogy and not this
trial's measured noise. No Dutch or Korean site is a development unit, so the trial cannot speak
for them (`NO_OTHER_SITES`).
"""

from fleetreserve import forecast_mix_pairs
from fleetreserve.forecast_energy_reading import (
    GAIN,
    LOSS,
    NO_OTHER_SITES,
    NONE,
    _mean,
    _ratio,
    own_dial,
    word,
)

ARMS = ("known", "empirical", "cautious", "cautious-need", "cautious-all")  # arms 1 to 5
REFERENCE_ARM = "known"  # arm 1: the figure the bars are a share of
MAIN_ARM = "empirical"  # arm 2: the side both comparisons are made against
JUDGED_ARM = "cautious-all"  # arm 5: the one judged comparison
INFORMATION_ARM = "cautious-need"  # arm 4: information only
COMPARISONS = {"judged": JUDGED_ARM, "information": INFORMATION_ARM}
HELPS = "all three energy bounds at level 0.05 help"
HURTS = "all three energy bounds at level 0.05 hurt"
MIXED = "no measurable difference; a mixed result is reported as mixed"
NO_JUDGED_UNIT = "no unit was judged: every judged unit is not judged (no bar can be defined)"
NOT_JUDGED = "not judged (the unit has no bar)"
NOT_JUDGED_NO_BAR = "not judged (arm 1's mean offer is 0, so no bar can be defined)"
INFORMATION_LABEL = "information only, read by no rule"
RULE = (
    "Reading rule, fixed before any result: ONE judged comparison, arm 5 (all three energy bounds "
    "at level 0.05) minus arm 2 (the empirical bounds), tier A, per judged unit, each arm at its "
    "own dial; G is the paired mean difference of the offer over the splits and the bar is the "
    "unit's share times the mean offer of arm 1 (known energy); it shows a gain only if G is "
    "above the bar and arm 5's mean test slot loss is at most the slot limit and its mean test "
    "driver loss at most the driver limit, a loss only if G is below minus the bar, else no "
    "measurable difference. One line: all three energy bounds at level 0.05 help only if one "
    "judged unit shows a gain and none a loss; they hurt if one shows a loss and none a gain; "
    "otherwise no measurable difference, and a mixed result is reported as mixed. A unit without "
    "a bar, and a unit where arm 1's mean offer is 0, is printed and not judged. Arm 4 (the "
    "needed energy alone at level 0.05) minus arm 2 is printed with the same figures as "
    "information only: it is read by no rule and has no word. Tier B is information."
)


def _cell(found: dict, trial: str) -> dict | None:
    """The paired cell of arm `trial` against arm 2, or None when an arm it needs has no dial."""
    needed = (REFERENCE_ARM, MAIN_ARM, trial)
    if any(found[arm] is None for arm in needed):
        return None
    cell = forecast_mix_pairs.paired_cell(found[MAIN_ARM]["entries"], found[trial]["entries"])
    known = _mean(found[REFERENCE_ARM]["entries"])
    main = cell["mean_main_offer_kwh_per_test_day"]
    gap = None if known is None or main is None else known - main
    difference = cell["mean_difference_kwh_per_test_day"]
    return {
        "trial_arm": trial,
        "cell": cell,
        "difference_share_of_known_offer": _ratio(difference, known),
        "recovered_share_of_the_gap": _ratio(difference, gap),
        "test_losses": {
            "slot": found[trial]["summary"]["mean_test_slot_loss"],
            "driver": found[trial]["summary"]["mean_test_driver_loss"],
        },
    }


def tier_block(found: dict) -> dict:
    """The figures of one unit and tier from `own_dial` of the five arms (a dict arm -> figures
    or None): module text. `known_mean`: arm 1's mean offer (None without a dial). `means` and
    `share_of_known_offer` give every arm that has a dial; `judged` and `information` are the
    paired cells of arm 5 and of arm 4 against arm 2, each None unless arms 1, 2 and the trial arm
    have a dial."""
    known = None if found[REFERENCE_ARM] is None else _mean(found[REFERENCE_ARM]["entries"])
    means = {arm: None if found[arm] is None else _mean(found[arm]["entries"]) for arm in ARMS}
    block = {
        "picked_dial": {arm: None if found[arm] is None else found[arm]["dial"] for arm in ARMS},
        "known_mean": known,
        "mean_offer_kwh_per_test_day": means,
        "share_of_known_offer": {arm: _ratio(means[arm], known) for arm in ARMS},
        "known_against_empirical": None,
        "judged": _cell(found, JUDGED_ARM),
        "information": _cell(found, INFORMATION_ARM),
    }
    if found[REFERENCE_ARM] is not None and found[MAIN_ARM] is not None:
        block["known_against_empirical"] = forecast_mix_pairs.paired_cell(
            found[REFERENCE_ARM]["entries"], found[MAIN_ARM]["entries"]
        )
    return block


def compare(documents: dict, offers: dict) -> dict:
    """The paired comparison of the five arms (module text): unit -> status and, per tier, the
    block of `tier_block`. `documents`: arm -> the results of `dial_choice.run`; `offers`: arm ->
    its per-split offers. `ValueError` when the arms are not of the same level, dials, seeds,
    scales, units or splits."""
    first = documents[ARMS[0]]
    for arm in ARMS[1:]:
        forecast_mix_pairs.check_same_run(first, documents[arm])
    found: dict = {}
    for unit, entry in first["units"].items():
        out = {"status": entry["status"], "counting_days": entry["counting_days"]}
        found[unit] = out
        if entry["status"] != "calibratable":
            continue
        out["tiers"] = {
            tier: tier_block(
                {arm: own_dial(documents[arm], offers[arm], unit, tier) for arm in ARMS}
            )
            for tier in first["run"]["tiers"]
        }
    return found


def share(block: dict | None) -> float | None:
    """G of the JUDGED comparison as a share of arm 1's mean offer; None without a block or a
    judged cell, or when arm 1's mean offer is not above 0."""
    if block is None or block["judged"] is None:
        return None
    return block["judged"]["difference_share_of_known_offer"]


def bar_defined(block: dict | None) -> bool:
    """False when the unit has a figure of arm 1 whose mean offer is not above 0: no bar can be
    defined for it (module text). True without a figure, where the first trial's wording stands."""
    if block is None or block["known_mean"] is None:
        return True
    return block["known_mean"] > 0


def read_units(comparison: dict, bars: dict, limits: dict) -> dict:
    """Per unit of `comparison`, its tier A word from the JUDGED comparison only (arm 5 minus arm
    2; module text). A judged unit (a bar in `bars`) gets the share, the bar, the word and a note;
    a unit without a bar gets `NOT_JUDGED` and no word; a judged unit where arm 1's mean offer is 0
    gets `NOT_JUDGED_NO_BAR` and no word; a judged unit missing from `comparison` or that cannot be
    split is given no share and no measurable difference. The information comparison is not read."""
    found = {}
    for unit in dict.fromkeys([*comparison, *bars]):
        block = comparison.get(unit, {}).get("tiers", {}).get("A")
        value = share(block)
        if unit not in bars:
            found[unit] = {"share": value, "bar": None, "word": None, "note": NOT_JUDGED}
            continue
        if not bar_defined(block):
            found[unit] = {
                "share": None,
                "bar": bars[unit],
                "word": None,
                "note": NOT_JUDGED_NO_BAR,
            }
            continue
        judged = None if block is None else block["judged"]
        losses = None if judged is None else judged["test_losses"]
        said, note = word(value, bars[unit], losses, limits)
        found[unit] = {"share": value, "bar": bars[unit], "word": said, "note": note}
    return found


def one_line(units: dict) -> str:
    """The one-line reading from the words of `read_units` (module text); units without a word
    (not judged) are not read; with no word at all the line says that no unit was judged."""
    words = [u["word"] for u in units.values() if u["word"] is not None]
    if not words:
        return NO_JUDGED_UNIT
    gains, losses = words.count(GAIN), words.count(LOSS)
    if gains and not losses:
        return HELPS
    if losses and not gains:
        return HURTS
    return MIXED


__all__ = [
    "ARMS",
    "COMPARISONS",
    "GAIN",
    "HELPS",
    "HURTS",
    "INFORMATION_ARM",
    "INFORMATION_LABEL",
    "JUDGED_ARM",
    "LOSS",
    "MAIN_ARM",
    "MIXED",
    "NONE",
    "NOT_JUDGED",
    "NOT_JUDGED_NO_BAR",
    "NO_JUDGED_UNIT",
    "NO_OTHER_SITES",
    "REFERENCE_ARM",
    "RULE",
    "bar_defined",
    "compare",
    "one_line",
    "read_units",
    "share",
    "tier_block",
]

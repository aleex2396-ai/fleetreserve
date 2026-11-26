"""The pairing and the reading rule of the forecast trial `energy` (`forecast_energy`; a trial made
after the sealed run, `after_sealed_run`), fixed before any result (the `reading` block and
`bars` of `trials-after-sealed-run/forecast-energy.json`).

Three arms are measured as `dial_choice` measures a model, on the same units, dial family, rung and
seeded splits, with the same fitted model and forecast table; they differ in the energy bounds only:

1. `known`: the car's true top charging speed, storable energy and needed energy;
2. `empirical`: those three bounds as quantiles at the run's level (0.2);
3. `cautious`: the top charging speed and the storable energy at level 0.05, the needed energy as
   in arm 2 (`forecast_empirical.CAUTIOUS`).

Pairing (`compare`). Per unit that can be split and per tier, each arm is read at its own dial (the
dial of `dial_splits.pick_dial`: the largest mean offer of the arm over the splits, a tie to the
smaller dial). The paired cell is arm 2 (as the main side) against arm 3 (as the trial side), split
by split, by `forecast_mix_pairs.paired_cell`: the difference arm 3 minus arm 2 of the calibrated
offer per counting test day, a split with no safe scale counting as an offer of 0. Arm 1's own-dial
offers are paired with arm 2's in a second cell, which checks that the splits are the same and
gives arm 1's mean offer, the figure the bars are a share of. Also given: arms 2's and 3's mean
offers as a share of arm 1's, the difference as a share of arm 1's mean offer, and the difference
as a share of the gap between arm 1 and arm 2 (none unless that gap is above 0). The two tiers are
separate blocks and are never pooled or averaged.

What is judged: tier A, per judged unit (a unit with a bar). With G the mean paired difference
(arm 3 minus arm 2) over the splits, the unit's bar is its share times arm 1's mean offer:

- "shows a gain" if G is above the bar and arm 3's mean test slot loss is at most the slot limit
  (0.10) and its mean test driver loss at most the driver limit (0.05) (the test losses of the
  tier A summary at arm 3's own dial);
- "shows a loss" if G is below minus the bar;
- otherwise "no measurable difference": a share equal to a bar is inside it; also a share above
  the bar with a loss limit exceeded or undefined, where the output says so (`note`), and a unit
  whose arm 1 mean offer is not above 0 or that has no figure, where no share is defined.

A unit without a bar is printed and not judged. The one-line reading is from the words of the
judged units: "the cautious bounds help" only when at least one unit shows a gain and none shows a
loss; "the cautious bounds hurt" when at least one shows a loss and none shows a gain; otherwise
"no measurable difference; a mixed result is reported as mixed". Tier B is numbers only,
information, and is not read by this rule. The bars are the neural model's seed spread at another
level, an analogy and not this trial's measured noise. No Dutch or Korean site is a development
unit, so the trial cannot speak for them (`NO_OTHER_SITES`).
"""

import numpy as np

from fleetreserve import crc, forecast_mix_pairs
from fleetreserve.forecast_mix_reading import GAIN, LOSS, NO_OTHER_SITES, NONE

ARMS = ("known", "empirical", "cautious")  # arm 1, arm 2, arm 3
HELPS = "the cautious bounds help"
HURTS = "the cautious bounds hurt"
MIXED = "no measurable difference; a mixed result is reported as mixed"
NOT_JUDGED = "not judged (the unit has no bar)"
RULE = (
    "Reading rule, fixed before any result: tier A, per judged unit, each arm at its own dial; G "
    "is the paired mean difference of the offer (arm 3, the cautious bounds, minus arm 2, the "
    "empirical bounds) over the splits and the bar is the unit's share times the mean offer of "
    "arm 1 (known energy); it shows a gain only if G is above the bar and arm 3's mean test slot "
    "loss is at most the slot limit and its mean test driver loss at most the driver limit, a loss "
    "only if G is below minus the bar, else no measurable difference. One line: the cautious "
    "bounds help only if one judged unit shows a gain and none a loss; they hurt if one shows a "
    "loss and none a gain; otherwise no measurable difference, and a mixed result is reported as "
    "mixed. A unit without a bar is printed and not judged. Tier B is information."
)


def _mean(entries: list[dict]) -> float | None:
    return float(np.mean([e["offer_kwh"] for e in entries])) if entries else None


def _name_at(configurations: dict, dial) -> str | None:
    """The one configuration of the dial `dial` that measured, or None."""
    if dial is None:
        return None
    names = [n for n, c in configurations.items() if c["driver_risk"] == dial and "tiers" in c]
    return names[0] if len(names) == 1 else None


def own_dial(document: dict, offers: dict, unit: str, tier: str) -> dict | None:
    """One arm's figures at its own dial (module text): the dial, the per-split entries and the
    tier summary of the results; None when the arm has no dial with a mean offer."""
    dial = document["dial_choice"].get(unit, {}).get(tier, {}).get("dial")
    name = _name_at(offers.get(unit, {}), dial)
    if name is None:
        return None
    return {
        "dial": dial,
        "entries": offers[unit][name]["tiers"][tier],
        "summary": document["units"][unit]["configurations"][name]["tiers"][tier],
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or not denominator > 0:
        return None
    return numerator / denominator


def tier_block(found: dict) -> dict:
    """The figures of one unit and tier from `own_dial` of the three arms (a dict arm -> figures
    or None): module text. The paired cells are none unless every arm has a dial."""
    block = {
        "picked_dial": {arm: None if found[arm] is None else found[arm]["dial"] for arm in ARMS},
        "cell": None,
        "known_against_empirical": None,
        "mean_offer_kwh_per_test_day": None,
        "share_of_known_offer": None,
        "difference_share_of_known_offer": None,
        "recovered_share_of_the_gap": None,
        "cautious_test_losses": None,
    }
    if any(found[arm] is None for arm in ARMS):
        return block
    cell = forecast_mix_pairs.paired_cell(
        found["empirical"]["entries"], found["cautious"]["entries"]
    )
    check = forecast_mix_pairs.paired_cell(found["known"]["entries"], found["empirical"]["entries"])
    known = _mean(found["known"]["entries"])
    empirical = cell["mean_main_offer_kwh_per_test_day"]
    cautious = cell["mean_trial_offer_kwh_per_test_day"]
    gap = None if known is None or empirical is None else known - empirical
    block.update(
        {
            "cell": cell,
            "known_against_empirical": check,
            "mean_offer_kwh_per_test_day": {
                "known": known,
                "empirical": empirical,
                "cautious": cautious,
            },
            "share_of_known_offer": {
                "empirical": _ratio(empirical, known),
                "cautious": _ratio(cautious, known),
            },
            "difference_share_of_known_offer": _ratio(
                cell["mean_difference_kwh_per_test_day"], known
            ),
            "recovered_share_of_the_gap": _ratio(cell["mean_difference_kwh_per_test_day"], gap),
            "cautious_test_losses": {
                "slot": found["cautious"]["summary"]["mean_test_slot_loss"],
                "driver": found["cautious"]["summary"]["mean_test_driver_loss"],
            },
        }
    )
    return block


def compare(documents: dict, offers: dict) -> dict:
    """The paired comparison of the three arms (module text): unit -> status and, per tier, the
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
    """G as a share of arm 1's mean offer (`difference_share_of_known_offer`); None without a block
    or when arm 1's mean offer is not above 0."""
    return None if block is None else block["difference_share_of_known_offer"]


def over(loss: float | None, limit: float) -> bool:
    """Whether a mean test loss is above its limit (with `crc.TOLERANCE`, as `dial_splits`)."""
    return loss is not None and loss > limit + crc.TOLERANCE


def word(
    value: float | None, bar: float, losses: dict | None, limits: dict
) -> tuple[str, str | None]:
    """(the word, a note or None) of a share against a unit's bar and the losses of arm 3 (module
    text); `losses` holds `slot` and `driver`, `limits` the same keys."""
    if value is None:
        return NONE, "no share is defined: no figure, or arm 1's mean offer is not above 0"
    if value < -bar:
        return LOSS, None
    if value > bar:
        if losses is None or losses["slot"] is None or losses["driver"] is None:
            return NONE, "above the bar, but the test losses of arm 3 are not defined"
        exceeded = [k for k in ("slot", "driver") if over(losses[k], limits[k])]
        if exceeded:
            return (
                NONE,
                f"above the bar, but arm 3's mean test {' and '.join(exceeded)} loss is over its "
                "limit",
            )
        return GAIN, None
    return NONE, None


def read_units(comparison: dict, bars: dict, limits: dict) -> dict:
    """Per unit of `comparison`, its tier A word: a judged unit (a bar in `bars`) gets the share,
    the bar, the word and a note; a unit without a bar gets `NOT_JUDGED` and no word; a judged unit
    missing from `comparison` or that cannot be split is given no share."""
    found = {}
    for unit in dict.fromkeys([*comparison, *bars]):
        block = comparison.get(unit, {}).get("tiers", {}).get("A")
        if unit not in bars:
            found[unit] = {"share": share(block), "bar": None, "word": None, "note": NOT_JUDGED}
            continue
        value = share(block)
        losses = None if block is None else block["cautious_test_losses"]
        said, note = word(value, bars[unit], losses, limits)
        found[unit] = {"share": value, "bar": bars[unit], "word": said, "note": note}
    return found


def one_line(units: dict) -> str:
    """The one-line reading from the words of `read_units` (module text); units without a word
    (not judged) are not read."""
    words = [u["word"] for u in units.values() if u["word"] is not None]
    gains, losses = words.count(GAIN), words.count(LOSS)
    if gains and not losses:
        return HELPS
    if losses and not gains:
        return HURTS
    return MIXED


__all__ = [
    "ARMS",
    "GAIN",
    "HELPS",
    "HURTS",
    "LOSS",
    "MIXED",
    "NONE",
    "NOT_JUDGED",
    "NO_OTHER_SITES",
    "RULE",
    "compare",
    "one_line",
    "own_dial",
    "read_units",
    "share",
    "tier_block",
    "word",
]

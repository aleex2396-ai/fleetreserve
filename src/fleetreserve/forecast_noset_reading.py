"""The reading rule of the forecast trial `noset`, fixed before any result (`forecast_noset`; the
`reading` block of `trials-after-sealed-run/forecast-noset.json`).

Two lines of figures (`forecast_noset_pairs`), both from the one measurement of each run:

- the JUDGED line, `plan_rule`: each run at the dial the project's development rule picks for it
  (plan section 6: the largest mean offer over the run's seeded splits at the unit and tier), the
  same rule for every arm and seed. That rule reads the test-day offers of the development units,
  for every arm alike, as every development figure of the project does. The bars apply to this
  line only;
- the CHECK line, `calibration_days_only`: each run at the dial chosen on half of each split's
  calibration days alone (variant S1 of `site_dial_whatif`). No bar and no verdict is formed on it:
  only the sign and the size of each difference are given. At a judged unit it can only contradict
  a gain or a loss of the judged line (below); where its difference is exactly 0 or undefined it is
  SILENT.

What is judged on the judged line: tier A, per judged unit, three comparisons of the no-encoder arm
(`hazard_tuned_no_set`): against `hazard_tuned` (`against_tuned`, THE judged comparison, which
differs in one thing, the set encoder), against `hazard_tuned_inputs` (`against_inputs`, the neural
model the project reports, which also has extra input columns) and against `lgbm_tail`
(`against_tail`, the main model, a different kind of model). An arm's figure at a unit is its mean
offer over the seeded splits; a neural arm has three seeds and so three such figures, and its
figure for the comparison is the mean of the three, taken split by split. The share of a comparison
is the paired mean difference of the offer (no-encoder arm minus the comparison arm) as a share of
the comparison arm's mean offer. Each judged unit has a bar, the spread that the neural model's
seed alone gave there in an earlier look (8% at the workplace unit, 30% at the pooled Norwegian
unit; sources in the settings file): an analogy, not this trial's own measure. The words are those
of `forecast_mix_reading`:

- "shows a gain" only when the share is above the unit's bar AND every one of the no-encoder
  arm's seed figures at the unit is above every one of the comparison arm's seed figures (the
  comparison arm `lgbm_tail` has one figure);
- "shows a loss" only when the share is below minus the bar AND every no-encoder seed figure is
  below every comparison seed figure;
- otherwise "no measurable difference" (a share equal to a bar is inside it; also when the
  comparison arm's mean offer is 0, where no share is defined).

A comparison that could not be measured at a unit (an arm refused or failed, or the unit not
split) gives the word NOT_READ.

The one line of a comparison is taken from the words of the judged units and from the check line:
"better" when at least one unit shows a gain and none a loss; "worse" when one shows a loss and none
a gain; "mixed" when one shows a gain and another a loss; "no measurable difference" when none shows
either. In addition the one line is "mixed" when, at a judged unit, the check line's difference has
the OPPOSITE sign to a gain or a loss that the judged line shows there (a difference within
`dial_splits.TIE` kWh of 0 has sign 0): a gain that only the dial rule that reads the test days
shows may come from the selection of the dial on those days. The check line is SILENT at a unit
where its difference is of sign 0 (for example every arm offers 0 under the stricter rule there, or
both arms offer 0), or is undefined (no figure: the check did not run or its cell is missing):
silence neither confirms nor contradicts, the judged line's word stands, and the output says for the
unit "check line silent (difference 0)" or "check line silent (no figure for the check line)". Where
the judged word is "no measurable difference" the check line cannot change it; its sign and size are
reported beside it. The output counts, per comparison, the judged units at which the check line was
silent, so that a reader sees how much of a reading rests on the judged line alone. NOT_READ if any
judged unit is NOT_READ.

The overall line takes the three comparisons: "better than all three" only when all three read
better, "worse than all three" only when all three read worse, otherwise "no clear ordering";
NOT_READ if any is.

Tier B is given as numbers only, as information, and is not read by this rule. No Dutch or Korean
site is a development unit, so the trial cannot speak for them (`NO_OTHER_SITES`).
"""

from fleetreserve import dial_splits
from fleetreserve import forecast_mix_reading as words

GAIN = words.GAIN
LOSS = words.LOSS
NONE = words.NONE
NOT_READ = "NOT_READ"
BETTER = "better"
WORSE = "worse"
NO_DIFFERENCE = "no measurable difference"
MIXED = "mixed"
OVERALL_BETTER = "better than all three"
OVERALL_WORSE = "worse than all three"
UNORDERED = "no clear ordering"
NO_OTHER_SITES = words.NO_OTHER_SITES
JUDGED = "against_tuned"
COMPARISONS = ("against_tuned", "against_inputs", "against_tail")
OTHER_ARM = {"against_tuned": "tuned", "against_inputs": "inputs", "against_tail": "tail"}
PLAN_RULE = "plan_rule"
CHECK = "calibration_days_only"
RULE = (
    "Reading rule, fixed before any result. Judged line: tier A, per judged unit, each run at the "
    "dial that the project's development rule picks for it (the largest mean offer over the "
    "splits, which reads the test-day offers of the development units, for every arm alike); an "
    "arm's figure is its mean offer over the splits, for a neural model the mean of its three "
    "seeds; the share is the paired mean difference of the offer (no-encoder model minus the "
    "comparison arm) over the splits as a share of the comparison arm's mean offer; it shows a "
    "gain only above the unit's bar and only if every one of the no-encoder model's three seed "
    "figures is above every seed figure of the comparison arm, a loss only below minus the bar and "
    "only if every one is below, else no measurable difference. One line per comparison (against "
    "hazard_tuned, the judged comparison; against hazard_tuned_inputs; against lgbm_tail): better "
    "if one unit shows a gain and none a loss, worse if one shows a loss and none a gain, mixed "
    "if one shows a gain and another a loss, no measurable difference if none shows either. Check "
    "line: each run at the dial chosen from calibration days alone; no bar and no verdict, only "
    "the sign and size of each difference; a line is mixed when, at a judged unit, the check "
    "line's difference has the opposite sign to a gain or a loss that the judged line shows there; "
    "a check line whose difference is exactly 0 or undefined is silent: it neither confirms nor "
    "contradicts, the judged line's word stands, and the silent units are counted; where the "
    "judged word is no measurable difference the check line cannot change it. Overall: better "
    "than all three only if all three lines read better, worse than all three only if all three "
    "read worse, otherwise no clear ordering. A comparison that could not be measured is NOT_READ. "
    "Tier B is information."
)


def sign_of(difference: float | None) -> int | None:
    """1, -1 or 0 (within `dial_splits.TIE` kWh of 0) for a paired mean difference; None for no
    figure."""
    if difference is None:
        return None
    if abs(difference) <= dial_splits.TIE:
        return 0
    return 1 if difference > 0 else -1


def read_unit(
    cell: dict | None, bar: float, noset_seeds: list | None, other_seeds: list | None
) -> dict:
    """The share, the bar and the word of one judged unit and comparison on the judged line
    (module text). `cell`: the paired cell with the comparison arm as `main`
    (`forecast_mix_pairs.paired_cell`); `noset_seeds`, `other_seeds`: each arm's per-seed figures
    (mean offers) at the unit. Any of the three missing gives NOT_READ."""
    if cell is None or not noset_seeds or not other_seeds:
        return {
            "share": None,
            "bar": bar,
            "difference": None,
            "word": NOT_READ,
            "word_by_the_bar_alone": NOT_READ,
        }
    value = words.share(cell)
    by_bar = words.word(value, bar)
    word = NONE
    if by_bar == GAIN and min(noset_seeds) > max(other_seeds):
        word = GAIN
    elif by_bar == LOSS and max(noset_seeds) < min(other_seeds):
        word = LOSS
    return {
        "share": value,
        "bar": bar,
        "difference": cell["mean_difference_kwh_per_test_day"],
        "word": word,
        "word_by_the_bar_alone": by_bar,
        "noset_seed_figures": list(noset_seeds),
        "comparison_seed_figures": list(other_seeds),
    }


def _tier_a(comparison: dict, unit: str, line: str) -> dict | None:
    return comparison.get(unit, {}).get("tiers", {}).get("A", {}).get(line)


def read_units(comparison: dict, bars: dict, which: str) -> dict:
    """Per judged unit of `bars` (unit -> bar), `read_unit` of the tier A cell `which` (one of
    `COMPARISONS`) on the judged line of `comparison` (`forecast_noset_pairs.compare`); a unit
    missing from it or without that cell gives NOT_READ."""
    if which not in COMPARISONS:
        raise ValueError(f"a comparison is one of {COMPARISONS}, got {which!r}")
    found = {}
    for unit, bar in bars.items():
        block = _tier_a(comparison, unit, PLAN_RULE)
        if block is None:
            found[unit] = read_unit(None, bar, None, None)
            continue
        figures = block["seed_figures"]
        found[unit] = read_unit(
            block.get(which), bar, figures.get("noset"), figures.get(OTHER_ARM[which])
        )
    return found


def check_units(comparison: dict, units, which: str) -> dict:
    """Per judged unit of `units`, the sign and the size of the tier A cell `which` on the check
    line (module text): `difference` (kWh per counting test day), `share` of the comparison arm's
    mean offer (None where undefined), `sign`; all None where the cell is missing. No verdict."""
    if which not in COMPARISONS:
        raise ValueError(f"a comparison is one of {COMPARISONS}, got {which!r}")
    found = {}
    for unit in units:
        block = _tier_a(comparison, unit, CHECK)
        cell = None if block is None else block.get(which)
        difference = None if cell is None else cell["mean_difference_kwh_per_test_day"]
        found[unit] = {
            "difference": difference,
            "share": words.share(cell),
            "sign": sign_of(difference),
        }
    return found


def silence_reason(check: dict | None) -> str | None:
    """None where the check line speaks at a unit (`check`: its entry of `check_units`, a nonzero
    sign); otherwise the text saying why it is silent: "check line silent (difference 0)" for a
    difference of sign 0, "check line silent (no figure for the check line)" for no figure."""
    sign = None if check is None else check.get("sign")
    if sign is None:
        return "check line silent (no figure for the check line)"
    if sign == 0:
        return "check line silent (difference 0)"
    return None


def silent_units(units: dict, check: dict) -> dict:
    """The judged units of `units` at which the check line is silent, each with its reason
    (`silence_reason`); `check` from `check_units`. A unit absent from `check` is silent."""
    found = {}
    for unit in units:
        reason = silence_reason(check.get(unit))
        if reason is not None:
            found[unit] = reason
    return found


def disagreements(units: dict, check: dict) -> list[str]:
    """The judged units at which the check line's difference has the opposite sign to a gain or a
    loss shown by the judged line (a gain with a negative check difference, a loss with a positive
    one); `units` from `read_units`, `check` from `check_units`. A silent check line, and a judged
    word of no measurable difference or NOT_READ, never list a unit."""
    found = []
    for unit, read in units.items():
        other = check.get(unit, {}).get("sign")
        if (read.get("word") == GAIN and other == -1) or (read.get("word") == LOSS and other == 1):
            found.append(unit)
    return found


def check_note(read: dict, check: dict | None) -> str:
    """The sentence that goes beside the check line's figures at one judged unit (`read` and
    `check`: the unit's entries of `read_units` and `check_units`): the silence reason, or whether
    the check line confirms or contradicts the judged line's gain or loss, or that it cannot change
    a judged word of no measurable difference."""
    word = read.get("word")
    if word == NOT_READ:
        return "the judged line is not read"
    reason = silence_reason(check)
    if reason is not None:
        return reason
    if word == NONE:
        return "the judged line shows no measurable difference and the check line cannot change it"
    opposite = (word == GAIN and check["sign"] == -1) or (word == LOSS and check["sign"] == 1)
    if opposite:
        return "THE SIGN IS OPPOSITE to the judged line's gain or loss: the reading is mixed"
    return "the sign agrees with the judged line's gain or loss"


def one_line(units: dict, check: dict) -> str:
    """The one line of a comparison from the words of `read_units` and the check line
    (`check_units`) (module text). A silent check line leaves the judged words as they are."""
    found = [u["word"] for u in units.values()]
    if not found or NOT_READ in found:
        return NOT_READ
    if disagreements(units, check):
        return MIXED
    gains, losses = found.count(GAIN), found.count(LOSS)
    if gains and not losses:
        return BETTER
    if losses and not gains:
        return WORSE
    if gains and losses:
        return MIXED
    return NO_DIFFERENCE


def overall(lines: dict) -> str:
    """The overall line from the one lines of the three comparisons (module text)."""
    found = [lines.get(which) for which in COMPARISONS]
    if None in found or NOT_READ in found:
        return NOT_READ
    if all(line == BETTER for line in found):
        return OVERALL_BETTER
    if all(line == WORSE for line in found):
        return OVERALL_WORSE
    return UNORDERED

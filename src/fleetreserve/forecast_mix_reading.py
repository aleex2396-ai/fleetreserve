"""The reading rule of the forecast trial `mix`, fixed before any result (`forecast_mix`; the
`reading` block and `bars` of `trials-after-sealed-run/forecast-mix.json`).

What is judged: tier A, per judged unit, each arm at its own dial (`forecast_mix_pairs`, cell
`own_dial`). The share of a unit is the mean paired difference of the offer (mix minus main) over
the seeded splits, as a share of the main arm's mean offer. Each judged unit has a bar, the spread
that the neural model's seed alone gave at that unit at kappa 0.3 (8% at the workplace unit, 30%
at the pooled Norwegian unit; sources in the settings file). The bars are an analogy, applied to a
mean over splits that share days, and not this trial's measured noise:

- "shows a gain" only when the share is above the unit's bar;
- "shows a loss" only when the share is below minus the bar;
- otherwise "no measurable difference" (a share equal to a bar is inside it); also when the main
  arm's mean offer is 0 or the unit has no figure, where no share is defined.

The one-line reading of the trial is taken from the level-0.5 line only, from the words of the
judged units:

- "the mix helps" only when at least one unit shows a gain and none shows a loss;
- "the mix hurts" when at least one unit shows a loss and none shows a gain;
- otherwise "no measurable difference; a mixed result is reported as mixed".

Tier B is given as numbers only, as information, and is not read by this rule. The extra line (the
mix at another level against the main model at its own) is judged by the same per-unit rule, and
never enters the one-line reading. No Dutch or Korean site is a development unit, so the trial
cannot speak for them (`NO_OTHER_SITES`).
"""

GAIN = "shows a gain"
LOSS = "shows a loss"
NONE = "no measurable difference"
HELPS = "the mix helps"
HURTS = "the mix hurts"
MIXED = "no measurable difference; a mixed result is reported as mixed"
NO_OTHER_SITES = (
    "No Dutch or Korean site is a development unit, so the trial cannot speak for them."
)
RULE = (
    "Reading rule, fixed before any result: tier A, per judged unit, each arm at its own dial; "
    "the share is the paired mean difference of the offer (mix minus main) over the splits as a "
    "share of the main arm's mean offer; it shows a gain only above the unit's bar, a loss only "
    "below minus the bar, else no measurable difference. One line: the mix helps only if one unit "
    "shows a gain and none a loss; it hurts if one shows a loss and none a gain; otherwise no "
    "measurable difference, and a mixed result is reported as mixed. The line is taken from the "
    "main level only; the extra line is information and never enters it. Tier B is information."
)


def share(cell: dict | None) -> float | None:
    """The paired mean difference of `cell` (`forecast_mix_pairs.paired_cell`) as a share of the
    main arm's mean offer; None without a cell or when that mean is not above 0."""
    if cell is None:
        return None
    main = cell["mean_main_offer_kwh_per_test_day"]
    difference = cell["mean_difference_kwh_per_test_day"]
    if main is None or difference is None or not main > 0:
        return None
    return difference / main


def word(value: float | None, bar: float) -> str:
    """The word of a share against a unit's bar (module text)."""
    if value is None:
        return NONE
    if value > bar:
        return GAIN
    if value < -bar:
        return LOSS
    return NONE


def read_units(comparison: dict, bars: dict) -> dict:
    """Per judged unit of `bars` (unit -> share), the share of its tier A own-dial cell, the bar
    and the word; a unit missing from `comparison` or that cannot be split is given no share."""
    found = {}
    for unit, bar in bars.items():
        entry = comparison.get(unit, {})
        cell = entry.get("tiers", {}).get("A", {}).get("own_dial")
        value = share(cell)
        found[unit] = {"share": value, "bar": bar, "word": word(value, bar)}
    return found


def one_line(units: dict) -> str:
    """The one-line reading from the words of `read_units` (module text)."""
    words = [u["word"] for u in units.values()]
    gains, losses = words.count(GAIN), words.count(LOSS)
    if gains and not losses:
        return HELPS
    if losses and not gains:
        return HURTS
    return MIXED

"""The reading rule of the forecast trial `mix` (`forecast_mix_reading`) on made-up cells. Written,
not run. Each test says which edit turns it red.

1. `share`: the paired mean difference over the main arm's mean offer; none for no cell, or a main
   mean that is not above 0. Red if the share is taken over the trial's mean.
2. `word` at each judged unit's committed bar: just above gives a gain, just below gives no
   difference, a share equal to the bar is inside it; the same on the negative side for a loss.
   Red if `>` becomes `>=` or `<` becomes `<=` in `word`, or if the loss side uses the bar of the
   gain side with the wrong sign.
3. `read_units` reads tier A, `own_dial`, per judged unit of the bars only. Red if it reads tier B
   or `same_dial`.
4. `one_line`: helps (a gain, no loss), hurts (a loss, no gain), and otherwise the mixed sentence,
   which is also the answer for no difference at all. Red if a gain beside a loss reads as helps.
5. The two bars are the committed ones, 8% and 30%, per unit.
"""

import json
from pathlib import Path

import pytest

from fleetreserve import forecast_mix
from fleetreserve import forecast_mix_reading as reading

SETTINGS = json.loads(Path(forecast_mix.SETTINGS_FILE).read_text())
BARS = forecast_mix.bars_of(SETTINGS)
WORKPLACE, NORWAY = "nrel-from-may-2019", "norway-development-pooled"


def cell(main: float, difference: float) -> dict:
    return {
        "mean_main_offer_kwh_per_test_day": main,
        "mean_difference_kwh_per_test_day": difference,
    }


def comparison(shares: dict, tier_b_share: float = 0.9, same_dial_share: float = 0.9) -> dict:
    """A comparison whose tier A own-dial cells have the given shares (main mean 10); tier B and
    same-dial cells carry another share, which must not be read."""
    found = {}
    for unit, value in shares.items():
        found[unit] = {
            "status": "calibratable",
            "tiers": {
                "A": {
                    "own_dial": cell(10.0, 10.0 * value),
                    "same_dial": cell(10.0, 10.0 * same_dial_share),
                },
                "B": {"own_dial": cell(10.0, 10.0 * tier_b_share)},
            },
        }
    return found


def test_the_bars_are_the_committed_ones_per_unit():
    assert BARS == {WORKPLACE: 0.08, NORWAY: 0.30}


def test_share_is_over_the_main_arms_mean():
    assert reading.share(cell(10.0, 1.0)) == pytest.approx(0.1)
    assert reading.share(cell(2.0, -1.0)) == pytest.approx(-0.5)
    assert reading.share(cell(0.0, 1.0)) is None
    assert reading.share(cell(None, None)) is None
    assert reading.share(None) is None


@pytest.mark.parametrize("unit", sorted(BARS))
def test_the_word_just_above_inside_and_just_below_each_units_bar(unit):
    bar = BARS[unit]
    assert reading.word(bar * 1.01, bar) == reading.GAIN
    assert reading.word(bar * 0.99, bar) == reading.NONE
    assert reading.word(bar, bar) == reading.NONE  # a share equal to the bar is inside it
    assert reading.word(0.0, bar) == reading.NONE
    assert reading.word(-bar * 0.99, bar) == reading.NONE
    assert reading.word(-bar, bar) == reading.NONE
    assert reading.word(-bar * 1.01, bar) == reading.LOSS
    assert reading.word(None, bar) == reading.NONE


def test_one_unit_is_judged_at_its_own_bar_not_the_other_units():
    """A share of 20% is a gain at the workplace unit and inside the Norwegian unit's bar."""
    units = reading.read_units(comparison({WORKPLACE: 0.20, NORWAY: 0.20}), BARS)
    assert units[WORKPLACE]["word"] == reading.GAIN and units[WORKPLACE]["bar"] == 0.08
    assert units[NORWAY]["word"] == reading.NONE and units[NORWAY]["bar"] == 0.30
    assert units[WORKPLACE]["share"] == pytest.approx(0.20)


def test_only_tier_a_at_each_arms_own_dial_is_read():
    found = comparison({WORKPLACE: 0.0, NORWAY: 0.0}, tier_b_share=0.9, same_dial_share=0.9)
    units = reading.read_units(found, BARS)
    assert {u["word"] for u in units.values()} == {reading.NONE}


def test_a_unit_without_a_figure_or_not_in_the_comparison_has_no_difference():
    found = comparison({WORKPLACE: 0.5})
    found[NORWAY] = {"status": "too few days", "counting_days": 5}
    units = reading.read_units(found, BARS)
    assert units[WORKPLACE]["word"] == reading.GAIN
    assert units[NORWAY] == {"share": None, "bar": 0.30, "word": reading.NONE}
    assert reading.read_units({}, BARS)[WORKPLACE]["word"] == reading.NONE


def line(workplace: float, norway: float) -> str:
    found = comparison({WORKPLACE: workplace, NORWAY: norway})
    return reading.one_line(reading.read_units(found, BARS))


@pytest.mark.parametrize(
    "workplace,norway,expected",
    [
        (0.20, 0.0, reading.HELPS),  # a gain at one unit, no loss at the other
        (0.0, 0.50, reading.HELPS),
        (0.20, 0.50, reading.HELPS),  # a gain at both
        (-0.20, 0.0, reading.HURTS),  # a loss at one unit, no gain at the other
        (0.0, -0.50, reading.HURTS),
        (-0.20, -0.50, reading.HURTS),
        (0.20, -0.50, reading.MIXED),  # a gain beside a loss: reported as mixed
        (-0.20, 0.50, reading.MIXED),
        (0.0, 0.0, reading.MIXED),  # no measurable difference at all
        (0.07, -0.29, reading.MIXED),  # inside both bars
    ],
)
def test_the_three_outcomes_of_the_one_line_reading(workplace, norway, expected):
    assert line(workplace, norway) == expected


def test_the_three_sentences():
    assert reading.HELPS == "the mix helps" and reading.HURTS == "the mix hurts"
    assert reading.MIXED == "no measurable difference; a mixed result is reported as mixed"
    assert "Dutch or Korean" in reading.NO_OTHER_SITES
    assert "development unit" in reading.NO_OTHER_SITES

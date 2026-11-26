"""The reading rule of the forecast trial `noset` (`forecast_noset_reading`) on made-up cells.
Written, not run. Each test says which edit turns it red.

1. The seed condition. A share above the bar is a gain only if every no-encoder seed figure is
   above every comparison seed figure; the same on the loss side. Red if the seed condition is
   removed from `read_unit`, or if it compares the means instead of the extremes (`min` / `max`).
2. The bar. A share equal to the bar is inside it, on both sides. Red if `>` becomes `>=` or `<`
   becomes `<=` in `words.word` (`forecast_mix_reading`), which `read_unit` calls.
3. NOT_READ. A missing cell or a missing seed list gives NOT_READ, never "no measurable
   difference". Red if the first test of `read_unit` is removed.
4. No share when the comparison arm's mean offer is 0: no measurable difference.
5. `read_units` reads tier A of the JUDGED line (`plan_rule`) and, for each comparison, the seed
   figures of its own comparison arm (`tuned` for `against_tuned`, `inputs` for `against_inputs`,
   `tail` for `against_tail`). Red if the arms are swapped, if tier B is read, or if the check
   line is read by the bars.
6. `check_units` gives the sign and size on the check line with no bar and no verdict. Red if a bar
   is applied to it.
7. `one_line`: better / worse / mixed / no measurable difference; mixed only when, at a judged
   unit, the check line's difference has the opposite sign to a gain or a loss of the judged line;
   a check line of difference exactly 0 or with no figure is silent (the judged word stands, the
   reason is given and the silent units are counted); a judged word of no measurable difference
   cannot be changed by the check line; NOT_READ if a judged unit is. Red if a gain beside a loss
   reads as better, if the opposite sign is not applied, if sign 0 or no figure counts as a
   disagreement, or if NOT_READ is passed over.
8. `overall`: better than all three, worse than all three, else no clear ordering. Red if two of
   three agreeing is enough.
9. The words are those of the mix trial's reading module.
"""

import json
from pathlib import Path

import pytest

from fleetreserve import forecast_mix_reading, forecast_noset
from fleetreserve import forecast_noset_reading as reading

SETTINGS = json.loads(Path(forecast_noset.SETTINGS_FILE).read_text())
BARS = forecast_noset.bars_of(SETTINGS)
WORKPLACE, NORWAY = "nrel-from-may-2019", "norway-development-pooled"


def cell(other: float, difference: float) -> dict:
    """The fields of `forecast_mix_pairs.paired_cell` that the reading reads (its `main` is the
    comparison arm)."""
    return {
        "mean_main_offer_kwh_per_test_day": other,
        "mean_difference_kwh_per_test_day": difference,
    }


def test_the_bars_are_the_committed_ones_per_unit():
    assert BARS == {WORKPLACE: 0.08, NORWAY: 0.30}


def test_the_words_are_those_of_the_mix_reading():
    assert (reading.GAIN, reading.LOSS, reading.NONE) == (
        forecast_mix_reading.GAIN,
        forecast_mix_reading.LOSS,
        forecast_mix_reading.NONE,
    )
    assert reading.NO_OTHER_SITES == forecast_mix_reading.NO_OTHER_SITES


# --- 1. the seed condition ----------------------------------------------------------------------


def test_a_share_above_the_bar_with_every_seed_above_is_a_gain():
    found = reading.read_unit(cell(10.0, 1.0), 0.08, [11.0, 11.5, 12.0], [9.0, 10.0, 10.9])
    assert found["word"] == reading.GAIN and found["share"] == pytest.approx(0.1)
    assert found["word_by_the_bar_alone"] == reading.GAIN
    assert found["difference"] == 1.0


def test_a_share_above_the_bar_with_overlapping_seeds_is_no_measurable_difference():
    # the lowest no-encoder seed (10.5) is below the highest comparison seed (10.9)
    found = reading.read_unit(cell(10.0, 1.0), 0.08, [10.5, 11.5, 12.0], [9.0, 10.0, 10.9])
    assert found["word"] == reading.NONE
    assert found["word_by_the_bar_alone"] == reading.GAIN


def test_a_seed_above_every_other_seed_but_a_share_inside_the_bar_is_no_measurable_difference():
    found = reading.read_unit(cell(10.0, 0.5), 0.08, [11.0, 11.5, 12.0], [9.0, 10.0, 10.9])
    assert found["word"] == reading.NONE


def test_a_share_below_minus_the_bar_with_every_seed_below_is_a_loss():
    found = reading.read_unit(cell(10.0, -1.0), 0.08, [8.0, 8.5, 9.0], [9.5, 10.0, 10.5])
    assert found["word"] == reading.LOSS
    overlapping = reading.read_unit(cell(10.0, -1.0), 0.08, [8.0, 8.5, 9.8], [9.5, 10.0, 10.5])
    assert overlapping["word"] == reading.NONE
    assert overlapping["word_by_the_bar_alone"] == reading.LOSS


def test_the_seed_condition_compares_the_extremes_not_the_means():
    # means 10.5 against 10.0, but the worst no-encoder seed (9.0) is below the best other (11.0)
    found = reading.read_unit(cell(10.0, 1.0), 0.08, [9.0, 11.5, 11.0], [9.0, 10.0, 11.0])
    assert found["word"] == reading.NONE


# --- 2. the bar ---------------------------------------------------------------------------------


def test_a_share_equal_to_the_bar_is_inside_it_on_both_sides():
    up = reading.read_unit(cell(10.0, 2.5), 0.25, [20.0], [1.0])
    down = reading.read_unit(cell(10.0, -2.5), 0.25, [1.0], [20.0])
    assert up["share"] == 0.25 and up["word"] == reading.NONE
    assert down["share"] == -0.25 and down["word"] == reading.NONE


# --- 3 and 4. NOT_READ and no share -------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        (None, 0.08, [1.0], [1.0]),
        (cell(10.0, 1.0), 0.08, None, [1.0]),
        (cell(10.0, 1.0), 0.08, [1.0], None),
        (cell(10.0, 1.0), 0.08, [], [1.0]),
    ],
)
def test_a_comparison_that_could_not_be_measured_is_not_read(arguments):
    found = reading.read_unit(*arguments)
    assert found["word"] == reading.NOT_READ and found["share"] is None
    assert found["difference"] is None


def test_a_comparison_arm_with_no_offer_has_no_share_and_no_difference_is_read():
    found = reading.read_unit(cell(0.0, 3.0), 0.08, [3.0, 3.0, 3.0], [0.0, 0.0, 0.0])
    assert found["share"] is None and found["word"] == reading.NONE


# --- 5. read_units ------------------------------------------------------------------------------


def line(
    tuned_cell, inputs_cell, tail_cell, noset=(10.0,), tuned=(5.0,), inputs=(6.0,), tail=(1.0,)
):
    return {
        "against_tuned": tuned_cell,
        "against_inputs": inputs_cell,
        "against_tail": tail_cell,
        "seed_figures": {
            "noset": list(noset),
            "tuned": list(tuned),
            "inputs": list(inputs),
            "tail": list(tail),
        },
    }


def comparison(**units) -> dict:
    """unit -> (judged line, check line); tier B is a trap that no rule may read."""
    trap = line(cell(1.0, 99.0), cell(1.0, 99.0), cell(1.0, 99.0), noset=(99.0,), tuned=(0.0,))
    found = {}
    for unit, (plan, check) in units.items():
        found[unit] = {
            "status": "calibratable",
            "tiers": {
                "A": {reading.PLAN_RULE: plan, reading.CHECK: check},
                "B": {reading.PLAN_RULE: trap, reading.CHECK: trap},
            },
        }
    return found


def test_each_comparison_reads_its_own_comparison_arms_seeds():
    # the no-encoder seeds (10) are above the tuned (5) and inputs (6) seeds at a 100% and a 67%
    # share, and below the tail seed (50) at a -80% share
    plan = line(cell(5.0, 5.0), cell(6.0, 4.0), cell(50.0, -40.0), tail=(50.0,))
    found = comparison(**{WORKPLACE: (plan, plan), NORWAY: (plan, plan)})
    one = reading.read_units(found, BARS, "against_tuned")
    two = reading.read_units(found, BARS, "against_inputs")
    three = reading.read_units(found, BARS, "against_tail")
    assert {u["word"] for u in one.values()} == {reading.GAIN}
    assert {u["word"] for u in two.values()} == {reading.GAIN}
    assert {u["word"] for u in three.values()} == {reading.LOSS}
    assert one[WORKPLACE]["comparison_seed_figures"] == [5.0]
    assert two[WORKPLACE]["comparison_seed_figures"] == [6.0]
    assert three[WORKPLACE]["comparison_seed_figures"] == [50.0]


def test_the_judged_comparison_is_the_one_against_hazard_tuned():
    assert reading.JUDGED == "against_tuned"
    assert reading.COMPARISONS == ("against_tuned", "against_inputs", "against_tail")


def test_tier_b_and_the_check_line_are_never_read_by_the_bars():
    quiet = line(cell(10.0, 0.0), cell(10.0, 0.0), cell(10.0, 0.0), noset=(10.0,), tuned=(10.0,))
    loud = line(cell(10.0, 9.0), cell(10.0, 9.0), cell(10.0, 9.0), noset=(50.0,), tuned=(1.0,))
    found = comparison(**{WORKPLACE: (quiet, loud), NORWAY: (quiet, loud)})
    for which in reading.COMPARISONS:
        words = {u["word"] for u in reading.read_units(found, BARS, which).values()}
        assert words == {reading.NONE}


def test_a_unit_that_cannot_be_split_or_is_missing_is_not_read():
    found = {WORKPLACE: {"status": "not calibratable", "counting_days": 3}}
    for which in reading.COMPARISONS:
        units = reading.read_units(found, BARS, which)
        assert {u["word"] for u in units.values()} == {reading.NOT_READ}
        assert reading.one_line(units, reading.check_units(found, BARS, which)) == reading.NOT_READ


def test_an_unknown_comparison_is_refused():
    with pytest.raises(ValueError):
        reading.read_units({}, BARS, "against_nothing")
    with pytest.raises(ValueError):
        reading.check_units({}, BARS, "against_nothing")


# --- 6. the check line: sign and size, no verdict -----------------------------------------------


def test_the_check_line_gives_the_sign_and_the_size_and_no_word():
    plan = line(cell(10.0, 1.0), cell(10.0, 1.0), cell(10.0, 1.0))
    check = line(cell(10.0, 0.01), cell(10.0, -3.0), cell(10.0, 0.0))
    found = comparison(**{WORKPLACE: (plan, check), NORWAY: (plan, check)})
    one = reading.check_units(found, BARS, "against_tuned")[WORKPLACE]
    assert one == {"difference": 0.01, "share": pytest.approx(0.001), "sign": 1}  # inside any bar
    two = reading.check_units(found, BARS, "against_inputs")[NORWAY]
    assert two["sign"] == -1 and two["share"] == pytest.approx(-0.3)
    three = reading.check_units(found, BARS, "against_tail")[WORKPLACE]
    assert three["sign"] == 0 and three["difference"] == 0.0
    assert "word" not in one


def test_a_missing_check_cell_has_no_sign():
    plan = line(cell(10.0, 1.0), cell(10.0, 1.0), cell(10.0, 1.0))
    check = line(None, cell(10.0, 1.0), cell(10.0, 1.0))
    found = comparison(**{WORKPLACE: (plan, check)})
    assert reading.check_units(found, [WORKPLACE], "against_tuned")[WORKPLACE] == {
        "difference": None,
        "share": None,
        "sign": None,
    }
    assert reading.check_units({}, [WORKPLACE], "against_tuned")[WORKPLACE]["sign"] is None


def test_the_sign_of_a_difference_within_the_tie_is_zero():
    assert reading.sign_of(1e-12) == 0 and reading.sign_of(-1e-12) == 0
    assert reading.sign_of(0.5) == 1 and reading.sign_of(-0.5) == -1
    assert reading.sign_of(None) is None


# --- 7. the lines -------------------------------------------------------------------------------


def judged(*found) -> dict:
    """Judged units with a word and a difference whose sign follows the word."""
    sign = {reading.GAIN: 2.0, reading.LOSS: -2.0, reading.NONE: 0.1, reading.NOT_READ: None}
    return {f"u{i}": {"word": w, "difference": sign[w]} for i, w in enumerate(found)}


def signs(*found) -> dict:
    return {f"u{i}": {"sign": s} for i, s in enumerate(found)}


def test_one_line_better_worse_mixed_no_difference_and_not_read():
    gain, loss, none, not_read = reading.GAIN, reading.LOSS, reading.NONE, reading.NOT_READ
    assert reading.one_line(judged(gain, none), signs(1, 1)) == reading.BETTER
    assert reading.one_line(judged(gain, gain), signs(1, 1)) == reading.BETTER
    assert reading.one_line(judged(loss, none), signs(-1, 1)) == reading.WORSE
    assert reading.one_line(judged(gain, loss), signs(1, -1)) == reading.MIXED
    assert reading.one_line(judged(none, none), signs(1, 1)) == reading.NO_DIFFERENCE
    assert reading.one_line(judged(gain, not_read), signs(1, 1)) == reading.NOT_READ
    assert reading.one_line({}, {}) == reading.NOT_READ


def test_a_check_sign_opposite_to_a_judged_gain_or_loss_makes_the_one_line_mixed():
    """Red if `disagreements` in `forecast_noset_reading` stops listing a gain (word GAIN) with
    check sign -1 or a loss (word LOSS) with check sign 1, for example by returning an empty list,
    by testing `other == 0` or by dropping the `read.get("word")` conditions."""
    gain, loss, none = reading.GAIN, reading.LOSS, reading.NONE
    # a gain on the judged line that the check line shows as a loss
    assert reading.one_line(judged(gain, none), signs(-1, 1)) == reading.MIXED
    assert reading.disagreements(judged(gain, none), signs(-1, 1)) == ["u0"]
    # a loss that the check line shows as a gain
    assert reading.one_line(judged(loss, none), signs(1, -1)) == reading.MIXED
    assert reading.disagreements(judged(loss, none), signs(1, -1)) == ["u0"]
    # the opposite sign at the second unit counts the same
    assert reading.one_line(judged(none, gain), signs(1, -1)) == reading.MIXED
    # the same signs agree: nothing is listed and the judged words stand
    assert reading.disagreements(judged(gain, loss), signs(1, -1)) == []
    assert reading.one_line(judged(gain, none), signs(1, -1)) == reading.BETTER
    assert reading.one_line(judged(loss, none), signs(-1, 1)) == reading.WORSE


def test_a_check_difference_of_exactly_zero_is_silent_and_the_judged_word_stands():
    """Red if `disagreements` goes back to listing a unit whenever the two signs differ (sign 0
    against a nonzero judged sign), or if `one_line` returns MIXED for a silent check line."""
    gain, loss, none = reading.GAIN, reading.LOSS, reading.NONE
    assert reading.one_line(judged(gain, none), signs(0, 0)) == reading.BETTER
    assert reading.one_line(judged(gain, gain), signs(0, 0)) == reading.BETTER
    assert reading.one_line(judged(loss, none), signs(0, 0)) == reading.WORSE
    assert reading.one_line(judged(gain, loss), signs(0, 0)) == reading.MIXED  # by the words alone
    assert reading.one_line(judged(none, none), signs(0, 0)) == reading.NO_DIFFERENCE
    assert reading.disagreements(judged(gain, loss), signs(0, 0)) == []
    # silence at one unit does not hide a contradiction at the other
    assert reading.one_line(judged(gain, gain), signs(0, -1)) == reading.MIXED
    assert reading.disagreements(judged(gain, gain), signs(0, -1)) == ["u1"]


def test_a_check_line_with_no_figure_is_silent_and_the_judged_word_stands():
    """Red if `one_line` returns NOT_READ when a check sign is None again, or if
    `silence_reason` lets None pass as a speaking check line."""
    gain, none = reading.GAIN, reading.NONE
    assert reading.one_line(judged(gain, none), signs(None, 1)) == reading.BETTER
    assert reading.one_line(judged(gain, none), {}) == reading.BETTER
    assert reading.disagreements(judged(gain, none), {}) == []
    assert reading.silent_units(judged(gain, none), {}) == {
        "u0": "check line silent (no figure for the check line)",
        "u1": "check line silent (no figure for the check line)",
    }


def test_a_judged_word_of_no_measurable_difference_cannot_be_changed_by_the_check_line():
    """Red if `disagreements` ignores the judged word and lists a unit of word NONE whose check
    sign is nonzero (the old rule listed any two signs that differ)."""
    none = reading.NONE
    for found in (signs(-1, 1), signs(1, -1), signs(-1, -1), signs(1, 1), signs(0, 0)):
        assert reading.disagreements(judged(none, none), found) == []
        assert reading.one_line(judged(none, none), found) == reading.NO_DIFFERENCE


def test_a_judged_unit_that_is_not_read_keeps_the_line_not_read_whatever_the_check_says():
    """Red if the NOT_READ test of `one_line` is moved below the silence or disagreement tests, or
    removed."""
    gain, not_read = reading.GAIN, reading.NOT_READ
    for found in (signs(1, 1), signs(0, 0), signs(-1, -1), signs(None, None)):
        assert reading.one_line(judged(gain, not_read), found) == reading.NOT_READ


def test_the_silent_units_are_counted_with_their_reasons():
    """Red if `silent_units` lists a unit whose check sign is nonzero, leaves out one whose sign is
    0 or None, or gives both reasons the same text."""
    gain, none = reading.GAIN, reading.NONE
    found = reading.silent_units(judged(gain, none), signs(0, 1))
    assert found == {"u0": "check line silent (difference 0)"}
    both = reading.silent_units(judged(gain, none), signs(1, None))
    assert both == {"u1": "check line silent (no figure for the check line)"}
    assert reading.silent_units(judged(gain, none), signs(1, -1)) == {}
    assert len(reading.silent_units(judged(gain, none), signs(0, 0))) == 2


def test_both_arms_at_zero_on_the_check_line_is_a_difference_of_zero_with_no_share():
    """Red if `check_units` gives such a cell a sign other than 0 or a share (the pooled Norwegian
    unit under the stricter rule: every arm offers 0)."""
    # on the judged line the share is 40%, above the bar of 30%, and the no-encoder seed figure (10)
    # is above the tuned one (5): a gain
    plan = line(cell(10.0, 4.0), cell(10.0, 4.0), cell(10.0, 4.0))
    zero = line(cell(0.0, 0.0), cell(0.0, 0.0), cell(0.0, 0.0), noset=(0.0,), tuned=(0.0,))
    found = comparison(**{NORWAY: (plan, zero)})
    check = reading.check_units(found, [NORWAY], "against_tuned")
    assert check[NORWAY] == {"difference": 0.0, "share": None, "sign": 0}
    units = reading.read_units(found, {NORWAY: 0.30}, "against_tuned")
    assert units[NORWAY]["word"] == reading.GAIN
    assert reading.silent_units(units, check) == {NORWAY: "check line silent (difference 0)"}
    assert reading.disagreements(units, check) == []
    assert reading.one_line(units, check) == reading.BETTER


def test_the_note_beside_the_check_line_says_silent_agrees_opposite_or_cannot_change():
    """Red if `check_note` words two of these cases alike or lets a silent check line read as
    agreement."""
    gain, none, not_read = reading.GAIN, reading.NONE, reading.NOT_READ
    word = {"word": gain}
    assert reading.check_note(word, {"sign": 0}) == "check line silent (difference 0)"
    assert reading.check_note(word, {"sign": None}).startswith("check line silent (no figure")
    assert reading.check_note(word, None).startswith("check line silent (no figure")
    assert reading.check_note(word, {"sign": 1}).startswith("the sign agrees")
    assert reading.check_note(word, {"sign": -1}).startswith("THE SIGN IS OPPOSITE")
    opposite = reading.check_note({"word": reading.LOSS}, {"sign": 1})
    assert opposite.startswith("THE SIGN IS OPPOSITE")
    assert reading.check_note({"word": none}, {"sign": -1}).endswith("cannot change it")
    assert reading.check_note({"word": none}, {"sign": 0}) == "check line silent (difference 0)"
    assert reading.check_note({"word": not_read}, {"sign": 1}) == "the judged line is not read"


# --- 8. overall ---------------------------------------------------------------------------------


def lines(tuned, inputs, tail) -> dict:
    return {"against_tuned": tuned, "against_inputs": inputs, "against_tail": tail}


def test_overall_needs_all_three_comparisons_to_agree():
    better, worse, mixed, none = reading.BETTER, reading.WORSE, reading.MIXED, reading.NO_DIFFERENCE
    assert reading.overall(lines(better, better, better)) == reading.OVERALL_BETTER
    assert reading.overall(lines(worse, worse, worse)) == reading.OVERALL_WORSE
    assert reading.overall(lines(better, better, mixed)) == reading.UNORDERED
    assert reading.overall(lines(better, better, none)) == reading.UNORDERED
    assert reading.overall(lines(better, worse, better)) == reading.UNORDERED
    assert reading.overall(lines(mixed, mixed, mixed)) == reading.UNORDERED
    assert reading.overall(lines(better, better, reading.NOT_READ)) == reading.NOT_READ
    assert reading.overall({"against_tuned": better, "against_inputs": better}) == reading.NOT_READ


def test_the_printed_rule_names_the_seed_condition_the_two_lines_and_not_read():
    assert "every one of the no-encoder model's three seed figures" in reading.RULE
    assert "reads the test-day offers of the development units" in reading.RULE
    assert "no bar and no verdict" in reading.RULE
    assert "sign" in reading.RULE and "NOT_READ" in reading.RULE
    assert "silent" in reading.RULE and "opposite sign" in reading.RULE
    assert "better than all three" in reading.RULE

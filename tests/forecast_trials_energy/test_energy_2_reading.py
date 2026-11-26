"""The pairing and the reading rule of the forecast trial `energy-2`
(`forecast_energy_2_reading`) on made-up results. Written to be run through the test slot. Each
test says which edit turns it red.

1. `tier_block` on known numbers (arm 1 `known` 10 in every split, arm 2 `empirical` 4, 6, 5, 5, arm
   3 `cautious` 6, 8, 7, 7, arm 4 `cautious-need` 5, 6, 7, 6, arm 5 `cautious-all` 7, 9, 8, 8): the
   JUDGED difference is arm 5 minus arm 2 (+3 in every split), the information difference arm 4
   minus arm 2 (+1, 0, +2, +1), the shares are over arm 1's mean offer, and each comparison carries
   its own arm's test losses. Red if the judged cell is made of arm 4 or arm 3, if the difference
   is turned, if the share is taken over arm 2's mean, or if the losses are another arm's.
2. Each arm is read at its own dial (five arms, five different dials). Red if one arm's dial is
   used for another or if the dial is looked up in the wrong arm's document.
3. No gap, no known offer, no dial: the figures that need them are None, never 0 or a division by
   zero; a missing arm 5 leaves the information cell and the other way round. Red if a guard is
   removed.
4. The same splits and the same run: `ValueError` otherwise, for arm 4 and arm 5 as for the others.
5. `word` at the committed bars through `read_units`: a gain, a loss, no difference, a share equal
   to the bar, arm 5's loss limits (slot, driver, undefined), never a loss softened by the limits.
   The limits are arm 5's: arm 3's and arm 4's losses are not read. Red if `>` becomes `>=`, if a
   limit is not read, or if another arm's losses are read.
6. `read_units` reads tier A, the bar of each unit (8% and 30%), arm 5's losses, never tier B; a
   unit without a bar is printed and not judged; a judged unit missing from the comparison or that
   cannot be split has no share and no measurable difference; a judged unit whose arm 1 mean offer
   is 0 is NOT JUDGED: no word, so the one-line reading does not read it (the first trial's reader
   gives such a unit "no measurable difference"). Red if the zero case gets a word, or if a unit
   with no figure at all is called not judged.
7. `one_line`: helps with a gain and no loss, hurts with a loss and no gain, otherwise the mixed
   sentence; a unit not judged is not read; with no word at all the line says no unit was judged.
   The words name the setting. Red if a gain beside a loss reads as helps.
8. Arm 4 cannot change the reading: its figures changed to a large gain or a large loss leave
   `read_units` and `one_line` as they were, while the information cell moves. Red if `read_units`
   or `one_line` reads the information cell.
9. The committed bars and limits are the ones the plan and the settings give; the rule text names
   the one judged comparison, the information comparison and the not-judged units.
"""

import json
from pathlib import Path

import pytest

from fleetreserve import crc, forecast_energy_2
from fleetreserve import forecast_energy_2_reading as reading
from fleetreserve import forecast_energy_reading as first_reading

SETTINGS = json.loads(Path(forecast_energy_2.SETTINGS_FILE).read_text())
BARS = forecast_energy_2.bars_of(SETTINGS)
LIMITS = forecast_energy_2.limits_of(SETTINGS)
WORKPLACE, NORWAY, OTHER = "nrel-from-may-2019", "norway-development-pooled", "nrel-before-may-2019"
OK = {"slot": 0.05, "driver": 0.02}  # losses inside the limits
OVER = {"slot": 0.5, "driver": 0.5}  # losses over both limits
DIAL = 0.02


def entries(offers) -> list[dict]:
    return [
        {"seed": seed, "offer_kwh": float(offer), "scale": 0.5}
        for seed, offer in enumerate(offers, start=1)
    ]


def found(*per_arm) -> dict:
    """`own_dial`-like figures of the five arms from (dial, offers, losses) triples."""
    out = {}
    for arm, (dial, offers, losses) in zip(reading.ARMS, per_arm, strict=True):
        out[arm] = {
            "dial": dial,
            "entries": entries(offers),
            "summary": {
                "mean_test_slot_loss": losses["slot"],
                "mean_test_driver_loss": losses["driver"],
            },
        }
    return out


KNOWN = (DIAL, [10, 10, 10, 10], OK)
EMPIRICAL = (DIAL, [4, 6, 5, 5], OK)
CAUTIOUS = (DIAL, [6, 8, 7, 7], OK)
NEED = (DIAL, [5, 6, 7, 6], OK)
ALL = (DIAL, [7, 9, 8, 8], OK)


def at_share(share: float, losses=OK, known=10.0, need=NEED, cautious=CAUTIOUS) -> dict:
    """Five arms whose judged difference (arm 5 minus arm 2) is `share` of arm 1's offer."""
    return found(
        (DIAL, [known] * 4, OK),
        (DIAL, [5.0] * 4, OK),
        cautious,
        need,
        (DIAL, [5.0 + known * share] * 4, losses),
    )


def comparison_of(per_unit: dict) -> dict:
    """A comparison whose units hold `tier_block` of the given five-arm figures in both tiers."""
    return {
        unit: {
            "status": "calibratable",
            "counting_days": 30,
            "tiers": {tier: reading.tier_block(arms) for tier in ("A", "B")},
        }
        for unit, arms in per_unit.items()
    }


# --- 1. known numbers -----------------------------------------------------------------------------


def test_tier_block_on_known_numbers():
    block = reading.tier_block(found(KNOWN, EMPIRICAL, CAUTIOUS, NEED, ALL))
    judged, information = block["judged"], block["information"]
    assert judged["trial_arm"] == "cautious-all" and information["trial_arm"] == "cautious-need"
    assert judged["cell"]["mean_main_offer_kwh_per_test_day"] == 5.0  # arm 2 is the main side
    assert judged["cell"]["mean_trial_offer_kwh_per_test_day"] == 8.0  # arm 5 is the trial side
    assert judged["cell"]["mean_difference_kwh_per_test_day"] == 3.0
    assert (judged["cell"]["splits_trial_offers_more"], judged["cell"]["splits_equal"]) == (4, 0)
    assert information["cell"]["mean_trial_offer_kwh_per_test_day"] == 6.0  # arm 4
    assert information["cell"]["mean_difference_kwh_per_test_day"] == 1.0
    assert (
        information["cell"]["splits_trial_offers_more"],
        information["cell"]["splits_equal"],
    ) == (
        3,
        1,
    )
    assert block["mean_offer_kwh_per_test_day"] == {
        "known": 10.0,
        "empirical": 5.0,
        "cautious": 7.0,
        "cautious-need": 6.0,
        "cautious-all": 8.0,
    }
    assert block["known_mean"] == 10.0
    assert block["share_of_known_offer"] == {
        "known": 1.0,
        "empirical": 0.5,
        "cautious": 0.7,
        "cautious-need": 0.6,
        "cautious-all": 0.8,
    }
    assert judged["difference_share_of_known_offer"] == pytest.approx(0.3)
    assert judged["recovered_share_of_the_gap"] == pytest.approx(0.6)
    assert information["difference_share_of_known_offer"] == pytest.approx(0.1)
    assert information["recovered_share_of_the_gap"] == pytest.approx(0.2)
    assert block["known_against_empirical"]["mean_difference_kwh_per_test_day"] == -5.0
    assert reading.share(block) == pytest.approx(0.3)


def test_each_comparison_carries_its_own_arms_losses():
    own = {"slot": 0.07, "driver": 0.03}
    block = reading.tier_block(
        found(KNOWN, EMPIRICAL, (DIAL, [6, 8, 7, 7], OVER), (DIAL, [5, 6, 7, 6], own), ALL)
    )
    assert block["judged"]["test_losses"] == OK  # arm 5's, not arm 3's or arm 4's
    assert block["information"]["test_losses"] == own  # arm 4's


def test_a_worse_judged_arm_has_a_negative_difference():
    block = reading.tier_block(found(KNOWN, ALL, CAUTIOUS, NEED, EMPIRICAL))  # arms 2 and 5 swapped
    assert block["judged"]["cell"]["mean_difference_kwh_per_test_day"] == -3.0
    assert reading.share(block) == pytest.approx(-0.3)


# --- 2. each arm at its own dial ------------------------------------------------------------------


def document_and_offers(arm_dial, by_dial):
    """Results and per-split offers of one arm: `by_dial` maps dial -> offers; the arm picked
    `arm_dial` for every unit and tier."""
    configurations, found_offers = {}, {}
    for dial, offers in [(None, [0, 0, 0, 0]), *by_dial.items()]:
        name = f"config-{dial}"
        tiers = {t: entries(offers) for t in ("A", "B")}
        found_offers[name] = {"driver_risk": dial, "tiers": tiers}
        configurations[name] = {
            "driver_risk": dial,
            "tiers": {
                t: {
                    "mean_test_slot_loss": 0.01 * dial if dial else None,
                    "mean_test_driver_loss": 0.001 * dial if dial else None,
                }
                for t in ("A", "B")
            },
        }
    document = {
        "run": {
            "kappa": 0.2,
            "dials": list(by_dial),
            "seeds": [1, 4],
            "splits": 4,
            "scales": [0.5, 1.0],
            "n_calibration": 20,
            "tiers": ["A", "B"],
            "rung": "one-way",
        },
        "units": {
            WORKPLACE: {
                "status": "calibratable",
                "counting_days": 30,
                "configurations": configurations,
            }
        },
        "dial_choice": {WORKPLACE: {t: {"dial": arm_dial} for t in ("A", "B")}},
    }
    return document, {WORKPLACE: found_offers}


def test_each_arm_is_read_at_its_own_dial():
    dials = {
        "known": 0.05,
        "empirical": 0.02,
        "cautious": 0.1,
        "cautious-need": 0.02,
        "cautious-all": 0.05,
    }
    figures = {  # arm -> offers at each dial; the arm's own dial is the one its figures favour
        "known": {0.02: [9] * 4, 0.05: [10] * 4, 0.1: [3] * 4},
        "empirical": {0.02: [5] * 4, 0.05: [1] * 4, 0.1: [1] * 4},
        "cautious": {0.02: [2] * 4, 0.05: [2] * 4, 0.1: [7] * 4},
        "cautious-need": {0.02: [6] * 4, 0.05: [1] * 4, 0.1: [1] * 4},
        "cautious-all": {0.02: [2] * 4, 0.05: [8] * 4, 0.1: [3] * 4},
    }
    documents, offers = {}, {}
    for arm in reading.ARMS:
        documents[arm], offers[arm] = document_and_offers(dials[arm], figures[arm])
    comparison = reading.compare(documents, offers)
    block = comparison[WORKPLACE]["tiers"]["A"]
    assert block["picked_dial"] == dials
    assert block["mean_offer_kwh_per_test_day"] == {
        "known": 10.0,
        "empirical": 5.0,
        "cautious": 7.0,
        "cautious-need": 6.0,
        "cautious-all": 8.0,
    }
    assert block["judged"]["cell"]["mean_difference_kwh_per_test_day"] == 3.0  # 8 at 0.05 - 5
    assert (
        block["information"]["cell"]["mean_difference_kwh_per_test_day"] == 1.0
    )  # 6 - 5, both 0.02
    # arm 5's losses are those of its own dial (0.05): 0.01 * 0.05 and 0.001 * 0.05
    assert block["judged"]["test_losses"] == {
        "slot": pytest.approx(0.0005),
        "driver": pytest.approx(0.00005),
    }
    assert set(comparison[WORKPLACE]["tiers"]) == {"A", "B"}  # the tiers are separate blocks


# --- 3. nothing to divide by ----------------------------------------------------------------------


def test_figures_that_need_a_gap_or_an_offer_or_a_dial_are_none():
    no_gap = reading.tier_block(found(KNOWN, (DIAL, [10] * 4, OK), CAUTIOUS, NEED, ALL))
    assert no_gap["judged"]["recovered_share_of_the_gap"] is None  # the gap of arms 1 and 2 is 0
    assert no_gap["judged"]["difference_share_of_known_offer"] == pytest.approx(-0.2)
    zero = reading.tier_block(found((DIAL, [0] * 4, OK), EMPIRICAL, CAUTIOUS, NEED, ALL))
    assert zero["judged"]["difference_share_of_known_offer"] is None
    assert zero["information"]["difference_share_of_known_offer"] is None
    assert set(zero["share_of_known_offer"].values()) == {None}
    assert zero["known_mean"] == 0.0 and reading.share(zero) is None
    no_dials = dict.fromkeys(reading.ARMS)
    nothing = reading.tier_block(no_dials)
    assert nothing["judged"] is None and nothing["information"] is None
    assert nothing["picked_dial"] == dict.fromkeys(reading.ARMS)
    assert nothing["known_mean"] is None and reading.share(nothing) is None
    assert reading.share(None) is None


def test_a_missing_arm_5_leaves_the_information_cell_and_the_other_way_round():
    arms = found(KNOWN, EMPIRICAL, CAUTIOUS, NEED, ALL)
    without_5 = {**arms, "cautious-all": None}
    block = reading.tier_block(without_5)
    assert block["judged"] is None and block["information"] is not None
    assert reading.share(block) is None
    without_4 = {**arms, "cautious-need": None}
    block = reading.tier_block(without_4)
    assert block["information"] is None and block["judged"] is not None
    assert block["mean_offer_kwh_per_test_day"]["cautious-need"] is None
    without_1 = {**arms, "known": None}
    block = reading.tier_block(without_1)
    assert block["judged"] is None and block["information"] is None
    assert block["known_against_empirical"] is None and block["known_mean"] is None


# --- 4. the same splits and the same run ----------------------------------------------------------


@pytest.mark.parametrize("arm", ["cautious-need", "cautious-all"])
def test_splits_of_other_seeds_are_refused(arm):
    arms = found(KNOWN, EMPIRICAL, CAUTIOUS, NEED, ALL)
    arms[arm]["entries"][1]["seed"] = 9
    with pytest.raises(ValueError, match="seeds"):
        reading.tier_block(arms)


@pytest.mark.parametrize("arm", ["cautious-need", "cautious-all"])
@pytest.mark.parametrize("key", ["kappa", "dials", "seeds", "scales", "tiers", "rung"])
def test_arms_of_another_run_are_refused(arm, key):
    documents, offers = {}, {}
    for name in reading.ARMS:
        documents[name], offers[name] = document_and_offers(DIAL, {DIAL: [1, 2, 3, 4]})
    documents[arm]["run"][key] = "other"
    with pytest.raises(ValueError, match=key):
        reading.compare(documents, offers)


# --- 5 and 6. the word and read_units -------------------------------------------------------------

BAR = 0.08


def word_at(share: float, losses=OK, unit=WORKPLACE) -> tuple:
    units = reading.read_units(comparison_of({unit: at_share(share, losses)}), BARS, LIMITS)
    return units[unit]["word"], units[unit]["note"]


@pytest.mark.parametrize(
    "share, losses, expected",
    [
        (BAR + 0.01, OK, reading.GAIN),
        (BAR, OK, reading.NONE),  # a share equal to the bar is inside it
        (-BAR - 0.01, OK, reading.LOSS),
        (-BAR, OK, reading.NONE),
        (0.0, OK, reading.NONE),
        (BAR + 0.1, {"slot": LIMITS["slot"], "driver": LIMITS["driver"]}, reading.GAIN),
        (BAR + 0.1, {"slot": LIMITS["slot"] + 0.001, "driver": 0.0}, reading.NONE),
        (BAR + 0.1, {"slot": 0.0, "driver": LIMITS["driver"] + 0.001}, reading.NONE),
        (BAR + 0.1, OVER, reading.NONE),
        (BAR + 0.1, {"slot": None, "driver": 0.0}, reading.NONE),
        (BAR + 0.1, {"slot": 0.0, "driver": None}, reading.NONE),
        (-BAR - 0.1, OVER, reading.LOSS),  # losses never soften a loss
    ],
)
def test_the_word_at_the_bar_and_the_limits_of_arm_5(share, losses, expected):
    assert word_at(share, losses)[0] == expected


def exact_block(value: float, known: float = 10.0, losses=OK) -> dict:
    """A comparison whose judged share is exactly `value` (no arithmetic on offers), so that a share
    equal to a bar is really equal to it."""
    block = {
        "known_mean": known,
        "judged": {"difference_share_of_known_offer": value, "test_losses": losses},
    }
    return {"status": "calibratable", "counting_days": 30, "tiers": {"A": block}}


@pytest.mark.parametrize(
    "value, expected",
    [
        (BAR, reading.NONE),  # exactly the bar: inside it
        (BAR + 1e-9, reading.GAIN),
        (-BAR, reading.NONE),  # exactly minus the bar: inside it
        (-BAR - 1e-9, reading.LOSS),
    ],
)
def test_a_share_exactly_at_the_bar_is_inside_it(value, expected):
    units = reading.read_units({WORKPLACE: exact_block(value)}, {WORKPLACE: BAR}, LIMITS)
    assert units[WORKPLACE]["word"] == expected


def test_a_gain_stopped_by_a_loss_limit_says_why():
    said, note = word_at(0.5, {"slot": 0.2, "driver": 0.01})
    assert said == reading.NONE and "slot" in note and "driver" not in note
    said, note = word_at(0.5, {"slot": 0.2, "driver": 0.9})
    assert "slot and driver" in note
    said, note = word_at(0.5, {"slot": None, "driver": 0.01})
    assert said == reading.NONE and "not defined" in note
    assert word_at(0.5) == (reading.GAIN, None)


def test_the_loss_limits_are_arm_5s_and_no_other_arms():
    arm_3_and_4_over = at_share(
        0.5, OK, need=(DIAL, [5, 6, 7, 6], OVER), cautious=(DIAL, [7] * 4, OVER)
    )
    units = reading.read_units(comparison_of({WORKPLACE: arm_3_and_4_over}), BARS, LIMITS)
    assert units[WORKPLACE]["word"] == reading.GAIN  # arm 5 is inside; arms 3 and 4 are not read
    arm_5_over = at_share(0.5, OVER)
    units = reading.read_units(comparison_of({WORKPLACE: arm_5_over}), BARS, LIMITS)
    assert units[WORKPLACE]["word"] == reading.NONE


def test_read_units_reads_tier_a_with_each_units_bar_and_not_tier_b():
    comparison = comparison_of({WORKPLACE: at_share(0.10), NORWAY: at_share(0.10)})
    units = reading.read_units(comparison, BARS, LIMITS)
    assert units[WORKPLACE]["word"] == reading.GAIN  # 10% is above the 8% bar of the workplace unit
    assert units[NORWAY]["word"] == reading.NONE  # and inside the 30% bar of the Norwegian unit
    assert units[WORKPLACE]["bar"] == 0.08 and units[NORWAY]["bar"] == 0.30
    tier_b_big = comparison_of({WORKPLACE: at_share(0.0), NORWAY: at_share(0.0)})
    for entry in tier_b_big.values():
        entry["tiers"]["B"] = reading.tier_block(at_share(0.9))
    again = reading.read_units(tier_b_big, BARS, LIMITS)
    assert {u["word"] for u in again.values()} == {reading.NONE}


def test_a_unit_without_a_bar_is_printed_and_not_judged():
    comparison = comparison_of({OTHER: at_share(0.9), WORKPLACE: at_share(0.0)})
    units = reading.read_units(comparison, BARS, LIMITS)
    assert units[OTHER] == {
        "share": pytest.approx(0.9),
        "bar": None,
        "word": None,
        "note": reading.NOT_JUDGED,
    }
    assert reading.one_line(units) == reading.MIXED  # a big share at an unjudged unit is no gain


def test_a_judged_unit_that_is_missing_or_cannot_be_split_has_no_share():
    cannot = {WORKPLACE: {"status": "too few days", "counting_days": 3}}
    units = reading.read_units(cannot, BARS, LIMITS)
    assert units[WORKPLACE]["share"] is None and units[WORKPLACE]["word"] == reading.NONE
    assert "no share is defined" in units[WORKPLACE]["note"]
    assert units[NORWAY]["share"] is None and units[NORWAY]["word"] == reading.NONE  # missing


def test_a_judged_unit_with_no_dial_has_no_share_and_no_measurable_difference():
    no_dial = dict.fromkeys(reading.ARMS)
    units = reading.read_units(comparison_of({WORKPLACE: no_dial}), BARS, LIMITS)
    assert units[WORKPLACE]["word"] == reading.NONE and units[WORKPLACE]["share"] is None


def test_a_unit_where_arm_1_offers_0_is_not_judged_and_has_no_word():
    zero = at_share(0.5, known=0.0)
    comparison = comparison_of({WORKPLACE: zero, NORWAY: at_share(0.5)})
    units = reading.read_units(comparison, BARS, LIMITS)
    assert units[WORKPLACE] == {
        "share": None,
        "bar": BARS[WORKPLACE],
        "word": None,
        "note": reading.NOT_JUDGED_NO_BAR,
    }
    assert "not judged" in reading.NOT_JUDGED_NO_BAR
    assert "no bar can be defined" in reading.NOT_JUDGED_NO_BAR
    assert units[NORWAY]["word"] == reading.GAIN  # the other unit is read as it always was
    assert reading.one_line(units) == reading.HELPS  # the zero unit is not read
    # the first trial's reader gives the same unit a word, which this reader does not
    old = first_reading.read_units(
        {
            WORKPLACE: {
                "status": "calibratable",
                "tiers": {
                    "A": {
                        "difference_share_of_known_offer": None,
                        "cautious_test_losses": OK,
                    }
                },
            }
        },
        {WORKPLACE: 0.08},
        LIMITS,
    )
    assert old[WORKPLACE]["word"] == first_reading.NONE


def test_bar_defined_is_false_only_for_a_known_mean_that_is_not_above_0():
    assert reading.bar_defined(reading.tier_block(at_share(0.1)))
    assert not reading.bar_defined(reading.tier_block(at_share(0.1, known=0.0)))
    assert reading.bar_defined(None)  # no figure at all: the first trial's wording stands
    assert reading.bar_defined(reading.tier_block(dict.fromkeys(reading.ARMS)))


# --- 7. the one line ------------------------------------------------------------------------------


def words(first: str | None, second: str | None, third: str | None = None) -> dict:
    out = {WORKPLACE: {"word": first}, NORWAY: {"word": second}}
    if third is not None:
        out[OTHER] = {"word": None}
    return out


def test_the_one_line_reading_names_the_setting():
    gain, loss, none = reading.GAIN, reading.LOSS, reading.NONE
    assert reading.one_line(words(gain, none)) == reading.HELPS
    assert reading.one_line(words(none, gain)) == reading.HELPS
    assert reading.one_line(words(gain, gain)) == reading.HELPS
    assert reading.one_line(words(loss, none)) == reading.HURTS
    assert reading.one_line(words(loss, loss)) == reading.HURTS
    assert reading.one_line(words(gain, loss)) == reading.MIXED  # a mixed result is mixed
    assert reading.one_line(words(loss, gain)) == reading.MIXED
    assert reading.one_line(words(none, none)) == reading.MIXED  # no difference at all
    assert reading.one_line(words(gain, none, "unjudged")) == reading.HELPS  # not judged: not read
    assert reading.one_line(words(None, None, "unjudged")) == reading.NO_JUDGED_UNIT
    assert reading.one_line({}) == reading.NO_JUDGED_UNIT
    assert reading.HELPS == "all three energy bounds at level 0.05 help"
    assert reading.HURTS == "all three energy bounds at level 0.05 hurt"
    assert reading.MIXED == "no measurable difference; a mixed result is reported as mixed"
    for line in (reading.HELPS, reading.HURTS):
        assert "the cautious bounds" not in line  # no line speaks of "the cautious bounds" alone


def test_the_one_lines_of_the_settings_file_are_those_of_the_code():
    block = SETTINGS["reading"]["one_line"]
    assert set(block) == {reading.HELPS, reading.HURTS, "otherwise", "no judged unit"}
    assert block["otherwise"] == reading.MIXED


# --- 8. arm 4 cannot change the reading -----------------------------------------------------------


def test_arm_4s_figures_cannot_change_the_one_line_reading_or_the_words():
    quiet = (DIAL, [5, 5, 5, 5], OK)  # arm 4 as arm 2: difference 0
    huge_gain = (DIAL, [50, 50, 50, 50], OK)
    huge_loss = (DIAL, [0, 0, 0, 0], OVER)
    per_arm4 = {}
    for name, arm_4 in (("quiet", quiet), ("gain", huge_gain), ("loss", huge_loss)):
        comparison = comparison_of(
            {WORKPLACE: at_share(0.02, need=arm_4), NORWAY: at_share(0.5, need=arm_4)}
        )
        units = reading.read_units(comparison, BARS, LIMITS)
        per_arm4[name] = (units, reading.one_line(units), comparison)
    assert per_arm4["quiet"][0] == per_arm4["gain"][0] == per_arm4["loss"][0]
    assert per_arm4["quiet"][1] == per_arm4["gain"][1] == per_arm4["loss"][1] == reading.HELPS
    assert per_arm4["quiet"][0][WORKPLACE]["word"] == reading.NONE
    assert per_arm4["quiet"][0][NORWAY]["word"] == reading.GAIN
    # while the information cell does move
    cells = [
        per_arm4[name][2][WORKPLACE]["tiers"]["A"]["information"]["cell"][
            "mean_difference_kwh_per_test_day"
        ]
        for name in ("quiet", "gain", "loss")
    ]
    assert cells == [0.0, 45.0, -5.0]
    for name in per_arm4:
        for entry in per_arm4[name][0].values():
            assert set(entry) == {"share", "bar", "word", "note"}  # no information figure in it


def test_the_only_comparison_read_is_arm_5_minus_arm_2():
    assert reading.COMPARISONS == {"judged": "cautious-all", "information": "cautious-need"}
    assert reading.MAIN_ARM == "empirical" and reading.REFERENCE_ARM == "known"
    assert SETTINGS["reading"]["judged_comparison"]["trial_arm"] == reading.JUDGED_ARM
    assert SETTINGS["reading"]["judged_comparison"]["main_arm"] == reading.MAIN_ARM
    assert SETTINGS["reading"]["information_comparison"]["trial_arm"] == reading.INFORMATION_ARM
    assert SETTINGS["reading"]["information_comparison"]["main_arm"] == reading.MAIN_ARM


# --- 9. the committed bars and limits, and the rule text ------------------------------------------


def test_the_committed_bars_and_limits_are_those_of_the_plan_and_the_settings():
    assert BARS == {WORKPLACE: 0.08, NORWAY: 0.30}
    assert (
        LIMITS
        == {"slot": crc.ALPHA_SLOT, "driver": crc.ALPHA_DRIVER}
        == {"slot": 0.1, "driver": 0.05}
    )
    assert SETTINGS["reading"]["no_other_sites"] == reading.NO_OTHER_SITES
    assert OTHER not in BARS
    first_bars = json.loads(Path(forecast_energy_2.first_trial.SETTINGS_FILE).read_text())
    assert first_bars["reading"]["bars"] == SETTINGS["reading"]["bars"]  # the first trial's bars


def test_the_rule_text_names_the_judged_comparison_the_information_one_and_the_unjudged_units():
    text = reading.RULE
    for part in (
        "ONE judged comparison",
        "arm 5",
        "arm 2",
        "arm 1",
        "its own dial",
        "bar",
        "slot limit",
        "driver limit",
        "Arm 4",
        "information only",
        "read by no rule",
        "not judged",
        "all three energy bounds at level 0.05",
    ):
        assert part in text, part
    assert "mixed" in text and "Tier B is information" in text
    assert "the cautious bounds" not in text

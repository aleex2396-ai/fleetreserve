"""The pairing and the reading rule of the forecast trial `energy` (`forecast_energy_reading`) on
made-up results. Written to be run through the test slot. Each test says which edit turns it red.

1. `tier_block` on known numbers (arm 1 `known` 10 in every split, arm 2 `empirical` 4, 6, 5, 5, arm
   3 `cautious` 6, 8, 7, 7): the paired difference is arm 3 minus arm 2 (+2 in every split), the
   share is over arm 1's mean offer (0.2), arm 2's and 3's shares of arm 1's offer (0.5 and 0.7),
   the recovered share of the gap (2 of 5). Red if the difference is turned (arm 2 minus arm 3), if
   the share is taken over arm 2's mean (0.4) or arm 3's, or if the gap is taken the wrong way.
2. Each arm is read at its own dial: the three arms pick different dials and the cell is made of
   each arm's own entries, so a cell made at one arm's dial for all differs. Red if the dial of
   arm 2 is used for arm 3, or if the dial is looked up in the wrong arm's document.
3. No gap, no known offer, no dial: the figures that need them are None, never 0 or a division by
   zero. Red if a guard of `_ratio` is removed.
4. The two runs must be of the same seeds in the same order, and the arms of the same run
   (level, dials, scales, units): `ValueError` otherwise. Red if a check is removed.
5. `word` at the committed bars: a share just above the bar with losses inside the limits is a
   gain, one equal to the bar is not; just below minus the bar is a loss, one equal to minus the
   bar is not; a share above the bar with a loss over its limit (slot, driver, either) or with an
   undefined loss is no measurable difference, with a note, and is never a loss; a limit equal to
   the loss is inside. Red if `>` becomes `>=`, if the slot or the driver limit is not read, if a
   loss limit exceeded turns the word into a loss, or if the negative side uses the wrong sign.
6. `read_units` reads tier A, the bars of each unit (8% and 30%), arm 3's losses, and never tier B;
   a unit without a bar is printed and not judged (no word, no gain, no loss); a judged unit
   missing from the comparison or that cannot be split has no share and no measurable
   difference. Red if tier B is read, if one bar is used for both units, or if a unit without a bar
   is judged.
7. `one_line`: helps with a gain and no loss, hurts with a loss and no gain, otherwise the mixed
   sentence (also for a gain beside a loss and for no difference at all); a unit that is not judged
   does not count. Red if a gain beside a loss reads as helps.
8. The committed bars and limits are the ones the plan and the settings give.
"""

import json
from pathlib import Path

import pytest

from fleetreserve import crc, forecast_energy
from fleetreserve import forecast_energy_reading as reading

SETTINGS = json.loads(Path(forecast_energy.SETTINGS_FILE).read_text())
BARS = forecast_energy.bars_of(SETTINGS)
LIMITS = forecast_energy.limits_of(SETTINGS)
WORKPLACE, NORWAY, OTHER = "nrel-from-may-2019", "norway-development-pooled", "nrel-before-may-2019"
OK = {"slot": 0.05, "driver": 0.02}  # losses inside the limits


def entries(offers) -> list[dict]:
    return [
        {"seed": seed, "offer_kwh": float(offer), "scale": 0.5}
        for seed, offer in enumerate(offers, start=1)
    ]


def found(*per_arm) -> dict:
    """`own_dial`-like figures of the three arms from (dial, offers, losses) triples."""
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


KNOWN = (0.02, [10, 10, 10, 10], OK)
EMPIRICAL = (0.02, [4, 6, 5, 5], OK)
CAUTIOUS = (0.02, [6, 8, 7, 7], OK)


# --- 1. known numbers -----------------------------------------------------------------------------


def test_tier_block_on_known_numbers():
    block = reading.tier_block(found(KNOWN, EMPIRICAL, CAUTIOUS))
    cell = block["cell"]
    assert cell["mean_main_offer_kwh_per_test_day"] == 5.0  # arm 2 is the main side
    assert cell["mean_trial_offer_kwh_per_test_day"] == 7.0  # arm 3 is the trial side
    assert cell["mean_difference_kwh_per_test_day"] == 2.0
    assert (cell["splits_trial_offers_more"], cell["splits_equal"]) == (4, 0)
    assert block["mean_offer_kwh_per_test_day"] == {
        "known": 10.0,
        "empirical": 5.0,
        "cautious": 7.0,
    }
    assert block["share_of_known_offer"] == {"empirical": 0.5, "cautious": 0.7}
    assert block["difference_share_of_known_offer"] == pytest.approx(0.2)
    assert block["recovered_share_of_the_gap"] == pytest.approx(0.4)
    assert block["cautious_test_losses"] == {"slot": 0.05, "driver": 0.02}
    assert (
        block["known_against_empirical"]["mean_difference_kwh_per_test_day"] == -5.0
    )  # arm 2 - arm 1
    assert reading.share(block) == pytest.approx(0.2)


def test_a_worse_cautious_arm_has_a_negative_difference():
    block = reading.tier_block(found(KNOWN, CAUTIOUS, EMPIRICAL))  # arms 2 and 3 swapped
    assert block["cell"]["mean_difference_kwh_per_test_day"] == -2.0
    assert reading.share(block) == pytest.approx(-0.2)


# --- 2. each arm at its own dial ------------------------------------------------------------------


def document_and_offers(arm_dials, by_dial):
    """Results and per-split offers of one arm: `by_dial` maps dial -> offers; the arm picked
    `arm_dials`'s dial for every unit and tier."""
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
            "dials": [d for d in by_dial],
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
        "dial_choice": {WORKPLACE: {t: {"dial": arm_dials} for t in ("A", "B")}},
    }
    return document, {WORKPLACE: found_offers}


def test_each_arm_is_read_at_its_own_dial():
    by_dial = {
        "known": {0.02: [9, 9, 9, 9], 0.05: [10, 10, 10, 10]},
        "empirical": {0.02: [5, 5, 5, 5], 0.05: [1, 1, 1, 1]},
        "cautious": {0.02: [2, 2, 2, 2], 0.05: [8, 8, 8, 8]},
    }
    picks = {"known": 0.05, "empirical": 0.02, "cautious": 0.05}
    documents, offers = {}, {}
    for arm in reading.ARMS:
        documents[arm], offers[arm] = document_and_offers(picks[arm], by_dial[arm])
    comparison = reading.compare(documents, offers)
    block = comparison[WORKPLACE]["tiers"]["A"]
    assert block["picked_dial"] == picks
    assert block["mean_offer_kwh_per_test_day"] == {
        "known": 10.0,
        "empirical": 5.0,
        "cautious": 8.0,
    }
    assert block["cell"]["mean_difference_kwh_per_test_day"] == 3.0  # 8 at 0.05 minus 5 at 0.02
    wrong = 2.0 - 5.0  # arm 3 at arm 2's dial 0.02 against arm 2: another figure
    assert wrong != block["cell"]["mean_difference_kwh_per_test_day"]
    # arm 3's losses are those of its own dial (0.05): 0.01 * 0.05 and 0.001 * 0.05
    assert block["cautious_test_losses"] == {
        "slot": pytest.approx(0.0005),
        "driver": pytest.approx(0.00005),
    }
    assert set(comparison[WORKPLACE]["tiers"]) == {"A", "B"}  # the tiers are separate blocks


# --- 3. nothing to divide by ----------------------------------------------------------------------


def test_figures_that_need_a_gap_or_an_offer_or_a_dial_are_none():
    no_gap = reading.tier_block(found(KNOWN, (0.02, [10, 10, 10, 10], OK), CAUTIOUS))
    assert no_gap["recovered_share_of_the_gap"] is None  # the gap between arms 1 and 2 is 0
    assert no_gap["difference_share_of_known_offer"] == pytest.approx(-0.3)
    zero = reading.tier_block(found((0.02, [0, 0, 0, 0], OK), EMPIRICAL, CAUTIOUS))
    assert zero["difference_share_of_known_offer"] is None and zero["share_of_known_offer"] == {
        "empirical": None,
        "cautious": None,
    }
    assert reading.share(zero) is None
    nothing = reading.tier_block({"known": None, "empirical": None, "cautious": None})
    assert nothing["cell"] is None and nothing["picked_dial"] == dict.fromkeys(reading.ARMS)
    assert reading.share(nothing) is None and reading.share(None) is None


# --- 4. the same splits and the same run ----------------------------------------------------------


def test_splits_of_other_seeds_are_refused():
    arms = found(KNOWN, EMPIRICAL, CAUTIOUS)
    arms["cautious"]["entries"][1]["seed"] = 9
    with pytest.raises(ValueError, match="seeds"):
        reading.tier_block(arms)


@pytest.mark.parametrize("key", ["kappa", "dials", "seeds", "scales", "tiers", "rung"])
def test_arms_of_another_run_are_refused(key):
    documents, offers = {}, {}
    for arm in reading.ARMS:
        documents[arm], offers[arm] = document_and_offers(0.02, {0.02: [1, 2, 3, 4]})
    documents["cautious"]["run"][key] = "other"
    with pytest.raises(ValueError, match=key):
        reading.compare(documents, offers)


# --- 5. the word ----------------------------------------------------------------------------------

BAR = 0.08


@pytest.mark.parametrize(
    "value, losses, expected",
    [
        (BAR + 1e-6, OK, reading.GAIN),
        (BAR, OK, reading.NONE),  # a share equal to the bar is inside it
        (-BAR - 1e-6, OK, reading.LOSS),
        (-BAR, OK, reading.NONE),
        (0.0, OK, reading.NONE),
        (BAR + 0.1, {"slot": LIMITS["slot"], "driver": LIMITS["driver"]}, reading.GAIN),
        (BAR + 0.1, {"slot": LIMITS["slot"] + 0.001, "driver": 0.0}, reading.NONE),
        (BAR + 0.1, {"slot": 0.0, "driver": LIMITS["driver"] + 0.001}, reading.NONE),
        (BAR + 0.1, {"slot": 0.5, "driver": 0.5}, reading.NONE),
        (BAR + 0.1, {"slot": None, "driver": 0.0}, reading.NONE),
        (BAR + 0.1, {"slot": 0.0, "driver": None}, reading.NONE),
        (BAR + 0.1, None, reading.NONE),
        (-BAR - 0.1, {"slot": 0.5, "driver": 0.5}, reading.LOSS),  # losses never soften a loss
        (None, OK, reading.NONE),
    ],
)
def test_the_word_at_the_bar_and_the_limits(value, losses, expected):
    assert reading.word(value, BAR, losses, LIMITS)[0] == expected


def test_a_gain_stopped_by_a_loss_limit_says_why():
    said, note = reading.word(0.5, BAR, {"slot": 0.2, "driver": 0.01}, LIMITS)
    assert said == reading.NONE and "slot" in note and "driver" not in note
    said, note = reading.word(0.5, BAR, {"slot": 0.2, "driver": 0.9}, LIMITS)
    assert "slot and driver" in note
    said, note = reading.word(0.5, BAR, None, LIMITS)
    assert said == reading.NONE and "not defined" in note
    said, note = reading.word(None, BAR, OK, LIMITS)
    assert said == reading.NONE and "no share is defined" in note
    assert reading.word(0.5, BAR, OK, LIMITS) == (reading.GAIN, None)


# --- 6. read_units --------------------------------------------------------------------------------


def comparison(shares: dict, tier_b_share: float = 0.9, losses=OK) -> dict:
    """A comparison whose tier A blocks have the given shares of arm 1's offer; tier B another."""
    out = {}
    for unit, value in shares.items():

        def block(share_value):
            return {
                "difference_share_of_known_offer": share_value,
                "cautious_test_losses": losses,
            }

        out[unit] = {
            "status": "calibratable",
            "tiers": {"A": block(value), "B": block(tier_b_share)},
        }
    return out


def test_read_units_reads_tier_a_with_each_units_bar_and_not_tier_b():
    units = reading.read_units(
        comparison({WORKPLACE: 0.10, NORWAY: 0.10}, tier_b_share=0.9), BARS, LIMITS
    )
    assert units[WORKPLACE]["word"] == reading.GAIN  # 10% is above the 8% bar of the workplace unit
    assert units[NORWAY]["word"] == reading.NONE  # and inside the 30% bar of the Norwegian unit
    assert units[WORKPLACE]["bar"] == 0.08 and units[NORWAY]["bar"] == 0.30
    tier_b_only = reading.read_units(
        comparison({WORKPLACE: 0.0, NORWAY: 0.0}, tier_b_share=0.9), BARS, LIMITS
    )
    assert {u["word"] for u in tier_b_only.values()} == {reading.NONE}


def test_read_units_uses_arm_3s_losses_for_the_gain():
    over = {"slot": 0.5, "driver": 0.0}
    units = reading.read_units(comparison({WORKPLACE: 0.5}, losses=over), BARS, LIMITS)
    assert units[WORKPLACE]["word"] == reading.NONE and "slot" in units[WORKPLACE]["note"]


def test_a_unit_without_a_bar_is_printed_and_not_judged():
    units = reading.read_units(comparison({OTHER: 0.9, WORKPLACE: 0.0}), BARS, LIMITS)
    assert units[OTHER] == {
        "share": 0.9,
        "bar": None,
        "word": None,
        "note": reading.NOT_JUDGED,
    }
    assert reading.one_line(units) == reading.MIXED  # a big share at an unjudged unit is no gain


def test_a_judged_unit_that_is_missing_or_cannot_be_split_has_no_share():
    cannot = {WORKPLACE: {"status": "too few days", "counting_days": 3}}
    units = reading.read_units(cannot, BARS, LIMITS)
    assert units[WORKPLACE]["share"] is None and units[WORKPLACE]["word"] == reading.NONE
    assert units[NORWAY]["share"] is None and units[NORWAY]["word"] == reading.NONE  # missing


# --- 7. the one line ------------------------------------------------------------------------------


def words(first: str | None, second: str | None, third: str | None = None) -> dict:
    out = {WORKPLACE: {"word": first}, NORWAY: {"word": second}}
    if third is not None:
        out[OTHER] = {"word": None}
    return out


def test_the_one_line_reading():
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


def test_the_rule_text_names_the_pairing_the_bar_and_the_limits():
    text = reading.RULE
    for part in ("arm 3", "arm 2", "arm 1", "its own dial", "bar", "slot limit", "driver limit"):
        assert part in text
    assert "mixed" in text and "Tier B is information" in text


# --- 8. the committed bars and limits -------------------------------------------------------------


def test_the_committed_bars_and_limits_are_those_of_the_plan_and_the_settings():
    assert BARS == {WORKPLACE: 0.08, NORWAY: 0.30}
    assert (
        LIMITS
        == {"slot": crc.ALPHA_SLOT, "driver": crc.ALPHA_DRIVER}
        == {"slot": 0.1, "driver": 0.05}
    )
    assert SETTINGS["reading"]["no_other_sites"] == reading.NO_OTHER_SITES
    assert OTHER not in BARS

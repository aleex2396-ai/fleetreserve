"""`day_intervals` on a unit whose hindsight ceiling of one test day is about -1e-14 (the rounding
of a true 0 in `exact`): the run's share of that ceiling is null (`gate_calibrate._share` gives no
share for a negative ceiling), and the check of reading I11 recomputes the share by that same
rule, so it agrees with the null instead of failing the command.

The unit `neg-unit` has ten calibration days of which five offer (too few for either tier: the
run's scale is 0, its offer 0) and six test days whose first hindsight ceiling is -1e-14, the
others 100; the best-causal ceiling is 80 on every day. Its twin has 0.0 in place of -1e-14 (a
ceiling of 0 is allowed: the run's share there is 0.0). The hand unit of `test_day_pooled`
(scale 0.3 in tier B) gives the pooled block a positive offer.

1. neg-unit and the hand unit: exit 0, no problem, every interval of neg-unit present except its
   share of the hindsight ceiling, which is "no figure" with a reason (its share of the other
   ceiling has its interval); the pooled block has its rungs.
2. The pooled figures with -1e-14 equal those with 0.0 in its place (to 1e-12): the offers do
   not read a ceiling at all and the shares read it only in the sums.
3. Breaks that stay red (exit 1, a problem named): the run's share 0.3 where the rule gives none
   (negative ceiling); the run's share null where no ceiling is negative and the sums are above
   0; the run's share 0.3 against a per-day share of 0.075; a NaN ceiling (a missing one) with a
   share in the run.
"""

import json

import pytest
import test_day_pooled as base
from interval_fake import NAME, RUNG, unit, write_run
from test_day_pooled import (  # noqa: F401  (the fixtures of the pooled tests apply here)
    CALIBRATION,
    FIRST,
    HAND,
    TEST_DRAWS,
    clean_git_environment,
    hand_unit,
    inject,
    results,
    run,
    verified,
)

from fleetreserve import crc, day_intervals
from fleetreserve.gate_config import CEILINGS

NEG = "neg-unit"
TINY = -1e-14


def neg_unit(first=TINY, run_share=None, nan=False):
    """(record, entry) of the unit with too few offering calibration days (scale 0 in both
    tiers), whose first hindsight ceiling is `first`; the run's share of that ceiling is
    `run_share` (None: null). `nan`: the per-day file lacks that ceiling (None in the file)."""
    ceilings = {CEILINGS[0]: [first] + [100.0] * 5, CEILINGS[1]: [80.0] * 6}
    test = {
        "values": [10.0, 20.0, 30.0, 0.0, 40.0, 50.0],
        "cars": [3, 4, 5, 3, 4, 5],
        "offers": [True, True, True, False, True, True],
        "ceilings": ceilings,
    }
    calibration = {"thresholds": [0.5] * 5, "offers": [True] * 5 + [False] * 5}
    tiers = {"A": (0.0, None, None), "B": (0.0, None, None)}
    record, entry = unit(NEG, calibration, test, set(crc.GRID), tiers, resample=(base.SEED, 5))
    scored = entry["rungs"][RUNG]
    share = None
    if run_share is not None:
        share = {"ratio_of_sums": run_share, "mean_of_daily_ratios": None}
    for found in (scored, scored["tier_b"]):
        found["test"]["share_of_ceiling"][CEILINGS[0]] = share
    if nan:
        record["days"]["ceiling_kwh"][CEILINGS[0]][10] = None
    return record, entry


def command(tmp_path, monkeypatch, make, name="out") -> int:
    """The command on the run of the units `make()` gives, built after the draws are injected (the
    made-up run evaluated its pairs under them)."""
    inject(
        monkeypatch,
        {**CALIBRATION, NEG: CALIBRATION[FIRST]},
        {**TEST_DRAWS, NEG: TEST_DRAWS[FIRST]},
    )
    folder = write_run(tmp_path / f"20200101T000000Z-{name}", make())
    return run(folder, tmp_path / name, monkeypatch, *HAND)


def pooled(tmp_path, name="out", tier="B") -> dict:
    block = results(tmp_path / name)["pooled"]["configurations"][NAME]
    return block["rungs"][RUNG][tier]


def problems_of(tmp_path, name="out") -> list[str]:
    return json.loads((tmp_path / name / day_intervals.RESULTS_NAME).read_text())["problems"]


def test_a_ceiling_of_minus_1e_14_and_the_runs_null_share_agree(tmp_path, monkeypatch):
    assert command(tmp_path, monkeypatch, lambda: [hand_unit(FIRST), neg_unit()]) == 0
    found = results(tmp_path / "out")
    assert found["problems"] == []
    per_unit = found["configurations"][NAME][NEG][RUNG]
    for tier in ("A", "B"):
        shown = per_unit[tier]
        assert shown["scale"]["interval"] == [0.0, 0.0]
        assert shown["offer_kwh_per_counting_test_day"]["interval"] == [0.0, 0.0]
        assert shown["offer_kwh_per_car_per_day"]["interval"] == [0.0, 0.0]
        hind = shown["share_of_ceiling"][CEILINGS[0]]
        assert hind["point"] is None and hind["interval"] is None and "null" in hind["reason"]
        other = shown["share_of_ceiling"][CEILINGS[1]]
        assert other["point"] == 0.0 and other["interval"] == [0.0, 0.0]
    block = found["pooled"]["configurations"][NAME]
    assert block["units"] == [FIRST, NEG] and set(block["rungs"][RUNG]) == {"A", "B"}
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert "FAILED" not in text and f"{NEG}: no interval" not in text
    assert f"{NAME} | pooled: no pooled figure" not in text


def test_the_pooled_figures_equal_those_with_the_ceiling_set_to_zero(tmp_path, monkeypatch):
    assert command(tmp_path, monkeypatch, lambda: [hand_unit(FIRST), neg_unit()], "tiny") == 0
    zero = lambda: [hand_unit(FIRST), neg_unit(first=0.0, run_share=0.0)]  # noqa: E731
    assert command(tmp_path, monkeypatch, zero, "zero") == 0
    for tier in ("A", "B"):
        tiny, flat = pooled(tmp_path, "tiny", tier), pooled(tmp_path, "zero", tier)
        for figure in ("sum_of_unit_offers", "total_over_total_days"):
            for key in ("offer_kwh_per_counting_test_day", "offer_kwh_per_car_per_day"):
                assert tiny[figure][key]["interval"] == pytest.approx(
                    flat[figure][key]["interval"], abs=1e-12
                )
                assert tiny[figure][key]["point"] == pytest.approx(flat[figure][key]["point"])
            for name in CEILINGS:
                a = tiny[figure]["share_of_ceiling"][name]
                b = flat[figure]["share_of_ceiling"][name]
                assert a["interval"] == pytest.approx(b["interval"], abs=1e-12)
                assert a["point"] == pytest.approx(b["point"], abs=1e-12)
    offer = pooled(tmp_path, "tiny")["sum_of_unit_offers"]["offer_kwh_per_counting_test_day"]
    assert offer["point"] == pytest.approx(7.5)  # the hand unit's offer; neg-unit offers 0
    assert offer["interval"][1] > 0


def test_the_runs_share_of_0_3_where_the_rule_gives_none_is_a_problem(tmp_path, monkeypatch):
    assert command(tmp_path, monkeypatch, lambda: [hand_unit(FIRST), neg_unit(run_share=0.3)]) == 1
    mine = [p for p in problems_of(tmp_path) if NEG in p]
    assert mine and all("share of" in p and "recomputed None" in p for p in mine)
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert "rungs" not in block and NEG in block["reason"]


def test_a_null_of_the_run_where_no_ceiling_is_negative_is_a_problem(tmp_path, monkeypatch):
    jobs = lambda: [hand_unit(FIRST), neg_unit(first=0.0, run_share=None)]  # noqa: E731
    assert command(tmp_path, monkeypatch, jobs) == 1
    found = [p for p in problems_of(tmp_path) if NEG in p]
    assert found and all("share of" in p and "recomputed 0.0" in p for p in found)


def test_a_share_of_0_3_against_a_per_day_share_of_0_075_is_a_problem(tmp_path, monkeypatch):
    def make():
        record, entry = hand_unit(FIRST)
        scored = entry["rungs"][RUNG]["tier_b"]["test"]
        assert scored["share_of_ceiling"][CEILINGS[0]]["ratio_of_sums"] == pytest.approx(0.075)
        scored["share_of_ceiling"][CEILINGS[0]]["ratio_of_sums"] = 0.3
        return [(record, entry), hand_unit(base.SECOND)]

    assert command(tmp_path, monkeypatch, make) == 1
    assert any(FIRST in p and "recomputed 0.075" in p for p in problems_of(tmp_path))


def test_a_nan_ceiling_with_a_share_in_the_run_is_a_problem(tmp_path, monkeypatch):
    jobs = lambda: [hand_unit(FIRST), neg_unit(first=100.0, run_share=0.0, nan=True)]  # noqa: E731
    assert command(tmp_path, monkeypatch, jobs) == 1
    assert any(NEG in p and "recomputed None" in p for p in problems_of(tmp_path))


def test_a_ceiling_of_minus_0_5_is_a_problem_even_with_the_runs_null_share(tmp_path, monkeypatch):
    assert command(tmp_path, monkeypatch, lambda: [hand_unit(FIRST), neg_unit(first=-0.5)]) == 1
    found = [p for p in problems_of(tmp_path) if NEG in p]
    assert found and all("below the rounding allowance" in p and "-0.5" in p for p in found)
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert "rungs" not in block and NEG in block["reason"]
    assert results(tmp_path / "out")["configurations"][NAME][NEG] is None


def test_a_ceiling_of_minus_1e_12_still_agrees_with_the_runs_null_share(tmp_path, monkeypatch):
    assert command(tmp_path, monkeypatch, lambda: [hand_unit(FIRST), neg_unit(first=-1e-12)]) == 0
    assert results(tmp_path / "out")["problems"] == []

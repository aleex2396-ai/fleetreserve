"""The paired comparison of the forecast trial `noset` (`forecast_noset_pairs`) on made-up
documents of `site_dial_whatif.run`. Written, not run. Each test says which edit turns it red.

Made-up numbers, one unit `U`, three splits (seeds 1 to 3), two dials (0.03 and 0.1), tier A. Per
document and split: the arm S1 (the check line), and the offer of each dial calibrated on all the
calibration days (`by_dial`), of which the plan's rule takes the dial with the largest mean:

  arm      seed   S1          dial 0.03     dial 0.1       plan rule: dial, offers
  tail     -      10,10,10    10,10,10      0,0,0          0.03: 10,10,10
  tuned    a      4,4,4       6,6,6         10,12,14       0.1: 10,12,14
  tuned    b      6,6,6       8,8,8         12,14,16       0.1: 12,14,16
  inputs   a      9,9,9       20,20,20      1,1,1          0.03: 20,20,20
  inputs   b      11,11,11    1,1,1         22,22,22       0.1: 22,22,22
  noset    a      6,6,6       14,16,18      1,1,1          0.03: 14,16,18
  noset    b      8,8,8       16,18,20      2,2,2          0.03: 16,18,20

so, per split mean over the seeds, on the plan-rule line: tail 10; tuned 11,13,15; inputs 21;
noset 15,17,19 (seed figures tuned 12 and 14, inputs 20 and 22, noset 16 and 18); on the check
line: tail 10; tuned 5; inputs 10; noset 7 (seed figures tuned 4 and 6, inputs 9 and 11, noset 6
and 8). The no-encoder arm minus the comparison arm, plan-rule line: against tuned +4, against
inputs -4, against tail +5, +7, +9; check line: +2, -3, -3.

1. The figures by hand on both lines: per-split mean over the seeds, then the difference (not the
   other way round), seed figures with their least and greatest, mean offers, counts. Red if
   `mean_over_seeds` is taken after pairing or over the wrong axis.
2. The plan-rule line reads `by_dial` at the dial of the plan's rule and never S1, `fixed`, S2 or
   S3; the check line reads S1 and never `by_dial`. Red if the line is swapped in `split_offers`.
3. The rule is applied to every run alike: each seed has its own dial (inputs: 0.03 and 0.1); the
   dial of a tier is picked from that tier's offers; a tie goes to the smaller dial. Red if one
   dial is taken for all the seeds of an arm or for both tiers, or if a tie goes to the larger.
4. A missing arm, or a document that holds no splits of the unit, gives no cell (None), on both
   lines, and the other cells stay. Red if a missing arm is read as 0 offers.
5. A unit that cannot be split has its status only.
6. Documents of different runs are refused, as are splits of different seeds.
"""

import copy

import pytest

from fleetreserve import forecast_noset_pairs as pairs

UNIT, OTHER = "U", "V"
RUN = {
    "kappa": 0.3,
    "dials": [0.03, 0.1],
    "seeds": [1, 3],
    "splits": 3,
    "scales": [0.5, 1.0],
    "n_calibration": 50,
    "tiers": ["A", "B"],
    "rung": "one-way",
}
SHIFT = 100.0
WRONG = {"offer_kwh_per_test_day": 999.0}


def tier(s1: float, d03: float, d1: float) -> dict:
    return {
        "S1": {"offer_kwh_per_test_day": s1},
        "by_dial": {"0.03": d03, "0.1": d1},
        "fixed": WRONG,
        "S2": WRONG,
        "S3": WRONG,
    }


def split(seed: int, a: tuple, b: tuple) -> dict:
    return {"seed": seed, "tiers": {"A": tier(*a), "B": tier(*b)}}


def document(s1, d03, d1, seeds=(1, 2, 3), b=None) -> dict:
    """A results document: per split the offers of tier A as given, tier B each plus SHIFT unless
    `b` (three lists, as the first three) gives them."""
    rows_a = list(zip(s1, d03, d1, strict=True))
    rows_b = (
        [(x + SHIFT, y + SHIFT, z + SHIFT) for x, y, z in rows_a]
        if b is None
        else list(zip(*b, strict=True))
    )
    return {
        "run": copy.deepcopy(RUN),
        "units": {
            UNIT: {
                "status": "calibratable",
                "counting_days": 60,
                "splits": [split(s, a, bb) for s, a, bb in zip(seeds, rows_a, rows_b, strict=True)],
            },
            OTHER: {"status": "not calibratable: 5 days", "counting_days": 5},
        },
    }


def documents() -> dict:
    return {
        "tail": [document([10, 10, 10], [10, 10, 10], [0, 0, 0])],
        "tuned": [
            document([4, 4, 4], [6, 6, 6], [10, 12, 14]),
            document([6, 6, 6], [8, 8, 8], [12, 14, 16]),
        ],
        "inputs": [
            document([9, 9, 9], [20, 20, 20], [1, 1, 1]),
            document([11, 11, 11], [1, 1, 1], [22, 22, 22]),
        ],
        "noset": [
            document([6, 6, 6], [14, 16, 18], [1, 1, 1]),
            document([8, 8, 8], [16, 18, 20], [2, 2, 2]),
        ],
    }


def block(found: dict, line: str, which: str = "A") -> dict:
    return found[UNIT]["tiers"][which][line]


PLAN, CHECK = pairs.PLAN_RULE, pairs.CHECK


# --- 1. the figures by hand ---------------------------------------------------------------------


def test_the_figures_of_the_arms_on_the_plan_rule_line_by_hand():
    one = block(pairs.compare(documents()), PLAN)
    assert one["seed_figures"] == {
        "tail": [10.0],
        "tuned": [12.0, 14.0],
        "inputs": [20.0, 22.0],
        "noset": [16.0, 18.0],
    }
    assert one["arms"]["tail"]["mean_offer"] == 10.0 and one["arms"]["tail"]["seeds"] == 1
    assert one["arms"]["tuned"]["mean_offer"] == pytest.approx(13.0)
    assert one["arms"]["inputs"]["mean_offer"] == pytest.approx(21.0)
    assert one["arms"]["noset"]["mean_offer"] == pytest.approx(17.0)
    assert one["arms"]["noset"]["seeds"] == 2
    assert (one["arms"]["noset"]["seed_min"], one["arms"]["noset"]["seed_max"]) == (16.0, 18.0)
    assert (one["arms"]["tail"]["seed_min"], one["arms"]["tail"]["seed_max"]) == (10.0, 10.0)


def test_the_figures_of_the_arms_on_the_check_line_by_hand():
    one = block(pairs.compare(documents()), CHECK)
    assert one["seed_figures"] == {
        "tail": [10.0],
        "tuned": [4.0, 6.0],
        "inputs": [9.0, 11.0],
        "noset": [6.0, 8.0],
    }
    assert one["arms"]["tuned"]["mean_offer"] == pytest.approx(5.0)
    assert one["arms"]["inputs"]["mean_offer"] == pytest.approx(10.0)
    assert one["arms"]["noset"]["mean_offer"] == pytest.approx(7.0)
    assert "dials_picked" not in one["arms"]["noset"]  # the check line has no dial rule


def test_the_paired_cells_on_the_plan_rule_line_by_hand():
    one = block(pairs.compare(documents()), PLAN)
    cell = one["against_tuned"]  # per split 15, 17, 19 against 11, 13, 15
    assert cell["splits"] == 3
    assert cell["mean_main_offer_kwh_per_test_day"] == pytest.approx(13.0)
    assert cell["mean_trial_offer_kwh_per_test_day"] == pytest.approx(17.0)
    assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(4.0)
    assert cell["splits_trial_offers_more"] == 3 and cell["splits_trial_offers_less"] == 0
    cell = one["against_inputs"]  # 15, 17, 19 against 21
    assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(-4.0)
    assert cell["splits_trial_offers_less"] == 3
    cell = one["against_tail"]  # 15, 17, 19 against 10, 10, 10
    assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(7.0)
    assert cell["difference_p50"] == pytest.approx(7.0)
    assert cell["ratio_of_means_trial_to_main"] == pytest.approx(1.7)
    cell = one["inputs_against_tail"]  # 21 against 10
    assert cell["mean_difference_kwh_per_test_day"] == pytest.approx(11.0)


def test_the_paired_cells_on_the_check_line_by_hand():
    one = block(pairs.compare(documents()), CHECK)
    assert one["against_tuned"]["mean_difference_kwh_per_test_day"] == pytest.approx(2.0)
    assert one["against_inputs"]["mean_difference_kwh_per_test_day"] == pytest.approx(-3.0)
    assert one["against_tail"]["mean_difference_kwh_per_test_day"] == pytest.approx(-3.0)
    assert one["inputs_against_tail"]["mean_difference_kwh_per_test_day"] == pytest.approx(0.0)
    assert one["inputs_against_tail"]["splits_equal"] == 3


def test_a_neural_arm_enters_with_the_mean_of_its_seeds_split_by_split():
    # seed a better on split 1, seed b better on split 3: the per-split mean differs from either
    found = {
        "tail": [document([10, 10, 10], [10, 10, 10], [0, 0, 0])],
        "tuned": [
            document([20, 10, 10], [20, 10, 10], [0, 0, 0]),
            document([10, 10, 20], [10, 10, 20], [0, 0, 0]),
        ],
        "inputs": [document([10, 10, 10], [10, 10, 10], [0, 0, 0])],
        "noset": [document([10, 10, 10], [10, 10, 10], [0, 0, 0])],
    }
    for line in (PLAN, CHECK):
        cell = block(pairs.compare(found), line)["against_tuned"]
        assert cell["splits"] == 3
        assert cell["mean_main_offer_kwh_per_test_day"] == pytest.approx(40.0 / 3)  # 15, 10, 15
        assert cell["splits_trial_offers_less"] == 2 and cell["splits_equal"] == 1


# --- 2. the line read ---------------------------------------------------------------------------


def test_each_line_reads_its_own_figure_and_never_the_other_arms_of_the_split_record():
    found = pairs.compare(documents())
    for which in ("A", "B"):
        for line in (PLAN, CHECK):
            for name in ("tail", "tuned", "inputs", "noset"):
                assert block(found, line, which)["arms"][name]["mean_offer"] < 500  # no 999
    # the lines differ: the plan rule's noset 17 against the check line's 7 (tier B: plus 100)
    assert block(found, PLAN)["arms"]["noset"]["mean_offer"] == pytest.approx(17.0)
    assert block(found, CHECK)["arms"]["noset"]["mean_offer"] == pytest.approx(7.0)
    assert block(found, PLAN, "B")["arms"]["noset"]["mean_offer"] == pytest.approx(117.0)


def test_a_line_that_is_not_one_of_the_two_is_refused():
    with pytest.raises(ValueError, match="a line is one of"):
        pairs.split_offers(documents()["tail"][0], UNIT, "A", "S2")


# --- 3. the rule applied to every run alike -----------------------------------------------------


def test_each_seed_has_its_own_dial_under_the_plan_rule():
    one = block(pairs.compare(documents()), PLAN)
    assert one["arms"]["inputs"]["dials_picked"] == [0.03, 0.1]
    assert one["arms"]["tuned"]["dials_picked"] == [0.1, 0.1]
    assert one["arms"]["noset"]["dials_picked"] == [0.03, 0.03]
    assert one["arms"]["tail"]["dials_picked"] == [0.03]


def test_the_dial_is_picked_from_the_offers_of_its_own_tier():
    # tier A: the dial 0.03 has the larger mean; tier B: the dial 0.1
    doc = document([1, 1, 1], [10, 10, 10], [1, 1, 1], b=([1, 1, 1], [1, 1, 1], [10, 10, 10]))
    found = pairs.compare({"tail": [doc], "tuned": [doc], "inputs": [doc], "noset": [doc]})
    assert block(found, PLAN, "A")["arms"]["tail"]["dials_picked"] == [0.03]
    assert block(found, PLAN, "B")["arms"]["tail"]["dials_picked"] == [0.1]
    assert block(found, PLAN, "A")["arms"]["tail"]["mean_offer"] == pytest.approx(10.0)
    assert block(found, PLAN, "B")["arms"]["tail"]["mean_offer"] == pytest.approx(10.0)


def test_a_tie_between_two_dials_goes_to_the_smaller_dial():
    doc = document([1, 1, 1], [5, 5, 5], [5, 5, 5])
    found = pairs.compare({"tail": [doc], "noset": [doc]})
    assert block(found, PLAN)["arms"]["tail"]["dials_picked"] == [0.03]


# --- 4. a missing arm ---------------------------------------------------------------------------


def test_a_missing_arm_gives_no_cell_on_both_lines_and_the_others_stay():
    for line in (PLAN, CHECK):
        found = documents()
        del found["tuned"]
        one = block(pairs.compare(found), line)
        assert one["against_tuned"] is None
        assert one["arms"]["tuned"] is None and one["seed_figures"]["tuned"] is None
        assert one["against_tail"] is not None and one["inputs_against_tail"] is not None
        found["tuned"] = []  # an empty list is the same as a missing arm
        assert block(pairs.compare(found), line)["against_tuned"] is None


def test_a_document_without_the_units_splits_gives_no_cell():
    found = documents()
    failed = document([10, 10, 10], [10, 10, 10], [0, 0, 0])
    del failed["units"][UNIT]["splits"]
    failed["units"][UNIT]["failed"] = "ValueError: made up"
    found["noset"][1] = failed
    for line in (PLAN, CHECK):
        one = block(pairs.compare(found), line)
        assert one["against_tuned"] is None and one["against_inputs"] is None
        assert one["against_tail"] is None and one["arms"]["noset"] is None
        assert one["inputs_against_tail"] is not None


def test_no_run_at_all_is_refused():
    with pytest.raises(ValueError, match="no run"):
        pairs.compare({})
    with pytest.raises(ValueError, match="no run"):
        pairs.compare({"tail": [], "inputs": []})


# --- 5. a unit that cannot be split -------------------------------------------------------------


def test_a_unit_that_cannot_be_split_has_its_status_only():
    found = pairs.compare(documents())
    assert found[OTHER] == {"status": "not calibratable: 5 days", "counting_days": 5}


# --- 6. documents of different runs -------------------------------------------------------------


def test_documents_of_different_runs_are_refused():
    found = documents()
    found["noset"][0]["run"]["kappa"] = 0.4
    with pytest.raises(ValueError, match="run.kappa"):
        pairs.compare(found)


def test_splits_of_different_seeds_are_refused():
    found = documents()
    found["noset"][0] = document([6, 6, 6], [14, 16, 18], [1, 1, 1], seeds=(1, 2, 4))
    with pytest.raises(ValueError, match="same seeds"):
        pairs.compare(found)


def test_the_units_of_the_runs_must_be_the_same():
    found = documents()
    del found["noset"][0]["units"][OTHER]
    with pytest.raises(ValueError, match="different units"):
        pairs.compare(found)

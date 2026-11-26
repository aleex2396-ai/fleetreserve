"""`forecast_mix_pairs`: the paired figures of one cell and the comparison of two runs, on made-up
results. Written, not run. Each test says which edit turns it red.

1. `paired_cell` on known numbers: the difference is trial minus main, split by split; the means,
   the standard error, the points, the counts and the ratio of means are as worked out by hand
   (main 1, 2, 3, 4; trial 2, 2, 1, 6). Red if the sign is turned (`b - a` to `a - b`), if the
   counts of more and less are swapped, or if the ratio is main over trial.
2. Splits of other seeds, or in another order, are refused. Red if the seed check is removed.
3. A main mean of 0 gives no ratio; no splits give no figures. Red if the `mean_main` test is
   removed.
4. `compare`: `dials` holds every configuration, both models at the same configuration;
   `same_dial` is both at the main model's dial; `own_dial` is the main model at its dial against
   the trial at its own dial. The three differ on the made-up numbers, so a cell taken from the
   wrong configuration is seen. Red if `trial_name` is replaced by `main_name` in the own-dial
   cell (it would equal the same-dial cell), or if the trial's dial is read from the main
   document.
5. A configuration that failed in either run is left out of every cell and named under `failed`;
   a model with no dial that has a mean offer gives no own-dial cell and no same-dial cell when
   it is the main model. Red if the `failed` filter is removed (a KeyError or a cell of a failed
   run).
6. Runs of another level, other dials, other seeds, or other units are refused. Red if a key of
   `check_same_run` is removed.
7. A unit that cannot be split has its status only; the tiers are separate blocks.
"""

import copy

import pytest

from fleetreserve.forecast_mix_pairs import CELL_FIELDS, check_same_run, compare, paired_cell


def entries(offers) -> list[dict]:
    return [
        {"seed": seed, "offer_kwh": float(offer), "scale": 0.5}
        for seed, offer in enumerate(offers, start=1)
    ]


def test_paired_cell_on_known_numbers():
    cell = paired_cell(entries([1, 2, 3, 4]), entries([2, 2, 1, 6]))
    assert set(cell) == set(CELL_FIELDS)
    assert cell["splits"] == 4
    assert cell["mean_main_offer_kwh_per_test_day"] == 2.5
    assert cell["mean_trial_offer_kwh_per_test_day"] == 2.75
    assert cell["mean_difference_kwh_per_test_day"] == 0.25
    assert cell["standard_error_of_the_difference"] == pytest.approx(0.8539, abs=1e-4)
    assert cell["difference_p50"] == 0.5
    assert cell["difference_p10"] == pytest.approx(-1.4)
    assert cell["difference_p90"] == pytest.approx(1.7)
    assert (
        cell["splits_trial_offers_more"],
        cell["splits_equal"],
        cell["splits_trial_offers_less"],
    ) == (2, 1, 1)
    assert cell["ratio_of_means_trial_to_main"] == pytest.approx(1.1)


def test_splits_of_other_seeds_or_order_are_refused():
    main = entries([1, 2, 3])
    other = entries([1, 2, 3])
    other[2]["seed"] = 9
    with pytest.raises(ValueError, match="same seeds"):
        paired_cell(main, other)
    with pytest.raises(ValueError, match="same seeds"):
        paired_cell(main, list(reversed(entries([1, 2, 3]))))
    with pytest.raises(ValueError, match="same seeds"):
        paired_cell(main, entries([1, 2]))


def test_a_zero_main_mean_gives_no_ratio_and_no_splits_give_no_figures():
    cell = paired_cell(entries([0, 0]), entries([0, 3]))
    assert cell["ratio_of_means_trial_to_main"] is None
    assert cell["mean_difference_kwh_per_test_day"] == 1.5
    empty = paired_cell([], [])
    assert empty["splits"] == 0 and empty["mean_difference_kwh_per_test_day"] is None
    assert empty["difference_p50"] is None and empty["standard_error_of_the_difference"] is None


# --- compare ------------------------------------------------------------------------------------

DIALS = {"plain": None, "d1": 0.01, "d2": 0.02}
MAIN_OFFERS = {
    "plain": ([5, 5, 5, 5], [4, 4, 4, 4]),
    "d1": ([10, 10, 10, 10], [8, 8, 8, 8]),
    "d2": ([9, 9, 9, 9], [7, 7, 7, 7]),
}
TRIAL_OFFERS = {
    "plain": ([6, 6, 6, 6], [4, 4, 4, 5]),
    "d1": ([11, 12, 11, 12], [9, 9, 9, 9]),
    "d2": ([14, 14, 14, 14], [8, 8, 8, 8]),
}


def run_record() -> dict:
    return {
        "kappa": 0.5,
        "dials": [0.01, 0.02],
        "seeds": [1, 4],
        "splits": 4,
        "scales": [0.1, 1.0],
        "n_calibration": 50,
        "tiers": ["A", "B"],
        "rung": "one-way",
    }


def one_run(offers: dict, picks: dict, failed=(), names=None, kappa=0.5):
    """(results document, per-split offers) of a made-up run with one unit that can be split and
    one that cannot; `picks`: the dial picked per tier (None: no pick); `names`: the name each
    configuration has in this run (a level is part of a name in the real runs)."""
    configurations = {}
    for key, dial in DIALS.items():
        name = key if names is None else names[key]
        if key in failed:
            configurations[name] = {"driver_risk": dial, "failed": "ValueError: made up"}
        else:
            a, b = offers[key]
            configurations[name] = {
                "driver_risk": dial,
                "tiers": {"A": entries(a), "B": entries(b)},
            }
    document = {
        "run": {**run_record(), "kappa": kappa},
        "units": {
            "big": {"status": "calibratable", "counting_days": 60},
            "small": {"status": "too few days", "counting_days": 5},
        },
        "dial_choice": {"big": {tier: {"dial": picks[tier]} for tier in ("A", "B")}},
    }
    return document, {"big": configurations}


def made_up_runs(trial_failed=(), main_picks=None, trial_picks=None):
    main = one_run(MAIN_OFFERS, main_picks or {"A": 0.01, "B": 0.02})
    trial = one_run(TRIAL_OFFERS, trial_picks or {"A": 0.02, "B": 0.01}, trial_failed)
    return compare(main[0], main[1], trial[0], trial[1])


def test_dials_same_dial_and_own_dial_each_take_the_configuration_they_name():
    found = made_up_runs()
    block = found["big"]["tiers"]["A"]
    assert list(block["dials"]) == ["plain", "d1", "d2"]
    assert block["dials"]["d1"]["dial"] == 0.01 and block["dials"]["plain"]["dial"] is None
    assert block["dials"]["d2"]["mean_difference_kwh_per_test_day"] == 5.0
    assert block["picked_dial"] == {"main": 0.01, "trial": 0.02}
    same = block["same_dial"]  # both at the main model's dial, 0.01: 10 against 11.5
    assert same["dial"] == 0.01
    assert same["mean_main_offer_kwh_per_test_day"] == 10.0
    assert same["mean_trial_offer_kwh_per_test_day"] == 11.5
    own = block["own_dial"]  # the main model at 0.01 against the trial at 0.02: 10 against 14
    assert own["mean_main_offer_kwh_per_test_day"] == 10.0
    assert own["mean_trial_offer_kwh_per_test_day"] == 14.0
    assert own["mean_difference_kwh_per_test_day"] == 4.0
    assert own != same


def test_the_tiers_are_separate_blocks_with_their_own_picks():
    found = made_up_runs()
    b = found["big"]["tiers"]["B"]
    assert b["picked_dial"] == {"main": 0.02, "trial": 0.01}
    assert b["same_dial"]["mean_main_offer_kwh_per_test_day"] == 7.0  # d2, tier B
    assert b["own_dial"]["mean_main_offer_kwh_per_test_day"] == 7.0
    assert b["own_dial"]["mean_trial_offer_kwh_per_test_day"] == 9.0  # the trial's d1, tier B
    assert found["big"]["tiers"]["A"] != b


def test_a_unit_that_cannot_be_split_has_its_status_only():
    found = made_up_runs()
    assert found["small"] == {"status": "too few days", "counting_days": 5}


def test_a_failed_configuration_is_left_out_and_named():
    found = made_up_runs(trial_failed=("d2",))
    assert found["big"]["failed"] == ["d2"]
    block = found["big"]["tiers"]["A"]
    assert list(block["dials"]) == ["plain", "d1"]
    assert block["same_dial"] is not None  # the main model's dial 0.01 is d1, which ran
    assert block["own_dial"] is None  # the trial's own dial, d2, did not run


def test_a_main_model_with_no_dial_gives_no_same_dial_and_no_own_dial_cell():
    found = made_up_runs(main_picks={"A": None, "B": 0.02})
    block = found["big"]["tiers"]["A"]
    assert block["same_dial"] is None and block["own_dial"] is None
    assert list(block["dials"]) == ["plain", "d1", "d2"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("kappa", 0.4),
        ("dials", [0.01]),
        ("seeds", [1, 5]),
        ("splits", 5),
        ("scales", [1.0]),
        ("n_calibration", 100),
        ("tiers", ["A"]),
        ("rung", "fixed"),
    ],
)
def test_runs_that_differ_in_level_dials_seeds_or_scales_are_refused(key, value):
    main, _ = one_run(MAIN_OFFERS, {"A": 0.01, "B": 0.01})
    trial = copy.deepcopy(main)
    trial["run"][key] = value
    with pytest.raises(ValueError, match=key):
        check_same_run(main, trial)


def test_runs_of_other_units_are_refused():
    main, _ = one_run(MAIN_OFFERS, {"A": 0.01, "B": 0.01})
    trial = copy.deepcopy(main)
    del trial["units"]["small"]
    with pytest.raises(ValueError, match="units"):
        check_same_run(main, trial)
    check_same_run(main, copy.deepcopy(main))


def test_two_runs_at_different_levels_are_paired_by_dial_not_by_name():
    """The extra line: the trial at another level, whose configuration names hold that level."""
    names = {"plain": "k4-plain", "d1": "k4-d1", "d2": "k4-d2"}
    main = one_run(MAIN_OFFERS, {"A": 0.01, "B": 0.02})
    trial = one_run(TRIAL_OFFERS, {"A": 0.02, "B": 0.01}, names=names, kappa=0.4)
    with pytest.raises(ValueError, match="kappa"):
        compare(main[0], main[1], trial[0], trial[1])
    found = compare(main[0], main[1], trial[0], trial[1], ignore=("kappa",))
    same_name = made_up_runs()
    assert found == same_name  # the figures are those of the same runs under the same names
    assert list(found["big"]["tiers"]["A"]["dials"]) == ["plain", "d1", "d2"]
    own = found["big"]["tiers"]["A"]["own_dial"]
    assert own["mean_main_offer_kwh_per_test_day"] == 10.0
    assert own["mean_trial_offer_kwh_per_test_day"] == 14.0


def test_a_dial_missing_from_the_other_run_is_left_out_and_named():
    main = one_run(MAIN_OFFERS, {"A": 0.01, "B": 0.02})
    names = {"plain": "p", "d1": "a", "d2": "b"}
    trial = one_run(TRIAL_OFFERS, {"A": 0.02, "B": 0.01}, names=names)
    del trial[1]["big"]["b"]
    found = compare(main[0], main[1], trial[0], trial[1])
    assert found["big"]["failed"] == ["d2"]
    assert list(found["big"]["tiers"]["A"]["dials"]) == ["plain", "d1"]

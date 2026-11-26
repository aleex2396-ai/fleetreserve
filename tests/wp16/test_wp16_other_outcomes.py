"""O2, O3, O5, O6, O7, O8, O9 and the comparison table from made-up summaries in the writers' forms.

Red when a figure is read from the wrong key, a missing figure becomes a number, a share of days is
made a mean of shares, or a step that did not pass is read.
"""

import pytest
from wp16_made_up import (
    BEST,
    HINDSIGHT,
    NAME_MAIN,
    UNITS,
    MadeUp,
    learn_document,
    matrix_same,
    sha,
)

from fleetreserve import (
    wp16_compare,
    wp16_o2,
    wp16_o3,
    wp16_o5,
    wp16_o6_o8,
    wp16_o7,
    wp16_o9,
    wp16_run,
)
from fleetreserve.wp16_cells import Missing
from fleetreserve.wp16_drive import Source, find_drive
from fleetreserve.wp16_learn import Runs

SAME = {u: NAME_MAIN for u in UNITS}


def source_of(tmp_path, steps, with_learn=False):
    made = MadeUp(tmp_path)
    if with_learn:
        data = made.commit_file("gate/sealed-matrix/lgbm_tail.json", matrix_same())
        made.commands["learn-lgbm_tail"] = [
            "made-up",
            "--matrix",
            "gate/sealed-matrix/lgbm_tail.json",
        ]
        learn = learn_document(sha(data), {u: (1.0, 2.0) for u in UNITS}, SAME)
        steps = [("learn-lgbm_tail", {"learn.json": learn}), *steps]
    made.drive(steps)
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def test_o2_reads_the_ratios_in_per_cent_and_names_a_unit_with_no_day_to_check(tmp_path):
    unit = {
        "days_in_m": 10,
        "ratio_caus_over_hind": {"mean": 0.99, "median": 1.0},
        "in_window_departure": {"days": 3, "ratio_caus_over_hind": {"mean": 0.95}},
        "aff_equals_caus": {"days": 9, "of_days": 10},
        "o2_pass": "passed",
        "days_in_m_not_built": 0,
        "faults": {"order_breaks": 0, "days_not_built": 0, "block_order_breaks_on_m_days": 0},
    }
    gone = {"o2_pass": "no day to check", "reason": "no window", "days_in_m_not_built": 0}
    report = {
        "every_day": {
            "units": {"u1": unit, "u2": gone},
            "pass_over_units": "passed",
            "no_day_to_check": ["u2"],
        }
    }
    source, _ = source_of(tmp_path, [("o2-rho0", {"ceiling-report.json": report})])
    (table,) = wp16_o2.build(source, UNITS)
    rows = {(r[0], r[1]): r for r in table.rows}
    row = rows[("main setting", "u1")]
    assert row[2:5] == (10, pytest.approx(99.0), 100.0)
    assert row[5] == pytest.approx(95.0)
    assert row[7:10] == (9, 10, "passed")
    assert rows[("main setting", "u2")][9] == "no day to check"
    assert "no window" in rows[("main setting", "u2")][10]
    assert isinstance(rows[("main setting", "u3")][2], Missing)  # not in the report
    assert "no entry writes it" in rows[("held setting", "all units")][2].text  # step not in drive


def day_entry(counting, in_m, meeting, checked, missing, violations):
    """A unit's (or the pooled) entry of `every_day` as `ceiling_summary.o2_summary` writes the
    day counts: `t4_check` holds the days meeting T4's hypotheses, those checked, those not
    checkable and one entry per violation."""
    return {
        "counting_days": counting,
        "days_in_m": in_m,
        "t4_check": {
            "days_meeting_hypotheses": meeting,
            "days_checked": checked,
            "days_not_checked_missing_value": missing,
            "violations": [
                {"day": f"day {k}", "block_hind": 1.0, "block_caus": 0.5} for k in range(violations)
            ],
            "held": violations == 0,
        },
    }


def day_rows(source) -> dict:
    (table,) = wp16_o2.build_day_counts(source, UNITS)
    assert table.name == "o2-day-counts"
    return {(row[0], row[1]): row for row in table.rows}


def test_o2_day_counts_are_the_reports_own_t4_check_counts_per_unit_and_setting(tmp_path):
    main = {
        "every_day": {
            "units": {
                "u1": day_entry(12, 10, 7, 6, 1, 2),
                "u2": {"o2_pass": "no day to check", "reason": "no window"},
            },
            "all_units": day_entry(33, 31, 9, 8, 1, 2),
        }
    }
    held = {"every_day": {"units": {"u1": day_entry(5, 5, 1, 1, 0, 0)}}}
    steps = [
        ("o2-rho0", {"ceiling-report.json": main}),
        ("o2-held-rho0", {"ceiling-report.json": held}),
    ]
    source, _ = source_of(tmp_path, steps)
    rows = day_rows(source)
    assert rows[("main setting", "u1")][2:] == (12, 10, 7, 6, 1, 2)
    assert rows[("held setting", "u1")][2:] == (5, 5, 1, 1, 0, 0)  # 0 is a count, not a gap
    pooled = rows[("main setting", wp16_o2.POOLED)]
    assert pooled[2:] == (33, 31, 9, 8, 1, 2)  # the file's own pooled entry, not a sum made here
    gone = rows[("main setting", "u2")]
    assert all(isinstance(cell, Missing) and "no window" in cell.text for cell in gone[2:])
    absent = rows[("main setting", "u3")]
    assert all(isinstance(c, Missing) and "not in the report" in c.text for c in absent[2:])
    # a held setting with no pooled entry: nothing is made up
    assert all(isinstance(cell, Missing) for cell in rows[("held setting", wp16_o2.POOLED)][2:])
    # red: `days_checked` read for the days meeting the hypotheses (6 for 7), `len(violations)`
    # replaced by `held`, the held setting read from `o2-rho0` (12 for 5) or a total summed over
    # the units in place of the report's pooled entry (7, not 9) turns a cell red.


def test_a_day_count_the_report_does_not_hold_is_no_figure_not_zero(tmp_path):
    broken = day_entry(12, 10, 7, 6, 1, 2)
    del broken["t4_check"]["days_checked"]
    broken["t4_check"]["violations"] = None
    nothing = {"counting_days": 4, "days_in_m": 4}  # no t4_check at all
    report = {"every_day": {"units": {"u1": broken, "u2": nothing}}}
    source, _ = source_of(tmp_path, [("o2-rho0", {"ceiling-report.json": report})])
    rows = day_rows(source)
    first = rows[("main setting", "u1")]
    assert first[2:6] == (12, 10, 7, Missing("no figure: the report holds no days_checked"))
    assert first[6] == 1
    assert first[7] == Missing("no figure: the report holds no violations list")
    second = rows[("main setting", "u2")]
    assert second[2:4] == (4, 4)
    assert all(isinstance(cell, Missing) for cell in second[4:])
    assert not wp16_run.any_unreadable(wp16_o2.build_day_counts(source, UNITS))
    # red: `.get(key, 0)` for a count, or `len(violations or [])`, shows 0 where the file has
    # nothing; a Missing that starts with "failed:", "not run:" or "unreadable:" turns the last
    # line red (it would move the exit status).


def test_the_o2_day_counts_of_a_step_that_did_not_pass_say_so_in_every_cell(tmp_path):
    source, _ = source_of(tmp_path, [("o2-rho0", "fail")])
    rows = day_rows(source)
    main = rows[("main setting", "all units")]
    assert all(isinstance(c, Missing) and c.text.startswith("failed: o2-rho0") for c in main[2:])
    held = rows[("held setting", "all units")]
    assert all(isinstance(c, Missing) and "no entry writes it" in c.text for c in held[2:])
    # red: a figure from the other setting, or 0, in a step's place (plan section 13, WP14).


def test_o3_gives_the_ratio_of_means_and_a_missing_bound_as_missing(tmp_path):
    def value(mean, ratio, missing=None):
        return {
            "days": 10,
            "days_with_value": 0 if missing else 10,
            "missing": missing or {},
            "kwh": {"mean": mean},
            "share": {"mean": ratio, "median": ratio},
            "ratio_of_means": ratio,
        }

    hourly = {
        "ceiling": {"days": 10, "days_with_value": 10, "missing": {}, "kwh": {"mean": 5.0}},
        "lp24": value(2.0, 0.4),
        "general_split": value(2.5, 0.5),
        "lp27_largest_found": value(2.2, 0.44),
        "lp27_largest_bound": value(None, None, {"timeout": 5}),
    }
    document = {
        "units": {"u1": {"hourly": hourly, "block": {}}},
        "lp27_cap": {"u1": {"days": 5, "reached": 3, "not_reached": 2}},
        "all_present": {
            "u1": {
                "hourly": {
                    "days": 4,
                    "lp24": {"share": {"mean": 0.3, "median": 0.31}, "ratio_of_means": 0.3},
                    "general_split": {"share": {"mean": 0.5, "median": 0.5}, "ratio_of_means": 0.5},
                    "lp27_largest_found": {
                        "share": {"mean": 0.35, "median": 0.34},
                        "ratio_of_means": 0.35,
                    },
                    "lp27_largest_bound": {"share": {"mean": None}, "ratio_of_means": None},
                },
                "block": {"days": 0},
            }
        },
    }
    source, _ = source_of(tmp_path, [("o3", {"o3.json": document})])
    (table, reach, same), figures = wp16_o3.build(source, UNITS)
    rows = {(r[0], r[1], r[2]): r for r in table.rows}
    assert {r[0] for r in table.rows} <= set(UNITS)  # per unit only: no row over the units
    assert rows[("u1", "hourly", "lp24")][7] == pytest.approx(40.0)
    bound = rows[("u1", "hourly", "lp27_largest_bound")]
    assert bound[5] == "timeout: 5"
    assert bound[7] is None  # printed as missing, never as 0
    assert reach.rows[0] == ("u1", 10, 5, 3, 2)
    # the figure and the same-days table read all_present (the same 4 days), not the rows above
    assert figures[0]["points"][0][1:] == (
        pytest.approx(30.0),
        pytest.approx(50.0),
        pytest.approx(35.0),
        None,
    )
    both = {(r[0], r[1], r[3]): r for r in same.rows if r[3] != ""}
    assert both[("u1", "hourly", "lp24")][2] == 4
    assert both[("u1", "hourly", "lp24")][4] == pytest.approx(30.0)
    assert both[("u1", "hourly", "lp27_largest_bound")][4] is None
    assert "different sets of days" in table.notes[0]


def test_o5_names_a_broken_property_and_the_days_that_break_the_offer_order(tmp_path):
    ratio = {
        "test_days": 5,
        "fixed_one_way_share": 1.0,
        "one_way_two_way_share": 0.8,
        "fixed_one_way_broken_days": [],
        "one_way_two_way_broken_days": ["u1 2026-01-02"],
    }
    ledger = {
        "checks": {
            "failed": True,
            "identity": {
                "days_checked": 10,
                "means_checked": 2,
                "largest_residual_kwh": 1e-12,
                "breaks": [],
            },
            "offer_order": {"breaks": ["u1: one-way to two-way, 1 day"]},
        },
        "units": {"u1": {"offer_ratio": {NAME_MAIN: {"A": ratio}}}},
    }
    eta = {
        "A": {"workplaces": 0.5, "Norway and Korea": 1.0},
        "B": {"workplaces": 2.0, "Norway and Korea": "offers nothing"},
    }
    drift = {
        "sealed_route": {
            "settings": {"file": "f.json", "git_object": "g", "sha256": "s", "eta": eta}
        }
    }
    steps = [
        ("ledger-lgbm_tail", {"ledger.json": ledger}),
        ("drift-lgbm_tail", {"drift-run.json": drift}),
    ]
    source, runs = source_of(tmp_path, steps, with_learn=True)
    properties, offers, sizes = wp16_o5.build(source, runs, UNITS)
    rows = {r[0][:12]: r for r in properties.rows}
    assert rows["T9 identity:"][2] == 0 and rows["T9 identity:"][4] is True
    order = next(r for r in properties.rows if r[0].startswith("offer order"))
    assert order[2] == 1 and order[4] is False
    verdict = next(r for r in properties.rows if r[0] == "the ledger's own verdict")
    assert verdict[4] is False
    t5 = next(r for r in properties.rows if r[0].startswith("T5"))
    assert t5[1:3] == (100, 0) and t5[4] is True
    first = offers.rows[0]
    assert first[0] == "u1" and first[5:] == (0, 1, False)
    assert sizes.rows == [  # all four step sizes, two tiers for each of the two site types
        ("A", "Norway and Korea", 1.0),
        ("A", "workplaces", 0.5),
        ("B", "Norway and Korea", "offers nothing"),
        ("B", "workplaces", 2.0),
    ]
    assert "no entry writes" not in " ".join(sizes.notes)
    assert "f.json" in sizes.notes[0]


def test_o6_and_o8_read_the_terms_where_the_ledger_writes_them(tmp_path):
    def share(**terms):
        return {"days": 4, "share_of_ceiling": terms}

    config = {
        "tiers": {
            "A": {
                "one-way": {
                    "tree_sample": share(offer=0.4, r=0.1, a=0.0, u=0.2, k=0.3),
                    "test_days": share(r_plus_a=0.12),
                }
            }
        }
    }
    entry = {
        "hourly_true_sets": {
            "format_loss": share(format_loss=0.05),
            "affine_minus_diagonal": share(affine_minus_diagonal=0.01),
            "tree_sample": {"a_not_zero_days": 0},
        },
        "block_product": {
            "drawn": 4,
            "tree_sample": share(r=0.02, a=0.0),
            "affine_minus_diagonal": share(affine_minus_diagonal=0.0),
            "price_of_the_block": share(price_of_the_block=0.03),
            "u": "not computed: no block offer in the gate",
            "k": "not computed: no block offer in the gate",
        },
        "configurations": {NAME_MAIN: config},
        "ceiling_ratio": {
            "v": 1.2,
            "one_way_zero_days": [],
            "two_way_offer_share_of_two_way_ceiling": {NAME_MAIN: {"A": 0.5}},
        },
        "offer_ratio": {NAME_MAIN: {"A": {"test_days": 5, "x": 1.1}}},
    }
    source, runs = source_of(
        tmp_path, [("ledger-lgbm_tail", {"ledger.json": {"units": {"u1": entry}}})], True
    )
    o6, o8 = wp16_o6_o8.build(source, runs, UNITS)
    hourly, block = o6.rows[0], o6.rows[1]
    assert hourly[:3] == ("u1", "hourly", 4)
    assert hourly[3:8] == (0.4, 0.1, 0.0, 0.2, 0.3)
    assert hourly[8:10] == (0.05, 0.01)
    assert block[4:6] == (0.02, 0.0) and block[10] == 0.03
    assert block[6].startswith("not computed") and isinstance(block[8], Missing)
    assert o8.rows[0] == (
        "u1",
        NAME_MAIN,
        5,
        1.1,
        1.2,
        0.5,
        0,
        "not run (the owner's decision of 16 November 2026)",
    )
    assert isinstance(o8.rows[1][3], Missing)  # u2 is not in the ledger


def test_o7_is_per_unit_and_no_share_is_pooled_over_units(tmp_path):
    def method(days, breaking, scale=0.5):
        if scale == 0:
            return {"scale": 0.0, "status": "no offer", "days_searched": days}
        return {
            "scale": scale,
            "status": "searched",
            "days_searched": days,
            "days_with_result": days,
            "days_breaking": breaking,
            "share_of_days_breaking": breaking / days,
        }

    def unit(asa, plan):
        both = {"methods": {"asa": asa, "fixed-plan": plan}}
        return {"variants": {"forecast": both, "oracle-departure": both}}

    document = {
        "methods": ["asa", "fixed-plan"],
        "units": {
            "u1": unit(method(10, 4), method(10, 0)),
            "u2": unit(method(40, 4), method(40, 0, 0)),  # no offer by the plan
        },
    }
    source, _ = source_of(tmp_path, [("corner", {"corner.json": document})])
    tables = wp16_o7.build(source, UNITS)
    assert [t.name for t in tables] == ["o7-corner-units"]  # no table over the units
    (units_table,) = tables
    asa = {
        r[2]: r for r in units_table.rows if r[0] == "without oracle departures" and r[1] == "asa"
    }
    assert asa["u1"][7] == pytest.approx(40.0) and asa["u2"][7] == pytest.approx(10.0)
    assert {r[2] for r in units_table.rows} <= set(UNITS)  # no pooled row
    plan = {
        r[2]: r
        for r in units_table.rows
        if r[0] == "with oracle departures" and r[1] == "fixed-plan"
    }
    assert plan["u2"][8] == "no offer" and plan["u2"][7] is None
    assert isinstance(asa["u3"][7], Missing)
    assert "no figure is pooled over units" in units_table.notes[0]


def test_o9_has_a_row_for_each_model_unit_and_k_and_says_when_one_was_not_scored(tmp_path):
    def one(availability, share, best):
        return {
            "offering_local_days": 3,
            "scale": 0.5,
            "offer_kwh_per_counting_scored_day": 1.0,
            "availability": availability,
            "offer_share": {
                HINDSIGHT: {"share_of_ceiling": share},
                BEST: {"share_of_ceiling": best},
            },
            "offering_scored_days": 4,
            "failing_scored_days": 0,
            "scale_zero_reason": None,
        }

    document = {
        "run": {"local_days": [0, 7], "ceilings": [HINDSIGHT, BEST]},
        "models": {"cold_started": "lgbm_tail", "reference": "lgbm"},
        "units": {"u1": {"status": "calibratable"}},
        "figures": {
            "lgbm_tail": {"u1": {"0": {"A": one(0.9, 0.2, 0.3)}, "7": {"A": one(0.92, 0.3, 0.4)}}},
            "lgbm": {"u1": {"0": {"A": one(0.91, 0.25, 0.35)}}},
        },
        "failed": ["lgbm | k=7: boom"],
    }
    steps = [("cold-start-workplaces", {"cold-start.json": document})]
    source, _ = source_of(tmp_path, steps)
    (table,), figures = wp16_o9.build(source, UNITS, {"u3": "no window"})
    rows = {(r[0], r[1], r[2]): r for r in table.rows if r[2] != ""}
    assert rows[("lgbm_tail", "u1", 7)][6:9] == (0.92, 0.3, 0.4)
    assert isinstance(rows[("lgbm", "u1", 7)][3], Missing)
    assert "boom" in rows[("lgbm", "u1", 7)][3].text
    assert any(r[1] == "u3" and "no window" in r[3].text for r in table.rows)
    assert any(r[1] == "u2" for r in table.rows)
    assert figures[0]["curves"][0][2][0] == (0, 0.9, 0.2)


def test_the_comparison_table_sums_each_method_and_names_the_units_it_left_out(tmp_path):
    def reading(offer):
        return {"mean_offer_kwh_per_test_day": offer}

    document = {
        "methods": ["re-solve", "asa", "llf"],
        "dial": 0.02,
        "configurations": {"forecast": NAME_MAIN},
        "failed": [],
        "units": {
            "u1": {
                "status": "calibratable",
                "gate_reading": {
                    "re-solve": reading(2.0),
                    "asa": reading(1.5),
                    "llf": reading(1.0),
                },
            },
            "u2": {
                "status": "calibratable",
                "gate_reading": {
                    "re-solve": reading(3.0),
                    "asa": reading(0.5),
                    "llf": reading(None),
                },
            },
        },
    }
    source, _ = source_of(tmp_path, [("compare", {"compare.json": document})])
    (table,) = wp16_compare.build(source, UNITS)
    rows = {r[0]: r for r in table.rows}
    assert rows["ASA (A)"][1].point == 2.0 and rows["ASA (A)"][2] == "2 of 3"
    assert rows["LLF (B)"][1].point == 1.0
    assert "u2: no figure" in rows["LLF (B)"][3] and "u3: no figure" in rows["LLF (B)"][3]
    assert rows["re-solve (the comparison run's own)"][1].point == 5.0

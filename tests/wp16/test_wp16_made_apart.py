"""The mixed-dial headline interval of O1 (`wp16_mixed`) with the reviewed correction of
`day_intervals` (round 3), and its mark "made apart".

Made-up per-day files of the real writers' form (`interval_fake`, `test_day_pooled` and
`test_day_intervals_negative_ceiling` of tests/wp12), read by the real `day_intervals` and
`day_bootstrap` functions; only the draws of the resamples are the injected hand lists of those
tests, and the run list names their seed (7) and 5 resamples.

Three units, headline configurations that differ by site type (so O1's interval is the mixed one):
`neg-unit` under NAME_MAIN, whose first test day's hindsight ceiling is -1e-14 and whose run share
of that ceiling is null (the rounding of a true 0; the case of the trace note), `first-unit` and
`second-unit` under NAME_OTHER (the hand units of `test_day_pooled`).

1. A null share from a ceiling of -1e-14: the block is kept (X has its interval ends, Z is a
   figure), the unit is pooled, and the block, its row (first cell and note) and the manifest's
   list carry the sentence "made apart at a later commit, after the run's output was seen; fills no
   slot of plan section 11".
   Red at f03149b: `day_intervals.check_points` there recomputed the share from the sums, got 0.0
   against the run's null and so returned a problem; `job_points` then made `Missed`, and the row
   had no interval ends: the assertion `row[4].low is not None` fails first (and so do the three
   about the sentence, which that commit does not write at all).
2. A test-day ceiling of -0.5 (not a rounding) still gives no interval, with the reason; such a row
   is not marked (it holds no figure made apart).
3. The manifest lists the marked rows and holds an empty list when there are none.
"""

import sys
from pathlib import Path

import pytest
from wp16_made_up import (
    NAME_MAIN,
    NAME_OTHER,
    MadeUp,
    learn_document,
    matrix_document,
    run_list_document,
    sha,
)

_WP12 = Path(__file__).resolve().parents[1] / "wp12"  # the made-up per-day files of the intervals
if str(_WP12) not in sys.path:
    sys.path.insert(0, str(_WP12))
from interval_fake import RUNG  # noqa: E402
from test_day_intervals_negative_ceiling import NEG, TINY, neg_unit  # noqa: E402
from test_day_pooled import (  # noqa: E402
    CALIBRATION,
    FIRST,
    SECOND,
    SEED,
    TEST_DRAWS,
    hand_unit,
    inject,
)

from fleetreserve import day_intervals, wp16_o1, wp16_run  # noqa: E402
from fleetreserve.wp16_cells import Interval, Missing, Table  # noqa: E402
from fleetreserve.wp16_drive import Source, find_drive  # noqa: E402
from fleetreserve.wp16_learn import Runs  # noqa: E402
from fleetreserve.wp16_mixed import pooled_block  # noqa: E402
from fleetreserve.wp16_o1 import SPECS, summary_row  # noqa: E402

SENTENCE = (
    "made apart at a later commit, after the run's output was seen; "
    "fills no slot of plan section 11"
)
HAZARD = "learn-hazard_tuned_inputs"
NINE = [NEG, FIRST, SECOND]
HEADLINE = {NEG: NAME_MAIN, FIRST: NAME_OTHER, SECOND: NAME_OTHER}


def made_up_hazard(tmp_path, monkeypatch, first=TINY):
    """A drive with the neural model's learn step: the per-day jobs of the three units (the first
    test day's hindsight ceiling of `neg-unit` is `first`), the run list of seed 7 and 5
    resamples; returns (source, runs)."""
    inject(
        monkeypatch,
        {**CALIBRATION, NEG: CALIBRATION[FIRST]},
        {**TEST_DRAWS, NEG: TEST_DRAWS[FIRST]},
    )
    pairs = [neg_unit(first=first), hand_unit(FIRST), hand_unit(SECOND)]
    for (record, _), unit in zip(pairs, NINE, strict=True):
        record["configurations"] = [HEADLINE[unit]]  # the job names the unit's own headline
    gate: dict = {"configurations": {}}
    for (_record, entry), unit in zip(pairs, NINE, strict=True):
        found = gate["configurations"].setdefault(HEADLINE[unit], {"units": {}})
        found["units"][unit] = entry
    matrix = matrix_document()
    matrix["sealed_choice"]["units"] = {"workplaces": [NEG], "others": [FIRST, SECOND]}
    listed = run_list_document(NINE)
    listed["fixed"] = {"seeds": {"intervals": SEED}, "counts": {"resamples": 5}}
    made = MadeUp(tmp_path)
    data = made.commit_file("gate/sealed-matrix/hazard_tuned_inputs.json", matrix)
    made.commit_file("run-lists/run-list.json", listed)
    made.commands[HAZARD] = ["made-up", "--matrix", "gate/sealed-matrix/hazard_tuned_inputs.json"]
    offers = {NEG: (1.0, 2.0), FIRST: (1.0, 3.0), SECOND: (1.0, 4.0)}
    files = {
        "learn.json": learn_document(sha(data), offers, HEADLINE),
        "learn-gate.json": gate,
        "learn-day-losses.json": {"rungs": [RUNG], "jobs": [record for record, _ in pairs]},
    }
    made.drive([(HAZARD, files)])
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def hazard_row(source, runs):
    spec = next(s for s in SPECS if s.learn == HAZARD)
    return summary_row(source, runs, spec, NINE, {}), spec


def test_a_null_share_from_a_ceiling_of_minus_1e_14_keeps_the_block_and_marks_it(
    tmp_path, monkeypatch
):
    source, runs = made_up_hazard(tmp_path, monkeypatch)
    (row, note), spec = hazard_row(source, runs)
    assert row[3] == "3 of 3"
    # the block is kept: X has its interval ends, Z is a figure (red at f03149b: no ends there)
    assert isinstance(row[4], Interval) and row[4].low is not None and row[4].high is not None
    assert row[4].why == ""
    assert isinstance(row[7], Interval) and row[7].point is not None
    assert "its own headline configuration" in note and "units pooled: 3" in note
    # the mark: first cell, note (so every file of the table), the block itself
    assert row[0] == f"{spec.title} [{SENTENCE}]"
    assert SENTENCE in note
    block, used = pooled_block(source, HAZARD, HEADLINE, NINE, {}, "one-way")
    assert used == NINE and block["made_apart"] == SENTENCE
    assert "sum_of_unit_offers" in block  # the block's own figures are still there
    assert day_intervals.APART_SENTENCE == SENTENCE  # the words are the one `--apart-from` stamps


def test_a_ceiling_of_minus_0_5_still_gives_no_interval_with_its_reason_and_no_mark(
    tmp_path, monkeypatch
):
    source, runs = made_up_hazard(tmp_path, monkeypatch, first=-0.5)
    (row, note), _ = hazard_row(source, runs)
    assert isinstance(row[4], Interval) and row[4].low is None  # the sum of the points only
    assert "below the rounding allowance" in row[4].why
    assert "below the rounding allowance" in note
    assert SENTENCE not in row[0] and SENTENCE not in note  # no figure was made apart


def test_o1_tables_mark_only_the_mixed_row_and_the_manifest_lists_it(tmp_path, monkeypatch):
    source, runs = made_up_hazard(tmp_path, monkeypatch)
    summary, per_unit, rho = wp16_o1.build(source, runs, NINE, {})
    marked = [r[0] for r in summary.rows if SENTENCE in str(r[0])]
    assert marked == [f"hazard_tuned_inputs, one-way re-solve [{SENTENCE}]"]
    assert not any(SENTENCE in str(r[0]) for r in rho.rows)
    assert not any(isinstance(c, Missing) and SENTENCE in c.text for r in rho.rows for c in r)
    listed = wp16_run.made_apart([summary, per_unit, rho])
    assert listed == {
        "sentence": SENTENCE,
        "rows": [{"table": "o1-summary", "row": marked[0]}],
    }


def test_the_manifest_holds_an_empty_list_when_no_row_is_made_apart():
    table = Table("t", "t", ("run",))
    table.add("lgbm_tail (main model), one-way re-solve")
    table.add(Missing("failed: learn-x (exit status 1)"))
    assert wp16_run.made_apart([table]) == {"sentence": SENTENCE, "rows": []}


@pytest.mark.parametrize("cell", [3.5, None])
def test_a_row_that_is_not_text_is_never_marked(cell):
    table = Table("t", "t", ("run",))
    table.add(cell)
    assert wp16_run.made_apart([table])["rows"] == []

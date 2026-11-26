"""The table of the days in the event E' (`wp16_eprime`) from made-up `learn.json` files in the
writers' form (`learn_run.e_prime_figures`; made-up numbers, nothing is run on real data).

Red when: a figure is read from the wrong key, configuration or unit; the two shares of days swap
places; a figure the run did not produce becomes 0 or a blank; counts that contradict one another
or a share that is not its count over its denominator are written as figures; a failed step is
filled from elsewhere; a WP15 run is left unmarked after a WP14 failure; or the table is left out
of the files of `wp16_run` or of its manifest. A module that read nothing fails every test that
names a number.
"""

import json

import pytest
from wp16_made_up import (
    COMMIT,
    NAME_BESIDE,
    NAME_MAIN,
    NAME_OTHER,
    UNITS,
    MadeUp,
    e_prime_entry,
    learn_document,
    matrix_document,
    matrix_same,
    not_calibratable,
    run_list_document,
    sha,
    with_e_prime,
)

from fleetreserve import wp16_eprime, wp16_figures, wp16_o1, wp16_run
from fleetreserve.wp16_cells import Missing
from fleetreserve.wp16_drive import Source, find_drive
from fleetreserve.wp16_failed import SUFFIX
from fleetreserve.wp16_learn import Runs, unit_test

MATRIX = "gate/sealed-matrix/lgbm_tail.json"
RUN = "lgbm_tail"
BESIDE = "lgbm_tail (departures seen at the next hour, beside reading)"
RHO_ONE = "lgbm_tail-rho0_1 (WP15 sensitivity)"
RHO_TWO = "lgbm_tail-rho0_2 (WP15 sensitivity)"
OFFERS = {u: (1.0, 2.0) for u in UNITS}


def figures_of(unit_one, unit_two, unit_three=None) -> dict:
    """unit -> configuration -> entry, for the matrix's headlines (u1: NAME_MAIN, u2 and u3:
    NAME_OTHER), with other configurations holding other numbers, so that a wrong pick shows."""
    found = {
        "u1": {NAME_MAIN: unit_one, NAME_OTHER: e_prime_entry(9, 9, 9, 9, 9)},
        "u2": {NAME_OTHER: unit_two, NAME_MAIN: e_prime_entry(9, 9, 9, 9, 9)},
    }
    if unit_three is not None:
        found["u3"] = {NAME_OTHER: unit_three}
    return found


def source_of(tmp_path, figures, *, extra=(), learn_steps=(), matrix=None, offers=OFFERS):
    """A made-up drive with `learn-lgbm_tail` (E' figures `figures`), `learn_steps` (more learn
    steps as (label, figures)) and `extra` steps; returns (source, runs)."""
    made = MadeUp(tmp_path)
    data = made.commit_file(MATRIX, matrix or matrix_document())
    made.commit_file("run-lists/run-list.json", run_list_document())
    steps = []
    for label, found in (("learn-lgbm_tail", figures), *learn_steps):
        made.commands[label] = ["made-up", "--matrix", MATRIX]
        learn = with_e_prime(learn_document(sha(data), offers), found)
        steps.append((label, {"learn.json": learn}))
    made.drive(steps + list(extra))
    source = Source(tmp_path, find_drive(tmp_path))
    return source, Runs(source)


def rows_of(source, runs, absent=None) -> dict:
    (table,) = wp16_eprime.build(source, runs, UNITS, absent or {})
    assert table.name == "eprime-days"
    assert all(len(row) == len(wp16_eprime.COLUMNS) for row in table.rows)
    return {(row[0], row[1]): row for row in table.rows}


def test_the_figures_land_in_the_cells_of_the_unit_and_its_headline_configuration(tmp_path):
    figures = figures_of(
        e_prime_entry(5, 4, 2, 12, 8),
        e_prime_entry(10, 6, 3, 30, 15),
        e_prime_entry(7, 7, 7, 21, 21),
    )
    rows = rows_of(*source_of(tmp_path, figures))
    # run, unit, configuration, counting days, certified days, days in E', share of certified
    # days, share of counting days, steered cars, steered cars in E'_i, share of steered cars
    assert rows[(RUN, "u1")] == (RUN, "u1", NAME_MAIN, 5, 4, 2, 0.5, 0.4, 12, 8, 8 / 12)
    assert rows[(RUN, "u2")] == (RUN, "u2", NAME_OTHER, 10, 6, 3, 0.5, 0.3, 30, 15, 0.5)
    assert rows[(RUN, "u3")] == (RUN, "u3", NAME_OTHER, 7, 7, 7, 1.0, 1.0, 21, 21, 1.0)
    # red: a module that reads nothing, the wrong configuration (the others hold 9s), the share
    # of counting days placed before the share of certified days (u1: 0.4 and 0.5).


def test_a_share_is_the_runs_own_and_is_not_computed_here(tmp_path):
    entry = e_prime_entry(5, 4, 2, 12, 8)
    entry["share_of_certified_test_days_in_e_prime"] = 0.5 + 1e-12  # inside the tolerance
    rows = rows_of(*source_of(tmp_path, figures_of(entry, e_prime_entry())))
    assert rows[(RUN, "u1")][6] == 0.5 + 1e-12  # the file's figure, not 2 / 4


def test_a_unit_whose_run_wrote_no_e_prime_has_no_figure_and_not_a_zero(tmp_path):
    figures = figures_of(e_prime_entry(5, 4, 2, 12, 8), e_prime_entry())
    source, runs = source_of(tmp_path, figures)
    rows = rows_of(source, runs)
    cells = rows[(RUN, "u3")][3:]  # u3 is in the run (a learn.json entry) with no `e_prime`
    assert len(cells) == 8
    assert all(isinstance(c, Missing) and c.text.startswith("no figure: ") for c in cells)
    assert all("e_prime" in c.text for c in cells)
    # red: a missing key read as 0 or as a blank.


def test_a_unit_the_run_does_not_hold_says_why_not(tmp_path):
    offers = {"u1": (1.0, 2.0), "u2": (1.0, 2.0)}
    figures = {"u1": {NAME_MAIN: e_prime_entry()}, "u2": {NAME_OTHER: e_prime_entry()}}
    rows = rows_of(*source_of(tmp_path, figures, offers=offers), absent={"u3": "no window"})
    assert rows[(RUN, "u3")][3].text == "no figure: no window (no window)"
    source, runs = source_of(tmp_path / "other", figures, offers=offers)
    assert rows_of(source, runs)[(RUN, "u3")][3].text == "no figure: the unit is not in the run"


def test_a_scored_unit_with_no_certified_day_has_a_real_zero_and_no_share(tmp_path):
    """A unit that WAS scored (it was split: 4 counting test days) on which no day was certified:
    its counts are real zeros and the shares over the certified days and the steered cars have a
    denominator of 0, so they are null in the file and "no figure" here; the share of counting
    days is a real 0 (0 of 4). A unit that was never scored is another case (next test)."""
    none_certified = e_prime_entry(4, 0, 0, 0, 0)
    assert none_certified["share_of_certified_test_days_in_e_prime"] is None
    rows = rows_of(*source_of(tmp_path, figures_of(none_certified, e_prime_entry())))
    row = rows[(RUN, "u1")]
    assert row[3:6] == (4, 0, 0) and row[8:10] == (0, 0)
    assert isinstance(row[6], Missing) and "certified_test_days is 0" in row[6].text
    assert row[7] == 0.0 and isinstance(row[10], Missing)


def test_a_unit_that_was_never_scored_is_no_figure_in_o1s_words_and_never_zero(tmp_path):
    figures = figures_of(e_prime_entry(5, 4, 2, 12, 8), e_prime_entry(), e_prime_entry())
    made = MadeUp(tmp_path)
    data = made.commit_file(MATRIX, matrix_document())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", MATRIX]
    learn = with_e_prime(learn_document(sha(data), OFFERS), figures)
    not_calibratable(learn, "u3", days=12)
    made.drive([("learn-lgbm_tail", {"learn.json": learn})])
    source = Source(tmp_path, find_drive(tmp_path))
    runs = Runs(source)
    # the writer's shape for the unit: counts of 0 and null shares, and no reason
    assert learn["units"]["u3"]["e_prime"][NAME_OTHER]["test_days"] == 0
    assert "reason" not in learn["units"]["u3"]["e_prime"][NAME_OTHER]
    cells = rows_of(source, runs)[(RUN, "u3")][3:]
    assert len(cells) == 8 and all(isinstance(c, Missing) for c in cells)
    (words,) = {c.text for c in cells}
    assert words == "no figure: not calibratable: 12 days; counts as not passing"
    # the same words as O1 gives for the same unit and run (its function, not a copy of the text)
    run = runs.get("learn-lgbm_tail", "intervals-lgbm_tail")
    assert words == unit_test(run, "u3", NAME_OTHER, "one-way", "A", {})[1]
    o1_row = [r for r in wp16_o1.unit_rows(runs, wp16_o1.SPECS[0], UNITS, {}) if r[1] == "u3"][0]
    assert o1_row[3].text == words
    # the exit status follows O1: "no figure" is not a failure
    assert not words.startswith(wp16_run.BAD_START)
    assert rows_of(source, runs)[(RUN, "u1")][3] == 5  # a scored unit of the same run is read
    # red: a reader that does not read the unit's status shows 0 and 0 for u3.


def test_a_unit_with_no_status_is_no_figure_and_not_a_zero(tmp_path):
    source, runs = source_of(tmp_path, figures_of(e_prime_entry(), e_prime_entry()))
    run = runs.get("learn-lgbm_tail", "intervals-lgbm_tail")
    del run.learn["units"]["u1"]["status"]
    cell = rows_of(source, runs)[(RUN, "u1")][3]
    assert cell.text == "no figure: the run holds no status for the unit"


def test_a_configuration_that_failed_keeps_its_own_reason(tmp_path):
    failed = {"reason": "failed: ValueError: boom"}
    unscored = {"reason": "the configuration failed, no day was scored"}
    rows = rows_of(*source_of(tmp_path, figures_of(failed, unscored)))
    assert all(c.text == "failed: ValueError: boom" for c in rows[(RUN, "u1")][3:])
    assert all(
        c.text == "no figure: the configuration failed, no day was scored"
        for c in rows[(RUN, "u2")][3:]
    )


def test_a_configuration_the_run_does_not_hold_is_named(tmp_path):
    figures = figures_of(e_prime_entry(), e_prime_entry())
    del figures["u1"][NAME_MAIN]
    rows = rows_of(*source_of(tmp_path, figures))
    assert f"no E' figures for configuration {NAME_MAIN}" in rows[(RUN, "u1")][3].text
    assert rows[(RUN, "u2")][3] == 5  # the other unit is read as it is


@pytest.mark.parametrize(
    "entry",
    [
        e_prime_entry(5, 4, 5, 12, 8, share_of_certified_test_days_in_e_prime=1.25),  # E' > days
        e_prime_entry(3, 4, 2, 12, 8),  # certified > counting
        e_prime_entry(5, 4, 2, 12, 13),  # cars in E'_i > steered cars
    ],
)
def test_counts_that_contradict_one_another_are_refused(tmp_path, entry):
    rows = rows_of(*source_of(tmp_path, figures_of(entry, e_prime_entry())))
    cells = rows[(RUN, "u1")][3:]
    assert all(isinstance(c, Missing) and c.text.startswith("unreadable: ") for c in cells)
    assert rows[(RUN, "u2")][3] == 5  # the other unit is read as it is
    # red: counts written as figures without the order of the counts being checked.


def test_a_share_that_is_not_its_count_over_its_denominator_is_refused_alone(tmp_path):
    wrong = e_prime_entry(5, 4, 2, 12, 8, share_of_test_days_in_e_prime=0.5)  # 2 of 5 is 0.4
    given = e_prime_entry(0, 0, 0, 0, 0, share_of_test_days_in_e_prime=0.0)  # null expected
    rows = rows_of(*source_of(tmp_path, figures_of(wrong, given)))
    one, two = rows[(RUN, "u1")], rows[(RUN, "u2")]
    assert one[7].text.startswith("unreadable: share_of_test_days_in_e_prime 0.5 is not")
    assert one[3:7] == (5, 4, 2, 0.5) and one[8:] == (12, 8, 8 / 12)  # the rest is read
    assert two[7].text.startswith("unreadable: share_of_test_days_in_e_prime 0.0 where test_days")


@pytest.mark.parametrize("bad", [True, -1, 2.5, "4", None])
def test_a_count_that_is_not_a_whole_number_is_no_figure(tmp_path, bad):
    entry = e_prime_entry(5, 4, 2, 12, 8)
    entry["certified_test_days"] = bad
    rows = rows_of(*source_of(tmp_path, figures_of(entry, e_prime_entry())))
    cells = rows[(RUN, "u1")][3:]
    assert all(isinstance(c, Missing) and "certified_test_days" in c.text for c in cells)


def test_the_beside_reading_takes_the_beside_configuration_or_says_there_is_none(tmp_path):
    figures = figures_of(e_prime_entry(5, 4, 2, 12, 8), e_prime_entry())
    figures["u1"][NAME_BESIDE] = e_prime_entry(6, 5, 1, 20, 4)
    rows = rows_of(*source_of(tmp_path, figures))
    assert rows[(BESIDE, "u1")][2:] == (NAME_BESIDE, 6, 5, 1, 0.2, 1 / 6, 20, 4, 0.2)
    assert "the matrix names no such configuration" in rows[(BESIDE, "u2")][3].text  # dial 0.03


def test_a_step_that_failed_is_not_filled_from_elsewhere(tmp_path):
    made = MadeUp(tmp_path)
    made.commit_file(MATRIX, matrix_document())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.drive([("learn-lgbm_tail", "fail")])
    source = Source(tmp_path, find_drive(tmp_path))
    rows = rows_of(source, Runs(source))
    for unit in UNITS:
        cells = rows[(RUN, unit)][3:]
        assert all(c.text == "failed: learn-lgbm_tail (exit status 1)" for c in cells)
    # a run the drive has no step for: "no entry writes it", for every unit
    assert rows[("hazard_tuned_inputs", "u1")][3].text.startswith("no entry writes it")


def test_the_runs_are_the_distinct_learn_steps_and_readings_of_o1():
    pairs = [(spec.learn, spec.reading) for spec, _ in wp16_eprime.runs_read()]
    assert len(pairs) == len(set(pairs))
    assert set(pairs) == {(s.learn, s.reading) for s in wp16_o1.SPECS}
    labels = [label for _, label in wp16_eprime.runs_read()]
    assert RUN in labels and BESIDE in labels and RHO_ONE in labels


def test_the_wp15_runs_are_marked_information_only_after_a_wp14_failure_and_not_before(tmp_path):
    figures = figures_of(e_prime_entry(), e_prime_entry())
    rho = [("learn-lgbm_tail-rho0_1", figures)]
    ok_source, ok_runs = source_of(tmp_path / "ok", figures, learn_steps=rho)
    bad_source, bad_runs = source_of(
        tmp_path / "bad", figures, learn_steps=rho, extra=[("o3", "fail")]
    )
    assert ok_source.drive.complete is True and bad_source.drive.complete is False
    assert not [k for k in rows_of(ok_source, ok_runs) if k[0].endswith(SUFFIX)]
    marked = {k[0] for k in rows_of(bad_source, bad_runs) if k[0].endswith(SUFFIX)}
    assert marked == {RHO_ONE + SUFFIX, RHO_TWO + SUFFIX}  # both WP15 runs, whether read or not
    assert RUN not in marked and BESIDE not in marked  # the headline run is never marked
    (table,) = wp16_eprime.build(bad_source, bad_runs, UNITS, {})
    assert [n for n in table.notes if n.startswith("Information only")]
    # the figures of the marked rows are the unmarked ones, as in O1
    (before,) = wp16_eprime.build(ok_source, ok_runs, UNITS, {})
    for old, new in zip(before.rows, table.rows, strict=True):
        assert old[1:] == new[1:]


@pytest.fixture
def stand_ins(monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)

    def render(spec, folder):
        name = f"{spec['kind']}.png"
        (folder / name).write_bytes(b"placeholder")
        return name

    monkeypatch.setattr(wp16_figures, "render", render)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)


def run_top(tmp_path, entry):
    """A top folder as `test_wp16_run` makes one, with E' figures in the learn step's file."""
    made = MadeUp(tmp_path)
    data = made.commit_file(MATRIX, matrix_same())
    made.commit_file("run-lists/run-list.json", run_list_document())
    made.commands["learn-lgbm_tail"] = ["made-up", "--matrix", MATRIX]
    truth = {"abstained": {}, "units": {}}
    gate = {
        "configurations": {
            NAME_MAIN: {
                "units": {
                    u: {"fixed_every_request": {"test_mean_offer_kwh_per_day": 0.25}} for u in UNITS
                }
            }
        }
    }
    figures = {u: {NAME_MAIN: entry} for u in UNITS}
    learn = with_e_prime(learn_document(sha(data), OFFERS, {u: NAME_MAIN for u in UNITS}), figures)
    made.drive(
        [
            ("sealed-truth", {"truth-summary.json": truth}),
            ("learn-lgbm_tail", {"learn.json": learn, "learn-gate.json": gate}),
        ]
    )


def test_the_command_writes_the_table_and_names_it_in_the_manifest(tmp_path, stand_ins):
    run_top(tmp_path, e_prime_entry(186, 186, 1, 400, 380))
    out = tmp_path / "out"
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    for name in ("eprime-days.csv", "eprime-days.md"):
        assert manifest["outputs"][name] == sha((out / name).read_bytes())
    lines = (out / "eprime-days.csv").read_text().splitlines()
    assert lines[0].startswith("# Drive ")  # the banner every table opens with
    assert any(line.startswith("lgbm_tail,u1,") and ",186,186,1," in line for line in lines)
    assert not any("total" in n or "pooled" in n for n in manifest["outputs"])  # nothing pooled
    assert "the days in E'" in (out / "outcomes.md").read_text()
    # red: the module not registered in `collect`.


def test_a_contradicting_count_makes_the_exit_status_1_and_a_missing_one_does_not(
    tmp_path, stand_ins
):
    run_top(tmp_path / "bad", e_prime_entry(5, 4, 5, 12, 8))
    assert wp16_run.main(["--root", str(tmp_path / "bad"), "--out", str(tmp_path / "o1")]) == 1
    run_top(tmp_path / "none", e_prime_entry(4, 0, 0, 0, 0))
    assert wp16_run.main(["--root", str(tmp_path / "none"), "--out", str(tmp_path / "o2")]) == 0

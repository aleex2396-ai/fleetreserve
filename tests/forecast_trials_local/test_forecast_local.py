"""The local-refit forecast trial (`forecast_local`) on made-up units and a made-up model. Written,
not run. Each test says which edit turns it red.

1. The refusal. Every name that is not a development unit is refused by `check_units`, in
   `--units` and in the settings file, and `main` ends with exit 2 before any matrix, table or
   truth build is read. Red if `check_units` accepts a name outside `DEVELOPMENT_UNITS`, or if
   `main` reads the matrix or the inputs before `check_units`.
2. The settings. The committed file holds the documented values; a file with another set of keys,
   a k that is not one of the plan's, a list of refit arms other than the two fixed ones (so
   neither can be dropped quietly) or a held-out unit is refused; the
   command has no option that sets one value of it. Red if a value of the committed file changes, if
   `read_settings` stops checking one of these, or if an option such as `--local-days` is added.
3. The fit rows (`local_fit_rows`). Only rows of the first k counting days (and the non-counting
   days among them) whose session had plugged out before the earliest later day's τ. Red if the
   line `plug_out <= cutoff` is removed (a session that outlasts the local days enters), if
   `first_forecast_time` takes the latest τ of the local days in place of the earliest τ of the
   later days (a pooled unit's earlier-τ site-day is then not guarded), or if `local_rows` is
   replaced by the whole table (a later day enters).
4. The days. The splits are those of the days after the first k, shifted by k: no first-k day in
   a calibration or test set, every later day in exactly one, the draw the hash of the date text
   alone (`tail_splits`). Red if the shift `d + k` or `tail_of` is dropped.
5. The scores. The first k days' losses are blanked (`without_first`); every arm is scored on
   the same splits; the paired difference is arm minus main per split. Red if `without_first`
   is removed, if the arms are swapped in `score_arms`, or if the arms get different splits.
5c. The reading (`verdict_word`, `one_line_reading`, `missing_judged_units`, `read_reading`): the
   word of a tier A cell against the unit's own bar, just above, just below, inside and on it, and
   the three outcomes of the one line; the block cannot be absent; the cited entry is found by its
   own words; the borrowed bars are declared as such; a unit with a bar and no judged cell makes
   the line "not read" with the reason shown. Red if one bar is used for both units, if `>` or
   `<` become `>=` or `<=`, if the loss test loses its minus, if `read_settings` stops reading
   the block, or if the `missing` guard goes from `one_line_reading` or `run`.
5b. The weights (`equal_mass_weight`, `refit_frames`): the local rows of the equal-mass arm weigh
   in total what the pool weighs, each pool row 1; the unweighted arm's frames carry no weight and
   hold the same rows. Red if the quotient is inverted or the weight column is put on the
   unweighted arm.
6. Whole runs on a made-up model that records what it was fitted on: the main arm is fitted on
   the training unit alone and each unit's two refit arms on it plus exactly that unit's local
   fit rows (the leak guards hold for each arm); a unit with too few days is left out; every
   output carries the label. Red if `refit_frames` holds a whole table in place of
   `local_fit_rows`, if the main fit gets the local rows, if `finish` is not called in `main`, or
   `write_note` in `write_outputs`.
7. The module never names the held-out loader.
8. The README names the command, the settings file, the output files and the label.
"""

import ast
import json
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build
from truth_tables_fake import date_of, stamp

from fleetreserve import (
    after_sealed_run,
    cold_start_days,
    dial_splits,
    forecast_local,
    gate_config,
    gate_run,
    run_verify,
    sealed_route,
)
from fleetreserve.day_split import split_days
from fleetreserve.forecast_model import KEY_COLUMNS, TABLE_COLUMNS, empty_table
from fleetreserve.truth_verify import CouldNotRun

TOP = Path(forecast_local.__file__).resolve().parents[2]
MODULE = Path(forecast_local.__file__)
README = TOP / "trials-after-sealed-run" / "README.md"
GRID = dial_splits.GRID


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


# --- made-up units without files ---------------------------------------------------------------


@dataclass
class StubUnit:
    """What `cold_start_days` and `forecast_local` read of a `UnitDays`."""

    unit: str
    rho: float
    days: tuple
    cars: pd.DataFrame


def stub_day(date: str, site: str = "A", tau_hour: float = 10.0):
    record = SimpleNamespace(date=date, site=site, tau=stamp(date, tau_hour))
    return SimpleNamespace(record=record)


def row(date: str, site: str, session: str, plug_out_hours) -> dict:
    """A cars-table row; `plug_out_hours` from midnight of `date` (above 24: a later day), or the
    text of an unreadable plug-out."""
    out = plug_out_hours if isinstance(plug_out_hours, str) else stamp(date, plug_out_hours)
    return {"date": date, "site": site, "session": session, "plug_out": out}


def stub(days: list, rows: list, name: str = "u") -> StubUnit:
    return StubUnit(name, 0.0, tuple(days), pd.DataFrame(rows))


# --- 1. the refusal ----------------------------------------------------------------------------

REFUSED = (
    *sealed_route.HELD_OUT_UNITS,
    "OSL_T",
    "TRO_R",
    "caltech",
    "OSL_S",
    "no-such-unit",
)


@pytest.mark.parametrize("name", REFUSED)
def test_a_unit_that_is_not_a_development_unit_is_refused(name):
    with pytest.raises(CouldNotRun, match="not a development unit"):
        forecast_local.check_units([name])
    with pytest.raises(CouldNotRun, match=name):
        forecast_local.check_units(["OSL_1", name])


def test_the_three_gate_units_are_accepted_once_each():
    gate = gate_config.load(TOP / "gate" / "matrix.json").gate_units
    assert forecast_local.check_units(gate) == tuple(gate)
    with pytest.raises(CouldNotRun, match="more than once"):
        forecast_local.check_units(["BER", "BER"])
    with pytest.raises(CouldNotRun, match="no unit"):
        forecast_local.check_units([])


def _forbidden(*args, **kwargs):
    raise AssertionError("a file, a matrix or a truth build was read")


def test_main_refuses_a_held_out_unit_before_any_file_is_read(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(gate_run, "check_inputs", _forbidden)
    monkeypatch.setattr(gate_run, "check_files", _forbidden)
    monkeypatch.setattr(forecast_local, "load", _forbidden)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = forecast_local.main(["--units", "nrel-from-may-2019", "OSL_T"])
    assert code == 2 and capsys.readouterr().out.startswith("COULD NOT RUN")
    assert list(tmp_path.iterdir()) == []


def test_main_refuses_a_held_out_unit_named_in_the_settings_before_any_data_is_read(
    monkeypatch, capsys, tmp_path
):
    document = json.loads(forecast_local.SETTINGS.read_text())
    document["units"] = ["nrel-from-may-2019", "OSL_T"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document))
    monkeypatch.setattr(gate_run, "check_inputs", _forbidden)
    monkeypatch.setattr(gate_run, "check_files", _forbidden)
    monkeypatch.setattr(forecast_local, "load", _forbidden)
    assert forecast_local.main(["--settings", str(path)]) == 2
    assert "OSL_T" in capsys.readouterr().out


# --- 2. the settings ---------------------------------------------------------------------------


def test_the_committed_settings_are_the_documented_ones():
    found = forecast_local.read_settings(forecast_local.SETTINGS)
    gate = gate_config.load(TOP / "gate" / "matrix.json")
    assert found.model == "lgbm_tail"
    assert found.refits == forecast_local.REFITS == ("unweighted-pool", "equal-mass")
    assert (found.kappa, found.dial, found.splits) == (0.5, 0.02, 200)
    assert found.local_days == 56 == cold_start_days.LARGEST  # the largest k of plan section 11
    assert found.units == tuple(gate.gate_units)  # the three development gate units


SOURCE_PHRASE = "The neural model's seed test"  # the words that begin the cited entry


def test_the_committed_reading_fixes_the_two_noise_bars_at_their_source_entry():
    found = forecast_local.read_settings(forecast_local.SETTINGS).reading
    bars = {"nrel-from-may-2019": 0.08, "norway-development-pooled": 0.3}
    assert found["noise_bar_share"] == bars
    lines = (TOP / "docs" / "prior-looks.md").read_text().splitlines()
    entries = [line for line in lines if line.lstrip(" -").startswith(SOURCE_PHRASE)]
    assert len(entries) == 1  # found by its own words, not by its line number
    assert "(8%)" in entries[0] and "(30%)" in entries[0]  # red if the figures are absent
    assert SOURCE_PHRASE in found["noise_bar_source"]
    assert "line 3" not in found["noise_bar_source"]  # no citation by line number


def test_the_reading_says_that_the_bars_are_borrowed_from_the_neural_model():
    source = forecast_local.read_settings(forecast_local.SETTINGS).reading["noise_bar_source"]
    for words in (
        "range over three seeds (with the plan's own runs) at one level, kappa 0.3",
        "one fixed seed",
        "an analogy and not this trial's measured noise",
        "up to 30% therefore reads no measurable difference",
    ):
        assert words in source, words  # red if the borrowed bars are not declared as such


def _settings(tmp_path, **changes) -> Path:
    document = json.loads(forecast_local.SETTINGS.read_text())
    document.update(changes)
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"local_days": 10}, "plan's k"),
        ({"local_days": 0}, "plan's k"),
        ({"refits": ["unweighted-pool"]}, "refits"),  # red if one arm can be dropped
        ({"refits": ["equal-mass"]}, "refits"),
        ({"refits": ["equal-mass", "unweighted-pool"]}, "refits"),
        ({"refits": ["unweighted-pool", "equal-mass", "weighted"]}, "refits"),
        ({"refits": "unweighted-pool"}, "refits"),
        ({"splits": 0}, "splits"),
        ({"units": ["nrel-from-may-2019", "KRO", "OSL_T"]}, "not a development unit"),
        ({"extra": 1}, "exactly the keys"),
        ({"kappa": "high"}, "kappa"),
    ],
)
def test_a_settings_file_that_departs_from_the_form_is_refused(tmp_path, changes, match):
    with pytest.raises(CouldNotRun, match=match):
        forecast_local.read_settings(_settings(tmp_path, **changes))


def test_a_settings_file_without_a_key_is_refused(tmp_path):
    document = json.loads(forecast_local.SETTINGS.read_text())
    del document["dial"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document))
    with pytest.raises(CouldNotRun, match="exactly the keys"):
        forecast_local.read_settings(path)


@pytest.mark.parametrize(
    "option", ["--local-days", "--kappa", "--dial", "--splits", "--model", "--refits"]
)
def test_the_command_has_no_option_that_sets_a_single_setting(option):
    with pytest.raises(SystemExit) as stopped:
        forecast_local.main([option, "1"])
    assert stopped.value.code == 2


# --- 3. the fit rows ---------------------------------------------------------------------------


def _one_site_unit() -> StubUnit:
    """Counting days on the dates 0, 1, 3, 4, ...; the date 2 has rows and does not count."""
    dates = [date_of(i) for i in (0, 1, 3, 4, 5, 6, 7)]
    days = [stub_day(d) for d in dates]
    rows = [
        row(date_of(0), "A", "s0", 15.0),
        row(date_of(1), "A", "s1", 15.0),
        row(date_of(2), "A", "non-counting-day", 15.0),
        row(date_of(3), "A", "ends-before-the-first-later-day", 24 + 5.0),  # next day 05:00
        row(date_of(3), "A", "ends-after-it-has-begun", 24 + 10.5),  # next day 10:30
        row(date_of(3), "A", "ends-days-later", 24 * 5 + 9.0),
        row(date_of(3), "A", "unreadable", ""),
        row(date_of(4), "A", "first-later-day", 15.0),
        row(date_of(7), "A", "last-day", 15.0),
    ]
    return stub(days, rows)


def test_the_fit_rows_are_the_first_days_sessions_that_ended_before_the_first_later_day():
    unit = _one_site_unit()  # k = 3: the dates 0, 1 and 3 are the local days; the date 4 follows
    kept = forecast_local.local_fit_rows(unit, 3)
    assert list(kept["session"]) == [
        "s0",
        "s1",
        "non-counting-day",
        "ends-before-the-first-later-day",
    ]
    # red if the line `plug_out <= cutoff` goes: "ends-after-it-has-begun" and the others enter;
    # red if `local_rows` is not used: "first-later-day" and "last-day" enter.


def test_the_counts_of_the_kept_rows_are_taken_from_the_rows_the_fit_keeps():
    unit = _one_site_unit()  # k = 3: seven local rows, four kept
    kept = forecast_local.local_fit_rows(unit, 3)
    assert forecast_local.local_row_counts(unit, 3, kept) == {
        "local_site_days_asked": 3,
        "local_site_days_with_a_kept_row": 3,  # the non-counting date 2 is not one of the three
        "rows_kept": 4,
        "rows_left_out": 3,  # red if the left-out count is not the local rows minus the kept
    }
    counts = forecast_local.local_row_counts(unit, 3, kept.iloc[0:0])
    assert counts["local_site_days_with_a_kept_row"] == 0 and counts["rows_left_out"] == 7


def test_the_first_forecast_time_is_the_earliest_tau_of_the_later_days_not_the_latest_local_one():
    days = [
        stub_day(date_of(0), "A", 9.0),
        stub_day(date_of(0), "B", 10.0),
        stub_day(date_of(0), "C", 8.0),
        stub_day(date_of(1), "A", 9.0),
    ]
    unit = stub(
        days,
        [
            row(date_of(0), "A", "a-ends-between", 9.5),
            row(date_of(0), "A", "a-ends-first", 7.5),
            row(date_of(0), "B", "b-ends-later", 11.0),
            row(date_of(0), "C", "c", 8.5),
        ],
    )
    assert forecast_local.first_forecast_time(unit, 2) == pd.Timestamp(stamp(date_of(0), 8.0))
    kept = forecast_local.local_fit_rows(unit, 2)
    assert list(kept["session"]) == ["a-ends-first"]
    # the cold start's own rule keeps the row that ends between 8:00 and 10:00 (the local end)
    known = cold_start_days.known_rows(unit, 2)
    assert "a-ends-between" in set(known["session"])


def test_the_fit_rows_need_a_later_day_and_a_k_the_unit_has():
    unit = _one_site_unit()
    with pytest.raises(ValueError):
        forecast_local.first_forecast_time(unit, 8)
    with pytest.raises(ValueError, match="no counting day after"):
        forecast_local.first_forecast_time(unit, len(unit.days))


# --- 4. the days -------------------------------------------------------------------------------


PAIR = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")


def _matrix(n_calibration: int = 20):
    document = matrix_document(PAIR)
    document["n_calibration"] = n_calibration
    return gate_config.parse(document)


def _plain_unit(n_days: int, name: str = "u") -> StubUnit:
    return stub([stub_day(date_of(i)) for i in range(n_days)], [], name)


def test_the_splits_are_those_of_the_later_days_shifted_by_k():
    unit, k, seeds = _plain_unit(50), 10, [1, 2, 3]
    plan, found = forecast_local.tail_splits("u", unit, _matrix(), k, seeds)
    assert plan.status == "calibratable" and (plan.calibration_days, plan.test_days) == (20, 20)
    keys = [f"{d.record.date} {d.record.site}" for d in unit.days]
    for seed, calibration, test in found:
        assert min(calibration + test) >= k  # red if `tail_of` or the shift is dropped
        assert sorted(calibration + test) == list(range(k, 50))
        assert not set(calibration) & set(test)
        expected = split_days("u", seed, keys[k:], n_calibration=20, min_days=21).calibration
        assert [keys[d] for d in calibration] == list(expected)  # the draw depends on dates alone
    assert [s for s, _, _ in found] == seeds
    assert len({tuple(c) for _, c, _ in found}) > 1  # the seeds draw different days


def test_a_unit_with_too_few_days_after_the_first_k_is_not_split():
    plan, found = forecast_local.tail_splits("u", _plain_unit(25), _matrix(), 7, [1])
    assert plan.status != "calibratable" and found == []  # 18 days left, 20 would calibrate


def test_the_half_rule_counts_the_days_after_the_first_k():
    document = matrix_document(PAIR)
    document["n_calibration"], document["calibration_half_below"] = 100, 200
    plan, found = forecast_local.tail_splits(
        "u", _plain_unit(250), gate_config.parse(document), 56, [1]
    )
    assert plan.status == "calibratable" and plan.calibration_days == 97  # 194 days: half
    assert len(found[0][1]) == 97 and len(found[0][2]) == 97


# --- 5. the scores -----------------------------------------------------------------------------


def _losses(days: int = 30, value: float = 1.0) -> dial_splits.UnitLosses:
    return dial_splits.UnitLosses(
        value=np.full(days, value),
        offering=np.ones(days, dtype=bool),
        slot=np.zeros((days, len(GRID))),
        driver=np.zeros((days, len(GRID))),
        ceiling=np.full(days, np.nan),
    )


def _dirty(losses: dial_splits.UnitLosses, k: int) -> dial_splits.UnitLosses:
    """The same losses with the first k days made to lose everything at every scale."""
    value, slot, driver = losses.value.copy(), losses.slot.copy(), losses.driver.copy()
    value[:k], slot[:k], driver[:k] = 100.0, 1.0, 1.0
    return replace(losses, value=value, slot=slot, driver=driver)


def test_the_first_k_days_cannot_reach_a_score_even_through_a_split_that_names_them():
    k = 5
    splits = [(1, list(range(0, 25)), list(range(25, 30)))]  # a split that includes days 0 to 4
    clean = forecast_local.score_arms(_losses(), {"a": _losses(), "b": _losses()}, splits, k)
    dirty = forecast_local.score_arms(
        _dirty(_losses(), k), {"a": _dirty(_losses(), k), "b": _dirty(_losses(), k)}, splits, k
    )
    for tier in ("A", "B"):
        for arm in ("main", "a", "b"):
            assert dirty[tier]["scores"][arm] == clean[tier]["scores"][arm]
            assert clean[tier]["scores"][arm][0].scale == GRID[-1]  # 20 clean offering days
    # red if `without_first` is removed: the dirty days lose everything and the scale is 0


def test_the_paired_difference_is_arm_minus_main_on_the_same_splits():
    splits = [(s, list(range(5, 25)), list(range(25, 30))) for s in (1, 2, 3)]
    refits = {"a": _losses(value=2.0), "b": _losses(value=3.0)}
    found = forecast_local.score_arms(_losses(value=1.0), refits, splits, 5)
    for tier in ("A", "B"):
        assert set(found[tier]["paired"]) == {"a", "b"}
        assert found[tier]["paired"]["b"]["mean_difference_kwh_per_test_day"] == pytest.approx(
            2 * GRID[-1]
        )
        paired = found[tier]["paired"]["a"]
        assert paired["splits_paired"] == 3 and paired["splits_left_out_value_missing"] == 0
        assert paired["mean_difference_kwh_per_test_day"] == pytest.approx(GRID[-1])
        assert paired["share_of_paired_splits_above_0"] == 1.0
        assert paired["method_mean_offer_kwh_per_test_day"] == pytest.approx(2 * GRID[-1])
        assert paired["reference_mean_offer_kwh_per_test_day"] == pytest.approx(GRID[-1])
        assert [s.seed for s in found[tier]["scores"]["main"]] == [1, 2, 3]
        assert [s.seed for s in found[tier]["scores"]["a"]] == [1, 2, 3]


def test_equal_arms_give_a_difference_of_zero_on_every_split():
    splits = [(s, list(range(5, 25)), list(range(25, 30))) for s in (1, 2)]
    found = forecast_local.score_arms(_losses(), {"a": _losses()}, splits, 5)
    paired = found["A"]["paired"]["a"]
    assert paired["mean_difference_kwh_per_test_day"] == 0.0
    assert paired["share_of_paired_splits_equal_0"] == 1.0


# --- 5c. the reading ---------------------------------------------------------------------------

BARS = {"nrel-from-may-2019": 0.08, "norway-development-pooled": 0.3}


@pytest.mark.parametrize("unit", list(BARS))
def test_the_word_of_a_tier_a_cell_follows_the_bar_of_its_unit(unit):
    bar = BARS[unit]
    word = forecast_local.verdict_word
    assert word(bar + 0.001, bar) == forecast_local.GAIN  # just above
    assert word(bar - 0.001, bar) == forecast_local.NOTHING  # just below, inside
    assert word(bar, bar) == forecast_local.NOTHING  # on the bar is inside it
    assert word(0.0, bar) == forecast_local.NOTHING
    assert word(-bar + 0.001, bar) == forecast_local.NOTHING
    assert word(-bar, bar) == forecast_local.NOTHING
    assert word(-bar - 0.001, bar) == forecast_local.LOSS  # red if the minus is lost
    assert word(None, bar) == forecast_local.UNDEFINED
    assert word(0.5, None) == forecast_local.NO_BAR


def test_one_share_is_a_gain_at_the_workplace_unit_and_no_difference_at_the_pooled_one():
    word = forecast_local.verdict_word
    assert word(0.10, BARS["nrel-from-may-2019"]) == forecast_local.GAIN
    assert word(0.10, BARS["norway-development-pooled"]) == forecast_local.NOTHING
    assert word(-0.10, BARS["nrel-from-may-2019"]) == forecast_local.LOSS
    assert word(-0.10, BARS["norway-development-pooled"]) == forecast_local.NOTHING


GAIN, LOSS, NOTHING = forecast_local.GAIN, forecast_local.LOSS, forecast_local.NOTHING


@pytest.mark.parametrize(
    "words, expected",
    [
        ([GAIN, NOTHING, NOTHING, NOTHING], forecast_local.HELPS),
        ([GAIN, GAIN, NOTHING, GAIN], forecast_local.HELPS),
        ([LOSS, NOTHING, NOTHING, NOTHING], forecast_local.HURTS),
        ([LOSS, LOSS, NOTHING, LOSS], forecast_local.HURTS),
        ([GAIN, LOSS, NOTHING, NOTHING], forecast_local.MIXED),  # a gain and a loss: mixed
        ([NOTHING] * 4, forecast_local.MIXED),
        ([forecast_local.UNDEFINED, GAIN], forecast_local.HELPS),
        ([GAIN, forecast_local.NO_BAR], forecast_local.NOT_READ),
        ([], forecast_local.NOT_READ),
    ],
)
def test_the_one_line_reading_has_its_three_outcomes(words, expected):
    assert forecast_local.one_line_reading(words) == expected
    assert forecast_local.MIXED == ("no measurable difference; a mixed result is reported as mixed")


@pytest.mark.parametrize(
    "words, missing, expected",
    [
        ([GAIN, NOTHING], [], forecast_local.HELPS),
        ([GAIN, NOTHING], ["norway-development-pooled"], forecast_local.NOT_READ),
        ([LOSS, NOTHING], ["nrel-from-may-2019"], forecast_local.NOT_READ),
        ([NOTHING, NOTHING], ["nrel-from-may-2019"], forecast_local.NOT_READ),
        ([GAIN, LOSS], ["nrel-from-may-2019"], forecast_local.NOT_READ),
    ],
)
def test_a_judged_unit_that_drops_out_makes_the_one_line_not_read(words, missing, expected):
    # red if the `missing` guard is removed from `one_line_reading`: the first row's words then
    # read as HELPS from one unit alone, whatever the unit that dropped out would have shown
    assert forecast_local.one_line_reading(words, missing) == expected


def test_only_a_unit_with_a_bar_and_no_judged_cell_is_missing():
    found = forecast_local.missing_judged_units
    done = {"status": "calibratable"}
    left_out = {"status": "left out: too few days after the first 56 days"}
    every = {SCORED[1]: done, SCORED[2]: done}
    assert found(BARS, every) == []
    assert found(BARS, {SCORED[0]: left_out, **every}) == []  # the bar-less unit is not judged
    assert found(BARS, {SCORED[1]: left_out, SCORED[2]: done}) == [SCORED[1]]
    assert found(BARS, {SCORED[1]: done, SCORED[2]: {"status": "failed"}}) == [SCORED[2]]
    assert found(BARS, {SCORED[1]: done}) == [SCORED[2]]  # a unit not run at all
    assert found(BARS, {}) == sorted(BARS)


def test_the_settings_reader_refuses_a_file_without_its_reading(tmp_path):
    document = json.loads(forecast_local.SETTINGS.read_text())
    del document["reading"]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document))
    with pytest.raises(CouldNotRun, match="exactly the keys"):
        forecast_local.read_settings(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop("noise_bar_share"),
        lambda r: r.update(extra="x"),
        lambda r: r.update(noise_bar_source=""),
        lambda r: r.update(noise_bar_share={}),
        lambda r: r.update(noise_bar_share={"nrel-from-may-2019": 0}),
        lambda r: r.update(noise_bar_share={"nrel-from-may-2019": 1.5}),
        lambda r: r.update(noise_bar_share={"OSL_T": 0.1}),
    ],
)
def test_the_settings_reader_refuses_a_broken_reading(tmp_path, change):
    document = json.loads(forecast_local.SETTINGS.read_text())
    change(document["reading"])
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document))
    with pytest.raises(CouldNotRun):
        forecast_local.read_settings(path)


# --- 5b. the weights ---------------------------------------------------------------------------


def _pool_and_local():
    pool = {
        "caltech": pd.DataFrame({"date": ["d"] * 30, "site": ["T"] * 30}),
        "jpl": pd.DataFrame({"date": ["d"] * 12, "site": ["J"] * 12}),
    }
    return pool, pd.DataFrame({"date": ["e"] * 8, "site": ["A"] * 8})


def test_the_equal_mass_weight_is_the_pool_rows_over_the_local_rows():
    assert forecast_local.equal_mass_weight(42, 8) == 5.25
    assert forecast_local.equal_mass_weight(40, 8) == 5.0
    for pool_rows, local_rows in ((0, 5), (5, 0)):
        with pytest.raises(ValueError, match="rows on both sides"):
            forecast_local.equal_mass_weight(pool_rows, local_rows)


def test_the_local_rows_of_the_equal_mass_arm_weigh_what_the_pool_weighs():
    pool, local = _pool_and_local()
    frames = forecast_local.refit_frames("equal-mass", pool, local, "A-unit")
    weight = forecast_local.WEIGHT_COLUMN
    assert list(frames)[:2] == ["caltech", "jpl"] and len(frames) == 3  # the pool comes first
    pool_weight = sum(frames[n][weight].sum() for n in pool)
    own = frames["local rows of A-unit"]
    assert pool_weight == 42.0 and all(frames[n][weight].eq(1.0).all() for n in pool)
    assert own[weight].sum() == pytest.approx(pool_weight)  # red if the quotient is inverted
    assert own[weight].eq(42 / 8).all() and len(own) == 8
    assert weight not in local.columns  # the input tables are not changed


def test_the_unweighted_arm_holds_the_same_rows_with_no_weight():
    pool, local = _pool_and_local()
    plain = forecast_local.refit_frames("unweighted-pool", pool, local, "A-unit")
    weighted = forecast_local.refit_frames("equal-mass", pool, local, "A-unit")
    assert list(plain) == list(weighted)
    for name, frame in plain.items():
        assert forecast_local.WEIGHT_COLUMN not in frame.columns  # red if a weight is put here
        kept = [c for c in weighted[name].columns if c != forecast_local.WEIGHT_COLUMN]
        assert frame.equals(weighted[name][kept])
    with pytest.raises(ValueError, match="no refit arm"):
        forecast_local.refit_frames("weighted", pool, local, "A-unit")


def test_the_arms_name_their_model_modules():
    assert forecast_local.arm_model_name("lgbm_tail", "unweighted-pool") == "lgbm_tail"
    assert forecast_local.arm_model_name("lgbm_tail", "equal-mass") == "lgbm_tail_equal_mass"


# --- 6. whole runs -----------------------------------------------------------------------------

TRAINING = "caltech"
K = 7
SPLITS = 3
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
SCORED = ("nrel-before-may-2019", "nrel-from-may-2019", "norway-development-pooled")
# three cars per site-day: all 7 local site-days at the single-site unit, 6 at the pooled one (the
# 7th site-day's cars are still plugged in at the first forecast day's decision time)
LOCAL_FIT_ROWS = {SCORED[1]: 3 * K, SCORED[2]: 3 * (K - 1)}
# Counts of the kept rows, derived by hand. Single site: 7 site-days (days 0 to 6), every car
# leaves the same day, the first forecast day is day 7: 21 rows kept, 0 left out. Pooled: both
# sites have the dates 0 to 29, so the site-days in order are (0 B), (0 C), (1 B), (1 C), (2 B),
# (2 C), (3 B); the 7th local site-day (3 B) shares its date with the first forecast site-day
# (3 C), whose decision time is before any car of (3 B) leaves: its 3 rows are left out, the other
# six site-days keep their 3 rows each: 18 kept, 3 left out, 6 site-days with a kept row.
REFIT_ARMS = ("unweighted-pool", "equal-mass")
LOCAL_COUNTS = {
    SCORED[1]: {
        "local_site_days_asked": K,
        "local_site_days_with_a_kept_row": 7,
        "rows_kept": 21,
        "rows_left_out": 0,
    },
    SCORED[2]: {
        "local_site_days_asked": K,
        "local_site_days_with_a_kept_row": 6,
        "rows_kept": 18,
        "rows_left_out": 3,
    },
}


class FakeTree:
    """A made-up tree model: it keeps the frame it was fitted on, and forecasts a departure of
    one hour for a car of a unit it was not fitted on and `seen_hat` hours for a car of one it
    was."""

    name = "fake_tree"
    fit_units = (TRAINING,)
    seen_hat = 2.0
    made: list = []

    def __init__(self):
        self.fitted = None
        FakeTree.made.append(self)

    def fit(self, frame):
        self.fitted = frame.copy()

    def predict(self, frame, kappas):
        seen = set(self.fitted["unit"])
        rows = []
        for key, own in zip(
            frame[list(KEY_COLUMNS)].itertuples(index=False),
            frame["unit"].isin(seen),
            strict=True,
        ):
            for kappa in kappas:
                made = dict.fromkeys(TABLE_COLUMNS, float("nan"))
                made.update(zip(KEY_COLUMNS, key, strict=True))
                made.update(kappa=float(kappa), rho=0.0, d_hat=self.seen_hat if own else 1.0)
                rows.append(made)
        return pd.DataFrame(rows, columns=list(TABLE_COLUMNS)) if rows else empty_table()


class FakeEqualMass(FakeTree):
    """The made-up model of the equal-mass arm: it needs the weight column, as the real one does."""

    name = "fake_tree_equal_mass"
    seen_hat = 3.0

    def fit(self, frame):
        if forecast_local.WEIGHT_COLUMN not in frame.columns:
            raise ValueError("no weight column")
        super().fit(frame)


def _register(monkeypatch, name, make):
    module = types.ModuleType(f"fleetreserve.models.{name}")
    module.make_model = make
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture(autouse=True)
def fresh_fakes(monkeypatch):
    FakeTree.made = []
    _register(monkeypatch, "fake_tree", lambda matrix: FakeTree())
    _register(monkeypatch, "fake_tree_equal_mass", lambda matrix: FakeEqualMass())


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(
        folder,
        {
            TRAINING: gate_cars(12, "T"),
            SCORED[0]: gate_cars(5),
            SCORED[1]: gate_cars(60),
            SCORED[2]: gate_cars(30, "B") + gate_cars(30, "C", early_every=9),
        },
    )
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    settings = folder / "settings.json"
    settings.write_text(
        json.dumps(
            {
                **json.loads(forecast_local.SETTINGS.read_text()),
                "model": "fake_tree",
                "kappa": 0.3,
                "local_days": K,
                "splits": SPLITS,
            }
        )
    )
    return run, tables, matrix, settings


def _argv(build, *extra) -> list[str]:
    run, tables, matrix, settings = build
    return [
        "--matrix",
        str(matrix),
        "--settings",
        str(settings),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        *extra,
    ]


def _run(build, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return forecast_local.main(_argv(build, *extra))


def _document(folder: Path) -> dict:
    return json.loads((folder / forecast_local.RESULTS_NAME).read_text())


def test_a_whole_run_measures_every_arm_and_stamps_every_output(
    build, tmp_path, monkeypatch, capsys
):
    code = _run(build, tmp_path / "out", monkeypatch)
    text = capsys.readouterr().out
    assert code == 0, text
    folder = tmp_path / "out"
    assert {p.name for p in folder.iterdir()} == {
        forecast_local.RESULTS_NAME,
        forecast_local.SUMMARY_NAME,
        *forecast_local.TABLE_NAMES.values(),
        forecast_local.TIMING_NAME,
        after_sealed_run.FILE_NAME,
    }
    document = _document(folder)
    assert document[after_sealed_run.KEY] == after_sealed_run.record()
    assert document["label"].startswith(after_sealed_run.WHAT)
    lines = text.splitlines()
    assert lines[:2] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    assert (folder / forecast_local.SUMMARY_NAME).read_text() == text
    assert (folder / after_sealed_run.FILE_NAME).read_text().splitlines() == lines[:2]
    assert document["run"]["refits"] == list(forecast_local.REFITS)
    assert document["run"]["local_days"] == K and document["run"]["splits"] == SPLITS
    assert document["run"]["kappa"] == 0.3 and document["run"]["dial"] == 0.02
    assert document["settings"]["file"] == "settings.json"
    assert text.count("(information only, not judged)") == 4  # tier B, two units, two arms
    assert text.count("against a bar of") == 4  # tier A, two units, two arms
    for word in ("main", "unweighted", "equal-mass", "unweighted-pool minus main"):
        assert word in text  # the three arms are printed side by side with both differences
    assert "equal-mass minus main" in text
    for line in document["definitions"]["reading"]:
        assert line in text  # the reading is printed before the units
    words = document["reading"]["words"]
    assert len(words) == 4 and set(words) <= {GAIN, LOSS, NOTHING, forecast_local.UNDEFINED}
    assert document["reading"]["units_with_a_bar_and_no_judged_cell"] == []
    assert document["reading"]["one_line"] == forecast_local.one_line_reading(words)
    assert f"Reading of the trial: {document['reading']['one_line']}" in text
    assert "Reason:" not in text  # the bar-less unit that is left out does not stop the reading
    left_out = document["units"][SCORED[0]]
    assert left_out["status"].startswith("left out") and "tiers" not in left_out
    for unit in SCORED[1:]:
        entry = document["units"][unit]
        rows = LOCAL_FIT_ROWS[unit]
        assert entry["status"] == "calibratable" and entry["local_fit_rows"] == rows
        assert entry["local_rows_by_arm"] == {arm: LOCAL_COUNTS[unit] for arm in REFIT_ARMS}
        counts = LOCAL_COUNTS[unit]
        assert (
            f"local site-days asked for {K}, with at least one kept row "
            f"{counts['local_site_days_with_a_kept_row']}; rows kept {counts['rows_kept']}, rows "
            f"left out by the cutoff {counts['rows_left_out']}"
        ) in text  # printed in the summary, per arm
        assert entry["equal_mass_weight"] == 36 / rows  # 36 training rows over the local rows
        assert entry["days_after_the_local_days"] == 60 - K
        for tier in ("A", "B"):
            found = entry["tiers"][tier]
            assert set(found) == {
                "main",
                "unweighted-pool",
                "equal-mass",
                "paired_minus_main",
                "main_all_days",
                "reading",
            }  # red if an arm is left out of the results
            for arm in ("main", *forecast_local.REFITS):
                assert found[arm]["splits"] == SPLITS
            assert set(found["paired_minus_main"]) == set(forecast_local.REFITS)
            for _arm, cell in found["reading"].items():
                assert ("word" in cell) == (tier == "A")  # tier B: numbers only
                if tier == "A":
                    assert cell["bar"] == BARS[unit]
                    expected = forecast_local.verdict_word(cell["share_of_main_offer"], cell["bar"])
                    assert cell["word"] == expected
            for pair in found["paired_minus_main"].values():
                assert pair["splits_paired"] == SPLITS
            assert found["main_all_days"]["splits"] == SPLITS


def test_the_main_arm_is_fitted_on_the_training_unit_and_each_refit_adds_the_first_days(
    build, tmp_path, monkeypatch, capsys
):
    assert _run(build, tmp_path / "out", monkeypatch) == 0, capsys.readouterr().out
    main, *refits = FakeTree.made
    assert set(main.fitted["unit"]) == {TRAINING}  # red if local rows reach the main fit
    assert main.fitted.get(forecast_local.WEIGHT_COLUMN) is None
    assert len(refits) == 4  # two arms for each unit that is split; the unit of 5 days has none
    # The fit rows are the first k counting days' rows whose session had plugged out before the
    # earliest later day's decision time (README, "Local fit rows"). At the single-site unit that
    # is every row of the first 7 days. At the pooled unit the 7th site-day (date 3, site B) and
    # the 8th (date 3, site C) share a date, so the 8th's decision time (10:00) falls before the
    # 7th's sessions leave (14:00 and later): those rows are not yet known and are left out.
    first_days = {
        SCORED[1]: {(date_of(i), "A") for i in range(K)},
        SCORED[2]: {(date_of(i // 2), "BC"[i % 2]) for i in range(K - 1)},  # site-days in order
    }
    order = [(unit, arm) for unit in SCORED[1:] for arm in forecast_local.REFITS]
    for model, (unit, arm) in zip(refits, order, strict=True):
        assert isinstance(model, FakeEqualMass) == (arm == "equal-mass")
        assert set(model.fitted["unit"]) == {TRAINING, unit}
        own = model.fitted[model.fitted["unit"] == unit]
        # the leak guards hold for each arm: the first k days' rows and no others
        assert set(zip(own["date"], own["site"], strict=True)) == first_days[unit]
        assert len(own) == LOCAL_FIT_ROWS[unit]  # red if `frames` holds the whole table
        training = model.fitted[model.fitted["unit"] == TRAINING]
        assert len(training) == len(main.fitted)
        if arm == "equal-mass":
            weight = model.fitted[forecast_local.WEIGHT_COLUMN]
            assert weight[own.index].sum() == pytest.approx(weight[training.index].sum())
        else:
            assert forecast_local.WEIGHT_COLUMN not in model.fitted.columns  # unchanged arm


def test_the_arms_forecast_the_same_cars_and_differ_in_the_fit(
    build, tmp_path, monkeypatch, capsys
):
    assert _run(build, tmp_path / "out", monkeypatch) == 0, capsys.readouterr().out
    folder = tmp_path / "out"
    tables = {
        arm: pd.read_csv(folder / name, dtype=str)
        for arm, name in forecast_local.TABLE_NAMES.items()
    }
    keys = ["unit", "site", "date", "session_id", "kappa"]
    for _arm, table in tables.items():
        assert set(table["unit"]) == set(SCORED[1:])
        assert tables["main"][keys].equals(table[keys])
    assert set(tables["main"]["d_hat"]) == {"1.0"}
    assert set(tables["unweighted-pool"]["d_hat"]) == {"2.0"}
    assert set(tables["equal-mass"]["d_hat"]) == {"3.0"}
    document = _document(folder)
    for arm, name in forecast_local.TABLE_NAMES.items():
        assert document["forecast_tables"][arm]["file"] == name


def test_a_whole_run_of_one_unit_leaves_the_others_out(build, tmp_path, monkeypatch, capsys):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", SCORED[1])
    assert code == 0, capsys.readouterr().out
    document = _document(tmp_path / "out")
    assert list(document["units"]) == [SCORED[1]]
    assert len(FakeTree.made) == 3  # the main model and the two refit arms of the one unit


def test_a_whole_run_without_one_judged_unit_does_not_read_the_one_line(
    build, tmp_path, monkeypatch, capsys
):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", SCORED[1])
    text = capsys.readouterr().out
    assert code == 0, text
    document = _document(tmp_path / "out")
    assert len(document["reading"]["words"]) == 2  # one unit is judged: two arms
    assert document["reading"]["units_with_a_bar_and_no_judged_cell"] == [SCORED[2]]
    # red if the guard is removed: the two words alone would read as helps, hurts or mixed
    assert document["reading"]["one_line"] == forecast_local.NOT_READ
    assert f"Reading of the trial: {forecast_local.NOT_READ}" in text
    assert f"Reason: no judged cell for {SCORED[2]}" in text


def test_a_whole_run_with_no_unit_that_can_be_split_could_not_run(
    build, tmp_path, monkeypatch, capsys
):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", SCORED[0])
    assert code == 2 and "COULD NOT RUN" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


def test_a_whole_run_refuses_a_unit_that_is_not_a_development_unit(
    build, tmp_path, monkeypatch, capsys
):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", SCORED[1], "OSL_T")
    assert code == 2 and "OSL_T" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []
    assert FakeTree.made == []  # nothing was built


@pytest.mark.parametrize("arms", [["unweighted-pool"], ["equal-mass"]])
def test_a_whole_run_with_one_arm_removed_from_the_settings_does_not_run(
    build, tmp_path, monkeypatch, capsys, arms
):
    settings = json.loads(build[3].read_text())
    settings["refits"] = arms
    path = tmp_path / "one-arm.json"
    path.write_text(json.dumps(settings))
    code = _run(build, tmp_path / "out", monkeypatch, "--settings", str(path))
    assert code == 2 and "refits" in capsys.readouterr().out  # red if an arm can be dropped
    assert list((tmp_path / "out").iterdir()) == [] and FakeTree.made == []


def test_finish_marks_the_document_always():
    document = forecast_local.finish({"label": "forecast trial local: x"})
    assert document[after_sealed_run.KEY] == after_sealed_run.record()
    assert document["label"].startswith(after_sealed_run.WHAT)


def test_a_model_that_reads_outcomes_is_refused(build, tmp_path, monkeypatch, capsys):
    class Reader(FakeTree):
        reads_other_day_outcomes = True

    _register(monkeypatch, "fake_tree", lambda matrix: Reader())
    code = _run(build, tmp_path / "out", monkeypatch)
    assert code == 1 and "REFUSED" in capsys.readouterr().out


# --- 7. no held-out loader ---------------------------------------------------------------------


def test_the_module_never_names_the_held_out_loader():
    text = MODULE.read_text()
    assert "heldout" not in text.lower().replace("held-out", "")
    modules = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            modules += [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    assert modules and not [m for m in modules if "held" in m.lower() or "sealed_route" in m]


# --- 8. the README -----------------------------------------------------------------------------


def test_the_readme_names_the_command_the_settings_and_the_output_files():
    text = README.read_text()
    assert "24 November 2026" in text and "after held-out results were seen" in text
    section = text[text.index("## Forecast trial local") :]
    assert "fleetreserve.forecast_local" in section and "forecast-local.json" in section
    for name in (
        forecast_local.RESULTS_NAME,
        forecast_local.SUMMARY_NAME,
        *forecast_local.TABLE_NAMES.values(),
        forecast_local.TIMING_NAME,
        after_sealed_run.FILE_NAME,
    ):
        assert name in section, name
    settings = forecast_local.read_settings(forecast_local.SETTINGS)
    assert f"k = {settings.local_days}" in section
    for word in (*forecast_local.REFITS, "equal_mass_weight", "whatever they show"):
        assert word in section, word  # both arms, the weight formula and the promise to report
    assert "(number of training-pool rows) / (number of local fit rows)" in section
    assert "nrel-before-may-2019" in section and "not calibratable" in section
    assert "docs/prior-looks.md" in section and SOURCE_PHRASE in section
    assert "line 342" not in section  # the entry is named by its words, not by a line number
    for words in (
        "range over three seeds (with the plan's own runs) at one level, kappa 0.3",
        "has one fixed seed",
        "an analogy and not this trial's measured noise",
        "up to 30% therefore reads",
        "no judged cell",
    ):
        assert words in " ".join(section.split()), words
    assert "equal-mass (weights and fitting route)" in section
    assert "the refit helps" in section and "mixed" in section
    flat = " ".join(section.split())
    assert "first k days' outcomes are all known before the first forecast day begins" not in flat
    assert "every row of them is kept" not in flat  # untrue at any unit
    assert "the first k days' outcomes are all known by then" not in flat
    for words in (
        "Rows of the first k site-days can be left out at any unit",
        "still plugged in at that decision time",
        "no readable plug-out",
        "At a pooled unit two sites can also have the same date",
        "count all local rows",
        "the first k counting site-days only",
    ):
        assert words in flat, words

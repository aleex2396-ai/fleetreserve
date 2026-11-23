"""The held-out units' windows and tables of `heldout_truth_build`, on made-up sessions in memory.

1. A Norway or Korea unit's window is the plan's rule on its calibration days, which are the days
   the calibration protocol draws (`gate_run.plan_unit`) at that window; the hand-made units give
   the windows worked out by hand in `heldout_truth_fake`.
2. The window does not change when the unit's test days are changed, and does change when its
   calibration days are changed in the same way; set over all days it would change.
3. Midnight: a portfolio whose plugged-in count peaks at 23:00 gets τ 22:00 and eight service
   hours to 07:00; a day is the date of its τ, a car is on that day only, its departure hour is
   counted from τ across midnight, and `days_from_truth` reads the tables into days.
4. A search that meets a τ twice without settling (a cycle) takes the first τ met twice with its
   calibration days and the S of the window the rule last gave at that τ (the search's last
   step), not the S of an earlier window at that τ nor the S rule applied at that τ to its own
   calibration days, and lists the cycle among the findings; a unit whose calibration days at
   that τ have no hour after τ at half their peak still gets that window; the search ends within
   25 steps. A unit with no calibration day at a τ the search reaches, at the start draw, or with
   no session, abstains with its reason.
5. Every table equals `truth.build` on the same sessions at the same window (no rule of its own),
   with hand-checked numbers for one car of each set; the held setting changes only the energy
   drawn before τ, and the windows are those of the main setting.
6. The a.s.r. units take the workplace window and the units of `load_asr.unit_of`; the Korean
   fast-charger rows are counted, per portfolio and in all.
"""

import hashlib
import io
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fake_heldout import asr_csv, asr_row
from fake_korea import korea_row
from heldout_truth_fake import (
    CYCLE_B_SHORT_DAY,
    KOREA_START,
    NORWAY_DAY,
    NORWAY_ODD_DAY,
    NORWAY_START,
    OSL_T_DAYS,
    TRO_R_DAYS,
    asr_bytes,
    asr_rows,
    at,
    cycle_rows,
    korea_bytes,
    korea_rows,
    norway_bytes,
    norway_days,
    norway_rows,
    three_type_cycle_rows,
    unchecked,
)

import fleetreserve
from fleetreserve import (
    days_from_truth,
    gate_config,
    gate_run,
    load_asr,
    load_heldout_norway,
    load_korea,
    load_norway,
    truth,
    truth_build,
)
from fleetreserve import heldout_truth_build as h
from fleetreserve.sessions import PHYSICAL_LIMIT_KW
from fleetreserve.windows import WORKPLACE, Window, site_days

MATRIX_PATH = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
MATRIX = gate_config.load(MATRIX_PATH)
RULE = h.SplitRule.of(MATRIX, "gate/matrix.json")
NORWAY_LIMIT = PHYSICAL_LIMIT_KW["norway"]


def _norway(tmp_path, rows=None):
    return unchecked(load_heldout_norway, norway_bytes(tmp_path, rows))


def _part(sessions, site):
    return sessions[sessions["site"] == site]


def _search(part, zone=load_norway.ZONE, limit=NORWAY_LIMIT):
    unit = str(part["site"].iloc[0])
    return h.search_window(part, unit=unit, zone=zone, limit_kw=limit, rule=RULE)


def _loaded(tmp_path):
    return {
        "norway": _norway(tmp_path),
        "asr": load_asr.from_bytes(asr_bytes()),
        "korea": load_korea.from_bytes(korea_bytes()),
    }


def test_the_hand_made_units_get_their_hand_worked_windows(tmp_path):
    sessions, _ = _norway(tmp_path)
    found = _search(_part(sessions, "OSL_T"))
    assert (found.window.tau_hour, found.window.service_hours) == (16, 6)
    assert found.reason == ""  # a found window carries no reason to abstain
    assert found.detail["peak_hour"] == 17 and found.detail["peak_median"] == 6.0
    assert found.detail["median_by_hour"][15:24] == [0.0, 4.0, 6, 6, 6, 6, 6, 4, 0]
    assert found.detail["search_taus"] == [16, 16]
    assert found.detail["calibration_days"] == len(found.calibration) == OSL_T_DAYS // 2
    assert found.detail["counting_days_at_tau"] == OSL_T_DAYS
    assert found.detail["days"] == OSL_T_DAYS // 2  # the rule saw the calibration days only
    assert found.detail["calibration_dates"] == [str(d) for d in found.calibration]
    result = h.build_heldout(_loaded(tmp_path), truth.MAIN, RULE)
    got = {u: (e["tau"], e["service"]) for u, e in result.windows.items()}
    assert got == {
        "asr-before-march-2023": ("08:00", "09:00 to 12:00"),
        "asr-from-march-2023": ("08:00", "09:00 to 12:00"),
        "OSL_T": ("16:00", "17:00 to 23:00"),
        "TRO_R": ("16:00", "17:00 to 23:00"),
        "korea-home": ("22:00", "23:00 to 07:00"),
        "korea-work": ("08:00", "09:00 to 17:00"),
        "korea-public-parking": ("09:00", "10:00 to 15:00"),
    }


@pytest.mark.parametrize("site", ["OSL_T", "TRO_R"])
def test_the_calibration_days_are_those_the_protocol_draws_at_the_window(tmp_path, site):
    sessions, _ = _norway(tmp_path)
    part = _part(sessions, site)
    found = _search(part)
    cars, days = truth.build(
        part,
        found.window,
        load_norway.ZONE,
        unit=site,
        days=site_days(part["plug_in"]),
        limit_kw=NORWAY_LIMIT,
        bound_by_charge_fields=False,
    )
    as_text = [
        pd.read_csv(io.BytesIO(truth_build.csv_bytes(t)), dtype=str, keep_default_na=False)
        for t in (cars, days)
    ]
    unit_days = days_from_truth.unit_days(
        *as_text, unit=site, rho=MATRIX.rho, no_fit=MATRIX.no_fit, small_fleet=MATRIX.small_fleet
    )
    plan = gate_run.plan_unit(site, unit_days, MATRIX)
    assert plan.split["calibration"] == [f"{d} {site}" for d in found.calibration]
    assert plan.calibration_days == len(found.calibration) > 0


def _with_days_replaced(tmp_path, chosen):
    days = norway_days(NORWAY_START, OSL_T_DAYS)
    rows = norway_rows("OSL_T", days, patterns=dict.fromkeys(chosen, NORWAY_ODD_DAY))
    return _part(_norway(tmp_path / "changed", rows)[0], "OSL_T")


def test_the_window_does_not_change_when_the_test_days_change(tmp_path):
    base = _search(_part(_norway(tmp_path / "base")[0], "OSL_T"))
    days = norway_days(NORWAY_START, OSL_T_DAYS)
    test_days = [d for d in days if d not in base.calibration]
    assert len(test_days) == OSL_T_DAYS - len(base.calibration) > 0
    assert base.detail["test_days_read"] == 0 and base.read == set(base.calibration)
    changed_part = _with_days_replaced(tmp_path / "test", test_days)
    changed = _search(changed_part)
    assert changed.window == base.window
    assert changed.calibration == base.calibration
    assert changed.detail == base.detail
    # the change is large: over all days the rule would give another window
    profile = h.occupancy_profile(
        changed_part["plug_in"], changed_part["plug_out"], load_norway.ZONE
    )
    assert h.window_from_profile(profile)[0].tau_hour != base.window.tau_hour
    # and the same change made to the calibration days does change the window
    moved = _search(_with_days_replaced(tmp_path / "calibration", list(base.calibration)))
    assert moved.window != base.window


def test_a_portfolio_peaking_at_23_crosses_midnight_and_a_day_is_the_date_of_its_tau(tmp_path):
    sessions, _ = load_korea.from_bytes(korea_bytes())
    part = _part(sessions, "korea-home")
    found = _search(part, load_korea.ZONE, load_korea.LIMIT_KW)
    assert found.detail["peak_hour"] == 23
    assert (found.window.tau_hour, found.window.service_hours) == (22, 8)
    entry = truth_build._window_entry(found.window, load_korea.ZONE)
    assert entry["service"] == "23:00 to 07:00"
    days = site_days(part["plug_in"])
    cars, day_table = truth.build(
        part,
        found.window,
        load_korea.ZONE,
        unit="korea-home",
        days=days,
        limit_kw=load_korea.LIMIT_KW,
        bound_by_charge_fields=False,
    )
    assert len(day_table) == 10 and day_table["counts"].all()
    assert (day_table["cars"] == 4).all()  # the car plugged in at 22:30 is not at τ
    first = day_table["tau"].iloc[0]
    assert str(first) == "2022-03-01 22:00:00+09:00"
    assert (cars["depart_hour"] == 9).all()  # 22:00 to 07:30 the next morning
    assert not cars["in_window_departure"].any()
    assert cars.groupby("session_id")["date"].nunique().max() == 1  # each car on one day
    assert (cars["plug_in"].dt.date == cars["date"]).all()
    assert (cars["plug_out"].dt.date == cars["date"] + timedelta(days=1)).all()
    one = cars.iloc[0]
    assert one["energy_before_kwh"] == pytest.approx(7.0 * 0.5)
    assert one["room_kwh"] == pytest.approx(30.0 - 3.5)
    assert one["rate_kw"] == pytest.approx(3.0)
    as_text = [
        pd.read_csv(io.BytesIO(truth_build.csv_bytes(t)), dtype=str, keep_default_na=False)
        for t in (cars, day_table)
    ]
    read = days_from_truth.unit_days(
        *as_text, unit="korea-home", rho=0.0, no_fit="cap", small_fleet="keep"
    )
    assert [d.day.date for d in read.days] == [KOREA_START + timedelta(days=k) for k in range(10)]
    assert {d.day.n_service for d in read.days} == {8}
    assert all(car.departure == 9 for d in read.days for car in d.day.cars)


def test_an_abstaining_search_records_the_taus_tried_and_the_days_read(tmp_path):
    sessions, _ = load_korea.from_bytes(korea_bytes())
    zone, limit = h.SITE_SETS["korea"]
    lodging = _part(sessions, "korea-lodging")
    found = _search(lodging, zone=zone, limit=limit)
    start = h.calibration_dates("korea-lodging", "korea-lodging", norway_days(KOREA_START, 5), RULE)
    assert len(start) == 2  # 5 days: 2 calibration days in the start draw, none at τ 17:00
    assert (found.window, found.calibration) == (None, ())
    assert found.detail == {"search_taus": [17]} and found.read == set(start)
    assert found.reason == "no calibration day at τ 17:00 (0 counting days)"
    visits = _search(_part(sessions, "korea-visits"), zone=zone, limit=limit)
    assert (visits.detail, visits.read) == ({"search_taus": []}, set())


def test_an_odd_number_of_counting_days_gives_one_test_day_more(tmp_path):
    rows = norway_rows("OSL_T", norway_days(NORWAY_START, 29))
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    sessions, _ = _norway(tmp_path, rows)
    found = _search(_part(sessions, "OSL_T"))
    assert found.detail["counting_days_at_tau"] == 29
    assert found.detail["calibration_days"] == len(found.calibration) == 14  # 29 // 2; 15 test


def _cycling(tmp_path, b_day=None):
    """The norway set with OSL_T the cycling site of `heldout_truth_fake`, TRO_R as usual."""
    rows, a_days, b_days = cycle_rows(b_day=b_day)
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    return _norway(tmp_path, rows), a_days, b_days


def test_a_search_that_cycles_takes_the_first_tau_met_twice_with_its_calibration_days(tmp_path):
    """Worked by hand (plan 2, the owner's rule of 15 November 2026; reading W1). OSL_T has 34
    calendar days, 4 of type B and 30 of type A (`heldout_truth_fake`); a day counts at τ with at
    least 3 cars plugged in at τ.
    - Start draw: 17 of the 34 calendar days (half, below 200 days). At most 4 are B, so at least
      13 are A, a strict majority of identical days, and each hour's median is A's count: the
      peak is 15:00 (10 cars at 15:00 and 16:00), so τ 14:00.
    - At τ 14:00 the 4 B days count (6 cars at 14:00) and no A day (none). 2 are drawn; their
      median is B's: the peak is 11:00 (10 cars from 11:00 to 13:00, the longest run), τ 10:00.
    - At τ 10:00 the 30 A days count (4 cars at 10:00) and no B day (none). 15 are drawn: τ
      14:00 again, already met: search τ [14, 10, 14].
    - The first τ met twice is 14:00, with the 2 calibration days drawn at 14:00. S is that of
      the window the rule last gave at 14:00, at the third step, on the 15 A days drawn at
      10:00: from 15:00 while A's median is at least half its peak (10 / 2 = 5): 10 cars at
      15:00 and 16:00, none at 17:00: two hours, 15:00 to 17:00. (The S rule applied with τ kept
      at 14:00 to the 2 B days would give three hours: 6 cars at 15:00, 16:00 and 17:00.) The
      rule's own τ on the B days is 10:00: that is the cycle, and τ stays 14:00.
    """
    (sessions, _), a_days, b_days = _cycling(tmp_path)
    found = _search(_part(sessions, "OSL_T"))
    assert (found.window.tau_hour, found.window.service_hours) == (14, 2)
    assert found.window.rule == h.CYCLE_RULE and found.reason == ""
    calibration = _protocol_draw("OSL_T", "OSL_T", b_days, 2, RULE.seed)
    assert found.calibration == calibration and set(calibration) < set(b_days)
    detail = found.detail
    assert detail["search_taus"] == [14, 10, 14]
    assert detail["search_cycle"] is True and detail["cycle_rule_tau_hour"] == 10
    assert (detail["peak_hour"], detail["peak_median"]) == (11, 10.0)
    assert detail["median_by_hour"][9:19] == [0, 0, 10, 10, 10, 6, 6, 6, 6, 0]
    assert detail["days"] == detail["calibration_days"] == 2
    assert detail["calibration_dates"] == [str(day) for day in calibration]
    assert detail["counting_days_at_tau"] == 4
    assert detail["window_days"] == (
        "calibration days of the unit at the first τ its search met twice; S from the window the "
        "rule last gave at that τ, on the calibration days at the τ before it (plan 2, a search "
        "that cycles; reading W1)"
    )
    days = norway_days(NORWAY_START, 34)
    start = _protocol_draw("OSL_T", "OSL_T", days, 17, RULE.seed)
    at_10 = _protocol_draw("OSL_T", "OSL_T", a_days, 15, RULE.seed)
    assert detail["start_days"] == 17 and len(set(start) & set(b_days)) <= 4
    assert found.read == set(start) | set(calibration) | set(at_10)
    assert detail["days_read"] == len(found.read)
    test_days = set(b_days) - set(calibration)  # counting days at 14:00 that are not drawn
    assert detail["test_days_read"] == len(found.read & test_days)
    # the command builds the unit at that window and lists the cycle among the findings
    loaded = {**_loaded(tmp_path / "all"), "norway": _cycling(tmp_path / "again")[0]}
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    entry = result.windows["OSL_T"]
    assert (entry["tau"], entry["service"], entry["rule"]) == (
        "14:00",
        "15:00 to 17:00",
        h.CYCLE_RULE,
    )
    assert result.summary["OSL_T"]["counting_days"] == 4
    assert (
        "OSL_T: the window search cycles through τ [14, 10, 14]: τ 14:00, the first τ met "
        "twice, stands with its calibration days, on which the rule's own τ is 10:00 (plan 2; "
        "reading W1)"
    ) in h.findings(result)
    line = [text for text in h.report_lines(result, {}) if text.startswith("- OSL_T: τ")]
    assert line[0].startswith("- OSL_T: τ 14:00, S 15:00 to 17:00 (2 hours); median peak 10.0 ")
    assert (
        "; search τ [14, 10, 14], a cycle: the rule's own τ on these days 10:00; S from the "
        "window the rule last gave at τ 14:00, on the calibration days at τ 10:00; days read "
    ) in line[0]


def test_a_cycle_whose_days_at_tau_have_no_hour_at_half_their_peak_after_tau_has_a_window(
    tmp_path,
):
    """As above with B short: τ, the calibration days and the cycle are the same (4 cars at 14:00
    still count). On those 2 B-short days the median at 15:00 is 0, below half their peak (5),
    so the S rule applied with τ kept at 14:00 would give no hour; S is that of the window the
    rule last gave at 14:00, on the 15 A days drawn at 10:00, as above: two hours, 15:00 to
    17:00. The unit has a window and its tables, and does not abstain."""
    (sessions, _), _, b_days = _cycling(tmp_path, b_day=CYCLE_B_SHORT_DAY)
    found = _search(_part(sessions, "OSL_T"))
    assert (found.window.tau_hour, found.window.service_hours) == (14, 2)
    assert found.window.rule == h.CYCLE_RULE and found.reason == ""
    assert found.calibration == _protocol_draw("OSL_T", "OSL_T", b_days, 2, RULE.seed)
    assert found.detail["search_taus"] == [14, 10, 14]
    assert found.detail["cycle_rule_tau_hour"] == 10
    assert found.detail["median_by_hour"][13:17] == [10, 4, 0, 0]  # B short: none at 15:00
    assert found.detail["days_read"] == len(found.read) > 0
    loaded = {
        **_loaded(tmp_path / "all"),
        "norway": _cycling(tmp_path / "again", CYCLE_B_SHORT_DAY)[0],
    }
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert "OSL_T" not in result.abstained and "OSL_T" in result.tables
    entry = result.windows["OSL_T"]
    assert (entry["tau"], entry["service"], entry["service_hours"]) == (
        "14:00",
        "15:00 to 17:00",
        2,
    )
    assert result.summary["OSL_T"]["counting_days"] == 4
    assert not [text for text in h.findings(result) if text.startswith("OSL_T: no tables")]


def _three_type_cycling(tmp_path):
    """The norway set with OSL_T the second cycling site of `heldout_truth_fake`, TRO_R as usual."""
    rows, a_days, b_days, c_days = three_type_cycle_rows()
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    return _norway(tmp_path, rows), a_days, b_days, c_days


def test_on_a_cycle_s_is_that_of_the_window_the_rule_last_gave_at_the_standing_tau(tmp_path):
    """Worked by hand (plan 2, the owner's decisions of 15 November 2026; reading W1(c)). OSL_T
    has 34 calendar days: 4 of type A, 4 of type B and 26 of type C (`heldout_truth_fake`); a day
    counts at τ with at least 3 cars plugged in at τ, and no C day counts at 10:00 or 14:00.
    - Start draw: 17 of the 34 days. At most 8 are A or B, so at least 9 are C, a strict majority
      of identical days, and each hour's median is C's count: 10 cars from 15:00 to 19:00, so
      the peak is 15:00 (the run at the highest median starts there), τ 14:00, and S runs from
      15:00 while the median is at least 10 / 2 = 5: five hours, 15:00 to 20:00.
    - At τ 14:00 the 4 B days count (6 cars at 14:00), no A or C day (none). 2 are drawn: B's
      peak is 11:00 (10 cars from 11:00 to 13:00), τ 10:00.
    - At τ 10:00 the 4 A days count (4 cars at 10:00), no B or C day (none). 2 are drawn: A's
      peak is 15:00 (10 cars at 15:00 and 16:00), τ 14:00 again, already met: search τ
      [14, 10, 14], a cycle; the rule's own τ on the B days is 10:00.
    - τ 14:00 stands with its 2 calibration days, the B days drawn at 14:00. S is that of the
      window the rule last gave at 14:00, at the third step on the 2 A days: from 15:00 while
      A's median is at least 10 / 2 = 5: 10 cars at 15:00 and 16:00, none at 17:00: two hours,
      15:00 to 17:00.
    Not the S of the first window the rule gave at 14:00 (five hours, on the start draw), and not
    the S rule applied with τ kept at 14:00 to the B days (6 cars at 15:00 to 17:00, none at
    18:00: three hours).
    """
    (sessions, _), a_days, b_days, c_days = _three_type_cycling(tmp_path)
    part = _part(sessions, "OSL_T")
    found = _search(part)
    assert found.detail["search_taus"] == [14, 10, 14]
    assert found.detail["search_cycle"] is True and found.detail["cycle_rule_tau_hour"] == 10
    assert (found.window.tau_hour, found.window.service_hours) == (14, 2)
    assert found.window.rule == h.CYCLE_RULE and found.reason == ""
    calibration = _protocol_draw("OSL_T", "OSL_T", b_days, 2, RULE.seed)
    assert found.calibration == calibration
    # the hand steps on the protocol's draws, written out apart from the module
    profile = h.occupancy_profile(part["plug_in"], part["plug_out"], load_norway.ZONE)
    start = _protocol_draw("OSL_T", "OSL_T", norway_days(NORWAY_START, 34), 17, RULE.seed)
    assert len(set(start) & set(c_days)) >= 9
    first = h.window_from_profile(profile.loc[list(start)])[0]
    assert (first.tau_hour, first.service_hours) == (14, 5)
    at_10 = _protocol_draw("OSL_T", "OSL_T", a_days, 2, RULE.seed)
    last = h.window_from_profile(profile.loc[list(at_10)])[0]
    assert (last.tau_hour, last.service_hours) == (14, 2)
    assert found.read == set(start) | set(calibration) | set(at_10)
    # the command builds the unit at that window, counting the 4 B days at 14:00
    loaded = {**_loaded(tmp_path / "all"), "norway": _three_type_cycling(tmp_path / "again")[0]}
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    entry = result.windows["OSL_T"]
    assert (entry["tau"], entry["service"], entry["service_hours"]) == (
        "14:00",
        "15:00 to 17:00",
        2,
    )
    assert result.summary["OSL_T"]["counting_days"] == 4


def test_the_search_ends_after_at_most_25_steps_as_tau_has_24_values(tmp_path, monkeypatch):
    """A made-up rule that gives a new τ at each step, 16:00, 17:00, ..., 15:00, with an S of
    1 + (step mod 7) hours, and with every day counting at every τ: the 25th step meets 16:00
    again and the search stops there. τ 16:00 stands with its calibration days, and S is that of
    the window the rule last gave at 16:00, the 25th step's: 1 + 4 = 5 hours, 17:00 to 22:00;
    not the first step's 2 hours, nor the real rule's 6 hours on those days."""
    sessions, _ = _norway(tmp_path)
    part = _part(sessions, "OSL_T")
    real = h.window_from_profile
    calls = []

    def stepping(profile):
        window, detail = real(profile)
        calls.append(1)
        step = len(calls)
        return Window((15 + step) % 24, 1 + step % 7, window.rule), detail

    def every_day_counts(_part, _window, _zone, *, days, **_options):
        return None, pd.DataFrame({"date": days, "counts": [True] * len(days)})

    monkeypatch.setattr(h, "window_from_profile", stepping)
    monkeypatch.setattr(h.truth, "build", every_day_counts)
    found = _search(part)
    assert found.detail["search_taus"] == [(16 + k) % 24 for k in range(24)] + [16]
    assert len(calls) == 25
    assert (found.window.tau_hour, found.window.service_hours) == (16, 5)
    assert found.window.rule == h.CYCLE_RULE
    assert found.detail["cycle_rule_tau_hour"] == 17  # the made-up rule's second step
    every = norway_days(NORWAY_START, OSL_T_DAYS)
    assert found.calibration == _protocol_draw("OSL_T", "OSL_T", every, 15, RULE.seed)


def test_units_without_a_calibration_day_or_a_session_abstain_with_their_reason(tmp_path):
    result = h.build_heldout(_loaded(tmp_path), truth.MAIN, RULE)
    # a unit whose search ran keeps its steps and its days read: korea-lodging read its 2 start
    # days (5 days, none counting at τ 17:00), korea-visits none (1 day, no start draw)
    assert result.abstained == {
        "korea-lodging": {
            "reason": "no calibration day at τ 17:00 (0 counting days)",
            "sessions_loaded": 15,
            "rows_removed": {},
            "search_taus": [17],
            "days_read": 2,
        },
        "korea-visits": {
            "reason": "no calibration day at the start draw",
            "sessions_loaded": 3,
            "rows_removed": {},
            "search_taus": [],
            "days_read": 0,
        },
    }
    assert "korea-lodging" not in result.tables and "korea-visits" not in result.tables
    findings = h.findings(result)
    assert any(text.startswith("korea-lodging: no tables, the unit abstains") for text in findings)
    loaded = _loaded(tmp_path / "again")
    sessions, dropped = loaded["norway"]
    loaded["norway"] = (sessions[sessions["site"] == "OSL_T"], dropped)
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert result.abstained["TRO_R"]["reason"] == "no session"
    assert "TRO_R" not in result.tables and "OSL_T" in result.tables
    # every TRO_R row without energy: rows were read and all dropped, its 36 rows counted as removed
    no_energy = [(user, start, end, 0.0) for user, start, end, _ in NORWAY_DAY]
    rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS))
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS), pattern=no_energy)
    loaded["norway"] = _norway(tmp_path / "empty", rows)
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert result.abstained["TRO_R"] == {
        "reason": "every row dropped by cleaning: 36 rows, by reason "
        "energy missing or not positive: 36",
        "sessions_loaded": 0,
        "rows_removed": {"energy missing or not positive": TRO_R_DAYS * len(NORWAY_DAY)},
    }
    assert TRO_R_DAYS * len(NORWAY_DAY) == 36
    # the first site with every row dropped: the second is still built
    rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS), pattern=no_energy)
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    loaded["norway"] = _norway(tmp_path / "first-empty", rows)
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert result.abstained["OSL_T"]["reason"].startswith("every row dropped by cleaning: 180 rows")
    assert "OSL_T" not in result.tables and "TRO_R" in result.tables
    # a unit that abstains in the window search keeps its removed rows, by reason
    lodging = [
        korea_row(
            "lodging-zero",
            "41",
            "C-korea-lodging",
            "Hotel",
            at(KOREA_START, 17.5),
            at(KOREA_START, 33.5),
            0.0,
        ),
        korea_row(
            "lodging-fast",
            "41",
            "F-korea-lodging",
            "Hotel",
            at(KOREA_START, 12.0),
            at(KOREA_START, 12.5),
            20.0,
            charger_type="1",
        ),
    ]
    loaded = {
        **_loaded(tmp_path / "lodging"),
        "korea": load_korea.from_bytes(korea_bytes(korea_rows() + lodging)),
    }
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert result.abstained["korea-lodging"] == {
        "reason": "no calibration day at τ 17:00 (0 counting days)",
        "sessions_loaded": 15,
        "rows_removed": {"energy missing or not positive": 1, h.FAST_REASON: 1},
        "search_taus": [17],
        "days_read": 2,
    }


def test_test_days_read_counts_the_read_days_that_are_test_days_at_the_window(tmp_path):
    days = norway_days(NORWAY_START, OSL_T_DAYS)
    # a third of the days have only two cars at τ 16:00, so they do not count there
    thin = [(f"w{k}", 15.5, 22.5, 10.0) for k in (1, 2)] + NORWAY_DAY[4:]
    rows = norway_rows("OSL_T", days, patterns={d: thin for d in days[::3]})
    part = _part(_norway(tmp_path, rows)[0], "OSL_T")
    found = _search(part)
    counting = [d for d in days if d not in days[::3]]
    assert found.detail["counting_days_at_tau"] == len(counting)
    test_days = set(counting) - set(found.calibration)
    assert found.detail["test_days_read"] == len(found.read & test_days)
    assert found.detail["days_read"] == len(found.read)
    assert set(found.calibration) <= found.read
    start = h.calibration_dates("OSL_T", "OSL_T", days, RULE)
    assert found.detail["start_days"] == len(start) == OSL_T_DAYS // 2
    assert set(start) <= found.read


def test_a_start_draw_with_more_counting_days_than_the_final_count_reads_test_days(tmp_path):
    days = norway_days(NORWAY_START, OSL_T_DAYS)
    start = h.calibration_dates("OSL_T", "OSL_T", days, RULE)
    # four days outside the start draw have only two cars at τ 16:00: 26 counting days remain,
    # so 13 calibration days, drawn by rank from the start draw's 15 days, which all count
    thin = [(f"w{k}", 15.5, 22.5, 10.0) for k in (1, 2)] + NORWAY_DAY[4:]
    thin_days = [d for d in days if d not in start][:4]
    rows = norway_rows("OSL_T", days, patterns=dict.fromkeys(thin_days, thin))
    found = _search(_part(_norway(tmp_path, rows)[0], "OSL_T"))
    assert found.detail["search_taus"] == [16, 16]
    assert (found.detail["counting_days_at_tau"], found.detail["calibration_days"]) == (26, 13)
    assert set(found.calibration) < set(start)
    assert found.detail["test_days_read"] == 2  # 15 start days, 13 of them calibration days
    loaded = {**_loaded(tmp_path), "norway": _norway(tmp_path, rows)}
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    listed = [text for text in h.findings(result) if "test days" in text]
    assert listed == ["OSL_T: the window search read 2 test days of its final window (reading W1)"]
    line = [text for text in h.report_lines(result, {}) if text.startswith("- OSL_T: τ")]
    assert line == [
        "- OSL_T: τ 16:00, S 17:00 to 23:00 (6 hours); median peak 6.0 cars at 17:00 over 13 "
        "calibration days (seed 1; search τ [16, 16]; days read 15, test days read 2); "
        "earliest-peak reading τ 16:00 would give 26 counting days"
    ]


def _direct(part, window, zone, limit, unit, days, setting, rates=None):
    return truth.build(
        part,
        window,
        zone,
        unit=unit,
        days=days,
        limit_kw=limit,
        bound_by_charge_fields=False,
        rates=rates,
        setting=setting,
    )


@pytest.mark.parametrize("setting", truth.SETTINGS)
def test_every_table_equals_truth_build_on_the_same_sessions(tmp_path, setting):
    loaded = _loaded(tmp_path)
    result = h.build_heldout(loaded, setting, RULE)
    assert list(result.tables) == [
        "asr-before-march-2023",
        "asr-from-march-2023",
        "OSL_T",
        "TRO_R",
        "korea-home",
        "korea-work",
        "korea-public-parking",
    ]
    asr_sessions = loaded["asr"][0]
    average, kind = truth.session_averages(asr_sessions, load_asr.LIMIT_KW)
    rates = truth.user_rates(asr_sessions, average, kind)
    for unit, before in zip(load_asr.UNITS, (True, False), strict=True):
        days = [d for d in site_days(asr_sessions["plug_in"]) if (d < date(2023, 3, 1)) == before]
        assert len(days) == 2
        expected = _direct(
            asr_sessions, WORKPLACE, load_asr.ZONE, load_asr.LIMIT_KW, unit, days, setting, rates
        )
        for got, want in zip(result.tables[unit], expected, strict=True):
            assert truth_build.csv_bytes(got) == truth_build.csv_bytes(want)
    for set_name, zone, limit in (
        ("norway", load_norway.ZONE, NORWAY_LIMIT),
        ("korea", load_korea.ZONE, load_korea.LIMIT_KW),
    ):
        sessions = loaded[set_name][0]
        for unit in result.tables:
            if unit not in set(sessions["site"]):
                continue
            part = _part(sessions, unit)
            entry = result.windows[unit]
            window = Window(int(entry["tau"][:2]), entry["service_hours"], "hand")
            expected = _direct(part, window, zone, limit, unit, site_days(part["plug_in"]), setting)
            for got, want in zip(result.tables[unit], expected, strict=True):
                assert truth_build.csv_bytes(got) == truth_build.csv_bytes(want)
    held = setting == truth.HELD
    # one car per set by hand: a.s.r. EV1 on 27 February 2023, OSL_T u1 on 6 January 2020
    cars = result.tables["asr-before-march-2023"][0]
    ev1 = cars[(cars["user_id"] == "EV1") & (cars["date"] == date(2023, 2, 27))].iloc[0]
    assert ev1["energy_before_kwh"] == pytest.approx(0.0 if held else 11.0 * 0.5)
    assert ev1["room_kwh"] == pytest.approx(20.0 if held else 14.5)
    # its rate is its highest average over both a.s.r. units (reading T2): 4 kW on 2 March
    assert (ev1["depart_hour"], ev1["rate_kw"]) == (9, pytest.approx(4.0))
    assert result.tables["asr-before-march-2023"][1]["cars"].tolist() == [3, 3]
    cars = result.tables["OSL_T"][0]
    u1 = cars[(cars["user_id"] == "u1") & (cars["date"] == NORWAY_START)].iloc[0]
    nominal = load_norway.NOMINAL_KW
    assert u1["energy_before_kwh"] == pytest.approx(0.0 if held else nominal * 0.5)
    assert u1["room_kwh"] == pytest.approx(10.0 - (0.0 if held else nominal * 0.5))
    assert (u1["depart_hour"], bool(u1["in_window_departure"])) == (6, True)
    assert u1["rate_kw"] == pytest.approx(10.0 / 7.0)


def test_the_held_setting_keeps_the_main_windows_and_counting_days(tmp_path):
    main = h.build_heldout(_loaded(tmp_path / "main"), truth.MAIN, RULE)
    held = h.build_heldout(_loaded(tmp_path / "held"), truth.HELD, RULE)
    assert held.windows == main.windows
    assert held.abstained == main.abstained
    for unit, (cars, days) in main.tables.items():
        held_cars, held_days = held.tables[unit]
        pd.testing.assert_frame_equal(held_days, days)
        assert (held_cars["energy_before_kwh"] == 0).all()
        assert (held_cars["room_kwh"] == held_cars["energy_kwh"]).all()
        same = [
            c
            for c in truth.CAR_COLUMNS
            if c not in ("energy_before_kwh", "energy_before_rule", "room_kwh", "room_fits")
        ]
        pd.testing.assert_frame_equal(held_cars[same], cars[same])


def test_the_asr_units_and_the_korean_fast_charger_rows_are_counted(tmp_path):
    result = h.build_heldout(_loaded(tmp_path), truth.MAIN, RULE)
    assert result.fast_charger_rows == 3
    fast = "fast charger (ChargerType 1)"
    assert result.summary["korea-home"]["rows_removed"] == {fast: 2}
    assert result.summary["korea-work"]["rows_removed"] == {fast: 1}
    assert result.summary["korea-public-parking"]["rows_removed"] == {}
    for unit in load_asr.UNITS:
        entry = result.summary[unit]
        assert entry["role"] == "held-out"
        assert (entry["sessions_loaded"], entry["days"], entry["counting_days"]) == (8, 2, 2)
        assert entry["rows_removed"] == {"energy missing or not positive": 2}
        assert entry["sites"] == ["asr"]
    assert result.checks["korea"]["rows_read"] == {
        "korea-home": 52,
        "korea-work": 31,
        "korea-public-parking": 16,
        "korea-lodging": 15,
        "korea-visits": 3,
    }
    assert result.checks["asr"]["rows_read"] == {"asr": 20}
    assert result.checks["norway"]["rows_read"] == {"OSL_T": 180, "TRO_R": 36}


def test_asr_units_without_a_session_or_a_day_abstain(tmp_path):
    loaded = _loaded(tmp_path)
    before_only = [row for row in asr_rows() if row["start_datetime"] < "2023-03-01"]
    # one row of 2 March without energy: removed, and counted in the unit with no day
    removed_late = [row for row in asr_rows() if row["start_datetime"] >= "2023-03-02"][:1]
    removed_late = [{**row, "total_energy": 0.0} for row in removed_late]
    loaded["asr"] = unchecked(load_asr, asr_csv(before_only + removed_late))
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert "asr-before-march-2023" in result.tables
    assert "asr-from-march-2023" not in result.tables
    assert result.abstained["asr-from-march-2023"] == {
        "reason": "every row dropped by cleaning: 1 rows, by reason "
        "energy missing or not positive: 1",
        "sessions_loaded": 0,
        "rows_removed": {"energy missing or not positive": 1},
    }
    # a period with no row at all is recorded as no session (the command refuses it): the other
    # period is still built by the build's functions
    from_only = [row for row in asr_rows() if row["start_datetime"] >= "2023-03-01"]
    loaded["asr"] = unchecked(load_asr, asr_csv(from_only))
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert "asr-from-march-2023" in result.tables
    assert "asr-before-march-2023" not in result.tables
    assert result.abstained["asr-before-march-2023"]["reason"] == h.NO_SESSION
    # no session left, 2 March left out: 10 rows removed before March, 5 from March
    none_left = [
        {**row, "total_energy": 0.0} for row in asr_rows() if row["start_datetime"] < "2023-03-02"
    ]
    loaded["asr"] = unchecked(load_asr, asr_csv(none_left))
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    removed = dict(zip(load_asr.UNITS, (10, 5), strict=True))
    for unit in load_asr.UNITS:
        assert result.abstained[unit] == {
            "reason": "every row dropped by cleaning: "
            f"{removed[unit]} rows, by reason energy missing or not positive: {removed[unit]}",
            "sessions_loaded": 0,
            "rows_removed": {"energy missing or not positive": removed[unit]},
        }
        assert unit not in result.tables
    # listed in the units' order, as the report lists them
    assert [u for u in result.abstained if u in load_asr.UNITS] == list(load_asr.UNITS)
    assert (
        "- asr-from-march-2023 (held-out): abstains, every row dropped by cleaning: 5 rows, by "
        "reason energy missing or not positive: 5; sessions 0, removed "
        "{'energy missing or not positive': 5}"
    ) in h.report_lines(result, {})
    # two reasons: listed in the order of their names, as `truth_build` lists them
    backwards = asr_row("EV9", "2023-02-27T10:00:00", "2023-02-27T09:00:00", 5.0, evse="EVSE-9")
    loaded["asr"] = unchecked(load_asr, asr_csv([*none_left, backwards]))
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert (
        "- asr-before-march-2023 (held-out): abstains, every row dropped by cleaning: 11 rows, "
        "by reason energy missing or not positive: 10, plug-out not after plug-in: 1; sessions 0, "
        "removed {'energy missing or not positive': 10, 'plug-out not after plug-in': 1}"
    ) in h.report_lines(result, {})


def test_a_unit_without_a_counting_day_is_reported_without_a_spread(tmp_path):
    loaded = _loaded(tmp_path)
    two_cars = [row for row in asr_rows() if row["EV_id_x"] in ("EV1", "EV2")]
    loaded["asr"] = load_asr.from_bytes(asr_csv(two_cars))
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert result.summary["asr-before-march-2023"]["counting_days"] == 0
    line = [t for t in h.report_lines(result, {}) if t.startswith("- asr-before-march-2023 (")]
    assert len(line) == 1
    assert (
        ", days 2, counting days 0, cars per counting day median None (p10 None, p90 None),"
        in (line[0])
    )


def test_the_report_puts_each_share_of_a_unit_in_its_place(tmp_path):
    loaded = _loaded(tmp_path)
    day = "2023-02-27"
    extra = [  # one car leaves in the first hour; two have drawn all their energy by τ 08:00
        asr_row("EV6", f"{day}T07:50:00", f"{day}T08:40:00", 5.0, evse="EVSE-6"),
        asr_row("EV7", f"{day}T04:00:00", f"{day}T12:00:00", 1.0),
        asr_row("EV8", f"{day}T04:00:00", f"{day}T12:00:00", 1.0),
    ]
    loaded["asr"] = load_asr.from_bytes(asr_csv(asr_rows() + extra))
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    entry = result.summary["asr-before-march-2023"]
    zero, hour_0 = entry["share_zero_room"], entry["share_depart_hour_zero"]
    # 9 cars on the unit's counting days: EV7 and EV8 with no room, EV6 leaving in hour 0
    assert (zero, hour_0) == pytest.approx((2 / 9, 1 / 9), abs=1e-6)
    line = [t for t in h.report_lines(result, {}) if t.startswith("- asr-before-march-2023 (")]
    assert f", zero room {zero}, " in line[0] and line[0].endswith(f", depart hour 0 {hour_0}")


def test_one_test_day_read_is_a_finding_and_the_findings_keep_the_units_order(tmp_path):
    result = h.build_heldout(_loaded(tmp_path), truth.MAIN, RULE)
    for entry in result.windows.values():
        if "test_days_read" in entry:
            entry["test_days_read"] = 1
    listed = [text.split(":")[0] for text in h.findings(result) if "test days" in text]
    assert listed == ["OSL_T", "TRO_R", "korea-home", "korea-work", "korea-public-parking"]


def test_the_daylight_saving_check_reads_the_clock_of_the_sets_zone(tmp_path):
    # OSL_T from 15 March to 11 April 2020 crosses the change of Sunday 29 March in Oslo, with
    # ten weekdays of six plug-ins on each side; TRO_R keeps its January days and no change.
    days = norway_days(date(2020, 3, 15), 28)
    rows = norway_rows("OSL_T", days) + norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    loaded = {**_loaded(tmp_path), "norway": _norway(tmp_path, rows)}
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    assert result.checks["norway"]["dst_shift"]["OSL_T"] == [
        {
            "change": "2020-03-29",
            "direction": "spring",
            "plug_ins_before": 60,
            "plug_ins_after": 60,
            "median_before_h": 15.5,
            "median_after_h": 15.5,
            "shift_h": 0.0,
        }
    ]
    assert result.checks["norway"]["dst_shift"]["TRO_R"] == []


def test_a_report_without_findings_says_none(tmp_path, monkeypatch):
    result = h.build_heldout(_loaded(tmp_path), truth.MAIN, RULE)
    monkeypatch.setattr(h, "findings", lambda _result: [])
    lines = h.report_lines(result, {})
    header = lines.index(HEADING)
    assert lines[header + 1 : header + 3] == ["- none", ""]


# The heading names every kind of line the findings hold (`h.findings`, `data_checks.findings`).
HEADING = (
    "FINDINGS (checks that contradict an assumption of the plan; any reason that removes more "
    "than 5% of a site's rows, the plan's own removal of fast-charger sessions included; units "
    "abstaining; window searches that cycle or read test days)"
)


def test_the_findings_heading_lists_a_large_fast_charger_share_as_a_removal_not_a_contradiction(
    tmp_path,
):
    # korea-home: 2 fast-charger rows of 52 read (10 days of 5 cars, and 2), about 3.8%; with
    # 4 more the share is 6 of 56, about 10.7%, above 5%, so the share is listed
    extra = [
        korea_row(
            f"more-fast-{k}",
            "99",
            "F-korea-home",
            "Apartment",
            at(KOREA_START, 12.0),
            at(KOREA_START, 12.5),
            20.0,
            charger_type="1",
        )
        for k in range(4)
    ]
    loaded = {
        **_loaded(tmp_path),
        "korea": load_korea.from_bytes(korea_bytes(korea_rows() + extra)),
    }
    result = h.build_heldout(loaded, truth.MAIN, RULE)
    lines = h.report_lines(result, {})
    found = lines[lines.index(HEADING) + 1 :]
    assert f"- korea korea-home: 10.7% of rows removed as '{h.FAST_REASON}'" in found
    assert h.FAST_REASON == "fast charger (ChargerType 1)"


def _protocol_draw(unit: str, site: str, days: list, count: int, seed: int) -> tuple:
    """The calibration days written out from plan section 6's protocol, apart from the module:
    each key "<date> <site>" ranked by SHA-256 of [unit, seed, key], the first `count` keys
    taken, returned in date order."""
    keys = {f"{day} {site}": day for day in days}
    rank = {k: hashlib.sha256(json.dumps([unit, seed, k]).encode("utf-8")).digest() for k in keys}
    chosen = sorted(keys, key=lambda k: (rank[k], k))[:count]
    return tuple(sorted(keys[k] for k in chosen))


@pytest.mark.parametrize(
    "rule, n_days, count",
    [
        (RULE, 250, 100),  # the matrix's rule: 100 days from 200 days on
        (RULE, 200, 100),
        (RULE, 199, 99),  # half the days, rounded down, below 200
        (RULE, 2, 1),
        (RULE, 1, 0),  # no test day would be left: none
        (h.SplitRule(7, 10, 30, "made-up"), 30, 10),  # a made-up rule, at its half-rule edge
        (h.SplitRule(7, 10, 30, "made-up"), 29, 14),
        (h.SplitRule(7, 10, None, "made-up"), 12, 10),  # no half rule
        (h.SplitRule(7, 10, None, "made-up"), 10, 0),
    ],
)
def test_calibration_dates_draw_the_protocol_s_days_from_dates_in_any_order(rule, n_days, count):
    assert (RULE.seed, RULE.n_calibration, RULE.half_below) == (1, 100, 200)
    days = [date(2022, 1, 1) + timedelta(days=k) for k in range(n_days)]
    shuffled = sorted(days, key=lambda d: hashlib.sha256(str(d).encode()).digest())
    assert shuffled != days or n_days < 3
    # the unit and the site differ, so that each is pinned to its own use
    got = h.calibration_dates("unit-u", "site-s", shuffled, rule)
    expected = _protocol_draw("unit-u", "site-s", days, count, rule.seed) if count else ()
    assert got == expected and len(got) == count

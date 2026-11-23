"""Losses and calibration are over offering days (plan sections 1 and 6), on made-up days.

An offering day is a counting day that does not abstain. A day that abstains offers nothing, has no
loss, is in neither mean and is not counted in n. The days here are stand-ins that carry only what
`gate_calibrate` reads, so that the rule is tested without a solver:

1. Most calibration days abstain and the few offering days fail from the first hour: the unit
   abstains, on the audited-set route and on the every-request route. Counted over all calibration
   days (the old rule), 4 failing days among 50 pass the bound and the unit would be given the top
   of the grid.
2. The abstain rule counts offering days: 10 offering days among 50 do not calibrate the driver
   loss (it needs 19); 19 do. The reason states the count.
3. Test results by hand: the offering and failing days, the mean losses over the offering test
   days, the mean offer per test day and per offering day.
4. The summary line of an offer shows the offering days out of the test days and the failing days.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from gate_fake import TOP

from fleetreserve import crc, gate_calibrate, gate_config, gate_day, gate_summary

RUNG = "fixed"


def _day(offers: bool, value: float = 4.0):
    return SimpleNamespace(
        abstains=not offers,
        value=value if offers else 0.0,
        n_cars=3,
        certificate=SimpleNamespace(policy=None),
        day=None,
    )


class _Cache:
    """Stands in for `gate_calibrate.DayCache`: the same losses at every scale above 0, 0 on a day
    that abstains (as `gate_day.evaluate` gives)."""

    def __init__(self, prepared, slot, driver):
        self.prepared = prepared
        self.slot, self.driver = slot, driver

    def get(self, days, beta):
        out = []
        for d in days:
            slot, driver = (
                (0.0, 0.0) if self.prepared[d].abstains else (self.slot[d], self.driver[d])
            )
            out.append(
                SimpleNamespace(rungs={RUNG: SimpleNamespace(slot_loss=slot, driver_loss=driver)})
            )
        return out


def _unit(offering: int, calibration: int = 50, slot: float = 0.0, driver: float = 0.0):
    """`calibration` days of which the first `offering` offer, each with the given losses."""
    prepared = [_day(d < offering) for d in range(calibration)]
    return prepared, _Cache(prepared, [slot] * calibration, [driver] * calibration)


def _audited(offering, **losses):
    prepared, cache = _unit(offering, **losses)
    found = gate_calibrate.calibrate_rung(cache, list(range(len(prepared))), RUNG)
    return found["beta_hat"], found["abstain"], found["reason"]


def _every_request(monkeypatch, offering, calibration=50, slot_threshold=np.inf):
    """The every-request route on days whose hour thresholds are all `slot_threshold` (a negative
    one loses at every scale above 0) and whose driver loss is 0."""
    monkeypatch.setattr(
        gate_calibrate, "hour_thresholds", lambda policy, rows: [slot_threshold] * 3
    )
    monkeypatch.setattr(gate_calibrate, "true_rows", lambda day: None)
    monkeypatch.setattr(
        gate_calibrate, "every_scale_terms", lambda p: ((np.zeros(0), np.zeros(0), 0.0), None)
    )
    prepared = [_day(d < offering) for d in range(calibration)]
    test = list(range(calibration))
    found = gate_calibrate.every_request(prepared, list(range(calibration)), test)
    return found["beta_used"], found["abstain"], found["reason"], found


# --- 1. the regression ---------------------------------------------------------------------------


def test_few_offering_days_that_always_fail_do_not_calibrate_audited_set():
    # 4 of 50 calibration days offer and each loses its whole day at every scale; the other 46
    # abstain. Over all 50 days the slot bound (4 + 1) / 51 <= 0.10 holds at every scale.
    beta, abstain, reason = _audited(offering=4, slot=1.0, driver=0.0)
    assert beta == 0.0 and abstain
    assert reason.startswith("abstains: 4 of 50 calibration days offer;")


def test_few_offering_days_that_always_fail_do_not_calibrate_every_request(monkeypatch):
    used, abstain, reason, found = _every_request(monkeypatch, offering=4, slot_threshold=-1.0)
    assert used == 0.0 and abstain
    assert reason.startswith("abstains: 4 of 50 calibration days offer;")
    assert found["offering_calibration_days"] == 4


def test_enough_offering_days_that_always_fail_get_scale_zero_not_a_diluted_mean():
    # 25 offering days that fail at every scale: the bound fails at every scale above 0.
    beta, abstain, reason = _audited(offering=25, calibration=200, slot=1.0)
    assert beta == 0.0 and abstain
    assert reason.startswith("no scale is safe: 25 of 200 calibration days offer;")


# --- 2. the abstain rule counts offering days ----------------------------------------------------


def test_10_offering_days_among_50_do_not_calibrate_the_driver_loss():
    beta, abstain, reason = _audited(offering=10)
    assert beta == 0.0 and abstain
    assert reason.startswith("abstains: 10 of 50 calibration days offer;")
    assert "driver loss at alpha 0.05 needs n >= 19" in reason
    assert crc.min_days(crc.ALPHA_DRIVER) == 19


def test_19_offering_days_among_50_calibrate():
    beta, abstain, reason = _audited(offering=19)
    assert beta == crc.GRID[-1] and not abstain and reason == ""


def test_the_every_request_route_counts_offering_days_too(monkeypatch):
    used, abstain, reason, _ = _every_request(monkeypatch, offering=10)
    assert used == 0.0 and abstain
    assert reason.startswith("abstains: 10 of 50 calibration days offer;")
    assert "needs n >= 19" in reason
    used, abstain, reason, found = _every_request(monkeypatch, offering=19)
    assert used == crc.GRID[-1] and not abstain and reason == ""
    assert found["offering_calibration_days"] == 19


# --- 3. test results by hand ---------------------------------------------------------------------


def test_test_results_are_means_over_offering_days_by_hand():
    # Five test days; days 0, 2 and 4 offer (values 4, 6 and 10 kWh at scale 1), days 1 and 3
    # abstain. Scale 0.5. Slot losses of the offering days 0, 0.25, 0.5; driver losses 0.02, 0,
    # 0.04. Offers 2, 0, 3, 0, 5 kWh.
    prepared = [_day(True, 4.0), _day(False), _day(True, 6.0), _day(False), _day(True, 10.0)]
    cache = _Cache(prepared, [0.0, 0.0, 0.25, 0.0, 0.5], [0.02, 0.0, 0.0, 0.0, 0.04])
    ceilings = [gate_day.Ceilings(20.0, 10.0, "optimal", 0.0)] * 5
    summary, rows = gate_calibrate.score_test(cache, prepared, ceilings, list(range(5)), RUNG, 0.5)
    assert summary["days"] == 5
    assert summary["offering_days"] == 3
    assert summary["failing_days"] == 2
    assert summary["mean_offer_kwh_per_day"] == pytest.approx(10.0 / 5)
    assert summary["mean_offer_kwh_per_offering_day"] == pytest.approx(10.0 / 3)
    assert summary["mean_slot_loss_offering_days"] == pytest.approx(0.75 / 3)
    assert summary["mean_driver_loss_offering_days"] == pytest.approx(0.06 / 3)
    assert summary["offer_kwh_per_car_per_day"] == pytest.approx(10.0 / 15)
    assert summary["share_of_ceiling"]["best-causal"]["ratio_of_sums"] == pytest.approx(10.0 / 50)
    assert [r["offer_kwh"] for r in rows] == pytest.approx([2.0, 0.0, 3.0, 0.0, 5.0])
    # no loss field is a mean over all days under a plain name
    assert "mean_slot_loss" not in summary and "mean_driver_loss" not in summary


def test_failing_days_count_slot_loss_not_driver_loss():
    # four offering test days; slot loss above 0 on two of them, driver loss above 0 on three
    prepared = [_day(True) for _ in range(4)]
    cache = _Cache(prepared, [0.0, 0.25, 0.5, 0.0], [0.1, 0.0, 0.2, 0.3])
    ceilings = [gate_day.Ceilings(20.0, 10.0, "optimal", 0.0)] * 4
    summary, _ = gate_calibrate.score_test(cache, prepared, ceilings, [0, 1, 2, 3], RUNG, 0.5)
    assert summary["offering_days"] == 4
    assert summary["failing_days"] == 2


def test_at_scale_zero_nothing_is_offered_so_there_is_no_loss_to_average():
    prepared = [_day(True, 4.0), _day(False), _day(True, 6.0)]
    cache = _Cache(prepared, [1.0, 0.0, 1.0], [0.5, 0.0, 0.5])  # losses it would have, if offered
    ceilings = [gate_day.Ceilings(20.0, 10.0, "optimal", 0.0)] * 3
    summary, _ = gate_calibrate.score_test(cache, prepared, ceilings, [0, 1, 2], RUNG, 0.0)
    assert summary["offering_days"] == 2  # days with a box at scale 1, whatever the scale
    assert summary["failing_days"] is None
    assert summary["mean_slot_loss_offering_days"] is None
    assert summary["mean_driver_loss_offering_days"] is None
    assert summary["mean_offer_kwh_per_day"] == 0.0


def test_no_offering_test_day_has_no_loss_mean():
    prepared = [_day(False), _day(False)]
    cache = _Cache(prepared, [0.0, 0.0], [0.0, 0.0])
    ceilings = [gate_day.Ceilings(20.0, 10.0, "optimal", 0.0)] * 2
    summary, _ = gate_calibrate.score_test(cache, prepared, ceilings, [0, 1], RUNG, 0.5)
    assert summary["offering_days"] == summary["failing_days"] == 0
    assert summary["mean_slot_loss_offering_days"] is None
    assert summary["mean_driver_loss_offering_days"] is None
    assert summary["mean_offer_kwh_per_offering_day"] is None
    assert summary["mean_offer_kwh_per_day"] == 0.0


def test_every_request_test_results_are_over_offering_test_days(monkeypatch):
    # 30 calibration days that all offer and never lose, so the scale is the top of the grid; then
    # the same thresholds on the test days: the negative threshold of the first test day loses
    # whole, the others do not.
    calibration = [_day(True) for _ in range(30)]
    test = [_day(True), _day(True), _day(False)]
    prepared = calibration + test
    thresholds = {id(test[0]): [-1.0] * 3}

    monkeypatch.setattr(
        gate_calibrate,
        "hour_thresholds",
        lambda policy, rows: thresholds.get(id(rows), [np.inf] * 3),
    )
    monkeypatch.setattr(gate_calibrate, "true_rows", lambda day: day)
    monkeypatch.setattr(
        gate_calibrate, "every_scale_terms", lambda p: ((np.zeros(0), np.zeros(0), 0.0), None)
    )
    for p in prepared:
        p.day = p  # `true_rows(p.day)` is then the stand-in itself
    found = gate_calibrate.every_request(prepared, list(range(30)), [30, 31, 32])
    assert found["beta_used"] == crc.GRID[-1]
    assert found["test_offering_days"] == 2
    assert found["test_failing_days"] == 1
    assert found["test_mean_slot_loss_offering_days"] == pytest.approx(0.5)
    assert found["test_mean_driver_loss_bound_offering_days"] == 0.0


# --- 4. the summary shows the counts -------------------------------------------------------------


def _rung(scale, offering, days, failing, reason="", offer=0.81):
    return {
        "beta_hat": scale,
        "reason": reason,
        "test": {
            "days": days,
            "offering_days": offering,
            "failing_days": failing,
            "mean_offer_kwh_per_day": offer,
            "share_of_ceiling": {
                "hindsight-free-decision-hour": {"ratio_of_sums": 0.1},
                "best-causal": {"ratio_of_sums": 0.1},
            },
        },
    }


def _document(matrix, main, unit, calibrated, oracle):
    def entry(rung):
        return {"units": {unit: {"rungs": {matrix.gate_rung: rung}}}}

    return {
        "trial": False,
        "partial": False,
        "t5": {"breaks_a": 0, "breaks_b": 0, "day_paths_checked": 1},
        "timeouts": {},
        "failed_configurations": [],
        "calibration_rule": {"n_calibration": 100, "calibration_half_below": 200},
        "units": {
            unit: {
                "calibratable": True,
                "status": "calibratable",
                "counting_days": 192,
                "calibration_days_used": 96,
                "test_days": 96,
                "at_least_100_counting_days": True,
            }
        },
        "configurations": {main.name: entry(calibrated), main.oracle_partner: entry(oracle)},
        "gate": {"by_pair": {}, "kappa_selected_per_unit": {}},
    }


def _offer_line(calibrated, oracle):
    matrix = gate_config.load(TOP / "gate" / "matrix.json")
    main = next(
        c for c in matrix.configurations if c.family == "main" and c.departure == "empirical"
    )
    unit = matrix.gate_units[1]
    text = "\n".join(
        gate_summary.summary_lines(_document(matrix, main, unit, calibrated, oracle), matrix)
    )
    line = next(
        row
        for row in text.splitlines()
        if row.lstrip().startswith(main.departure_seen) and unit in row
    )
    return line, text


def test_the_summary_line_of_an_offer_shows_scale_offering_and_failing_days():
    line, text = _offer_line(_rung(0.3, 27, 192, 1), _rung(1.0, 103, 192, 0))
    assert "scale 0.3, 27/192 offering, 1 failing" in line
    assert "scale 1, 103/192 offering, 0 failing" in line
    assert "scale, offering days/test days, failing days" in text
    assert "whose box at scale 1 has value above zero" in text
    assert "test days with an offer above zero" not in text


def test_a_scale_of_zero_is_not_shown_as_a_safe_offer_and_the_cause_is_told():
    too_few = "abstains: 11 of 50 calibration days offer; too few days (n = 11): needs n >= 19"
    unsafe = "no scale is safe: 30 of 50 calibration days offer; the bound fails everywhere"
    line, _ = _offer_line(
        _rung(0.0, 11, 192, None, too_few, 0.0), _rung(0.0, 11, 192, None, unsafe, 0.0)
    )
    assert "scale 0 (abstains: 11 of 50 calibration days offer;" in line
    assert "scale 0 (no scale is safe: 30 of 50 calibration days offer;" in line
    assert "11/192 offering, no offer made" in line
    assert "failing" not in line


def test_the_gate_outcome_lines_show_scale_offering_and_failing_days():
    row = {
        "calibrated_share": 0.12,
        "ratio": 2.0,
        "offering_days": 27,
        "failing_days": 1,
        "test_days": 192,
        "scale": 0.3,
    }
    assert "(scale 0.3, 27/192 offering, 1 failing)" in gate_summary._decision_text(
        {
            "ceiling": "c",
            "ratio_rule_on": "r",
            "go": False,
            "n_pass_share": 0,
            "n_pass_both": 0,
            "units": {"u": row},
        }
    )
    row = dict(row, scale=0.0, failing_days=None)
    assert "(scale 0, 27/192 offering, no offer made)" in gate_summary._decision_text(
        {
            "ceiling": "c",
            "ratio_rule_on": "r",
            "go": False,
            "n_pass_share": 0,
            "n_pass_both": 0,
            "units": {"u": row},
        }
    )

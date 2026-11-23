"""Counts of cars that other readings of the truth rules would change, on a made-up fleet."""

import pandas as pd
import pytest
from test_truth import DAY, build, sessions

from fleetreserve import truth_readings


def _fleet():
    return [
        # User u: a 6 kW session two days before, a 4 kW session the day before, and today's
        # 2 kW session.
        {
            "id": "u-earlier",
            "user": "u",
            "plug_in": "2019-04-29 08:00",
            "plug_out": "2019-04-29 10:00",
            "energy": 12.0,
            "charge_end": "2019-04-29 10:00",
        },
        {
            "id": "u-before",
            "user": "u",
            "plug_in": "2019-04-30 08:00",
            "plug_out": "2019-04-30 10:00",
            "energy": 8.0,
            "charge_end": "2019-04-30 10:00",
        },
        {
            "id": "u-today",
            "user": "u",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 17:00",
            "energy": 4.0,
            "charge_end": f"{DAY} 10:00",
        },
        # User w: only today's session, so it has no history ended before τ.
        {
            "id": "w-today",
            "user": "w",
            "plug_in": f"{DAY} 09:00",
            "plug_out": f"{DAY} 17:00",
            "energy": 3.0,
            "charge_end": f"{DAY} 10:00",
        },
        {"id": "anonymous", "plug_in": f"{DAY} 09:30", "plug_out": f"{DAY} 17:00"},
        # User x: 10 kWh charged from 08:00 to 13:00, 2 kW.
        {
            "id": "x-today",
            "user": "x",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 17:00",
            "energy": 10.0,
            "charge_end": f"{DAY} 13:00",
        },
    ]


def test_counts_of_changed_cars():
    rows = _fleet()
    cars, _ = build(rows)
    table = sessions(rows)
    counts = truth_readings.count_changes(cars, table, limit_kw=19.2, bound_by_charge_fields=False)
    assert counts["cars_on_counting_days"] == 4
    # Main rates: u 6 kW (12 kWh over 2 h), w 3 kW (3 kWh over the hour 09:00 to 10:00), x 2 kW,
    # anonymous 6.66. History ended before τ: u keeps 6 kW (the highest of its two earlier
    # sessions, not the latest); w and x have none and fall to 6.66: two cars change.
    assert counts["rate_history_ended_before_tau"] == 2
    # Over plug-in to plug-out: u's best is 12/2 = 6 kW (unchanged), w's is 3/8 kW and x's
    # 10/9 kW: two changes.
    assert counts["rate_average_over_plug_in_to_plug_out"] == 2
    # Bounding by the charging end changes nothing here (all charging ends at or after τ).
    assert counts["energy_before_charge_bound_switched"] == 0
    # At the car's rate instead of 6.66 kW: x draws 2 h x 2 kW = 4 kWh before τ instead of all
    # 10 kWh; u (2 h x 6 kW, capped at its 4 kWh), w (capped at its 3 kWh) and the anonymous car
    # (rate 6.66) do not change.
    assert counts["energy_before_at_rate_not_nominal_cap"] == 1
    # The build report describes every reading it counts.
    assert set(truth_readings.READINGS) == set(counts) - {"cars_on_counting_days"}


def test_history_before_tau_takes_the_running_highest():
    rows = _fleet()
    cars, _ = build(rows)
    table = sessions(rows)
    from fleetreserve.truth import session_averages

    average, _ = session_averages(table, 19.2)
    rows_for_cars = table.set_index("session_id").loc[list(cars["session_id"])].reset_index()
    rows_for_cars["tau"] = cars["tau"].array
    before = truth_readings.rate_history_before(rows_for_cars, table, average)
    by_id = dict(zip(cars["session_id"], before, strict=True))
    assert by_id["u-today"] == pytest.approx(6.0)
    assert by_id["w-today"] == pytest.approx(6.66)
    assert by_id["anonymous"] == pytest.approx(6.66)
    assert isinstance(rows_for_cars["tau"].dtype, pd.DatetimeTZDtype)

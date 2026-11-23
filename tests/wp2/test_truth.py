"""Truth tables (plan sections 1 to 3) on made-up session tables, with hand-worked numbers."""

import numpy as np
import pandas as pd
import pytest
from fake_raw import TEST_WINDOW

from fleetreserve import truth
from fleetreserve.sessions import COLUMNS
from fleetreserve.windows import Window

ZONE = "America/Los_Angeles"
DAY = "2019-05-01"
TAU = pd.Timestamp(f"{DAY} 10:00", tz=ZONE)


def _t(text: str | None) -> pd.Timestamp:
    return pd.NaT if text is None else pd.Timestamp(text, tz=ZONE)


def sessions(rows: list[dict]) -> pd.DataFrame:
    """Rows with keys id, plug_in, plug_out (local text) and optional others."""
    table = []
    for row in rows:
        table.append(
            {
                "site": row.get("site", "s"),
                "role": "train",
                "session_id": row["id"],
                "user_id": row.get("user"),
                "station": None,
                "plug_in": _t(row["plug_in"]),
                "plug_out": _t(row["plug_out"]),
                "charge_start": _t(row.get("charge_start")),
                "charge_end": _t(row.get("charge_end")),
                "energy_kwh": row.get("energy", 5.0),
                "max_power_kw": np.nan,
                "declared_departure": pd.NaT,
                "declared_energy_kwh": np.nan,
                "nominal_kw": row.get("nominal", 6.66),
                "nominal_rule": "source cap",
            }
        )
    frame = pd.DataFrame(table)
    for name in truth_time_columns():
        frame[name] = pd.to_datetime(frame[name], utc=True).dt.tz_convert(ZONE)
    return frame[list(COLUMNS)]


def truth_time_columns():
    return ("plug_in", "plug_out", "charge_start", "charge_end", "declared_departure")


def build(rows, days=(DAY,), window=TEST_WINDOW, bound=False, limit=19.2):
    dates = [pd.Timestamp(day).date() for day in days]
    return truth.build(
        sessions(rows),
        window,
        ZONE,
        unit="u",
        days=dates,
        limit_kw=limit,
        bound_by_charge_fields=bound,
    )


def test_plugged_in_at_tau_boundaries():
    cars, days = build(
        [
            {"id": "plugs-in-at-tau", "plug_in": f"{DAY} 10:00", "plug_out": f"{DAY} 15:00"},
            {"id": "plugs-out-at-tau", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 10:00"},
            {"id": "one-second-late", "plug_in": f"{DAY} 10:00:01", "plug_out": f"{DAY} 15:00"},
            {
                "id": "leaves-one-second-after",
                "plug_in": f"{DAY} 08:00",
                "plug_out": f"{DAY} 10:00:01",
            },
        ]
    )
    assert sorted(cars["session_id"]) == ["leaves-one-second-after", "plugs-in-at-tau"]
    assert list(days["cars"]) == [2]


@pytest.mark.parametrize(
    ("plug_out", "depart_hour", "in_window"),
    [
        ("10:59:59", 0, True),  # leaves inside the decision hour
        ("11:00:00", 1, True),  # exactly one hour after τ
        ("12:59:00", 2, True),  # rounded down, not to the nearest hour
        ("16:00:00", 6, True),  # leaves during the last service hour (16:00 to 17:00)
        ("17:00:00", 7, False),  # stays through S
    ],
)
def test_departure_hour_is_rounded_down(plug_out, depart_hour, in_window):
    cars, _ = build([{"id": "a", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} {plug_out}"}])
    assert cars.loc[0, "depart_hour"] == depart_hour
    assert bool(cars.loc[0, "in_window_departure"]) is in_window
    assert cars.loc[0, "plug_out"] == _t(f"{DAY} {plug_out}")


def test_room_hand_worked():
    # Nominal cap 6.66 kW, plug-in 08:30, τ 10:00: 1.5 hours at 6.66 kW = 9.99 kWh drawn before
    # τ. A 15 kWh session keeps room 15 - 9.99 = 5.01 kWh; an 8 kWh session drew all 8 before τ
    # (the bound by session energy) and keeps room 0.
    cars, _ = build(
        [
            {"id": "big", "plug_in": f"{DAY} 08:30", "plug_out": f"{DAY} 17:00", "energy": 15.0},
            {"id": "small", "plug_in": f"{DAY} 08:30", "plug_out": f"{DAY} 17:00", "energy": 8.0},
        ]
    )
    by_id = cars.set_index("session_id")
    assert by_id.loc["big", "energy_before_kwh"] == pytest.approx(9.99)
    assert by_id.loc["big", "room_kwh"] == pytest.approx(5.01)
    assert by_id.loc["small", "energy_before_kwh"] == pytest.approx(8.0)
    assert by_id.loc["small", "room_kwh"] == 0.0


def test_room_bounded_by_charge_fields_hand_worked():
    # Plug-in 08:00, charging recorded 09:00 to 10:30, nominal 6.1 kW, τ 10:00. Bounded: only
    # the hour of charging before τ counts, 1 x 6.1 = 6.1 kWh, room 9.5 - 6.1 = 3.4.
    # Unbounded, two hours at 6.1 kW would draw all 9.5 kWh. A car with no charging time
    # recorded is not bounded and is marked so.
    rows = [
        {
            "id": "charged",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 16:30",
            "energy": 9.5,
            "nominal": 6.1,
            "charge_start": f"{DAY} 09:00",
            "charge_end": f"{DAY} 10:30",
        },
        {
            "id": "no-fields",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 16:30",
            "energy": 9.5,
            "nominal": 6.1,
        },
    ]
    bounded, _ = build(rows, bound=True)
    plain, _ = build(rows, bound=False)
    by_id = bounded.set_index("session_id")
    assert by_id.loc["charged", "energy_before_kwh"] == pytest.approx(6.1)
    assert by_id.loc["charged", "room_kwh"] == pytest.approx(3.4)
    assert by_id.loc["charged", "energy_before_rule"] == truth.BEFORE_RULE_BOUNDED
    assert by_id.loc["no-fields", "energy_before_kwh"] == pytest.approx(9.5)
    assert by_id.loc["no-fields", "energy_before_rule"] == truth.BEFORE_RULE_UNBOUNDED
    assert plain.set_index("session_id").loc["charged", "room_kwh"] == 0.0


@pytest.mark.parametrize(
    ("plug_in", "charge_start", "charge_end", "energy"),
    [
        # 3 seconds of recorded charging the day before τ, 12 kWh: at the nominal cap the
        # charging time carries only 0.0055 kWh, yet the charging has ended.
        ("2019-04-30 08:00", "2019-04-30 08:00:00", "2019-04-30 08:00:03", 12.0),
        # Same day: charging recorded 08:00 to 08:30, ended before τ 10:00; half an hour at
        # 6.1 kW carries 3.05 of the 9.5 kWh.
        (f"{DAY} 07:30", f"{DAY} 08:00", f"{DAY} 08:30", 9.5),
        # Charging ends exactly at τ.
        (f"{DAY} 07:30", f"{DAY} 09:00", f"{DAY} 10:00", 9.5),
    ],
)
def test_charging_ended_by_tau_leaves_no_room(plug_in, charge_start, charge_end, energy):
    # NREL ("bounded by its charge fields"): a car whose recorded charging ended at or before τ
    # drew its whole session energy before τ, so its room at τ is 0.
    rows = [
        {
            "id": "ended",
            "plug_in": plug_in,
            "plug_out": f"{DAY} 16:30",
            "energy": energy,
            "nominal": 6.1,
            "charge_start": charge_start,
            "charge_end": charge_end,
        }
    ]
    cars, _ = build(rows, bound=True)
    row = cars.set_index("date").loc[pd.Timestamp(DAY).date()]
    assert row["energy_before_kwh"] == pytest.approx(energy)
    assert row["room_kwh"] == 0.0
    assert row["energy_before_rule"] == truth.BEFORE_RULE_ENDED


def test_rate_is_the_users_highest_average_with_fallback():
    rows = [
        # User u: 2 kWh charged in half an hour (4 kW) on this day; on another day 6 kWh over
        # one hour of charging (6 kW). Both of u's cars get 6 kW.
        {
            "id": "u-today",
            "user": "u",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 17:00",
            "energy": 2.0,
            "charge_end": f"{DAY} 08:30",
        },
        {
            "id": "u-later",
            "user": "u",
            "plug_in": "2019-05-02 08:00",
            "plug_out": "2019-05-02 17:00",
            "energy": 6.0,
            "charge_end": "2019-05-02 09:00",
        },
        # User v: charging end recorded one minute after plug-in with 5 kWh (300 kW, above the
        # physical limit): its average falls back to plug-in to plug-out, 5 kWh / 5 h = 1 kW.
        {
            "id": "v",
            "user": "v",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 13:00",
            "energy": 5.0,
            "charge_end": f"{DAY} 08:01",
        },
        # No user id: nominal cap.
        {"id": "anonymous", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00"},
    ]
    cars, _ = build(rows, days=(DAY, "2019-05-02"))
    by_key = cars.set_index(["session_id"])
    assert by_key.loc["u-today", "rate_kw"] == pytest.approx(6.0)
    assert by_key.loc["u-later", "rate_kw"] == pytest.approx(6.0)
    assert by_key.loc["u-today", "rate_average_kind"] == "charging time"
    assert by_key.loc["v", "rate_kw"] == pytest.approx(1.0)
    assert by_key.loc["v", "rate_average_kind"] == "plug-in to plug-out"
    assert by_key.loc["anonymous", "rate_kw"] == 6.66
    assert by_key.loc["anonymous", "rate_rule"] == truth.RATE_FALLBACK


def test_charging_end_after_plug_out_is_clipped_to_plug_out():
    rows = [
        {
            "id": "a",
            "user": "a",
            "plug_in": f"{DAY} 08:00",
            "plug_out": f"{DAY} 12:00",
            "energy": 8.0,
            "charge_end": f"{DAY} 13:00",
        }
    ]
    cars, _ = build(rows)
    assert cars.loc[0, "rate_kw"] == pytest.approx(2.0)  # 8 kWh over 4 hours, not 5


def test_counting_day_needs_three_cars():
    two = [{"id": f"c{n}", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00"} for n in range(2)]
    three = [*two, {"id": "c2", "plug_in": f"{DAY} 09:00", "plug_out": f"{DAY} 17:00"}]
    cars, days = build(two)
    assert not days.loc[0, "counts"] and not cars["day_counts"].any()
    cars, days = build(three)
    assert days.loc[0, "counts"] and cars["day_counts"].all()
    assert list(cars["cars_in_day"]) == [3, 3, 3]


def test_a_car_staying_two_days_is_in_both_fleets_and_tau_follows_daylight_saving():
    # 10 March 2019 is the spring change in Los Angeles: τ is 10:00 PDT (UTC-7) that day, PST
    # (UTC-8) the day before.
    rows = [{"id": "stay", "plug_in": "2019-03-09 09:00", "plug_out": "2019-03-10 12:30"}]
    cars, days = build(rows, days=("2019-03-09", "2019-03-10"))
    assert list(cars["date"].astype(str)) == ["2019-03-09", "2019-03-10"]
    assert [t.isoformat() for t in days["tau"]] == [
        "2019-03-09T10:00:00-08:00",
        "2019-03-10T10:00:00-07:00",
    ]
    # 2019-03-09 10:00 PST to 2019-03-10 12:30 PDT is 25.5 hours of real time.
    assert list(cars["depart_hour"]) == [25, 2]
    # The second day's energy before τ counts from the plug-in on the first day.
    assert cars.loc[1, "energy_before_kwh"] == pytest.approx(5.0)


def test_room_fits_compares_room_with_rate_times_whole_hours():
    rows = [
        {
            "id": "a",
            "user": "a",
            "plug_in": f"{DAY} 09:30",
            "plug_out": f"{DAY} 11:30",
            "energy": 8.0,
            "charge_end": f"{DAY} 11:30",
        }
    ]
    cars, _ = build(rows)
    # Rate 8 kWh over 2 h = 4 kW; before τ 0.5 h x 6.66 = 3.33; room 4.67; one whole hour
    # before departure gives 4 kWh < 4.67: the room does not fit.
    assert cars.loc[0, "room_kwh"] == pytest.approx(4.67)
    assert not cars.loc[0, "room_fits"]


def test_unsorted_decision_times_are_refused():
    table = sessions([{"id": "a", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00"}])
    taus = pd.DatetimeIndex([TAU, TAU - pd.Timedelta(days=1)])
    with pytest.raises(ValueError):
        truth.pairs_at_tau(table, taus)


def test_norway_style_window_puts_tau_at_its_clock_hour():
    window = Window(tau_hour=21, service_hours=8, rule="test")
    rows = [{"id": "a", "plug_in": f"{DAY} 18:00", "plug_out": "2019-05-02 07:00"}]
    cars, days = build(rows, window=window)
    assert days.loc[0, "tau"].isoformat() == "2019-05-01T21:00:00-07:00"
    assert cars.loc[0, "depart_hour"] == 10
    assert not cars.loc[0, "in_window_departure"]

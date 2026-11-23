"""Clock, unit and midnight checks (plan section 3, WP2) on made-up sessions, including a clock
that ignores daylight saving, which the checks must report as a finding."""

import numpy as np
import pandas as pd

from fleetreserve import data_checks

ZONE = "America/Denver"


def _weekday_arrivals(first: str, last: str, local_hour: float, zone: str = ZONE):
    """Plug-ins at a fixed local clock time on every weekday, as naive local wall clock."""
    days = [d for d in pd.date_range(first, last, freq="D") if d.day_of_week < 5]
    stamps = []
    for day in days:
        for minute in range(0, 60, 2):  # 30 cars a day spread over one hour
            stamps.append(day + pd.Timedelta(hours=local_hour, minutes=minute))
    return pd.Series(pd.DatetimeIndex(stamps))


def _table(plug_in: pd.Series, energy: float = 10.0, hours: float = 8.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plug_in": plug_in,
            "plug_out": plug_in + pd.Timedelta(hours=hours),
            "energy_kwh": energy,
            "nominal_kw": 6.6,
            "max_power_kw": np.nan,
        }
    )


def test_change_days_of_a_year():
    assert data_checks.change_days("2019-01-01", "2019-12-31", ZONE) == [
        (pd.Timestamp("2019-03-10").date(), "spring"),
        (pd.Timestamp("2019-11-03").date(), "autumn"),
    ]


def test_right_clock_shows_no_shift_and_no_finding():
    naive = _weekday_arrivals("2019-02-01", "2019-12-15", 8.0)
    table = _table(naive.dt.tz_localize(ZONE))
    rows = data_checks.dst_shift(table, ZONE)
    assert [row["direction"] for row in rows] == ["spring", "autumn"]
    assert all(abs(row["shift_h"]) < 0.1 for row in rows)
    assert data_checks.findings("x", {"dst_shift": {"site": rows + rows}}) == []


def test_clock_without_daylight_saving_is_a_finding():
    # The cars arrive at 08:00 local clock time, but the file was written in standard time all
    # year (UTC-7), and is read as local time with daylight saving: summer arrivals look one
    # hour early, so the median moves by about -1 h in spring and +1 h in autumn.
    true_local = _weekday_arrivals("2018-02-01", "2019-12-15", 8.0).dt.tz_localize(ZONE)
    written = true_local.dt.tz_convert("Etc/GMT+7").dt.tz_localize(None)
    misread = written.dt.tz_localize(ZONE, nonexistent="NaT", ambiguous="NaT").dropna()
    rows = data_checks.dst_shift(_table(misread), ZONE)
    spring = [row["shift_h"] for row in rows if row["direction"] == "spring"]
    autumn = [row["shift_h"] for row in rows if row["direction"] == "autumn"]
    assert len(spring) == 2 and len(autumn) == 2
    assert all(abs(shift + 1.0) < 0.1 for shift in spring)
    assert all(abs(shift - 1.0) < 0.1 for shift in autumn)
    found = data_checks.findings("nrel", {"dst_shift": {"nrel": rows}})
    assert len(found) == 2 and all("contradicted" in text for text in found)


def test_gap_and_doubled_hour_counts():
    naive = pd.Series(
        pd.to_datetime(
            [
                "2019-03-10 02:30:00",  # in the gap
                "2019-03-10 02:45:00",  # in the gap
                "2019-03-10 02:50:00",  # in the gap
                "2019-11-03 01:30:00",  # in the doubled hour
                "2019-01-15 02:10:00",  # an ordinary day at 02:00
                "2019-01-16 09:00:00",
            ]
        )
    )
    numbers = data_checks.gap_and_doubled(naive, ZONE)
    assert numbers["changes"] == 2
    assert numbers["in_gap_hour"] == 3 and numbers["in_doubled_hour"] == 1
    found = data_checks.findings("x", {"gap_and_doubled": {"times": numbers}})
    assert len(found) == 1 and "gap hour" in found[0]


def test_units_and_a_unit_finding():
    plug_in = pd.Series(pd.date_range("2019-05-01 08:00", periods=10, freq="D", tz=ZONE))
    good = data_checks.units(_table(plug_in, energy=10.0, hours=5.0), 19.2)
    assert good["energy_kwh_median"] == 10.0 and good["average_kw_median"] == 2.0
    assert good["share_average_above_nominal"] == 0.0
    assert data_checks.findings("x", {"units": {"s": good}}) == []
    in_watt_hours = data_checks.units(_table(plug_in, energy=10000.0, hours=5.0), 19.2)
    found = data_checks.findings("x", {"units": {"s": in_watt_hours}})
    assert any("units contradicted" in text for text in found)


def test_midnight_crossings():
    plug_in = pd.Series(
        pd.to_datetime(["2019-05-01 08:00", "2019-05-01 20:00", "2019-05-01 20:00"]).tz_localize(
            ZONE
        )
    )
    table = _table(plug_in)
    table.loc[2, "plug_out"] = table.loc[2, "plug_in"] + pd.Timedelta(hours=30)
    numbers = data_checks.midnight(table)
    assert numbers["share_ending_on_a_later_date"] == round(2 / 3, 6)
    assert numbers["share_longer_than_24h"] == round(1 / 3, 6)
    assert numbers["longest_h"] == 30.0


def test_drop_findings():
    dropped = pd.DataFrame({"reason": ["no plug-out"] * 6 + ["fast-charging station"]})
    shares = data_checks.drop_shares(100, dropped)
    assert shares == {"fast-charging station": 0.01, "no plug-out": 0.06}
    found = data_checks.findings("x", {"drop_shares": {"s": shares}})
    assert len(found) == 2
    small = data_checks.drop_shares(1000, dropped.iloc[:6])
    assert data_checks.findings("x", {"drop_shares": {"s": small}}) == []


def test_count_against_plan_finding():
    assert data_checks.findings("acn", {"counts_against_plan": {"a": (10, 10)}}) == []
    assert len(data_checks.findings("acn", {"counts_against_plan": {"a": (9, 10)}})) == 1


def test_stated_duration_read_two_ways():
    raw = pd.DataFrame(
        {
            # Spans the autumn change in Oslo (27 October 2019): 10 hours of wall clock, 11 of
            # real time.
            "plugin_time": pd.to_datetime(["2019-10-26 20:00", "2019-05-01 18:00"]),
            "plugout_time": pd.to_datetime(["2019-10-27 06:00", "2019-05-02 07:00"]),
            "connection_time": [11.0, 13.0],
        }
    )
    numbers = data_checks.stated_duration(raw, "Europe/Oslo")
    assert numbers["sessions_across_a_change"] == 1
    assert numbers["across_change_stated_equals_elapsed"] == 1
    assert numbers["stated_equals_wall_clock"] == 1

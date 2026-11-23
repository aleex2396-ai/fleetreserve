"""Windows (plan section 2): the Norway rule on made-up occupancy profiles, the count of cars
plugged in at an instant at its boundaries, and local clock hours on daylight-saving days."""

import numpy as np
import pandas as pd
import pytest

from fleetreserve.windows import (
    WORKPLACE,
    local_instants,
    occupancy_profile,
    peak_hour,
    plugged_in_counts,
    site_days,
    window_from_profile,
)


def _profile(medians_by_day: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(medians_by_day, columns=range(24))


def test_workplace_window():
    assert (WORKPLACE.tau_hour, WORKPLACE.service_hours) == (8, 3)


def test_rule_on_an_evening_peak():
    # Medians: 0 until 16:00, then 1, 2, 4, 4, 4, 3, 2, 1 for 16:00 to 23:00. The peak (4)
    # starts at 18:00, so τ = 17:00; half the peak is 2, so S = 18:00 to 22:00 (5 hours: the
    # medians 4, 4, 4, 3, 2), stopping at 23:00 where the median 1 is below 2.
    day = [0.0] * 16 + [1, 2, 4, 4, 4, 3, 2, 1]
    window, detail = window_from_profile(_profile([day, day, day]))
    assert (window.tau_hour, window.service_hours) == (17, 5)
    assert detail["peak_hour"] == 18 and detail["peak_median"] == 4.0


def test_median_over_days_not_mean():
    # Three days; at 20:00 the counts are 0, 0, 9 (median 0, mean 3). At 21:00 they are 2, 2, 2.
    days = [[0.0] * 24 for _ in range(3)]
    days[2][20] = 9
    for day in days:
        day[21] = 2
    window, detail = window_from_profile(_profile(days))
    assert detail["peak_hour"] == 21
    assert window.tau_hour == 20


def test_plateau_crossing_midnight_starts_at_its_first_hour_and_s_is_capped_at_8():
    # Median 5 from 22:00 to 05:00 (a circular run of 8 hours), 3 at 06:00 and 07:00, 0 after.
    day = [5.0] * 6 + [3.0, 3.0] + [0.0] * 14 + [5.0, 5.0]
    window, detail = window_from_profile(_profile([day]))
    assert detail["peak_hour"] == 22
    assert (window.tau_hour, window.service_hours) == (21, 8)
    # The other reading (earliest clock hour at the peak) would put τ at 23:00.
    assert detail["alternative_tau_hour_earliest_peak"] == 23


def test_equal_runs_go_to_the_earliest_start():
    median = np.zeros(24)
    median[[3, 4, 15, 16]] = 2.0
    assert peak_hour(median) == 3
    median[17] = 2.0  # the later run is now longer
    assert peak_hour(median) == 15


def test_counts_at_an_instant_include_plug_in_and_exclude_plug_out():
    zone = "Europe/Oslo"
    instant = pd.Timestamp("2020-01-10 22:00", tz=zone)
    plug_in = pd.Series(
        pd.to_datetime(["2020-01-10 22:00", "2020-01-10 20:00", "2020-01-10 21:00"]).tz_localize(
            zone
        )
    )
    plug_out = pd.Series(
        pd.to_datetime(
            ["2020-01-11 07:00:00", "2020-01-10 22:00:00", "2020-01-10 22:00:01"]
        ).tz_localize(zone)
    )
    # In: the first (plugs in at the instant) and the third; out: the second (plugs out at it).
    assert list(plugged_in_counts(plug_in, plug_out, [instant])) == [2]


def test_local_hours_on_daylight_saving_days():
    days = pd.to_datetime(["2019-03-31", "2019-10-27"]).date
    at_two = local_instants(days, 2, "Europe/Oslo")
    # 02:00 does not exist on 31 March: moved forward to 03:00 CEST; on 27 October the first
    # 02:00 (CEST) is taken.
    assert [t.isoformat() for t in at_two] == [
        "2019-03-31T03:00:00+02:00",
        "2019-10-27T02:00:00+02:00",
    ]
    at_ten = local_instants(days, 10, "Europe/Oslo")
    assert [t.isoformat() for t in at_ten] == [
        "2019-03-31T10:00:00+02:00",
        "2019-10-27T10:00:00+01:00",
    ]


def test_profile_counts_cars_per_day_and_hour_over_all_days():
    zone = "Europe/Oslo"
    plug_in = pd.Series(pd.to_datetime(["2020-01-10 18:00", "2020-01-12 18:30"]).tz_localize(zone))
    plug_out = pd.Series(pd.to_datetime(["2020-01-11 07:00", "2020-01-13 07:00"]).tz_localize(zone))
    profile = occupancy_profile(plug_in, plug_out, zone)
    # Days from the first plug-in date to the last: 10, 11 and 12 January (11th has no plug-in).
    assert [str(day) for day in profile.index] == ["2020-01-10", "2020-01-11", "2020-01-12"]
    assert profile.loc[profile.index[0], 18] == 1
    assert profile.loc[profile.index[1], 6] == 1 and profile.loc[profile.index[1], 7] == 0
    assert profile.loc[profile.index[2], 18] == 0 and profile.loc[profile.index[2], 19] == 1
    assert site_days(plug_in) == list(profile.index)


@pytest.mark.parametrize("bad_hour", [7])
def test_service_stops_at_the_first_hour_below_half(bad_hour):
    day = [4.0] * 24
    day[bad_hour] = 1.0  # below half of 4
    day[0] = 4.0
    window, detail = window_from_profile(_profile([day]))
    # Every hour but 07:00 is at the peak: the run starts at 08:00, τ = 07:00, and S runs for 8
    # hours (the cap) since 08:00 to 15:00 are all at the peak.
    assert detail["peak_hour"] == 8
    assert (window.tau_hour, window.service_hours) == (7, 8)

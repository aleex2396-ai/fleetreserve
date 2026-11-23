"""Decision time τ and service hours S per site (plan section 2, "Windows").

A window is the clock hour of τ and the number of service hours. Hour t of a day is the hour
starting at τ + t hours (real time); hour 0 is the decision hour and the service hours are
1, ..., |S|.

- Workplaces (ACN, NREL, a.s.r.): one window, `WORKPLACE`, the single home of its value. The
  rule (plan section 2) is a choice among the pairs of τ in 7:00, ..., 12:00 local and |S| in
  2, 3, 4, 6 (S starts one hour after τ): the pair with the largest known-car ceiling
  (`known_car_ceiling`, hourly product, ρ = 0, main truth; a reading of the plan's V_caus as the
  best causal value with the decision-hour power fixed at τ, the gate's best-causal ceiling, not
  b(S) - p(S) with a free decision hour) summed over the counting days of
  Caltech and JPL, ties to the earlier τ and then the shorter S. `window_choice` makes the
  choice as a recorded run, and the pair it names is used unchanged at NREL and a.s.r.
  `WORKPLACE` holds the pair of the recorded run `20261112T015029Z-wp2-window-choice`, made at
  commit facb1e3: τ = 08:00 local, S = 09:00 to 12:00, three service hours (summed ceiling
  120,778.967 kWh over Caltech and JPL, no tie). The window used before the rule, τ = 10:00,
  S = 11:00 to 17:00, six service hours, was set without a rule and scored 15,530.768 kWh in
  the same run.
- Norway sites: from the site's occupancy profile, the number of cars plugged in at each whole
  clock hour of every day. Its median over days, per clock hour, peaks at some hour; τ is one
  hour before the peak, and S is the hours that follow τ, at most 8, while the median stays at
  least half the peak.

Readings of the plan's sentence, each named in the code below:
- "all days" are all calendar days from the site's first plug-in date to its last plug-in
  date, days without any session included (they count zero cars).
- A car is counted at an hour if it is plugged in at the hour's first instant (plug-in <= that
  instant < plug-out), the same rule as "plugged in at τ".
- "The peak": the clock hour where the longest run of hours at the highest median starts,
  reading the 24 hours as a circle (so a run may cross midnight). Ties between runs of equal
  length go to the earliest starting clock hour. The alternative, the earliest clock hour at the
  highest median, is returned too, for the report.
- "The next hours ... with median >= half the peak": consecutive hours from the peak hour on,
  stopping at the first hour below half the peak or after 8 hours.
- A local clock hour that does not exist on a day (the daylight-saving gap) is moved forward to
  the first existing instant; one that occurs twice is taken at its first occurrence.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_SERVICE_HOURS = 8


@dataclass(frozen=True)
class Window:
    tau_hour: int  # clock hour of τ, 0 to 23
    service_hours: int  # |S|; service hour t (1 to |S|) starts at τ + t hours
    rule: str


# The workplace window; see the module text for the rule that sets it.
WORKPLACE = Window(tau_hour=8, service_hours=3, rule="workplace: τ 08:00, S 09:00 to 12:00")


def local_instants(dates: pd.Series | list, hour: int, zone: str) -> pd.DatetimeIndex:
    """The instant `hour`:00 local on each date (see the module text for gap and doubled hour)."""
    naive = pd.DatetimeIndex(pd.to_datetime([str(day) for day in dates])) + pd.Timedelta(hours=hour)
    first = np.ones(len(naive), dtype=bool)  # True: the first of a doubled hour
    return naive.tz_localize(zone, nonexistent="shift_forward", ambiguous=first)


def utc_values(times) -> np.ndarray:
    """Timezone-aware times as UTC numpy values (microseconds)."""
    index = pd.DatetimeIndex(times).tz_convert("UTC").tz_localize(None)
    return index.to_numpy(dtype="datetime64[us]")


def site_days(plug_in: pd.Series) -> list:
    """All calendar days from the first plug-in date to the last plug-in date (a reading)."""
    first, last = plug_in.min().date(), plug_in.max().date()
    return list(pd.date_range(first, last, freq="D").date)


def plugged_in_counts(plug_in: pd.Series, plug_out: pd.Series, instants) -> np.ndarray:
    """Number of sessions with plug_in <= instant < plug_out, for each instant."""
    starts = np.sort(utc_values(plug_in))
    ends = np.sort(utc_values(plug_out))
    points = utc_values(instants)
    began = np.searchsorted(starts, points, side="right")
    ended = np.searchsorted(ends, points, side="right")
    return (began - ended).astype(int)


def occupancy_profile(plug_in: pd.Series, plug_out: pd.Series, zone: str) -> pd.DataFrame:
    """Cars plugged in at each whole clock hour (columns 0 to 23) of each site day (rows)."""
    days = site_days(plug_in)
    columns = {}
    for hour in range(24):
        columns[hour] = plugged_in_counts(plug_in, plug_out, local_instants(days, hour, zone))
    return pd.DataFrame(columns, index=pd.Index(days, name="date"))


def _runs_at_maximum(median: np.ndarray) -> list[tuple[int, int]]:
    """(start hour, length) of each circular run of hours at the highest median."""
    top = median == median.max()
    if top.all():
        return [(0, 24)]
    runs = []
    for hour in range(24):
        if top[hour] and not top[hour - 1]:
            length = 0
            while top[(hour + length) % 24]:
                length += 1
            runs.append((hour, length))
    return runs


def peak_hour(median: np.ndarray) -> int:
    runs = _runs_at_maximum(np.asarray(median, dtype=float))
    longest = max(length for _, length in runs)
    return min(start for start, length in runs if length == longest)


def window_from_profile(profile: pd.DataFrame) -> tuple[Window, dict]:
    """The Norway rule on an occupancy profile; also the medians and the alternative τ."""
    median = np.array([float(np.median(profile[hour])) for hour in range(24)])
    peak = peak_hour(median)
    half = median[peak] / 2.0
    service = 0
    while service < MAX_SERVICE_HOURS and median[(peak + service) % 24] >= half:
        service += 1
    earliest = int(np.flatnonzero(median == median.max())[0])
    window = Window(
        tau_hour=(peak - 1) % 24,
        service_hours=service,
        rule="Norway: τ one hour before the median peak, S while median >= half the peak",
    )
    detail = {
        "median_by_hour": [round(float(value), 6) for value in median],
        "peak_hour": int(peak),
        "peak_median": float(median[peak]),
        "days": len(profile),
        "alternative_tau_hour_earliest_peak": (earliest - 1) % 24,
    }
    return window, detail

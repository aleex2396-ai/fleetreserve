"""Checks of the clocks, units and midnight crossings that plan section 3 assumes (WP2).

Each check returns plain numbers; `findings` turns them into statements, by the rules below,
where an assumption of the plan is contradicted or a removal reason takes a large share of the
rows (a removal the plan itself asks for included). A finding is reported, never fixed here.

- Clock, daylight-saving shift: around every daylight-saving change in a site's span, the
  median local clock time of weekday plug-ins in the 14 days before and the 14 days after are
  compared. People arrive by the local clock, so with the right clock the median barely moves;
  a clock that ignores daylight saving moves it by about one hour. Finding: the median shift
  over the changes of one direction (spring or autumn) is at least 0.5 hour in size, with at
  least two changes measured (at least 20 plug-ins on each side).
- Clock, daylight-saving gap and doubled hour (sources written in local wall clock: NREL,
  Norway): written times inside the hour that does not exist on a spring change day, and
  inside the hour that occurs twice on an autumn change day, against the number expected from
  that clock hour on other days. Finding: at least 3 written times in the gap and at least half
  the expected number.
- Units: the energy per session (median, spread) and the average power, energy over plug-in to
  plug-out. Finding: median energy outside 1 to 100 kWh, or more than 1% of sessions above the
  physical limit.
- Midnight crossings: sessions whose plug-out falls on a later local date than their plug-in,
  and sessions longer than 24 hours. Reported, no finding rule.
- Rows removed: a removal reason that takes more than 5% of a source's rows, and any
  `fast-charging station` removal (the plan names one nominal cap per source), are findings.
"""

import numpy as np
import pandas as pd

from fleetreserve.sessions import hours_between

SIDE_DAYS = 14
MIN_EACH_SIDE = 20
SHIFT_FINDING_HOURS = 0.5
DROP_FINDING_SHARE = 0.05


def clock_hours(times: pd.Series) -> np.ndarray:
    """Local clock time of day in hours (10:30 is 10.5)."""
    return (times.dt.hour + times.dt.minute / 60.0 + times.dt.second / 3600.0).to_numpy(float)


def change_days(first, last, zone: str) -> list[tuple]:
    """(date, "spring" or "autumn") for each daylight-saving change from `first` to `last`."""
    days = pd.date_range(first, last, freq="D")
    noon = (days + pd.Timedelta(hours=12)).tz_localize(zone)
    offset = np.array([moment.utcoffset().total_seconds() for moment in noon])
    out = []
    for k in range(1, len(days)):
        if offset[k] != offset[k - 1]:
            out.append((days[k].date(), "spring" if offset[k] > offset[k - 1] else "autumn"))
    return out


def dst_shift(sessions: pd.DataFrame, zone: str) -> list[dict]:
    """Median weekday plug-in clock time before and after each daylight-saving change."""
    plug_in = sessions["plug_in"]
    weekday = plug_in.dt.day_of_week < 5
    dates = plug_in.dt.date
    hours = clock_hours(plug_in)
    rows = []
    for day, direction in change_days(dates.min(), dates.max(), zone):
        before = weekday & (dates >= day - pd.Timedelta(days=SIDE_DAYS)) & (dates < day)
        after = weekday & (dates >= day) & (dates < day + pd.Timedelta(days=SIDE_DAYS))
        n_before, n_after = int(before.sum()), int(after.sum())
        if min(n_before, n_after) < MIN_EACH_SIDE:
            continue
        median_before = float(np.median(hours[before.to_numpy()]))
        median_after = float(np.median(hours[after.to_numpy()]))
        rows.append(
            {
                "change": str(day),
                "direction": direction,
                "plug_ins_before": n_before,
                "plug_ins_after": n_after,
                "median_before_h": round(median_before, 3),
                "median_after_h": round(median_after, 3),
                "shift_h": round(median_after - median_before, 3),
            }
        )
    return rows


def gap_and_doubled(naive: pd.Series, zone: str) -> dict:
    """Written local times in the daylight-saving gap and doubled hour, against expectation."""
    naive = naive.dropna()
    changes = change_days(naive.min().date(), naive.max().date(), zone)
    if not changes:
        return {"changes": 0}
    dates = naive.dt.date
    probe = pd.DatetimeIndex([pd.Timestamp(str(day)) for day, _ in changes])
    # The changed hour is the clock hour where the offset moves on that day (02:00 in both zones
    # used here); found from the zone itself rather than assumed.
    hours = []
    for start in probe:
        steps = pd.date_range(start, start + pd.Timedelta(hours=23), freq="h")
        local = steps.tz_localize(zone, nonexistent="NaT", ambiguous="NaT")
        missing = np.flatnonzero(pd.isna(local))
        hours.append(int(steps[missing[0]].hour) if len(missing) else -1)
    in_hour = {}
    for (day, direction), hour in zip(changes, hours, strict=True):
        hit = int(((dates == day) & (naive.dt.hour == hour)).sum())
        in_hour.setdefault(direction, 0)
        in_hour[direction] += hit
    change_dates = {day for day, _ in changes}
    others = ~dates.isin(change_dates)
    n_days = max(len(set(dates[others])), 1)
    per_day = float((others & (naive.dt.hour == 2)).sum()) / n_days
    springs = sum(1 for _, direction in changes if direction == "spring")
    autumns = len(changes) - springs
    return {
        "changes": len(changes),
        "in_gap_hour": in_hour.get("spring", 0),
        "expected_in_gap_hour": round(per_day * springs, 2),
        "in_doubled_hour": in_hour.get("autumn", 0),
        "expected_in_doubled_hour_once": round(per_day * autumns, 2),
    }


def weekday_plug_in_hours(sessions: pd.DataFrame) -> list[int]:
    plug_in = sessions["plug_in"]
    hours = plug_in[plug_in.dt.day_of_week < 5].dt.hour
    return [int(n) for n in np.bincount(hours.to_numpy(), minlength=24)]


def units(sessions: pd.DataFrame, limit_kw: float) -> dict:
    energy = sessions["energy_kwh"].to_numpy(float)
    average = energy / hours_between(sessions["plug_out"], sessions["plug_in"])
    nominal = sessions["nominal_kw"].to_numpy(float)
    out = {
        "sessions": len(sessions),
        "energy_kwh_min": round(float(energy.min()), 4),
        "energy_kwh_p01": round(float(np.percentile(energy, 1)), 4),
        "energy_kwh_median": round(float(np.median(energy)), 4),
        "energy_kwh_p99": round(float(np.percentile(energy, 99)), 4),
        "energy_kwh_max": round(float(energy.max()), 4),
        "average_kw_median": round(float(np.median(average)), 4),
        "average_kw_max": round(float(average.max()), 4),
        "share_average_above_nominal": round(float(np.mean(average > nominal)), 6),
        "share_average_above_limit": round(float(np.mean(average > limit_kw)), 6),
        "limit_kw": limit_kw,
    }
    if sessions["max_power_kw"].notna().any():
        has = (
            sessions["max_power_kw"].notna()
            & sessions["charge_start"].notna()
            & sessions["charge_end"].notna()
        )
        part = sessions[has]
        hours = hours_between(part["charge_end"], part["charge_start"])
        positive = hours > 0
        charging_average = part["energy_kwh"].to_numpy(float)[positive] / hours[positive]
        peak = part["max_power_kw"].to_numpy(float)[positive]
        out["sessions_with_charging_time_and_peak"] = int(positive.sum())
        out["share_charging_average_above_peak_by_5pct"] = round(
            float(np.mean(charging_average > 1.05 * peak)), 6
        )
        ratio = part["energy_kwh"] / part["declared_energy_kwh"]
        out["energy_over_declared_median"] = round(float(ratio[np.isfinite(ratio)].median()), 4)
    return out


def midnight(sessions: pd.DataFrame) -> dict:
    later_day = sessions["plug_out"].dt.date > sessions["plug_in"].dt.date
    hours = hours_between(sessions["plug_out"], sessions["plug_in"])
    return {
        "share_ending_on_a_later_date": round(float(later_day.mean()), 6),
        "share_longer_than_24h": round(float(np.mean(hours > 24)), 6),
        "longest_h": round(float(hours.max()), 3),
    }


def stated_duration(raw: pd.DataFrame, zone: str) -> dict:
    """Norway: the file's `connection_time` against plug-out minus plug-in, read two ways."""
    has = raw["plugout_time"].notna() & raw["connection_time"].notna()
    part = raw[has]
    wall = hours_between(part["plugout_time"], part["plugin_time"])
    plug_in = part["plugin_time"].dt.tz_localize(zone, nonexistent="NaT", ambiguous="NaT")
    plug_out = part["plugout_time"].dt.tz_localize(zone, nonexistent="NaT", ambiguous="NaT")
    real = hours_between(plug_out, plug_in)
    stated = part["connection_time"].to_numpy(float)
    across = np.abs(wall - real) > 0.5
    tolerance = 0.02
    return {
        "sessions": int(has.sum()),
        "stated_equals_wall_clock": int(np.sum(np.abs(stated - wall) < tolerance)),
        "sessions_across_a_change": int(across.sum()),
        "across_change_stated_equals_wall_clock": int(
            np.sum(np.abs(stated - wall)[across] < tolerance)
        ),
        "across_change_stated_equals_elapsed": int(
            np.sum(np.abs(stated - real)[across] < tolerance)
        ),
    }


def drop_shares(rows_read: int, dropped: pd.DataFrame) -> dict:
    counts = dropped["reason"].value_counts().sort_index()
    return {reason: round(int(n) / max(rows_read, 1), 6) for reason, n in counts.items()}


def findings(source: str, result: dict) -> list[str]:
    """Statements of the finding rules of the module text."""
    out = []
    for site, rows in result.get("dst_shift", {}).items():
        for direction in ("spring", "autumn"):
            shifts = [row["shift_h"] for row in rows if row["direction"] == direction]
            if len(shifts) >= 2 and abs(float(np.median(shifts))) >= SHIFT_FINDING_HOURS:
                out.append(
                    f"{source} {site}: weekday plug-ins move by {np.median(shifts):+.2f} h at "
                    f"the {direction} daylight-saving changes ({len(shifts)} changes): the "
                    "assumed clock is contradicted"
                )
    gap = result.get("gap_and_doubled", {})
    for what, numbers in gap.items():
        n_in, expected = numbers.get("in_gap_hour", 0), numbers.get("expected_in_gap_hour", 0)
        if n_in >= 3 and n_in >= expected / 2:
            out.append(
                f"{source} {what}: {n_in} written times fall in the daylight-saving gap hour "
                f"({expected} expected without daylight saving): the assumed clock is "
                "contradicted"
            )
    for site, numbers in result.get("units", {}).items():
        median = numbers["energy_kwh_median"]
        if not 1.0 <= median <= 100.0:
            out.append(f"{source} {site}: median energy {median} kWh: units contradicted")
        if numbers["share_average_above_limit"] > 0.01:
            out.append(
                f"{source} {site}: {numbers['share_average_above_limit']:.1%} of sessions "
                "average above the physical limit"
            )
    for site, shares in result.get("drop_shares", {}).items():
        for reason, share in shares.items():
            if share > DROP_FINDING_SHARE or reason == "fast-charging station":
                out.append(f"{source} {site}: {share:.1%} of rows removed as '{reason}'")
    for site, (found, stated) in result.get("counts_against_plan", {}).items():
        if found != stated:
            out.append(f"{source} {site}: {found} rows read, the plan or paper states {stated}")
    return out

"""Truth tables: the cars plugged in at τ on each day, with departure, rate and room (plan 1-3).

For one site's session table (or NREL's pooled table) and its window, every calendar day of
the site gets τ (its window's clock hour, local time) and its fleet: the sessions with
plug-in <= τ < plug-out. The day counts if at least COUNTING_CARS cars are in it (plan 1).

Per car and day (all times local, timezone-aware):
- `depart_hour` = floor((plug-out - τ) / 1 hour) (plan 2, "plug-out hour, rounded down"): the
  car can draw power in hour t, the hour starting at τ + t hours, only if t < depart_hour. A
  car plugged in at τ that leaves within the decision hour has depart_hour 0. The exact
  plug-out is kept.
- `in_window_departure`: depart_hour <= |S|, the last service hour (plan 1).
- `rate_kw`, main rule (plan 3, "user's highest observed average"): the highest session average
  power among all sessions of the same user id at the same site, any date, this one included
  (a reading: the truth is hindsight, so it uses every session). A session's average is its
  energy over its charging time where the source records the end of charging (ACN
  `doneChargingTime`; NREL `start_charge` to `termin_charge`), clipped to plug-in and plug-out;
  otherwise, or when that average is above the source's physical limit, or the charging time
  is not positive, its energy over plug-in to plug-out. A car without a user id has no
  history; its rate is the nominal cap (plan 3, "cars without history", share reported).
- `energy_before_kwh` (plan 3, main rule "pre-decision charging uncontrolled from arrival at
  nominal rate"): min(energy, nominal cap x hours from plug-in to τ). For NREL ("bounded by its
  charge fields") the hours are those of plug-in to τ that also lie inside the recorded
  charging time, `start_charge` to `termin_charge`, where recorded; and when the recorded
  charging ended (`termin_charge`) at or before τ, the energy before τ is the whole session
  energy, so the room is 0, however short the recorded charging time (both a reading of
  "NREL bounded by its charge fields"). ACN would be bounded by its per-session time series
  (`/ts`), which are not held here; its `doneChargingTime` does not bound it.
- Setting (plan 3). `main` is the rule above. `held` is the second full setting, reported apart
  and never pooled with the main one: no charging before τ, so `energy_before_kwh` is 0 for
  every car, its rule is `BEFORE_RULE_HELD`, and room is the session energy. Windows, sessions,
  rates, counting days, departures and declared columns are those of the main setting.
- `room_kwh` = energy - energy_before_kwh (plan 1, "Room e_i").
- `room_fits` = room <= rate x depart_hour: whether the room can be drawn at the rate in the
  whole hours before departure. Reported, never used to remove a car.
- `declared_at`, `declared_departure_h` (plan section 3, "Declared energy, departure: ACN, NREL
  only"): the departure the driver declared, as hours from τ (may be fractional, and negative
  when the declared time had passed by τ), and when it was stated (the session table's
  `declared_at`: ACN `modifiedAt` of the first app entry, NREL `request_entry_time`). Rule: both
  are written only when the statement was made at or before τ and both times are recorded;
  otherwise both are missing, so a statement made after τ never reaches the table. A session
  table without `declared_at` (Norway) gives missing values throughout.
- `plug_out_from_declared`: the session's plug-out was taken from the declared departure (the
  session table's flag; false when the table has none). Such a car's true departure is its
  declared one, so a declared-departure forecast of it would be scored against itself.
"""

import numpy as np
import pandas as pd

from fleetreserve.sessions import DECLARED_AT, FROM_DECLARED, hours_between
from fleetreserve.windows import Window, local_instants, utc_values

COUNTING_CARS = 3
MAIN = "main"
HELD = "held"
SETTINGS = (MAIN, HELD)
RATE_HISTORY = "user's highest average, all sessions"
RATE_FALLBACK = "nominal cap: no user id"
BEFORE_RULE = "from plug-in at the nominal cap"
BEFORE_RULE_BOUNDED = "from plug-in at the nominal cap, inside the charging time"
BEFORE_RULE_UNBOUNDED = "from plug-in at the nominal cap, no charging time recorded"
BEFORE_RULE_ENDED = "whole session energy: recorded charging ended at or before τ"
BEFORE_RULE_HELD = "held: no charging before τ"
CAR_COLUMNS = (
    "unit",
    "site",
    "date",
    "tau",
    "day_counts",
    "cars_in_day",
    "session_id",
    "user_id",
    "station",
    "plug_in",
    "plug_out",
    "depart_hour",
    "in_window_departure",
    "energy_kwh",
    "nominal_kw",
    "nominal_rule",
    "rate_kw",
    "rate_rule",
    "rate_average_kind",
    "energy_before_kwh",
    "energy_before_rule",
    "room_kwh",
    "room_fits",
    "declared_at",
    "declared_departure_h",
    "plug_out_from_declared",
)
DAY_COLUMNS = ("unit", "site", "date", "tau", "service_hours", "cars", "counts")


def session_averages(sessions: pd.DataFrame, limit_kw: float, use_charging_time: bool = True):
    """Each session's average power (kW) and which time it was taken over."""
    connection = sessions["energy_kwh"].to_numpy() / hours_between(
        sessions["plug_out"], sessions["plug_in"]
    )
    kind = np.full(len(sessions), "plug-in to plug-out", dtype=object)
    if not use_charging_time:
        return connection, kind
    start = sessions["charge_start"].fillna(sessions["plug_in"])
    start = start.where(start > sessions["plug_in"], sessions["plug_in"])
    end = sessions["charge_end"].where(sessions["charge_end"] < sessions["plug_out"])
    end = end.fillna(sessions["plug_out"])  # a missing end is excluded below
    charging_hours = hours_between(end, start)
    with np.errstate(divide="ignore", invalid="ignore"):
        charging = sessions["energy_kwh"].to_numpy() / charging_hours
    usable = sessions["charge_end"].notna().to_numpy() & (charging_hours > 0)
    usable &= charging <= limit_kw
    average = np.where(usable, charging, connection)
    kind = np.where(usable, "charging time", kind)
    return average, kind


def user_rates(sessions: pd.DataFrame, average: np.ndarray, kind: np.ndarray) -> pd.DataFrame:
    """Per session: the user's highest average over all their sessions at the site, the kind of
    time it was taken over, and the rule; the nominal cap for a session without a user id."""
    frame = pd.DataFrame(
        {
            "site": sessions["site"].to_numpy(),
            "user": sessions["user_id"].to_numpy(),
            "average": average,
            "kind": kind,
        }
    )
    known = frame["user"].notna()
    best = (
        frame[known]
        .sort_values(["site", "user", "average"], ascending=[True, True, False], kind="stable")
        .drop_duplicates(["site", "user"])
        .set_index(["site", "user"])
    )
    keys = pd.MultiIndex.from_arrays([frame["site"], frame["user"]])
    rate = best["average"].reindex(keys).to_numpy()
    rate_kind = best["kind"].reindex(keys).to_numpy()
    nominal = sessions["nominal_kw"].to_numpy()
    has = known.to_numpy()
    return pd.DataFrame(
        {
            "rate_kw": np.where(has, rate, nominal),
            "rate_rule": np.where(has, RATE_HISTORY, RATE_FALLBACK),
            "rate_average_kind": np.where(has, rate_kind, "none"),
        },
        index=sessions.index,
    )


def day_taus(days: list, window: Window, zone: str) -> pd.DatetimeIndex:
    return local_instants(days, window.tau_hour, zone)


def pairs_at_tau(sessions: pd.DataFrame, taus: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(session position, day position) for every session plugged in at a day's τ
    (plug-in <= τ < plug-out). `taus` must be increasing."""
    points = utc_values(taus)
    if len(points) > 1 and not (np.diff(points) > np.timedelta64(0, "us")).all():
        raise ValueError("decision times must be increasing")
    first = np.searchsorted(points, utc_values(sessions["plug_in"]), side="left")
    stop = np.searchsorted(points, utc_values(sessions["plug_out"]), side="left")
    count = np.maximum(stop - first, 0)
    session_pos = np.repeat(np.arange(len(sessions)), count)
    offsets = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    day_pos = np.repeat(first, count) + offsets
    return session_pos, day_pos


def energy_before(
    cars: pd.DataFrame, tau: pd.Series, bound_by_charge_fields: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Energy drawn before τ at the nominal cap from plug-in (see the module text)."""
    start = cars["plug_in"]
    end = tau
    rule = np.full(len(cars), BEFORE_RULE, dtype=object)
    ended = np.zeros(len(cars), dtype=bool)
    if bound_by_charge_fields:
        has_start = cars["charge_start"].notna()
        has_end = cars["charge_end"].notna()
        start = start.where(~has_start | (cars["charge_start"] <= start), cars["charge_start"])
        end = end.where(~has_end | (cars["charge_end"] >= end), cars["charge_end"])
        recorded = (has_start | has_end).to_numpy()
        rule = np.where(recorded, BEFORE_RULE_BOUNDED, BEFORE_RULE_UNBOUNDED)
        ended = (has_end & (cars["charge_end"] <= tau.array)).to_numpy()
        rule = np.where(ended, BEFORE_RULE_ENDED, rule)
    hours = np.maximum(hours_between(end, start), 0.0)
    energy = cars["energy_kwh"].to_numpy()
    drawn = np.minimum(energy, cars["nominal_kw"].to_numpy() * hours)
    return np.where(ended, energy, drawn), rule


def declared_at_tau(cars: pd.DataFrame, tau: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """`declared_at` and `declared_departure_h` of each car (see the module text)."""
    if DECLARED_AT in cars.columns:
        stated = cars[DECLARED_AT]
    else:
        stated = pd.Series(pd.NaT, index=cars.index, dtype=tau.dtype)
    recorded = (stated.notna() & cars["declared_departure"].notna()).to_numpy()
    known = recorded & (utc_values(stated.where(recorded, tau)) <= utc_values(tau))
    hours = np.where(known, hours_between(cars["declared_departure"], tau), np.nan)
    return stated.where(known), hours


def build(
    sessions: pd.DataFrame,
    window: Window,
    zone: str,
    *,
    unit: str,
    days: list,
    limit_kw: float,
    bound_by_charge_fields: bool,
    rates: pd.DataFrame | None = None,
    setting: str = MAIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cars table and the days table of one site (or pooled NREL) over `days`.

    `setting` is `MAIN` or `HELD` (module text): it changes only the energy drawn before τ.

    `rates` (from `user_rates`) may be passed when the rate history must cover more sessions
    than the days given, for example a regime of a pooled site.
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, not {setting!r}")
    sessions = sessions.reset_index(drop=True)
    if rates is None:
        average, kind = session_averages(sessions, limit_kw)
        rates = user_rates(sessions, average, kind)
    rates = rates.reset_index(drop=True)
    days = sorted(days)
    taus = day_taus(days, window, zone)
    session_pos, day_pos = pairs_at_tau(sessions, taus)
    cars = sessions.iloc[session_pos].reset_index(drop=True)
    car_rates = rates.iloc[session_pos].reset_index(drop=True)
    tau = pd.Series(taus[day_pos], name="tau").dt.tz_convert(zone)
    cars["date"] = [days[k] for k in day_pos]
    cars["tau"] = tau
    cars["depart_hour"] = np.floor(hours_between(cars["plug_out"], tau)).astype(int)
    cars["in_window_departure"] = cars["depart_hour"] <= window.service_hours
    cars[["rate_kw", "rate_rule", "rate_average_kind"]] = car_rates.to_numpy()
    cars["rate_kw"] = cars["rate_kw"].astype(float)
    if setting == HELD:
        drawn = np.zeros(len(cars))
        rule = np.full(len(cars), BEFORE_RULE_HELD, dtype=object)
    else:
        drawn, rule = energy_before(cars, tau, bound_by_charge_fields)
    cars["energy_before_kwh"] = drawn
    cars["energy_before_rule"] = rule
    cars["room_kwh"] = cars["energy_kwh"] - cars["energy_before_kwh"]
    cars["room_fits"] = cars["room_kwh"] <= cars["rate_kw"] * cars["depart_hour"] + 1e-9
    cars["declared_at"], cars["declared_departure_h"] = declared_at_tau(cars, tau)
    flag = cars[FROM_DECLARED] if FROM_DECLARED in cars.columns else False
    cars["plug_out_from_declared"] = pd.Series(flag, index=cars.index).astype(bool)
    per_day = np.bincount(day_pos, minlength=len(days))
    cars["cars_in_day"] = per_day[day_pos]
    cars["day_counts"] = cars["cars_in_day"] >= COUNTING_CARS
    cars["unit"] = unit
    cars = cars.sort_values(["date", "session_id"], kind="stable").reset_index(drop=True)
    day_table = pd.DataFrame(
        {
            "unit": unit,
            "site": sessions["site"].iloc[0] if len(sessions) else "",
            "date": days,
            "tau": pd.Series(taus).dt.tz_convert(zone),
            "service_hours": window.service_hours,
            "cars": per_day,
            "counts": per_day >= COUNTING_CARS,
        }
    )
    return cars[list(CAR_COLUMNS)], day_table[list(DAY_COLUMNS)]

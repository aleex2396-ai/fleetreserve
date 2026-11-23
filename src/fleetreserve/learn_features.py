"""The feature and label table of the learned departure models (plan sections 7, 13 WP8).

`build_table(cars, rho=, events=None)` reads a cars table in the layout `days_from_truth` gives
it (`UnitDays.cars`: the truth table's columns `truth.CAR_COLUMNS` as text, plus the columns the
rules add) and returns one row per row of `cars`, in the same order: the key columns
(`forecast_model.KEY_COLUMNS`), `counting` (the row's day counts), the features
(`FEATURE_COLUMNS`, `EARLIER_COLUMNS`, `DECLARED_COLUMNS`, `TAIL_HISTORY_COLUMNS`, then any event
columns) and the labels (`LABEL_COLUMNS`). `feature_frame` keeps the keys, `counting` and the
features only: the frame a learned model's `predict` is given (`learn_run`). Every learned model
reads its inputs from this one function.

Features, each known at the decision time τ of the row's day:

- `unit` and `site` (also key columns);
- date parts of the row's date: `weekday` (Monday 0), `month`, `day_of_year`;
- `arrival_hour`: the clock hour of plug-in; `arrived_earlier_day`: plugged in on an earlier
  calendar day; `hours_since_plug_in`: hours from plug-in to τ;
- `fleet_at_tau`: the cars plugged in at τ on the row's site-day (`cars_in_day`);
- `user_known`: the row has a user id (an empty id, or the rate rule "nominal cap: no user id",
  means none);
- the user's history (`HISTORY_COLUMNS`): over the user's earlier sessions at the same site in
  the same cars table, each counted once, where an earlier session is one listed on a date before
  the row's date that had also plugged out at or before the row's τ (so a session plugged in over
  several days never enters its own history, and nothing of the row's own day enters it):
  `history_sessions` their count; `history_mean_stay_h`, `history_median_stay_h` the mean and
  median of plug-out minus plug-in (hours); `history_mean_departure_clock_h`,
  `history_median_departure_clock_h` those of the local clock time of plug-out (hours after local
  midnight, minutes as a fraction); `history_mean_energy_kwh` the mean session energy. Fill for no
  history (no user id, or no earlier session): count 0 and every other history column NaN.
  The history is read from the cars table given, so it holds only sessions plugged in at some
  day's τ, and for a unit split by date (NREL at 1 May 2019) only that unit's own dates;
- `declared_departure_h` (`DECLARED_COLUMNS`): the departure the driver declared, in hours from
  τ (negative when the declared time had passed by τ), as the cars table holds it (`truth`: only
  a statement made at or before τ is written). The rule is applied again here: the value is NaN
  unless the row's `declared_at` is a time at or before its τ, so a table holding a later
  statement cannot pass it on. NaN where nothing was declared (all Norwegian rows). It is not in
  `FEATURE_COLUMNS`, which the pooled LightGBM and CQR-GBM models read, so their inputs are
  unchanged; a model reads it by name;
- the lower tail of the user's history (`TAIL_HISTORY_COLUMNS`): over the same earlier sessions as
  `HISTORY_COLUMNS` (one selection, no second rule). The departure value of an earlier session is
  its exact departure in hours from the τ of the day it left: the smallest `departure_exact` over
  the days it is listed. With at least one earlier session: `history_min_departure_h`,
  `history_q10_departure_h`, `history_q25_departure_h` the minimum, 10% and 25% quantile (linear
  interpolation) of the departure values; `history_share_departure_under_3h` the share with a
  departure value below 3; `history_share_departure_before_6` the share that left before the next
  local 06:00 after their own day's τ (departure value below the hours from that day's τ clock
  time to the next 06:00 on the local clock of the τ text, a τ at exactly 06:00 taking 24; the
  clock change of a daylight-saving night is not applied); `history_same_weekday_sessions` the
  count of earlier sessions first listed on the row's weekday and
  `history_same_weekday_median_departure_h` the median of their departure values (count 0 and NaN
  when there is none);
  `history_last_departure_h` the departure value of the earlier session with the latest plug-out
  and `history_hours_since_last_plug_out` the hours from that plug-out to the row's τ (a tie in
  plug-out is broken by the later first-listed date, then the later session id);
  `history_last5_min_departure_h` the minimum departure value over the five earlier sessions with
  the latest plug-outs (the same order; all of them when fewer than five). Fill for no history:
  `history_same_weekday_sessions` 0 and every other tail column NaN. Not in `FEATURE_COLUMNS`
  either, so the pooled LightGBM and CQR-GBM models' inputs are unchanged; a model reads them by
  name. They are columns of the table built here, not of the cars table, so the lists of cars-table
  columns below do not carry them; their inputs (`plug_out`, `departure_exact` and `tau` of sessions
  that ended by τ) are the earlier sessions' outcomes named at the end of this text;
- the inputs of the earlier analysis's per-car models (plan section 12; `earlier_inputs`) that the
  columns above do not give (`EARLIER_COLUMNS`): `plug_in_clock_h`, the plug-in time in hours
  after the local midnight of the row's date, to the second, negative for a car plugged in on an
  earlier day (read from the local clock of the plug-in text, so the clock change of a
  daylight-saving night is not applied); and, over the same earlier sessions as
  `HISTORY_COLUMNS` (one selection, no second rule), `history_q10_stay_h` the 10% quantile of
  their stays (plug-out minus plug-in, hours), `history_q10_energy_kwh` and
  `history_median_energy_kwh` the 10% quantile and the median of their session energies (linear
  interpolation), and `history_days_since_last_plug_in` the days from the latest plug-in among
  them to the row's τ. Fill for no history: NaN in the four history columns. Not in
  `FEATURE_COLUMNS` or `TAIL_HISTORY_COLUMNS`, so the inputs of every model that reads those are
  unchanged; a model reads them by name;
- event columns (optional): `events` is a table with `site`, `date` and further columns whose
  names start with `event_`, at most one row per site and date; they are joined on (site, date),
  NaN where no event row matches. With `events=None` (the default) there are none.

Holidays: `holiday_flags(table, countries)` adds `public_holiday` from the pinned `holidays`
package (a package of the `learn` dependency group), imported inside that function, so this
module also loads in the default environment. `countries` maps each site to a country code and,
optionally, a subdivision; it is the caller's choice, as the plan does not list it. The flag
applies the package's calendars as they stand; plan section 7's dating of laws by enactment and
of one-off holidays by announcement is not applied here.

Context columns (`CONTEXT_COLUMNS`), each a function of the row's `site` and `date` alone and so
known at τ: `public_holiday`, the flag above with the committed mapping `SITE_COUNTRIES` (the
package's calendars as they stand; the dating of laws by enactment is not applied); `site_type`,
1 for a workplace and 0 otherwise, from the committed mapping `SITE_TYPES`, which follows plan
section 2's split of the windows ("workplaces (ACN, NREL, a.s.r.)"; Norway and Korea are the
other type). `context_columns(frame)` gives the two columns for the rows of any frame with `site`
and `date`, and `with_context(frame)` the frame with them. They are not columns of `build_table`
(which loads without the `holidays` package) and not in `FEATURE_COLUMNS` or
`TAIL_HISTORY_COLUMNS`, so the inputs of the models that read those are unchanged; a model that
reads them calls `with_context` on its fit table and on the frame it is given to forecast. A site
in neither mapping raises `ValueError`: no site gets a default.

The site mappings, by the `site` column of a cars table (plan section 3): the ACN sites
(`caltech`, `jpl`, `office001`: United States, California; workplaces); `nrel` (United States,
Colorado; workplace); the twelve Norwegian sites of every role (Norway; the other type); the
a.s.r. site `asr` (`load_asr.SITE`; Netherlands; workplace); and the five Korean portfolios,
under the site names `load_korea` writes (`load_korea.UNITS`; South Korea; the other type). The
data-set name `korea` is no site: a cars table of the Korean portfolios carries the portfolio
names.

Labels (never in `feature_frame`): `label_departure_h`, the exact departure in hours from τ
(`departure_exact`); `label_rate_kw`, the truth rate (`rate`); `label_room_kwh`, the room after
the rules (`room`, NaN for a car the rules leave out of the fleet); `label_need_kwh`,
`(1 - rho) * label_room_kwh`, the protected energy.

Columns of the cars table and whether they are known at τ (`KNOWN_AT_TAU`, `NOT_KNOWN_AT_TAU`;
every column is in exactly one of the two, a test checks it). Not known at τ, and so never a
feature of the row's own car:

- `plug_out`, `depart_hour`, `in_window_departure`, `departure_exact`, `depart`: the departure;
- `energy_kwh`: the session's total energy, known at plug-out;
- `rate_kw`, `rate`: the user's highest average power over all sessions at the site, this one and
  later ones included (truth's main rate rule is hindsight); `rate_average_kind`: which time that
  average was taken over;
- `energy_before_kwh`: `min(energy, nominal x hours from plug-in to τ)`, capped by the session
  total, so for a car that would be full by τ it gives the total away; for NREL the nominal is
  the session's highest measured power and the hours end at the recorded charging end. The part
  known at τ, the hours since plug-in, is the feature `hours_since_plug_in`; `energy_before_rule`:
  for NREL, whether charge times were recorded and whether charging ended by τ, read from fields
  written at the session's end;
- `nominal_kw`, `nominal_rule`: for NREL `max_charge_power`, the highest power measured over the
  whole session (`sessions`); for ACN and Norway a constant per source, which `site` carries;
- `room_kwh`, `room_table`, `room`, `room_fits`, `no_fit`: the room, session energy minus energy
  before τ;
- `outcome`, `in_fleet`: set from the departure hour and the room by the rules of
  `days_from_truth`;
- `plug_out_from_declared`: whether the plug-out was taken from the declared departure, decided
  from the plug-out fields recorded at the session's end.

Known at τ besides the keys and the inputs above: `declared_at` and `declared_departure_h`,
written only for a statement made at or before τ.

The user's earlier sessions are the only place outcomes enter the features: there `plug_out`,
`energy_kwh` and `departure_exact` of sessions that ended by τ are read, which are known at τ.
"""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.load_korea import UNITS as KOREA_SITES
from fleetreserve.truth import RATE_FALLBACK

HISTORY_COLUMNS = (
    "history_sessions",
    "history_mean_stay_h",
    "history_median_stay_h",
    "history_mean_departure_clock_h",
    "history_median_departure_clock_h",
    "history_mean_energy_kwh",
)
FEATURE_COLUMNS = (
    "weekday",
    "month",
    "day_of_year",
    "arrival_hour",
    "arrived_earlier_day",
    "hours_since_plug_in",
    "fleet_at_tau",
    "user_known",
    *HISTORY_COLUMNS,
)
DECLARED_COLUMNS = ("declared_departure_h",)
TAIL_HISTORY_COLUMNS = (
    "history_min_departure_h",
    "history_q10_departure_h",
    "history_q25_departure_h",
    "history_share_departure_under_3h",
    "history_share_departure_before_6",
    "history_same_weekday_sessions",
    "history_same_weekday_median_departure_h",
    "history_last_departure_h",
    "history_hours_since_last_plug_out",
    "history_last5_min_departure_h",
)
EARLIER_HISTORY_COLUMNS = (
    "history_q10_stay_h",
    "history_q10_energy_kwh",
    "history_median_energy_kwh",
    "history_days_since_last_plug_in",
)
EARLIER_COLUMNS = ("plug_in_clock_h", *EARLIER_HISTORY_COLUMNS)
SHORT_STAY_H = 3.0
EARLY_CLOCK_H = 6.0
LAST_SESSIONS = 5
LABEL_COLUMNS = ("label_departure_h", "label_rate_kw", "label_room_kwh", "label_need_kwh")
CONTEXT_COLUMNS = ("public_holiday", "site_type")
WORKPLACE, OTHER_SITE = 1, 0
NORWAY_SITES = (
    *("OSL_S", "ASK", "TRO", "BAR", "BOD"),  # plan section 3: train
    *("OSL_1", "BER", "KRO", "OSL_2", "BAR_2"),  # development
    *("OSL_T", "TRO_R"),  # the two sites kept for the final run
)
# site -> (country,) or (country, subdivision), as `holiday_flags` reads it (module text).
SITE_COUNTRIES: dict[str, tuple[str, ...]] = {
    "caltech": ("US", "CA"),
    "jpl": ("US", "CA"),
    "office001": ("US", "CA"),
    "nrel": ("US", "CO"),
    **dict.fromkeys(NORWAY_SITES, ("NO",)),
    "asr": ("NL",),
    **dict.fromkeys(KOREA_SITES, ("KR",)),
}
# site -> `WORKPLACE` or `OTHER_SITE` (plan section 2; module text).
SITE_TYPES: dict[str, int] = {
    "caltech": WORKPLACE,
    "jpl": WORKPLACE,
    "office001": WORKPLACE,
    "nrel": WORKPLACE,
    **dict.fromkeys(NORWAY_SITES, OTHER_SITE),
    "asr": WORKPLACE,
    **dict.fromkeys(KOREA_SITES, OTHER_SITE),
}
EVENT_PREFIX = "event_"
KNOWN_AT_TAU = (
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
    "rate_rule",
    "arrival_hour",
    "arrived_earlier_day",
    "counting",
    "declared_at",
    "declared_departure_h",
)
NOT_KNOWN_AT_TAU = (
    "plug_out",
    "depart_hour",
    "in_window_departure",
    "departure_exact",
    "depart",
    "energy_kwh",
    "rate_kw",
    "rate",
    "rate_average_kind",
    "energy_before_kwh",
    "energy_before_rule",
    "nominal_kw",
    "nominal_rule",
    "room_kwh",
    "room_table",
    "room",
    "room_fits",
    "no_fit",
    "outcome",
    "in_fleet",
    "plug_out_from_declared",
)


def _utc(text: pd.Series) -> pd.Series:
    return pd.to_datetime(text, utc=True, format="ISO8601")


def _seconds(moments: pd.Series) -> np.ndarray:
    """Seconds since 1970 (UTC) of timezone-aware times."""
    return ((moments - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def _clock_hours(text: pd.Series) -> np.ndarray:
    """The local clock time of an ISO 8601 text with offset, in hours after local midnight."""
    return (text.str[11:13].astype(int) + text.str[14:16].astype(int) / 60.0).to_numpy(float)


def _hours_to_next_early_clock(tau_clock: np.ndarray) -> np.ndarray:
    """Hours from a τ clock time (hours after local midnight) to the next local 06:00 after it."""
    wait = (EARLY_CLOCK_H - tau_clock) % 24.0
    return np.where(wait == 0.0, 24.0, wait)


def _history(cars: pd.DataFrame, tau: pd.Series, known: np.ndarray) -> pd.DataFrame:
    """The history columns of every row: `HISTORY_COLUMNS`, `TAIL_HISTORY_COLUMNS` and
    `EARLIER_HISTORY_COLUMNS` (module text), all from the one selection of earlier sessions."""
    n = len(cars)
    names = (*HISTORY_COLUMNS, *TAIL_HISTORY_COLUMNS, *EARLIER_HISTORY_COLUMNS)
    out = {name: np.full(n, math.nan) for name in names}
    out["history_sessions"] = np.zeros(n)
    out["history_same_weekday_sessions"] = np.zeros(n)
    plug_in = _utc(cars["plug_in"])
    plug_out = _utc(cars["plug_out"])
    frame = pd.DataFrame(
        {
            "site": cars["site"].to_numpy(str),
            "user": cars["user_id"].to_numpy(str),
            "session": cars["session_id"].to_numpy(str),
            "date": cars["date"].to_numpy(str),
            "tau": _seconds(tau),
            "in": _seconds(plug_in),
            "out": _seconds(plug_out),
            "stay": ((plug_out - plug_in) / pd.Timedelta(hours=1)).to_numpy(float),
            "clock": _clock_hours(cars["plug_out"]),
            "energy": cars["energy_kwh"].astype(float).to_numpy(),
            "departure": cars["departure_exact"].astype(float).to_numpy(),
            "tau_clock": _clock_hours(cars["tau"].astype(str)),
            "weekday": pd.to_datetime(cars["date"], format="%Y-%m-%d").dt.day_of_week.to_numpy(int),
            "known": known,
            "position": np.arange(n),
        }
    )
    for _, rows in frame[frame["known"]].groupby(["site", "user"], sort=False):
        # One entry per session: its first listed date; its plug-out, stay, clock and energy are
        # the same on every row of the session.
        sessions = rows.sort_values("date", kind="stable").drop_duplicates("session")
        first = sessions["date"].to_numpy(str)
        ended = sessions["out"].to_numpy(float)
        values = {
            "stay": sessions["stay"].to_numpy(float),
            "clock": sessions["clock"].to_numpy(float),
            "energy": sessions["energy"].to_numpy(float),
            "in": sessions["in"].to_numpy(float),
        }
        # The departure value of a session: its row with the smallest exact departure (the day it
        # left), and the 06:00 wait from that day's τ.
        left = (
            rows.sort_values(["departure", "date"], kind="stable")
            .drop_duplicates("session")
            .set_index("session")
            .loc[sessions["session"]]
        )
        departure = left["departure"].to_numpy(float)
        before_six = departure < _hours_to_next_early_clock(left["tau_clock"].to_numpy(float))
        first_weekday = pd.to_datetime(first, format="%Y-%m-%d").day_of_week.to_numpy(int)
        by_plug_out = np.empty(len(sessions), dtype=int)
        by_plug_out[np.lexsort((sessions["session"].to_numpy(str), first, ended))] = np.arange(
            len(sessions)
        )
        for date_text, row_tau, position, weekday in zip(
            rows["date"], rows["tau"], rows["position"], rows["weekday"], strict=True
        ):
            earlier = (first < date_text) & (ended <= row_tau)
            count = int(earlier.sum())
            out["history_sessions"][position] = count
            if count == 0:
                continue
            out["history_mean_stay_h"][position] = values["stay"][earlier].mean()
            out["history_median_stay_h"][position] = np.median(values["stay"][earlier])
            out["history_mean_departure_clock_h"][position] = values["clock"][earlier].mean()
            out["history_median_departure_clock_h"][position] = np.median(values["clock"][earlier])
            out["history_mean_energy_kwh"][position] = values["energy"][earlier].mean()
            d = departure[earlier]
            out["history_min_departure_h"][position] = d.min()
            out["history_q10_departure_h"][position], out["history_q25_departure_h"][position] = (
                np.quantile(d, [0.1, 0.25])
            )
            out["history_share_departure_under_3h"][position] = (d < SHORT_STAY_H).mean()
            out["history_share_departure_before_6"][position] = before_six[earlier].mean()
            same_day = earlier & (first_weekday == weekday)
            out["history_same_weekday_sessions"][position] = int(same_day.sum())
            if same_day.any():
                out["history_same_weekday_median_departure_h"][position] = np.median(
                    departure[same_day]
                )
            chosen = np.flatnonzero(earlier)
            chosen = chosen[np.argsort(by_plug_out[chosen])]
            out["history_last_departure_h"][position] = departure[chosen[-1]]
            out["history_hours_since_last_plug_out"][position] = (
                row_tau - ended[chosen[-1]]
            ) / 3600.0
            last_few = departure[chosen[-LAST_SESSIONS:]]
            out["history_last5_min_departure_h"][position] = last_few.min()
            stays, energies = values["stay"][earlier], values["energy"][earlier]
            out["history_q10_stay_h"][position] = np.quantile(stays, 0.1)
            out["history_q10_energy_kwh"][position] = np.quantile(energies, 0.1)
            out["history_median_energy_kwh"][position] = np.median(energies)
            out["history_days_since_last_plug_in"][position] = (
                row_tau - values["in"][earlier].max()
            ) / 86400.0
    return pd.DataFrame(out, index=cars.index)


def _plug_in_clock(cars: pd.DataFrame) -> np.ndarray:
    """`plug_in_clock_h` (module text): hours from the local midnight of the row's date to the
    plug-in, on the local clock of the plug-in text, to the second; the date and the time may be
    parted by "T" or a space."""
    text = cars["plug_in"].astype(str)
    clock = pd.to_datetime(text.str[:10] + " " + text.str[11:19], format="%Y-%m-%d %H:%M:%S")
    midnight = pd.to_datetime(cars["date"], format="%Y-%m-%d")
    return ((clock - midnight) / pd.Timedelta(hours=1)).to_numpy(float)


def _declared(cars: pd.DataFrame, tau: pd.Series) -> np.ndarray:
    """`declared_departure_h` where `declared_at` is at or before τ, else NaN (module text)."""
    text = cars["declared_departure_h"].astype(str)
    hours = pd.to_numeric(text.where(text != ""), errors="coerce").to_numpy(float)
    stated = _utc(cars["declared_at"].astype(str).replace("", None))
    by_tau = (stated.notna() & (stated <= tau)).to_numpy(bool)
    return np.where(by_tau, hours, math.nan)


def _events(table: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    extra = [c for c in events.columns if c not in ("site", "date")]
    if "site" not in events.columns or "date" not in events.columns:
        raise ValueError("an events table has the columns site and date")
    bad = [c for c in extra if not str(c).startswith(EVENT_PREFIX)]
    if bad:
        raise ValueError(f"event columns must start with {EVENT_PREFIX!r}, got {bad}")
    keyed = events.assign(site=events["site"].astype(str), date=events["date"].astype(str))
    if keyed.duplicated(["site", "date"]).any():
        raise ValueError("an events table has at most one row per site and date")
    joined = table[["site", "date"]].merge(keyed, on=["site", "date"], how="left")
    return joined[extra].set_axis(table.index)


def build_table(
    cars: pd.DataFrame, *, rho: float, events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Keys, `counting`, features and labels of every row of `cars` (module text)."""
    if not 0 <= rho < 1:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    cars = cars.reset_index(drop=True)
    tau = _utc(cars["tau"])
    dates = pd.to_datetime(cars["date"], format="%Y-%m-%d")
    user = cars["user_id"].astype(str)
    known = ((user != "") & (cars["rate_rule"] != RATE_FALLBACK)).to_numpy(bool)
    table = pd.DataFrame({c: cars[c].astype(str) for c in KEY_COLUMNS})
    table["counting"] = cars["counting"].astype(bool)
    table["weekday"] = dates.dt.day_of_week.to_numpy(int)
    table["month"] = dates.dt.month.to_numpy(int)
    table["day_of_year"] = dates.dt.day_of_year.to_numpy(int)
    table["arrival_hour"] = cars["arrival_hour"].astype(int)
    table["arrived_earlier_day"] = cars["arrived_earlier_day"].astype(bool)
    table["hours_since_plug_in"] = (tau - _utc(cars["plug_in"])) / pd.Timedelta(hours=1)
    table["fleet_at_tau"] = cars["cars_in_day"].astype(int)
    table["user_known"] = known
    history = _history(cars, tau, known)
    table = pd.concat([table, history[list(HISTORY_COLUMNS)]], axis=1)
    table["plug_in_clock_h"] = _plug_in_clock(cars)
    table = pd.concat([table, history[list(EARLIER_HISTORY_COLUMNS)]], axis=1)
    table["declared_departure_h"] = _declared(cars, tau)
    table = pd.concat([table, history[list(TAIL_HISTORY_COLUMNS)]], axis=1)
    if events is not None:
        table = pd.concat([table, _events(table, events)], axis=1)
    room = cars["room"].astype(float)
    table["label_departure_h"] = cars["departure_exact"].astype(float)
    table["label_rate_kw"] = cars["rate"].astype(float)
    table["label_room_kwh"] = room
    table["label_need_kwh"] = (1 - rho) * room
    return table


def feature_frame(table: pd.DataFrame) -> pd.DataFrame:
    """The table without its labels: keys, `counting`, features and event columns."""
    return table.drop(columns=[c for c in LABEL_COLUMNS if c in table.columns])


def holiday_flags(table: pd.DataFrame, countries: Mapping[str, tuple]) -> pd.DataFrame:
    """The table with `public_holiday`: whether the row's date is a public holiday of its site's
    country (and subdivision), by the pinned `holidays` package. `countries` maps a site to
    `(country,)` or `(country, subdivision)`; a site missing from it raises `ValueError`."""
    import holidays

    missing = sorted(set(table["site"]) - set(countries))
    if missing:
        raise ValueError(f"no country given for the sites {missing}")
    years = sorted({int(d[:4]) for d in table["date"]})
    calendars = {}
    for site, place in countries.items():
        country, subdivision = (place[0], place[1] if len(place) > 1 else None)
        calendars[site] = holidays.country_holidays(country, subdiv=subdivision, years=years)
    flags = [d in calendars[s] for s, d in zip(table["site"], table["date"], strict=True)]
    return table.assign(public_holiday=np.array(flags, dtype=bool))


def context_columns(
    frame: pd.DataFrame,
    countries: Mapping[str, tuple] | None = None,
    types: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """`CONTEXT_COLUMNS` for the rows of `frame`, read from its `site` and `date` only (module
    text). The mappings default to `SITE_COUNTRIES` and `SITE_TYPES`; a site missing from either
    raises `ValueError`."""
    countries = SITE_COUNTRIES if countries is None else countries
    types = SITE_TYPES if types is None else types
    keys = pd.DataFrame({"site": frame["site"].astype(str), "date": frame["date"].astype(str)})
    missing = sorted(set(keys["site"]) - set(types))
    if missing:
        raise ValueError(f"no site type given for the sites {missing}")
    flags = holiday_flags(keys, countries)["public_holiday"].to_numpy(bool)
    kinds = np.array([types[s] for s in keys["site"]], dtype=int)
    return pd.DataFrame({"public_holiday": flags, "site_type": kinds}, index=frame.index)


def with_context(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` with `CONTEXT_COLUMNS` (set anew from `site` and `date` when it has them)."""
    context = context_columns(frame)
    return frame.assign(**{name: context[name] for name in CONTEXT_COLUMNS})

"""Calendar inputs of the departure model, each known at the decision time τ (plan section 7).

Plan section 7: "calendars (pinned python-holidays, laws dated by enactment, one-off holidays by
announcement; school terms; OxCGRT; JPL 9/80 Fridays)". OxCGRT is `oxcgrt_inputs`. This module
gives two columns (`COLUMNS`) and states what is not built.

`public_holiday_dated` (`dated_holiday_flags`): 1 when the row's date is a public holiday of its
site's calendar in the pinned `holidays` package (a package of the `learn` dependency group,
imported inside the function, so this module loads in the default environment) AND that
holiday's law or announcement was made before the row's date. It differs from
`learn_features.public_holiday`, which applies the package's calendars as they stand. Calendars:
`learn_features.SITE_COUNTRIES` as it stands, which holds the five Korean portfolios under the
unit names of the Korean loader (`load_korea.UNITS`); a site missing from it raises
`ValueError`. Dating, per holiday the package lists:

- a holiday whose name is in `LONG_STANDING` for its country was set by a law in force before
  every data set's first day (the earliest is 1 Jan 2017), so it is known at τ of any day here;
- a holiday made by a law or announcement inside the data sets' spans is in `ENACTED`, keyed by
  (country, date), with the day the law was enacted or the decision announced; it counts only on
  a row whose date is after that day. Entries, each from a source read through a web search
  summary on 15 Nov 2026: Juneteenth, the federal law signed 17 Jun 2021 (its observed day
  18 Jun 2021 and the day itself, 19 Jun 2021); Colorado's Frances Xavier Cabrini Day, which
  replaced Columbus Day by HB20-1031, signed 20 Mar 2020 (5 Oct 2020, 4 Oct 2021); Korea's
  alternative holidays for Liberation Day, National Foundation Day and Hangul Day, extended to
  them by the amendment of the Regulations on Public Holidays of Government Offices approved by
  the Cabinet on 3 Aug 2021 (announced by the ministry on 15 Jul 2021; 16 Aug, 4 Oct and 11 Oct
  2021, 10 Oct 2022). The enactment day is used, the later of the two. [Alternative: the
  announcement day.];
- any other holiday the package lists raises `ValueError` naming it, so a package update or a
  new span cannot add an undated holiday.

The dated rule changes no flag on the data sets' days (every listed law came before its first
holiday); it is applied so that it would. Not covered: one-off closures the package does not
list (for example the United States' federal Christmas Eve closures of 2018 to 2020 by executive
order, which bind federal employees, not the sites' staff), and the Korean election days, listed
by the package and set by statute long before (the Public Official Election Act), so in
`LONG_STANDING`.

`nine_eighty_friday` (`nine_eighty_flags`): 1 on the Fridays off of JPL's 9/80 schedule (eight
nine-hour days and one eight-hour day over two weeks, every second Friday off), at the sites of
`NINE_EIGHTY_SITES` (`jpl`), 0 at every other row. The schedule is a fixed calendar published
to its staff in advance, so its value is known at τ. Which of the two alternating sets of
Fridays is off (`parity`: the Friday's `fortnight_parity`, 0 or 1) is not in any source this
project holds; it must be given. `nine_eighty_phase(cars)` estimates it from a cars table of
the training site `jpl`: the parity whose Fridays have the fewer cars plugged in at τ on average
over the counting days. Running it reads the JPL training days, a look at data that a recorded
run must make and `docs/prior-looks.md` must list; it has not been run. [Alternative: the parity
from JPL's own published calendar, not downloaded (no source allowed for it).] The schedule may
have been suspended or changed while the site's staff worked from home in 2020 and 2021; the
estimate does not detect that.

School terms are `school_inputs`.
"""

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd

COLUMNS = ("public_holiday_dated", "nine_eighty_friday")
NINE_EIGHTY_SITES = ("jpl",)
REFERENCE_FRIDAY = date(1970, 1, 2)  # a Friday; fortnights are counted from it
LANGUAGE = "en_US"
LONG_STANDING: dict[str, frozenset[str]] = {
    "US": frozenset(
        {
            "Cesar Chavez Day",
            "Cesar Chavez Day (observed)",
            "Christmas Day",
            "Christmas Day (observed)",
            "Day After Thanksgiving",
            "Independence Day",
            "Independence Day (observed)",
            "Labor Day",
            "Martin Luther King Jr. Day",
            "Memorial Day",
            "New Year's Day",
            "New Year's Day (observed)",
            "Presidents' Day",
            "Presidents' Day; Susan B. Anthony Day",
            "Susan B. Anthony Day",
            "Thanksgiving Day",
            "Veterans Day",
            "Veterans Day (observed)",
            "Washington-Lincoln Day",
        }
    ),
    "NO": frozenset(
        {
            "Ascension Day",
            "Christmas Day",
            "Constitution Day",
            "Easter Monday",
            "Easter Sunday",
            "Good Friday",
            "Labor Day",
            "Maundy Thursday",
            "New Year's Day",
            "Pentecost",
            "Pentecost Monday",
            "Second Day of Christmas",
        }
    ),
    "NL": frozenset(
        {
            "Ascension Day",
            "Christmas Day",
            "Easter Monday",
            "Easter Sunday",
            "Good Friday",
            "King's Day",
            "Liberation Day",
            "New Year's Day",
            "Pentecost",
            "Pentecost Monday",
            "Second Day of Christmas",
        }
    ),
    "KR": frozenset(
        {
            "Alternative holiday for Chuseok",
            "Buddha's Birthday",
            "Children's Day",
            "Christmas Day",
            "Chuseok",
            "Hangul Day",
            "Independence Movement Day",
            "Korean New Year",
            "Liberation Day",
            "Local Election Day",
            "Memorial Day",
            "National Foundation Day",
            "New Year's Day",
            "Presidential Election Day",
            "The day preceding Chuseok",
            "The day preceding Korean New Year",
            "The second day of Chuseok",
            "The second day of Korean New Year",
        }
    ),
}
_JUNETEENTH = date(2021, 6, 17)
_CABRINI = date(2020, 3, 20)
_KOREA_ALTERNATIVE = date(2021, 8, 3)
ENACTED: dict[tuple[str, date], date] = {
    ("US", date(2021, 6, 18)): _JUNETEENTH,
    ("US", date(2021, 6, 19)): _JUNETEENTH,
    ("US", date(2020, 10, 5)): _CABRINI,
    ("US", date(2021, 10, 4)): _CABRINI,
    ("KR", date(2021, 8, 16)): _KOREA_ALTERNATIVE,
    ("KR", date(2021, 10, 4)): _KOREA_ALTERNATIVE,
    ("KR", date(2021, 10, 11)): _KOREA_ALTERNATIVE,
    ("KR", date(2022, 10, 10)): _KOREA_ALTERNATIVE,
}


def holiday_known(country: str, day: date, name: str, row_date: date) -> bool:
    """Whether a holiday the package lists for `day` was law or announced before `row_date`
    (module text); raises `ValueError` for a holiday with no dating."""
    if (country, day) in ENACTED:
        return ENACTED[(country, day)] < row_date
    if all(part.strip() in LONG_STANDING.get(country, ()) for part in name.split(";")):
        return True
    raise ValueError(f"no dating for the holiday {name!r} of {country} on {day.isoformat()}")


def dated_holiday_flags(frame: pd.DataFrame, countries: Mapping[str, tuple] | None = None):
    """`public_holiday_dated` for the rows of `frame`, read from `site` and `date` only."""
    import holidays

    from fleetreserve.learn_features import SITE_COUNTRIES

    countries = SITE_COUNTRIES if countries is None else countries
    sites = frame["site"].astype(str).to_numpy()
    missing = sorted(set(sites) - set(countries))
    if missing:
        raise ValueError(f"no country given for the sites {missing}")
    days = pd.to_datetime(frame["date"].astype(str), format="%Y-%m-%d").dt.date.to_numpy()
    years = sorted({d.year for d in days})
    flags = np.zeros(len(frame), dtype=int)
    for site in pd.unique(sites):
        place = countries[site]
        calendar = holidays.country_holidays(
            place[0], subdiv=place[1] if len(place) > 1 else None, years=years, language=LANGUAGE
        )
        for row in np.flatnonzero(sites == site):
            day = days[row]
            if day in calendar:
                flags[row] = int(holiday_known(place[0], day, calendar[day], day))
    return pd.DataFrame({COLUMNS[0]: flags}, index=frame.index)


def fortnight_parity(days) -> np.ndarray:
    """Per date, the parity (0 or 1) of the whole weeks since `REFERENCE_FRIDAY`, so the Fridays
    of one parity are every second Friday, a fortnight apart."""
    offset = np.array([(d - REFERENCE_FRIDAY).days for d in days], dtype=int)
    return (offset // 7) % 2


def nine_eighty_flags(frame: pd.DataFrame, parity: int | None) -> pd.DataFrame:
    """`nine_eighty_friday` for the rows of `frame`, read from `site` and `date` only (module
    text); `parity` is the fortnight parity of the Fridays off, which must be given."""
    if parity not in (0, 1):
        raise ValueError("the parity of JPL's Fridays off is not set (module text)")
    days = pd.to_datetime(frame["date"].astype(str), format="%Y-%m-%d").dt.date.to_numpy()
    friday = np.array([d.weekday() == 4 for d in days], dtype=bool)
    site = np.isin(frame["site"].astype(str).to_numpy(), NINE_EIGHTY_SITES)
    off = friday & site & (fortnight_parity(days) == parity)
    return pd.DataFrame({COLUMNS[1]: off.astype(int)}, index=frame.index)


def nine_eighty_phase(days: pd.DataFrame) -> dict:
    """The estimate of the Fridays-off parity from a days table of JPL (module text): the rows
    with `site` in `NINE_EIGHTY_SITES`, `date`, `cars` (plugged in at τ) and `counts`. Returns
    the parity with the fewer mean cars on counting Fridays, each parity's day count and mean,
    and None as the parity when either parity has no counting Friday or the means are equal."""
    rows = days[np.isin(days["site"].astype(str), NINE_EIGHTY_SITES)]
    counts = rows["counts"].astype(str).isin(["True", "true", "1"])
    dates = pd.to_datetime(rows["date"].astype(str), format="%Y-%m-%d").dt.date.to_numpy()
    friday = np.array([d.weekday() == 4 for d in dates], dtype=bool) & counts.to_numpy()
    cars = rows["cars"].astype(int).to_numpy()
    parity = fortnight_parity(dates)
    result: dict = {"parity": None}
    means = []
    for p in (0, 1):
        chosen = friday & (parity == p)
        result[f"fridays_{p}"] = int(chosen.sum())
        means.append(float(cars[chosen].mean()) if chosen.any() else float("nan"))
        result[f"mean_cars_{p}"] = means[-1]
    if all(np.isfinite(means)) and means[0] != means[1]:
        result["parity"] = int(np.argmin(means))
    return result

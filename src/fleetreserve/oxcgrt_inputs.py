"""The OxCGRT input of the departure model, known at the decision time τ (plan section 7).

Source: the Oxford COVID-19 Government Response Tracker (Hale et al. 2021, Nature Human
Behaviour 5:529), its final dataset v1 (June 2023), file
`data/timeseries_indices/OxCGRT_timeseries_StringencyIndex_v1.csv` of the repository
OxCGRT/covid-policy-dataset at commit e7f66ee, CC BY 4.0, downloaded by
`tools/fetch_data.py --context` to `data/raw/oxcgrt/`. Layout: one row per jurisdiction, the
columns `CountryCode`, `CountryName`, `RegionCode`, `RegionName`, `CityCode`, `CityName`,
`Jurisdiction`, then one column per day written like `01Jan2020`, holding the stringency index
(0 to 100) of that day or nothing.

Column (`COLUMNS`): `oxcgrt_stringency`, for a row of date d (the local date of its day) at a
site whose jurisdiction is `context_places.SITE_OXCGRT[site]`, the index of day d - `LAG_DAYS`.

Why it is known at τ, and the lag assumed. OxCGRT published in real time from March 2020 and
its live repository was "updated every hour" (its README), but a day's value was coded by
volunteers after the fact, and the README warns that a recent fall in an index may be missing
data rather than a real change. No lag is stated by OxCGRT. `LAG_DAYS` = 14: a tracker built on
OxCGRT (the KFF global COVID-19 tracker) states a two-week lag in its reporting, and 14 days also
covers the gaps of a few days users reported in the live files. [Alternatives: 7 days; the exact
value published by τ, from the hourly history of the legacy repository OxCGRT/covid-policy-tracker,
which would remove the assumption.] What remains assumed: the values are those of the final
dataset, revised after the fact, not the vintage that was online at τ; a revision that changed
a day's value more than 14 days later is not excluded by this rule.

Fill outside the coded days, a reading each: a day before the first coded day (1 Jan 2020) gets
0, since no government response to COVID-19 existed before it (known at τ for any day before
2020); a day after the last coded day (`LAST_CODED`, 31 Dec 2022, fixed once: the final dataset
codes the jurisdictions in use from 1 Jan 2020 to 31 Dec 2022 with no empty cell inside, as a
review of the downloaded file found on 15 Nov 2026) gets the value of that day, which had been
published by τ at the 14-day lag (the a.s.r. days from 2023). The day is a constant, not the
last non-empty cell of the row's jurisdiction, so the value of a day never depends on whether a
later cell is filled. [Alternatives: NaN before or after the coded days.] An empty cell up to
`LAST_CODED`, and an empty cell at it for a day after it, gives NaN.

Jurisdictions (`context_places.SITE_OXCGRT`): national rows for Norway, the Netherlands and
Korea; for the ACN sites the California row (`US_CA`) and for NREL the Colorado row (`US_CO`),
which the file holds as state rows of the United States. A site missing from the table raises
`KeyError`; a jurisdiction missing from the file raises `LayoutError`.
"""

import io
from collections.abc import Mapping
from datetime import date, timedelta

import numpy as np
import pandas as pd

from fleetreserve.context_places import SITE_OXCGRT

LAG_DAYS = 14
COLUMNS = ("oxcgrt_stringency",)
ID_COLUMNS = (
    "CountryCode",
    "CountryName",
    "RegionCode",
    "RegionName",
    "CityCode",
    "CityName",
    "Jurisdiction",
)
FIRST_CODED = date(2020, 1, 1)
LAST_CODED = date(2022, 12, 31)
BEFORE_VALUE = 0.0


class LayoutError(ValueError):
    pass


def read_oxcgrt(data: bytes) -> dict[str, pd.Series]:
    """Per jurisdiction key, its daily index indexed by date (module text). The key is the
    `RegionCode` where there is one (`US_CA`), else the `CountryCode` of a national row
    (`NOR`); city rows are left out."""
    try:
        table = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, encoding="utf-8")
    except (ValueError, UnicodeDecodeError):
        raise LayoutError("the OxCGRT file cannot be read as CSV") from None
    if tuple(table.columns[: len(ID_COLUMNS)]) != ID_COLUMNS:
        raise LayoutError("the OxCGRT file does not start with its identifier columns")
    day_names = list(table.columns[len(ID_COLUMNS) :])
    try:
        days = [pd.to_datetime(name, format="%d%b%Y").date() for name in day_names]
    except ValueError:
        raise LayoutError("an OxCGRT day column is not written like 01Jan2020") from None
    steps = [b - a for a, b in zip(days, days[1:], strict=False)]
    if days[0] != FIRST_CODED or any(step != timedelta(days=1) for step in steps):
        raise LayoutError("the OxCGRT day columns are not consecutive from 1 Jan 2020")
    out = {}
    for _, row in table.iterrows():
        if row["CityCode"] != "":
            continue
        if row["RegionCode"] == "" and row["Jurisdiction"] != "NAT_TOTAL":
            continue
        key = row["RegionCode"] or row["CountryCode"]
        if key in out:
            raise LayoutError("the OxCGRT file lists a jurisdiction twice")
        text = row[day_names]
        try:
            values = pd.to_numeric(text.where(text != ""), errors="raise").astype(float)
        except ValueError:
            raise LayoutError("an OxCGRT value is not a number") from None
        out[key] = pd.Series(values.to_numpy(), index=pd.Index(days, name="date"))
    return out


def _value_at(series: pd.Series, day: date, last_coded: date) -> float:
    if day < FIRST_CODED:
        return BEFORE_VALUE
    if day > last_coded:
        day = last_coded
    return float(series.get(day, np.nan))


def oxcgrt_columns(
    frame: pd.DataFrame,
    index: Mapping[str, pd.Series],
    site_jurisdictions: Mapping[str, str] = SITE_OXCGRT,
    lag_days: int = LAG_DAYS,
    last_coded: date = LAST_CODED,
) -> pd.DataFrame:
    """`COLUMNS` for the rows of `frame`, read from its `site` and `date` only (module text)."""
    if lag_days < LAG_DAYS:
        raise ValueError(f"the lag must be at least {LAG_DAYS} days (module text)")
    lag = timedelta(days=lag_days)
    keys = frame["site"].astype(str).map(lambda site: site_jurisdictions[site])
    missing = sorted(set(keys) - set(index))
    if missing:
        raise LayoutError(f"the OxCGRT file has no row for {missing}")
    days = pd.to_datetime(frame["date"].astype(str), format="%Y-%m-%d").dt.date
    cache: dict[tuple[str, date], float] = {}
    values = []
    for key, day in zip(keys, days, strict=True):
        if (key, day) not in cache:
            cache[(key, day)] = _value_at(index[key], day - lag, last_coded)
        values.append(cache[(key, day)])
    return pd.DataFrame({COLUMNS[0]: np.array(values, dtype=float)}, index=frame.index)

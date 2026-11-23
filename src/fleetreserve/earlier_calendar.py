"""The academic calendar the earlier analysis's per-car models read (plan section 12), as day flags.

The earlier analysis's package (`ev-fleet-offer`, commit 05a6cf9, module `academic`) gives three
flags per local date, inputs known at the decision time: `in_term` (instruction in session, from
the beginning of instruction to the end of the term, study periods and final examinations
included), `summer` (between the end of the third term and the next beginning of instruction) and
`holiday` (an Institute holiday). Days that are neither in term nor summer are the winter and
spring recesses. The tables below are that module's, unchanged: the Caltech registrar's academic
calendars for 2017-18 to 2021-22 for the terms and the holidays they name, and the Institute's
events calendar for Independence Day, Labor Day and the year-end closure (24 December to
1 January), with the federal date and the same closure rule for the years it did not list.

Outside the span the tables cover (`FIRST` to `LAST`) the earlier module refuses the date; here
every flag is NaN there (reading E6 of `models.earlier_gbm`), so a unit with later dates is
forecast with the flags missing rather than not at all.
"""

import numpy as np
import pandas as pd

# (beginning of instruction, end of term), local dates, inclusive.
TERMS = (
    ("2018-04-02", "2018-06-15"),
    ("2018-10-01", "2018-12-14"),
    ("2019-01-07", "2019-03-20"),
    ("2019-04-01", "2019-06-14"),
    ("2019-10-01", "2019-12-13"),
    ("2020-01-06", "2020-03-18"),
    ("2020-03-30", "2020-06-12"),
    ("2020-09-29", "2020-12-11"),
    ("2021-01-04", "2021-03-17"),
    ("2021-03-29", "2021-06-11"),
    ("2021-09-27", "2021-12-10"),
    ("2022-01-03", "2022-03-16"),
    ("2022-03-28", "2022-06-10"),
)
# Third-term ends to the next beginning of instruction, exclusive of both.
SUMMERS = (
    ("2018-06-15", "2018-10-01"),
    ("2019-06-14", "2019-10-01"),
    ("2020-06-12", "2020-09-29"),
    ("2021-06-11", "2021-09-27"),
)
HOLIDAYS = (
    "2018-05-28",
    "2018-07-04",
    "2018-09-03",
    "2018-11-22",
    "2018-11-23",
    "2019-01-21",
    "2019-02-18",
    "2019-05-27",
    "2019-07-04",
    "2019-07-05",
    "2019-09-02",
    "2019-11-28",
    "2019-11-29",
    "2020-01-20",
    "2020-02-17",
    "2020-05-25",
    "2020-07-03",
    "2020-09-07",
    "2020-11-26",
    "2020-11-27",
    "2021-01-18",
    "2021-02-15",
    "2021-05-31",
    "2021-07-05",
    "2021-09-06",
    "2021-11-25",
    "2021-11-26",
    "2022-01-17",
    "2022-02-21",
    "2022-05-30",
)
CLOSURE_YEARS = (2017, 2018, 2019, 2020, 2021)  # 24 December of the year to 1 January after
FIRST, LAST = pd.Timestamp("2018-01-01"), pd.Timestamp("2022-06-30")
FLAGS = ("in_term", "summer", "holiday")


def _holidays() -> pd.DatetimeIndex:
    closures = [pd.date_range(f"{y}-12-24", f"{y + 1}-01-01", freq="D") for y in CLOSURE_YEARS]
    named = pd.DatetimeIndex(pd.to_datetime(list(HOLIDAYS)))
    return named.append(closures).unique().sort_values()


def day_flags(dates: pd.Series) -> pd.DataFrame:
    """`FLAGS` as 1.0 or 0.0 for local dates (text `YYYY-MM-DD`), NaN outside `FIRST` to `LAST`
    (module text); one row per date, index 0 to n - 1."""
    days = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).astype(str), format="%Y-%m-%d"))
    in_term = np.zeros(len(days), dtype=bool)
    summer = np.zeros(len(days), dtype=bool)
    for first, last in TERMS:
        in_term |= (days >= first) & (days <= last)
    for after, before in SUMMERS:
        summer |= (days > after) & (days < before)
    holiday = days.isin(_holidays())
    covered = (days >= FIRST) & (days <= LAST)
    out = {
        name: np.where(covered, flag.astype(float), np.nan)
        for name, flag in zip(FLAGS, (in_term, summer, holiday), strict=True)
    }
    return pd.DataFrame(out)

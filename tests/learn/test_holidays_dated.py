"""`calendar_inputs.public_holiday_dated` with the pinned `holidays` package (learn lane): a
holiday counts only on a day after its law or announcement; every holiday the package lists on
the data sets' spans has a dating; on those spans the dated flag equals the undated one.
"""

from datetime import date

import pandas as pd
import pytest

from fleetreserve import calendar_inputs, context_places, learn_features, load_korea


def flags(rows, countries=None):
    frame = pd.DataFrame(rows, columns=["site", "date"])
    got = calendar_inputs.dated_holiday_flags(frame, countries)
    return got["public_holiday_dated"].tolist()


def test_dated_holidays_by_hand():
    # Juneteenth observed on Fri 18 Jun 2021 (signed 17 Jun); Cabrini Day 5 Oct 2020 in Colorado
    # only; Korea's alternative Liberation Day 16 Aug 2021; 17 Jun 2021 is no holiday
    rows = [
        ("jpl", "2021-06-18"),
        ("nrel", "2020-10-05"),
        ("caltech", "2020-10-05"),
        ("korea-visits", "2021-08-16"),
        ("jpl", "2021-06-17"),
        ("OSL_T", "2020-05-17"),
    ]
    assert flags(rows) == [1, 1, 0, 1, 0, 1]


def test_the_five_korean_unit_names_take_korea_s_calendar():
    # the alternative Liberation Day 16 Aug 2021 and New Year's Day; 17 Aug 2021 is no holiday
    names = list(load_korea.UNITS)
    assert len(names) == 5
    rows = [(n, "2021-08-16") for n in names] + [(names[0], "2022-01-01"), (names[1], "2021-08-17")]
    assert flags(rows) == [1] * 5 + [1, 0]


def test_a_holiday_enacted_after_the_day_does_not_count(monkeypatch):
    late = dict(calendar_inputs.ENACTED)
    late[("US", date(2021, 6, 18))] = date(2021, 6, 18)
    monkeypatch.setattr(calendar_inputs, "ENACTED", late)
    assert flags([("jpl", "2021-06-18"), ("jpl", "2021-06-19")]) == [0, 1]


def test_an_undated_holiday_is_refused(monkeypatch):
    shorter = dict(calendar_inputs.LONG_STANDING)
    shorter["NO"] = shorter["NO"] - {"Constitution Day"}
    monkeypatch.setattr(calendar_inputs, "LONG_STANDING", shorter)
    with pytest.raises(ValueError, match="no dating"):
        flags([("OSL_T", "2020-05-17")])


def _span_rows():
    rows = []
    for site, places in context_places.SITE_PLACES.items():
        if site not in learn_features.SITE_COUNTRIES:
            continue
        first = min(context_places.PLACES[p].first for p in places)
        last = max(context_places.PLACES[p].last for p in places)
        for day in pd.date_range(first, last, freq="D"):
            rows.append((site, f"{day:%Y-%m-%d}"))
    return rows


def test_every_listed_holiday_on_the_spans_is_dated_and_none_is_moved():
    rows = _span_rows()
    frame = pd.DataFrame(rows, columns=["site", "date"])
    dated = calendar_inputs.dated_holiday_flags(frame)["public_holiday_dated"]
    plain = learn_features.holiday_flags(frame, learn_features.SITE_COUNTRIES)["public_holiday"]
    assert dated.sum() > 0
    assert dated.tolist() == plain.astype(int).tolist()

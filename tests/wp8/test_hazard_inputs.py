"""The new input columns of the departure model (`hazard_inputs` and the modules it calls), on
made-up files and cars, each with a case computed by hand, and the rule that each column is
known at τ: values of times after the cut it states are changed and the column is not.
"""

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from truth_tables_fake import FakeCar, date_of, tables

from fleetreserve import (
    calendar_inputs,
    context_places,
    hazard_inputs,
    learn_features,
    load_korea,
    oxcgrt_inputs,
    regime_inputs,
    school_inputs,
    weather_inputs,
)
from fleetreserve.days_from_truth import apply_car_rules

# --- weather --------------------------------------------------------------------------------


def weather_bytes(start="2019-01-01T00:00", hours=48, temp=float, precip=None, snow=None) -> bytes:
    times = pd.date_range(start, periods=hours, freq="h")
    rows = [
        f"{t:%Y-%m-%dT%H:%M},{temp(k)},{1.0 if precip is None else precip(k)},"
        f"{(2.0 if k == 24 else 0.0) if snow is None else snow(k)}"
        for k, t in enumerate(times)
    ]
    head = [
        weather_inputs.META_HEADER,
        "59.86,10.81,0.0,0,GMT,GMT",
        "",
        ",".join(weather_inputs.HEADER),
    ]
    return ("\n".join(head + rows) + "\n").encode("utf-8")


def frame(site, tau):
    return pd.DataFrame({"site": [site], "tau": [tau]})


def test_weather_by_hand():
    # τ 10:30 at +01:00 is 09:30 UTC; minus 8 h is 01:30, so c is 01:00 UTC on 2 Jan, hour 25 of
    # the file; the window is hours 2 to 25: temperatures 2..25, precipitation 1 each, snow 2 at
    # hour 24.
    weather = {"oslo": weather_inputs.read_weather(weather_bytes())}
    got = weather_inputs.weather_columns(frame("OSL_T", "2019-01-02T10:30:00+01:00"), weather)
    assert got.iloc[0].tolist() == [25.0, 13.5, 24.0, 2.0]


def test_weather_reads_nothing_after_its_cut():
    rows = frame("OSL_T", "2019-01-02T10:30:00+01:00")
    base = weather_inputs.weather_columns(
        rows, {"oslo": weather_inputs.read_weather(weather_bytes())}
    )
    later = weather_bytes(
        temp=lambda k: 999.0 if k > 25 else float(k),
        precip=lambda k: 50.0 if k > 25 else 1.0,
        snow=lambda k: 50.0 if k > 25 else (2.0 if k == 24 else 0.0),
    )
    after = weather_inputs.weather_columns(rows, {"oslo": weather_inputs.read_weather(later)})
    pd.testing.assert_frame_equal(base, after)
    at_cut = weather_bytes(temp=lambda k: 999.0 if k == 25 else float(k))
    moved = weather_inputs.weather_columns(rows, {"oslo": weather_inputs.read_weather(at_cut)})
    assert moved.iloc[0, 0] == 999.0


def test_weather_window_before_the_file_is_missing():
    weather = {"oslo": weather_inputs.read_weather(weather_bytes())}
    got = weather_inputs.weather_columns(frame("OSL_T", "2019-01-01T12:00:00+00:00"), weather)
    # c is 04:00 UTC on 1 Jan, hour 4: the temperature is there, the 24 hours before it are not
    assert got.iloc[0, 0] == 4.0
    assert got.iloc[0, 1:].isna().all()


def test_weather_of_several_places_is_their_mean():
    weather = {
        "a": weather_inputs.read_weather(weather_bytes()),
        "b": weather_inputs.read_weather(weather_bytes(temp=lambda k: k + 10.0)),
    }
    got = weather_inputs.weather_columns(
        frame("p", "2019-01-02T09:30:00+00:00"), weather, site_places={"p": ("a", "b")}
    )
    assert got.iloc[0, 0] == 30.0 and got.iloc[0, 1] == 18.5


def test_weather_delay_below_eight_hours_is_refused():
    weather = {"oslo": weather_inputs.read_weather(weather_bytes())}
    with pytest.raises(ValueError):
        weather_inputs.weather_columns(
            frame("OSL_T", "2019-01-02T09:30:00+00:00"), weather, delay_hours=7
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.replace(b",0,GMT,GMT", b",3600,GMT,GMT"),
        lambda b: b.replace(b"snowfall (cm)", b"rain (mm)"),
        lambda b: b.replace(b"2019-01-01T05:00", b"2019-01-01T06:00", 1),
        lambda b: b.replace(b"2019-01-01T05:00,5.0", b"2019-01-01T05:00,warm", 1),
        lambda b: b"\xff" + b,
    ],
)
def test_weather_layout_other_than_fetched_is_refused(change):
    with pytest.raises(weather_inputs.LayoutError):
        weather_inputs.read_weather(change(weather_bytes()))


# --- OxCGRT ---------------------------------------------------------------------------------

DAYS = pd.date_range("2020-01-01", periods=40, freq="D")


def oxcgrt_bytes(nor=None) -> bytes:
    head = ",".join([*oxcgrt_inputs.ID_COLUMNS, *(f"{d:%d%b%Y}" for d in DAYS)])
    nor = nor or (lambda k: "" if k == 20 or k > 34 else f"{k:.2f}")
    rows = [
        ",".join(["NOR", "Norway", "", "", "", "", "NAT_TOTAL", *(nor(k) for k in range(40))]),
        ",".join(
            [
                "USA",
                "United States",
                "US_CA",
                "California",
                "",
                "",
                "STATE_TOTAL",
                *(f"{100 - k:.2f}" for k in range(40)),
            ]
        ),
        ",".join(["USA", "United States", "", "", "", "", "NAT_TOTAL", *(["50"] * 40)]),
        ",".join(
            ["BRA", "Brazil", "BR_SP", "Sao Paulo", "BR_SP_X", "City", "CITY_TOTAL", *(["7"] * 40)]
        ),
    ]
    return ("\n".join([head, *rows]) + "\n").encode("utf-8")


def oxcgrt(rows, data=None):
    index = oxcgrt_inputs.read_oxcgrt(data or oxcgrt_bytes())
    table = pd.DataFrame(rows, columns=["site", "date"])
    # the made-up file's last coded day is day 34 (4 Feb 2020)
    got = oxcgrt_inputs.oxcgrt_columns(table, index, last_coded=DAYS[34].date())
    return got["oxcgrt_stringency"].tolist()


def test_oxcgrt_by_hand():
    # 20 Jan less 14 days is 6 Jan, day 5: Norway 5, California 95; 10 Jan less 14 days is
    # 27 Dec 2019, before the first coded day: 0; 1 Mar less 14 days is 16 Feb, after Norway's
    # last coded day (day 34, 4 Feb): 34; 4 Feb less 14 is 21 Jan, day 20, an empty cell: NaN.
    got = oxcgrt(
        [
            ("OSL_T", "2020-01-20"),
            ("caltech", "2020-01-20"),
            ("OSL_T", "2020-01-10"),
            ("OSL_T", "2020-03-01"),
            ("OSL_T", "2020-02-04"),
        ]
    )
    assert got[:4] == [5.0, 95.0, 0.0, 34.0] and math.isnan(got[4])


def test_oxcgrt_after_the_last_coded_day_does_not_depend_on_later_cells():
    # coded to 31 Dec 2022 (50), empty after it but for 5 Jan 2023 (99); 22 Jan 2023 less 14 days
    # is 8 Jan, after the fixed last coded day: its value 50, whatever a later cell holds
    days = pd.Index([date(2022, 12, 25) + timedelta(days=d) for d in range(20)], name="date")
    values = [50.0 if d <= date(2022, 12, 31) else math.nan for d in days]
    values[days.get_loc(date(2023, 1, 5))] = 99.0
    table = pd.DataFrame({"site": ["asr"], "date": ["2023-01-22"]})
    index = {context_places.SITE_OXCGRT["asr"]: pd.Series(values, index=days)}
    assert oxcgrt_inputs.oxcgrt_columns(table, index)["oxcgrt_stringency"].tolist() == [50.0]


def test_oxcgrt_reads_nothing_after_its_lag():
    later = oxcgrt_bytes(nor=lambda k: "99" if k > 5 else f"{k:.2f}")
    assert oxcgrt([("OSL_T", "2020-01-20")], later) == [5.0]


def test_oxcgrt_keys_national_and_state_rows_and_leaves_out_cities():
    assert set(oxcgrt_inputs.read_oxcgrt(oxcgrt_bytes())) == {"NOR", "US_CA", "USA"}


def test_oxcgrt_lag_below_fourteen_days_is_refused():
    index = oxcgrt_inputs.read_oxcgrt(oxcgrt_bytes())
    table = pd.DataFrame({"site": ["OSL_T"], "date": ["2020-01-20"]})
    with pytest.raises(ValueError):
        oxcgrt_inputs.oxcgrt_columns(table, index, lag_days=13)


def test_oxcgrt_missing_jurisdiction_and_bad_layout_are_refused():
    index = oxcgrt_inputs.read_oxcgrt(oxcgrt_bytes())
    table = pd.DataFrame({"site": ["asr"], "date": ["2020-01-20"]})
    with pytest.raises(oxcgrt_inputs.LayoutError):
        oxcgrt_inputs.oxcgrt_columns(table, index)
    with pytest.raises(oxcgrt_inputs.LayoutError):
        oxcgrt_inputs.read_oxcgrt(oxcgrt_bytes().replace(b"03Jan2020", b"04Jan2020", 1))
    with pytest.raises(oxcgrt_inputs.LayoutError):
        oxcgrt_inputs.read_oxcgrt(oxcgrt_bytes(nor=lambda k: "high"))


# --- regimes --------------------------------------------------------------------------------


def test_regimes_by_hand():
    rows = pd.DataFrame(
        [
            ("nrel", "2019-04-30"),
            ("nrel", "2019-05-01"),
            ("caltech", "2019-05-01"),
            ("asr", "2022-08-31"),
            ("asr", "2022-09-01"),
            ("asr", "2023-02-28"),
            ("asr", "2023-03-01"),
            ("OSL_T", "2020-03-11"),
            ("OSL_T", "2020-03-12"),
            ("asr", "2023-05-05"),
            ("asr", "2023-05-06"),
        ],
        columns=["site", "date"],
    )
    got = regime_inputs.regime_columns(rows)
    assert got["regime_nrel_controller"].tolist() == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert got["regime_asr_paid"].tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    assert got["regime_asr_controller"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
    assert got["regime_covid"].tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]


# --- JPL 9/80 Fridays -----------------------------------------------------------------------


def test_fortnight_parity_by_hand():
    # 4 Jan 2019 is 17,899 days, 2,557 whole weeks, after Friday 2 Jan 1970: odd
    days = pd.to_datetime(["1970-01-02", "1970-01-09", "2019-01-04", "2019-01-11"]).date
    assert calendar_inputs.fortnight_parity(days).tolist() == [0, 1, 1, 0]


def test_nine_eighty_fridays_by_hand():
    rows = pd.DataFrame(
        [
            ("jpl", "2019-01-04"),
            ("jpl", "2019-01-11"),
            ("caltech", "2019-01-04"),
            ("jpl", "2019-01-03"),
            ("jpl", "2019-01-18"),
            ("jpl", "2019-01-05"),  # a Saturday of the same fortnight parity as 4 Jan
        ],
        columns=["site", "date"],
    )
    got = calendar_inputs.nine_eighty_flags(rows, parity=1)["nine_eighty_friday"].tolist()
    assert got == [1, 0, 0, 0, 1, 0]
    with pytest.raises(ValueError):
        calendar_inputs.nine_eighty_flags(rows, parity=None)


def test_nine_eighty_phase_picks_the_parity_with_fewer_cars():
    fridays = pd.date_range("2019-01-04", periods=8, freq="7D")
    days = pd.DataFrame(
        {
            "site": ["jpl"] * 8 + ["caltech"],
            "date": [f"{d:%Y-%m-%d}" for d in fridays] + ["2019-01-04"],
            # parity 1 (4 Jan, 18 Jan, ...) has 5 cars, parity 0 has 20; the last JPL day does
            # not count; caltech's day is another site
            "cars": [5, 20, 5, 20, 5, 20, 5, 1, 0],
            "counts": ["True"] * 7 + ["False", "True"],
        }
    )
    got = calendar_inputs.nine_eighty_phase(days)
    assert got == {
        "parity": 1,
        "fridays_0": 3,
        "mean_cars_0": 20.0,
        "fridays_1": 4,
        "mean_cars_1": 5.0,
    }


# --- the assembly ---------------------------------------------------------------------------


def _cars(site="OSL_1"):
    fakes = [
        FakeCar(
            site,
            date_of(k),
            7.0 + j,
            17.0 + j,
            room=1.0 + j,
            energy=4.0 + j + 0.1 * k,
            session=f"{site}-{k}-{j}",
        )
        for k in range(3)
        for j in range(3)
    ]
    car_table, _ = tables(fakes)
    return apply_car_rules(car_table, 0.0, "cap")


def test_no_new_column_leaves_the_table_as_it_is():
    cars = _cars()
    table = learn_features.build_table(cars, rho=0.0)
    again = hazard_inputs.with_inputs(table, cars)
    assert again is table
    pd.testing.assert_frame_equal(again, learn_features.build_table(cars, rho=0.0))


def test_new_columns_are_appended_and_the_old_ones_unchanged():
    cars = _cars()
    table = learn_features.build_table(cars, rho=0.0)
    names = ("energy_drawn_kwh", "regime_covid")
    got = hazard_inputs.with_inputs(table, cars, names)
    assert list(got.columns) == [*table.columns, *names]
    pd.testing.assert_frame_equal(got[list(table.columns)], table)
    # reading E3: the fake's constant cap 6.6 kW x (τ 10:00 - plug-in 7 + j) = 6.6 (3 - j), not
    # the fake's energy before τ, energy - room = 3 + 0.1 k
    expected = [6.6 * (3 - j) for _ in range(3) for j in range(3)]
    assert sorted(got["energy_drawn_kwh"].tolist()) == pytest.approx(sorted(expected))


def test_energy_drawn_refuses_a_nominal_power_that_varies_at_a_site_but_nrel():
    cars = _cars()
    cars.loc[cars.index[0], "nominal_kw"] = "7.200000"
    with pytest.raises(ValueError):
        hazard_inputs.energy_drawn(cars)
    nrel = _cars("nrel")
    nrel.loc[nrel.index[0], "nominal_kw"] = "7.200000"
    # NREL's session power is replaced by the stand-in, so no refusal and no trace of it
    assert set(hazard_inputs.energy_drawn(nrel).round(9)) == {6.6, 13.2, 19.8}


def test_weather_through_the_assembly_uses_the_site_place():
    cars = _cars("OSL_1")
    table = learn_features.build_table(cars, rho=0.0)
    # the fake's first day is 1 Jan 2020, τ 10:00 at -06:00: 16:00 UTC, so c = 08:00 UTC, hour 32
    # of a file from 31 Dec 2019 00:00 whose temperature is its hour number; the next days 56, 80
    sources = hazard_inputs.ContextSources(
        weather={
            "oslo": weather_inputs.read_weather(
                weather_bytes(start="2019-12-31T00:00", hours=24 * 6)
            )
        }
    )
    got = hazard_inputs.with_inputs(table, cars, ("weather_temperature_c",), sources)
    assert date_of(0) == "2020-01-01"
    assert got["weather_temperature_c"].tolist() == [32.0] * 3 + [56.0] * 3 + [80.0] * 3


def test_misaligned_cars_unknown_names_and_missing_sources_are_refused():
    cars = _cars()
    table = learn_features.build_table(cars, rho=0.0)
    with pytest.raises(ValueError):
        hazard_inputs.with_inputs(table, cars.iloc[::-1], ("regime_covid",))
    with pytest.raises(ValueError):
        hazard_inputs.with_inputs(table, cars, ("energy_before_kwh",))
    with pytest.raises(ValueError):
        hazard_inputs.with_inputs(table, cars, ("weather_temperature_c",))
    with pytest.raises(ValueError):
        hazard_inputs.with_inputs(table, cars, ("regime_covid", "regime_covid"))


def test_every_new_column_is_named_once_and_none_is_an_old_one():
    assert len(set(hazard_inputs.NEW_COLUMNS)) == len(hazard_inputs.NEW_COLUMNS)
    old = {
        *learn_features.FEATURE_COLUMNS,
        *learn_features.TAIL_HISTORY_COLUMNS,
        *learn_features.CONTEXT_COLUMNS,
        *learn_features.LABEL_COLUMNS,
    }
    assert not old & set(hazard_inputs.NEW_COLUMNS)
    assert np.isin(["public_holiday_dated", "nine_eighty_friday"], hazard_inputs.NEW_COLUMNS).all()


# --- the place table ------------------------------------------------------------------------


def test_every_site_of_the_site_tables_has_places_and_a_jurisdiction():
    for site in learn_features.SITE_COUNTRIES:
        assert all(p in context_places.PLACES for p in context_places.SITE_PLACES[site])
        assert context_places.SITE_OXCGRT[site]
    assert context_places.SITE_OXCGRT["nrel"] == "US_CO"
    assert context_places.SITE_OXCGRT["jpl"] == "US_CA"


def test_the_context_tables_name_exactly_the_sites_the_loaders_write():
    # the Korean portfolios under the Korean loader's unit names; the data-set name `korea` is
    # no site, so no table keeps it
    sites = set(learn_features.SITE_COUNTRIES)
    assert set(context_places.SITE_PLACES) == set(context_places.SITE_OXCGRT) == sites
    for column in school_inputs.COLUMNS:
        assert set(school_inputs.SITE_CALENDARS[column]) == sites
    assert set(context_places.KOREA_SITES) == set(load_korea.UNITS)
    assert "korea" not in sites


def test_places_read_by_hand():
    # degrees and minutes: Oslo 59 deg 54', 10 deg 45'; Bodo 67 deg 16'
    oslo = context_places.PLACES["oslo"]
    assert (oslo.latitude, oslo.longitude) == (59.9, 10.75)
    assert context_places.PLACES["bodo"].latitude == 67.2667
    # NREL from the archive's first day; Krokkleiva 15 Jan to 6 May 2021, a week each side
    assert context_places.PLACES["golden-nrel"].first.isoformat() == "2017-01-01"
    krokkleiva = context_places.PLACES["krokkleiva"]
    assert (krokkleiva.first.isoformat(), krokkleiva.last.isoformat()) == (
        "2021-01-08",
        "2021-05-13",
    )
    assert len(context_places.SITE_PLACES["korea-work"]) == 7

"""Every new input column of `hazard_inputs` is known at τ, checked on made-up sessions run
through the real path (`truth.build`, the cars table as CSV text, `apply_car_rules`,
`build_table`, `with_inputs`): cars that are the same up to τ and differ only after it (their
departure, their session energy, NREL's highest power) get the same value in every column. And
`energy_drawn_kwh` by hand (reading E3 of `hazard_inputs`)."""

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from fleetreserve import context_places, hazard_inputs, learn_features, school_inputs, truth
from fleetreserve.days_from_truth import apply_car_rules
from fleetreserve.windows import Window


def _sessions(site, zone, rows):
    out = []
    for k, r in enumerate(rows):
        out.append(
            {
                "site": site,
                "session_id": f"s{k}",
                "user_id": f"u{k}",
                "station": str(k),
                "plug_in": pd.Timestamp(r["plug_in"]).tz_localize(zone),
                "plug_out": pd.Timestamp(r["plug_out"]).tz_localize(zone),
                "energy_kwh": r["energy"],
                "nominal_kw": r["nominal"],
                "nominal_rule": r["rule"],
                "charge_start": pd.Timestamp(r["start"]).tz_localize(zone)
                if "start" in r
                else None,
                "charge_end": pd.Timestamp(r["end"]).tz_localize(zone) if "end" in r else None,
                "declared_departure": None,
            }
        )
    frame = pd.DataFrame(out)
    for c in ("charge_start", "charge_end", "declared_departure"):
        frame[c] = pd.to_datetime(frame[c], utc=True).dt.tz_convert(zone)
    return frame


def _sources(site):
    hours = pd.date_range("2019-01-01", "2020-01-01", freq="h", tz="UTC")
    k = np.arange(len(hours), dtype=float)
    weather = pd.DataFrame(
        {"temperature": k % 30, "precipitation": k % 3, "snowfall": k % 2}, index=hours
    )
    days = pd.Index([date(2020, 1, 1) + timedelta(days=d) for d in range(30)], name="date")
    return hazard_inputs.ContextSources(
        weather={place: weather for place in context_places.SITE_PLACES[site]},
        oxcgrt={context_places.SITE_OXCGRT[site]: pd.Series(np.arange(30.0), index=days)},
        nine_eighty_parity=0,
        school=school_inputs.load_terms(),
    )


def _columns(sessions, window, zone, unit, bound, setting=truth.MAIN):
    day = sessions["plug_in"].iloc[0].date()
    cars, _ = truth.build(
        sessions,
        window,
        zone,
        unit=unit,
        days=[day],
        limit_kw=50.0,
        bound_by_charge_fields=bound,
        setting=setting,
    )
    text = io.StringIO()
    cars.to_csv(text, index=False)
    text.seek(0)
    ruled = apply_car_rules(pd.read_csv(text, dtype=str, keep_default_na=False), 0.0, "cap")
    table = learn_features.build_table(ruled, rho=0.0)
    names = hazard_inputs.NEW_COLUMNS
    got = hazard_inputs.with_inputs(table, ruled, names, _sources(sessions["site"].iloc[0]))
    return got.set_index("session_id")[list(names)].sort_index()


def _same_rows(got):
    first = got.iloc[0]
    for _, row in got.iloc[1:].iterrows():
        pd.testing.assert_series_equal(row, first, check_names=False)


def norway(setting=truth.MAIN):
    # τ 17:00; three cars plug in at 15:00 and charge at 3.6 kW until they leave 0.5, 1 and 4 h
    # after τ, so a meter at τ reads 7.2 kWh for each; the source's cap is 7.4 kW
    rows = [
        {
            "plug_in": "2019-03-05 15:00",
            "plug_out": str(pd.Timestamp("2019-03-05 17:00") + pd.Timedelta(hours=after)),
            "energy": 3.6 * (2 + after),
            "nominal": 7.4,
            "rule": "source cap",
        }
        for after in (0.5, 1.0, 4.0)
    ]
    sessions = _sessions("OSL_1", "Europe/Oslo", rows)
    return _columns(sessions, Window(17, 3, "fake"), "Europe/Oslo", "OSL_1", False, setting)


def nrel():
    # τ 08:00; four cars plug in at 07:00 and charge at 2 kW from plug-in until they leave 0.5,
    # 3, 3 and 6 h after τ; the third reaches 7.2 kW once after τ (its max_charge_power), the
    # others 6.6 kW
    rows = []
    for after, peak in ((0.5, 6.6), (3.0, 6.6), (3.0, 7.2), (6.0, 6.6)):
        out = str(pd.Timestamp("2019-06-04 08:00") + pd.Timedelta(hours=after))
        rows.append(
            {
                "plug_in": "2019-06-04 07:00",
                "plug_out": out,
                "start": "2019-06-04 07:00",
                "end": out,
                "energy": 2.0 * (1.0 + after),
                "nominal": peak,
                "rule": "max_charge_power floored at 3.3 kW",
            }
        )
    sessions = _sessions("nrel", "America/Denver", rows)
    return _columns(sessions, Window(8, 3, "fake"), "America/Denver", "nrel", True)


def test_cars_the_same_up_to_tau_get_the_same_inputs_norway():
    got = norway()
    _same_rows(got)
    # by hand, reading E3: the source's constant cap 7.4 kW x 2 h since plug-in
    assert got["energy_drawn_kwh"].tolist() == pytest.approx([14.8] * 3)


def test_cars_the_same_up_to_tau_get_the_same_inputs_nrel():
    got = nrel()
    _same_rows(got)
    # by hand: NREL's constant stand-in 6.6 kW x 1 h since plug-in
    assert got["energy_drawn_kwh"].tolist() == pytest.approx([6.6] * 4)


def test_the_held_setting_draws_nothing_before_tau():
    got = norway(truth.HELD)
    assert got["energy_drawn_kwh"].tolist() == [0.0] * 3

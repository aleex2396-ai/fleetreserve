"""Declared departure in the session and truth tables (plan section 3, "Declared energy,
departure: ACN, NREL only"), on made-up data.

1. `truth.build`: a declared departure stated at or before τ is written as hours from τ with its
   statement time; one stated after τ, one with no statement time and one with no declared time
   are missing; a session over two days gets each day's own hours; every other column equals the
   table built without declared values; a session table without `declared_at` gives missing
   values and no plug-out flag; the session table's flag reaches the cars table.
2. NREL loader: the statement time is `request_entry_time`; the plug-out-from-declared flag is set
   exactly on the rows whose times were filled from `expected_departure`, never under the main
   reading or the `charge_end` reading.
3. ACN loader: the statement time is the `modifiedAt` of the first entry; none for a record
   without entries or with no readable `modifiedAt`.
4. The truth build writes the columns and the summary's declared share; no NREL car of the main
   reading has its plug-out from the declared departure.
5. `truth_verify` catches a statement made after τ, a declared value without its statement time,
   and a car whose plug-out came from its declared departure.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from fake_raw import (
    TEST_WINDOW,
    acn_record,
    acn_time,
    full_tree,
    nrel_row,
    raw_root,
    write_acn,
    write_nrel,
)

from fleetreserve import load_acn, load_nrel, truth, truth_build, truth_verify
from fleetreserve.sessions import COLUMNS

ZONE = "America/Los_Angeles"
DAY, NEXT = "2019-05-01", "2019-05-02"
DECLARED = ("declared_at", "declared_departure_h", "plug_out_from_declared")


def _t(text):
    return pd.NaT if text is None else pd.Timestamp(text, tz=ZONE)


def _sessions(rows, with_stated=True) -> pd.DataFrame:
    table = []
    for row in rows:
        entry = dict.fromkeys(COLUMNS)
        entry.update(
            site="s",
            role="train",
            session_id=row["id"],
            user_id="u",
            plug_in=_t(row["plug_in"]),
            plug_out=_t(row["plug_out"]),
            charge_start=pd.NaT,
            charge_end=pd.NaT,
            energy_kwh=5.0,
            max_power_kw=np.nan,
            declared_departure=_t(row.get("declared")),
            declared_energy_kwh=np.nan,
            nominal_kw=6.66,
            nominal_rule="source cap",
        )
        entry["declared_at"] = _t(row.get("stated"))
        entry["plug_out_from_declared"] = row.get("flag", False)
        table.append(entry)
    frame = pd.DataFrame(table)
    for name in ("plug_in", "plug_out", "charge_start", "charge_end", "declared_departure"):
        frame[name] = pd.to_datetime(frame[name], utc=True).dt.tz_convert(ZONE)
    frame["declared_at"] = pd.to_datetime(frame["declared_at"], utc=True).dt.tz_convert(ZONE)
    columns = [*COLUMNS, "declared_at", "plug_out_from_declared"] if with_stated else list(COLUMNS)
    return frame[columns]


def _build(sessions, days=(DAY,)):
    return truth.build(
        sessions,
        TEST_WINDOW,
        ZONE,
        unit="u",
        days=[pd.Timestamp(d).date() for d in days],
        limit_kw=19.2,
        bound_by_charge_fields=False,
    )


ROWS = [
    {"id": "before", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00",
     "declared": f"{DAY} 16:30", "stated": f"{DAY} 08:05"},
    {"id": "at-tau", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00",
     "declared": f"{DAY} 15:00", "stated": f"{DAY} 10:00"},
    {"id": "after", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00",
     "declared": f"{DAY} 16:00", "stated": f"{DAY} 10:00:01"},
    {"id": "no-time", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00",
     "declared": f"{DAY} 16:00"},
    {"id": "none", "plug_in": f"{DAY} 08:00", "plug_out": f"{DAY} 17:00",
     "stated": f"{DAY} 08:00"},
    {"id": "two-day", "plug_in": f"{DAY} 08:00", "plug_out": f"{NEXT} 12:00",
     "declared": f"{NEXT} 11:00", "stated": f"{DAY} 08:10", "flag": True},
]  # fmt: skip


def test_declared_departure_is_kept_only_when_stated_by_tau():
    cars, _ = _build(_sessions(ROWS), days=(DAY, NEXT))
    by = {(r.session_id, str(r.date)): r for r in cars.itertuples(index=False)}
    assert by[("before", DAY)].declared_departure_h == 6.5
    assert by[("before", DAY)].declared_at.isoformat() == "2019-05-01T08:05:00-07:00"
    assert by[("at-tau", DAY)].declared_departure_h == 5.0
    assert by[("two-day", DAY)].declared_departure_h == 25.0
    assert by[("two-day", NEXT)].declared_departure_h == 1.0
    for name in ("after", "no-time", "none"):
        assert math.isnan(by[(name, DAY)].declared_departure_h), name
        assert pd.isna(by[(name, DAY)].declared_at), name
    flags = {key: row.plug_out_from_declared for key, row in by.items()}
    assert flags[("two-day", DAY)] and flags[("two-day", NEXT)]
    assert sum(flags.values()) == 2
    assert list(cars.columns[-3:]) == list(DECLARED)


def test_the_other_columns_do_not_change():
    plain = [{k: v for k, v in r.items() if k in ("id", "plug_in", "plug_out")} for r in ROWS]
    with_declared, days_a = _build(_sessions(ROWS), days=(DAY, NEXT))
    without, days_b = _build(_sessions(plain, with_stated=False), days=(DAY, NEXT))
    old = [c for c in truth.CAR_COLUMNS if c not in DECLARED]
    pd.testing.assert_frame_equal(with_declared[old], without[old])
    pd.testing.assert_frame_equal(days_a, days_b)
    assert without["declared_departure_h"].isna().all() and without["declared_at"].isna().all()
    assert not without["plug_out_from_declared"].any()


def _nrel_rows():
    return [
        nrel_row(  # 2018: no plug-in or plug-out recorded
            request_entry_time="2018-06-01 08:00:00",
            expected_departure="2018-06-01 17:00:00",
            termin_charge="2018-06-01 11:00:00",
            energy_charged="5",
        ),
        nrel_row(  # plug-in recorded, plug-out not: both are filled under a fallback
            request_entry_time="2019-04-01 08:00:00",
            expected_departure="2019-04-01 16:00:00",
            start_date_time="2019-04-01 08:00:00",
            termin_charge="2019-04-01 12:00:00",
            energy_charged="5",
        ),
        nrel_row(
            request_entry_time="2019-05-15 08:00:05",
            expected_departure="2019-05-15 16:00:00",
            start_date_time="2019-05-15 08:00:00",
            end_date_time="2019-05-15 16:30:00",
            energy_charged="9.5",
        ),
    ]


def test_nrel_statement_time_and_the_plug_out_flag(tmp_path):
    root = raw_root(tmp_path)
    write_nrel(root, _nrel_rows())
    main, _ = load_nrel.load(root)
    assert list(main["session_id"]) == ["row-3"]
    assert main.iloc[0]["declared_at"].isoformat() == "2019-05-15T08:00:05-06:00"
    assert not main["plug_out_from_declared"].any()
    by_charge, _ = load_nrel.load(root, missing_plug_out="charge_end")
    assert len(by_charge) == 3 and not by_charge["plug_out_from_declared"].any()
    by_declared, _ = load_nrel.load(root, missing_plug_out="declared_departure")
    flags = dict(zip(by_declared["session_id"], by_declared["plug_out_from_declared"], strict=True))
    assert flags == {"row-1": True, "row-2": True, "row-3": False}
    filled = by_declared.set_index("session_id")
    assert (
        filled.loc[["row-1", "row-2"], "plug_out"]
        == filled.loc[["row-1", "row-2"], "declared_departure"]
    ).all()


def test_acn_statement_time_is_the_first_entry(tmp_path):
    entries = [
        {"modifiedAt": acn_time("2019-03-05 16:10:00"),
         "requestedDeparture": acn_time("2019-03-06 01:00:00"), "kWhRequested": 20.0},
        {"modifiedAt": acn_time("2019-03-05 16:01:00"),
         "requestedDeparture": acn_time("2019-03-05 23:00:00"), "kWhRequested": 10.0},
    ]  # fmt: skip
    unreadable = [
        {"modifiedAt": "yesterday", "requestedDeparture": acn_time("2019-03-05 23:00:00")}
    ]
    records = [
        acn_record("s1", "2019-03-05 16:00:00", "2019-03-06 02:00:00", 8.5, inputs=entries),
        acn_record("s2", "2019-03-05 16:00:00", "2019-03-06 02:00:00", 8.5),
        acn_record("s3", "2019-03-05 16:00:00", "2019-03-06 02:00:00", 8.5, inputs=unreadable),
    ]
    root = raw_root(tmp_path)
    write_acn(root, "caltech", "2019-03", records)
    sessions, _ = load_acn.load("caltech", root)
    stated = dict(zip(sessions["session_id"], sessions["declared_at"], strict=True))
    assert stated["s1"].isoformat() == "2019-03-05T08:01:00-08:00"
    assert pd.isna(stated["s2"]) and pd.isna(stated["s3"])
    assert not sessions["plug_out_from_declared"].any()


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("build")
    root = full_tree(tmp_path)
    out = tmp_path / "out"
    result = truth_build.build_all(root, TEST_WINDOW)
    written = truth_build.write(result, out)
    summary = json.loads(
        json.dumps(truth_build.summary_document(result, written), sort_keys=True, default=str)
    )
    return summary, out


def _read(out, unit):
    cars = pd.read_csv(out / f"{unit}-cars.csv", dtype=str, keep_default_na=False)
    days = pd.read_csv(out / f"{unit}-days.csv", dtype=str, keep_default_na=False)
    return cars, days


def test_the_build_writes_the_declared_columns(built):
    summary, out = built
    # Made-up NREL rows: stated 08:00:05, declared 16:00, τ 10:00. ACN rows have no app entry.
    cars, _ = _read(out, "nrel-from-may-2019")
    assert set(cars["declared_departure_h"]) == {"6.000000"}
    assert set(cars["declared_at"]) == {"2019-05-15T08:00:05-06:00"}
    assert set(cars["plug_out_from_declared"]) == {"False"}
    units = summary["units"]
    for unit in truth_build.NREL_UNITS:
        assert units[unit]["share_declared_departure_known_at_tau"] == 1.0
        assert units[unit]["cars_plug_out_from_declared"] == 0
    assert units["caltech"]["share_declared_departure_known_at_tau"] == 0.0
    assert units[truth_build.POOLED]["share_declared_departure_known_at_tau"] == 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"declared_at": "2019-05-15T10:00:01-06:00"}, "1 declared departures stated after τ"),
        ({"declared_at": ""}, "a declared departure without its statement time, or the reverse"),
        (
            {"plug_out_from_declared": "True"},
            "1 cars have a plug-out taken from the declared departure",
        ),
    ],
)
def test_truth_verify_catches_a_declared_value_it_must_not_hold(built, change, message):
    _, out = built
    cars, days = _read(out, "nrel-from-may-2019")
    assert truth_verify.check_tables(cars, days, {"nrel": 10}) == []
    broken = cars.copy()
    for column, value in change.items():
        broken.loc[0, column] = value
    assert truth_verify.check_tables(broken, days, {"nrel": 10}) == [message]

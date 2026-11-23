"""The WP2 pass-mark checks on tables built from a made-up raw folder: they pass on good
tables, and each broken rule is caught."""

import json

import pandas as pd
import pytest
from fake_raw import TEST_WINDOW, full_tree

from fleetreserve import truth_build, truth_verify


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
    return summary, out, root


def _read(out, unit):
    cars = pd.read_csv(out / f"{unit}-cars.csv", dtype=str, keep_default_na=False)
    days = pd.read_csv(out / f"{unit}-days.csv", dtype=str, keep_default_na=False)
    return cars, days


@pytest.mark.parametrize("role", ["train", "development"])
def test_good_tables_pass(built, role):
    summary, out, root = built
    status, lines = truth_verify.verify_role(role, summary, out, root)
    assert status == 0, lines
    assert all(line.startswith("passed: ") for line in lines)
    assert len(lines) == len(truth_build.UNITS[role])


def test_fallback_shares_pass_and_a_missing_one_fails(built):
    summary, _, _ = built
    assert truth_verify.verify_fallbacks(summary)[0] == 0
    broken = json.loads(json.dumps(summary))
    del broken["units"]["jpl"]["fallback_shares"]["sessions_missing_user_id"]
    status, lines = truth_verify.verify_fallbacks(broken)
    assert status == 1 and any("jpl" in line for line in lines)
    broken["units"]["jpl"]["fallback_shares"]["sessions_missing_user_id"] = 1.5
    assert truth_verify.verify_fallbacks(broken)[0] == 1
    del broken["units"]["BOD"]
    assert truth_verify.verify_fallbacks(broken)[0] == 1


def _break(cars, days, how):
    cars, days = cars.copy(), days.copy()
    if how == "room above energy":
        cars.loc[0, "room_kwh"] = "99.000000"
    elif how == "room not energy minus before":
        cars.loc[0, "energy_before_kwh"] = "0.000000"
    elif how == "depart hour off by one":
        cars.loc[0, "depart_hour"] = str(int(cars.loc[0, "depart_hour"]) + 1)
    elif how == "car twice on a day":
        cars = pd.concat([cars, cars.iloc[[0]]], ignore_index=True)
    elif how == "counting day with two cars":
        days.loc[days["counts"] == "True", "cars"] = "2"
    elif how == "time without offset":
        cars.loc[0, "plug_in"] = cars.loc[0, "plug_in"][:19]
    elif how == "car not present at tau":
        cars.loc[0, "plug_out"] = cars.loc[0, "tau"]
    elif how == "rate zero":
        cars.loc[0, "rate_kw"] = "0.000000"
    elif how == "in-window flag wrong":
        cars.loc[0, "in_window_departure"] = (
            "False" if cars.loc[0, "in_window_departure"] == "True" else "True"
        )
    elif how == "tau at another hour":
        days.loc[0, "tau"] = days.loc[0, "tau"].replace("T10:", "T11:")
    return cars, days


@pytest.mark.parametrize(
    ("how", "message"),
    [
        ("room above energy", "room outside 0 to the session energy"),
        ("room not energy minus before", "room is not energy minus energy before τ"),
        ("depart hour off by one", "depart_hour is not floor((plug-out - τ) / 1 hour)"),
        ("car twice on a day", "a session is listed twice on one day"),
        (
            "counting day with two cars",
            "a day counts without at least 3 cars, or has 3 and does not count",
        ),
        ("time without offset", "plug_in: 1 times without an offset or unreadable"),
        ("car not present at tau", "1 cars not plugged in at τ"),
        ("rate zero", "rate_kw is not positive everywhere"),
        ("in-window flag wrong", "in_window_departure is not depart_hour <= |S|"),
        ("tau at another hour", "days: τ not at the window's clock hour"),
    ],
)
def test_each_broken_rule_is_caught(built, how, message):
    _, out, _ = built
    cars, days = _read(out, "caltech")
    assert truth_verify.check_tables(cars, days, {"caltech": 10}) == []
    broken_cars, broken_days = _break(cars, days, how)
    assert message in truth_verify.check_tables(broken_cars, broken_days, {"caltech": 10})


def test_room_after_recorded_charging_ended_is_caught(built):
    # NREL: the made-up car 2 of each day has charging recorded 09:00 to 09:30, before τ 10:00,
    # so it drew all of its energy before τ. A table giving it room fails.
    _, out, root = built
    unit = "nrel-from-may-2019"
    cars, days = _read(out, unit)
    ends = truth_verify.nrel_charge_ends(root)
    assert truth_verify.check_tables(cars, days, {"nrel": 10}, ends) == []
    ended = cars["session_id"] == "row-7"
    assert ended.sum() == 1 and cars.loc[ended, "room_kwh"].item() == "0.000000"
    broken = cars.copy()
    broken.loc[ended, "energy_before_kwh"] = "6.100000"
    broken.loc[ended, "room_kwh"] = "3.400000"
    message = "1 cars keep room although their recorded charging ended at or before τ"
    assert truth_verify.check_tables(broken, days, {"nrel": 10}, ends) == [message]
    assert truth_verify.check_tables(broken, days, {"nrel": 10}) == []


def test_changed_table_fails_against_the_recorded_hash(built, tmp_path):
    summary, out, root = built
    copy = tmp_path / "tables"
    copy.mkdir()
    for path in out.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    with (copy / "jpl-cars.csv").open("a") as handle:
        handle.write("\n")
    status, lines = truth_verify.verify_role("train", summary, copy, root)
    assert status == 1 and any(line.startswith("FAILED: jpl") for line in lines)


def test_missing_table_cannot_run(built, tmp_path):
    summary, _, root = built
    with pytest.raises(truth_verify.CouldNotRun):
        truth_verify.verify_role("train", summary, tmp_path / "empty", root)


def test_build_at_another_commit_cannot_run(monkeypatch, tmp_path):
    monkeypatch.setattr(truth_verify, "read_verified", lambda path: ({"commit": "a" * 40}, b"{}"))
    with pytest.raises(truth_verify.CouldNotRun):
        truth_verify.load_summary(tmp_path, "b" * 40)


def test_no_raw_data_cannot_run(monkeypatch, tmp_path):
    monkeypatch.setattr(truth_verify, "RAW_ROOT", tmp_path / "data" / "raw")
    assert truth_verify.main(["train"]) == 2


def test_no_recorded_build_cannot_run(tmp_path):
    with pytest.raises(truth_verify.CouldNotRun):
        truth_verify.latest_run(tmp_path)

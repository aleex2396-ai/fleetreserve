"""The common session table's shared rules: the raw-data path guard, the table check, drops."""

import numpy as np
import pandas as pd
import pytest
from fake_raw import raw_root

from fleetreserve.sessions import (
    COLUMNS,
    Dropper,
    OutsideRawData,
    finish,
    require_raw_path,
)


def test_path_inside_raw_is_accepted(tmp_path):
    root = raw_root(tmp_path)
    (root / "nrel").mkdir()
    target = root / "nrel" / "file.csv"
    target.write_text("x")
    assert require_raw_path(target, root) == target.resolve()


def test_path_outside_raw_is_refused(tmp_path):
    root = raw_root(tmp_path)
    beside = tmp_path / "data" / "other" / "file.csv"
    beside.parent.mkdir(parents=True)
    beside.write_text("x")
    with pytest.raises(OutsideRawData):
        require_raw_path(beside, root)
    with pytest.raises(OutsideRawData):
        require_raw_path(root / ".." / "other" / "file.csv", root)


def test_link_inside_raw_pointing_outside_is_refused(tmp_path):
    root = raw_root(tmp_path)
    outside = tmp_path / "elsewhere.csv"
    outside.write_text("x")
    link = root / "link.csv"
    link.symlink_to(outside)
    with pytest.raises(OutsideRawData):
        require_raw_path(link, root)


def test_root_that_is_not_data_raw_is_refused(tmp_path):
    other = tmp_path / "data" / "notraw"
    other.mkdir(parents=True)
    with pytest.raises(OutsideRawData):
        require_raw_path(other / "file.csv", other)


def _table(**changes) -> pd.DataFrame:
    zone = "Europe/Oslo"
    base = {
        "site": ["A", "A"],
        "role": ["train", "train"],
        "session_id": ["s2", "s1"],
        "user_id": ["u", None],
        "station": [None, None],
        "plug_in": pd.to_datetime(["2020-01-01 18:00", "2020-01-01 17:00"]).tz_localize(zone),
        "plug_out": pd.to_datetime(["2020-01-02 07:00", "2020-01-02 06:00"]).tz_localize(zone),
        "charge_start": pd.Series([pd.NaT, pd.NaT], dtype=f"datetime64[us, {zone}]"),
        "charge_end": pd.Series([pd.NaT, pd.NaT], dtype=f"datetime64[us, {zone}]"),
        "energy_kwh": [5.0, 6.0],
        "max_power_kw": [np.nan, np.nan],
        "declared_departure": pd.Series([pd.NaT, pd.NaT], dtype=f"datetime64[us, {zone}]"),
        "declared_energy_kwh": [np.nan, np.nan],
        "nominal_kw": [7.4, 7.4],
        "nominal_rule": ["source cap", "source cap"],
    }
    base.update(changes)
    return pd.DataFrame(base)


def test_finish_sorts_by_plug_in_and_keeps_columns():
    out = finish(_table())
    assert list(out.columns) == list(COLUMNS)
    assert list(out["session_id"]) == ["s1", "s2"]


@pytest.mark.parametrize(
    "changes",
    [
        {"energy_kwh": [5.0, 0.0]},
        {"role": ["train", "test"]},
        {"session_id": ["s1", "s1"]},
        {"nominal_kw": [7.4, -1.0]},
        {
            "plug_out": pd.to_datetime(["2020-01-01 18:00", "2020-01-02 06:00"]).tz_localize(
                "Europe/Oslo"
            )
        },
        {"plug_in": pd.to_datetime(["2020-01-01 18:00", "2020-01-01 17:00"])},
    ],
)
def test_finish_refuses_a_broken_rule(changes):
    with pytest.raises(ValueError):
        finish(_table(**changes))


def test_dropper_records_every_removed_row_with_its_reason():
    frame = pd.DataFrame({"x": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
    dropper = Dropper(
        pd.Series(["A"] * 4, index=frame.index),
        pd.Series(["a", "b", "c", "d"], index=frame.index),
        pd.Series([None] * 4, index=frame.index),
    )
    kept = dropper.drop(frame, frame["x"] > 2, "too big")
    kept = dropper.drop(kept, kept["x"] == 1, "one")
    assert list(kept["x"]) == [2]
    table = dropper.table()
    assert list(table["session_id"]) == ["c", "d", "a"]
    assert list(table["reason"]) == ["too big", "too big", "one"]

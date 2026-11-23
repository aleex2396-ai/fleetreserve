"""`heldout_truth_build` on a held-out file in which every row of a unit is dropped by the
cleaning rules: a result about the data (the unit abstains, exit 0), not a reading fault. A unit
with no row at all in the held-out files stays a reading fault (exit 2).

The sets are the made-up ones of `heldout_truth_fake`, held in memory (`load_sets` replaced).
A loader now refuses a Norwegian site or a.s.r. period whose rows are all removed, or that has no
row (`test_load_*_units.py`, `test_heldout_unit_refusal.py`); a Korean portfolio of fast-charger
rows only still abstains. So the tests below that give the command such a Norwegian or a.s.r. table
read it with `heldout_truth_fake.unchecked` (the loader without its unit check): they test the
command's guard for tables that did not come from a loader.

1. TRO_R with every row without energy (36 rows): exit 0, tables for the other units, TRO_R
   recorded under `abstained` with the reason "every row dropped by cleaning: 36 rows, by reason
   energy missing or not positive: 36", no session, its rows removed by reason; the report names
   it among the findings.
2. The same with two reasons among the dropped rows: the reason lists each, in the order of
   their names.
3. a.s.r. with every row dropped (both units): exit 0, each unit abstains with its own rows
   counted; a.s.r. with no row at all stays exit 2.
   a.s.r. with rows in one period only, none (kept or dropped) in the other: exit 2 with the
   reading-fault line naming the empty period, nothing written.
4. TRO_R with no row at all stays exit 2 with the reading-fault text and nothing written.
"""

import json
from pathlib import Path

import pandas as pd
from fake_heldout import ASR_FILE, asr_csv
from fake_korea import KOREA_FILE
from heldout_truth_fake import (
    NORWAY_DAY,
    NORWAY_START,
    OSL_T_DAYS,
    TRO_R_DAYS,
    asr_rows,
    norway_bytes,
    norway_days,
    norway_rows,
    sealed_files,
    unchecked,
)

import fleetreserve
from fleetreserve import (
    gate_config,
    heldout_loader,
    load_asr,
    load_heldout_norway,
    load_korea,
    truth_build,
)
from fleetreserve import heldout_truth_build as h

MATRIX = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
RULE = h.SplitRule.of(gate_config.load(MATRIX), "gate/matrix.json")
ENERGY = "energy missing or not positive"
FAULT = "{unit}: no session in the held-out files: a reading fault, not a result"
NO_ENERGY = [(user, start, end, 0.0) for user, start, end, _ in NORWAY_DAY]


def _sets(tmp_path):
    files = sealed_files(tmp_path / "made")
    return {
        "norway": load_heldout_norway.from_bytes(files[heldout_loader.NORWAY]),
        "asr": load_asr.from_bytes(files[ASR_FILE]),
        "korea": load_korea.from_bytes(files[KOREA_FILE]),
    }


def _norway(tmp_path, tro_r_rows):
    rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS)) + tro_r_rows
    return unchecked(load_heldout_norway, norway_bytes(tmp_path / "norway", rows))


def _run(tmp_path, monkeypatch, loaded, capsys):
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    out = tmp_path / "out"
    monkeypatch.setattr(h, "load_sets", lambda _root: loaded)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    status = h.main(["--matrix", str(MATRIX), "--out", str(out)], root=root)
    return status, capsys.readouterr().out, out, run_dir


def _summary(run_dir):
    return json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())


def test_a_unit_whose_rows_are_all_dropped_abstains_with_a_true_reason(
    tmp_path, monkeypatch, capsys
):
    loaded = _sets(tmp_path)
    tro_r = norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS), pattern=NO_ENERGY)
    loaded["norway"] = _norway(tmp_path, tro_r)
    status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
    assert status == 0
    assert "reading fault" not in printed
    summary = _summary(run_dir)
    reason = f"every row dropped by cleaning: 36 rows, by reason {ENERGY}: 36"
    assert summary["abstained"]["TRO_R"] == {
        "reason": reason,
        "sessions_loaded": 0,
        "rows_removed": {ENERGY: 36},
    }
    assert "TRO_R" not in summary["units"] and "OSL_T" in summary["units"]
    assert (out / "OSL_T-days.csv").is_file() and not (out / "TRO_R-days.csv").exists()
    assert f"TRO_R: no tables, the unit abstains: {reason}" in printed


def test_the_reason_lists_each_reason_of_the_dropped_rows_in_the_order_of_their_names(tmp_path):
    loaded = _sets(tmp_path)
    tro_r = norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS), pattern=NO_ENERGY)
    loaded["norway"] = _norway(tmp_path, tro_r)
    sessions, dropped = loaded["norway"]
    extra = dropped[dropped["site"] == "TRO_R"].iloc[:2].copy()
    extra["reason"] = "a second reason"
    loaded["norway"] = (sessions, pd.concat([dropped, extra], ignore_index=True))
    result = h.build_heldout(loaded, "main", RULE)
    assert result.abstained["TRO_R"]["reason"] == (
        f"every row dropped by cleaning: 38 rows, by reason a second reason: 2, {ENERGY}: 36"
    )


def test_an_asr_file_with_every_row_dropped_abstains_both_units(tmp_path, monkeypatch, capsys):
    loaded = _sets(tmp_path)
    all_zero = [{**row, "total_energy": 0.0} for row in asr_rows()]
    loaded["asr"] = unchecked(load_asr, asr_csv(all_zero))
    status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
    assert status == 0
    assert "reading fault" not in printed
    abstained = _summary(run_dir)["abstained"]
    for unit in load_asr.UNITS:
        assert abstained[unit]["reason"].startswith("every row dropped by cleaning: ")
        removed = abstained[unit]["rows_removed"][ENERGY]
        assert abstained[unit]["reason"].endswith(f"by reason {ENERGY}: {removed}")
        assert abstained[unit]["sessions_loaded"] == 0


def test_an_asr_file_with_no_row_at_all_is_still_a_reading_fault(tmp_path, monkeypatch, capsys):
    loaded = _sets(tmp_path)
    sessions, dropped = loaded["asr"]
    loaded["asr"] = (sessions.iloc[0:0], dropped.iloc[0:0])
    status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
    assert status == 2
    for unit in load_asr.UNITS:
        assert FAULT.format(unit=unit) in printed
    assert not out.exists() and list(run_dir.iterdir()) == []


def test_an_asr_period_with_no_row_while_the_other_has_sessions_is_a_reading_fault(
    tmp_path, monkeypatch, capsys
):
    for empty, kept in (
        ("asr-before-march-2023", lambda row: row["start_datetime"] >= "2023-03-01"),
        ("asr-from-march-2023", lambda row: row["start_datetime"] < "2023-03-01"),
    ):
        loaded = _sets(tmp_path)
        loaded["asr"] = unchecked(load_asr, asr_csv([row for row in asr_rows() if kept(row)]))
        status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
        assert status == 2
        assert printed.startswith("COULD NOT RUN: " + FAULT.format(unit=empty))
        other = next(unit for unit in load_asr.UNITS if unit != empty)
        assert other not in printed and "nothing written" in printed
        assert not out.exists() and list(run_dir.iterdir()) == []


def test_a_norway_site_with_no_row_at_all_is_still_a_reading_fault(tmp_path, monkeypatch, capsys):
    loaded = _sets(tmp_path)
    loaded["norway"] = _norway(tmp_path, [])
    status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
    assert status == 2
    assert printed.startswith("COULD NOT RUN: " + FAULT.format(unit="TRO_R"))
    assert "every row dropped" not in printed
    assert not out.exists() and list(run_dir.iterdir()) == []

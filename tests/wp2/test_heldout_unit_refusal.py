"""`heldout_truth_build` on made-up sealed repositories in which one named unit has no row, or only
rows that the loader removes: the loader refuses the layout, so the command exits 1, logs the
opening and the refusal (the parsing-fix route of plan section 13 applies) and writes nothing.

The exit 2 of the command for a unit with no row at all (`heldout_truth_build.NO_SESSION`) is
reached by no unit of a file that went through the gated routes: each loader refuses first. The
command keeps that exit as a guard for tables that did not come from a loader; its own tests
(`test_heldout_no_session.py`) give it such tables.

A Korean portfolio of fast-charger rows only still abstains, exit 0.
"""

import json

import pytest
from fake_heldout import ASR_FILE, asr_csv, sealed_repo
from fake_korea import KOREA_FILE
from fake_raw import NORWAY_HEADER
from heldout_truth_fake import (
    KOREA_PATTERNS,
    NORWAY_START,
    OSL_T_DAYS,
    asr_rows,
    korea_bytes,
    korea_rows,
    norway_bytes,
    norway_days,
    norway_rows,
    sealed_files,
)

from fleetreserve import heldout_loader, truth_build
from fleetreserve import heldout_truth_build as h
from fleetreserve.heldout_loader import NORWAY
from fleetreserve.sessions import TOP

MATRIX = TOP / "gate" / "matrix.json"
SETS = {"norway": 1, "asr": 2, "korea": 3}  # the openings made when the set refuses


def _files(tmp_path, set_name, kind):
    files = sealed_files(tmp_path / "made")
    if set_name == "norway":
        rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS))
        if kind == "all removed":
            rows += norway_rows(
                "TRO_R", norway_days(NORWAY_START, 3), pattern=[("u", 15.5, 22.5, 0.0)]
            )
        files[NORWAY] = norway_bytes(tmp_path / "n", rows, header=NORWAY_HEADER)
    elif set_name == "asr":
        rows = [row for row in asr_rows() if row["start_datetime"] < "2023-03-01"]
        if kind == "all removed":
            rows += [
                {**row, "total_energy": 0.0}
                for row in asr_rows()
                if row["start_datetime"] >= "2023-03-01"
            ]
        files[ASR_FILE] = asr_csv(rows)
    else:
        patterns = {k: v for k, v in KOREA_PATTERNS.items() if k != "korea-visits"}
        rows = korea_rows(patterns)
        if kind == "all removed":
            rows += [
                {**row, "Demand": 0.0}
                for row in korea_rows({"korea-visits": KOREA_PATTERNS["korea-visits"]})
            ]
        files[KOREA_FILE] = korea_bytes(rows)
    return files


def _run(root, monkeypatch, run_dir):
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    return h.main(["--matrix", str(MATRIX)], root=root)


@pytest.mark.parametrize("kind", ["no row", "all removed"])
@pytest.mark.parametrize("set_name", list(SETS))
def test_a_unit_with_no_usable_row_is_a_logged_layout_refusal_and_the_command_exits_1(
    tmp_path, monkeypatch, capsys, set_name, kind
):
    root = sealed_repo(tmp_path, _files(tmp_path, set_name, kind))
    run_dir = tmp_path / "run"
    assert _run(root, monkeypatch, run_dir) == 1
    printed = capsys.readouterr().out
    assert printed.startswith(f"FAILED: the {set_name} file does not have its published layout: ")
    assert "COULD NOT RUN" not in printed and "reading fault" not in printed
    assert f"openings logged by this run: {SETS[set_name]}" in printed
    refusals = heldout_loader.logged_refusals(root)
    assert [line["set"] for line in refusals] == [set_name]
    assert "has no row kept" in refusals[0]["layout"]
    assert refusals[0]["layout"] in printed
    assert len(heldout_loader.logged_openings(root)) == SETS[set_name]
    assert list(run_dir.iterdir()) == [] and not (root / "data" / "derived").exists()


def test_a_portfolio_of_fast_charger_rows_only_still_abstains_with_exit_0(
    tmp_path, monkeypatch, capsys
):
    files = sealed_files(tmp_path / "made")
    patterns = {k: v for k, v in KOREA_PATTERNS.items() if k != "korea-visits"}
    rows = korea_rows(patterns)
    fast = [
        {**row, "ChargerType": "1"}
        for row in korea_rows({"korea-visits": KOREA_PATTERNS["korea-visits"]})
    ]
    files[KOREA_FILE] = korea_bytes(rows + fast)
    root = sealed_repo(tmp_path, files)
    run_dir = tmp_path / "run"
    assert _run(root, monkeypatch, run_dir) == 0
    summary = json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())
    assert "korea-visits" in summary["abstained"]
    assert heldout_loader.logged_refusals(root) == []

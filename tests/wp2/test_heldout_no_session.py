"""`heldout_truth_build` on a held-out file in which a unit has no session at all: a reading
fault, not a result (exit 2), while a unit that abstains for its window search stays an abstention.

The sets are the made-up ones of `heldout_truth_fake`, held in memory (`load_sets` replaced).
Each loader now refuses a file in which a named unit has no row (`test_load_*_units.py`,
`test_heldout_unit_refusal.py`: exit 1, logged), so no file read through a loader reaches this
exit 2. The command keeps it as a guard for tables that did not come from a loader; these tests
give it such tables (`heldout_truth_fake.unchecked`: the loader without its unit check).

1. A Norway site with no session (TRO_R taken out of the file), and a.s.r. with no session (both
   a.s.r. units): exit 2, one line per unit "<unit>: no session in the held-out files: a reading
   fault, not a result", and nothing written.
2. The made-up sets as they are (`korea-lodging` and `korea-visits` abstain for their window
   search: fewer than 2 calibration days): exit 0, both recorded as abstained with that reason, as
   before.
"""

import json
from pathlib import Path

from fake_heldout import ASR_FILE
from fake_korea import KOREA_FILE
from heldout_truth_fake import (
    NORWAY_START,
    OSL_T_DAYS,
    norway_bytes,
    norway_days,
    norway_rows,
    sealed_files,
    unchecked,
)

import fleetreserve
from fleetreserve import heldout_loader, load_asr, load_heldout_norway, load_korea, truth_build
from fleetreserve import heldout_truth_build as h

MATRIX = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
FAULT = "{unit}: no session in the held-out files: a reading fault, not a result"


def _sets(tmp_path):
    files = sealed_files(tmp_path / "made")
    return {
        "norway": load_heldout_norway.from_bytes(files[heldout_loader.NORWAY]),
        "asr": load_asr.from_bytes(files[ASR_FILE]),
        "korea": load_korea.from_bytes(files[KOREA_FILE]),
    }


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


def test_a_norway_site_with_no_session_is_a_reading_fault(tmp_path, monkeypatch, capsys):
    loaded = _sets(tmp_path)
    rows = norway_rows("OSL_T", norway_days(NORWAY_START, OSL_T_DAYS))
    loaded["norway"] = unchecked(load_heldout_norway, norway_bytes(tmp_path, rows))
    status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
    assert status == 2
    assert printed.startswith("COULD NOT RUN: " + FAULT.format(unit="TRO_R"))
    assert not out.exists() and list(run_dir.iterdir()) == []


def test_an_asr_file_with_no_session_is_a_reading_fault(tmp_path, monkeypatch, capsys):
    loaded = _sets(tmp_path)
    sessions, dropped = loaded["asr"]
    loaded["asr"] = (sessions.iloc[0:0], dropped.iloc[0:0])
    status, printed, out, run_dir = _run(tmp_path, monkeypatch, loaded, capsys)
    assert status == 2
    for unit in load_asr.UNITS:
        assert FAULT.format(unit=unit) in printed
    assert not out.exists() and list(run_dir.iterdir()) == []


def test_a_window_search_abstention_stays_an_abstention(tmp_path, monkeypatch, capsys):
    status, _, out, run_dir = _run(tmp_path, monkeypatch, _sets(tmp_path), capsys)
    assert status == 0
    summary = json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())
    assert set(summary["abstained"]) == {"korea-lodging", "korea-visits"}
    for entry in summary["abstained"].values():
        assert entry["reason"] != "no session"
    assert out.exists()

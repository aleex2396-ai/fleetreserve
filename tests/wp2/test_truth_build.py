"""The truth build command on a made-up raw folder holding every source: units, outputs,
repeatability, and its exit statuses."""

import json

import pytest
from fake_raw import NORWAY_SITES, full_tree, write_nrel

from fleetreserve import truth_build


@pytest.fixture
def built(tmp_path, monkeypatch):
    root = full_tree(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(run_dir))
    out = tmp_path / "out"
    status = truth_build.main(["--raw-root", str(root), "--out", str(out)])
    return status, root, out, run_dir


def test_every_unit_is_built_with_its_tables(built):
    status, _, out, run_dir = built
    assert status == 0
    summary = json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())
    expected = set(truth_build.TRAIN_UNITS) | set(truth_build.DEVELOPMENT_UNITS)
    assert set(summary["units"]) == expected
    for unit in expected:
        assert (out / f"{unit}-cars.csv").is_file() and (out / f"{unit}-days.csv").is_file()
        entry = summary["units"][unit]
        assert set(entry["fallback_shares"]) == set(truth_build.FALLBACK_KEYS)
        assert entry["counting_days"] >= 1
    assert (run_dir / truth_build.REPORT_NAME).read_text().startswith("FINDINGS")


def test_unit_numbers_on_the_made_up_data(built):
    _, _, _, run_dir = built
    units = json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())["units"]
    # ACN: three cars plugged in at τ on 1 May 2019, one without a user id.
    assert units["caltech"]["counting_days"] == 1
    assert units["caltech"]["fallback_shares"]["rate_nominal_cap_no_user_id"] == round(1 / 3, 6)
    # NREL: three cars on 10 April 2019 (before the controller) and three on 15 May 2019; the
    # 2018 row has no plug-in or plug-out time. One car a day has no charge fields.
    before, after = units["nrel-before-may-2019"], units["nrel-from-may-2019"]
    assert (before["counting_days"], after["counting_days"]) == (1, 1)
    assert before["rows_removed"] == {"no plug-in or plug-out time recorded": 1}
    assert after["rows_removed"] == {}
    assert before["fallback_shares"]["nominal_cap_floor_max_charge_power_missing"] == round(
        1 / 3, 6
    )
    # Norway: two counting days per site; the pooled unit is the union of the five sites.
    for site in NORWAY_SITES["development"]:
        assert units[site]["counting_days"] == 2
    pooled = units[truth_build.POOLED]
    assert pooled["counting_days"] == 10
    assert pooled["sites"] == sorted(NORWAY_SITES["development"])
    assert pooled["sessions_loaded"] == 40
    sites = NORWAY_SITES["development"]
    assert pooled["cars_on_counting_days"] == sum(
        units[site]["cars_on_counting_days"] for site in sites
    )
    assert pooled["cars_on_counting_days"] == 30


def test_windows_reported_per_site(built):
    _, _, _, run_dir = built
    windows = json.loads((run_dir / truth_build.SUMMARY_NAME).read_text())["windows"]
    assert windows["caltech"]["tau"] == "08:00" and windows["caltech"]["service_hours"] == 3
    assert windows["ASK"]["tau"] == "17:00" and windows["ASK"]["service_hours"] == 8


def test_same_inputs_give_the_same_bytes(built, tmp_path, monkeypatch):
    _, root, out, run_dir = built
    again_run = tmp_path / "run2"
    again_run.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(again_run))
    again = tmp_path / "out2"
    assert truth_build.main(["--raw-root", str(root), "--out", str(again)]) == 0
    first = sorted(path.name for path in out.iterdir())
    assert first == sorted(path.name for path in again.iterdir())
    for name in first:
        assert (out / name).read_bytes() == (again / name).read_bytes()
    for name in (truth_build.SUMMARY_NAME, truth_build.REPORT_NAME):
        assert (run_dir / name).read_bytes() == (again_run / name).read_bytes()


def test_missing_raw_file_cannot_run(tmp_path, monkeypatch):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    root = full_tree(tmp_path)
    (root / "nrel" / "dataSet_Oct2021.csv").unlink()
    assert truth_build.main(["--raw-root", str(root), "--out", str(tmp_path / "o")]) == 2


def test_wrong_layout_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)
    root = full_tree(tmp_path)
    write_nrel(root, [], header=("driverId", "station"))
    assert truth_build.main(["--raw-root", str(root), "--out", str(tmp_path / "o")]) == 1

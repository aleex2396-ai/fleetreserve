"""The fallback-share item of the WP2 pass mark, `tools/wp2_fallback_check.py`, on made-up truth
summaries.

A share the truth build wrote as `null` passes only when its denominator, rebuilt from the same
unit's entry and cars table, is exactly 0 (an undefined share, 0 of 0); a share that is missing,
not a number or outside 0 to 1 fails, as in `truth_verify.verify_fallbacks`. The history share's
denominator is counted exactly from the unit's cars table (rows of a counting day whose rate came
from history), after the table's hash is compared with the recorded one. The job's own exit
statuses are checked with the recorded-build lookups of `truth_verify` replaced, so no data or
run is read.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from fleetreserve import truth, truth_build, truth_verify

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "wp2_fallback_check.py"
_spec = importlib.util.spec_from_file_location("wp2_fallback_check", SCRIPT)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

UNITS = truth_build.UNITS["train"] + truth_build.UNITS["development"]
UNDEFINED_LINE = (
    "rate_average_without_charging_time undefined: 0 of 0 "
    "(counting cars whose rate came from history)"
)


def good_unit(**changes):
    entry = {
        "sessions_loaded": 500,
        "cars_on_counting_days": 31,
        "fallback_shares": {
            "rate_nominal_cap_no_user_id": 0.25,
            "sessions_missing_user_id": 0.1,
            "nominal_cap_floor_max_charge_power_missing": 0.0,
            "energy_before_no_charging_time": 0.5,
            "rate_average_without_charging_time": 0.75,
        },
    }
    entry.update(changes)
    return entry


def summary_of(**units):
    document = {"units": {unit: good_unit() for unit in UNITS}}
    document["units"].update(units)
    return document


def shares_of(**shares):
    entry = good_unit()
    entry["fallback_shares"].update(shares)
    return entry


NO_TABLES = Path("no-tables-folder")  # a test that reads a table it should not read cannot run
HISTORY = truth.RATE_HISTORY
FALLBACK = truth.RATE_FALLBACK


def judge(summary, tables=NO_TABLES):
    return check.verify_fallbacks(summary, tables)


def write_cars(folder, unit, rows, entry):
    """Write `<unit>-cars.csv` with one row per (day_counts, rate_rule) of `rows` and record its
    SHA-256 in the unit's `entry`, as the truth build does."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = ["day_counts,rate_rule"] + [f'{counts},"{rule}"' for counts, rule in rows]
    data = ("\n".join(lines) + "\n").encode()
    (folder / f"{unit}-cars.csv").write_bytes(data)
    entry["sha256"] = {"cars": hashlib.sha256(data).hexdigest(), "days": "0" * 64}


def office001_case(tables, rows=None):
    """The unit that raised the item: 31 cars on counting days, every rate from the nominal cap,
    so no car's rate came from history and the share over those cars is null. `rows` replaces the
    cars table's rows."""
    entry = shares_of(rate_nominal_cap_no_user_id=1.0, rate_average_without_charging_time=None)
    rows = rows if rows is not None else [("True", FALLBACK)] * 31
    write_cars(tables, "office001", rows, entry)
    return entry


def test_every_share_a_number_passes_with_the_line_verify_fallbacks_gives():
    summary = summary_of()
    status, lines = judge(summary)
    assert status == 0, lines
    assert (status, lines) == truth_verify.verify_fallbacks(summary)


def test_office001_undefined_history_share_passes_and_is_shown(tmp_path):
    status, lines = judge(summary_of(office001=office001_case(tmp_path)), tmp_path)
    assert status == 0, lines
    assert any(line.startswith("passed: office001: ") for line in lines)
    assert any(UNDEFINED_LINE in line for line in lines)
    assert not any(line.startswith("FAILED") for line in lines)
    # the line is shown once, under the one unit that has an undefined share
    assert sum(UNDEFINED_LINE in line for line in lines) == 1


def test_null_history_share_with_one_counting_history_row_fails(tmp_path):
    rows = [("True", FALLBACK)] * 30 + [("True", HISTORY)]
    status, lines = judge(summary_of(office001=office001_case(tmp_path, rows)), tmp_path)
    assert status == 1
    assert any(
        line.startswith("FAILED: office001:") and "rate_average_without_charging_time" in line
        for line in lines
    )


def test_null_history_share_passes_when_history_rows_are_only_on_days_that_do_not_count(tmp_path):
    rows = [("True", FALLBACK)] * 31 + [("False", HISTORY)] * 3
    status, lines = judge(summary_of(office001=office001_case(tmp_path, rows)), tmp_path)
    assert status == 0, lines
    assert any(UNDEFINED_LINE in line for line in lines)


def test_a_third_rate_rule_with_no_history_row_still_counts_only_history_rows(tmp_path):
    # Shows that the count is of rows whose rule is exactly the history rule: a row with an
    # unknown rule text is neither a history row nor an error, so with no history row the
    # denominator is 0 and the null passes; the same table with one history row fails.
    other = "an unknown rule text"
    rows = [("True", FALLBACK)] * 20 + [("True", other)] * 11
    status, _ = judge(summary_of(office001=office001_case(tmp_path, rows)), tmp_path)
    assert status == 0
    status, _ = judge(
        summary_of(office001=office001_case(tmp_path, [*rows, ("True", HISTORY)])), tmp_path
    )
    assert status == 1


def test_null_history_share_with_a_table_that_differs_from_the_recorded_hash_fails(tmp_path):
    entry = office001_case(tmp_path)
    entry["sha256"]["cars"] = "f" * 64
    status, lines = judge(summary_of(office001=entry), tmp_path)
    assert status == 1
    assert any(
        line.startswith("FAILED: office001:") and "differs from the recorded" in line
        for line in lines
    )
    # a table changed after the build, in the way the null would then pass: also fails
    entry = office001_case(tmp_path)
    (tmp_path / "office001-cars.csv").write_bytes(b"day_counts,rate_rule\nFalse,x\n")
    assert judge(summary_of(office001=entry), tmp_path)[0] == 1


def test_null_history_share_with_a_missing_table_could_not_run(tmp_path):
    entry = office001_case(tmp_path)
    (tmp_path / "office001-cars.csv").unlink()
    with pytest.raises(truth_verify.CouldNotRun, match="office001-cars.csv is missing"):
        judge(summary_of(office001=entry), tmp_path)


def test_a_table_without_the_rule_columns_fails(tmp_path):
    entry = office001_case(tmp_path)
    data = b"a,b\n1,2\n"
    (tmp_path / "office001-cars.csv").write_bytes(data)
    entry["sha256"]["cars"] = hashlib.sha256(data).hexdigest()
    assert judge(summary_of(office001=entry), tmp_path)[0] == 1


def test_a_table_is_not_read_when_no_history_share_is_null():
    # NO_TABLES does not exist: a read would end the check as "could not run"
    assert judge(summary_of(jpl=shares_of(sessions_missing_user_id=1.2)))[0] == 1


def test_a_missing_key_fails():
    entry = good_unit()
    del entry["fallback_shares"]["sessions_missing_user_id"]
    status, lines = judge(summary_of(jpl=entry))
    assert status == 1
    assert any(
        line.startswith("FAILED: jpl:") and "sessions_missing_user_id" in line for line in lines
    )


def test_a_share_above_one_fails():
    status, lines = judge(summary_of(jpl=shares_of(sessions_missing_user_id=1.2)))
    assert status == 1
    assert any(line.startswith("FAILED: jpl:") for line in lines)


def test_a_share_that_is_not_a_number_fails():
    for value in ("0.5", [0.5], {}, float("nan")):
        entry = shares_of(energy_before_no_charging_time=value)
        assert judge(summary_of(jpl=entry))[0] == 1, value


def test_no_fallback_shares_and_a_missing_unit_fail():
    entry = good_unit()
    del entry["fallback_shares"]
    status, lines = judge(summary_of(jpl=entry))
    assert status == 1 and any("no fallback shares logged" in line for line in lines)
    summary = summary_of()
    del summary["units"]["BOD"]
    assert judge(summary)[0] == 1


def test_a_share_over_the_sessions_is_undefined_only_without_sessions():
    shown = "sessions_missing_user_id undefined: 0 of 0 (sessions loaded)"
    entry = shares_of(sessions_missing_user_id=None)
    entry["sessions_loaded"] = 0
    status, lines = judge(summary_of(jpl=entry))
    assert status == 0, lines
    assert any(shown in line for line in lines)
    for loaded in (1, None, "0"):
        entry["sessions_loaded"] = loaded
        assert judge(summary_of(jpl=entry))[0] == 1, loaded
    del entry["sessions_loaded"]
    assert judge(summary_of(jpl=entry))[0] == 1


@pytest.mark.parametrize(
    "key",
    [
        "rate_nominal_cap_no_user_id",
        "nominal_cap_floor_max_charge_power_missing",
        "energy_before_no_charging_time",
    ],
)
def test_shares_over_counting_cars_are_undefined_only_without_counting_cars(key):
    entry = shares_of(**{key: None})
    entry["cars_on_counting_days"] = 0
    status, lines = judge(summary_of(jpl=entry))
    assert status == 0, lines
    assert any(f"{key} undefined: 0 of 0 (counting cars)" in line for line in lines)
    entry["cars_on_counting_days"] = 31
    assert judge(summary_of(jpl=entry))[0] == 1
    del entry["cars_on_counting_days"]
    assert judge(summary_of(jpl=entry))[0] == 1


# --- the job's exit statuses ------------------------------------------------------------------


@pytest.fixture
def lookups(monkeypatch, tmp_path):
    """The recorded-build lookups of `truth_verify`, replaced by a made-up run folder."""
    (tmp_path / "data" / "raw").mkdir(parents=True)
    run = tmp_path / "runs.noindex" / "2026-01-01-wp2-truth"
    run.mkdir(parents=True)
    monkeypatch.setattr(truth_verify, "TOP", tmp_path)
    monkeypatch.setattr(truth_verify, "RAW_ROOT", tmp_path / "data" / "raw")
    monkeypatch.setattr(truth_verify, "_head_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        truth_verify, "require_matching_setting", lambda summary, tables, name: truth_verify.MAIN
    )
    return tmp_path


def serve(monkeypatch, summary):
    monkeypatch.setattr(truth_verify, "load_summary", lambda run, head: summary)


def test_job_passes_office001_and_prints_the_undefined_line(lookups, monkeypatch, capsys):
    tables = lookups / "tables"
    serve(monkeypatch, summary_of(office001=office001_case(tables)))
    assert check.main(["--tables", str(tables)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "run: 2026-01-01-wp2-truth"
    assert UNDEFINED_LINE in out


def test_job_with_a_missing_cars_table_could_not_run(lookups, monkeypatch, capsys):
    tables = lookups / "tables"
    entry = office001_case(tables)
    (tables / "office001-cars.csv").unlink()
    serve(monkeypatch, summary_of(office001=entry))
    assert check.main(["--tables", str(tables)]) == 2
    assert "office001-cars.csv is missing" in capsys.readouterr().out


def test_job_with_a_changed_cars_table_fails(lookups, monkeypatch, capsys):
    tables = lookups / "tables"
    entry = office001_case(tables)
    entry["sha256"]["cars"] = "f" * 64
    serve(monkeypatch, summary_of(office001=entry))
    assert check.main(["--tables", str(tables)]) == 1
    assert "FAILED: office001:" in capsys.readouterr().out


def test_job_prints_the_same_lines_as_truth_verify_when_every_share_is_a_number(
    lookups, monkeypatch, capsys
):
    serve(monkeypatch, summary_of())
    assert check.main([]) == 0
    mine = capsys.readouterr().out
    assert truth_verify.main(["fallbacks"]) == 0
    assert mine == capsys.readouterr().out


def test_job_fails_with_exit_1_when_a_share_is_outside_0_to_1(lookups, monkeypatch, capsys):
    serve(monkeypatch, summary_of(jpl=shares_of(sessions_missing_user_id=1.2)))
    assert check.main([]) == 1
    assert "FAILED: jpl:" in capsys.readouterr().out


def test_job_with_no_recorded_summary_could_not_run(lookups, monkeypatch, capsys):
    (lookups / "runs.noindex" / "2026-01-01-wp2-truth").rmdir()
    assert check.main([]) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN: ")


def test_job_with_no_raw_folder_could_not_run(lookups, monkeypatch, capsys):
    monkeypatch.setattr(truth_verify, "RAW_ROOT", lookups / "data" / "no-raw")
    serve(monkeypatch, summary_of())
    assert check.main([]) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN: ")


def test_job_with_a_summary_of_another_commit_could_not_run(lookups, monkeypatch, capsys):
    def other_commit(run, head):
        raise truth_verify.CouldNotRun("the recorded truth build is at another commit")

    monkeypatch.setattr(truth_verify, "load_summary", other_commit)
    assert check.main([]) == 2
    assert "another commit" in capsys.readouterr().out


def test_job_with_a_summary_that_does_not_verify_fails(lookups, monkeypatch, capsys):
    def not_verified(run, head):
        raise truth_verify.NotVerified("hash differs")

    monkeypatch.setattr(truth_verify, "load_summary", not_verified)
    assert check.main([]) == 1
    assert capsys.readouterr().out.startswith("FAILED: the recorded truth build is not verified")


def test_job_with_a_summary_of_the_other_setting_could_not_run(lookups, monkeypatch, capsys):
    document = summary_of()
    document["setting"] = "held"
    serve(monkeypatch, document)
    assert check.main([]) == 2
    assert "setting" in capsys.readouterr().out


def test_the_summary_is_read_as_json_would_give_it(tmp_path):
    # a null written by the truth build comes back as None, a share as a float
    text = json.dumps(summary_of(office001=office001_case(tmp_path)))
    assert judge(json.loads(text), tmp_path)[0] == 0

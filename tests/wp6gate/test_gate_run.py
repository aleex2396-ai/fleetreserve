"""The gate run command (`gate_run`) on made-up gate units with a made-up verified truth build.

Units (`gate_fake.gate_units`): NREL before May 2019 with 10 days (not calibratable), NREL from May
2019 with 60 days, the pooled Norwegian unit with 60 site-days at two sites on the same dates. The
build is made once per module; the command runs once for the whole matrix (`full`, 1 worker), and
once more only where a test changes what the run does (a fault injected, `--limit-days`, the
number of workers). The matrix: the main family at kappa 0.1 under both readings of
`departure_seen`, each with its oracle partner.

1. The full run ends with exit 0 and writes every output; the small unit is reported "not
   calibratable: 10 days", gets no number, and passes neither rule in every gate decision; every
   day has the same role in the split under every configuration; T5 is checked on day-paths and
   no break is written.
2. Both readings of `departure_seen` are reported, each with its own kappa selection and decision
   lines, and they replay differently (the one-way rung's re-solves differ).
3. The results files are identical for 1 and 2 workers.
4. A trial (`--limit-days`) says "TRIAL, not a gate result" at the top of every output.
5. A T5 break injected through the replay seam is counted, written out and leads the summary; the
   run still ends with exit 0 (a break is a result, not a failure of the run).
6. Exit status 2 for a missing table, an unverified or stale truth build, a bad matrix and an
   unknown configuration; 1 when a configuration fails (the others still run).
"""

import csv
import dataclasses
import json
from pathlib import Path

import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    HOUR_START,
    ROUNDED,
    gate_units,
    matrix_document,
    write_build,
)

from fleetreserve import gate_calibrate, gate_day, gate_run, run_verify

NAMES = ROUNDED + HOUR_START
EMPIRICAL = (ROUNDED[0], HOUR_START[0])
OUTPUTS = (
    gate_run.RESULTS_NAME,
    gate_run.DAYS_NAME,
    gate_run.BREAKS_NAME,
    gate_run.SUMMARY_NAME,
    gate_run.TIMING_NAME,
)
COMPARED = OUTPUTS[:4]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def run_gate(build, out: Path, monkeypatch, *extra) -> int:
    run, tables, matrix = build
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    return gate_run.main([*arguments, *extra])


def _module_run(build, folder: Path, *extra, patch=None):
    """One run for a module fixture: (exit status, output folder)."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out = folder / "out"
        return run_gate(build, out, monkeypatch, *extra), out


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    return _module_run(build, tmp_path_factory.mktemp("full"), "--workers", "1")


def _rows(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


def _document(out: Path) -> dict:
    return json.loads((out / gate_run.RESULTS_NAME).read_text())


def test_a_full_run_writes_every_output(full):
    code, out = full
    assert code == 0
    for name in OUTPUTS:
        assert (out / name).is_file()
    document = _document(out)
    assert (document["trial"], document["partial"]) == (False, False)
    assert document["t5"]["day_paths_checked"] > 0
    assert document["t5"]["breaks_a"] == document["t5"]["breaks_b"] == 0
    assert _rows(out / gate_run.BREAKS_NAME) == []
    assert document["failed_configurations"] == []
    for name in NAMES:
        units = document["configurations"][name]["units"]
        for unit in GATE_UNITS[1:]:
            entry = units[unit]
            assert entry["calibration_days"] == 50 and entry["test_days"] == 10
            assert set(entry["rungs"]) == {"fixed", "one-way", "two-way"}
    text = (out / gate_run.SUMMARY_NAME).read_text()
    assert "PROVISIONAL" in text and "TRIAL" not in text
    assert text.splitlines()[0] == "Truth tables: main setting."
    assert text.splitlines()[1].startswith("T5 on every evaluated day-path:")
    assert "Decided by the project owner on 16 Nov 2026" in text
    assert "11 Nov 2026" not in text
    outcomes = [
        d["go"]
        for entry in document["gate"]["by_pair"].values()
        if entry["family"] == "main"
        for d in entry["decisions"]
    ] + [
        d["go"]
        for seen, entry in document["gate"]["kappa_selected_per_unit"].items()
        if seen == "rounded-hour"  # the main reading; hour-start is reported beside it
        for d in entry["decisions"]
    ]
    assert len(outcomes) == 1 * 6 + 1 * 6
    expected = (
        f"The main-family outcomes differ: {sum(outcomes)} go, {len(outcomes) - sum(outcomes)} "
        "no go;"
        if 0 < sum(outcomes) < len(outcomes)
        else f"All {len(outcomes)} main-family outcomes agree: "
    )
    assert expected in text


def test_a_unit_too_small_is_reported_and_passes_nothing(full):
    _, out = full
    document = _document(out)
    small = GATE_UNITS[0]
    assert document["units"][small]["status"] == "not calibratable: 10 days"
    assert not document["units"][small]["calibratable"]
    for name in NAMES:
        entry = document["configurations"][name]["units"][small]
        assert entry["reason"] == "not calibratable: 10 days" and "rungs" not in entry
    for name in EMPIRICAL:
        decisions = document["gate"]["by_pair"][name]["decisions"]
        assert len(decisions) == 2 * 3
        for decision in decisions:
            row = decision["units"][small]
            assert row == {"passes_share": False, "passes_ratio": False, "reason": row["reason"]}
            assert "not calibratable" in row["reason"]
            if decision["ratio_rule_on"] == "all_units":
                assert decision["go"] is False
    text = (out / gate_run.SUMMARY_NAME).read_text()
    assert f"{small} counts as not passing (not calibratable: 10 days)" in text
    assert "not decided" not in text
    small_rows = [r for r in _rows(out / gate_run.DAYS_NAME) if r["unit"] == small]
    assert {r["role"] for r in small_rows} == {"not calibratable"}
    assert all(r["one-way_offer_kwh"] == "" for r in small_rows)


def test_the_offering_counts_and_losses_match_the_day_rows(full):
    """Per rung, the test results are recounted from the per-day file: a day offers when its
    `abstain_reason` is empty, fails when it offers and its slot loss is above 0, and the loss
    means are over the offering test days."""
    _, out = full
    document = _document(out)
    rows = _rows(out / gate_run.DAYS_NAME)
    checked = offering_seen = 0
    for name in NAMES:
        for unit in GATE_UNITS[1:]:
            test_rows = [
                r
                for r in rows
                if r["configuration"] == name and r["unit"] == unit and r["role"] == "test"
            ]
            offering = [r for r in test_rows if r["abstain_reason"] == ""]
            for rung, found in document["configurations"][name]["units"][unit]["rungs"].items():
                test = found["test"]
                slot = [float(r[f"{rung}_slot_loss"]) for r in offering]
                driver = [float(r[f"{rung}_driver_loss"]) for r in offering]
                assert test["days"] == len(test_rows)
                assert test["offering_days"] == len(offering)
                if found["beta_hat"] > 0:
                    assert test["failing_days"] == sum(loss > 0 for loss in slot)
                    assert test["mean_slot_loss_offering_days"] == pytest.approx(
                        sum(slot) / len(slot) if slot else None
                    )
                    assert test["mean_driver_loss_offering_days"] == pytest.approx(
                        sum(driver) / len(driver) if driver else None
                    )
                else:  # nothing is offered at scale 0: no loss to average, and why
                    assert test["failing_days"] is None
                    assert test["mean_slot_loss_offering_days"] is None
                    assert test["mean_driver_loss_offering_days"] is None
                    assert found["reason"].startswith(("abstains: ", "no scale is safe: "))
                every = [float(r[f"{rung}_offer_kwh"]) for r in test_rows]
                assert test["mean_offer_kwh_per_day"] == pytest.approx(sum(every) / len(every))
                offers = [float(r[f"{rung}_offer_kwh"]) for r in offering]
                assert test["mean_offer_kwh_per_offering_day"] == pytest.approx(
                    sum(offers) / len(offers) if offers else None
                )
                checked += 1
                offering_seen += len(offering)
    assert checked == len(NAMES) * len(GATE_UNITS[1:]) * 3 and offering_seen > 0
    definitions = document["definitions"]
    assert "above zero_value_kwh (1e-06)" in definitions["offering_day"]
    assert "offering calibration days only" in definitions["evaluations"]
    text = (out / gate_run.SUMMARY_NAME).read_text()
    assert " offering, " in text and " failing" in text


def test_the_split_is_the_same_for_every_configuration(full):
    _, out = full
    rows = _rows(out / gate_run.DAYS_NAME)
    assert len(rows) == len(NAMES) * (10 + 60 + 60)
    roles = {}
    for r in rows:
        roles.setdefault((r["unit"], r["site"], r["date"]), set()).add(r["role"])
    assert all(len(found) == 1 for found in roles.values())
    pooled = [k for k in roles if k[0] == GATE_UNITS[2]]
    assert len(pooled) == 60 and len({k[2] for k in pooled}) == 30  # two sites per date


def test_both_readings_of_departure_seen_are_reported_and_replay_differently(full):
    _, out = full
    document = _document(out)
    pairs = document["gate"]["by_pair"]
    assert {name: pairs[name]["departure_seen"] for name in EMPIRICAL} == {
        ROUNDED[0]: "rounded-hour",
        HOUR_START[0]: "hour-start",
    }
    chosen = document["gate"]["kappa_selected_per_unit"]
    assert set(chosen) == {"hour-start", "rounded-hour"}
    assert chosen["rounded-hour"]["configuration_by_unit"][GATE_UNITS[1]] == ROUNDED[0]
    assert chosen["hour-start"]["configuration_by_unit"][GATE_UNITS[1]] == HOUR_START[0]
    text = (out / gate_run.SUMMARY_NAME).read_text()
    for seen in ("hour-start", "rounded-hour"):
        assert f"PROVISIONAL departure seen {seen}, kappa 0.1:" in text
        assert f"PROVISIONAL departure seen {seen}, kappa selected:" in text
    for unit in GATE_UNITS[1:]:
        found = [
            document["configurations"][name]["units"][unit]["rungs"]["one-way"]["resolves"]
            for name in EMPIRICAL
        ]
        assert found[0] != found[1]


def test_the_results_do_not_depend_on_the_number_of_workers(build, tmp_path, monkeypatch):
    found = {}
    for workers in ("1", "2"):
        out = tmp_path / workers
        assert run_gate(build, out, monkeypatch, "--only", ROUNDED[0], "--workers", workers) == 0
        found[workers] = {name: (out / name).read_bytes() for name in COMPARED}
    assert found["1"] == found["2"]
    document = json.loads(found["1"][gate_run.RESULTS_NAME])
    assert document["partial"] is True and document["label"] == gate_run.PARTIAL_TEXT


def test_a_trial_is_marked_in_every_output(build, tmp_path, monkeypatch):
    out = tmp_path / "trial"
    assert run_gate(build, out, monkeypatch, "--only", ROUNDED[0], "--limit-days", "55") == 0
    document = _document(out)
    assert document["trial"] is True and document["label"] == gate_run.TRIAL_TEXT
    for name in (gate_run.DAYS_NAME, gate_run.BREAKS_NAME):
        assert gate_run.TRIAL_TEXT in (out / name).read_text().splitlines()[0]
    # The summary's first line names the truth setting; the trial marker follows at once.
    summary = (out / gate_run.SUMMARY_NAME).read_text().splitlines()
    assert summary[0] == "Truth tables: main setting."
    assert gate_run.TRIAL_TEXT in summary[1]
    timing = json.loads((out / gate_run.TIMING_NAME).read_text())
    assert timing["label"] == gate_run.TRIAL_TEXT
    assert document["units"][GATE_UNITS[1]]["counting_days"] == 55


def test_an_injected_t5_break_is_counted_and_written(build, tmp_path, monkeypatch):
    real = gate_day.replay_day
    done = []

    def lowered(day, *args, **kwargs):
        result = real(day, *args, **kwargs)
        if done or day.unit != GATE_UNITS[1]:
            return result
        done.append(day.date.isoformat())
        two_way = list(result.by_rung["two-way"])
        final = result.by_rung["one-way"][0].final.copy()
        final[0] -= 0.5
        two_way[0] = dataclasses.replace(two_way[0], final=final)
        return dataclasses.replace(result, by_rung={**result.by_rung, "two-way": tuple(two_way)})

    monkeypatch.setattr(gate_day, "replay_day", lowered)
    out = tmp_path / "broken"
    assert run_gate(build, out, monkeypatch, "--only", ROUNDED[0]) == 0
    document = _document(out)
    assert (document["t5"]["breaks_a"], document["t5"]["breaks_b"]) == (0, 1)
    (row,) = _rows(out / gate_run.BREAKS_NAME)
    assert (row["unit"], row["date"], row["kind"], row["rung"]) == (
        GATE_UNITS[1],
        done[0],
        "b",
        "two-way",
    )
    assert float(row["predecessor_value"]) - float(row["value"]) == pytest.approx(0.5)
    summary = (out / gate_run.SUMMARY_NAME).read_text().splitlines()
    assert summary[0] == "Truth tables: main setting."
    assert summary[1].startswith("PARTIAL RUN") and summary[2].startswith("T5 BREAKS")


def test_a_missing_table_could_not_run(build, tmp_path, monkeypatch, capsys):
    run, tables, matrix = build
    other = tmp_path / "tables"
    other.mkdir()
    for path in tables.iterdir():
        if path.name != f"{GATE_UNITS[2]}-days.csv":
            (other / path.name).write_bytes(path.read_bytes())
    code = gate_run.main(["--matrix", str(matrix), "--tables", str(other), "--truth-run", str(run)])
    assert code == 2 and "missing" in capsys.readouterr().out


def test_a_changed_table_could_not_run(build, tmp_path, capsys):
    run, tables, matrix = build
    other = tmp_path / "tables"
    other.mkdir()
    for path in tables.iterdir():
        (other / path.name).write_bytes(path.read_bytes())
    path = other / f"{GATE_UNITS[1]}-cars.csv"
    path.write_text(path.read_text() + "\n")
    code = gate_run.main(["--matrix", str(matrix), "--tables", str(other), "--truth-run", str(run)])
    assert code == 2 and "differs" in capsys.readouterr().out


def test_a_stale_or_unverified_build_could_not_run(build, tmp_path, monkeypatch, capsys):
    run, tables, matrix = build
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    monkeypatch.setattr(gate_run, "_head_commit", lambda: "b" * 40)
    assert gate_run.main(arguments) == 2
    assert "not at the checked-out commit" in capsys.readouterr().out
    assert gate_run.main([*arguments[:4], "--truth-run", str(tmp_path / "none")]) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out


def test_a_bad_matrix_or_name_could_not_run(build, tmp_path, capsys):
    run, tables, matrix = build
    bad = tmp_path / "bad.json"
    document = matrix_document(NAMES)
    document["configurations"][0]["kappa"] = 0.15
    bad.write_text(json.dumps(document))
    base = ["--tables", str(tables), "--truth-run", str(run)]
    assert gate_run.main(["--matrix", str(bad), *base]) == 2
    assert gate_run.main(["--matrix", str(matrix), *base, "--only", "no-such"]) == 2
    assert gate_run.main(["--matrix", str(matrix), *base, "--workers", "0"]) == 2
    assert capsys.readouterr().out.count("COULD NOT RUN") == 3


def test_a_failed_configuration_exits_1_and_the_others_run(build, tmp_path, monkeypatch):
    real = gate_calibrate.run_job

    def failing(prepared, *args, **kwargs):
        if prepared[0].unit == GATE_UNITS[2]:
            raise RuntimeError("made-up failure")
        return real(prepared, *args, **kwargs)

    monkeypatch.setattr(gate_calibrate, "run_job", failing)
    out = tmp_path / "failed"
    assert run_gate(build, out, monkeypatch, "--only", ROUNDED[0]) == 1
    document = _document(out)
    assert document["failed_configurations"] == [
        f"{ROUNDED[0]} | {GATE_UNITS[2]}: RuntimeError: made-up failure"
    ]
    assert "rungs" in document["configurations"][ROUNDED[0]]["units"][GATE_UNITS[1]]
    assert "FAILED CONFIGURATIONS" in (out / gate_run.SUMMARY_NAME).read_text()

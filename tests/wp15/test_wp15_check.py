"""`tools/wp15_check.py`, the one command of WP15's pass mark, on sealed-run drives written by
the real driver (`run_driver`) through the REAL gate of the run from the tag, in throwaway git
repositories (`tests/wp12/drive_fake.py`) made ones of a made-up freeze
(`tests/wp2/fake_heldout.py`: `data/` ignored, a copy of the running package under
`src/fleetreserve/`, a made-up plan frozen at the tag `frozen-plan`), as the WP14 check's tests
make them.

The made-up run list has the entries WP15's readings W1 to W3 name (the labels of the committed
list's entries, `run-lists/run-list.json` on the run list's branch): `learn-lgbm_tail`,
`learn-lgbm_tail-rho0_1`, `learn-lgbm_tail-rho0_2`, `intervals-lgbm_tail-rho0_1`,
`intervals-lgbm_tail-rho0_2` and `drift-lgbm_tail`, with the stand-in of the sealed truth build
first, each written with the options the run list's checker asks of the sealed run. Each learn,
interval and drift step is one stand-in (`STAND_IN`): it copies into its run folder the files of
the ignored folder `data/made-up/<its label>/`, the results the real command would write, made up
here in the real files' form (`learn.json`, `day-intervals.json`, `drift-run.json`,
`drift-run-days.csv`); a file `exit` there sets its exit status and a file `sleep` the seconds it
sleeps first. The driver's start check is replaced for the stand-ins (`drive_fake.no_start_check`),
and each made-up matrix holds a `sealed_choice` whose dial of Norway and Korea is the list's fixed
`dial-others`, the drift step's dial, as the run list's rule 8 asks.
"""

import csv
import hashlib
import importlib
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
from sealed_watch import watch

from fleetreserve import drift_run, run_driver, sealed_matrix
from fleetreserve.gate_config import CEILINGS

TOP = Path(__file__).resolve().parents[2]
TOOL = TOP / "tools" / "wp15_check.py"
for _helpers in (TOP / "tools", TOP / "tests" / "wp2", TOP / "tests" / "wp12"):
    if str(_helpers) not in sys.path:
        sys.path.insert(0, str(_helpers))
fake = importlib.import_module("fake_heldout")
drive_fake = importlib.import_module("drive_fake")
check = importlib.import_module("wp15_check")

LIST = drive_fake.LIST_PATH
SEAL = ["--purpose", "sealed-run"]
REHEARSE = ["--purpose", "rehearsal"]
STAND_IN = """import os
import shutil
import sys
import time

run_dir = os.environ["FLEETRESERVE_RUN_DIR"]
label = os.path.basename(run_dir).split("-", 1)[1]
source = os.path.join("data", "made-up", label)
control = {}
if os.path.isdir(source):
    for name in sorted(os.listdir(source)):
        if name in ("exit", "sleep", "dirty", "no-opening"):
            with open(os.path.join(source, name)) as handle:
                control[name] = handle.read().strip()
        else:
            shutil.copyfile(os.path.join(source, name), os.path.join(run_dir, name))
time.sleep(float(control.get("sleep", "0")))
if "dirty" in control:  # the tree changes, as the driver's own check reads it
    with open("tracked.txt", "a") as handle:
        handle.write("changed by a step\\n")
sys.exit(int(control.get("exit", "0")))
"""
# The stand-in of the sealed truth build also writes the line the opener writes to the openings log,
# as the real build's opening of a sealed file does: WP15 runs WP14's own check on each complete
# drive (decision 99 (5)), and its mark 1 fails a complete drive that holds no opening, and its mark
# 2 an earlier drive that opened sealed data and did not stop at a layout refusal: a file
# `data/made-up/sealed-truth/no-opening` makes a drive's stand-in write none (an earlier drive).
OPENING = """import datetime
import json
import subprocess

if "no-opening" not in control:
    quiet = ["-c", "maintenance.auto=false", "-c", "gc.auto=0"]
    run = dict(capture_output=True, text=True)
    head = subprocess.run(["git", *quiet, "rev-parse", "HEAD"], **run).stdout
    ref = "refs/tags/frozen-plan^{commit}"
    tag = subprocess.run(["git", *quiet, "rev-parse", ref], **run).stdout
    head, tag = head.strip(), tag.strip()
    now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = {"time_utc": now, "set": "asr", "file": "202410DatasetEVOfficeParking_v0.csv",
            "sha256": "0" * 64, "commit": head, "tag_commit": tag, "command": sys.argv}
    if head != tag:
        line["fix_of_tag"] = tag
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "heldout-openings.log"), "a") as handle:
        handle.write(json.dumps(line) + "\\n")
"""
STAND_INS = ("learn_run", "day_intervals", "drift_run", drive_fake.SEALED_TRUTH)
MAIN = "gate/sealed-matrix/lgbm_tail.json"
MATRICES = {None: MAIN, **{r: f"gate/sealed-matrix/lgbm_tail-rho{r:g}.json" for r in check.RHOS}}
CONFIGURATIONS = ("main-k0.5-none-empirical", "main-k0.5-d0.02-empirical")
SEEDED = ["--interval-seed", "{seed:intervals}", "--interval-resamples", "2000"]
ROUTE = ["--held-out-truth-run", "{run:sealed-truth}"]
DIAL = "0.02"  # each site type's dial in the made-up matrices, the list's fixed `dial-others`
CHOICE = {"reading": "R2", "kappa": 0.5}  # the parts the run list's rule 8 and parser read
CHOICE["dials"] = {sealed_matrix.WORKPLACES: float(DIAL), sealed_matrix.OTHERS: float(DIAL)}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    """Drop every GIT_* variable and git's global and system settings for the test."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def nine() -> list[str]:
    from fleetreserve import sealed_route

    return list(sealed_route.HELD_OUT_UNITS)


# --- the made-up list and results ---------------------------------------------------------------


def stand_in(module: str, *arguments: str) -> list[str]:
    return [sys.executable, "-m", f"fleetreserve.{module}", *arguments]


def learn_entry(rho=None, label=None) -> dict:
    return drive_fake.entry(
        label or check.learn_label(rho),
        stand_in("learn_run", "--model", "lgbm_tail", *SEEDED),
        extra={"sealed-run": ["--matrix", MATRICES[rho], *ROUTE]},
    )


def intervals_entry(rho, reads=None) -> dict:
    learn = reads or check.learn_label(rho)
    return drive_fake.entry(
        check.intervals_label(rho),
        stand_in("day_intervals", "--run", f"{{run:{learn}}}", "--seed", "{seed:intervals}"),
        after=[learn],
    )


def drift_entry(max_seconds=60) -> dict:
    return drive_fake.entry(
        check.DRIFT_LABEL,
        stand_in(
            "drift_run",
            "--model",
            "lgbm_tail",
            "--dial",
            "{name:dial-others}",
            "--units",
            "{units}",
        ),
        units="drift",
        extra={"sealed-run": ["--matrix", MAIN, *ROUTE]},
        max_seconds=max_seconds,
    )


def entries(*, drop=(), drift_seconds=60) -> list[dict]:
    found = [
        drive_fake.entry("sealed-truth", stand_in(drive_fake.SEALED_TRUTH), only="sealed-run"),
        learn_entry(),
        *(e for r in check.RHOS for e in (learn_entry(r), intervals_entry(r))),
        drift_entry(drift_seconds),
    ]
    return [e for e in found if e["label"] not in drop]


def run_list(found: list[dict], drift_units=(check.DRIFT_UNIT,)) -> dict:
    """The made-up list of `found`, with the group `drift` when an entry names it."""
    grouped = any(e["units"] == "drift" for e in found)
    document = drive_fake.document(
        found,
        seeds={"intervals": 20261122},
        units={
            "rehearsal": {"drift": ["norway-development-pooled"]} if grouped else {},
            "sealed-run": {"drift": list(drift_units)} if grouped else {},
        },
    )
    if any("{name:dial-others}" in e["command"] for e in found):  # a name no entry reads: L2
        document["fixed"]["names"]["dial-others"] = DIAL  # the drift step's dial (rule 8)
    return document


def rungs(*, offer=1.5, shares=(0.4, 0.6)) -> dict:
    test = {
        "days": 20,
        "offering_days": 18,
        "mean_offer_kwh_per_day": offer,
        "share_of_ceiling": {
            c: {"ratio_of_sums": s, "mean_of_daily_ratios": s}
            for c, s in zip(CEILINGS, shares, strict=True)
        },
    }
    return {
        r: {"beta_hat": 0.8, "reason": None, "test": json.loads(json.dumps(test))}
        for r in ("fixed", "one-way")
    }


def learn_json(units, matrix_bytes: bytes) -> dict:
    return {
        "model": "lgbm_tail",
        "units": {
            u: {"configurations": {c: {"rungs": rungs()} for c in CONFIGURATIONS}} for u in units
        },
        "matrix": {"file": "m.json", "sha256": hashlib.sha256(matrix_bytes).hexdigest()},
        "sealed_route": {"kind": "held-out"},
        "failed_configurations": [],
    }


def bounded(point) -> dict:
    return {"point": point, "interval": [point * 0.5, point * 1.5]}


def intervals_json(units) -> dict:
    cell = {
        "offer_kwh_per_counting_test_day": bounded(1.5),
        "share_of_ceiling": {c: bounded(0.5) for c in CEILINGS},
    }
    by_rung = {r: {"A": json.loads(json.dumps(cell))} for r in ("fixed", "one-way")}
    return {
        "configurations": {
            c: {u: json.loads(json.dumps(by_rung)) for u in units} for c in CONFIGURATIONS
        },
        "problems": [],
        "sealed_route": {"kind": "held-out"},
    }


def drift_files(*, blank_loss=False) -> tuple[dict, str]:
    rows = []
    for tier in ("A", "B"):
        for day in range(3):
            row = dict.fromkeys(drift_run.DAY_COLUMNS, "")
            row.update(
                unit=check.DRIFT_UNIT,
                tier=tier,
                start="zero",
                eta="0.1",
                day=str(day),
                offering="true",
                step=str(day + 1),
                offer_kwh="2.5",
                slot_loss="0.0",
                driver_loss="0.0",
            )
            rows.append(row)
    if blank_loss:
        rows[1]["slot_loss"] = ""
    handle = io.StringIO()
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(drift_run.DAY_COLUMNS)
    for row in rows:
        writer.writerow([row[c] for c in drift_run.DAY_COLUMNS])
    table = handle.getvalue()
    document = {
        "run": {"model": "lgbm_tail", "tiers": ["A", "B"], "units": [check.DRIFT_UNIT]},
        "units": {
            check.DRIFT_UNIT: {
                "status": "run",
                "guarded_days": 3,
                "paths": {t: {"zero": {"0.1": {"tier": t}}} for t in ("A", "B")},
            }
        },
        "failed_units": [],
        "days_file": {
            "file": drift_run.DAYS_NAME,
            "rows": len(rows),
            "sha256": hashlib.sha256(table.encode()).hexdigest(),
        },
        "sealed_route": {"kind": "held-out"},
    }
    return document, table


def put(root: Path, label: str, name: str, content) -> None:
    folder = root / "data" / "made-up" / label
    folder.mkdir(parents=True, exist_ok=True)
    text = content if isinstance(content, str) else json.dumps(content, indent=1) + "\n"
    (folder / name).write_text(text, encoding="utf-8")


def made_results(root: Path) -> None:
    """The results each step's stand-in copies into its run folder (module text); the sealed truth
    build's summary records no abstained unit."""
    put(root, "sealed-truth", "truth-summary.json", {"units": {}, "abstained": {}})
    for rho, path in MATRICES.items():
        put(
            root,
            check.learn_label(rho),
            "learn.json",
            learn_json(nine(), (root / path).read_bytes()),
        )
    for rho in check.RHOS:
        put(root, check.intervals_label(rho), "day-intervals.json", intervals_json(nine()))
    document, table = drift_files()
    put(root, check.DRIFT_LABEL, drift_run.RESULTS_NAME, document)
    put(root, check.DRIFT_LABEL, drift_run.DAYS_NAME, table)


def made(tmp_path, monkeypatch, found=None, drift_units=(check.DRIFT_UNIT,)) -> Path:
    """The throwaway repository frozen at the tag with the run list of `found` (default: every
    entry), the sealed run's drift group `drift_units`, and the made-up results in its ignored
    folder; its top folder."""
    root = drive_fake.make_repo(tmp_path)
    for module in STAND_INS:
        text = STAND_IN.replace("time.sleep(", OPENING + "time.sleep(", 1)
        (root / "fleetreserve" / f"{module}.py").write_text(
            text if module == drive_fake.SEALED_TRUTH else STAND_IN
        )
    with (root / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("data/\n")
    for rho, path in MATRICES.items():
        drive_fake.write_matrix(root, path, nine())
        matrix = json.loads((root / path).read_text())
        matrix["sealed_choice"] = CHOICE  # the run list's rule 8 holds the drift dial against it
        if rho is not None:
            matrix["rho"] = rho
        (root / path).write_text(json.dumps(matrix, indent=2) + "\n")
    fake.copy_package(root)
    drive_fake.commit_list(root, run_list(entries() if found is None else found, drift_units))
    fake.freeze(root)
    made_results(root)
    drive_fake.no_start_check(monkeypatch)  # the stand-ins' starts (drive_fake, module text)
    monkeypatch.chdir(root)
    return root


def drive(expected: int, purpose=SEAL) -> None:
    """One drive by the real driver (the sealed run's through the real gate), its exit status
    checked; the next drive starts in another second, so the logs' names differ."""
    assert run_driver.main([LIST, *purpose]) == expected
    time.sleep(1.1)


def run(root: Path, capsys) -> tuple[int, list[str]]:
    capsys.readouterr()  # the driver's own output
    status = check.main(["--root", str(root)])
    return status, capsys.readouterr().out.splitlines()


def logs(root: Path) -> list[Path]:
    return drive_fake.drive_logs(root)


def entry_line(root: Path, label: str) -> dict:
    (log,) = [p for p in logs(root) if p.name.endswith("-sealed-run.jsonl")]
    return next(r for r in drive_fake.read_log(log) if r.get("label") == label and "run" in r)


def rewritten(path: Path, change) -> None:
    """The drive log with `change(records)` applied and its end's hash recomputed, as the driver
    would write it with those lines."""
    records = [json.loads(line) for line in path.read_bytes().splitlines()]
    change(records)
    above = b"".join(run_driver.encode(record) for record in records[:-1])
    end = records[-1]
    end["log_sha256"] = run_driver.end_digest(above, end)
    path.write_bytes(above + run_driver.encode(end))


def part_lines(lines: list[str], part: str) -> list[str]:
    return [line for line in lines if part in line]


# --- all reported --------------------------------------------------------------------------------


def test_all_three_parts_reported_from_the_complete_drive_pass(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 0, lines
    assert lines[0] == check.HEADER
    assert "readings W1 to W3" in lines[0] and "not plan text" in lines[0]
    assert "provisional" not in lines[0] and "listed for the owner" not in lines[0]
    for part in check.PARTS:
        assert f"{part}: reported" in lines, lines
    (log,) = logs(root)
    assert lines[-1].startswith("passed: WP15, 3 of 3 parts reported, from the entries ")
    assert lines[-1].endswith(f"of {LIST}, in {log.name}")
    for label in check.needed_labels():
        assert label in lines[-1]
    rho = [line for line in lines if line.startswith("  rho 0.1: ")]
    assert len(rho) == 1 and "18 configuration-unit cells of 9 units" in rho[0], lines


def test_a_complete_drive_prints_the_same_lines_and_exits_0_as_before_decision_99(
    tmp_path, monkeypatch, capsys
):
    """Every line but the header (whose words changed: no "provisional") of a complete drive, as
    the check printed them at 7580c12, the folders and log named by their labels; green there and
    at the head of the branch that made the incomplete drive an information-only reading."""
    root = made(tmp_path, monkeypatch)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 0, lines
    named = [re.sub(r"\d{8}T\d{6}Z-", "<start>-", x) for x in lines[1:]]
    both = (
        "holds the offer and its share of both ceilings for 18 configuration-unit cells of 9 units"
    )
    rho = [
        f"  rho {r}: learn-lgbm_tail-rho{u} (runs.noindex/<start>-learn-lgbm_tail-rho{u}) {both}, "
        f"intervals-lgbm_tail-rho{u} (runs.noindex/<start>-intervals-lgbm_tail-rho{u}) their "
        "intervals"
        for r, u in (("0.1", "0_1"), ("0.2", "0_2"))
    ]
    drift = (
        "  drift-lgbm_tail (runs.noindex/<start>-drift-lgbm_tail) on OSL_T: tier A: 3 guarded "
        "days, 3 offering steps; tier B: 3 guarded days, 3 offering steps"
    )
    regimes = (
        "  learn-lgbm_tail (runs.noindex/<start>-learn-lgbm_tail) scores asr-before-march-2023, "
        "asr-from-march-2023 as units of their own, 4 configuration-unit cells with the offer and "
        "both shares"
    )
    assert named[:-1] == [
        *rho,
        "(a) rho 0.1 and 0.2: reported",
        drift,
        "(b) chronological calibration at OSL_T: reported",
        regimes,
        "(c) regimes: reported",
    ]
    assert named[-1] == (
        "passed: WP15, 3 of 3 parts reported, from the entries learn-lgbm_tail-rho0_1, "
        "intervals-lgbm_tail-rho0_1, learn-lgbm_tail-rho0_2, intervals-lgbm_tail-rho0_2, "
        "drift-lgbm_tail, learn-lgbm_tail of run-lists/run-list.json, in <start>-sealed-run.jsonl"
    )


def test_the_labels_read_are_the_readings_w1_to_w3():
    assert check.needed_labels() == (
        "learn-lgbm_tail-rho0_1",
        "intervals-lgbm_tail-rho0_1",
        "learn-lgbm_tail-rho0_2",
        "intervals-lgbm_tail-rho0_2",
        "drift-lgbm_tail",
        "learn-lgbm_tail",
    )
    assert check.ASR_UNITS == ("asr-before-march-2023", "asr-from-march-2023")


# --- which run: never the rehearsal, never stale, never a run that did not pass -------------------


def test_a_rehearsal_drive_alone_could_not_run(tmp_path, monkeypatch, capsys):
    """The rehearsal's drive of the same list, whose steps copy the same made-up results, is
    never read: with no sealed-run drive log the item could not run."""
    root = made(tmp_path, monkeypatch)
    drive(0, REHEARSE)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any("only rehearsal drive logs are present (1)" in line for line in lines), lines
    assert any(line.startswith("COULD NOT RUN: no sealed-run drive log") for line in lines), lines
    assert not any(line.endswith(": reported") for line in lines), lines


def test_no_drive_log_could_not_run(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert lines[-1].startswith("COULD NOT RUN: WP15, 0 of 3 parts reported"), lines


def test_a_drive_whose_start_names_another_commit_is_stale(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    drive(0)
    (log,) = logs(root)
    rewritten(log, lambda records: records[0].update(commit="d" * 40))
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any(f"is stale: written at commit {'d' * 40}" in line for line in lines), lines
    assert not any(line.endswith(": reported") for line in lines), lines


def test_results_recorded_at_another_commit_could_not_run(tmp_path, monkeypatch, capsys):
    """The manifest of the rho 0.1 learn run rewritten to name another commit, the drive log
    rewritten to record that manifest's SHA-256: the run does not verify at the checked-out
    commit, so no part is reported."""
    root = made(tmp_path, monkeypatch)
    drive(0)
    label = check.learn_label(0.1)
    line = entry_line(root, label)
    manifest_file = root / line["run"] / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["commit"] = "d" * 40
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    def change(records):
        for record in records:
            if record.get("label") == label and "run" in record:
                record["manifest_sha256"] = digest

    (log,) = logs(root)
    rewritten(log, change)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any(f"{label}: {line['run']} was recorded at another commit" in x for x in lines), lines
    assert not any(x.endswith(": reported") for x in lines), lines


def test_a_failed_rho_run_makes_the_drive_incomplete_and_the_item_exits_1(
    tmp_path, monkeypatch, capsys
):
    """The rho 0.1 learn run ends with exit status 1: the sealed-run drive goes on (its interval
    step, which reads that run, is logged not run), no complete drive is left, the held-out run is
    incomplete, part (a) is lost, the other parts are information, and the exit status is 1
    (decision 99 (5); tests/wp15/test_wp15_incomplete.py has the rest)."""
    root = made(tmp_path, monkeypatch)
    put(root, check.learn_label(0.1), "exit", "1\n")
    drive(1)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert lines[1].startswith("the held-out run is incomplete: ")
    assert "failed step 3 (learn-lgbm_tail-rho0_1)" in lines[1], lines
    assert any(
        x.startswith("information: (a) rho 0.1 and 0.2: lost, it needs a step that was not read: ")
        and "learn-lgbm_tail-rho0_1 (failed: learn-lgbm_tail-rho0_1 ended with exit status 1)" in x
        for x in lines
    ), lines
    assert not any(x.endswith(": reported") for x in lines), lines
    assert lines[-1].startswith("FAILED: WP15, the held-out run is incomplete"), lines


def test_a_failed_rho_run_and_a_failed_drift_run_are_both_named_and_the_interval_run_is_not_run(
    tmp_path, monkeypatch, capsys
):
    """Two needed steps fail in one drive (the drive goes on past each), and the interval step
    that reads the first one's run is logged `not_run` and never started: the incomplete line
    names the three, part (a) and part (b) are lost, each naming its own step, and part (c),
    whose step passed, is readable as information; exit 1."""
    root = made(tmp_path, monkeypatch)
    put(root, check.learn_label(0.1), "exit", "1\n")
    put(root, check.DRIFT_LABEL, "exit", "1\n")
    drive(1)
    (log,) = drive_fake.drive_logs(root)
    records = drive_fake.read_log(log)
    assert [r["label"] for r in records if r["record"] == "failed"] == [
        check.learn_label(0.1),
        check.DRIFT_LABEL,
    ]
    assert [r["label"] for r in records if r["record"] == "not_run"] == [check.intervals_label(0.1)]
    status, lines = run(root, capsys)
    assert status == 1, lines
    for text in (
        "failed step 3 (learn-lgbm_tail-rho0_1)",
        "failed step 7 (drift-lgbm_tail)",
        "not run step 4 (intervals-lgbm_tail-rho0_1)",
    ):
        assert text in lines[1], lines
    assert any(
        x.startswith("information: (a) rho 0.1 and 0.2: lost")
        and "learn-lgbm_tail-rho0_1 (failed: learn-lgbm_tail-rho0_1 ended with exit status 1)" in x
        for x in lines
    ), lines
    assert any(
        x.startswith("information: (b) chronological calibration at OSL_T: lost")
        and "drift-lgbm_tail (failed: drift-lgbm_tail ended with exit status 1)" in x
        for x in lines
    ), lines
    assert any(x.startswith("information: (c) regimes: readable") for x in lines), lines


def test_a_timed_out_drift_run_makes_the_drive_incomplete_and_is_named(
    tmp_path, monkeypatch, capsys
):
    """The drift run sleeps past its time limit: the recorder stops it, the drive logs it failed
    and goes on, the held-out run is incomplete (exit 1, never a pass), and part (b) is lost,
    naming the timeout."""
    root = made(tmp_path, monkeypatch, entries(drift_seconds=2))
    put(root, check.DRIFT_LABEL, "sleep", "60\n")
    drive(1)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith("information: (b) chronological calibration at OSL_T: lost")
        and "drift-lgbm_tail hit its time limit" in x
        for x in lines
    ), lines
    assert not any(x.endswith(": reported") for x in lines), lines


def test_a_complete_drive_after_a_failed_one_is_the_run_read(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    put(root, check.learn_label(0.2), "exit", "1\n")
    put(root, "sealed-truth", "no-opening", "1\n")  # the earlier drive opened nothing
    drive(1)
    (root / "data" / "made-up" / check.learn_label(0.2) / "exit").unlink()
    (root / "data" / "made-up" / "sealed-truth" / "no-opening").unlink()
    drive(0)
    status, lines = run(root, capsys)
    assert status == 0, lines


def test_two_complete_drives_could_not_run(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    drive(0)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any("2 complete sealed-run drives" in x for x in lines), lines


# --- the entries come from the committed list -----------------------------------------------------


@pytest.mark.parametrize("label", ["intervals-lgbm_tail-rho0_2", "drift-lgbm_tail"])
def test_an_entry_missing_from_the_list_could_not_run_naming_it(
    tmp_path, monkeypatch, capsys, label
):
    root = made(tmp_path, monkeypatch, entries(drop=(label,)))
    drive(0)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any(f"has no entry {label}, which reading W" in x for x in lines), lines


def test_a_rho_entry_on_the_main_matrix_could_not_run(tmp_path, monkeypatch, capsys):
    """The rho 0.2 learn entry scores the main matrix (rho 0): it is not the sensitivity W1
    names, so part (a) could not run."""
    found = entries()
    for e in found:
        if e["label"] == check.learn_label(0.2):
            e["extra"]["sealed-run"][1] = MAIN
    root = made(tmp_path, monkeypatch, found)
    put(root, check.learn_label(0.2), "learn.json", learn_json(nine(), (root / MAIN).read_bytes()))
    drive(0)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any("the matrix of learn-lgbm_tail-rho0_2 is at rho 0.0, not 0.2" in x for x in lines)


def test_intervals_of_another_run_could_not_run(tmp_path, monkeypatch, capsys):
    found = [e for e in entries() if e["label"] != check.intervals_label(0.1)]
    found.insert(3, intervals_entry(0.1, reads=check.learn_label()))
    root = made(tmp_path, monkeypatch, found)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any("does not read the run of learn-lgbm_tail-rho0_1" in x for x in lines), lines


# --- the figures each part needs -----------------------------------------------------------------


def test_a_share_of_a_ceiling_missing_fails(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    document = learn_json(nine(), (root / MATRICES[0.2]).read_bytes())
    test = document["units"]["TRO_R"]["configurations"][CONFIGURATIONS[1]]["rungs"]["one-way"]
    test["test"]["share_of_ceiling"][CEILINGS[1]] = None
    put(root, check.learn_label(0.2), "learn.json", document)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith("FAILED: (a) rho 0.1 and 0.2: learn-lgbm_tail-rho0_2 lacks 1 figure: ")
        and f"{CONFIGURATIONS[1]} on TRO_R, one-way: no share of the {CEILINGS[1]} ceiling" in x
        for x in lines
    ), lines


def test_a_unit_of_the_matrix_missing_from_the_results_fails(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    units = [u for u in nine() if u != "korea-work"]
    put(
        root,
        check.learn_label(0.1),
        "learn.json",
        learn_json(units, (root / MATRICES[0.1]).read_bytes()),
    )
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert any(
        "korea-work: no figure: the learn run has no entry for it, and the held-out truth build "
        "recorded no abstention for it" in x
        for x in lines
    ), lines
    assert not any("None" in x or "no configuration scored" in x for x in lines), lines


def test_an_interval_missing_fails(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    document = intervals_json(nine())
    document["configurations"][CONFIGURATIONS[0]]["OSL_T"]["fixed"]["A"][
        "offer_kwh_per_counting_test_day"
    ]["interval"] = None
    put(root, check.intervals_label(0.1), "day-intervals.json", document)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert any("no interval of the offer" in x for x in lines), lines


def test_a_drift_day_without_its_loss_fails(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    document, table = drift_files(blank_loss=True)
    put(root, check.DRIFT_LABEL, drift_run.RESULTS_NAME, document)
    put(root, check.DRIFT_LABEL, drift_run.DAYS_NAME, table)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert any("tier A of OSL_T lacks a day's losses" in x for x in lines), lines


def test_a_drift_run_on_other_units_too_could_not_run(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch, drift_units=("OSL_T", "TRO_R"))
    drive(0)
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any("runs on ['OSL_T', 'TRO_R'], not on OSL_T alone" in x for x in lines), lines


def test_the_asr_units_pooled_into_one_fail(tmp_path, monkeypatch, capsys):
    """The main model's learn run scores one pooled a.s.r. unit in place of the two: part (c)
    fails, naming both."""
    root = made(tmp_path, monkeypatch)
    units = ["asr-pooled", *(u for u in nine() if not u.startswith("asr"))]
    put(root, check.learn_label(), "learn.json", learn_json(units, (root / MAIN).read_bytes()))
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert any(
        x.startswith("FAILED: (c) regimes: learn-lgbm_tail: the a.s.r. units are not scored")
        and "missing ['asr-before-march-2023', 'asr-from-march-2023']" in x
        and "other a.s.r. units ['asr-pooled']" in x
        for x in lines
    ), lines


# --- reported, with the reason (reading V6, the owner's decision of 19 November 2026) -----------

NOT_SPLIT = "not calibratable: 40 days"  # `gate_run.plan_unit`'s status of a unit not split


def not_split(document: dict, unit: str, *, status=NOT_SPLIT, test_days=0, rungs=None) -> None:
    """`unit` of a made-up `learn.json` as `learn_run` writes a unit that cannot be split: its
    status, no test day, and each configuration's rungs the reason alone (`learn_eval._rungs`),
    unless `rungs` is given."""
    document["units"][unit] = {
        "counting_days": 40,
        "calibration_days_used": 0,
        "test_days": test_days,
        "status": status,
        "configurations": {
            c: {"rungs": {"reason": status} if rungs is None else rungs} for c in CONFIGURATIONS
        },
    }


def test_a_unit_that_cannot_be_split_is_reported_with_the_reason(tmp_path, monkeypatch, capsys):
    """TRO_R cannot be split at rho 0.1: its configurations hold only the reason, and the interval
    run, which has no per-day record of it, holds none of its cells. Part (a) is reported, with
    the reason; exit 0 (exit 1 before decision 74)."""
    root = made(tmp_path, monkeypatch)
    document = learn_json(nine(), (root / MATRICES[0.1]).read_bytes())
    not_split(document, "TRO_R")
    put(root, check.learn_label(0.1), "learn.json", document)
    others = [u for u in nine() if u != "TRO_R"]
    put(root, check.intervals_label(0.1), "day-intervals.json", intervals_json(others))
    drive(0)
    status, lines = run(root, capsys)
    assert status == 0, lines
    (rho,) = [x for x in lines if x.startswith("  rho 0.1:")]
    assert "16 configuration-unit cells of 9 units" in rho, lines
    assert rho.endswith(f"; with the reason, no figure: TRO_R: {NOT_SPLIT} (2 configurations)")
    assert (
        f"(a) rho 0.1 and 0.2: reported, with the reason: rho 0.1: TRO_R: {NOT_SPLIT} "
        "(2 configurations)"
    ) in lines, lines
    assert "(b) chronological calibration at OSL_T: reported" in lines, lines


def test_an_asr_unit_that_cannot_be_split_is_reported_with_the_reason(
    tmp_path, monkeypatch, capsys
):
    """The same in the main model's learn run: part (c) reported, with the reason (and part (a)
    untouched, its learn runs being the rho runs)."""
    root = made(tmp_path, monkeypatch)
    document = learn_json(nine(), (root / MAIN).read_bytes())
    not_split(document, check.ASR_UNITS[0])
    put(root, check.learn_label(), "learn.json", document)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 0, lines
    assert (
        f"(c) regimes: reported, with the reason: {check.ASR_UNITS[0]}: {NOT_SPLIT} "
        "(2 configurations)"
    ) in lines, lines
    assert "(a) rho 0.1 and 0.2: reported" in lines, lines


@pytest.mark.parametrize(
    "shape",
    [
        {"rungs": {}},  # a unit not split, a configuration with no figure and no reason
        {"rungs": {"reason": "another reason"}},  # a reason other than the unit's status
        {"status": "calibratable", "test_days": 20, "rungs": {"reason": "configuration failed"}},
        {"test_days": 20},  # the unit's status says not split, but it has test days
        {"status": "calibratable"},  # no test day, the reason its status, but split by its status
    ],
    ids=["no-reason", "other-reason", "failed-configuration", "test-days", "split-status"],
)
def test_a_configuration_without_its_figures_and_not_split_still_fails(
    tmp_path, monkeypatch, capsys, shape
):
    """A missing figure with no reason, or with any reason but the unit's not being split, still
    fails part (a) and part (c) (exit 1)."""
    root = made(tmp_path, monkeypatch)
    for rho, unit in ((0.1, "TRO_R"), (None, check.ASR_UNITS[0])):
        document = learn_json(nine(), (root / MATRICES[rho]).read_bytes())
        not_split(document, unit, **shape)
        put(root, check.learn_label(rho), "learn.json", document)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    for part, label, unit in (
        ("(a) rho 0.1 and 0.2", check.learn_label(0.1), "TRO_R"),
        ("(c) regimes", check.learn_label(), check.ASR_UNITS[0]),
    ):
        assert any(
            x.startswith(f"FAILED: {part}: {label} lacks ") and f"on {unit}, fixed: no offer" in x
            for x in lines
        ), lines
    assert not any("with the reason" in x for x in lines), lines


def drift_offering_nothing(eta=None, record=True) -> tuple[dict, str]:
    """The drift guard's results when no tier holds a path for OSL_T (reading E29): no row in the
    day table, the guarded days counted, and the route's record of the step sizes (`eta`; by
    default "offers nothing" for both tiers), left out when `record` is false."""
    document, _ = drift_files()
    table = ",".join(drift_run.DAY_COLUMNS) + "\n"
    document["units"][check.DRIFT_UNIT]["paths"] = {}
    document["days_file"].update(rows=0, sha256=hashlib.sha256(table.encode()).hexdigest())
    nothing = "offers nothing"
    steps = {"site_type": "other", "eta": eta or {"A": nothing, "B": nothing}}
    if record:
        document["sealed_route"]["step_sizes"] = {check.DRIFT_UNIT: steps}
    return document, table


def test_a_drift_run_in_which_no_tier_offers_anything_is_reported_with_the_reason(
    tmp_path, monkeypatch, capsys
):
    """Exit 0, part (b) printed with the reason (exit 1 before decision 74)."""
    root = made(tmp_path, monkeypatch)
    document, table = drift_offering_nothing()
    put(root, check.DRIFT_LABEL, drift_run.RESULTS_NAME, document)
    put(root, check.DRIFT_LABEL, drift_run.DAYS_NAME, table)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 0, lines
    reason = (
        'no tier offers anything on OSL_T: its run\'s record gives tiers A, B "offers nothing" '
        "(reading E29)"
    )
    assert f"(b) chronological calibration at OSL_T: reported, with the reason: {reason}" in lines
    assert any(
        x.startswith("  drift-lgbm_tail (") and "3 guarded days; with the reason, no figure" in x
        for x in lines
    ), lines


@pytest.mark.parametrize(
    "eta, record",
    [({"A": "offers nothing", "B": 2.0}, True), (None, False)],
    ids=["tier-b-has-a-step-size", "no-record"],
)
def test_no_tier_with_a_path_without_the_reason_still_fails(
    tmp_path, monkeypatch, capsys, eta, record
):
    """No tier holds a path, but the run's record does not give every tier "offers nothing":
    the tiers' figures are missing with no reason, part (b) fails (exit 1)."""
    root = made(tmp_path, monkeypatch)
    document, table = drift_offering_nothing(eta, record)
    put(root, check.DRIFT_LABEL, drift_run.RESULTS_NAME, document)
    put(root, check.DRIFT_LABEL, drift_run.DAYS_NAME, table)
    drive(0)
    status, lines = run(root, capsys)
    assert status == 1, lines
    assert (
        "FAILED: (b) chronological calibration at OSL_T: drift-lgbm_tail: OSL_T has no per-day "
        "loss in any tier"
    ) in lines, lines


def test_a_results_file_changed_after_its_run_could_not_run(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    drive(0)
    line = entry_line(root, check.learn_label())
    file = root / line["run"] / "learn.json"
    file.write_text(file.read_text().replace("1.5", "2.5"))
    status, lines = run(root, capsys)
    assert status == 2, lines
    assert any("learn.json does not verify" in x for x in lines), lines


# --- nothing sealed opened; the script's exit statuses ----------------------------------------


def test_the_check_opens_lists_or_scans_nothing_sealed(tmp_path, monkeypatch, capsys):
    root = made(tmp_path, monkeypatch)
    drive(0)
    sealed = root / fake.SEALED / fake.ASR_FILE[0] / fake.ASR_FILE[1]
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"made-up sealed bytes\n")
    with watch(root / fake.SEALED) as seen:
        status, lines = run(root, capsys)
    assert status == 0, lines
    assert seen == []


def script(*args, extra=None) -> subprocess.CompletedProcess:
    environment = fake.tool_environment()
    environment.update(extra or {})
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        timeout=300,
    )


def test_the_script_gives_each_exit_status(tmp_path, monkeypatch):
    root = made(tmp_path, monkeypatch)
    assert script("--root", str(root)).returncode == 2  # no drive log
    drive(0)
    done = script("--root", str(root))
    assert done.returncode == 0, done.stdout
    assert done.stdout.splitlines()[0] == check.HEADER
    assert script("everything").returncode == 2  # a wrong call


def test_the_script_fails_on_a_failed_part(tmp_path, monkeypatch):
    root = made(tmp_path, monkeypatch)
    put(root, check.learn_label(0.1), "exit", "1\n")
    drive(1)
    assert script("--root", str(root)).returncode == 1


def test_a_failed_import_could_not_run(tmp_path):
    broken = tmp_path / "fleetreserve"
    broken.mkdir()
    (broken / "__init__.py").write_text("raise ImportError('a made-up broken package')\n")
    done = script(extra={"PYTHONPATH": str(tmp_path)})
    assert done.returncode == 2
    assert done.stdout.startswith("COULD NOT RUN: the check cannot start: ")

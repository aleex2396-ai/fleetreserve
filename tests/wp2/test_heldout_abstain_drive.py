"""The driver (`run_driver.drive`, purpose `sealed-run`) goes past a held-out unit the held-out
truth build abstained, to the last step.

A made-up repository frozen at the made-up tag `frozen-plan` (`fake_heldout.sealed_repo`, the
made-up sealed files of `heldout_truth_fake`, its package copied in) is driven through the real
gate of the run from the tag. The list is four steps, each a REAL command of the package run in
this process (the driver's recorder call is replaced by `_in_process`, which runs the command's
`main` with `FLEETRESERVE_RUN_DIR` set, records the run as the recorder does, and hands the driver
its `entry` line): the held-out truth build (`heldout_truth_build`, which abstains `korea-visits`:
one day, no calibration day), `learn_run`, outcome O2's `ceiling_report --every-day` and `o3_run`,
the last three naming OSL_T and korea-visits and the held-out build of step 1 as
`{run:sealed-truth}`. The list file and its checks (`run_list.parse`, the start check) are not the
subject: `prepare` and `run_list.instantiate` are replaced by the four steps.

1. The drive ends with exit status 0, every step's entry has exit status 0, and each of the three
   scoring steps wrote the abstention into its results.
2. With the build's abstention record removed from its summary after step 1 (a unit absent from
   the build the build did not record as abstained) each of the three scoring steps ends with exit
   status 2 ("not in the recorded truth build"): only a recorded abstention is let through. The
   sealed-run drive goes on after a failed entry (`run_driver` module text), so each is logged as
   failed, none stops the drive, the drive's exit status is 1, and WP14's check fails mark 1
   naming each.
"""

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import test_heldout_route_scoring as scoring
from fake_heldout import git, sealed_repo
from gate_fake import gate_cars, write_build
from heldout_truth_fake import sealed_files

from fleetreserve import (
    ceiling_report,
    gate_run,
    learn_run,
    o3_run,
    run_driver,
    run_list,
)
from fleetreserve import heldout_truth_build as h

LABELS = ("sealed-truth", "sealed-learn", "sealed-o2", "sealed-o3")
MODULES = {
    "heldout_truth_build": h.main,
    "learn_run": learn_run.main,
    "ceiling_report": ceiling_report.main,
    "o3_run": o3_run.main,
}
ABSTAINED = "korea-visits"
UNITS = ("OSL_T", ABSTAINED)
MODEL = scoring.MODEL


@pytest.fixture
def made(tmp_path, monkeypatch):
    """The made-up repository at its tag, a development build recorded at the tag's commit, the
    stand-in model, the commands' reading of the checked-out commit set to the tag's."""
    root = sealed_repo(tmp_path, sealed_files(tmp_path / "made"))
    tag = git(root, "rev-parse", "HEAD")
    derived = root / "data" / "derived"
    (derived / "development").mkdir(parents=True)
    run, tables = write_build(derived / "development", {scoring.FIT_UNIT: gate_cars(40, "T")})
    scoring._record(run, tag)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: tag)
    with (root / ".git" / "info" / "exclude").open("a") as handle:
        handle.write("runs.noindex/\n")  # the drive's log, outside the tracked files
    monkeypatch.chdir(root)

    def make_model(matrix):
        return scoring.RouteModel(MODEL, matrix.rho)

    module = types.ModuleType(f"fleetreserve.models.{MODEL}")
    module.make_model = make_model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    matrix = scoring._matrix(tmp_path, UNITS)
    return {"root": root, "tag": tag, "run": run, "tables": tables, "matrix": matrix}


def _steps(m: dict) -> list[run_list.Step]:
    def step(k, label, module, *arguments):
        command = ["python", "-m", f"fleetreserve.{module}", *arguments]
        places = tuple(
            (place, text[len("{run:") : -1])
            for place, text in enumerate(command)
            if text.startswith("{run:")
        )
        return run_list.Step(k, label, tuple(command), places, "main", 3600.0, ())

    held = "{run:sealed-truth}"
    dev = ["--tables", str(m["tables"]), "--truth-run", str(m["run"])]
    return [
        step(0, LABELS[0], "heldout_truth_build", "--matrix", str(scoring.MATRIX)),
        step(
            1,
            LABELS[1],
            "learn_run",
            *("--model", MODEL, "--matrix", str(m["matrix"]), *dev, "--held-out-truth-run", held),
        ),
        step(
            2,
            LABELS[2],
            "ceiling_report",
            *("--units", *UNITS, "--every-day", "--workers", "1", *dev),
            *("--held-out-truth-run", held),
        ),
        step(
            3,
            LABELS[3],
            "o3_run",
            *("--units", *UNITS, "--days", "1", "--seed", "1", "--base-set", "day-average"),
            *("--lp27-days", "1", *dev, "--held-out-truth-run", held),
        ),
    ]


def _in_process(m: dict, monkeypatch, after_first=None):
    """The driver's `run_entry` replaced: the command's `main` runs here with the recorder's
    environment, its run folder is recorded under `data/derived/runs` as the recorder records one
    (a manifest listing every output's SHA-256, at the tag's commit), and the `entry` line has
    the run's exit status. `after_first(run_folder)` is called after step 1."""
    root, runs = m["root"], m["root"] / "data" / "derived" / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    def run_entry(top, k, entry, command):
        module = command[command.index("-m") + 1].split(".", 1)[1]
        work = runs / f"20261116T00000{k}Z-{entry.label}"
        work.mkdir()
        with pytest.MonkeyPatch.context() as patched:
            patched.setenv("FLEETRESERVE_RUN_DIR", str(work))
            status = MODULES[module](command[command.index("-m") + 2 :], root=root)
        outputs = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(work.iterdir())
        }
        manifest = {
            "commit": m["tag"],
            "command": command,
            "start": "s",
            "end": "e",
            "exit_code": status,
            "timed_out": False,
            "outputs": outputs,
        }
        (work / "manifest.json").write_text(json.dumps(manifest))
        if k == 0 and after_first is not None:
            after_first(work)
        digest = hashlib.sha256((work / "manifest.json").read_bytes()).hexdigest()
        now = run_driver.format_time(run_driver.utc_now())  # a time of the driver's form
        return {
            "record": "entry",
            "index": k,
            "label": entry.label,
            "run": work.relative_to(root).as_posix(),
            "exit_status": status,
            "timed_out": False,
            "interrupted": None,
            "start": now,
            "end": now,
            "seconds": 0.0,
            "manifest_sha256": digest,
            "recorder_status": status,
            "recorder_output": "",
        }

    monkeypatch.setattr(run_driver, "run_entry", run_entry)
    monkeypatch.setattr(
        run_driver, "prepare", lambda top, path: (m["tag"], "run-list.json", b"{}", None)
    )
    monkeypatch.setattr(run_list, "instantiate", lambda parsed, purpose: _steps(m))


def _drive(m: dict) -> int:
    return run_driver.drive(Path("run-list.json"), purpose=run_list.SEALED_RUN, start_dir=m["root"])


def _log(m: dict) -> list[dict]:
    (path,) = list((m["root"] / "runs.noindex" / "drives").glob("*-sealed-run.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_drive_goes_past_the_abstained_unit_to_the_last_step(made, monkeypatch, capsys):
    _in_process(made, monkeypatch)
    assert _drive(made) == 0
    log = _log(made)
    entries = [r for r in log if r["record"] == "entry"]
    assert [(e["label"], e["exit_status"]) for e in entries] == [(label, 0) for label in LABELS]
    assert log[-1]["record"] == "end" and log[-1]["completed"] is True
    assert not [r for r in log if r["record"] == "stop"]
    root = made["root"]
    results = {
        LABELS[1]: ("learn.json", ("sealed_route", "held_out_inputs")),
        LABELS[2]: ("ceiling-report.json", ("inputs",)),
        LABELS[3]: ("o3.json", ("inputs",)),
    }
    for entry in entries[1:]:
        name, keys = results[entry["label"]]
        document = json.loads((root / entry["run"] / name).read_text())
        for key in keys:
            document = document[key]
        assert list(document["abstained"]) == [ABSTAINED], entry["label"]


def test_a_unit_the_build_does_not_record_as_abstained_fails_each_scoring_step(
    made, monkeypatch, capsys
):
    def drop_record(run: Path) -> None:
        summary = json.loads((run / "truth-summary.json").read_text())
        del summary["abstained"][ABSTAINED]
        (run / "truth-summary.json").write_text(json.dumps(summary))
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["outputs"]["truth-summary.json"] = hashlib.sha256(
            (run / "truth-summary.json").read_bytes()
        ).hexdigest()
        (run / "manifest.json").write_text(json.dumps(manifest))

    _in_process(made, monkeypatch, after_first=drop_record)
    assert _drive(made) == run_driver.FAILED
    printed = capsys.readouterr().out
    log = _log(made)
    entries = [r for r in log if r["record"] == "entry"]
    assert [(e["label"], e["exit_status"]) for e in entries] == [
        (LABELS[0], 0),
        *[(label, 2) for label in LABELS[1:]],
    ]
    failed = [r for r in log if r["record"] == "failed"]
    assert [(r["label"], r["reason"]) for r in failed] == [
        (label, "the run ended with exit status 2") for label in LABELS[1:]
    ]
    assert not [r for r in log if r["record"] in ("stop", "not_run")]
    assert log[-1]["completed"] is False and log[-1]["exit_status"] == run_driver.FAILED
    assert [f["label"] for f in log[-1]["failed"]] == list(LABELS[1:])
    # each scoring command named the unit in its own output (the reason the drive log does not hold)
    assert printed.count("korea-visits is not in the recorded truth build") == 3, printed
    # WP14's check: no complete drive, so mark 1 fails (exit 1) naming every failed entry
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    monkeypatch.syspath_prepend(tools)
    check = importlib.import_module("sealed_run_check")
    folder = made["root"] / "runs.noindex" / "recorder-run"
    folder.mkdir(parents=True)
    monkeypatch.setenv(check.RUN_DIR_VARIABLE, str(folder))
    status = check.main(["--root", str(made["root"])])
    capsys.readouterr()
    marks = {m["mark"]: m for m in json.loads((folder / check.RESULTS_NAME).read_text())["marks"]}
    assert status == 1, marks
    assert marks["one logged run"]["status"] == "failed", marks["one logged run"]
    (named,) = [
        line for line in marks["one logged run"]["lines"] if "no complete sealed-run" in line
    ]
    for k, label in enumerate(LABELS[1:], start=2):
        assert f"failed step {k} ({label}): the run ended with exit status 2" in named, named

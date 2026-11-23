"""The driver's start check (`run_driver` reading D7, `run_starts`): before its log and its first
step, for either purpose, the driver starts each step of the sealed run that runs a command of
the sealed route by the REAL command's code at a made-up tag, and refuses the drive when a start
is refused (the review of 4ae83be, finding 2, probe P3).

The working copy is made up (`test_run_starts.working_copy`): the committed default matrices,
the matrices `sealed_matrix` writes from made-up kappa-choice results, the drift guard's made-up
step-size file, a run list of three entries (the sealed truth build, a `compare_run` of the
main model's matrix on the sealed route, a truth build of the rehearsal) and a made-up sealed
file in the ignored `data/heldout/`. The sealed run's gate is made to pass, as at the tag. Each
step is run by a stand-in of `run_driver.run_entry` that records the step and opens the made-up
sealed file, as the sealed truth build would, so the watch of the data folder
(`tests/wp6gate/sealed_watch.py`) sees any step that starts.

1. `compare_run` without `--splits 1` (probe P3): the drive of each purpose is refused (exit 2)
   with the start check's line naming `--splits`; no step starts, no log is written, and no
   opening, listing or scan of the data folder, its made-up sealed file included, is seen.
2. With `--splits 1`: the start check passes and the first step starts (the stand-in is called
   with the sealed truth build and opens the made-up sealed file, which the watch sees), so the
   check of item 1 is the start check's refusal, not a drive that never starts.
3. A start that cannot be decided (the probe ends with its own `UNDECIDED` status, as when a model
   package is missing; `run_starts` reading T4) refuses the drive of each purpose (exit 2) with
   the line saying the list cannot be decided; no step starts and no log is written (`run_driver`
   reading D7; the review of 3d4fa78, note 1).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from drive_fake import entry
from test_run_starts import MODEL, ROUTE, git, working_copy
from test_sealed_matrix import files, honest

from fleetreserve import (
    cold_start_gate,
    run_driver,
    run_list,
    run_starts,
    run_verify,
    sealed_matrix,
)
from fleetreserve.sealed_route import TRUTH_BUILD_MODULE

_WP6GATE = Path(__file__).resolve().parents[1] / "wp6gate"  # the watch of sealed files
if str(_WP6GATE) not in sys.path:
    sys.path.insert(0, str(_WP6GATE))
import sealed_watch  # noqa: E402

LIST = Path("run-lists") / "run-list.json"
MATRIX = "gate/sealed-matrix/lgbm_tail.json"
SEALED_FILE = Path("data") / "heldout" / "made-up" / "sessions.csv"


class Started(Exception):
    """A step of the drive started (the stand-in of `run_driver.run_entry`)."""


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)


def listed(splits: list[str]) -> dict:
    compare = [*MODEL, "--dial", "{name:dial-workplaces}", "--matrix", MATRIX, *splits]
    return {
        "format": "run-list-2",
        "about": "a made-up run list",
        "units": {"rehearsal": {}, "sealed-run": {}},
        "fixed": {"seeds": {}, "counts": {}, "names": {"dial-workplaces": "0.05"}},
        "entries": [
            entry(
                "sealed-truth",
                [sys.executable, "-m", f"fleetreserve.{TRUTH_BUILD_MODULE}"],
                only="sealed-run",
            ),
            entry(
                "made-up-compare",
                [
                    sys.executable,
                    "-m",
                    "fleetreserve.compare_run",
                    *compare,
                    *ROUTE[:-1],
                    "{run:sealed-truth}",
                ],
                only="sealed-run",
            ),
            entry(
                "made-up-truth",
                [sys.executable, "-m", "fleetreserve.truth_build"],
                only="rehearsal",
            ),
        ],
    }


def copy_with(tmp_path: Path, document: dict) -> Path:
    top = working_copy(tmp_path)
    assert sealed_matrix.main([*files(tmp_path, honest()), "--out", str(top / "gate")]) == 0
    written = json.loads((top / MATRIX).read_text())["sealed_choice"]["dials"]
    assert written[sealed_matrix.WORKPLACES] == 0.05  # the list's fixed name dial-workplaces
    (top / LIST).parent.mkdir()
    (top / LIST).write_text(json.dumps(document, indent=1) + "\n")
    sealed = top / SEALED_FILE
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b"made-up sealed bytes\n")  # ignored: data/ is in .gitignore
    git(top, "add", "-A")
    git(top, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "list")
    assert git(top, "status", "--porcelain") == ""
    return top


def standing_in(monkeypatch, top: Path) -> list[str]:
    """The gate made to pass, and each step run by a stand-in that opens the made-up sealed file
    and stops the drive; the labels of the steps started."""
    started = []
    monkeypatch.setattr(
        cold_start_gate, "require_gate", lambda root: git(root, "rev-parse", "HEAD").strip()
    )

    def run_entry(root, k, step, command):
        started.append(step.label)
        (top / SEALED_FILE).read_bytes()
        raise Started(step.label)

    monkeypatch.setattr(run_driver, "run_entry", run_entry)
    return started


@pytest.mark.parametrize("purpose", run_list.PURPOSES)
def test_a_start_the_start_check_refuses_refuses_the_drive_before_any_step(
    tmp_path, monkeypatch, capsys, purpose
):
    top = copy_with(tmp_path, listed([]))
    started = standing_in(monkeypatch, top)
    with sealed_watch.watch(top / "data") as seen:
        code = run_driver.drive(LIST, purpose=purpose, start_dir=top)
    out = capsys.readouterr().out
    assert code == 2, out
    assert any(
        line.startswith("FAULT: entry 2 (made-up-compare): started at a made-up tag")
        and "--splits" in line
        and line.endswith("(the start check)")
        for line in out.splitlines()
    ), out
    assert started == []  # no step started
    assert seen == []  # nothing under the data folder opened, listed or scanned
    assert not (top / "runs.noindex").exists()  # no log, no run


def test_the_drive_starts_its_first_step_once_the_start_check_passes(tmp_path, monkeypatch, capsys):
    top = copy_with(tmp_path, listed(["--splits", "1"]))
    started = standing_in(monkeypatch, top)
    with sealed_watch.watch(top / "data") as seen, pytest.raises(Started):
        run_driver.drive(LIST, purpose=run_list.SEALED_RUN, start_dir=top)
    assert started == ["sealed-truth"], capsys.readouterr().out
    assert [path for _, path in seen] == [str(top / SEALED_FILE)]  # the watch sees an opening


@pytest.mark.parametrize("purpose", run_list.PURPOSES)
def test_a_start_that_cannot_be_decided_refuses_the_drive_before_any_step(
    tmp_path, monkeypatch, capsys, purpose
):
    top = copy_with(tmp_path, listed(["--splits", "1"]))  # a list whose starts pass
    started = standing_in(monkeypatch, top)
    real = run_starts.subprocess.run
    probes = []

    def undecided(arguments, *args, **kwargs):
        if f"{run_starts.PACKAGE}.run_starts" not in arguments:
            return real(arguments, *args, **kwargs)  # git, for the made-up repository
        probes.append(arguments)
        line = f"{run_starts.UNDECIDED_LINE}the package made_up is not installed (reading T4)"
        return subprocess.CompletedProcess(arguments, run_starts.UNDECIDED, line + "\n", "")

    monkeypatch.setattr(run_starts.subprocess, "run", undecided)
    with sealed_watch.watch(top / "data") as seen:
        code = run_driver.drive(LIST, purpose=purpose, start_dir=top)
    out = capsys.readouterr().out
    assert code == 2, out
    assert len(probes) == 1  # the one step of the route was probed
    assert "cannot be decided" in out and "the package made_up is not installed" in out, out
    assert started == []  # no step started
    assert seen == []
    assert not (top / "runs.noindex").exists()  # no log, no run

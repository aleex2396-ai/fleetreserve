"""The run list's start check (`run_starts`): each step of the sealed run that runs a command of the
sealed route is started by its own code at a made-up tag, up to its first read of data.

1. The three starts of the review of the run list's rules at c8370cb (its finding 2), each of
   which passed the list's rules and refused itself at its start: `compare_run` without
   `--splits 1`, `corner_run --days 10`, `drift_run --etas 1`; and the others that finding names
   by reading the code: `corner_run` with a `--seed` other than the matrix's split seed or
   `--methods` without the fixed plan, `drift_run --calibration-days`, and a model that reads
   other days' outcomes in `learn_run`, `compare_run` and `corner_run`. Each is a fault naming
   the command's own refusal.
2. The honest start of every command of the route (`run_list.ROUTE_COMMANDS`, `ceiling_report`
   with `--every-day` among them) on the nine held-out units reaches its first read of data
   (its stop of `run_starts.first_reads`), so the stops are pinned to each command; `drift_run`
   needs its step-size file committed at the tag, and without it its start is a fault naming
   the file.
3. By the checker's command line, on the main model's matrix that `sealed_matrix` writes from
   made-up kappa-choice results, its dial by its fixed name: a well-formed list whose
   `compare_run` step has no `--splits 1` is faulty (exit 1, the start check's line); with
   `--splits 1` it is well formed (exit 0) and its last line says the starts were made;
   `--no-starts` says they were not.
4. The made-up repository holds the working copy's commit and the running package, at the tag,
   clean, with no data folder.
5. The main model's sensitivity matrices at rho 0.1 and 0.2 that `sealed_matrix` writes (plan
   WP15): the start of a `learn_run` on each, on the nine held-out units, reaches its first read
   of data.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from drive_fake import entry, write_matrix
from git_run import run_git

from fleetreserve import run_list, run_starts, sealed_route

NINE = list(sealed_route.HELD_OUT_UNITS)
TOP = Path(__file__).resolve().parents[2]
ROUTE = [
    "--tables",
    "data/derived/development",
    "--truth-run",
    "data/derived/runs/development",
    "--held-out-truth-run",
    "runs.noindex/20261122T000000Z-sealed-truth",
]
DRIFT_ETA = {
    "rule": "largest-below-first-failure",
    "start": "zero",
    "grouping": "site-type",
    "units": ["caltech", "OSL_S"],
    "results": {"run_folder": "runs.noindex/made-up", "commit": "0" * 40},
    "eta": {"A": {"workplace": 0.5, "other": 5.0}, "B": {"workplace": 0.5, "other": 2.0}},
}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(folder: Path, *args: str) -> str:
    done = run_git(["git", "-C", str(folder), *args], capture_output=True, text=True, check=True)
    return done.stdout


def working_copy(folder: Path, *, drift_eta: bool = True) -> Path:
    """A made-up working copy: the committed default matrices, a matrix of the nine held-out
    units and, unless asked not to, a made-up step-size file of the drift guard."""
    root = folder / "copy"
    root.mkdir()
    git(root, "init", "-q")
    for path in ("gate/matrix.json", "learn/matrix.json"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes((TOP / path).read_bytes())
    write_matrix(root, "gate/nine.json", NINE)
    if drift_eta:
        (root / "learn" / "drift-eta.json").write_text(json.dumps(DRIFT_ETA, indent=1) + "\n")
    (root / ".gitignore").write_text("data/\nruns.noindex/\n")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "m")
    return root


@pytest.fixture(scope="module")
def made_up(tmp_path_factory):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        os.environ.pop(key)
    folder = tmp_path_factory.mktemp("starts")
    top = working_copy(folder)
    return run_starts.made_up_repository(top, folder).resolve()


def test_a_learn_run_on_each_sensitivity_matrix_reaches_its_first_read(tmp_path):
    from test_sealed_matrix import files, honest

    from fleetreserve import run_verify, sealed_matrix

    top = working_copy(tmp_path)
    check = run_verify._check_commit
    run_verify._check_commit = lambda folder, commit: None
    try:
        written = sealed_matrix.main([*files(tmp_path, honest()), "--out", str(top / "gate")])
    finally:
        run_verify._check_commit = check
    assert written == 0
    git(top, "add", "-A", "--", "gate")
    git(top, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "rho")
    root = run_starts.made_up_repository(top, tmp_path).resolve()
    for rho in sealed_matrix.SENSITIVITY_RHOS:
        matrix = f"gate/sealed-matrix/lgbm_tail-rho{rho:g}.json"
        assert json.loads((root / matrix).read_text())["rho"] == rho
        arguments = HONEST["learn_run"][:]
        arguments[arguments.index("gate/nine.json")] = matrix
        assert run_starts.start_fault(step("learn_run", *arguments), root) is None, rho


def step(module: str, *arguments: str) -> run_list.Step:
    return run_list.Step(
        position=0,
        label="made-up",
        command=(sys.executable, "-m", f"fleetreserve.{module}", *arguments),
        runs=(),
        setting="main",
        max_seconds=60,
        after=(),
    )


MODEL = ["--model", "km"]
DIAL = ["--dial", "0.02"]
NINE_MATRIX = ["--matrix", "gate/nine.json"]
HONEST = {
    "learn_run": [
        *MODEL,
        *NINE_MATRIX,
        *ROUTE,
        "--interval-seed",
        "1",
        "--interval-resamples",
        "2000",
    ],
    "compare_run": [*MODEL, *DIAL, *NINE_MATRIX, "--splits", "1", *ROUTE],
    "corner_run": [*MODEL, *DIAL, *NINE_MATRIX, "--days", "50", "--seed", "1", *ROUTE],
    "ledger_run": [
        "--run",
        "runs.noindex/20261122T000000Z-scored",
        *NINE_MATRIX,
        "--tree-days",
        "100",
        "--seed",
        "1",
        *ROUTE,
    ],
    "drift_run": [*MODEL, *DIAL, *NINE_MATRIX, "--units", *NINE, *ROUTE],
    "o3_run": [
        "--units",
        *NINE,
        "--days",
        "200",
        "--seed",
        "1",
        "--base-set",
        "day-average",
        *ROUTE,
    ],
    "cold_start": [*MODEL, "--reference", "declared", *DIAL, "--units", *NINE, *ROUTE],
    "ceiling_report": ["--units", *NINE, "--every-day", *ROUTE],
}
# (command, arguments, a text of the command's own refusal)
REFUSED = {
    "compare_run without --splits 1 (finding 2)": (
        "compare_run",
        [*MODEL, *DIAL, *NINE_MATRIX, *ROUTE],
        "--splits",
    ),
    "compare_run --splits 200": (
        "compare_run",
        [*MODEL, *DIAL, *NINE_MATRIX, "--splits", "200", *ROUTE],
        "--splits",
    ),
    "corner_run --days 10 (finding 2)": (
        "corner_run",
        [*MODEL, *DIAL, *NINE_MATRIX, "--days", "10", "--seed", "1", *ROUTE],
        "--days 10 is below",
    ),
    "corner_run --seed 2": (
        "corner_run",
        [*MODEL, *DIAL, *NINE_MATRIX, "--days", "50", "--seed", "2", *ROUTE],
        "--seed 2 is not the matrix's split_seed",
    ),
    "corner_run --methods without the fixed plan": (
        "corner_run",
        [*MODEL, *DIAL, *NINE_MATRIX, "--methods", "re-solve", *ROUTE],
        "without fixed-plan",
    ),
    "drift_run --etas 1 (finding 2)": (
        "drift_run",
        [*MODEL, *DIAL, *NINE_MATRIX, "--units", *NINE, "--etas", "1", *ROUTE],
        "--etas",
    ),
    "drift_run --calibration-days 50": (
        "drift_run",
        [*MODEL, *DIAL, *NINE_MATRIX, "--units", *NINE, "--calibration-days", "50", *ROUTE],
        "--calibration-days",
    ),
    "learn_run with a model that reads other days' outcomes": (
        "learn_run",
        ["--model", "empirical", *NINE_MATRIX, *ROUTE],
        "builds a scored unit's forecasts",
    ),
    "compare_run with a model that reads other days' outcomes": (
        "compare_run",
        ["--model", "empirical", *DIAL, *NINE_MATRIX, "--splits", "1", *ROUTE],
        "builds a scored unit's forecasts",
    ),
    "corner_run with a model that reads other days' outcomes": (
        "corner_run",
        ["--model", "empirical", *DIAL, *NINE_MATRIX, *ROUTE],
        "builds a scored unit's forecasts",
    ),
}


@pytest.mark.parametrize(
    ("module", "arguments", "refusal"), list(REFUSED.values()), ids=list(REFUSED)
)
def test_a_start_the_command_refuses_is_a_fault(made_up, module, arguments, refusal):
    found = run_starts.start_fault(step(module, *arguments), made_up)
    assert found is not None
    assert "started at a made-up tag" in found and refusal in found, found


@pytest.mark.parametrize("module", list(run_list.ROUTE_COMMANDS))
def test_the_honest_start_of_every_route_command_reaches_its_first_read(made_up, module):
    if module == "gate_run":  # it refuses every run on the route (sealed_route reading S3)
        found = run_starts.start_fault(step(module, *NINE_MATRIX, *ROUTE), made_up)
        assert found is not None and "builds a scored unit's forecasts" in found, found
        return
    assert set(HONEST) == set(run_list.ROUTE_COMMANDS) - {"gate_run"}
    found = run_starts.start_fault(step(module, *HONEST[module]), made_up)
    assert found is None, found


def test_the_probe_stops_at_a_stop_of_its_command(made_up):
    done = subprocess.run(
        [sys.executable, "-m", "fleetreserve.run_starts", "compare_run", str(made_up)]
        + HONEST["compare_run"],
        cwd=made_up,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == run_starts.REACHED, (done.stdout, done.stderr)
    assert done.stdout.splitlines()[-1] == "REACHED DATA: sealed_route.check_split_rule"


def test_drift_run_without_its_step_size_file_at_the_tag_is_a_fault(tmp_path):
    top = working_copy(tmp_path, drift_eta=False)
    root = run_starts.made_up_repository(top, tmp_path).resolve()
    found = run_starts.start_fault(step("drift_run", *HONEST["drift_run"]), root)
    assert found is not None and "learn/drift-eta.json" in found, found


def test_the_made_up_repository_is_the_commit_and_the_running_package_at_the_tag(made_up):
    assert git(made_up, "status", "--porcelain") == ""
    assert git(made_up, "rev-parse", "HEAD") == git(made_up, "rev-parse", "frozen-plan^{commit}")
    assert (made_up / "gate" / "nine.json").is_file()
    assert not (made_up / "data").exists()
    package = Path(run_starts.__file__).parent
    assert (made_up / "src" / "fleetreserve" / "run_starts.py").read_bytes() == (
        package / "run_starts.py"
    ).read_bytes()


# --- 3. by the checker's command line ------------------------------------------------------------


def listed(compare: list[str]) -> dict:
    return {
        "format": "run-list-2",
        "about": "a made-up run list",
        "units": {"rehearsal": {}, "sealed-run": {}},
        "fixed": {"seeds": {}, "counts": {}, "names": {}},
        "entries": [
            entry(
                "sealed-truth",
                [sys.executable, "-m", f"fleetreserve.{sealed_route.TRUTH_BUILD_MODULE}"],
                only="sealed-run",
            ),
            entry(
                "made-up-compare",
                [sys.executable, "-m", "fleetreserve.compare_run", *compare],
                only="sealed-run",
            ),
            entry(
                "made-up-truth",
                [sys.executable, "-m", "fleetreserve.truth_build"],
                only="rehearsal",
            ),
        ],
    }


@pytest.mark.parametrize("splits", [[], ["--splits", "1"]], ids=["no splits", "splits 1"])
def test_the_checker_runs_the_starts(tmp_path, capsys, monkeypatch, splits):
    from test_sealed_matrix import files, honest

    from fleetreserve import run_verify, sealed_matrix

    top = working_copy(tmp_path)
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    assert sealed_matrix.main([*files(tmp_path, honest()), "--out", str(top / "gate")]) == 0
    path = top / "run-lists" / "run-list.json"
    path.parent.mkdir()
    route = [*ROUTE[:-1], "{run:sealed-truth}"]
    # the dial by its fixed name, at the dial the main model's matrix gives (rule 8 of run_list)
    dial = ["--dial", "{name:dial-workplaces}"]
    matrix = ["--matrix", "gate/sealed-matrix/lgbm_tail.json"]
    document = listed([*MODEL, *dial, *matrix, *splits, *route])
    document["fixed"]["names"]["dial-workplaces"] = "0.05"
    path.write_text(json.dumps(document))
    git(top, "add", "-A")
    git(top, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "list")
    code = run_list.main([str(path)])
    lines = capsys.readouterr().out.splitlines()
    if splits:
        assert code == 0, lines
        assert lines[-2] == (
            "starts: the starts of 1 step of the sealed route made at a made-up tag, none refused"
        ), lines
    else:
        assert code == 1, lines
        assert any("(the start check)" in line and "--splits" in line for line in lines), lines
    assert run_list.main([str(path), "--no-starts"]) == 0
    shown = capsys.readouterr().out.splitlines()[-2]
    assert shown == "starts: the starts not made (--no-starts)", shown

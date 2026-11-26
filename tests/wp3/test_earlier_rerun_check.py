"""`tools/earlier_rerun_check.py` on made-up recorded runs.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real. The command of each run only writes a made-up results file.
Long runs are made by rewriting the manifest's start and end, since a run of many hours cannot be
made here. How long a run took never decides an item (plan section 13, WP3).
"""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve.run_record import record_run

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
check = importlib.import_module("earlier_rerun_check")
earlier_rerun = importlib.import_module("earlier_rerun")

WRITE = (
    "import os, sys; "
    "open(os.path.join(os.environ['FLEETRESERVE_RUN_DIR'], 'earlier-rerun.json'), 'w')"
    ".write(sys.argv[1]); sys.exit(int(sys.argv[2]))"
)
SLEEP_AFTER_WRITE = WRITE.replace("sys.exit(int(sys.argv[2]))", "import time; time.sleep(60)")
GOOD = {
    "perfect_knowledge_kwh": 45.6,
    "learned_share": 0.004,
    "population_share": 0.039,
}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> None:
    run_git(["git", "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n")
    (root / "tracked.txt").write_text("one\n")
    git(root, "add", ".gitignore", "tracked.txt")
    git(root, "commit", "-q", "-m", "first")
    return root


def make_run(
    repo: Path,
    measured: dict | None = None,
    *,
    label: str = "wp3-rerun",
    exit_code: int = 0,
    results_text: str | None = None,
    time_out: bool = False,
) -> Path:
    """Record a run whose command writes the results file; return its run folder."""
    text = results_text if results_text is not None else json.dumps({"measured": measured or GOOD})
    runs = repo / "runs.noindex"
    before = set(runs.glob("*")) if runs.exists() else set()
    if time_out:
        status = record_run(
            label,
            [sys.executable, "-c", SLEEP_AFTER_WRITE, text, "0"],
            start_dir=repo,
            max_seconds=1,
            grace_seconds=0.2,
        )
    else:
        status = record_run(
            label, [sys.executable, "-c", WRITE, text, str(exit_code)], start_dir=repo
        )
    assert status == exit_code or time_out
    (folder,) = set(runs.glob("*")) - before
    return folder


def rename(folder: Path, stamp: str, label: str = "wp3-rerun") -> Path:
    target = folder.parent / f"{stamp}-{label}"
    folder.rename(target)
    return target


def set_span(folder: Path, seconds: int) -> None:
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["start"] = "2026-11-11T00:00:00Z"
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    manifest["end"] = f"2026-11-11T{hours:02d}:{minutes:02d}:{secs:02d}Z"
    (folder / "manifest.json").write_text(json.dumps(manifest))


def run_check(repo: Path, item: str, *extra: str) -> int:
    return check.main([f"--{item}", "--top", str(repo), *extra])


# --- the limits ----------------------------------------------------------------------------------

# (item, measured value, expected exit status). The edges are exactly 0.1 away from the published
# value in the item's own unit (kWh, or percentage points).
LIMIT_CASES = [
    ("kwh", 45.6, 0),
    ("kwh", 45.7, 0),  # exactly 0.1 above: 45.7 - 45.6 is 0.10000000000000142 in binary floats
    ("kwh", 45.5, 0),  # exactly 0.1 below
    ("kwh", 45.70000000000001, 1),  # one unit in the last place beyond the limit
    ("kwh", 45.49999999999999, 1),
    ("kwh", 45.71, 1),
    ("kwh", 0.0, 1),
    ("learned", 0.004, 0),
    ("learned", 0.005, 0),  # 0.5 %, exactly 0.1 point above
    ("learned", 0.003, 0),  # 0.3 %, exactly 0.1 point below
    ("learned", 0.0050000000000001, 1),
    ("learned", 0.0029999999999999, 1),
    ("learned", 0.0, 1),
    ("learned", 0.039, 1),
    ("population", 0.039, 0),
    ("population", 0.04, 0),  # 4.0 %
    ("population", 0.038, 0),  # 3.8 %
    ("population", 0.0401, 1),
    ("population", 0.0379, 1),
    ("population", 0.004, 1),
]


@pytest.mark.parametrize(("item", "value", "expected"), LIMIT_CASES)
def test_limits_are_inclusive_and_exact(repo, capsys, item, value, expected):
    name = check.ITEMS[item][0]
    folder = make_run(repo, {**GOOD, name: value})
    assert run_check(repo, item, "--run", str(folder)) == expected, capsys.readouterr().out


def test_the_edge_would_fail_with_plain_float_arithmetic():
    # Why the check uses decimal numbers: a plain float comparison rejects an exact edge.
    assert abs(45.7 - 45.6) > 0.1
    assert check.difference(check.exact(45.7), check.exact(45.6)) == check.exact(0.1)


def test_a_pass_prints_the_measured_and_the_published_value(repo, capsys):
    make_run(repo, {**GOOD, "learned_share": 0.0041234567})
    capsys.readouterr()
    assert run_check(repo, "learned") == 0
    out = capsys.readouterr().out
    assert "measured 0.41234567 %" in out and "published 0.4 %" in out
    assert "difference 0.01234567" in out


def test_a_failure_says_which_way_it_missed(repo, capsys):
    make_run(repo, {**GOOD, "perfect_knowledge_kwh": 47.0})
    capsys.readouterr()  # the recorder's own lines
    assert run_check(repo, "kwh") == 1
    out = capsys.readouterr().out
    assert out.startswith("FAILED: outside the limit") and "measured 47 kWh" in out


def test_the_outcome_line_names_the_run_folder_it_read(repo, capsys):
    """Red when `check_number` drops `; run <folder name>` from the end of its outcome text: the
    line then ends at `inclusive)` and names no run, whichever run the item read."""
    old = rename(make_run(repo, {**GOOD, "perfect_knowledge_kwh": 45.6}), "20260101T000000Z")
    new = rename(make_run(repo, {**GOOD, "perfect_knowledge_kwh": 50.0}), "20260102T000000Z")
    capsys.readouterr()  # the recorder's own lines
    assert run_check(repo, "kwh", "--run", str(old)) == 0
    out = capsys.readouterr().out
    assert out.startswith("within the limit: ") and out.strip().endswith(f"; run {old.name}")
    assert run_check(repo, "kwh") == 1  # no --run: the newest, which fails
    out = capsys.readouterr().out
    assert out.startswith("FAILED: outside the limit: ")
    assert out.strip().endswith(f"(limit 0.1, inclusive); run {new.name}"), out


def test_each_item_reads_only_its_own_number(repo):
    make_run(repo, {**GOOD, "learned_share": 0.5})  # only the learned share is far off
    assert run_check(repo, "kwh") == 0
    assert run_check(repo, "learned") == 1
    assert run_check(repo, "population") == 0


# --- run time is never judged --------------------------------------------------------------------


@pytest.mark.parametrize("seconds", [1, 21601, 86399])
def test_a_finished_run_has_its_numbers_judged_however_long_it_took(repo, seconds):
    folder = make_run(repo, {**GOOD, "learned_share": 0.5})
    set_span(folder, seconds)
    assert run_check(repo, "kwh") == 0
    assert run_check(repo, "learned") == 1
    assert run_check(repo, "population") == 0


def test_there_is_no_time_item(repo):
    make_run(repo)
    with pytest.raises(SystemExit) as raised:
        check.main(["--time", "--top", str(repo)])
    assert raised.value.code == 2


# --- which run, and whether it verifies ----------------------------------------------------------


def test_no_run_folder_cannot_run(repo, capsys):
    capsys.readouterr()
    for item in ("kwh", "learned", "population"):
        assert run_check(repo, item) == 2
    assert "no run named" in capsys.readouterr().out


def test_a_run_folder_given_that_does_not_exist_cannot_run(repo):
    assert run_check(repo, "kwh", "--run", str(repo / "runs.noindex" / "nothing")) == 2


def test_other_labels_do_not_count(repo):
    make_run(repo, label="wp3-rerun-extra")
    make_run(repo, label="wp1-passmark")
    assert run_check(repo, "kwh") == 2


def test_the_newest_rerun_is_the_one_read(repo, capsys):
    old = rename(make_run(repo, {**GOOD, "perfect_knowledge_kwh": 45.6}), "20260101T000000Z")
    new = rename(make_run(repo, {**GOOD, "perfect_knowledge_kwh": 50.0}), "20260102T000000Z")
    assert old.name < new.name
    assert run_check(repo, "kwh") == 1  # the newer, failing run, not the older passing one
    assert run_check(repo, "kwh", "--run", str(old)) == 0


def test_a_newer_run_without_a_manifest_is_not_skipped_for_an_older_good_one(repo):
    rename(make_run(repo), "20260101T000000Z")
    unfinished = repo / "runs.noindex" / "20260102T000000Z-wp3-rerun"
    unfinished.mkdir()
    (unfinished / "earlier-rerun.json").write_text(json.dumps({"measured": GOOD}))
    assert run_check(repo, "kwh") == 2


def test_a_changed_results_file_is_not_verified(repo, capsys):
    folder = make_run(repo)
    (folder / "earlier-rerun.json").write_text(
        json.dumps({"measured": {**GOOD, "learned_share": 0.004}}) + " "
    )
    assert run_check(repo, "learned") == 2
    assert "is not verified" in capsys.readouterr().out


def test_a_run_that_failed_is_not_verified(repo, capsys):
    make_run(repo, exit_code=1)
    capsys.readouterr()
    for item in ("kwh", "learned", "population"):
        assert run_check(repo, item) == 2
    assert "exit code 1" in capsys.readouterr().out


def test_a_run_without_the_results_file_cannot_run(repo):
    folder = make_run(repo)
    (folder / "earlier-rerun.json").unlink()
    assert run_check(repo, "kwh") == 2


@pytest.mark.parametrize(
    "results",
    [
        "{}",
        '{"measured": {}}',
        '{"measured": {"perfect_knowledge_kwh": "45.6"}}',
        '{"measured": {"perfect_knowledge_kwh": NaN}}',
        '{"measured": {"perfect_knowledge_kwh": Infinity}}',
        '{"measured": {"perfect_knowledge_kwh": true}}',
        '{"measured": {"perfect_knowledge_kwh": null}}',
        "not json",
    ],
)
def test_a_results_file_without_a_usable_number_cannot_run(repo, results):
    make_run(repo, results_text=results)
    assert run_check(repo, "kwh") == 2


def test_a_manifest_that_cannot_be_read_cannot_run(repo):
    folder = make_run(repo)
    (folder / "manifest.json").write_text("{")
    assert run_check(repo, "kwh") == 2


# --- a run stopped by the recorder's stop limit ---------------------------------------------------


def test_a_stopped_run_could_not_run_and_says_the_stop_limit_is_no_pass_mark(repo, capsys):
    folder = make_run(repo, time_out=True)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["timed_out"] is True
    capsys.readouterr()
    for item in ("kwh", "learned", "population"):
        assert run_check(repo, item) == 2
        out = capsys.readouterr().out
        assert out.startswith(f"COULD NOT RUN: {folder.name} was stopped by the run recorder")
        assert "stop limit of 1 s" in out and "not a pass mark" in out
        assert "hours" not in out and "published values cited" not in out


def test_a_stopped_run_does_not_pass_even_when_its_results_file_exists_and_matches(repo):
    # The command wrote matching numbers and then ran past the stop limit: still no pass.
    folder = make_run(repo, time_out=True)
    assert (folder / "earlier-rerun.json").is_file()
    for item in ("kwh", "learned", "population"):
        assert run_check(repo, item) == 2


# --- arguments -----------------------------------------------------------------------------------


def test_exactly_one_item_is_required(repo):
    for arguments in ([], ["--kwh", "--learned"], ["--learned", "--population"]):
        with pytest.raises(SystemExit) as raised:
            check.main([*arguments, "--top", str(repo)])
        assert raised.value.code == 2


def test_the_published_values_are_the_plans():
    assert earlier_rerun.PUBLISHED == {
        "perfect_knowledge_kwh": 45.6,
        "learned_share": 0.004,
        "population_share": 0.039,
    }
    assert earlier_rerun.TOLERANCE_KWH == 0.1
    assert earlier_rerun.TOLERANCE_POINTS == 0.1

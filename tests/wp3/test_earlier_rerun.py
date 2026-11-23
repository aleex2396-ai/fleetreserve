"""`tools/earlier_rerun.py` on a made-up earlier package and made-up month files.

The stage runs for real as a separate process, with a stand-in package (see
`fake_earlier_package.py`) whose tag check really runs git. No real package and no real data are
used. The tool is called in this process (so its pinned file list can be pointed at the stand-in)
and once as a separate process, as the run recorder starts it.
"""

import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import fake_earlier_package as fake
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
earlier_rerun = importlib.import_module("earlier_rerun")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """No git variable of the caller, a stated token-file variable, and the stand-in's pins."""
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("ACN_API_TOKEN_FILE", "not-a-real-file")  # must not reach the stage
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)


@pytest.fixture
def run_dir(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "run"
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    return folder


def use_pins(monkeypatch, top: Path) -> None:
    monkeypatch.setattr(earlier_rerun, "PINNED_FILES", fake.pins_for(top))


def run_tool(monkeypatch, top: Path, *arguments: str) -> int:
    use_pins(monkeypatch, top)
    return earlier_rerun.main(["--top", str(top), *arguments])


def results_of(run_dir: Path) -> dict:
    return json.loads((run_dir / "earlier-rerun.json").read_text())


def seen_by_stage(run_dir: Path) -> dict:
    return json.loads((run_dir / "evaluation" / "rows.json").read_text())


# --- the whole path ------------------------------------------------------------------------------


def test_full_path_writes_the_three_numbers_and_the_record(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    package = top / "reference" / "part1"
    before = fake.files_under(package)
    assert run_tool(monkeypatch, top, "--workers", "3") == 0
    out = capfd.readouterr().out
    assert "wrote evaluate evaluation" in out  # the stage's own line comes straight through
    assert "no file of the earlier analysis was changed" in out.lower()
    results = results_of(run_dir)
    assert results["measured"] == {
        "perfect_knowledge_kwh": 45.63456789012345,  # full precision, not rounded
        "learned_share": 0.00412345678901234,
        "population_share": 0.0391,
    }
    assert results["published"]["perfect_knowledge_kwh"] == 45.6
    assert results["published"]["learned_share"] == 0.004
    assert results["published"]["population_share"] == 0.039
    assert "published" in results["published"]["label"]
    assert results["scored_days"] == {"perfect_knowledge": 120, "learned": 118, "population": 117}
    assert results["workers"] == 3
    assert results["package_commit_text"] == "commit 05a6cf9"
    assert "exported unchanged" in results["package_provenance"]
    assert results["disclosure"] == earlier_rerun.DISCLOSURE
    assert "scratch" in results["disclosure"]
    assert results["elapsed_seconds"] >= results["stage_seconds"] >= 0
    assert results["offers_counted_from"] == "2019-09-01"
    assert results["row_filters"]["learned_share"]["who"] == "4d"
    # The package was not changed or added to (no bytecode either).
    assert fake.files_under(package) == before


def test_the_stage_sees_the_data_links_the_tag_and_a_clean_environment(
    tmp_path, run_dir, monkeypatch
):
    top = fake.build_top(tmp_path)
    monkeypatch.setenv("GIT_INDEX_FILE", "inherited-from-the-caller")
    assert run_tool(monkeypatch, top, "--workers", "2") == 0
    seen = seen_by_stage(run_dir)
    # Exactly the months the stage may open: the two sealed months in the made-up folder are absent.
    assert seen["months"] == [f"{m}.json.gz" for m in fake.months_between("2018-04", "2020-02")]
    assert "2020-03.json.gz" in os.listdir(top / "data" / "raw" / "acn" / "caltech")
    links = sorted((run_dir / "dataroot" / "raw" / "caltech").iterdir())
    assert len(links) == 23 and all(link.is_symlink() for link in links)
    # The tag check was satisfied by the scratch repository inside the run folder, and only there.
    assert seen["git_dir"] == str(run_dir / "scratch-tag.git")
    assert seen["git_index_file"] is None
    assert seen["token_file_variable"] is None
    assert seen["no_bytecode"] == "1"
    assert seen["workers"] == 2
    assert seen["cwd"] == str((top / "reference" / "part1").resolve())


def test_without_the_scratch_repository_the_stand_in_refuses_as_the_real_package_does(tmp_path):
    top = fake.build_top(tmp_path)
    package = top / "reference" / "part1"
    interpreter = package / ".venv" / "bin" / "python"
    probe = "from fleetoffer import data; print(data.plan_is_tagged())"

    def tagged(git_dir):
        done = subprocess.run(
            [str(interpreter), "-c", probe],
            cwd=package,
            env=earlier_rerun.package_environment(package, git_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        return done.stdout.strip()

    assert tagged(None) == "False"
    scratch_parent = tmp_path / "scratch_parent"
    scratch_parent.mkdir()
    assert tagged(earlier_rerun.make_scratch_repository(scratch_parent)) == "True"


def test_scratch_repository_holds_one_empty_commit_with_the_tag(tmp_path):
    scratch = earlier_rerun.make_scratch_repository(tmp_path)
    environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "PATH": os.environ["PATH"],
    }

    def git(*arguments):
        return subprocess.run(
            ["git", f"--git-dir={scratch}", *arguments],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        ).stdout.strip()

    assert git("tag", "--list") == "analysis-plan"
    assert git("rev-list", "--count", "analysis-plan") == "1"
    assert git("ls-tree", "-r", "analysis-plan") == ""
    assert not list(scratch.glob("hooks/*"))  # no sample files


def test_a_separate_process_does_the_same(tmp_path, run_dir):
    # Called as the recorder calls it; the real pinned list does not match the stand-in, so the
    # tool refuses (exit 2) before it runs anything. This shows the real list is in force.
    top = fake.build_top(tmp_path)
    done = subprocess.run(
        [sys.executable, str(TOOLS / "earlier_rerun.py"), "--top", str(top)],
        capture_output=True,
        text=True,
        env={**os.environ, "FLEETRESERVE_RUN_DIR": str(run_dir)},
        check=False,
    )
    assert done.returncode == 2
    assert "COULD NOT RUN: the earlier package is not the stated export" in done.stdout
    assert not (run_dir / "evaluation").exists()


# --- the stage or the reading fails --------------------------------------------------------------


def test_a_failing_stage_is_exit_1_and_writes_no_results(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path, stage_exit=3)
    assert run_tool(monkeypatch, top) == 1
    assert "FAILED: the stage ended with exit code 3" in capfd.readouterr().out
    assert not (run_dir / "earlier-rerun.json").exists()


def test_a_failing_accounting_step_is_exit_1(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    # The stand-in accounting step reads rows.json, so remove it once the stage is done.
    original = earlier_rerun.run_stage

    def stage_then_lose_table(*args):
        original(*args)
        (run_dir / "evaluation" / "rows.json").unlink()

    monkeypatch.setattr(earlier_rerun, "run_stage", stage_then_lose_table)
    assert run_tool(monkeypatch, top) == 1
    assert "FAILED: the package's own accounting step failed" in capfd.readouterr().out
    assert not (run_dir / "earlier-rerun.json").exists()


def without(rows: list[dict], **match) -> list[dict]:
    return [row for row in rows if not all(row.get(k) == v for k, v in match.items())]


@pytest.mark.parametrize(
    ("label", "rows", "named"),
    [
        (
            "no perfect-knowledge row of the average form",
            without(fake.GOOD_ROWS, form="average", step="perfect knowledge"),
            "perfect_knowledge_kwh",
        ),
        (
            "no 4d row",
            without(fake.GOOD_ROWS, who="4d"),
            "learned_share",
        ),
        (
            "no 5d row",
            without(fake.GOOD_ROWS, who="5d"),
            "population_share",
        ),
        (
            "two 4d rows",
            [*fake.GOOD_ROWS, {**fake.GOOD_ROWS[3], "share_of_perfect": 0.5}],
            "learned_share",
        ),
        (
            "the 5d row has a share that is not a number",
            [
                *without(fake.GOOD_ROWS, who="5d"),
                {**fake.GOOD_ROWS[4], "share_of_perfect": math.nan},
            ],
            "population_share",
        ),
        (
            "the perfect-knowledge size is text",
            [
                *without(fake.GOOD_ROWS, form="average", step="perfect knowledge"),
                {**fake.GOOD_ROWS[1], "size_mean_kwh": "45.6"},
            ],
            "perfect_knowledge_kwh",
        ),
    ],
)
def test_a_missing_or_unusable_row_is_a_failure_that_lists_the_rows(
    tmp_path, run_dir, monkeypatch, capfd, label, rows, named
):
    top = fake.build_top(tmp_path, rows=rows)
    assert run_tool(monkeypatch, top) == 1, label
    out = capfd.readouterr().out
    assert f"FAILED: {named}" in out, label
    assert not (run_dir / "earlier-rerun.json").exists(), label
    if "not a number" not in out:
        assert "rows of the average form" in out and "'perfect knowledge'" in out


def test_a_pinned_file_changed_during_the_run_is_a_failure(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path, touch_package=True)
    assert run_tool(monkeypatch, top) == 1
    assert "FAILED: the earlier package changed during the run: data.py differs" in (
        capfd.readouterr().out
    )
    assert not (run_dir / "earlier-rerun.json").exists()


# --- it cannot run -------------------------------------------------------------------------------


def test_no_run_folder_variable_cannot_run(tmp_path, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    assert run_tool(monkeypatch, top) == 2
    assert "FLEETRESERVE_RUN_DIR must name an existing run folder" in capfd.readouterr().out


def test_a_run_folder_that_does_not_exist_cannot_run(tmp_path, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path / "not_made"))
    assert run_tool(monkeypatch, top) == 2


def test_changed_provenance_text_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path, provenance="exported unchanged at commit 1234567\n")
    assert run_tool(monkeypatch, top) == 2
    assert "PROVENANCE.txt does not say: commit 05a6cf9" in capfd.readouterr().out
    assert not (run_dir / "evaluation").exists()


def test_a_changed_pinned_file_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    use_pins(monkeypatch, top)
    levels = top / "reference" / "part1" / "src" / "fleetoffer" / "levels.py"
    levels.write_text(levels.read_text() + "# edited\n")
    assert earlier_rerun.main(["--top", str(top)]) == 2
    out = capfd.readouterr().out
    assert "not the stated export" in out and "levels.py differs" in out
    assert not (run_dir / "evaluation").exists()


def test_a_missing_pinned_file_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    use_pins(monkeypatch, top)
    (top / "reference" / "part1" / "src" / "fleetoffer" / "evaluation.py").unlink()
    assert earlier_rerun.main(["--top", str(top)]) == 2
    assert "evaluation.py is missing" in capfd.readouterr().out


def test_no_package_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    assert earlier_rerun.main(["--top", str(tmp_path)]) == 2
    assert "the earlier package is not in reference/part1" in capfd.readouterr().out


def test_no_interpreter_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    (top / "reference" / "part1" / ".venv" / "bin" / "python").unlink()
    assert run_tool(monkeypatch, top) == 2
    assert "interpreter is missing" in capfd.readouterr().out


def test_modules_that_do_not_import_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path, broken_import=True)
    assert run_tool(monkeypatch, top) == 2
    out = capfd.readouterr().out
    assert "do not import with its own interpreter" in out and "broken on purpose" in out
    assert not (run_dir / "evaluation").exists()


@pytest.mark.parametrize("missing", ["2018-04", "2019-11", "2020-02"])
def test_a_missing_month_cannot_run_and_the_stage_is_not_started(
    tmp_path, run_dir, monkeypatch, capfd, missing
):
    top = fake.build_top(tmp_path)
    (top / "data" / "raw" / "acn" / "caltech" / f"{missing}.json.gz").unlink()
    assert run_tool(monkeypatch, top) == 2
    out = capfd.readouterr().out
    assert "1 Caltech month file(s) missing or empty" in out and f"{missing}.json.gz" in out
    assert not (run_dir / "evaluation").exists()
    assert not (run_dir / "dataroot").exists()


def test_an_empty_month_file_cannot_run(tmp_path, run_dir, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    (top / "data" / "raw" / "acn" / "caltech" / "2019-10.json.gz").write_bytes(b"")
    assert run_tool(monkeypatch, top) == 2
    assert "2019-10.json.gz" in capfd.readouterr().out


def test_months_after_february_2020_are_not_required(tmp_path, run_dir, monkeypatch):
    top = fake.build_top(tmp_path, months=fake.months_between("2018-04", "2020-02"))
    assert run_tool(monkeypatch, top) == 0


# --- check only, arguments -----------------------------------------------------------------------


def test_check_only_needs_no_run_folder_and_writes_nothing(tmp_path, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    before = fake.files_under(tmp_path)
    assert run_tool(monkeypatch, top, "--check-only") == 0
    assert "ready: the earlier package is the stated export; 23 month files found" in (
        capfd.readouterr().out
    )
    assert fake.files_under(tmp_path) == before


def test_check_only_reports_a_missing_month(tmp_path, monkeypatch, capfd):
    top = fake.build_top(tmp_path)
    (top / "data" / "raw" / "acn" / "caltech" / "2019-09.json.gz").unlink()
    assert run_tool(monkeypatch, top, "--check-only") == 2
    assert "2019-09.json.gz" in capfd.readouterr().out


@pytest.mark.parametrize("workers", ["0", "-1", "many"])
def test_bad_worker_counts_are_refused(tmp_path, workers):
    with pytest.raises(SystemExit) as raised:
        earlier_rerun.main(["--top", str(tmp_path), "--workers", workers])
    assert raised.value.code == 2


def test_the_default_is_six_workers(tmp_path, run_dir, monkeypatch):
    top = fake.build_top(tmp_path)
    assert run_tool(monkeypatch, top) == 0
    assert seen_by_stage(run_dir)["workers"] == 6 == earlier_rerun.DEFAULT_WORKERS


def test_the_pinned_list_names_files_that_decide_the_numbers():
    # The real pins are checked against the real package by the smoke check; here only their shape.
    assert {"pipeline.py", "levels.py", "evaluation.py", "data.py", "offer.py"} <= set(
        earlier_rerun.PINNED_FILES
    )
    assert all(len(value) == 64 for value in earlier_rerun.PINNED_FILES.values())
    assert earlier_rerun.months_needed()[0] == "2018-04"
    assert earlier_rerun.months_needed()[-1] == "2020-02"
    assert len(earlier_rerun.months_needed()) == 23

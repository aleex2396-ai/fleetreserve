"""`tools/dev_day_check.py` on made-up recorded runs (`recorded_fakes`) whose results file is the
one the command really writes on the made-up build (`dev_day_fakes.reference_document`), changed
where a test needs it: each item at pass, fail and could not run, configurations, coverage,
timeouts, staleness, which run is read, and runs the command made with `--units` naming other
units than its default list."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from dev_day_fakes import SAMPLE, reference_document, run_command
from recorded_fakes import (
    clean_git_environment,
    docs_commit,
    earlier_note,
    make_repo,
    make_run,
    new_commit,
)

from fleetreserve.dev_day_checks import RESULTS_NAME, UNITS

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
check = importlib.import_module("dev_day_check")

LABEL = "wp5-dev-days"
T2_CONFIG = [
    "--energy-bounds",
    "known",
    "--uncertifiable-car",
    "leave-unsteered",
    "--kappa",
    "0.1",
    "--departure-floor",
    "as-planned",
]
ARGS = {
    "order": ["--order"],
    "t4_log": ["--t4-log"],
    "t2": ["--t2", "--sample-size", str(SAMPLE), *T2_CONFIG],
    "two_way": ["--two-way", "--sample-size", str(SAMPLE)],
}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    clean_git_environment(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def good() -> dict:
    return json.loads(reference_document())


def recorded(repo, doc, **options):
    return make_run(repo, LABEL, RESULTS_NAME, doc, **options)


def run_check(repo: Path, item: str, *extra: str) -> int:
    return check.main([*ARGS[item], "--top", str(repo), *extra])


def fail(doc: dict, item: str) -> dict:
    doc["checks"][item]["verdict"] = "failed"
    doc["checks"][item]["failure_lines"] = ["made-up day: a made-up violation"]
    return doc


def test_the_reference_run_passes_every_item(repo, capsys):
    recorded(repo, good())
    capsys.readouterr()
    for item in ARGS:
        assert run_check(repo, item) == 0, item
    out = capsys.readouterr().out
    assert "certified on" in out and "days in E'" in out  # shown with the T2 claim
    assert "V_hind = V_caus held on all" in out


@pytest.mark.parametrize("item", list(ARGS))
def test_a_failed_check_in_a_run_that_ended_with_exit_1_fails_the_item(repo, capsys, item):
    recorded(repo, fail(good(), item), exit_code=1)
    capsys.readouterr()
    assert run_check(repo, item) == 1
    assert "FAILED: made-up day: a made-up violation" in capsys.readouterr().out


@pytest.mark.parametrize("item", list(ARGS))
def test_another_check_failing_leaves_a_passed_item_could_not_run(repo, capsys, item):
    other = "order" if item != "order" else "t2"
    recorded(repo, fail(good(), other), exit_code=1)
    capsys.readouterr()
    assert run_check(repo, item) == 2
    assert "not verified and cannot pass" in capsys.readouterr().out


@pytest.mark.parametrize("item", list(ARGS))
def test_a_check_with_a_timeout_could_not_run(repo, capsys, item):
    doc = good()
    check_doc = doc["checks"][item]
    check_doc.update(
        verdict="could not run",
        covered_days=check_doc["of_days"] - 1,
        timeout_days=1,
        could_not_run_reasons=["covered N of M days: 1 with a timeout"],
    )
    recorded(repo, doc, exit_code=2)
    capsys.readouterr()
    assert run_check(repo, item) == 2
    assert "1 with a timeout" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("item", "key", "step"),
    [
        ("order", "covered_days", -1),
        ("t2", "timeout_days", 1),
        ("two_way", "other_missing_days", 1),
        ("t4_log", "covered_days", -1),
    ],
)
def test_a_passed_verdict_without_full_cover_could_not_run(repo, item, key, step):
    doc = good()
    doc["checks"][item][key] += step
    recorded(repo, doc)
    assert run_check(repo, item) == 2


def test_a_run_that_ended_otherwise_than_0_1_or_2_is_not_read(repo, capsys):
    recorded(repo, fail(good(), "order"), exit_code=3)
    capsys.readouterr()
    assert run_check(repo, "order") == 2
    assert "is not verified" in capsys.readouterr().out


@pytest.mark.parametrize("item", ["order", "t4_log"])
def test_a_check_over_fewer_days_than_the_counting_days_could_not_run(repo, item):
    doc = good()
    doc["checks"][item]["of_days"] -= 1
    doc["checks"][item]["covered_days"] -= 1
    recorded(repo, doc)
    assert run_check(repo, item) == 2


def test_a_t4_log_not_listing_every_day_could_not_run(repo):
    doc = good()
    doc["t4_log_days"] = doc["t4_log_days"][:-1]
    recorded(repo, doc)
    assert run_check(repo, "t4_log") == 2


def test_a_t4_log_with_a_hypothesis_day_unchecked_could_not_run(repo, capsys):
    doc = good()
    doc["checks"]["t4_log"]["hypothesis_days_checked"] -= 1
    recorded(repo, doc)
    capsys.readouterr()
    assert run_check(repo, "t4_log") == 2
    assert "was not checked on every hypothesis day" in capsys.readouterr().out


def test_an_unknown_verdict_could_not_run(repo):
    doc = good()
    doc["checks"]["order"]["verdict"] = "fine"
    recorded(repo, doc)
    assert run_check(repo, "order") == 2


# --- configuration -------------------------------------------------------------------------------


@pytest.mark.parametrize("item", ["t2", "two_way"])
def test_the_pass_mark_sample_size_is_not_met_by_a_smaller_sample(repo, capsys, item):
    recorded(repo, good())
    args = [a if a != str(SAMPLE) else "100" for a in ARGS[item]]
    capsys.readouterr()
    assert check.main([*args, "--top", str(repo)]) == 2
    assert f"the run's sample has {SAMPLE} days, not 100" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("flag", "other"),
    [
        ("--energy-bounds", "empirical"),
        ("--uncertifiable-car", "day-abstains"),
        ("--kappa", "0.2"),
        ("--departure-floor", "floored"),
    ],
)
def test_t2_under_another_configuration_could_not_run(repo, capsys, flag, other):
    recorded(repo, good())
    args = list(ARGS["t2"])
    args[args.index(flag) + 1] = other
    capsys.readouterr()
    assert check.main([*args, "--top", str(repo)]) == 2
    assert "as this item asks" in capsys.readouterr().out


@pytest.mark.parametrize(("key", "value"), [("rho", 0.1), ("budget_kwh", 2.0)])
def test_a_run_at_another_rho_or_budget_could_not_run(repo, key, value):
    doc = good()
    doc["configuration"][key] = value
    recorded(repo, doc)
    for item in ARGS:
        assert run_check(repo, item) == 2


def test_t2_without_its_configuration_could_not_run(repo, capsys):
    recorded(repo, good())
    assert check.main(["--t2", "--sample-size", str(SAMPLE), "--top", str(repo)]) == 2
    assert "this item needs --energy-bounds" in capsys.readouterr().out


# --- which run, and whether it verifies ----------------------------------------------------------


def test_no_run_could_not_run(repo, capsys):
    for item in ARGS:
        assert run_check(repo, item) == 2
    assert "no run named" in capsys.readouterr().out


def test_the_newest_run_is_read_not_an_older_passing_one(repo):
    recorded(repo, good(), stamp="20260101T000000Z")
    recorded(repo, fail(good(), "order"), exit_code=1, stamp="20260102T000000Z")
    assert run_check(repo, "order") == 1


def test_a_newer_run_that_timed_out_is_not_skipped(repo):
    recorded(repo, good(), stamp="20260101T000000Z")
    recorded(repo, good(), time_out=True, stamp="20260102T000000Z")
    assert run_check(repo, "order") == 2


def test_a_run_at_an_older_commit_is_stale(repo, capsys):
    recorded(repo, good())
    new_commit(repo)
    capsys.readouterr()
    assert run_check(repo, "order") == 2
    assert "is stale" in capsys.readouterr().out


def test_a_run_from_an_earlier_commit_with_only_docs_changed_passes_and_says_so(repo, capsys):
    run = recorded(repo, good())
    earlier = docs_commit(repo)
    capsys.readouterr()
    assert run_check(repo, "order") == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(
        f"item: {check.WORDS['order']}; run {run.name}{earlier_note(earlier)}"
    )


def _order_could_not_run() -> dict:
    doc = good()
    doc["checks"]["order"].update(verdict="could not run", could_not_run_reasons=["made up"])
    return doc


@pytest.mark.parametrize(
    "make, exit_code, words",
    [
        (lambda: fail(good(), "t2"), 1, "ended with exit code 1, so it is not verified"),
        (_order_could_not_run, 2, ": made up"),
    ],
    ids=["another check failed", "the item's check could not run"],
)
def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(
    repo, capsys, make, exit_code, words
):
    run = recorded(repo, make(), exit_code=exit_code)
    earlier = docs_commit(repo)
    capsys.readouterr()
    assert run_check(repo, "order") == 2
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("COULD NOT RUN: ")
    assert f"{run.name}{earlier_note(earlier)}" in lines[-1]
    assert words in lines[-1]
    naming = [line for line in lines if run.name in line]
    assert len(naming) == 2
    assert all(f"{run.name}{earlier_note(earlier)}" in line for line in naming)


def test_a_changed_results_file_could_not_run(repo):
    folder = recorded(repo, good())
    (folder / RESULTS_NAME).write_text((folder / RESULTS_NAME).read_text() + " ")
    assert run_check(repo, "order") == 2


# --- the units the run read ----------------------------------------------------------------------

# Runs of the command with `--units`: another list (the training unit caltech and the development
# unit OSL_1, on made-up days of `ceiling_fakes`) and a subset of the default list. Each ends with
# exit 0 and passes every check of its own, so only its units keep it from the pass mark.
OTHER_LAYOUT = {"OSL_1": (("A", 25),), "caltech": (("A", 15), ("B", 15))}
UNIT_RUNS = {
    "another list": (OTHER_LAYOUT, ["caltech", "OSL_1"]),
    "a subset": (None, ["nrel-before-may-2019", "nrel-from-may-2019"]),
}


@pytest.fixture(scope="module")
def unit_documents(tmp_path_factory):
    found = {}
    for name, (layout, units) in UNIT_RUNS.items():
        with pytest.MonkeyPatch.context() as patch:
            root = tmp_path_factory.mktemp("units") / "build"
            code, out = run_command(root, patch, layout=layout, units=units)
        assert code == 0, name
        found[name] = json.loads((out / RESULTS_NAME).read_text())
    return found


@pytest.mark.parametrize("item", list(ARGS))
@pytest.mark.parametrize("case", list(UNIT_RUNS))
def test_a_run_on_other_units_could_not_run(repo, capsys, unit_documents, case, item):
    recorded(repo, unit_documents[case])
    capsys.readouterr()
    assert run_check(repo, item) == 2
    named = ", ".join(sorted(UNIT_RUNS[case][1]))
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: the run read the units {named}, not the units of this pass mark "
        f"({', '.join(UNITS)}: dev_day_checks.UNITS)\n"
    )


def test_a_run_whose_results_do_not_name_its_units_could_not_run(repo, capsys):
    doc = good()
    del doc["sample"]["counting_days"]
    recorded(repo, doc)
    capsys.readouterr()
    assert run_check(repo, "order") == 2
    assert "does not say which units the run read" in capsys.readouterr().out

"""`tools/flexitroid_check.py` on made-up recorded runs (`recorded_fakes`): pass, differs and
could not run (no run, stale, not verified, timed out, fewer than 50 days, a day that could not
run, a file that does not add up, a day of another unit, days of a part of the units only), and
which run is read."""

import importlib
import sys
from pathlib import Path

import pytest
from recorded_fakes import (
    clean_git_environment,
    docs_commit,
    earlier_note,
    make_repo,
    make_run,
    new_commit,
)

from fleetreserve.ceiling_report import UNITS
from fleetreserve.flexitroid_days import RESULTS_NAME, counts
from fleetreserve.flexitroid_match import COULD_NOT_RUN, DIFFERS, MATCH

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
check = importlib.import_module("flexitroid_check")

LABEL = "wp4-flexitroid"


def day(k: int, verdict: str = MATCH) -> dict:
    return {
        "unit": UNITS[k % len(UNITS)],
        "site": "A",
        "date": f"2020-01-{k + 1:02d}" if k < 28 else f"2020-02-{k - 27:02d}",
        "verdict": verdict,
        "functions_only_verdict": MATCH if verdict != COULD_NOT_RUN else COULD_NOT_RUN,
        "problems": ["made-up difference"] if verdict == DIFFERS else [],
        "could_not_run": ["made-up timeout"] if verdict == COULD_NOT_RUN else [],
    }


def document(n_days: int = 50, special: dict[int, str] | None = None) -> dict:
    days = [day(k, (special or {}).get(k, MATCH)) for k in range(n_days)]
    return {
        "sample": {"size": 50, "seed": 2026},
        "flexitroid": {"version": "0.1.0", "commit": "0" * 40},
        "days": days,
        "counts": counts(days),
    }


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    clean_git_environment(monkeypatch)


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def run_check(repo: Path, *extra: str) -> int:
    return check.main(["--top", str(repo), *extra])


def recorded(repo, doc, **options):
    return make_run(repo, LABEL, RESULTS_NAME, doc, **options)


def test_all_50_days_matching_in_a_verified_run_passes(repo, capsys):
    recorded(repo, document())
    assert run_check(repo) == 0
    assert "all 50 development days match" in capsys.readouterr().out


def test_a_day_that_differs_fails(repo, capsys):
    recorded(repo, document(special={7: DIFFERS}), exit_code=1)
    assert run_check(repo) == 1
    out = capsys.readouterr().out
    assert "FAILED: 1 of 50 days differ" in out
    assert "made-up difference" in out


def test_a_day_that_differs_fails_even_beside_days_that_could_not_run(repo):
    recorded(repo, document(n_days=30, special={3: COULD_NOT_RUN, 4: DIFFERS}), exit_code=1)
    assert run_check(repo) == 1


def test_a_day_that_could_not_run_is_never_a_match(repo, capsys):
    recorded(repo, document(special={12: COULD_NOT_RUN}), exit_code=2)
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert "COULD NOT RUN: 1 of 50 days could not run" in out
    assert "made-up timeout" in out


def test_fewer_than_50_days_could_not_run(repo, capsys):
    recorded(repo, document(n_days=49))
    assert run_check(repo) == 2
    assert "covers 49 days, not the 50" in capsys.readouterr().out


def test_no_run_could_not_run(repo, capsys):
    assert run_check(repo) == 2
    assert "no run named <UTC start>-wp4-flexitroid" in capsys.readouterr().out


def test_a_stale_run_could_not_run(repo, capsys):
    recorded(repo, document())
    new_commit(repo)
    assert run_check(repo) == 2
    assert "is stale" in capsys.readouterr().out


def test_a_run_from_an_earlier_commit_with_only_docs_changed_passes_and_says_so(repo, capsys):
    run = recorded(repo, document())
    earlier = docs_commit(repo)
    capsys.readouterr()
    assert run_check(repo) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"item: {check.ITEM}; run {run.name}{earlier_note(earlier)}")


def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(repo, capsys):
    run = recorded(repo, document(), exit_code=2)
    earlier = docs_commit(repo)
    capsys.readouterr()
    assert run_check(repo) == 2
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith(f"COULD NOT RUN: {run.name}{earlier_note(earlier)}")
    assert lines[-1].endswith("not verified and cannot pass the item")
    naming = [line for line in lines if run.name in line]
    assert len(naming) == 2
    assert all(f"{run.name}{earlier_note(earlier)}" in line for line in naming)


def test_a_run_that_did_not_end_with_exit_0_cannot_pass(repo, capsys):
    recorded(repo, document(), exit_code=2)
    assert run_check(repo) == 2
    assert "not verified and cannot pass" in capsys.readouterr().out


def test_a_timed_out_run_could_not_run(repo, capsys):
    recorded(repo, document(), time_out=True)
    assert run_check(repo) == 2
    assert "not verified" in capsys.readouterr().out


def test_counts_that_do_not_add_up_could_not_run(repo, capsys):
    doc = document()
    doc["counts"]["strict"][MATCH] = 51
    recorded(repo, doc)
    assert run_check(repo) == 2
    assert "do not agree with the days listed" in capsys.readouterr().out


def test_a_day_listed_twice_could_not_run(repo, capsys):
    doc = document()
    doc["days"][1] = dict(doc["days"][0])
    doc["counts"] = counts(doc["days"])
    recorded(repo, doc)
    assert run_check(repo) == 2
    assert "listed twice" in capsys.readouterr().out


def test_a_day_from_another_unit_could_not_run(repo, capsys):
    doc = document()
    doc["days"][0]["unit"] = "heldout-unit"
    recorded(repo, doc)
    assert run_check(repo) == 2
    assert "is not a development unit" in capsys.readouterr().out


def test_a_run_on_a_subset_of_the_units_could_not_run(repo, capsys):
    # As `flexitroid_days --units nrel-before-may-2019 nrel-from-may-2019` would write it: the
    # block sample's draw gives every unit named at least one day, so a run on the three units
    # lists days of all three, and one on two of them lists days of those two only.
    doc = document()
    for k, entry in enumerate(doc["days"]):
        entry["unit"] = UNITS[k % 2]
    doc["counts"] = counts(doc["days"])
    recorded(repo, doc)
    capsys.readouterr()
    assert run_check(repo) == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: the run's days are of the units nrel-before-may-2019, "
        f"nrel-from-may-2019, not of every unit of this pass mark ({', '.join(UNITS)}: "
        "ceiling_report.UNITS)\n"
    )


def test_the_newest_run_is_read_and_an_older_pass_does_not_hide_it(repo):
    recorded(repo, document(), stamp="20260101T000000Z")
    recorded(repo, document(special={0: DIFFERS}), exit_code=1, stamp="20260102T000000Z")
    assert run_check(repo) == 1
    older = repo / "runs.noindex" / f"20260101T000000Z-{LABEL}"
    assert run_check(repo, "--run", str(older)) == 0


def test_on_this_checkout_without_a_run_it_could_not_run(tmp_path, capsys):
    # The repository's own folder in a fresh checkout has no recorded run of this label.
    empty = make_repo(tmp_path / "empty")
    assert check.main(["--top", str(empty)]) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN: no run named")

"""`day_intervals --apart-from <commit>` (reading I19): a run through the sealed-data gate that
`run_evidence` reads only at its own commit is read at a later commit when the option names the
run's commit, that commit is an ancestor of the checked-out one, and only
`src/fleetreserve/day_intervals.py` and paths of the keep-list differ. Made-up repositories
(`test_day_pooled`'s) with the hand run of that file recorded at the first commit.

1. Accepted with the note: a sealed run (its results hold the route's record), the later commit
   changing `src/fleetreserve/day_intervals.py` and a `tests/` path: exit 0; `day-intervals.json`
   holds `made_apart` (the run's commit, this commit, the differing paths, the sentence), the
   summary's first line says it, and standard output starts with it.
2. Without the option the old refusal stays (exit 2, "is stale").
3. Refused (exit 2, nothing written, the reason named): another path under src/ differs; the
   option names a commit that is not the run's; the run's commit is not an ancestor of the
   checked-out one; a results file's hash is off; an option that is not a full commit id.
4. A run at the checked-out commit read with the option has no `made_apart`.
"""

import json

import pytest
from test_day_pooled import (  # noqa: F401  (the fixtures of the pooled tests apply here)
    CALIBRATION,
    HAND,
    ROUTE_KEY,
    TEST_DRAWS,
    clean_git_environment,
    commit,
    git,
    inject,
    made_up_repository,
    recorded,
    results,
    sealed_record,
    verified,
)

from fleetreserve import day_intervals, learn_run

MODULE = "src/fleetreserve/day_intervals.py"
SENTENCE = (
    "made apart at a later commit, after the run's output was seen; fills no slot of plan "
    "section 11"
)


def intervals(root, folder, monkeypatch, head, *extra):
    """`day_intervals` on `folder` with `head` checked out; the exit status and the output
    folder."""
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: head)
    out = root / "runs.noindex" / "20200101T000009Z-intervals"
    out.mkdir(exist_ok=True)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return day_intervals.main(["--run", str(folder), *HAND, *extra]), out


def later(root, *, other=False):
    """A commit that changes the module under test and a tests/ path (and, with `other`, another
    path under src/)."""
    (root / "src" / "fleetreserve").mkdir(parents=True, exist_ok=True)
    (root / MODULE).write_text("changed = True\n")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "x.py").write_text("changed = True\n")
    if other:
        (root / "src" / "fleetreserve" / "other.py").write_text("changed = True\n")
    return commit(root, "later")


def setup(tmp_path, monkeypatch, label="1"):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    root, first = made_up_repository(tmp_path)
    return root, first, recorded(root, first, label, sealed=True)


def test_a_sealed_run_is_read_apart_with_the_note(tmp_path, monkeypatch, capsys):
    root, first, folder = setup(tmp_path, monkeypatch)
    second = later(root)
    status, out = intervals(root, folder, monkeypatch, second, "--apart-from", first)
    assert status == 0
    found = results(out)
    assert found["made_apart"] == {
        "run_commit": first,
        "this_commit": second,
        "differing_paths": [MODULE, "tests/x.py"],
        "note": SENTENCE,
    }
    assert found[ROUTE_KEY] == sealed_record()
    summary = (out / day_intervals.SUMMARY_NAME).read_text().splitlines()
    assert summary[0].startswith(SENTENCE.capitalize())
    assert first in summary[0] and second in summary[0] and MODULE in summary[0]
    assert capsys.readouterr().out.splitlines()[0] == summary[0]


def test_without_the_option_the_stale_refusal_stays(tmp_path, monkeypatch, capsys):
    root, first, folder = setup(tmp_path, monkeypatch)
    second = later(root)
    status, out = intervals(root, folder, monkeypatch, second)
    assert status == 2
    text = capsys.readouterr().out
    assert "is stale" in text and f"{MODULE} differs between them" in text
    assert list(out.iterdir()) == []


def refused(root, folder, monkeypatch, capsys, head, argument, *reason) -> None:
    status, out = intervals(root, folder, monkeypatch, head, "--apart-from", argument)
    text = capsys.readouterr().out
    assert status == 2 and text.startswith("COULD NOT RUN: ")
    for word in reason:
        assert word in text
    assert list(out.iterdir()) == []


def test_another_path_under_src_refuses(tmp_path, monkeypatch, capsys):
    root, first, folder = setup(tmp_path, monkeypatch)
    second = later(root, other=True)
    refused(root, folder, monkeypatch, capsys, second, first, "src/fleetreserve/other.py differs")


def test_a_commit_that_is_not_the_runs_refuses(tmp_path, monkeypatch, capsys):
    root, first, folder = setup(tmp_path, monkeypatch)
    (root / "docs" / "notes.md").write_text("Changed notes.\n")
    middle = commit(root, "a change on the keep-list")
    second = later(root)
    refused(
        root,
        folder,
        monkeypatch,
        capsys,
        second,
        middle,
        "is not the commit the run was recorded at",
    )


def test_a_run_commit_that_is_not_an_ancestor_refuses(tmp_path, monkeypatch, capsys):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    root, first = made_up_repository(tmp_path)
    git(root, "checkout", "-q", "-b", "side")
    (root / "docs" / "side.md").write_text("Side.\n")
    side = commit(root, "on a side branch")
    folder = recorded(root, side, "1", sealed=True)
    git(root, "checkout", "-q", "main")
    second = later(root)
    refused(root, folder, monkeypatch, capsys, second, side, "is not an ancestor")


def test_a_results_file_whose_hash_is_off_refuses(tmp_path, monkeypatch, capsys):
    root, first, folder = setup(tmp_path, monkeypatch)
    second = later(root)
    file = folder / learn_run.GATE_NAME
    document = json.loads(file.read_text())
    document["label"] = "changed after the run"
    file.write_text(json.dumps(document))
    refused(
        root, folder, monkeypatch, capsys, second, first, "--apart-from", "changed since the run"
    )


@pytest.mark.parametrize("argument", ["abc", "A" * 40, "0" * 39])
def test_an_option_that_is_not_a_full_commit_id_refuses(tmp_path, monkeypatch, capsys, argument):
    root, first, folder = setup(tmp_path, monkeypatch)
    second = later(root)
    refused(root, folder, monkeypatch, capsys, second, argument, "full commit id")


def test_a_run_at_the_checked_out_commit_has_no_note(tmp_path, monkeypatch):
    root, first, folder = setup(tmp_path, monkeypatch)
    status, out = intervals(root, folder, monkeypatch, first, "--apart-from", first)
    assert status == 0
    assert "made_apart" not in results(out)
    assert not (out / day_intervals.SUMMARY_NAME).read_text().startswith("Made apart")

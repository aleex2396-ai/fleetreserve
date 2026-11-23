"""What `run_evidence` reads of a held-out truth build (`heldout_truth_build.main`), on a made-up
sealed repository.

The build opens the three sealed sets through the gated routes, so its summary holds the route's
record, the key `sealed_route` (`gate_summary.ROUTE_KEY`; reading T4 there), and `read_evidence`
never reads the build from an earlier commit (its point 4, reading E3). A development build's
summary (`truth_build`) has no such key.

1. The summary the command writes holds the key with the record `ROUTE_RECORD`.
2. That build, recorded at a commit of the made-up repository, is read at that commit and is
   stale from a later one whose only change is under `docs/` (a change `read_evidence` accepts for
   a run that did not go through the gate); the same summary without the key is read from the
   later commit, so the key alone decides.
"""

import json
import shutil
from pathlib import Path

import pytest
from fake_heldout import git, record_run, sealed_repo
from heldout_truth_fake import sealed_files

import fleetreserve
from fleetreserve import heldout_truth_build as h
from fleetreserve import run_evidence, truth_build
from fleetreserve.gate_summary import ROUTE_KEY

MATRIX = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
COMMAND = ["uv", "run", "--frozen", "python", "-m", "fleetreserve.heldout_truth_build"]
SUMMARY = truth_build.SUMMARY_NAME


@pytest.fixture
def built(tmp_path, monkeypatch):
    """(the made-up repository, the run folder of the real command's outputs)."""
    repo = sealed_repo(tmp_path, sealed_files(tmp_path / "made"))
    out = tmp_path / "run"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert h.main(["--matrix", str(MATRIX)], root=repo) == 0
    return repo, out


def _later(repo: Path) -> str:
    """A commit that changes `docs/` only (on the keep-list of `run_evidence`)."""
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "notes.md").write_text("changed\n")
    git(repo, "add", "--", "docs/notes.md")
    git(repo, "commit", "-q", "--no-verify", "-m", "a change under docs")
    return git(repo, "rev-parse", "HEAD")


def test_the_summary_holds_the_routes_key(built):
    _, out = built
    document = json.loads((out / SUMMARY).read_text())
    assert document[ROUTE_KEY] == h.ROUTE_RECORD
    assert document[ROUTE_KEY]["kind"] == "held-out"


def test_a_held_out_truth_build_from_an_earlier_commit_is_stale(built):
    repo, out = built
    run = record_run(repo, "wp2-heldout-truth", out, COMMAND)
    at = git(repo, "rev-parse", "HEAD")
    assert run_evidence.read_evidence(run, SUMMARY, at).origin == run_evidence.AT_CHECKED_OUT
    later = _later(repo)
    with pytest.raises(run_evidence.CannotCheck) as raised:
        run_evidence.read_evidence(run, SUMMARY, later)
    assert f"its results hold the route's record {ROUTE_KEY}" in str(raised.value)
    assert "(reading E3)" in str(raised.value)


def test_the_same_summary_without_the_key_is_read_from_the_later_commit(built, tmp_path):
    repo, out = built
    document = json.loads((out / SUMMARY).read_text())
    del document[ROUTE_KEY]
    plain = tmp_path / "plain"
    shutil.copytree(out, plain)
    (plain / SUMMARY).write_text(json.dumps(document))
    run = record_run(repo, "wp2-heldout-truth", plain, COMMAND)
    later = _later(repo)
    assert run_evidence.read_evidence(run, SUMMARY, later).origin == run_evidence.FROM_EARLIER

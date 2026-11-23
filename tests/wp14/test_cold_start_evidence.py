"""What `run_evidence` reads of a cold start (`cold_start.main`) on held-out units, in a made-up
repository.

A held-out cold start passes the sealed-data gate, so its results hold the route's record, the key
`sealed_route` (`gate_summary.ROUTE_KEY`), beside `gate`, and `read_evidence` never reads the run
from an earlier commit (its point 4, reading E3). A development run has neither key.

1. The results of a held-out run written by the command hold the key, with the kind, the label and
   the tag's commit and fix commit that `gate` holds; a development run holds neither key.
2. That run, recorded at a commit of a made-up repository, is read at that commit and is stale
   from a later one whose only change is under `docs/` (a change `read_evidence` accepts for a
   run that did not go through the gate); the same results without the key are read from the
   later commit, so the key alone decides.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

# `fake_heldout` (the made-up repository and the recorder) lives in a folder the conftest does
# not add to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp2"))

from cold_start_fake import FIT_UNIT, cold_cars, register, write_build  # noqa: E402
from fake_heldout import git, record_run, sealed_repo  # noqa: E402
from gate_fake import COMMIT, matrix_document  # noqa: E402

from fleetreserve import (  # noqa: E402
    cold_start,
    cold_start_gate,
    gate_run,
    run_evidence,
    run_verify,
    sealed_route,
)
from fleetreserve.gate_summary import ROUTE_KEY  # noqa: E402

HELD_OUT = "OSL_T"
DEVELOPMENT = "nrel-from-may-2019"
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
MODELS = ("cold_one", "cold_two")
TAG, FIX = "f" * 40, "e" * 40
COMMAND = ["uv", "run", "--frozen", "python", "-m", "fleetreserve.cold_start"]


def _run_command(folder: Path, monkeypatch, *, found) -> Path:
    """The real command on made-up units in `folder`; the folder its outputs are written to. With
    `found` (what the gate returns, patched) the scored unit is a held-out one; without, the gate
    must not be called and the scored unit is a development one."""
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    register(monkeypatch, MODELS[0])
    register(monkeypatch, MODELS[1], shift=0.5)

    def gate(root=None):
        if found is None:
            pytest.fail("the gate is called for a development unit")
        return found

    monkeypatch.setattr(cold_start_gate, "require_gate", gate)
    (folder / "dev").mkdir()
    units = {DEVELOPMENT: cold_cars(62, skip=(20,), thin=(30,)), FIT_UNIT: cold_cars(5, "T")}
    run, tables = write_build(folder / "dev", units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    out = folder / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = [
        *("--model", MODELS[0], "--reference", MODELS[1], "--dial", "0.02"),
        *("--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)),
    ]
    if found is None:
        arguments += ["--units", DEVELOPMENT]
    else:
        (folder / "held").mkdir()
        held_run, held_tables = write_build(folder / "held", {HELD_OUT: cold_cars(58)})
        arguments += [
            *("--units", HELD_OUT),
            *("--held-out-tables", str(held_tables), "--held-out-truth-run", str(held_run)),
        ]
    assert cold_start.main(arguments) == 0
    return out


@pytest.fixture
def held_out(tmp_path, monkeypatch):
    return _run_command(tmp_path, monkeypatch, found=sealed_route.GateCommit(TAG, FIX))


def _results(folder: Path) -> dict:
    return json.loads((folder / cold_start.RESULTS_NAME).read_text())


def test_a_held_out_run_holds_the_routes_key_and_a_development_run_does_not(
    held_out, tmp_path_factory, monkeypatch
):
    document = _results(held_out)
    assert document[ROUTE_KEY] == {
        "kind": cold_start_gate.HELD_OUT,
        "label": cold_start.LABELS[cold_start_gate.HELD_OUT],
        "tag_commit": TAG,
        "fix_commit": FIX,
    }
    assert document["gate"] == {"tag_commit": TAG, "fix_commit": FIX}
    developed = _results(
        _run_command(tmp_path_factory.mktemp("development"), monkeypatch, found=None)
    )
    assert ROUTE_KEY not in developed and "gate" not in developed


def _repository(tmp_path) -> Path:
    return sealed_repo(tmp_path / "made-up", {}, tag=False, package=False)


def _later(repo: Path) -> str:
    """A commit that changes `docs/` only (on the keep-list of `run_evidence`)."""
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "notes.md").write_text("changed\n")
    git(repo, "add", "--", "docs/notes.md")
    git(repo, "commit", "-q", "--no-verify", "-m", "a change under docs")
    return git(repo, "rev-parse", "HEAD")


def test_a_held_out_cold_start_from_an_earlier_commit_is_stale(held_out, tmp_path):
    repo = _repository(tmp_path)
    run = record_run(repo, "cold-start", held_out, COMMAND)
    at = git(repo, "rev-parse", "HEAD")
    assert run_evidence.read_evidence(run, cold_start.RESULTS_NAME, at).origin == (
        run_evidence.AT_CHECKED_OUT
    )
    later = _later(repo)
    with pytest.raises(run_evidence.CannotCheck) as raised:
        run_evidence.read_evidence(run, cold_start.RESULTS_NAME, later)
    assert f"its results hold the route's record {ROUTE_KEY}" in str(raised.value)
    assert "(reading E3)" in str(raised.value)


def test_the_same_results_without_the_key_are_read_from_the_later_commit(held_out, tmp_path):
    results = _results(held_out)
    del results[ROUTE_KEY]
    plain = tmp_path / "plain"
    shutil.copytree(held_out, plain)
    (plain / cold_start.RESULTS_NAME).write_text(json.dumps(results))
    repo = _repository(tmp_path)
    run = record_run(repo, "cold-start", plain, COMMAND)
    later = _later(repo)
    evidence = run_evidence.read_evidence(run, cold_start.RESULTS_NAME, later)
    assert evidence.origin == run_evidence.FROM_EARLIER

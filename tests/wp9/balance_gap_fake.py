"""Made-up recorded runs for the tests of the balance-gap measurement (`balance_gap_run`).

`build(folder)`: the made-up build of `learn_fake.small_gate_units` (5, 24 and 24 days of
`gate_fake.gate_cars`, one service hour; the first unit not calibratable) and the learn matrix of
`learn_fake.small_matrix` (20 calibration days; the empirical configuration at kappa 0.1 and its
oracle partner; tiers A and B), written as `matrix.json` in `folder`. `gate_run` and
`learn_run --model empirical` are run on it, each into a folder of its own, and recorded
(`record`). Returns the paths by name: `folder` (the top folder the recorded commands name their
matrix from), `truth`, `tables`, `matrix`, `gate`, `learn`.

`record(folder, source, module, label, ...)`: a recorded run folder `<UTC start>-<label>` holding
the files of `source` and a manifest listing their SHA-256, as `run_record` writes it, with the
command `uv run --frozen python -m <module> --matrix matrix.json <extra>` at `gate_fake.COMMIT`
(the commit checks are replaced by the caller). `exit_code` and `commit` make a run that does not
verify or is stale. `rehash(run)` lists the run's files again with the SHA-256 they have now.

`read_rows(path)`: the rows of a CSV table, a leading label line (`# ...`) skipped.
"""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from gate_fake import COMMIT, write_build
from learn_fake import small_gate_units, small_matrix

from fleetreserve import gate_run, learn_run, run_verify

GATE_MODULE = "fleetreserve.gate_run"
LEARN_MODULE = "fleetreserve.learn_run"


def verified(monkeypatch) -> None:
    """The commit checks replaced: every recorded commit exists and `COMMIT` is checked out."""
    from fleetreserve import balance_gap_run

    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setattr(balance_gap_run, "_head_commit", lambda: COMMIT)


def record(
    folder: Path,
    source: Path,
    module: str,
    label: str,
    *,
    extra=(),
    exit_code: int = 0,
    commit: str = COMMIT,
) -> Path:
    """A recorded run of `module` holding the files of `source` (module text)."""
    run = folder / f"20260102T000000Z-{label}"
    shutil.copytree(source, run)
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.iterdir())}
    command = ["uv", "run", "--frozen", "python", "-m", module, "--matrix", "matrix.json", *extra]
    manifest = {
        "commit": commit,
        "command": command,
        "start": "2026-01-02T00:00:00Z",
        "end": "2026-01-02T00:01:00Z",
        "exit_code": exit_code,
        "timed_out": False,
        "outputs": outputs,
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


def rehash(run: Path) -> None:
    """List the run's files again in its manifest with the SHA-256 they have now."""
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["outputs"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run.iterdir())
        if p.name != "manifest.json"
    }
    (run / "manifest.json").write_text(json.dumps(manifest))


def build(folder: Path) -> dict:
    """The made-up build, matrix and the two recorded runs (module text)."""
    truth, tables = write_build(folder, small_gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    paths = {"folder": folder, "truth": truth, "tables": tables, "matrix": matrix}
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(truth)]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        for name, command, extra in (
            ("gate", gate_run.main, []),
            ("learn", learn_run.main, ["--model", "empirical"]),
        ):
            source = folder / f"{name}-source"
            source.mkdir()
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(source))
            assert command([*arguments, *extra]) == 0
    paths["gate"] = record(folder, folder / "gate-source", GATE_MODULE, "gate")
    paths["learn"] = record(
        folder, folder / "learn-source", LEARN_MODULE, "learn", extra=["--model", "empirical"]
    )
    return paths


def read_rows(path: Path) -> list[dict]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))

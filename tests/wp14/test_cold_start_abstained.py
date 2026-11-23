"""The cold start (`cold_start`) on a held-out unit the held-out truth build abstained.

As the held-out test of `test_cold_start_command` (the gate patched to pass, the made-up held-out
build of `cold_start_fake.write_build`, whose summary here also records `korea-visits` as
abstained with a reason and no tables): the run on OSL_T and korea-visits goes on with OSL_T,
exits 0, prints one line
`korea-visits: abstained in the held-out truth build <run>: <reason>; not scored, counted as
not passing` and writes the abstention into `held_out_inputs`; a unit the summary neither
lists nor records as abstained still could not run (exit 2); a run on abstained units alone has
no unit left (exit 2).
"""

import hashlib
import json
from pathlib import Path

import test_cold_start_command as cold
from cold_start_fake import cold_cars, write_build
from test_cold_start_command import _arguments, _document

from fleetreserve import cold_start, cold_start_gate

build = cold.build  # the fixtures of the cold start's command tests, by their names
watched = cold.watched
ABSTAINED = "korea-visits"
REASON = "no calibration day at the start draw"


def _held_build(folder: Path, abstained: dict | None) -> tuple[Path, Path]:
    run, tables = write_build(folder, {"OSL_T": cold_cars(58)})
    if abstained is not None:
        summary = run / "truth-summary.json"
        document = json.loads(summary.read_text())
        document["abstained"] = abstained
        summary.write_text(json.dumps(document))
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["outputs"] = {summary.name: hashlib.sha256(summary.read_bytes()).hexdigest()}
        (run / "manifest.json").write_text(json.dumps(manifest))
    return run, tables


def _held(run: Path, tables: Path) -> list[str]:
    return ["--held-out-tables", str(tables), "--held-out-truth-run", str(run)]


def _gate(monkeypatch) -> None:
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: "f" * 40)


def test_the_unit_with_tables_is_scored_and_the_abstained_unit_is_recorded(
    build, tmp_path, watched, monkeypatch, capsys
):
    _gate(monkeypatch)
    run, tables = _held_build(tmp_path, {ABSTAINED: {"reason": REASON, "sessions_loaded": 3}})
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = [*_arguments(build, "OSL_T", ABSTAINED), *_held(run, tables)]
    assert cold_start.main(arguments) == 0
    line = (
        f"{ABSTAINED}: abstained in the held-out truth build {run.name}: {REASON}; "
        "not scored, counted as not passing"
    )
    assert line in capsys.readouterr().out.splitlines()
    assert line in (out / cold_start.SUMMARY_NAME).read_text().splitlines()
    document = _document(out)
    assert list(document["units"]) == ["OSL_T"]
    assert document["held_out_inputs"]["abstained"] == {ABSTAINED: REASON}
    assert set(document["held_out_inputs"]["table_sha256"]) == {"OSL_T-cars.csv", "OSL_T-days.csv"}


def test_a_unit_the_build_does_not_record_as_abstained_still_could_not_run(
    build, tmp_path, watched, monkeypatch, capsys
):
    _gate(monkeypatch)
    run, tables = _held_build(tmp_path, {})
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = [*_arguments(build, "OSL_T", ABSTAINED), *_held(run, tables)]
    assert cold_start.main(arguments) == 2
    assert f"COULD NOT RUN: {ABSTAINED} is not in the recorded truth build" in (
        capsys.readouterr().out
    )
    assert not any(out.iterdir())


def test_a_run_on_abstained_units_alone_has_no_unit_left(
    build, tmp_path, watched, monkeypatch, capsys
):
    _gate(monkeypatch)
    run, tables = _held_build(tmp_path, {ABSTAINED: {"reason": REASON}})
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    assert cold_start.main([*_arguments(build, ABSTAINED), *_held(run, tables)]) == 2
    shown = capsys.readouterr().out
    assert shown.startswith("COULD NOT RUN: no unit is left to score")
    assert not any(out.iterdir())

"""A configuration that raises an error on one unit, in `learn_run` off the sealed route (the
development and rehearsal routes), is recorded and ends the step with exit 1, as before the
sealed route's own reading (`test_heldout_learn_failed`): the unit holds the reason
`configuration failed` and the error in `failed`, and the offer shows the error alone.
"""

import json

from gate_fake import COMMIT, GATE_UNITS, write_build
from learn_fake import PAIR, small_gate_units, small_matrix

from fleetreserve import gate_run, learn_run, run_verify

ERROR = "RuntimeError: made to fail"


def test_a_failed_configuration_on_a_development_unit_still_exits_1(tmp_path, monkeypatch, capsys):
    truth, tables = write_build(tmp_path, small_gate_units(), service_hours=2)
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    real = gate_run._prepare_task
    unit, name = GATE_UNITS[1], PAIR[0]

    def task(job):
        if job[0] == unit and job[2].name == name:
            raise RuntimeError("made to fail")
        return real(job)

    monkeypatch.setattr(gate_run, "_prepare_task", task)
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(truth)]
    status = learn_run.main([*arguments, "--model", "empirical"])
    shown = capsys.readouterr().out
    assert status == 1, shown
    results = json.loads((out / "learn.json").read_text())
    gate = json.loads((out / "learn-gate.json").read_text())
    assert results["failed_configurations"] == [f"{name} | {unit}: {ERROR}"]
    assert results["units"][unit]["configurations"][name]["rungs"] == {"reason": ERROR}
    entry = gate["configurations"][name]["units"][unit]
    assert entry == {"failed": ERROR, "reason": "configuration failed"}
    assert f"FAILED CONFIGURATIONS: {name} | {unit}: {ERROR}" in shown

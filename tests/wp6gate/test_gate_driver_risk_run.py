"""The gate run with driver-risk rows, on the made-up gate units of `gate_fake`.

A configuration pair with the dial 0.02 (main family at kappa 0.3, departures seen from the
rounded-down hour, one alternate of the committed matrix) runs beside the same pair without rows.
The command ends with exit 0; the results document, the pair table and the summary carry the dial;
the configurations without rows are computed apart from those with it (different work keys); and
the made-up days certify a value with the rows that is at most the value without them.
"""

import csv
import json

import pytest
from gate_fake import gate_units, matrix_document, write_build

from fleetreserve import gate_run, run_verify

PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
RISKY = ("main-rounded-hour-k0.3-eps0.02-empirical", "main-rounded-hour-k0.3-eps0.02-oracle")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    folder = tmp_path_factory.mktemp("risk")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(PLAIN + RISKY)))
    out = folder / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: "a" * 40)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
        code = gate_run.main([*arguments, "--workers", "1"])
    return code, json.loads((out / gate_run.RESULTS_NAME).read_text()), out


def test_the_run_ends_and_the_results_carry_the_dial(result):
    code, document, _ = result
    assert code == 0 and document["failed_configurations"] == []
    for name in PLAIN:
        assert document["configurations"][name]["configuration"]["driver_risk"] is None
    for name in RISKY:
        assert document["configurations"][name]["configuration"]["driver_risk"] == 0.02
        assert document["configurations"][name]["work_shared_with"] == []
    pairs = document["gate"]["by_pair"]
    assert pairs[PLAIN[0]]["driver_risk"] is None and pairs[RISKY[0]]["driver_risk"] == 0.02


def test_the_summary_names_the_dial_of_the_alternate(result):
    _, _, out = result
    text = (out / gate_run.SUMMARY_NAME).read_text()
    assert PLAIN[0] not in text.split("Alternates")[-1]
    assert "driver-risk rows at epsilon 0.02" in text.split("Alternates")[-1]


def _values(out, name):
    lines = [
        line
        for line in (out / gate_run.DAYS_NAME).read_text().splitlines()
        if not line.startswith("#")
    ]
    return {
        (r["unit"], r["site"], r["date"]): float(r["certified_value_kwh"])
        for r in csv.DictReader(lines)
        if r["configuration"] == name
    }


def test_the_rows_never_raise_a_certified_value_and_leave_the_oracle_partner_unchanged(result):
    _, _, out = result
    plain, risky = _values(out, PLAIN[0]), _values(out, RISKY[0])
    assert plain.keys() == risky.keys() and len(plain) > 100
    assert all(risky[k] <= plain[k] + 1e-7 for k in plain)
    oracle_plain, oracle_risky = _values(out, PLAIN[1]), _values(out, RISKY[1])
    assert oracle_plain.keys() == oracle_risky.keys() and len(oracle_plain) > 100
    assert all(oracle_risky[k] == pytest.approx(oracle_plain[k], abs=1e-7) for k in oracle_plain)

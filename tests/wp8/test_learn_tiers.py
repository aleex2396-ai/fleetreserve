"""The two tiers of plan section 6 in the WP8 evaluation (`learn_run`, `learn_eval`).

1. `learn.json` carries, for each offer rung of a configuration and of its oracle partner, tier A
   as before and tier B under `tier_b` (scale, reason, test-day scores, stay-to-bound check), equal
   to the gate-format results the offers are read from.
2. The learn summary gives each offer under both labels, on separate lines, and no line carries
   both labels.
3. A results document without tier B (a matrix listing only tier A) has no `tier_b` key and no
   tier B line.
"""

import json

import pytest
from gate_fake import COMMIT, write_build
from learn_fake import PAIR, small_gate_units, small_matrix

from fleetreserve import gate_run, learn_eval, learn_run, run_verify


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def _build(folder, tiers):
    run, tables = write_build(folder, small_gate_units())
    document = small_matrix()
    document["report"]["tiers"] = tiers
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, out):
    run, tables, matrix = build
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = learn_run.main(
            ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
            + ["--model", "empirical"]
        )
    return code, json.loads((out / learn_run.RESULTS_NAME).read_text()), out


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    folder = tmp_path_factory.mktemp("both")
    return _run(_build(folder, ["A", "B"]), folder / "out")


@pytest.fixture(scope="module")
def only_a(tmp_path_factory):
    folder = tmp_path_factory.mktemp("only_a")
    return _run(_build(folder, ["A"]), folder / "out")


def test_learn_json_has_tier_b_equal_to_the_gate_results(both):
    code, results, out = both
    assert code == 0
    gate = json.loads((out / learn_run.GATE_NAME).read_text())
    found = 0
    for unit in results["units"]:
        entry = results["units"][unit]["configurations"].get(PAIR[0])
        if entry is None or "reason" in entry["rungs"]:
            continue
        sources = (
            (entry["rungs"], gate["configurations"][PAIR[0]]["units"][unit]),
            (entry["oracle_partner"]["rungs"], gate["configurations"][PAIR[1]]["units"][unit]),
        )
        for rungs, source in sources:
            for rung in learn_eval.OFFER_RUNGS:
                tier_b = source["rungs"][rung]["tier_b"]
                assert rungs[rung]["tier_b"] == {
                    "beta_hat": tier_b["beta_hat"],
                    "reason": tier_b["reason"],
                    "test": tier_b["test"],
                    "stay_to_bound": tier_b["stay_to_bound"],
                }
                assert rungs[rung]["test"] == source["rungs"][rung]["test"]
                found += 1
    assert found > 0
    assert "gate_tier_b" in gate


def test_the_learn_summary_gives_each_tier_on_its_own_line(both):
    _, results, _ = both
    lines = learn_run.summary_lines(results)
    a = [x for x in lines if "tier A (driver-protected)" in x]
    b = [x for x in lines if "tier B (deadline contract)" in x]
    assert a and b  # a unit that is not calibratable has a tier A line (its reason) only
    assert not [x for x in lines if "tier A" in x and "tier B" in x]
    for k, line in enumerate(lines):
        if "tier B (deadline contract)" in line:
            assert "tier A (driver-protected)" in lines[k - 1]
    assert all("cars staying to their bound" in x for x in b)
    assert not any("cars staying to their bound" in x for x in a)


def test_without_tier_b_there_is_no_tier_b_anywhere(only_a):
    code, results, out = only_a
    assert code == 0
    assert "tier_b" not in json.dumps(results)
    gate = json.loads((out / learn_run.GATE_NAME).read_text())
    assert "gate_tier_b" not in gate
    text = "\n".join(learn_run.summary_lines(results))
    assert "tier A (driver-protected)" in text and "tier B" not in text

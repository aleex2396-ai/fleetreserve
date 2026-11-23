"""The gate summary and tables under the two tiers (`gate_summary`, `gate_run`; plan section 6).

1. On a made-up results document where the tiers disagree (tier A "no go", tier B "go"), the
   summary prints the offer rows and the gate outcomes of both tiers, each line under its own
   label, and no row or outcome line carries both labels (the two sentences that define the tiers
   do). Each tier has its own agreement line, which counts that
   tier's decisions only. Nothing is added or averaged across tiers: the shares of the two tiers
   appear each on its own line and no number made from both appears.
2. The summary says that the plan reads the gate at tier A, with tier B beside it, and prints no
   single verdict across tiers.
3. A document without tier B results (a matrix listing only tier A, or an older document) prints
   tier A alone, and tier B's gate says it has no result.
4. The gate tables of a real run of made-up units where the tiers differ (every fourth day a car
   leaves after the service hour and before its forecast departure): tier B offers where tier A
   abstains; `gate_tier_b` holds tier B's decisions, `gate` tier A's, and the tier B decisions are
   computed from tier B's scale and test days.
"""

import json

import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    ROUNDED,
    TOP,
    gate_cars,
    matrix_document,
    write_build,
)

from fleetreserve import gate_config, gate_run, gate_summary, run_verify

MATRIX = gate_config.load(TOP / "gate" / "matrix.json")
UNITS = MATRIX.gate_units
NAME = next(
    c.name for c in MATRIX.configurations if c.family == "main" and c.departure == "empirical"
)
CONFIGURATION = MATRIX.configuration(NAME)
A, B = gate_config.TIER_LABELS["A"], gate_config.TIER_LABELS["B"]


def _test(offer: float, share: float, days: int = 10, failing: int | None = 0) -> dict:
    return {
        "days": days,
        "offering_days": days,
        "failing_days": failing,
        "mean_offer_kwh_per_day": offer,
        "share_of_ceiling": {
            "hindsight-free-decision-hour": {"ratio_of_sums": share},
            "best-causal": {"ratio_of_sums": share},
        },
    }


def _rung(scale: float, offer: float, share: float, stay=None) -> dict:
    found = {"beta_hat": scale, "reason": "", "test": _test(offer, share)}
    if stay:
        found["stay_to_bound"] = stay
    return found


STAY = {"cars_checked": 123, "cars_short": 4, "largest_shortfall_kwh": 0.5}


def _row(tier_a: dict, tier_b: dict | None) -> dict:
    rung = dict(tier_a)
    if tier_b is not None:
        rung["tier_b"] = tier_b
    return {"rungs": {MATRIX.gate_rung: rung}}


def _data(share: float, days: int = 4) -> dict:
    return {
        "values": [share * 10.0] * days,
        "ceilings": {c: [10.0] * days for c in gate_config.CEILINGS},
        "beta": 1.0,
    }


def _tables(cal_share: float, oracle_share: float) -> dict:
    """The gate tables of a made-up document in which every unit has the given shares."""
    test_data = {}
    for unit in UNITS:
        test_data[(NAME, unit)] = _data(cal_share)
        test_data[(CONFIGURATION.oracle_partner, unit)] = _data(oracle_share)
    return gate_summary.gate_tables(MATRIX, test_data)


def _document(tier_b=True) -> dict:
    units = {
        unit: _row(
            _rung(0.0, 0.0, 0.0),
            _rung(1.0, 4.4, 0.44, STAY) if tier_b else None,
        )
        for unit in UNITS
    }
    partner = {unit: _row(_rung(1.0, 9.9, 0.99), _rung(1.0, 9.9, 0.99, STAY)) for unit in UNITS}
    document = {
        "trial": False,
        "partial": False,
        "t5": {"breaks_a": 0, "breaks_b": 0, "day_paths_checked": 7},
        "timeouts": {},
        "failed_configurations": [],
        "calibration_rule": {"n_calibration": 100, "calibration_half_below": 200},
        "units": {},
        "configurations": {
            NAME: {"units": units},
            CONFIGURATION.oracle_partner: {"units": partner},
        },
        "gate": _tables(0.0, 0.99),  # tier A offers nothing: no go
    }
    if tier_b:
        document["gate_tier_b"] = _tables(0.44, 0.99)  # tier B offers 44%: go on the share rule
    return document


@pytest.fixture(scope="module")
def lines():
    return gate_summary.summary_lines(_document(), MATRIX)


def test_the_tiers_disagree_on_the_made_up_document():
    document = _document()
    go = lambda gate: {d["go"] for e in gate["by_pair"].values() for d in e["decisions"]}  # noqa: E731
    assert go(document["gate"]) == {False}
    assert True in go(document["gate_tier_b"])


def test_both_tiers_are_printed_each_line_under_its_own_label(lines):
    text = "\n".join(lines)
    assert A in text and B in text
    assert not [x for x in lines if A in x and B in x and x.startswith("  ")]
    rows = [x for x in lines if f"  {CONFIGURATION.departure_seen:<14}" in x and UNITS[1] in x]
    assert len(rows) == 2
    row_a, row_b = rows
    assert A in row_a and "scale 0 (" in row_a
    assert B in row_b and "scale 1," in row_b and "44.00%" in row_b
    assert "44.00%" not in row_a
    assert "cars staying to their bound: 123 checked, 4 short, largest shortfall 0.5 kWh" in row_b
    assert "cars staying to their bound" not in row_a
    outcomes = [x for x in lines if x.lstrip().startswith("PROVISIONAL") and "ceiling " in x]
    assert outcomes
    for x in outcomes:
        assert (A in x) != (B in x)
    go_a = [x for x in outcomes if A in x]
    go_b = [x for x in outcomes if B in x]
    assert all(": no go " in x for x in go_a) and any(": go " in x for x in go_b)


def test_each_tier_has_its_own_agreement_line_over_its_own_decisions(lines):
    marks = [k for k, x in enumerate(lines) if x.startswith("Gate outcomes")]
    assert [A in lines[k] for k in marks] == [True, False] and B in lines[marks[1]]
    blocks = [lines[k : lines.index("", k)] for k in marks]
    last_a, last_b = blocks[0][-1], blocks[1][-1]
    assert last_a.startswith("All ") and "agree: no go" in last_a and last_a.endswith(f"{A} only.")
    assert last_b.startswith("The main-family outcomes differ") and last_b.endswith(f"{B} only.")
    assert "differ" not in last_a and "agree" not in last_b


def test_the_summary_says_the_plan_reads_the_gate_at_tier_a_and_gives_no_overall_verdict(lines):
    text = "\n".join(lines)
    assert "The plan reads the gate at tier A (plan section 13, WP6); tier B, where" in text
    assert "has not been decided" not in text and "a call for the project owner" not in text
    assert "the gate is read at tier A (plan section 13, WP6; shown, never pooled: " in text
    assert "which tier decides the gate" not in text
    for forbidden in ("overall", "both tiers agree", "combined", "average of the tiers"):
        assert forbidden not in text
    assert text.count("Gate outcomes") == 2  # one block per tier, no third


def test_a_document_without_tier_b_prints_tier_a_alone_and_says_tier_b_has_no_result():
    only_a = gate_summary.summary_lines(_document(tier_b=False), MATRIX)
    text = "\n".join(only_a)
    assert f"Gate outcomes, {B}: no result in this document." in text
    assert [x for x in only_a if A in x and ceiling_line(x)]
    assert not [x for x in only_a if B in x and ceiling_line(x)]
    assert f"no {B} result" in text


def ceiling_line(line: str) -> bool:
    return "ceiling " in line and line.lstrip().startswith("PROVISIONAL")


def test_a_matrix_listing_one_tier_prints_that_tier_only():
    document = _matrix_document()
    document["report"]["tiers"] = ["B"]
    one = gate_config.parse(document)
    only_b = gate_summary.summary_lines(_document(), one)
    assert [x for x in only_b if x.startswith("Gate outcomes")] == [
        x for x in only_b if x.startswith(f"Gate outcomes, {B}")
    ]
    assert [x for x in only_b if B in x and ceiling_line(x)]
    assert not [x for x in only_b if A in x and ceiling_line(x)]


def _matrix_document() -> dict:
    return json.loads((TOP / "gate" / "matrix.json").read_text())


# --- a real run where the tiers differ -----------------------------------------------------------


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    folder = tmp_path_factory.mktemp("real")
    units = {
        GATE_UNITS[0]: gate_cars(10),
        GATE_UNITS[1]: gate_cars(60, early_every=4, early_out=12.5),
        GATE_UNITS[2]: gate_cars(30, "B", early_every=4, early_out=12.5)
        + gate_cars(30, "C", early_every=4, early_out=12.5),
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(ROUNDED)))
    out = folder / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = gate_run.main(
            ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
        )
    return code, json.loads((out / gate_run.RESULTS_NAME).read_text()), out


def test_a_real_run_gives_each_tier_its_own_tables(real):
    code, document, out = real
    assert code == 0
    unit = GATE_UNITS[1]
    found = document["configurations"][ROUNDED[0]]["units"][unit]["rungs"]["one-way"]
    assert found["beta_hat"] == 0 and found["tier_b"]["beta_hat"] > 0
    a = document["gate"]["by_pair"][ROUNDED[0]]["decisions"]
    b = document["gate_tier_b"]["by_pair"][ROUNDED[0]]["decisions"]
    assert len(a) == len(b) == 6
    for decision in a:
        assert decision["units"][unit]["scale"] == 0
    for decision in b:
        assert decision["units"][unit]["scale"] == found["tier_b"]["beta_hat"]
    assert document["definitions"]["tiers"]["deciding_tier"] == "A"
    text = (out / gate_run.SUMMARY_NAME).read_text()
    assert A in text and B in text and "The plan reads the gate at tier A" in text
    assert not [x for x in text.splitlines() if A in x and B in x and x.startswith("  ")]


def test_the_days_file_has_tier_b_columns(real):
    _, _, out = real
    header = (out / gate_run.DAYS_NAME).read_text().splitlines()[0].split(",")
    assert "one-way_tier_b_offer_kwh" in header and "one-way_offer_kwh" in header
    rows = [
        dict(zip(header, line.split(","), strict=True))
        for line in (out / gate_run.DAYS_NAME).read_text().splitlines()[1:]
    ]
    offered = [
        r
        for r in rows
        if r["unit"] == GATE_UNITS[1] and r["role"] == "test" and r["configuration"] == ROUNDED[0]
    ]
    assert offered
    assert all(float(r["one-way_offer_kwh"]) == 0 for r in offered)
    assert any(float(r["one-way_tier_b_offer_kwh"]) > 0 for r in offered)

"""The ledger command (`ledger_run`) on the hand-worked run of `ledger_fake` (its module text gives
every value by hand): one service hour, two units of two days, one test day each, the same three
cars on both days.

`nrel-from-may-2019`, where the decision hour's power matters: V_hind 3, V_caus = V_aff = V_diag
= 1, so R = 2 (above 0), A = 0, V_aff - V_diag = 0; LP (24) with the day-average base set:
alpha 1, box 1, F = 2; block V_hind 3, its R 2 and A 0, the price of the block 0 (one service
hour). `norway-development-pooled`, where it does not: V_hind = V_caus = V_aff = V_diag = 3, so
R = A = 0; alpha 3, box 3, F = 0.

Per configuration, tier and rung, with the run's V_cert and scale (`ledger_fake.HAND_CERTIFIED`,
`HAND_SCALES`), on the one test day, worked here:

- NREL, empirical, tier A: V_cert 0.8, U = 1 - 0.8 = 0.2. Fixed plan (scale 0.5): offer 0.4,
  K 0.4, and 0.4 + 2 + 0 + 0.2 + 0.4 = 3. One-way (0.75): offer 0.6, K 0.2. Two-way (1.25):
  offer 1, K = -0.25 x 0.8 = -0.2.
- NREL, empirical, tier B: scales 1, 1.5, 2: offers 0.8, 1.2, 1.6; K 0, -0.4, -0.8.
- NREL, oracle: V_cert 1, scale 1 everywhere: offer 1, U 0, K 0.
- Norway, empirical, tier A: V_cert 2.5, U 0.5; fixed plan: offer 1.25, K 1.25, and
  1.25 + 0 + 0 + 0.5 + 1.25 = 3.
- Norway, oracle: V_cert 3, U 0, offer 3 at scale 1, K 0.

Each mean is over the one test day, so it equals the day's term, and each share is the term over
3. The command ends with exit 0, and its day rows hold the same numbers.
"""

import csv
import json

import pytest
from gate_fake import COMMIT, ROUNDED
from ledger_fake import HAND_UNITS, write_hand_run

from fleetreserve import gate_run, ledger_run, run_verify

EMPIRICAL, ORACLE = ROUNDED
NREL, NORWAY = HAND_UNITS
# (configuration, unit, tier, rung) -> (offer, R, A, U, K), by hand (module text).
HAND = {
    (EMPIRICAL, NREL, "A", "fixed"): (0.4, 2.0, 0.0, 0.2, 0.4),
    (EMPIRICAL, NREL, "A", "one-way"): (0.6, 2.0, 0.0, 0.2, 0.2),
    (EMPIRICAL, NREL, "A", "two-way"): (1.0, 2.0, 0.0, 0.2, -0.2),
    (EMPIRICAL, NREL, "B", "fixed"): (0.8, 2.0, 0.0, 0.2, 0.0),
    (EMPIRICAL, NREL, "B", "one-way"): (1.2, 2.0, 0.0, 0.2, -0.4),
    (EMPIRICAL, NREL, "B", "two-way"): (1.6, 2.0, 0.0, 0.2, -0.8),
    (ORACLE, NREL, "A", "fixed"): (1.0, 2.0, 0.0, 0.0, 0.0),
    (ORACLE, NREL, "B", "two-way"): (1.0, 2.0, 0.0, 0.0, 0.0),
    (EMPIRICAL, NORWAY, "A", "fixed"): (1.25, 0.0, 0.0, 0.5, 1.25),
    (EMPIRICAL, NORWAY, "A", "two-way"): (3.125, 0.0, 0.0, 0.5, -0.625),
    (EMPIRICAL, NORWAY, "B", "one-way"): (3.75, 0.0, 0.0, 0.5, -1.25),
    (ORACLE, NORWAY, "A", "one-way"): (3.0, 0.0, 0.0, 0.0, 0.0),
}
CEILING = 3.0


@pytest.fixture(scope="module")
def ledger(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hand")
    paths = write_hand_run(folder)
    out = folder / "ledger"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setattr(ledger_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = ledger_run.main(
            [
                "--run",
                str(paths["run"]),
                "--matrix",
                str(paths["matrix"]),
                "--tables",
                str(paths["tables"]),
                "--truth-run",
                str(paths["truth"]),
                "--tree-days",
                "1",
                "--seed",
                "3",
            ]
        )
    return code, json.loads((out / ledger_run.RESULTS_NAME).read_text()), out


def test_the_hand_run_passes_every_check(ledger):
    code, document, _ = ledger
    assert code == 0, document["checks"]
    assert document["checks"]["failed"] is False
    assert document["checks"]["identity"]["breaks"] == []


def test_the_true_set_terms_are_the_hand_values(ledger):
    _, document, _ = ledger
    expected = {NREL: (2.0, 0.0, 1.0, 2.0), NORWAY: (0.0, 0.0, 3.0, 0.0)}  # R, A, box, F
    for unit, (r, a, box, f) in expected.items():
        true = document["units"][unit]["hourly_true_sets"]
        assert true["test_days"]["days"] == 1
        assert true["test_days"]["mean_ceiling_kwh"] == pytest.approx(CEILING)
        assert true["test_days"]["mean_kwh"]["r_plus_a"] == pytest.approx(r + a, abs=1e-7)
        tree = true["tree_sample"]
        assert tree["days"] == 1 and tree["tree_outcomes"] == {"optimal": 1}
        assert tree["mean_kwh"]["r"] == pytest.approx(r, abs=1e-7)
        assert tree["mean_kwh"]["a"] == pytest.approx(a, abs=1e-7)
        assert tree["a_not_zero_days"] == 0
        affine = true["affine_minus_diagonal"]
        assert affine["days"] == 1
        assert affine["mean_kwh"]["affine_minus_diagonal"] == pytest.approx(0.0, abs=1e-7)
        fmt = true["format_loss"]
        assert fmt["days"] == 1 and fmt["sets"] == "true" and fmt["base_set"] == "day-average"
        assert fmt["mean_kwh"]["box"] == pytest.approx(box, abs=1e-7)
        assert fmt["mean_kwh"]["format_loss"] == pytest.approx(f, abs=1e-7)


def test_r_is_above_zero_where_the_decision_hour_matters_and_zero_where_it_does_not(ledger):
    _, document, _ = ledger
    nrel = document["units"][NREL]["hourly_true_sets"]["tree_sample"]["mean_kwh"]["r"]
    norway = document["units"][NORWAY]["hourly_true_sets"]["tree_sample"]["mean_kwh"]["r"]
    assert nrel > 1.0
    assert norway == pytest.approx(0.0, abs=1e-7)


def test_the_block_product_is_the_hand_values(ledger):
    _, document, _ = ledger
    for unit, r in ((NREL, 2.0), (NORWAY, 0.0)):
        block = document["units"][unit]["block_product"]
        assert block["drawn"] == 1
        assert block["tree_sample"]["mean_ceiling_kwh"] == pytest.approx(CEILING, abs=1e-7)
        assert block["tree_sample"]["mean_kwh"]["r"] == pytest.approx(r, abs=1e-7)
        assert block["tree_sample"]["mean_kwh"]["a"] == pytest.approx(0.0, abs=1e-7)
        price = block["price_of_the_block"]
        assert price["mean_kwh"]["price_of_the_block"] == pytest.approx(0.0, abs=1e-7)
        assert block["u"] == block["k"] == "not computed: no block offer in the gate"


def test_every_rung_and_tier_is_the_hand_values(ledger):
    _, document, _ = ledger
    for (name, unit, tier, rung), (offer, r, a, u, k) in HAND.items():
        found = document["units"][unit]["configurations"][name]["tiers"][tier][rung]
        tree = found["tree_sample"]
        assert tree["days"] == 1
        for term, value in zip(("offer", "r", "a", "u", "k"), (offer, r, a, u, k), strict=True):
            assert tree["mean_kwh"][term] == pytest.approx(value, abs=1e-7), (name, unit, term)
            assert tree["share_of_ceiling"][term] == pytest.approx(value / CEILING, abs=1e-7)
        test = found["test_days"]
        assert test["mean_kwh"]["r_plus_a"] == pytest.approx(r + a, abs=1e-7)
        assert test["mean_kwh"]["offer"] == pytest.approx(offer, abs=1e-12)
        assert found["run_mean_offer_kwh_per_test_day"] == pytest.approx(offer, abs=1e-12)
        assert abs(test["identity_residual_kwh"]) <= 1e-9 * CEILING
        assert abs(tree["identity_residual_kwh"]) <= 1e-9 * CEILING


def test_the_day_rows_hold_the_hand_values(ledger):
    _, _, out = ledger
    rows = list(csv.DictReader((out / ledger_run.DAYS_NAME).read_text().splitlines()))
    assert len(rows) == 2 * 2 * 2 * 3  # configurations, units, tiers, rungs; one test day each
    chosen = {(r["configuration"], r["unit"], r["tier"], r["rung"]): r for r in rows}
    row = chosen[(EMPIRICAL, NREL, "A", "fixed")]
    assert row["in_tree_sample"] == "True"
    for column, value in (
        ("offer_kwh", 0.4),
        ("r_kwh", 2.0),
        ("a_kwh", 0.0),
        ("u_kwh", 0.2),
        ("k_kwh", 0.4),
        ("ceiling_kwh", 3.0),
    ):
        assert float(row[column]) == pytest.approx(value, abs=1e-7), column
    assert abs(float(row["identity_residual_r_a_apart_kwh"])) <= 1e-9 * CEILING

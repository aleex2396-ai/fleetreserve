"""The figures of the gate reading in `dial_choice` (plan section 13, WP6), over all splits.

The plan reads the gate on the Tier A mean of the calibrated re-solve offer over the development
splits. `dial_choice` gives, per gate unit, configuration and ceiling, over all the splits: the
mean offer (a split with no safe scale counts as 0), the mean ceiling and the offer as a share of
it, the oracle partner's offer (calibrated by the same rule on the same splits) and its ratio to
the offer. It forms no verdict.

1. The arithmetic (`dial_splits.gate_split`, `dial_splits.summarise_gate`) against a hand count,
   including a split with no safe scale, a split with a missing ceiling and the division by zero.
2. On made-up days, on the split the gate run uses (the matrix's seed): the ceiling, the share of
   it and the oracle offer equal the gate run's own figures (`gate_calibrate.run_job`,
   `gate_stat.unit_offer`) at the largest grid scale not above the gate run's scale.
3. The command: the figures are in the results and the summary beside the plan's thresholds, the
   numbers `dial_choice` gave before are unchanged, a missing ceiling is counted and never read as
   0, workers do not change the figures, and no verdict word is written.
"""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    gate_cars,
    gate_units,
    matrix_document,
    write_build,
    write_tables,
)

from fleetreserve import (
    dial_choice,
    dial_splits,
    gate_calibrate,
    gate_config,
    gate_day,
    gate_run,
    gate_stat,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.dial_splits import GRID, UnitLosses
from fleetreserve.forecast_empirical import EmpiricalForecaster

H, B = gate_config.CEILINGS  # hindsight-free-decision-hour, best-causal
PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
NAMES = PLAIN + LOW
SPLITS = 6
VERDICT = re.compile(r"\b(go|no go|pass|passes|passed|passing|fail|fails|failed|failing)\b", re.I)


# --- 1. the arithmetic against a hand count ------------------------------------------------------


def _losses(values, offering=None) -> UnitLosses:
    n = len(values)
    zeros = np.zeros((n, len(GRID)))
    return UnitLosses(
        value=np.array(values, dtype=float),
        offering=np.ones(n, dtype=bool) if offering is None else np.array(offering, dtype=bool),
        slot=zeros,
        driver=zeros.copy(),
        ceiling=np.full(n, np.nan),
    )


DAYS = 40
V = np.arange(1, DAYS + 1, dtype=float)  # the certified value of day d is d + 1
# The oracle partner offers twice as much, but abstains on days 0 to 4 (so with 15 offering
# calibration days in a calibration set that holds them it has too few and offers nothing).
ORACLE_OFFERING = [d >= 5 for d in range(DAYS)]
HIND = V + 10.0
BEST = V + 5.0
BEST[25] = np.nan  # a timeout on day 25
CEILINGS = {H: HIND, B: BEST}
FIRST = list(range(0, 20))
SECOND = list(range(20, 40))
SPLIT_A = (1, FIRST, SECOND)  # 20 offering calibration days: scale 1 (oracle: 15, none)
SPLIT_B = (2, FIRST[:10], FIRST[10:])  # 10 offering calibration days: no safe scale
SPLIT_C = (3, SECOND, FIRST)  # 20 offering calibration days: scale 1 for both


def _gate_splits() -> list:
    losses = _losses(V)
    oracle = _losses(2 * V, ORACLE_OFFERING)
    return [
        dial_splits.gate_split(losses, oracle, CEILINGS, *split)
        for split in (SPLIT_A, SPLIT_B, SPLIT_C)
    ]


def test_the_premises_of_the_hand_count_hold():
    losses, oracle = _losses(V), _losses(2 * V, ORACLE_OFFERING)
    scales = [
        dial_splits.score_split(x, *s, "A").scale
        for x in (losses, oracle)
        for s in (SPLIT_A, SPLIT_B, SPLIT_C)
    ]
    assert scales == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_the_all_splits_mean_counts_a_split_with_no_safe_scale_as_zero():
    found = _gate_splits()
    assert [s.offer_kwh for s in found] == pytest.approx([30.5, 0.0, 10.5])
    summary = dial_splits.summarise_gate(found, [H, B])
    # (30.5 + 0 + 10.5) / 3, not the mean (30.5 + 10.5) / 2 of the splits that offer
    assert summary["mean_offer_kwh_per_test_day"] == pytest.approx(41.0 / 3)
    assert summary["tier"] == "A" and summary["splits"] == 3


def test_the_share_of_the_ceiling_is_the_mean_offer_over_the_mean_ceiling():
    summary = dial_splits.summarise_gate(_gate_splits(), [H, B])
    mean_ceiling = (40.5 + 25.5 + 20.5) / 3  # days 20-39, 10-19, 0-19 of V + 10
    found = summary["ceilings"][H]
    assert found["mean_ceiling_kwh_per_test_day"] == pytest.approx(mean_ceiling)
    assert found["share_of_ceiling"] == pytest.approx((41.0 / 3) / mean_ceiling)
    assert (found["splits_with_ceiling"], found["splits_left_out_ceiling_missing"]) == (3, 0)


def test_a_missing_ceiling_leaves_the_split_out_and_is_counted_never_read_as_zero():
    found = _gate_splits()
    assert found[0].ceiling_kwh[B] is None and found[1].ceiling_kwh[B] is not None
    summary = dial_splits.summarise_gate(found, [H, B])
    best = summary["ceilings"][B]
    assert (best["splits_with_ceiling"], best["splits_left_out_ceiling_missing"]) == (2, 1)
    # splits B and C: ceilings 20.5 and 15.5, offers 0 and 10.5
    assert best["mean_ceiling_kwh_per_test_day"] == pytest.approx(18.0)
    assert best["mean_offer_kwh_per_test_day_on_those_splits"] == pytest.approx(5.25)
    assert best["share_of_ceiling"] == pytest.approx(5.25 / 18.0)
    # the mean offer over all splits is not changed by the missing ceiling
    assert summary["mean_offer_kwh_per_test_day"] == pytest.approx(41.0 / 3)
    # a ceiling of 0 for the missing day would have given another mean
    assert best["mean_ceiling_kwh_per_test_day"] != pytest.approx((30.5 + 20.5 + 15.5) / 3)


def test_the_oracle_offer_is_calibrated_on_its_own_losses_on_the_same_splits():
    found = _gate_splits()
    assert [s.oracle_offer_kwh for s in found] == pytest.approx([0.0, 0.0, 21.0])
    summary = dial_splits.summarise_gate(found, [H, B])
    assert summary["mean_oracle_offer_kwh_per_test_day"] == pytest.approx(7.0)
    assert summary["oracle_to_offer_ratio"] == pytest.approx(7.0 / (41.0 / 3))


def test_a_mean_offer_of_zero_gives_no_ratio_and_no_division():
    losses, oracle = _losses(V), _losses(2 * V)
    found = [dial_splits.gate_split(losses, oracle, CEILINGS, *SPLIT_B)]  # no safe scale
    summary = dial_splits.summarise_gate(found, [H])
    assert summary["mean_offer_kwh_per_test_day"] == 0.0
    assert summary["oracle_to_offer_ratio"] is None
    assert "mean offer is 0" in summary["oracle_to_offer_ratio_note"]
    assert summary["ceilings"][H]["share_of_ceiling"] == 0.0  # an offer of 0 is a share of 0


def test_a_ceiling_mean_that_is_not_positive_gives_no_share():
    zero = {H: np.zeros(DAYS)}
    found = [dial_splits.gate_split(_losses(V), _losses(V), zero, *SPLIT_A)]
    summary = dial_splits.summarise_gate(found, [H])
    assert summary["ceilings"][H]["share_of_ceiling"] is None


def test_every_split_missing_its_ceiling_gives_no_share():
    nothing = {B: np.full(DAYS, np.nan)}
    found = [dial_splits.gate_split(_losses(V), _losses(V), nothing, *SPLIT_A)]
    best = dial_splits.summarise_gate(found, [B])["ceilings"][B]
    assert best["splits_with_ceiling"] == 0 and best["splits_left_out_ceiling_missing"] == 1
    assert best["share_of_ceiling"] is None and best["mean_ceiling_kwh_per_test_day"] is None


def test_the_unit_and_its_oracle_partner_must_have_the_same_days():
    with pytest.raises(ValueError, match="same days"):
        dial_splits.gate_split(_losses(V), _losses(V[:-1]), CEILINGS, *SPLIT_A)


# --- 2. against the gate run on its own split ----------------------------------------------------


def _gate_floor(scale: float) -> float:
    return max((g for g in GRID if g <= scale), default=0.0)


@pytest.fixture(scope="module")
def against_the_gate(tmp_path_factory):
    folder = tmp_path_factory.mktemp("against")
    matrix = gate_config.parse(matrix_document(NAMES))
    write_tables(folder, "early", gate_cars(60, early_every=6, early_out=12.0))
    days = load_unit_days(folder, "early", rho=0.0, no_fit="cap", small_fleet="keep")
    plan = gate_run.plan_unit("early", days, matrix)
    ceilings = [gate_day.day_ceilings(t.day) for t in days.days]

    def batch(pairs):
        return [gate_day.evaluate(p, b).compact() for p, b in pairs]

    def prepare(name):
        forecaster = EmpiricalForecaster(days, reference="other-days", min_cell=20)
        c = matrix.configuration(name)
        return [
            gate_day.prepare_day(t, forecaster, c, gate_run.settings(matrix)) for t in days.days
        ]

    found = {}
    for name in (LOW[0], LOW[1]):
        prepared = prepare(name)
        result, *_ = gate_calibrate.run_job(prepared, ceilings, plan.roles, batch, tier_b=False)
        losses, _ = dial_choice.measure(prepared, [c.best_causal for c in ceilings], batch)
        found[name] = (prepared, result["rungs"]["one-way"]["beta_hat"], losses)
    arrays = {
        name: np.array(
            [
                np.nan if v is None else v
                for v in (gate_calibrate.ceiling_values(c, name) for c in ceilings)
            ]
        )
        for name in gate_config.CEILINGS
    }
    (split,) = dial_choice.split_indices("early", days, matrix, [matrix.split_seed])
    test = [d for d, role in enumerate(plan.roles) if role == "test"]
    reading = dial_splits.gate_split(found[LOW[0]][2], found[LOW[1]][2], arrays, *split)
    return matrix, ceilings, test, found, reading, split


def test_the_ceiling_and_its_share_are_those_of_the_gate_runs_first_decision(against_the_gate):
    _, ceilings, test, found, reading, split = against_the_gate
    prepared, beta, _ = found[LOW[0]]
    assert beta > 0  # the comparison is not between two empty offers
    assert split[2] == test  # the same test days
    for name in gate_config.CEILINGS:
        gate_ceiling = [gate_calibrate.ceiling_values(ceilings[d], name) for d in test]
        assert reading.ceiling_kwh[name] == pytest.approx(sum(gate_ceiling) / len(test))
        unit = gate_stat.unit_offer(
            [prepared[d].value for d in test], gate_ceiling, _gate_floor(beta)
        )
        assert reading.offer_kwh / reading.ceiling_kwh[name] == pytest.approx(unit.share_of_sums)
        summary = dial_splits.summarise_gate([reading], gate_config.CEILINGS)
        assert summary["ceilings"][name]["share_of_ceiling"] == pytest.approx(unit.share_of_sums)


def test_the_oracle_offer_is_that_of_the_gate_runs_oracle_partner(against_the_gate):
    _, _, test, found, reading, _ = against_the_gate
    prepared, beta, _ = found[LOW[1]]
    assert beta > 0 and reading.oracle_offer_kwh > 0
    expected = _gate_floor(beta) * sum(prepared[d].value for d in test) / len(test)
    assert reading.oracle_offer_kwh == pytest.approx(expected)
    own = _gate_floor(found[LOW[0]][1]) * sum(found[LOW[0]][0][d].value for d in test) / len(test)
    assert reading.offer_kwh == pytest.approx(own)


# --- 3. the command ------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _run(build, out: Path, *extra, patch=None):
    run, tables, matrix = build
    arguments = ["--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)]
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        if patch is not None:
            patch(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = dial_choice.main(
            ["--model", "empirical", *arguments, "--splits", str(SPLITS), *extra]
        )
    return code, out


def _results(out: Path) -> dict:
    return json.loads((out / dial_choice.RESULTS_NAME).read_text())


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    return _run(build, tmp_path_factory.mktemp("full") / "out")


def test_every_split_unit_and_configuration_has_the_figures(full):
    code, out = full
    assert code == 0
    document = _results(out)
    run = document["run"]
    assert run["ceilings"] == [H, B] and run["gate_ceiling"] == H
    assert run["thresholds"] == {"share_of_ceiling": 0.1, "oracle_to_offer_ratio": 1.5}
    pairs = {PLAIN[0]: PLAIN[1], LOW[0]: LOW[1]}
    for unit in GATE_UNITS[1:]:
        for name, found in document["units"][unit]["configurations"].items():
            reading = found["gate_reading"]
            assert reading["oracle_partner"] == pairs[name] and reading["splits"] == SPLITS
            # the mean offer over all splits is the number dial_choice always gave
            assert reading["mean_offer_kwh_per_test_day"] == pytest.approx(
                found["tiers"]["A"]["mean_offer_kwh_per_test_day"]
            )
            assert set(reading["ceilings"]) == {H, B}
            for ceiling in reading["ceilings"].values():
                assert ceiling["splits_with_ceiling"] == SPLITS
                assert ceiling["splits_left_out_ceiling_missing"] == 0
            first = reading["ceilings"][H]
            assert first["share_of_ceiling"] == pytest.approx(
                reading["mean_offer_kwh_per_test_day"] / first["mean_ceiling_kwh_per_test_day"]
            )
            if reading["mean_offer_kwh_per_test_day"] > 0:
                assert reading["oracle_to_offer_ratio"] == pytest.approx(
                    reading["mean_oracle_offer_kwh_per_test_day"]
                    / reading["mean_offer_kwh_per_test_day"]
                )
            else:
                assert reading["oracle_to_offer_ratio"] is None
    assert "gate_reading" not in document["units"][GATE_UNITS[0]]["configurations"]


def test_the_oracle_figures_are_not_the_calibrated_ones(full):
    _, out = full
    document = _results(out)
    readings = [
        found["gate_reading"]
        for unit in GATE_UNITS[1:]
        for found in document["units"][unit]["configurations"].values()
    ]
    assert any(
        r["mean_oracle_offer_kwh_per_test_day"] != r["mean_offer_kwh_per_test_day"]
        for r in readings
    )
    assert any(r["mean_oracle_offer_kwh_per_test_day"] > 0 for r in readings)


def test_the_summary_shows_the_figures_beside_the_thresholds(full):
    _, out = full
    text = (out / dial_choice.SUMMARY_NAME).read_text()
    assert "Gate reading, tier A (driver-protected), mean over all 6 splits" in text
    assert f"ceiling {H} (the ceiling of the gate run's first decision)" in text
    assert f"ceiling {B}\n" in text
    assert "oracle/offer" in text and "splits without ceiling" in text
    assert "not applied here: the offer is at least 10% of the ceiling" in text
    assert "the oracle-departure offer is at least 1.5 times the offer" in text
    assert "of ceiling (offering splits only)" in text


def test_nothing_in_the_output_reads_as_a_verdict(full):
    _, out = full
    text = (out / dial_choice.SUMMARY_NAME).read_text()
    assert not VERDICT.findall(text), VERDICT.findall(text)
    assert "no verdict" in text

    def keys(value):
        if isinstance(value, dict):
            for k, v in value.items():
                yield k
                yield from keys(v)
        elif isinstance(value, list):
            for v in value:
                yield from keys(v)

    document = _results(out)
    for unit in document["units"].values():
        for found in unit["configurations"].values():
            assert not [k for k in keys(found["gate_reading"]) if VERDICT.search(k)]


def test_a_missing_ceiling_is_counted_and_never_read_as_zero(build, tmp_path_factory, full):
    def patch(monkeypatch):
        real = gate_day.day_ceilings

        def with_a_timeout(day):
            found = real(day)
            return dataclasses.replace(found, best_causal=None, best_causal_outcome="timeout")

        monkeypatch.setattr(gate_day, "day_ceilings", with_a_timeout)

    code, out = _run(build, tmp_path_factory.mktemp("missing") / "out", patch=patch)
    assert code == 0
    document, plain = _results(out), _results(full[1])
    assert document["timeouts"]["best_causal_ceilings"] > 0
    for unit in GATE_UNITS[1:]:
        for name, found in document["units"][unit]["configurations"].items():
            reading = found["gate_reading"]
            best, hindsight = reading["ceilings"][B], reading["ceilings"][H]
            assert best["splits_with_ceiling"] == 0
            assert best["splits_left_out_ceiling_missing"] == SPLITS
            assert (
                best["share_of_ceiling"] is None and best["mean_ceiling_kwh_per_test_day"] is None
            )
            assert hindsight["splits_left_out_ceiling_missing"] == 0
            assert hindsight["share_of_ceiling"] is not None
            same = plain["units"][unit]["configurations"][name]["gate_reading"]
            assert reading["mean_offer_kwh_per_test_day"] == same["mean_offer_kwh_per_test_day"]
            assert hindsight == same["ceilings"][H]


def test_a_missing_ceiling_on_some_days_leaves_out_only_the_splits_that_test_on_them(
    build, tmp_path_factory, full
):
    calls = []

    def patch(monkeypatch):
        real = gate_day.day_ceilings

        def sometimes_missing(day):
            found = real(day)
            calls.append(1)
            if len(calls) % 11 == 0:
                return dataclasses.replace(found, best_causal=None, best_causal_outcome="timeout")
            return found

        monkeypatch.setattr(gate_day, "day_ceilings", sometimes_missing)

    code, out = _run(build, tmp_path_factory.mktemp("some") / "out", patch=patch)
    assert code == 0
    document, plain = _results(out), _results(full[1])
    left_out = []
    for unit in GATE_UNITS[1:]:
        for name, found in document["units"][unit]["configurations"].items():
            reading = found["gate_reading"]
            best = reading["ceilings"][B]
            assert best["splits_with_ceiling"] + best["splits_left_out_ceiling_missing"] == SPLITS
            left_out.append(best["splits_left_out_ceiling_missing"])
            same = plain["units"][unit]["configurations"][name]["gate_reading"]
            assert reading["mean_offer_kwh_per_test_day"] == same["mean_offer_kwh_per_test_day"]
            assert reading["ceilings"][H] == same["ceilings"][H]
            if best["splits_with_ceiling"]:
                assert best["mean_ceiling_kwh_per_test_day"] > 0
    assert any(n > 0 for n in left_out) and any(n < SPLITS for n in left_out)


def test_the_figures_do_not_depend_on_the_number_of_workers(build, tmp_path_factory, full):
    code, out = _run(build, tmp_path_factory.mktemp("workers") / "out", "--workers", "2")
    assert code == 0
    one, two = _results(full[1]), _results(out)
    for unit in GATE_UNITS[1:]:
        assert one["units"][unit]["configurations"] == two["units"][unit]["configurations"]
    assert one["timeouts"] == two["timeouts"]


def test_the_timeouts_of_the_oracle_partner_are_counted_apart(full):
    document = _results(full[1])
    assert document["timeouts"]["oracle_certificates"] == 0
    assert document["timeouts"]["oracle_re_solves"] == 0

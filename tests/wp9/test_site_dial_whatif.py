"""The site-dial what-if (`site_dial_whatif`) and its summary (`site_dial_summary`).

On made-up `dial_splits.UnitLosses` (no solver), 60 counting days that all offer: calibration
days 0 to 39 (n = 40), test days 40 to 59, choosing part of S1 days 0 to 19 (given by hand),
calibrating part days 20 to 39. The grid is 0.05, 0.1, 0.2, ..., 0.8, 1. Tier A's bound on the
driver loss at alpha 0.05 holds while the loss sum is at most 0.05 (n + 1) - 1: 1.05 on 40 days,
0.05 on 20 days; at the union level 0.025 (two dials) 0.025 on 40 days.

1. A unit whose best dial differs from the fixed one gains under S1, S2 and S3. Fixed dial 0.02:
   value 10, driver loss 0.04 on every day from scale 0.5 on (40 days: sum 1.6 > 1.05, so scale
   0.4; offer 0.4 x 10 = 4). Dial 0.1: value 12, no loss (scale 1, offer 12). Every variant takes
   0.1 and offers 12: +8 over the fixed arm, +8 over its reference (the fixed dial calibrated the
   variant's way: 0.4 x 10 in each). Tier B (slot loss only, none): 10 against 12, +2.
2. All dials alike (both dials value 10, slot loss 0.06 and driver loss 0.02 from scale 0.5 on):
   variant minus reference is exactly 0 in every variant and both tiers, the tie goes to the fixed
   dial, and S3 minus fixed is exactly 0. The price of the device shows in both tiers: on 40 days
   the slot sum 2.4 <= 0.1 x 41 - 1 = 3.1 and the driver sum 0.8 <= 1.05 hold to scale 1 (offer
   10); on S1's 20 calibrating days the slot sum 1.2 > 1.1 fails at 0.5 (scale 0.4, offer 4, -6);
   at S2's slot level 0.05 on 40 days 2.4 > 1.05 fails at 0.5 (-6).
3. S1 chooses on the choosing part only and calibrates on the calibrating part only. (a) Dial
   0.1's losses only on the calibrating part: chosen (12 > 10 on the choosing part), scale 0.4 on
   the calibrating part, offer 4.8. (b) Its losses only on the choosing part: there its scale is
   0.4 (offer 4.8 < 10), so the fixed dial is taken, though on all 40 days 0.1 would win (sum 0.8
   <= 1.05, offer 12).
4. S2's division of the risk is the stated one: levels (0.10/K, 0.05/K) in tier A and (0.10/K, 1)
   in tier B; with K = 8 dials the fewest offering calibration days are 79 (slot) and 159 (driver,
   tier A); on 100 calibration days without any loss, tier A under S2 takes no dial and offers 0
   (every dial "too few"), tier B offers at scale 1, and the plan's levels (S3, fixed) offer at
   scale 1 in both tiers. A case where the divided level lowers the scale: 2 dials, driver loss
   0.02 from 0.5 on, 40 days: S2's scale 0.4, the plan's 1.
5. S3 takes the largest calibrated offer, not the largest value: dial 0.1 (value 15) with scale 0.4
   offers 6 on the calibration days, the fixed dial (value 10, no loss) 10; S3 takes 0.02.
6. The test losses are those of the test days only: no loss on calibration days, slot 0.3 and
   driver 0.2 at every scale on test days; every arm reports 0.3 and 0.2 and the splits over the
   limits; scales 1 (calibration sees no loss).
7. The choice rule: ties to the fixed dial, else the smallest; None when no offer is above 0.
8. The choosing part: half the calibration days rounded down, within them, a function of the
   unit, the seed and the days' keys (not of their order), other for another seed, and not the
   days the split's own draw ranks first.
9. The figures over splits by hand: mean, p5, p95 skip missing values; higher, equal, lower; dials
   taken with "none"; the plan's rule per dial.

On the made-up gate units of `gate_fake` (60 days, 50 calibration days, one service hour), with
two dials (0.03, the fixed one, and 0.1) and 6 splits. On these units every dial of the family
gives the same days, losses and values (a scratch run measured all eight), so the runs below make
dial 0.1's certified value 1.5 times dial 0.03's on every day it offers (`richer_dial`), its
losses unchanged: then a choice taken by mistake from the wrong dial's figures shows.

10. A run ends with exit 0 and writes its four files; the summary starts with the what-if label,
    names the unit that cannot be split, gives a table per tier with four arms per unit.
11. End to end, a variant that takes dial 0.1 offers exactly 1.5 times its reference (same losses,
    so the same scale): S1 and S2 minus reference is 0.5 x the reference's offer on every split
    where they take 0.1 (and 0 where no dial offered on the choosing days); S3 minus fixed is 0.5 x
    the fixed offer wherever S3 takes a dial; each variant takes 0.1 and gains on some split.
12. The fixed arm is `dial_choice`'s figure: per unit, tier and dial, the mean over splits of the
    dial calibrated on all calibration days equals the mean offer `dial_choice` writes for the
    same build, matrix, splits and values, and the dials' means differ.
13. A timeout is a recorded outcome: a certificate that times out makes the day abstain under its
    dial (offering days down by one per dial) and is counted; a re-solve that times out is
    counted.
14. Refusals before any day (exit 2, nothing written): a gate unit without a fixed dial, a dial
    not of the family, a unit not of the matrix, a unit named twice, a rotation matrix, no dial
    family, `--splits 0`.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from gate_fake import COMMIT, GATE_UNITS, gate_units, matrix_document, write_build

from fleetreserve import (
    crc,
    dial_choice,
    dial_splits,
    gate_day,
    gate_run,
    run_verify,
    site_dial_summary,
    site_dial_whatif,
)
from fleetreserve.day_split import split_days

GRID = np.array(dial_splits.GRID)
CAL, TEST, PICK = list(range(40)), list(range(40, 60)), list(range(20))
FIXED, OTHER = 0.02, 0.1


def unit(value, driver=None, slot=None, days=60) -> dial_splits.UnitLosses:
    """A made-up unit: every day offers with certified value `value`; `driver`, `slot`: per-day
    losses of shape (days, len(GRID)), none by default."""
    shape = (days, len(GRID))
    return dial_splits.UnitLosses(
        value=np.full(days, float(value)),
        offering=np.ones(days, dtype=bool),
        slot=np.zeros(shape) if slot is None else slot,
        driver=np.zeros(shape) if driver is None else driver,
        ceiling=np.full(days, np.nan),
    )


def from_scale(loss: float, scale: float, on=range(60), days=60) -> np.ndarray:
    """Per-day losses: `loss` on the days `on` at every grid scale from `scale` on, else 0."""
    out = np.zeros((days, len(GRID)))
    scales = np.asarray(dial_splits.GRID)
    out[np.ix_(list(on), np.flatnonzero(scales >= scale))] = loss
    return out


def arms(losses, tier="A", pick=PICK, fixed=FIXED):
    return site_dial_whatif.split_arms(losses, fixed, 1, CAL, TEST, pick, tier)


# --- 1. a better dial gains ----------------------------------------------------------------------


def test_a_unit_whose_best_dial_is_not_the_fixed_one_gains_under_every_variant():
    losses = {FIXED: unit(10, driver=from_scale(0.04, 0.5)), OTHER: unit(12)}
    found = arms(losses)
    assert found["fixed"]["dial"] == FIXED
    assert found["fixed"]["scale"] == 0.4
    assert found["fixed"]["offer_kwh_per_test_day"] == pytest.approx(4.0)
    for variant in ("S1", "S2", "S3"):
        arm = found[variant]
        assert arm["dial"] == OTHER and arm["scale"] == 1.0, variant
        assert arm["offer_kwh_per_test_day"] == pytest.approx(12.0)
        assert arm["minus_fixed"] == pytest.approx(8.0)
        assert arm["reference_offer_kwh_per_test_day"] == pytest.approx(4.0)
        assert arm["minus_reference"] == pytest.approx(8.0)
    assert found["S1"]["offering_calibration_days"] == 20  # the calibrating part
    assert found["S2"]["offering_calibration_days"] == 40
    assert found["S1"]["choosing_offers"] == {"0.02": pytest.approx(4.0), "0.1": 12.0}
    tier_b = arms(losses, "B")
    assert tier_b["fixed"]["offer_kwh_per_test_day"] == pytest.approx(10.0)
    for variant in ("S1", "S2", "S3"):
        assert tier_b[variant]["minus_fixed"] == pytest.approx(2.0)


# --- 2. all dials alike --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["A", "B"])
def test_when_every_dial_is_alike_the_choice_changes_nothing(tier):
    slot, driver = from_scale(0.06, 0.5), from_scale(0.02, 0.5)
    losses = {
        FIXED: unit(10, driver=driver, slot=slot),
        OTHER: unit(10, driver=driver, slot=slot),
    }
    found = arms(losses, tier)
    for variant in ("S1", "S2", "S3"):
        assert found[variant]["dial"] == FIXED, variant
        assert found[variant]["minus_reference"] == 0.0, variant
    assert found["S3"]["minus_fixed"] == 0.0
    assert found["fixed"]["offer_kwh_per_test_day"] == pytest.approx(10.0)
    assert found["S1"]["offer_kwh_per_test_day"] == pytest.approx(4.0)
    assert found["S2"]["offer_kwh_per_test_day"] == pytest.approx(4.0)
    assert found["S1"]["minus_fixed"] == pytest.approx(-6.0)
    assert found["S2"]["minus_fixed"] == pytest.approx(-6.0)


# --- 3. S1's two parts ---------------------------------------------------------------------------


def test_s1_calibrates_its_scale_on_the_calibrating_part_only():
    later = from_scale(0.04, 0.5, on=range(20, 40))
    losses = {FIXED: unit(10), OTHER: unit(12, driver=later)}
    found = arms(losses)["S1"]
    assert found["choosing_offers"] == {"0.02": 10.0, "0.1": 12.0}
    assert found["dial"] == OTHER
    assert found["scale"] == 0.4  # 20 days x 0.04 = 0.8 > 0.05 at 0.5
    assert found["offer_kwh_per_test_day"] == pytest.approx(4.8)
    assert arms(losses)["S3"]["scale"] == 1.0  # on all 40 days 0.8 <= 1.05


def test_s1_chooses_on_the_choosing_part_only():
    earlier = from_scale(0.04, 0.5, on=range(20))
    losses = {FIXED: unit(10), OTHER: unit(12, driver=earlier)}
    found = arms(losses)
    assert found["S1"]["choosing_offers"] == {"0.02": 10.0, "0.1": pytest.approx(4.8)}
    assert found["S1"]["dial"] == FIXED
    assert found["S1"]["offer_kwh_per_test_day"] == pytest.approx(10.0)
    assert found["S3"]["dial"] == OTHER  # on all 40 days dial 0.1 wins
    assert found["S3"]["offer_kwh_per_test_day"] == pytest.approx(12.0)


# --- 4. S2's division of the risk ----------------------------------------------------------------


def test_the_union_levels_are_the_plans_levels_over_the_number_of_dials():
    assert site_dial_whatif.union_levels("A", 8) == pytest.approx((0.0125, 0.00625))
    assert site_dial_whatif.union_levels("B", 8) == pytest.approx((0.0125, 1.0))
    assert site_dial_whatif.union_levels("A", 2) == pytest.approx((0.05, 0.025))
    assert crc.min_days(0.0125) == 79 and crc.min_days(0.00625) == 159
    with pytest.raises(ValueError):
        site_dial_whatif.union_levels("A", 0)


def test_with_eight_dials_and_100_calibration_days_s2_abstains_in_tier_a():
    dials = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
    losses = {d: unit(10 + 10 * d, days=120) for d in dials}
    calibration, test = list(range(100)), list(range(100, 120))
    found = site_dial_whatif.split_arms(losses, FIXED, 1, calibration, test, range(50), "A")
    assert found["S2"]["dial"] is None and found["S2"]["scale"] == 0.0
    assert found["S2"]["offer_kwh_per_test_day"] == 0.0
    assert found["S2"]["reason"] == dial_splits.TOO_FEW
    assert set(found["S2"]["choosing_offers"].values()) == {0.0}
    assert found["S3"]["scale"] == 1.0 and found["fixed"]["scale"] == 1.0
    assert found["S3"]["dial"] == 0.1  # the largest value, nothing lost
    tier_b = site_dial_whatif.split_arms(losses, FIXED, 1, calibration, test, range(50), "B")
    assert tier_b["S2"]["dial"] == 0.1 and tier_b["S2"]["scale"] == 1.0


def test_the_divided_level_lowers_s2s_scale():
    alike = from_scale(0.02, 0.5)
    losses = {FIXED: unit(10, driver=alike), OTHER: unit(11, driver=alike)}
    found = arms(losses)
    assert found["S2"]["choosing_offers"] == {"0.02": pytest.approx(4.0), "0.1": pytest.approx(4.4)}
    assert found["S2"]["dial"] == OTHER and found["S2"]["scale"] == 0.4
    assert found["S3"]["dial"] == OTHER and found["S3"]["scale"] == 1.0


# --- 5. S3 takes the largest calibrated offer ----------------------------------------------------


def test_s3_takes_the_largest_calibrated_offer_not_the_largest_value():
    losses = {FIXED: unit(10), OTHER: unit(15, driver=from_scale(0.04, 0.5))}
    found = arms(losses)["S3"]
    assert found["choosing_offers"] == {"0.02": 10.0, "0.1": pytest.approx(6.0)}
    assert found["dial"] == FIXED and found["minus_fixed"] == 0.0


# --- 6. test losses from test days only ----------------------------------------------------------


@pytest.mark.parametrize("tier", ["A", "B"])
def test_the_failure_rates_are_those_of_the_test_days(tier):
    slot, driver = np.zeros((60, len(GRID))), np.zeros((60, len(GRID)))
    slot[TEST, :], driver[TEST, :] = 0.3, 0.2
    losses = {FIXED: unit(10, driver=driver, slot=slot), OTHER: unit(12, driver=driver, slot=slot)}
    found = arms(losses, tier)
    for arm in site_dial_whatif.ARMS:
        assert found[arm]["scale"] == 1.0, arm
        assert found[arm]["slot_loss"] == pytest.approx(0.3), arm
        assert found[arm]["driver_loss"] == pytest.approx(0.2), arm
        assert found[arm]["offering_test_days"] == 20
    figures = site_dial_summary.arm_figures([found["S1"]] * 3, tier, (FIXED, OTHER))
    assert figures["slot_loss_over_limit"] == 3
    assert figures["mean_test_slot_loss"] == pytest.approx(0.3)
    if tier == "A":
        assert figures["driver_loss_over_limit"] == 3
    else:
        assert "driver_loss_over_limit" not in figures


# --- 7. the choice rule --------------------------------------------------------------------------


def test_the_choice_rule():
    assert site_dial_whatif.choose({0.01: 5.0, 0.02: 5.0, 0.1: 4.0}, 0.02) == 0.02
    assert site_dial_whatif.choose({0.01: 5.0, 0.02: 4.0, 0.1: 5.0}, 0.02) == 0.01
    assert site_dial_whatif.choose({0.01: 5.0, 0.02: 5.0 - 1e-10}, 0.02) == 0.02  # within TIE
    assert site_dial_whatif.choose({0.01: 4.0, 0.02: 5.0, 0.1: 6.0}, 0.02) == 0.1
    assert site_dial_whatif.choose({0.01: 0.0, 0.02: 0.0}, 0.02) is None


def test_s1_falls_back_to_the_fixed_dial_when_no_dial_offers_on_the_choosing_part():
    losses = {FIXED: unit(10), OTHER: unit(12)}
    found = arms(losses, pick=range(10))  # 10 days: too few for the driver loss at 0.05
    assert found["S1"]["no_dial_offered"] and found["S1"]["dial"] == FIXED
    assert found["S1"]["scale"] == 1.0 and found["S1"]["offering_calibration_days"] == 30


def test_a_choosing_part_outside_the_calibration_days_is_refused():
    losses = {FIXED: unit(10), OTHER: unit(12)}
    with pytest.raises(ValueError, match="within the calibration days"):
        arms(losses, pick=[45])
    with pytest.raises(ValueError, match="not a dial of the family"):
        arms(losses, fixed=0.05)


# --- 8. the choosing part ------------------------------------------------------------------------


def test_the_choosing_part_is_half_the_calibration_days_by_a_seeded_draw():
    keys = [f"2020-01-{d:02d} A" for d in range(1, 31)] + [
        f"2020-02-{d:02d} A" for d in range(1, 21)
    ]
    calibration = list(range(0, 50, 2)) + [1]  # 26 days
    found = site_dial_whatif.pick_part("u", 3, keys, calibration)
    assert len(found) == 13 and set(found) <= set(calibration) and found == sorted(found)
    assert site_dial_whatif.pick_part("u", 3, keys, list(reversed(calibration))) == found
    assert site_dial_whatif.pick_part("u", 4, keys, calibration) != found
    first = split_days("u", 3, [keys[d] for d in calibration], n_calibration=13, min_days=14)
    assert {keys[d] for d in found} != set(first.calibration)  # not the split's own ranking
    assert site_dial_whatif.pick_part("u", 3, keys, [5]) == []
    assert len(site_dial_whatif.pick_part("u", 3, keys, [5, 7, 9])) == 1


# --- 9. figures over splits ----------------------------------------------------------------------


def _variant(offer, minus_fixed, dial, slot=None, no_dial=False):
    return {
        "dial": dial,
        "scale": 0.0 if dial is None else 0.5,
        "offer_kwh_per_test_day": offer,
        "slot_loss": slot,
        "driver_loss": slot,
        "no_dial_offered": no_dial,
        "minus_fixed": minus_fixed,
        "reference_offer_kwh_per_test_day": offer,
        "minus_reference": 0.0,
    }


def test_the_figures_over_splits_by_hand():
    found = site_dial_summary.arm_figures(
        [
            _variant(2.0, 1.0, 0.1, 0.12),
            _variant(0.0, 0.0, None, None, True),
            _variant(4.0, -1.0, 0.02, 0.08),
            _variant(6.0, 0.0, 0.1, 0.10),
        ],
        "A",
        (0.02, 0.1),
    )
    assert found["offer_kwh_per_test_day"]["mean"] == pytest.approx(3.0)
    assert found["offer_kwh_per_test_day"]["p5"] == pytest.approx(0.3)  # 0 + 0.15 x 2
    assert found["offer_kwh_per_test_day"]["p95"] == pytest.approx(5.7)  # 4 + 0.85 x 2
    assert found["minus_fixed"]["higher"] == 1 and found["minus_fixed"]["lower"] == 1
    assert found["minus_fixed"]["equal"] == 2 and found["minus_fixed"]["mean"] == 0.0
    assert found["dial_taken"] == {"0.02": 1, "0.1": 2, "none": 1}
    assert found["splits_that_offer"] == 3 and found["splits_with_test_losses"] == 3
    assert found["mean_test_slot_loss"] == pytest.approx(0.1)
    assert found["slot_loss_over_limit"] == 1 and found["driver_loss_over_limit"] == 3
    assert found["no_dial_offered_on_choosing_days"] == 1


def test_the_plans_rule_per_dial_by_hand():
    records = [
        {"tiers": {"A": {"by_dial": {"0.02": a, "0.1": b}}}} for a, b in ((4.0, 6.0), (8.0, 4.0))
    ]
    found = site_dial_summary.by_dial(records, (0.02, 0.1))["A"]
    assert found["mean_offer_kwh_per_test_day"] == {"0.02": 6.0, "0.1": 5.0}
    assert found["dial_by_the_plans_rule"] == 0.02


# --- the command on made-up gate units -----------------------------------------------------------


PLAIN = ("main-rounded-hour-k0.3-empirical", "main-rounded-hour-k0.3-oracle")
LOW = ("main-rounded-hour-k0.3-eps0.03-empirical", "main-rounded-hour-k0.3-eps0.03-oracle")
HIGH = ("main-rounded-hour-k0.3-eps0.1-empirical", "main-rounded-hour-k0.3-eps0.1-oracle")
NAMES = PLAIN + LOW + HIGH
SPLITS = "6"
FIXED_DIALS = [f"{u}=0.03" for u in GATE_UNITS]
FILES = (
    site_dial_whatif.RESULTS_NAME,
    site_dial_whatif.SUMMARY_NAME,
    site_dial_whatif.TABLE_NAME,
    site_dial_whatif.TIMING_NAME,
)


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, gate_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _arguments(build, matrix=None) -> list[str]:
    run, tables, committed = build
    return ["--matrix", str(matrix or committed), "--tables", str(tables), "--truth-run", str(run)]


def _what_if(build, out: Path, *extra, fixed=FIXED_DIALS, matrix=None, patch=None) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        if patch is not None:
            patch(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        arguments = ["--model", "empirical", "--splits", SPLITS, "--fixed-dials", *fixed]
        return site_dial_whatif.main([*_arguments(build, matrix), *arguments, *extra])


def _read(out: Path) -> dict:
    return json.loads((out / site_dial_whatif.RESULTS_NAME).read_text())


def richer_dial(monkeypatch):
    """Dial 0.1's certified value 1.5 times dial 0.03's on every day it offers, its losses
    unchanged (module text)."""
    original = gate_day.prepare_day

    def prepare(truth, forecaster, configuration, settings):
        found = original(truth, forecaster, configuration, settings)
        if configuration.driver_risk == 0.1 and not found.abstains:
            return dataclasses.replace(found, value=1.5 * found.value)
        return found

    monkeypatch.setattr(gate_day, "prepare_day", prepare)


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    out = tmp_path_factory.mktemp("full") / "out"
    return _what_if(build, out, patch=richer_dial), out


def test_a_run_writes_every_file_and_its_summary(full):
    code, out = full
    assert code == 0
    for name in FILES:
        assert (out / name).is_file()
    document = _read(out)
    assert document["label"] == site_dial_summary.LABEL
    assert document["failed_units"] == [] and document["run"]["dials"] == [0.03, 0.1]
    assert document["run"]["union_levels"]["A"] == pytest.approx([0.05, 0.025])
    small, usable, pooled = (document["units"][u] for u in GATE_UNITS)
    assert small["status"].startswith("not calibratable") and "splits" not in small
    for entry in (usable, pooled):
        assert len(entry["splits"]) == 6 and set(entry["dials"]) == {"0.03", "0.1"}
        for record in entry["splits"]:
            assert record["choosing_days"] == 25 and record["calibrating_days"] == 25
            assert set(record["tiers"]) == {"A", "B"}
        for tier in ("A", "B"):
            assert set(entry["over_splits"][tier]) == set(site_dial_whatif.ARMS)
    text = (out / site_dial_whatif.SUMMARY_NAME).read_text()
    lines = text.splitlines()
    assert lines[0] == site_dial_summary.LABEL
    assert any(line.startswith(f"{GATE_UNITS[0]}: not calibratable") for line in lines)
    assert "NO GUARANTEE ARGUMENT" in text
    assert text.count("tier A (driver-protected)\n") == 1
    rows = [line for line in lines if line.startswith(f"  {GATE_UNITS[1]} ")]
    assert len(rows) == 2 * 4  # two tiers, four arms


def test_end_to_end_a_variant_taking_the_richer_dial_gains_exactly_its_share(full):
    document = _read(full[1])
    gains = {"S1": 0, "S2": 0, "S3": 0}
    for unit in GATE_UNITS[1:]:
        for record in document["units"][unit]["splits"]:
            for arms_of_tier in record["tiers"].values():
                fixed = arms_of_tier["fixed"]["offer_kwh_per_test_day"]
                s3 = arms_of_tier["S3"]
                if s3["dial"] is not None:
                    assert s3["dial"] == 0.1
                    assert s3["minus_fixed"] == pytest.approx(0.5 * fixed)
                    gains["S3"] += s3["minus_fixed"] > 0
                for variant in ("S1", "S2"):
                    arm = arms_of_tier[variant]
                    reference = arm["reference_offer_kwh_per_test_day"]
                    if arm["dial"] == 0.1:
                        assert arm["minus_reference"] == pytest.approx(0.5 * reference)
                        gains[variant] += arm["minus_reference"] > 0
                    else:
                        assert arm["no_dial_offered"] and arm["minus_reference"] == 0.0
    assert all(count > 0 for count in gains.values()), gains


def test_the_fixed_arm_is_the_dial_choice_figure(build, full, tmp_path):
    _, out = full
    document = _read(out)
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        richer_dial(monkeypatch)
        (tmp_path / "dial").mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path / "dial"))
        code = dial_choice.main(["--model", "empirical", *_arguments(build), "--splits", SPLITS])
    assert code == 0
    theirs = json.loads((tmp_path / "dial" / dial_choice.RESULTS_NAME).read_text())
    compared = 0
    for unit in GATE_UNITS[1:]:
        for name, dial in ((LOW[0], "0.03"), (HIGH[0], "0.1")):
            found = theirs["units"][unit]["configurations"][name]
            assert document["units"][unit]["dials"][dial]["offering_days"] == found["offering_days"]
            for tier in ("A", "B"):
                ours = document["units"][unit]["by_dial"][tier]["mean_offer_kwh_per_test_day"]
                assert ours[dial] == found["tiers"][tier]["mean_offer_kwh_per_test_day"]
                compared += 1
        fixed = document["units"][unit]["over_splits"]["A"]["fixed"]["offer_kwh_per_test_day"]
        found = theirs["units"][unit]["configurations"][LOW[0]]["tiers"]["A"]
        assert fixed["mean"] == found["mean_offer_kwh_per_test_day"]
        means = document["units"][unit]["by_dial"]["A"]["mean_offer_kwh_per_test_day"]
        assert means["0.1"] != means["0.03"]  # the dials differ, so a mix-up would show
    assert compared == 8
    offers = [
        r["tiers"]["A"]["fixed"]["offer_kwh_per_test_day"]
        for u in GATE_UNITS[1:]
        for r in document["units"][u]["splits"]
    ]
    assert any(o > 0 for o in offers)  # the comparison is not one of zeros


def test_a_certificate_timeout_abstains_and_is_counted(build, full, tmp_path):
    original = gate_day.prepare_day
    first = sorted({c.date for c in gate_units()[GATE_UNITS[1]]})[0]

    def patch(monkeypatch):
        def prepare(truth, forecaster, configuration, settings):
            found = original(truth, forecaster, configuration, settings)
            if (truth.record.date, truth.record.site) == (first, "A") and not found.abstains:
                return dataclasses.replace(
                    found,
                    abstain_reason="no certificate (timeout)",
                    certificate_outcome="timeout",
                    value=0.0,
                )
            return found

        monkeypatch.setattr(gate_day, "prepare_day", prepare)

    base = _read(full[1])
    code = _what_if(build, tmp_path / "out", patch=patch)
    document = _read(tmp_path / "out")
    assert code == 0
    for dial in ("0.03", "0.1"):
        before = base["units"][GATE_UNITS[1]]["dials"][dial]
        after = document["units"][GATE_UNITS[1]]["dials"][dial]
        assert after["offering_days"] == before["offering_days"] - 1
        assert after["certificate_timeouts"] == 1
    assert document["timeouts"]["certificates"] == 2 and base["timeouts"]["certificates"] == 0
    assert (
        "Timeouts: certificates 2" in (tmp_path / "out" / site_dial_whatif.SUMMARY_NAME).read_text()
    )


def test_a_re_solve_timeout_is_counted(build, full, tmp_path):
    original = gate_run._evaluate_task

    def patch(monkeypatch):
        def evaluate(job):
            found = original(job)
            prepared, beta = job
            if beta == 1.0:
                rung = found.rungs[dial_choice.RUNG]
                resolves = {**rung.resolves, "timeout": rung.resolves.get("timeout", 0) + 1}
                rungs = {
                    **found.rungs,
                    dial_choice.RUNG: dataclasses.replace(rung, resolves=resolves),
                }
                return dataclasses.replace(found, rungs=rungs)
            return found

        monkeypatch.setattr(gate_run, "_evaluate_task", evaluate)

    base = _read(full[1])
    assert _what_if(build, tmp_path / "out", patch=patch) == 0
    document = _read(tmp_path / "out")
    offering = sum(
        document["units"][u]["dials"][d]["offering_days"]
        for u in GATE_UNITS[1:]
        for d in ("0.03", "0.1")
    )
    assert base["timeouts"]["re_solves"] == 0
    assert document["timeouts"]["re_solves"] == offering > 0  # one per offering day at scale 1


@pytest.mark.parametrize(
    ("fixed", "why"),
    [
        (FIXED_DIALS[:2], "no fixed dial for"),
        ([*FIXED_DIALS[:2], f"{GATE_UNITS[2]}=0.02"], "not a dial of the family"),
        ([*FIXED_DIALS, "other-unit=0.03"], "not a gate unit"),
        ([*FIXED_DIALS, FIXED_DIALS[0]], "named twice"),
        ([*FIXED_DIALS[:2], GATE_UNITS[2]], "takes <unit>=<dial>"),
    ],
)
def test_bad_fixed_dials_could_not_run(build, tmp_path, capsys, fixed, why):
    assert _what_if(build, tmp_path / "out", fixed=fixed) == 2
    assert why in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


def test_other_refusals_before_any_day(build, tmp_path, capsys):
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(matrix_document(PLAIN)))
    assert _what_if(build, tmp_path / "a", matrix=bare) == 2
    assert "no dial family" in capsys.readouterr().out
    rotation = tmp_path / "rotation.json"
    document = matrix_document(NAMES)
    document["gate_units"] = ["jpl"]
    document["required_units"] = 1
    document["rotation_whatif"] = True
    rotation.write_text(json.dumps(document))
    code = _what_if(build, tmp_path / "b", fixed=["jpl=0.03"], matrix=rotation)
    assert code == 2 and "COULD NOT RUN: a rotation what-if matrix" in capsys.readouterr().out
    assert _what_if(build, tmp_path / "c", "--splits", "0") == 2
    assert "--splits must be at least 1" in capsys.readouterr().out
    for name in ("a", "b", "c"):
        assert list((tmp_path / name).iterdir()) == []

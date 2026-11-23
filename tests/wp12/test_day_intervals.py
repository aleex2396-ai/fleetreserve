"""The intervals over days (`day_intervals`), on made-up recorded runs (`interval_fake`) whose
resamples' pairs were evaluated by the scoring run's own code (`day_resample`) with a made-up
step replay.

1. A unit worked by hand, five resamples injected in place of the seeded draws (the made-up run
   evaluated its pairs under the same draws). Ten calibration days c0 to c9, c9 not offering; the
   offering days lose their whole slot (one service hour) at every scale above their thresholds
   0.3, 0.5, 0.5, 0.8 and five times 2.0, and no driver energy. Tier B (slot loss alone, alpha
   0.10) on n offering days holds while (lost days + 1) / (n + 1) <= 0.10: on 9 or 10 days no
   day may lose, so the scale is the largest scale the search reaches at or below the smallest
   threshold drawn; below 9 offering days it abstains. Tier A needs 19 offering days and abstains
   in every resample. The run's own scales: every grid scale, the halving from 0.3 to 0.4 its
   tier B search asked, and the halving from 0.5 to 0.6.
   Six test days t0 to t5: certified values 10, 20, 30, 0 (t3 does not offer), 40, 50 kWh; cars
   3, 4, 5, 3, 4, 5; ceilings 100 and 80 kWh each. The run's tier B scale is 0.3 (its search on
   c0 to c9), its test losses there: slot 0, 1, 0, 0, 0, 0 and driver 0.02, 0.1, 0, 0, 0.04, 0.
   Resamples (calibration positions; test positions) and their tier B scale and offer:
     r1 (0..9; 0..5), the run's days: 0.3; offer 0.3 x 150 / 6 = 7.5 kWh.
     r2 (1,1,2,...,9; 0,0,1,1,2,2): smallest threshold 0.5 on 9 offering days: 0.5; offer
        0.5 x 120 / 6 = 10.
     r3 (0,0,0,3,...,9; 3,3,3,3,4,5): 0.3 (c0 three times loses above 0.3); offer 0.3 x 90 / 6
        = 4.5.
     r4 (3,...,8,8,8,8,8; 5 x 6): 10 offering days, c3 loses above 0.8, so 1.0 fails; the halving
        from 0.8 to 1.0 asks 0.9, 0.85, 0.825, ..., which the run's own searches never asked; the
        scoring run evaluated them for this resample (c3 loses at each), so the search ends at
        0.8 exactly (at c0cec9e, without those pairs, it was only known to lie in
        [0.8, 0.99921875]); offer 0.8 x 50 = 40.
     r5 (0,...,7,9,9; 1,2,3,4,5,5): 8 offering days, too few: abstains, offer 0.
   Percentiles (numpy "linear": position 4 p among the 5 sorted values): offers 0, 4.5, 7.5, 10,
   40 give [0 + 0.1 x 4.5, 10 + 0.9 x 30] = [0.45, 37]; scales 0, 0.3, 0.3, 0.5, 0.8 give
   [0.03, 0.77]; per car 0, 0.3 x 90 / 21, 1.875, 2.5, 8 give [0.1 x 27 / 21, 7.45]; the share
   of the 100 kWh ceiling is the offer over 100. Every resample is exact; one of five abstains.
   Losses at the run's 0.3 over the drawn offering test days: slot 0.2, 1/3, 0, 0, 0.2 and
   driver 0.032, 0.04, 0.02, 0, 0.028: availability [2/3 + 0.1 x (0.8 - 2/3), 1] = [0.68, 1],
   shortfall [0.1 x 0.02, 0.032 + 0.9 x 0.008] = [0.002, 0.0392].
2. The same unit with the pair (c3, 0.9) failing in the run: r4's search is run with c3's losses
   at 0.9 as 1 (it fails there and goes down to 0.8) and as 0 (it holds and climbs; at 0.95,
   0.925, ... c3 loses, so it ends at 0.9): r4 lies in [0.8, 0.9], counted as one resample with a
   failed pair, offers 40 and 45: the offer interval is [0.45, 10 + 0.9 x 35] = [0.45, 41.5],
   an outer bound, and the summary says so.
3. The hand case of the review of c0cec9e (from its verdict): 22 calibration days (c21 not
   offering); c0 loses its slot above 0.2, c1 above 0.4, c2 0.08 of its driver energy above 0.3,
   c3 the same above 0.25, c4 to c20 nothing. The run's three searches on its 21 offering days
   evaluate the grid up to 0.5 and the halvings of [0.3, 0.4] and [0.4, 0.5]; tier A 0.3 (at
   0.4 two driver losses of 0.08 exceed 0.05 x 22 - 1), tier B 0.4 (at 0.5 two slots lost). Six
   test days: values 12, 0 (t1 does not offer), 6, 18, 24, 30; cars 3, 2, 3, 4, 4, 5; ceilings
   40 and 30. Resamples:
     r1 (own days; 0..5): A 0.3, B 0.4.
     r2 (2,3,4,4,4,5..21; 0,0,2,3,4,4): A 0.3; B: no slot loss on 21 offering days, the grid
        holds to its top: 2.0 (at c0cec9e [0.5, 2.0]).
     r3 (0,0,0,3..21; 1,1,1,1,4,5): c0 three times loses above 0.2: A and B 0.2 (at c0cec9e
        [0.2, 0.29921875]).
     r4 (0..18,21,21,21; 5 x 6): 19 offering days, 0.05 x 20 - 1 = 0 driver energy may be lost:
        A holds at 0.25 (c3 loses above it) and fails at 0.275, 0.2625, ...: 0.25 (at c0cec9e
        [0.2, 0.29921875]); B 0.4.
   Tier A offers 0.3 x 90 / 6 = 4.5, 0.3 x 96 / 6 = 4.8, 0.2 x 54 / 6 = 1.8, 0.25 x 180 / 6 =
   7.5: sorted 1.8, 4.5, 4.8, 7.5, position 3 p: [1.8 + 0.075 x 2.7, 4.8 + 0.925 x 2.7] =
   [2.0025, 7.2975] (at c0cec9e the upper end was 8.66). Tier B 6, 32, 1.8, 12: [1.8 + 0.075 x
   4.2, 12 + 0.925 x 20] = [2.115, 30.5].
4. Every day identical: every interval has zero width.
5. With the seed fixed the results are identical across two runs; a run whose pairs were
   evaluated for another seed gives other results under that seed and is refused under the first.
6. Refused, could not run (2): a run that ended with exit 1, a file changed since the run, a run
   recorded at another commit, a manifest naming another command, a folder with no results file
   of the two commands, a run recorded without the resamples' pairs, or with those of another
   seed or number of resamples; a negative seed. No seed: the argument parser stops (2).
7. A per-day file that disagrees with the run's scale fails (1) and gives that job no interval.
8. A learn run's files are read the same way.
"""

import json

import numpy as np
import pytest
from interval_fake import COMMIT, NAME, RUNG, asked, unit, write_manifest, write_run

from fleetreserve import crc, day_bootstrap, day_intervals, run_verify
from fleetreserve.dial_splits import tier_alphas
from fleetreserve.gate_config import CEILINGS

THRESHOLDS = [0.3, 0.5, 0.5, 0.8, 2.0, 2.0, 2.0, 2.0, 2.0]
CALIBRATION_DRAWS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
    [3, 4, 5, 6, 7, 8, 8, 8, 8, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 9],
]
TEST_DRAWS = [
    [0, 1, 2, 3, 4, 5],
    [0, 0, 1, 1, 2, 2],
    [3, 3, 3, 3, 4, 5],
    [5, 5, 5, 5, 5, 5],
    [1, 2, 3, 4, 5, 5],
]
TEST = {
    "values": [10.0, 20.0, 30.0, 0.0, 40.0, 50.0],
    "cars": [3, 4, 5, 3, 4, 5],
    "offers": [True, True, True, False, True, True],
    "ceilings": {CEILINGS[0]: [100.0] * 6, CEILINGS[1]: [80.0] * 6},
}
SLOT = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
DRIVER = [0.02, 0.1, 0.0, 0.0, 0.04, 0.0]
UNIT = "made-up-unit"
SEED = 7


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: COMMIT)


def hand_job(fails=()):
    run_days = THRESHOLDS
    second = [0.5, 0.5, 0.5, 0.8, 2.0, 2.0, 2.0, 2.0, 2.0]  # r2's offering days
    scales = {*crc.GRID, *asked(run_days, "B"), *asked(second, "B")}
    assert 0.9 not in scales
    calibration = {"thresholds": THRESHOLDS, "offers": [True] * 9 + [False]}
    tiers = {"A": (0.0, None, None), "B": (0.3, SLOT, DRIVER)}
    resample = (SEED, len(CALIBRATION_DRAWS))
    return unit(UNIT, calibration, TEST, scales, tiers, resample=resample, fails=fails)


def inject(monkeypatch, calibration, test):
    def draws(seed, name, n_calibration, n_test, resamples):
        assert (n_calibration, n_test, resamples) == (
            len(calibration[0]),
            len(test[0]),
            len(calibration),
        )
        return np.array(calibration), np.array(test)

    monkeypatch.setattr(day_bootstrap, "draws", draws)


def run(folder, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return day_intervals.main(["--run", str(folder), *extra])


def results(out) -> dict:
    return json.loads((out / day_intervals.RESULTS_NAME).read_text())


HAND = ("--seed", str(SEED), "--resamples", "5")


def test_each_resample_by_hand(monkeypatch):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    record, _ = hand_job()
    job = day_bootstrap.Job.from_record(record, [RUNG])
    found = [job.scale(RUNG, "B", row) for row in CALIBRATION_DRAWS]
    assert [(s.low, s.high, s.exact) for s in found] == [
        (0.3, 0.3, True),
        (0.5, 0.5, True),
        (0.3, 0.3, True),
        (0.8, 0.8, True),
        (0.0, 0.0, True),
    ]
    assert found[4].too_few and not any(s.failed_pair or s.missing_pair for s in found)
    assert [job.scale(RUNG, "A", row).high for row in CALIBRATION_DRAWS] == [0.0] * 5


def test_the_command_agrees_with_the_hand_case(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [hand_job()])
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    found = results(tmp_path / "out")["configurations"][NAME][UNIT][RUNG]
    b = found["B"]
    assert b["scale"]["resamples_exact"] == 5 and b["scale"]["resamples"] == 5
    assert b["scale"]["resamples_with_failed_pairs"] == 0
    assert b["scale"]["interval"] == pytest.approx([0.03, 0.77])
    assert b["abstaining"]["resamples_at_scale_zero"] == 1
    assert b["abstaining"]["resamples_too_few_offering_calibration_days"] == 1
    assert b["abstaining"]["share_of_resamples_abstaining"] == 0.2
    offer = b["offer_kwh_per_counting_test_day"]
    assert offer["point"] == pytest.approx(7.5)
    assert offer["interval"] == pytest.approx([0.45, 37.0])
    assert offer["lower_end_between"] == pytest.approx([0.45, 0.45])
    assert offer["upper_end_between"] == pytest.approx([37.0, 37.0])
    per_car = b["offer_kwh_per_car_per_day"]
    assert per_car["point"] == pytest.approx(1.875)
    assert per_car["interval"] == pytest.approx([0.1 * 27 / 21, 7.45])
    share = b["share_of_ceiling"][CEILINGS[0]]
    assert share["point"] == pytest.approx(0.075)
    assert share["interval"] == pytest.approx([0.0045, 0.37])
    assert b["availability"]["point"] == pytest.approx(0.8)
    assert b["availability"]["interval"] == pytest.approx([0.68, 1.0])
    assert b["shortfall"]["point"] == pytest.approx(0.032)
    assert b["shortfall"]["interval"] == pytest.approx([0.002, 0.0392])
    assert b["slot_loss"]["interval"] == pytest.approx([0.0, 0.32])
    assert b["driver_loss"]["interval"] == b["shortfall"]["interval"]
    a = found["A"]
    assert a["offer_kwh_per_counting_test_day"]["interval"] == [0.0, 0.0]
    assert a["abstaining"]["share_of_resamples_abstaining"] == 1.0
    assert a["availability"]["interval"] is None
    lines = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    assert any(
        "calibration days and its test days are drawn with replacement, each group as many days "
        "as the run has"
        in line
        and "the test days alone are drawn, at the run's scale" in line
        for line in lines
    )
    assert "Every scale is exact: in 10 of 10 resamples" in "\n".join(lines)
    (mine,) = [line for line in lines if line.startswith(f"{NAME} | {UNIT} | {RUNG} | ")][1:]
    assert "offer 7.500 [0.450, 37.000] kWh per counting test day" in mine
    assert "every scale exact (5 of 5)" in mine and "outer bound" not in mine
    assert "abstains in 1 of 5 resamples (1 for too few offering calibration days)" in mine


def test_a_failed_pair_is_bounded_counted_and_named(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [hand_job(fails=[(3, 0.9)])])
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    b = results(tmp_path / "out")["configurations"][NAME][UNIT][RUNG]["B"]
    assert b["scale"]["resamples_exact"] == 4
    assert b["scale"]["resamples_with_failed_pairs"] == 1
    assert b["scale"]["resamples_with_pairs_not_evaluated"] == 0
    offer = b["offer_kwh_per_counting_test_day"]
    assert offer["interval"] == pytest.approx([0.45, 41.5])
    assert offer["upper_end_between"] == pytest.approx([37.0, 41.5])
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert "Not every scale is exact: 1 of 10 resamples" in text
    (mine,) = [line for line in text.splitlines() if line.startswith(f"{NAME} | {UNIT} | ")][1:]
    assert "offer 7.500, outer bound [0.450, 41.500] kWh per counting test day" in mine
    assert "lower end between 0.450 and 0.450, upper end between 37.000 and 41.500" in mine
    assert "4 of 5 scales exact, 1 of 5 resamples met a failed evaluation" in mine


REVIEW_CALIBRATION = [
    list(range(22)),
    [2, 3, 4, 4, 4, *range(5, 22)],
    [0, 0, 0, *range(3, 22)],
    [*range(19), 21, 21, 21],
]
REVIEW_TEST = [[0, 1, 2, 3, 4, 5], [0, 0, 2, 3, 4, 4], [1, 1, 1, 1, 4, 5], [5] * 6]


def review_job():
    thresholds = [0.2, 0.4] + [99.0] * 19
    drivers = [None, None, (0.08, 0.3), (0.08, 0.25)] + [None] * 17

    def loss_at(beta, found):
        found.append(beta)
        slot = np.array([1.0 * (beta > t) for t in thresholds])
        driver = np.array([0.0 if d is None else d[0] * (beta > d[1]) for d in drivers])
        return slot, driver

    scales, chosen = set(), {}
    for search, alphas in (("A", tier_alphas("A")), ("B", tier_alphas("B")), ("D", (1.0, 0.05))):
        found = []
        chosen[search] = crc.calibrate_search(lambda b, f=found: loss_at(b, f), 21, *alphas).beta
        scales.update(found)
    assert (chosen["A"], chosen["B"]) == (0.3, 0.4)
    assert max(scales) == 0.5 and {0.05, 0.1, 0.2, 0.3, 0.4} < scales
    assert any(0.3 < b < 0.4 for b in scales) and any(0.4 < b < 0.5 for b in scales)
    calibration = {"thresholds": thresholds, "offers": [True] * 21 + [False], "driver": drivers}
    test = {
        "values": [12.0, 0.0, 6.0, 18.0, 24.0, 30.0],
        "cars": [3, 2, 3, 4, 4, 5],
        "offers": [True, False, True, True, True, True],
        "ceilings": {CEILINGS[0]: [40.0] * 6, CEILINGS[1]: [30.0] * 6},
    }
    tiers = {
        "A": (0.3, [0, 0, 0.5, 0, 1, 0], [0.01, 0, 0, 0.03, 0, 0.02]),
        "B": (0.4, [0, 0, 1, 0, 1, 0.5], [0.02, 0, 0.01, 0.05, 0, 0.02]),
    }
    return unit("H", calibration, test, scales, tiers, resample=(SEED, 4))


def test_the_hand_case_of_the_review_is_exact(tmp_path, monkeypatch):
    inject(monkeypatch, REVIEW_CALIBRATION, REVIEW_TEST)
    record, entry = review_job()
    job = day_bootstrap.Job.from_record(record, [RUNG])
    for tier, expected in (("A", [0.3, 0.3, 0.2, 0.25]), ("B", [0.4, 2.0, 0.2, 0.4])):
        found = [job.scale(RUNG, tier, row) for row in REVIEW_CALIBRATION]
        assert [(s.low, s.high, s.exact) for s in found] == [(b, b, True) for b in expected]
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [(record, entry)])
    assert run(folder, tmp_path / "out", monkeypatch, "--seed", str(SEED), "--resamples", "4") == 0
    found = results(tmp_path / "out")["configurations"][NAME]["H"][RUNG]
    assert found["A"]["offer_kwh_per_counting_test_day"]["interval"] == pytest.approx(
        [2.0025, 7.2975]
    )
    assert found["A"]["scale"]["interval"] == pytest.approx([0.20375, 0.3])
    assert found["B"]["offer_kwh_per_counting_test_day"]["interval"] == pytest.approx([2.115, 30.5])
    assert found["B"]["scale"]["interval"] == pytest.approx([0.215, 1.88])


def identical_job():
    thresholds = [0.5] * 20
    scales = {*crc.GRID, *asked(thresholds, "A"), *asked(thresholds, "B")}
    calibration = {"thresholds": thresholds, "offers": [True] * 20}
    test = {
        "values": [10.0] * 10,
        "cars": [4] * 10,
        "offers": [True] * 10,
        "ceilings": {CEILINGS[0]: [100.0] * 10, CEILINGS[1]: [80.0] * 10},
    }
    losses = ([0.0] * 10, [0.01] * 10)
    tiers = {"A": (0.5, *losses), "B": (0.5, *losses)}
    return unit(UNIT, calibration, test, scales, tiers, resample=(1, 2000))


def _intervals(found: dict):
    """Every interval of one rung and tier."""
    for key, value in found.items():
        if key == "share_of_ceiling":
            for share in value.values():
                yield share["interval"]
        elif key not in ("abstaining",):
            yield value["interval"]


def test_identical_days_give_intervals_of_zero_width(tmp_path, monkeypatch):
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [identical_job()])
    assert run(folder, tmp_path / "out", monkeypatch, "--seed", "1") == 0
    found = results(tmp_path / "out")["configurations"][NAME][UNIT][RUNG]
    for tier in ("A", "B"):
        assert found[tier]["scale"]["resamples"] == 2000
        assert found[tier]["scale"]["resamples_exact"] == 2000
        assert found[tier]["offer_kwh_per_counting_test_day"]["interval"] == [5.0, 5.0]
        for interval in _intervals(found[tier]):
            assert interval[0] == interval[1]


def varied_job(seed: int):
    rng = np.random.default_rng(11)
    thresholds = [float(t) for t in rng.choice([0.1, 0.25, 0.4, 0.55, 0.7, 2.0], size=30)]
    scales = {*crc.GRID, *asked(thresholds, "A"), *asked(thresholds, "B")}
    test = {
        "values": [float(v) for v in rng.uniform(0, 30, size=20)],
        "cars": [int(c) for c in rng.integers(3, 9, size=20)],
        "offers": [True] * 20,
        "ceilings": {CEILINGS[0]: [100.0] * 20, CEILINGS[1]: [80.0] * 20},
    }
    calibration = {"thresholds": thresholds, "offers": [True] * 30}

    def scale_of(tier):
        losses = lambda b: (np.array([1.0 * (b > t) for t in thresholds]), np.zeros(30))  # noqa: E731
        return crc.calibrate_search(losses, 30, *tier_alphas(tier)).beta

    slot = [float(s) for s in rng.choice([0.0, 1.0], size=20, p=[0.9, 0.1])]
    driver = [float(d) for d in rng.uniform(0, 0.05, size=20)]
    tiers = {"A": (scale_of("A"), slot, driver), "B": (scale_of("B"), slot, driver)}
    return unit(UNIT, calibration, test, scales, tiers, resample=(seed, 200))


def test_a_fixed_seed_repeats_and_another_seed_differs(tmp_path, monkeypatch):
    three = write_run(tmp_path / "20200101T000000Z-seed-3", [varied_job(3)])
    four = write_run(tmp_path / "20200101T000000Z-seed-4", [varied_job(4)])
    texts = []
    for k, (folder, seed) in enumerate(((three, "3"), (three, "3"), (four, "4"))):
        out = tmp_path / f"out{k}"
        assert run(folder, out, monkeypatch, "--seed", seed, "--resamples", "200") == 0
        texts.append((out / day_intervals.RESULTS_NAME).read_text())
    assert texts[0] == texts[1]
    first = json.loads(texts[0])["configurations"]
    other = json.loads(texts[2])["configurations"]
    assert first != other
    offer = first[NAME][UNIT][RUNG]["B"]["offer_kwh_per_counting_test_day"]["interval"]
    assert offer[0] < offer[1]
    assert first[NAME][UNIT][RUNG]["B"]["scale"]["resamples_exact"] == 200
    assert run(three, tmp_path / "out-other", monkeypatch, "--seed", "4", "--resamples", "200") == 2


def test_draws_depend_on_the_seed_and_the_unit():
    first = day_bootstrap.draws(5, "u", 10, 6, 50)
    again = day_bootstrap.draws(5, "u", 10, 6, 50)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not np.array_equal(first[0], day_bootstrap.draws(6, "u", 10, 6, 50)[0])
    assert not np.array_equal(first[0], day_bootstrap.draws(5, "v", 10, 6, 50)[0])
    assert first[0].shape == (50, 10) and first[1].shape == (50, 6)
    assert (np.diff(first[0], axis=1) >= 0).all() and first[0].max() < 10


FAULTS = [
    "exit-1",
    "changed",
    "stale",
    "other-command",
    "no-results",
    "both-results",
    "no-pairs",
    "other-seed",
    "other-count",
]


@pytest.mark.parametrize("fault", FAULTS)
def test_runs_that_do_not_verify_are_refused(tmp_path, monkeypatch, fault):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    job = hand_job()
    if fault == "no-pairs":
        del job[0]["resampled"]
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [job])
    arguments = list(HAND)
    if fault == "exit-1":
        write_manifest(folder, "fleetreserve.gate_run", exit_code=1)
    elif fault == "changed":
        path = folder / "gate.json"
        path.write_text(path.read_text() + " ")
    elif fault == "stale":
        write_manifest(folder, "fleetreserve.gate_run", commit="d" * 40)
    elif fault == "other-command":
        write_manifest(folder, "fleetreserve.compare_run")
    elif fault == "no-results":
        (folder / "gate.json").unlink()
        write_manifest(folder, "fleetreserve.gate_run")
    elif fault == "both-results":
        (folder / "learn-gate.json").write_text((folder / "gate.json").read_text())
        write_manifest(folder, "fleetreserve.gate_run")
    elif fault == "other-seed":
        arguments[1] = str(SEED + 1)
    elif fault == "other-count":
        monkeypatch.setattr(day_bootstrap, "draws", lambda *a: pytest.fail("no draws"))
        arguments[3] = "6"
    out = tmp_path / "out"
    assert run(folder, out, monkeypatch, *arguments) == 2
    assert list(out.iterdir()) == []


def test_bad_seeds_are_refused(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [hand_job()])
    assert run(folder, tmp_path / "out", monkeypatch, "--seed", "-1", "--resamples", "5") == 2
    with pytest.raises(SystemExit) as stopped:
        day_intervals.main(["--run", str(folder)])
    assert stopped.value.code == 2


def test_a_file_that_disagrees_with_the_run_fails(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    record, entry = hand_job()
    record["calibration"]["slot_loss"][RUNG][0] = [0.0] * len(record["calibration"]["scales"])
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [(record, entry)])
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 1
    found = results(tmp_path / "out")
    assert found["configurations"][NAME][UNIT] is None
    assert any("scale recomputed 0.5" in p for p in found["problems"])
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert "FAILED" in text


def test_a_learn_run_is_read_the_same_way(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION_DRAWS, TEST_DRAWS)
    folder = write_run(
        tmp_path / "20200101T000000Z-made-up", [hand_job()], module="fleetreserve.learn_run"
    )
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    found = results(tmp_path / "out")
    assert found["run"]["command"] == "fleetreserve.learn_run"
    offer = found["configurations"][NAME][UNIT][RUNG]["B"]["offer_kwh_per_counting_test_day"]
    assert offer["interval"] == pytest.approx([0.45, 37.0])

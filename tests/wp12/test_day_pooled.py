"""The pooled figures of the intervals over days (`day_intervals` readings I13 to I17) and the
route record and earlier-commit note of its files, on made-up recorded runs (`interval_fake`).

1. Two units worked by hand, five resamples injected in place of the seeded draws (each unit has
   its own list, as the unit's own draw would be). Tier B, scale 0.3 in the run. Both units have
   the calibration days of `test_day_intervals`' hand unit (ten days, c9 not offering), so their
   scales in resamples 1 to 4 are that unit's: 0.3, 0.5, 0.3, 0.8. Resample 5 draws c0 to c8 and
   c8 again for the second unit (10 offering days, no day loses below 0.3: scale 0.3) and c0 to
   c7 with c9 (not offering) twice for the first (8 offering days: too few, abstains). First
   unit: 6 test days, values 10, 20, 30, 0, 40, 50, cars 3, 4, 5, 3, 4, 5, ceiling 100 each.
   Second unit: 4 test days, values 20, 40, 60, 0, cars 2, 3, 4, 5, ceiling 50 each. Test
   positions of the second unit: (0,1,2,3), (0,0,1,1), (1,1,3,3), (2,2,2,2), (0,1,2,3); the first
   unit's are those of the hand unit.

   Offered kWh over the drawn test days (scale x sum of values), per unit and resample:
     first   0.3 x 150 = 45;  0.5 x 120 = 60;  0.3 x 90 = 27;  0.8 x 300 = 240;  0
     second  0.3 x 120 = 36;  0.5 x 120 = 60;  0.3 x 80 = 24;  0.8 x 240 = 192;  0.3 x 120 = 36
   Cars over the drawn test days: first 24, 24, 21, 30, 26; second 14, 10, 16, 16, 14.
   (a) the sum of the units' offers per counting test day (kWh / 6 and kWh / 4):
        r1 7.5 + 9 = 16.5;  r2 10 + 15 = 25;  r3 4.5 + 6 = 10.5;  r4 40 + 48 = 88;  r5 0 + 9 = 9
       sorted 9, 10.5, 16.5, 25, 88; the numpy "linear" 2.5% and 97.5% percentiles of five values
       lie at positions 0.1 and 3.9: [9 + 0.1 x 1.5, 25 + 0.9 x 63] = [9.15, 81.7].
       The run's own: 0.3 x 150 / 6 + 0.3 x 120 / 4 = 7.5 + 9 = 16.5.
   (b) the total over the 10 counting test days of both units:
        r1 81 / 10 = 8.1;  r2 12;  r3 5.1;  r4 43.2;  r5 3.6
       sorted 3.6, 5.1, 8.1, 12, 43.2: [3.6 + 0.1 x 1.5, 12 + 0.9 x 31.2] = [3.75, 40.08].
       The run's own: 81 / 10 = 8.1. With 2 units, (a) = 2 x (b) would hold if both had as many
       counting test days; here 16.5 is not 2 x 8.1 = 16.2.
   Per car, (b): the total over the total cars, 81 / 38, 120 / 34, 51 / 37, 432 / 46, 36 / 40;
       (a): the sum of the offers per day over the sum of the cars per day (24 / 6 + 14 / 4 = 7.5
       in r1): 16.5 / 7.5, 25 / 6.5, 10.5 / 7.5, 88 / 9, 9 / (26 / 6 + 14 / 4).
   Share of the first ceiling (100 a day in the first unit, 50 in the second): (b) over 600 + 200
       = 800 kWh, (a) over 100 + 50 = 150 kWh a day.
   The abstaining first unit contributes 0 in resample 5, and tier A (19 offering days needed)
   abstains in every resample of both units: its pooled offers are 0 with interval [0, 0].
2. The first unit's pair (c3, 0.9) failing in the run: resample 4 of the first unit lies between
   scales 0.8 and 0.9 (offer 40 to 45), so the pooled (a) of resample 4 lies between 88 and 93:
   lower percentiles over the lower bounds (9.15 and 81.7), upper over the upper bounds (sorted 9,
   10.5, 16.5, 25, 93: 9.15 and 25 + 0.9 x 68 = 86.2): interval [9.15, 86.2], lower end between
   9.15 and 9.15, upper end between 81.7 and 86.2, an outer bound in four of five exact resamples.
3. Two units with every day identical (twenty offering calibration days at threshold 0.5, ten
   test days of value 10): each unit's offer is 5, so (a) is 10 and (b) is 5 in every resample:
   zero-width intervals; with as many counting test days in both, (a) is 2 x (b).
4. The route record: a made-up run through the sealed route (`sealed_route.mark`, the writer of
   the record) read by `day_intervals` gives `day-intervals.json` the same `sealed_route` record,
   and `run_evidence.read_evidence` refuses that file from an earlier commit of a made-up
   repository (reading E3) where it reads the file of a development run from the same commit; a
   development run's intervals file has no such key. A sealed run recorded at an earlier commit
   could not run (2).
5. The summary names the earlier commit after the run's name when the run read was recorded at an
   earlier commit and accepted (only paths of the keep-list differ), and not otherwise.
6. A unit of the run that holds no per-day record (one the run could not split) is named in the
   pooled block, with the run's reason, and on the summary's pooled line; no pooled number
   changes; with no such unit the list is empty and the line says so. The route record is copied
   whatever its type.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from interval_fake import COMMIT, NAME, RUNG, asked, unit, write_manifest, write_run

from fleetreserve import crc, day_bootstrap, day_intervals, run_evidence, run_verify, sealed_route
from fleetreserve.gate_config import CEILINGS
from fleetreserve.gate_summary import ROUTE_KEY

_WP2 = Path(__file__).resolve().parents[1] / "wp2"  # the made-up repositories
if str(_WP2) not in sys.path:
    sys.path.insert(0, str(_WP2))
from fake_heldout import commit, git  # noqa: E402

FIRST, SECOND = "first-unit", "second-unit"
SEED = 7
THRESHOLDS = [0.3, 0.5, 0.5, 0.8, 2.0, 2.0, 2.0, 2.0, 2.0]
LAST = [0, 1, 2, 3, 4, 5, 6, 7, 9, 9]
CALIBRATION = {
    FIRST: [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 8, 8, 8, 8],
        LAST,
    ],
    SECOND: [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 0, 3, 4, 5, 6, 7, 8, 9],
        [3, 4, 5, 6, 7, 8, 8, 8, 8, 8],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 8],
    ],
}
TEST_DRAWS = {
    FIRST: [
        [0, 1, 2, 3, 4, 5],
        [0, 0, 1, 1, 2, 2],
        [3, 3, 3, 3, 4, 5],
        [5] * 6,
        [1, 2, 3, 4, 5, 5],
    ],
    SECOND: [[0, 1, 2, 3], [0, 0, 1, 1], [1, 1, 3, 3], [2, 2, 2, 2], [0, 1, 2, 3]],
}
TESTS = {
    FIRST: {
        "values": [10.0, 20.0, 30.0, 0.0, 40.0, 50.0],
        "cars": [3, 4, 5, 3, 4, 5],
        "offers": [True, True, True, False, True, True],
        "ceilings": {CEILINGS[0]: [100.0] * 6, CEILINGS[1]: [80.0] * 6},
    },
    SECOND: {
        "values": [20.0, 40.0, 60.0, 0.0],
        "cars": [2, 3, 4, 5],
        "offers": [True, True, True, False],
        "ceilings": {CEILINGS[0]: [50.0] * 4, CEILINGS[1]: [40.0] * 4},
    },
}
LOSSES = {
    FIRST: ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.02, 0.1, 0.0, 0.0, 0.04, 0.0]),
    SECOND: ([0.0, 0.0, 0.0, 0.0], [0.01, 0.0, 0.02, 0.0]),
}
HAND = ("--seed", str(SEED), "--resamples", "5")


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: COMMIT)


@pytest.fixture(autouse=True)
def clean_git_environment(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def inject(monkeypatch, calibration, test):
    """The draws of each unit replaced by its own list (by the unit's name)."""

    def draws(seed, name, n_calibration, n_test, resamples):
        found = (len(calibration[name][0]), len(test[name][0]), len(calibration[name]))
        assert (n_calibration, n_test, resamples) == found
        return np.array(calibration[name]), np.array(test[name])

    monkeypatch.setattr(day_bootstrap, "draws", draws)


def hand_unit(name, fails=()):
    second = [0.5, 0.5, 0.5, 0.8, 2.0, 2.0, 2.0, 2.0, 2.0]  # the second resample's offering days
    scales = {*crc.GRID, *asked(THRESHOLDS, "B"), *asked(second, "B")}
    calibration = {"thresholds": THRESHOLDS, "offers": [True] * 9 + [False]}
    tiers = {"A": (0.0, None, None), "B": (0.3, *LOSSES[name])}
    return unit(name, calibration, TESTS[name], scales, tiers, resample=(SEED, 5), fails=fails)


def hand_run(folder, fails=(), **options):
    return write_run(folder, [hand_unit(FIRST, fails), hand_unit(SECOND)], **options)


def run(folder, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return day_intervals.main(["--run", str(folder), *extra])


def results(out) -> dict:
    return json.loads((out / day_intervals.RESULTS_NAME).read_text())


def pooled(out, tier="B") -> dict:
    block = results(out)["pooled"]["configurations"][NAME]
    return block["rungs"][RUNG][tier]


def test_the_pooled_figures_agree_with_the_hand_case(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up")
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    found = results(tmp_path / "out")
    block = found["pooled"]["configurations"][NAME]
    assert block["units"] == [FIRST, SECOND]
    assert block["counting_test_days"] == {FIRST: 6, SECOND: 4}
    b = pooled(tmp_path / "out")
    assert b["resamples"] == 5 and b["resamples_every_scale_exact"] == 5
    key = "offer_kwh_per_counting_test_day"
    a_offer, b_offer = b["sum_of_unit_offers"][key], b["total_over_total_days"][key]
    assert a_offer["point"] == pytest.approx(16.5)
    assert a_offer["interval"] == pytest.approx([9.15, 81.7])
    assert b_offer["point"] == pytest.approx(8.1)
    assert b_offer["interval"] == pytest.approx([3.75, 40.08])
    assert a_offer["point"] != pytest.approx(2 * b_offer["point"])  # 16.5 against 16.2
    assert a_offer["lower_end_between"] == pytest.approx([9.15, 9.15])
    car = "offer_kwh_per_car_per_day"
    a_car, b_car = b["sum_of_unit_offers"][car], b["total_over_total_days"][car]
    assert b_car["point"] == pytest.approx(81 / 38)
    five = 36 / 40
    assert b_car["interval"] == pytest.approx(
        [five + 0.1 * (51 / 37 - five), 120 / 34 + 0.9 * (432 / 46 - 120 / 34)]
    )
    assert a_car["point"] == pytest.approx(16.5 / 7.5)
    five = 9 / (26 / 6 + 14 / 4)
    assert a_car["interval"] == pytest.approx(
        [five + 0.1 * (1.4 - five), 25 / 6.5 + 0.9 * (88 / 9 - 25 / 6.5)]
    )
    share = "share_of_ceiling"
    a_share = b["sum_of_unit_offers"][share][CEILINGS[0]]
    b_share = b["total_over_total_days"][share][CEILINGS[0]]
    assert b_share["point"] == pytest.approx(81 / 800)
    assert b_share["interval"] == pytest.approx([37.5 / 800, 400.8 / 800])
    assert a_share["point"] == pytest.approx(16.5 / 150)
    assert a_share["interval"] == pytest.approx([9.15 / 150, 81.7 / 150])
    other = b["total_over_total_days"][share][CEILINGS[1]]
    assert other["point"] == pytest.approx(81 / (480 + 160))
    tier_a = pooled(tmp_path / "out", "A")
    for figure in ("sum_of_unit_offers", "total_over_total_days"):
        assert tier_a[figure][key]["interval"] == [0.0, 0.0]
        assert tier_a[figure][key]["point"] == 0.0
    # the per-unit blocks are the units' own, as before
    mine = found["configurations"][NAME][FIRST][RUNG]["B"]["offer_kwh_per_counting_test_day"]
    assert mine["interval"] == pytest.approx([0.45, 37.0])
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    (line,) = [t for t in text if t.startswith(f"{NAME} | pooled over 2 units | {RUNG} | ")][1:]
    assert "offer (a) 16.500 [9.150, 81.700] kWh, (b) 8.100 [3.750, 40.080] kWh" in line
    assert "per counting test day; per car (a) 2.200" in line
    assert "every unit's scale exact in every resample (5 of 5)" in line


def test_a_failed_pair_of_one_unit_makes_the_pooled_interval_an_outer_bound(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up", fails=[(3, 0.9)])
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    b = pooled(tmp_path / "out")
    assert b["resamples_every_scale_exact"] == 4
    offer = b["sum_of_unit_offers"]["offer_kwh_per_counting_test_day"]
    assert offer["interval"] == pytest.approx([9.15, 86.2])
    assert offer["lower_end_between"] == pytest.approx([9.15, 9.15])
    assert offer["upper_end_between"] == pytest.approx([81.7, 86.2])
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert "outer bounds: every unit's scale exact in 4 of 5 resamples" in text


def identical_unit(name):
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
    return unit(name, calibration, test, scales, tiers, resample=(1, 2000))


def test_units_with_identical_days_give_pooled_intervals_of_zero_width(tmp_path, monkeypatch):
    folder = write_run(
        tmp_path / "20200101T000000Z-made-up", [identical_unit("one"), identical_unit("two")]
    )
    assert run(folder, tmp_path / "out", monkeypatch, "--seed", "1") == 0
    for tier in ("A", "B"):
        found = pooled(tmp_path / "out", tier)
        assert found["resamples"] == found["resamples_every_scale_exact"] == 2000
        a = found["sum_of_unit_offers"]["offer_kwh_per_counting_test_day"]
        b = found["total_over_total_days"]["offer_kwh_per_counting_test_day"]
        assert a["interval"] == [10.0, 10.0] and a["point"] == 10.0
        assert b["interval"] == [5.0, 5.0] and b["point"] == 5.0
        for figure in ("sum_of_unit_offers", "total_over_total_days"):
            for key in ("offer_kwh_per_car_per_day",):
                low, high = found[figure][key]["interval"]
                assert low == high
            for share in found[figure]["share_of_ceiling"].values():
                assert share["interval"][0] == share["interval"][1]


def test_a_unit_without_an_interval_leaves_no_pooled_figure(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    record, entry = hand_unit(SECOND)
    record["calibration"]["slot_loss"][RUNG][0] = [0.0] * len(record["calibration"]["scales"])
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [hand_unit(FIRST), (record, entry)])
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 1
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert "rungs" not in block and SECOND in block["reason"]
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert f"{NAME} | pooled: no pooled figure" in text


# --- the route's record and the earlier-commit note --------------------------------------------


def sealed_record() -> dict:
    document = {}
    route = sealed_route.Route(True, tag_commit="a" * 40)
    sealed_route.mark(document, route, {"truth_run": "made-up"}, fit_units=("caltech",))
    return document[ROUTE_KEY]


def made_up_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    (root / ".gitignore").write_text("runs.noindex/\n")
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("Notes.\n")
    return root, commit(root, "first")


def recorded(root, first, label, *, sealed):
    """A learn run of the hand units recorded at `first` in the made-up repository."""
    inject_target = root / "runs.noindex" / f"20200101T00000{label}Z-made-up"
    extra = {ROUTE_KEY: sealed_record()} if sealed else None
    folder = hand_run(inject_target, module="fleetreserve.learn_run", extra=extra)
    write_manifest(folder, "fleetreserve.learn_run", commit=first)
    return folder


def intervals_of(root, folder, monkeypatch, head):
    """`day_intervals` on `folder` with `head` as the checked-out commit; the exit status and the
    run folder its files went to."""
    monkeypatch.setattr(day_intervals, "head_commit", lambda top: head)
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    out = root / "runs.noindex" / "20200101T000009Z-intervals"
    out.mkdir(exist_ok=True)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return day_intervals.main(["--run", str(folder), *HAND]), out


def test_the_sealed_route_record_is_copied_and_the_file_is_refused_from_an_earlier_commit(
    tmp_path, monkeypatch
):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    root, first = made_up_repository(tmp_path)
    sealed = recorded(root, first, "1", sealed=True)
    status, out = intervals_of(root, sealed, monkeypatch, first)
    assert status == 0
    found = results(out)
    assert found[ROUTE_KEY] == sealed_record()
    assert found[ROUTE_KEY]["kind"] == sealed_route.HELD_OUT
    write_manifest(out, "fleetreserve.day_intervals", commit=first)
    assert run_evidence.read_evidence(out, day_intervals.RESULTS_NAME, first).data[ROUTE_KEY]
    (root / "docs" / "notes.md").write_text("Changed notes.\n")
    second = commit(root, "a change on the keep-list")
    with pytest.raises(run_evidence.CannotCheck) as refused:
        run_evidence.read_evidence(out, day_intervals.RESULTS_NAME, second)
    assert "(reading E3)" in str(refused.value) and ROUTE_KEY in str(refused.value)


def test_a_development_run_has_no_route_record_and_is_read_from_an_earlier_commit(
    tmp_path, monkeypatch
):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    root, first = made_up_repository(tmp_path)
    development = recorded(root, first, "2", sealed=False)
    status, out = intervals_of(root, development, monkeypatch, first)
    assert status == 0
    assert ROUTE_KEY not in results(out)
    assert "accepted_from_earlier_commit" not in results(out)["run"]
    assert "accepted from the earlier commit" not in (out / day_intervals.SUMMARY_NAME).read_text()
    write_manifest(out, "fleetreserve.day_intervals", commit=first)
    (root / "docs" / "notes.md").write_text("Changed notes.\n")
    second = commit(root, "a change on the keep-list")
    kept = run_evidence.read_evidence(out, day_intervals.RESULTS_NAME, second)
    assert kept.origin == run_evidence.FROM_EARLIER
    again = root / "runs.noindex" / "20200101T000009Z-intervals"
    for name in again.iterdir():
        name.unlink()
    status, out = intervals_of(root, development, monkeypatch, second)
    assert status == 0
    note = run_evidence.earlier_note(first)
    assert results(out)["run"]["accepted_from_earlier_commit"] == note
    summary = (out / day_intervals.SUMMARY_NAME).read_text().splitlines()
    (head,) = [line for line in summary if line.startswith("95% intervals over days")]
    assert f"of the recorded run 20200101T000002Z-made-up{note} (fleetreserve.learn_run," in head


def test_a_sealed_run_from_an_earlier_commit_could_not_run(tmp_path, monkeypatch, capsys):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    root, first = made_up_repository(tmp_path)
    sealed = recorded(root, first, "3", sealed=True)
    (root / "docs" / "notes.md").write_text("Changed notes.\n")
    second = commit(root, "a change on the keep-list")
    status, out = intervals_of(root, sealed, monkeypatch, second)
    assert status == 2
    assert "(reading E3)" in capsys.readouterr().out
    assert list(out.iterdir()) == []


# --- the units a pooled block leaves out ---------------------------------------------------------

THIRD = "third-unit"
UNSPLIT = "not calibratable: 5 days"


def with_third_unit(folder, entry=None):
    """The hand run's gate results with a third unit that the run could not split (no per-day
    record), the manifest written again."""
    path = folder / "gate.json"
    document = json.loads(path.read_text())
    document["configurations"][NAME]["units"][THIRD] = entry or {"reason": UNSPLIT}
    path.write_text(json.dumps(document))
    write_manifest(folder, "fleetreserve.gate_run")


def test_a_unit_the_run_could_not_split_is_named_in_the_pooled_block(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up")
    with_third_unit(folder)
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert block["units"] == [FIRST, SECOND]
    assert block["left_out"] == [{"unit": THIRD, "reason": UNSPLIT}]
    b = pooled(tmp_path / "out")  # no pooled number changes
    offer = b["sum_of_unit_offers"]["offer_kwh_per_counting_test_day"]
    assert offer["point"] == pytest.approx(16.5)
    assert offer["interval"] == pytest.approx([9.15, 81.7])
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    (line,) = [t for t in text if t.startswith(f"{NAME} | pooled over 2 units | {RUNG} | ")][1:]
    assert line.endswith(f"; units left out of the pooling: {THIRD} ({UNSPLIT})")


def test_with_no_unit_left_out_the_list_is_empty_and_the_summary_says_none(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up")
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert block["left_out"] == []
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    (line,) = [t for t in text if t.startswith(f"{NAME} | pooled over 2 units | {RUNG} | ")][1:]
    assert line.endswith("; units left out of the pooling: none")


def test_a_failed_unit_of_the_run_is_left_out_with_the_runs_reason(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up")
    failed = {"failed": "ValueError: made up", "reason": "configuration failed"}
    with_third_unit(folder, failed)
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert block["left_out"] == [{"unit": THIRD, "reason": "configuration failed"}]


@pytest.mark.parametrize(
    "entry",
    [
        {"failed": "failed: ValueError: made up", "reason": "failed: ValueError: made up"},
        {"failed": "ValueError: made up", "reason": "configuration failed"},
    ],
)
def test_the_pooled_line_counts_a_unit_whose_configuration_failed_among_the_units(
    tmp_path, monkeypatch, entry
):
    """Two units pooled and a third whose configuration failed: "2 of 3 units", never "2 units"."""
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up")
    with_third_unit(folder, entry)
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text().splitlines()
    lines = [t for t in text if t.startswith(f"{NAME} | pooled over ") and f"| {RUNG} |" in t]
    assert len(lines) == 2
    for line in lines:
        assert line.startswith(f"{NAME} | pooled over 2 of 3 units | {RUNG} | ")
        assert f"units left out of the pooling: {THIRD} ({entry['reason']})" in line


def test_the_units_left_out_are_named_when_a_unit_has_no_interval_too(tmp_path, monkeypatch):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    record, entry = hand_unit(SECOND)
    record["calibration"]["slot_loss"][RUNG][0] = [0.0] * len(record["calibration"]["scales"])
    folder = write_run(tmp_path / "20200101T000000Z-made-up", [hand_unit(FIRST), (record, entry)])
    with_third_unit(folder)
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 1
    block = results(tmp_path / "out")["pooled"]["configurations"][NAME]
    assert "rungs" not in block and SECOND in block["reason"]
    assert block["left_out"] == [{"unit": THIRD, "reason": UNSPLIT}]
    text = (tmp_path / "out" / day_intervals.SUMMARY_NAME).read_text()
    assert f"; units left out of the pooling: {THIRD} ({UNSPLIT})" in text


@pytest.mark.parametrize("record", ["a made-up record", ["a", "list"], {"kind": "a dict"}])
def test_the_route_record_is_copied_whatever_its_type(tmp_path, monkeypatch, record):
    inject(monkeypatch, CALIBRATION, TEST_DRAWS)
    folder = hand_run(tmp_path / "20200101T000000Z-made-up", extra={ROUTE_KEY: record})
    assert run(folder, tmp_path / "out", monkeypatch, *HAND) == 0
    assert results(tmp_path / "out")[ROUTE_KEY] == record

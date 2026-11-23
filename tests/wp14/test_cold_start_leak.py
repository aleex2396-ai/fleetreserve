"""No day after the first k reaches the model's fit, its forecasts or the calibration at k
(`cold_start.run`, `cold_start_days.frame_at`).

A made-up development unit of 30 counting days (1 to 30 January 2020, three cars a day, each car j
keeping one user id) is run at k = 0, 10 and 20 (scored days: the ten after the first 20), with a
stand-in model whose forecasts follow the smallest earlier departure of the user's history
(`cold_start_fake.HistoryModel`). Two copies differ from it in one planted value each:

- on 16 January (day 15, after the first 10, before the first 20) the first car leaves 0.4 hours
  after τ instead of 4.41. At k = 0 and 10 every forecast table, every figure and every day row is
  the same as without the plant; at k = 20, where that day is a local day, the forecast tables
  differ, so the planted value moves the output once it may be seen;
- on 25 January (day 24, a scored day) the third car leaves 0.6 hours after τ instead of 6.73. At
  every k the forecast tables, the calibration (scale, offering local days and the reason for a
  scale of 0, per tier) and the rows of the local days are the same; the day's own row of the
  days file differs, so the planted value reaches the scoring of that day and nothing else.

The model is fitted once, on the training unit's rows only, and the same in all three runs; every
frame it is given at k holds only the k local days and the scored days.
"""

import pandas as pd
import pytest
from cold_start_fake import FIT_UNIT, HistoryModel, cold_cars, write_build
from gate_fake import matrix_document

from fleetreserve import cold_start, compare_run, gate_config
from fleetreserve.days_from_truth import load_unit_days

UNIT = "nrel-from-may-2019"
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
LOCAL_DAYS = (0, 10, 20)
SEEN_LATE = "A-15-0"  # 16 January
SCORED_DAY = "A-24-2"  # 25 January


def _cars(planted: dict) -> list:
    cars = cold_cars(30)
    for car in cars:
        if car.session in planted:
            car.plug_out = planted[car.session]
    return cars


def _run(folder, planted: dict):
    folder.mkdir()
    _, tables = write_build(folder, {UNIT: _cars(planted), FIT_UNIT: cold_cars(5, "T")})
    units = {
        name: load_unit_days(tables, name, rho=0.0, no_fit="cap", small_fleet="keep")
        for name in (UNIT, FIT_UNIT)
    }
    matrix = gate_config.parse(matrix_document(NAMES))
    configuration = compare_run.dial_configurations(matrix, 0.02)["forecast"]
    model = HistoryModel("stand_in", 0.0)
    checked = cold_start.Checked("stand_in", model, (FIT_UNIT,), None)
    found = cold_start.run(
        matrix,
        configuration,
        {UNIT: units[UNIT]},
        [checked],
        {FIT_UNIT: units[FIT_UNIT].cars},
        workers=1,
        local_days=LOCAL_DAYS,
    )
    return found, model


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    top = tmp_path_factory.mktemp("leak")
    return {
        "plain": _run(top / "plain", {}),
        "seen late": _run(top / "late", {SEEN_LATE: 10.4}),
        "scored": _run(top / "scored", {SCORED_DAY: 10.6}),
    }


def _rows(found, k, **where) -> list[dict]:
    rows = [r for r in found[1] if r["k"] == k]
    return [r for r in rows if all(r[key] == value for key, value in where.items())]


def _calibration(found, k) -> dict:
    by_tier = found[0]["figures"]["stand_in"][UNIT][str(k)]
    keys = ("scale", "scale_zero_reason", "offering_local_days", "local_days")
    return {tier: {key: f[key] for key in keys} for tier, f in by_tier.items()}


def test_the_runs_scored_every_k(runs):
    for found, _ in runs.values():
        assert found[0]["failed"] == []
        assert set(found[0]["figures"]["stand_in"][UNIT]) == {"0", "10", "20"}


def test_a_day_after_k_and_before_the_scored_days_moves_nothing_at_k(runs):
    plain, _ = runs["plain"]
    late, _ = runs["seen late"]
    for k in (0, 10):
        pd.testing.assert_frame_equal(plain[2][("stand_in", k)], late[2][("stand_in", k)])
        assert (
            plain[0]["figures"]["stand_in"][UNIT][str(k)]
            == (late[0]["figures"]["stand_in"][UNIT][str(k)])
        )
        assert _rows(plain, k) == _rows(late, k)
    # positive control: at k = 20 the day is local, and the planted value is seen
    assert not plain[2][("stand_in", 20)].equals(late[2][("stand_in", 20)])


def test_a_scored_day_moves_no_forecast_and_no_calibration(runs):
    plain, _ = runs["plain"]
    scored, _ = runs["scored"]
    for k in LOCAL_DAYS:
        pd.testing.assert_frame_equal(plain[2][("stand_in", k)], scored[2][("stand_in", k)])
        assert _calibration(plain, k) == _calibration(scored, k)
        assert _rows(plain, k, role="local") == _rows(scored, k, role="local")
        others = [r for r in _rows(plain, k, role="scored") if r["date"] != "2020-01-25"]
        assert others == [r for r in _rows(scored, k, role="scored") if r["date"] != "2020-01-25"]
    # positive control: the planted value reaches the scoring of its own day
    day = [_rows(f, 20, role="scored", date="2020-01-25") for f in (plain, scored)]
    assert day[0] != day[1]


def test_the_fit_is_on_the_training_unit_alone_and_the_same_in_every_run(runs):
    fitted = [model.fitted for _, model in runs.values()]
    for frames in fitted:
        assert len(frames) == 1
        assert set(frames[0]["unit"]) == {FIT_UNIT}
    pd.testing.assert_frame_equal(fitted[0][0], fitted[1][0])
    pd.testing.assert_frame_equal(fitted[0][0], fitted[2][0])


def test_every_frame_holds_the_local_days_and_the_scored_days_only(runs):
    dates = [f"2020-01-{d:02d}" for d in range(1, 31)]
    for _, model in runs.values():
        assert len(model.predicted) == len(LOCAL_DAYS)
        for k, frame in zip(LOCAL_DAYS, model.predicted, strict=True):
            assert sorted(set(frame["date"])) == sorted({*dates[:k], *dates[20:]})

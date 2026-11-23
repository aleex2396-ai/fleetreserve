"""The calibration days a held-out unit's window was searched on are the calibration days its scored
run draws (the scope page's item W10), under the committed run matrix.

`heldout_truth_build` searches a Norway or Korea unit's window on its calibration days
(`heldout_truth_build.calibration_dates`: the days that count at τ in `truth.build`, at least 3
cars plugged in, drawn by the matrix's split rule) and records them in its summary
(`windows.<unit>.calibration_dates`). A scored run draws its calibration days from the counting
days `days_from_truth` reads back from the written tables (`gate_run.plan_unit`, with the
matrix's readings). Under the committed matrix (`gate/matrix.json`: `small_fleet` "keep", seed 1,
100 calibration days or half the days under 200) the two lists are equal, which this test pins,
on made-up sessions with the two kinds of day that could part them:

- a day with only 2 cars plugged in at τ, which counts in neither list;
- a day with 3 cars at τ, one of which leaves within the decision hour, so its fleet holds 2 cars
  (`days_from_truth` reading 1): it counts at τ, and the reading "keep" keeps it.

Made-up repository and tag (`fake_heldout.sealed_repo`); OSL_T over 31 days from 6 January 2020
(the 5th day thin, the 10th with the early leaver, the rest `heldout_truth_fake.NORWAY_DAY`), the
other sealed files `heldout_truth_fake`'s; the build of each setting, main and held.

Where the lists part: a scoring matrix with `small_fleet` "drop" leaves the early leaver's day out
of the scored days, or one with another split seed or count draws other days, while the build
drew on its own matrix's rule. The sealed route refuses such a run: the day check
(`sealed_route.check_split_days`) passes under the committed matrix and refuses under
`small_fleet` "drop" (29 days, so 14 calibration days against the build's 15), naming the unit,
both counts and the readings; a unit whose window the build did not search on calibration days
(an a.s.r. unit, at the workplace window) is not compared. The seed and counts are compared with
the build's recorded rule before any table is read (`sealed_route.check_split_rule`, through the
commands).
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fake_heldout import NORWAY_FILE, sealed_repo
from heldout_truth_fake import NORWAY_DAY, NORWAY_START, norway_bytes, norway_days, norway_rows
from heldout_truth_fake import sealed_files as made_up_files

import fleetreserve
from fleetreserve import gate_config, gate_run, sealed_route
from fleetreserve import heldout_truth_build as h
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.truth_verify import CouldNotRun

MATRIX = Path(fleetreserve.__file__).resolve().parents[2] / "gate" / "matrix.json"
DAYS = norway_days(NORWAY_START, 31)
THIN = DAYS[4]  # 2 cars at τ 16:00: not a counting day
SHORT = DAYS[9]  # 3 cars at τ, one leaving at 16:30: a counting day whose fleet holds 2 cars
THIN_DAY = NORWAY_DAY[:2] + NORWAY_DAY[4:]
SHORT_DAY = [*NORWAY_DAY[:2], ("u3", 15.5, 16.5, 1.0), *NORWAY_DAY[4:]]


def sealed_files(folder: Path) -> dict:
    files = made_up_files(folder)
    rows = norway_rows("OSL_T", DAYS, patterns={THIN: THIN_DAY, SHORT: SHORT_DAY})
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, 6))
    files[NORWAY_FILE] = norway_bytes(folder / "norway", rows)
    return files


@pytest.fixture(scope="module", params=["main", "held"])
def built(tmp_path_factory, request):
    """The build of each setting (both search the window under the main setting)."""
    folder = tmp_path_factory.mktemp(f"calibration-days-{request.param}")
    root = sealed_repo(folder, sealed_files(folder / "made"))
    run = folder / "run"
    run.mkdir()
    with pytest.MonkeyPatch.context() as patched:
        patched.setenv("FLEETRESERVE_RUN_DIR", str(run))
        arguments = ["--matrix", str(MATRIX), "--setting", request.param]
        assert h.main(arguments, root=root) == 0
    summary = json.loads((run / "truth-summary.json").read_text())
    return h.default_out(root, request.param), summary


def _scored_calibration_dates(tables: Path, unit: str, matrix) -> list[str]:
    days = load_unit_days(
        tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    plan = gate_run.plan_unit(unit, days, matrix)
    return [key.split(" ")[0] for key in plan.split.get("calibration", [])]


def test_the_made_up_days_are_the_two_kinds(built):
    tables, summary = built
    matrix = gate_config.load(MATRIX)
    days = load_unit_days(
        tables, "OSL_T", rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    dates = [str(t.record.date) for t in days.days]
    assert summary["windows"]["OSL_T"]["tau"] == "16:00"
    assert len(dates) == 30 and str(THIN) not in dates
    (short,) = [t.record for t in days.days if t.record.date == str(SHORT)]
    assert (short.cars_at_tau, short.cars_in_fleet, short.depart_zero_cars) == (3, 2, 1)


def test_the_scored_calibration_days_are_those_the_window_was_searched_on(built):
    tables, summary = built
    matrix = gate_config.load(MATRIX)
    searched = {u: w for u, w in summary["windows"].items() if "calibration_dates" in w}
    assert sorted(searched) == [
        "OSL_T",
        "TRO_R",
        "korea-home",
        "korea-public-parking",
        "korea-work",
    ]
    for unit, window in searched.items():
        assert _scored_calibration_dates(tables, unit, matrix) == window["calibration_dates"]
    assert len(searched["OSL_T"]["calibration_dates"]) == 15


def _units(tables: Path, matrix) -> dict:
    return {
        unit: load_unit_days(
            tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for unit in ("OSL_T", "asr-before-march-2023")
    }


def test_the_route_refuses_a_scored_unit_whose_calibration_days_are_not_the_builds(built):
    tables, summary = built
    matrix = gate_config.load(MATRIX)
    sealed_route.check_split_days(matrix, _units(tables, matrix), summary, "R")  # the same days
    dropped = replace(matrix, small_fleet="drop")  # leaves the early leaver's day out: 29 days
    with pytest.raises(CouldNotRun) as raised:
        sealed_route.check_split_days(dropped, _units(tables, dropped), summary, "R")
    assert str(raised.value).startswith(
        "OSL_T: the 14 calibration days this run draws are not the 15 the held-out build R "
        "searched its window on ("
    )
    assert str(raised.value).endswith(
        "); the matrix's readings (no_fit cap, small_fleet drop) make other counting days than "
        "the build's, so on the sealed route the unit is not scored"
    )

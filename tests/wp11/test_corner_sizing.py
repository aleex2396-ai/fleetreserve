"""The corner search of a sizing method (`also-x`, `cvar`: kind "sizing" of `compare_methods`;
reading W5 of `corner_run`), fast, without a whole run of the command.

1. On the hand day `SINGLE` (`corner_fake`: one car, rate 1 kW, room 1.5 kWh, box
   `0.5 + 0.5 beta zeta_t` in service hours 1 and 2) at scale 1, by hand: the corners ask for
   (0, 0), (1, 0), (0, 1) and (1, 1) kW. Only (+1, +1) asks for more than the car's room (2 kWh
   against 1.5), so it fails in hour 2 (slot loss 1/2) whatever the re-solve plans; the three
   others ask for at most 1 kW in an hour and 1 kWh in all, which the car meets, and recovery
   fills it, so no driver loss. A sizing method is searched as the re-solve is: that one corner
   breaks, and its day (re-solves by outcome included) equals the re-solve's day at the same
   scale in every field but the method's name.
2. At scale 0.5 (requests 0.25 to 0.75 kW, at most 1.5 kWh in all) every corner holds, for the
   sizing methods as for the re-solve: the search is at the scale given.
3. The calibration's sets of losses (`measured_losses`): one per method, the sizing methods
   reading the re-solve's, measured once; and `count_path_hours` counts them.
4. The definitions of a run with a sizing method add reading W5, the resolves text that names
   the sizing methods, the sizing counts under the calibration counts and the timeout key; a run
   without one has the definitions it had.
5. The timeouts by hand: a sizing method's programmes timed out in the calibration are summed
   under `sizing_programmes_calibration` (`oracle_` in front for the oracle variant); a
   calibration without sizing counts adds no such key.
"""

import dataclasses
from collections import Counter
from types import SimpleNamespace

import pytest
from corner_fake import SINGLE

from fleetreserve import compare_methods, corner_run, corner_search, corner_summary

RE_SOLVE = compare_methods.select([compare_methods.RE_SOLVE])[0]
SIZING = tuple(m for m in compare_methods.METHODS if m.kind == "sizing")
MATRIX = SimpleNamespace(gate_units=("c", "b", "a"), n_calibration=100, calibration_half_below=60)


def test_the_registry_holds_the_two_sizing_methods():
    assert [m.name for m in SIZING] == ["also-x", "cvar"]


# --- 1. and 2. the hand day ----------------------------------------------------------------------


@pytest.mark.parametrize("method", SIZING, ids=lambda m: m.name)
def test_a_sizing_method_breaks_single_on_the_upper_corner_only(method):
    found = corner_search.search_day(SINGLE, method, 1.0)
    assert found.method == method.name and found.beta == 1.0
    assert found.status == "breaks" and found.n_paths == 4
    assert [k for k, p in enumerate(found.paths) if p.breaks] == [3]
    assert found.breaking_paths == found.slot_breaking_paths == 1
    assert found.driver_short_paths == 0
    assert found.worst_slot_loss == pytest.approx(0.5)
    assert found.worst_driver_loss == pytest.approx(0.0)
    assert found.worst_path == (1, 1) and found.worst_first_failure == 2
    assert all(p.first_failure is None for p in found.paths[:3])
    assert found.solver_counts is None
    resolved = corner_search.search_day(SINGLE, RE_SOLVE, 1.0)
    assert found == dataclasses.replace(resolved, method=method.name)


@pytest.mark.parametrize("method", SIZING, ids=lambda m: m.name)
def test_a_sizing_method_is_searched_at_the_scale_given(method):
    found = corner_search.search_day(SINGLE, method, 0.5)
    assert found.status == "holds" and found.breaking_paths == 0
    assert found.worst_slot_loss == 0.0 and found.worst_path is None
    resolved = corner_search.search_day(SINGLE, RE_SOLVE, 0.5)
    assert found == dataclasses.replace(resolved, method=method.name)


# --- 3. the calibration's sets of losses ---------------------------------------------------------


def test_the_sizing_methods_share_the_re_solves_losses():
    every = compare_methods.METHODS
    assert corner_run.measured_losses(every) == len(every) - 2
    assert corner_run.measured_losses(compare_methods.select(["also-x"])) == 1
    assert corner_run.measured_losses(compare_methods.select(["also-x", "cvar"])) == 1
    assert corner_run.measured_losses(compare_methods.select(["re-solve", "cvar"])) == 1
    assert corner_run.measured_losses(compare_methods.select(["llf", "cvar"])) == 2
    assert corner_run.measured_losses(compare_methods.select(["llf", "edf"])) == 2


def test_the_calibration_path_hours_count_the_sets_measured():
    day = SimpleNamespace(day=SimpleNamespace(n_service=2), abstains=False)
    prepared = {0: day, 1: day}
    found = corner_run.count_path_hours(prepared, [0], [1], 3, 1, measured=2)
    # 8 audited rows of 2 hours, 10 grid scales, 2 sets; 4 corners of 2 hours for 3 methods
    assert found["calibration"] == 8 * 2 * 10 * 2
    assert found["search_upper_bound"] == 4 * 2 * 3
    assert corner_run.count_path_hours(prepared, [0], [1], 3, 1)["calibration"] == 8 * 2 * 10 * 3


# --- 4. the definitions --------------------------------------------------------------------------


def test_the_definitions_with_a_sizing_method_add_w5_and_its_counts():
    plain = corner_run.definitions(MATRIX)
    assert corner_run.definitions(MATRIX, ()) == plain
    assert "W5" not in plain["readings"]
    found = corner_run.definitions(MATRIX, ["also-x", "cvar"])
    assert found["readings"] == {**plain["readings"], "W5": corner_run.SIZING_READING}
    assert found["fields"]["method.resolves"] == corner_summary.RESOLVES_WITH_SIZING
    assert "sizing" in corner_summary.RESOLVES_WITH_SIZING
    assert found["fields"]["variant.calibration_evaluations"].startswith(
        plain["fields"]["variant.calibration_evaluations"] + "; for a sizing method (also-x, cvar)"
    )
    assert found["timeouts"].startswith(plain["timeouts"] + "; sizing_programmes_calibration: ")
    changed = {"method.resolves", "variant.calibration_evaluations"}
    assert {k for k in plain["fields"] if found["fields"][k] != plain["fields"][k]} == changed
    assert set(found) == set(plain) and found["label"] == plain["label"]


# --- 5. the timeouts -----------------------------------------------------------------------------


def _entry(sized):
    evaluations = {"re_solves_of_the_one_way_rung": {"optimal": 2}}
    if sized is not None:
        evaluations["sizing_solver_outcomes"] = sized
    return {"calibration_evaluations": evaluations, "methods": {}}


def test_the_sizing_programmes_timed_out_in_the_calibration_are_counted():
    sized = {
        "also-x": {"calibration_sets": 1, "solves": {"optimal": 20, "timeout": 3}},
        "cvar": {"calibration_sets": 1, "solves": {"optimal": 1, "timeout": 1}},
    }
    timeouts: Counter = Counter()
    corner_run._count_timeouts(timeouts, "forecast", [], _entry(sized))
    corner_run._count_timeouts(timeouts, "oracle-departure", [], _entry({"cvar": sized["cvar"]}))
    assert timeouts["sizing_programmes_calibration"] == 4
    assert timeouts["oracle_sizing_programmes_calibration"] == 1
    plain: Counter = Counter()
    corner_run._count_timeouts(plain, "forecast", [], _entry(None))
    assert not any("sizing" in key for key in plain)

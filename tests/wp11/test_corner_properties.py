"""Properties of the corner search on seeded made-up prepared days (`compare_fake`), every method
of the registry (the slow tests of the corner search: every method, `asa` and `asa-es` among them,
on every corner of 12 and 3 days; the rules they rest on are also pinned by the fast tests of
`test_corner_hand`, `test_corner_made_up` and `test_corner_summary`).

1. The audited reading (the day's audited rows read from the corner runs) never exceeds the corner
   worst case and never breaks where the corners hold; it equals the day-level losses that
   `compare_day.evaluate` (controllers) and the one-way rung of `gate_day.evaluate` (re-solve)
   give on the audited paths; a controller's solver counts on the audited paths equal
   `compare_day.evaluate`'s. The default list, `asa`, `asa-es` and `diagonal-split` among it, is
   run.
2. The controllers run on the corners with the day's own `departure_seen`: with each made-up day
   set to the other reading, the audited reading still equals `compare_day.evaluate`, and differs
   from what the other reading gives on some day.
"""

import dataclasses

import pytest
from corner_fake import offer_days

from fleetreserve import compare_day, compare_methods, corner_search, gate_day

DAYS = offer_days(24, seed=11)  # the days of test_corner_made_up


@pytest.mark.parametrize("beta", (0.3, 1.0))
def test_audited_reading_is_within_the_corner_worst_case(beta):
    methods = compare_methods.select(None)
    controllers = tuple(m for m in methods if m.kind == "controller")
    for prepared in DAYS[:12]:
        by_compare = compare_day.evaluate(prepared, beta, controllers)
        resolve = gate_day.evaluate(prepared, beta).rungs["one-way"]
        for method in methods:
            found = corner_search.search_day(prepared, method, beta)
            assert found.status in ("breaks", "holds")
            assert found.audited_worst_slot_loss <= found.worst_slot_loss
            assert found.audited_worst_driver_loss <= found.worst_driver_loss
            assert found.audited_breaking_rows <= len(prepared.paths)
            if found.audited_breaks:
                assert found.status == "breaks"
            direct = by_compare.get(method.name, resolve)
            assert found.audited_worst_slot_loss == pytest.approx(direct.slot_loss, abs=1e-12)
            assert found.audited_worst_driver_loss == pytest.approx(direct.driver_loss, abs=1e-12)
            if method.kind == "controller":
                # solver counts as compare_day sums them, on the same paths
                _, _, counts = corner_search.run_paths(prepared, method, beta, prepared.paths)
                assert counts == direct.solver_counts
                assert (found.solver_counts is None) == (direct.solver_counts is None)
    names = {m.name for m in methods}  # the default list runs these
    assert {"asa", "asa-es", "diagonal-split"} <= names


def test_the_controllers_run_with_the_days_own_departure_reading():
    controllers = tuple(m for m in compare_methods.select(None) if m.kind == "controller")
    other = {"hour-start": "rounded-hour", "rounded-hour": "hour-start"}
    differs = set()
    # days 17 and 18 (one of each reading) are those of the 24 on which the reading changes the
    # priority rules' corner runs
    for k in (16, 17, 18):
        prepared = DAYS[k]
        swapped = dataclasses.replace(prepared, departure_seen=other[prepared.departure_seen])
        by_compare = compare_day.evaluate(swapped, 1.0, controllers)
        for method in controllers:
            found = corner_search.search_day(swapped, method, 1.0)
            direct = by_compare[method.name]
            assert found.audited_worst_driver_loss == pytest.approx(direct.driver_loss, abs=1e-12)
            assert found.audited_worst_slot_loss == pytest.approx(direct.slot_loss, abs=1e-12)
            mine = corner_search.search_day(prepared, method, 1.0)
            if [_losses(p) for p in mine.paths] != [_losses(p) for p in found.paths]:
                differs.add((k, prepared.departure_seen))
    assert {k for k, _ in differs} == {17, 18}
    assert {seen for _, seen in differs} == {"hour-start", "rounded-hour"}


def _losses(path):
    return (path.first_failure, path.slot_loss, path.driver_loss)

"""`t3_replay`: the corner tree's own policy replayed on the true cars along the audited paths.

Hand-made cases with known answers:

- the plan's T3 unit example (one car, two hours, rate 1, energy in [1, 1.5], box (0.6, 0.6) +-
  lam (1, 1)) through the per-day check the command runs (`check_rows`): the tree's lam is 0.10, so
  V_tree = 2 * 0.10 * 2 = 0.4 kWh, and the tree's policy matches on every corner path;
- the example's upper-corner box, lam = 0.15 (the largest lam on the upper-corner path alone): a
  policy that meets every request of that box gives the car 0.45 + 0.45 = 0.9 kWh on the
  all-lower path, below its 1 kWh, so that one path is a mismatch (a true row broken, every request
  met) and the other three match;
- a hand-made day of three cars where both products' trees match on all 8 audited rows;
- a command past a car's rate and room is cut to them, so the request of that path is missed;
- a cut of 5e-5 kW, above the row tolerance (1e-6) and below the request tolerance (1e-4), is a
  mismatch by the cut alone, both above a car's rate in hour 1 and in the decision hour (hour 0);
- the paths file's `cuts_above_tolerance` column holds the count of cuts, 2 on a path cut twice;
- a V_tree other than the policy's box is a mismatch by the value alone.

The command end to end on the made-up development build (`ceiling_fakes`, scaled down): every row
matches (exit 0), the rows are the gate's audited paths, the files do not depend on the worker
count; a hand-broken policy on one day is a named mismatch (exit 1); a timeout on one day is
counted missing, never a match (exit 2); every tree is solved under the plan's 300 s limit, and
the document records 300; held-out and training units are refused, and a tables
folder inside a working copy's data folder is refused, with nothing under it opened (an opener
hook, `sealed_watch`).
"""

import csv
import datetime
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest
from ceiling_fakes import setup_build
from sealed_watch import watch

from fleetreserve import gate_run, products, sealed_route, t3_replay
from fleetreserve.audit_paths import audited_paths
from fleetreserve.carsets import CarRows, stack, true_rows
from fleetreserve.ceiling_report import write_csv
from fleetreserve.day import Day, TrueCar
from fleetreserve.tree import TreePolicy, TreeResult, corner_tree, path_nodes

# --- the plan's T3 unit example ------------------------------------------------------------------


def _example_car() -> CarRows:
    """One car, hours 0 (the decision hour, no power) to 2, rate 1, stored energy at most 1.5, at
    least 1 kWh by the end: rows a . u <= b over (u_0, u_1, u_2)."""
    return CarRows(
        2,
        np.array(
            [
                [1, 0, 0],  # power-high: u_0 <= 0
                [0, 1, 0],  # u_1 <= 1
                [0, 0, 1],  # u_2 <= 1
                [-1, 0, 0],  # power-low: u_t >= 0
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0],  # stored-high: running sums <= 1.5
                [1, 1, 0],
                [1, 1, 1],
                [-1, -1, -1],  # need: u_0 + u_1 + u_2 >= 1
            ],
            dtype=float,
        ),
        np.array([0, 1, 1, 0, 0, 0, 1.5, 1.5, 1.5, -1.0]),
        ("power-high",) * 3 + ("power-low",) * 3 + ("stored-high",) * 3 + ("need",),
    )


EXAMPLE_ROWS = stack([_example_car()])
EXAMPLE_BOX = products.fixed_shape((0.6, 0.6), (1.0, 1.0))
ALL_PATHS = np.array(list(itertools.product((-1.0, 1.0), repeat=2)))


def test_the_plans_t3_example_matches_on_every_corner_path():
    found = t3_replay.check_rows(EXAMPLE_ROWS, EXAMPLE_BOX, "example", ALL_PATHS)
    assert found.status == "solved" and found.outcome == "optimal"
    assert found.v_tree == pytest.approx(0.4, abs=1e-6)  # lam 0.10, value 2 * 0.10 * 2
    assert [c.verdict for c in found.paths] == [t3_replay.MATCH] * 4
    for c in found.paths:
        assert c.requests_met and c.rows_broken == 0
        assert c.replayed_value == pytest.approx(0.4, abs=1e-6)


def test_the_plans_t3_example_on_its_audited_rows():
    paths = audited_paths("example", datetime.date(2020, 1, 1), 2)
    found = t3_replay.check_rows(EXAMPLE_ROWS, EXAMPLE_BOX, "example", paths)
    assert found.rows == len(found.paths) == 8
    assert all(c.verdict == t3_replay.MATCH for c in found.paths)


def _upper_corner_policy(lam: float) -> TreePolicy:
    """The one car's power at every node is the request there, for the box (0.6, 0.6) +- lam."""
    m, w = np.array([0.6, 0.6]), np.array([lam, lam])
    powers = np.zeros((1, 7))
    for path in ALL_PATHS:
        powers[0, path_nodes(path)[1:]] = m + w * path
    return TreePolicy(m, w, powers)


def test_the_upper_corner_box_breaks_only_the_all_lower_path():
    policy = _upper_corner_policy(0.15)
    found = [t3_replay.replay_path(EXAMPLE_ROWS, policy, 0.6, path) for path in ALL_PATHS]
    verdicts = {c.corners: c.verdict for c in found}
    assert verdicts == {
        (-1, -1): t3_replay.MISMATCH,
        (-1, 1): t3_replay.MATCH,
        (1, -1): t3_replay.MATCH,
        (1, 1): t3_replay.MATCH,
    }
    lower = found[0]
    assert lower.requests_met and lower.rows_broken == 1
    assert lower.least_row_slack == pytest.approx(-0.1, abs=1e-9)  # 0.9 kWh of the 1 needed


# --- a hand-made day -----------------------------------------------------------------------------

DAY = Day(
    "u",
    datetime.date(2020, 1, 1),
    3,
    2,
    0.0,
    (TrueCar(4, 3.0, 6.0), TrueCar(6, 2.0, 4.0), TrueCar(3, 4.0, 2.0)),
)


def test_a_hand_made_day_matches_on_every_audited_row_for_both_products():
    found = t3_replay.check_day(t3_replay.DayTask("u", "A", "2020-01-01", DAY))
    assert [p.product for p in found.products] == list(t3_replay.PRODUCTS)
    paths = audited_paths("u", datetime.date(2020, 1, 1), 3)
    for p in found.products:
        assert p.status == "solved" and p.v_tree > 0
        assert [c.corners for c in p.paths] == [tuple(int(z) for z in row) for row in paths]
        assert all(c.verdict == t3_replay.MATCH for c in p.paths)


def _hourly_tree():
    return corner_tree(true_rows(DAY), products.hourly())


def test_a_command_past_a_cars_limits_is_cut_to_them_and_the_request_is_missed():
    # The example's box at lam 0.10, the car told 1.5 kW in hour 2 on the all-upper path, where
    # the request is 0.7: its rate allows 1, its room 1.5 - 0.7 = 0.8, so it draws 0.8 (cut 0.7)
    # and the request is missed by 0.1 kWh, with every true row kept by what was drawn.
    policy = _upper_corner_policy(0.10)
    powers = np.array(policy.node_powers)
    powers[0, path_nodes((1, 1))[-1]] = 1.5
    broken = replace(policy, node_powers=powers)
    found = t3_replay.replay_path(EXAMPLE_ROWS, broken, 0.4, (1, 1))
    assert found.verdict == t3_replay.MISMATCH
    assert not found.requests_met and found.first_missed_hour == 2
    assert found.largest_request_miss == pytest.approx(0.1, abs=1e-9)
    assert found.largest_cut == pytest.approx(0.7, abs=1e-9)
    assert found.rows_broken == 0
    assert found.replayed_value == pytest.approx(0.6, abs=1e-9)  # 2 * (0.1 + 0.2)
    other = t3_replay.replay_path(EXAMPLE_ROWS, broken, 0.4, (1, -1))
    assert other.verdict == t3_replay.MATCH


# Two cars of the example (each rate 1, room 1.5, at least 1 kWh, no power in hour 0) under a box of
# no width at 1 kW in each hour: car 0 draws 0 then 1, car 1 draws 1 then 0, on every path.
TWO_CARS = stack([_example_car(), _example_car()])


def _two_car_policy(hour_one) -> TreePolicy:
    powers = np.zeros((2, 7))
    powers[:, 1:3] = np.array(hour_one)[:, None]  # both nodes of hour 1
    powers[:, 3:7] = np.array([[1.0], [0.0]])  # every node of hour 2
    return TreePolicy(np.array([1.0, 1.0]), np.zeros(2), powers)


def test_two_cars_cut_the_opposite_way_in_one_hour_are_a_named_mismatch():
    right = t3_replay.replay_path(TWO_CARS, _two_car_policy([0.0, 1.0]), 0.0, (1, -1))
    assert right.verdict == t3_replay.MATCH
    # Car 0 told -0.5 kW and car 1 told 1.5 kW (0.5 above its rate) in hour 1: each draws what it
    # did before, so the requests, the true rows and the value all hold, and only the cuts differ.
    wrong = t3_replay.replay_path(TWO_CARS, _two_car_policy([-0.5, 1.5]), 0.0, (1, -1))
    assert wrong.requests_met and wrong.rows_broken == 0 and wrong.value_gap == 0.0
    assert wrong.verdict == t3_replay.MISMATCH
    assert right.cuts == 0 and wrong.cuts == 2
    assert (wrong.cut_car, wrong.cut_hour) == (0, 1)
    assert wrong.largest_cut == pytest.approx(0.5, abs=1e-12)
    day = t3_replay.DayCheck(
        "u",
        "A",
        "2020-01-01",
        2,
        2,
        (t3_replay.ProductCheck("hourly", "solved", "optimal", 0.0, 1, (wrong,), 0.0),),
    )
    (line,) = t3_replay.mismatch_lines([day])
    assert line.endswith("commands cut 2 (largest: car 0 in hour 1, 0.5 kW)")


# Above the row tolerance (1e-6 kW) and below the request tolerance (1e-4 kWh): the summed draw
# stays on the request, so only the cut check (d) can see a cut of this size.
SMALL_CUT = 5e-5


def test_a_small_cut_above_a_cars_rate_alone_is_a_mismatch():
    # Car 1 told 1 + 5e-5 kW in hour 1, its rate 1: it draws 1, car 0 draws 0, so the request of
    # 1 kWh is met exactly, every true row holds and the value gap is 0; only the cut shows.
    policy = _two_car_policy([0.0, 1.0 + SMALL_CUT])
    found = t3_replay.replay_path(TWO_CARS, policy, 0.0, (1, -1))
    assert found.requests_met and found.rows_broken == 0 and found.value_gap == 0.0
    assert found.verdict == t3_replay.MISMATCH and found.cuts == 1
    assert (found.cut_car, found.cut_hour) == (1, 1)
    assert found.largest_cut == pytest.approx(SMALL_CUT, abs=1e-12)


def test_a_small_cut_in_the_decision_hour_is_a_mismatch_on_every_path():
    # Car 1 told 5e-5 kW in hour 0, the decision hour, where neither car may draw (u_0 <= 0): it
    # draws 0. Hour 0 has no request and hours 1 and 2 are drawn as before, so only the cut
    # shows, on every path, as the root node is on every path.
    policy = _two_car_policy([0.0, 1.0])
    powers = np.array(policy.node_powers)
    powers[1, 0] = SMALL_CUT
    broken = replace(policy, node_powers=powers)
    for path in ALL_PATHS:
        assert t3_replay.replay_path(TWO_CARS, policy, 0.0, path).verdict == t3_replay.MATCH
        found = t3_replay.replay_path(TWO_CARS, broken, 0.0, path)
        assert found.requests_met and found.rows_broken == 0 and found.value_gap == 0.0
        assert found.verdict == t3_replay.MISMATCH and found.cuts == 1
        assert (found.cut_car, found.cut_hour) == (1, 0)
        assert found.largest_cut == pytest.approx(SMALL_CUT, abs=1e-12)


def test_the_paths_file_holds_the_count_of_cuts(tmp_path):
    # The two-car case above: no cut on the right policy, 2 cuts (largest car 0, hour 1, 0.5 kW)
    # on the wrong one, written by the command's own writer (`path_rows`, `write_csv`).
    right = t3_replay.replay_path(TWO_CARS, _two_car_policy([0.0, 1.0]), 0.0, (1, -1))
    wrong = t3_replay.replay_path(TWO_CARS, _two_car_policy([-0.5, 1.5]), 0.0, (1, -1))
    found = t3_replay.ProductCheck("hourly", "solved", "optimal", 0.0, 2, (right, wrong), 0.0)
    day = t3_replay.DayCheck("u", "A", "2020-01-01", 2, 2, (found,))
    path = tmp_path / t3_replay.PATHS_NAME
    write_csv(path, t3_replay.PATHS_HEADER, t3_replay.path_rows([day]))
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["verdict"] for r in rows] == [t3_replay.MATCH, t3_replay.MISMATCH]
    assert [r["cuts_above_tolerance"] for r in rows] == ["0", "2"]
    assert [r["largest_cut_car"] for r in rows] == ["", "0"]
    assert [r["largest_cut_hour"] for r in rows] == ["", "1"]
    assert float(rows[1]["largest_cut_kw"]) == pytest.approx(0.5, abs=1e-9)


def test_a_cut_at_the_row_tolerance_is_not_a_mismatch():
    tiny = 1.0 + t3_replay.ROW_TOLERANCE / 2
    found = t3_replay.replay_path(TWO_CARS, _two_car_policy([0.0, tiny]), 0.0, (1, -1))
    assert found.verdict == t3_replay.MATCH and found.cuts == 0


def test_a_v_tree_other_than_the_policys_box_is_a_mismatch_by_the_value_alone():
    rows = true_rows(DAY)
    tree = _hourly_tree()
    path = -np.ones(3)
    right = t3_replay.replay_path(rows, tree.policy, tree.value, path)
    assert right.verdict == t3_replay.MATCH
    wrong = t3_replay.replay_path(rows, tree.policy, tree.value + 0.01, path)
    assert wrong.requests_met and wrong.rows_broken == 0
    assert wrong.verdict == t3_replay.MISMATCH and wrong.value_gap == pytest.approx(0.01)


def test_the_replay_steps_down_the_tree_as_the_tree_numbers_its_nodes():
    for path in itertools.product((-1.0, 1.0), repeat=4):
        node, stepped = 0, [0]
        for hour, corner in enumerate(path, start=1):
            node = t3_replay._step(node, hour, corner)
            stepped.append(node)
        assert stepped == list(path_nodes(path))


def test_nothing_checked_could_not_run():
    assert t3_replay.verdict(t3_replay.counts([])) == "could not run"


# --- the command on a made-up build --------------------------------------------------------------

SCALE = 0.2
N_DAYS = 15  # counting days at that scale (ceiling_fakes): 4, 5 and 3 + 3
N_EMPTY = 1  # days of kind 4 (every car leaves within the decision hour): day 4 of nrel-from


def _run(root, patch, *, workers=1, units=None, tables=None):
    root.mkdir(parents=True)
    run, made = setup_build(root, patch, scale=SCALE)
    patch.setattr(gate_run, "_head_commit", lambda: "a" * 40)
    out = root / "out"
    out.mkdir()
    patch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    argv = ["--truth-run", str(run), "--tables", str(tables or made), "--workers", str(workers)]
    code = t3_replay.main([*argv, *([] if units is None else ["--units", *units])])
    return code, out, made


def _document(out):
    return json.loads((out / t3_replay.RESULTS_NAME).read_text())


def _paths(out):
    with (out / t3_replay.PATHS_NAME).open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("t3")
    with pytest.MonkeyPatch.context() as patch:
        one = _run(root / "one", patch, workers=1)
    with pytest.MonkeyPatch.context() as patch:
        two = _run(root / "two", patch, workers=2)
    return {"codes": (one[0], two[0]), "one": one[1], "two": two[1]}


def test_every_audited_row_matches_on_the_made_up_build(runs):
    assert runs["codes"] == (0, 0)
    total = _document(runs["one"])["counts"]["all"]
    solved = N_DAYS - N_EMPTY
    assert total["days"] == N_DAYS
    assert total["day_products_solved"] == 2 * solved
    assert total["day_products_empty"] == 2 * N_EMPTY
    assert total["paths_checked"] == total["matches"] == 2 * solved * 8
    assert total["mismatches"] == total["missing"] == 0
    text = (runs["one"] / t3_replay.TEXT_NAME).read_text()
    assert text.splitlines()[-1].startswith("T3 passed")
    assert "MISMATCH" not in text and "MISSING" not in text


def test_the_rows_are_the_gates_audited_paths(runs):
    rows = _paths(runs["one"])
    assert len(rows) == N_DAYS * 2 * 8
    for row in rows:
        if row["verdict"] != t3_replay.MATCH:
            assert row["verdict"] == t3_replay.EMPTY
            continue
        date = datetime.date.fromisoformat(row["date"])
        paths = audited_paths(row["unit"], date, int(row["service_hours"]))
        expected = "".join("+" if z > 0 else "-" for z in paths[int(row["row"]) - 1])
        assert row["path"] == expected


def test_the_files_do_not_depend_on_the_worker_count(runs):
    for name in (t3_replay.RESULTS_NAME, t3_replay.PATHS_NAME, t3_replay.TEXT_NAME):
        assert (runs["one"] / name).read_bytes() == (runs["two"] / name).read_bytes(), name
    seconds = json.loads((runs["two"] / t3_replay.SECONDS_NAME).read_text())
    assert seconds["workers"] == 2


def _first_solved_day(out):
    row = next(r for r in _paths(out) if r["verdict"] == t3_replay.MATCH)
    return row["unit"], row["site"], row["date"]


def _on_one_day(day, change):
    """`corner_tree` with `change(result, rows)` applied on the one day `day` (unit, site, date)
    only, as `check_day` (patched by `_seam`) names the day it works on."""

    def patched(rows, product, time_limit):
        found = corner_tree(rows, product, time_limit=time_limit)
        if patched.day == day:
            return change(found, rows)
        return found

    patched.day = None
    return patched


def _seam(monkeypatch, day, change):
    patched = _on_one_day(day, change)
    original = t3_replay.check_day

    def check_day(task):
        patched.day = (task.unit, task.site, task.date)
        return original(task)

    monkeypatch.setattr(t3_replay, "corner_tree", patched)
    monkeypatch.setattr(t3_replay, "check_day", check_day)


def test_a_hand_broken_policy_is_a_named_mismatch_and_exit_1(runs, tmp_path, monkeypatch):
    unit, site, date = _first_solved_day(runs["one"])

    def broken(found, rows):
        if found.product.kind != "hourly":
            return found
        assert found.value > 0  # so some hour's upper request m_t + w_t >= 2 w_t is above 0
        powers = np.array(found.policy.node_powers)
        upper = path_nodes(np.ones(rows.n_service))[1:]  # the all-upper path, row 2 of each set
        powers[:, upper] = 0.0
        return replace(found, policy=replace(found.policy, node_powers=powers))

    _seam(monkeypatch, (unit, site, date), broken)
    code, out, _ = _run(tmp_path / "broken", monkeypatch)
    assert code == 1
    total = _document(out)["counts"]["all"]
    assert total["mismatches"] >= 1 and total["missing"] == 0
    text = (out / t3_replay.TEXT_NAME).read_text().splitlines()
    assert text[0].startswith(f"MISMATCH: {unit} {site} {date} hourly row 2 path +")
    assert text[-1].startswith("T3 FAILED")
    bad = [r for r in _paths(out) if r["verdict"] == t3_replay.MISMATCH]
    assert {(r["unit"], r["site"], r["date"], r["product"]) for r in bad} == {
        (unit, site, date, "hourly")
    }


def test_a_timeout_is_missing_never_a_match_and_exit_2(runs, tmp_path, monkeypatch):
    unit, site, date = _first_solved_day(runs["one"])

    def timed_out(found, rows):
        return TreeResult("timeout", found.product, None, None, None, 300.0, found.sizes)

    _seam(monkeypatch, (unit, site, date), timed_out)
    code, out, _ = _run(tmp_path / "timeout", monkeypatch)
    assert code == 2
    document = _document(out)
    total = document["counts"]["all"]
    solved = N_DAYS - N_EMPTY
    assert total["missing"] == 2 * 8  # both products of the one day
    assert total["matches"] == total["paths_checked"] == 2 * (solved - 1) * 8
    assert total["mismatches"] == 0
    missing = [r for r in _paths(out) if r["verdict"] == t3_replay.MISSING]
    assert len(missing) == 16
    assert {(r["unit"], r["site"], r["date"], r["tree_outcome"]) for r in missing} == {
        (unit, site, date, "timeout")
    }
    assert all(r["v_tree_kwh"] == "" and r["replayed_value_kwh"] == "" for r in missing)
    text = (out / t3_replay.TEXT_NAME).read_text().splitlines()
    head = f"MISSING: {unit} {site} {date} hourly"
    assert text[0] == f"{head}: tree timeout; its 8 audited rows missing"
    assert text[-1].startswith("T3 COULD NOT RUN")


def test_every_tree_is_solved_under_the_plans_300_second_limit(tmp_path, monkeypatch):
    # Plan section 13: 300 s for every other programme. The stand-in records each call's limit
    # and solves as `corner_tree` does; the number is written out, not read from the module.
    limits = []

    def recorded(rows, product, time_limit=None):
        limits.append(time_limit)
        return corner_tree(rows, product, time_limit=time_limit)

    monkeypatch.setattr(t3_replay, "corner_tree", recorded)
    code, out, _ = _run(tmp_path / "limit", monkeypatch)
    assert code == 0
    assert limits == [300] * (2 * (N_DAYS - N_EMPTY))  # both products of every solved day
    assert _document(out)["configuration"]["tree_seconds"] == 300
    assert "tree time limit 300.0 s." in (out / t3_replay.TEXT_NAME).read_text()


# --- refusals ------------------------------------------------------------------------------------


@pytest.mark.parametrize("unit", [sealed_route.HELD_OUT_UNITS[2], "caltech"])
def test_held_out_and_training_units_are_refused_with_nothing_opened(
    unit, tmp_path, monkeypatch, capsys
):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / f"{unit}-cars.csv").write_text("x\n")
    (sealed / f"{unit}-days.csv").write_text("x\n")
    with watch(sealed) as seen:
        code, out, _ = _run(tmp_path / "build", monkeypatch, units=[unit], tables=sealed)
    assert code == 2
    assert seen == []
    assert capsys.readouterr().out.startswith("COULD NOT RUN: --units names")
    assert list(out.iterdir()) == []


def test_tables_in_a_working_copys_data_folder_are_refused_before_any_open(
    tmp_path, monkeypatch, capsys
):
    copy = tmp_path / "copy"
    (copy / ".git").mkdir(parents=True)
    sealed = copy / "data" / "heldout"
    sealed.mkdir(parents=True)
    for kind in ("cars", "days"):
        (sealed / f"nrel-before-may-2019-{kind}.csv").write_text("x\n")
    with watch(copy / "data") as seen:
        code, out, _ = _run(
            tmp_path / "build", monkeypatch, units=["nrel-before-may-2019"], tables=sealed
        )
    assert code == 2
    assert seen == []
    assert "refused before it is opened" in capsys.readouterr().out
    assert list(out.iterdir()) == []

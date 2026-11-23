"""The command of outcome O3 (`o3_run`) on made-up tables (`o3_fake`): the seeded sample, the time
limits, missing values and the exit status, the refusals and the worker count.

1. The sample: on a unit of `N_DAYS` counting days, `--days 4 --seed 7` draws the pinned days
   `PINNED`, which are the rule written out in the test (the 4 days with the smallest SHA-256 of
   [unit, seed, "site date"] as JSON), and the same on a second run; seed 8 draws the rule's
   other days; `--days` above the counting days gives every day, marked.
2. Time limits: `--lp24-seconds`, `--other-seconds` and `--box-seconds` reach the solver: LP (24)
   once at its limit; LP (27) and every programme of its search over the maps (at least LP (27),
   the reversed order, the draws and direction sets of the default effort `o3_run.EFFORT_SHARE`,
   and LP (27)'s bound's map ranges, 2 n q per product) at the others' limit; the two general
   splits per product, the block ceiling, every box programme of the search (at least one per
   candidate and product) and the bound's root relaxation per product at the box limit, and no
   other limit (reading D4).
   At 0.001 s each, every value whose solve did not end optimal is missing (empty, never 0),
   counted in `missing_values`, and the exit status is 0 (reading R3); HiGHS's interior point
   method checks its clock only at some points and returns most such solves late but optimal
   (`lp` module text), so how many time out is not fixed. With every solve given `TINY` seconds
   answered as the solver answers one that reached its time limit (`solver_timeout_fake`: a real
   limit that small races the clock in the same way), every value but the hourly ceiling (no
   solve) is missing, counted by outcome and named on the summary's third line, and the exit
   status is still 0.
3. Exit status 1: a programme that ends "infeasible" (made-up solver answer) is a fault.
4. Refusals, exit 2, with no output file: an unverified truth build (its manifest records exit 1),
   a build at another commit, a table differing from the build, a unit that is not a training,
   development or held-out unit, a unit named twice, a held-out unit away from the tag (refused
   by the sealed route's gate; the route itself is tested in `tests/wp2/test_heldout_route_o3.py`),
   the template base set (not built), a unit with no counting day; and `--days`, `--seed`,
   `--base-set` and `--units` have no default.
5. One and two worker processes give the same results file and days table.
6. LP (27)'s columns of the days table and its keys in the results file say "found" or "bound"
   (`lp27_smallest_found`, `lp27_largest_found`, `lp27_largest_bound`), as the summary's labels
   do; a present bound's outcome is closed, ceiling or open (`o3_lp27_bound`).
7. LP (27)'s cost arguments (reading R4): `--lp27-effort` outside (0, 1] (0, -0.5, 1.5, nan) and
   `--lp27-bound-nodes` below 1 are refused, exit 2, no output file; the values given reach
   every day's task (the settings `o3_lp27.scaled_effort` gives and the node budget), are
   recorded in the results file and named on the summary's second line; the defaults reach the
   tasks and are recorded when none is given. The defaults are share 0.1, at which the ascent
   has one start from each source, and one node (reading R4's measured times).
"""

import csv
import hashlib
import json

import pytest
from gate_fake import COMMIT
from o3_fake import build, trust_commit, varied_cars
from solver_timeout_fake import force_timeouts

from fleetreserve import exact, gate_run, o3_lp27, o3_lp27_bound, o3_run, sparse_programme
from fleetreserve.lp import LpResult, solve_lp
from fleetreserve.o3_day import BOUND, KEYS, METHODS, PRODUCTS, compute_day
from fleetreserve.o3_lp27_bound import CEILING, CLOSED, OPEN

UNIT = "nrel-before-may-2019"
N_DAYS = 12
PINNED = ["A 2020-01-01", "A 2020-01-06", "A 2020-01-09", "A 2020-01-11"]
TINY = 1e-6


def _smallest_ranks(unit, seed, texts, count):
    """The sample rule written out: the `count` texts with the smallest SHA-256 of
    [unit, seed, text] as JSON, in text order."""
    ranked = sorted(
        texts, key=lambda t: hashlib.sha256(json.dumps([unit, seed, t]).encode()).digest()
    )
    return sorted(ranked[:count])


def _arguments(run, tables, *extra, units=(UNIT,), days="4", seed="7"):
    return [
        "--units",
        *units,
        "--days",
        days,
        "--seed",
        seed,
        "--base-set",
        "day-average",
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        *extra,
    ]


@pytest.fixture
def made_up(tmp_path, monkeypatch):
    trust_commit(monkeypatch)
    run, tables = build(tmp_path, {UNIT: varied_cars(N_DAYS, seed=3)}, service_hours=3)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return run, tables, out


def _document(out):
    return json.loads((out / o3_run.RESULTS_NAME).read_text())


def _rows(out):
    with (out / o3_run.DAYS_NAME).open() as handle:
        return list(csv.DictReader(handle))


def test_the_sample_is_pinned_by_seed_and_tables(made_up):
    texts = [f"A 2020-01-{day:02d}" for day in range(1, N_DAYS + 1)]
    assert _smallest_ranks(UNIT, 7, texts, 4) == PINNED
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables)) == 0
    first = _document(out)["sample"][UNIT]
    assert first["days"] == PINNED
    assert first["counting_days"] == N_DAYS and first["sampled_days"] == 4
    assert not first["fewer_than_asked"]
    assert [f"{r['site']} {r['date']}" for r in _rows(out) if r["product"] == "hourly"] == PINNED
    assert o3_run.main(_arguments(run, tables)) == 0
    assert _document(out)["sample"][UNIT]["days"] == PINNED
    assert o3_run.main(_arguments(run, tables, seed="8")) == 0
    assert _document(out)["sample"][UNIT]["days"] == _smallest_ranks(UNIT, 8, texts, 4) != PINNED
    assert o3_run.main(_arguments(run, tables, days="50")) == 0
    every = _document(out)["sample"][UNIT]
    assert every["sampled_days"] == N_DAYS and every["fewer_than_asked"]
    assert every["days"] == texts


def _spy(monkeypatch):
    seen = []

    def spy(c, **kwargs):
        seen.append(kwargs["time_limit"])
        return solve_lp(c, **kwargs)

    monkeypatch.setattr(sparse_programme, "solve_lp", spy)
    monkeypatch.setattr(exact, "solve_lp", spy)
    monkeypatch.setattr(o3_lp27, "solve_lp", spy)
    return seen


def test_each_time_limit_reaches_its_solves(made_up, monkeypatch):
    run, tables, _ = made_up
    seen = _spy(monkeypatch)
    bound_seen = []
    solve = o3_lp27_bound._solve

    def bound_spy(lp, objective, time_limit):
        bound_seen.append(time_limit)
        return solve(lp, objective, time_limit)

    monkeypatch.setattr(o3_lp27_bound, "_solve", bound_spy)
    limits = ["--lp24-seconds", "70", "--other-seconds", "50", "--box-seconds", "30"]
    assert o3_run.main(_arguments(run, tables, *limits, days="1")) == 0
    effort = o3_lp27.scaled_effort(o3_run.EFFORT_SHARE)
    candidates = 2 + effort.draws + len(o3_lp27._directions(3, effort.range_draws))
    assert set(seen) == {30.0, 50.0, 70.0}
    assert seen.count(70.0) == 1
    assert seen.count(50.0) >= candidates
    assert seen.count(30.0) >= 5 + 2 * candidates
    # the bound: 2 n q map ranges per product (n = 3, q >= 1), one root relaxation per product
    assert set(bound_seen) == {30.0, 50.0}
    assert bound_seen.count(50.0) >= 2 * 2 * 3 and bound_seen.count(30.0) >= 2


def _check_missing_recorded(out):
    """Every value whose outcome is not optimal (or exact, or for the bound closed, ceiling or
    open) is empty and counted as missing."""
    counted = {}
    for row in _rows(out):
        for key in KEYS:
            outcome, value = row[f"{key}_outcome"], row[f"{key}_kwh"]
            present = (CLOSED, CEILING, OPEN) if key == BOUND else ("optimal", "exact")
            if outcome in present:
                assert value != "", (key, row)
            else:
                assert value == "", (key, row)
                counted[outcome] = counted.get(outcome, 0) + 1
    assert _document(out)["missing_values"] == counted
    return counted


def test_a_limit_of_0_001_seconds(made_up, monkeypatch):
    run, tables, out = made_up
    seen = _spy(monkeypatch)
    limits = ["--lp24-seconds", "0.001", "--other-seconds", "0.001", "--box-seconds", "0.001"]
    assert o3_run.main(_arguments(run, tables, *limits)) == 0
    assert seen and set(seen) == {0.001}
    counted = _check_missing_recorded(out)
    assert set(counted) <= {"timeout"}
    print(f"values missing at 0.001 s: {counted}")


def test_every_solve_timed_out_is_missing_and_the_exit_status_is_0(made_up, monkeypatch):
    force_timeouts(monkeypatch, TINY)
    run, tables, out = made_up
    limits = ["--lp24-seconds", str(TINY), "--other-seconds", str(TINY), "--box-seconds", str(TINY)]
    assert o3_run.main(_arguments(run, tables, *limits)) == 0
    counted = _check_missing_recorded(out)
    # 4 days x (2 products x 7 values - the hourly ceiling, which needs no solve).
    assert counted == {"timeout": 4 * 13}
    document = _document(out)
    assert document["failures"] == []
    hourly = document["units"][UNIT]["hourly"]
    assert hourly["ceiling"]["days_with_value"] == 4
    for method in METHODS:
        assert hourly[method]["days_with_value"] == 0
        assert hourly[method]["missing"] == {"timeout": 4}
        assert hourly[method]["kwh"]["mean"] is None and hourly[method]["share"]["days"] == 0
    third = (out / o3_run.SUMMARY_NAME).read_text().splitlines()[2]
    assert third == "MISSING VALUES (never read as zero): 52 (52 timeout)"


def test_the_lp27_columns_and_keys_say_found_or_bound(made_up):
    run, tables, out = made_up
    limits = ["--lp24-seconds", str(TINY), "--other-seconds", str(TINY), "--box-seconds", str(TINY)]
    assert o3_run.main(_arguments(run, tables, *limits, days="1")) == 0
    header = list(_rows(out)[0])
    assert [c for c in header if c.startswith("lp27_")] == [
        "lp27_smallest_found_kwh",
        "lp27_smallest_found_outcome",
        "lp27_largest_found_kwh",
        "lp27_largest_found_outcome",
        "lp27_largest_bound_kwh",
        "lp27_largest_bound_outcome",
        "lp27_smallest_found_share",
        "lp27_largest_found_share",
        "lp27_largest_bound_share",
        "lp27_trace",
        "lp27_skipped",
    ]
    for product in PRODUCTS:
        keys = [k for k in _document(out)["units"][UNIT][product] if k.startswith("lp27_")]
        assert keys == ["lp27_largest_bound", "lp27_largest_found", "lp27_smallest_found"]


@pytest.mark.parametrize(
    "extra",
    [
        ("--lp27-effort", "0"),
        ("--lp27-effort", "-0.5"),
        ("--lp27-effort", "1.5"),
        ("--lp27-effort", "nan"),
        ("--lp27-bound-nodes", "0"),
    ],
)
def test_lp27_cost_arguments_out_of_range_are_refused(made_up, extra, capsys):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, *extra, days="1")) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert _no_outputs(out)


def test_lp27_cost_arguments_reach_the_days_and_are_recorded(made_up, monkeypatch):
    run, tables, out = made_up
    tasks = []

    def spy(task):
        tasks.append((task.effort, task.bound_nodes))
        return compute_day(task)

    monkeypatch.setattr(o3_run, "compute_day", spy)
    limits = ["--lp24-seconds", str(TINY), "--other-seconds", str(TINY), "--box-seconds", str(TINY)]
    given = ["--lp27-effort", "0.5", "--lp27-bound-nodes", "7"]
    assert o3_run.main(_arguments(run, tables, *limits, *given, days="2")) == 0
    effort = o3_lp27.scaled_effort(0.5)
    assert tasks == [(effort, 7), (effort, 7)]
    assert _document(out)["arguments"]["lp27"] == {
        "effort_share": 0.5,
        "settings": {
            "draws": effort.draws,
            "range_draws": effort.range_draws,
            "mixtures": effort.mixtures,
            "ascent_starts": effort.ascent_starts,
            "mixture_starts": effort.mixture_starts,
            "other_starts": effort.other_starts,
            "rounds": effort.rounds,
        },
        "bound_nodes": 7,
    }
    second = (out / o3_run.SUMMARY_NAME).read_text().splitlines()[1]
    assert second.endswith(
        f"LP (27) search effort 0.5 (draws {effort.draws}, range_draws {effort.range_draws}, "
        f"mixtures {effort.mixtures}, ascent_starts {effort.ascent_starts}, mixture_starts "
        f"{effort.mixture_starts}, other_starts {effort.other_starts}, rounds {effort.rounds}), "
        "bound relaxations per product 7."
    )
    tasks.clear()
    assert o3_run.main(_arguments(run, tables, *limits, days="1")) == 0
    recorded = _document(out)["arguments"]["lp27"]
    assert recorded["effort_share"] == o3_run.EFFORT_SHARE
    assert recorded["bound_nodes"] == o3_run.RUN_BOUND_NODES
    assert tasks == [(o3_lp27.scaled_effort(o3_run.EFFORT_SHARE), o3_run.RUN_BOUND_NODES)]


def test_the_cost_defaults_are_those_chosen_from_measured_times():
    """Reading R4: share 0.1 (the ascent's fewest starts) and the root alone."""
    assert (o3_run.EFFORT_SHARE, o3_run.RUN_BOUND_NODES) == (0.1, 1)
    effort = o3_lp27.scaled_effort(o3_run.EFFORT_SHARE)
    assert (effort.ascent_starts, effort.mixture_starts) == (1, 1)


def test_an_infeasible_programme_is_a_fault(made_up, monkeypatch):
    run, tables, out = made_up

    def infeasible(c, **kwargs):
        return LpResult("infeasible", None, None, 0.1, kwargs["time_limit"], 2, "made up")

    monkeypatch.setattr(exact, "solve_lp", infeasible)
    assert o3_run.main(_arguments(run, tables, days="1")) == 1
    document = _document(out)
    assert len(document["failures"]) == 1
    assert "block: ceiling ended 'infeasible'" in document["failures"][0]
    assert "FAILED: " in (out / o3_run.SUMMARY_NAME).read_text()


def _no_outputs(out):
    return not any(out.iterdir())


def test_an_unverified_truth_build_is_refused(made_up, capsys):
    run, tables, out = made_up
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["exit_code"] = 1
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert o3_run.main(_arguments(run, tables)) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert _no_outputs(out)


def test_a_build_at_another_commit_is_refused(made_up, monkeypatch):
    run, tables, out = made_up
    monkeypatch.setattr(gate_run, "_head_commit", lambda: "b" * 40)
    assert o3_run.main(_arguments(run, tables)) == 2
    assert _no_outputs(out)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    assert o3_run.main(_arguments(run, tables)) == 0


def test_a_changed_table_is_refused(made_up):
    run, tables, out = made_up
    path = tables / f"{UNIT}-cars.csv"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(path.read_text() + "\n")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != before
    assert o3_run.main(_arguments(run, tables)) == 2
    assert _no_outputs(out)


@pytest.mark.parametrize(
    "units",
    [("not-a-unit",), (UNIT, UNIT)],
    ids=["unknown unit", "named twice"],
)
def test_units_outside_the_training_and_development_lists_are_refused(made_up, units):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, units=units)) == 2
    assert _no_outputs(out)


def test_a_held_out_unit_away_from_the_tag_could_not_run(made_up, tmp_path, capsys):
    """Refused by the sealed route's gate (the top folder given is no repository, so no tag),
    before any table is opened."""
    run, tables, out = made_up
    nowhere = tmp_path / "no-repository"
    nowhere.mkdir()
    arguments = _arguments(run, tables, "--held-out-truth-run", str(run), units=("OSL_T",))
    assert o3_run.main(arguments, root=nowhere) == 2
    shown = capsys.readouterr().out
    assert shown.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert shown.endswith("; nothing read\n") and shown.count("\n") == 1
    assert _no_outputs(out)


def test_the_template_base_set_is_not_built(made_up, capsys):
    run, tables, out = made_up
    arguments = _arguments(run, tables)
    arguments[arguments.index("day-average")] = "template"
    assert o3_run.main(arguments) == 2
    assert "not built" in capsys.readouterr().out
    assert _no_outputs(out)


def test_a_unit_with_no_counting_day_could_not_run(tmp_path, monkeypatch):
    trust_commit(monkeypatch)
    thin = [car for car in varied_cars(3, seed=1) if not car.session.endswith(("-2", "-3"))]
    run, tables = build(tmp_path, {UNIT: thin}, service_hours=3)
    assert o3_run.main(_arguments(run, tables)) == 2


@pytest.mark.parametrize("dropped", ["--days", "--seed", "--base-set", "--units"])
def test_the_run_list_arguments_have_no_default(made_up, dropped):
    run, tables, _ = made_up
    arguments = _arguments(run, tables)
    at = arguments.index(dropped)
    del arguments[at : at + 2]
    with pytest.raises(SystemExit) as stopped:
        o3_run.main(arguments)
    assert stopped.value.code == 2


def test_the_worker_count_does_not_change_the_results(made_up, tmp_path, monkeypatch):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables)) == 0
    one = ((out / o3_run.RESULTS_NAME).read_text(), (out / o3_run.DAYS_NAME).read_text())
    two_out = tmp_path / "two"
    two_out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(two_out))
    assert o3_run.main(_arguments(run, tables, "--workers", "2")) == 0
    two = ((two_out / o3_run.RESULTS_NAME).read_text(), (two_out / o3_run.DAYS_NAME).read_text())
    assert one == two
    assert len(_rows(out)) == 4 * len(PRODUCTS)

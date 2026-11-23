"""O3's days per method and LP (27)'s time cap per unit (`o3_run` readings R8 to R10, `o3_day`
reading D9; plan section 11, O3: LP (27) on 100 sampled days per unit, LP (24) and the general
split on 200, LP (27)'s run time capped at about two hours per unit, the days not reached
recorded as missing), on made-up tables (`o3_fake`) and tiny limits.

1. The two day counts: `--days 6 --lp27-days 3` draws the 6 days with the smallest SHA-256 ranks
   of [unit, seed, "site date"] (written out in the test) and LP (27) runs on the 3 of them with
   the smallest ranks, so the 3 are among the 6 and the draw is the same whatever `--lp27-days`
   is. LP (24), the general splits and the ceilings have a value on all 6 days; LP (27)'s three
   values are empty on the other 3 days, with the outcome "not a day of LP (27)'s sample", which
   is not counted as a missing value (`missing_values`); LP (27)'s statistics and the unit's
   LP (27) line count the 3 days. `--lp27-days` above the sampled days gives LP (27) on all of
   them.
2. The days are started in the order of their hash ranks, the units one after another.
3. The cap, on a made-up clock (`o3_day.now`) that moves 40 s at each reading, with a cap of
   100 s: the first two days with cars of a unit are reached, the others are not (LP (27)'s three
   values empty, outcome "not reached within the unit's LP (27) time cap", counted per unit in
   `lp27_cap` and printed on the unit's line and in the third line), they are the days with the
   largest ranks, every other value is there, and the exit status is 0. With two units the clock
   of each starts with its own first day. With a cap of 1e-9 s on the real clock, no day is
   reached. A cap far away reaches every day (the default 7200 s too).
4. Every solve keeps its own limit: with the cap far away and every solve given `TINY` seconds
   answered as the solver answers one that reached its limit (`solver_timeout_fake`), LP (27)'s
   values are missing with the outcome "timeout", not as cap days.
5. Quick-look numbers: the defaults are the plan's (100 days, 7200 s); `departures` names each
   number that differs from the plan's, and nothing when none does; the summary prints the
   list. `--lp27-days` below 1 and a cap that is not a positive number are refused (exit 2, no
   output file). One and two workers give the same results with `--lp27-days` below `--days`.
"""

import hashlib
import json
import math

import pytest
from o3_fake import build, trust_commit, varied_cars
from solver_timeout_fake import force_timeouts

from fleetreserve import o3_day, o3_run
from fleetreserve.lp import BOX_SECONDS, LP24_SECONDS, OTHER_SECONDS
from fleetreserve.o3_day import CAP_REACHED, LP27_KEYS, NOT_SAMPLED, PRODUCTS, compute_day

UNIT = "nrel-before-may-2019"
OTHER = "nrel-from-may-2019"
N_DAYS = 12
SEED = 7
TINY = 1e-6
TEXTS = [f"A 2020-01-{day:02d}" for day in range(1, N_DAYS + 1)]


def _by_rank(unit, seed, texts):
    """The texts in the order of their SHA-256 of [unit, seed, text] as JSON (the draw's rule)."""
    return sorted(
        texts, key=lambda t: (hashlib.sha256(json.dumps([unit, seed, t]).encode()).digest(), t)
    )


def _arguments(run, tables, *extra, units=(UNIT,), days="6", seed=str(SEED)):
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
    units = {UNIT: varied_cars(N_DAYS, seed=3), OTHER: varied_cars(N_DAYS, seed=5)}
    run, tables = build(tmp_path, units, service_hours=3)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return run, tables, out


def _document(out):
    return json.loads((out / o3_run.RESULTS_NAME).read_text())


def _rows(out):
    import csv

    with (out / o3_run.DAYS_NAME).open() as handle:
        return list(csv.DictReader(handle))


def _day_of(row):
    return f"{row['site']} {row['date']}"


def _lp27_cells(row):
    return [(row[f"{key}_kwh"], row[f"{key}_outcome"]) for key in LP27_KEYS]


# --- 1. the two day counts -------------------------------------------------------------------


def test_lp27_runs_on_the_first_days_of_the_one_draw_and_the_other_methods_on_all(made_up):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, "--lp27-days", "3")) == 0
    ranked = _by_rank(UNIT, SEED, TEXTS)
    document = _document(out)
    sample = document["sample"][UNIT]
    assert sample["days"] == sorted(ranked[:6])
    assert sample["lp27_days"] == sorted(ranked[:3])
    assert set(sample["lp27_days"]) < set(sample["days"])
    rows = [r for r in _rows(out) if r["product"] == "hourly"]
    assert [_day_of(r) for r in rows] == sorted(ranked[:6])
    for row in rows:
        asked = _day_of(row) in ranked[:3]
        assert row["general_split_kwh"] != "" and row["ceiling_kwh"] != ""
        assert row["lp24_kwh"] != "" or row["lp24_outcome"] != "optimal"
        for kwh, outcome in _lp27_cells(row):
            if asked:
                assert outcome != NOT_SAMPLED
            else:
                assert (kwh, outcome) == ("", NOT_SAMPLED)
    # the unit's statistics: LP (24) and the split on 6 days, LP (27) on its 3
    hourly = document["units"][UNIT]["hourly"]
    assert hourly["lp24"]["days"] == 6 and hourly["general_split"]["days"] == 6
    for key in LP27_KEYS:
        assert hourly[key]["days"] == 3
        assert NOT_SAMPLED not in hourly[key]["missing"]
    assert NOT_SAMPLED not in document["missing_values"]
    assert document["lp27_cap"][UNIT]["days"] == 3
    assert document["lp27_cap"][UNIT]["not_reached"] == 0
    text = (out / o3_run.SUMMARY_NAME).read_text()
    assert (
        "  LP (27): 3 days asked for, 3 reached, 0 not reached within the cap of 7200.0 s" in text
    )
    assert f"UNIT {UNIT}: 6 of {N_DAYS} counting days sampled" in text


def test_the_draw_does_not_depend_on_the_number_of_lp27_days(made_up):
    run, tables, out = made_up
    drawn = {}
    for count in ("2", "4", "6"):
        assert o3_run.main(_arguments(run, tables, "--lp27-days", count)) == 0
        sample = _document(out)["sample"][UNIT]
        drawn[count] = (sample["days"], sample["lp27_days"])
    assert drawn["2"][0] == drawn["4"][0] == drawn["6"][0]
    assert set(drawn["2"][1]) < set(drawn["4"][1]) < set(drawn["6"][1])
    assert drawn["6"][1] == drawn["6"][0]


def test_lp27_days_above_the_sampled_days_give_lp27_on_all_of_them(made_up):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, "--lp27-days", "50")) == 0
    sample = _document(out)["sample"][UNIT]
    assert sample["lp27_days"] == sample["days"] and len(sample["days"]) == 6
    assert all(o[1] != NOT_SAMPLED for row in _rows(out) for o in _lp27_cells(row))


# --- 2. the order the days are started in ---------------------------------------------------


def test_the_days_are_started_in_the_order_of_their_hash_ranks_unit_after_unit(
    made_up, monkeypatch
):
    run, tables, _ = made_up
    started = []

    def spy(task):
        started.append((task.unit, f"{task.site} {task.date}", task.rank, task.lp27))
        return compute_day(task)

    monkeypatch.setattr(o3_run, "compute_day", spy)
    arguments = _arguments(run, tables, "--lp27-days", "2", units=(OTHER, UNIT), days="4")
    assert o3_run.main(arguments) == 0
    expected = []
    for unit in sorted((UNIT, OTHER)):
        for rank, text in enumerate(_by_rank(unit, SEED, TEXTS)[:4]):
            expected.append((unit, text, rank, rank < 2))
    assert started == expected


# --- 3. the cap ------------------------------------------------------------------------------


def _clock(monkeypatch, step=40.0):
    """`o3_day.now` as a made-up clock: 0 at its first reading, `step` more at each next."""
    readings = []

    def now():
        readings.append(step * len(readings))
        return readings[-1]

    monkeypatch.setattr(o3_day, "now", now)
    return readings


def _reached_and_not(out, unit):
    """(reached days, not reached days) of the unit's LP (27) days, in the order of the rows."""
    reached, capped = [], []
    for row in _rows(out):
        if row["unit"] != unit or row["product"] != "hourly" or int(row["cars"]) == 0:
            continue  # a day with no car needs no solve and no clock reading
        cells = _lp27_cells(row)
        if all(outcome == CAP_REACHED for _, outcome in cells):
            assert all(kwh == "" for kwh, _ in cells)
            capped.append(_day_of(row))
        elif all(outcome != NOT_SAMPLED for _, outcome in cells):
            reached.append(_day_of(row))
    return reached, capped


def _days_with_cars(out, unit):
    return [
        _day_of(r)
        for r in _rows(out)
        if r["unit"] == unit and r["product"] == "hourly" and int(r["cars"]) > 0
    ]


def test_a_unit_whose_cap_is_spent_has_its_later_lp27_days_missing_not_zero(
    made_up, monkeypatch, capsys
):
    run, tables, out = made_up
    _clock(monkeypatch)
    arguments = _arguments(run, tables, "--lp27-cap-seconds", "100", "--lp27-days", "6")
    assert o3_run.main(arguments) == 0
    ranked = _by_rank(UNIT, SEED, TEXTS)[:6]
    reached, capped = _reached_and_not(out, UNIT)
    assert len(_days_with_cars(out, UNIT)) >= 4  # the made-up days are not all empty
    assert len(reached) == 2 and len(capped) >= 2
    first_two = [d for d in ranked if d in _days_with_cars(out, UNIT)][:2]
    assert sorted(reached) == sorted(first_two)
    # the days not reached are the last ranks: no reached day ranks after a day not reached
    assert max(ranked.index(d) for d in reached) < min(ranked.index(d) for d in capped)
    document = _document(out)
    reach = document["lp27_cap"][UNIT]
    assert (reach["days"], reach["reached"], reach["not_reached"]) == (
        6,
        6 - len(capped),
        len(capped),
    )
    assert sorted(reach["not_reached_days"]) == sorted(capped)
    # every other value is there on the days not reached; LP (27)'s are missing, never 0
    for row in _rows(out):
        if _day_of(row) in capped:
            assert row["ceiling_kwh"] != "" and row["general_split_kwh"] != ""
            assert row["lp27_largest_found_kwh"] == "" and row["lp27_largest_found_share"] == ""
    hourly = document["units"][UNIT]["hourly"]
    for key in LP27_KEYS:
        assert hourly[key]["missing"].get(CAP_REACHED) == len(capped)
        assert hourly[key]["days_with_value"] <= 2
    assert document["missing_values"][CAP_REACHED] == len(capped) * len(PRODUCTS) * len(LP27_KEYS)
    text = (out / o3_run.SUMMARY_NAME).read_text()
    assert (
        f"  LP (27): 6 days asked for, {6 - len(capped)} reached, {len(capped)} not reached "
        "within the cap of 100.0 s (missing, never zero)"
    ) in text
    assert f"{CAP_REACHED}" in text.splitlines()[2]
    assert text.splitlines()[2].startswith("MISSING VALUES (never read as zero): ")
    assert capsys.readouterr().out == text


def test_each_unit_has_its_own_cap_clock(made_up, monkeypatch):
    run, tables, out = made_up
    readings = _clock(monkeypatch)
    arguments = _arguments(
        run, tables, "--lp27-cap-seconds", "100", "--lp27-days", "6", units=(UNIT, OTHER)
    )
    assert o3_run.main(arguments) == 0
    for unit in (UNIT, OTHER):
        reached, capped = _reached_and_not(out, unit)
        assert len(reached) == 2, (unit, reached, capped)
        assert len(_days_with_cars(out, unit)) >= 4 and len(capped) >= 2
    # one reading per unit's start, one per day with cars
    cars = sum(len(_days_with_cars(out, unit)) for unit in (UNIT, OTHER))
    assert len(readings) == 2 + cars


def test_a_cap_spent_before_the_first_day_leaves_no_day_for_lp27(made_up):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, "--lp27-cap-seconds", "1e-9", days="4")) == 0
    reached, capped = _reached_and_not(out, UNIT)
    assert reached == [] and len(capped) == len(_days_with_cars(out, UNIT)) >= 2
    assert _document(out)["failures"] == []


def test_a_cap_far_away_reaches_every_day(made_up):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, "--lp27-cap-seconds", "1e9", days="4")) == 0
    assert _reached_and_not(out, UNIT)[1] == []
    assert _document(out)["lp27_cap"][UNIT]["not_reached"] == 0
    assert CAP_REACHED not in _document(out)["missing_values"]


# --- 4. every solve keeps its own limit -------------------------------------------------------


def test_solves_that_time_out_inside_the_cap_are_timeouts_not_cap_days(made_up, monkeypatch):
    force_timeouts(monkeypatch, TINY)
    run, tables, out = made_up
    limits = ["--lp24-seconds", str(TINY), "--other-seconds", str(TINY), "--box-seconds", str(TINY)]
    assert o3_run.main(_arguments(run, tables, *limits, "--lp27-days", "3", days="4")) == 0
    document = _document(out)
    assert document["lp27_cap"][UNIT]["not_reached"] == 0
    hourly = document["units"][UNIT]["hourly"]
    for key in LP27_KEYS:
        assert hourly[key]["days"] == 3
        assert hourly[key]["missing"] == {"timeout": 3}
        assert hourly[key]["kwh"]["mean"] is None
    assert CAP_REACHED not in document["missing_values"]


# --- 5. the numbers of the plan and the options ----------------------------------------------


def test_the_defaults_are_the_plans_numbers_and_departures_are_named():
    assert (o3_run.PLAN_DAYS, o3_run.PLAN_LP27_DAYS, o3_run.PLAN_CAP_SECONDS) == (200, 100, 7200.0)
    parsed = o3_run.parser().parse_args(
        ["--units", UNIT, "--days", "200", "--seed", "1", "--base-set", "day-average"]
    )
    assert (parsed.lp27_days, parsed.lp27_cap_seconds) == (100, 7200.0)
    assert (parsed.lp24_seconds, parsed.other_seconds, parsed.box_seconds) == (
        LP24_SECONDS,
        OTHER_SECONDS,
        BOX_SECONDS,
    )
    assert o3_run.departures(parsed) == []
    quick = o3_run.parser().parse_args(
        ["--units", UNIT, "--days", "6", "--seed", "1", "--base-set", "day-average"]
        + ["--lp27-days", "3", "--lp27-cap-seconds", "60", "--box-seconds", "5"]
    )
    assert o3_run.departures(quick) == [
        "days per unit 6, the plan's 200",
        "LP (27) days per unit 3, the plan's 100",
        "LP (27) cap per unit (s) 60, the plan's 7200",
        "box time limit (s) 5, the plan's 30",
    ]


def test_the_summary_names_the_departures_from_the_plan(made_up):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, "--lp27-days", "3", days="4")) == 0
    document = _document(out)
    assert document["arguments"]["departures_from_plan"] == [
        "days per unit 4, the plan's 200",
        "LP (27) days per unit 3, the plan's 100",
    ]
    lines = (out / o3_run.SUMMARY_NAME).read_text().splitlines()
    assert lines[4].startswith("LP (27) DAYS: LP (27) is run on the first 3 days of each unit's")
    assert lines[5] == (
        "DEPARTURES FROM THE PLAN'S NUMBERS (sections 11 O3 and 13): days per unit 4, the plan's "
        "200; LP (27) days per unit 3, the plan's 100."
    )


@pytest.mark.parametrize(
    "extra",
    [
        ("--lp27-days", "0"),
        ("--lp27-days", "-3"),
        ("--lp27-cap-seconds", "0"),
        ("--lp27-cap-seconds", "-5"),
        ("--lp27-cap-seconds", "nan"),
        ("--lp27-cap-seconds", "inf"),
    ],
)
def test_lp27_days_and_cap_out_of_range_are_refused(made_up, extra, capsys):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, *extra, days="2")) == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert not any(out.iterdir())


def test_the_worker_count_does_not_change_the_results_with_fewer_lp27_days(
    made_up, tmp_path, monkeypatch
):
    run, tables, out = made_up
    assert o3_run.main(_arguments(run, tables, "--lp27-days", "2", days="4")) == 0
    one = ((out / o3_run.RESULTS_NAME).read_text(), (out / o3_run.DAYS_NAME).read_text())
    two_out = tmp_path / "two"
    two_out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(two_out))
    arguments = _arguments(run, tables, "--lp27-days", "2", "--workers", "2", days="4")
    assert o3_run.main(arguments) == 0
    two = ((two_out / o3_run.RESULTS_NAME).read_text(), (two_out / o3_run.DAYS_NAME).read_text())
    assert one == two
    assert math.isfinite(float(_document(out)["arguments"]["lp27_cap_seconds"]))

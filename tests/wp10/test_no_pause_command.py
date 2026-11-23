"""The no-pause sensitivity of the comparison command (`compare_run --no-pause-share`), on made-up
gate units with two service hours (`compare_fake.two_hour_cars`, five cars a day).

Units: the first gate unit holds 2 days and cannot be split; the second holds 28 days of site A;
the pooled third holds 14 days of site B and 14 of site C on the same dates, and each car of site
C has the session identifier of the car of site B in the same place, so the two sites' cars differ
only by their site; 20 days calibrate in each of 6 splits. Methods: `re-solve`, `fixed-plan`,
`llf`. Every drawn car is recounted here with this file's own hash rule (`no_pause` module text),
from the sites and session identifiers of the made-up tables.

1. At share 1/3 the run ends with exit 0 and writes its five files under the names of
   `compare_run.output_names` and no file under the names of the run without the option; every
   file names the sensitivity: the results (`no_pause` with share and seed, the label), the
   summary (its first line and one line per unit that can be split), the split table's header
   (the ceilings ending `-no-pause-cars-removed`), the timing file (share and seed) and the
   forecast table by its name. The forecast table is the same bytes as the run without the option.
2. The cars given to `gate_day.prepare_day` as `no_pause` are exactly the recounted ones, for
   every counting day of every unit that can be split and for both variants alike; every prepared
   day evaluated for the re-solve rung or the controllers, at every scale, leaves exactly those
   cars unsteered with the reason `gate_day.NO_PAUSE`, and its audited paths are the day's. On the
   pooled unit some car of site B and the car of site C with the same session identifier and
   date are drawn differently.
3. The written counts equal a recount: per unit `no_pause` (from this file's draw), each
   variant's `unsteered_cars` (every reason, from the prepared days of that variant) and the
   summary's line for the unit, whole. The timeouts are those of the run without the option plus
   `best_causal_ceilings_no_pause_cars_removed`; a timeout of a ceiling of the fleet without the
   drawn cars is counted under that key alone and leaves that day's ceiling missing. A timeout of
   either ceiling is counted only for the outcome "timeout" (not "no car").
4. The results, the summary, the split table and the forecast table at share 1/3 are the same
   bytes for 1 and 3 workers.
5. With `--no-pause-share 0` (with or without a seed) every file but the timing file is the same
   bytes as the run without the option; no file of the run without the option holds "no-pause"
   (in any case), "no_pause" or "cannot pause". The run without the option gives
   `gate_day.prepare_day` no drawn car on any day, and no car has the reason `gate_day.NO_PAUSE`.
6. Another seed draws other cars, and its results and timing file say so.
7. Ceilings: the ceilings named as in the matrix are the whole fleet's (their split-table columns
   and the gate reading's mean ceilings equal those of the run without the option); the columns
   ending `-no-pause-cars-removed` are the means over each split's test days of
   `gate_day.day_ceilings` of the day without the recounted cars, at most the whole fleet's and
   below it on some split.
8. A share other than 0 and 1/3 ("0.5", "1/2", "0.33") is refused with exit 2 and a message, and
   no file is written; `compare_run.run` refuses a share not of the two, and its defaults are
   share 0 and seed 1.
9. The written names and words, as text here: the five file names, the names and hashes of the
   two tables in the results, the label, the keys of `no_pause`, the run's ceiling names, the
   reason "cannot pause (no-pause sensitivity)" among the sorted keys of `unsteered_cars`, the
   summary's first line whole and the unit's line right under its heading; a clause each of the
   counts definition and of readings N4 and N5.
10. The unit line of the summary for 0 and for some cars drawn, for a unit without cars, and for
   0, 1 and 2 days with every car drawn; no line without the key `no_pause`.
11. A session identifier holding `|` on a counting day of the second unit stops the run at share
   1/3 with exit 2, one line naming that unit, and no file; one in the unsplit first unit is not
   named; at share 0 the same units run with exit 0.
12. The keys of the results without the option (the run without it, at share 0 and at share 0
   with seed 9), level by level (document, run, timeouts, units, variants, methods, tiers, gate
   reading and its ceilings), are those written at 22930ef with the variants' record
   `measured_on` added (`compare_run` module text), as literal sets here, and so are the
   file names, the split table's header and the timing file's keys. At share 1/3 they are those
   plus `no_pause` (document and split units), `unsteered_cars` (variants),
   `best_causal_ceilings_no_pause_cars_removed` (timeouts) and the ceilings ending
   `-no-pause-cars-removed` (run, gate reading, split table's header), and the timing file adds
   `no_pause`; without those, the key paths of the whole document (below the keys of outcomes
   and reasons found in the data) are exactly those of the run without the option.
"""

import csv
import dataclasses
import hashlib
import inspect
import io
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from compare_fake import NAMES, TWO_HOURS, two_hour_cars
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import (
    compare_run,
    compare_splits,
    compare_summary,
    dial_choice,
    gate_config,
    gate_day,
    gate_run,
    no_pause,
    run_verify,
)
from fleetreserve.audit_paths import audited_paths
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_calibrate import ceiling_values

SPLITS = 6
METHODS = ["re-solve", "fixed-plan", "llf"]
SPLIT_UNITS = GATE_UNITS[1:]
SUFFIX = "-no-pause-cars-removed"


def _own_draw(site, session, seed) -> bool:
    digest = hashlib.sha256(f"no-pause|{seed}|{site}|{session}".encode()).digest()
    return 3 * int.from_bytes(digest[:8], "big") < 2**64


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: two_hour_cars(28),
        GATE_UNITS[2]: two_hour_cars(14, "B")
        + [
            dataclasses.replace(c, session=c.session.replace("C-", "B-", 1))
            for c in two_hour_cars(14, "C")
        ],
    }
    return _write(folder, units)


def _write(folder: Path, units: dict):
    run, tables = write_build(folder, units, TWO_HOURS)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _with_separator(cars: list, k: int) -> list:
    return [
        dataclasses.replace(c, session=f"{c.session}|x") if i == k else c
        for i, c in enumerate(cars)
    ]


@pytest.fixture(scope="module")
def separator_build(tmp_path_factory):
    """The units of `build` but a car of the unsplit first unit and one of the second unit hold
    `|` in their session identifiers."""
    units = {
        GATE_UNITS[0]: _with_separator(gate_cars(2), 1),
        GATE_UNITS[1]: _with_separator(two_hour_cars(28), 7),
        GATE_UNITS[2]: two_hour_cars(14, "B") + two_hour_cars(14, "C"),
    }
    return _write(tmp_path_factory.mktemp("separator-build"), units)


def _run(build, folder: Path, *extra, record=None):
    run, tables, matrix = build
    folder.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        if record is not None:
            _record(monkeypatch, record)
        arguments = ["--model", "empirical", "--dial", "0.03", "--matrix", str(matrix)]
        arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
        code = compare_run.main([*arguments, "--methods", *METHODS, *extra])
    return code, folder


def _record(monkeypatch, record):
    prepare = gate_day.prepare_day
    evaluate_re_solve = compare_run._evaluate_task
    evaluate_controllers = compare_run._compare_task

    def prepare_day(truth, forecaster, configuration, settings, no_pause=()):
        key = (truth.day.unit, truth.record.site, truth.record.date)
        found = prepare(truth, forecaster, configuration, settings, no_pause=no_pause)
        record["prepared"].append((key, configuration.name, tuple(no_pause), found))
        return found

    def re_solve(job):
        record["evaluated"].append(job[0])
        return evaluate_re_solve(job)

    def controllers(job):
        record["evaluated"].append(job[0])
        return evaluate_controllers(job)

    monkeypatch.setattr(gate_day, "prepare_day", prepare_day)
    monkeypatch.setattr(compare_run, "_evaluate_task", re_solve)
    monkeypatch.setattr(compare_run, "_compare_task", controllers)


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    top = tmp_path_factory.mktemp("runs")
    record = {"prepared": [], "evaluated": []}
    plain = {"prepared": [], "evaluated": []}
    found = {
        "third": _run(build, top / "third", "--no-pause-share", "1/3", record=record),
        "third-w3": _run(build, top / "third-w3", "--no-pause-share", "1/3", "--workers", "3"),
        "plain": _run(build, top / "plain", record=plain),
        "zero": _run(build, top / "zero", "--no-pause-share", "0"),
        "zero-seed": _run(
            build, top / "zero-seed", "--no-pause-share", "0", "--no-pause-seed", "9"
        ),
        "seed2": _run(build, top / "seed2", "--no-pause-share", "1/3", "--no-pause-seed", "2"),
    }
    return found, record, plain


@pytest.fixture(scope="module")
def units(build):
    _, tables, matrix = build
    loaded = gate_config.load(matrix)
    return {
        u: load_unit_days(
            tables, u, rho=loaded.rho, no_fit=loaded.no_fit, small_fleet=loaded.small_fleet
        )
        for u in GATE_UNITS
    }


@pytest.fixture(scope="module")
def recount(units):
    """unit -> list over counting days of the drawn positions, by this file's own rule."""
    return {
        u: [
            tuple(
                i
                for i, car in enumerate(t.fleet)
                if _own_draw(car.site, car.session_id, no_pause.DEFAULT_SEED)
            )
            for t in units[u].days
        ]
        for u in SPLIT_UNITS
    }


def _document(folder: Path, share="1/3") -> dict:
    names = compare_run.output_names(no_pause.read_share(share))
    return json.loads((folder / names["results"]).read_text())


def test_the_files_of_the_sensitivity_name_it(runs):
    code, out = runs[0]["third"]
    assert code == 0
    names = compare_run.output_names(no_pause.read_share("1/3"))
    plain = compare_run.output_names(no_pause.read_share("0"))
    assert sorted(p.name for p in out.iterdir()) == sorted(names.values())
    assert not set(plain.values()) & set(names.values())
    document = _document(out)
    assert document["no_pause"]["share"] == "1/3"
    assert document["no_pause"]["seed"] == no_pause.DEFAULT_SEED
    assert set(document["no_pause"]["readings"]) == {"N1", "N2", "N3", "N4", "N5"}
    assert "NO-PAUSE SENSITIVITY" in document["label"]
    assert document["definitions"]["label"] == document["label"]
    assert document["forecast_table"]["file"] == names["table"]
    assert document["split_table"]["file"] == names["splits"]
    summary = (out / names["summary"]).read_text().splitlines()
    assert summary[0].startswith("NO-PAUSE SENSITIVITY (plan section 9")
    assert "share 1/3 of the cars (seed 1)" in summary[0]
    header = (out / names["splits"]).read_text().splitlines()[0]
    assert f"mean_best-causal{SUFFIX}_ceiling_kwh_per_test_day" in header.split(",")
    timing = json.loads((out / names["timing"]).read_text())
    assert timing["no_pause"] == {"share": "1/3", "seed": no_pause.DEFAULT_SEED}
    assert "no-pause" in names["table"]
    plain_table = runs[0]["plain"][1] / plain["table"]
    assert (out / names["table"]).read_bytes() == plain_table.read_bytes()


def test_the_drawn_cars_are_the_recounted_ones_for_both_variants(runs, units, recount):
    record = runs[1]
    by_day = {}
    for key, configuration, drawn, _ in record["prepared"]:
        by_day.setdefault(key, {})[configuration] = drawn
    for unit in SPLIT_UNITS:
        for truth, expected in zip(units[unit].days, recount[unit], strict=True):
            key = (truth.day.unit, truth.record.site, truth.record.date)
            assert by_day[key] == {NAMES[2]: expected, NAMES[3]: expected}
    assert sum(len(d) for u in SPLIT_UNITS for d in recount[u]) > 0
    pooled = {(t.record.site, t.record.date): t for t in units[GATE_UNITS[2]].days}
    apart = 0
    for (site, date), truth in pooled.items():
        if site != "B":
            continue
        other = pooled[("C", date)]
        for b, c in zip(truth.fleet, other.fleet, strict=True):
            assert b.session_id == c.session_id
            apart += _own_draw("B", b.session_id, 1) != _own_draw("C", c.session_id, 1)
    assert apart > 0


def test_every_evaluated_day_leaves_the_drawn_cars_unsteered(runs, units, recount):
    record = runs[1]
    expected = {}
    for unit in SPLIT_UNITS:
        for truth, drawn in zip(units[unit].days, recount[unit], strict=True):
            expected[(truth.day.unit, truth.record.site, truth.record.date)] = drawn
    seen = set()
    for prepared in record["evaluated"]:
        key = (prepared.unit, prepared.site, prepared.date)
        drawn = expected[key]
        assert tuple(i for i, why in prepared.unsteered if why == gate_day.NO_PAUSE) == drawn
        assert not set(drawn) & set(prepared.steered)
        n = prepared.day.n_service
        assert np.array_equal(prepared.paths, audited_paths(prepared.unit, prepared.day.date, n))
        seen.add(key)
    assert all(len(expected[k]) < 5 for k in seen)  # a day with every car drawn abstains
    assert len(seen) > 20 and any(expected[k] for k in seen)


def test_the_written_counts_equal_the_recount(runs, units, recount):
    code, out = runs[0]["third"]
    document = _document(out)
    summary = (out / compare_run.output_names(no_pause.read_share("1/3"))["summary"]).read_text()
    reasons = {}
    for _, configuration, _, prepared in runs[1]["prepared"]:
        found = reasons.setdefault((prepared.unit, configuration), Counter())
        found.update(why for _, why in prepared.unsteered)
    for unit in SPLIT_UNITS:
        draws = recount[unit]
        cars = sum(len(t.day.cars) for t in units[unit].days)
        drawn = sum(len(d) for d in draws)
        every = sum(1 for t, d in zip(units[unit].days, draws, strict=True) if len(d) == 5)
        entry = document["units"][unit]
        assert entry["no_pause"] == {
            "cars": cars,
            "cars_drawn": drawn,
            "days_every_car_drawn": every,
        }
        for variant in entry["variants"].values():
            assert variant["unsteered_cars"][gate_day.NO_PAUSE] == drawn
            assert variant["unsteered_cars"] == dict(reasons[(unit, variant["configuration"])])
        assert len(entry["variants"]["forecast"]["unsteered_cars"]) > 1  # another reason too
        if every == 1:
            days = "1 day with every car drawn, which abstains"
        else:
            days = f"{every} days with every car drawn, which abstain"
        share = f"{100 * drawn / cars:.0f}%"
        line = f" cars that cannot pause: {drawn} of {cars} cars of the counting days ({share}); "
        assert line + days in summary.splitlines()
    assert "no_pause" not in document["units"][GATE_UNITS[0]]
    plain = _document(runs[0]["plain"][1], "0")["timeouts"]
    assert document["timeouts"] == {**plain, "best_causal_ceilings_no_pause_cars_removed": 0}


class _Pool:
    def __init__(self, found):
        self.found = found

    def map(self, task, jobs):
        return [self.found[i] for i, _ in enumerate(jobs)]


def test_a_timeout_of_a_ceiling_without_the_drawn_cars_is_counted_apart(build):
    matrix = gate_config.load(build[2])
    found = [
        gate_day.Ceilings(3.0, 2.0, "optimal", 0.0),
        gate_day.Ceilings(4.0, None, "timeout", 0.0),
    ]
    timeouts = Counter()
    arrays = compare_run._reduced_ceilings(_Pool(found), "u", [(0,), ()], matrix, timeouts)
    assert dict(timeouts) == {"best_causal_ceilings_no_pause_cars_removed": 1}
    hindsight, best = (f"{c}{SUFFIX}" for c in matrix.ceilings)
    assert list(arrays) == [hindsight, best]
    assert arrays[hindsight].tolist() == [3.0, 4.0]
    assert arrays[best][0] == 2.0 and np.isnan(arrays[best][1])


def test_a_whole_fleet_ceiling_counts_timeouts_only(build):
    matrix = gate_config.load(build[2])
    found = [
        gate_day.Ceilings(3.0, 2.0, "optimal", 0.0),
        gate_day.Ceilings(0.0, 0.0, "no car", 0.0),
        gate_day.Ceilings(4.0, None, "timeout", 0.0),
    ]
    timeouts = Counter()
    best, arrays = compare_run._ceilings(_Pool(found), "u", 3, matrix, timeouts)
    assert dict(timeouts) == {"best_causal_ceilings": 1} and best == [2.0, 0.0, None]
    assert list(arrays) == list(matrix.ceilings)
    timeouts = Counter()
    compare_run._reduced_ceilings(_Pool(found), "u", [(), (), ()], matrix, timeouts)
    assert dict(timeouts) == {"best_causal_ceilings_no_pause_cars_removed": 1}


def test_the_written_names_and_words(runs, build):
    code, out = runs[0]["third"]
    stem = "compare-no-pause-third"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{stem}{end}"
        for end in ("-forecasts.csv", "-splits.csv", "-summary.txt", "-timing.json", ".json")
    )
    document = _document(out)
    assert document["label"] == (
        f"{compare_run.LABEL}; NO-PAUSE SENSITIVITY of plan section 9: a seeded share 1/3 of the "
        "cars (seed 1) charge as logged, without flexibility, and are left unsteered"
    )
    for key, end in (("forecast_table", "-forecasts.csv"), ("split_table", "-splits.csv")):
        assert document[key]["file"] == f"{stem}{end}"
        digest = hashlib.sha256((out / f"{stem}{end}").read_bytes()).hexdigest()
        assert document[key]["sha256"] == digest
    assert set(document["no_pause"]) == {"share", "seed", "readings", "counts"}
    assert document["no_pause"]["counts"].startswith("units.<unit>.no_pause, over the unit's")
    assert "a car on several counting days once per day; " in document["no_pause"]["counts"]
    readings = document["no_pause"]["readings"]
    assert "plan section 9 names the sensitivity under two-way charging" in readings["N4"]
    assert "a drawn car is only not forecast on a day it is drawn" in readings["N5"]
    matrix = gate_config.load(build[2])
    assert document["run"]["ceilings"] == [*matrix.ceilings, *(c + SUFFIX for c in matrix.ceilings)]
    for unit in SPLIT_UNITS:
        for variant in document["units"][unit]["variants"].values():
            assert "cannot pause (no-pause sensitivity)" in variant["unsteered_cars"]
            assert list(variant["unsteered_cars"]) == sorted(variant["unsteered_cars"])
    summary = (out / f"{stem}-summary.txt").read_text().splitlines()
    assert summary[0] == (
        "NO-PAUSE SENSITIVITY (plan section 9; Brinkel et al. (2024)): a seeded share 1/3 of the "
        "cars (seed 1) charge as logged, without flexibility, and are left unsteered by every "
        "method; this is not the run without the sensitivity. of ceiling: share of the whole "
        "fleet's best-causal ceiling, the cars that cannot pause included; the gate-reading table "
        "also gives the shares of the ceilings of the fleet without those cars (names ending "
        "-no-pause-cars-removed)."
    )
    for unit in SPLIT_UNITS:
        at = summary.index(next(x for x in summary if x.startswith(f"{unit} (")))
        assert summary[at + 1].startswith(" cars that cannot pause: ")


@pytest.mark.parametrize(
    ("counts", "line"),
    [
        ((6, 0, 0), "0 of 6 cars of the counting days (0%); 0 days with every car drawn, which "),
        ((0, 0, 0), "0 of 0 cars of the counting days (-); 0 days with every car drawn, which "),
        ((9, 3, 1), "3 of 9 cars of the counting days (33%); 1 day with every car drawn, which "),
        ((9, 7, 2), "7 of 9 cars of the counting days (78%); 2 days with every car drawn, which "),
    ],
)
def test_the_unit_line_of_the_summary(counts, line):
    cars, drawn, every = counts
    entry = {"no_pause": {"cars": cars, "cars_drawn": drawn, "days_every_car_drawn": every}}
    end = "abstains" if every == 1 else "abstain"
    assert compare_summary._no_pause_unit(entry) == [f" cars that cannot pause: {line}{end}"]
    assert compare_summary._no_pause_unit({}) == []
    assert compare_summary._no_pause_head({}) == []


def test_run_refuses_a_share_not_of_the_plan_and_defaults_to_none_drawn():
    with pytest.raises(no_pause.NoPauseError):
        compare_run.run(None, {}, None, {}, [], splits=1, workers=1, no_pause_share=Fraction(1, 2))
    parameters = inspect.signature(compare_run.run).parameters
    assert parameters["no_pause_share"].default == 0
    assert parameters["no_pause_seed"].default == 1


def test_the_files_do_not_depend_on_the_number_of_workers(runs):
    names = compare_run.output_names(no_pause.read_share("1/3"))
    one, three = runs[0]["third"][1], runs[0]["third-w3"][1]
    assert runs[0]["third-w3"][0] == 0
    for kind in ("results", "summary", "splits", "table"):
        assert (one / names[kind]).read_bytes() == (three / names[kind]).read_bytes(), kind


def test_share_zero_is_the_run_without_the_option(runs):
    names = compare_run.output_names(no_pause.read_share("0"))
    plain = runs[0]["plain"][1]
    assert runs[0]["plain"][0] == runs[0]["zero"][0] == runs[0]["zero-seed"][0] == 0
    prepared = runs[2]["prepared"]
    assert len(prepared) > 20 and all(drawn == () for _, _, drawn, _ in prepared)
    assert all(why != gate_day.NO_PAUSE for *_, p in prepared for _, why in p.unsteered)
    for other in ("zero", "zero-seed"):
        folder = runs[0][other][1]
        assert sorted(p.name for p in folder.iterdir()) == sorted(names.values())
        for kind in ("results", "summary", "splits", "table"):
            assert (plain / names[kind]).read_bytes() == (folder / names[kind]).read_bytes()
    for kind in ("results", "summary", "splits", "table", "timing"):
        assert b"no-pause" not in (plain / names[kind]).read_bytes().lower()
        assert b"no_pause" not in (plain / names[kind]).read_bytes()
        assert b"cannot pause" not in (plain / names[kind]).read_bytes()


def test_another_seed_draws_other_cars(runs, units):
    first = _document(runs[0]["third"][1])
    second = _document(runs[0]["seed2"][1])
    assert second["no_pause"]["seed"] == 2
    timing = runs[0]["seed2"][1] / compare_run.output_names(no_pause.read_share("1/3"))["timing"]
    assert json.loads(timing.read_text())["no_pause"] == {"share": "1/3", "seed": 2}
    counts = [(first["units"][u]["no_pause"], second["units"][u]["no_pause"]) for u in SPLIT_UNITS]
    other = [
        sum(_own_draw(c.site, c.session_id, 2) for t in units[u].days for c in t.fleet)
        for u in SPLIT_UNITS
    ]
    assert [b["cars_drawn"] for _, b in counts] == other
    differ = sum(
        _own_draw(c.site, c.session_id, 1) != _own_draw(c.site, c.session_id, 2)
        for u in SPLIT_UNITS
        for t in units[u].days
        for c in t.fleet
    )
    assert differ > 20


def _rows(folder: Path, share: str) -> list[dict]:
    names = compare_run.output_names(no_pause.read_share(share))
    return list(csv.DictReader(io.StringIO((folder / names["splits"]).read_text())))


def test_the_ceilings_of_both_fleets(runs, build, units, recount):
    third, plain = runs[0]["third"][1], runs[0]["plain"][1]
    rows, plain_rows = _rows(third, "1/3"), _rows(plain, "0")
    assert len(rows) == len(plain_rows)
    matrix = gate_config.load(build[2])
    lower = []
    for row, other in zip(rows, plain_rows, strict=True):
        for name in matrix.ceilings:
            column = f"mean_{name}_ceiling_kwh_per_test_day"
            assert row[column] == other[column]
    for unit in SPLIT_UNITS:
        days = units[unit]
        found = [
            gate_day.day_ceilings(no_pause.without(t.day, d))
            for t, d in zip(days.days, recount[unit], strict=True)
        ]
        splits = dial_choice.split_indices(unit, days, matrix, range(1, SPLITS + 1))
        for name in matrix.ceilings:
            per_day = np.array([ceiling_values(c, name) for c in found], dtype=float)
            column = f"mean_{name}{SUFFIX}_ceiling_kwh_per_test_day"
            for seed, _, test in splits:
                mine = [r for r in rows if r["unit"] == unit and int(r["seed"]) == seed]
                assert mine and {r[column] for r in mine} == {repr(float(per_day[test].mean()))}
            for r in (r for r in rows if r["unit"] == unit):
                reduced = float(r[column])
                full = float(r[f"mean_{name}_ceiling_kwh_per_test_day"])
                assert reduced <= full
                lower.append(reduced < full)
    assert any(lower)
    document, other = _document(third), _document(plain, "0")
    for unit in SPLIT_UNITS:
        for method in METHODS:
            mine = document["units"][unit]["gate_reading"][method]["ceilings"]
            theirs = other["units"][unit]["gate_reading"][method]["ceilings"]
            for name in matrix.ceilings:
                for field in ("mean_ceiling_kwh_per_test_day", "splits_with_ceiling"):
                    assert mine[name][field] == theirs[name][field]
            assert set(mine) == {*matrix.ceilings, *(n + SUFFIX for n in matrix.ceilings)}


@pytest.mark.parametrize("share", ["0.5", "1/2", "0.33"])
def test_another_share_is_refused(build, tmp_path, capsys, share):
    code, out = _run(build, tmp_path / "out", "--no-pause-share", share)
    assert code == 2
    assert "COULD NOT RUN" in capsys.readouterr().out
    assert list(out.iterdir()) == []


def test_an_identifier_holding_the_separator_stops_the_run_at_one_third(
    separator_build, tmp_path, capsys
):
    code, out = _run(separator_build, tmp_path / "third", "--no-pause-share", "1/3")
    printed = [x for x in capsys.readouterr().out.splitlines() if x.startswith("COULD NOT RUN")]
    assert code == 2 and list(out.iterdir()) == []
    assert printed == [
        f"COULD NOT RUN: unit {GATE_UNITS[1]}: a site or session identifier holds '|': 'A', "
        "'A-1-2|x'; --no-pause-share 1/3 draws each car from the text "
        "no-pause|<seed>|<site>|<session id>, so no site or session identifier may hold '|'"
    ]
    code, out = _run(separator_build, tmp_path / "zero", "--no-pause-share", "0")
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(PLAIN_FILES)


PLAIN_FILES = (
    "compare.json",
    "compare-summary.txt",
    "compare-forecasts.csv",
    "compare-timing.json",
    "compare-splits.csv",
)
# The keys of compare.json without the option, level by level, as `main` (22930ef) writes them,
# with the variants' record `measured_on` added.
DOCUMENT_KEYS = {
    "configurations",
    "definitions",
    "dial",
    "failed",
    "fit_units",
    "forecast_table",
    "inputs",
    "label",
    "matrix",
    "methods",
    "model",
    "paired_differences",
    "run",
    "split_table",
    "timeouts",
    "units",
}
RUN_KEYS = {
    "calibration_half_below",
    "ceilings",
    "kappa",
    "n_calibration",
    "scales",
    "seeds",
    "splits",
    "tiers",
    "variants",
}
TIMEOUT_KEYS = {
    "best_causal_ceilings",
    "certificates",
    "oracle_certificates",
    "oracle_re_solves",
    "re_solves",
}
UNSPLIT_UNIT_KEYS = {
    "calibration_days_used",
    "counting_days",
    "split_of_the_gate_run",
    "status",
    "test_days",
    "variants",
}
SPLIT_UNIT_KEYS = UNSPLIT_UNIT_KEYS | {"gate_reading"}
VARIANT_KEYS = {"configuration", "evaluations", "measured_on", "methods", "offering_days"}
TIER_B_KEYS = {
    "mean_offer_kwh_per_test_day",
    "mean_scale",
    "mean_share_of_best_causal_ceiling",
    "mean_test_driver_loss",
    "mean_test_slot_loss",
    "offer_p10",
    "offer_p50",
    "offer_p90",
    "share_of_splits_that_offer",
    "slot_loss_over_limit",
    "splits",
    "splits_at_scale_zero_by_reason",
    "splits_that_offer",
    "splits_with_ceiling_share",
    "splits_with_test_losses",
}
TIER_A_KEYS = TIER_B_KEYS | {"driver_loss_over_limit"}
GATE_KEYS = {
    "ceilings",
    "mean_offer_kwh_per_test_day",
    "mean_oracle_offer_kwh_per_test_day",
    "oracle_to_offer_ratio",
    "oracle_to_offer_ratio_note",
    "splits",
    "tier",
}
GATE_CEILING_KEYS = {
    "mean_ceiling_kwh_per_test_day",
    "mean_offer_kwh_per_test_day_on_those_splits",
    "share_of_ceiling",
    "splits_left_out_ceiling_missing",
    "splits_with_ceiling",
}
# dictionaries whose keys are outcomes or reasons found in the data, not names the code fixes
DATA_KEYED = {
    "splits_at_scale_zero_by_reason",
    "re_solves_of_the_one_way_rung",
    "controller_solver_outcomes",
    "unsteered_cars",
}


def _levels(document: dict, ceilings, extra=lambda level: set()):
    """Checks every level of `document` against the keys without the option plus `extra(level)`."""
    assert set(document) == DOCUMENT_KEYS | set(extra("document"))
    assert set(document["run"]) == RUN_KEYS
    assert document["run"]["ceilings"] == [*ceilings, *extra("ceilings")]
    assert set(document["timeouts"]) == TIMEOUT_KEYS | set(extra("timeouts"))
    assert set(document["units"]) == set(GATE_UNITS)
    assert set(document["units"][GATE_UNITS[0]]) == UNSPLIT_UNIT_KEYS
    assert document["units"][GATE_UNITS[0]]["variants"] == {}
    for unit in SPLIT_UNITS:
        entry = document["units"][unit]
        assert set(entry) == SPLIT_UNIT_KEYS | set(extra("unit"))
        assert set(entry["variants"]) == {"forecast", "oracle-departure"}
        for variant in entry["variants"].values():
            assert set(variant) == VARIANT_KEYS | set(extra("variant"))
            assert set(variant["methods"]) == set(METHODS)
            for method in variant["methods"].values():
                assert set(method) == {"A", "B"}
                assert set(method["A"]) == TIER_A_KEYS and set(method["B"]) == TIER_B_KEYS
        assert set(entry["gate_reading"]) == set(METHODS)
        for method in entry["gate_reading"].values():
            assert set(method) == GATE_KEYS
            assert set(method["ceilings"]) == {*ceilings, *extra("ceilings")}
            assert all(set(c) == GATE_CEILING_KEYS for c in method["ceilings"].values())


def _paths(value, at=()):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield (*at, key)
            if key not in DATA_KEYED:
                yield from _paths(inner, (*at, key))
    elif isinstance(value, list):
        for inner in value:
            yield from _paths(inner, (*at, "[]"))


def test_the_keys_and_files_without_the_option_are_those_of_main(runs, build):
    ceilings = list(gate_config.load(build[2]).ceilings)
    header = [*compare_splits.BASE_COLUMNS, *(compare_splits.ceiling_column(c) for c in ceilings)]
    for name in ("plain", "zero", "zero-seed"):
        folder = runs[0][name][1]
        assert sorted(p.name for p in folder.iterdir()) == sorted(PLAIN_FILES)
        _levels(json.loads((folder / "compare.json").read_text()), ceilings)
        assert (folder / "compare-splits.csv").read_text().splitlines()[0] == ",".join(header)
        timing = json.loads((folder / "compare-timing.json").read_text())
        assert set(timing) == {"total_seconds", "units"}
    assert len(header) == 18


def test_the_keys_and_files_at_one_third_are_those_and_the_no_pause_keys(runs, build):
    ceilings = list(gate_config.load(build[2]).ceilings)
    reduced = [c + SUFFIX for c in ceilings]
    added = {
        "document": {"no_pause"},
        "ceilings": reduced,
        "timeouts": {"best_causal_ceilings_no_pause_cars_removed"},
        "unit": {"no_pause"},
        "variant": {"unsteered_cars"},
    }
    third, plain = runs[0]["third"][1], runs[0]["plain"][1]
    document = _document(third)
    _levels(document, ceilings, lambda level: added.get(level, ()))
    documented = [("no_pause",), ("timeouts", "best_causal_ceilings_no_pause_cars_removed")]
    for unit in SPLIT_UNITS:
        documented.append(("units", unit, "no_pause"))
        for variant in ("forecast", "oracle-departure"):
            documented.append(("units", unit, "variants", variant, "unsteered_cars"))
        for method in METHODS:
            for name in reduced:
                documented.append(("units", unit, "gate_reading", method, "ceilings", name))
    mine = set(_paths(document))
    assert all(p in mine for p in documented)
    left = {p for p in mine if not any(p[: len(d)] == d for d in documented)}
    assert left == set(_paths(_document(plain, "0")))
    names = compare_run.output_names(no_pause.read_share("1/3"))
    header = (third / names["splits"]).read_text().splitlines()[0].split(",")
    plain_header = (plain / "compare-splits.csv").read_text().splitlines()[0].split(",")
    assert header == plain_header + [compare_splits.ceiling_column(c) for c in reduced]
    timing = json.loads((third / names["timing"]).read_text())
    assert set(timing) == {"total_seconds", "units", "no_pause"}

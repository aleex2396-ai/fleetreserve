"""The diagnosis command (`asa_diagnosis_run`) on the made-up gate units of `gate_fake`.

Units (`gate_fake.gate_cars`, the cars of the comparison tests, with two service hours so that a
later hour depends on earlier ones): the first gate unit with 2 days (it cannot be split), the
second with 24 days and an early leaver on every fifth day, the pooled unit with 12 days from
each of two sites; 20 calibration days, the dial 0.03.

Runs: "rule" (the defaults: `asa-es` against `llf`, the scale rule over 6 splits, 1 worker), the
same with 2 workers, "fixed" (`--scale 1 --reference fixed-plan`), "swap" (`--scale 1 --method
llf --reference asa-es`), "two" (`--scale 1 --reference fixed-plan --days 2`) and "boxes"
(`--scale 1 --reference diagonal-split --days 2`, a reference whose counts include boxes), "four"
(`--scale 1 --reference fixed-plan --days 4`) and "four3" (the same with 3 workers). Every run but
"two" passes `--splits 6`; "two" runs on the default.

1. Every run ends with exit 0 and writes the results, the hour table, the state-car table, the
   summary, the forecast table and the timing file; the unit that cannot be split is reported
   with its status and not diagnosed; every unit that can be split has both variants. The rule
   run, on the defaults, diagnoses up to 50 days under a 300 s limit, and the rules written name
   the splits and days of each run.
2. The scale rule: each variant of the rule run is diagnosed at the scale recomputed here from
   the reference's tier A calibration (`compare_run.measure_controllers`, `dial_splits`), or is
   "no reference offer" when the reference offers on no split; both statuses occur.
3. Every key of the results document of every run is defined: every key of every dictionary
   outside `definitions` is a key of `definitions`, except the keys of the maps named in
   `definitions.name_maps` (a check shown to find a key added on purpose); every column of both
   tables has a definitions entry, the hour table's header is `HOURS_COLUMNS` and the state-car
   table's is unit, variant, day, day_index, path, hour, then the fields of `StateCar`; booleans
   are written "True" or "False".
4. The counts add up: per unit and variant, the causes add up to the method-only hours, which
   equal the hour table's rows of that unit and variant; the four groups add up to the diagnosed
   days times 8 paths times 2 service hours; each row of cause "state" has one car row per car of
   its day, numbered from 0; each row's `day_index` is the index of its `day` among the unit's
   counting days. The "fixed" and "swap" runs have rows; every row of "swap" (a method
   that solves no programme) of cause "choice" names the goal "rule".
5. The days diagnosed are the first `--days` offering days in the order of D2
   (`asa_diagnosis_run.day_order`); "two" diagnoses the first two.
6. The files are byte-identical for 1 and 2 workers. For 1 and 3 workers ("four", "four3"), each
   day diagnosed is diagnosed again here, without the pool, by `asa_diagnosis_day.diagnose_day`
   on the day its key names: the hour and state-car tables written from those diagnoses, with
   each row's unit, variant, day and day_index, are byte-identical to the run's, and so are each
   variant's figures, `top_days` included; at least one variant has days with differing rows.
7. `--scale 1` diagnoses every variant at 1, with `scale_source` "--scale".
8. The counts of one unit and variant (`asa_diagnosis_run._variant`) with a made-up pool: four
   prepared days, two abstaining (one by a certificate timeout) and two offering, each of these
   diagnosed with goal tests 2 timeouts and 1 optimal: offering days 2, certificate timeouts 1
   under the variant's own key, goal-test timeouts 4, the days in the order of D2; the
   preparation task is given no drawn car. Seven made-up days, one abstaining, each diagnosed
   with rows that name it (d + 1 rows for day d, so every two days differ): for 6 and 3 days
   wanted, the days diagnosed are the first offering days of D2, and every hour row and state
   car is written under the key and index of the day it was diagnosed on, the figures and
   `top_days` with them (a swap of any two days' diagnoses, the first two included, fails). The
   scale rule on the second gate unit's real splits, with a made-up tier A scale of 1 on the
   splits of seeds 1 to 3 and 0.5 on any other: over `--splits 5` it reads seeds 1 to 5 and
   gives scale 1 and [[0.5, 2], [1.0, 3]] (seeds 2 to 6 would give 0.5), and every day is
   diagnosed at that scale.
9. Exit status 2 for a scale not on the grid, the method named as reference, a method that is
   not a controller (`re-solve`), an unknown method, and a number of days, splits or workers
   below 1. Exit status 1, with every failure written, when every variant fails (a made-up
   `_variant` that raises).
"""

import csv
import json
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from compare_fake import NAMES
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import (
    asa_diagnosis_day,
    asa_diagnosis_run,
    asa_diagnosis_summary,
    compare_methods,
    compare_run,
    dial_choice,
    dial_splits,
    gate_day,
    gate_run,
    learn_run,
    lp,
    run_verify,
)
from fleetreserve.asa_diagnosis_day import DayDiagnosis
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load

SPLITS = 6
KEYS = ["unit", "variant", "day", "day_index"]
HOURS = 2
PATHS = 8
FILES = (
    asa_diagnosis_run.RESULTS_NAME,
    asa_diagnosis_run.HOURS_NAME,
    asa_diagnosis_run.CARS_NAME,
    asa_diagnosis_run.SUMMARY_NAME,
    asa_diagnosis_run.TABLE_NAME,
)
RUNS = {
    "rule": ("--workers", "1"),
    "rule2": ("--workers", "2"),
    "fixed": ("--scale", "1", "--reference", "fixed-plan"),
    "swap": ("--scale", "1", "--method", "llf", "--reference", "asa-es"),
    "two": ("--scale", "1", "--reference", "fixed-plan", "--days", "2"),
    "boxes": ("--scale", "1", "--reference", "diagonal-split", "--days", "2"),
    "four": ("--scale", "1", "--reference", "fixed-plan", "--days", "4"),
    "four3": ("--scale", "1", "--reference", "fixed-plan", "--days", "4", "--workers", "3"),
}


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        GATE_UNITS[1]: gate_cars(24, early_every=5),
        GATE_UNITS[2]: gate_cars(12, "B") + gate_cars(12, "C", early_every=4),
    }
    run, tables = write_build(folder, units, HOURS)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


def _run(build, folder: Path, *extra, splits=True):
    run, tables, matrix = build
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        folder.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        arguments = [
            "--model",
            "empirical",
            "--dial",
            "0.03",
            "--matrix",
            str(matrix),
            "--tables",
            str(tables),
            "--truth-run",
            str(run),
            *(("--splits", str(SPLITS)) if splits else ()),
            *extra,
        ]
        return asa_diagnosis_run.main(arguments), folder


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    # "two" has --scale, so the splits are not read: it runs on the default --splits.
    top = tmp_path_factory.mktemp("runs")
    return {
        name: _run(build, top / name, *extra, splits=name != "two") for name, extra in RUNS.items()
    }


def _document(folder: Path) -> dict:
    return json.loads((folder / asa_diagnosis_run.RESULTS_NAME).read_text())


def _rows(folder: Path, name: str) -> list[dict]:
    with open(folder / name, newline="") as handle:
        return list(csv.DictReader(handle))


def _diagnosed(document):
    for unit, entry in document["units"].items():
        for variant, found in entry["variants"].items():
            if found["status"] == "diagnosed":
                yield unit, variant, found


def _prepared_and_plan(build, unit, variant):
    _, tables, matrix_file = build
    matrix = load(matrix_file)
    configurations = compare_run.dial_configurations(matrix, 0.03)
    loaded = {
        name: load_unit_days(
            tables, name, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        for name in matrix.gate_units
    }
    model = learn_run.load_model("empirical", matrix)
    factory, _ = dial_choice.make_factory(
        "empirical", model, matrix, loaded, {}, configurations["forecast"].kappa
    )
    days = loaded[unit]
    forecaster = factory(days, matrix)
    plan = gate_run.plan_unit(unit, days, matrix)
    prepared = [
        gate_day.prepare_day(t, forecaster, configurations[variant], gate_run.settings(matrix))
        for t in days.days
    ]
    return matrix, days, plan, prepared


# --- 1. the runs ---------------------------------------------------------------------------------


def test_every_run_writes_every_file_and_reports_every_unit(runs):
    for name, (code, out) in runs.items():
        assert code == 0, name
        for file in (*FILES, asa_diagnosis_run.TIMING_NAME):
            assert (out / file).is_file(), (name, file)
        document = _document(out)
        assert document["failed"] == []
        first = document["units"][GATE_UNITS[0]]
        assert first["status"].startswith("not calibratable") and first["variants"] == {}
        for unit in GATE_UNITS[1:]:
            assert set(document["units"][unit]["variants"]) == set(compare_run.VARIANTS)
        summary = (out / asa_diagnosis_run.SUMMARY_NAME).read_text()
        assert f"{GATE_UNITS[0]}: not calibratable" in summary
    document = _document(runs["rule"][1])
    assert document["method"] == "asa-es" and document["reference"] == "llf"
    assert document["run"]["days"] == 50
    assert document["run"]["splits"] == SPLITS
    assert document["run"]["time_limit_seconds"] == lp.OTHER_SECONDS
    assert "seeds 1 to 6 on which it offers" in document["scale_rule"]
    assert document["days_rule"].startswith("the first 50 offering days")
    two = _document(runs["two"][1])
    assert two["run"]["splits"] == dial_choice.SPLITS and two["run"]["days"] == 2
    assert two["scale_rule"] == "--scale 1 for every unit and variant"
    assert two["days_rule"].startswith("the first 2 offering days")


# --- 2. the scale rule ---------------------------------------------------------------------------


def test_the_scale_of_the_rule_is_recomputed(build, runs):
    document = _document(runs["rule"][1])
    (llf,) = compare_methods.select(["llf"])
    statuses = set()
    for unit in GATE_UNITS[1:]:
        for variant in compare_run.VARIANTS:
            matrix, days, _, prepared = _prepared_and_plan(build, unit, variant)
            losses, _ = compare_run.measure_controllers(
                prepared,
                [None] * len(prepared),
                (llf,),
                lambda jobs: [compare_run._compare_task(j) for j in jobs],
            )
            splits = dial_choice.split_indices(unit, days, matrix, range(1, SPLITS + 1))
            scores = [dial_splits.score_split(losses["llf"], s, c, t, "A") for s, c, t in splits]
            offered = [s.scale for s in scores if s.offer_kwh > 0]
            entry = document["units"][unit]["variants"][variant]
            assert entry["scale_source"] == "rule"
            statuses.add(entry["status"])
            if not offered:
                assert entry["status"] == "no reference offer"
                continue
            best = max(set(offered), key=lambda g: (offered.count(g), g))
            assert entry["status"] == "diagnosed"
            assert entry["scale"] == best and entry["offering_splits"] == len(offered)
    assert statuses == {"diagnosed", "no reference offer"}


def test_the_scale_rule_counts_offering_splits_and_breaks_a_tie_to_the_larger(monkeypatch):
    # Made-up split scores: 0.2 twice, 0.5 twice, 1.0 once with an offer, and 1.0 three times
    # with no offer (not counted): a tie of 0.2 and 0.5 goes to 0.5.
    made_up = [(0.2, 1.0), (0.5, 1.0), (0.2, 2.0), (1.0, 3.0), (0.5, 0.5)] + [(1.0, 0.0)] * 3
    scores = iter(made_up)

    def score(losses, seed, calibration, test, tier):
        assert tier == "A"
        scale, offer = next(scores)
        return dial_splits.SplitScore(seed, scale, "", 20, 4, offer, 0.0, 0.0, None)

    monkeypatch.setattr(dial_splits, "score_split", score)
    splits = [(seed, [], []) for seed in range(1, len(made_up) + 1)]
    scale, pairs, offering = asa_diagnosis_run.reference_scale(None, splits)
    assert scale == 0.5 and offering == 5
    assert pairs == [[0.2, 2], [0.5, 2], [1.0, 1]]


def test_the_day_order_starts_with_the_calibration_days_of_split_seed_1(build):
    _, tables, matrix_file = build
    matrix = load(matrix_file)
    for unit in GATE_UNITS[1:]:
        days = load_unit_days(
            tables, unit, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
        )
        plan = gate_run.plan_unit(unit, days, matrix)
        ((_, calibration, test),) = dial_choice.split_indices(unit, days, matrix, [1])
        order = asa_diagnosis_run.day_order(unit, plan.keys)
        assert sorted(order[: len(calibration)]) == calibration
        assert sorted(order[len(calibration) :]) == test


# --- 3. every key defined ------------------------------------------------------------------------


def _undefined(value, defined, maps, parent=None) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "definitions":
                continue
            if parent not in maps and key not in defined:
                found.append(key)
            found += _undefined(child, defined, maps, key)
    elif isinstance(value, list):
        for child in value:
            found += _undefined(child, defined, maps, parent)
    return found


def test_every_key_and_column_is_defined(runs):
    for _, folder in runs.values():
        document = _document(folder)
        definitions = document["definitions"]
        maps = set(asa_diagnosis_run.NAME_MAPS)
        assert _undefined(document, set(definitions), maps) == []
        assert set(asa_diagnosis_run.HOURS_COLUMNS) == set(definitions["hours_columns"])
        assert set(asa_diagnosis_run.CARS_COLUMNS) == set(definitions["state_cars_columns"])
        with open(folder / asa_diagnosis_run.HOURS_NAME, newline="") as handle:
            assert next(csv.reader(handle)) == list(asa_diagnosis_run.HOURS_COLUMNS)
        state = [f.name for f in fields(asa_diagnosis_day.StateCar)]
        with open(folder / asa_diagnosis_run.CARS_NAME, newline="") as handle:
            assert next(csv.reader(handle)) == [*KEYS, "path", "hour", *state]
        flags = {r["method_first_failure"] for r in _rows(folder, asa_diagnosis_run.HOURS_NAME)}
        for row in _rows(folder, asa_diagnosis_run.CARS_NAME):
            flags |= {row["present"], row["left_earlier"]}
        assert flags <= {"True", "False"}
    boxes = [
        entry["figures"]["solver_counts"]["reference"]["boxes"]
        for _, _, entry in _diagnosed(_document(runs["boxes"][1]))
    ]
    assert boxes and all(sum(b.values()) > 0 for b in boxes)


def test_the_key_check_finds_an_undefined_key(runs):
    document = _document(runs["rule"][1])
    definitions = set(document["definitions"])
    document["units"][GATE_UNITS[1]]["made_up"] = 1
    assert _undefined(document, definitions, set(asa_diagnosis_run.NAME_MAPS)) == ["made_up"]


# --- 4. the counts add up ------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rule", "fixed", "swap"])
def test_the_counts_add_up(build, runs, name):
    out = runs[name][1]
    document = _document(out)
    hours = _rows(out, asa_diagnosis_run.HOURS_NAME)
    cars = _rows(out, asa_diagnosis_run.CARS_NAME)
    assert document["hours_table"]["rows"] == len(hours)
    assert document["state_cars_table"]["rows"] == len(cars)
    total_rows = total_cars = 0
    for unit, variant, entry in _diagnosed(document):
        figures = entry["figures"]
        mine = [r for r in hours if r["unit"] == unit and r["variant"] == variant]
        assert sum(figures["causes"].values()) == figures["groups"]["method_only"] == len(mine)
        assert sum(figures["groups"].values()) == entry["days_diagnosed"] * PATHS * HOURS
        for cause, count in figures["causes"].items():
            assert sum(r["cause"] == cause for r in mine) == count
        total_rows += len(mine)
        _, _, plan, prepared = _prepared_and_plan(build, unit, variant)
        assert all(plan.keys[int(r["day_index"])] == r["day"] for r in mine)
        for row in (r for r in mine if r["cause"] == "state"):
            found = [
                int(c["car"])
                for c in cars
                if (c["unit"], c["variant"], c["day"], c["path"], c["hour"])
                == (row["unit"], row["variant"], row["day"], row["path"], row["hour"])
            ]
            assert found == list(range(len(prepared[int(row["day_index"])].day.cars)))
            total_cars += len(found)
    assert total_rows == len(hours)
    assert total_cars == len(cars)
    if name != "rule":
        assert hours
    if name == "swap":
        assert {r["goal"] for r in hours if r["cause"] == "choice"} <= {"rule"}


# --- 5. the days diagnosed -----------------------------------------------------------------------


def test_the_days_are_the_first_offering_days_of_the_seeded_order(build, runs):
    fixed = _document(runs["fixed"][1])
    two = _document(runs["two"][1])
    checked = 0
    for unit, variant, entry in _diagnosed(fixed):
        _, _, plan, prepared = _prepared_and_plan(build, unit, variant)
        order = asa_diagnosis_run.day_order(unit, plan.keys)
        offering = [plan.keys[d] for d in order if not prepared[d].abstains]
        assert entry["days"] == offering[: asa_diagnosis_run.DAYS]
        found = two["units"][unit]["variants"][variant]
        assert found["days"] == offering[:2] and found["days_diagnosed"] == 2
        assert entry["offering_days"] == len(offering)
        checked += 1
    assert checked == 4


# --- 6. workers ----------------------------------------------------------------------------------


def test_the_files_are_the_same_for_one_and_two_workers(runs):
    for name in FILES:
        assert (runs["rule"][1] / name).read_bytes() == (runs["rule2"][1] / name).read_bytes()


@pytest.mark.parametrize("name", ["four", "four3"])
def test_each_row_is_written_under_the_day_it_was_diagnosed_on(build, runs, name, tmp_path):
    out = runs[name][1]
    document = _document(out)
    method, reference = compare_methods.select(["asa-es", "fixed-plan"])
    hours, cars, differing = [], [], 0
    for unit, variant, entry in _diagnosed(document):
        _, _, plan, prepared = _prepared_and_plan(build, unit, variant)
        order = asa_diagnosis_run.day_order(unit, plan.keys)
        chosen = [d for d in order if not prepared[d].abstains][:4]
        assert entry["days"] == [plan.keys[d] for d in chosen]
        diagnosed = []
        for d in chosen:
            day = asa_diagnosis_day.diagnose_day(
                prepared[d], 1.0, method, reference, time_limit=lp.OTHER_SECONDS
            )
            diagnosed.append((plan.keys[d], day))
            keys = {"unit": unit, "variant": variant, "day": plan.keys[d], "day_index": d}
            for row in day.rows:
                fields_of = asdict(row)
                for car in fields_of.pop("state_cars"):
                    cars.append({**keys, "path": row.path, "hour": row.hour, **car})
                hours.append({**keys, **fields_of})
        figures = asa_diagnosis_summary.variant_figures(diagnosed)
        assert entry["figures"] == json.loads(json.dumps(figures))
        differing += len({repr(day.rows) for _, day in diagnosed}) > 1
    assert hours and differing
    for rows, file, columns in (
        (hours, asa_diagnosis_run.HOURS_NAME, asa_diagnosis_run.HOURS_COLUMNS),
        (cars, asa_diagnosis_run.CARS_NAME, asa_diagnosis_run.CARS_COLUMNS),
    ):
        asa_diagnosis_run.write_rows(rows, tmp_path / file, columns)
        assert (tmp_path / file).read_bytes() == (out / file).read_bytes(), file


# --- 7. a given scale ----------------------------------------------------------------------------


def test_a_given_scale_is_used_for_every_variant(runs):
    document = _document(runs["fixed"][1])
    for unit in GATE_UNITS[1:]:
        for entry in document["units"][unit]["variants"].values():
            assert entry["scale"] == 1.0 and entry["scale_source"] == "--scale"
            assert entry["status"] == "diagnosed"


# --- 8. the counts of one unit and variant --------------------------------------------------------


@pytest.mark.parametrize("variant", compare_run.VARIANTS)
def test_the_counts_of_one_variant_with_a_made_up_pool(variant):
    keys = ["2020-01-01 s", "2020-01-02 s", "2020-01-03 s", "2020-01-04 s"]
    prepared = [
        SimpleNamespace(abstains=True, certificate_outcome="timeout"),
        SimpleNamespace(abstains=False, certificate_outcome="optimal"),
        SimpleNamespace(abstains=True, certificate_outcome="optimal"),
        SimpleNamespace(abstains=False, certificate_outcome="optimal"),
    ]

    class Pool:
        def map(self, fn, jobs):
            if fn is compare_run._prepare_task:
                assert [job[3] for job in jobs] == [()] * len(keys)  # no car drawn
                return [prepared[i] for _, i, _, _ in jobs]
            assert fn is asa_diagnosis_run._diagnose_task
            return [DayDiagnosis(goal_tests=Counter(timeout=2, optimal=1)) for _ in jobs]

    out = {"variant": variant, "hours": [], "cars": []}
    out["timeouts"] = Counter(certificates=0, oracle_certificates=0, goal_tests=0)
    plan = SimpleNamespace(keys=keys)
    options = {"scale": 1.0, "time_limit": 1.0}
    entry = asa_diagnosis_run._variant(
        Pool(), "u", plan, 50, SimpleNamespace(name="c"), (None, None), options, out
    )
    order = [d for d in asa_diagnosis_run.day_order("u", keys) if d in (1, 3)]
    assert entry["offering_days"] == 2 and entry["days"] == [keys[d] for d in order]
    key = "certificates" if variant == "forecast" else "oracle_certificates"
    other = "oracle_certificates" if variant == "forecast" else "certificates"
    assert (out["timeouts"][key], out["timeouts"][other]) == (1, 0)
    assert out["timeouts"]["goal_tests"] == 4


_ROW = asa_diagnosis_day.HourRow(
    path=0,
    hour=1,
    request=1.0,
    method_delivered=0.0,
    reference_delivered=1.0,
    shortfall=1.0,
    method_first_failure=True,
    method_low=0.0,
    method_high=0.0,
    reference_low=0.0,
    reference_high=1.0,
    believed_most=0.0,
    believed_on_true=0.0,
    deficit_plugged=0.0,
    deficit_departure=0.0,
    deficit_rate=0.0,
    deficit_cap=0.0,
    overplanned=0.0,
    cars_left_earlier=0,
    charged_ahead=0.0,
    cause="other",
    goal="",
)


def _named_diagnosis(d: int) -> DayDiagnosis:
    """Day d's made-up diagnosis: d + 1 method-only rows, each with path d and a request naming
    the day and hour, the causes in turn from d; a row of cause "state" has one car, numbered d."""
    rows = []
    for h in range(d + 1):
        cause = asa_diagnosis_day.CAUSES[(d + h) % len(asa_diagnosis_day.CAUSES)]
        car = asa_diagnosis_day.StateCar(d, 2, 3.0, False, True, 0.5, 0.0, 0.0, 1.0)
        rows.append(
            replace(
                _ROW,
                path=d,
                hour=h + 1,
                request=10.0 * d + h,
                shortfall=10.0 * d + h,
                deficit_cap=float(d) if cause == "belief" else 0.0,
                cars_left_earlier=1 if cause == "state" else 0,
                cause=cause,
                goal="rule" if cause == "choice" else "",
                state_cars=(car,) if cause == "state" else (),
            )
        )
    groups = Counter(method_only=d + 1, neither=2 * HOURS * PATHS - d - 1)
    return DayDiagnosis(groups=groups, rows=rows, goal_tests=Counter())


class _NamedPool:
    """A made-up pool: the prepared days given (each naming its index as `d`); each diagnosed
    day d gets `_named_diagnosis(d)`. Keeps the diagnosis jobs."""

    def __init__(self, prepared):
        self.prepared = prepared
        self.jobs = []

    def map(self, fn, jobs):
        if fn is compare_run._prepare_task:
            return [self.prepared[i] for _, i, _, _ in jobs]
        assert fn is asa_diagnosis_run._diagnose_task
        jobs = list(jobs)
        self.jobs += jobs
        return [_named_diagnosis(prepared.d) for prepared, *_ in jobs]


@pytest.mark.parametrize("wanted", [6, 3])
@pytest.mark.parametrize("variant", compare_run.VARIANTS)
def test_each_day_s_rows_are_written_under_its_own_key_with_a_made_up_pool(variant, wanted):
    # Seven days, day 4 abstaining; every two days' diagnoses differ (their rows name the day),
    # so a swap of any two diagnoses, the first two included, writes some row under another day.
    keys = [f"2020-01-0{d + 1} s" for d in range(7)]
    prepared = [
        SimpleNamespace(abstains=d == 4, certificate_outcome="optimal", d=d) for d in range(7)
    ]
    pool = _NamedPool(prepared)
    out = {"variant": variant, "hours": [], "cars": []}
    out["timeouts"] = Counter(certificates=0, oracle_certificates=0, goal_tests=0)
    entry = asa_diagnosis_run._variant(
        pool,
        "u",
        SimpleNamespace(keys=keys),
        wanted,
        SimpleNamespace(name="c"),
        (None, None),
        {"scale": 1.0, "time_limit": 1.0},
        out,
    )
    order = [d for d in asa_diagnosis_run.day_order("u", keys) if d != 4][:wanted]
    full = asa_diagnosis_run.day_order("u", keys)
    assert len(order) == wanted and full != sorted(full)
    assert len({repr(_named_diagnosis(d).rows) for d in order}) == wanted
    assert [job[0].d for job in pool.jobs] == order
    assert entry["days"] == [keys[d] for d in order] and entry["days_diagnosed"] == wanted
    hours, cars = [], []
    for d in order:
        named = {"unit": "u", "variant": variant, "day": keys[d], "day_index": d}
        for row in _named_diagnosis(d).rows:
            fields_of = asdict(row)
            for car in fields_of.pop("state_cars"):
                cars.append({**named, "path": d, "hour": row.hour, **car})
            hours.append({**named, **fields_of})
    assert out["hours"] == hours and out["cars"] == cars and cars
    figures = asa_diagnosis_summary.variant_figures([(keys[d], _named_diagnosis(d)) for d in order])
    assert entry["figures"] == figures
    assert [top["day"] for top in figures["top_days"]] == [keys[d] for d in sorted(order)[::-1]][:5]


def test_the_scale_rule_reads_the_splits_of_seeds_one_to_n(build, monkeypatch):
    # The reference's tier A calibrated scale on a split is made up from the split's own
    # calibration days: 1 on the splits of seeds 1 to 3, 0.5 on any other. Over seeds 1 to 5 the
    # rule gives 1 (three splits to two); over seeds 2 to 6 it would give 0.5 (two to three).
    unit, splits = GATE_UNITS[1], 5
    matrix, days, plan, prepared = _prepared_and_plan(build, unit, "forecast")
    prepared = [
        SimpleNamespace(abstains=p.abstains, certificate_outcome=p.certificate_outcome, d=d)
        for d, p in enumerate(prepared)
    ]

    first = [c for _, c, _ in dial_choice.split_indices(unit, days, matrix, range(1, 4))]

    def scale_of(calibration):
        return 1.0 if list(calibration) in first else 0.5

    def score(losses, seed, calibration, test, tier):
        assert losses == "reference losses" and tier == "A"
        return SimpleNamespace(scale=scale_of(calibration), offer_kwh=1.0)

    def expected(seeds):
        found = Counter(
            scale_of(c) for _, c, _ in dial_choice.split_indices(unit, days, matrix, seeds)
        )
        pairs = [[scale, found[scale]] for scale in sorted(found)]
        return max(found, key=lambda scale: (found[scale], scale)), pairs

    right = expected(range(1, splits + 1))
    shifted = expected(range(2, splits + 2))
    assert right[0] != shifted[0] and right[1] != shifted[1]  # it can tell the seeds apart
    monkeypatch.setattr(dial_splits, "score_split", score)
    monkeypatch.setattr(
        compare_run, "measure_controllers", lambda *args: ({"ref": "reference losses"}, 0)
    )
    pool = _NamedPool(prepared)
    out = {"variant": "forecast", "hours": [], "cars": []}
    out["timeouts"] = Counter(certificates=0, oracle_certificates=0, goal_tests=0)
    options = {"scale": None, "splits": splits, "matrix": matrix, "time_limit": 1.0}
    methods = (SimpleNamespace(name="m"), SimpleNamespace(name="ref"))
    entry = asa_diagnosis_run._variant(
        pool, unit, plan, 3, SimpleNamespace(name="c"), methods, options, out
    )
    assert (entry["scale"], entry["reference_scales"]) == right
    assert entry["offering_splits"] == splits and entry["scale_source"] == "rule"
    assert len(pool.jobs) == 3 and {job[1] for job in pool.jobs} == {right[0]}


# --- 9. refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        ("--scale", "0.55"),
        ("--reference", "asa-es"),
        ("--method", "re-solve"),
        ("--method", "no-such-method"),
        ("--days", "0"),
        ("--splits", "0"),
        ("--workers", "0"),
    ],
)
def test_refused_arguments_exit_2(build, tmp_path, extra, capsys):
    code, _ = _run(build, tmp_path / "refused", *extra)
    assert code == 2
    assert "COULD NOT RUN" in capsys.readouterr().out


def test_a_failed_variant_is_written_and_exits_1(build, tmp_path, monkeypatch):
    def fails(pool, name, plan, days, configuration, methods, options, out):
        raise ValueError(f"made up for {out['variant']}")

    monkeypatch.setattr(asa_diagnosis_run, "_variant", fails)
    code, out = _run(build, tmp_path / "failed", "--scale", "1")
    assert code == 1
    document = _document(out)
    expected = [
        f"{unit} | {variant}: ValueError: made up for {variant}"
        for unit in GATE_UNITS[1:]
        for variant in compare_run.VARIANTS
    ]
    assert document["failed"] == sorted(expected)
    for unit in GATE_UNITS[1:]:
        for variant, entry in document["units"][unit]["variants"].items():
            assert entry["failed"] == f"ValueError: made up for {variant}"
    summary = (out / asa_diagnosis_run.SUMMARY_NAME).read_text()
    assert "FAILED:\n" in summary and f"  {sorted(expected)[0]}\n" in summary

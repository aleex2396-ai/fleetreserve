"""How the corner-search command (`corner_run`) puts a run together, with everything below it
replaced by recording stand-ins (fast; the whole command on made-up units is run by
`test_corner_command`, the parts by `test_corner_run_parts`).

1. `run`: one pool for the run, closed at the end even when a unit fails outside a variant; per
   unit in the matrix's order its counts and status, a unit that cannot be split with no variants;
   per variant the preparation jobs of every counting day, the searched days (the first `--days`
   offering test days in the searched order), the path-hours counted before the search (and the
   progress function called each time), the variant's entry and rows; a variant that fails is
   recorded with its error and the others go on. The document: its label, definitions,
   configurations, dial, methods, settings, units, the pass mark summed over the checked variants
   (agreed only when something was checked and all agree), the timeouts sorted by key and the
   failures sorted.
2. `definitions`, `_json` and `_csv` written in full.
3. `main`: every refusal before the run with its exit status and its one line; the inputs read
   in order (the gate units, then the fit units not among them) and passed on; the files written
   into the run folder (results, rows, summary, timing, the timing also as the run goes) and none
   without one; the exit status 1 on a failure or a pass mark not agreed, else 0; the defaults of
   every option.
"""

import csv
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from fleetreserve import (
    compare_methods,
    compare_run,
    corner_run,
    corner_search,
    corner_summary,
    dial_choice,
    dial_splits,
    gate_run,
    learn_run,
    rotation_whatif,
)
from fleetreserve.gate_config import MatrixError
from fleetreserve.truth_build import OUT
from fleetreserve.truth_verify import CouldNotRun

# --- 1. run --------------------------------------------------------------------------------------

MATRIX = SimpleNamespace(gate_units=("c", "b", "a"), n_calibration=100, calibration_half_below=60)
CONFIGURATIONS = {
    "forecast": SimpleNamespace(name="cf", driver_risk=0.02, kappa=0.3),
    "oracle-departure": SimpleNamespace(name="co", driver_risk=0.5, kappa=0.9),
}
METHODS = (
    SimpleNamespace(name="m1", kind="controller"),
    SimpleNamespace(name="m2", kind="re-solve"),
)
UNITS = {"c": "days of c", "b": "days of b", "a": "days of a"}
PLANS = {
    "c": SimpleNamespace(
        keys=list(range(7)), calibration_days=2, test_days=5, status="calibratable"
    ),
    "b": SimpleNamespace(keys=list(range(3)), calibration_days=1, test_days=2, status="too few"),
    "a": SimpleNamespace(
        keys=list(range(6)), calibration_days=2, test_days=4, status="calibratable"
    ),
}
SPLITS = {"c": ([0, 1], [2, 3, 4, 5, 6]), "a": ([0, 1], [2, 3, 4, 5])}
ABSTAINING = {("a", 3), ("a", 5)}
# per unit and variant: the error raised, or the pass mark's counts
OUTCOMES = {
    ("c", "forecast"): (3, 2, 3, True),
    ("c", "oracle-departure"): ValueError("boom"),
    ("a", "forecast"): (5, 1, 4, False),
    ("a", "oracle-departure"): KeyError("x"),
}


class Pool:
    made = []

    def __init__(self, workers, units, matrix, factory):
        self.args, self.jobs, self.closed = (workers, units, matrix, factory), [], False
        Pool.made.append(self)

    def map(self, task, jobs):
        assert task is compare_run._prepare_task
        self.jobs.append(list(jobs))
        return [_prepared(unit, i, configuration) for unit, i, configuration, _ in jobs]

    def close(self):
        self.closed = True


VARIANT_OF = {c.name: v for v, c in CONFIGURATIONS.items()}


def _prepared(unit, i, configuration):
    return SimpleNamespace(
        abstains=(unit, i) in ABSTAINING,
        date=f"{unit}-{i}",
        site=f"site-{unit}",
        day=SimpleNamespace(n_service=1 + i % 3),
        certificate_outcome="timeout" if i == 0 else "optimal",
        unit=unit,
        variant=VARIANT_OF[configuration.name],
    )


def _found(unit, variant, searched):
    pairs, strict, agreed, verdict = OUTCOMES[(unit, variant)]
    return {
        "offering_calibration_days": 2,
        "calibration_evaluations": {
            "re_solves_of_the_one_way_rung": {"timeout": 1},
            "controller_solver_outcomes": {},
        },
        "searched_days": list(searched),
        "methods": {"m1": {"resolves": {"timeout": 2}}, "m2": {"status": "no offer"}},
        "pass_mark": {
            "pairs": pairs,
            "pairs_agreed_strict": strict,
            "pairs_agreed": agreed,
            "agreed": verdict,
        },
    }


SCALES = {"m1": (0.4, ""), "m2": (0.0, "no scale")}


def _by_method(searched):
    return {m: [corner_search.no_offer(m, SCALES[m][0]) for _ in searched] for m in SCALES}


@pytest.fixture
def recorded(monkeypatch):
    calls = {"measure": [], "progress": []}
    Pool.made = []
    monkeypatch.setattr(compare_run, "Pool", Pool)

    def plan_unit(name, unit_days, matrix):
        assert unit_days == UNITS[name] and matrix is MATRIX
        return PLANS[name]

    def split_indices(name, unit_days, matrix, seeds):
        assert unit_days == UNITS[name] and matrix is MATRIX and seeds == [5]
        return [(5, *SPLITS[name])]

    def measure(prepared, calibration, searched, methods, pool, hours, counted):
        unit, variant = prepared[0].unit, prepared[0].variant
        calls["measure"].append((unit, variant, list(calibration), list(searched), dict(hours)))
        assert methods is METHODS and pool is Pool.made[-1]
        outcome = OUTCOMES[(unit, variant)]
        if isinstance(outcome, Exception):
            raise outcome
        hours["search"] = 1000 + len(searched)
        counted()
        return _found(unit, variant, searched), _by_method(searched), SCALES

    monkeypatch.setattr(gate_run, "plan_unit", plan_unit)
    monkeypatch.setattr(corner_run.dial_choice, "split_indices", split_indices)
    monkeypatch.setattr(corner_run, "measure_variant", measure)
    return calls


def _run(calls, **options):
    def progress(timing):
        calls["progress"].append(timing)

    options = {"days": 3, "seed": 5, "workers": 2, "progress": progress, **options}
    return corner_run.run(MATRIX, UNITS, "factory", CONFIGURATIONS, METHODS, **options)


def _searched(unit):
    order = corner_run.day_order(unit, 5, SPLITS[unit][1])
    return [d for d in order if (unit, d) not in ABSTAINING][:3]


def test_the_run_is_put_together_unit_by_unit_and_variant_by_variant(recorded):
    document, rows, timing = _run(recorded)
    (pool,) = Pool.made
    assert pool.args == (2, UNITS, MATRIX, "factory") and pool.closed
    # every counting day prepared, with no car drawn, once per variant
    assert pool.jobs == [
        corner_run.prepare_jobs(unit, len(PLANS[unit].keys), CONFIGURATIONS[variant])
        for unit in ("c", "a")
        for variant in CONFIGURATIONS
    ]
    searched = {unit: _searched(unit) for unit in ("c", "a")}
    assert len(searched["c"]) == 3 and len(searched["a"]) == 2  # a has 2 offering test days
    forecast = CONFIGURATIONS["forecast"]
    hours = {
        unit: corner_run.count_path_hours(
            [_prepared(unit, i, forecast) for i in PLANS[unit].keys],
            SPLITS[unit][0],
            searched[unit],
            2,
            corner_run.CHECK_DAYS,
        )
        for unit in ("c", "a")
    }
    assert recorded["measure"] == [
        (unit, variant, SPLITS[unit][0], searched[unit], hours[unit])
        for unit in ("c", "a")
        for variant in CONFIGURATIONS
    ]
    # progress: once the path-hours are counted, and once the search's are known
    assert len(recorded["progress"]) == 6 and all(t is timing for t in recorded["progress"])
    with_search = {u: {**hours[u], "search": 1000 + len(searched[u])} for u in hours}
    assert timing["path_hours"] == {
        "c": {"forecast": with_search["c"], "oracle-departure": hours["c"]},
        "a": {"forecast": with_search["a"], "oracle-departure": hours["a"]},
    }
    assert list(timing["units"]) == ["c", "a"] and set(timing) == {
        "units",
        "path_hours",
        "total_seconds",
    }
    expected_rows = []
    for unit in ("c", "a"):
        prepared = [_prepared(unit, i, forecast) for i in PLANS[unit].keys]
        for m in ("m1", "m2"):
            expected_rows += corner_summary.day_rows(
                unit,
                "forecast",
                m,
                SCALES[m][0],
                searched[unit],
                prepared,
                _by_method(searched[unit])[m],
            )
    assert rows == expected_rows and len(rows) == 2 * (3 + 2)

    def variant(unit, offering):
        return {
            "configuration": "cf",
            "offering_test_days": offering,
            "days_asked": 3,
            "days_searched": len(searched[unit]),
            **_found(unit, "forecast", searched[unit]),
        }

    assert document == {
        "label": corner_run.LABEL,
        "definitions": corner_run.definitions(MATRIX),
        "configurations": {"forecast": "cf", "oracle-departure": "co"},
        "dial": 0.02,
        "methods": ["m1", "m2"],
        "run": {
            "seed": 5,
            "days": 3,
            "tier": "A",
            "check_days": corner_run.CHECK_DAYS,
            "check_scales": list(dial_splits.GRID),
            "slot_tolerance": corner_search.SLOT_TOLERANCE,
            "driver_tolerance": corner_search.DRIVER_TOLERANCE,
            "kappa": 0.3,
            "variants": ["forecast", "oracle-departure"],
        },
        "units": {
            "c": {
                "counting_days": 7,
                "calibration_days": 2,
                "test_days": 5,
                "status": "calibratable",
                "variants": {
                    "forecast": variant("c", 5),
                    "oracle-departure": {"configuration": "co", "failed": "ValueError: boom"},
                },
            },
            "b": {
                "counting_days": 3,
                "calibration_days": 1,
                "test_days": 2,
                "status": "too few",
                "variants": {},
            },
            "a": {
                "counting_days": 6,
                "calibration_days": 2,
                "test_days": 4,
                "status": "calibratable",
                "variants": {
                    "forecast": variant("a", 2),
                    "oracle-departure": {"configuration": "co", "failed": "KeyError: 'x'"},
                },
            },
        },
        "pass_mark": {"pairs": 8, "pairs_agreed_strict": 3, "pairs_agreed": 7, "agreed": False},
        "timeouts": {
            "certificates": 2,
            "controller_solves_calibration": 0,
            "controller_solves_search": 0,
            "re_solves_calibration": 2,
            "re_solves_search": 4,
        },
        "failed": [
            "a | oracle-departure: KeyError: 'x'",
            "c | oracle-departure: ValueError: boom",
        ],
    }
    assert list(document["timeouts"]) == sorted(document["timeouts"])


def test_the_pass_mark_is_agreed_when_every_check_agrees(recorded, monkeypatch):
    monkeypatch.setitem(OUTCOMES, ("a", "forecast"), (5, 5, 5, True))
    document, _, _ = _run(recorded)
    assert document["pass_mark"] == {
        "pairs": 8,
        "pairs_agreed_strict": 7,
        "pairs_agreed": 8,
        "agreed": True,
    }


def test_nothing_checked_is_not_agreed(recorded, monkeypatch):
    monkeypatch.setattr(MATRIX, "gate_units", ("b",))
    document, rows, timing = _run(recorded, progress=None)
    assert document["pass_mark"] == {
        "pairs": 0,
        "pairs_agreed_strict": 0,
        "pairs_agreed": 0,
        "agreed": False,
    }
    assert rows == [] and timing["units"] == {} and timing["path_hours"] == {}
    assert document["timeouts"] == {} and document["failed"] == []


def test_the_pool_is_closed_when_a_unit_fails_outside_a_variant(recorded, monkeypatch):
    def broken(name, unit_days, matrix):
        raise RuntimeError("no plan")

    monkeypatch.setattr(gate_run, "plan_unit", broken)
    with pytest.raises(RuntimeError, match="no plan"):
        _run(recorded)
    assert Pool.made[-1].closed


# --- 2. definitions and the file forms -----------------------------------------------------------


def test_the_definitions_in_full():
    assert corner_run.definitions(MATRIX) == {
        "readings": corner_run.READINGS,
        "fields": {**corner_run.FIELD_DEFINITIONS, **corner_summary.DEFINITIONS},
        "variants": {
            "forecast": "the dial's configuration of the matrix",
            "oracle-departure": "that configuration's oracle partner (field oracle_partner)",
        },
        "calibration": (
            "tier A: slot loss at alpha 0.10 and driver loss at alpha 0.05 on the split's "
            "offering calibration days (n_calibration 100, calibration_half_below 60)"
        ),
        "timeouts": (
            "certificates: days whose certificate timed out (they offer nothing); "
            "re_solves_calibration: one-way re-solves timed out in the calibration; "
            "re_solves_search: re-solves timed out on searched corners; "
            "controller_solves_calibration, controller_solves_search: solves of a controller "
            "that solves programmes timed out, in the calibration and on searched corners (each "
            "hour so planned by its fall-back); the keys with oracle_ in front: the same under "
            "the oracle-departure variant"
        ),
        "label": corner_run.LABEL,
    }
    found = corner_run.definitions(MATRIX)
    assert found["readings"] is not corner_run.READINGS  # a copy, not the module's own


def test_the_json_and_csv_forms():
    assert corner_run._json({"b": 1, "a": [1.5]}) == '{\n "a": [\n  1.5\n ],\n "b": 1\n}\n'
    with pytest.raises(ValueError):
        corner_run._json({"a": float("nan")})
    text = corner_run._csv([["u", 1, "x,y"], ["v", "", 0.25]])
    assert text.endswith("\n") and "\r" not in text
    lines = text.split("\n")
    assert lines[0] == ",".join(corner_summary.ROW_FIELDS)
    assert lines[1:] == ['u,1,"x,y"', "v,,0.25", ""]
    assert list(csv.reader(io.StringIO(text)))[1:] == [["u", "1", "x,y"], ["v", "", "0.25"]]


# --- 3. main -------------------------------------------------------------------------------------

DOCUMENT = {
    "label": "made up",
    "dial": 0.02,
    "configurations": {"forecast": "cf"},
    "methods": [],
    "run": {"seed": 1, "days": 2, "check_days": 1, "check_scales": [1.0], "slot_tolerance": 1e-9},
    "units": {},
    "timeouts": {},
}


@pytest.fixture
def command(monkeypatch, tmp_path):
    calls = {}
    matrix_file = tmp_path / "the-matrix.json"
    matrix_file.write_bytes(b"made-up matrix")
    matrix = SimpleNamespace(gate_units=("c", "a"), rho=0.25, no_fit=0.5, small_fleet=3)
    model = SimpleNamespace(name="model-x", fit_units=["a", "f"])
    state = {"failed": [], "agreed": True, "raise": {}}

    def raising(name, value):
        def call(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            if name in state["raise"]:
                raise state["raise"][name]
            return value(*args, **kwargs) if callable(value) else value

        return call

    def run(*args, **options):
        calls.setdefault("run", []).append((args, options))
        files = sorted(p.name for p in folder.iterdir()) if folder.exists() else None
        options["progress"]({"partial": 1})
        calls["timing while running"] = (files, _read(folder / corner_run.TIMING_NAME))
        document = {
            **json.loads(json.dumps(DOCUMENT)),
            "pass_mark": {"pairs": 1, "pairs_agreed_strict": 1, "pairs_agreed": 1},
            "failed": list(state["failed"]),
        }
        document["pass_mark"]["agreed"] = state["agreed"]
        return document, [["r", 1]], {"total_seconds": 2.5}

    def mark(document, m, fit_units):
        assert m is matrix and fit_units == ("a", "f")
        document["marked"] = True

    folder = tmp_path / "run"
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    monkeypatch.setattr(dial_choice, "MATRIX", matrix_file)
    monkeypatch.setattr(corner_run, "load", raising("load", matrix))
    monkeypatch.setattr(
        compare_run, "dial_configurations", raising("dial_configurations", CONFIGURATIONS)
    )
    monkeypatch.setattr(learn_run, "load_model", raising("load_model", model))
    monkeypatch.setattr(learn_run, "check_model", raising("check_model", None))
    monkeypatch.setattr(gate_run, "check_inputs", raising("check_inputs", {"inputs": "checked"}))
    monkeypatch.setattr(
        corner_run,
        "load_unit_days",
        raising("load_unit_days", lambda tables, name, **kw: SimpleNamespace(cars=f"cars {name}")),
    )
    monkeypatch.setattr(
        dial_choice, "make_factory", raising("make_factory", ("the factory", [1, 2, 3]))
    )
    monkeypatch.setattr(corner_run, "run", run)
    monkeypatch.setattr(rotation_whatif, "mark", mark)
    return SimpleNamespace(
        calls=calls, state=state, folder=folder, matrix=matrix, model=model, file=matrix_file
    )


def _read(path):
    return path.read_text() if path.exists() else None


ARGS = ["--model", "lgbm_tail", "--dial", "0.02"]


def test_main_reads_its_inputs_in_order_and_writes_every_file(command, capsys):
    tables, truth = command.file.parent / "tables", command.file.parent / "truth"
    argv = [*ARGS, "--days", "4", "--workers", "3", "--seed", "9", "--tables", str(tables)]
    argv += ["--truth-run", str(truth), "--methods", "llf", "fixed-plan"]
    assert corner_run.main(argv) == 0
    calls = command.calls
    assert calls["load"] == [((command.file,), {})]
    assert calls["dial_configurations"] == [((command.matrix, 0.02), {})]
    assert calls["load_model"] == [(("lgbm_tail", command.matrix), {})]
    assert calls["check_model"] == [(("lgbm_tail", command.model, ("a", "f"), ("c", "a")), {})]
    assert calls["check_inputs"] == [((truth, tables, ("c", "a", "f")), {})]
    shape = {"rho": 0.25, "no_fit": 0.5, "small_fleet": 3}
    assert calls["load_unit_days"] == [((tables, n), shape) for n in ("c", "a", "f")]
    units = {n: SimpleNamespace(cars=f"cars {n}") for n in ("c", "a")}
    assert calls["make_factory"] == [
        (
            (
                "lgbm_tail",
                command.model,
                command.matrix,
                units,
                {"a": "cars a", "f": "cars f"},
                0.3,
            ),
            {},
        )
    ]
    ((args, options),) = calls["run"]
    assert args == (
        command.matrix,
        units,
        "the factory",
        CONFIGURATIONS,
        compare_methods.select(["llf", "fixed-plan"]),
    )
    assert {k: v for k, v in options.items() if k != "progress"} == {
        "days": 4,
        "seed": 9,
        "workers": 3,
    }
    # the timing is written as the run goes, before any other file
    assert calls["timing while running"] == ([], corner_run._json({"partial": 1}))
    document = {
        **DOCUMENT,
        "pass_mark": {"pairs": 1, "pairs_agreed_strict": 1, "pairs_agreed": 1, "agreed": True},
        "failed": [],
        "model": "model-x",
        "fit_units": ["a", "f"],
        "forecast_table_rows": 3,
        "marked": True,
        "inputs": {"inputs": "checked"},
        "matrix": {
            "file": "the-matrix.json",
            "sha256": hashlib.sha256(b"made-up matrix").hexdigest(),
        },
    }
    text = "\n".join(corner_summary.summary_lines(document)) + "\n"
    assert capsys.readouterr().out == text
    folder = command.folder
    assert sorted(p.name for p in folder.iterdir()) == sorted(
        [
            corner_run.RESULTS_NAME,
            corner_run.ROWS_NAME,
            corner_run.SUMMARY_NAME,
            corner_run.TIMING_NAME,
        ]
    )
    assert (folder / "corner.json").read_text() == corner_run._json(document)
    assert (folder / "corner-days.csv").read_text() == corner_run._csv([["r", 1]])
    assert (folder / "corner-summary.txt").read_text() == text
    assert (folder / "corner-timing.json").read_text() == corner_run._json({"total_seconds": 2.5})


def test_main_defaults(command):
    assert corner_run.main(ARGS) == 0
    assert command.calls["load"] == [((command.file,), {})]
    assert command.calls["check_inputs"][0][0][:2] == (None, OUT)
    assert command.calls["load_unit_days"][0][0][0] == OUT
    ((args, options),) = command.calls["run"]
    assert args[4] == compare_methods.select(None)
    assert (options["days"], options["seed"], options["workers"]) == (50, 1, 1)
    assert (corner_run.DAYS, corner_run.SEED) == (50, 1)


def test_main_without_a_run_folder_writes_nothing(command, monkeypatch, capsys):
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR")
    assert corner_run.main(ARGS) == 0
    assert list(command.folder.iterdir()) == []
    assert command.calls["timing while running"] == ([], None)
    assert capsys.readouterr().out.startswith("Corner search for model model-x")


@pytest.mark.parametrize(
    ("failed", "agreed", "code"),
    [([], True, 0), (["u | forecast: boom"], True, 1), ([], False, 1), (["x"], False, 1)],
)
def test_main_exit_status(command, failed, agreed, code):
    command.state["failed"], command.state["agreed"] = failed, agreed
    assert corner_run.main(ARGS) == code


REFUSALS = [
    (["--workers", "0"], None, None, 2, "COULD NOT RUN: --workers must be at least 1"),
    (["--days", "0"], None, None, 2, "COULD NOT RUN: --days must be at least 1"),
    (["--seed", "0"], None, None, 2, "COULD NOT RUN: --seed must be at least 1"),
    (["--methods", "llf", "llf"], None, None, 2, "COULD NOT RUN: a method is named twice"),
    ([], "load", MatrixError("bad matrix"), 2, "COULD NOT RUN: bad matrix"),
    ([], "dial_configurations", MatrixError("no dial"), 2, "COULD NOT RUN: no dial"),
    ([], "load_model", MatrixError("no model"), 2, "COULD NOT RUN: no model"),
    ([], "check_model", learn_run.FitOnScored("fit on a"), 2, "COULD NOT RUN: fit on a"),
    ([], "check_model", learn_run.Refused("not run"), 1, "REFUSED: not run"),
    ([], "check_inputs", CouldNotRun("no truth"), 2, "COULD NOT RUN: no truth"),
    (
        [],
        "load_unit_days",
        ValueError("bad day"),
        1,
        "FAILED: a day could not be built from the tables: bad day",
    ),
    ([], "make_factory", learn_run.Refused("no factory"), 1, "REFUSED: no factory"),
    ([], "make_factory", ValueError("v"), 1, "FAILED: the model could not be scored: v"),
]


@pytest.mark.parametrize(("extra", "where", "problem", "code", "line"), REFUSALS)
def test_main_refuses_before_the_run(command, capsys, extra, where, problem, code, line):
    if where is not None:
        command.state["raise"][where] = problem
    assert corner_run.main([*ARGS, *extra]) == code
    assert capsys.readouterr().out == line + "\n"
    assert "run" not in command.calls and list(command.folder.iterdir()) == []

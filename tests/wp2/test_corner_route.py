"""`corner_run` scoring held-out units through the sealed route (`sealed_route`), on made-up data.

The fixtures of `test_heldout_route_scoring` are imported from it, not copied: the held-out build
at the made-up tag `frozen-plan` and the made-up development build (caltech), both recorded at the
tag's commit, the held setting's held-out build, and the stand-in models `route_model` (fitted on
caltech) and the look-ahead one. Only the made-up repository is this file's own (`repo`): the
sealed files of `heldout_truth_fake` with OSL_T's days replaced by `OSL_T_DAYS` days of
`STAYING_DAY`, because on that file's own OSL_T days no day offers (each car at τ leaves inside the
window, with no recovery hour after it). On these days four cars are plugged in at τ 16:00 and stay
to 00:30, and six more, 16:30 to 21:30, hold the median above half the peak to 22:00 only, so S is
17:00 to 22:00 (five service hours, 32 corners a day) and the four cars have two recovery hours.
40 counting days give 20 calibration days, enough for tier A's driver bound (19 at 0.05). For the
runs that search, `route_model` forecasts near that truth (`CornerModel`).

The matrix holds the dial family of the corner tests (`corner_fake.NAMES`: kappa 0.3, dials 0.03
and 0.1, each with its oracle partner) with the held-out build's split rule (the committed
matrix's: seed 1, 100 calibration days or half the days under 200); its gate unit is OSL_T. The
methods searched are the fixed plan and LLF (`--methods`), so a run stays short.

1. On the route `corner_run` fits the model on the development build's caltech tables only and
   searches OSL_T from the held-out build (exit 0): the route is recorded in `corner.json` (the
   tag's commit, the held-out build's check with the build's own hashes, the fit units, no kappa
   selection, what the route's checks stop and do not), `inputs` holds the development build's
   check of caltech's tables only, the label says it is a held-out result and the development
   label is not used, the summary's first two lines are the route's, the searched days are test
   days of the build's split (none of the calibration days the build searched OSL_T's window on),
   every written field has a definition (the closed-form check's under its route name,
   `closed_form_check`), no sealed file is opened and the held-out loader logs no new opening.
1b. No output of the route run calls its closed-form check WP11's pass mark (readings H5 and H6
   of `corner_run`): of the summary only its third line, the label, names WP11 or a pass mark,
   and it says the check is not WP11's pass mark; the three headings name the check a
   closed-form property check on held-out days; every line of `corner.json`, the timing file and
   the rows that names WP11 or a pass mark says the same; no field, definition or path-hour
   count is named `pass_mark`.
1c. The gate once more: a model whose module, loaded after the route's first gate, is code from
   outside the package is refused by the second gate (`sealed_route.gate_again`; exit 2, one
   line), the model built, no table read and nothing sealed opened.
1d. The check at one scale (readings H5 and H7 of `corner_run`): every (day, scale) pair the
   check evaluates is at the fixed plan's calibrated scale of its unit and variant, the value
   `calibrate` returned (called once per variant), and the counts and path-hours are those of
   one scale (20 days of 32 corners and five service hours: 3,200 path-hours per variant); made
   to disagree at that scale the run exits 1 ("DISAGREED"), made to disagree only at the grid's
   other scales it records no disagreement (the decision's consequence, not a property of the
   fixed plan); a unit and variant whose fixed plan offers nothing is listed as not checked and
   does not decide the exit status; with nothing checked the run-level field is null, the
   summary says "nothing checked on this run" and the exit status is the search's (0, or 1 with
   a failed variant); `--methods` without the fixed plan is refused before any table is read or
   the model loaded (the matrix file is read first).
1e. The route's label, written into the summary and `corner.json`, says where the gate passes in
   the one text true at the tag and at a logged parsing fix of it (`sealed_route.AT_THE_TAG`).
2. The search sets nothing: with the search's results replaced (each day searched at scale 2 in
   place of the method's calibrated scale, above 0 for both methods), every method's scale, every
   searched day and every calibration count are unchanged; only the break figures move.
3. Off the tag (a made-up repository without it) the run could not run (exit 2, one line), no
   table is read, the model is not loaded, and no file of the made-up sealed folder or of the
   held-out build's tables is opened, listed or scanned (`sealed_watch`).
4. Refused, each with exit 2, one line and no table read: units of two kinds in the matrix; the
   two tables folders swapped; the held-out arguments with a development matrix; no held-out
   truth run; the held-out tables named in the sealed folder, the matrix file named there, and the
   held-out tables written as a URL (nothing in the sealed folder opened). Builds of two settings
   (the held setting's held-out build with the main development build) are refused with exit 2
   and one line after the development build's tables are checked, as in the learn_run route
   tests: no held-out table is read.
5. The leak rules, each refused before any day is prepared: a model that builds a scored unit's
   forecasts from its own other days (`empirical`, exit 2); a look-ahead model (exit 1); `--seed`
   other than the split the build drew (exit 2); a matrix of another split rule (exit 2, naming
   both rules); `--days` below the plan's 50 (exit 2); calibration days other than the build's
   (a scored day left out of the parsed days, as a matrix reading that makes other counting days
   would; exit 2, the model not fitted).
"""

import csv
import dataclasses
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest
import test_heldout_route_scoring as scoring
from fake_heldout import NORWAY_FILE, sealed_repo
from gate_fake import COMMITTED, matrix_document
from heldout_truth_fake import (
    NORWAY_START,
    TRO_R_DAYS,
    norway_bytes,
    norway_days,
    norway_rows,
    sealed_files,
)
from sealed_watch import watch
from swap_after_read import grown, swapped_after_first_read, table_refusal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp11"))

from corner_fake import NAMES  # noqa: E402

from fleetreserve import (  # noqa: E402
    corner_run,
    corner_search,
    corner_summary,
    dial_splits,
    heldout_loader,
    sealed_route,
)

# The two builds, the held setting's held-out build and the stand-in models: the fixtures of the
# learn_run route tests, reused; each reads this file's `repo`.
builds = scoring.builds
held_setting = scoring.held_setting
setup = scoring.setup

SCORED = scoring.SCORED
FIT_UNIT = scoring.FIT_UNIT
MODEL = scoring.MODEL
DIAL = "0.03"
METHODS = ("fixed-plan", "llf")
_opens = scoring._opens
_tables_read = scoring._tables_read
# The closed-form check's name on the route (reading H6 of `corner_run`).
HELD_CHECK = "closed_form_check"

# (user, plug-in hours, plug-out hours, kWh) on each of OSL_T's days (module text).
STAYING_DAY = [(f"u{k}", 15.5, 24.5, 10.0) for k in range(1, 5)] + [
    (f"u{k}", 16.5, 21.5, 8.0) for k in range(5, 11)
]
OSL_T_DAYS = 40


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """The made-up repository at its tag: `heldout_truth_fake`'s sealed files, OSL_T's days of
    `STAYING_DAY` (module text)."""
    folder = tmp_path_factory.mktemp("corner-route")
    rows = norway_rows(SCORED, norway_days(NORWAY_START, OSL_T_DAYS), pattern=STAYING_DAY)
    rows += norway_rows("TRO_R", norway_days(NORWAY_START, TRO_R_DAYS))
    files = {**sealed_files(folder / "made"), NORWAY_FILE: norway_bytes(folder / "osl", rows)}
    return sealed_repo(folder, files)


# Forecasts near the truth of `STAYING_DAY`'s cars at τ: departure 8 hours after τ (true 8.5),
# rate 1.1 kW (true 10/9), room and need 6.3 kWh (true room: 10 kWh less 3.7 before τ).
FORECAST = {"d_hat": 8.0, "d_median": 8.0, "r_hat": 1.1, "e_cap": 6.3, "e_need": 6.3}


class CornerModel(scoring.RouteModel):
    """The stand-in model of the learn_run route tests (fitted on caltech), forecasting
    `FORECAST` for every car."""

    def predict(self, frame, kappas):
        return super().predict(frame, kappas).assign(**FORECAST)


@pytest.fixture
def near(setup, monkeypatch):
    """`setup`, with `route_model` forecasting near the made-up truth (`CornerModel`)."""

    def make_model(matrix):
        model = CornerModel(MODEL, matrix.rho)
        setup["built"].append(model)
        return model

    module = types.ModuleType(f"fleetreserve.models.{MODEL}")
    module.make_model = make_model
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return setup


def _matrix(folder: Path, units, **changes) -> Path:
    document = matrix_document(NAMES)
    document["gate_units"] = list(units)
    document["required_units"] = 1
    document["n_calibration"] = COMMITTED["n_calibration"]  # the held-out build's split rule
    document["calibration_half_below"] = COMMITTED["calibration_half_below"]
    document.update(changes)
    path = folder / f"corner-matrix-{len(list(folder.iterdir()))}.json"
    path.write_text(json.dumps(document))
    return path


def _arguments(s: dict, matrix: Path, *extra, tables=None, run=None, model=MODEL, **held) -> list:
    """The command line; `held_tables` and `held_run` (None: left out) name the held-out build."""
    found = ["--model", model, "--dial", DIAL, "--matrix", str(matrix)]
    found += ["--tables", str(tables or s["tables"]), "--truth-run", str(run or s["run"])]
    if held.get("held_tables") is not None:
        found += ["--held-out-tables", str(held["held_tables"])]
    if held.get("held_run") is not None:
        found += ["--held-out-truth-run", str(held["held_run"])]
    return [*found, "--methods", *METHODS, *extra]


def _route_arguments(s: dict, matrix: Path, *extra, **options) -> list:
    """The route's command line: the held-out build's run named, its tables folder by default."""
    return _arguments(s, matrix, *extra, held_run=s["held_run"], **options)


def _document(s: dict) -> dict:
    return json.loads((s["out"] / corner_run.RESULTS_NAME).read_text())


def _rows(s: dict) -> list[dict]:
    with (s["out"] / corner_run.ROWS_NAME).open() as handle:
        return list(csv.DictReader(handle))


def _sealed(repo: Path) -> Path:
    return repo / "data" / "heldout"


# --- 1. the route, end to end --------------------------------------------------------------------


def test_the_route_searches_the_sealed_tables_with_a_model_fitted_on_the_development_build(
    repo, near, capsys
):
    s = near
    matrix = _matrix(s["tmp"], [SCORED])
    openings = heldout_loader.logged_openings(repo)
    with watch(_sealed(repo)) as seen:
        status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert (status, seen) == (0, []), "\n".join(shown)
    assert shown[0] == (
        f"{sealed_route.LABEL} (tag commit {s['tag']}): the scored units' tables from the "
        f"held-out build {s['held_run'].name}, the fit units' ({FIT_UNIT}) from the development "
        f"build {s['run'].name}."
    )
    assert shown[1] == scoring.CHECKS_TEXT
    assert shown[2].startswith(f"Corner search for model {MODEL}, dial {DIAL}: split seed 1,")
    assert shown[2].endswith(f" {corner_run.HELD_OUT_LABEL}.")
    assert corner_run.LABEL not in "\n".join(shown)
    (model,) = s["built"]
    assert [sorted(set(frame["site"])) for frame in model.fitted] == [["T"]]
    assert [sorted(set(frame["unit"])) for frame in model.predicted] == [[SCORED]]
    document = _document(s)
    assert document["label"] == document["definitions"]["label"] == corner_run.HELD_OUT_LABEL
    route = document["sealed_route"]
    assert route["tag_commit"] == s["tag"] and route["fit_units"] == [FIT_UNIT]
    assert route["checks"] == scoring.CHECKS_TEXT
    assert route["kappa_selection"] == sealed_route.NO_SELECTION
    assert route["held_out_inputs"]["truth_run"] == s["held_run"].name
    assert route["held_out_inputs"]["table_sha256"] == scoring._held_out_hashes(
        s["summary"], SCORED
    )
    assert document["inputs"]["truth_run"] == s["run"].name
    assert set(document["inputs"]["table_sha256"]) == {
        f"{FIT_UNIT}-cars.csv",
        f"{FIT_UNIT}-days.csv",
    }
    unit = document["units"][SCORED]
    assert (unit["counting_days"], unit["calibration_days"], unit["test_days"]) == (40, 20, 20)
    assert document[HELD_CHECK]["agreed"] is True and document["failed"] == []
    calibration = set(s["summary"]["windows"][SCORED]["calibration_dates"])
    assert len(calibration) == 20
    rows = _rows(s)
    for variant, found in unit["variants"].items():
        dates = {r["date"] for r in rows if r["variant"] == variant}
        assert len(dates) == found["days_searched"] > 0
        assert dates.isdisjoint(calibration)  # test days of the build's split only
    assert set(document["definitions"]["readings"]) >= {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    assert _route_undefined(document, rows) == set()
    assert heldout_loader.logged_openings(repo) == openings


def _route_written(document: dict, rows) -> set:
    """The fields a run on the route writes, by the levels of `_written` in
    tests/wp11/test_corner_command.py, the check's under its route name (reading H6 of
    `corner_run`)."""
    written = {f"document.{k}" for k in document}
    written |= {f"document.run.{k}" for k in document["run"]}
    written |= {f"document.{HELD_CHECK}.{k}" for k in document[HELD_CHECK]}
    written |= {f"document.matrix.{k}" for k in document.get("matrix", {})}
    for unit in document["units"].values():
        written |= {f"unit.{k}" for k in unit}
        for variant in unit["variants"].values():
            written |= {f"variant.{k}" for k in variant}
            for method in variant.get("methods", {}).values():
                written |= {f"method.{k}" for k in method}
            check = variant.get(HELD_CHECK, {})
            written |= {f"{HELD_CHECK}.{k}" for k in check}
            for wrong in check.get("disagreements", []):
                written |= {f"{HELD_CHECK}.disagreements.{k}" for k in wrong}
    written |= {f"row.{k}" for row in rows for k in row}
    return written


def _route_undefined(document: dict, rows) -> set:
    """The written fields of a run on the route without a definition (`_undefined` of
    tests/wp11/test_corner_command.py)."""
    return _route_written(document, rows) - set(document["definitions"]["fields"])


# --- 1b. the closed-form check's names on the route ----------------------------------------------

# A line that names the check as WP11's pass mark, in any spelling of the outputs.
NAMED = re.compile(r"wp11|pass[ _-]?mark", re.IGNORECASE)
# The one sentence of a route run that says what the check is not (`corner_run.HELD_OUT_LABEL`,
# readings H5 and H6).
NOT_THE_PASS_MARK = "not WP11's pass mark"
ROUTE_HEADINGS = (
    "Closed-form property check on held-out days: the first 20 searched days of each unit and "
    "variant at the fixed plan's calibrated scale of that unit and variant alone (none where it "
    "offers nothing or where no day was searched); ",
    " Closed-form property check on held-out days: fixed-plan closed form against the corner "
    "worst case",
    "Closed-form property check over all units: ",
)


def test_the_route_label_names_the_tag_or_a_logged_parsing_fix():
    """Item 1e: the label's place of the gate is true at the tag and at a logged parsing fix."""
    assert corner_run.HELD_OUT_LABEL.startswith(
        "corner search on held-out units through the sealed-data route at the tag frozen-plan, "
        "or at a logged parsing fix of it: a held-out result (plan section 11, O7); "
    )


def test_no_output_of_a_route_run_calls_its_closed_form_check_what_it_is_not(repo, near, capsys):
    """On held-out days the closed-form check is a property check, not WP11's pass mark (reading
    H5): the summary names WP11 or a pass mark on one line only, its third, the label, which
    says so; its three headings say what the check is; every line of the results, the timing
    file and the rows that names WP11 or a pass mark says it is not WP11's pass mark; no field
    is named `pass_mark` (reading H6)."""
    s = near
    matrix = _matrix(s["tmp"], [SCORED])
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert status == 0, "\n".join(shown)
    assert (s["out"] / corner_run.SUMMARY_NAME).read_text().splitlines() == shown
    assert [line for line in shown if NAMED.search(line)] == [shown[2]]
    assert shown[2].endswith(f" {corner_run.HELD_OUT_LABEL}.") and NOT_THE_PASS_MARK in shown[2]
    paragraph, unit_head, total = ROUTE_HEADINGS
    assert sum(line.startswith(paragraph) for line in shown) == 1
    assert shown.count(unit_head) == 1  # one gate unit
    assert [line for line in shown if line.startswith(total)] == [
        f"{total}40 of 40 pairs agree strictly, 40 with the tolerance: agreed."
    ]  # 20 test days checked, at the fixed plan's calibrated scale alone, under 2 variants
    for name in (corner_run.RESULTS_NAME, corner_run.TIMING_NAME, corner_run.ROWS_NAME):
        lines = (s["out"] / name).read_text().splitlines()
        named = [line for line in lines if NAMED.search(line)]
        assert [line for line in named if NOT_THE_PASS_MARK not in line] == [], name
    document = _document(s)
    assert corner_summary.CHECK_KEY not in document and HELD_CHECK in document
    for found in document["units"][SCORED]["variants"].values():
        assert corner_summary.CHECK_KEY not in found and found[HELD_CHECK]["agreed"] is True
    fields = document["definitions"]["fields"]
    assert [k for k in fields if corner_summary.CHECK_KEY in k.split(".")] == []
    assert (
        fields[f"variant.{HELD_CHECK}"]
        == corner_run.ROUTE_CHECK_DEFINITIONS[f"variant.{HELD_CHECK}"]
    )
    assert document["definitions"]["readings"]["K3"] == corner_run.ROUTE_K3
    timing = json.loads((s["out"] / corner_run.TIMING_NAME).read_text())
    for found in timing["path_hours"][SCORED].values():
        assert set(found) == {"calibration", "search_upper_bound", "search", HELD_CHECK}


# --- 1c. the gate once more, after the model is loaded -------------------------------------------


def test_the_gate_is_passed_again_once_the_models_code_is_loaded(repo, setup, monkeypatch, capsys):
    """The model's module is loaded after the route's first gate, so its code is checked by a
    second (`sealed_route.gate_again`) before any table is read: here the module, loaded from a
    file outside the package, is refused there (exit 2), after the model is built and before any
    table is read or the made-up sealed folder opened."""
    s = setup
    module = sys.modules[f"fleetreserve.models.{MODEL}"]
    make = module.make_model
    elsewhere = s["tmp"] / f"{MODEL}.py"

    def make_model(matrix):
        monkeypatch.setattr(module, "__file__", str(elsewhere), raising=False)
        return make(matrix)

    monkeypatch.setattr(module, "make_model", make_model)
    matrix = _matrix(s["tmp"], [SCORED])
    with _opens() as opened, watch(_sealed(repo)) as seen:
        status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out
    assert (status, seen) == (2, [])
    assert shown.startswith(
        "COULD NOT RUN: REFUSED BY THE GATE: the running code is not the tag's: modules outside "
        f"the package folder: {elsewhere.resolve()}"
    )
    assert shown.endswith("; nothing read\n") and shown.count("\n") == 1
    assert len(s["built"]) == 1  # the model was built, then refused
    assert _tables_read(opened) == []
    assert not (s["out"] / corner_run.RESULTS_NAME).exists()


# --- 1d. the closed-form check at one scale ------------------------------------------------------

FIXED = "fixed-plan"
# Path-hours of the check on one day of OSL_T (module text): 32 corners times 5 service hours.
DAY_PATH_HOURS = 32 * 5


@pytest.fixture
def recorded(monkeypatch):
    """The fixed plan's scale as `calibrate` returned it, per call in order (`chosen`), and the
    scales of each call of the check (`checked`); `zero` (a set of call numbers, from 0) makes
    `calibrate` return the fixed plan's scale as 0 on those calls."""
    found = {"chosen": [], "checked": [], "zero": set()}
    calibrate, check = corner_run.calibrate, corner_search.closed_form_check

    def calibrated(prepared, calibration, methods, pool):
        scales, counts = calibrate(prepared, calibration, methods, pool)
        if len(found["chosen"]) in found["zero"]:
            scales[FIXED] = (0.0, dial_splits.NO_SAFE_SCALE)
        found["chosen"].append(scales[FIXED][0])
        return scales, counts

    def checked(prepared, betas):
        found["checked"].append(tuple(betas))
        return check(prepared, betas)

    monkeypatch.setattr(corner_run, "calibrate", calibrated)
    monkeypatch.setattr(corner_search, "closed_form_check", checked)
    return found


def test_on_the_route_the_check_runs_at_the_fixed_plans_calibrated_scale_alone(
    repo, near, recorded, capsys
):
    s = near
    matrix = _matrix(s["tmp"], [SCORED])
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert status == 0, "\n".join(shown)
    document = _document(s)
    variants = document["units"][SCORED]["variants"]
    assert list(variants) == ["forecast", "oracle-departure"]
    chosen = recorded["chosen"]
    assert len(chosen) == 2 and all(scale > 0 for scale in chosen)  # one calibration per variant
    expected = []
    for scale, found in zip(chosen, variants.values(), strict=True):
        assert found["methods"][FIXED]["scale"] == scale  # the value the search uses
        check = found[HELD_CHECK]
        assert (check["days_checked"], check["pairs"], check["scales"]) == (20, 20, [scale])
        expected += [(scale,)] * 20
    assert recorded["checked"] == expected  # every pair evaluated at the chosen scale
    total = document[HELD_CHECK]
    assert (total["pairs"], total["agreed"], total["not_checked"]) == (40, True, [])
    assert document["run"]["check_scales"] == corner_run.ROUTE_CHECK_SCALES
    timing = json.loads((s["out"] / corner_run.TIMING_NAME).read_text())
    for found in timing["path_hours"][SCORED].values():
        assert found[HELD_CHECK] == 20 * DAY_PATH_HOURS * 1 == 3200  # 20 days at one scale
    assert _route_undefined(document, _rows(s)) == set()


def _disagreeing(monkeypatch, recorded, at_chosen: bool) -> None:
    """The check made to disagree (reason 'unexplained') at the fixed plan's chosen scale, or
    at every other scale of the grid."""
    check = corner_search.closed_form_check

    def wrong(prepared, betas):
        return tuple(
            dataclasses.replace(c, agreed=False, agreed_strict=False, reason="unexplained")
            if (c.beta in recorded["chosen"]) == at_chosen
            else c
            for c in check(prepared, betas)
        )

    monkeypatch.setattr(corner_search, "closed_form_check", wrong)


@pytest.mark.parametrize("at_chosen", [True, False], ids=["at-the-chosen-scale", "elsewhere"])
def test_a_disagreement_counts_only_at_the_chosen_scale(
    repo, near, recorded, monkeypatch, at_chosen, capsys
):
    """At the chosen scale a disagreement gives exit 1 and 'DISAGREED'. Only at the grid's other
    scales none is recorded: that is the consequence of the decision to check held-out test days
    at the chosen scale alone (reading H5), not a property of the fixed plan, whose closed form
    is not compared there."""
    s = near
    _disagreeing(monkeypatch, recorded, at_chosen)
    matrix = _matrix(s["tmp"], [SCORED])
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out.splitlines()
    document = _document(s)
    total = document[HELD_CHECK]
    head = corner_summary.HEADINGS[True][2]
    variants = document["units"][SCORED]["variants"].values()
    if at_chosen:
        assert (status, total["agreed"], total["pairs_agreed"]) == (1, False, 0)
        assert shown[-2] == (
            f"{head}: 0 of 40 pairs agree strictly, 0 with the tolerance: DISAGREED."
        )
        for scale, found in zip(recorded["chosen"], variants, strict=True):
            wrong = found[HELD_CHECK]["disagreements"]
            assert len(wrong) == 20 and {w["scale"] for w in wrong} == {scale}
    else:
        assert (status, total["agreed"], total["pairs_agreed"]) == (0, True, 40)
        assert all(found[HELD_CHECK]["disagreements"] == [] for found in variants)


def test_a_unit_and_variant_whose_fixed_plan_offers_nothing_is_listed_not_checked(
    repo, near, recorded, capsys
):
    s = near
    recorded["zero"] = {0}  # the forecast variant's calibration
    matrix = _matrix(s["tmp"], [SCORED])
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert status == 0, "\n".join(shown)
    reason = f"the fixed plan offers nothing (scale 0, {dial_splits.NO_SAFE_SCALE})"
    document = _document(s)
    variants = document["units"][SCORED]["variants"]
    assert variants["forecast"][HELD_CHECK] == {"status": "not checked", "reason": reason}
    assert variants["forecast"]["methods"][FIXED]["status"] == corner_search.NO_OFFER
    oracle = variants["oracle-departure"][HELD_CHECK]
    assert (oracle["pairs"], oracle["agreed"], oracle["scales"]) == (
        20,
        True,
        [recorded["chosen"][1]],
    )
    assert document[HELD_CHECK] == {
        "pairs": 20,
        "pairs_agreed_strict": 20,
        "pairs_agreed": 20,
        "agreed": True,  # of what was checked
        "not_checked": [f"{SCORED} | forecast: {reason}"],
    }
    assert recorded["checked"] == [(recorded["chosen"][1],)] * 20  # nothing under forecast
    assert f"  forecast: not checked: {reason}" in shown
    head = corner_summary.HEADINGS[True][2]
    assert shown[-2] == (
        f"{head}: 20 of 20 pairs agree strictly, 20 with the tolerance: agreed; 1 unit and "
        "variant(s) not checked, listed above."
    )
    timing = json.loads((s["out"] / corner_run.TIMING_NAME).read_text())
    hours = timing["path_hours"][SCORED]
    assert (hours["forecast"][HELD_CHECK], hours["oracle-departure"][HELD_CHECK]) == (0, 3200)
    assert _route_undefined(document, _rows(s)) == set()


@pytest.mark.parametrize("search", ["ran", "failed", "no-day"])
def test_with_nothing_checked_the_exit_status_is_the_searchs(
    repo, near, recorded, monkeypatch, search, capsys
):
    """Both variants' fixed plans offer nothing, or no day is searched (the test days' order made
    empty, as where no test day offers; the fixed plan's scale above 0): nothing is checked, the
    run-level `agreed` is null, not true, and the summary's line over all units says so; the
    exit status is then the search's alone: 0, or 1 when the second variant's search fails."""
    s = near
    recorded["zero"] = set() if search == "no-day" else {0, 1}
    if search == "no-day":
        monkeypatch.setattr(corner_run, "day_order", lambda unit, seed, days: [])
    if search == "failed":
        original = corner_run.search
        calls = []

        def failing(*arguments):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("a made-up failure of the search")
            return original(*arguments)

        monkeypatch.setattr(corner_run, "search", failing)
    matrix = _matrix(s["tmp"], [SCORED])
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert (s["out"] / corner_run.RESULTS_NAME).exists(), "\n".join(shown)
    document = _document(s)
    reason = f"the fixed plan offers nothing (scale 0, {dial_splits.NO_SAFE_SCALE})"
    if search == "no-day":
        reason = "no day searched (no test day offers under the variant)"
        assert all(scale > 0 for scale in recorded["chosen"])
    unchecked = ["forecast"] if search == "failed" else ["forecast", "oracle-departure"]
    total = document[HELD_CHECK]
    assert total == {
        "pairs": 0,
        "pairs_agreed_strict": 0,
        "pairs_agreed": 0,
        "agreed": None,
        "not_checked": [f"{SCORED} | {variant}: {reason}" for variant in unchecked],
    }
    assert recorded["checked"] == []
    head = corner_summary.HEADINGS[True][2]
    line = f"{head}: nothing checked on this run; {len(unchecked)} unit and variant(s) not checked"
    assert [x for x in shown if x.startswith(head)] == [f"{line}, listed above."]
    if search == "failed":
        assert status == 1
        assert document["failed"] == [
            f"{SCORED} | oracle-departure: RuntimeError: a made-up failure of the search"
        ]
    else:
        assert (status, document["failed"]) == (0, [])
    assert _route_undefined(document, _rows(s)) == set()


def test_methods_without_the_fixed_plan_are_refused_on_the_route(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _route_arguments(s, matrix, "--methods", "llf")  # the last --methods is read
    assert _refused(repo, s, arguments, capsys) == (
        "COULD NOT RUN: --methods without fixed-plan: on the sealed route the closed-form check "
        "on held-out test days runs at the fixed plan's calibrated scale, which only a run that "
        "searches fixed-plan calibrates; nothing read\n"
    )
    assert s["built"] == []  # refused before the model is loaded


# --- 2. the search sets nothing ------------------------------------------------------------------


def _settings(document: dict) -> dict:
    """Every setting the search could feed: per variant the methods' scales and reasons, the
    searched days and the calibration's counts."""
    return {
        name: {
            "scales": {m: (f["scale"], f["scale_reason"]) for m, f in found["methods"].items()},
            "searched_days": found["searched_days"],
            "calibration": found["calibration_evaluations"],
            "offering_calibration_days": found["offering_calibration_days"],
        }
        for name, found in document["units"][SCORED]["variants"].items()
    }


def _figures(document: dict) -> dict:
    return {
        name: {m: f.get("worst_slot_loss") for m, f in found["methods"].items()}
        for name, found in document["units"][SCORED]["variants"].items()
    }


def test_the_corner_search_sets_no_setting_of_any_method(repo, near, monkeypatch, capsys):
    s = near
    matrix = _matrix(s["tmp"], [SCORED])
    assert corner_run.main(_route_arguments(s, matrix), root=repo) == 0
    plain = _document(s)
    for found in plain["units"][SCORED]["variants"].values():
        assert all(f["scale"] > 0 for f in found["methods"].values())
    original = corner_search.search_day

    def elsewhere(prepared, method, beta):
        return original(prepared, method, 2.0)  # results of another scale

    monkeypatch.setattr(corner_search, "search_day", elsewhere)
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    capsys.readouterr()
    assert status in (0, 1)
    moved = _document(s)
    assert _settings(moved) == _settings(plain)
    assert _figures(moved) != _figures(plain)  # the replaced results were read


# --- 3. off the tag ------------------------------------------------------------------------------


def test_off_the_tag_the_route_could_not_run_and_opens_nothing_sealed(repo, setup, capsys):
    s = setup
    untagged = sealed_repo(s["tmp"] / "untagged", {}, tag=False)
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _route_arguments(s, matrix, held_tables=s["held_tables"])
    with watch(_sealed(repo), s["held_tables"]) as seen, _opens() as opened:
        status = corner_run.main(arguments, root=untagged)
    shown = capsys.readouterr().out
    assert (status, seen) == (2, [])
    assert shown.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert shown.endswith("; nothing read\n") and shown.count("\n") == 1
    assert _tables_read(opened) == []
    assert s["built"] == []  # refused before the model is loaded
    assert not (s["out"] / corner_run.RESULTS_NAME).exists()


# --- 4. refusals ---------------------------------------------------------------------------------


def _refused(repo: Path, s: dict, arguments: list, capsys) -> str:
    """The one line printed by a run that could not run (exit 2) and read no table."""
    with _opens() as opened, watch(_sealed(repo)) as seen:
        status = corner_run.main(arguments, root=repo)
    shown = capsys.readouterr().out
    assert (status, seen) == (2, [])
    assert shown.startswith("COULD NOT RUN: ") and shown.count("\n") == 1
    assert _tables_read(opened) == []
    assert not (s["out"] / corner_run.RESULTS_NAME).exists()
    return shown


def test_units_of_two_kinds_are_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED, "nrel-from-may-2019"])
    shown = _refused(repo, s, _route_arguments(s, matrix), capsys)
    assert shown.startswith("COULD NOT RUN: the units are refused: the matrix's gate units ")
    assert "never both" in shown and s["built"] == []


def test_the_two_tables_folders_swapped_are_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(
        s,
        matrix,
        tables=s["held_tables"],
        run=s["held_run"],
        held_tables=s["tables"],
        held_run=s["run"],
    )
    assert _refused(repo, s, arguments, capsys) == (
        f"COULD NOT RUN: a folder is refused: the tables folder {s['held_tables']} lies in the "
        f"held-out build's folder {s['held_tables']}; nothing read\n"
    )


def test_the_held_out_arguments_with_development_units_are_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], ["nrel-from-may-2019"])
    assert _refused(repo, s, _route_arguments(s, matrix), capsys) == (
        "COULD NOT RUN: --held-out-truth-run is for held-out units, and none of the matrix's "
        "gate units ['nrel-from-may-2019'] is a held-out unit; nothing read\n"
    )


def test_held_out_units_need_the_held_out_truth_run(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(s, matrix, held_tables=s["held_tables"])
    assert _refused(repo, s, arguments, capsys) == (
        "COULD NOT RUN: held-out units need --held-out-truth-run\n"
    )


def test_builds_of_two_settings_are_refused(repo, setup, held_setting, capsys):
    s = setup
    run, tables = held_setting
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _arguments(s, matrix, held_tables=tables, held_run=run)
    with _opens() as opened, watch(_sealed(repo)) as seen:
        status = corner_run.main(arguments, root=repo)
    assert (status, seen) == (2, [])
    assert capsys.readouterr().out == (
        f"COULD NOT RUN: the held-out tables in {tables} are of the held setting but the "
        f"development build {s['run'].name} is of the main setting; a run reads the two builds "
        "of one setting\n"
    )
    # `sealed_route.read_two_builds` checks the development build's tables first, as in the
    # learn_run route tests: no held-out table is read.
    assert [path for path in _tables_read(opened) if str(tables) in path] == []
    assert not (s["out"] / corner_run.RESULTS_NAME).exists()


@pytest.fixture
def sealed_matrix(repo, setup):
    """A copy of a held-out matrix placed in the made-up sealed folder (removed after the test)."""
    path = _sealed(repo) / "corner-matrix.json"
    shutil.copyfile(_matrix(setup["tmp"], [SCORED]), path)
    yield path
    path.unlink()


@pytest.mark.parametrize("place", ["sealed-tables", "sealed-matrix", "url-tables"])
def test_paths_into_the_sealed_folder_are_refused_unopened(
    repo, setup, sealed_matrix, place, capsys
):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    sealed_tables = _sealed(repo) / "norway"
    if place == "sealed-tables":
        arguments = _route_arguments(s, matrix, held_tables=sealed_tables)
        expected = "COULD NOT RUN: a folder is refused: the folder "
    elif place == "sealed-matrix":
        arguments = _route_arguments(s, sealed_matrix)
        expected = "COULD NOT RUN: a file is refused before it is opened: "
    else:
        written = f"file://{sealed_tables}"
        arguments = _route_arguments(s, matrix, held_tables=written)
        # the command line's type Path writes "file://" as "file:/"
        expected = f"COULD NOT RUN: the path {Path(written)} is written as a URL (file:)"
    assert _refused(repo, s, arguments, capsys).startswith(expected)


# --- 5. the leak rules ---------------------------------------------------------------------------


def test_a_model_that_reads_the_scored_units_other_days_is_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _route_arguments(s, matrix, model="empirical")
    assert _refused(repo, s, arguments, capsys) == (
        "COULD NOT RUN: model empirical builds a scored unit's forecasts from that unit's own "
        "other days, test days included; on the sealed route a scored unit's days reach a model "
        "only through its inputs, as in the cold start (its reading C12); nothing read\n"
    )


def test_a_look_ahead_model_is_refused_on_the_route(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    arguments = _route_arguments(s, matrix, model=scoring.LOOK_AHEAD)
    with _opens() as opened:
        status = corner_run.main(arguments, root=repo)
    shown = capsys.readouterr().out
    assert status == 1
    assert shown.startswith(
        f"REFUSED: model {scoring.LOOK_AHEAD} reads values of times after the decision time"
    )
    assert _tables_read(opened) == []


def test_a_seed_other_than_the_builds_split_is_refused(repo, setup, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    shown = _refused(repo, s, _route_arguments(s, matrix, "--seed", "2"), capsys)
    assert shown == (
        "COULD NOT RUN: --seed 2 is not the matrix's split_seed 1; on the sealed route a "
        "held-out unit has one split, the one its held-out build drew, and --seed names it; "
        "nothing read\n"
    )
    assert s["built"] == []


def test_a_matrix_of_another_split_rule_is_refused(repo, setup, capsys):
    s = setup
    rule = s["summary"]["calibration_rule"]
    assert rule["seed"] == 1
    matrix = _matrix(s["tmp"], [SCORED], split_seed=2)
    shown = _refused(repo, s, _route_arguments(s, matrix, "--seed", "2"), capsys)
    assert shown == (
        "COULD NOT RUN: the matrix's split rule (split_seed 2, n_calibration "
        f"{rule['n_calibration']}, calibration_half_below {rule['half_below']}) is not the rule "
        f"the held-out build {s['held_run'].name} drew its calibration days by (split_seed 1, "
        f"n_calibration {rule['n_calibration']}, calibration_half_below {rule['half_below']}); "
        "on the sealed route every scored unit is split by its build's rule\n"
    )


@pytest.mark.parametrize("days", ["49", "1"])
def test_fewer_days_than_the_plans_50_are_refused(repo, setup, days, capsys):
    s = setup
    matrix = _matrix(s["tmp"], [SCORED])
    shown = _refused(repo, s, _route_arguments(s, matrix, "--days", days), capsys)
    assert shown == (
        f"COULD NOT RUN: --days {days} is below the 50 fixed random days per unit of plan "
        "section 11 (O7); on the sealed route fewer days are a trial; nothing read\n"
    )
    assert s["built"] == []


def test_calibration_days_other_than_the_builds_are_refused(repo, setup, monkeypatch, capsys):
    s = setup
    original = corner_run.parse_unit_days

    def one_day_fewer(table_bytes, name, **readings):
        found = original(table_bytes, name, **readings)
        if name != SCORED:
            return found
        return dataclasses.replace(found, days=found.days[1:])

    monkeypatch.setattr(corner_run, "parse_unit_days", one_day_fewer)
    matrix = _matrix(s["tmp"], [SCORED])
    status = corner_run.main(_route_arguments(s, matrix), root=repo)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown.startswith(
        f"COULD NOT RUN: {SCORED}: the 19 calibration days this run draws are not the 20 the "
        f"held-out build {s['held_run'].name} searched its window on ("
    )
    assert shown.count("\n") == 1
    (model,) = s["built"]
    assert model.fitted == [] and model.predicted == []
    assert not (s["out"] / corner_run.RESULTS_NAME).exists()


# --- a table changed between its hash check and its parse ----------------------------------------


def test_a_held_out_table_changed_after_its_check_is_refused(repo, near, capsys):
    """The scored unit's table holds other bytes (`swap_after_read.grown`) from just after its
    first read, the read its SHA-256 is checked on: the run could not run and writes nothing."""
    s = near
    matrix = _matrix(s["tmp"], [SCORED])
    path = s["held_tables"] / f"{SCORED}-days.csv"
    with swapped_after_first_read(path, grown(path)) as swaps:
        status = corner_run.main(_route_arguments(s, matrix), root=repo)
    assert (status, capsys.readouterr().out) == (2, table_refusal(path))
    assert swaps == [str(path.resolve())]
    assert not (s["out"] / corner_run.RESULTS_NAME).exists()

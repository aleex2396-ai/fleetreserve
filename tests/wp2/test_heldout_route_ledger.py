"""The ledger of plan T9 (`ledger_run`) on held-out units through the sealed route (`sealed_route`).

The made-up repository at its made-up tag `frozen-plan` is made here (`repo`: `fake_heldout.
sealed_repo` holding the made-up sealed files of `heldout_truth_fake`, as in
`test_heldout_route_scoring`), so every call of the held-out loader below is given a root this
file makes (the rule of `test_heldout_routes`, item 4). Its held-out truth build and its
development build (the training unit caltech) are the fixtures `builds` and `held_setting` of
`test_heldout_route_scoring`, imported from it, as are its stand-in model, matrix and watches.
`learn_run` scores OSL_T (30 counting days, 15 calibration and 15 test days) through the route
once, with a stand-in model fitted on caltech whose forecasts give boxes (`BoxModel`), under the
committed matrix's two configurations whose energy bounds are the forecasts (`NAMES`), and the run
is recorded at the tag's commit in the made-up repository's `data/derived/runs` (kept out of git
there, so the tree stays clean; `ledger_fake.record`). The ledger runs on it with `--tree-days 3
--seed 7` and the made-up repository as the top folder, once (`_route_ledger`, kept with the
route run for the tests of item 1).

1. On the route the ledger ends with exit 0 and reads the scored unit's two tables only from the
   held-out build (no development table, no file under the made-up sealed folder opened, listed or
   scanned, no opening logged by the held-out loader). Its first three summary lines are the
   route's label (a held-out result: the tag's commit, the route run, the held-out build), the
   route's `checks` text and the hindsight text, which says that U is V_aff on the true sets less
   the run's certified value V_cert, from its forecast sets (and each row's U is that);
   `ledger.json` records the route and lists the readings L1 to L13. The identity holds on every
   day row, each row's offer and V_cert are the run's, some test day certifies a box, and the tree
   sample is the seeded draw of the test days' names alone. The run's calibrated scale is 0, so its
   offers are 0 on every row: offer + (R + A) + U + K = 0 + 5.714284 + (0 - 24) + (1 - 0) 24 =
   5.714284 on each test day of the first configuration (V_hind 5.714284 and V_aff 0, on the true
   sets, and V_cert 24, on the forecast sets, are the run's numbers). Outcome O8's ceiling ratio
   (`ceiling_ratio`) is printed on a line that says it is a held-out unit through the route, at
   the matrix's B=5: on each of the 15 test days the one-way ceiling it recomputes is the run's
   5.714284 (one-way sum 85.714 kWh), the two-way one is not below it, and V is the two-way sum
   over the one-way sum.
1b. Offers above 0: the route run with its scales made up (`SCALES`, per tier and rung, as
   `ledger_fake`'s hand-worked run makes a run's numbers), each test day's offer the scale times
   its V_cert and each mean offer the mean of the day offers, every other number the run's own,
   recorded at the tag's commit. The ledger ends with exit 0, and on each test day of the first
   configuration, tier A, fixed plan (scale 0.5): offer 0.5 x 24 = 12, R + A = 5.714284 - 0 =
   5.714284, U = 0 - 24 = -24, K = (1 - 0.5) 24 = 12, and 12 + 5.714284 - 24 + 12 = 5.714284, the
   ceiling; at the two-way rung (scale 1.25) the offer is 30 and K = -6, and 30 + 5.714284 - 24 - 6
   = 5.714284. The scales differ between tiers and rungs, so a ledger that read another tier's or
   rung's scale or offer would break the identity, which the run of item 1, whose scales are all
   0, cannot show.
2. Away from the tag (a made-up repository without it): exit 2, one line, nothing of the run, no
   table and nothing under the sealed folder opened.
3. Refused before anything is read (exit 2, one line): a matrix mixing held-out and development
   units; the held-out options with a development matrix; the held-out build given as the
   development tables; the run folder given inside the data folder outside its derived tables;
   no held-out truth run.
4. The run must be a run through the route recorded at the checked-out commit (`ledger_run`'s
   reading P-h, exit 2): a run whose results hold no record of the route; a run recorded at
   another commit (one line naming the run, its commit and the checked-out commit, refused before
   the run's results are read); a record naming another tag commit, or a fix commit at the tag
   (one line naming the record's commits and the gate's in full); a route run handed with a
   development matrix (another matrix).
4b. At a logged parsing fix, in a second made-up repository (`fixed`) with HEAD checked out at the
   fix commit F (a correction of a test file of the a.s.r. loader and its entry, after a refusal
   of the a.s.r. layout logged at the tag by hand), where the gate passes by its own checks: with
   the two builds made at F and `learn_run` run at F, the ledger reads that run (exit 0) and its
   first summary line and route record name the tag's commit and F. Refused with the line of
   item 4: the run `learn_run` made at the tag (only a test file and the plan differ, so the
   earlier-run rule alone would have read it but for its rule E3), and the run of F recorded at
   the correction before F or at a commit after F. A development run at F is refused for holding
   no record of the route. A run recorded at F whose route record names another fix commit, or no
   fix commit, is refused with one line naming the record's commits and the gate's in full.
5. The builds, setting and tables must be the run's (exit 2): the held setting's held-out build
   with the main development build (no held-setting table read); a run that read another held-out
   build, another development build, or another scored table.
6. The calibration days are the held-out build's own (exit 2): a held-out build that recorded
   another split rule (refused before any table is read), and one whose recorded calibration days
   of OSL_T are not those the split draws.
"""

import csv
import io
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import test_heldout_route_scoring as scoring
from fake_heldout import (
    ASR_FILE,
    approve,
    correction,
    entry,
    freeze,
    git,
    log_refusal,
    sealed_repo,
)
from gate_fake import COMMITTED, gate_cars, write_build
from heldout_truth_fake import sealed_files
from sealed_watch import watch
from swap_after_read import CHANGED, grown, swapped_after_first_read, table_refusal
from test_heldout_route_scoring import (
    CHECKS_TEXT,
    FIT_UNIT,
    SCORED,
    RouteModel,
    _opens,
    _record,
    _tables_read,
)

from fleetreserve import gate_run, heldout_loader, learn_run, ledger_run, sealed_route
from fleetreserve.block_sample import draw
from fleetreserve.rungs import RUNGS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp14"))
from ledger_fake import LEARN_MODULE, record  # noqa: E402

# The builds of the made-up repository, as fixtures of this module (module text).
builds = scoring.builds
held_setting = scoring.held_setting
MODEL = "route_box_model"
# The committed matrix's configurations whose energy bounds are the model's forecasts, so the
# stand-in model's forecasts give boxes (`BoxModel`).
NAMES = ("alt-energy-empirical-k0.1-empirical", "alt-energy-empirical-k0.1-oracle")
SAMPLE = ["--tree-days", "3", "--seed", "7"]
HINDSIGHT_TEXT = (
    "The ceiling, the losses R and A, the format loss F, V_aff - V_diag and the block product's "
    "terms are hindsight quantities of the scored test days, computed on their true sets; U is "
    "V_aff, computed on their true sets, less the run's certified value V_cert, which the run "
    "computed on its forecast sets; the offer and K come from the run's own scale and certified "
    "value. Every term is measured after the run scored its offers, and none is used to choose "
    "anything (no scale, configuration, kappa, dial or day)."
)
# The readings of `ledger_run`'s module text, listed in `ledger.json`.
READINGS = [f"L{k}" for k in range(1, 14)]
# The made-up scales of item 1b, per tier and rung.
SCALES = {
    "A": {"fixed": 0.5, "one-way": 0.75, "two-way": 1.25},
    "B": {"fixed": 0.25, "one-way": 0.5, "two-way": 1.0},
}


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    """The made-up repository at its made-up tag (module text)."""
    folder = tmp_path_factory.mktemp("route-ledger-repo")
    return sealed_repo(folder, sealed_files(folder / "made"))


class BoxModel(RouteModel):
    """The stand-in model of the route's tests, with forecasts that give boxes: every car stays to
    hour 7 after τ (past the service hours) at 2 kW, room 12 kWh, needing 6 kWh."""

    def predict(self, frame, kappas):
        table = super().predict(frame, kappas)
        table["d_hat"] = table["d_median"] = 7.0
        table["r_hat"], table["e_cap"], table["e_need"] = 2.0, 12.0, 6.0
        return table


def _matrix(folder: Path, units, **changes) -> Path:
    """The route tests' matrix (`test_heldout_route_scoring._matrix`, the held-out build's split
    rule) with the configurations `NAMES`."""
    path = scoring._matrix(folder, units, **changes)
    document = json.loads(path.read_text())
    document["configurations"] = [c for c in COMMITTED["configurations"] if c["name"] in NAMES]
    path.write_text(json.dumps(document))
    return path


def _at_head(patched, commit: str) -> None:
    """The commands' reading of the checked-out commit (the project's own repository otherwise)
    set to `commit`, which every caller has checked out in the made-up repository."""
    patched.setattr(gate_run, "_head_commit", lambda: commit)
    patched.setattr(ledger_run, "_head_commit", lambda: commit)


def _learn(root: Path, head: str, matrix: Path, b: dict, source: Path) -> None:
    """`learn_run` on OSL_T through the route with HEAD at `head`, into `source`, on the builds
    `b`."""
    source.mkdir()
    with pytest.MonkeyPatch.context() as patched:
        _at_head(patched, head)
        module = types.ModuleType(f"fleetreserve.models.{MODEL}")
        module.make_model = lambda matrix: BoxModel(MODEL, matrix.rho)
        patched.setitem(sys.modules, module.__name__, module)
        patched.setenv("FLEETRESERVE_RUN_DIR", str(source))
        arguments = ["--model", MODEL, "--matrix", str(matrix), "--tables", str(b["tables"])]
        arguments += ["--truth-run", str(b["run"])]
        arguments += ["--held-out-tables", str(b["held_tables"])]
        arguments += ["--held-out-truth-run", str(b["held_run"])]
        assert learn_run.main(arguments, root=root) == 0


@pytest.fixture(scope="module")
def route_run(repo, builds, tmp_path_factory):
    """`learn_run` on OSL_T through the route, recorded at the tag's commit (module text)."""
    folder = tmp_path_factory.mktemp("route-ledger")
    matrix = _matrix(folder, [SCORED])
    source = folder / "learn-source"
    _learn(repo, builds["tag"], matrix, builds, source)
    runs = repo / "data" / "derived" / "runs"
    run = record(runs, source, LEARN_MODULE, "learn-route", commit=builds["tag"])
    found = {"folder": folder, "matrix": matrix, "source": source, "runs": runs, "run": run}
    return {**found, "tag": builds["tag"]}


@pytest.fixture
def at_tag(builds, monkeypatch):
    _at_head(monkeypatch, builds["tag"])


def _arguments(s: dict, r: dict, run=None, matrix=None, **changes) -> list[str]:
    given = {
        "--tables": s["tables"],
        "--truth-run": s["run"],
        "--held-out-tables": s["held_tables"],
        "--held-out-truth-run": s["held_run"],
    }
    given.update({f"--{k.replace('_', '-')}": v for k, v in changes.items()})
    found = ["--run", str(run or r["run"]), "--matrix", str(matrix or r["matrix"]), *SAMPLE]
    for option, value in given.items():
        if value is not None:
            found += [option, str(value)]
    return found


def _ledger(arguments, root: Path, out: Path, monkeypatch) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return ledger_run.main(arguments, root=root)


def _rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.read_text().splitlines()))


def _truth_tables(seen) -> list[str]:
    """The truth tables opened: `_tables_read` without the run's own day table and the ledger's."""
    own = (learn_run.DAYS_NAME, ledger_run.DAYS_NAME)
    return [path for path in _tables_read(seen) if Path(path).name not in own]


# --- 1. the ledger through the route -------------------------------------------------------------


def _route_ledger(root: Path, builds: dict, route_run: dict) -> dict:
    """The ledger on the route run, with every opened path, every look at the sealed folder and
    every opening the held-out loader logged meanwhile. It runs once, at the first call; the
    outcome is kept in `route_run` (a fixture of the module) for the later calls. Its callers are
    the tests of item 1, each with the root `repo` (module text)."""
    if "ledger" not in route_run:
        out = route_run["folder"] / "route-ledger-out"
        openings = heldout_loader.logged_openings(root)
        with pytest.MonkeyPatch.context() as patched:
            _at_head(patched, builds["tag"])
            with watch(root / "data" / "heldout") as sealed, _opens() as seen:
                status = _ledger(_arguments(builds, route_run), root, out, patched)
        route_run["ledger"] = {
            "status": status,
            "out": out,
            "seen": seen,
            "sealed": sealed,
            "logged": heldout_loader.logged_openings(root)[len(openings) :],
        }
    return route_run["ledger"]


def test_the_route_ledger_reads_the_scored_tables_from_the_held_out_build_only(
    repo, builds, route_run
):
    found = _route_ledger(repo, builds, route_run)
    assert found["status"] == 0
    assert sorted(_truth_tables(found["seen"])) == sorted(
        str(builds["held_tables"] / f"{SCORED}-{kind}.csv") for kind in ("cars", "days")
    )
    assert found["sealed"] == [] and found["logged"] == []


def test_the_route_ledger_says_it_is_a_held_out_result(repo, builds, route_run):
    out = _route_ledger(repo, builds, route_run)["out"]
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    assert lines[0] == (
        f"{sealed_route.LABEL} (tag commit {builds['tag']}): a held-out result, the ledger of the "
        f"route run {route_run['run'].name}, the scored units' tables from the held-out build "
        f"{builds['held_run'].name} (the development build {builds['run'].name} checked for its "
        "setting, none of its tables read)."
    )
    assert lines[1] == CHECKS_TEXT
    assert lines[2] == HINDSIGHT_TEXT
    assert lines[3].startswith("Ledger of plan T9 (outcome O6), hourly product, from the recorded")
    ledger = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    route = ledger["sealed_route"]
    assert route["tag_commit"] == builds["tag"] and route["run"] == route_run["run"].name
    assert route["held_out_inputs"]["truth_run"] == builds["held_run"].name
    held = builds["summary"]["units"][SCORED]["sha256"]
    assert route["held_out_inputs"]["table_sha256"] == {
        f"{SCORED}-{kind}.csv": held[kind] for kind in ("cars", "days")
    }
    assert route["development_inputs"] == {
        "setting": "main",
        "truth_run": builds["run"].name,
        "table_sha256": {},
    }
    assert route["run_fit_units"] == [FIT_UNIT]
    assert route["run_kappa_selection"] == sealed_route.NO_SELECTION
    assert route["checks"] == CHECKS_TEXT and route["hindsight"] == HINDSIGHT_TEXT
    assert ledger["readings"] == READINGS


def _test_rows(run: Path) -> dict:
    """The run's test day rows by (configuration, date, site)."""
    return {
        (r["configuration"], r["date"], r["site"]): r
        for r in _rows(run / learn_run.DAYS_NAME)
        if r["role"] == "test"
    }


def test_the_route_ledgers_third_line_says_u_is_v_aff_less_the_runs_certified_value(
    repo, builds, route_run
):
    out = _route_ledger(repo, builds, route_run)["out"]
    third = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()[2]
    on_true_sets, about_u = third.split("; ")[:2]
    assert " U" not in on_true_sets and on_true_sets.endswith("computed on their true sets")
    assert about_u == (
        "U is V_aff, computed on their true sets, less the run's certified value V_cert, which the "
        "run computed on its forecast sets"
    )
    run_rows = _test_rows(route_run["run"])
    for row in _rows(out / ledger_run.DAYS_NAME):
        day = run_rows[(row["configuration"], row["date"], row["site"])]
        best, certified = float(day["ceiling_best_causal_kwh"]), float(day["certified_value_kwh"])
        assert float(row["u_kwh"]) == best - certified


def test_the_route_ledger_holds_the_identity_on_the_runs_numbers(repo, builds, route_run):
    out = _route_ledger(repo, builds, route_run)["out"]
    ledger = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    assert ledger["checks"]["failed"] is False
    run_rows = {
        (r["configuration"], r["date"], r["site"]): r
        for r in _rows(route_run["run"] / learn_run.DAYS_NAME)
        if r["role"] == "test"
    }
    assert len(run_rows) == 2 * 15  # two configurations, 15 test days
    rows = _rows(out / ledger_run.DAYS_NAME)
    assert len(rows) == 2 * 2 * len(RUNGS) * 15  # configurations, tiers, rungs, test days
    for row in rows:
        day = run_rows[(row["configuration"], row["date"], row["site"])]
        tier = "" if row["tier"] == "A" else "tier_b_"
        assert float(row["offer_kwh"]) == float(day[f"{row['rung']}_{tier}offer_kwh"])
        assert float(row["certified_kwh"]) == float(day["certified_value_kwh"])
        assert float(row["ceiling_kwh"]) == float(day["ceiling_hindsight_kwh"])
        terms = sum(float(row[f"{n}_kwh"]) for n in ("offer", "r_plus_a", "u", "k"))
        assert abs(terms - float(row["ceiling_kwh"])) <= 1e-9 * max(1.0, float(row["ceiling_kwh"]))
    assert max(float(r["certified_kwh"]) for r in rows) > 0.1  # some day certifies a box
    # The worked identity of the module text, item 1: every scale 0, so every offer 0.
    assert {float(r["scale"]) for r in rows} == {0.0}
    first = [r for r in rows if r["configuration"] == NAMES[0]]
    assert len(first) == 2 * len(RUNGS) * 15
    for row in first:
        assert float(row["offer_kwh"]) == 0.0 and float(row["best_causal_kwh"]) == 0.0
        assert float(row["certified_kwh"]) == 24.0 and float(row["u_kwh"]) == -24.0
        assert float(row["k_kwh"]) == 24.0
        assert abs(float(row["ceiling_kwh"]) - 5.714284) <= 1e-9
        assert abs(float(row["r_plus_a_kwh"]) - 5.714284) <= 1e-9


def test_the_route_ledger_prints_the_ceiling_ratio_as_a_held_out_result(repo, builds, route_run):
    out = _route_ledger(repo, builds, route_run)["out"]
    ledger = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    rows = _rows(out / ledger_run.RATIO_NAME)
    run_rows = _test_rows(route_run["run"])
    assert len(rows) == 15 and {r["unit"] for r in rows} == {SCORED}
    for row in rows:
        recorded = run_rows[(NAMES[0], row["date"], row["site"])]["ceiling_hindsight_kwh"]
        assert row["one_way_ceiling_kwh"] == row["run_ceiling_hindsight_kwh"] == recorded
        assert float(row["two_way_ceiling_kwh"]) >= float(row["one_way_ceiling_kwh"]) - 1e-9
        assert float(row["budget_kwh"]) == 5.0
    one = sum(float(r["one_way_ceiling_kwh"]) for r in rows)
    two = sum(float(r["two_way_ceiling_kwh"]) for r in rows)
    assert abs(one - 15 * 5.714284) <= 1e-6
    found = ledger["units"][SCORED]["ceiling_ratio"]
    assert found["test_days"] == 15 and found["v"] == pytest.approx(two / one, rel=1e-12)
    assert ledger["checks"]["ceiling_ratio_one_way_reproduced"]["breaks"] == []
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    (line,) = [line for line in lines if line.startswith("  O8 ceiling ratio V")]
    assert line.startswith(
        "  O8 ceiling ratio V at B=5 kWh, held-out unit through the sealed route, true sets, 15 "
        f"test days: one-way ceiling sum 85.714 kWh, two-way {two:.3f} kWh, V = {two / one:.3f}"
    )


def test_the_route_ledgers_tree_sample_is_drawn_from_the_test_days_names(repo, builds, route_run):
    out = _route_ledger(repo, builds, route_run)["out"]
    ledger = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    rows = _rows(route_run["run"] / learn_run.DAYS_NAME)
    names = sorted({f"{r['site']} {r['date']}" for r in rows if r["role"] == "test"})
    assert len(names) == 15
    assert ledger["tree_sample"]["days"][SCORED] == list(draw(SCORED, 7, names, 3))


# --- 1b. a route run whose offers are above 0 ----------------------------------------------------


def _with_offers(route_run: dict) -> Path:
    """The route run with offers above 0 (module text, item 1b), recorded at the tag's commit."""
    source = route_run["folder"] / "source-offers"
    shutil.copytree(route_run["source"], source)
    days = source / learn_run.DAYS_NAME
    lines = days.read_text().splitlines(keepends=True)
    top = [line for line in lines if line.startswith("#")]
    reader = csv.DictReader([line for line in lines if not line.startswith("#")])
    rows, offers = list(reader), {}
    for row in rows:
        if row["role"] != "test":
            continue
        certified = float(row["certified_value_kwh"])
        for tier, scales in SCALES.items():
            for rung, scale in scales.items():
                column = f"{rung}_offer_kwh" if tier == "A" else f"{rung}_tier_b_offer_kwh"
                row[column] = repr(scale * certified)
                offers.setdefault((row["configuration"], tier, rung), []).append(scale * certified)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=reader.fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    days.write_text("".join(top) + text.getvalue())
    path = source / learn_run.GATE_NAME
    document = json.loads(path.read_text())
    for (name, tier, rung), found in offers.items():
        entry = document["configurations"][name]["units"][SCORED]["rungs"][rung]
        scored = entry if tier == "A" else entry["tier_b"]
        scored["beta_hat"] = SCALES[tier][rung]
        scored["test"]["mean_offer_kwh_per_day"] = sum(found) / len(found)
    path.write_text(json.dumps(document))
    return record(route_run["runs"], source, LEARN_MODULE, "offers", commit=route_run["tag"])


def test_a_route_run_with_offers_above_0_holds_the_identity_at_its_scales(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch
):
    run = _with_offers(route_run)
    out = tmp_path / "out"
    assert _ledger(_arguments(builds, route_run, run=run), repo, out, monkeypatch) == 0
    ledger = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    assert ledger["checks"]["failed"] is False and ledger["sealed_route"]["run"] == run.name
    run_rows = _test_rows(run)
    rows = _rows(out / ledger_run.DAYS_NAME)
    assert len(rows) == 2 * 2 * len(RUNGS) * 15  # configurations, tiers, rungs, test days
    for row in rows:
        day = run_rows[(row["configuration"], row["date"], row["site"])]
        scale, certified = SCALES[row["tier"]][row["rung"]], float(day["certified_value_kwh"])
        assert float(row["scale"]) == scale and float(row["certified_kwh"]) == certified
        assert float(row["offer_kwh"]) == scale * certified > 0
        assert abs(float(row["k_kwh"]) - (1 - scale) * certified) <= 1e-12 * certified
        terms = sum(float(row[f"{n}_kwh"]) for n in ("offer", "r_plus_a", "u", "k"))
        assert abs(terms - float(row["ceiling_kwh"])) <= 1e-9 * max(1.0, float(row["ceiling_kwh"]))
    # The worked identity of the module text, item 1b.
    for rung, offer, k in (("fixed", 12.0, 12.0), ("two-way", 30.0, -6.0)):
        worked = [
            r for r in rows if (r["configuration"], r["tier"], r["rung"]) == (NAMES[0], "A", rung)
        ]
        assert len(worked) == 15
        for row in worked:
            assert (float(row["offer_kwh"]), float(row["k_kwh"])) == (offer, k)
            assert float(row["u_kwh"]) == -24.0
            assert abs(float(row["r_plus_a_kwh"]) - 5.714284) <= 1e-9
            assert abs(float(row["ceiling_kwh"]) - 5.714284) <= 1e-9
    tiers = ledger["units"][SCORED]["configurations"][NAMES[0]]["tiers"]
    assert tiers["A"]["fixed"]["mean_offer_kwh_per_test_day"] == 12.0
    assert tiers["A"]["fixed"]["run_mean_offer_kwh_per_test_day"] == 12.0
    assert tiers["B"]["fixed"]["mean_offer_kwh_per_test_day"] == 6.0


# --- 2. away from the tag ------------------------------------------------------------------------


def test_away_from_the_tag_the_ledger_could_not_run_and_opens_nothing(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    untagged = sealed_repo(tmp_path / "untagged", {}, tag=False)
    arguments = _arguments(builds, route_run)
    with watch(repo / "data" / "heldout") as sealed, _opens() as seen:
        status = _ledger(arguments, untagged, tmp_path / "out", monkeypatch)
    shown = capsys.readouterr().out
    assert status == 2
    assert shown.startswith("COULD NOT RUN: REFUSED BY THE GATE: ")
    assert shown.endswith("; nothing read\n") and shown.count("\n") == 1
    assert _tables_read(seen) == [] and sealed == []
    assert [path for path in seen if str(route_run["run"]) in path] == []
    assert list((tmp_path / "out").iterdir()) == []


# --- 3. refused before anything is read ----------------------------------------------------------


def _refused_before_reading(repo, arguments, out, monkeypatch, capsys) -> str:
    with _opens() as seen:
        status = _ledger(arguments, repo, out, monkeypatch)
    shown = capsys.readouterr().out
    assert status == 2 and shown.count("\n") == 1, shown
    assert [path for path in seen if str(repo / "data") in path] == []
    return shown


def test_a_matrix_mixing_held_out_and_development_units_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    matrix = _matrix(tmp_path, [SCORED, "nrel-from-may-2019"])
    arguments = _arguments(builds, route_run, matrix=matrix)
    shown = _refused_before_reading(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown.startswith("COULD NOT RUN: the units are refused: the matrix's gate units ")
    assert "never both" in shown


def test_the_held_out_options_with_a_development_matrix_are_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    matrix = _matrix(tmp_path, ["nrel-from-may-2019"])
    arguments = _arguments(builds, route_run, matrix=matrix, held_out_tables=None)
    shown = _refused_before_reading(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == (
        "COULD NOT RUN: --held-out-truth-run is for held-out units, and none of the matrix's "
        "gate units ['nrel-from-may-2019'] is a held-out unit; nothing read\n"
    )


def test_the_held_out_build_given_as_the_development_tables_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    arguments = _arguments(builds, route_run, tables=builds["held_tables"])
    shown = _refused_before_reading(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == (
        f"COULD NOT RUN: a folder is refused: the tables folder {builds['held_tables']} lies in "
        f"the held-out build's folder {builds['held_tables']}; nothing read\n"
    )


def test_a_run_folder_in_the_data_folder_outside_its_derived_tables_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    elsewhere = repo / "data" / "elsewhere" / route_run["run"].name
    shutil.copytree(route_run["run"], elsewhere)
    try:
        arguments = _arguments(builds, route_run, run=elsewhere)
        shown = _refused_before_reading(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    finally:
        shutil.rmtree(elsewhere.parent)
    assert shown == (
        f"COULD NOT RUN: a folder is refused: the folder {elsewhere} lies in the data folder "
        f"{repo / 'data'} outside its derived tables {repo / 'data' / 'derived'}; nothing read\n"
    )


def test_held_out_units_need_the_held_out_truth_run(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    arguments = _arguments(builds, route_run, held_out_truth_run=None)
    shown = _refused_before_reading(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == "COULD NOT RUN: held-out units need --held-out-truth-run\n"


# --- 4. the run must be a run through the route at the checked-out commit ------------------------


def _variant(route_run, label: str, change=None, **recorded) -> Path:
    """The route run's files, its gate document changed by `change`, recorded as a new run (at the
    tag's commit unless `recorded` names another)."""
    recorded.setdefault("commit", route_run["tag"])
    source = route_run["folder"] / f"source-{label}"
    shutil.copytree(route_run["source"], source)
    if change is not None:
        path = source / learn_run.GATE_NAME
        document = json.loads(path.read_text())
        change(document)
        path.write_text(json.dumps(document))
    return record(route_run["runs"], source, LEARN_MODULE, label, **recorded)


def _could_not_run(repo, arguments, out, monkeypatch, capsys) -> tuple[str, list]:
    with _opens() as seen:
        status = _ledger(arguments, repo, out, monkeypatch)
    shown = capsys.readouterr().out
    assert status == 2 and shown.startswith("COULD NOT RUN: ") and shown.count("\n") == 1, shown
    assert list(out.iterdir()) == []
    return shown, _truth_tables(seen)


def test_a_run_with_no_record_of_the_route_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    run = _variant(route_run, "no-route-record", lambda d: d.pop("sealed_route"))
    arguments = _arguments(builds, route_run, run=run)
    shown, tables = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == (
        f"COULD NOT RUN: {run.name} holds no record of the sealed route; on held-out units the "
        "ledger reads only a run made through the route\n"
    )
    assert tables == []


def _not_at_head(run: Path, commit: str, head: str, tag: str | None = None) -> str:
    """The one line refusing a run recorded at `commit` with HEAD at `head`: the tag's commit,
    or (with `tag`) a logged parsing fix of the tag's commit `tag`."""
    where = "the tag's commit"
    if tag is not None:
        where = f"the logged parsing fix of the tag's commit {tag}"
    return (
        f"COULD NOT RUN: {run.name} was recorded at commit {commit}, not at the checked-out "
        f"commit {head} ({where}); on the sealed route the ledger reads a scoring run only "
        "when it was recorded at the checked-out commit, so after a logged parsing fix every "
        "scoring run is made again at the fix commit\n"
    )


def test_a_route_run_recorded_at_another_commit_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    other = git(repo, "commit-tree", "HEAD^{tree}", "-m", "another commit")  # HEAD unchanged
    run = _variant(route_run, "other-commit", commit=other)
    arguments = _arguments(builds, route_run, run=run)
    shown, tables = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == _not_at_head(run, other, builds["tag"])
    assert tables == []


def test_a_route_record_of_another_tag_commit_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    def other_tag(document):
        document["sealed_route"]["tag_commit"] = "b" * 40

    run = _variant(route_run, "other-tag", other_tag)
    arguments = _arguments(builds, route_run, run=run)
    shown, tables = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == _other_record(run, ("b" * 40, None), (builds["tag"], None))
    assert tables == []


def _passed_at(tag: str, fix: str | None) -> str:
    where = f"the tag frozen-plan ({tag})"
    return where if fix is None else f"the logged parsing fix {fix} of {where}"


def _other_record(run: Path, recorded: tuple, gate: tuple) -> str:
    """The one line refusing a run whose route record names the (tag commit, fix commit)
    `recorded`, with the gate passing at `gate`."""
    return (
        f"COULD NOT RUN: {run.name} went through the sealed route at {_passed_at(*recorded)} by "
        f"its record, not at {_passed_at(*gate)}, where the gate passes now; on the sealed route "
        "the ledger reads a run only when its record names the tag's commit and the fix commit "
        "the gate passes at\n"
    )


def test_at_the_tag_a_route_record_naming_a_fix_commit_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    def claims_a_fix(document):
        document["sealed_route"]["fix_commit"] = "f" * 40

    run = _variant(route_run, "claims-a-fix", claims_a_fix)
    arguments = _arguments(builds, route_run, run=run)
    shown, tables = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == _other_record(run, (builds["tag"], "f" * 40), (builds["tag"], None))
    assert tables == []


# --- 4b. at a logged parsing fix, HEAD at the fix commit -----------------------------------------


def _builds_at(root: Path, head: str, label: str) -> dict:
    """The two builds made with HEAD at `head` by their writers (the held-out truth build through
    the gated opener, the development build by `gate_fake.write_build`), each recorded at `head`."""
    derived = root / "data" / "derived"
    held_run = derived / "runs" / f"20261116T00000{label}Z-wp14-heldout-truth"
    scoring._held_out_build(root, held_run)
    _record(held_run, head)
    (derived / f"development-{label}").mkdir()
    run, tables = write_build(derived / f"development-{label}", {FIT_UNIT: gate_cars(40, "T")})
    _record(run, head)
    held_tables = sealed_route.held_out_tables(root)
    return {"run": run, "tables": tables, "held_run": held_run, "held_tables": held_tables}


@pytest.fixture(scope="module")
def fixed(tmp_path_factory):
    """A second made-up repository: its made-up plan frozen at the tag, the a.s.r. layout logged
    as refused at the tag (by hand), a parsing fix (a correction of a test file of the a.s.r.
    loader, then its entry: the fix commit F, approved at once in the approvals record by hand, as
    the opener requires (reading H7)), and after F a second correction (`later`). With
    HEAD at the tag the two builds are made and `learn_run` scores OSL_T (`tag_run`); with HEAD at
    F the two builds are made again and `learn_run` scores again (`fix_run`). HEAD stays at F."""
    folder = tmp_path_factory.mktemp("route-ledger-fix")
    files = sealed_files(folder / "made")
    root = sealed_repo(folder, files, tag=False)
    tag = freeze(root)
    matrix = _matrix(folder, [SCORED])
    runs = root / "data" / "derived" / "runs"
    at_tag = _builds_at(root, tag, "1")
    _learn(root, tag, matrix, at_tag, folder / "source-tag")
    tag_run = record(runs, folder / "source-tag", LEARN_MODULE, "learn-at-tag", commit=tag)
    log_refusal(root, ASR_FILE, files[ASR_FILE], tag)
    corrected = correction(root, "asr")
    fix = entry(root, corrected)
    approve(root, fix)
    later = correction(root, "asr", name="later")
    git(root, "checkout", "-q", "--detach", fix)
    at_fix = _builds_at(root, fix, "2")
    _learn(root, fix, matrix, at_fix, folder / "source-fix")
    fix_run = record(runs, folder / "source-fix", LEARN_MODULE, "learn-at-fix", commit=fix)
    return {
        "root": root,
        "folder": folder,
        "tag": tag,
        "corrected": corrected,
        "fix": fix,
        "later": later,
        "matrix": matrix,
        "runs": runs,
        "builds": at_fix,
        "tag_run": tag_run,
        "fix_run": fix_run,
        "source": folder / "source-fix",
    }


@pytest.fixture
def at_fix(fixed, monkeypatch):
    """HEAD at F, checked out by `fixed`, and the gate passing there by its own checks."""
    assert git(fixed["root"], "rev-parse", "HEAD") == fixed["fix"]
    found = sealed_route.require_gate(fixed["root"])
    assert (found, found.fix_commit) == (fixed["tag"], fixed["fix"])
    _at_head(monkeypatch, fixed["fix"])


def _fix_arguments(fixed: dict, run: Path) -> list[str]:
    b = fixed["builds"]
    found = ["--run", str(run), "--matrix", str(fixed["matrix"]), *SAMPLE]
    found += ["--tables", str(b["tables"]), "--truth-run", str(b["run"])]
    found += ["--held-out-tables", str(b["held_tables"])]
    found += ["--held-out-truth-run", str(b["held_run"])]
    return found


def test_at_a_fix_a_run_made_at_the_fix_commit_is_read(fixed, at_fix, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert _ledger(_fix_arguments(fixed, fixed["fix_run"]), fixed["root"], out, monkeypatch) == 0
    lines = (out / ledger_run.SUMMARY_NAME).read_text().splitlines()
    assert lines[0].startswith(
        f"{sealed_route.LABEL} (tag commit {fixed['tag']}; at the logged parsing fix "
        f"{fixed['fix']}): a held-out result, the ledger of the route run "
        f"{fixed['fix_run'].name}, "
    )
    ledger = json.loads((out / ledger_run.RESULTS_NAME).read_text())
    assert ledger["checks"]["failed"] is False
    route = ledger["sealed_route"]
    assert (route["tag_commit"], route["fix_commit"]) == (fixed["tag"], fixed["fix"])


def _fix_refusal(fixed, run, tmp_path, monkeypatch, capsys) -> tuple[str, list]:
    arguments = _fix_arguments(fixed, run)
    return _could_not_run(fixed["root"], arguments, tmp_path / "out", monkeypatch, capsys)


def test_at_a_fix_a_run_made_at_the_tag_is_refused(fixed, at_fix, tmp_path, monkeypatch, capsys):
    """Only a test file and the plan differ between the tag and F, so the earlier-run rule of
    `run_evidence` would take the run but for its rule E3; this rule refuses it first."""
    run = fixed["tag_run"]
    shown, tables = _fix_refusal(fixed, run, tmp_path, monkeypatch, capsys)
    assert shown == _not_at_head(run, fixed["tag"], fixed["fix"], fixed["tag"])
    assert tables == []


def _fix_variant(fixed: dict, label: str, commit: str, change=None) -> Path:
    """The run made at F, its gate document changed by `change`, recorded at `commit`."""
    source = fixed["folder"] / f"source-{label}"
    shutil.copytree(fixed["source"], source)
    if change is not None:
        path = source / learn_run.GATE_NAME
        document = json.loads(path.read_text())
        change(document)
        path.write_text(json.dumps(document))
    return record(fixed["runs"], source, LEARN_MODULE, label, commit=commit)


@pytest.mark.parametrize("where", ["corrected", "later"])
def test_at_a_fix_a_run_at_another_commit_of_the_line_is_refused(
    fixed, at_fix, tmp_path, monkeypatch, capsys, where
):
    run = _fix_variant(fixed, f"at-{where}", fixed[where])
    shown, tables = _fix_refusal(fixed, run, tmp_path, monkeypatch, capsys)
    assert shown == _not_at_head(run, fixed[where], fixed["fix"], fixed["tag"])
    assert tables == []


def _another_fix(document):
    document["sealed_route"]["fix_commit"] = "f" * 40


def _no_fix(document):
    document["sealed_route"].pop("fix_commit")


def test_at_a_fix_a_run_of_the_fix_commit_whose_record_names_another_fix_is_refused(
    fixed, at_fix, tmp_path, monkeypatch, capsys
):
    run = _fix_variant(fixed, "record-other-fix", fixed["fix"], _another_fix)
    shown, tables = _fix_refusal(fixed, run, tmp_path, monkeypatch, capsys)
    gate = (fixed["tag"], fixed["fix"])
    assert shown == _other_record(run, (fixed["tag"], "f" * 40), gate)
    assert tables == []


def test_at_a_fix_a_run_of_the_fix_commit_whose_record_names_no_fix_is_refused(
    fixed, at_fix, tmp_path, monkeypatch, capsys
):
    run = _fix_variant(fixed, "record-no-fix", fixed["fix"], _no_fix)
    shown, tables = _fix_refusal(fixed, run, tmp_path, monkeypatch, capsys)
    gate = (fixed["tag"], fixed["fix"])
    assert shown == _other_record(run, (fixed["tag"], None), gate)
    assert tables == []


def test_at_a_fix_a_development_run_is_refused(fixed, at_fix, tmp_path, monkeypatch, capsys):
    run = _fix_variant(fixed, "dev-at-fix", fixed["fix"], lambda d: d.pop("sealed_route"))
    shown, tables = _fix_refusal(fixed, run, tmp_path, monkeypatch, capsys)
    assert shown == (
        f"COULD NOT RUN: {run.name} holds no record of the sealed route; on held-out units the "
        "ledger reads only a run made through the route\n"
    )
    assert tables == []


def test_a_route_run_handed_with_a_development_matrix_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    matrix = _matrix(tmp_path, ["nrel-from-may-2019"])
    arguments = _arguments(builds, route_run, matrix=matrix)
    arguments = arguments[: arguments.index("--held-out-tables")]
    shown, tables = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown.startswith(f"COULD NOT RUN: {route_run['run'].name} ran on another matrix: ")
    assert tables == []


# --- 5. the builds, the setting and the tables are the run's -------------------------------------


def test_the_held_setting_held_out_build_with_the_main_development_build_is_refused(
    repo, builds, route_run, held_setting, at_tag, tmp_path, monkeypatch, capsys
):
    run, tables = held_setting
    arguments = _arguments(builds, route_run, held_out_tables=tables, held_out_truth_run=run)
    shown, read = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown == (
        f"COULD NOT RUN: the held-out tables in {tables} are of the held setting but the "
        f"development build {builds['run'].name} is of the main setting; a run reads the two "
        "builds of one setting\n"
    )
    assert [path for path in read if str(tables) in path] == []


def _other_held_out_build(document):
    document["sealed_route"]["held_out_inputs"]["truth_run"] = "20261116T000009Z-another"


def _other_development_build(document):
    document["inputs"]["truth_run"] = "20261116T000009Z-another"


def _other_scored_table(document):
    document["sealed_route"]["held_out_inputs"]["table_sha256"][f"{SCORED}-cars.csv"] = "0" * 64


OTHER = {  # case -> (the change to the run's gate document, the refusal's text after the run name)
    "held-out build": (
        _other_held_out_build,
        " read the held-out build 20261116T000009Z-another, not the held-out build {held} given "
        "here",
    ),
    "development build": (
        _other_development_build,
        " read the development build 20261116T000009Z-another, not the development build {dev} "
        "given here",
    ),
    "scored table": (_other_scored_table, f"{SCORED}-cars.csv is not the table {{run}} read"),
}


@pytest.mark.parametrize("case", sorted(OTHER))
def test_a_run_that_read_another_build_or_table_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys, case
):
    change, text = OTHER[case]
    run = _variant(route_run, "other-" + case.replace(" ", "-"), change)
    arguments = _arguments(builds, route_run, run=run)
    shown, _ = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    reason = text.format(held=builds["held_run"].name, dev=builds["run"].name, run=run.name)
    expected = reason if case == "scored table" else run.name + reason
    assert shown == f"COULD NOT RUN: {expected}\n"


# --- 6. the calibration days are the held-out build's own ----------------------------------------


def _changed_build(repo, builds, label: str, change) -> Path:
    """A copy of the held-out build's run, under the same name in another folder of the made-up
    repository's derived tables, its summary changed by `change` and recorded at the tag."""
    copy = repo / "data" / "derived" / label / builds["held_run"].name
    shutil.copytree(builds["held_run"], copy)
    path = copy / "truth-summary.json"
    summary = json.loads(path.read_text())
    change(summary)
    path.write_text(json.dumps(summary))
    _record(copy, builds["tag"])
    return copy


def test_a_held_out_build_of_another_split_rule_is_refused_before_any_table(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    def other_seed(summary):
        summary["calibration_rule"]["seed"] = 2

    held_run = _changed_build(repo, builds, "other-rule", other_seed)
    arguments = _arguments(builds, route_run, held_out_truth_run=held_run)
    shown, tables = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown.startswith("COULD NOT RUN: the matrix's split rule (split_seed 1, ")
    assert f"the held-out build {held_run.name} drew its calibration days by (split_seed 2, " in (
        shown
    )
    assert tables == []


def test_calibration_days_other_than_the_held_out_builds_are_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys
):
    def one_day_less(summary):
        summary["windows"][SCORED]["calibration_dates"].pop()

    held_run = _changed_build(repo, builds, "other-days", one_day_less)
    arguments = _arguments(builds, route_run, held_out_truth_run=held_run)
    shown, _ = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert shown.startswith(
        f"COULD NOT RUN: {SCORED}: the 15 calibration days this run draws are not the 14 the "
        f"held-out build {held_run.name} searched its window on (1 drawn only here"
    )


# --- a file changed between its hash check and its parse -----------------------------------------


@pytest.mark.parametrize("which", ["table", "results", "days"])
def test_a_file_changed_after_its_check_is_refused(
    repo, builds, route_run, at_tag, tmp_path, monkeypatch, capsys, which
):
    """The scored unit's held-out table, or the scoring run's results or day table, holds other
    bytes (`swap_after_read.grown`) from just after its first read, the read its SHA-256 is
    checked on: the ledger could not run and writes nothing. The results file is read once more
    by `run_evidence.read_evidence` before the ledger's own read, so that read already finds the
    file changed since the run."""
    run = route_run["run"]
    path = {
        "table": builds["held_tables"] / f"{SCORED}-cars.csv",
        "results": run / learn_run.GATE_NAME,
        "days": run / learn_run.DAYS_NAME,
    }[which]
    arguments = _arguments(builds, route_run)
    with swapped_after_first_read(path, grown(path)) as swaps:
        shown, _ = _could_not_run(repo, arguments, tmp_path / "out", monkeypatch, capsys)
    assert swaps == [str(path.resolve())]
    assert (
        shown
        == {
            "table": table_refusal(path),
            "results": (
                f"COULD NOT RUN: {path.name} of {run.name} is not verified: the file has changed "
                "since the run (its hash differs)\n"
            ),
            "days": f"COULD NOT RUN: {path.name} of {run.name}: {CHANGED}\n",
        }[which]
    )

"""A held-out unit the held-out truth build abstained is recorded and left out of a run on the
sealed route, which goes on with the units that have tables (`sealed_route.read_two_builds`).

The made-up repository, its tag and the held-out build are those of `test_heldout_route_scoring`
(`heldout_truth_fake`'s sealed files). Its build writes the tables of OSL_T and TRO_R and records
`korea-visits` as abstained (one day, no calibration day: `summary["abstained"]`, no tables). A
run on the units OSL_T and korea-visits is the made-up case.

1. `learn_run`, `compare_run`, `corner_run`, `ceiling_report` and `o3_run` on OSL_T and
   korea-visits exit 0, score OSL_T alone, print one line
   `korea-visits: abstained in the held-out truth build <run>: <reason>; not scored, counted as
   not passing`, and write the abstention into the route's record (`held_out_inputs`, or
   `inputs` for the two commands that name their units).
2. A unit absent from the build that the build did not record as abstained still could not run
   (exit 2): the summary's abstention record is removed in a copy of the build, or the unit is
   one the build never heard of.
3. A run with no abstained unit is as before: the record holds no `abstained` key.
4. A run on the abstained units alone has no unit left to score (exit 2, naming each).
5. The development route is as before: a development unit absent from a development build still
   could not run.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
import test_heldout_route_ledger as ledger_tests
import test_heldout_route_scoring as scoring
from fake_heldout import sealed_repo
from gate_fake import COMMITTED, gate_cars, matrix_document, write_build
from heldout_truth_fake import sealed_files

from fleetreserve import (
    ceiling_report,
    compare_run,
    corner_run,
    gate_run,
    learn_run,
    ledger_run,
    o3_run,
    run_verify,
    sealed_route,
)
from fleetreserve.ceiling_summary import pass_over
from fleetreserve.truth_verify import CouldNotRun

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wp11"))
from corner_fake import NAMES as CORNER_NAMES  # noqa: E402

builds = scoring.builds
setup = scoring.setup

SCORED = "OSL_T"
ABSTAINED = "korea-visits"
UNITS = (SCORED, ABSTAINED)
MODEL = scoring.MODEL
DIAL = "0.03"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    folder = tmp_path_factory.mktemp("abstain-route")
    return sealed_repo(folder, sealed_files(folder / "made"))


def _line(s: dict, unit: str = ABSTAINED) -> str:
    reason = s["summary"]["abstained"][unit]["reason"]
    return (
        f"{unit}: abstained in the held-out truth build {s['held_run'].name}: {reason}; "
        "not scored, counted as not passing"
    )


def _matrix(folder: Path, units, names=scoring.NAMES) -> Path:
    document = matrix_document(names)
    document["gate_units"] = list(units)
    document["required_units"] = 1
    document["n_calibration"] = COMMITTED["n_calibration"]
    document["calibration_half_below"] = COMMITTED["calibration_half_below"]
    path = folder / f"matrix-{len(list(folder.iterdir()))}.json"
    path.write_text(json.dumps(document))
    return path


def _held(s: dict) -> list[str]:
    return ["--held-out-tables", str(s["held_tables"]), "--held-out-truth-run", str(s["held_run"])]


def _scoring_arguments(s: dict, units=UNITS, names=scoring.NAMES, *extra) -> list[str]:
    matrix = _matrix(s["tmp"], units, names)
    return [
        "--model",
        MODEL,
        "--matrix",
        str(matrix),
        "--tables",
        str(s["tables"]),
        "--truth-run",
        str(s["run"]),
        *_held(s),
        *extra,
    ]


def _named_arguments(s: dict, units=UNITS, *extra) -> list[str]:
    return [
        "--units",
        *units,
        "--tables",
        str(s["tables"]),
        "--truth-run",
        str(s["run"]),
        "--workers",
        "1",
        *_held(s),
        *extra,
    ]


def _results(s: dict, name: str) -> dict:
    return json.loads((s["out"] / name).read_text())


def test_the_made_up_build_abstains_the_unit(builds):
    assert ABSTAINED in builds["summary"]["abstained"]
    assert ABSTAINED not in builds["summary"]["units"]
    assert SCORED in builds["summary"]["units"]


# --- 1. each command leaves the abstained unit out and goes on ------------------------------------


def test_learn_run_scores_the_unit_with_tables_and_records_the_abstention(repo, setup, capsys):
    s = setup
    assert learn_run.main(_scoring_arguments(s), root=repo) == 0
    shown = capsys.readouterr().out.splitlines()
    assert shown[2] == _line(s)  # after the route's two lines
    results = _results(s, "learn.json")
    assert list(results["units"]) == [SCORED]
    held = results["sealed_route"]["held_out_inputs"]
    assert held["abstained"] == {ABSTAINED: s["summary"]["abstained"][ABSTAINED]["reason"]}
    assert set(held["table_sha256"]) == {f"{SCORED}-cars.csv", f"{SCORED}-days.csv"}


def test_learn_run_scores_a_korean_portfolio_beside_the_abstained_one(repo, setup, capsys):
    """The run's draw of the calibration days is the build's for a Korean portfolio too
    (`sealed_route.check_split_days`), so the unit is scored and the abstained one left out."""
    s = setup
    units = ("korea-home", ABSTAINED)
    assert learn_run.main(_scoring_arguments(s, units), root=repo) == 0
    assert _line(s) in capsys.readouterr().out.splitlines()
    assert list(_results(s, "learn.json")["units"]) == ["korea-home"]


def test_compare_run_scores_the_unit_with_tables_and_records_the_abstention(repo, setup, capsys):
    s = setup
    names = (
        "main-rounded-hour-k0.3-empirical",
        "main-rounded-hour-k0.3-oracle",
        "main-rounded-hour-k0.3-eps0.03-empirical",
        "main-rounded-hour-k0.3-eps0.03-oracle",
    )
    arguments = _scoring_arguments(s, UNITS, names, "--dial", DIAL, "--splits", "1")
    arguments += ["--methods", "re-solve", "fixed-plan"]
    assert compare_run.main(arguments, root=repo) == 0
    assert _line(s) in capsys.readouterr().out.splitlines()
    document = _results(s, compare_run.RESULTS_NAME)
    held = document["sealed_route"]["held_out_inputs"]
    assert list(held["abstained"]) == [ABSTAINED]
    assert set(document["units"]) == {SCORED}


def test_corner_run_scores_the_unit_with_tables_and_records_the_abstention(repo, setup, capsys):
    s = setup
    arguments = _scoring_arguments(s, UNITS, CORNER_NAMES, "--dial", DIAL)
    arguments += ["--methods", "fixed-plan", "llf"]
    status = corner_run.main(arguments, root=repo)
    shown = capsys.readouterr().out.splitlines()
    assert _line(s) in shown, "\n".join(shown)
    assert status in (0, 1)  # the corner search's own verdict is not this test's
    held = _results(s, corner_run.RESULTS_NAME)["sealed_route"]["held_out_inputs"]
    assert list(held["abstained"]) == [ABSTAINED]


def test_ceiling_report_reads_the_unit_with_tables_and_records_the_abstention(repo, setup, capsys):
    s = setup
    assert ceiling_report.main(_named_arguments(s, UNITS, "--every-day"), root=repo) == 0
    assert _line(s) in capsys.readouterr().out.splitlines()
    document = _results(s, ceiling_report.RESULTS_NAME)
    every = document["every_day"]
    assert list(every["units"]) == [SCORED, ABSTAINED]
    left = every["units"][ABSTAINED]
    assert left["o2_pass"] == "no day to check"  # decision 79 (3), not a could-not-run verdict
    assert left["reason"] == f"abstained ({s['summary']['abstained'][ABSTAINED]['reason']})"
    # the verdict is judged on the unit that has a day; the abstained unit is listed beside it
    assert every["pass_over_units"] == pass_over([every["units"][SCORED]["o2_pass"]])
    assert ABSTAINED in every["no_day_to_check"]
    assert list(document["inputs"]["abstained"]) == [ABSTAINED]


def test_o3_run_samples_the_unit_with_tables_and_records_the_abstention(repo, setup, capsys):
    s = setup
    extra = ["--days", "1", "--seed", "1", "--base-set", "day-average", "--lp27-days", "1"]
    assert o3_run.main(_named_arguments(s, UNITS, *extra), root=repo) == 0
    assert _line(s) in capsys.readouterr().out.splitlines()
    document = _results(s, o3_run.RESULTS_NAME)
    assert list(document["units"]) == [SCORED]
    assert list(document["inputs"]["abstained"]) == [ABSTAINED]


def test_ledger_run_reads_a_run_made_without_the_abstained_unit(
    repo, builds, tmp_path, monkeypatch, capsys
):
    matrix = ledger_tests._matrix(tmp_path, UNITS)
    source = tmp_path / "learn-source"
    ledger_tests._learn(repo, builds["tag"], matrix, builds, source)
    assert list(json.loads((source / "learn.json").read_text())["units"]) == [SCORED]
    runs = repo / "data" / "derived" / "runs"
    run = ledger_tests.record(
        runs, source, ledger_tests.LEARN_MODULE, "learn-abstain", commit=builds["tag"]
    )
    out = tmp_path / "ledger-out"
    with monkeypatch.context() as patched:
        ledger_tests._at_head(patched, builds["tag"])
        arguments = ledger_tests._arguments(builds, {"run": run, "matrix": matrix})
        status = ledger_tests._ledger(arguments, repo, out, patched)
    shown = capsys.readouterr().out.splitlines()
    assert status == 0, "\n".join(shown[:5])
    reason = builds["summary"]["abstained"][ABSTAINED]["reason"]
    line = (
        f"{ABSTAINED}: abstained in the held-out truth build {builds['held_run'].name}: {reason}; "
        "not scored, counted as not passing"
    )
    assert line in shown[:6], shown[:6]
    held = json.loads((out / ledger_run.RESULTS_NAME).read_text())["sealed_route"][
        "held_out_inputs"
    ]
    assert list(held["abstained"]) == [ABSTAINED]


# --- the count: an abstained unit stays in it and does not pass ----------------------------------


def test_with_one_of_nine_units_abstained_the_count_is_8_of_9_at_best(repo, builds):
    """The gate's decision over the nine sealed units (`gate/sealed-matrix/lgbm_tail.json`): eight
    units pass both rules, the ninth abstained. It stays among the nine, passes neither rule and
    shows as `abstained (<reason>)`; the count of passing units is 8 of 9 at best, never 8 of 8,
    and the reading of the 1.5 rule that asks every unit to pass is not met."""
    from gate_fake import TOP

    from fleetreserve import gate_config, gate_summary

    matrix = gate_config.load(TOP / "gate" / "sealed-matrix" / "lgbm_tail.json")
    assert len(matrix.gate_units) == 9
    abstained = matrix.gate_units[-1]
    record = {"truth_run": "R", "abstained": {abstained: "no calibration day at the start draw"}}

    def passing() -> dict:
        return {
            "values": [1.2] * 4,
            "ceilings": {c: [10.0] * 4 for c in gate_config.CEILINGS},
            "beta": 1.0,
        }

    left = sealed_route.abstained_gate_data(record)
    assert left == {abstained: {"reason": "abstained (no calibration day at the start draw)"}}
    pairs = {u: (passing(), {**passing(), "values": [3.0] * 4}) for u in matrix.gate_units[:-1]}
    pairs[abstained] = (left[abstained], left[abstained])
    for rule in gate_summary.gate_stat.RATIO_RULES:
        found = gate_summary.decide(pairs, matrix, "best-causal", rule)
        assert len(found["units"]) == 9
        assert found["n_pass_share"] == 8 < len(found["units"])
        row = found["units"][abstained]
        assert row["passes_share"] is False and row["passes_ratio"] is False
        assert row["reason"] == "abstained (no calibration day at the start draw)"
        assert (found["go"] is False) == (rule == "all_units")


def test_learn_run_keeps_the_abstained_unit_in_its_gate_decisions(repo, setup):
    s = setup
    assert learn_run.main(_scoring_arguments(s), root=repo) == 0
    gate = _results(s, "learn-gate.json")
    reason = f"abstained ({s['summary']['abstained'][ABSTAINED]['reason']})"
    assert gate["counted_units_not_scored"] == {ABSTAINED: reason}
    for key in ("gate", "gate_tier_b"):
        for pair in gate[key]["by_pair"].values():
            for decision in pair["decisions"]:
                assert list(decision["units"]) == [SCORED, ABSTAINED]
                row = decision["units"][ABSTAINED]
                assert (row["passes_share"], row["passes_ratio"], row["reason"]) == (
                    False,
                    False,
                    reason,
                )


# --- 2. only a recorded abstention is let through -------------------------------------------------


def _copied_build(s: dict, tmp: Path, change) -> Path:
    """A copy of the held-out build's run folder whose summary `change` has altered, recorded at
    the tag's commit again; the tables folder is the original."""
    run = (
        s["held_run"].parent / f"20261117T0000{len(list(s['held_run'].parent.iterdir())):02d}Z-copy"
    )
    shutil.copytree(s["held_run"], run)  # inside the made-up repository: git checks its commit
    summary = json.loads((run / "truth-summary.json").read_text())
    change(summary)
    (run / "truth-summary.json").write_text(json.dumps(summary))
    scoring._record(run, s["tag"])
    return run


def _drop_record(summary: dict) -> None:
    del summary["abstained"][ABSTAINED]


def _drop_reason(summary: dict) -> None:
    summary["abstained"][ABSTAINED] = {"sessions_loaded": 3}


@pytest.mark.parametrize("change", [_drop_record, _drop_reason])
def test_a_unit_without_a_recorded_abstention_still_could_not_run(repo, setup, capsys, change):
    s = setup
    held_run = _copied_build(s, s["tmp"], change)
    arguments = _scoring_arguments(s)
    arguments[arguments.index("--held-out-truth-run") + 1] = str(held_run)
    assert learn_run.main(arguments, root=repo) == 2
    shown = capsys.readouterr().out
    assert f"COULD NOT RUN: {ABSTAINED} is not in the recorded truth build" in shown
    assert not (s["out"] / "learn.json").exists()
    arguments = _named_arguments(s, UNITS, "--every-day")
    arguments[arguments.index("--held-out-truth-run") + 1] = str(held_run)
    assert ceiling_report.main(arguments, root=repo) == 2
    assert (
        f"COULD NOT RUN: {ABSTAINED} is not in the recorded truth build" in capsys.readouterr().out
    )


def test_a_unit_recorded_as_written_is_read_not_left_out(repo, setup):
    """A unit the summary lists with tables is read even when its name is also under `abstained`."""
    s = setup

    def both(summary: dict) -> None:
        summary["abstained"][SCORED] = {"reason": "recorded beside its tables"}

    held_run = _copied_build(s, s["tmp"], both)
    assert sealed_route.abstained_in_build(held_run, (SCORED,)) == {}


# --- 3. a run with no abstained unit is as before -------------------------------------------------


def test_a_run_with_no_abstained_unit_records_none(repo, setup, capsys):
    s = setup
    assert learn_run.main(_scoring_arguments(s, (SCORED,)), root=repo) == 0
    shown = capsys.readouterr().out
    assert "abstained" not in shown
    held = _results(s, "learn.json")["sealed_route"]["held_out_inputs"]
    assert sorted(held) == ["setting", "table_sha256", "truth_run"]


# --- 4. no unit left ------------------------------------------------------------------------------


def test_a_run_on_abstained_units_alone_has_no_unit_left_to_score(repo, setup, capsys):
    s = setup
    both = ("korea-lodging", ABSTAINED)
    assert learn_run.main(_scoring_arguments(s, both), root=repo) == 2
    shown = capsys.readouterr().out
    assert shown.startswith("COULD NOT RUN: no unit is left to score: every unit named abstained")
    assert _line(s) in shown and _line(s, "korea-lodging") in shown
    assert ceiling_report.main(_named_arguments(s, both, "--every-day"), root=repo) == 2
    assert capsys.readouterr().out.startswith("COULD NOT RUN: no unit is left to score")
    assert sorted(p.name for p in s["out"].iterdir()) == []


# --- 5. the development route is as before --------------------------------------------------------


def test_a_development_unit_a_build_does_not_hold_still_could_not_run(tmp_path):
    """Nothing is left out off the route: `gate_run.read_inputs`, which the development route
    reads through, refuses a unit its summary lacks, though the summary lists it as abstained."""
    run, tables = write_build(tmp_path, {scoring.FIT_UNIT: gate_cars(5, "T")})
    summary = json.loads((run / "truth-summary.json").read_text())
    summary["abstained"] = {"nrel-from-may-2019": {"reason": "no day"}}
    (run / "truth-summary.json").write_text(json.dumps(summary))
    scoring._record(run, "0" * 40)
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patched.setattr(gate_run, "_head_commit", lambda: "0" * 40)
        with pytest.raises(CouldNotRun, match="nrel-from-may-2019 is not in the recorded truth"):
            gate_run.read_inputs(run, tables, ("nrel-from-may-2019",))

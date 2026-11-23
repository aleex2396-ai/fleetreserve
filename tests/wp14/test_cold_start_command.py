"""The cold-start command (`cold_start.main`) on made-up development units, and its refusals.

The made-up build: `nrel-from-may-2019` with cars from 1 January to 2 March 2020 but none on 21
January and two only on 31 January (a day that does not count), so 60 counting days; the pooled
Norwegian unit with 10 days at each of two sites (20 site-days); the training unit `caltech` with 5
days. Two stand-in models (`cold_start_fake.HistoryModel`, the second forecasting half an hour
later), each fitted on `caltech`.

1. A run on the two development units ends with exit 0 and writes every output; the gate of the
   sealed-data path is not called for development units; the summary's first line says the run is
   a rehearsal.
2. The scored days are the stated ones, written out by hand: the counting days after the first
   56, 28 February to 2 March, the same at every k, in the results and in the days file; the local
   days of each k are the first k counting days; the pooled unit, with no counting day after its
   first 56, gets no figures.
3. Each figure equals its value recomputed from the days file, and each forecast table holds the
   local and scored days only.
4. A held-out unit is refused outside the gate (a folder that is not a git repository, and one
   without the tag), with nothing read and no model loaded; so are a training unit, an unknown
   name and a mix of development and held-out units; a model that reads other days' outcomes and
   the same model twice. With the gate passing (patched), a held-out unit's tables are read only
   from the held-out folder and checked against the held-out truth build, and a held-out run
   without that build is refused; with the gate returning a fix commit, as it does at a logged
   parsing fix after the tag, the results' gate record names the tag's commit and the fix's; with
   the gate passing again after the models are loaded at a later fix than at first, the run
   could not run (exit 2, the gate's message, no table checked or read).
5. The folders a run reads: the default held-out tables are under the root the command is given;
   a folder in a held-out place (`--tables` in the sealed folder, through a link to it, or in the
   held-out build's folder; `--truth-run` or `--held-out-tables` in the sealed folder) is refused
   before any table is checked or read, and a folder written inside the sealed folder before any
   link in it is followed; under the top folder's `data/` only its derived tables are read.
"""

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from cold_start_fake import FIT_UNIT, cold_cars, register, write_build
from gate_fake import COMMIT, matrix_document
from swap_after_read import grown, swapped_after_first_read, table_refusal

from fleetreserve import (
    cold_start,
    cold_start_gate,
    gate_run,
    learn_run,
    run_verify,
    sealed_route,
    truth,
)
from fleetreserve.heldout_loader import SealRefused
from fleetreserve.heldout_truth_build import UNITS as HELD_OUT_UNITS
from fleetreserve.truth_verify import CouldNotRun

UNIT, POOLED = "nrel-from-may-2019", "norway-development-pooled"
NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)
SCORED = ["2020-02-28 A", "2020-02-29 A", "2020-03-01 A", "2020-03-02 A"]
MODELS = ("cold_one", "cold_two")


def _counting_dates() -> list[str]:
    """1 January to 2 March 2020 but 21 and 31 January, written out by month."""
    january = [f"2020-01-{d:02d}" for d in range(1, 32) if d not in (21, 31)]
    february = [f"2020-02-{d:02d}" for d in range(1, 30)]
    return [*january, *february, "2020-03-01", "2020-03-02"]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def _models(monkeypatch):
    register(monkeypatch, MODELS[0])
    register(monkeypatch, MODELS[1], shift=0.5)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        UNIT: cold_cars(62, skip=(20,), thin=(30,)),
        POOLED: cold_cars(10, "B") + cold_cars(10, "C"),
        FIT_UNIT: cold_cars(5, "T"),
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _arguments(build, *units, model=MODELS[0], reference=MODELS[1]) -> list[str]:
    run, tables, matrix = build
    return [
        *("--model", model, "--reference", reference, "--dial", "0.02"),
        *("--units", *units),
        *("--matrix", str(matrix), "--tables", str(tables), "--truth-run", str(run)),
    ]


def _no_gate(root=None):
    raise AssertionError("the gate is called for development units")


@pytest.fixture(scope="module")
def full(build, tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        _models(monkeypatch)
        monkeypatch.setattr(cold_start_gate, "require_gate", _no_gate)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        status = cold_start.main(_arguments(build, UNIT, POOLED))
    return status, out


def _document(out: Path) -> dict:
    return json.loads((out / cold_start.RESULTS_NAME).read_text())


def _days(out: Path) -> list[dict]:
    lines = (out / cold_start.DAYS_NAME).read_text().splitlines()
    assert lines[0].startswith("# REHEARSAL")
    return list(csv.DictReader(lines[1:]))


# --- 1. the run ----------------------------------------------------------------------------------


def test_a_development_run_ends_with_exit_0_and_writes_every_output(full):
    status, out = full
    assert status == 0
    document = _document(out)
    assert document["failed"] == []
    assert document["kind"] == cold_start_gate.DEVELOPMENT
    for name in (cold_start.SUMMARY_NAME, cold_start.DAYS_NAME, cold_start.TIMING_NAME):
        assert (out / name).is_file()
    tables = document["forecast_tables"]
    assert len(tables) == 10
    for entry in tables.values():
        assert hashlib.sha256((out / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert document["models"]["cold_started"] == "cold_one"
    assert document["models"]["reference"] == "cold_two"
    assert document["run"]["local_days"] == [0, 7, 14, 28, 56]
    assert document["run"]["dial"] == 0.02
    assert "gate" not in document and "held_out_inputs" not in document


def test_the_summary_says_it_is_a_rehearsal(full):
    _, out = full
    lines = (out / cold_start.SUMMARY_NAME).read_text().splitlines()
    assert lines[0].startswith("REHEARSAL on development units: not outcome O9")
    assert f"{UNIT}: 60 counting days; 4 scored days (2020-02-28 A to 2020-03-02 A)" in lines
    assert f"{POOLED}: no scored day: 20 counting days, none after the first 56" in lines
    assert "  tier A (driver-protected)" in lines and "  tier B (deadline contract)" in lines


# --- 2. the scored days are the stated ones -------------------------------------------------------


def test_the_scored_days_are_the_counting_days_after_the_first_56(full):
    _, out = full
    entry = _document(out)["units"][UNIT]
    assert entry["counting_days"] == 60
    assert entry["scored_days"] == SCORED
    dates = _counting_dates()
    assert len(dates) == 60
    for k in (0, 7, 14, 28, 56):
        assert entry["local_days"][str(k)] == [f"{d} A" for d in dates[:k]]


def test_the_days_file_scores_the_same_days_at_every_k(full):
    _, out = full
    rows = _days(out)
    dates = _counting_dates()
    for model in MODELS:
        for k in (0, 7, 14, 28, 56):
            for tier in ("A", "B"):
                chosen = [
                    r for r in rows if (r["model"], r["k"], r["tier"]) == (model, str(k), tier)
                ]
                scored = [f"{r['date']} {r['site']}" for r in chosen if r["role"] == "scored"]
                local = [r["date"] for r in chosen if r["role"] == "local"]
                assert scored == SCORED
                assert local == dates[:k]
    assert {r["unit"] for r in rows} == {UNIT}


def test_a_unit_without_a_scored_day_gets_no_figures(full):
    _, out = full
    document = _document(out)
    assert document["units"][POOLED]["scored_days"] == []
    assert document["units"][POOLED]["local_days"] == {}
    for model in MODELS:
        assert set(document["figures"][model]) == {UNIT}


# --- 3. the figures and the tables ---------------------------------------------------------------


def test_each_figure_equals_its_value_from_the_days_file(full):
    _, out = full
    document, rows = _document(out), _days(out)
    seen_offer = False
    for model in MODELS:
        for k in (0, 7, 14, 28, 56):
            for tier in ("A", "B"):
                found = document["figures"][model][UNIT][str(k)][tier]
                scored = [
                    r
                    for r in rows
                    if (r["model"], r["k"], r["tier"], r["role"]) == (model, str(k), tier, "scored")
                ]
                offers = [float(r["offer_kwh"]) for r in scored]
                offer = sum(offers) / len(offers)
                assert found["offer_kwh_per_counting_scored_day"] == pytest.approx(offer)
                ceiling = sum(float(r["hindsight_kwh"]) for r in scored) / len(scored)
                share = found["offer_share"]["hindsight-free-decision-hour"]["share_of_ceiling"]
                assert share == pytest.approx(offer / ceiling)
                offering = [r for r in scored if r["offering"] == "True"]
                if found["scale"] > 0 and offering:
                    seen_offer = True
                    slot = sum(float(r["slot_loss_at_scale"]) for r in offering) / len(offering)
                    assert found["availability"] == pytest.approx(1 - slot)
                else:
                    assert found["availability"] is None
                if k in (0, 7):  # fewer than 9 local days: both tiers abstain (reading C7)
                    assert found["scale"] == 0.0 and offer == 0.0
    assert seen_offer


def test_each_forecast_table_holds_the_local_and_scored_days_only(full):
    _, out = full
    document = _document(out)
    dates = _counting_dates()
    for name, entry in document["forecast_tables"].items():
        k = int(name.split("k=")[1])
        with open(out / entry["file"]) as handle:
            found = {row["date"] for row in csv.DictReader(handle)}
        assert found == {*dates[:k], *(s.split()[0] for s in SCORED)}


# --- 4. refusals ---------------------------------------------------------------------------------


@pytest.fixture
def watched(monkeypatch):
    """Records every table read (by its path, or parsed from the bytes its check read), input
    check (`gate_run.read_inputs`, which `gate_run.check_inputs` and the held-out route's
    `read_two_builds` call) and model load."""
    seen = []
    loads, parses = cold_start.load_unit_days, cold_start.parse_unit_days
    checks, models = gate_run.read_inputs, learn_run.load_model

    def load(folder, unit, **readings):
        seen.append(("read", unit))
        return loads(folder, unit, **readings)

    def parse(read, unit, **readings):
        seen.append(("read", unit))
        return parses(read, unit, **readings)

    def check(run, tables, units):
        seen.append(("check", tuple(units)))
        return checks(run, tables, units)

    def model(name, matrix):
        seen.append(("model", name))
        return models(name, matrix)

    monkeypatch.setattr(cold_start, "load_unit_days", load)
    monkeypatch.setattr(cold_start, "parse_unit_days", parse)
    monkeypatch.setattr(cold_start.gate_run, "read_inputs", check, raising=True)
    monkeypatch.setattr(cold_start.learn_run, "load_model", model, raising=True)
    _verified(monkeypatch)
    _models(monkeypatch)
    return seen


def test_a_held_out_unit_is_refused_outside_the_gate(build, tmp_path, watched, capsys):
    outside = tmp_path / "not-a-repository"
    outside.mkdir()
    untagged = tmp_path / "untagged"
    untagged.mkdir()
    subprocess.run(["git", "init", "--quiet", str(untagged)], check=True)
    for root in (outside, untagged):
        for unit in HELD_OUT_UNITS:
            assert cold_start.main(_arguments(build, unit), root=root) == 2
    shown = capsys.readouterr().out
    assert shown.count("REFUSED BY THE GATE") == 2 * len(HELD_OUT_UNITS)
    assert "no tag frozen-plan" in shown
    assert watched == []


def test_training_unknown_and_mixed_units_are_refused_before_anything_is_read(
    build, tmp_path, watched, capsys
):
    for units in ((FIT_UNIT,), ("nowhere",), (UNIT, "OSL_T"), (UNIT, UNIT)):
        assert cold_start.main(_arguments(build, *units), root=tmp_path) == 2
    shown = capsys.readouterr().out
    assert shown.count("COULD NOT RUN: the units are refused") == 4
    assert "training units" in shown and "never both" in shown
    assert watched == []


def test_an_outcome_reader_and_the_same_model_twice_are_refused(build, watched, capsys):
    assert cold_start.main(_arguments(build, UNIT, model="empirical")) == 1
    assert "REFUSED: model empirical reads other days' outcomes" in capsys.readouterr().out
    assert cold_start.main(_arguments(build, UNIT, reference=MODELS[0])) == 2
    assert not [event for event in watched if event[0] in ("read", "check")]


def test_with_the_gate_passing_a_held_out_unit_is_read_from_the_held_out_build(
    build, tmp_path, watched, monkeypatch, capsys
):
    calls = []

    def gate(root=None):
        calls.append(root)
        return "f" * 40

    monkeypatch.setattr(cold_start_gate, "require_gate", gate)
    held_run, held_tables = write_build(tmp_path, {"OSL_T": cold_cars(58)})
    base = _arguments(build, "OSL_T")
    assert cold_start.main([*base, "--held-out-tables", str(held_tables)]) == 2
    assert "held-out units need --held-out-truth-run" in capsys.readouterr().out
    assert [e for e in watched if e[0] in ("read", "check")] == []
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    arguments = [
        *base,
        *("--held-out-tables", str(held_tables), "--held-out-truth-run", str(held_run)),
    ]
    assert cold_start.main(arguments) == 0
    # once in the refused call; twice in the run: before reading anything, after loading models
    assert len(calls) == 3
    document = _document(out)
    assert document["kind"] == cold_start_gate.HELD_OUT
    assert document["gate"] == {"tag_commit": "f" * 40}
    assert set(document["held_out_inputs"]["table_sha256"]) == {"OSL_T-cars.csv", "OSL_T-days.csv"}
    assert set(document["inputs"]["table_sha256"]) == {
        f"{FIT_UNIT}-cars.csv",
        f"{FIT_UNIT}-days.csv",
    }
    assert document["units"]["OSL_T"]["scored_days"] == ["2020-02-26 A", "2020-02-27 A"]
    assert ("check", ("OSL_T",)) in watched and ("check", (FIT_UNIT,)) in watched
    assert not (Path(build[1]) / "OSL_T-cars.csv").exists()


def test_at_a_parsing_fix_the_gate_record_names_the_fix_commit(
    build, tmp_path, watched, monkeypatch
):
    found = sealed_route.GateCommit("f" * 40, "e" * 40)  # as the gate returns at a fix
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: found)
    held_run, held_tables = write_build(tmp_path, {"OSL_T": cold_cars(58)})
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    held = ("--held-out-tables", str(held_tables), "--held-out-truth-run", str(held_run))
    assert cold_start.main([*_arguments(build, "OSL_T"), *held]) == 0
    assert _document(out)["gate"] == {"tag_commit": "f" * 40, "fix_commit": "e" * 40}


def test_the_gate_again_at_another_fix_than_at_first_is_refused(
    build, tmp_path, watched, monkeypatch, capsys
):
    tag, first, later = "f" * 40, "e" * 40, "d" * 40
    passes = iter([sealed_route.GateCommit(tag, first), sealed_route.GateCommit(tag, later)])
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: next(passes))
    held_run, held_tables = write_build(tmp_path, {"OSL_T": cold_cars(58)})
    held = ("--held-out-tables", str(held_tables), "--held-out-truth-run", str(held_run))
    assert cold_start.main([*_arguments(build, "OSL_T"), *held]) == 2
    assert capsys.readouterr().out == (
        "COULD NOT RUN: REFUSED BY THE GATE: the gate passes now at the logged parsing fix "
        f"{later} of the tag frozen-plan ({tag}), not at the logged parsing fix {first} of the "
        f"tag frozen-plan ({tag}), where it passed when this run started; nothing read\n"
    )
    assert [e for e in watched if e[0] in ("read", "check")] == []


def test_require_gate_raises_the_loaders_refusal(tmp_path):
    with pytest.raises(SealRefused, match="not the top folder of a git repository"):
        cold_start_gate.require_gate(tmp_path)


# --- 5. the folders a run reads ------------------------------------------------------------------


def test_the_default_held_out_tables_are_under_the_root_given(tmp_path):
    derived = tmp_path / "data" / "derived"
    assert cold_start_gate.held_out_tables(tmp_path) == derived / "truth-heldout"
    assert cold_start_gate.held_out_tables(tmp_path, truth.HELD) == derived / "truth-heldout-held"


def test_the_command_takes_the_default_held_out_tables_under_its_root(
    build, tmp_path, watched, monkeypatch, capsys
):
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: "f" * 40)
    checks = cold_start.gate_run.read_inputs
    asked = []

    def check(run, tables, units):
        if tuple(units) != ("OSL_T",):
            return checks(run, tables, units)
        asked.append(Path(tables))
        raise CouldNotRun("stopped here by the test")

    monkeypatch.setattr(cold_start.gate_run, "read_inputs", check)
    arguments = [*_arguments(build, "OSL_T"), "--held-out-truth-run", str(tmp_path / "run")]
    assert cold_start.main(arguments, root=tmp_path) == 2
    assert "stopped here by the test" in capsys.readouterr().out
    assert asked == [tmp_path / "data" / "derived" / "truth-heldout"]


def _place(root: Path, case: str) -> tuple[str, Path]:
    """(the argument, the folder) of one refused case of the test below."""
    sealed = root / "data" / "heldout" / "x"
    sealed.mkdir(parents=True)
    if case == "tables linked into the sealed folder":
        link = root / "linked"
        link.symlink_to(sealed, target_is_directory=True)
        return "--tables", link
    return {
        "tables in the sealed folder": ("--tables", sealed),
        "tables in the held-out build's folder": (
            "--tables",
            root / "data" / "derived" / "truth-heldout",
        ),
        "truth run in the sealed folder": ("--truth-run", sealed / "run"),
        "held-out tables in the sealed folder": ("--held-out-tables", sealed),
    }[case]


@pytest.mark.parametrize(
    "case",
    [
        "tables in the sealed folder",
        "tables linked into the sealed folder",
        "tables in the held-out build's folder",
        "truth run in the sealed folder",
        "held-out tables in the sealed folder",
    ],
)
def test_a_folder_in_a_held_out_place_is_refused_before_anything_is_read(
    build, tmp_path, watched, monkeypatch, capsys, case
):
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: "f" * 40)
    option, folder = _place(tmp_path, case)
    held_out = option == "--held-out-tables"
    arguments = _arguments(build, "OSL_T" if held_out else UNIT)
    if held_out:
        arguments += ["--held-out-truth-run", str(tmp_path / "run")]
    if option in arguments:  # --tables and --truth-run are set by _arguments: replace the value
        arguments[arguments.index(option) + 1] = str(folder)
    else:
        arguments += [option, str(folder)]
    assert cold_start.main(arguments, root=tmp_path) == 2
    shown = capsys.readouterr().out
    assert "COULD NOT RUN: a folder is refused" in shown and "nothing read" in shown
    assert watched == []


def test_a_folder_written_in_the_sealed_folder_is_refused_before_any_link_is_followed(
    tmp_path, monkeypatch
):
    sealed = tmp_path / "data" / "heldout"
    resolve = Path.resolve

    def no_call(self, strict=False):
        if sealed in self.parents:  # a path inside the sealed folder, not the folder itself
            raise AssertionError(f"a link was followed in {self}")
        return resolve(self, strict=strict)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "resolve", no_call)
        with pytest.raises(cold_start_gate.FoldersRefused, match="outside its derived tables"):
            cold_start_gate.check_folders(tmp_path, sealed / "x")
        with pytest.raises(cold_start_gate.FoldersRefused, match="outside its derived tables"):
            cold_start_gate.check_folders(tmp_path, tmp_path / "tables", (sealed / "run",))


def test_under_the_data_folder_only_the_derived_tables_are_read(tmp_path):
    data = tmp_path / "data"
    # read: the derived tables, the held-out tables as --held-out-tables, folders outside data/
    cold_start_gate.check_folders(
        tmp_path,
        data / "derived" / "truth",
        (tmp_path / "runs.noindex" / "run", data / "derived" / "truth-heldout", None),
    )
    cold_start_gate.check_folders(tmp_path, tmp_path / "heldout" / "x")
    # refused: anything else under data/, whatever its name, written plainly or not
    for refused in (data / "heldout-copy", data / "raw", data, data / "derived" / ".." / "x"):
        with pytest.raises(cold_start_gate.FoldersRefused, match="outside its derived tables"):
            cold_start_gate.check_folders(tmp_path, refused)
        with pytest.raises(cold_start_gate.FoldersRefused, match="outside its derived tables"):
            cold_start_gate.check_folders(tmp_path, tmp_path / "tables", (refused,))


# --- a held-out table changed between its hash check and its parse -------------------------------


@pytest.mark.parametrize("name", ["OSL_T-cars.csv", "OSL_T-days.csv"])
def test_a_held_out_table_changed_after_its_check_is_refused(
    build, tmp_path, watched, monkeypatch, capsys, name
):
    """With the gate passing (patched), the held-out table holds other bytes
    (`swap_after_read.grown`) from just after its first read, the read its SHA-256 is checked
    on: the run could not run and writes nothing, never scoring a table other than the one
    checked."""
    monkeypatch.setattr(cold_start_gate, "require_gate", lambda root=None: "f" * 40)
    held_run, held_tables = write_build(tmp_path, {"OSL_T": cold_cars(58)})
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    held = ("--held-out-tables", str(held_tables), "--held-out-truth-run", str(held_run))
    path = held_tables / name
    with swapped_after_first_read(path, grown(path)) as swaps:
        status = cold_start.main([*_arguments(build, "OSL_T"), *held])
    assert (status, capsys.readouterr().out) == (2, table_refusal(path))
    assert swaps == [str(path.resolve())]
    assert list(out.iterdir()) == []

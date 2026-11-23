"""`tools/compare_shared_check.py`, the WP10 pass-mark item, on made-up recorded runs.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real. The throwaway repository commits the project's
`gate/matrix.json` and a stand-in package `fleetreserve` whose `compare_run`, `dial_choice` and
`truth_build` only copy made-up files into the run folder and exit with a given status, so each
run's recorded command has the form of a real run (`python -m fleetreserve.compare_run --model
<name> --dial <dial> ...`). Most tests use a made-up requirement of three methods (`re-solve`,
`llf`, `cvar`), one model (`km`) and 3 splits.

1. A truth build, a dial-choice run and the comparison with and without the draw, all at the
   checked-out commit and sharing their days: shown (0), one line per requirement.
2. A comparison run missing a method: failed (1), the line naming it.
3. A method on another box (another certified value, other offering or calibration days or
   another ceiling on one split), a split-table row missing or a stray row: failed (1).
4. A stale run, a run without a recorded truth build (none named, its folder missing, at another
   commit, of the held setting, a table differing): could not run (2).
5. No comparison run, no dial-choice run: could not run (2), "NO RUN".
6. Two dials (one per site type): one comparison per dial, each read for its own units.
7. A run under the label whose results name another model, dial or draw, or that cannot decide
   (another number of splits or seeds, another matrix, the held setting, a run that exited 1,
   another command, a tier or a variant missing, a unit that cannot be split): could not run (2).
8. The no-pause run with other paths, another forecast table, other methods or other truth tables
   than the run without the draw: failed (1); with no run without the draw: could not run (2).
9. Output files: one missing or not listed, the results naming another hash or number of rows,
   another split-table header: failed (1); a file changed after the run: could not run (2).
10. A unit or variant missing a result, a summary, a gate reading or its record `measured_on`,
    or a record of other than 8 path rows a day or of other days: failed (1).
11. A listed method with no registry name or one not in the registry is "no method", a
    requirement no command writes is "not built": could not run (2), each also alone among
    passing runs; a failure wins.
12. An unexpected error in one requirement's check (the others still checked) or outside them:
    could not run (2), never Python's own exit status 1.
13. The newest run under a label is the one read.
14. The real lists (the owner's decision of 16 November 2026): the eleven methods of plan section
    10 that the registry implements, every one of them, the one model `lgbm_tail`, the dial
    sources and no requirement not built. Made-up runs of the eleven methods for `lgbm_tail`,
    with and without the draw, pass the real requirement (0); a run lacking one of the eleven
    fails it (1); no run with the draw: could not run (2).
"""

import csv
import hashlib
import importlib
import io
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest
import truth_summary_real
from git_run import run_git

from fleetreserve import compare_methods, compare_run, compare_splits, truth_build
from fleetreserve.gate_config import parse
from fleetreserve.run_record import record_run

TOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOP / "tools"))
check = importlib.import_module("compare_shared_check")

MATRIX_BYTES = (TOP / "gate" / "matrix.json").read_bytes()
MATRIX_SHA = hashlib.sha256(MATRIX_BYTES).hexdigest()
MATRIX = parse(json.loads(MATRIX_BYTES))
UNITS = list(MATRIX.gate_units)
CEILINGS = list(MATRIX.ceilings)
VARIANTS = list(compare_run.VARIANTS)
TIERS = ["A", "B"]
NREL = [u for u in UNITS if u.startswith("nrel-")]
NORWAY = [u for u in UNITS if u not in NREL]
SPLITS = 3
DIAL = 0.02
DAYS = 30
THREE = ("re-solve", "llf", "cvar")
SETUP = check.Setup(
    methods=tuple(check.Method(f"made-up {name}", name) for name in THREE),
    models=("km",),
    splits=SPLITS,
    not_built=(),
)
THIRD = Fraction(1, 3)

STAND_IN = """import os
import shutil
import sys

source = os.environ["FAKE_FROM"]
for name in sorted(os.listdir(source)):
    shutil.copy(os.path.join(source, name), os.path.join(os.environ["FLEETRESERVE_RUN_DIR"], name))
sys.exit(int(os.environ["FAKE_EXIT"]))
"""


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch):
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def git(repo: Path, *args: str) -> None:
    run_git(["git", "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "gate").mkdir(parents=True)
    (root / "fleetreserve").mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    (root / ".gitignore").write_text("runs.noindex/\n__pycache__/\n")
    (root / "tracked.txt").write_text("one\n")
    (root / "gate" / "matrix.json").write_bytes(MATRIX_BYTES)
    (root / "fleetreserve" / "__init__.py").write_text("")
    for module in ("compare_run", "dial_choice", "truth_build", "other"):
        (root / "fleetreserve" / f"{module}.py").write_text(STAND_IN)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


class Maker:
    """Records made-up runs in the throwaway repository."""

    def __init__(self, repo: Path, monkeypatch, folder: Path):
        self.repo, self.monkeypatch, self.folder = repo, monkeypatch, folder
        self.count = 0
        self.labels = set()
        self.truth_run = None
        self.salt = ""

    def record(self, label, module, arguments, files: dict, exit_code=0) -> Path:
        if label in self.labels:  # a run folder is named by its start to the second
            second = int(time.time())
            while int(time.time()) == second:
                time.sleep(0.02)
        self.labels.add(label)
        self.count += 1
        source = self.folder / f"files-{self.count}"
        source.mkdir(parents=True)
        for name, text in files.items():
            (source / name).write_text(text)
        self.monkeypatch.setenv("FAKE_FROM", str(source))
        self.monkeypatch.setenv("FAKE_EXIT", str(exit_code))
        runs = self.repo / "runs.noindex"
        before = set(runs.glob("*")) if runs.exists() else set()
        command = [sys.executable, "-m", module, *arguments]
        assert record_run(label, command, start_dir=self.repo) == exit_code
        (made,) = set(runs.glob("*")) - before
        return made

    def truth(self, setting="main", salt="") -> Path:
        """A recorded truth build whose summary is the text of the real writer
        (`truth_summary_real`): the main setting has no `setting` key, the held setting has it."""
        written = {
            unit: {kind: table_hash(unit, kind, salt) for kind in ("cars", "days")}
            for unit in UNITS
        }
        files = {truth_build.SUMMARY_NAME: truth_summary_real.summary_text(written, setting)}
        made = self.record("wp2-truth", "fleetreserve.truth_build", [], files)
        self.truth_run, self.salt = made.name, salt
        return made

    def dials(self, model="km", workplace=DIAL, norway=DIAL, change=None) -> Path:
        document = {
            "model": model,
            "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
            "inputs": inputs(self.truth_run, self.salt),
            "run": {"splits": SPLITS if model == "km" else check.SPLITS},
            "dial_choice": {
                "nrel-from-may-2019": {"A": {"dial": workplace}, "B": {"dial": 0.1}},
                "norway-development-pooled": {"A": {"dial": norway}, "B": {"dial": 0.1}},
            },
        }
        if change is not None:
            change(document)
        arguments = ["--model", model, "--matrix", "gate/matrix.json", "--splits", "200"]
        return self.record(
            check.dial_label(model),
            "fleetreserve.dial_choice",
            arguments,
            {"dial-choice.json": json.dumps(document)},
        )

    def compare(
        self,
        model="km",
        dial=DIAL,
        share=None,
        *,
        methods=THREE,
        splits=SPLITS,
        change=None,
        finish=None,
        omit=(),
        exit_code=0,
        module="fleetreserve.compare_run",
        forecasts="car,departure\n1,2.0\n",
    ) -> Path:
        made = (self.truth_run, self.salt, forecasts)
        files = compare_files(model, dial, share, made, methods, splits, change, finish, omit)
        arguments = ["--model", model, "--dial", f"{dial:g}", "--matrix", "gate/matrix.json"]
        if share is not None:
            arguments += ["--no-pause-share", "1/3"]
        label = check.compare_label(model, dial, None if share is None else "1/3")
        return self.record(label, module, arguments, files, exit_code)

    def everything(self, model="km") -> tuple[Path, Path, Path, Path]:
        truth = self.truth()
        dials = self.dials(model)
        main = self.compare(model)
        third = self.compare(model, share=THIRD)
        return truth, dials, main, third


@pytest.fixture
def make(repo, monkeypatch, tmp_path) -> Maker:
    return Maker(repo, monkeypatch, tmp_path / "made")


def table_hash(unit: str, kind: str, salt: str = "") -> str:
    return hashlib.sha256(f"{unit}-{kind}{salt}".encode()).hexdigest()


def inputs(truth_run, salt="") -> dict:
    return {
        "setting": "main",
        "truth_run": truth_run,
        "table_sha256": {
            f"{unit}-{kind}.csv": table_hash(unit, kind, salt)
            for unit in UNITS
            for kind in ("cars", "days")
        },
    }


def _digest(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _scale(method: str, seed: int) -> float:
    if method in ("llf", "cvar") and seed == 1:
        return 0.0
    return 0.3 if method == "llf" else 0.5


def split_rows(methods, splits) -> list[dict]:
    """Rows of the split table in the order unit, variant, tier, method, seed: every method on the
    same days, with the same certified value (offer over scale) on each split."""
    rows = []
    for u, unit in enumerate(UNITS):
        for v, variant in enumerate(VARIANTS):
            for tier in TIERS:
                for method in methods:
                    for seed in range(1, splits + 1):
                        value = 10.0 + u + 2.0 * v + 0.37 * seed
                        scale = _scale(method, seed)
                        rows.append(
                            {
                                "unit": unit,
                                "variant": variant,
                                "tier": tier,
                                "method": method,
                                "seed": seed,
                                "calibration_days": 10,
                                "test_days": 20,
                                "offering_calibration_days": 9,
                                "scale": scale,
                                "scale_zero_reason": "made up" if scale == 0 else None,
                                "offer_kwh_per_counting_test_day": scale * value,
                                "offering_test_days": 18,
                                "failing_test_days": None if scale == 0 else 1,
                                "mean_test_slot_loss": None if scale == 0 else 0.05,
                                "mean_test_driver_loss": None if scale == 0 else 0.01,
                                "share_of_best_causal_ceiling": None if scale == 0 else 0.2,
                                **{
                                    compare_splits.ceiling_column(c): 40.0 + u + k + 0.1 * seed
                                    for k, c in enumerate(CEILINGS)
                                },
                            }
                        )
    return rows


def compare_files(model, dial, share, made, methods, splits, change, finish, omit) -> dict:
    """The five files of a made-up comparison run (`compare_run.output_names`); `made`: the
    truth build's folder name, its tables' salt and the forecast table's text."""
    truth_run, salt, forecasts_text = made
    names = compare_run.output_names(share or 0)
    drawn = "third" if share else "none"
    document = {
        "model": model,
        "dial": dial,
        "methods": list(methods),
        "configurations": {variant: f"made-up {variant}" for variant in VARIANTS},
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
        "inputs": inputs(truth_run, salt),
        "run": {
            "splits": splits,
            "seeds": [1, splits],
            "tiers": list(TIERS),
            "variants": list(VARIANTS),
            "ceilings": list(CEILINGS),
        },
        "units": {
            unit: {
                "status": "calibratable",
                "counting_days": DAYS,
                "variants": {
                    variant: {
                        "configuration": f"made-up {variant}",
                        "offering_days": DAYS - 2,
                        "methods": {
                            m: {t: {"mean_offer_kwh_per_test_day": 1.0} for t in TIERS}
                            for m in methods
                        },
                        "measured_on": {
                            "days": DAYS,
                            "path_rows": 8 * DAYS,
                            "boxes_sha256": _digest(unit, variant, "boxes", drawn),
                            "paths_sha256": _digest(unit, variant, "paths"),
                            "steered_sha256": _digest(unit, variant, "steered", drawn),
                        },
                    }
                    for variant in VARIANTS
                },
                "gate_reading": {m: {"mean_offer_kwh_per_test_day": 1.0} for m in methods},
            }
            for unit in UNITS
        },
        "failed": [],
    }
    if share:
        document["no_pause"] = {"share": "1/3", "seed": 1}
    rows = split_rows(methods, splits)
    if change is not None:
        change(document, rows)
    table = io.StringIO()
    writer = csv.writer(table, lineterminator="\n")
    columns = compare_splits.columns(document["run"]["ceilings"])
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(row[c]) for c in columns])
    splits_text = table.getvalue()
    document["split_table"] = {
        "file": names["splits"],
        "rows": len(rows),
        "sha256": hashlib.sha256(splits_text.encode()).hexdigest(),
    }
    document["forecast_table"] = {
        "file": names["table"],
        "rows": 1,
        "sha256": hashlib.sha256(forecasts_text.encode()).hexdigest(),
    }
    if finish is not None:
        finish(document)
    files = {
        names["results"]: json.dumps(document),
        names["summary"]: "made-up summary: " + ", ".join(methods) + "\n",
        names["table"]: forecasts_text,
        names["splits"]: splits_text,
        names["timing"]: "{}\n",
    }
    return {name: text for name, text in files.items() if name not in omit}


def _cell(value) -> str:
    if value is None:
        return ""
    return repr(value) if isinstance(value, float) else str(value)


def run_check(repo: Path, setup=SETUP) -> int:
    return check.main(["--top", str(repo)], setup=setup)


def line_of(out: str, start: str) -> str:
    (found,) = [line for line in out.splitlines() if line.startswith(start)]
    return found


MAIN = "km at dial 0.02:"
WITH_DRAW = "km at dial 0.02, no-pause 1/3 seed 1:"


def test_everything_shown_passes(make, repo, capsys):
    _, dials, main, third = make.everything()
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    groups = len(UNITS) * len(VARIANTS) * len(TIERS) * SPLITS
    line = line_of(out, "dials: km: ")
    assert dials.name in line and "nrel-before-may-2019 0.02 (read at nrel-from-may-2019)" in line
    line = line_of(out, f"shown: {MAIN} ")
    assert f"{main.name}; read for {', '.join(UNITS)}; 3 of 3 methods in its files" in line
    assert f"{len(UNITS) * len(VARIANTS)} unit-variants" in line
    assert f"split table {groups * 3} rows" in line
    # on the first split only re-solve offers, so the value is compared on the other splits
    assert f"on {groups} unit, variant, tier and split groups" in line
    assert f"certified value compared on {groups - groups // SPLITS} of them" in line
    line = line_of(out, f"shown: {WITH_DRAW} ")
    assert third.name in line and f"those of {main.name}" in line
    assert out.splitlines()[-1] == "3 of 3 requirements shown, 0 failed, 0 could not run"


def test_a_run_missing_a_method_fails(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare(methods=("re-solve", "llf"))
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    out = capsys.readouterr().out
    line = line_of(out, f"FAILED: {MAIN} ")
    assert "methods 2 of 3; missing cvar" in line
    assert out.splitlines()[-1] == "1 of 3 requirements shown, 2 failed, 0 could not run"


def _row(rows, unit, variant, tier, method, seed):
    (found,) = [
        r
        for r in rows
        if (r["unit"], r["variant"], r["tier"], r["method"], r["seed"])
        == (unit, variant, tier, method, seed)
    ]
    return found


def _other_value(document, rows):
    _row(rows, UNITS[1], VARIANTS[0], "A", "llf", 2)["offer_kwh_per_counting_test_day"] *= 1.001


def _other_offering_days(document, rows):
    _row(rows, UNITS[1], VARIANTS[0], "A", "llf", 2)["offering_test_days"] = 17


def _other_calibration_days(document, rows):
    _row(rows, UNITS[1], VARIANTS[0], "A", "llf", 2)["offering_calibration_days"] = 8


def _other_ceiling(document, rows):
    _row(rows, UNITS[1], VARIANTS[0], "A", "llf", 2)[compare_splits.ceiling_column(CEILINGS[1])] = (
        1.0
    )


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (_other_value, "llf has the certified value"),
        (_other_offering_days, "llf has offering_test_days 17"),
        (_other_calibration_days, "llf has offering_calibration_days 8"),
        (_other_ceiling, f"llf has {compare_splits.ceiling_column(CEILINGS[1])} 1.0"),
    ],
    ids=["certified value", "offering test days", "offering calibration days", "a ceiling"],
)
def test_a_method_on_another_box_fails(make, repo, capsys, change, said):
    make.truth()
    make.dials()
    make.compare(change=change)
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    line = line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")
    assert f"{UNITS[1]} {VARIANTS[0]} tier A split 2: {said}" in line


def _drop_row(document, rows):
    rows.remove(_row(rows, UNITS[2], VARIANTS[1], "B", "cvar", 3))


def _stray_row(document, rows):
    rows.append({**rows[-1], "seed": SPLITS + 1})


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (_drop_row, f"{UNITS[2]} {VARIANTS[1]} tier B split 3: no row of cvar"),
        (_stray_row, f"a row of {UNITS[-1]} {VARIANTS[-1]} tier B split {SPLITS + 1}"),
    ],
    ids=["a row missing", "a stray row"],
)
def test_a_split_table_row_missing_or_stray_fails(make, repo, capsys, change, said):
    make.truth()
    make.dials()
    make.compare(change=change)
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    assert said in line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")


def test_a_stale_run_could_not_run(make, repo, capsys):
    make.everything()
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "next")
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert "stale" in line_of(out, "COULD NOT RUN: dials of km: ")
    assert "its dials are not known" in line_of(out, "COULD NOT RUN: comparisons of km: ")


def head(repo: Path) -> str:
    done = run_git(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_runs_from_an_earlier_commit_with_only_docs_changed_are_shown_and_say_so(
    make, repo, capsys
):
    _, dials, main, third = make.everything()
    earlier = head(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("later\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    capsys.readouterr()  # what the run recorder printed
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    note = (
        f" (accepted from the earlier commit {earlier[:12]}: every committed path that differs "
        "between it and the checked-out commit is on the keep-list of run_evidence; uncommitted "
        "files were not compared)"
    )
    assert out.splitlines()[0].startswith(
        f"WP10 comparison runs checked at commit {head(repo)[:12]} (a run accepted from an "
        "earlier commit says so on its line), on the committed gate/matrix.json"
    )
    assert line_of(out, "dials: km: ").startswith(f"dials: km: {dials.name}{note}; ")
    assert line_of(out, f"shown: {MAIN} ").startswith(f"shown: {MAIN} {main.name}{note}; read ")
    line = line_of(out, f"shown: {WITH_DRAW} ")
    assert line.startswith(f"shown: {WITH_DRAW} {third.name}{note}; read ")
    assert f"those of {main.name}{note}" in line
    assert out.splitlines()[-1] == "3 of 3 requirements shown, 0 failed, 0 could not run"


@pytest.mark.parametrize(
    ("setting", "options", "start", "said"),
    [
        ("main", {"exit_code": 1}, f"COULD NOT RUN: {MAIN} ", "ended with exit code 1"),
        ("main", {"change": lambda d, r: d.update(model="cox")}, f"COULD NOT RUN: {MAIN} ", "cox"),
        ("held", {}, "COULD NOT RUN: dials of km: ", "is of the setting 'held'"),
    ],
    ids=["exit 1", "another model", "a truth build of the held setting"],
)
def test_runs_from_an_earlier_commit_that_cannot_decide_are_named_with_the_note(
    make, repo, capsys, setting, options, start, said
):
    runs = [make.truth(setting=setting), make.dials(), make.compare(**options)]
    runs.append(make.compare(share=THIRD))
    earlier = head(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("later\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    capsys.readouterr()  # what the run recorder printed
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    note = f" (accepted from the earlier commit {earlier[:12]}: "
    assert said in line_of(out, start)
    # A run's name as a whole (the run without the draw can be named like the start of the name
    # of the run with it when both started in the same second).
    named = {run: re.compile(re.escape(run.name) + r"(?![\w-])") for run in runs}
    naming = [(run, line) for run in runs for line in out.splitlines() if named[run].search(line)]
    assert naming
    assert [line for run, line in naming if f"{run.name}{note}" not in line] == []


def test_a_stale_comparison_could_not_run(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare()
    make.compare(share=THIRD)
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "next")
    make.truth()
    make.dials()
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert line_of(out, "dials: km: ")
    assert "stale" in line_of(out, f"COULD NOT RUN: {MAIN} ")


def _no_truth_named(document, rows):
    document["inputs"].pop("truth_run")


def _truth_named(name):
    def change(document, rows):
        document["inputs"]["truth_run"] = name

    return change


def _table_differs(document, rows):
    document["inputs"]["table_sha256"][f"{UNITS[0]}-days.csv"] = "0" * 64


def _a_table_not_listed(document, rows):
    document["inputs"]["table_sha256"].pop(f"{UNITS[2]}-cars.csv")


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (_no_truth_named, "names no recorded truth build"),
        (_truth_named("20260101T000000Z-wp2-truth"), "does not exist"),
        (_truth_named("20260101T000000Z-wp2-truth-held"), "names no recorded truth build"),
        (_table_differs, f"{UNITS[0]}-days.csv"),
        (_a_table_not_listed, f"lists no {UNITS[2]}-cars.csv"),
    ],
    ids=["none named", "its folder missing", "the held label", "a table differs", "not listed"],
)
def test_a_comparison_without_its_truth_build_could_not_run(make, repo, capsys, change, said):
    make.truth()
    make.dials()
    make.compare(change=change)
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    assert said in line_of(capsys.readouterr().out, f"COULD NOT RUN: {MAIN} ")


def test_a_truth_build_at_another_commit_could_not_run(make, repo, capsys):
    make.truth()
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "next")
    make.dials()
    make.compare()
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert "stale" in line_of(out, "COULD NOT RUN: dials of km: ")


def test_a_truth_build_of_the_held_setting_could_not_run(make, repo, capsys):
    make.truth(setting="held")
    make.dials()
    make.compare()
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    assert "'held'" in line_of(capsys.readouterr().out, "COULD NOT RUN: dials of km: ")


def test_a_real_main_truth_summary_has_no_setting_and_the_item_passes(make, repo, capsys):
    """The truth build writes `setting` only for the held setting (`truth_build.summary_text`), so
    a main summary has no such key and is read as main (`truth_verify.summary_setting`)."""
    truth, *_ = make.everything()
    assert "setting" not in json.loads((truth / truth_build.SUMMARY_NAME).read_text())
    assert run_check(repo) == 0
    assert "3 of 3 requirements shown, 0 failed, 0 could not run" in capsys.readouterr().out


def test_the_real_writer_writes_the_held_setting_and_not_the_main_one():
    written = {unit: {kind: table_hash(unit, kind) for kind in ("cars", "days")} for unit in UNITS}
    main = json.loads(truth_summary_real.summary_text(written))
    held = json.loads(truth_summary_real.summary_text(written, "held"))
    assert "setting" not in main
    assert held["setting"] == "held"
    assert {k: v for k, v in held.items() if k != "setting"} == main


def test_a_truth_build_of_an_unknown_setting_could_not_run(make, repo, capsys):
    truth = make.truth(setting="elsewhere")
    assert json.loads((truth / truth_build.SUMMARY_NAME).read_text())["setting"] == "elsewhere"
    make.dials()
    make.compare()
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    line = line_of(capsys.readouterr().out, "COULD NOT RUN: dials of km: ")
    assert "unknown setting: 'elsewhere'" in line


def test_a_missing_truth_build_folder_could_not_run(make, repo, capsys):
    truth, *_ = make.everything()
    for path in sorted(truth.iterdir()):
        path.unlink()
    truth.rmdir()
    assert run_check(repo) == 2
    assert "does not exist" in line_of(capsys.readouterr().out, "COULD NOT RUN: dials of km: ")


def test_no_comparison_run_could_not_run(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert check.compare_label("km", DIAL) in line_of(out, f"NO RUN: {MAIN} ")
    assert "the run without the draw could not be read" in line_of(
        out, f"COULD NOT RUN: {WITH_DRAW} "
    )


def test_no_dial_choice_run_could_not_run(make, repo, capsys):
    make.truth()
    make.compare()
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert check.dial_label("km") in line_of(out, "NO RUN: dials of km: ")
    assert line_of(out, "COULD NOT RUN: comparisons of km: ")


def test_two_dials_need_two_comparisons(make, repo, capsys):
    make.truth()
    make.dials(workplace=0.03, norway=0.02)
    make.compare(dial=0.03)
    make.compare(dial=0.03, share=THIRD)
    make.compare(dial=0.02)
    make.compare(dial=0.02, share=THIRD)
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    assert f"read for {', '.join(NREL)};" in line_of(out, "shown: km at dial 0.03: ")
    assert f"read for {', '.join(NORWAY)};" in line_of(out, f"shown: {MAIN} ")
    assert out.splitlines()[-1] == "5 of 5 requirements shown, 0 failed, 0 could not run"


def test_two_dials_with_one_comparison_could_not_run(make, repo, capsys):
    make.truth()
    make.dials(workplace=0.03, norway=0.02)
    make.compare(dial=0.02)
    make.compare(dial=0.02, share=THIRD)
    assert run_check(repo) == 2
    assert line_of(capsys.readouterr().out, "NO RUN: km at dial 0.03: ")


@pytest.mark.parametrize(
    ("kind", "change", "said"),
    [
        ("main", lambda d, r: d.update(model="cox"), "compares the model 'cox'"),
        ("main", lambda d, r: d.update(dial=0.03), "at the dial 0.03"),
        ("main", lambda d, r: d.update(no_pause={"share": "1/3", "seed": 1}), "draws"),
        ("third", lambda d, r: d.pop("no_pause"), "draws no car"),
        ("third", lambda d, r: d["no_pause"].update(seed=2), "seed 2"),
    ],
    ids=["another model", "another dial", "a draw", "no draw", "another seed"],
)
def test_a_run_of_another_comparison_could_not_run(make, repo, capsys, kind, change, said):
    make.truth()
    make.dials()
    make.compare(change=change if kind == "main" else None)
    make.compare(share=THIRD, change=change if kind == "third" else None)
    assert run_check(repo) == 2
    start = f"COULD NOT RUN: {MAIN} " if kind == "main" else f"COULD NOT RUN: {WITH_DRAW} "
    assert said in line_of(capsys.readouterr().out, start)


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda d, r: d["run"].update(splits=2), "2 splits"),
        (lambda d, r: d["run"].update(seeds=[2, 4]), "seeds"),
        (lambda d, r: d["matrix"].update(sha256="0" * 64), "gate/matrix.json"),
        (lambda d, r: d["matrix"].pop("sha256"), "gate/matrix.json"),
        (lambda d, r: d["inputs"].update(setting="held"), "setting"),
        (lambda d, r: d["run"].update(tiers=["A"]), "tiers"),
        (lambda d, r: d["run"].update(variants=["forecast"]), "variants"),
        (lambda d, r: d["units"][UNITS[0]].update(status="too few days"), "cannot be split"),
    ],
    ids=[
        "splits",
        "seeds",
        "another matrix",
        "no matrix hash",
        "held setting",
        "a tier",
        "a variant",
        "a unit not split",
    ],
)
def test_a_run_that_cannot_decide_could_not_run(make, repo, capsys, change, said):
    make.truth()
    make.dials()
    make.compare(change=change)
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    assert said in line_of(capsys.readouterr().out, f"COULD NOT RUN: {MAIN} ")


def test_a_run_that_exited_1_could_not_run(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare(exit_code=1)
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    assert "exit code 1" in line_of(capsys.readouterr().out, f"COULD NOT RUN: {MAIN} ")


def test_a_run_of_another_command_could_not_run(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare(module="fleetreserve.other")
    make.compare(share=THIRD)
    assert run_check(repo) == 2
    line = line_of(capsys.readouterr().out, f"COULD NOT RUN: {MAIN} ")
    assert "python -m fleetreserve.compare_run" in line


def _other_paths(document, rows):
    record = document["units"][UNITS[1]]["variants"][VARIANTS[1]]["measured_on"]
    record["paths_sha256"] = "0" * 64


@pytest.mark.parametrize(
    ("arguments", "said"),
    [
        ({"change": _other_paths}, f"{UNITS[1]} {VARIANTS[1]}: audited paths differ"),
        ({"methods": ("re-solve", "llf", "cvar", "edf")}, "methods differ"),
        ({"forecasts": "car,departure\n1,3.0\n"}, "forecast table differs"),
    ],
    ids=["other paths", "other methods", "another forecast table"],
)
def test_a_no_pause_run_unlike_its_twin_fails(make, repo, capsys, arguments, said):
    make.truth()
    make.dials()
    make.compare()
    make.compare(share=THIRD, **arguments)
    assert run_check(repo) == 1
    assert said in line_of(capsys.readouterr().out, f"FAILED: {WITH_DRAW} ")


def test_a_no_pause_run_on_other_truth_tables_fails(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare()
    make.truth(salt="other")  # a second truth build at the same commit, with other tables
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    assert "truth tables differ" in line_of(capsys.readouterr().out, f"FAILED: {WITH_DRAW} ")


@pytest.mark.parametrize(
    "omitted", ["summary", "timing", "table", "splits"], ids=lambda kind: f"no {kind} file"
)
def test_a_missing_output_file_fails(make, repo, capsys, omitted):
    make.truth()
    make.dials()
    make.compare(omit=(compare_run.output_names(0)[omitted],))
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    line = line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")
    assert f"the output file {compare_run.output_names(0)[omitted]} is missing" in line


@pytest.mark.parametrize(
    ("finish", "said"),
    [
        (lambda d: d["split_table"].update(sha256="2" * 64), "the results give compare-splits.csv"),
        (lambda d: d["forecast_table"].update(file="x.csv"), "the results name no"),
        (lambda d: d["split_table"].update(rows=1), "the results say 1"),
        (lambda d: d["run"].update(ceilings=CEILINGS[:1]), "header"),
    ],
    ids=["a split hash", "a forecast name", "a row count", "a header"],
)
def test_output_files_the_results_misname_fail(make, repo, capsys, finish, said):
    make.truth()
    make.dials()
    make.compare(finish=finish)
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    assert said in line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")


def test_an_output_file_changed_after_the_run_could_not_run(make, repo, capsys):
    _, _, main, _ = make.everything()
    (main / compare_run.SUMMARY_NAME).write_text("edited\n")
    assert run_check(repo) == 2
    line = line_of(capsys.readouterr().out, f"COULD NOT RUN: {MAIN} ")
    assert f"{compare_run.SUMMARY_NAME} of {main.name} is not verified" in line


def test_an_output_file_not_in_the_manifest_fails(make, repo, capsys):
    _, _, main, _ = make.everything()
    manifest = json.loads((main / "manifest.json").read_text())
    manifest["outputs"].pop(compare_run.TIMING_NAME)
    (main / "manifest.json").write_text(json.dumps(manifest))
    assert run_check(repo) == 1
    line = line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")
    assert f"{compare_run.TIMING_NAME} is not listed in the manifest" in line


def _variant(document):
    return document["units"][UNITS[2]]["variants"][VARIANTS[1]]


@pytest.mark.parametrize(
    ("change", "said"),
    [
        (lambda d, r: d["units"].pop(UNITS[2]), f"{UNITS[2]} is not in the results"),
        (lambda d, r: d["units"][UNITS[2]]["variants"].pop(VARIANTS[1]), "no results"),
        (lambda d, r: _variant(d)["methods"]["llf"].pop("B"), "llf: no summary for tier B"),
        (lambda d, r: _variant(d)["methods"].pop("cvar"), "cvar: no summary for tier A"),
        (lambda d, r: d["units"][UNITS[2]]["gate_reading"].pop("llf"), "llf: no gate reading"),
        (lambda d, r: _variant(d).pop("measured_on"), "no record measured_on"),
        (lambda d, r: _variant(d)["measured_on"].update(path_rows=7 * DAYS), "not 8 a day"),
        (lambda d, r: _variant(d)["measured_on"].update(days=DAYS - 1), "counting days"),
        (lambda d, r: _variant(d)["measured_on"].pop("boxes_sha256"), "no boxes_sha256"),
    ],
    ids=[
        "a unit",
        "a variant",
        "a tier",
        "a method's summaries",
        "a gate reading",
        "the record",
        "path rows",
        "days",
        "a digest",
    ],
)
def test_a_result_or_record_missing_fails(make, repo, capsys, change, said):
    make.truth()
    make.dials()
    make.compare(change=change)
    make.compare(share=THIRD)
    assert run_check(repo) == 1
    assert said in line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")


def test_a_method_without_an_implementation_is_no_method(make, repo, capsys):
    make.everything()
    methods = (
        *SETUP.methods,
        check.Method("the earlier analysis", None),
        check.Method("made up", "no-such-method"),
    )
    setup = check.Setup(methods, ("km",), SPLITS, ("a sensitivity",))
    assert run_check(repo, setup) == 2
    out = capsys.readouterr().out
    assert line_of(out, "NO METHOD: the earlier analysis: plan section 10 names it; no method ")
    assert "not a method of fleetreserve.compare_methods" in line_of(out, "NO METHOD: no-such-")
    assert line_of(out, "NOT BUILT: a sensitivity")
    assert line_of(out, f"shown: {MAIN} ")
    assert out.splitlines()[-1] == "3 of 6 requirements shown, 0 failed, 3 could not run"


def test_a_method_with_no_registry_name_alone_could_not_run(make, repo, capsys):
    make.everything()
    methods = (*SETUP.methods, check.Method("a made-up method with no code", None))
    setup = check.Setup(methods, ("km",), SPLITS, ())
    assert run_check(repo, setup) == 2
    out = capsys.readouterr().out
    assert line_of(out, "NO METHOD: a made-up method with no code: plan section 10 names it; ")
    assert line_of(out, f"shown: {MAIN} ") and line_of(out, f"shown: {WITH_DRAW} ")
    assert out.splitlines()[-1] == "3 of 4 requirements shown, 0 failed, 1 could not run"


def test_a_requirement_not_built_alone_could_not_run(make, repo, capsys):
    make.everything()
    setup = check.Setup(SETUP.methods, ("km",), SPLITS, ("a made-up requirement, not built",))
    assert run_check(repo, setup) == 2
    out = capsys.readouterr().out
    assert line_of(out, "NOT BUILT: a made-up requirement, not built")
    assert line_of(out, f"shown: {MAIN} ") and line_of(out, f"shown: {WITH_DRAW} ")
    assert out.splitlines()[-1] == "3 of 4 requirements shown, 0 failed, 1 could not run"


def test_a_failure_wins_over_could_not_run(make, repo, capsys):
    make.truth()
    make.dials()
    make.compare(methods=("re-solve", "llf"))
    setup = check.Setup(SETUP.methods, ("km",), SPLITS, ("a sensitivity",))
    assert run_check(repo, setup) == 1
    out = capsys.readouterr().out
    assert line_of(out, f"FAILED: {MAIN} ")
    assert line_of(out, f"NO RUN: {WITH_DRAW} ")


def _unreadable_manifest(folder: Path) -> None:
    (folder / "manifest.json").write_text("[" * 10_000 + "]" * 10_000)


def test_an_unexpected_error_in_one_check_could_not_run(make, repo, capsys):
    _, _, main, _ = make.everything()
    _unreadable_manifest(main)
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert "RecursionError" in line_of(out, f"COULD NOT RUN: {MAIN} ")
    assert line_of(out, "dials: km: ")


def test_an_unexpected_error_outside_the_checks_could_not_run(
    make, repo, capsys, tmp_path, monkeypatch
):
    make.everything()
    (tmp_path / "no-programs").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "no-programs"))  # git cannot be started
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("COULD NOT RUN: ") and "FileNotFoundError" in out


def test_the_newest_run_is_the_one_read(make, repo, capsys):
    make.truth()
    make.dials()
    made = make.compare()
    old = made.parent / f"20000101T000000Z-{check.compare_label('km', DIAL)}"
    made.rename(old)
    new = make.compare(methods=("re-solve", "llf"))
    make.compare(share=THIRD)
    assert old.name < new.name
    assert run_check(repo) == 1
    assert new.name in line_of(capsys.readouterr().out, f"FAILED: {MAIN} ")


# The methods of plan section 10 that the registry implements, in its order (the owner's decision
# of 16 November 2026: the check requires the 11 built methods): the words, the registry name.
REAL_METHODS = [
    ("the calibrated one-way re-solve, the reference", "re-solve"),
    ("diagonal split (includes homothets, Zhao et al. 2017)", "diagonal-split"),
    ("fixed plan without re-solving", "fixed-plan"),
    ("ASA as an hourly tracking MPC, linear-programme form", "asa"),
    ("ASA as an hourly tracking MPC, with the equal-sharing term", "asa-es"),
    ("LLF", "llf"),
    ("EDF", "edf"),
    ("smoothed LLF (Chen et al. 2022)", "smoothed-llf"),
    ("maximum-entropy feedback on forecast departures (Li et al.)", "max-entropy-feedback"),
    ("ALSO-X (Lunde et al. 2025)", "also-x"),
    ("CVaR (Lunde et al. 2025)", "cvar"),
]
ELEVEN = tuple(name for _, name in REAL_METHODS)
MAIN_MODEL = "lgbm_tail"
REAL_MAIN = f"{MAIN_MODEL} at dial 0.02:"
REAL_WITH_DRAW = f"{MAIN_MODEL} at dial 0.02, no-pause 1/3 seed 1:"


def test_the_real_lists_are_those_of_the_plan():
    assert [(m.plan_name, m.registry) for m in check.METHODS] == REAL_METHODS
    named = {m.registry for m in check.METHODS if m.registry is not None}
    assert named == {m.name for m in compare_methods.METHODS}
    assert check.MODELS == ("lgbm_tail",)
    assert not hasattr(check, "HAZARD_MAIN")  # the model is named in the tool, not imported
    assert check.DIAL_SOURCES == {
        "nrel-before-may-2019": "nrel-from-may-2019",
        "nrel-from-may-2019": "nrel-from-may-2019",
        "norway-development-pooled": "norway-development-pooled",
    }
    assert set(check.DIAL_SOURCES) == set(UNITS)
    assert check.NOT_BUILT == ()
    assert (check.SPLITS, check.NO_PAUSE_SHARE, check.NO_PAUSE_SEED) == (200, "1/3", 1)
    assert check.compare_label("lgbm_tail", 0.02) == "wp10-compare-lgbm_tail-eps002"
    assert check.compare_label("x", 0.005, "1/3") == "wp10-compare-x-eps0005-no-pause"


def _real_runs(make, methods=ELEVEN, draw=True) -> tuple[Path, Path | None]:
    """A truth build, the dial-choice run of the main model and its comparison at 200 splits of
    `methods` without the draw and, when `draw`, with it."""
    make.truth()
    make.dials(MAIN_MODEL)
    main = make.compare(MAIN_MODEL, methods=methods, splits=check.SPLITS)
    if not draw:
        return main, None
    return main, make.compare(MAIN_MODEL, share=THIRD, methods=methods, splits=check.SPLITS)


def test_the_real_requirement_passes_on_the_eleven_methods_of_the_main_model(make, repo, capsys):
    main, third = _real_runs(make)
    assert check.main(["--top", str(repo)]) == 0
    out = capsys.readouterr().out
    assert line_of(out, f"dials: {MAIN_MODEL}: ")
    line = line_of(out, f"shown: {REAL_MAIN} ")
    assert f"{main.name}; read for {', '.join(UNITS)}; 11 of 11 methods in its files" in line
    rows = len(UNITS) * len(VARIANTS) * len(TIERS) * len(ELEVEN) * check.SPLITS
    assert f"split table {rows} rows" in line
    line = line_of(out, f"shown: {REAL_WITH_DRAW} ")
    assert third.name in line and f"those of {main.name}" in line
    assert not [text for text in out.splitlines() if text.startswith(("NO METHOD", "NOT BUILT"))]
    assert out.splitlines()[-1] == "3 of 3 requirements shown, 0 failed, 0 could not run"


def test_the_real_requirement_fails_on_runs_lacking_one_of_the_eleven(make, repo, capsys):
    _real_runs(make, methods=tuple(m for m in ELEVEN if m != "asa-es"))
    assert check.main(["--top", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "methods 10 of 11; missing asa-es" in line_of(out, f"FAILED: {REAL_MAIN} ")
    assert "methods 10 of 11; missing asa-es" in line_of(out, f"FAILED: {REAL_WITH_DRAW} ")
    assert out.splitlines()[-1] == "1 of 3 requirements shown, 2 failed, 0 could not run"


def test_the_real_requirement_without_the_no_pause_run_could_not_run(make, repo, capsys):
    _real_runs(make, draw=False)
    assert check.main(["--top", str(repo)]) == 2
    out = capsys.readouterr().out
    assert line_of(out, f"shown: {REAL_MAIN} ")
    label = check.compare_label(MAIN_MODEL, DIAL, check.NO_PAUSE_SHARE)
    assert label in line_of(out, f"NO RUN: {REAL_WITH_DRAW} ")
    assert out.splitlines()[-1] == "2 of 3 requirements shown, 0 failed, 1 could not run"

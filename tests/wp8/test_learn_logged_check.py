"""`tools/learn_logged_check.py`, the WP8 pass-mark item, on made-up recorded learn runs.

Runs are made with the real run recorder in a throwaway git repository inside pytest's temporary
folder, so their manifests are real. The throwaway repository commits the project's
`gate/matrix.json` and a stand-in package `fleetreserve` whose `learn_run` only writes a made-up
`learn.json` and `learn-gate.json` (read from environment variables) and exits with a given
status, so each run's recorded command has the form of a real run: `python -m
fleetreserve.learn_run --model <name> --matrix gate/matrix.json`. Most tests use a made-up list
of two models.

1. Every listed model with a logged run at the checked-out commit: passed (0), one line per model
   naming its run folder and the three values, with how many offers are 0.
2. A model with no run, a run at another commit or on another matrix, a trial or partial run, a
   run that exited 1, a run of another model or command, a run on the held truth setting, a
   changed results file: could not run (2).
3. The concordance index missing, or any of the three values missing or malformed: failed (1);
   a failed model wins over one that could not run.
4. A listed model with no module or with a name that is not a Python name: "no model", could not
   run (2).
5. A committed matrix the parser cannot take, an unexpected error in one model's check (the other
   models still checked, a failed one still winning) or outside them: could not run (2), never
   Python's own exit status 1 for an uncaught error.
6. A unit outside a model's listed units is not required; a listed unit the matrix does not score
   could not run (2).
7. The newest run under a model's label is the one read.
8. The real list holds the forecast models plan section 7 names, with the WP6 gate's model
   `lgbm_tail` after pooled LightGBM, each with a module, and it passes once every one is logged.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from git_run import run_git

from fleetreserve.gate_config import parse
from fleetreserve.run_record import record_run

TOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOP / "tools"))
check = importlib.import_module("learn_logged_check")

MATRIX_BYTES = (TOP / "gate" / "matrix.json").read_bytes()
MATRIX_SHA = hashlib.sha256(MATRIX_BYTES).hexdigest()
MATRIX = parse(json.loads(MATRIX_BYTES))
UNITS = list(MATRIX.gate_units)
KAPPAS = [repr(float(k)) for k in sorted({c.kappa for c in MATRIX.configurations})]
OFFERED = [c.name for c in MATRIX.configurations if c.departure != "oracle"]
RUNGS = ("fixed", "one-way")
NREL = tuple(u for u in UNITS if u.startswith("nrel-"))
OTHER = [u for u in UNITS if u not in NREL]

FAKE_LEARN_RUN = """import os
import sys

out = os.environ["FLEETRESERVE_RUN_DIR"]
for name, key in (("learn.json", "FAKE_LEARN"), ("learn-gate.json", "FAKE_GATE")):
    if key in os.environ:
        with open(os.path.join(out, name), "w") as handle:
            handle.write(os.environ[key])
sys.exit(int(os.environ["FAKE_EXIT"]))
"""
GATE = {"trial": False, "partial": False}
KM = check.Model("Kaplan-Meier", "km")
COX = check.Model("Cox", "cox")
TWO = (KM, COX)


def good(model: str) -> dict:
    """A made-up `learn.json` of `model` with every value of the committed matrix logged."""
    return {
        "model": model,
        "inputs": {"setting": "main"},
        "matrix": {"file": "matrix.json", "sha256": MATRIX_SHA},
        "units": {
            unit: {
                "status": "calibratable",
                "kappas": {
                    kappa: {
                        "pinball_loss": {"departure_hours": {"mean": 0.5 + 0.1 * k}},
                        "concordance_index": {"value": 0.6 + 0.01 * u},
                    }
                    for k, kappa in enumerate(KAPPAS)
                },
                "configurations": {
                    name: {
                        "rungs": {
                            rung: {"beta_hat": 1.0, "test": {"mean_offer_kwh_per_day": 2.0 + u}}
                            for rung in RUNGS
                        }
                    }
                    for name in OFFERED
                },
            }
            for u, unit in enumerate(UNITS)
        },
    }


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
    (root / "fleetreserve" / "learn_run.py").write_text(FAKE_LEARN_RUN)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "first")
    return root


def make_run(
    repo: Path,
    monkeypatch,
    model: str = "km",
    results: dict | None = None,
    *,
    gate: dict | None = GATE,
    exit_code: int = 0,
    command_model: str | None = None,
    module: str = "fleetreserve.learn_run",
) -> Path:
    monkeypatch.setenv("FAKE_LEARN", json.dumps(good(model) if results is None else results))
    if gate is None:
        monkeypatch.delenv("FAKE_GATE", raising=False)
    else:
        monkeypatch.setenv("FAKE_GATE", json.dumps(gate))
    monkeypatch.setenv("FAKE_EXIT", str(exit_code))
    command = [sys.executable, "-m", module, "--model", command_model or model]
    command += ["--matrix", "gate/matrix.json", "--workers", "1"]
    runs = repo / "runs.noindex"
    before = set(runs.glob("*")) if runs.exists() else set()
    status = record_run(check.label(model), command, start_dir=repo)
    assert status == exit_code
    (folder,) = set(runs.glob("*")) - before
    return folder


def edited(model: str, change) -> dict:
    results = good(model)
    change(results)
    return results


def run_check(repo: Path, models=TWO) -> int:
    return check.main(["--top", str(repo)], models=models)


def line_of(out: str, registry: str) -> str:
    (found,) = [line for line in out.splitlines() if f" {registry} (" in line]
    return found


def test_every_listed_model_with_a_logged_run_passes(repo, monkeypatch, capsys):
    km = make_run(repo, monkeypatch, "km")
    cox = make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    scores, offers = len(UNITS) * len(KAPPAS), len(UNITS) * len(OFFERED) * len(RUNGS)
    highest = 0.5 + 0.1 * (len(KAPPAS) - 1)  # `good` logs 0.5 + 0.1 k at the k-th level
    for registry, folder, plan_name in (("km", km, "Kaplan-Meier"), ("cox", cox, "Cox")):
        line = line_of(out, registry)
        assert line.startswith(f"logged: {registry} ({plan_name}): {folder.name}; ")
        assert f"departure pinball loss {scores} of {scores} (0.5 to {highest:g} h)" in line
        assert f"concordance index {scores} of {scores} (0.6 to 0.62)" in line
        assert f"offer {offers} of {offers} (2 to 4 kWh per test day; 0 of {offers} are 0)" in line
    assert out.splitlines()[-1].startswith("2 of 2 models logged")


def test_the_logged_line_counts_the_offers_that_are_0(repo, monkeypatch, capsys):
    def abstain(results):  # an abstaining calibration writes the test-day offer 0.0
        for configuration in results["units"][UNITS[0]]["configurations"].values():
            for rung in configuration["rungs"].values():
                rung["test"]["mean_offer_kwh_per_day"] = 0.0

    make_run(repo, monkeypatch, "km", edited("km", abstain))
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    offers, zeros = len(UNITS) * len(OFFERED) * len(RUNGS), len(OFFERED) * len(RUNGS)
    line = line_of(out, "km")
    assert line.startswith("logged: km (Kaplan-Meier): ")
    counted = f"offer {offers} of {offers}"
    assert f"{counted} (0 to 4 kWh per test day; {zeros} of {offers} are 0)" in line
    assert f"{counted} (2 to 4 kWh per test day; 0 of {offers} are 0)" in line_of(out, "cox")


def test_a_model_without_a_run_could_not_run(repo, monkeypatch, capsys):
    make_run(repo, monkeypatch, "km")
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert line_of(out, "km").startswith("logged: ")
    line = line_of(out, "cox")
    assert line.startswith("NO RUN: cox (Cox): ") and check.label("cox") in line


def test_a_run_at_another_commit_could_not_run(repo, monkeypatch, capsys):
    make_run(repo, monkeypatch, "km")
    make_run(repo, monkeypatch, "cox")
    (repo / "tracked.txt").write_text("two\n")
    git(repo, "commit", "-q", "-am", "next")
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert line_of(out, "km").startswith("COULD NOT RUN: km (Kaplan-Meier): ")
    assert "stale" in line_of(out, "km")


def head(repo: Path) -> str:
    done = run_git(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def test_runs_from_an_earlier_commit_with_only_docs_changed_pass_and_say_so(
    repo, monkeypatch, capsys
):
    km = make_run(repo, monkeypatch, "km")
    cox = make_run(repo, monkeypatch, "cox")
    earlier = head(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("later\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    capsys.readouterr()  # what the run recorder printed
    assert run_check(repo) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == (
        f"WP8 learn runs checked at commit {head(repo)[:12]} (a run accepted from an earlier "
        f"commit says so on its line), on the committed gate/matrix.json ({MATRIX_SHA[:12]})"
    )
    note = (
        f" (accepted from the earlier commit {earlier[:12]}: every committed path that differs "
        "between it and the checked-out commit is on the keep-list of run_evidence; uncommitted "
        "files were not compared); "
    )
    assert line_of(out, "km").startswith(f"logged: km (Kaplan-Meier): {km.name}{note}")
    assert line_of(out, "cox").startswith(f"logged: cox (Cox): {cox.name}{note}")


@pytest.mark.parametrize(
    "options, words",
    [
        ({"results": edited("km", lambda r: r["inputs"].update(setting="held"))}, "'held'"),
        ({"exit_code": 1}, "ended with exit code 1"),
        ({"gate": {"trial": True, "partial": False}}, "trial"),
        ({"gate": None}, "is not verified: the file is missing"),
    ],
    ids=["held setting", "exit 1", "trial", "no gate file"],
)
def test_a_run_from_an_earlier_commit_that_cannot_decide_is_named_with_the_note(
    repo, monkeypatch, capsys, options, words
):
    km = make_run(repo, monkeypatch, "km", **options)
    make_run(repo, monkeypatch, "cox")
    earlier = head(repo)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("later\n")
    git(repo, "add", "docs/notes.md")
    git(repo, "commit", "-q", "-m", "docs")
    capsys.readouterr()  # what the run recorder printed
    assert run_check(repo) == 2
    line = line_of(capsys.readouterr().out, "km")
    assert line.startswith("COULD NOT RUN: km (Kaplan-Meier): ")
    assert f"{km.name} (accepted from the earlier commit {earlier[:12]}: " in line
    assert words in line


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["matrix"].update(sha256=hashlib.sha256(b"a smaller matrix").hexdigest()),
        lambda r: r["matrix"].pop("sha256"),
        lambda r: r.pop("matrix"),
    ],
    ids=["another sha256", "no sha256", "no matrix"],
)
def test_a_run_on_another_matrix_could_not_run(repo, monkeypatch, capsys, change):
    make_run(repo, monkeypatch, "km", edited("km", change))
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    line = line_of(out, "km")
    assert line.startswith("COULD NOT RUN: km ") and "gate/matrix.json" in line
    assert line_of(out, "cox").startswith("logged: ")


def test_no_committed_matrix_could_not_run(repo, monkeypatch, capsys):
    git(repo, "rm", "-q", "gate/matrix.json")
    git(repo, "commit", "-q", "-m", "no matrix")
    make_run(repo, monkeypatch, "km")
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    assert "gate/matrix.json" in capsys.readouterr().out


def _concordance(unit, kappa):
    return lambda r: r["units"][unit]["kappas"][kappa]["concordance_index"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: _concordance(UNITS[0], KAPPAS[1])(r).update(value=None),
        lambda r: _concordance(UNITS[0], KAPPAS[1])(r).pop("value"),
        lambda r: r["units"][UNITS[0]]["kappas"][KAPPAS[1]].pop("concordance_index"),
    ],
    ids=["none", "no value", "no entry"],
)
def test_a_run_missing_the_concordance_index_fails(repo, monkeypatch, capsys, change):
    make_run(repo, monkeypatch, "km", edited("km", change))
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 1
    out = capsys.readouterr().out
    scores = len(UNITS) * len(KAPPAS)
    line = line_of(out, "km")
    assert line.startswith("MISSING: km (Kaplan-Meier): ")
    assert f"concordance index {scores - 1} of {scores}" in line
    assert f"{UNITS[0]} kappa {KAPPAS[1]}" in line
    assert f"departure pinball loss {scores} of {scores}" in line


@pytest.mark.parametrize(
    "gate",
    [
        {"trial": True, "partial": False},
        {"trial": False, "partial": True},
        {"partial": False},
        {"trial": False},
        None,
    ],
    ids=["trial", "partial", "trial not said", "partial not said", "no gate file"],
)
def test_a_trial_or_partial_run_could_not_run(repo, monkeypatch, capsys, gate):
    make_run(repo, monkeypatch, "km", gate=gate)
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    assert line_of(capsys.readouterr().out, "km").startswith("COULD NOT RUN: km ")


def _pinball(r):
    return r["units"][UNITS[1]]["kappas"][KAPPAS[0]]["pinball_loss"]["departure_hours"]


def _offer(r):
    return r["units"][UNITS[2]]["configurations"][OFFERED[0]]["rungs"]["one-way"]["test"]


@pytest.mark.parametrize(
    "change",
    [
        lambda r: _pinball(r).update(mean=None),
        lambda r: _pinball(r).update(mean="0.5"),
        lambda r: _pinball(r).update(mean=True),
        lambda r: _pinball(r).update(mean=-0.1),
        lambda r: _pinball(r).update(mean=float("nan")),
        lambda r: r["units"][UNITS[1]]["kappas"].pop(KAPPAS[0]),
        lambda r: _concordance(UNITS[1], KAPPAS[0])(r).update(value=1.5),
        lambda r: _offer(r).update(mean_offer_kwh_per_day=None),
        lambda r: _offer(r).update(mean_offer_kwh_per_day=-1.0),
        lambda r: r["units"][UNITS[2]]["configurations"][OFFERED[0]]["rungs"].pop("fixed"),
        lambda r: r["units"][UNITS[2]]["configurations"].update(
            {OFFERED[0]: {"rungs": {"reason": "not calibratable: 1 days"}}}
        ),
        lambda r: r["units"][UNITS[2]]["configurations"].pop(OFFERED[0]),
        lambda r: r["units"].pop(UNITS[2]),
        lambda r: r.update(units=[]),
    ],
    ids=[
        "pinball none",
        "pinball text",
        "pinball true",
        "pinball negative",
        "pinball nan",
        "a kappa",
        "concordance above 1",
        "offer none",
        "offer negative",
        "a rung",
        "no rung scored",
        "a configuration",
        "a unit",
        "units not a mapping",
    ],
)
def test_a_missing_or_malformed_value_fails(repo, monkeypatch, capsys, change):
    make_run(repo, monkeypatch, "km", edited("km", change))
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 1
    out = capsys.readouterr().out
    assert line_of(out, "km").startswith("MISSING: km ")
    assert line_of(out, "cox").startswith("logged: ")


def test_a_listed_model_with_no_module_is_no_model(repo, monkeypatch, capsys):
    make_run(repo, monkeypatch, "km")
    make_run(repo, monkeypatch, "no_such_model")
    models = (KM, check.Model("the earlier GBM", None), check.Model("made up", "no_such_model"))
    assert run_check(repo, models) == 2
    out = capsys.readouterr().out
    assert "NO MODEL: the earlier GBM: " in out
    assert "NO MODEL: no_such_model (made up): " in out
    assert "fleetreserve.models.no_such_model" in out


@pytest.mark.parametrize("name", ["km.sub", "no_such.sub", "", "km sub"])
def test_a_malformed_registry_name_is_no_model(repo, monkeypatch, capsys, name):
    make_run(repo, monkeypatch, "km")
    assert run_check(repo, (KM, check.Model("made up", name))) == 2
    out = capsys.readouterr().out
    assert line_of(out, "km").startswith("logged: ")
    (line,) = [line for line in out.splitlines() if "(made up)" in line]
    assert line.startswith(f"NO MODEL: {name} (made up): ") and "not a Python name" in line


def _committed(repo: Path, matrix: bytes) -> None:
    (repo / "gate" / "matrix.json").write_bytes(matrix)
    git(repo, "commit", "-q", "-am", "another matrix")


@pytest.mark.parametrize(
    "matrix",
    [
        json.dumps({**json.loads(MATRIX_BYTES), "rho": 10**400}).encode(),
        b"[" * 10_000 + b"]" * 10_000,
    ],
    ids=["a number too large for a float", "nested too deep"],
)
def test_a_committed_matrix_the_parser_cannot_take_could_not_run(repo, monkeypatch, capsys, matrix):
    _committed(repo, matrix)
    make_run(repo, monkeypatch, "km")
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("COULD NOT RUN: the committed gate/matrix.json cannot ")


def _unreadable_manifest(folder: Path) -> None:
    (folder / "manifest.json").write_text("[" * 10_000 + "]" * 10_000)


@pytest.mark.parametrize("cox_missing", [False, True], ids=["the other logged", "the other failed"])
def test_an_unexpected_error_in_one_models_check_could_not_run(
    repo, monkeypatch, capsys, cox_missing
):
    _unreadable_manifest(make_run(repo, monkeypatch, "km"))
    change = lambda r: _concordance(UNITS[0], KAPPAS[0])(r).update(value=None)  # noqa: E731
    make_run(repo, monkeypatch, "cox", edited("cox", change) if cox_missing else None)
    assert run_check(repo) == (1 if cox_missing else 2)
    out = capsys.readouterr().out
    line = line_of(out, "km")
    assert line.startswith("COULD NOT RUN: km (Kaplan-Meier): ") and "RecursionError" in line
    assert line_of(out, "cox").startswith("MISSING: " if cox_missing else "logged: ")


def test_an_unexpected_error_outside_the_models_could_not_run(repo, monkeypatch, capsys, tmp_path):
    make_run(repo, monkeypatch, "km")
    make_run(repo, monkeypatch, "cox")
    (tmp_path / "no-programs").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "no-programs"))  # git cannot be started
    assert run_check(repo) == 2
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("COULD NOT RUN: ")
    assert "FileNotFoundError" in out


def test_a_failed_model_wins_over_ones_that_could_not_run(repo, monkeypatch, capsys):
    change = lambda r: _concordance(UNITS[0], KAPPAS[0])(r).update(value=None)  # noqa: E731
    make_run(repo, monkeypatch, "km", edited("km", change))
    assert run_check(repo, (KM, COX, check.Model("the earlier GBM", None))) == 1
    out = capsys.readouterr().out
    assert line_of(out, "km").startswith("MISSING: ")
    assert line_of(out, "cox").startswith("NO RUN: ")


def test_a_run_that_exited_1_could_not_run(repo, monkeypatch, capsys):
    make_run(repo, monkeypatch, "km", exit_code=1)
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    assert "exit code 1" in line_of(capsys.readouterr().out, "km")


@pytest.mark.parametrize(
    "other",
    [{"command_model": "cox"}, {"module": "fleetreserve.size_whatif"}],
    ids=["another model", "another command"],
)
def test_a_run_of_another_model_or_command_could_not_run(repo, monkeypatch, capsys, other):
    if "module" in other:
        (repo / "fleetreserve" / "size_whatif.py").write_text(FAKE_LEARN_RUN)
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "another command")
    make_run(repo, monkeypatch, "km", **other)
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    line = line_of(capsys.readouterr().out, "km")
    assert line.startswith("COULD NOT RUN: km ") and "learn_run --model km" in line


@pytest.mark.parametrize(
    "change",
    [lambda r: r["inputs"].update(setting="held"), lambda r: r.pop("inputs")],
    ids=["held", "no setting"],
)
def test_a_run_on_the_held_truth_setting_could_not_run(repo, monkeypatch, capsys, change):
    make_run(repo, monkeypatch, "km", edited("km", change))
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    assert "setting" in line_of(capsys.readouterr().out, "km")


def test_a_changed_results_file_is_not_verified(repo, monkeypatch, capsys):
    folder = make_run(repo, monkeypatch, "km", edited("km", lambda r: r["units"].pop(UNITS[0])))
    (folder / "learn.json").write_text(json.dumps(good("km")))
    make_run(repo, monkeypatch, "cox")
    assert run_check(repo) == 2
    assert "not verified" in line_of(capsys.readouterr().out, "km")


def test_the_newest_run_is_the_one_read(repo, monkeypatch, capsys):
    made = make_run(repo, monkeypatch, "km")
    old = made.parent / f"20000101T000000Z-{check.label('km')}"  # two runs in a second share a name
    made.rename(old)
    change = lambda r: _concordance(UNITS[0], KAPPAS[0])(r).update(value=None)  # noqa: E731
    new = make_run(repo, monkeypatch, "km", edited("km", change))
    make_run(repo, monkeypatch, "cox")
    assert old.name < new.name
    assert run_check(repo) == 1
    assert new.name in line_of(capsys.readouterr().out, "km")


def _no_forecast_on(units):
    def change(results):
        for unit in units:
            for score in results["units"][unit]["kappas"].values():
                score["pinball_loss"]["departure_hours"]["mean"] = None
                score["concordance_index"]["value"] = None

    return change


def test_a_unit_outside_a_models_units_is_not_required(repo, monkeypatch, capsys):
    declared = check.Model("declared departure", "declared", units=NREL)
    make_run(repo, monkeypatch, "declared", edited("declared", _no_forecast_on(OTHER)))
    assert run_check(repo, (declared,)) == 0
    line = line_of(capsys.readouterr().out, "declared")
    scores, offers = len(NREL) * len(KAPPAS), len(NREL) * len(OFFERED) * len(RUNGS)
    assert line.startswith("logged: declared (declared departure): ")
    assert f"departure pinball loss {scores} of {scores}" in line
    assert f"offer {offers} of {offers}" in line
    assert f"units {', '.join(NREL)} only" in line


def test_a_unit_inside_a_models_units_is_required(repo, monkeypatch, capsys):
    declared = check.Model("declared departure", "declared", units=NREL)
    make_run(repo, monkeypatch, "declared", edited("declared", _no_forecast_on(NREL[:1])))
    assert run_check(repo, (declared,)) == 1


def test_a_listed_unit_the_matrix_does_not_score_could_not_run(repo, monkeypatch, capsys):
    make_run(repo, monkeypatch, "km")
    assert run_check(repo, (check.Model("Kaplan-Meier", "km", units=("no-such-unit",)),)) == 2
    assert "no-such-unit" in line_of(capsys.readouterr().out, "km")


# The forecast models plan section 7 names, in its order: name, registry name in
# fleetreserve.models, the units scored (None: every gate unit). The main hazard model's registry
# name is the tool's constant HAZARD_MAIN, changed in one line there and nowhere else. The earlier
# GBM and population models are earlier_gbm and earlier_population (the tool's constants
# EARLIER_GBM and EARLIER_POPULATION). Pooled LightGBM has a second entry, lgbm_tail, the model the
# WP6 gate is read with (plan 13 WP6).
REAL_MODELS = [
    ("hazard model (main)", check.HAZARD_MAIN, None),
    ("hazard model with the plan's section 7 inputs", "hazard_tuned_inputs", None),
    ("hazard model, no set encoder", "hazard_tuned_no_set", None),
    ("hazard model, no site embedding", "hazard_tuned_no_site", None),
    ("hazard model, no day factor", "hazard_tuned_no_day", None),
    ("hazard model, calendar-only inputs", "hazard_tuned_calendar", None),
    ("Kaplan-Meier (site, arrival hour)", "km", None),
    ("Cox", "cox", None),
    ("Weibull AFT", "weibull_aft", None),
    ("pooled LightGBM", "lgbm", None),
    (
        "pooled LightGBM with the lower tail of the user's past departures (the WP6 gate's model)",
        "lgbm_tail",
        None,
    ),
    ("CQR-GBM", "cqr", None),
    ("the earlier GBM model", "earlier_gbm", None),
    ("the earlier population model", "earlier_population", None),
    ("declared departure (ACN, NREL)", "declared", ("nrel-before-may-2019", "nrel-from-may-2019")),
]


def test_the_real_list_is_the_forecast_models_of_plan_section_7():
    assert [(m.plan_name, m.registry, m.units) for m in check.MODELS] == REAL_MODELS
    assert check.HAZARD_MAIN.startswith("hazard")
    for model in check.MODELS:
        assert importlib.util.find_spec(f"fleetreserve.models.{model.registry}") is not None
        if model.units is not None:
            assert set(model.units) <= set(UNITS)


def test_the_real_list_passes_once_every_listed_model_is_logged(repo, monkeypatch, capsys):
    for model in check.MODELS:
        make_run(repo, monkeypatch, model.registry)
    assert check.main(["--top", str(repo)]) == 0
    out = capsys.readouterr().out
    assert sum(line.startswith("logged: ") for line in out.splitlines()) == len(check.MODELS)
    assert "NO MODEL" not in out

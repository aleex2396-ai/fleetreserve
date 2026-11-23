"""`model_pairs`: two sets of recorded `dial_choice` runs compared split by split, on made-up gate
units (the empirical model) with the real writers of `dial_choice` and `kappa_choice`'s own
readings.

The runs: the six runs of the plan's grid (`--kappa`) and a what-if run at 0.6. A run is copied to
a recorded folder with a made-up manifest, as `run_record` writes it; `forge` rewrites a recorded
run (its offers, its fields) and mends what the reader checks (the means, the hash of the results
file named by the per-split file, the manifest), so a test changes one thing.

1. Hand-worked: offers 10, 8, 6, 4, 2, 0 against 9, 8, 7, 4, 1, 0 give the differences 1, 0, -1,
   0, 1, 0, the mean 1/6 and the counts 2 more, 3 equal, 1 less; the interval is the percentile
   bootstrap over those six, from the seed, the same every time, and a constant difference gives
   an interval of no width.
2. The pairing is by seed: a second set whose file lists its splits in another order gives the
   same figures.
3. The pair: given on the command line, or the set's best by `kappa_choice` reading R1 (grid set of
   six runs, what-if set over its levels); a set of the grid without its six cannot give a best.
4. Each refusal: no per-split file, a changed per-split file, a per-split file naming another
   results file or whose offers do not have the results' mean, a set mixing the grid and what-if
   runs or two fits or repeating a level, sets differing in commit (unless readable under
   `run_evidence`), matrix, truth inputs, split seeds or units; a pair with no run at its level.
5. A pair without figures in a unit is "not defined" there, never 0.
6. The what-if label and fit stay on every line that names a what-if set.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from gate_fake import (
    COMMIT,
    GATE_UNITS,
    PLAN_LEVELS,
    gate_cars,
    gate_units,
    matrix_document_six,
    write_build,
)

from fleetreserve import dial_choice, gate_run, kappa_choice, model_pairs, run_evidence, run_verify

PLAINS = tuple(
    f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")
)
DIALS = tuple(
    f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")
)
NAMES = PLAINS + DIALS
SPLITS = 6
CALIBRATABLE = GATE_UNITS[1:]
UNIT = CALIBRATABLE[0]
RESULTS = dial_choice.RESULTS_NAME
WHAT_IF = dial_choice.WHAT_IF_NAMES["results"]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)
    monkeypatch.setattr(run_evidence, "head_commit", lambda top: COMMIT)


def units() -> dict:
    """The made-up gate units of `test_dial_choice_kappa`."""
    found = gate_units()
    found[GATE_UNITS[1]] = gate_cars(60, early_every=3, early_out=12.0)
    found[GATE_UNITS[2]] = gate_cars(30, "B", early_every=3, early_out=12.0) + gate_cars(
        30, "C", early_every=4, early_out=12.0
    )
    return found


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(folder: Path, commit: str = COMMIT) -> None:
    """A made-up manifest in `folder`, as `run_record` writes it."""
    outputs = {p.name: _sha(p) for p in sorted(folder.iterdir()) if p.name != "manifest.json"}
    manifest = {"commit": commit, "command": ["python", "-m", "fleetreserve.dial_choice"]}
    manifest |= {"start": "s", "end": "e", "exit_code": 0, "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest))


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """label -> recorded run folder: the six of the grid ("k0.05" ...) and "w0.6" (what-if)."""
    folder = tmp_path_factory.mktemp("made")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    found = {}
    for label, extra in [(f"k{k:g}", ("--kappa", str(k))) for k in PLAN_LEVELS] + [
        ("w0.6", ("--what-if-kappa", "0.6"))
    ]:
        out = folder / label
        out.mkdir()
        arguments = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
        arguments += ["--truth-run", str(run), "--splits", str(SPLITS), *extra]
        with pytest.MonkeyPatch.context() as monkeypatch:
            _verified(monkeypatch)
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
            assert dial_choice.main(arguments) == 0
        _record(out)
        found[label] = out
    return found


def copy(made, tmp_path, name: str, labels=None) -> dict:
    """Copies of the recorded runs `labels` (default all) under `tmp_path/name`."""
    out = {}
    for label in labels or made:
        out[label] = tmp_path / name / label
        shutil.copytree(made[label], out[label])
    return out


def results_file(folder: Path) -> Path:
    return folder / (WHAT_IF if (folder / WHAT_IF).exists() else RESULTS)


def split_file(folder: Path) -> Path:
    what_if = (folder / WHAT_IF).exists()
    return (
        folder / (dial_choice.WHAT_IF_NAMES if what_if else dial_choice.PLAN_NAMES)["split_offers"]
    )


def forge(folder: Path, results=None, split=None, commit: str = COMMIT) -> None:
    """Rewrites a recorded run (module text): `results` and `split` change the two documents; the
    means of the results are made again from the per-split offers, the per-split file names the
    new results file, and the manifest is made again."""
    path, other = results_file(folder), split_file(folder)
    document, rows = json.loads(path.read_text()), json.loads(other.read_text())
    if results is not None:
        results(document)
    if split is not None:
        split(rows)
    for unit, entry in rows["units"].items():
        if unit not in document["units"]:
            continue
        for name, configuration in entry["configurations"].items():
            for tier, found in configuration.get("tiers", {}).items():
                offers = [r["offer_kwh"] for r in found]
                cell = document["units"][unit]["configurations"][name]["tiers"][tier]
                cell["mean_offer_kwh_per_test_day"] = float(np.mean(offers))
    path.write_text(json.dumps(document, indent=1))
    rows["results"] = {"file": path.name, "sha256": _sha(path)}
    other.write_text(json.dumps(rows, indent=1))
    _record(folder, commit)


def set_offers(folder: Path, unit: str, dial, tier: str, offers: list, order=None) -> None:
    """Replaces the offers of one cell with `offers` (seed 1, 2, ...), listed in `order`."""

    def change(rows: dict) -> None:
        for configuration in rows["units"][unit]["configurations"].values():
            if configuration["driver_risk"] == dial:
                found = [
                    {"seed": s, "offer_kwh": float(o), "scale": 1.0 if o else 0.0}
                    for s, o in enumerate(offers, start=1)
                ]
                configuration["tiers"][tier] = [found[i] for i in (order or range(len(found)))]

    forge(folder, split=change)


def argv(first, second, *extra) -> list[str]:
    return ["--first", *(str(p) for p in first), "--second", *(str(p) for p in second), *extra]


def run(first, second, *extra, tmp_path=None, monkeypatch=None) -> tuple[int, Path]:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return model_pairs.main(argv(first, second, *extra)), out


def read(out: Path) -> dict:
    return json.loads((out / model_pairs.RESULTS_NAME).read_text())


def grid(folders, labels=None) -> list[Path]:
    return [results_file(folders[f"k{k:g}"]) for k in (labels or PLAN_LEVELS)]


# --- 1. hand-worked ------------------------------------------------------------------------------


def test_hand_worked_differences_counts_and_interval(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one", ["k0.3"])
    two = copy(made, tmp_path, "two", ["k0.3"])
    set_offers(one["k0.3"], UNIT, 0.03, "A", [10, 8, 6, 4, 2, 0])
    set_offers(two["k0.3"], UNIT, 0.1, "A", [9, 8, 7, 4, 1, 0])
    code, out = run(
        grid(one, [0.3]),
        grid(two, [0.3]),
        "--first-pair",
        "0.3:0.03",
        "--second-pair",
        "0.3:0.1",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 0
    cell = read(out)["units"][UNIT]["tiers"]["A"]
    assert [r["difference"] for r in cell["per_split"]] == [1, 0, -1, 0, 1, 0]
    assert [r["seed"] for r in cell["per_split"]] == [1, 2, 3, 4, 5, 6]
    assert cell["mean_difference"] == pytest.approx(1 / 6)
    assert (cell["first_offers_more"], cell["equal"], cell["first_offers_less"]) == (2, 3, 1)
    assert cell["mean_offer_first"] == pytest.approx(30 / 6)
    assert cell["mean_offer_second"] == pytest.approx(29 / 6)
    low, high = cell["interval_95"]
    assert low <= cell["mean_difference"] <= high
    generator = np.random.default_rng(1)
    d = np.array([1, 0, -1, 0, 1, 0], dtype=float)
    means = d[generator.integers(0, 6, size=(2000, 6))].mean(axis=1)
    assert [low, high] == [float(q) for q in np.percentile(means, [2.5, 97.5])]
    assert "optimistic for both" in capsys.readouterr().out


def test_a_constant_difference_has_an_interval_of_no_width(made, tmp_path, monkeypatch):
    one = copy(made, tmp_path, "one", ["k0.3"])
    two = copy(made, tmp_path, "two", ["k0.3"])
    set_offers(one["k0.3"], UNIT, 0.03, "A", [5, 6, 7, 8, 9, 10])
    set_offers(two["k0.3"], UNIT, 0.03, "A", [3, 4, 5, 6, 7, 8])
    code, out = run(
        grid(one, [0.3]),
        grid(two, [0.3]),
        "--first-pair",
        "0.3:0.03",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    cell = read(out)["units"][UNIT]["tiers"]["A"]
    assert code == 0 and cell["interval_95"] == [2.0, 2.0] and cell["mean_difference"] == 2.0
    assert (cell["first_offers_more"], cell["equal"], cell["first_offers_less"]) == (6, 0, 0)


def test_the_interval_is_the_same_every_time_and_moves_with_the_seed(made, tmp_path, monkeypatch):
    one = copy(made, tmp_path, "one", ["k0.3"])
    two = copy(made, tmp_path, "two", ["k0.3"])
    set_offers(one["k0.3"], UNIT, 0.03, "A", [10, 8, 6, 4, 2, 0])
    set_offers(two["k0.3"], UNIT, 0.03, "A", [6.9, 9.7, 3.7, 0, 6.2, 0])
    found = []
    for name, extra in (("a", ()), ("b", ()), ("c", ("--bootstrap-seed", "7"))):
        sub = tmp_path / name
        sub.mkdir()
        code, out = run(
            grid(one, [0.3]),
            grid(two, [0.3]),
            "--first-pair",
            "0.3:0.03",
            "--second-pair",
            "0.3:0.03",
            *extra,
            tmp_path=sub,
            monkeypatch=monkeypatch,
        )
        assert code == 0
        found.append((out / model_pairs.RESULTS_NAME).read_text())
    assert found[0] == found[1]
    cells = [json.loads(text)["units"][UNIT]["tiers"]["A"] for text in found]
    assert cells[0]["interval_95"] == cells[1]["interval_95"] != cells[2]["interval_95"]
    assert cells[0]["mean_difference"] == cells[2]["mean_difference"]


# --- 2. paired by seed ---------------------------------------------------------------------------


def test_the_pairing_is_by_seed_not_by_position(made, tmp_path, monkeypatch):
    one = copy(made, tmp_path, "one", ["k0.3"])
    two = copy(made, tmp_path, "two", ["k0.3"])
    set_offers(one["k0.3"], UNIT, 0.03, "A", [10, 8, 6, 4, 2, 0])
    set_offers(two["k0.3"], UNIT, 0.03, "A", [9, 8, 7, 4, 1, 0], order=[5, 3, 1, 0, 2, 4])
    code, out = run(
        grid(one, [0.3]),
        grid(two, [0.3]),
        "--first-pair",
        "0.3:0.03",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    cell = read(out)["units"][UNIT]["tiers"]["A"]
    assert code == 0
    assert [r["difference"] for r in cell["per_split"]] == [1, 0, -1, 0, 1, 0]


# --- 3. the pair ---------------------------------------------------------------------------------


def test_the_best_pair_is_kappa_choices_r1_per_site_type(made, tmp_path, monkeypatch):
    one = copy(made, tmp_path, "one")
    two = copy(made, tmp_path, "two")
    code, out = run(grid(one), grid(two), tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert code == 0
    document = read(out)
    expected = kappa_choice.joint([kappa_choice.read_run(p) for p in grid(one)])["readings"]["R1"]
    assert set(document["units"]) == set(kappa_choice.SITE_TYPE_UNITS.values())
    for site_type, unit in kappa_choice.SITE_TYPE_UNITS.items():
        for record in document["units"][unit]["pairs"]:
            assert record["kappa"] == expected[site_type]["kappa"]
            assert record["dial"] == expected[site_type]["dial"]
            assert record["chosen"] == f"R1 of {site_type}"
        cell = document["units"][unit]["tiers"]["A"]
        assert cell["mean_offer_first"] == pytest.approx(expected[site_type]["mean_offer"])
        assert cell["mean_difference"] == 0
        assert set(document["units"][unit]["tiers"]) == {"A", "B"}


def test_a_grid_set_without_its_six_runs_gives_no_best_pair(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one")
    code, _ = run(
        grid(one, [0.3, 0.4]),
        grid(one),
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and "needs one run at each level" in capsys.readouterr().out


def test_a_what_if_set_gives_its_best_pair_over_its_own_levels(made, tmp_path, monkeypatch):
    one = copy(made, tmp_path, "one")
    code, out = run(
        grid(one), [results_file(one["w0.6"])], tmp_path=tmp_path, monkeypatch=monkeypatch
    )
    assert code == 0
    document = read(out)
    assert document["sets"]["second"]["kind"] == "what-if"
    for unit in kappa_choice.SITE_TYPE_UNITS.values():
        second = document["units"][unit]["pairs"][1]
        assert second["kappa"] == 0.6 and second["chosen"].startswith("R1 of ")
        assert document["units"][unit]["pairs"][0]["kappa"] in PLAN_LEVELS


def test_the_what_if_label_and_fit_stay_on_the_lines_that_name_the_set(
    made, tmp_path, monkeypatch, capsys
):
    one = copy(made, tmp_path, "one")

    def with_fit(document):
        document["what_if"]["fit"] = {"seed": 2, "levels_fitted": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]}

    forge(one["w0.6"], results=with_fit, split=with_fit)
    code, out = run(
        grid(one),
        [results_file(one["w0.6"])],
        "--second-pair",
        "0.6:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    text = capsys.readouterr().out
    assert code == 0
    line = next(x for x in text.splitlines() if x.startswith("Second"))
    assert "what-if" in line and "seed 2" in line and "levels fitted" in line and "0.6" in line
    assert "what-if" in read(out)["sets"]["second"]["description"]
    assert "(the plan's grid" in next(x for x in text.splitlines() if x.startswith("First"))


def test_a_pair_the_set_has_no_run_for_could_not_run(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one", ["k0.3"])
    code, _ = run(
        grid(one, [0.3]),
        grid(one, [0.3]),
        "--first-pair",
        "0.4:0.03",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and "holds no run at kappa 0.4" in capsys.readouterr().out


# --- 4. refusals ---------------------------------------------------------------------------------


def _pair_run(one, two, tmp_path, monkeypatch, labels=("k0.3",), *extra):
    return run(
        [results_file(one[x]) for x in labels],
        [results_file(two[x]) for x in labels],
        "--first-pair",
        "0.3:0.03",
        "--second-pair",
        "0.3:0.03",
        *extra,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )


def test_a_run_without_the_per_split_file_is_refused_saying_so(made, tmp_path, monkeypatch, capsys):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])
    split_file(two["k0.3"]).unlink()
    _record(two["k0.3"])
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 2 and "has to be made again" in capsys.readouterr().out


def test_a_per_split_file_changed_after_the_run_is_refused(made, tmp_path, monkeypatch, capsys):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])
    path = split_file(two["k0.3"])
    path.write_text(path.read_text() + " ")
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 1 and "REFUSED" in capsys.readouterr().out


def test_a_per_split_file_that_names_another_results_file_is_refused(
    made, tmp_path, monkeypatch, capsys
):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])
    path = split_file(two["k0.3"])
    rows = json.loads(path.read_text())
    rows["results"]["sha256"] = "0" * 64
    path.write_text(json.dumps(rows))
    _record(two["k0.3"])
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 2 and "does not name the results file" in capsys.readouterr().out


def test_offers_whose_mean_is_not_the_results_mean_are_refused(made, tmp_path, monkeypatch, capsys):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])
    path = split_file(two["k0.3"])
    rows = json.loads(path.read_text())
    cell = next(
        c for c in rows["units"][UNIT]["configurations"].values() if c["driver_risk"] == 0.03
    )
    cell["tiers"]["A"][0]["offer_kwh"] += 1.0
    rows["results"] = {"file": RESULTS, "sha256": _sha(results_file(two["k0.3"]))}
    path.write_text(json.dumps(rows))
    _record(two["k0.3"])
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 2 and "is not the results' mean offer" in capsys.readouterr().out


def test_a_set_cannot_mix_the_grid_and_what_if_runs(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one")
    code, _ = run(
        [results_file(one["k0.3"]), results_file(one["w0.6"])],
        [results_file(one["k0.3"])],
        "--first-pair",
        "0.3:0.03",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and "never both" in capsys.readouterr().out


def test_a_set_cannot_repeat_a_level(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one")
    shutil.copytree(made["k0.3"], tmp_path / "again")
    code, _ = run(
        [results_file(one["k0.3"]), results_file(tmp_path / "again")],
        [results_file(one["k0.3"])],
        "--first-pair",
        "0.3:0.03",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and "two runs at kappa 0.3" in capsys.readouterr().out


def test_a_set_cannot_hold_what_if_runs_of_two_fits(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one")
    second = tmp_path / "w-other"
    shutil.copytree(made["w0.6"], second)

    def fit(seed):
        def change(document):
            document["what_if"]["fit"] = {"seed": seed, "levels_fitted": [0.05]}

        return change

    forge(one["w0.6"], results=fit(1), split=fit(1))
    forge(second, results=fit(2), split=fit(2))
    both = [results_file(one["w0.6"]), results_file(second)]
    code, _ = run(
        both,
        [results_file(one["k0.3"])],
        "--first-pair",
        "0.6:0.03",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and "one fit only" in capsys.readouterr().out


def test_sets_of_another_commit_are_refused_unless_run_evidence_reads_them(
    made, tmp_path, monkeypatch, capsys
):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])
    forge(two["k0.3"], commit="b" * 40)
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 2 and "not readable under run_evidence" in capsys.readouterr().out
    seen = []

    def readable(run_folder, results_name, head, **options):
        seen.append(run_folder.name)
        return None

    monkeypatch.setattr(run_evidence, "read_evidence", readable)
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 0 and seen


@pytest.mark.parametrize(
    "field,text,mutate",
    [
        ("matrix", "differ in matrix", lambda d: d["matrix"].update(sha256="0" * 64)),
        ("inputs", "differ in inputs", lambda d: d["inputs"].update(changed="yes")),
        ("seeds", "differ in seeds", lambda d: d["run"].update(seeds=[1, 5])),
        ("units", "differ in their units", lambda d: d["units"].pop(CALIBRATABLE[1])),
    ],
)
def test_sets_that_differ_are_refused(made, tmp_path, monkeypatch, capsys, field, text, mutate):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])

    def results(document):
        mutate(document)

    def split(rows):
        if field in ("matrix", "inputs"):
            mutate(rows)
        if field == "seeds":
            mutate(rows)
            for entry in rows["units"].values():
                for c in entry["configurations"].values():
                    for tier in c.get("tiers", {}):
                        c["tiers"][tier] = [r for r in c["tiers"][tier] if r["seed"] <= 5]
        if field == "units":
            rows["units"].pop(CALIBRATABLE[1])

    forge(two["k0.3"], results=results, split=split)
    code, _ = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 2 and text in capsys.readouterr().out


# --- 5. a pair without figures --------------------------------------------------------------------


def test_a_configuration_that_failed_is_not_defined_never_zero(made, tmp_path, monkeypatch):
    one, two = copy(made, tmp_path, "one", ["k0.3"]), copy(made, tmp_path, "two", ["k0.3"])

    def fail(document):
        for entry in document["units"][UNIT]["configurations"].values():
            if entry["driver_risk"] == 0.03:
                for key in [k for k in entry if k != "driver_risk"]:
                    del entry[key]
                entry["failed"] = "made-up failure"

    def fail_split(rows):
        for entry in rows["units"][UNIT]["configurations"].values():
            if entry["driver_risk"] == 0.03:
                for key in [k for k in entry if k != "driver_risk"]:
                    del entry[key]
                entry["failed"] = "made-up failure"

    forge(two["k0.3"], results=fail, split=fail_split)
    code, out = _pair_run(one, two, tmp_path, monkeypatch)
    assert code == 0
    document = read(out)
    assert document["units"][UNIT]["tiers"]["A"]["defined"] is False
    assert "made-up failure" in " ".join(document["units"][UNIT]["tiers"]["A"]["why"])
    other = document["units"][CALIBRATABLE[1]]["tiers"]["A"]
    assert other["defined"] is True
    assert "not defined" in (out / model_pairs.SUMMARY_NAME).read_text()


def test_no_cell_to_compare_could_not_run(made, tmp_path, monkeypatch, capsys):
    one = copy(made, tmp_path, "one", ["k0.3"])
    code, _ = run(
        grid(one, [0.3]),
        grid(one, [0.3]),
        "--first-pair",
        "0.3:0.77",
        "--second-pair",
        "0.3:0.03",
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert code == 2 and "no unit and tier has a pair" in capsys.readouterr().out

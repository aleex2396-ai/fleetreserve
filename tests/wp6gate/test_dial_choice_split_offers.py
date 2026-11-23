"""The per-split file of `dial_choice` (`dial-choice-split-offers.json`, and its what-if name), on
made-up gate units built with `gate_fake` (the empirical model).

1. A run writes the file beside the others, under its plan name or, for a what-if run, its what-if
   name; the manifest of a recorded run lists and hashes it with the other outputs.
2. For every unit, configuration, tier and cell, the mean of the per-split offers is the summary's
   mean offer, the count is the run's splits, the seeds are 1 to N in order, the number of splits
   with a positive scale is the summary's splits that offer, and an offer of 0 goes with scale 0.
3. The file carries the run's identity: label, model, fit units, matrix and hash, truth inputs, the
   run's seeds, kappa and dials, the what-if record of a what-if run, and the SHA-256 of the
   results file beside it.
4. The results, the summary, the forecast table, the timing and the certificate timeouts are what
   they were: the checked-in figures of a run without the new file are not read here; the
   comparison at the earlier commit is in the hand-back. Here: the results text does not hold the
   per-split values.
5. A unit that cannot be split has no configurations; a configuration that failed has no entries.
"""

import hashlib
import json
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

from fleetreserve import dial_choice, gate_run, run_verify

PLAINS = tuple(
    f"main-rounded-hour-k{k:g}-{d}" for k in PLAN_LEVELS for d in ("empirical", "oracle")
)
DIALS = tuple(
    f"main-rounded-hour-k0.3-eps{e:g}-{d}" for e in (0.03, 0.1) for d in ("empirical", "oracle")
)
NAMES = PLAINS + DIALS
SPLITS = 6
CALIBRATABLE = GATE_UNITS[1:]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def units() -> dict:
    """The made-up gate units of `test_dial_choice_kappa`."""
    found = gate_units()
    found[GATE_UNITS[1]] = gate_cars(60, early_every=3, early_out=12.0)
    found[GATE_UNITS[2]] = gate_cars(30, "B", early_every=3, early_out=12.0) + gate_cars(
        30, "C", early_every=4, early_out=12.0
    )
    return found


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    return run, tables, matrix


def _run(build, out: Path, *extra) -> int:
    run, tables, matrix = build
    arguments = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
    arguments += ["--truth-run", str(run), "--splits", str(SPLITS), *extra]
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir(parents=True)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return dial_choice.main(arguments)


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    """label -> run folder: a run without an option, at kappa 0.4, and a what-if run at 0.6."""
    folder = tmp_path_factory.mktemp("runs")
    found = {}
    for label, extra in (
        ("plain", ()),
        ("kappa", ("--kappa", "0.4")),
        ("what-if", ("--what-if-kappa", "0.6")),
    ):
        found[label] = folder / label
        assert _run(build, found[label], *extra) == 0
    return found


def _names(label: str) -> dict:
    return dial_choice.WHAT_IF_NAMES if label == "what-if" else dial_choice.PLAN_NAMES


def _read(runs, label: str) -> tuple[dict, dict]:
    names = _names(label)
    results = json.loads((runs[label] / names["results"]).read_text())
    split = json.loads((runs[label] / names["split_offers"]).read_text())
    return results, split


# --- 1. the file --------------------------------------------------------------------------------


def test_the_file_has_its_plan_name_and_its_what_if_name():
    assert dial_choice.SPLIT_OFFERS_NAME == "dial-choice-split-offers.json"
    assert dial_choice.PLAN_NAMES["split_offers"] == "dial-choice-split-offers.json"
    assert dial_choice.WHAT_IF_NAMES["split_offers"] == "dial-choice-what-if-split-offers.json"


@pytest.mark.parametrize("label", ["plain", "kappa", "what-if"])
def test_a_run_writes_the_file_beside_the_others(runs, label):
    written = {p.name for p in runs[label].iterdir()}
    assert _names(label)["split_offers"] in written
    other = _names("what-if" if label != "what-if" else "plain")["split_offers"]
    assert other not in written


# --- 2. the values ------------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["plain", "kappa", "what-if"])
def test_the_mean_of_the_per_split_offers_is_the_summarys_mean_offer_in_every_cell(runs, label):
    results, split = _read(runs, label)
    cells = 0
    for unit in CALIBRATABLE:
        configurations = results["units"][unit]["configurations"]
        assert set(split["units"][unit]["configurations"]) == set(configurations)
        for name, found in configurations.items():
            entry = split["units"][unit]["configurations"][name]
            assert entry["driver_risk"] == found["driver_risk"]
            for tier, figures in found["tiers"].items():
                rows = entry["tiers"][tier]
                offers = [r["offer_kwh"] for r in rows]
                assert [r["seed"] for r in rows] == list(range(1, SPLITS + 1))
                assert float(np.mean(offers)) == figures["mean_offer_kwh_per_test_day"]
                assert sum(r["scale"] > 0 for r in rows) == figures["splits_that_offer"]
                assert all((r["offer_kwh"] == 0) == (r["scale"] == 0) for r in rows)
                cells += 1
    assert cells == len(CALIBRATABLE) * 2 * len(results["units"][CALIBRATABLE[0]]["configurations"])


def test_the_offers_differ_between_splits_somewhere(runs):
    """A file of equal offers would pass a mean test and say nothing about pairing."""
    _, split = _read(runs, "plain")
    spread = [
        len({r["offer_kwh"] for r in entry["tiers"]["A"]})
        for unit in CALIBRATABLE
        for entry in split["units"][unit]["configurations"].values()
    ]
    assert max(spread) > 1


# --- 3. the identity ----------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["plain", "kappa", "what-if"])
def test_the_file_carries_the_runs_identity(runs, label):
    results, split = _read(runs, label)
    for key in ("label", "model", "fit_units", "matrix", "inputs", "run", "forecast_table"):
        assert split[key] == results[key], key
    assert split["what_if"] == results["what_if"] if label == "what-if" else "what_if" not in split
    names = _names(label)
    assert split["results"] == {
        "file": names["results"],
        "sha256": hashlib.sha256((runs[label] / names["results"]).read_bytes()).hexdigest(),
    }
    assert split["run"]["seeds"] == [1, SPLITS]


def test_a_what_if_file_carries_the_what_if_label(runs):
    _, split = _read(runs, "what-if")
    assert split["label"].startswith("dial choice: what-if")
    assert split["what_if"]["kappa"] == 0.6


# --- 4. nothing else holds the values ------------------------------------------------------------


def test_the_results_hold_no_per_split_values(runs):
    results, _ = _read(runs, "plain")
    assert '"offer_kwh"' not in json.dumps(results) and '"scale"' not in json.dumps(results)


# --- 5. units that cannot be split and configurations that failed --------------------------------


def test_a_unit_that_cannot_be_split_has_no_configurations(runs):
    results, split = _read(runs, "plain")
    small = [u for u in results["units"] if u not in CALIBRATABLE]
    assert small
    for unit in small:
        assert split["units"][unit]["configurations"] == {}
        assert split["units"][unit]["status"] == results["units"][unit]["status"]


def test_a_failed_configuration_has_no_entries(build, tmp_path, monkeypatch):
    real = dial_choice.measure
    calls = {"n": 0}

    def failing(prepared, best_causal, evaluate_batch):
        calls["n"] += 1
        if calls["n"] == 3:  # call 2 is the first configuration's oracle partner
            raise RuntimeError("made-up failure")
        return real(prepared, best_causal, evaluate_batch)

    monkeypatch.setattr(dial_choice, "measure", failing)
    out = tmp_path / "failed"
    assert _run(build, out) == 1
    results = json.loads((out / dial_choice.RESULTS_NAME).read_text())
    split = json.loads((out / dial_choice.SPLIT_OFFERS_NAME).read_text())
    failed = [
        (u, n)
        for u, e in results["units"].items()
        for n, c in e["configurations"].items()
        if "failed" in c
    ]
    assert failed
    for unit, name in failed:
        entry = split["units"][unit]["configurations"][name]
        assert (
            "tiers" not in entry
            and entry["failed"] == results["units"][unit]["configurations"][name]["failed"]
        )

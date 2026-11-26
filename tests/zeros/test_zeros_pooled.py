"""Pooled calibration across units (`pooled_calibration`), on made-up losses and made-up units.
Written, not run. Each test says which edit turns it red.

1. One unit: the pooled scale is the unit's own on-grid scale (`dial_splits.on_grid_scale`), for
   both tiers and for splits that leave out non-offering days. Red if `pooled_scale` reads the
   wrong tier's alphas, skips the `found.offering[days]` filter (the line `used = ...`), or
   evaluates the grid with a bisection (a `tolerance` below 1 in the `crc.calibrate_search` call).
2. Two units of 10 days each: neither calibrates alone in tier A (it needs 19 offering days) and
   the pool of 20 does; the pool's scale is limited by the units' joint losses. Red if
   `pooled_scale` pools only the first unit (the `np.concatenate` of the unit lists), or if the
   scale is taken per unit.
3. `score_pooled`: each unit's test days are scored at the pool's scale, and its own scale is the
   unit's alone. Red if the `replace(..., rule=_rule(...))` line is removed (the unit would be
   scored at its own scale) or if the seeds are not matched.
4. The refusal: every name that is not a development site unit (held-out, training, the pooled
   Norwegian unit) is refused by `check_units`, and `main` ends with exit 2 before any table or
   truth build is read. Red if `check_units` accepts a name outside `UNITS`.
5. The module never names the held-out loader: no import of a module with "held" in its name and
   no such word in its text. Red if an import or a call of the held-out loader is added.
6. The stamp: `finish` marks the document, and a whole run (the empirical model on made-up units
   named like the Norwegian sites) writes the key, the label, the first lines of the summary and
   the label file. Red if `finish` is not called in `main`, or `write_note` in `write_outputs`.
7. A whole run with one unit: its own and its pooled figures are equal, tier by tier. A whole run
   with three units: every figure the command reports is present and the pool pools the three.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from gate_fake import COMMIT, gate_cars, matrix_document, write_build

from fleetreserve import (
    after_sealed_run,
    crc,
    dial_choice,
    dial_splits,
    gate_config,
    gate_run,
    pooled_calibration,
    run_verify,
    sealed_route,
)
from fleetreserve.truth_verify import CouldNotRun

GRID = dial_splits.GRID
LEVELS = len(GRID)
MODULE = Path(pooled_calibration.__file__)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)


def losses(offering, slot=None, driver=None) -> dial_splits.UnitLosses:
    days = len(offering)
    zero = np.zeros((days, LEVELS))
    return dial_splits.UnitLosses(
        value=np.ones(days),
        offering=np.array(offering, dtype=bool),
        slot=zero if slot is None else slot,
        driver=zero.copy() if driver is None else driver,
        ceiling=np.full(days, np.nan),
    )


# --- 1. one unit -------------------------------------------------------------------------------


def _four_failing_days(days: int = 40, failing: int = 4) -> dial_splits.UnitLosses:
    """`days` offering days; `failing` of them lose everything from the scale GRID[5] on."""
    slot = np.zeros((days, LEVELS))
    slot[:failing, 5:] = 1.0
    return losses([True] * days, slot)


@pytest.mark.parametrize("tier", ["A", "B"])
def test_one_unit_pooled_scale_is_the_units_own_scale(tier):
    unit = _four_failing_days()
    calibration = list(range(40))
    found = pooled_calibration.pooled_scale({"u": unit}, {"u": calibration}, tier)
    alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
    own, _ = dial_splits.on_grid_scale(unit.slot, unit.driver, alpha_slot, alpha_driver)
    assert found.scale == own == GRID[4]  # four days over the allowance of 3.1 at GRID[5]
    assert found.offering_calibration_days == 40 and found.by_unit == {"u": 40}


def test_the_pooled_scale_is_the_top_scale_when_the_bound_never_fails():
    unit = _four_failing_days(failing=3)  # 3 <= 0.1 * 41 - 1
    found = pooled_calibration.pooled_scale({"u": unit}, {"u": list(range(40))}, "B")
    assert found.scale == GRID[-1] and found.reason == ""


def test_the_pooled_scale_matches_the_on_grid_rule_on_random_losses():
    rng = np.random.default_rng(7)
    for tier in ("A", "B"):
        alpha_slot, alpha_driver = dial_splits.tier_alphas(tier)
        for trial in range(20):
            days = 30
            slot = (rng.random((days, LEVELS)) < 0.01 * (trial + 1)).astype(float)
            driver = (rng.random((days, LEVELS)) < 0.005 * (trial + 1)).astype(float)
            offering = rng.random(days) < 0.8
            unit = losses(offering, slot, driver)
            calibration = sorted(rng.choice(days, size=24, replace=False))
            used = [d for d in calibration if offering[d]]
            expected, _ = dial_splits.on_grid_scale(
                slot[used], driver[used], alpha_slot, alpha_driver
            )
            found = pooled_calibration.pooled_scale({"u": unit}, {"u": calibration}, tier)
            assert found.scale == expected
            assert found.offering_calibration_days == len(used)


def test_days_that_do_not_offer_are_not_in_the_pool():
    unit = _four_failing_days(days=40)
    hidden = losses([False] * 40, np.ones((40, LEVELS)))
    found = pooled_calibration.pooled_scale(
        {"u": unit, "z": hidden}, {"u": list(range(40)), "z": list(range(40))}, "B"
    )
    assert found.by_unit == {"u": 40, "z": 0} and found.scale == GRID[4]


# --- 2. two units ------------------------------------------------------------------------------


def _pair():
    """Unit a: 10 offering days, no loss. Unit b: 10 offering days, two of them lose everything
    from the scale GRID[4] on (the allowance of 20 days is 1.1 in tier A and tier B alike)."""
    slot = np.zeros((10, LEVELS))
    slot[:2, 4:] = 1.0
    return losses([True] * 10), losses([True] * 10, slot)


def test_two_units_that_cannot_calibrate_alone_calibrate_together():
    a, b = _pair()
    calibration = {"a": list(range(10)), "b": list(range(10))}
    for name, unit in (("a", a), ("b", b)):
        alone = pooled_calibration.pooled_scale({name: unit}, {name: calibration[name]}, "A")
        assert alone.scale == 0.0 and alone.reason == dial_splits.TOO_FEW
    both = pooled_calibration.pooled_scale({"a": a, "b": b}, calibration, "A")
    assert both.offering_calibration_days == 20 and both.by_unit == {"a": 10, "b": 10}
    assert both.scale == GRID[3] and both.reason == ""  # two days over 1.1 at GRID[4]


def test_a_pool_with_no_unit_abstains():
    found = pooled_calibration.pooled_scale({}, {}, "A")
    assert found.scale == 0.0 and found.reason == dial_splits.TOO_FEW


def test_the_pooled_scale_is_a_search_result_of_crc_on_the_pooled_losses():
    a, b = _pair()
    both = pooled_calibration.pooled_scale({"a": a, "b": b}, {"a": range(10), "b": range(10)}, "B")
    slot = np.concatenate([a.slot, b.slot])
    driver = np.concatenate([a.driver, b.driver])
    index = {g: k for k, g in enumerate(GRID)}
    direct = crc.calibrate_search(
        lambda beta: (slot[:, index[beta]], driver[:, index[beta]]),
        20,
        crc.ALPHA_SLOT,
        1.0,
        grid=GRID,
        tolerance=1.0,
    )
    assert both.scale == direct.beta == GRID[3]


# --- 3. scoring --------------------------------------------------------------------------------


def _scored_pair():
    """The pair of `_pair` with four test days each (indices 10 to 13): in b the test days lose
    0.5 at the scale GRID[3]."""
    days = 14
    a_slot, b_slot = np.zeros((days, LEVELS)), np.zeros((days, LEVELS))
    b_slot[:2, 4:] = 1.0
    b_slot[10:, 3] = 0.5
    units = {"a": losses([True] * days, a_slot), "b": losses([True] * days, b_slot)}
    splits = {n: [(1, list(range(10)), [10, 11, 12, 13])] for n in units}
    return units, splits


def test_each_units_test_days_are_scored_at_the_pooled_scale_and_beside_its_own():
    units, splits = _scored_pair()
    pools, own, pooled = pooled_calibration.score_pooled(units, splits, "A")
    assert [p.scale for p in pools] == [GRID[3]]
    assert own["a"][0].scale == 0.0 and own["b"][0].scale == 0.0  # 10 days: too few alone
    assert pooled["a"][0].scale == GRID[3] and pooled["b"][0].scale == GRID[3]
    assert pooled["a"][0].offer_kwh == pytest.approx(GRID[3])  # the scale times a value of 1
    assert pooled["a"][0].slot_loss == 0.0 and pooled["b"][0].slot_loss == 0.5
    assert own["a"][0].offer_kwh == 0.0 and own["b"][0].slot_loss is None
    summary = pooled_calibration.pool_summary(pools, pooled, splits, "A")
    assert summary["splits_with_scale_above_zero"] == 1 and summary["scales"] == {"0.3": 1}
    assert summary["mean_test_slot_loss"] == pytest.approx(0.25)  # 4 test days at 0 and 4 at 0.5
    assert summary["slot_loss_over_limit"]["splits"] == 1


def test_splits_of_different_seeds_are_refused():
    units, splits = _scored_pair()
    splits["b"] = [(2, *splits["b"][0][1:])]
    with pytest.raises(ValueError, match="same seeds"):
        pooled_calibration.score_pooled(units, splits, "A")


# --- 4. the refusal ----------------------------------------------------------------------------

REFUSED = (
    *sealed_route.HELD_OUT_UNITS,
    "OSL_T",
    "TRO_R",
    "caltech",
    "OSL_S",
    "norway-development-pooled",
    "no-such-unit",
)


@pytest.mark.parametrize("name", REFUSED)
def test_a_unit_that_is_not_a_development_site_unit_is_refused(name):
    with pytest.raises(CouldNotRun, match="not a development unit"):
        pooled_calibration.check_units([name])
    with pytest.raises(CouldNotRun, match=name):
        pooled_calibration.check_units(["OSL_1", name])


def test_the_development_units_are_accepted_once_each():
    assert pooled_calibration.check_units(pooled_calibration.DEFAULT_UNITS) == (
        "OSL_1",
        "BER",
        "KRO",
        "OSL_2",
        "BAR_2",
    )
    assert pooled_calibration.check_units(["nrel-from-may-2019"]) == ("nrel-from-may-2019",)
    with pytest.raises(CouldNotRun, match="more than once"):
        pooled_calibration.check_units(["BER", "BER"])
    with pytest.raises(CouldNotRun, match="no unit"):
        pooled_calibration.check_units([])


def test_main_refuses_a_held_out_unit_before_any_file_is_read(monkeypatch, capsys, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("a file or a truth build was read")

    monkeypatch.setattr(gate_run, "check_inputs", forbidden)
    monkeypatch.setattr(gate_run, "check_files", forbidden)
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(tmp_path))
    code = pooled_calibration.main(["--model", "empirical", "--units", "OSL_1", "OSL_T"])
    assert code == 2 and capsys.readouterr().out.startswith("COULD NOT RUN")
    assert list(tmp_path.iterdir()) == []


# --- 5. no held-out loader ---------------------------------------------------------------------


def test_the_module_never_names_the_held_out_loader():
    text = MODULE.read_text()
    assert "heldout" not in text.lower().replace("held-out", "")
    modules = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            modules += [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    assert modules and not [m for m in modules if "held" in m.lower() or "sealed_route" in m]


# --- 6. and 7. the stamp, and whole runs -------------------------------------------------------


def test_finish_marks_the_document_and_always():
    document = pooled_calibration.finish({"label": "pooled calibration: x"})
    assert document[after_sealed_run.KEY] == after_sealed_run.record()
    assert document["label"].startswith(after_sealed_run.WHAT)


NAMES = (
    "main-rounded-hour-k0.3-empirical",
    "main-rounded-hour-k0.3-oracle",
    "main-rounded-hour-k0.3-eps0.02-empirical",
    "main-rounded-hour-k0.3-eps0.02-oracle",
)


def _site(site: str, early_every: int):
    return gate_cars(60, site, early_every=early_every, early_out=12.0)


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(
        folder, {"OSL_1": _site("A", 3), "BER": _site("B", 4), "KRO": _site("C", 5)}
    )
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document(NAMES)))
    return run, tables, matrix


def _argv(build, *extra) -> list[str]:
    run, tables, matrix = build
    return [
        "--model",
        "empirical",
        "--matrix",
        str(matrix),
        "--tables",
        str(tables),
        "--truth-run",
        str(run),
        "--kappa",
        "0.3",
        "--splits",
        "4",
        *extra,
    ]


def _run(build, out, monkeypatch, *extra) -> int:
    out.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
    return pooled_calibration.main(_argv(build, *extra))


def test_a_whole_run_pools_the_units_and_stamps_every_output(build, tmp_path, monkeypatch, capsys):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", "OSL_1", "BER", "KRO")
    text = capsys.readouterr().out
    assert code == 0, text
    folder = tmp_path / "out"
    names = {p.name for p in folder.iterdir()}
    assert {
        pooled_calibration.RESULTS_NAME,
        pooled_calibration.SUMMARY_NAME,
        pooled_calibration.TABLE_NAME,
        pooled_calibration.TIMING_NAME,
        after_sealed_run.FILE_NAME,
    } == names
    document = json.loads((folder / pooled_calibration.RESULTS_NAME).read_text())
    assert document[after_sealed_run.KEY] == after_sealed_run.record()
    assert document["label"].startswith(after_sealed_run.WHAT)
    lines = text.splitlines()
    assert lines[:2] == [after_sealed_run.WHAT, after_sealed_run.SENTENCE]
    assert (folder / pooled_calibration.SUMMARY_NAME).read_text() == text
    assert (folder / after_sealed_run.FILE_NAME).read_text().splitlines() == lines[:2]
    assert document["run"]["units_pooled"] == ["BER", "KRO", "OSL_1"]
    assert document["run"]["kappa"] == 0.3 and document["run"]["dial"] == 0.02
    for tier in ("A", "B"):
        assert document["pool"][tier]["splits"] == 4
        for unit in ("OSL_1", "BER", "KRO"):
            found = document["units"][unit]["tiers"][tier]
            assert set(found) == {"own_scale", "pooled_scale"}
            assert found["pooled_scale"]["splits"] == 4


def test_a_whole_run_of_one_unit_gives_its_own_figures_as_the_pooled_ones(
    build, tmp_path, monkeypatch, capsys
):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", "BER")
    assert code == 0, capsys.readouterr().out
    document = json.loads((tmp_path / "out" / pooled_calibration.RESULTS_NAME).read_text())
    for tier in ("A", "B"):
        found = document["units"]["BER"]["tiers"][tier]
        assert found["own_scale"] == found["pooled_scale"]


def test_a_whole_run_refuses_a_unit_that_is_not_a_development_unit(
    build, tmp_path, monkeypatch, capsys
):
    code = _run(build, tmp_path / "out", monkeypatch, "--units", "BER", "OSL_T")
    assert code == 2 and "OSL_T" in capsys.readouterr().out
    assert list((tmp_path / "out").iterdir()) == []


def test_the_dial_of_the_run_is_found_in_the_family_at_the_main_level():
    matrix = gate_config.load(Path(pooled_calibration.MATRIX))
    family = dial_choice.find_family(matrix)
    _, (_, dials) = dial_choice.family_at(matrix, family, pooled_calibration.KAPPA)
    found = [d for d in dials if d.driver_risk == pooled_calibration.DIAL]
    assert len(found) == 1 and found[0].kappa == 0.5 and "-k0.5-" in found[0].name

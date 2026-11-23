"""The re-solve sizing what-if (`size_resolve_whatif`) scoring development units with finished
sizings (`--development-units`), on made-up units (`gate_fake.gate_cars`: 36 days of three cars
each, an early leaver on one day in 10 to 17 on every other unit) and a made-up verified truth
build: three training units labelled (caltech, jpl, OSL_S) and one development unit (OSL_1), 3
splits, 2 control draws, one worker; once with OSL_1 and once without.

1. `--development-units` takes development units only (`truth_build.DEVELOPMENT_UNITS`), each
   named once: a training unit, a unit that is neither, or a unit named twice is refused before
   any table is opened (exit 2, "COULD NOT RUN", no file written).
2. With `--development-units OSL_1` the run ends with exit 0: OSL_1 has its entry beside the
   training units, with every sizing in both tiers and, for the net, conformal training and the
   multiplier, its shuffled control over the run's 2 draws with the gains by draw; it is in no
   pool; the run lists it under `development_units`; the label and the definitions say that
   development units are scored with sizings fitted on the training units only. In the summary's
   part for the development units, each control line is followed by the sizing's gain over each
   draw, with the smallest, the largest and the draws at or above the sizing; the summary of the
   run without development units has no such line.
3. The training units' figures are those of the run without OSL_1: equal entries, pools, fit
   records, control counts and run fields in the results, the same summary lines (the label line
   apart) before the development units' part, the same day rows, OSL_1's rows after them.
4. The finished sizings: OSL_1's booster and constant predictions, its net predictions and its
   conformal training's outputs are those of one fit on every labelled training day (the rows of
   the run without OSL_1) read at OSL_1's labelled days, the cars those of the labelled units;
   the fits' records are under `development_fits`, each network of the net fitted and validated
   on the training units' labelled days, as many as they have.
5. The default number of control draws is 19.
"""

import json

import numpy as np
import pandas as pd
import pytest
import size_net_fake
from gate_fake import COMMIT, gate_cars, write_build
from test_size_whatif import document

from fleetreserve import (
    gate_run,
    run_verify,
    size_ct,
    size_gbm,
    size_net,
    size_net_inputs,
    size_resolve_summary,
    size_resolve_whatif,
    size_scale,
    size_whatif,
)
from fleetreserve.days_from_truth import load_unit_days
from fleetreserve.gate_config import load
from fleetreserve.learn_eval import pinball_loss
from fleetreserve.truth_build import DEVELOPMENT_UNITS, TRAIN_UNITS

SPLITS = 3
DAYS = 36
DRAWS = 2
UNITS = ("caltech", "jpl", "OSL_S")
DEVELOPMENT = "OSL_1"
LEVELS = [float(level) for level in size_gbm.LEVELS]


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        unit: gate_cars(DAYS, unit, early_every=0 if j % 2 == 0 else 9 + j)
        for j, unit in enumerate((*TRAIN_UNITS, DEVELOPMENT))
    }
    run, tables = write_build(folder, units)
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document()))
    return {"run": run, "tables": tables, "matrix": matrix}


def arguments(build, *extra):
    return [
        "--model",
        "lgbm_tail_rotation",
        "--dial",
        "0.03",
        "--matrix",
        str(build["matrix"]),
        "--tables",
        str(build["tables"]),
        "--truth-run",
        str(build["run"]),
        "--splits",
        str(SPLITS),
        "--units",
        *UNITS,
        "--control-draws",
        str(DRAWS),
        *extra,
    ]


@pytest.fixture(scope="module")
def runs(build, tmp_path_factory):
    out = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        patch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        patch.setattr(size_net_inputs, "load_context", size_net_fake.context)
        for kind, extra in (
            ("training", ()),
            ("development", ("--development-units", DEVELOPMENT)),
        ):
            folder = tmp_path_factory.mktemp(kind)
            patch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
            code = size_resolve_whatif.main(arguments(build, *extra))
            out[kind] = {
                "code": code,
                "results": json.loads((folder / size_resolve_whatif.RESULTS_NAME).read_text()),
                "summary": (folder / size_resolve_whatif.SUMMARY_NAME).read_text().splitlines(),
                "days": pd.read_csv(folder / size_resolve_whatif.DAYS_NAME, dtype={"date": str}),
                "day_text": (folder / size_resolve_whatif.DAYS_NAME).read_text().splitlines(),
            }
    return out


# --- 1. ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("names", "says"),
    (
        (("jpl",), "jpl: not a development unit"),
        (("OSL_T",), "OSL_T: not a development unit"),
        ((DEVELOPMENT, "BER", DEVELOPMENT), "names OSL_1 twice"),
    ),
)
def test_only_development_units_named_once_are_taken(
    build, tmp_path, monkeypatch, capsys, names, says
):
    folder = tmp_path / "out"
    folder.mkdir()
    monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
    argv = arguments(build, "--development-units", *names)
    argv[argv.index("--tables") + 1] = str(tmp_path / "no-such-folder")
    code = size_resolve_whatif.main(argv)
    text = capsys.readouterr().out
    assert code == 2 and text.startswith("COULD NOT RUN: --development-units") and says in text
    assert list(folder.iterdir()) == []
    assert "OSL_T" not in DEVELOPMENT_UNITS and "BER" in DEVELOPMENT_UNITS


# --- 2. ----------------------------------------------------------------------------------------


def test_the_development_unit_is_scored_beside_the_training_units(runs):
    found = runs["development"]
    assert found["code"] == 0
    results = found["results"]
    assert results["failed_units"] == [] and results["failed_controls"] == []
    assert set(results["units"]) == {*UNITS, DEVELOPMENT}
    assert results["run"]["units"] == list(UNITS)
    assert results["run"][size_resolve_whatif.DEVELOPMENT] == [DEVELOPMENT]
    entry = results["units"][DEVELOPMENT]
    assert entry["status"] == "calibratable" and entry["evaluations"]["day_scale_pairs"] > 0
    for tier in size_scale.TIERS:
        figures = entry["tiers"][tier]
        assert set(figures) == set(results["units"]["jpl"]["tiers"][tier])
        assert figures["constant"]["splits"] == SPLITS
        for name in ("ct-scale", "ct-abstain", "net-0.1-scale", "day-type-abstain"):
            control = figures[name]["difference_from_shuffled"]
            assert control["draws"] == DRAWS and len(control["by_draw"]["gain_kwh"]) == DRAWS
    assert all(DEVELOPMENT not in p["units"] for p in results["pooled_by_site_type"].values())
    assert results["label"] == size_resolve_summary.DEVELOPMENT_LABEL
    assert results["definitions"]["development_units"] == size_resolve_summary.DEVELOPMENT_TEXT
    summary = found["summary"]
    assert summary[0] == size_resolve_summary.DEVELOPMENT_LABEL
    head = f"development units: {size_resolve_summary.DEVELOPMENT_TEXT}"
    assert summary.count(head) == 1
    site_type = entry["site_type"]
    assert any(line.startswith(f"{DEVELOPMENT} ({site_type}, calibratable): ") for line in summary)
    assert summary.index(head) < [k for k, s in enumerate(summary) if s.startswith(DEVELOPMENT)][0]
    assert runs["training"]["results"]["label"] == size_resolve_summary.WHAT


def test_the_summary_gives_each_draws_gain_for_the_development_unit(runs):
    summary = runs["development"]["summary"]
    entry = runs["development"]["results"]["units"][DEVELOPMENT]
    part = summary[summary.index(f"development units: {size_resolve_summary.DEVELOPMENT_TEXT}") :]
    expected = {}
    for tier in size_scale.TIERS:
        for sizing, figures in entry["tiers"][tier].items():
            if "difference_from_shuffled" in figures:
                found = figures["difference_from_shuffled"]["by_draw"]
                expected[(tier, sizing)] = (
                    "        gains by draw, kWh (the sizing's offer minus each draw's): "
                    + ", ".join(f"{gain:.3f}" for gain in found["gain_kwh"])
                    + f"; smallest {found['smallest_gain_kwh']:.3f}, largest "
                    f"{found['largest_gain_kwh']:.3f}; draws at or above the sizing "
                    f"{found['draws_at_or_above']}"
                )
    # per tier: net-<level>-<arm> for 5 levels and 2 arms, ct-<arm> and day-type-<arm>
    assert len(expected) == 2 * (5 * 2 + 2 + 2)
    shown, tier = {}, None
    control = f"      against its shuffled control {size_resolve_summary.CONTROL}-"
    for k, line in enumerate(part):
        if line.startswith("  tier "):
            tier = line.removeprefix("  tier ").removesuffix(":")
        if line.lstrip().startswith("gains by draw"):
            assert part[k - 1].startswith(control)
            sizing = part[k - 1].removeprefix(control).split(",")[0]
            assert (tier, sizing) not in shown
            shown[(tier, sizing)] = line
    assert shown == expected
    assert not any("gains by draw" in line for line in runs["training"]["summary"])


# --- 3. ----------------------------------------------------------------------------------------


def test_the_training_units_figures_are_those_without_the_development_unit(runs):
    alone, beside = runs["training"]["results"], runs["development"]["results"]
    assert runs["training"]["code"] == 0
    for unit in UNITS:
        assert beside["units"][unit] == alone["units"][unit]
        assert beside["control_evaluations"][unit] == alone["control_evaluations"][unit]
    for key in ("pooled_by_site_type", "net_fits", "ct_fits", "net_inputs", "matrix"):
        assert beside[key] == alone[key]
    run = dict(beside["run"])
    assert run.pop(size_resolve_whatif.DEVELOPMENT) == [DEVELOPMENT]
    assert run == alone["run"]
    assert {k: v for k, v in beside["definitions"].items() if k != "development_units"} == (
        alone["definitions"]
    )
    tables = beside["inputs"]["table_sha256"]
    assert {k: v for k, v in tables.items() if not k.startswith(DEVELOPMENT)} == (
        alone["inputs"]["table_sha256"]
    )
    assert sorted(k for k in tables if k.startswith(DEVELOPMENT)) == [
        f"{DEVELOPMENT}-cars.csv",
        f"{DEVELOPMENT}-days.csv",
    ]
    lines = runs["training"]["summary"]
    assert runs["development"]["summary"][1 : len(lines)] == lines[1:]
    assert not any(line.startswith(DEVELOPMENT) for line in lines)
    text = runs["development"]["day_text"]
    assert text[: len(runs["training"]["day_text"])] == runs["training"]["day_text"]
    rest = runs["development"]["days"].iloc[len(runs["training"]["days"]) :]
    assert len(rest) > 0 and set(rest["unit"]) == {DEVELOPMENT}


# --- 4. ----------------------------------------------------------------------------------------


def labelled_cars(build, units) -> pd.DataFrame:
    """The cars plugged in at the decision time of `units` (`size_net_inputs`)."""
    matrix = load(build["matrix"])
    sources = size_net_fake.context().sources
    frames = []
    for unit in units:
        days = load_unit_days(
            build["tables"],
            unit,
            rho=matrix.rho,
            no_fit=matrix.no_fit,
            small_fleet=matrix.small_fleet,
        )
        frames.append(size_net_inputs.car_frame(days.cars, rho=matrix.rho, sources=sources))
    return pd.concat(frames, ignore_index=True)


def test_the_development_sizes_come_from_one_fit_on_every_training_day(runs, build):
    train = runs["training"]["days"]
    days = runs["development"]["days"]
    mine = days[days["unit"] == DEVELOPMENT].reset_index(drop=True)
    assert set(train["unit"]) == set(UNITS) and len(mine) > 0
    context = list(size_whatif.CONTEXT)
    x, y = train[context].astype(float).to_numpy(), train[size_gbm.LABEL].to_numpy(float)
    x_new = mine[context].astype(float).to_numpy()
    results = runs["development"]["results"]
    pinball = results["units"][DEVELOPMENT]["pinball"]
    train_cars, my_cars = labelled_cars(build, UNITS), labelled_cars(build, (DEVELOPMENT,))
    rows = ["unit", "site", "date", size_gbm.LABEL]
    fits = results["development_fits"]
    assert sorted(fits) == ["ct", "net"]
    for level in LEVELS:
        expected = size_gbm.booster_predictions(x, y, x_new, context, level)
        np.testing.assert_array_equal(mine[f"gbm_q{level:g}"].to_numpy(float), expected)
        net = size_net.fit_days(train[rows], train_cars, (level,))
        found = size_net.predict_days(net, mine[rows], my_cars)[:, 0]
        np.testing.assert_allclose(mine[f"net_q{level:g}"].to_numpy(float), found, rtol=1e-15)
        assert fits["net"][f"{level:g}"] == net.record
        labels = mine[size_gbm.LABEL].to_numpy(float)
        constant = np.full(len(mine), size_gbm.constant(y, level))
        loss = pinball_loss(labels, constant, level)
        assert pinball[f"{level:g}"]["constant"] == pytest.approx(loss, rel=1e-12)
    ct_rows = [*rows, size_ct.VALUE]
    start = size_ct.fit_start(train[ct_rows], train_cars)
    for arm in size_scale.ARMS:
        fitted = size_ct.fit_days(train[ct_rows], train_cars, arm, None, start)
        expected = size_ct.predict_days(fitted, mine[ct_rows], my_cars)
        np.testing.assert_array_equal(mine[f"ct_{arm}"].to_numpy(float), expected)
        assert fits["ct"][arm] == fitted.record
    for record in fits["net"].values():
        assert record["training_days"] + record["validation_days"] == len(train)
    line = size_resolve_summary.development_fit_line(fits)
    assert line in runs["development"]["summary"]


# --- 5. ----------------------------------------------------------------------------------------


def test_the_default_number_of_control_draws_is_19():
    assert size_resolve_whatif.CONTROL_DRAWS == 19

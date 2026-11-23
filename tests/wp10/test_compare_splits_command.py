"""The per-split rows and the paired differences of the comparison command (`compare_run`), on a
made-up gate unit with two service hours, where the hindsight ceiling and the best-causal ceiling
differ (`compare_fake.two_hour_cars`).

Units: the first and the third gate unit hold 2 days each and cannot be split; the second holds 28
days, 20 of which calibrate in each split. One run, at the dial 0.03, of every method of the
registry but `asa` and `asa-es`, with 6 splits and one worker. Everything recomputed below is
recomputed by this file's own arithmetic, not by `dial_splits` or `compare_paired`.

1. `compare-splits.csv`: its SHA-256 and number of rows are those under `split_table`; it holds
   one row per unit that can be split, variant, tier, method and split, in that order.
2. Every summary field of `compare.json` (per unit, variant, method and tier) and every figure of
   each method's gate reading is recomputed from the file.
3. The paired differences of `compare.json` are recomputed from the file.
4. The ceilings are recomputed here (`gate_day.day_ceilings` of the unit's days, the splits of
   `dial_choice.split_indices`): the two ceilings differ on every day; the file's ceiling columns
   are their means over each split's test days; the file's share of the best-causal ceiling, and
   so the results' "of ceiling" figure, is the offer over the best-causal ceiling and not over the
   hindsight ceiling.
5. The summary's "of ceiling" and "driver loss" cells of the tier tables, and the shares of each
   ceiling, the oracle offer and the oracle-to-offer ratio of the gate-reading table, equal the
   numbers recomputed here (from the file's offers and losses and the ceilings of item 4).
6. The summary is the summary of the results without the paired differences, followed by the
   paired lines; each row of the paired tables shows the figures of the results.
"""

import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import pytest
from compare_fake import NAMES, TWO_HOURS, two_hour_cars
from gate_fake import COMMIT, GATE_UNITS, gate_cars, matrix_document, write_build

from fleetreserve import (
    compare_methods,
    compare_run,
    compare_summary,
    dial_choice,
    gate_config,
    gate_day,
    gate_run,
    run_verify,
)
from fleetreserve.days_from_truth import load_unit_days

SPLITS = 6
METHODS = [m.name for m in compare_methods.METHODS if m.name not in ("asa", "asa-es")]
UNIT = GATE_UNITS[1]
ALPHAS = {"A": (0.10, 0.05), "B": (0.10, 1.0)}
TOLERANCE = 1e-9  # crc.TOLERANCE, the margin of "over the limit"
CLOSE = 1e-12


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    units = {
        GATE_UNITS[0]: gate_cars(2),
        UNIT: two_hour_cars(28),
        GATE_UNITS[2]: gate_cars(2, "B"),
    }
    run, tables = write_build(folder, units, TWO_HOURS)
    document = matrix_document(NAMES)
    document["n_calibration"] = 20
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(document))
    return run, tables, matrix


@pytest.fixture(scope="module")
def out(build, tmp_path_factory):
    run, tables, matrix = build
    folder = tmp_path_factory.mktemp("run")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(folder))
        arguments = ["--model", "empirical", "--dial", "0.03", "--matrix", str(matrix)]
        arguments += ["--tables", str(tables), "--truth-run", str(run), "--splits", str(SPLITS)]
        assert compare_run.main([*arguments, "--methods", *METHODS]) == 0
    return folder


@pytest.fixture(scope="module")
def document(out):
    return json.loads((out / compare_run.RESULTS_NAME).read_text())


def _number(text: str):
    return None if text == "" else float(text)


@pytest.fixture(scope="module")
def rows(out):
    text = (out / compare_run.SPLITS_NAME).read_text()
    found = []
    for row in csv.DictReader(io.StringIO(text)):
        parsed = dict(row)
        for key, value in row.items():
            if key not in ("unit", "variant", "tier", "method", "scale_zero_reason"):
                parsed[key] = _number(value)
        found.append(parsed)
    return found


def _select(rows, unit, variant, tier, method):
    chosen = [
        r
        for r in rows
        if (r["unit"], r["variant"], r["tier"], r["method"])
        == (
            unit,
            variant,
            tier,
            method,
        )
    ]
    return sorted(chosen, key=lambda r: r["seed"])


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def _quantile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def _close(found, expected, where=""):
    if isinstance(expected, dict):
        assert set(found) == set(expected), where
        for key in expected:
            _close(found[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, float) and not isinstance(found, str):
        assert found == pytest.approx(expected, abs=CLOSE, rel=CLOSE), where
    else:
        assert found == expected, where


# --- 1. the file ---------------------------------------------------------------------------------


def test_the_file_is_the_one_named_in_the_results(out, document, rows):
    data = (out / compare_run.SPLITS_NAME).read_bytes()
    assert document["split_table"] == {
        "file": compare_run.SPLITS_NAME,
        "rows": len(rows),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    keys = [(r["unit"], r["variant"], r["tier"], r["method"], r["seed"]) for r in rows]
    variants, tiers = document["run"]["variants"], document["run"]["tiers"]
    expected = [
        (UNIT, v, t, m, float(s))
        for v in variants
        for t in tiers
        for m in METHODS
        for s in range(1, SPLITS + 1)
    ]
    assert keys == expected


# --- 2. the summary fields and the gate reading --------------------------------------------------


def _summary(found, tier):
    alpha_slot, alpha_driver = ALPHAS[tier]
    n = len(found)
    offers = [r["offer_kwh_per_counting_test_day"] for r in found]
    offering = [r for r in found if r["scale"] > 0]
    slot = [r["mean_test_slot_loss"] for r in offering if r["mean_test_slot_loss"] is not None]
    driver = [
        r["mean_test_driver_loss"] for r in offering if r["mean_test_driver_loss"] is not None
    ]
    shares = [
        r["share_of_best_causal_ceiling"]
        for r in offering
        if r["share_of_best_causal_ceiling"] is not None
    ]

    def over(losses, limit):
        count = sum(1 for x in losses if x > limit + TOLERANCE)
        return {
            "limit": limit,
            "splits": count,
            "share_of_all_splits": count / n,
            "share_of_splits_that_offer": count / len(offering) if offering else None,
        }

    out = {
        "splits": n,
        "mean_offer_kwh_per_test_day": _mean(offers),
        "offer_p10": _quantile(offers, 0.1),
        "offer_p50": _quantile(offers, 0.5),
        "offer_p90": _quantile(offers, 0.9),
        "splits_that_offer": len(offering),
        "share_of_splits_that_offer": len(offering) / n,
        "mean_scale": _mean(r["scale"] for r in found),
        "splits_at_scale_zero_by_reason": dict(
            sorted(Counter(r["scale_zero_reason"] for r in found if r["scale"] <= 0).items())
        ),
        "splits_with_test_losses": len(slot),
        "mean_test_slot_loss": _mean(slot),
        "mean_test_driver_loss": _mean(driver),
        "slot_loss_over_limit": over(slot, alpha_slot),
        "splits_with_ceiling_share": len(shares),
        "mean_share_of_best_causal_ceiling": _mean(shares),
    }
    if tier == "A":
        out["driver_loss_over_limit"] = over(driver, alpha_driver)
    return out


def test_every_summary_field_is_recomputed_from_the_file(document, rows):
    checked = 0
    for variant in document["run"]["variants"]:
        methods = document["units"][UNIT]["variants"][variant]["methods"]
        for method in METHODS:
            for tier in document["run"]["tiers"]:
                found = _select(rows, UNIT, variant, tier, method)
                assert len(found) == SPLITS
                _close(methods[method][tier], _summary(found, tier), f"{variant}.{method}.{tier}")
                checked += 1
    assert checked == 2 * 2 * len(METHODS)


def _gate(forecast, oracle, ceiling_names):
    offer = _mean(r["offer_kwh_per_counting_test_day"] for r in forecast)
    oracle_offer = _mean(r["offer_kwh_per_counting_test_day"] for r in oracle)
    by_ceiling = {}
    for name in ceiling_names:
        column = f"mean_{name}_ceiling_kwh_per_test_day"
        kept = [r for r in forecast if r[column] is not None]
        mean_ceiling = _mean(r[column] for r in kept)
        there = _mean(r["offer_kwh_per_counting_test_day"] for r in kept)
        usable = mean_ceiling is not None and mean_ceiling > 0
        by_ceiling[name] = {
            "splits_with_ceiling": len(kept),
            "splits_left_out_ceiling_missing": len(forecast) - len(kept),
            "mean_ceiling_kwh_per_test_day": mean_ceiling,
            "mean_offer_kwh_per_test_day_on_those_splits": there,
            "share_of_ceiling": there / mean_ceiling if usable else None,
        }
    return {
        "tier": "A",
        "splits": len(forecast),
        "mean_offer_kwh_per_test_day": offer,
        "mean_oracle_offer_kwh_per_test_day": oracle_offer,
        "oracle_to_offer_ratio": None if not offer else oracle_offer / offer,
        "oracle_to_offer_ratio_note": (
            "not a number: the mean offer is 0" if not offer else "oracle mean over offer mean"
        ),
        "ceilings": by_ceiling,
    }


def test_every_gate_reading_is_recomputed_from_the_file(document, rows):
    reading = document["units"][UNIT]["gate_reading"]
    ceilings = document["run"]["ceilings"]
    for method in METHODS:
        forecast = _select(rows, UNIT, "forecast", "A", method)
        oracle = _select(rows, UNIT, "oracle-departure", "A", method)
        _close(reading[method], _gate(forecast, oracle, ceilings), method)
    assert any(reading[m]["mean_offer_kwh_per_test_day"] > 0 for m in METHODS), (
        "some method must offer in tier A"
    )


# --- 3. the paired differences -------------------------------------------------------------------


def _verdict(p5, p95):
    return "ahead" if p5 > 0 else "behind" if p95 < 0 else "not separated"


def _pair(rows, variant, tier, method, reference):
    ours = {
        r["seed"]: r["offer_kwh_per_counting_test_day"]
        for r in _select(rows, UNIT, variant, tier, method)
    }
    theirs = {
        r["seed"]: r["offer_kwh_per_counting_test_day"]
        for r in _select(rows, UNIT, variant, tier, reference)
    }
    seeds = sorted(set(ours) & set(theirs))
    differences = [ours[s] - theirs[s] for s in seeds]
    n = len(differences)
    reference_mean = _mean(theirs[s] for s in seeds)
    mean = _mean(differences)
    p5, p95 = _quantile(differences, 0.05), _quantile(differences, 0.95)
    return {
        "method": method,
        "reference": reference,
        "splits_paired": n,
        "splits_left_out_value_missing": SPLITS - n,
        "mean_difference_kwh_per_test_day": mean,
        "difference_p5_kwh_per_test_day": p5,
        "difference_p95_kwh_per_test_day": p95,
        "share_of_paired_splits_above_0": sum(d > 0 for d in differences) / n,
        "share_of_paired_splits_equal_0": sum(d == 0 for d in differences) / n,
        "share_of_paired_splits_below_0": sum(d < 0 for d in differences) / n,
        "method_mean_offer_kwh_per_test_day": _mean(ours[s] for s in seeds),
        "reference_mean_offer_kwh_per_test_day": reference_mean,
        "mean_difference_share_of_reference_mean_offer": (
            None if reference_mean == 0 else mean / reference_mean
        ),
        "verdict": _verdict(p5, p95),
    }


def test_the_paired_differences_are_recomputed_from_the_file(document, rows):
    paired = document["paired_differences"]["units"]
    assert set(paired) == {UNIT}
    others = [m for m in METHODS if m != "re-solve"]
    verdicts = Counter()
    for variant in document["run"]["variants"]:
        for tier in document["run"]["tiers"]:
            found = paired[UNIT][variant][tier]
            against = [_pair(rows, variant, tier, m, "re-solve") for m in others]
            other = [
                _pair(rows, variant, tier, b, a)
                for i, a in enumerate(others)
                for b in others[i + 1 :]
            ]
            assert len(found["against_re_solve"]) == len(against)
            for got, expected in zip(found["against_re_solve"], against, strict=True):
                _close(got, expected, f"{variant}.{tier}")
            for got, expected in zip(found["other_pairs"], other, strict=True):
                _close(got, expected, f"{variant}.{tier}")
            verdicts.update(r["verdict"] for r in found["against_re_solve"] + found["other_pairs"])
    assert verdicts["behind"] > 0 and verdicts["not separated"] > 0, verdicts


# --- 4. the ceilings -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ceilings(build):
    _, tables, path = build
    matrix = gate_config.load(path)
    days = load_unit_days(
        tables, UNIT, rho=matrix.rho, no_fit=matrix.no_fit, small_fleet=matrix.small_fleet
    )
    found = [gate_day.day_ceilings(t.day) for t in days.days]
    splits = dial_choice.split_indices(UNIT, days, matrix, range(1, SPLITS + 1))
    per_split = {
        seed: (
            _mean(found[d].hindsight for d in test),
            _mean(found[d].best_causal for d in test),
        )
        for seed, _, test in splits
    }
    return found, per_split


def test_the_ceilings_differ_and_the_file_carries_their_means(ceilings, rows):
    days, per_split = ceilings
    assert all(c.hindsight > c.best_causal + 0.5 for c in days)
    for row in rows:
        hindsight, best = per_split[int(row["seed"])]
        assert row["mean_hindsight-free-decision-hour_ceiling_kwh_per_test_day"] == pytest.approx(
            hindsight, abs=CLOSE
        )
        assert row["mean_best-causal_ceiling_kwh_per_test_day"] == pytest.approx(best, abs=CLOSE)


def test_the_share_of_ceiling_is_over_the_best_causal_ceiling(ceilings, rows):
    _, per_split = ceilings
    checked = 0
    for row in rows:
        share = row["share_of_best_causal_ceiling"]
        if share is None:
            assert row["scale"] == 0
            continue
        hindsight, best = per_split[int(row["seed"])]
        offer = row["offer_kwh_per_counting_test_day"]
        assert share == pytest.approx(offer / best, rel=1e-9)
        assert share != pytest.approx(offer / hindsight, rel=1e-3)
        checked += 1
    assert checked > 0


# --- 5. the summary's cells ----------------------------------------------------------------------


def _share_text(value):
    return "-" if value is None else f"{100 * value:.0f}%"


def _block(out, document):
    lines = (out / compare_run.SUMMARY_NAME).read_text().splitlines()
    start = lines.index(next(x for x in lines if x.startswith(f"{UNIT} (")))
    end = lines.index("", start)
    return lines[start:end]


def test_the_of_ceiling_and_driver_loss_cells(out, document, rows, ceilings):
    _, per_split = ceilings
    tier, checked, with_share = None, 0, 0
    for line in _block(out, document):
        if line.startswith(" tier "):
            tier = line.split()[1]
        elif line.startswith(" Gate reading"):
            tier = None
        elif tier and line.split()[:1] in ([m] for m in METHODS):
            cells = line.split()
            for k, variant in enumerate(document["run"]["variants"]):
                found = _select(rows, UNIT, variant, tier, cells[0])
                offering = [r for r in found if r["scale"] > 0]
                shares = [
                    r["offer_kwh_per_counting_test_day"] / per_split[int(r["seed"])][1]
                    for r in offering
                ]
                driver = [
                    r["mean_test_driver_loss"]
                    for r in offering
                    if r["mean_test_driver_loss"] is not None
                ]
                assert cells[2 + 6 * k] == _share_text(_mean(shares))
                mean_driver = _mean(driver)
                assert cells[4 + 6 * k] == ("-" if mean_driver is None else f"{mean_driver:.3f}")
                with_share += bool(shares)
            checked += 1
    assert checked == 2 * len(METHODS) and with_share > 0


def test_the_gate_reading_cells(out, document, rows, ceilings):
    _, per_split = ceilings
    lines = _block(out, document)
    start = lines.index(next(x for x in lines if x.startswith(" Gate reading")))
    header = lines[start + 1].split()
    assert header[:2] == ["method", "offer"]
    ratios = 0
    for line in lines[start + 2 : start + 2 + len(METHODS)]:
        cells = line.split()
        forecast = _select(rows, UNIT, "forecast", "A", cells[0])
        oracle = _select(rows, UNIT, "oracle-departure", "A", cells[0])
        offer = _mean(r["offer_kwh_per_counting_test_day"] for r in forecast)
        oracle_offer = _mean(r["offer_kwh_per_counting_test_day"] for r in oracle)
        hindsight = _mean(per_split[int(r["seed"])][0] for r in forecast)
        best = _mean(per_split[int(r["seed"])][1] for r in forecast)
        assert cells[1] == f"{offer:.2f}"
        assert cells[2] == _share_text(offer / hindsight)
        assert cells[3] == _share_text(offer / best)
        assert cells[4] == f"{oracle_offer:.2f}"
        assert cells[5] == ("-" if not offer else f"{oracle_offer / offer:.2f}")
        ratios += bool(offer)
    assert ratios > 0, "some method must have an oracle-to-offer ratio"


# --- 6. the paired lines come after the summary of the results ----------------------------------


def test_the_paired_lines_follow_the_earlier_summary(out, document):
    lines = (out / compare_run.SUMMARY_NAME).read_text().splitlines()
    earlier = {k: v for k, v in document.items() if k != "paired_differences"}
    earlier["units"] = {u: document["units"][u] for u in GATE_UNITS}  # the file's key order
    base = compare_summary.summary_lines(earlier)
    assert lines[: len(base)] == base
    rest = lines[len(base) :]
    assert rest[:2] == ["", compare_summary.PAIRED_TEXT]
    shown = [x.split() for x in rest if x.startswith("  ") and x.split()[0] in METHODS]
    table = document["paired_differences"]["units"][UNIT]
    expected = [
        row
        for tier in document["run"]["tiers"]
        for variant in document["run"]["variants"]
        for name in ("against_re_solve", "other_pairs")
        for row in table[variant][tier][name]
    ]
    assert len(shown) == len(expected)
    for cells, row in zip(shown, expected, strict=True):
        assert cells[:2] == [row["method"], row["reference"]]
        assert cells[2] == f"{row['mean_difference_kwh_per_test_day']:+.3f}"
        assert cells[3] == f"{row['difference_p5_kwh_per_test_day']:+.3f}"
        assert cells[4] == f"{row['difference_p95_kwh_per_test_day']:+.3f}"
        share = row["mean_difference_share_of_reference_mean_offer"]
        reference = row["reference_mean_offer_kwh_per_test_day"]
        if reference == 0:
            assert cells[8:10] == ["no", "offer"]
        else:
            assert cells[8] == f"{100 * share:+.1f}%"
            assert share == pytest.approx(row["mean_difference_kwh_per_test_day"] / reference)
        assert cells[-2:] == [str(row["splits_paired"]), "0"]
        assert " ".join(cells).count(row["verdict"]) == 1
    for unit in (GATE_UNITS[0], GATE_UNITS[2]):
        assert f"{unit}: no paired differences: the unit cannot be split" in rest
    assert Path(out / compare_run.SPLITS_NAME).is_file()

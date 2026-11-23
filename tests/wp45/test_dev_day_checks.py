"""`python -m fleetreserve.dev_day_checks` end to end on a small made-up build (`dev_day_fakes`):
the files, the sample, worker independence, the refusals, and injected violations and timeouts
(each through a small seam; no check is weakened)."""

import csv
import json

import numpy as np
import pytest
from dev_day_fakes import CONFIG, N_DAYS, SAMPLE, run_command

from fleetreserve import ceiling_values, dev_day_checks, dev_day_values, t2_day, two_way_day
from fleetreserve.block_sample import allocate, day_text, draw
from fleetreserve.certificate import Certificate
from fleetreserve.policy import AffinePolicy

FILES = (
    dev_day_checks.RESULTS_NAME,
    dev_day_checks.TEXT_NAME,
    dev_day_checks.DAYS_NAME,
    dev_day_checks.SAMPLE_NAME,
)


def _table(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("dev")
    with pytest.MonkeyPatch.context() as patch:
        one = run_command(root / "one", patch, workers=1)
    with pytest.MonkeyPatch.context() as patch:
        two = run_command(root / "two", patch, workers=2)
    return {"codes": (one[0], two[0]), "one": one[1], "two": two[1]}


def _document(folder):
    return json.loads((folder / dev_day_checks.RESULTS_NAME).read_text())


def test_every_check_passes_on_the_made_up_build(runs):
    assert runs["codes"] == (0, 0)
    checks = _document(runs["one"])["checks"]
    for name in dev_day_checks.CHECKS:
        assert checks[name]["verdict"] == "passed", name
        assert checks[name]["covered_days"] == checks[name]["of_days"] > 0
    assert checks["order"]["of_days"] == N_DAYS
    assert checks["t2"]["of_days"] == checks["two_way"]["of_days"] == SAMPLE
    assert checks["t2"]["certified_days"] >= 1 and checks["t2"]["days_in_e_prime"] >= 1
    assert checks["t2"]["forecast_rows_checked"] > 0
    assert checks["t2"]["cars_checked_in_e_prime_i"] > 0
    assert checks["two_way"]["days_both_products_matched"] == SAMPLE
    assert "all checks passed" in (runs["one"] / dev_day_checks.TEXT_NAME).read_text()


def test_the_results_files_do_not_depend_on_the_worker_count(runs):
    for name in FILES:
        assert (runs["one"] / name).read_bytes() == (runs["two"] / name).read_bytes(), name
        assert "seconds" not in (runs["one"] / name).read_text()
    seconds = json.loads((runs["two"] / dev_day_checks.SECONDS_NAME).read_text())
    assert seconds["workers"] == 2


def test_the_files_cover_every_day_and_the_sample(runs):
    days = _table(runs["one"] / dev_day_checks.DAYS_NAME)
    assert len(days) == N_DAYS
    order = [(r["unit"], r["date"], r["site"]) for r in days]
    assert order == sorted(order)
    assert sum(r["in_t2_sample"] == "True" for r in days) == SAMPLE
    sample = _table(runs["one"] / dev_day_checks.SAMPLE_NAME)
    assert len(sample) == SAMPLE
    document = _document(runs["one"])
    assert len(document["t4_log_days"]) == N_DAYS
    assert document["configuration"] == {
        "rho": 0.0,
        "budget_kwh": 5.0,
        "energy_bounds": "known",
        "uncertifiable_car": "leave-unsteered",
        "kappa": 0.1,
        "departure_floor": "as-planned",
        "forecast_departure": "empirical kappa-quantile by site and arrival hour",
        "forecast_reference": "other-days",
        "forecast_min_cell": 20,
    }


def test_the_sample_is_drawn_with_its_own_label(runs):
    document = _document(runs["one"])
    days = _table(runs["one"] / dev_day_checks.DAYS_NAME)
    listed = {}
    for r in days:
        listed.setdefault(r["unit"], []).append(day_text(r["site"], r["date"]))
    shares = allocate({u: len(d) for u, d in listed.items()}, SAMPLE)
    expected = {
        day_text(*text.split(" ", 1))
        for unit, texts in listed.items()
        for text in draw(f"t2-check {unit}", document["sample"]["seed"], texts, shares[unit])
    }
    found = {day_text(r["site"], r["date"]) for r in days if r["in_t2_sample"] == "True"}
    assert found == expected
    assert document["sample"]["sampled_days"] == shares


def test_a_sample_other_than_100_is_marked(runs):
    text = (runs["one"] / dev_day_checks.TEXT_NAME).read_text()
    assert f"the T2 sample has {SAMPLE} days, not the pass mark's 100" in text
    assert _document(runs["one"])["sample"]["pass_mark_size"] == 100


def test_the_t4_log_splits_the_days(runs):
    t4 = _document(runs["one"])["checks"]["t4_log"]
    groups = t4["groups"]
    assert sum(g["days"] for g in groups.values()) == N_DAYS
    assert all(g["days"] > 0 for g in groups.values())
    assert (
        t4["hypothesis_days"]
        == t4["hypothesis_days_checked"]
        == groups["meets T4's hypotheses"]["days"]
    )


# --- injected violations, each through a seam ----------------------------------------------------


def test_an_injected_order_violation_is_a_failed_line_and_exit_1(tmp_path, monkeypatch, capsys):
    real = ceiling_values.block_values

    def broken(functions, rows):
        found = real(functions, rows)
        value, outcome, seconds = found["block_hind"]
        found["block_hind"] = (None if value is None else 0.5 * value - 1.0, outcome, seconds)
        return found

    monkeypatch.setattr(ceiling_values, "block_values", broken)
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 1
    text = capsys.readouterr().out
    assert text.splitlines()[0].startswith("FAILED: ")
    assert "block_caus" in text and "above block_hind" in text
    assert "CHECKS FAILED" in text
    assert _document(out)["checks"]["order"]["verdict"] == "failed"


def test_an_injected_t2_violation_is_a_failed_line_and_exit_1(tmp_path, monkeypatch, capsys):
    real = t2_day.t2_certificate

    def broken(rows) -> Certificate:
        found = real(rows)
        p = found.policy
        g = np.array(p.g)
        g[0, 1] += 50.0  # car 0 draws 50 kW more in hour 1 on every path
        bad = AffinePolicy(p.m, p.w, g, p.G)
        return Certificate(
            found.outcome,
            found.product,
            found.structure,
            found.shape_rule,
            found.value,
            found.scale,
            bad,
            found.first_value,
            found.seconds,
            found.sizes,
        )

    monkeypatch.setattr(t2_day, "t2_certificate", broken)
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 1
    text = capsys.readouterr().out
    assert "FAILED: " in text and "breaks a row of its true set" in text
    assert "miss the request" in text
    assert _document(out)["checks"]["t2"]["verdict"] == "failed"


def test_an_injected_replay_mismatch_is_a_failed_line(tmp_path, monkeypatch, capsys):
    real = AffinePolicy.worst_case
    monkeypatch.setattr(
        AffinePolicy, "worst_case", lambda self, rows, beta=1.0: real(self, rows, beta) + 1e-3
    )
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 1
    assert "closed-form worst case" in capsys.readouterr().out
    assert _document(out)["checks"]["t2"]["replay_mismatches"] > 0


def test_an_injected_two_way_mismatch_is_a_failed_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        two_way_day, "enlarged_scale", lambda rows, policy: (1.5, "optimal", True, 0.0)
    )
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 1
    assert "tree does not match" in capsys.readouterr().out
    assert _document(out)["checks"]["two_way"]["verdict"] == "failed"


def test_a_timeout_is_missing_and_the_check_could_not_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dev_day_values, "hourly_diagonal", lambda task: (None, "timeout", 30.0))
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 2
    order = _document(out)["checks"]["order"]
    assert order["verdict"] == "could not run"
    assert order["timeout_days"] > 0
    assert order["covered_days"] == order["of_days"] - order["timeout_days"]
    assert "COULD NOT RUN: a check could not run" in capsys.readouterr().out


def test_a_certificate_timeout_leaves_t2_could_not_run(tmp_path, monkeypatch):
    real = t2_day.t2_certificate

    def timed_out(rows):
        found = real(rows)
        return Certificate(
            "timeout", found.product, "causal", "x", None, None, None, None, 30.0, found.sizes
        )

    monkeypatch.setattr(t2_day, "t2_certificate", timed_out)
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 2
    t2 = _document(out)["checks"]["t2"]
    assert t2["verdict"] == "could not run" and t2["timeout_days"] > 0


# --- configurations and refusals -----------------------------------------------------------------


def test_day_abstains_gives_days_without_a_certificate(tmp_path, monkeypatch):
    config = list(CONFIG)
    config[config.index("leave-unsteered")] = "day-abstains"
    code, out = run_command(tmp_path / "b", monkeypatch, config=config)
    t2 = _document(out)["checks"]["t2"]
    abstained = t2["days_without_certificate_by_rule"]
    assert abstained.get(t2_day.ABSTAINED, 0) > 0
    assert code in (0, 2)  # either outcome is the configuration's, not a failure


def test_a_check_that_looked_at_nothing_could_not_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(t2_day, "forecast_set_empty", lambda forecast, n, r: True)
    code, out = run_command(tmp_path / "b", monkeypatch)
    assert code == 2
    t2 = _document(out)["checks"]["t2"]
    assert t2["certified_days"] == 0 and t2["verdict"] == "could not run"
    assert "no sampled day was certified" in capsys.readouterr().out


def test_a_sample_larger_than_the_counting_days_could_not_run(tmp_path, monkeypatch, capsys):
    code, _ = run_command(tmp_path / "b", monkeypatch, sample=500)
    assert code == 2
    assert "fewer than the sample size 500" in capsys.readouterr().out


@pytest.mark.parametrize(
    "left_out", ["--energy-bounds", "--uncertifiable-car", "--kappa", "--departure-floor"]
)
def test_every_forecast_argument_is_required(left_out, capsys):
    config = list(CONFIG)
    k = config.index(left_out)
    del config[k : k + 2]
    with pytest.raises(SystemExit) as stopped:
        dev_day_checks.main(["--t2-sample-size", "3", *config])
    assert stopped.value.code == 2
    assert f"the following arguments are required: {left_out}" in capsys.readouterr().err


def test_a_bad_kappa_could_not_run(capsys):
    config = list(CONFIG)
    config[config.index("0.1")] = "1.5"
    assert dev_day_checks.main(["--t2-sample-size", "3", *config]) == 2
    assert "COULD NOT RUN: --kappa must be in (0, 1)" in capsys.readouterr().out

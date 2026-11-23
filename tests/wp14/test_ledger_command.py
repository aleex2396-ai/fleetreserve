"""The ledger command (`ledger_run`) on made-up scoring runs of several units, and its refusals.

The made-up build: the three gate units of `ledger_fake.ledger_units` (5, 24 and 24 days; the
first not calibratable), on the learn matrix of `learn_fake.small_matrix` (20 calibration days, so
4 test days per split unit; the empirical configuration and its oracle partner; tiers A and B).
`gate_run` and `learn_run --model empirical` are run on it and recorded (`ledger_fake.record`),
and the ledger is run on each with `--tree-days 3 --seed 7`.

1. Both ledgers end with exit 0; the first unit has no ledger (not calibratable). Each
   `ledger.json` lists every reading of `ledger_run`'s module text, L1 to L13.
2. The identity on every day, recomputed from the written day rows: offer + (R + A) + U + K and
   offer + R + A + U + K equal the ceiling to 1e-9 of max(1, ceiling) on every row (R and A apart
   on the tree sample's rows). There is a row per configuration, split unit, tier, rung and test
   day.
3. Each day's terms recomputed from the run's own files (its day table and results): the offer is
   the run's, R + A = V_hind - V_aff, U = V_aff - V_cert, K = (1 - scale) V_cert with the run's
   scale for the rung and tier.
4. The means: each is the mean of its days' rows, the means add up to the mean ceiling (1e-9), the
   shares add up to 1, and the offer mean is the run's own `test.mean_offer_kwh_per_day`.
5. The learn run's ledger is the gate run's (the same scored numbers), and 2 workers write the
   same bytes as 1.
6. A wrong term (K formed from V_aff instead of V_cert) breaks the identity: exit 1, and the
   summary leads with the failed check.
6b. A recorded one-way ceiling other than the one the ceiling ratio recomputes: the gate run with
   one test day of `nrel-from-may-2019` outside the tree sample given a one-way ceiling 0.5 kWh
   higher in every configuration's row, and each configuration's recorded sum 0.5 kWh higher, so
   that the identity, the sums and the sampled days all still agree: exit 1, one break per
   configuration naming the day, and the summary leads with that check.
7. A tree that hit its time limit is missing and counted, never a failed check; an A below 0 is
   reported as measured (never clipped) and breaks the plan's order (exit 1).
8. Refusals, exit 2 with nothing read further: a run that ended with exit 1, a day table changed
   after the run, a run recorded at another commit, a command that is not a scoring run, another
   matrix, a trial run, a run that read other tables, day rows that are not the tables' days, and
   `--tree-days 0`; `--tree-days` and `--seed` have no default.
"""

import csv
import json
import re
import shutil

import pytest
from gate_fake import COMMIT, write_build
from learn_fake import small_matrix
from ledger_fake import GATE_MODULE, LEARN_MODULE, ledger_units, record, rehash

from fleetreserve import gate_run, learn_run, ledger_day, ledger_run, ledger_terms, run_verify
from fleetreserve.block_sample import draw
from fleetreserve.gate_config import CEILINGS
from fleetreserve.rungs import RUNGS

TOLERANCE = 1e-9
SAMPLE = ["--tree-days", "3", "--seed", "7"]


def _verified(monkeypatch):
    monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
    monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
    monkeypatch.setattr(ledger_run, "_head_commit", lambda: COMMIT)


@pytest.fixture(autouse=True)
def verified(monkeypatch):
    _verified(monkeypatch)


def _inputs(paths) -> list[str]:
    return ["--tables", str(paths["tables"]), "--truth-run", str(paths["truth"])]


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    truth, tables = write_build(folder, ledger_units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(small_matrix()))
    paths = {"folder": folder, "truth": truth, "tables": tables, "matrix": matrix}
    arguments = ["--matrix", str(matrix), *_inputs(paths)]
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        for name, command, extra in (
            ("gate", gate_run.main, []),
            ("learn", learn_run.main, ["--model", "empirical"]),
        ):
            source = folder / f"{name}-source"
            source.mkdir()
            monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(source))
            assert command([*arguments, *extra]) == 0
    paths["gate"] = record(folder, folder / "gate-source", GATE_MODULE, "gate")
    paths["learn"] = record(folder, folder / "learn-source", LEARN_MODULE, "learn")
    return paths


def _ledger(paths, run, out, *extra) -> int:
    with pytest.MonkeyPatch.context() as monkeypatch:
        _verified(monkeypatch)
        out.mkdir()
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        return ledger_run.main(
            ["--run", str(run), "--matrix", str(paths["matrix"]), *_inputs(paths), *extra]
        )


@pytest.fixture(scope="module")
def ledgers(build, tmp_path_factory):
    folder = tmp_path_factory.mktemp("ledgers")
    codes = {
        name: _ledger(build, build[name], folder / name, *SAMPLE) for name in ("gate", "learn")
    }
    return codes, folder


def _rows(path) -> list[dict]:
    return list(csv.DictReader(path.read_text().splitlines()))


def _number(text):
    return None if text == "" else float(text)


def test_both_ledgers_pass(ledgers):
    codes, folder = ledgers
    assert codes == {"gate": 0, "learn": 0}
    document = json.loads((folder / "gate" / ledger_run.RESULTS_NAME).read_text())
    assert document["checks"]["failed"] is False
    units = document["units"]
    assert units["nrel-before-may-2019"]["status"].startswith("not calibratable")
    assert "hourly_true_sets" not in units["nrel-before-may-2019"]
    for unit in ("nrel-from-may-2019", "norway-development-pooled"):
        assert units[unit]["test_days"] == 4 and len(units[unit]["tree_sample_days"]) == 3
    summary = (folder / "gate" / ledger_run.SUMMARY_NAME).read_text()
    assert "CHECK FAILED" not in summary and "R = V_hind - V_caus" in summary


def test_the_ledger_lists_every_reading_of_its_module_text(ledgers):
    _, folder = ledgers
    written = re.findall(r"^  (L\d+)\. ", ledger_run.__doc__, flags=re.MULTILINE)
    assert written == [f"L{k}" for k in range(1, 14)]
    for name in ("gate", "learn"):
        document = json.loads((folder / name / ledger_run.RESULTS_NAME).read_text())
        assert document["readings"] == written


def test_the_identity_holds_on_every_day_row(ledgers):
    _, folder = ledgers
    rows = _rows(folder / "gate" / ledger_run.DAYS_NAME)
    assert len(rows) == 2 * 2 * 2 * len(RUNGS) * 4  # configurations, units, tiers, rungs, days
    split = 0
    for row in rows:
        ceiling = float(row["ceiling_kwh"])
        bound = TOLERANCE * max(1.0, abs(ceiling))
        joint = sum(float(row[f"{n}_kwh"]) for n in ("offer", "r_plus_a", "u", "k"))
        assert abs(joint - ceiling) <= bound, row
        if row["in_tree_sample"] == "True":
            split += 1
            apart = sum(float(row[f"{n}_kwh"]) for n in ("offer", "r", "a", "u", "k"))
            assert abs(apart - ceiling) <= bound, row
        else:
            assert row["r_kwh"] == row["a_kwh"] == row["tree_kwh"] == ""
    assert split == 2 * 2 * 2 * len(RUNGS) * 3
    r_plus_a = [float(r["r_plus_a_kwh"]) for r in rows]
    assert max(r_plus_a) > 0.1  # the identity is not checked on days where R + A is only 0


def test_each_term_is_the_runs_numbers(build, ledgers):
    _, folder = ledgers
    document = json.loads((build["gate"] / gate_run.RESULTS_NAME).read_text())
    run_rows = {
        (r["configuration"], r["unit"], r["date"], r["site"]): r
        for r in _rows(build["gate"] / gate_run.DAYS_NAME)
        if r["role"] == "test"
    }
    rows = _rows(folder / "gate" / ledger_run.DAYS_NAME)
    assert {(r["configuration"], r["unit"], r["date"], r["site"]) for r in rows} == set(run_rows)
    for row in rows:
        day = run_rows[(row["configuration"], row["unit"], row["date"], row["site"])]
        rungs = document["configurations"][row["configuration"]]["units"][row["unit"]]["rungs"]
        scored = rungs[row["rung"]] if row["tier"] == "A" else rungs[row["rung"]]["tier_b"]
        column = (
            f"{row['rung']}_offer_kwh" if row["tier"] == "A" else f"{row['rung']}_tier_b_offer_kwh"
        )
        hind = float(day["ceiling_hindsight_kwh"])
        best = float(day["ceiling_best_causal_kwh"])
        certified = float(day["certified_value_kwh"])
        assert float(row["offer_kwh"]) == float(day[column])
        assert float(row["ceiling_kwh"]) == hind
        assert float(row["scale"]) == scored["beta_hat"]
        assert float(row["r_plus_a_kwh"]) == pytest.approx(hind - best, abs=1e-12)
        assert float(row["u_kwh"]) == pytest.approx(best - certified, abs=1e-12)
        assert float(row["k_kwh"]) == pytest.approx((1 - scored["beta_hat"]) * certified, abs=1e-12)


def test_the_means_add_up_and_are_the_runs(build, ledgers):
    _, folder = ledgers
    ledger = json.loads((folder / "gate" / ledger_run.RESULTS_NAME).read_text())
    run = json.loads((build["gate"] / gate_run.RESULTS_NAME).read_text())
    rows = _rows(folder / "gate" / ledger_run.DAYS_NAME)
    checked = 0
    for unit in ("nrel-from-may-2019", "norway-development-pooled"):
        for name, found in ledger["units"][unit]["configurations"].items():
            rungs = run["configurations"][name]["units"][unit]["rungs"]
            for tier, by_rung in found["tiers"].items():
                for rung, entry in by_rung.items():
                    mine = [
                        r
                        for r in rows
                        if (r["configuration"], r["unit"], r["tier"], r["rung"])
                        == (name, unit, tier, rung)
                    ]
                    scored = rungs[rung] if tier == "A" else rungs[rung]["tier_b"]
                    assert (
                        entry["run_mean_offer_kwh_per_test_day"]
                        == (scored["test"]["mean_offer_kwh_per_day"])
                    )
                    for block, names, chosen in (
                        ("test_days", ledger_terms.JOINT, mine),
                        ("tree_sample", ledger_terms.SPLIT, [r for r in mine if r["r_kwh"]]),
                    ):
                        means = entry[block]
                        assert means["days"] == len(chosen)
                        ceiling = sum(float(r["ceiling_kwh"]) for r in chosen) / len(chosen)
                        assert means["mean_ceiling_kwh"] == pytest.approx(ceiling, abs=1e-12)
                        for term in names:
                            mean = sum(float(r[f"{term}_kwh"]) for r in chosen) / len(chosen)
                            assert means["mean_kwh"][term] == pytest.approx(mean, abs=1e-12)
                        total = sum(means["mean_kwh"][t] for t in names)
                        assert abs(total - ceiling) <= TOLERANCE * max(1.0, ceiling)
                        shares = sum(means["share_of_ceiling"][t] for t in names)
                        assert shares == pytest.approx(1.0, abs=1e-9)
                        assert entry["test_days"]["mean_kwh"]["offer"] == pytest.approx(
                            scored["test"]["mean_offer_kwh_per_day"], abs=1e-12
                        )
                        checked += 1
    assert checked == 2 * 2 * 2 * len(RUNGS) * 2


def test_the_tree_sample_is_the_seeded_draw(build, ledgers):
    _, folder = ledgers
    ledger = json.loads((folder / "gate" / ledger_run.RESULTS_NAME).read_text())
    rows = [r for r in _rows(build["gate"] / gate_run.DAYS_NAME) if r["role"] == "test"]
    for unit in ("nrel-from-may-2019", "norway-development-pooled"):
        texts = sorted({f"{r['site']} {r['date']}" for r in rows if r["unit"] == unit})
        assert ledger["tree_sample"]["days"][unit] == list(draw(unit, 7, texts, 3))
    assert ledger["tree_sample"]["seed"] == 7 and ledger["tree_sample"]["tree_days_per_unit"] == 3


def test_the_learn_runs_ledger_is_the_gate_runs(ledgers):
    _, folder = ledgers
    gate = json.loads((folder / "gate" / ledger_run.RESULTS_NAME).read_text())
    learn = json.loads((folder / "learn" / ledger_run.RESULTS_NAME).read_text())
    assert learn["run"]["command"] == LEARN_MODULE and gate["run"]["command"] == GATE_MODULE
    del gate["run"], learn["run"]
    assert learn == gate
    for name in (ledger_run.DAYS_NAME, ledger_run.SAMPLE_NAME):
        assert (folder / "learn" / name).read_bytes() == (folder / "gate" / name).read_bytes()


def test_two_workers_write_the_same_bytes(build, ledgers, tmp_path):
    _, folder = ledgers
    assert _ledger(build, build["gate"], tmp_path / "two", *SAMPLE, "--workers", "2") == 0
    for name in (
        ledger_run.RESULTS_NAME,
        ledger_run.DAYS_NAME,
        ledger_run.SAMPLE_NAME,
        ledger_run.SUMMARY_NAME,
    ):
        assert (tmp_path / "two" / name).read_bytes() == (folder / "gate" / name).read_bytes()


def test_a_wrong_term_breaks_the_identity(build, tmp_path, monkeypatch):
    right = ledger_terms.day_terms

    def wrong(ceiling, best_causal, certified, beta, offer, tree=None):
        terms = right(ceiling, best_causal, certified, beta, offer, tree)
        terms["k"] = (1.0 - beta) * best_causal  # V_aff in place of V_cert
        return terms

    monkeypatch.setattr(ledger_terms, "day_terms", wrong)
    assert _ledger(build, build["gate"], tmp_path / "wrong", *SAMPLE) == 1
    document = json.loads((tmp_path / "wrong" / ledger_run.RESULTS_NAME).read_text())
    assert document["checks"]["failed"] is True and document["checks"]["identity"]["breaks"]
    summary = (tmp_path / "wrong" / ledger_run.SUMMARY_NAME).read_text()
    assert "CHECK FAILED: the identity" in summary


def _one_way_ceiling_raised(build, unit: str, site: str, date: str, by: float):
    """The gate run with one test day's recorded one-way ceiling raised by `by` kWh in every
    configuration's row, and each configuration's recorded sum of it raised with it (module text,
    item 6b)."""

    def edit(folder):
        path = folder / gate_run.DAYS_NAME
        rows = _rows(path)
        for row in rows:
            if (row["unit"], row["site"], row["date"]) == (unit, site, date):
                row["ceiling_hindsight_kwh"] = repr(float(row["ceiling_hindsight_kwh"]) + by)
        path.write_text(gate_run._csv(rows, gate_run.DAY_COLUMNS, None))
        results = folder / gate_run.RESULTS_NAME
        document = json.loads(results.read_text())
        for entry in document["configurations"].values():
            entry["units"][unit]["ceiling_sums_test_days_kwh"][CEILINGS[0]] += by
        results.write_text(json.dumps(document))

    return _variant(build, "one-way-ceiling-raised", edit)


def test_a_recorded_one_way_ceiling_other_than_the_recomputed_one_fails(build, ledgers, tmp_path):
    _, folder = ledgers
    unit = "nrel-from-may-2019"
    sampled = json.loads((folder / "gate" / ledger_run.RESULTS_NAME).read_text())["tree_sample"]
    names = sorted({r["configuration"] for r in _rows(build["gate"] / gate_run.DAYS_NAME)})
    tests = [
        (r["site"], r["date"])
        for r in _rows(build["gate"] / gate_run.DAYS_NAME)
        if (r["configuration"], r["unit"], r["role"]) == (names[0], unit, "test")
    ]
    site, date = next(d for d in tests if f"{d[0]} {d[1]}" not in sampled["days"][unit])
    run = _one_way_ceiling_raised(build, unit, site, date, 0.5)
    assert _ledger(build, run, tmp_path / "raised", *SAMPLE) == 1
    checks = json.loads((tmp_path / "raised" / ledger_run.RESULTS_NAME).read_text())["checks"]
    breaks = checks["ceiling_ratio_one_way_reproduced"]["breaks"]
    assert sorted(b.split(":")[0] for b in breaks) == [f"{n} {unit} {date} {site}" for n in names]
    assert all(": the one-way ceiling " in b and ", the run's " in b for b in breaks)
    for other in ("identity", "run_ceiling_sums_reproduced", "sampled_days_are_the_runs"):
        assert checks[other]["breaks"] == [], other  # the day is not sampled; the sums agree
    summary = (tmp_path / "raised" / ledger_run.SUMMARY_NAME).read_text().splitlines()
    assert summary[0].startswith(
        "CHECK FAILED: the ceiling ratio's one-way ceilings against the run's, 2 breaks: "
    )


def _changed_tree(monkeypatch, change, unit="nrel-from-may-2019"):
    """Every sampled day solved as usual, then the values of the first one of `unit` changed by
    `change`."""
    right = ledger_run.solve_day
    seen = []

    def solve(task):
        found = right(task)
        if task.unit != unit:
            return found
        seen.append(found.key)
        if len(seen) == 1:
            values, outcomes = change(dict(found.values), dict(found.outcomes))
            return ledger_day.SampleDay(
                found.unit,
                found.site,
                found.date,
                found.cars,
                found.n_service,
                found.error,
                values,
                outcomes,
                found.seconds,
            )
        return found

    monkeypatch.setattr(ledger_run, "solve_day", solve)


def test_a_timed_out_tree_is_missing_and_counted(build, tmp_path, monkeypatch):
    def timed_out(values, outcomes):
        values["hourly_tree"] = None
        outcomes["hourly_tree"] = "time limit"
        return values, outcomes

    _changed_tree(monkeypatch, timed_out)
    assert _ledger(build, build["gate"], tmp_path / "timeout", *SAMPLE) == 0
    document = json.loads((tmp_path / "timeout" / ledger_run.RESULTS_NAME).read_text())
    first = "nrel-from-may-2019"
    tree = document["units"][first]["hourly_true_sets"]["tree_sample"]
    assert tree["drawn"] == 3 and tree["days"] == 2
    assert tree["tree_outcomes"] == {"optimal": 2, "time limit": 1}
    entry = document["units"][first]["configurations"]["main-k0.1-empirical"]["tiers"]["A"]
    assert entry["fixed"]["tree_sample"]["days"] == 2 and entry["fixed"]["test_days"]["days"] == 4
    other = document["units"]["norway-development-pooled"]["hourly_true_sets"]["tree_sample"]
    assert other["days"] == 3


def test_an_a_below_zero_is_reported_as_measured_and_breaks_the_order(build, tmp_path, monkeypatch):
    def lower(values, outcomes):
        values["hourly_tree"] = values["hourly_causal"] - 0.25
        return values, outcomes

    _changed_tree(monkeypatch, lower)
    assert _ledger(build, build["gate"], tmp_path / "below", *SAMPLE) == 1
    document = json.loads((tmp_path / "below" / ledger_run.RESULTS_NAME).read_text())
    tree = document["units"]["nrel-from-may-2019"]["hourly_true_sets"]["tree_sample"]
    assert tree["a_not_zero_days"] == 1
    assert tree["largest_abs_a_kwh"] == pytest.approx(0.25, abs=1e-6)
    assert document["checks"]["order"]["breaks"]
    assert document["checks"]["identity"]["breaks"] == []
    rows = _rows(tmp_path / "below" / ledger_run.DAYS_NAME)
    assert min(_number(r["a_kwh"]) for r in rows if r["a_kwh"]) == pytest.approx(-0.25, abs=1e-6)
    summary = (tmp_path / "below" / ledger_run.SUMMARY_NAME).read_text()
    assert "CHECK FAILED: the plan's order" in summary


# --- refusals ------------------------------------------------------------------------------------


def _variant(build, label, edit, module=GATE_MODULE, **recorded):
    """The gate run's files changed by `edit(folder)`, recorded as a new run."""
    source = build["folder"] / f"source-{label}"
    shutil.copytree(build["folder"] / "gate-source", source)
    edit(source)
    return record(build["folder"], source, module, label, **recorded)


def _edit_results(change):
    def edit(folder):
        path = folder / gate_run.RESULTS_NAME
        document = json.loads(path.read_text())
        change(document)
        path.write_text(json.dumps(document))

    return edit


def _same(folder):
    return None


def _first_test_row_to_calibration(folder):
    path = folder / gate_run.DAYS_NAME
    lines = path.read_text().splitlines(keepends=True)
    for k, line in enumerate(lines):
        if ",test," in line:
            lines[k] = line.replace(",test,", ",calibration,", 1)
            break
    path.write_text("".join(lines))


def _set_trial(document):
    document["trial"] = True


def _other_table(document):
    document["inputs"]["table_sha256"]["nrel-from-may-2019-cars.csv"] = "0" * 64


def _other_matrix(document):
    document["matrix"]["sha256"] = "0" * 64


REFUSED = {  # case -> (the run, a part of the reason printed)
    "exit-1": (lambda b: _variant(b, "exit-1", _same, exit_code=1), "exit code 1"),
    "stale": (lambda b: _variant(b, "stale", _same, commit="b" * 40), "is stale"),
    "not-scoring": (
        lambda b: _variant(b, "not-scoring", _same, module="fleetreserve.compare_run"),
        "is not a recorded run of",
    ),
    "trial": (lambda b: _variant(b, "trial", _edit_results(_set_trial)), "is a trial run"),
    "other-table": (
        lambda b: _variant(b, "other-table", _edit_results(_other_table)),
        "is not the table",
    ),
    "other-matrix": (
        lambda b: _variant(b, "other-matrix", _edit_results(_other_matrix)),
        "ran on another matrix",
    ),
    "other-days": (
        lambda b: _variant(b, "other-days", _first_test_row_to_calibration),
        "are not the days and roles of the tables",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_run_that_cannot_be_read_could_not_run(build, tmp_path, case, capsys):
    make, reason = REFUSED[case]
    assert _ledger(build, make(build), tmp_path / "out", *SAMPLE) == 2
    printed = capsys.readouterr().out
    assert "COULD NOT RUN" in printed and reason in printed, printed
    assert not (tmp_path / "out" / ledger_run.RESULTS_NAME).exists()


def test_a_day_table_changed_after_the_run_could_not_run(build, tmp_path, capsys):
    run = _variant(build, "changed-days", _same)
    with (run / gate_run.DAYS_NAME).open("a") as handle:
        handle.write("\n")
    assert _ledger(build, run, tmp_path / "out", *SAMPLE) == 2
    assert "is not verified" in capsys.readouterr().out
    rehash(run)  # the same files listed again: read
    assert _ledger(build, run, tmp_path / "again", *SAMPLE) == 0


def test_a_tree_sample_of_no_days_could_not_run(build, tmp_path):
    assert _ledger(build, build["gate"], tmp_path / "out", "--tree-days", "0", "--seed", "7") == 2


@pytest.mark.parametrize("dropped", ["--tree-days", "--seed"])
def test_the_sample_has_no_default(build, tmp_path, dropped):
    given = {"--tree-days": "3", "--seed": "7"}
    del given[dropped]
    extra = [part for pair in given.items() for part in pair]
    with pytest.raises(SystemExit) as stopped:
        _ledger(build, build["gate"], tmp_path / "out", *extra)
    assert stopped.value.code == 2

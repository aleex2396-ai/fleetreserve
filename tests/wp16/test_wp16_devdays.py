"""The development figures of `wp16_devdays` and the two options of `wp16_run` that read them, on
made-up run folders in the writers' form (`dev_day_checks`, `dial_choice`; made-up numbers).

Each test names, in its last comment, the change of the code that turns it red. A module that
reads nothing fails every test that names a number.
"""

import json
import shutil
from pathlib import Path

import pytest
from test_wp16_run import made_up_top
from wp16_made_up import (
    COMMIT,
    MadeUp,
    dev_days_document,
    dial_choice_document,
    distinct_dev_days_document,
    distinct_dial_choice_document,
    sha,
)

from fleetreserve import (
    dev_day_checks,
    dev_day_verdicts,
    dial_choice,
    wp16_devdays,
    wp16_figures,
    wp16_run,
)
from fleetreserve.wp16_cells import Missing
from fleetreserve.wp16_drive import Source

DEV_LABEL = "wp5-dev-days"
DIAL_LABEL = "wp6-dial-choice-m1"
DEV_COMMAND = ["python", "-m", "fleetreserve.dev_day_checks", "--t2-sample-size", "10"]
DIAL_COMMAND = ["python", "-m", "fleetreserve.dial_choice", "--model", "m1"]
DEV_OPTION, DIAL_OPTION = "dev_days_run", "dial_choice_run"
NEW_FILES = {
    "dev-days-checks.csv",
    "dev-days-checks.md",
    "dev-dial-choice.csv",
    "dev-dial-choice.md",
}


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    monkeypatch.setattr(wp16_run, "head_commit", lambda root: COMMIT)

    def render(spec, folder):
        name = f"{spec['kind']}.png"
        (folder / name).write_bytes(b"placeholder")
        return name

    monkeypatch.setattr(wp16_figures, "render", render)
    monkeypatch.delenv("FLEETRESERVE_RUN_DIR", raising=False)


def make_run(top, label, name, document, command, exit_code=0) -> Path:
    """A made-up run folder as the recorder writes one; returns its path."""
    made = MadeUp(top)
    made.commands[label] = command
    return Path(top) / made.run(label, {name: document}, exit_code)["run"]


def dev_run(top, document=None, exit_code=0) -> Path:
    document = dev_days_document() if document is None else document
    return make_run(top, DEV_LABEL, "dev-day-checks.json", document, DEV_COMMAND, exit_code)


def dial_run(top, document=None, exit_code=0) -> Path:
    document = dial_choice_document() if document is None else document
    return make_run(top, DIAL_LABEL, "dial-choice.json", document, DIAL_COMMAND, exit_code)


def edit_manifest(folder: Path, **changes) -> None:
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def table_of(top, option, folder):
    """The one table `build` makes for `option` (and the manifest's record)."""
    source = Source(top, None)  # the options read no step of the drive
    tables, info = wp16_devdays.build(source, {option: folder})
    (table,) = tables
    assert all(len(row) == len(wp16_devdays.COLUMNS) for row in table.rows)
    return table, info, source


def cells(table) -> dict:
    """key in the file -> the figure's cell (the keys of one table are not repeated)."""
    found = {row[5]: row[4] for row in table.rows}
    assert len(found) == len(table.rows)
    return found


def leaves(document, path=()) -> dict:
    """The scalar entries of a JSON document by the dotted key the tables use."""
    if isinstance(document, dict):
        out = {}
        for name, value in document.items():
            out.update(leaves(value, (*path, name)))
        return out
    if isinstance(document, list):
        return {}
    return {wp16_devdays.key_of(path): document}


def texts(table) -> list:
    return [cell.text for cell in cells(table).values() if isinstance(cell, Missing)]


# --- the names are the writers' names ---------------------------------------------


def test_the_names_the_reader_uses_are_the_writers_names():
    assert wp16_devdays.DEV_DAYS_FILE == dev_day_checks.RESULTS_NAME
    assert wp16_devdays.DIAL_FILE == dial_choice.RESULTS_NAME
    assert dev_day_checks.__name__ == wp16_devdays.DEV_DAYS_MODULE
    assert dial_choice.__name__ == wp16_devdays.DIAL_MODULE
    assert wp16_devdays.CHECKS == dev_day_checks.CHECKS
    assert wp16_devdays.T4_GROUPS == dev_day_verdicts.T4_GROUPS
    assert (wp16_devdays.PASSED, wp16_devdays.FAILED, wp16_devdays.COULD_NOT_RUN) == (
        dev_day_verdicts.PASSED,
        dev_day_verdicts.FAILED,
        dev_day_verdicts.COULD_NOT_RUN,
    )
    # red: a writer that renames its results file, a check, a T4 group or a verdict.


# --- the figures are the summary's ----------------------------------------------------


def test_the_development_day_figures_are_the_files_own_and_none_is_left_out(tmp_path):
    document = dev_days_document()
    table, info, source = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    assert table.name == "dev-days-checks" and "Development figures" in table.title
    assert {row[0] for row in table.rows} == {"development"}
    found = cells(table)
    t2 = found["checks.t2.days_in_e_prime"], found["checks.t2.share_of_certified_days_in_e_prime"]
    assert t2 == (4, 0.5)
    assert found["checks.t2.replay_mismatches"] == 6
    assert found["checks.t2.cars_outside_e_prime_i_breaking_a_true_row"] == 7
    assert found["checks.t2.days_without_certificate_by_rule.no certificate: no steered car"] == 2
    group = "checks.t4_log.groups.other days, in-window departure"
    assert found[f"{group}.mean_gap_tree_minus_affine_kwh"] == 1.5
    assert found["checks.t4_log.groups.meets T4's hypotheses.days_tree_below_hind"] == 1
    assert found["configuration.energy_bounds"] == "known" and found["configuration.kappa"] == 0.1
    assert found["sample.counting_days.d1"] == 30 and found["checks.two_way.verdict"] == "passed"
    assert found["checks.two_way.days_both_products_matched"] == 9
    assert found["checks.order.failure_lines"] == wp16_devdays.NONE_LISTED
    # every scalar of the file is a row, with the file's value, whatever its key
    for key, value in leaves(document).items():
        assert found[key] == value, key
    assert not texts(table)
    assert source.hashes[f"{info[DEV_OPTION]['run']}/dev-day-checks.json"] == sha(
        (tmp_path / info[DEV_OPTION]["run"] / "dev-day-checks.json").read_bytes()
    )
    assert info[DEV_OPTION]["commit"] == COMMIT and info[DEV_OPTION]["exit_code"] == 0
    # red: a module that reads nothing; a figure read from another key (the made-up values differ
    # from key to key); a quantity dropped from the lists of keys (the loop over the scalars);
    # a row that is not marked "development".


def test_the_dial_choice_figures_are_the_files_own(tmp_path):
    document = dial_choice_document()
    table, info, source = table_of(tmp_path, DIAL_OPTION, dial_run(tmp_path, document))
    assert table.name == "dev-dial-choice"
    found = cells(table)
    cfg = "units.d1.configurations.cfg-0.02"
    assert found[f"{cfg}.tiers.A.mean_offer_kwh_per_test_day"] == 7.25
    assert found[f"{cfg}.tiers.A.offer_p90"] == 15.0
    assert found[f"{cfg}.tiers.A.share_of_splits_that_offer"] == 0.75
    assert found[f"{cfg}.tiers.A.driver_loss_over_limit.share_of_splits_that_offer"] == 0.02
    assert found[f"{cfg}.tiers.B.mean_offer_kwh_per_test_day"] == 8.5
    assert not [key for key in found if "tiers.B.driver_loss_over_limit" in key]
    gate = f"{cfg}.gate_reading"
    assert found[f"{gate}.mean_oracle_offer_kwh_per_test_day"] == 21.75
    assert found[f"{gate}.ceilings.hindsight-free-decision-hour.share_of_ceiling"] == 7.25 / 64.0
    assert found[f"{gate}.ceilings.best-causal.splits_left_out_ceiling_missing"] == 2
    assert found["dial_choice.d1.A.dial"] == 0.02 and found["dial_choice.d1.B.dial"] == 0.01
    assert found["dial_choice.d1.B.mean_offer_kwh_per_test_day"] == 9.5
    assert found["units.d2.status"] == "not calibratable: 3 days"
    assert not [key for key in found if key.startswith("units.d2.configurations")]
    assert found["run.kappa"] == 0.3 and found["model"] == "m1"
    assert found["timeouts.certificates"] == 1
    for key, value in leaves(document).items():  # a row that exists carries the file's value
        if key in found:
            assert found[key] == value, key
    assert not texts(table)
    assert info[DIAL_OPTION]["file"] == "dial-choice.json"
    assert len(source.hashes) == 2  # the manifest and the file of the run
    # red: a figure read from another key, configuration, tier or unit (each differs in the made-up
    # file); the share of the ceiling taken from another ceiling; a tier B row for the driver loss.


# --- every label names the key its figure is stored under -------------------------
#
# The expected pairs below are written out here, from the meaning of each key of the writers, and
# not taken from the module's own lists: a label that moved to another key shows as a difference.

NONE = "none (the file lists none)"
COMMIT_ROW = ("run", "", "commit of the run", "manifest.commit", COMMIT)
T4_GROUPS = (
    "meets T4's hypotheses",
    "other days, no in-window departure",
    "other days, in-window departure",
)
DEV_SCOPES = {
    "order": "order check, every day",
    "t4_log": "T4 log, block product, every day",
    "t2": "T2's check, sampled days",
    "two_way": "two-way corner tests, sampled days",
}
DEV_CONFIGURATION = (
    ("rho", "rho"),
    ("two-way budget, kWh", "budget_kwh"),
    ("energy bounds", "energy_bounds"),
    ("car with no usable forecast", "uncertifiable_car"),
    ("kappa", "kappa"),
    ("departure", "departure_floor"),
    ("forecast of the departure", "forecast_departure"),
    ("forecast reference", "forecast_reference"),
    ("smallest cell of the forecast, cars", "forecast_min_cell"),
)
DEV_SAMPLE = (
    ("T2 sample, days", "size"),
    ("T2 sample of the pass mark, days", "pass_mark_size"),
    ("seed", "seed"),
    ("sample label", "label"),
)
DEV_VERDICT = (
    ("verdict", "verdict"),
    ("days the check should cover", "of_days"),
    ("days covered", "covered_days"),
    ("days with a timeout", "timeout_days"),
    ("days with another missing value", "other_missing_days"),
)
DEV_T4_GROUP = (
    ("days", "days"),
    ("days with a hindsight and a tree value", "days_with_hind_and_tree"),
    ("days with the tree below hindsight", "days_tree_below_hind"),
    ("mean gap hindsight minus tree, kWh", "mean_gap_hind_minus_tree_kwh"),
    ("days with a tree and an affine value", "days_with_tree_and_affine"),
    ("days with the affine value below the tree", "days_affine_below_tree"),
    ("mean gap tree minus affine, kWh", "mean_gap_tree_minus_affine_kwh"),
)
DEV_T2 = (
    ("sampled days certified", "certified_days"),
    ("days in E'", "days_in_e_prime"),
    ("share of sampled days in E'", "share_of_sampled_days_in_e_prime"),
    ("share of certified days in E'", "share_of_certified_days_in_e_prime"),
    ("steered cars on certified days", "steered_cars_on_certified_days"),
    ("share of steered cars in E'_i", "share_of_steered_cars_in_e_prime_i"),
    ("cars on sampled days", "cars_on_sampled_days"),
    ("cars left unsteered, no forecast", "cars_left_unsteered_no_forecast"),
    ("cars left unsteered, empty forecast set", "cars_left_unsteered_empty_forecast_set"),
    ("forecast rows checked", "forecast_rows_checked"),
    ("closed form against replay, mismatches", "replay_mismatches"),
    ("cars in E'_i checked", "cars_checked_in_e_prime_i"),
    ("cars in E'_i breaking a true row", "true_row_violations_in_e_prime_i"),
    (
        "cars outside E'_i breaking a true row (counted, not judged)",
        "cars_outside_e_prime_i_breaking_a_true_row",
    ),
    ("corner paths with a request missed", "paths_with_request_missed"),
)
DIAL_RUN = (
    ("label of the run", ("label",)),
    ("model", ("model",)),
    ("seeded splits per unit", ("run", "splits")),
    ("kappa", ("run", "kappa")),
    ("rung read", ("run", "rung")),
    ("ceiling of the gate run's first decision", ("run", "gate_ceiling")),
    ("plan threshold: the offer's share of the ceiling", ("run", "thresholds", "share_of_ceiling")),
    (
        "plan threshold: the oracle-departure offer over the offer",
        ("run", "thresholds", "oracle_to_offer_ratio"),
    ),
)
DIAL_UNIT = (
    ("status", "status"),
    ("counting days", "counting_days"),
    ("calibration days in each split", "calibration_days_used"),
    ("test days in each split", "test_days"),
)
DIAL_TIER = (
    ("mean offer", ("mean_offer_kwh_per_test_day",)),
    ("offer, p10 over splits", ("offer_p10",)),
    ("offer, p50 over splits", ("offer_p50",)),
    ("offer, p90 over splits", ("offer_p90",)),
    ("splits that offer", ("splits_that_offer",)),
    ("share of splits that offer", ("share_of_splits_that_offer",)),
    ("mean scale", ("mean_scale",)),
    ("mean test slot loss (offering splits)", ("mean_test_slot_loss",)),
    (
        "share of offering splits with the slot loss over its limit",
        ("slot_loss_over_limit", "share_of_splits_that_offer"),
    ),
    ("mean test driver loss (offering splits)", ("mean_test_driver_loss",)),
    (
        "share of offering splits with the driver loss over its limit",
        ("driver_loss_over_limit", "share_of_splits_that_offer"),  # tier A only
    ),
    (
        "offer as a share of the best-causal ceiling (offering splits only)",
        ("mean_share_of_best_causal_ceiling",),
    ),
)
DIAL_GATE = (
    ("mean offer over all splits", "mean_offer_kwh_per_test_day"),
    ("oracle-departure offer over all splits", "mean_oracle_offer_kwh_per_test_day"),
    ("splits", "splits"),
    ("oracle offer over mean offer", "oracle_to_offer_ratio"),
)
DIAL_CEILING = (
    ("splits with the ceiling", "splits_with_ceiling"),
    ("splits left out, ceiling missing", "splits_left_out_ceiling_missing"),
    ("mean ceiling", "mean_ceiling_kwh_per_test_day"),
    ("mean offer on those splits", "mean_offer_kwh_per_test_day_on_those_splits"),
    ("mean offer over mean ceiling", "share_of_ceiling"),
)
DIAL_CONFIGURATIONS = (("cfg-none", "no rows"), ("cfg-0.01", "eps 0.01"), ("cfg-0.02", "eps 0.02"))
DIAL_CEILINGS = ("hindsight-free-decision-hour", "best-causal")
# the figures the file's own rules force to be equal, by their keys (each set shares one value)
FORCED_DEV = {
    frozenset({"counting_days", "checks.order.of_days", "checks.t4_log.of_days"}),
    frozenset({"sample.size", "checks.t2.of_days", "checks.two_way.of_days"}),
    frozenset({"checks.t4_log.hypothesis_days", "checks.t4_log.groups.meets T4's hypotheses.days"}),
}
FORCED_DIAL = {
    frozenset({"run.dials.0", "dial_choice.d1.B.dial"}),
    frozenset({"run.dials.1", "dial_choice.d1.A.dial"}),
    frozenset(
        {
            "dial_choice.d1.A.mean_offer_kwh_per_test_day",
            "units.d1.configurations.cfg-0.02.tiers.A.mean_offer_kwh_per_test_day",
        }
    ),
    frozenset(
        {
            "dial_choice.d1.B.mean_offer_kwh_per_test_day",
            "units.d1.configurations.cfg-0.01.tiers.B.mean_offer_kwh_per_test_day",
        }
    ),
    frozenset({"units.d2.calibration_days_used", "units.d2.test_days"}),
}


def get(document, path):
    """The value at `path` (a name in a dictionary, a place from 0 in a list)."""
    here = document
    for name in path:
        here = here[name]
    return here


def entry(document, scope, reading, quantity, *path):
    """(scope, reading, quantity, key, the figure the table must show there)."""
    value = get(document, path)
    shown = NONE if value in ([], {}) else value
    return (scope, reading, quantity, ".".join(str(name) for name in path), shown)


def entries(document, scope, reading, quantity, *path):
    """The rows of a dictionary or list of the file: one per entry, or one that says none."""
    value = get(document, path)
    if isinstance(value, dict) and value:
        return [
            entry(document, scope, reading, f"{quantity}: {name}", *path, name)
            for name in sorted(value)
        ]
    if isinstance(value, list) and value:
        return [
            entry(document, scope, reading, f"{quantity} {k + 1}", *path, k)
            for k in range(len(value))
        ]
    return [entry(document, scope, reading, quantity, *path)]


def expected_dev_rows(document) -> list:
    rows = []
    for quantity, tail in DEV_CONFIGURATION:
        rows.append(entry(document, "configuration", "", quantity, "configuration", tail))
    for quantity, tail in DEV_SAMPLE:
        rows.append(entry(document, "sample", "", quantity, "sample", tail))
    rows += entries(document, "sample", "", "counting days", "sample", "counting_days")
    rows += entries(document, "sample", "", "sampled days", "sample", "sampled_days")
    rows.append(entry(document, "all days", "", "counting days", "counting_days"))
    for name, scope in DEV_SCOPES.items():
        for quantity, tail in DEV_VERDICT:
            rows.append(entry(document, scope, "", quantity, "checks", name, tail))
        rows += entries(document, scope, "", "failure line", "checks", name, "failure_lines")
        reasons = "could_not_run_reasons"
        rows += entries(document, scope, "", "could not run because", "checks", name, reasons)
    scope = DEV_SCOPES["t4_log"]
    t4 = ("checks", "t4_log")
    rows.append(entry(document, scope, "", "days meeting T4's hypotheses", *t4, "hypothesis_days"))
    both = "of them with V_hind and V_caus"
    rows.append(entry(document, scope, "", both, *t4, "hypothesis_days_checked"))
    for group in T4_GROUPS:
        for quantity, tail in DEV_T4_GROUP:
            rows.append(entry(document, scope, group, quantity, *t4, "groups", group, tail))
    above = "day with V_caus above V_aff"
    rows += entries(document, scope, "", above, *t4, "findings_tree_above_affine")
    scope = DEV_SCOPES["t2"]
    for quantity, tail in DEV_T2:
        rows.append(entry(document, scope, "", quantity, "checks", "t2", tail))
    by_rule = "days without a certificate by the rule"
    rule_key = "days_without_certificate_by_rule"
    rows += entries(document, scope, "", by_rule, "checks", "t2", rule_key)
    scope = DEV_SCOPES["two_way"]
    matched = "days on which both products' trees matched"
    matched_key = "days_both_products_matched"
    rows.append(entry(document, scope, "", matched, "checks", "two_way", matched_key))
    return rows


def expected_dial_rows(document) -> list:
    rows = []
    for quantity, path in DIAL_RUN:
        rows.append(entry(document, "run", "", quantity, *path))
    rows += entries(document, "run", "", "dial", "run", "dials")
    rows += entries(document, "run", "", "tier", "run", "tiers")
    rows += entries(document, "run", "", "ceiling", "run", "ceilings")
    rows += entries(document, "run", "", "timeouts", "timeouts")
    rows += entries(document, "run", "", "configuration that failed", "failed_configurations")
    for unit in ("d1", "d2"):  # d2 is too short to split: the file holds no more for it
        for quantity, tail in DIAL_UNIT:
            rows.append(entry(document, unit, "", quantity, "units", unit, tail))
    for tier in ("A", "B"):
        chosen = f"tier {tier}, dial chosen"
        dial = "dial with the largest mean offer"
        rows.append(entry(document, "d1", chosen, dial, "dial_choice", "d1", tier, "dial"))
        offer, there = "its mean offer", ("dial_choice", "d1", tier, "mean_offer_kwh_per_test_day")
        rows.append(entry(document, "d1", chosen, offer, *there))
    for name, where in DIAL_CONFIGURATIONS:
        base = ("units", "d1", "configurations", name)
        for tier in ("A", "B"):
            for quantity, tail in DIAL_TIER:
                if tail[0] == "driver_loss_over_limit" and tier != "A":
                    continue
                reading = f"tier {tier}, {where}"
                rows.append(entry(document, "d1", reading, quantity, *base, "tiers", tier, *tail))
        gate = f"gate reading (tier A), {where}"
        for quantity, tail in DIAL_GATE:
            rows.append(entry(document, "d1", gate, quantity, *base, "gate_reading", tail))
        for ceiling in DIAL_CEILINGS:
            for quantity, tail in DIAL_CEILING:
                path = (*base, "gate_reading", "ceilings", ceiling, tail)
                rows.append(entry(document, "d1", f"{gate}, ceiling {ceiling}", quantity, *path))
    return rows


def assert_pairs(table, expected) -> None:
    """The table has exactly the expected labels, each with the key and the figure expected."""
    shown = {(row[1], row[2], row[3]): row for row in table.rows}
    assert len(shown) == len(table.rows)  # no label twice
    assert set(shown) == {(scope, reading, quantity) for scope, reading, quantity, _, _ in expected}
    for scope, reading, quantity, key, figure in expected:
        row = shown[(scope, reading, quantity)]
        assert row[5] == key, (scope, reading, quantity, "is stored under", row[5], "not", key)
        assert row[4] == figure, (scope, reading, quantity, "shows", row[4], "not", figure)


def repeated(table) -> set:
    """The sets of keys whose figures are equal numbers."""
    by_value: dict = {}
    for row in table.rows:
        if isinstance(row[4], int | float) and not isinstance(row[4], bool):
            by_value.setdefault(row[4], set()).add(row[5])
    return {frozenset(keys) for keys in by_value.values() if len(keys) > 1}


def test_every_label_of_the_development_day_table_names_the_key_of_its_figure(tmp_path):
    document = distinct_dev_days_document()
    table, _, _ = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    assert not texts(table)  # the made-up file contradicts nothing
    assert repeated(table) == FORCED_DEV  # every other figure has a value of its own
    assert_pairs(table, [*expected_dev_rows(document), COMMIT_ROW])
    # Traced by hand: with the texts "days in E'" and "sampled days certified" swapped in the
    # module's list of T2 keys, the row "days in E'" has the key checks.t2.certified_days (700),
    # the expected one is checks.t2.days_in_e_prime (280): red on the key and on the figure. The
    # same for the two shares of E' (0.35 over the 800 sampled days and 0.4 over the 700
    # certified ones, which the swap exchanges) and for the share of steered cars in E'_i.
    # red: any two quantities, scopes or readings of this table swapped between their keys, a
    # row left out or added, or a figure shown that is not the one under the key.


def test_every_label_of_the_dial_choice_table_names_the_key_of_its_figure(tmp_path):
    document = distinct_dial_choice_document()
    table, _, _ = table_of(tmp_path, DIAL_OPTION, dial_run(tmp_path, document))
    assert not texts(table)
    assert repeated(table) == FORCED_DIAL
    assert_pairs(table, [*expected_dial_rows(document), COMMIT_ROW])
    # Traced by hand: with the texts "mean ceiling" and "mean offer on those splits" swapped in the
    # module's list of ceiling figures, the row "mean ceiling" holds the key
    # ...ceilings.<name>.mean_offer_kwh_per_test_day_on_those_splits, not ...mean_ceiling_kwh_per_
    # test_day: red on the key (the two values also differ). The same for the swap of the two
    # ceilings' names in a reading, or of the tier A and tier B rows of one configuration.
    # red: any two quantities, tiers, configurations, ceilings or units of this table swapped
    # between their keys, a row left out or added, or a figure that is not the one under the key.


def test_a_relative_folder_is_read_under_the_top_folder(tmp_path):
    folder = dev_run(tmp_path)
    table, _, _ = table_of(tmp_path, DEV_OPTION, Path(folder.relative_to(tmp_path)))
    assert cells(table)["checks.t2.days_in_e_prime"] == 4
    # red: a relative folder opened from the current folder.


def test_an_option_that_is_not_given_makes_no_table(tmp_path):
    source = Source(tmp_path, None)
    assert wp16_devdays.build(source, {}) == ([], {})
    assert wp16_devdays.build(source, {DEV_OPTION: None, DIAL_OPTION: None}) == ([], {})
    assert source.hashes == {}
    # red: a table made, or a file read, without the option.


# --- a figure that is not there, and a figure that is contradicted ---------------


def test_a_missing_or_null_figure_is_no_figure_and_never_a_zero(tmp_path):
    document = dev_days_document()
    del document["checks"]["t2"]["days_in_e_prime"]
    document["checks"]["t2"]["cars_on_sampled_days"] = None
    document["sample"]["seed"] = "2026"
    table, _, _ = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    found = cells(table)
    assert found["checks.t2.days_in_e_prime"] == Missing(
        "no figure: the file holds no checks.t2.days_in_e_prime"
    )
    assert found["checks.t2.cars_on_sampled_days"] == Missing(
        "no figure: the file writes null for checks.t2.cars_on_sampled_days"
    )
    wrong = found["sample.seed"]
    assert isinstance(wrong, Missing) and wrong.text.startswith("unreadable: sample.seed is '2026'")
    assert found["checks.t2.share_of_sampled_days_in_e_prime"] == 0.4  # others are untouched
    assert wp16_run.any_unreadable([table])  # the wrong type is unreadable; the other two are not
    document["sample"]["seed"] = 2026
    again, _, _ = table_of(tmp_path / "again", DEV_OPTION, dev_run(tmp_path / "again", document))
    assert not wp16_run.any_unreadable([again])  # "no figure" is not a failure
    # red: a missing or null figure written as 0 or as a blank; "no figure" counted as unreadable;
    # a figure of the wrong type copied.


def test_counts_that_contradict_one_another_are_unreadable(tmp_path):
    document = dev_days_document()
    document["checks"]["t2"]["days_in_e_prime"] = 9  # above the 8 certified days
    table, _, _ = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    found = cells(table)
    cell = found["checks.t2.days_in_e_prime"]
    assert isinstance(cell, Missing) and cell.text.startswith("unreadable: ")
    assert "checks.t2.certified_days 8" in cell.text
    assert found["checks.t2.certified_days"] == 8  # a cell not contradicted stays a figure
    # the share of sampled days is no longer 9 over 10 either
    assert isinstance(found["checks.t2.share_of_sampled_days_in_e_prime"], Missing)
    assert wp16_run.any_unreadable([table])
    # red: counts compared in the wrong direction or not at all.


def test_a_share_that_is_not_its_count_over_its_denominator_is_unreadable(tmp_path):
    document = dev_days_document()
    document["checks"]["t2"]["share_of_certified_days_in_e_prime"] = 0.75  # 4 of 8 is 0.5
    table, _, _ = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    found = cells(table)
    assert isinstance(found["checks.t2.share_of_certified_days_in_e_prime"], Missing)
    assert found["checks.t2.share_of_sampled_days_in_e_prime"] == 0.4
    # a share the writer rounded to six digits is inside the tolerance
    document["checks"]["t2"]["share_of_certified_days_in_e_prime"] = 0.5 + 4e-7
    ok, _, _ = table_of(tmp_path / "ok", DEV_OPTION, dev_run(tmp_path / "ok", document))
    assert cells(ok)["checks.t2.share_of_certified_days_in_e_prime"] == 0.5 + 4e-7
    # red: a share compared against nothing, or with no tolerance for the six-digit rounding.


def test_a_verdict_that_does_not_follow_from_its_lines_and_cover_is_unreadable(tmp_path):
    document = dev_days_document()
    document["checks"]["order"]["failure_lines"] = ["a line"]  # a pass with a failure line
    document["checks"]["t4_log"]["covered_days"] = 49  # a pass that did not cover every day
    table, _, _ = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    found = cells(table)
    assert isinstance(found["checks.order.verdict"], Missing)
    assert isinstance(found["checks.t4_log.verdict"], Missing)
    assert found["checks.t2.verdict"] == "passed"
    assert found["checks.order.failure_lines.0"] == "a line"
    # red: a verdict copied without being set against the failure lines and the cover.


def test_the_groups_of_the_t4_log_must_add_up_to_the_days(tmp_path):
    document = dev_days_document()
    document["checks"]["t4_log"]["groups"]["meets T4's hypotheses"]["days"] = 13
    table, _, _ = table_of(tmp_path, DEV_OPTION, dev_run(tmp_path, document))
    found = cells(table)
    assert isinstance(found["checks.t4_log.of_days"], Missing)
    assert found["checks.t4_log.groups.meets T4's hypotheses.days"] == 13  # the figure is copied
    # red: no comparison of the groups' days with the days of the check.


def test_a_dial_whose_offer_is_not_its_configurations_is_unreadable(tmp_path):
    document = dial_choice_document()
    document["dial_choice"]["d1"]["A"]["mean_offer_kwh_per_test_day"] = 7.5
    table, _, _ = table_of(tmp_path, DIAL_OPTION, dial_run(tmp_path, document))
    found = cells(table)
    cell = found["dial_choice.d1.A.mean_offer_kwh_per_test_day"]
    assert isinstance(cell, Missing) and cell.text.startswith("unreadable: ")
    assert found["dial_choice.d1.B.mean_offer_kwh_per_test_day"] == 9.5
    assert wp16_run.any_unreadable([table])
    # red: the chosen dial's offer taken on trust.


def test_a_gate_share_a_spread_and_a_split_count_that_do_not_fit_are_unreadable(tmp_path):
    document = dial_choice_document()
    cfg = document["units"]["d1"]["configurations"]["cfg-0.01"]
    cfg["gate_reading"]["ceilings"]["best-causal"]["share_of_ceiling"] = 0.9
    cfg["tiers"]["A"]["offer_p10"] = 99.0  # above p50
    cfg["tiers"]["B"]["splits_that_offer"] = 201  # above the 200 splits
    table, _, _ = table_of(tmp_path, DIAL_OPTION, dial_run(tmp_path, document))
    found = cells(table)
    base = "units.d1.configurations.cfg-0.01"
    for key in (
        f"{base}.gate_reading.ceilings.best-causal.share_of_ceiling",
        f"{base}.tiers.A.offer_p10",
        f"{base}.tiers.B.splits_that_offer",
        f"{base}.tiers.B.share_of_splits_that_offer",
    ):
        assert isinstance(found[key], Missing) and found[key].text.startswith("unreadable: "), key
    assert found[f"{base}.tiers.A.offer_p50"] == 4.5  # a neighbour is a figure
    # red: any of the three comparisons left out.


def test_a_failed_configuration_and_a_failed_gate_reading_are_failed_rows(tmp_path):
    document = dial_choice_document()
    units = document["units"]["d1"]["configurations"]
    units["cfg-0.02"] = {"driver_risk": 0.02, "failed": "ValueError: boom"}
    units["cfg-0.01"]["gate_reading"] = {"failed": "KeyError: x", "oracle_partner": "oracle"}
    table, _, _ = table_of(tmp_path, DIAL_OPTION, dial_run(tmp_path, document))
    bad = texts(table)
    assert "failed: d1 | cfg-0.02: ValueError: boom" in bad
    assert "failed: d1 | cfg-0.01: gate reading: KeyError: x" in bad
    assert wp16_run.any_unreadable([table])  # exit status 1
    # red: a failed configuration skipped, or written as no figure.


def test_a_what_if_run_is_refused(tmp_path):
    document = dial_choice_document()
    document["what_if"] = {"what": "what-if"}
    table, info, _ = table_of(tmp_path, DIAL_OPTION, dial_run(tmp_path, document))
    (row,) = table.rows
    assert isinstance(row[4], Missing) and row[4].text.startswith("unreadable: ")
    assert "what_if" in row[4].text and info == {}
    # red: a what-if run read as a run of the plan.


# --- the run folder is read through its manifest ---------------------------------


def only_row(table) -> str:
    (row,) = table.rows
    assert isinstance(row[4], Missing)
    return row[4].text


def test_a_run_that_did_not_exit_0_is_not_read_as_passed(tmp_path):
    folder = dev_run(tmp_path, exit_code=1)  # the file is there, a check failed
    table, info, source = table_of(tmp_path, DEV_OPTION, folder)
    assert only_row(table) == f"failed: {folder.name} (it ended with exit status 1)"
    assert info == {} and source.hashes == {}  # nothing of it was taken
    assert wp16_run.any_unreadable([table])
    # red: the exit code not looked at, so the file's figures are read.


@pytest.mark.parametrize(
    ("change", "start"),
    [
        ({"timed_out": True}, "failed: "),
        ({"interrupted": "SIGTERM"}, "failed: "),
        ({"error": "could not start"}, "failed: "),
        ({"exit_code": None}, "unreadable: "),
        ({"commit": 5}, "unreadable: "),  # a commit that is not text (none at all is no figure)
        ({"command": ["python", "-m", "fleetreserve.dial_choice"]}, "unreadable: "),
        ({"command": "dev_day_checks"}, "unreadable: "),
        ({"outputs": {}}, "unreadable: "),
    ],
)
def test_a_manifest_that_does_not_show_a_clean_run_of_the_command_is_not_read(
    tmp_path, change, start
):
    folder = dev_run(tmp_path)
    edit_manifest(folder, **change)
    table, info, _ = table_of(tmp_path, DEV_OPTION, folder)
    assert only_row(table).startswith(start) and info == {}
    # red: a time limit, an interruption, a start error, a missing exit code, a commit that is not
    # text, another command or an unlisted file not looked at in the manifest.


def test_a_file_that_is_not_the_one_the_run_wrote_is_unreadable(tmp_path):
    folder = dev_run(tmp_path)
    (folder / "dev-day-checks.json").write_text(json.dumps(dev_days_document(), indent=2))
    table, _, source = table_of(tmp_path, DEV_OPTION, folder)
    assert "it is not the file its run wrote" in only_row(table)
    assert source.hashes == {}
    # red: the hash of the file not compared with the manifest's.


def test_a_file_that_is_not_json_or_not_an_object_is_unreadable(tmp_path):
    for k, content in enumerate((b"{not json", b"[1, 2]")):
        folder = make_run(tmp_path / str(k), DEV_LABEL, "dev-day-checks.json", content, DEV_COMMAND)
        table, _, _ = table_of(tmp_path / str(k), DEV_OPTION, folder)
        assert only_row(table).startswith("unreadable: ")
    # red: a file that does not parse taken as an empty summary.


def test_a_folder_that_is_not_a_plain_run_folder_under_the_top_is_unreadable(tmp_path):
    top = tmp_path / "top"
    folder = dev_run(top)
    other = tmp_path / "elsewhere"
    other.mkdir()
    for given in (
        other,  # not under the top folder
        folder.parent / ".." / folder.parent.name / folder.name,  # a ".." part
        folder / "..",  # a ".." part
        folder.parent,  # not a run folder name
    ):
        table, info, _ = table_of(top, DEV_OPTION, given)
        assert only_row(table).startswith("unreadable: ") and info == {}
    link = top / "runs.noindex" / "20261123T130000Z-linked"
    link.symlink_to(folder)  # a link is never followed
    table, _, _ = table_of(top, DEV_OPTION, link)
    assert only_row(table).startswith("unreadable: ")
    # red: a path opened without the plain reader (a ".." or a link followed).


def test_a_run_that_could_not_run_is_labelled_so_and_a_failed_one_is_not(tmp_path):
    folder = dev_run(tmp_path, exit_code=2)  # a check that could not run exits 2
    table, info, source = table_of(tmp_path, DEV_OPTION, folder)
    assert only_row(table) == f"could not run: {folder.name} (it ended with exit status 2)"
    assert info == {} and source.hashes == {}
    assert wp16_run.any_unreadable([table])  # the exit status of wp16_run stays 1
    folder = dial_run(tmp_path / "dial", exit_code=2)
    table, _, _ = table_of(tmp_path / "dial", DIAL_OPTION, folder)
    assert only_row(table).startswith("could not run: ")
    other = dev_run(tmp_path / "other", exit_code=3)  # any other non-zero code is a failure
    table, _, _ = table_of(tmp_path / "other", DEV_OPTION, other)
    assert only_row(table) == f"failed: {other.name} (it ended with exit status 3)"
    made_up_top(tmp_path / "top", "pass")
    out = tmp_path / "out"
    argv = ["--root", str(tmp_path / "top"), "--out", str(out)]
    could_not_run = dev_run(tmp_path / "top", exit_code=2)
    assert wp16_run.main([*argv, "--dev-days-run", str(could_not_run)]) == 1
    assert "could not run: " in (out / "dev-days-checks.csv").read_text()
    # red: exit status 2 labelled "failed:" as any other non-zero code; "could not run:" missing
    # from the starts that give exit status 1 (`wp16_run.BAD_START`), so that it ends with 0.


def test_a_run_folder_under_a_data_folder_is_refused_and_a_copy_elsewhere_is_read(tmp_path):
    made_up_top(tmp_path, "pass")
    folder = dev_run(tmp_path)
    inside = tmp_path / "data" / "copies" / folder.name  # a perfect copy of a run folder
    shutil.copytree(folder, inside)
    for given in (inside, Path("data") / "copies" / folder.name):  # absolute, then relative
        table, info, source = table_of(tmp_path, DEV_OPTION, given)
        words = only_row(table)
        assert words.startswith("unreadable: ") and "data/" in words
        assert info == {} and source.hashes == {}
    deeper = tmp_path / "reference" / "copies" / "data" / folder.name  # any part named data
    shutil.copytree(folder, deeper)
    table, _, _ = table_of(tmp_path, DEV_OPTION, deeper)
    assert only_row(table).startswith("unreadable: ")
    near = tmp_path / "reference" / "copies" / folder.name  # a copy elsewhere is read as before
    shutil.copytree(folder, near)
    table, _, _ = table_of(tmp_path, DEV_OPTION, near)
    assert cells(table)["checks.t2.days_in_e_prime"] == 4
    out = tmp_path / "out"
    argv = ["--root", str(tmp_path), "--out", str(out), "--dev-days-run", str(inside)]
    assert wp16_run.main(argv) == 1
    assert "unreadable: " in (out / "dev-days-checks.csv").read_text()
    # red: no look at the parts of the folder's path (the copy under data/ would be read as a run:
    # its table then holds many rows, not one); or a check that refuses more than a part named
    # data (the copy under reference/ would be refused).


# --- the commit of the run -------------------------------------------------------


def commit_cells(table) -> list:
    return [row for row in table.rows if row[3] == "commit of the run"]


def test_each_table_carries_the_commit_its_manifest_gives_copied_as_written(tmp_path):
    written = "0123abc-as-written"  # not a 40-digit hash: the text is copied, not rewritten
    for k, (option, make) in enumerate(((DEV_OPTION, dev_run), (DIAL_OPTION, dial_run))):
        top = tmp_path / str(k)
        folder = make(top)
        edit_manifest(folder, commit=written)
        table, info, _ = table_of(top, option, folder)
        (row,) = commit_cells(table)
        assert row[0] == "development" and row[4] == written and row[5] == "manifest.commit"
        assert info[option]["commit"] == written
        assert f"at commit {written}" in table.notes[-1]
    # red: the commit left out of a table, taken from another source (the made-up default is 40
    # letters a) or rewritten; the manifest of wp16_run not carrying it.


@pytest.mark.parametrize("how", ["missing", "null", "empty"])
def test_a_manifest_with_no_commit_gives_no_figure_for_it_and_is_still_read(tmp_path, how):
    made_up_top(tmp_path, "pass")
    folder = dev_run(tmp_path)
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    if how == "missing":
        del manifest["commit"]
    else:
        manifest["commit"] = None if how == "null" else ""
    path.write_text(json.dumps(manifest))
    table, info, _ = table_of(tmp_path, DEV_OPTION, folder)
    (row,) = commit_cells(table)
    assert row[4] == Missing("no figure: no commit in the manifest")
    assert cells(table)["checks.t2.days_in_e_prime"] == 4  # the run is read all the same
    assert not wp16_run.any_unreadable([table])  # "no figure" is not a failure
    assert info[DEV_OPTION]["commit"] == "no figure: no commit in the manifest"
    assert "with no commit in its manifest" in table.notes[-1]
    out = tmp_path / "out"
    argv = ["--root", str(tmp_path), "--out", str(out), "--dev-days-run", str(folder)]
    assert wp16_run.main(argv) == 0
    # red: a manifest with no commit refused, or its commit shown as a blank or a made-up value.


# --- where "development" is said -------------------------------------------------


def test_the_drive_banner_is_the_first_note_and_the_development_note_follows(tmp_path):
    made_up_top(tmp_path, "pass")
    out = tmp_path / "out"
    dev, dial = dev_run(tmp_path), dial_run(tmp_path)
    options = ["--dev-days-run", str(dev), "--dial-choice-run", str(dial)]
    assert wp16_run.main(["--root", str(tmp_path), "--out", str(out), *options]) == 0
    for name, start in (
        ("dev-days-checks", "Development figures read from one finished run of dev_day_checks"),
        ("dev-dial-choice", "Development figures read from one finished run of dial_choice"),
    ):
        page = (out / f"{name}.md").read_text()
        notes = [line for line in page.splitlines() if line.startswith("- ")]
        assert notes[0].startswith("- Drive ") and notes[1].startswith(f"- {start}")
        assert "Development figures" in page.splitlines()[0]  # the title
    assert (out / "dev-days-checks.csv").read_text().count("development,") > 50  # first column
    # red: the development note put ahead of the banner, or left out of the notes (the module text
    # says the banner comes first and the note follows).


def test_a_run_of_another_command_is_unreadable(tmp_path):
    table, _, _ = table_of(tmp_path, DIAL_OPTION, dev_run(tmp_path))  # the wrong run for the option
    assert "does not run python -m fleetreserve.dial_choice" in only_row(table)
    # red: the option's table read from a run of another command.


# --- the two options of wp16_run -------------------------------------------------


def test_without_the_options_the_files_and_the_manifest_are_unchanged_and_with_them_one_table_each(
    tmp_path,
):
    made_up_top(tmp_path, "pass")
    dev, dial = dev_run(tmp_path), dial_run(tmp_path)
    plain, both = tmp_path / "plain", tmp_path / "both"
    argv = ["--root", str(tmp_path)]
    assert wp16_run.main([*argv, "--out", str(plain)]) == 0
    options = ["--dev-days-run", str(dev), "--dial-choice-run", str(dial)]
    assert wp16_run.main([*argv, "--out", str(both), *options]) == 0
    before = {p.name for p in plain.iterdir()}
    after = {p.name for p in both.iterdir()}
    assert not before & NEW_FILES and after == before | NEW_FILES
    for name in before - {"outcomes.md", "wp16-manifest.json"}:  # nothing else moves
        assert (plain / name).read_bytes() == (both / name).read_bytes(), name
    outcomes = (both / "outcomes.md").read_text()
    assert outcomes.startswith((plain / "outcomes.md").read_text())  # the new tables come last
    assert "## Development figures: the checks on the development days" in outcomes
    assert "## Development figures: the dial choice" in outcomes
    plain_manifest = json.loads((plain / "wp16-manifest.json").read_text())
    manifest = json.loads((both / "wp16-manifest.json").read_text())
    assert "development_runs" not in plain_manifest
    runs = manifest["development_runs"]
    assert runs[DEV_OPTION]["run"] == str(dev.relative_to(tmp_path))
    assert runs[DIAL_OPTION]["file_sha256"] == sha((dial / "dial-choice.json").read_bytes())
    for name in NEW_FILES:
        assert manifest["outputs"][name] == sha((both / name).read_bytes())
    for name, digest in plain_manifest["outputs"].items():
        if name != "outcomes.md":
            assert manifest["outputs"][name] == digest
    read = manifest["summaries_read"]
    assert f"{runs[DEV_OPTION]['run']}/dev-day-checks.json" in read
    assert f"{runs[DIAL_OPTION]['run']}/manifest.json" in read
    assert (both / "dev-days-checks.csv").read_text().count("development,") > 50
    # red: an existing file, its hash or the exit status changed by the option's presence; a table
    # written without its option or left out of the files and the manifest with it; a development
    # run read but not named in the manifest.


def test_one_option_gives_one_table_only(tmp_path):
    made_up_top(tmp_path, "pass")
    out = tmp_path / "out"
    argv = ["--root", str(tmp_path), "--out", str(out), "--dev-days-run", str(dev_run(tmp_path))]
    assert wp16_run.main(argv) == 0
    names = {p.name for p in out.iterdir()}
    assert {"dev-days-checks.csv", "dev-days-checks.md"} <= names
    assert not names & {"dev-dial-choice.csv", "dev-dial-choice.md"}
    manifest = json.loads((out / "wp16-manifest.json").read_text())
    assert set(manifest["development_runs"]) == {DEV_OPTION}
    # red: both tables written for one option.


def test_a_wrong_hash_a_failed_run_or_a_contradiction_gives_exit_status_1(tmp_path):
    made_up_top(tmp_path, "pass")
    base = ["--root", str(tmp_path)]
    folder = dev_run(tmp_path)
    (folder / "dev-day-checks.json").write_text("{}")  # not the file the manifest hashes
    out = tmp_path / "wrong-hash"
    assert wp16_run.main([*base, "--out", str(out), "--dev-days-run", str(folder)]) == 1
    page = (out / "dev-days-checks.csv").read_text()
    assert "it is not the file its run wrote" in page and "checks.t2" not in page
    failed, dial = tmp_path / "failed", dial_run(tmp_path, exit_code=1)
    assert wp16_run.main([*base, "--out", str(failed), "--dial-choice-run", str(dial)]) == 1
    assert "it ended with exit status 1" in (failed / "dev-dial-choice.csv").read_text()
    document = dev_days_document()
    document["checks"]["t2"]["days_in_e_prime"] = 9
    contradicted, second = tmp_path / "contradicted", dev_run(tmp_path / "second", document)
    assert wp16_run.main([*base, "--out", str(contradicted), "--dev-days-run", str(second)]) == 1
    # red: a table of one of these written with exit status 0.


def test_a_run_folder_that_does_not_exist_gives_a_table_that_says_so_and_exit_status_1(tmp_path):
    made_up_top(tmp_path, "pass")
    out = tmp_path / "out"
    missing = tmp_path / "runs.noindex" / "20261124T000000Z-wp5-dev-days"
    argv = ["--root", str(tmp_path), "--out", str(out), "--dev-days-run", str(missing)]
    assert wp16_run.main(argv) == 1
    page = (out / "dev-days-checks.csv").read_text()
    assert "unreadable: 20261124T000000Z-wp5-dev-days/manifest.json" in page
    # red: a folder that cannot be read ending the command or being left out in silence.

"""The main reading of when a departure is seen (plan section 5, "Rungs"; matrix and summary).

Decided 12 November 2026: a rung sees a car gone from the start of its rounded-down departure hour
("rounded-hour", main); seen only at the start of the following hour ("hour-start") is reported
beside it, never pooled.

1. In the committed `gate/matrix.json` every configuration of the family "main" sees departures
   as "rounded-hour"; the configurations that see them at the start of the following hour are in
   the family "beside"; every configuration name of the four-level matrix is unchanged and the
   two levels 0.4 and 0.5 add names of the same pattern; the dial family
   and the other alternates keep their family; the `about` text says which reading is main.
2. The summary lists the reading of `departure_seen` among the decided matters and no longer among
   the open ones; the dial and the tier are open.
3. The main-family tables, outcomes and agreement line use the rounded-hour reading only; the
   hour-start results are in their own block, labelled as reported beside the main reading, with
   their own offer rows and outcome lines, and they do not enter the main agreement line.
4. The tools that read a run (`tools/gate_t5_check.py`) pick no configuration by family or reading.
"""

import json
import re

import pytest
from gate_fake import TOP

from fleetreserve import gate_config, gate_summary

MATRIX = gate_config.load(TOP / "gate" / "matrix.json")
KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)  # plan section 2's grid
DIALS = ("0", "0.005", "0.01", "0.02", "0.03", "0.05", "0.075", "0.1")
BESIDE_BLOCK = (
    "departures seen at the start of the following hour (reported beside the main reading)"
)


def _names(prefix: str) -> list[str]:
    return [f"{prefix}{k}-{d}" for k in KAPPAS for d in ("empirical", "oracle")]


def test_the_main_family_is_rounded_hour_and_the_hour_start_family_is_beside():
    by_family = {}
    for c in MATRIX.configurations:
        by_family.setdefault(c.family, []).append(c)
    assert set(by_family) == {"main", "beside", "alternate"}
    assert {c.departure_seen for c in by_family["main"]} == {"rounded-hour"}
    assert [c.name for c in by_family["main"]] == _names("main-rounded-hour-k")
    assert {c.departure_seen for c in by_family["beside"]} == {"hour-start"}
    assert [c.name for c in by_family["beside"]] == _names("main-k")
    for c in by_family["main"] + by_family["beside"]:
        assert (c.energy_bounds, c.driver_risk, c.departure_floor) == ("known", None, "as-planned")


def test_every_configuration_name_is_unchanged_and_the_alternates_keep_their_family():
    expected = (
        _names("main-k")
        + _names("main-rounded-hour-k")
        + [
            f"alt-{what}-k0.1-{d}"
            for what in ("energy-empirical", "day-abstains", "floored")
            for d in ("empirical", "oracle")
        ]
        + [f"main-rounded-hour-k0.3-eps{e}-{d}" for e in DIALS for d in ("empirical", "oracle")]
    )
    assert sorted(c.name for c in MATRIX.configurations) == sorted(expected)
    for c in MATRIX.configurations:
        if c.name.startswith(("alt-", "main-rounded-hour-k0.3-eps")):
            assert c.family == "alternate"


def test_the_about_text_says_which_reading_is_main():
    about = MATRIX_ABOUT
    assert "rounded-down departure hour" in about
    assert "reported beside" in about and "start of the following hour" in about


MATRIX_ABOUT = json.loads((TOP / "gate" / "matrix.json").read_text())["about"]


def test_the_family_names_are_the_three_the_summary_reads():
    assert gate_config.FAMILIES == ("main", "beside", "alternate")


# --- the summary ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def text() -> str:
    return "\n".join(gate_summary.summary_lines(_document(), MATRIX))


def _decision(go: bool) -> dict:
    return {
        "ceiling": "best-causal",
        "ratio_rule_on": "all_units",
        "go": go,
        "n_pass_share": 0,
        "n_pass_both": 0,
        "units": {},
    }


def _tables(main_go: bool, beside_go: bool) -> dict:
    by_pair = {}
    for c in MATRIX.configurations:
        if c.departure != "empirical" or c.family == "alternate" or c.driver_risk is not None:
            continue
        go = main_go if c.family == "main" else beside_go
        by_pair[c.name] = {
            "family": c.family,
            "kappa": c.kappa,
            "departure_seen": c.departure_seen,
            "decisions": [_decision(go)],
        }
    selected = {
        "rounded-hour": {"configuration_by_unit": {}, "decisions": [_decision(main_go)]},
        "hour-start": {"configuration_by_unit": {}, "decisions": [_decision(beside_go)]},
    }
    return {"by_pair": by_pair, "kappa_selected_per_unit": selected}


def _document(main_go: bool = False, beside_go: bool = True) -> dict:
    return {
        "trial": False,
        "partial": False,
        "t5": {"breaks_a": 0, "breaks_b": 0, "day_paths_checked": 7},
        "timeouts": {},
        "failed_configurations": [],
        "calibration_rule": {"n_calibration": 100, "calibration_half_below": 200},
        "units": {},
        "configurations": {},
        "gate": _tables(main_go, beside_go),
        "gate_tier_b": _tables(main_go, beside_go),
    }


def _sentence(text: str, start: str) -> str:
    line = next(x for x in text.splitlines() if x.startswith(start))
    return line


def test_the_reading_is_decided_and_no_longer_open(text):
    decided = _sentence(text, "Decided by the project owner on 16 Nov 2026")
    assert "Decided by the project owner on 12 Nov 2026" in decided
    assert "rounded-down departure hour" in decided and "reported beside it" in decided
    still_open = _sentence(text, "Still open")
    assert "sees a car gone" not in still_open and "departure_seen" not in still_open
    assert "which reading of the 1.5 rule" in still_open
    assert "which tier decides the gate" not in still_open
    assert "the gate is read at tier A" in still_open
    assert "the dial of the driver-risk rows is chosen by a rule fixed in advance" in still_open
    assert "kappa and its selection" not in still_open


def test_the_main_agreement_line_counts_the_main_reading_only(text):
    # 6 kappas of the main family and the selection of the main reading: 7 decisions, all no go;
    # the beside family's 7 decisions are go and would make the line say "differ" if counted
    assert text.count("All 7 main-family outcomes agree: no go;") == 2  # tier A and tier B
    assert "main-family outcomes differ" not in text
    marks = [k for k, x in enumerate(text.splitlines()) if x.startswith("Gate outcomes")]
    assert len(marks) == 2  # one block per tier; the beside block is not a gate-outcomes block


def test_the_hour_start_results_are_in_their_own_labelled_block(text):
    lines = text.splitlines()
    heads = [k for k, x in enumerate(lines) if BESIDE_BLOCK in x]
    assert heads, "no block for departures seen at the start of the following hour"
    block = lines[heads[0] :]
    end = next((k for k, x in enumerate(block) if x.startswith("Alternates")), len(block))
    inside = block[:end]
    outcomes = [x for x in inside if x.lstrip().startswith("PROVISIONAL")]
    assert outcomes and all("departure seen hour-start" in x for x in outcomes)
    assert all(": go " in x for x in outcomes if "ceiling" in x)
    before = lines[: heads[0]]
    assert not [x for x in before if "departure seen hour-start" in x]
    assert [x for x in before if "departure seen rounded-hour" in x]
    assert "never pooled" in lines[heads[0]]


def test_the_gate_outcome_lines_of_the_main_block_are_rounded_hour(text):
    lines = text.splitlines()
    first = next(k for k, x in enumerate(lines) if x.startswith("Gate outcomes"))
    block = lines[first : lines.index("", first)]
    outcome = [x for x in block if x.lstrip().startswith("PROVISIONAL")]
    assert outcome and all("departure seen rounded-hour" in x for x in outcome)
    assert re.search(r"All 7 main-family outcomes agree", block[-1])


def _rung(offer: float) -> dict:
    test = {
        "days": 10,
        "offering_days": 10,
        "failing_days": 0,
        "mean_offer_kwh_per_day": offer,
        "share_of_ceiling": {
            "hindsight-free-decision-hour": {"ratio_of_sums": offer / 10},
            "best-causal": {"ratio_of_sums": offer / 10},
        },
    }
    found = {"beta_hat": 1.0, "reason": "", "test": test}
    return {**found, "tier_b": found}


def test_the_offer_rows_of_each_reading_are_in_their_own_table():
    document = _document()
    for name, offer in (("main-rounded-hour-k0.1", 2.0), ("main-k0.1", 3.0)):
        for kind, value in (("empirical", offer), ("oracle", 2 * offer)):
            unit = {u: {"rungs": {MATRIX.gate_rung: _rung(value)}} for u in MATRIX.gate_units}
            document["configurations"][f"{name}-{kind}"] = {"units": unit}
    lines = gate_summary.summary_lines(document, MATRIX)
    head = next(k for k, x in enumerate(lines) if BESIDE_BLOCK in x)
    rows = [
        (k, x) for k, x in enumerate(lines) if x.startswith(("  rounded-hour ", "  hour-start "))
    ]
    assert rows
    for k, x in rows:
        if x.startswith("  rounded-hour "):
            assert k < head and " 2.000 " in x
        else:
            assert k > head and " 3.000 " in x
    assert len(rows) == 2 * len(MATRIX.tiers) * len(MATRIX.gate_units)


# --- the tools -----------------------------------------------------------------------------------


def test_the_t5_tool_picks_no_configuration_by_family_or_reading():
    source = (TOP / "tools" / "gate_t5_check.py").read_text()
    for word in ("family", "departure_seen", "rounded-hour", "hour-start", "configurations"):
        assert word not in source

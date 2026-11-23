"""The gate decisions and the wording of the gate summary (`gate_summary`), on made-up numbers.

1. A unit that is not calibratable (or has no result) counts as not passing: with one such unit,
   "go" needs both other units to pass; under the reading "all units" of the 1.5 rule it is never
   "go"; with two such units it is never "go".
2. The summary names what the project owner decided, each with the date the record gives (rate
   and energy known on 16 Nov 2026, a car with no usable forecast left unsteered on 18 Nov 2026),
   says that "counts as not passing" is a reading with no recorded decision, names what is still
   open and that the plan fixes kappa, and keeps the PROVISIONAL prefix on every outcome line.
3. The last line of the outcome block says whether all outcomes of the main family agree, computed
   from the decisions of the results document: all "no go", all "go", mixed (with counts) and an
   empty main family (no agreement is claimed). Alternates do not enter it.
"""

from gate_fake import TOP

from fleetreserve import gate_config, gate_summary

MATRIX = gate_config.load(TOP / "gate" / "matrix.json")
UNITS = MATRIX.gate_units
CEILING = "best-causal"


def _data(share: float, days: int = 4) -> dict:
    """Test data with a calibrated offer of `share` of a ceiling of 10 kWh per day."""
    return {
        "values": [share * 10.0] * days,
        "ceilings": {c: [10.0] * days for c in gate_config.CEILINGS},
        "beta": 1.0,
    }


def _pair(cal_share: float, oracle_share: float) -> tuple[dict, dict]:
    return _data(cal_share), _data(oracle_share)


def _missing() -> tuple[dict, dict]:
    reason = {"reason": "not calibratable: 19 days"}
    return reason, reason


def _decisions(pairs: dict) -> dict[str, bool]:
    return {
        rule: gate_summary.decide(pairs, MATRIX, CEILING, rule)["go"]
        for rule in gate_summary.gate_stat.RATIO_RULES
    }


def test_a_unit_that_is_not_calibratable_counts_as_not_passing():
    passing = _pair(0.12, 0.30)  # 12% of the ceiling, oracle 2.5 times the calibrated offer
    failing = _pair(0.05, 0.30)  # 5% of the ceiling
    pairs = {UNITS[0]: _missing(), UNITS[1]: passing, UNITS[2]: passing}
    assert _decisions(pairs) == {
        "all_units": False,  # a unit without a number cannot show the 1.5 rule
        "all_share_passing_units": True,
        "required_units_both": True,
    }
    row = gate_summary.decide(pairs, MATRIX, CEILING, "required_units_both")["units"][UNITS[0]]
    assert row["passes_share"] is False and row["passes_ratio"] is False
    one_fails = {UNITS[0]: _missing(), UNITS[1]: passing, UNITS[2]: failing}
    assert _decisions(one_fails) == {rule: False for rule in gate_summary.gate_stat.RATIO_RULES}
    both_fail = {UNITS[0]: _missing(), UNITS[1]: failing, UNITS[2]: failing}
    assert _decisions(both_fail) == {rule: False for rule in gate_summary.gate_stat.RATIO_RULES}
    two_missing = {UNITS[0]: _missing(), UNITS[1]: _missing(), UNITS[2]: passing}
    assert _decisions(two_missing) == {rule: False for rule in gate_summary.gate_stat.RATIO_RULES}


def test_a_unit_without_a_number_is_worded_as_not_passing():
    pairs = {UNITS[0]: _missing(), UNITS[1]: _pair(0.12, 0.30), UNITS[2]: _pair(0.12, 0.30)}
    decision = gate_summary.decide(pairs, MATRIX, CEILING, "required_units_both")
    text = gate_summary._decision_text(decision)
    assert f"{UNITS[0]} counts as not passing (not calibratable: 19 days)" in text
    assert "not decided" not in text
    assert text.startswith(f"ceiling {CEILING}, 1.5 rule on required_units_both: go ")


def _decision(go: bool) -> dict:
    return {
        "ceiling": CEILING,
        "ratio_rule_on": "all_units",
        "go": go,
        "n_pass_share": 0,
        "n_pass_both": 0,
        "units": {},
    }


def _document(main_pairs: list[bool], selected: list[bool], alternates: list[bool] = ()) -> dict:
    by_pair = {
        f"main-{i}": {
            "family": "main",
            "kappa": 0.1,
            "departure_seen": "hour-start",
            "decisions": [_decision(go)],
        }
        for i, go in enumerate(main_pairs)
    }
    by_pair.update(
        {
            f"alt-{i}": {"family": "alternate", "decisions": [_decision(go)]}
            for i, go in enumerate(alternates)
        }
    )
    return {
        "gate": {
            "by_pair": by_pair,
            "kappa_selected_per_unit": {
                "hour-start": {
                    "configuration_by_unit": {},
                    "decisions": [_decision(go) for go in selected],
                }
            },
        }
    }


VARIED = "kappa, the ceiling, the reading of the 1.5 rule and the per-unit kappa selection"
HELD = (
    "The hourly shape rule, the departure floor and which cars count for driver loss are held at "
    "one value in the main family"
)


def test_all_no_go_agree():
    line = gate_summary.agreement_line(_document([False, False, False], [False, False]))
    assert line == (
        "All 5 main-family outcomes agree: no go; varying "
        f"{VARIED} does not change the outcome. {HELD}, so this says nothing about them."
    )


def test_all_go_agree():
    line = gate_summary.agreement_line(_document([True, True], [True]))
    assert line == (
        "All 3 main-family outcomes agree: go; varying "
        f"{VARIED} does not change the outcome. {HELD}, so this says nothing about them."
    )


def test_a_mixed_family_says_it_differs_with_counts():
    line = gate_summary.agreement_line(_document([True, False, False], [False, True]))
    assert line == (
        "The main-family outcomes differ: 2 go, 3 no go; the outcome depends on at least one of "
        f"{VARIED}."
    )


def test_a_single_odd_outcome_among_many_is_not_hidden():
    line = gate_summary.agreement_line(_document([False] * 30, [False] * 29 + [True]))
    assert line.startswith("The main-family outcomes differ: 1 go, 59 no go")


def test_the_agreement_line_does_not_claim_all_open_decisions():
    for line in (
        gate_summary.agreement_line(_document([False], [False])),
        gate_summary.agreement_line(_document([True, False], [False])),
    ):
        assert "open decisions" not in line
        assert "above" not in line


def test_an_empty_main_family_claims_no_agreement():
    line = gate_summary.agreement_line(_document([], []))
    assert "outcomes agree" not in line
    assert line == "No main-family outcome was computed; nothing can be said about agreement."


def test_alternates_do_not_enter_the_agreement():
    line = gate_summary.agreement_line(_document([False, False], [False], alternates=[True]))
    assert line.startswith("All 3 main-family outcomes agree: no go")


def _summary(document_gate: dict) -> list[str]:
    document = {
        "trial": False,
        "partial": False,
        "t5": {"breaks_a": 0, "breaks_b": 0, "day_paths_checked": 7},
        "timeouts": {},
        "failed_configurations": [],
        "calibration_rule": {"n_calibration": 100, "calibration_half_below": 200},
        "units": {},
        "configurations": {},
        "gate": document_gate,
    }
    return gate_summary.summary_lines(document, MATRIX)


def test_the_summary_names_what_is_decided_and_what_is_open():
    text = "\n".join(_summary(_document([False, False], [False])["gate"]))
    assert "Decided by the project owner on 16 Nov 2026: each car's rate and energy need" in text
    assert "Decided by the project owner on 18 Nov 2026: a car with no usable forecast" in text
    assert "11 Nov 2026" not in text
    for decided in (
        "each car's rate and energy need are taken as known",
        "a car with no usable forecast is left unsteered",
        "A unit that is not calibratable counts as not passing (a reading taken by this report; "
        'no decision on it is recorded), so with one such unit "go" needs both other units',
    ):
        assert decided in text
    assert "Still open, so every outcome line below is PROVISIONAL" in text
    line = next(x for x in text.splitlines() if x.startswith("Still open"))
    still_open, settled = line.split("Fixed by the plan:")
    for open_one in (
        "which reading of the 1.5 rule (the plan does not say on which units it must hold",
        "the departure floor (no decision on it is recorded; the main family uses as-planned)",
        "which cars count for driver loss (no decision on it is recorded; the main family uses "
        "steered-only)",
    ):
        assert open_one in still_open
    for settled_one in (
        "which ceiling is the denominator",
        "which tier decides the gate",
        "the dial of the driver-risk rows",
        "the hourly shape rule",
    ):
        assert settled_one not in still_open
    for fixed in (
        "the ceiling in the denominator is the hindsight ceiling (plan section 13, WP6; shown: "
        "the hindsight ceiling and the best-causal ceiling)",
        "the gate is read at tier A (plan section 13, WP6; shown, never pooled: "
        f"{gate_config.TIER_LABELS['A']} and {gate_config.TIER_LABELS['B']})",
        "the dial of the driver-risk rows is chosen by a rule fixed in advance (plan section 6; "
        "the main family uses none; alternates at the dials 0, 0.005, 0.01, 0.02, 0.03, 0.05, "
        "0.075, 0.1 are shown)",
        "the hourly shape rule is the one WP5 sets (plan section 2; the main family uses "
        "largest-smallest-width)",
        "kappa is the largest calibrated development offer (plan section 7;",
    ):
        assert fixed in settled
    assert "kappa and its selection" not in text
    assert "open decisions 2" not in text and "open decision 1," not in text


def test_the_agreement_line_ends_the_outcome_block():
    lines = _summary(_document([False, True], [False])["gate"])
    marker = lines.index(next(x for x in lines if x.startswith("Gate outcomes")))
    block = lines[marker : lines.index("", marker)]
    assert block[-1].startswith("The main-family outcomes differ: 1 go, 2 no go")
    outcome_lines = [x for x in block if "ceiling " in x]
    assert outcome_lines and all(x.lstrip().startswith("PROVISIONAL") for x in outcome_lines)


def test_the_alternates_are_comparisons_and_the_side_not_chosen_is_marked_per_line():
    pairs = {u: _pair(0.12, 0.30) for u in UNITS}
    decision = gate_summary.decide(pairs, MATRIX, CEILING, "all_units")
    names = (
        "alt-energy-empirical-k0.1-empirical",
        "alt-day-abstains-k0.1-empirical",
        "alt-floored-k0.1-empirical",
    )
    assert {c.name for c in MATRIX.configurations} >= set(names)
    gate = {
        "by_pair": {
            name: {"family": "alternate", "departure_seen": "hour-start", "decisions": [decision]}
            for name in names
        },
        "kappa_selected_per_unit": {},
    }
    lines = _summary(gate)
    start = next(i for i, x in enumerate(lines) if x.startswith("Alternates"))
    assert "comparisons" in lines[start]
    found = {name: next(x for x in lines[start:] if f"{name} (" in x) for name in names}
    assert all(x.lstrip().startswith("PROVISIONAL") for x in found.values())
    assert "side not chosen" in found[names[0]] and "side not chosen" in found[names[1]]
    assert "side not chosen" not in found[names[2]]


def test_the_dial_sentence_follows_the_matrix_of_the_run_from_the_tag():
    sealed = gate_config.load(TOP / "gate" / "sealed-matrix" / "lgbm_tail.json")
    text = gate_summary._open_text(sealed, selected=False)
    assert "the main family uses none, 0.02)" in text
    assert "alternates at the dials" not in text


def test_no_sentence_of_the_open_text_has_an_empty_value_for_any_committed_matrix():
    """A matrix with no main family (the forecast-energy matrices) names its configurations,
    not an empty main family, in every clause."""
    paths = sorted((TOP / "gate").rglob("*.json"))
    matrices = []
    for path in paths:
        try:
            matrices.append((path, gate_config.load(path)))
        except (ValueError, KeyError, TypeError):
            continue  # not a run matrix (the training files)
    assert len(matrices) >= 15
    for path, matrix in matrices:
        for selected in (True, False):
            text = gate_summary._open_text(matrix, selected=selected)
            for empty in ("uses ;", "uses )", "uses ,", "use ;", "use )"):
                assert empty not in text, (path.name, empty)
    energy = gate_config.load(TOP / "gate" / "sealed-matrix" / "lgbm_tail-forecast-energy.json")
    text = gate_summary._open_text(energy, selected=False)
    assert "this matrix has no main family; its alternates use the dials none, 0.02" in text
    assert "its configurations use" in text
    assert "the main family uses" not in text

"""The committed settings file and the README section of the forecast trial `energy-2`, against the
code, the plan and the first energy trial's settings. Written to be run through the test slot. Each
test says which edit turns it red.

1. The committed settings are the ones the code and the plan give: approved to run, level 0.2 and a
   level of the plan's grid with a dial family in the committed matrix, rate and room level 0.05
   (`forecast_empirical.CAUTIOUS_LEVEL`, the lowest level of the grid), the level of the needed
   energy of arms 4 and 5 the same 0.05, need level of arms 2 and 3 equal to the level, 200 splits,
   the model `lgbm_tail`, the five arms in order (the five values of
   `gate_config.MATRIX_ENERGY_BOUNDS`), the bars 8% and 30% and the loss limits 0.10 and 0.05
   (`crc`), every unit of the committed matrix named and no held-out name. The guard of a run
   compares with the committed file on disk and not with git; this test is what pins that the
   committed file is the one the code and the plan give. Red if a value in the file changes without
   the code, or the code without the file.
2. Every value this trial shares with the first energy trial's settings equals it (model, level,
   rate and room level, need level, splits, units, the bars, the loss limits, the judged tier, the
   dial family, the rung, the calibration, the words). Red if one differs.
3. The reading block fixes what the brief fixed: the one judged comparison (arm 5 against arm 2),
   the information comparison (arm 4 against arm 2, no word), the loss limits read from arm 5, the
   not-judged rule for an arm 1 offer of 0, the one-line words that name the setting, what a gain,
   a loss and no measurable difference would and would not show, the untried list, the re-run rule.
4. The bars rest on the prior-looks entry found by its words.
5. The README has the section after the first energy trial's, headed so that the scan test finds
   it, with the one command (under `run_record`, with `--max-seconds 172800`), every output file,
   the label, the approval, what a gain, a loss and no measurable difference would and would not
   show, the untried settings, the sentence that the trial cannot speak for the Dutch or Korean
   sites, that the bars are an analogy, that the guard compares with the file on disk and that a run
   starts from a clean commit, and the duration as a guess with its basis; no absolute path, no
   "next step", and no sentence that says "the cautious bounds help" without the setting. Red if a
   command, an output or a sentence is dropped. The command is in the fence of every other trial's
   command, with `--first-run` and a placeholder for the first trial's run folder (the first
   trial's test counts the commands of its own section only, which ends at the
   next line that begins `## `), and the section's label sentence names the date of the run.
6. The new modules name no loader of the sealed data and import none; the command writes only under
   the run folder.
"""

import ast
import json
from pathlib import Path

import pytest
from gate_fake import TOP

from fleetreserve import (
    crc,
    dial_choice,
    driver_risk,
    forecast_energy,
    forecast_energy_2,
    forecast_energy_2_reading,
    gate_config,
    sealed_route,
)
from fleetreserve.forecast_empirical import CAUTIOUS_LEVEL, KAPPAS

SETTINGS = json.loads(Path(forecast_energy_2.SETTINGS_FILE).read_text())
FIRST = json.loads(Path(forecast_energy.SETTINGS_FILE).read_text())
README = (TOP / "trials-after-sealed-run" / "README.md").read_text()
SEED_ENTRY = "The neural model's seed test"
HEADING = (
    "## Trial forecast-energy-2: all three energy bounds cautious, and the needed energy alone"
)


def test_the_committed_settings_are_the_ones_the_code_and_the_plan_give():
    assert SETTINGS["approved_to_run"] is True
    assert SETTINGS["level"] == 0.2 and SETTINGS["level"] in driver_risk.LEVELS
    assert SETTINGS["rate_cap_level"] == CAUTIOUS_LEVEL == min(KAPPAS) == 0.05
    assert SETTINGS["cautious_need_level"] == CAUTIOUS_LEVEL
    assert SETTINGS["need_level"] == SETTINGS["level"]
    assert SETTINGS["splits"] == 200 and SETTINGS["model"] == "lgbm_tail"
    assert (
        SETTINGS["arms"]
        == list(forecast_energy_2.ARMS)
        == list(gate_config.MATRIX_ENERGY_BOUNDS)
        == ["known", "empirical", "cautious", "cautious-need", "cautious-all"]
    )
    assert forecast_energy_2.bars_of(SETTINGS) == {
        "nrel-from-may-2019": 0.08,
        "norway-development-pooled": 0.30,
    }
    assert forecast_energy_2.limits_of(SETTINGS) == {
        "slot": crc.ALPHA_SLOT,
        "driver": crc.ALPHA_DRIVER,
    }
    assert SETTINGS["reading"]["no_other_sites"] == forecast_energy_2_reading.NO_OTHER_SITES
    matrix = gate_config.load(forecast_energy_2.MATRIX)
    for unit in matrix.gate_units:
        assert unit in SETTINGS["units"]
    assert set(forecast_energy_2.bars_of(SETTINGS)) <= set(matrix.gate_units)
    _, (plain, dials) = dial_choice.family_at(
        matrix, dial_choice.find_family(matrix), SETTINGS["level"]
    )
    assert plain.kappa == SETTINGS["level"] and dials
    held_out = set(sealed_route.HELD_OUT_UNITS)
    text = Path(forecast_energy_2.SETTINGS_FILE).read_text()
    assert not {name for name in held_out if name in text}
    assert forecast_energy_2.REQUIRED_SETTINGS == (
        "approved_to_run",
        "level",
        "rate_cap_level",
        "need_level",
        "cautious_need_level",
        "splits",
        "model",
        "arms",
        "reading",
    )
    for key in (
        "sources",
        "arm_definitions",
        "units",
        "calibration",
        "pairing",
        "question",
        "first_run_comparison",
    ):
        assert SETTINGS[key]
    assert forecast_energy_2.load_settings() == SETTINGS
    assert "/Users" not in text and "/private" not in text and "reference/" not in text


def test_the_approval_quotes_the_owners_decision_in_neutral_words():
    about = SETTINGS["about"]
    assert (
        "APPROVED TO RUN by the owner's decision of 25 November 2026: 'Yes, build and run'" in about
    )
    assert "cannot grant it" in about and "the first energy trial's figures stand" in about.lower()


def test_what_this_trial_shares_with_the_first_energy_trial_is_equal():
    for key in (
        "model",
        "level",
        "rate_cap_level",
        "need_level",
        "splits",
        "judged_tier",
        "fit_units",
        "dial_family",
        "rung",
        "calibration",
    ):
        assert SETTINGS[key] == FIRST[key], key
    for unit in gate_config.load(forecast_energy_2.MATRIX).gate_units:
        assert unit in SETTINGS["units"] and unit in FIRST["units"]
    for key in ("bars", "unit_not_judged", "tier_B", "no_other_sites", "judged_units"):
        assert SETTINGS["reading"][key] == FIRST["reading"][key], key
    assert {k: SETTINGS["reading"]["loss_limits"][k] for k in ("slot", "driver")} == {
        k: FIRST["reading"]["loss_limits"][k] for k in ("slot", "driver")
    }
    first_arms = FIRST["arms"]
    assert SETTINGS["arms"][: len(first_arms)] == first_arms  # arms 1 to 3 are the first trial's
    for arm in first_arms:
        assert arm in SETTINGS["arm_definitions"]


def test_the_arm_definitions_name_the_levels_of_the_code():
    arms = SETTINGS["arm_definitions"]
    assert set(arms) == set(forecast_energy_2.ARMS)
    assert "0.8" in arms["empirical"] and "true" in arms["known"]
    assert "0.05" in arms["cautious"] and "0.8" in arms["cautious"]
    assert "0.05" in arms["cautious-need"] and "0.95" in arms["cautious-need"]
    assert "information only" in arms["cautious-need"]
    for part in ("0.05", "0.95", "all three"):
        assert part in arms["cautious-all"]
    assert "judged" in arms["cautious-all"]


# --- 3. the reading block -------------------------------------------------------------------------


def test_the_reading_block_states_one_judged_comparison_and_one_information_comparison():
    block = SETTINGS["reading"]
    assert block["judged_comparison"]["trial_arm"] == "cautious-all"
    assert block["judged_comparison"]["main_arm"] == "empirical"
    assert block["information_comparison"]["trial_arm"] == "cautious-need"
    assert block["information_comparison"]["main_arm"] == "empirical"
    assert "read by no rule" in block["information_comparison"]["role"]
    assert "no gain, loss or no-difference word" in block["information_comparison"]["role"]
    assert "ONE judged comparison" in block["rule"] and "arm 5" in block["rule"]
    assert "arm 5's tier A summary" in block["loss_limits"]["source"]
    assert "arm 4 minus arm 2" in block["information_comparison"]["role"]


def test_the_words_the_one_lines_and_the_not_judged_rule_are_those_of_the_code():
    block = SETTINGS["reading"]
    assert set(block["words"]) == {
        "shows a gain",
        "shows a loss",
        "no measurable difference",
        "not judged",
    }
    assert "arm 1's mean offer is 0" in block["words"]["not judged"]
    assert "no bar can be defined" in block["words"]["not judged"]
    assert set(block["one_line"]) == {
        forecast_energy_2_reading.HELPS,
        forecast_energy_2_reading.HURTS,
        "otherwise",
        "no judged unit",
    }
    assert block["one_line"]["otherwise"] == forecast_energy_2_reading.MIXED
    assert block["unit_not_judged"].startswith("nrel-before-may-2019 has no bar")
    assert "NOT JUDGED" in block["rule"]
    assert "no arm, level, bar, limit or rule is added" in block["no_further_arm"]


UNTRIED = ("levels between 0.05 and 0.2", "other departure levels", "another model")


def test_what_a_result_would_and_would_not_show_is_claimed_no_wider_than_the_setting_tried():
    shows = SETTINGS["reading"]["what_a_result_shows"]
    for part in ("A gain would show", "A loss would show", "No measurable difference would show"):
        assert part in shows
    loss = shows[shows.index("A loss would show") : shows.index("No measurable difference")]
    for item in (*UNTRIED, "energy rows in the certificate"):
        assert item in loss, item
    assert "one setting" in loss and "would not show that other settings would do the same" in loss
    assert "would not show that no change of levels helps" in loss
    assert "next step" not in json.dumps(SETTINGS) and "recovers the gap" not in loss
    assert "would not show that the held-out zero came from the energy bounds" in shows
    assert "not that the bounds are equal and not that no setting helps" in shows
    assert "arm 4 with arm 2 is information" in shows
    assert "the cautious bounds help" not in json.dumps(SETTINGS)
    assert "the cautious bounds hurt" not in json.dumps(SETTINGS)


def test_the_rerun_rule_is_exact_equality_and_a_fault_is_not_chosen_between():
    rule = SETTINGS["first_run_comparison"]
    assert rule["tolerance"] == "none"
    for part in (
        "forecast-energy.json",
        "exact equality",
        "FAULT",
        "not chosen between",
        "not replaced",
        "--first-run",
        "not an override",
    ):
        assert part in rule["rule"], part
    assert rule["figures"].startswith("per judged unit and per arm of arms 1 to 3")
    assert forecast_energy_2.FIRST_RUN_ARMS == ("known", "empirical", "cautious")
    assert forecast_energy_2.FIRST_RESULTS_NAME == "forecast-energy.json"


# --- 4. the bars ----------------------------------------------------------------------------------


def prior_looks_entry(opening: str) -> str:
    """The one entry (a bullet line) of `docs/prior-looks.md` that opens with `opening`, found by
    its words and never by its line number."""
    lines = (TOP / "docs" / "prior-looks.md").read_text().splitlines()
    found = [line for line in lines if line.strip().startswith(f"- {opening}")]
    assert len(found) == 1, opening
    return found[0]


def test_the_bars_rest_on_the_prior_looks_entry_found_by_its_words():
    entry = prior_looks_entry(SEED_ENTRY)
    assert "kappa 0.3" in entry
    assert "about 1.5 kWh (8%) at workplaces and 0.5 kWh (30%) at Norway" in entry
    for found in SETTINGS["reading"]["bars"].values():
        assert "docs/prior-looks.md" in found["source"] and SEED_ENTRY in found["source"]
        assert "analogy" in found["source"] and "not this trial's measured noise" in found["source"]


# --- 5. the README --------------------------------------------------------------------------------

COMMAND = (
    "python -m fleetreserve.run_record --name forecast-energy-2 --long --max-seconds 172800 -- "
    "uv run --frozen --group learn python -m fleetreserve.forecast_energy_2 --workers 6 "
    "--first-run <run folder of the first energy trial>"
)


def raw_section() -> str:
    """The README section of the trial, from its heading to the next heading of its level."""
    start = README.index(HEADING)
    end = README.find("\n## ", start + 1)
    return README[start : len(README) if end == -1 else end]


def section() -> str:
    """The section with every run of whitespace made one space."""
    return " ".join(raw_section().split())


def test_the_section_is_after_the_first_energy_trials_and_headed_so_the_scan_finds_it():
    assert README.index("## Trial forecast-energy: more cautious energy bounds") < README.index(
        HEADING
    )
    assert README.count("\n## Trial forecast-energy-2") == 1
    assert HEADING.startswith("## Trial forecast-energy-2")
    text = section()
    assert text.count("python -m fleetreserve.run_record") == 1 and COMMAND in text
    assert text.count("``` python -m") == 1  # the one command, in the fence of every other trial
    for output in (
        "forecast-energy-2.json",
        "forecast-energy-2-summary.txt",
        "forecast-energy-2-timing.json",
        "forecast-energy-2-forecasts.csv",
        "forecast-energy-2-<arm>-dial-choice.json",
        "forecast-energy-2-<arm>-split-offers.json",
        "made-after-sealed-run.txt",
        "not-the-trial.txt",
    ):
        assert output in text, output
    assert "--max-seconds 172800" in text and "--not-the-trial" in text
    assert "--first-run" in text and "forecast-energy.json" in text
    for arm in ("`known`", "`empirical`", "`cautious`", "`cautious-need`", "`cautious-all`"):
        assert arm in text
    assert "/Users" not in README and "/private" not in README


def test_the_committed_command_compares_arms_1_to_3_with_the_first_run():
    """The design fixes that arms 1 to 3 are compared with the first run, and the comparison is
    left out unless `--first-run` is given: the one committed command carries the option with a
    placeholder for the first trial's run folder (no real folder name, no absolute path), and the
    section says the trial is run with it. The program keeps the option optional. Red if the option
    or the sentence is removed."""
    text = section()
    assert text.count("--first-run <run folder of the first energy trial>") == 1
    command = text[text.index("``` python -m") :]
    command = command[: command.index(" ```")]
    assert command.endswith("--workers 6 --first-run <run folder of the first energy trial>")
    assert "The trial is run with `--first-run`, so that arms 1 to 3 are compared" in text
    assert "still works and says that the first run was not compared" in text


def test_the_section_says_where_a_first_run_fault_is_shown_and_what_it_leaves_alone():
    """Read `reading` first, then `first_run_check`; `first_run_faults` of `reading` holds the list
    (empty when equal, null when not compared); the summary's first lines say the count, equal or
    not compared; a fault moves no figure, reading word or exit status. Red if a part is dropped."""
    text = section()
    assert "read `reading` first, then `first_run_check`" in text
    assert "`first_run_faults` of `reading`" in text
    assert "an empty list when the first run was compared and every figure is equal" in text
    assert "null when the first run was not compared" in text
    assert (
        'the count of FAULT lines, "First run compared: equal", or "First run not compared"' in text
    )
    assert "A fault changes no figure, no reading word and not the exit status" in text


def test_the_section_says_the_label_names_the_date_the_run_started():
    """The label sentence carries the UTC date of the run's start, from the clock; the README does
    not say that it names 24 November 2026. Red if the old sentence is back or the new one is
    dropped."""
    text = section()
    assert "names the UTC date on which the run started" in text
    assert "read from the clock once at the start of the run" in text
    assert "The label text names 24 November 2026" not in text
    assert "The sealed run ended on 24 November 2026." in text


def test_the_section_says_what_a_gain_a_loss_and_no_difference_would_and_would_not_show():
    text = section()
    assert "APPROVED TO RUN by the owner's decision of 25 November 2026" in text
    assert "approved_to_run: true" in text and "cannot grant it" in text
    assert "What a result would and would not show:" in text
    for part in ("- A gain:", "- A loss:", "- No measurable difference:"):
        assert part in text
    assert "It would not show that the held-out zero came from the energy bounds" in text
    assert (
        "No Dutch or Korean site is a development unit, so the trial cannot speak for them" in text
    )
    assert (
        "The trial was run once, on 25 November 2026, at commit 4ebe5fe: start 12:43:30 UTC, "
        "end 13:06:01 UTC, exit 0." in text
    )
    assert "has not been run" not in text
    assert "calibration, the certificate, the loss limits" in text
    loss = text[text.index("- A loss:") : text.index("- No measurable difference:")]
    assert (
        "tries one setting" in loss
        and "does not show that other settings would do the same" in loss
    )
    assert "does not show that no change of levels helps" in loss
    for item in (*UNTRIED, "energy rows in the certificate"):
        assert item in loss, item


def test_the_section_states_the_untried_the_analogy_the_guard_and_the_duration_as_a_guess():
    text = section()
    assert "analogy and not this trial's measured noise" in text
    assert (
        "compares the values of a run with the committed settings file on disk, not with git"
        in text
    )
    assert "a run starts from a clean commit" in text
    assert "Duration, a guess and not a measurement: about 25 minutes, between 15 and 45" in text
    assert "five measurement runs and one fit" in text and "took about 15 minutes" in text
    assert "not checked again here" in text


def test_the_section_states_the_bars_the_limits_and_the_one_line_words_naming_the_setting():
    text = section()
    assert "8% at the workplace unit, 30% at the pooled Norwegian unit" in text
    assert "0.10" in text and "0.05" in text
    for words in (
        "shows a gain",
        "shows a loss",
        "no measurable difference",
        forecast_energy_2_reading.HELPS,
        forecast_energy_2_reading.HURTS,
        "ONE judged comparison",
        "arm 5 minus arm 2",
        "information only",
        "not judged",
        "FAULT",
    ):
        assert words in text, words
    assert "the cautious bounds help" not in text and "the cautious bounds hurt" not in text
    assert "next step" not in text


def test_the_second_section_does_not_change_the_first_trials_section():
    """The first trial's section keeps its own command and words, and is cut off at its own end."""
    start = README.index("## Trial forecast-energy: more cautious energy bounds")
    first = README[start : README.index(HEADING)]
    assert first.count("python -m fleetreserve.run_record") == 1
    assert "forecast-energy-2" not in first and "cautious-need" not in first


# --- 6. no sealed data, nothing written outside the run folder -----------------------------------


@pytest.mark.parametrize("module", [forecast_energy_2, forecast_energy_2_reading])
def test_the_new_modules_name_no_loader_of_the_sealed_data(module):
    text = Path(module.__file__).read_text()
    assert "heldout" not in text.lower().replace("held-out", "")
    tree = ast.parse(text)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules += [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            modules += [a.name for a in node.names]
    assert modules and not [m for m in modules if "held" in m.lower() or "sealed_route" in m]


def test_the_command_writes_only_under_the_run_folder():
    tree = ast.parse(Path(forecast_energy_2.__file__).read_text())
    writes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("write_text", "write_bytes", "open"):
                writes.append(ast.unparse(node.func.value))  # the path written
            elif node.func.attr in ("write_table", "write_note"):
                writes.append(" ".join(ast.unparse(a) for a in node.args))
    assert len(writes) >= 8
    for found in writes:
        assert "folder" in found, found  # every write is on a path built from the run folder


def test_the_command_opens_the_first_run_folder_for_reading_only():
    """The one read of the first trial's folder is `read_first_run`: `read_bytes` of the one file
    `FIRST_RESULTS_NAME`, and nothing else in the module names a first-run path."""
    text = Path(forecast_energy_2.__file__).read_text()
    tree = ast.parse(text)
    reads = [
        ast.unparse(node.func.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("read_bytes", "read_text")
    ]
    assert any("FIRST_RESULTS_NAME" in found or found == "path" for found in reads)
    assert text.count("FIRST_RESULTS_NAME") >= 2
    assert "listdir" not in text and "iterdir" not in text and "glob" not in text

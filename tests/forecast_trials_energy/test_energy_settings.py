"""The committed settings file and the README section of the forecast trial `energy`, against the
code and the plan. Written to be run through the test slot. Each test says which edit turns it red.

1. The committed settings are the ones the code and the plan give: approved to run, level 0.2 and a
   level of the plan's grid with a dial family in the committed matrix, rate and room level 0.05
   (`forecast_empirical.CAUTIOUS_LEVEL`, the lowest level of the grid), need level equal to the
   level, 200 splits, the model `lgbm_tail`, the three arms in order, the bars 8% and 30% and the
   loss limits 0.10 and 0.05 (`crc`), every unit of the committed matrix named and no held-out name.
   Red if a value in the file changes without the code, or the code without the file.
2. The bars rest on the prior-looks entry found by its words. Red if the entry or its figures
   change.
3. The README has the section after the no-encoder one, with the one command (under `run_record`,
   with `--max-seconds 172800`), every output file, the label, the approval, what a gain, a loss and
   no measurable difference would and would not show, and the sentence that the trial cannot speak
   for the Dutch or Korean sites; no absolute path. The section ends at the next line that begins
   `## `, so the one command counted is the trial's own and a later trial's section is not read.
   Red if a command, an output or a sentence is dropped, or a second command is in the section.
4. The new modules name no loader of the sealed data and import none.
5. Nothing the guarantee rests on is touched by the trial: its modules write only under the run
   folder named by `FLEETRESERVE_RUN_DIR` and import no calibration, certificate or plan module for
   writing. Red if a module of the trial opens a file for writing outside the run folder.
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
    forecast_energy_reading,
    gate_config,
    sealed_route,
)
from fleetreserve.forecast_empirical import CAUTIOUS_LEVEL, KAPPAS

SETTINGS = json.loads(Path(forecast_energy.SETTINGS_FILE).read_text())
README = (TOP / "trials-after-sealed-run" / "README.md").read_text()
SEED_ENTRY = "The neural model's seed test"


def test_the_committed_settings_are_the_ones_the_code_and_the_plan_give():
    assert SETTINGS["approved_to_run"] is True
    assert SETTINGS["level"] == 0.2 and SETTINGS["level"] in driver_risk.LEVELS
    assert SETTINGS["rate_cap_level"] == CAUTIOUS_LEVEL == min(KAPPAS) == 0.05
    assert SETTINGS["need_level"] == SETTINGS["level"]
    assert SETTINGS["splits"] == 200 and SETTINGS["model"] == "lgbm_tail"
    assert SETTINGS["arms"] == list(forecast_energy.ARMS) == ["known", "empirical", "cautious"]
    assert forecast_energy.bars_of(SETTINGS) == {
        "nrel-from-may-2019": 0.08,
        "norway-development-pooled": 0.30,
    }
    assert forecast_energy.limits_of(SETTINGS) == {
        "slot": crc.ALPHA_SLOT,
        "driver": crc.ALPHA_DRIVER,
    }
    assert SETTINGS["reading"]["no_other_sites"] == forecast_energy_reading.NO_OTHER_SITES
    matrix = gate_config.load(forecast_energy.MATRIX)
    for unit in matrix.gate_units:
        assert unit in SETTINGS["units"]
    assert set(forecast_energy.bars_of(SETTINGS)) <= set(matrix.gate_units)
    _, (plain, dials) = dial_choice.family_at(
        matrix, dial_choice.find_family(matrix), SETTINGS["level"]
    )
    assert plain.kappa == SETTINGS["level"] and dials
    held_out = set(sealed_route.HELD_OUT_UNITS)
    assert not {
        name for name in held_out if name in Path(forecast_energy.SETTINGS_FILE).read_text()
    }
    assert forecast_energy.REQUIRED_SETTINGS == (
        "approved_to_run",
        "level",
        "rate_cap_level",
        "need_level",
        "splits",
        "model",
        "arms",
        "reading",
    )
    for key in ("sources", "arm_definitions", "units", "calibration", "pairing", "question"):
        assert SETTINGS[key]
    assert forecast_energy.load_settings() == SETTINGS


def test_the_arm_definitions_name_the_levels_of_the_code():
    arms = SETTINGS["arm_definitions"]
    assert set(arms) == set(forecast_energy.ARMS)
    assert "0.05" in arms["cautious"] and "0.8" in arms["cautious"] and "0.8" in arms["empirical"]
    assert "true" in arms["known"]


def test_the_reading_block_states_the_words_the_one_lines_and_what_a_result_shows():
    block = SETTINGS["reading"]
    assert set(block["words"]) == {"shows a gain", "shows a loss", "no measurable difference"}
    assert set(block["one_line"]) == {
        forecast_energy_reading.HELPS,
        forecast_energy_reading.HURTS,
        "otherwise",
    }
    assert block["one_line"]["otherwise"] == forecast_energy_reading.MIXED
    shows = block["what_a_result_shows"]
    for part in ("A gain would show", "A loss would show", "No measurable difference would show"):
        assert part in shows
    assert block["unit_not_judged"].startswith("nrel-before-may-2019 has no bar")


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


# --- the README -----------------------------------------------------------------------------------

COMMAND = (
    "python -m fleetreserve.run_record --name forecast-energy --long --max-seconds 172800 -- "
    "uv run --frozen --group learn python -m fleetreserve.forecast_energy --workers 6"
)


def section() -> str:
    """The README section of the trial, from its heading to the next line that begins `## ` (or
    the end of the file), with every run of whitespace made one space. A later trial's section is
    not part of it."""
    start = README.index("## Trial forecast-energy: more cautious energy bounds")
    end = README.find("\n## ", start + 1)
    return " ".join(README[start : len(README) if end == -1 else end].split())


def test_the_section_is_after_the_no_encoder_one_and_names_the_command_and_the_outputs():
    assert README.index("## Trial forecast-noset") < README.index("## Trial forecast-energy")
    text = section()
    assert text.count("``` python -m") == 1 and COMMAND in text
    for output in (
        "forecast-energy.json",
        "forecast-energy-summary.txt",
        "forecast-energy-timing.json",
        "forecast-energy-forecasts.csv",
        "forecast-energy-<arm>-dial-choice.json",
        "forecast-energy-<arm>-split-offers.json",
        "made-after-sealed-run.txt",
        "not-the-trial.txt",
    ):
        assert output in text, output
    assert "--max-seconds 172800" in text and "--not-the-trial" in text
    assert "/Users" not in README and "/private" not in README


def test_the_section_says_what_a_gain_a_loss_and_no_difference_would_and_would_not_show():
    text = section()
    assert "APPROVED TO RUN by the owner's decision of 24 November 2026" in text
    assert "approved_to_run: true" in text and "cannot grant it" in text
    assert "What a result would and would not show:" in text
    for part in ("- A gain:", "- A loss:", "- No measurable difference:"):
        assert part in text
    assert "It would not show that the held-out zero came from the energy bounds" in text
    assert (
        "No Dutch or Korean site is a development unit, so the trial cannot speak for them" in text
    )
    assert (
        "The trial was run once, on 25 November 2026, at commit e288876: start 01:17:57 UTC, "
        "end 01:33:00 UTC, exit 0." in text
    )
    assert "has not been run" not in text
    for arm in ("`known`", "`empirical`", "`cautious`"):
        assert arm in text
    assert "calibration, the certificate, the loss limits" in text


def test_the_section_states_the_bars_the_limits_and_the_one_line_words():
    text = section()
    assert "8% at the workplace unit, 30% at the pooled Norwegian unit" in text
    assert "0.10" in text and "0.05" in text
    for words in (
        "shows a gain",
        "shows a loss",
        "no measurable difference",
        forecast_energy_reading.HELPS,
        forecast_energy_reading.HURTS,
    ):
        assert words in text, words


# --- 4 and 5. no sealed data, nothing written outside the run folder -----------------------------


@pytest.mark.parametrize("module", [forecast_energy, forecast_energy_reading])
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
    tree = ast.parse(Path(forecast_energy.__file__).read_text())
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


# --- 6. a loss is claimed no wider than the one setting tried -------------------------------------

UNTRIED = (
    "levels between 0.05 and",
    "a stricter need bound",
    "other departure levels",
    "other models",
    "need not move in one direction with the level",
)


def test_a_loss_is_claimed_no_wider_than_the_one_setting_tried():
    """The trial tries one setting. A loss shows only that level 0.05 for rate and storable energy,
    at these units and this departure level, makes the offer smaller than the empirical bounds;
    the README and the settings list what stays untried and name no single next step. Red if the
    old sentence ("no change of the two levels recovers the gap", "the next step would be") is
    back, or if an untried item is dropped."""
    flat = section()
    start = flat.index("- A loss:")
    loss = flat[start : flat.index("- No measurable difference:")]
    shows = SETTINGS["reading"]["what_a_result_shows"]
    loss_sentence = shows[
        shows.index("A loss would show") : shows.index("No measurable difference")
    ]
    for text in (loss, loss_sentence):
        assert "level 0.05" in text and "smaller" in text
        for item in UNTRIED:
            assert item in text, item
        assert "next step" not in text and "no change of the two levels" not in text
        assert "recovers the gap" not in text
    assert "next step" not in flat and "next step" not in json.dumps(SETTINGS)

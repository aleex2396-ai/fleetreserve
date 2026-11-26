"""The `cautious` value of `energy_bounds` (`forecast_empirical.CAUTIOUS`) stays inside the energy
forecast trial. Written to be run through the test slot. Each test says which edit turns it red.

The parser of the matrix files accepts the value, so nothing else refuses a hand edit that puts it
into a configuration of a matrix, a run list or a pass-mark file of the plan. This file does.

1. Every committed JSON file under `gate/`, `learn/`, `run-lists/` and `passmarks/` (found by
   walking the folders, so a file added later is read too) holds no value, and no key, equal to
   the value. Red if one does; the message names the file and the key path.
2. In `trials-after-sealed-run/` the value may appear only in the files of the energy trial: a file
   whose name starts with `forecast-energy`, and the section of `README.md` that begins at the
   heading of that trial and ends at the next heading of the same level. Any other JSON value or
   key equal to the value, or any other README line that holds the word, makes the test red.
3. The helpers themselves are seen to go red and to stay green: a copy of the committed
   `gate/matrix.json` with the value put into one alternate-family configuration, a made-up
   run list and a made-up README, each in a temporary folder the helpers are pointed at, and the
   untouched copies, which give nothing.

The second energy trial (`forecast_energy_2`) adds two values, `cautious-need` and `cautious-all`
(`forecast_empirical.CAUTIOUS_NEED`, `CAUTIOUS_ALL`); the same rules are extended to the three
values, in their own tests below, with the first trial's tests left as they were:

4. No committed JSON file under the four plan folders holds any of the THREE values, as a value or
   as a key. Each of the three has its own red case (a made-up copy of `gate/matrix.json` with the
   value in one alternate-family configuration) and its own green case (the untouched copy).
5. In `trials-after-sealed-run/` the two NEW values appear only in the files whose name starts with
   `forecast-energy-2` and in the section of `README.md` that begins "## Trial forecast-energy-2";
   the first trial's files and its README section hold neither (a narrower rule than for
   `cautious`, which both trials' files may hold). Red if a first-trial file, another trial's file
   or another README section holds one: each is made up in a temporary folder and seen red, and the
   untouched made-up folder gives nothing.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from fleetreserve.forecast_empirical import CAUTIOUS, CAUTIOUS_ALL, CAUTIOUS_NEED

TOP = Path(__file__).resolve().parents[2]
PLAN_FOLDERS = ("gate", "learn", "run-lists", "passmarks")
TRIAL_FOLDER = "trials-after-sealed-run"
TRIAL_FILE_PREFIX = "forecast-energy"
TRIAL_HEADING = "## Trial forecast-energy"
WORD = re.compile(rf"\b{re.escape(CAUTIOUS)}\b", re.IGNORECASE)
NEW_VALUES = (CAUTIOUS_NEED, CAUTIOUS_ALL)
THREE_VALUES = (CAUTIOUS, *NEW_VALUES)
SECOND_FILE_PREFIX = "forecast-energy-2"
SECOND_HEADING = "## Trial forecast-energy-2"


def whole_word(value):
    """The value as a whole word: not part of a longer hyphenated name (`cautious` is not found in
    `cautious-need`)."""
    return re.compile(rf"(?<![\w-]){re.escape(value)}(?![\w-])", re.IGNORECASE)


def find_value(node, path="", value=CAUTIOUS):
    """Every key path in a loaded JSON document whose key or value equals `value`, as text."""
    found = []
    if isinstance(node, dict):
        for key, child in node.items():
            here = f"{path}/{key}"
            if key == value:
                found.append(f"{here} (key)")
            found.extend(find_value(child, here, value))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            found.extend(find_value(child, f"{path}/{index}", value))
    elif node == value:
        found.append(path or "/")
    return found


def hits_in_plan_files(top=TOP, values=(CAUTIOUS,)):
    """(file, key path) of every use of any of `values` in the JSON files of the four plan folders
    under `top`.

    A folder that does not exist is skipped by name; none of the four existing is an error.
    """
    folders = [top / name for name in PLAN_FOLDERS if (top / name).is_dir()]
    if not folders:
        raise FileNotFoundError(f"none of {', '.join(PLAN_FOLDERS)} exists under {top}")
    hits = []
    for folder in folders:
        for file in sorted(folder.rglob("*.json")):
            document = json.loads(file.read_text())
            hits.extend(
                (file.relative_to(top).as_posix(), p)
                for value in values
                for p in find_value(document, "", value)
            )
    return hits


def readme_outside_the_trial(text, heading=TRIAL_HEADING):
    """The numbers and text of the README lines outside the section that begins with `heading`
    (the energy trial's own section by default)."""
    outside, inside = [], False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            inside = line.startswith(heading)
        if not inside:
            outside.append((number, line))
    return outside


def hits_in_trial_folder(
    top=TOP,
    value=CAUTIOUS,
    prefix=TRIAL_FILE_PREFIX,
    heading=TRIAL_HEADING,
    word=WORD,
):
    """(file, place) of every use of `value` in `trials-after-sealed-run/` outside the files whose
    name starts with `prefix` and outside the README section that begins with `heading`. The
    defaults are the first energy trial's, for `cautious`."""
    folder = top / TRIAL_FOLDER
    if not folder.is_dir():
        return []
    hits = []
    for file in sorted(folder.rglob("*")):
        if not file.is_file() or file.name.startswith(prefix):
            continue
        name = file.relative_to(top).as_posix()
        if file.suffix == ".json":
            hits.extend((name, p) for p in find_value(json.loads(file.read_text()), "", value))
        elif file.suffix == ".md" and file.name == "README.md":
            hits.extend(
                (name, f"line {n}")
                for n, line in readme_outside_the_trial(file.read_text(), heading)
                if word.search(line)
            )
        elif word.search(file.read_text(errors="replace")):
            hits.append((name, "text"))
    return hits


def hits_of_a_new_value(value, top=TOP):
    """`hits_in_trial_folder` for one of the two new values: only the second trial's files and its
    README section may hold it, and it is found as a whole word only."""
    return hits_in_trial_folder(top, value, SECOND_FILE_PREFIX, SECOND_HEADING, whole_word(value))


def test_no_committed_file_of_the_plan_uses_the_trials_value():
    hits = hits_in_plan_files()
    assert not hits, f"the trial's value {CAUTIOUS!r} is in a file of the plan: {hits}"


def test_the_value_stays_in_the_energy_trial_inside_the_trial_folder():
    hits = hits_in_trial_folder()
    assert not hits, f"the trial's value {CAUTIOUS!r} is outside the energy trial: {hits}"


def test_the_energy_trial_does_use_its_own_value():
    """Guards the test above against an empty search: the trial's own settings file names it."""
    own = TOP / TRIAL_FOLDER / "forecast-energy.json"
    assert find_value(json.loads(own.read_text()))
    section = (TOP / TRIAL_FOLDER / "README.md").read_text()
    assert len(readme_outside_the_trial(section)) < len(section.splitlines())


def test_the_value_in_an_alternate_family_configuration_of_a_matrix_is_found(tmp_path):
    (tmp_path / "gate").mkdir()
    copy = tmp_path / "gate" / "matrix.json"
    shutil.copy(TOP / "gate" / "matrix.json", copy)
    assert hits_in_plan_files(tmp_path) == []  # the untouched copy gives nothing
    matrix = json.loads(copy.read_text())
    index = next(i for i, c in enumerate(matrix["configurations"]) if c["family"] == "alternate")
    matrix["configurations"][index]["energy_bounds"] = CAUTIOUS
    copy.write_text(json.dumps(matrix))
    assert hits_in_plan_files(tmp_path) == [
        ("gate/matrix.json", f"/configurations/{index}/energy_bounds")
    ]


def test_the_value_in_a_learning_matrix_a_run_list_or_a_pass_mark_file_is_found(tmp_path):
    made = (("learn", "matrix.json"), ("run-lists", "a.json"), ("passmarks", "b.json"))
    for folder, name in made:
        (tmp_path / folder / "deeper").mkdir(parents=True)
        (tmp_path / folder / "deeper" / name).write_text(json.dumps({"x": [{"y": CAUTIOUS}]}))
    assert hits_in_plan_files(tmp_path) == [
        (f"{folder}/deeper/{name}", "/x/0/y") for folder, name in made
    ]


def test_a_key_equal_to_the_value_is_found(tmp_path):
    (tmp_path / "learn").mkdir()
    (tmp_path / "learn" / "k.json").write_text(json.dumps({CAUTIOUS: 1}))
    assert hits_in_plan_files(tmp_path) == [("learn/k.json", f"/{CAUTIOUS} (key)")]


def test_a_missing_folder_is_skipped_and_all_four_missing_is_an_error(tmp_path):
    (tmp_path / "passmarks").mkdir()
    (tmp_path / "passmarks" / "ok.json").write_text(json.dumps({"x": "known"}))
    assert hits_in_plan_files(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        hits_in_plan_files(tmp_path / "nowhere")


def test_the_trial_folder_allows_the_energy_files_and_refuses_the_others(tmp_path):
    folder = tmp_path / TRIAL_FOLDER
    folder.mkdir()
    (folder / "forecast-energy.json").write_text(json.dumps({"arm": CAUTIOUS}))
    (folder / "README.md").write_text(
        f"## Trial a: x\nno such word here\n{TRIAL_HEADING}: y\nthe {CAUTIOUS} arm\n"
        f"## Trial b: z\nplain text\n"
    )
    assert hits_in_trial_folder(tmp_path) == []
    (folder / "forecast-mix.json").write_text(json.dumps({"energy_bounds": CAUTIOUS}))
    with (folder / "README.md").open("a") as handle:
        handle.write(f"the {CAUTIOUS.upper()} bounds again\n")
    assert hits_in_trial_folder(tmp_path) == [
        (f"{TRIAL_FOLDER}/README.md", "line 7"),
        (f"{TRIAL_FOLDER}/forecast-mix.json", "/energy_bounds"),
    ]


# --- the second energy trial's two values ---------------------------------------------------------


def test_no_committed_file_of_the_plan_uses_any_of_the_three_values():
    hits = hits_in_plan_files(values=THREE_VALUES)
    assert not hits, f"a value of the energy trials is in a file of the plan: {hits}"


@pytest.mark.parametrize("value", NEW_VALUES)
def test_a_new_value_stays_in_the_second_energy_trial_inside_the_trial_folder(value):
    hits = hits_of_a_new_value(value)
    assert not hits, f"the value {value!r} is outside the second energy trial: {hits}"


@pytest.mark.parametrize("value", NEW_VALUES)
def test_the_second_energy_trial_does_use_its_own_values_and_the_first_does_not(value):
    """Guards the test above against an empty search, and pins the narrower rule: the second
    trial's settings and README section name the value, the first trial's file and section do
    not."""
    own = TOP / TRIAL_FOLDER / "forecast-energy-2.json"
    assert find_value(json.loads(own.read_text()), "", value)
    first = TOP / TRIAL_FOLDER / "forecast-energy.json"
    assert not find_value(json.loads(first.read_text()), "", value)
    text = (TOP / TRIAL_FOLDER / "README.md").read_text()
    pattern = whole_word(value)
    inside_second = [
        line for _, line in readme_outside_the_trial(text, "## nothing") if pattern.search(line)
    ]
    outside_second = [
        line for _, line in readme_outside_the_trial(text, SECOND_HEADING) if pattern.search(line)
    ]
    assert inside_second and not outside_second  # in the README, only in the second section


def test_a_whole_word_search_does_not_find_cautious_in_the_longer_names():
    for value in NEW_VALUES:
        assert whole_word(CAUTIOUS).search(f"the {value} arm") is None
        assert whole_word(value).search(f"the {value} arm")
    assert whole_word(CAUTIOUS_NEED).search("cautious-need-all") is None
    assert whole_word(CAUTIOUS).search("the cautious arm") and whole_word(CAUTIOUS).search(
        "(cautious)"
    )
    assert WORD.search("the cautious-need arm")  # the first trial's rule finds the stem, as before


@pytest.mark.parametrize("value", THREE_VALUES)
def test_each_of_the_three_values_in_an_alternate_family_configuration_is_found(tmp_path, value):
    (tmp_path / "gate").mkdir()
    copy = tmp_path / "gate" / "matrix.json"
    shutil.copy(TOP / "gate" / "matrix.json", copy)
    assert hits_in_plan_files(tmp_path, THREE_VALUES) == []  # the green case: the untouched copy
    matrix = json.loads(copy.read_text())
    index = next(i for i, c in enumerate(matrix["configurations"]) if c["family"] == "alternate")
    matrix["configurations"][index]["energy_bounds"] = value
    copy.write_text(json.dumps(matrix))
    assert hits_in_plan_files(tmp_path, THREE_VALUES) == [  # the red case
        ("gate/matrix.json", f"/configurations/{index}/energy_bounds")
    ]
    others = [v for v in THREE_VALUES if v != value]
    assert hits_in_plan_files(tmp_path, others) == []  # each value is found as itself only


@pytest.mark.parametrize("value", THREE_VALUES)
def test_each_of_the_three_values_as_a_key_or_in_the_other_plan_folders_is_found(tmp_path, value):
    for folder, name in (
        ("learn", "matrix.json"),
        ("run-lists", "a.json"),
        ("passmarks", "b.json"),
    ):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).write_text(json.dumps({"x": [{"y": value}], value: 1}))
    assert hits_in_plan_files(tmp_path, THREE_VALUES) == [
        (f"{folder}/{name}", path)
        for folder, name in (
            ("learn", "matrix.json"),
            ("run-lists", "a.json"),
            ("passmarks", "b.json"),
        )
        for path in ("/x/0/y", f"/{value} (key)")
    ]


def made_up_trial_folder(tmp_path, value):
    """A made-up trial folder holding `value` only where the second trial may hold it."""
    folder = tmp_path / TRIAL_FOLDER
    folder.mkdir()
    (folder / "forecast-energy-2.json").write_text(json.dumps({"arm": value}))
    (folder / "forecast-energy.json").write_text(json.dumps({"arm": CAUTIOUS}))
    (folder / "README.md").write_text(
        f"## Trial a: x\nno such word here\n{TRIAL_HEADING}: first\nthe {CAUTIOUS} arm\n"
        f"{SECOND_HEADING}: second\nthe {value} arm and the {CAUTIOUS} arm\n"
        f"## Trial b: z\nplain text\n"
    )
    return folder


@pytest.mark.parametrize("value", NEW_VALUES)
def test_a_new_value_is_allowed_in_the_second_trials_files_and_refused_elsewhere(tmp_path, value):
    folder = made_up_trial_folder(tmp_path, value)
    assert hits_of_a_new_value(value, tmp_path) == []  # the green case
    assert hits_in_trial_folder(tmp_path) == []  # `cautious` in both trials' files: as before
    (folder / "forecast-energy.json").write_text(json.dumps({"arm": value}))  # the first trial's
    assert hits_of_a_new_value(value, tmp_path) == [
        (f"{TRIAL_FOLDER}/forecast-energy.json", "/arm")
    ]
    (folder / "forecast-energy.json").write_text(json.dumps({"arm": CAUTIOUS}))
    (folder / "forecast-mix.json").write_text(json.dumps({"energy_bounds": value}))
    with (folder / "README.md").open("a") as handle:
        handle.write(f"the {value.upper()} bounds again\n")
    assert hits_of_a_new_value(value, tmp_path) == [
        (f"{TRIAL_FOLDER}/README.md", "line 9"),
        (f"{TRIAL_FOLDER}/forecast-mix.json", "/energy_bounds"),
    ]


@pytest.mark.parametrize("value", NEW_VALUES)
def test_a_new_value_in_the_first_trials_readme_section_is_refused(tmp_path, value):
    folder = made_up_trial_folder(tmp_path, value)
    text = (
        (folder / "README.md")
        .read_text()
        .replace(
            f"{TRIAL_HEADING}: first\nthe {CAUTIOUS} arm",
            f"{TRIAL_HEADING}: first\nthe {value} arm",
        )
    )
    (folder / "README.md").write_text(text)
    assert hits_of_a_new_value(value, tmp_path) == [(f"{TRIAL_FOLDER}/README.md", "line 4")]
    assert hits_in_trial_folder(tmp_path) == []  # `cautious` is still free in the first section

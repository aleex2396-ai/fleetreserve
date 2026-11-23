"""The run list's form and its checker (`fleetreserve.run_list`), on made-up lists.

Every list here is made up. Its commands name real modules of the package with made-up
arguments: the checker reads only the `python -m fleetreserve.<module>` call and the
placeholders, never what the arguments mean (reading L4).

1. A well-formed list (`good_list`: an entry of the sealed run only, a group of units that
   differs between the purposes and one that does not, extra arguments of the sealed run, a
   `{run:<label>}` in each purpose) is well formed (exit 0): one `difference` line for each of
   its three differences, and a last line with each purpose's entries and time limits.
2. Each fault, put into an otherwise well-formed list, makes it faulty (exit 1) with a line that
   names it: an unknown module, a command that is no `python -m fleetreserve.<module>` call, a
   missing, non-positive, non-finite, inexact or too large time limit, a duplicate label, also
   when letter case is ignored, a label `run_record` refuses, a label whose run folder name would
   be longer than the file system allows (the recorder's own agreement with the checker on labels
   is tested in `test_run_driver.py`, where the real recorder runs), a rehearsal group naming a
   unit that is no training or development unit, a group of the sealed run mixing the kinds, a
   group a purpose lacks or that no entry names, `{units}` without a group or a group without
   `{units}`, `{units}` in extra arguments, extra arguments of an entry of one purpose or for no
   purpose, a dependency or a run placeholder on a later, a missing or the same entry or on one
   that does not run in the purpose, a reference to a seed the fixed block lacks, a fixed value no
   entry refers to, a purpose with no entry, a missing or an unknown field (the old `purpose` and
   `kind` among them), the old format, text that is not JSON.
3. A list that cannot be read: could not decide (exit 2).
4. `instantiate` gives each purpose its entries with the fixed values, the purpose's units and its
   extra arguments put in, by hand; `fill_runs` puts each earlier run folder in place of its
   placeholder, and nothing else (a fixed name that looks like a placeholder stays as it is).
"""

import copy
import json
from pathlib import Path

import pytest

from fleetreserve import run_list

TRUTH = ["uv", "run", "--frozen", "python", "-m", "fleetreserve.truth_build"]
SEALED_TRUTH = ["uv", "run", "--frozen", "python", "-m", "fleetreserve.heldout_truth_build"]
LEARN = ["uv", "run", "--frozen", "--group", "learn", "python", "-m"]
SCORED_DEVELOPMENT = ["nrel-before-may-2019", "nrel-from-may-2019"]
SCORED_SEALED = ["OSL_T", "TRO_R"]
SEALED_ROUTE = ["--held-out-tables", "data/derived/sealed-tables", "--held-out-truth-run"]


def good_list() -> dict:
    """A made-up list of both purposes with four entries and a fixed block of every kind."""
    return {
        "format": "run-list-2",
        "about": "a made-up run list",
        "units": {
            "rehearsal": {"scored": list(SCORED_DEVELOPMENT), "fitted": ["caltech", "jpl"]},
            "sealed-run": {"scored": list(SCORED_SEALED), "fitted": ["caltech", "jpl"]},
        },
        "fixed": {
            "seeds": {"bootstrap": 20261116, "day-sample": 7},
            "counts": {"bootstrap-resamples": 2000, "day-sample-days": 150},
            "names": {"base-set": "day-average"},
        },
        "entries": [
            {
                "label": "wp2-truth",
                "only": None,
                "command": list(TRUTH),
                "units": None,
                "extra": {},
                "setting": "main",
                "max_seconds": 3600,
                "after": [],
            },
            {
                "label": "sealed-truth",
                "only": "sealed-run",
                "command": list(SEALED_TRUTH),
                "units": None,
                "extra": {},
                "setting": "main",
                "max_seconds": 7200,
                "after": [],
            },
            {
                "label": "rehearsal-cold-start",
                "only": None,
                "command": [
                    *LEARN,
                    "fleetreserve.cold_start",
                    "--units",
                    "{units}",
                    "--days",
                    "{count:day-sample-days}",
                    "--seed",
                    "{seed:day-sample}",
                    "--base-set",
                    "{name:base-set}",
                ],
                "units": "scored",
                "extra": {"sealed-run": [*SEALED_ROUTE, "{run:sealed-truth}"]},
                "setting": "main",
                "max_seconds": 14400,
                "after": ["wp2-truth"],
            },
            {
                "label": "rehearsal-intervals",
                "only": None,
                "command": [
                    "python3.12",
                    "-m",
                    "fleetreserve.gate_summary",
                    "--fit",
                    "{units}",
                    "--resamples",
                    "{count:bootstrap-resamples}",
                    "--seed",
                    "{seed:bootstrap}",
                    "--from",
                    "{run:rehearsal-cold-start}",
                ],
                "units": "fitted",
                "extra": {},
                "setting": "held",
                "max_seconds": 600.5,
                "after": ["wp2-truth", "rehearsal-cold-start"],
            },
        ],
    }


GOOD_DIFFERENCES = [
    "difference: entry 2 (sealed-truth): runs in the sealed-run only",
    "difference: entry 3 (rehearsal-cold-start): its units, the group 'scored': rehearsal "
    "nrel-before-may-2019 nrel-from-may-2019; sealed-run OSL_T TRO_R",
    "difference: entry 3 (rehearsal-cold-start): arguments for the sealed-run only: "
    "--held-out-tables data/derived/sealed-tables --held-out-truth-run {run:sealed-truth}",
]


def check(tmp_path: Path, document, capsys) -> tuple[int, list[str]]:
    path = tmp_path / "list.json"
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    code = run_list.main([str(path), "--no-starts"])  # stand-in commands: no start check
    return code, capsys.readouterr().out.splitlines()


def test_a_well_formed_list_is_well_formed_and_its_differences_are_listed(tmp_path, capsys):
    code, lines = check(tmp_path, good_list(), capsys)
    assert code == 0, lines
    assert lines[:-2] == GOOD_DIFFERENCES
    assert lines[-2] == "starts: the starts not made (--no-starts)"
    assert lines[-1].startswith("well formed: ")
    assert lines[-1].endswith(
        ": the rehearsal 3 entries, time limits adding up to 18600.5 s (5.2 h); the sealed-run 4 "
        "entries, time limits adding up to 25800.5 s (7.2 h); 3 differences between the purposes"
    )


def test_a_list_of_one_entry_says_one_entry(tmp_path, capsys):
    document = good_list()
    document["entries"] = document["entries"][:1]
    document["units"] = {"rehearsal": {}, "sealed-run": {}}
    document["fixed"] = {"seeds": {}, "counts": {}, "names": {}}
    code, lines = check(tmp_path, document, capsys)
    assert code == 0, lines
    assert lines[0] == "starts: the starts not made (--no-starts)"
    assert lines[1].endswith(
        ": the rehearsal 1 entry, time limits adding up to 3600 s (1.0 h); the sealed-run 1 entry, "
        "time limits adding up to 3600 s (1.0 h); 0 differences between the purposes"
    )
    assert len(lines) == 2


def _entry(document, k: int) -> dict:
    return document["entries"][k]


def unknown_module(d):
    _entry(d, 2)["command"][7] = "fleetreserve.no_such_module"


def no_module_call(d):
    _entry(d, 0)["command"] = ["bash", "-c", "true"]


def module_call_without_python(d):
    _entry(d, 0)["command"] = ["uv", "run", "-m", "fleetreserve.truth_build"]


def two_module_calls(d):
    _entry(d, 0)["command"] = [*TRUTH, "python", "-m", "fleetreserve.gate_run"]


def no_time_limit(d):
    del _entry(d, 2)["max_seconds"]


def zero_time_limit(d):
    _entry(d, 2)["max_seconds"] = 0


def negative_time_limit(d):
    _entry(d, 2)["max_seconds"] = -60


def text_time_limit(d):
    _entry(d, 2)["max_seconds"] = "600"


def boolean_time_limit(d):
    _entry(d, 2)["max_seconds"] = True


def time_limit_past_exact_floats(d):
    _entry(d, 2)["max_seconds"] = 2**53 + 1  # the recorder reads it as 2**53


def time_limit_past_any_float(d):
    _entry(d, 2)["max_seconds"] = 10**400


def duplicate_label(d):
    _entry(d, 3)["label"] = "rehearsal-cold-start"
    _entry(d, 3)["command"][-1] = "{run:wp2-truth}"
    _entry(d, 3)["after"] = ["wp2-truth"]


def labels_equal_but_for_case(d):
    _entry(d, 3)["label"] = "Rehearsal-Cold-Start"  # one run folder name on a case-blind system
    _entry(d, 3)["command"][-1] = "{run:wp2-truth}"
    _entry(d, 3)["after"] = ["wp2-truth"]


def label_with_space(d):
    _entry(d, 3)["label"] = "a b"


def label_with_slash(d):
    _entry(d, 3)["label"] = "a/b"


def label_with_dot(d):
    _entry(d, 3)["label"] = "a.b"


def empty_label(d):
    _entry(d, 3)["label"] = ""


def label_too_long(d):
    _entry(d, 3)["label"] = "a" * 250  # its run folder name, <UTC start>-<label>, is 267 bytes


def sealed_unit_in_a_rehearsal_group(d):
    d["units"]["rehearsal"]["scored"] = ["nrel-before-may-2019", "OSL_T"]


def misspelt_development_unit(d):
    d["units"]["rehearsal"]["scored"] = ["nrel-before-may-2019", "OSL1"]


def mixed_kinds_in_a_sealed_run_group(d):
    d["units"]["sealed-run"]["scored"] = ["OSL_T", "nrel-before-may-2019"]


def empty_group(d):
    d["units"]["sealed-run"]["scored"] = []


def no_units_of_a_purpose(d):
    del d["units"]["sealed-run"]


def group_a_purpose_lacks(d):
    del d["units"]["sealed-run"]["scored"]


def group_no_entry_names(d):
    d["units"]["rehearsal"]["spare"] = ["BER"]


def group_without_its_placeholder(d):
    _entry(d, 2)["command"][8:10] = []  # drops --units {units}


def placeholder_without_a_group(d):
    _entry(d, 0)["command"].append("{units}")


def units_placeholder_in_extra_arguments(d):
    _entry(d, 2)["extra"]["sealed-run"].append("{units}")


def extra_arguments_of_an_entry_of_one_purpose(d):
    _entry(d, 1)["extra"] = {"sealed-run": ["--more"]}


def extra_arguments_for_no_purpose(d):
    _entry(d, 2)["extra"]["batch"] = ["--more"]


def empty_extra_arguments(d):
    _entry(d, 2)["extra"]["sealed-run"] = []


def unknown_only(d):
    _entry(d, 1)["only"] = "batch"


def run_of_a_later_entry(d):
    _entry(d, 2)["extra"]["sealed-run"][-1] = "{run:rehearsal-intervals}"


def run_of_a_missing_entry(d):
    _entry(d, 2)["extra"]["sealed-run"][-1] = "{run:no-such-entry}"


def run_of_itself(d):
    _entry(d, 3)["command"][-1] = "{run:rehearsal-intervals}"


def run_of_an_entry_that_does_not_run_in_the_purpose(d):
    _entry(d, 3)["command"][-1] = "{run:sealed-truth}"  # the rehearsal has no sealed-truth


def depends_on_a_later_entry(d):
    _entry(d, 2)["after"] = ["wp2-truth", "rehearsal-intervals"]


def depends_on_a_missing_entry(d):
    _entry(d, 2)["after"] = ["no-such-entry"]


def depends_on_itself(d):
    _entry(d, 2)["after"] = ["rehearsal-cold-start"]


def depends_on_an_entry_that_does_not_run_in_the_purpose(d):
    _entry(d, 3)["after"].append("sealed-truth")


def seed_the_block_lacks(d):
    _entry(d, 3)["command"][8] = "{seed:bootstrapp}"  # a misspelt name: also leaves one unused


def count_the_block_lacks(d):
    _entry(d, 2)["command"][11] = "{count:days}"


def unused_fixed_value(d):
    d["fixed"]["seeds"]["spare"] = 3


def malformed_placeholder(d):
    _entry(d, 2)["command"][13] = "{seed day-sample}"


def placeholder_inside_an_argument(d):
    _entry(d, 2)["extra"]["sealed-run"][-1] = "--run={run:sealed-truth}"


def no_entry_in_a_purpose(d):
    d["entries"] = [_entry(d, 1)]
    d["units"] = {"rehearsal": {}, "sealed-run": {}}
    d["fixed"] = {"seeds": {}, "counts": {}, "names": {}}


def unknown_field(d):
    _entry(d, 0)["seed"] = 1


def the_old_kind_field(d):
    _entry(d, 0)["kind"] = "development"


def the_old_purpose_field(d):
    d["purpose"] = "rehearsal"


def the_old_format(d):
    d["format"] = "run-list-1"


def missing_after(d):
    del _entry(d, 0)["after"]


def missing_only(d):
    del _entry(d, 0)["only"]


def unknown_setting(d):
    _entry(d, 0)["setting"] = "even"


def seed_below_zero(d):
    d["fixed"]["seeds"]["bootstrap"] = -1


def zero_count(d):
    d["fixed"]["counts"]["bootstrap-resamples"] = 0


def no_entries(d):
    d["entries"] = []


COLD = "entry 3 (rehearsal-cold-start)"
INTERVALS = "entry 4 (rehearsal-intervals)"
FAULTS = [
    (unknown_module, COLD, "no module of the package"),
    (no_module_call, "entry 1 (wp2-truth)", "not one `python -m fleetreserve.<module>` call"),
    (module_call_without_python, "entry 1", "not one `python -m fleetreserve.<module>` call"),
    (two_module_calls, "entry 1", "not one `python -m fleetreserve.<module>` call"),
    (no_time_limit, COLD, "has no time limit"),
    (zero_time_limit, COLD, "is not a positive finite number of seconds"),
    (negative_time_limit, COLD, "is not a positive finite number of seconds"),
    (text_time_limit, COLD, "is not a positive finite number of seconds"),
    (boolean_time_limit, COLD, "is not a positive finite number of seconds"),
    (time_limit_past_exact_floats, COLD, "reads back as 9007199254740992.0 seconds"),
    (time_limit_past_any_float, COLD, "is too large to be read as a number of seconds"),
    (duplicate_label, "entry 4 (rehearsal-cold-start)", "its label is that of an earlier entry"),
    (
        labels_equal_but_for_case,
        "entry 4 (Rehearsal-Cold-Start)",
        "its label is that of an earlier entry (entry 3, rehearsal-cold-start) when letter case",
    ),
    (label_with_space, "entry 4 (a b)", "is refused by run_record"),
    (label_with_slash, "entry 4 (a/b)", "is refused by run_record"),
    (label_with_dot, "entry 4 (a.b)", "is refused by run_record"),
    (empty_label, "entry 4", "is refused by run_record"),
    (label_too_long, "entry 4", "its run folder name would be 267 bytes long, more than the"),
    (
        sealed_unit_in_a_rehearsal_group,
        "units.rehearsal.scored",
        "names OSL_T, which is no training or development unit: the rehearsal runs on those alone",
    ),
    (misspelt_development_unit, "units.rehearsal.scored", "names OSL1, which is no training"),
    (mixed_kinds_in_a_sealed_run_group, "units.sealed-run.scored", "mixes the two kinds of units"),
    (empty_group, "units.sealed-run.scored", "is not a non-empty list of distinct unit names"),
    (no_units_of_a_purpose, "units", "lacks the field 'sealed-run'"),
    (group_a_purpose_lacks, COLD, "names the units group 'scored', which units.sealed-run lacks"),
    (group_no_entry_names, "units.rehearsal.spare", "no entry of the rehearsal names it"),
    (group_without_its_placeholder, COLD, "names the units group 'scored' but its command has no"),
    (placeholder_without_a_group, "entry 1", "its command holds {units} but it names no units"),
    (
        units_placeholder_in_extra_arguments,
        COLD,
        "its extra arguments hold {units}, which only its command may hold",
    ),
    (
        extra_arguments_of_an_entry_of_one_purpose,
        "entry 2 (sealed-truth)",
        "runs in the sealed-run only, so its arguments go in its command and its extra is {}",
    ),
    (extra_arguments_for_no_purpose, COLD, "its extra names 'batch', which is no purpose"),
    (
        empty_extra_arguments,
        COLD,
        "its extra arguments for the sealed-run are not a non-empty list of non-empty texts",
    ),
    (unknown_only, "entry 2", "its only 'batch' is not null or one of rehearsal, sealed-run"),
    (run_of_a_later_entry, COLD, "refers to the run of 'rehearsal-intervals', which is no earlier"),
    (run_of_a_missing_entry, COLD, "refers to the run of 'no-such-entry', which is no earlier"),
    (run_of_itself, INTERVALS, "refers to its own run"),
    (
        run_of_an_entry_that_does_not_run_in_the_purpose,
        INTERVALS,
        "refers to the run of 'sealed-truth' in the rehearsal, where 'sealed-truth' does not run",
    ),
    (depends_on_a_later_entry, COLD, "depends on 'rehearsal-intervals', which is no earlier"),
    (depends_on_a_missing_entry, COLD, "depends on 'no-such-entry', which is no earlier"),
    (depends_on_itself, COLD, "depends on itself"),
    (
        depends_on_an_entry_that_does_not_run_in_the_purpose,
        INTERVALS,
        "depends on 'sealed-truth' in the rehearsal, where 'sealed-truth' does not run",
    ),
    (seed_the_block_lacks, INTERVALS, "names the seed 'bootstrapp', which fixed.seeds lacks"),
    (count_the_block_lacks, COLD, "names the count 'days', which fixed.counts lacks"),
    (unused_fixed_value, "fixed.seeds.spare", "no entry refers to it"),
    (malformed_placeholder, COLD, "holds a brace but is not one placeholder"),
    (placeholder_inside_an_argument, COLD, "holds a brace but is not one placeholder"),
    (no_entry_in_a_purpose, "the list", "no entry runs in the rehearsal"),
    (unknown_field, "entry 1", "has the unknown field 'seed'"),
    (the_old_kind_field, "entry 1 (wp2-truth)", "has the unknown field 'kind'"),
    (the_old_purpose_field, "the list", "has the unknown field 'purpose'"),
    (the_old_format, "the list", "its format 'run-list-1' is not 'run-list-2'"),
    (missing_after, "entry 1", "lacks the field 'after'"),
    (missing_only, "entry 1", "lacks the field 'only'"),
    (unknown_setting, "entry 1", "its setting 'even' is not one of main, held"),
    (seed_below_zero, "fixed.seeds.bootstrap", "is not a whole number >= 0"),
    (zero_count, "fixed.counts.bootstrap-resamples", "is not a whole number >= 1"),
    (no_entries, "the list", "its entries are not a non-empty list"),
]


@pytest.mark.parametrize(("change", "where", "what"), FAULTS, ids=[f[0].__name__ for f in FAULTS])
def test_each_fault_makes_the_list_faulty_and_is_named(tmp_path, capsys, change, where, what):
    document = copy.deepcopy(good_list())
    change(document)
    code, lines = check(tmp_path, document, capsys)
    assert code == 1, lines
    assert lines[-1].startswith("faulty: ")
    named = [line for line in lines[:-1] if line.startswith(f"FAULT: {where}") and what in line]
    assert named, lines


@pytest.mark.parametrize(
    "text",
    ["{not json", '{"format": NaN}', "[1, 2]", ""],
    ids=["broken", "nan", "not-an-object", "empty"],
)
def test_text_that_is_not_a_json_object_is_faulty(tmp_path, capsys, text):
    code, lines = check(tmp_path, text, capsys)
    assert code == 1
    assert lines[-1].startswith("faulty: ")
    assert lines[0].startswith("FAULT: the list: ")


def test_a_list_that_cannot_be_read_cannot_be_decided(tmp_path, capsys):
    code = run_list.main([str(tmp_path / "no-such-list.json")])
    lines = capsys.readouterr().out.splitlines()
    assert code == 2
    assert lines == [lines[0]]
    assert lines[0].startswith("COULD NOT RUN: the run list ")


COLD_START = [*LEARN, "fleetreserve.cold_start", "--units"]
COLD_REST = ["--days", "150", "--seed", "7", "--base-set", "day-average"]
INTERVALS_COMMAND = [
    "python3.12",
    "-m",
    "fleetreserve.gate_summary",
    "--fit",
    "caltech",
    "jpl",
    "--resamples",
    "2000",
    "--seed",
    "20261116",
    "--from",
    "{run:rehearsal-cold-start}",
]


def test_each_purpose_runs_its_entries_with_its_units_and_its_arguments(tmp_path):
    parsed = run_list.parse(good_list(), name_max=run_list.name_limit(tmp_path))
    rehearsal = run_list.instantiate(parsed, "rehearsal")
    sealed = run_list.instantiate(parsed, "sealed-run")
    assert [s.label for s in rehearsal] == [
        "wp2-truth",
        "rehearsal-cold-start",
        "rehearsal-intervals",
    ]
    assert [s.position for s in rehearsal] == [0, 2, 3]
    assert [s.label for s in sealed] == [
        "wp2-truth",
        "sealed-truth",
        "rehearsal-cold-start",
        "rehearsal-intervals",
    ]
    assert rehearsal[0].command == sealed[0].command == tuple(TRUTH)
    assert sealed[1].command == tuple(SEALED_TRUTH)
    assert list(rehearsal[1].command) == [*COLD_START, *SCORED_DEVELOPMENT, *COLD_REST]
    assert list(sealed[2].command) == [
        *COLD_START,
        *SCORED_SEALED,
        *COLD_REST,
        *SEALED_ROUTE,
        "{run:sealed-truth}",
    ]
    assert list(rehearsal[2].command) == list(sealed[3].command) == INTERVALS_COMMAND
    assert (rehearsal[2].setting, rehearsal[2].max_seconds) == ("held", 600.5)
    assert rehearsal[2].after == ("wp2-truth", "rehearsal-cold-start")
    assert rehearsal[1].runs == ()
    assert sealed[2].runs == ((len(sealed[2].command) - 1, "sealed-truth"),)
    assert run_list.fill_runs(sealed[2], {"sealed-truth": "runs.noindex/x-sealed-truth"})[-1] == (
        "runs.noindex/x-sealed-truth"
    )
    folder = "runs.noindex/20261116T000000Z-rehearsal-cold-start"
    filled = run_list.fill_runs(rehearsal[2], {"rehearsal-cold-start": folder})
    assert filled == [*INTERVALS_COMMAND[:-1], folder]
    with pytest.raises(KeyError):
        run_list.fill_runs(rehearsal[2], {})


def test_a_fixed_name_that_looks_like_a_placeholder_is_not_filled_as_one(tmp_path):
    document = good_list()
    document["fixed"]["names"]["base-set"] = "{run:wp2-truth}"
    parsed = run_list.parse(document, name_max=run_list.name_limit(tmp_path))
    (step,) = [s for s in run_list.instantiate(parsed, "rehearsal") if s.position == 2]
    assert run_list.fill_runs(step, {"wp2-truth": "runs.noindex/x-wp2-truth"})[-1] == (
        "{run:wp2-truth}"
    )

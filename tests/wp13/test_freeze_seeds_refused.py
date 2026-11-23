"""The seeds item of `tools/freeze_check.py lists` (reading F12, part (h) of its module text) on a
step whose own command refuses a seed option given the step's other options (the owner's decision
80 of 19 November 2026): today only `ceiling_report --seed` with `--every-day`, which draws no
sample. The made-up repositories are those of `test_freeze_items.py` (its builder reused), with
outcome O2's two `ceiling_report` steps in the run list: the project's own when its list holds
them, otherwise the two entries of the run list on branch `wp12-o2-two` at 1c0bc7d, copied here
unchanged (`O2_ENTRIES`). No test reads the project's own data or runs folder.

1. The O2 steps as committed, each with `--every-day` and no `--seed`, pass the item in both
   purposes, and nothing in the made-up sealed folder is opened, listed or scanned.
2. The same step without `--every-day` and without `--seed` fails: the seed is left at the
   command's own default, as before.
3. Unchanged: a literal seed on an O2 step fails; a shortened seed option and a seed option given
   twice on an O2 step fail; a seed left at its default on a step whose command accepts it
   (`compare_run --no-pause-seed`) fails; a command with two seed options, one declared refused,
   still needs the other.
4. The refusal is the command's own (drift): `ceiling_report`'s parser and `seed_refusal` refuse
   `--seed` with `--every-day` and accept each alone, its `main` refuses through `seed_refusal`,
   and every declaration of a run-list command is enforced by that command. A made-up command
   declaring an exclusion that neither its parser nor its check enforces is caught; one whose
   parser refuses the pair is taken as refusing it; a step whose own arguments its parser refuses
   is not taken as refusing.
"""

import argparse
import copy
import dataclasses
import importlib
import sys
import types
from pathlib import Path

import pytest
from sealed_watch import watch

from fleetreserve import ceiling_report
from fleetreserve import run_list as rl

TOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TOP / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
freeze_check = importlib.import_module("freeze_check")
items = importlib.import_module("test_freeze_items")

RUN_LIST = items.RUN_LIST
MODULE = "fleetreserve.ceiling_report"
PURPOSES = ("rehearsal", "sealed-run")
DEFAULT_LINE = "leaves ceiling_report's --seed at the command's own default, a seed not fixed"


def _o2(label, tables, truth, extra, setting):
    command = ["uv", "run", "--frozen", "--group", "learn", "--group", "baselines", "python"]
    command += ["-m", MODULE, "--units", "{units}", "--every-day", "--rho", "0", "--workers", "4"]
    command += ["--tables", tables, "--truth-run", f"{{run:{truth}}}"]
    return {
        "label": label,
        "only": None,
        "command": command,
        "units": "nine",
        "extra": {"sealed-run": extra},
        "setting": setting,
        "max_seconds": 21600,
        "after": [truth],
    }


# The two entries of `run-lists/run-list.json` on branch wp12-o2-two at 1c0bc7d, unchanged.
O2_ENTRIES = [
    _o2(
        "o2-rho0",
        "data/derived/truth",
        "dev-truth",
        ["--held-out-truth-run", "{run:sealed-truth}"],
        "main",
    ),
    _o2(
        "o2-held-rho0",
        "data/derived/truth-held",
        "dev-truth-held",
        [
            "--held-out-tables",
            "data/derived/truth-heldout-held",
            "--held-out-truth-run",
            "{run:sealed-truth-held}",
        ],
        "held",
    ),
]


def is_o2(found: dict) -> bool:
    return MODULE in found["command"]


def with_o2(document: dict) -> dict:
    """`document` holding the O2 steps: the project's own, or `O2_ENTRIES` added."""
    if not any(is_o2(e) for e in document["entries"]):
        document["entries"].extend(copy.deepcopy(O2_ENTRIES))
    return document


def o2_text() -> dict:
    return with_o2(items.run_list_text())


def labels(document: dict) -> list[str]:
    return [e["label"] for e in document["entries"] if is_o2(e)]


def where(document: dict, label: str, purpose: str) -> str:
    k = next(k for k, e in enumerate(document["entries"]) if e["label"] == label)
    return f"entry {k + 1} ({label}), {purpose}"


# --- the item end to end, on a made-up repository ---------------------------------------------


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = items.build(tmp_path_factory.mktemp("freeze-seeds"))
    items.edit(root, RUN_LIST, with_o2)
    return root


@pytest.fixture
def repo(base, tmp_path):
    root = tmp_path / "repo"
    items.shutil.copytree(base, root, symlinks=True)
    return root


def test_the_committed_o2_steps_with_every_day_and_no_seed_pass_the_item(repo, capsys):
    document = items.read_json(repo, RUN_LIST)
    assert labels(document) == ["o2-rho0", "o2-held-rho0"]
    for found in document["entries"]:
        if is_o2(found):
            assert "--every-day" in found["command"]
            assert not any("seed" in p for p in (*found["command"], *found["extra"]["sealed-run"]))
    with watch(repo / items.SEALED) as seen:
        status, lines = items.run("lists", repo, capsys)
    assert seen == []
    assert status == 0, lines
    assert len(lines) == 1
    assert lines[0].startswith("held-out list, run matrix, seeds: ")
    assert "the 4 seed options left out of steps whose own command refuses them there" in lines[0]


def test_a_literal_seed_on_an_o2_step_still_fails(repo, capsys):
    def literal(document):
        items.entry(document, "o2-rho0")["command"] += ["--seed", "7"]

    items.edit(repo, RUN_LIST, literal)
    document = items.read_json(repo, RUN_LIST)
    status, lines = items.run("lists", repo, capsys)
    assert status == 1
    for purpose in PURPOSES:
        assert (
            f"FAILED: {where(document, 'o2-rho0', purpose)}: its --seed is '7', not one of the run "
            "list's fixed seeds {seed:<name>}: a literal seed"
        ) in lines


# --- the item's seeds part on a parsed run list ---------------------------------------------------


def parsed(document: dict, tmp_path: Path):
    return rl.parse(document, name_max=rl.name_limit(tmp_path))


def changed(run_list, label: str, change):
    """`run_list` with the entry `label` changed by `change(command, extra) -> (command, extra)`;
    built past `run_list.parse`, so the seeds part alone decides (the run list's own rules on an
    O2 step's options, rule 12 on branch wp12-o2-two, are not the seeds item's)."""
    entries = []
    for found in run_list.entries:
        if found.label == label:
            extra = {p: list(a) for p, a in found.extra.items()}
            command, extra = change(list(found.command), extra)
            found = dataclasses.replace(
                found,
                command=tuple(command),
                extra={p: tuple(a) for p, a in extra.items()},
            )
        entries.append(found)
    return dataclasses.replace(run_list, entries=tuple(entries))


def seeds_lines(run_list) -> list[str]:
    lines: list[str] = []
    found = freeze_check.Findings(lines.append)
    freeze_check._seeds(freeze_check._import_route(), run_list, found)
    return lines


@pytest.fixture
def o2_list(tmp_path):
    document = o2_text()
    return document, parsed(document, tmp_path)


def test_the_o2_steps_give_no_seeds_fault(o2_list):
    _, run_list = o2_list
    assert seeds_lines(run_list) == []


def test_an_o2_step_without_every_day_and_without_seed_fails(o2_list):
    document, run_list = o2_list
    lines = seeds_lines(
        changed(run_list, "o2-rho0", lambda c, e: ([p for p in c if p != "--every-day"], e))
    )
    assert lines == [
        f"FAILED: {where(document, 'o2-rho0', purpose)}: {DEFAULT_LINE} in the run list"
        for purpose in PURPOSES
    ]


def test_every_day_given_in_one_purpose_only_frees_that_purpose_alone(o2_list):
    document, run_list = o2_list

    def sealed_only(command, extra):
        extra["sealed-run"] = [*extra.get("sealed-run", []), "--every-day"]
        return [p for p in command if p != "--every-day"], extra

    lines = seeds_lines(changed(run_list, "o2-rho0", sealed_only))
    assert lines == [
        f"FAILED: {where(document, 'o2-rho0', 'rehearsal')}: {DEFAULT_LINE} in the run list"
    ]


@pytest.mark.parametrize("short", ["--every", "--every-d", "--every-da"])
def test_a_shortened_every_day_does_not_free_the_seed_of_an_o2_step(o2_list, short):
    """The command's parser reads each of these as `--every-day`, but the exemption needs the
    option given in full: the seed is still left at the command's default, in both purposes."""
    document, run_list = o2_list

    def shortened(command, extra):
        return [short if p == "--every-day" else p for p in command], extra

    step = next(e for e in run_list.entries if e.label == "o2-rho0")
    assert "--every-day" in step.command and short not in step.command
    swapped = changed(run_list, "o2-rho0", shortened)
    step = next(e for e in swapped.entries if e.label == "o2-rho0")
    assert short in step.command and "--every-day" not in step.command
    lines = seeds_lines(swapped)
    assert lines == [
        f"FAILED: {where(document, 'o2-rho0', purpose)}: {DEFAULT_LINE} in the run list"
        for purpose in PURPOSES
    ]


def test_a_shortened_seed_option_on_an_o2_step_still_fails(o2_list):
    document, run_list = o2_list
    lines = seeds_lines(changed(run_list, "o2-rho0", lambda c, e: ([*c, "--see", "1"], e)))
    for purpose in PURPOSES:
        assert (
            f"FAILED: {where(document, 'o2-rho0', purpose)}: its argument '--see' shortens --seed"
        ) in lines


def test_a_seed_given_twice_on_an_o2_step_still_fails(o2_list):
    document, run_list = o2_list
    twice = ["--seed", "{seed:o3}", "--seed", "{seed:ledger}"]
    lines = seeds_lines(changed(run_list, "o2-rho0", lambda c, e: ([*c, *twice], e)))
    for purpose in PURPOSES:
        assert (
            f"FAILED: {where(document, 'o2-rho0', purpose)}: gives --seed 2 times, two seeds for "
            "one draw"
        ) in lines


def test_a_seed_left_at_its_default_where_the_command_accepts_it_still_fails(o2_list):
    document, run_list = o2_list

    def left(command, extra):
        at = command.index("--no-pause-seed")
        return command[:at] + command[at + 2 :], extra

    lines = seeds_lines(changed(run_list, "compare", left))
    assert lines == [
        f"FAILED: {where(document, 'compare', purpose)}: leaves compare_run's "
        "--no-pause-seed at the command's own default, a seed not fixed in the run list"
        for purpose in PURPOSES
    ]


# --- the refusal is the command's own -------------------------------------------------------------

SEEDS = (1, 20261122)


def test_ceiling_reports_parser_and_check_refuse_seed_with_every_day_alone():
    def refusal(*argv):
        return ceiling_report.seed_refusal(ceiling_report.parser().parse_args(list(argv)))

    assert ceiling_report.SEED_REFUSED_WITH == {"--seed": ("--every-day",)}
    assert refusal("--every-day", "--seed", "1") == ceiling_report.SEED_WITH_EVERY_DAY
    assert refusal("--seed", "1") is None
    assert refusal("--every-day") is None


def test_ceiling_reports_main_refuses_through_its_seed_refusal(monkeypatch, capsys):
    monkeypatch.setattr(ceiling_report, "seed_refusal", lambda args: "made-up refusal")
    assert ceiling_report.main(["--seed", "1"]) == 2
    assert capsys.readouterr().out == "COULD NOT RUN: made-up refusal\n"


def test_every_declaration_of_a_run_list_command_is_enforced_by_that_command():
    declaring = sorted(m for m in rl.step_modules() if freeze_check._declares_refusal(m))
    assert "ceiling_report" in declaring
    for module in declaring:
        command = importlib.import_module(f"{rl.PACKAGE}.{module}")
        for option, excluders in command.SEED_REFUSED_WITH.items():
            for excluder in excluders:
                assert freeze_check._refused_seed(command, option, [excluder], SEEDS) == (
                    True,
                    None,
                ), (module, option, excluder)
            assert freeze_check._refused_seed(command, option, [], SEEDS) == (False, None)


def made_up(*, check=None, exclusive=False, options=("--seed",), declared=None):
    """A made-up command module: its parser has `options` (each an int, default None) and
    `--every-day`, mutually exclusive with `--seed` when `exclusive`; its check is `check`."""
    command = types.ModuleType("made_up_command")

    def parser():
        found = argparse.ArgumentParser()
        group = found.add_mutually_exclusive_group() if exclusive else found
        group.add_argument("--every-day", action="store_true")
        for option in options:
            group.add_argument(option, type=int, default=None)
        return found

    command.parser = parser
    command.SEED_REFUSED_WITH = declared or {"--seed": ("--every-day",)}
    if check is not None:
        command.seed_refusal = check
    return command


def enforcing(args):
    return "refused" if args.every_day and args.seed is not None else None


def test_a_made_up_declaration_its_command_does_not_enforce_is_caught():
    for command in (made_up(check=lambda args: None), made_up()):
        refused, fault = freeze_check._refused_seed(command, "--seed", ["--every-day"], SEEDS)
        assert not refused
        assert fault == (
            "made_up_command declares --seed refused with --every-day (SEED_REFUSED_WITH), but "
            "neither its parser nor its seed_refusal refuses --seed 1 with the step's other "
            "arguments: a seed left at the command's own default, not fixed in the run list"
        )


def test_a_made_up_command_whose_parser_or_check_refuses_the_pair_is_refusing():
    for command in (made_up(exclusive=True), made_up(check=enforcing)):
        assert freeze_check._refused_seed(command, "--seed", ["--every-day"], SEEDS) == (True, None)
        assert freeze_check._refused_seed(command, "--seed", [], SEEDS) == (False, None)


def test_a_check_refusing_one_fixed_seed_only_is_not_refusing():
    command = made_up(check=lambda args: "refused" if args.seed == 1 else None)
    refused, fault = freeze_check._refused_seed(command, "--seed", ["--every-day"], SEEDS)
    assert not refused
    assert "refuses --seed 20261122 with the step's other arguments" in fault


def test_a_step_whose_own_arguments_its_parser_refuses_is_not_refusing():
    command = made_up(check=enforcing)
    refused, fault = freeze_check._refused_seed(
        command, "--seed", ["--every-day", "--made-up", "1"], SEEDS
    )
    assert not refused
    assert fault == (
        "made_up_command's parser refuses the step's own arguments, so whether it refuses --seed "
        "with --every-day is not shown: a seed left at the command's own default, not fixed in "
        "the run list"
    )


def test_a_second_seed_option_not_declared_is_still_needed():
    command = made_up(check=lambda args: "refused" if args.every_day else None)
    command.parser = made_up(options=("--seed", "--split-seed")).parser
    assert freeze_check._refused_seed(command, "--seed", ["--every-day"], SEEDS) == (True, None)
    assert freeze_check._refused_seed(command, "--split-seed", ["--every-day"], SEEDS) == (
        False,
        None,
    )

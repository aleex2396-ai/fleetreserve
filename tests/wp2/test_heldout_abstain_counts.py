"""The counts over the held-out units name a unit the held-out truth build abstained, on the made-up
repository, tag and held-out build of `test_heldout_abstain` (OSL_T scored, korea-visits abstained).

1. `corner_run`: the line over all units says how many abstained held-out units were not checked
   and count as not passing; a run with no abstained unit has no such clause.
2. The commands whose summaries hold no count over units (`learn_run`, `compare_run`, `ledger_run`,
   `o3_run`, `cold_start`) leave the unit's count to their first lines (`abstention_lines`), tested
   in `test_heldout_abstain`; this file adds no figure for them.
"""

from test_heldout_abstain import (  # noqa: F401
    ABSTAINED,
    CORNER_NAMES,
    DIAL,
    UNITS,
    _scoring_arguments,
    builds,
    repo,
    setup,
)

from fleetreserve import corner_run, corner_summary


def test_the_corner_line_over_all_units_counts_the_abstained_unit(repo, setup, capsys):  # noqa: F811
    s = setup
    arguments = _scoring_arguments(s, UNITS, CORNER_NAMES, "--dial", DIAL)
    arguments += ["--methods", "fixed-plan", "llf"]
    assert corner_run.main(arguments, root=repo) in (0, 1)
    shown = capsys.readouterr().out.splitlines()
    head = corner_summary.HEADINGS[True][2]
    (line,) = [x for x in shown if x.startswith(head)]
    assert line.endswith(
        "1 abstained held-out unit(s) not checked, counted as not passing, named above."
    ), line


def test_the_corner_line_without_an_abstained_unit_has_no_such_clause():
    found = {"pairs": 4, "pairs_agreed_strict": 4, "pairs_agreed": 4, "not_checked": []}
    head = corner_summary.HEADINGS[True][2]
    line = corner_summary._total_line(head, {**found, "agreed": True}, True, {})
    assert "abstained" not in line
    assert corner_summary._total_line(head, {**found, "agreed": True}, True, {"x": "r"}).endswith(
        "agreed; 1 abstained held-out unit(s) not checked, counted as not passing, named above."
    )

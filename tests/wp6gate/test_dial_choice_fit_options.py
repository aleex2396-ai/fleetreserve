"""The refusals of `dial_choice --what-if-fit-seed` and `--what-if-fit-levels` (the options of the
neural hazard family's fit, readings W5 and W6), on the made-up gate units of
`test_dial_choice_what_if` and the empirical model, which none of them may be given with.

1. The empirical model is not of the neural hazard family: exit 2 with a plain message, whichever
   option is given, and nothing is written.
2. A negative seed: exit 2.
3. `--what-if-fit-levels grid` with `--what-if-kappa` at a level off the grid: exit 2 (a network
   fitted at the plan's six levels has no output at 0.6).
4. `--what-if-fit-levels grid` with a matrix whose levels are not the plan's six (five, or the
   committed four): exit 2.
5. A level choice other than `all` and `grid` is refused by the argument parser (exit 2).
6. The module text names both options.
"""

import json

import pytest
from gate_fake import COMMIT, matrix_document, matrix_document_six, write_build
from test_dial_choice_what_if import NAMES, SPLITS, units

from fleetreserve import dial_choice, gate_run, run_verify


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    return run, tables, matrix


def _main(build, tmp_path, *extra, matrix=None) -> int:
    run, tables, own = build
    arguments = ["--model", "empirical", "--matrix", str(matrix or own), "--tables", str(tables)]
    arguments += ["--truth-run", str(run), "--splits", str(SPLITS), *extra]
    out = tmp_path / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(run_verify, "_check_commit", lambda folder, commit: None)
        monkeypatch.setattr(gate_run, "_head_commit", lambda: COMMIT)
        monkeypatch.setenv("FLEETRESERVE_RUN_DIR", str(out))
        code = dial_choice.main(arguments)
    assert list(out.iterdir()) == []
    return code


@pytest.mark.parametrize(
    "option",
    [
        ("--what-if-fit-seed", "1"),
        ("--what-if-fit-levels", "all"),
        ("--what-if-fit-levels", "grid"),
    ],
)
def test_the_empirical_model_is_not_of_the_hazard_family(build, tmp_path, capsys, option):
    assert _main(build, tmp_path, *option) == 2
    text = capsys.readouterr().out
    assert (
        "COULD NOT RUN" in text
        and "neural hazard family" in text
        and "empirical is not one" in text
    )


def test_a_negative_seed_is_refused(build, tmp_path, capsys):
    assert _main(build, tmp_path, "--what-if-fit-seed", "-1") == 2
    assert "0 or more" in capsys.readouterr().out


def test_grid_cannot_read_a_level_off_the_grid(build, tmp_path, capsys):
    assert _main(build, tmp_path, "--what-if-kappa", "0.6", "--what-if-fit-levels", "grid") == 2
    text = capsys.readouterr().out
    assert "cannot read kappa 0.6" in text and "has no output there" in text


@pytest.mark.parametrize("left_out", ["-k0.05-", "-k0.4-", "-k0.5-"])
def test_grid_needs_a_matrix_with_the_plans_six_levels(build, tmp_path, capsys, left_out):
    names = [n for n in NAMES if left_out not in n]
    matrix = tmp_path / "five-levels.json"
    matrix.write_text(json.dumps(matrix_document_six(names)))
    assert _main(build, tmp_path, "--what-if-fit-levels", "grid", matrix=matrix) == 2
    assert "whose levels are the plan's grid" in capsys.readouterr().out


def test_grid_refuses_the_committed_matrix_that_lists_four_levels(build, tmp_path, capsys):
    names = [n for n in NAMES if "-k0.4-" not in n and "-k0.5-" not in n]
    matrix = tmp_path / "four-levels.json"
    matrix.write_text(json.dumps(matrix_document(names)))
    assert _main(build, tmp_path, "--what-if-fit-levels", "grid", matrix=matrix) == 2
    assert "[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]" in capsys.readouterr().out


def test_another_level_choice_is_refused_by_the_parser(build, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        _main(build, tmp_path, "--what-if-fit-levels", "some")
    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_module_text_names_both_options():
    assert "--what-if-fit-seed" in dial_choice.__doc__
    assert "--what-if-fit-levels" in dial_choice.__doc__

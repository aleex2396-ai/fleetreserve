"""The refusals of `dial_choice --what-if-fit-seeds N1,N2,...` (an average over seeds of the neural
hazard family's networks, `seed_average`), on the made-up gate units of `test_dial_choice_what_if`
and the empirical model; the runs themselves are in `tests/learn/test_dial_choice_seed_average.py`.

1. The empirical model is not of the neural hazard family: exit 2 with a plain message, and
   nothing is written.
2. Given with `--what-if-fit-seed`: exit 2.
3. One seed, a seed listed twice, a negative seed, a seed that is not a whole number, an empty
   list: exit 2, each with its own message.
4. `parse_seeds` puts the seeds in increasing order; the module text names the option.
"""

import json

import pytest
from gate_fake import COMMIT, matrix_document_six, write_build
from test_dial_choice_what_if import NAMES, SPLITS, units

from fleetreserve import dial_choice, gate_run, run_verify
from fleetreserve.gate_config import MatrixError


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    folder = tmp_path_factory.mktemp("build")
    run, tables = write_build(folder, units())
    matrix = folder / "matrix.json"
    matrix.write_text(json.dumps(matrix_document_six(NAMES)))
    return run, tables, matrix


def _main(build, tmp_path, *extra) -> int:
    run, tables, matrix = build
    arguments = ["--model", "empirical", "--matrix", str(matrix), "--tables", str(tables)]
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


def test_the_empirical_model_is_not_of_the_hazard_family(build, tmp_path, capsys):
    assert _main(build, tmp_path, "--what-if-fit-seeds", "1,2") == 2
    text = capsys.readouterr().out
    assert "COULD NOT RUN" in text and "neural hazard family" in text
    assert "empirical is not one" in text


def test_not_given_with_a_single_seed(build, tmp_path, capsys):
    extra = ("--what-if-fit-seed", "1", "--what-if-fit-seeds", "1,2")
    assert _main(build, tmp_path, *extra) == 2
    assert "give --what-if-fit-seed or --what-if-fit-seeds, not both" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("listed", "said"),
    [
        ("1", "averages two or more seeds"),
        ("1,1", "lists a seed twice"),
        ("2,1,2", "lists a seed twice"),
        ("1,-2", "whole numbers of 0 or more"),
        ("1,a", "whole numbers of 0 or more"),
        ("1.5,2", "whole numbers of 0 or more"),
        ("", "whole numbers of 0 or more"),
    ],
)
def test_a_list_that_is_not_two_or_more_distinct_seeds_is_refused(
    build, tmp_path, capsys, listed, said
):
    assert _main(build, tmp_path, "--what-if-fit-seeds", listed) == 2
    text = capsys.readouterr().out
    assert "COULD NOT RUN" in text and said in text


def test_the_seeds_are_put_in_increasing_order():
    assert dial_choice.parse_seeds("3,0, 2,1") == (0, 1, 2, 3)
    with pytest.raises(MatrixError):
        dial_choice.parse_seeds("3")


def test_the_module_text_names_the_option():
    assert "--what-if-fit-seeds" in dial_choice.__doc__

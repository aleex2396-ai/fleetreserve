"""The candidate grid of the tuning and its choice rule (`learn_tune_grid`).

1. The committed grid holds 48 candidates, every combination of its lists, the present settings
   among them; a grid without the present settings, with a repeated value, another key or a bad
   number is refused.
2. The choice rule: the lowest score wins whatever the candidate's size; within 1e-9 of the
   lowest the fewer leaves win, then the larger `min_data_in_leaf`, then the fewer rounds; a
   difference above 1e-9 is not a tie.
3. The rank counts the candidates with a strictly lower score.
"""

import json

import pytest

from fleetreserve.learn_tune_grid import (
    GRID,
    PRESENT,
    GridError,
    Settings,
    choose,
    load_grid,
    rank,
    settings_from,
)


def test_the_committed_grid_holds_the_48_candidates_and_the_present_settings():
    grid = load_grid()
    assert grid.file == "tune-grid.json" == GRID.name
    assert len(grid.candidates) == 48 == len(set(grid.candidates))
    assert {c.num_leaves for c in grid.candidates} == {7, 15, 31, 63}
    assert {c.min_data_in_leaf for c in grid.candidates} == {20, 50, 100, 200}
    assert {(c.learning_rate, c.rounds) for c in grid.candidates} == {
        (0.05, 200),
        (0.05, 400),
        (0.02, 1000),
    }
    assert Settings(15, 20, 0.05, 200) == PRESENT
    assert PRESENT in grid.candidates
    assert len(grid.sha256) == 64
    assert PRESENT.overrides() == {
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "learning_rate": 0.05,
    }


def _grid(tmp_path, **changes):
    document = {
        "about": "made up",
        "num_leaves": [7, 15],
        "min_data_in_leaf": [20],
        "learning_rate_and_rounds": [[0.05, 200]],
        **changes,
    }
    for key, value in list(document.items()):
        if value is None:
            del document[key]
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(document))
    return path


def test_a_small_grid_is_read_in_the_order_of_the_file(tmp_path):
    grid = load_grid(_grid(tmp_path))
    assert grid.candidates == (Settings(7, 20, 0.05, 200), Settings(15, 20, 0.05, 200))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"num_leaves": [7, 31]}, "present settings"),
        ({"num_leaves": [15, 15]}, "twice"),
        ({"num_leaves": []}, "non-empty"),
        ({"num_leaves": [15, 0]}, "whole number"),
        ({"min_data_in_leaf": [20.5]}, "whole number"),
        ({"learning_rate_and_rounds": [[0.05, 200, 1]]}, "pairs"),
        ({"learning_rate_and_rounds": [[0.0, 200]]}, "learning_rate"),
        ({"extra": 1}, "exactly the keys"),
        ({"about": None}, "exactly the keys"),
    ],
)
def test_a_bad_grid_is_refused(tmp_path, changes, message):
    with pytest.raises(GridError, match=message):
        load_grid(_grid(tmp_path, **changes))


def test_a_missing_grid_file_is_refused(tmp_path):
    with pytest.raises(GridError, match="cannot be read"):
        load_grid(tmp_path / "none.json")


def test_settings_are_read_only_from_exactly_the_four_keys():
    assert settings_from(PRESENT.as_dict()) == PRESENT
    with pytest.raises(GridError):
        settings_from({**PRESENT.as_dict(), "seed": 1})
    with pytest.raises(GridError):
        settings_from({"num_leaves": 15})


SMALL = Settings(7, 20, 0.05, 200)
LARGE = Settings(63, 20, 0.05, 200)


def test_the_lowest_score_wins_whatever_the_size():
    assert choose([(SMALL, 2.0), (LARGE, 1.0)]) == LARGE
    assert choose([(LARGE, 1.0), (SMALL, 2.0)]) == LARGE
    assert choose([(SMALL, 1.0), (LARGE, 2.0)]) == SMALL


def test_a_tie_goes_to_the_fewer_leaves():
    # the larger candidate is lower by less than 1e-9: a tie, so the fewer leaves win
    assert choose([(LARGE, 1.0), (SMALL, 1.0 + 5e-10)]) == SMALL
    assert choose([(SMALL, 1.0 + 5e-10), (LARGE, 1.0)]) == SMALL


def test_a_difference_above_the_tolerance_is_not_a_tie():
    assert choose([(LARGE, 1.0), (SMALL, 1.0 + 2e-9)]) == LARGE


def test_a_tie_in_leaves_goes_to_the_larger_min_data_in_leaf():
    few, many = Settings(15, 20, 0.05, 200), Settings(15, 100, 0.05, 200)
    assert choose([(few, 1.0), (many, 1.0)]) == many
    assert choose([(many, 1.0), (few, 1.0)]) == many
    # leaves come first: fewer leaves beat a larger min_data_in_leaf
    assert choose([(Settings(7, 20, 0.05, 200), 1.0), (many, 1.0)]) == Settings(7, 20, 0.05, 200)


def test_a_tie_in_leaves_and_min_data_goes_to_the_fewer_rounds():
    short, long = Settings(15, 50, 0.05, 200), Settings(15, 50, 0.05, 400)
    longest = Settings(15, 50, 0.02, 1000)
    assert choose([(long, 1.0), (longest, 1.0), (short, 1.0)]) == short
    assert choose([(longest, 1.0), (long, 1.0)]) == long
    # min_data_in_leaf comes before the rounds
    assert choose([(short, 1.0), (Settings(15, 100, 0.02, 1000), 1.0)]) == Settings(
        15, 100, 0.02, 1000
    )


def test_ties_are_measured_from_the_lowest_score_only():
    # 1.0, 1.0 + 8e-10 and 1.0 + 1.6e-9: the third is not within 1e-9 of the lowest
    a, b, c = Settings(63, 20, 0.05, 200), Settings(31, 20, 0.05, 200), Settings(7, 20, 0.05, 200)
    assert choose([(a, 1.0), (b, 1.0 + 8e-10), (c, 1.0 + 1.6e-9)]) == b


def test_no_candidate_or_a_score_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError):
        choose([])
    with pytest.raises(ValueError):
        choose([(SMALL, float("nan"))])


def test_the_rank_counts_strictly_lower_scores():
    scored = [(SMALL, 2.0), (LARGE, 1.0), (PRESENT, 2.0)]
    assert rank(scored, LARGE) == 1
    assert rank(scored, SMALL) == 2 == rank(scored, PRESENT)

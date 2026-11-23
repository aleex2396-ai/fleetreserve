"""The tuned models were chosen with a score over 0.05, 0.1, 0.2 and 0.3 (`learn_tune`, reading T1);
plan section 2's grid has two more levels, 0.4 and 0.5, and the score keeps its four.

1. `learn_tune` and `hazard_tune` score over `forecast_empirical.TUNING_KAPPAS`, the home's own
   object, which is the grid's levels up to 0.3, and not over the grid `KAPPAS` (six levels).
2. The module text of `learn_tune` states the reading and its alternative.
"""

import inspect

from fleetreserve import forecast_empirical, hazard_tune, learn_tune

FOUR = (0.05, 0.1, 0.2, 0.3)


def test_the_tuning_scores_read_the_four_tuning_levels_and_not_the_six_of_the_grid():
    assert forecast_empirical.TUNING_KAPPAS == FOUR
    assert len(forecast_empirical.KAPPAS) == 6
    assert learn_tune.TUNING_KAPPAS is forecast_empirical.TUNING_KAPPAS
    assert hazard_tune.TUNING_KAPPAS is forecast_empirical.TUNING_KAPPAS
    assert not hasattr(learn_tune, "KAPPAS") and not hasattr(hazard_tune, "KAPPAS")
    default = inspect.signature(learn_tune.fold_scores).parameters["kappas"].default
    assert default is forecast_empirical.TUNING_KAPPAS


def test_the_module_text_of_learn_tune_states_the_reading_and_its_alternative():
    text = learn_tune.__doc__
    assert "Reading T1" in text and "TUNING_KAPPAS" in text
    assert "Alternative: score over all six levels and tune again" in text

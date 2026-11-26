"""The tree model and the neural model averaged: the forecast trial `mix` (a trial made after the
sealed run, `after_sealed_run`; plan section 7).

Members, fixed here:

- `models.lgbm_tail`, the main forecasting model of plan section 7;
- `models.hazard_tuned_inputs`, the neural variant that plan section 7 names ("the one built on the
  plan's own inputs"), at the settings of the tuned settings file and the one seed of those
  settings, 0 (`models.hazard_tuned`: "the seed is its default (0)"). One network, no seed average.

Fit: each member is fitted as its own module fits it, on the cars tables of `fit_units`, the
training units (both members name `truth_build.TRAIN_UNITS`; the model refuses members whose fit
units differ). Nothing is fitted on a development unit, so no development day, test or
calibration, reaches the fit.

Predict: both members forecast the same frame at the same levels, and the table is
`forecast_average.average_tables` of the two tables: every number is the mean of the two members'
values at the same car and level, with equal weights (one half each). There is no weight, no
search and no tuned setting in this module. A NaN of either member is a NaN of the mix
(`forecast_average`, module text).

The mix reads the same frame as the neural member: it declares that member's extra input columns
and holds its context sources, so `model_inputs.extra_of` and `learn_run.forecast_frame` add the
columns the neural member reads; the tree member reads its own columns of the same frame and
ignores the rest (`models.lgbm.features`).

`make_model` imports the two members inside the function: both need the packages of the `learn`
dependency group, and the tree member's module imports LightGBM before the neural member's module
imports PyTorch (the order the project requires). A command refused before it builds a model never
imports them.
"""

import importlib
from collections.abc import Sequence

import pandas as pd

from fleetreserve.forecast_average import average_tables
from fleetreserve.truth_build import TRAIN_UNITS

MEMBERS = ("lgbm_tail", "hazard_tuned_inputs")


class MixModel:
    name = "mix_tail_inputs"
    look_ahead = False

    def __init__(self, *, tail, neural) -> None:
        if tuple(tail.fit_units) != tuple(neural.fit_units):
            raise ValueError("the mix members are not fitted on the same units")
        if tuple(tail.fit_units) != TRAIN_UNITS:
            raise ValueError(f"the mix members are fitted on the training units {TRAIN_UNITS}")
        self.tail = tail
        self.neural = neural
        self.fit_units = tuple(tail.fit_units)
        # What the neural member reads beyond the cars tables (`model_inputs.extra_of`).
        self.extra_inputs = tuple(getattr(neural, "extra_inputs", ()))
        self.context_sources = getattr(neural, "context_sources", None)
        self.context_files = dict(getattr(neural, "context_files", {}))

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit the tree member, then the neural one, each on `frame` (module text)."""
        self.tail.fit(frame)
        self.neural.fit(frame)

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        levels = [float(k) for k in kappas]
        return average_tables(
            [self.tail.predict(frame, levels), self.neural.predict(frame, levels)]
        )


def make_model(matrix) -> MixModel:
    # One module after the other, in this order: LightGBM, then PyTorch (`hazard_net`).
    tail = importlib.import_module(f"fleetreserve.models.{MEMBERS[0]}")
    neural = importlib.import_module(f"fleetreserve.models.{MEMBERS[1]}")
    return MixModel(tail=tail.make_model(matrix), neural=neural.make_model(matrix))

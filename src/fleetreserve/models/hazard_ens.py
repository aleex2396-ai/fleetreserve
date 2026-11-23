"""Seed ensemble of the tuned neural hazard model (plan sections 7, 13 WP8).

Members: one per seed of `SEEDS` (0, 1, 2, 3, 4). Each is `models.hazard.HazardModel` with the
settings of `learn/hazard-tuned.json` at its seed and every other setting at its default, so the
members are the model of `models.hazard_tuned` fitted with five seeds. The number of members
was fixed at five before any score of this model existed; it is a constant, not a tuned setting.
`make_model` reads the settings with `hazard_tune_settings.load_tuned`, so it refuses where
`models.hazard_tuned` refuses, before anything is fitted.

Fit: the members are fitted one after another on the same cars tables (the fit units of
`models.hazard_tuned`), each as `HazardModel.fit` fits it, on the thread count of its settings
(one). An error in a member's fit is raised; no member is left out.

Forecast, per car:

- the departure distribution is the mean, with equal weights, of the members' survival curves
  over the time grid (the same as the mean of their probabilities of each bin);
- `d_hat`, `d_median` and `survival_h1` to `survival_h8` are read from that mean curve by the code
  the single model uses (`HazardModel.departure_columns`, `hazard_curves.departure_table`),
  with its rules for NaN: an hour is NaN for every car of a set when fewer than
  `min_presence_support` fit rows of the set's site type have a departure at or above it
  (`models.hazard.presence_support_of`), and for a car
  whose probability of being present at its start, read from the mean curve, is below
  `min_presence_probability`. The fit-row counts are those of the first member; every member
  counts the same rows, which `fit` checks;
- `r_hat`, `e_cap` and `e_need` are the plain means over the members of each member's value.

A quantile of the mean curve is in general not the mean of the members' quantiles; the model
gives the former. With one member (`seeds=(0,)`) the table is `models.hazard_tuned`'s, bit for
bit (tested); the model built by `make_model` has five.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from fleetreserve.forecast_model import empty_table
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.hazard_tune_settings import load_tuned
from fleetreserve.models.hazard import HazardModel, SetCurves, forecast_table
from fleetreserve.models.hazard_tuned import required
from fleetreserve.models.lgbm import matrix_kappas
from fleetreserve.truth_build import TRAIN_UNITS

SEEDS = (0, 1, 2, 3, 4)


class HazardEnsembleModel:
    name = "hazard_ens"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(
        self, *, rho: float, kappas: Sequence[float], tuned: Setting, seeds: Sequence[int] = SEEDS
    ) -> None:
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"the seeds of an ensemble are distinct and at least one: {seeds}")
        self.rho = rho
        self.tuned = required(tuned)
        self.seeds = tuple(int(seed) for seed in seeds)
        self.members = [
            HazardModel(rho=rho, kappas=kappas, settings=tuned.hazard(seed)) for seed in self.seeds
        ]
        self.fitted = False

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit every member, one after another (module text)."""
        self.fitted = False
        for member in self.members:
            member.fit(frame)
        first = self.members[0].presence_support
        for member in self.members[1:]:
            if any(not np.array_equal(first[k], member.presence_support[k]) for k in first):
                raise ValueError("hazard_ens: the members counted different fit rows")
        self.fitted = True

    def mean_curves(self, frame: pd.DataFrame) -> tuple[SetCurves, list[SetCurves]]:
        """The sets of a non-empty `frame` with the mean of the members' curves (its `energy` is
        the first member's, which `predict` does not read), and each member's own
        (`HazardModel.set_curves`)."""
        found = [member.set_curves(frame) for member in self.members]
        first = found[0]
        survival = [
            np.mean(np.stack([f.survival[k] for f in found]), axis=0)
            for k in range(len(first.rows))
        ]
        return SetCurves(first.n, first.rows, first.kinds, survival, first.energy), found

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("predict before fit")
        asked = [member.check_levels(kappas) for member in self.members][0]
        if len(frame) == 0:
            return empty_table()
        mean, found = self.mean_curves(frame)
        departure = self.members[0].departure_columns(mean, asked)
        each = [m.energy_columns(f, asked) for m, f in zip(self.members, found, strict=True)]
        energy = {
            k: {
                name: np.mean(np.stack([e[k][name] for e in each]), axis=0)
                for name in ("r_hat", "e_cap", "e_need")
            }
            for k in asked
        }
        return forecast_table(frame, kappas, self.rho, departure, energy)


def make_model(matrix) -> HazardEnsembleModel:
    tuned = load_tuned()
    return HazardEnsembleModel(rho=matrix.rho, kappas=matrix_kappas(matrix), tuned=tuned)

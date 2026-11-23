"""Tuned neural hazard model: `models.hazard` at the settings of the tuned settings file (plan
sections 7, 13 WP8; the plan's main forecasting model).

Everything of `models.hazard` holds (the fit units, the inputs, the sets, the table, the
levels). The one change: `width`, `learning_rate`, `kl_weight` (the divergence weight of the day
factor) and `epochs` are those of `learn/hazard-tuned.json`, the candidate
`fleetreserve.hazard_tune` chose on the training sites alone, each training site held out in turn;
no development unit is read for it. Every other setting is the default of `HazardSettings`, and
the seed is its default (0): the tuning averaged its scores over the seeds of the grid file, and
the seed is not a setting. In particular the two settings of the survival-bound rule,
`min_presence_support` and `min_presence_probability`, are not in the grid (the score of the
tuning is the pinball loss of `d_hat`, which they never change), so they stay at their defaults.

`make_model` reads the settings with `hazard_tune_settings.load_tuned` and so refuses to build the
model, before anything is fitted, when the settings file is absent, cannot be used, records another
grid than the committed one or holds settings that are not a candidate of it (`learn_run` then
ends with exit 2, could not run). The class cannot be built
without settings either.
"""

from collections.abc import Sequence

from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.hazard_tune_settings import load_tuned
from fleetreserve.models.hazard import HazardModel
from fleetreserve.models.lgbm import matrix_kappas


def required(setting) -> Setting:
    if not isinstance(setting, Setting):
        raise ValueError("the tuned model is built with the settings of the tuned settings file")
    return setting


class HazardTunedModel(HazardModel):
    name = "hazard_tuned"

    def __init__(self, *, rho: float, kappas: Sequence[float], tuned: Setting) -> None:
        super().__init__(rho=rho, kappas=kappas, settings=required(tuned).hazard())
        self.tuned = tuned


def make_model(matrix) -> HazardTunedModel:
    return HazardTunedModel(rho=matrix.rho, kappas=matrix_kappas(matrix), tuned=load_tuned())

"""Tuned pooled LightGBM: `models.lgbm_ctx` at the settings of the tuned settings file (plan
sections 7, 13 WP8; a learned departure model of section 7).

Everything of `models.lgbm_ctx` holds (the fit units, the input columns `models.lgbm_ctx.COLUMNS`,
the boosters' quantities, levels and rows, the clipping at 0, the survival columns left NaN). The
one change: `num_leaves`, `min_data_in_leaf`, `learning_rate` and the number of rounds are those
of `learn/tuned.json`, the candidate `fleetreserve.learn_tune` chose on the training sites alone,
each training site held out in turn; no development unit is read for it. Every other LightGBM
parameter stays as `models.lgbm.PARAMETERS` has it (the same seed and deterministic mode).

`make_model` reads the settings with `learn_tune_settings.load_tuned` and so refuses to build the
model, before anything is fitted, when the settings file is absent or records another grid than
the committed one (`learn_run` then ends with exit 2, could not run). The class cannot be built
without settings either.
"""

from collections.abc import Sequence

from fleetreserve.learn_tune_grid import Settings
from fleetreserve.learn_tune_settings import load_tuned
from fleetreserve.models.lgbm import matrix_kappas
from fleetreserve.models.lgbm_ctx import LgbmCtxModel


def required(settings) -> Settings:
    if not isinstance(settings, Settings):
        raise ValueError("a tuned model is built with the settings of the tuned settings file")
    return settings


class LgbmTunedModel(LgbmCtxModel):
    name = "lgbm_tuned"

    def __init__(self, *, rho: float, kappas: Sequence[float], settings: Settings) -> None:
        super().__init__(rho=rho, kappas=kappas, settings=required(settings))


def make_model(matrix) -> LgbmTunedModel:
    return LgbmTunedModel(rho=matrix.rho, kappas=matrix_kappas(matrix), settings=load_tuned())

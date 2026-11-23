"""Pooled LightGBM with the lower tail of the user's history and the survival-tightened bound:
`models.lgbm_tail` plus the conditional-fit rule (plan sections 1, 7, 13 WP8; a learned departure
model of section 7).

Every column `models.lgbm_tail` fills is filled with the same number (the same boosters on the
same columns, `models.lgbm_tail.COLUMNS`). The one change is that of `models.lgbm_surv`: the
columns `survival_h1` to `survival_h8` hold the bound of `models.conditional_fit`, fitted on the
same training table and on `models.lgbm_tail.COLUMNS`. `models.lgbm_tail` is kept as it is.
"""

from fleetreserve.models.lgbm import matrix_kappas
from fleetreserve.models.lgbm_surv import LgbmSurvModel
from fleetreserve.models.lgbm_tail import COLUMNS as TAIL_COLUMNS


class LgbmTailSurvModel(LgbmSurvModel):
    name = "lgbm_tail_surv"
    COLUMNS = TAIL_COLUMNS


def make_model(matrix) -> LgbmTailSurvModel:
    return LgbmTailSurvModel(rho=matrix.rho, kappas=matrix_kappas(matrix))

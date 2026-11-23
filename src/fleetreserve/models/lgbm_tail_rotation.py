"""Pooled LightGBM with the lower tail of the user's history, fitted without the scored training
units: `models.lgbm_tail` for a rotation what-if (plan sections 3, 7, 13 WP8; a learned departure
model of section 7).

Everything of `models.lgbm_tail` holds: the input columns (`models.lgbm_tail.COLUMNS`), the settings
(`models.lgbm.PARAMETERS`, `ROUNDS`), the level and quantity of every booster, the rows each is
fitted on, the clipping at 0, the survival columns left NaN and the table format. The code is that
of `models.lgbm_tail` (this class only narrows the fit units). The one change is which training
units the model is fitted on.

Rule. The plan fits on all eight training units (`truth_build.TRAIN_UNITS`) and scores development
units. A rotation what-if scores a training unit (the matrix field `rotation_whatif`,
`gate_config`), and a unit is never scored by a model fitted on it. So `fit_units` is
`truth_build.TRAIN_UNITS` minus the gate units of the run matrix the model is made for
(`make_model(matrix)`), in the order of `TRAIN_UNITS`. `fit` keeps only the rows of those units
(`unit` column), so a frame that also holds a left-out unit's rows gives the same model as one
that does not; `learn_run.check_model` refuses a model whose `fit_units` meet the gate units, and
this model passes it because its fit units exclude them.

For a matrix whose gate units hold no training unit (the committed matrices), the fit units are
all eight and the forecast table is that of `models.lgbm_tail`, byte for byte. A matrix that leaves
no training unit to fit on is refused (`MatrixError`, exit 2 of the commands).

Limit. When several training units are gate units of one matrix, all of them are left out
together, and the model is fitted on the rest: one unit at a time needs one matrix per unit (each
run then fits on seven units, not on the same set). The units left out are the matrix's, not the
model's choice, so two runs with different matrices are fitted on different units and their
figures are not on the same footing as a run of the plan.
"""

from collections.abc import Sequence

import pandas as pd

from fleetreserve.gate_config import MatrixError
from fleetreserve.models.lgbm import matrix_kappas
from fleetreserve.models.lgbm_tail import LgbmTailModel
from fleetreserve.truth_build import TRAIN_UNITS


def fit_units_for(gate_units: Sequence[str]) -> tuple[str, ...]:
    """The training units that are not in `gate_units`, in the order of `TRAIN_UNITS`; raises
    `MatrixError` when none is left."""
    kept = tuple(u for u in TRAIN_UNITS if u not in gate_units)
    if not kept:
        raise MatrixError(
            f"the matrix scores every training unit {list(TRAIN_UNITS)}, so no training unit is "
            "left to fit the model on"
        )
    return kept


class LgbmTailRotationModel(LgbmTailModel):
    name = "lgbm_tail_rotation"

    def __init__(self, *, rho: float, kappas: Sequence[float], fit_units: Sequence[str]) -> None:
        super().__init__(rho=rho, kappas=kappas)
        units = tuple(fit_units)
        if not units or any(u not in TRAIN_UNITS for u in units) or len(set(units)) != len(units):
            raise ValueError(
                f"fit_units must be distinct, non-empty training units {list(TRAIN_UNITS)}, "
                f"got {list(units)}"
            )
        self.fit_units = units

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit as `models.lgbm_tail` on the rows of `fit_units` only (module text)."""
        rows = frame[frame["unit"].isin(self.fit_units)].reset_index(drop=True)
        if rows.empty:
            raise ValueError(f"the frame holds no row of the fit units {list(self.fit_units)}")
        super().fit(rows)


def make_model(matrix) -> LgbmTailRotationModel:
    return LgbmTailRotationModel(
        rho=matrix.rho,
        kappas=matrix_kappas(matrix),
        fit_units=fit_units_for(matrix.gate_units),
    )

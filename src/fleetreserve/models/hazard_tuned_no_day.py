"""The tuned neural hazard model with no day factor (plan section 7, "Ablations of the
hazard model"; section 13 WP8): `models.hazard_tuned` with one change, the part
`no_day_factor` of `hazard_parts.PARTS`, which `fleetreserve.hazard_tune` also measures.

The day factor is off (`HazardSettings.day_factor`): the network has no prior or posterior
network and the loss no divergence term, so the tuned divergence weight has no effect.

Everything else is that of `models.hazard_tuned`: the fit units, the inputs, the sets, the table,
the levels, the settings of `learn/hazard-tuned.json` read by `hazard_tune_settings.load_tuned`
(so `make_model` refuses, before anything is fitted, where `models.hazard_tuned` refuses), every
other setting at its default and the seed 0.
"""

from collections.abc import Sequence

from fleetreserve.hazard_parts import PARTS
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.hazard_tune_settings import load_tuned
from fleetreserve.models.hazard_tuned import HazardTunedModel
from fleetreserve.models.lgbm import matrix_kappas

PART = "no_day_factor"


class HazardTunedNoDayModel(HazardTunedModel):
    name = "hazard_tuned_no_day"

    def __init__(self, *, rho: float, kappas: Sequence[float], tuned: Setting) -> None:
        super().__init__(rho=rho, kappas=kappas, tuned=tuned)
        self.settings = tuned.hazard(**PARTS[PART])


def make_model(matrix) -> HazardTunedNoDayModel:
    tuned = load_tuned()
    return HazardTunedNoDayModel(rho=matrix.rho, kappas=matrix_kappas(matrix), tuned=tuned)

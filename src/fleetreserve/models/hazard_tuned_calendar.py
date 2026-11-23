"""The tuned neural hazard model with calendar-only inputs (plan section 7, "Ablations of the
hazard model"; section 13 WP8): `models.hazard_tuned` with one change, the part
`calendar_only_inputs` of `hazard_parts.PARTS`, which `fleetreserve.hazard_tune` also measures.

Calendar-only inputs (`HazardSettings.inputs` is `hazard_parts.CALENDAR_INPUTS`): every input
but the date parts and `public_holiday` is a constant 0 for the network (`hazard_data`, module
text). The set encoder, the site embedding and the day factor stay on.

Everything else is that of `models.hazard_tuned`: the fit units, the input columns of the table
(of which the network uses only those named above), the sets, the table, the levels, the
settings of `learn/hazard-tuned.json` read by `hazard_tune_settings.load_tuned` (so
`make_model` refuses, before anything is fitted, where `models.hazard_tuned` refuses), every
other setting at its default and the seed 0.
"""

from collections.abc import Sequence

from fleetreserve.hazard_parts import PARTS
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.hazard_tune_settings import load_tuned
from fleetreserve.models.hazard_tuned import HazardTunedModel
from fleetreserve.models.lgbm import matrix_kappas

PART = "calendar_only_inputs"


class HazardTunedCalendarModel(HazardTunedModel):
    name = "hazard_tuned_calendar"

    def __init__(self, *, rho: float, kappas: Sequence[float], tuned: Setting) -> None:
        super().__init__(rho=rho, kappas=kappas, tuned=tuned)
        self.settings = tuned.hazard(**PARTS[PART])


def make_model(matrix) -> HazardTunedCalendarModel:
    tuned = load_tuned()
    return HazardTunedCalendarModel(rho=matrix.rho, kappas=matrix_kappas(matrix), tuned=tuned)

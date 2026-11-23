"""The tuned neural hazard model with no set encoder (plan section 7, "Ablations of the
hazard model"; section 13 WP8): `models.hazard_tuned` with one change, the part
`no_set_encoder` of `hazard_parts.PARTS`, which `fleetreserve.hazard_tune` also measures.

The set encoder is off (`HazardSettings.set_encoder`; `hazard_net`, module text): a car's
encoding, and so its energy columns, depend on its own inputs only. The fleet vector is then the
mean of the encodings of the set's cars, and the day factor's prior is read from it, so with the
day factor on (as here) a car's departure columns can still change with the other cars of its
site-day.

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

PART = "no_set_encoder"


class HazardTunedNoSetModel(HazardTunedModel):
    name = "hazard_tuned_no_set"

    def __init__(self, *, rho: float, kappas: Sequence[float], tuned: Setting) -> None:
        super().__init__(rho=rho, kappas=kappas, tuned=tuned)
        self.settings = tuned.hazard(**PARTS[PART])


def make_model(matrix) -> HazardTunedNoSetModel:
    tuned = load_tuned()
    return HazardTunedNoSetModel(rho=matrix.rho, kappas=matrix_kappas(matrix), tuned=tuned)

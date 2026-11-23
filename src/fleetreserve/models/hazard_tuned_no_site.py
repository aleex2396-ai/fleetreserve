"""The tuned neural hazard model with no site embedding (plan section 7, "Ablations of the
hazard model"; section 13 WP8): `models.hazard_tuned` with one change, the part
`no_site_embedding` of `hazard_parts.PARTS`, which `fleetreserve.hazard_tune` also measures.

The site embedding is off (`HazardSettings.site_embedding`): the network has no site vectors and
the hazard head is given no site input. The site is still read where `models.hazard` reads it
apart from that: the inputs made from it (`site_type` and the holiday flags,
`learn_features.with_context`) and the grouping of the cars into sets.

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

PART = "no_site_embedding"


class HazardTunedNoSiteModel(HazardTunedModel):
    name = "hazard_tuned_no_site"

    def __init__(self, *, rho: float, kappas: Sequence[float], tuned: Setting) -> None:
        super().__init__(rho=rho, kappas=kappas, tuned=tuned)
        self.settings = tuned.hazard(**PARTS[PART])


def make_model(matrix) -> HazardTunedNoSiteModel:
    tuned = load_tuned()
    return HazardTunedNoSiteModel(rho=matrix.rho, kappas=matrix_kappas(matrix), tuned=tuned)

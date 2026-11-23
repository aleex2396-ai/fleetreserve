"""LOOK-AHEAD WHAT-IF, not a model of the plan: `models.hazard_tuned_inputs` that also reads the
weather observed in the service hours of the day itself (the owner's decision of 15 Nov 2026 on
the weather input, plan section 7: a what-if on development units gives a model the day's own
weather, labelled as a leak and never a model input, as an upper bound on what a forecast could
add).

The one change: `extra_inputs` is `models.hazard_tuned_inputs.EXTRA_INPUTS` followed by
`weather_lookahead.COLUMNS` (the mean temperature, precipitation and snowfall of the 8 hours after
τ), which are not known at τ. Everything else, the settings (reading T1 there) included, is that
of `models.hazard_tuned_inputs`. If its development figures show a gain over that model, a real
forecast archive is a separate decision (a new download).

Where it is refused (`model_inputs`, module text): this module is listed in
`forecast_model.LOOK_AHEAD_MODELS` and declares `look_ahead`; `learn_run.check_model` refuses it
unless the caller allows look-ahead, which only `learn_run.main` (the development evaluation)
does, so `dial_choice` and every command built on `dial_choice.make_factory` (where a model is
chosen or its offers compared) refuse it with exit 1; `model_inputs.extra_of` and
`ExtraInputs.add` refuse its columns where look-ahead is not allowed; and `predict` refuses any
row whose unit is not a development unit (`truth_build.DEVELOPMENT_UNITS`), so no held-out or
training unit is ever forecast by it, whatever the command. It is FITTED on the training units
with the look-ahead columns (the what-if needs a model that has learned from them), and never
forecast on them. `learn_run` labels its results
(`look_ahead`) and the first lines of its summary.
"""

from collections.abc import Sequence

import pandas as pd

from fleetreserve import model_inputs, weather_lookahead
from fleetreserve.hazard_tune_settings import load_tuned
from fleetreserve.models.hazard_tuned_inputs import EXTRA_INPUTS, HazardTunedInputsModel
from fleetreserve.models.lgbm import matrix_kappas
from fleetreserve.truth_build import DEVELOPMENT_UNITS


class HazardTunedInputsLookaheadModel(HazardTunedInputsModel):
    name = "hazard_tuned_inputs_lookahead"
    extra_inputs = (*EXTRA_INPUTS, *weather_lookahead.COLUMNS)
    look_ahead = True

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        units = sorted(set(frame["unit"].astype(str))) if "unit" in frame.columns else []
        outside = [u for u in units if u not in DEVELOPMENT_UNITS]
        if outside:
            raise ValueError(
                f"the look-ahead what-if forecasts development units only, not {outside}"
            )
        return super().predict(frame, kappas)


def make_model(matrix) -> HazardTunedInputsLookaheadModel:
    tuned = load_tuned()
    return HazardTunedInputsLookaheadModel(
        rho=matrix.rho,
        kappas=matrix_kappas(matrix),
        tuned=tuned,
        context=model_inputs.load_context(),
    )

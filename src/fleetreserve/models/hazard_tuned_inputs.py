"""The tuned neural hazard model with the plan inputs of `hazard_inputs` as extra input columns
(plan section 7; section 13 WP8, phase two).

`models.hazard_tuned` with one change: besides `hazard_data.INPUT_COLUMNS`, each car gets the
columns `EXTRA_INPUTS` (`model_inputs`): the energy drawn before τ under reading E3 (nominal power
times hours since plug-in), the four regime flags, the four weather columns of the recent past,
the OxCGRT index, the public holidays dated by enactment, JPL's 9/80 Fridays, and the school
terms. Each is known at τ (`hazard_inputs`, and `tests/learn/test_energy_drawn.py`: cars equal up
to τ get equal values). `EXTRA_INPUTS` is the one declared list: dropping a column is deleting
its line.

What the network gets (`model_inputs.InputsScaler`): the inputs of `hazard_data.Scaler`, then
per extra column its standardised value (0 where missing) and a 0/1 missing flag. The car-token
network takes that wider vector (`hazard_net.HazardNet`, `n_inputs`); everything else of the
network, the sets, the fit, the curves, the table and the levels is that of `models.hazard`.

Where the columns come from: the fit table is built in `fit` from the cars tables it is given,
the columns appended by `model_inputs.ExtraInputs`; the forecast frame gets them in
`learn_run.forecast_frame` (`model_inputs`, module text). The sources are read once, in
`make_model`, by `model_inputs.load_context`: the weather files of every place and the OxCGRT
file under `data/raw/` (`tools/fetch_data.py --context`), the committed school terms
(`learn/school-terms.csv`) and the 9/80 parity (`learn/nine-eighty-phase.json`, written by
`nine_eighty_settings` from a recorded `nine_eighty_run`). A missing or unreadable one ends the
command with "could not run" (exit 2), before anything is fitted, as a missing tuned settings file
does.

Reading T1: the settings are those of `learn/hazard-tuned.json`, tuned for `models.hazard` without
the extra columns; they are not tuned again for this model. [Alternative: a tuning run of this
model on the training sites, `hazard_tune` with the extra columns, which the command does not
yet take.]
"""

from collections.abc import Sequence

import pandas as pd

from fleetreserve import model_inputs
from fleetreserve.forecast_model import KEY_COLUMNS
from fleetreserve.hazard_data import INPUT_COLUMNS
from fleetreserve.hazard_tune_grid import Setting
from fleetreserve.hazard_tune_settings import load_tuned
from fleetreserve.learn_features import with_context
from fleetreserve.models.hazard_tuned import HazardTunedModel
from fleetreserve.models.lgbm import matrix_kappas

# The extra input columns, one per line (module text).
EXTRA_INPUTS = (
    "energy_drawn_kwh",
    "regime_nrel_controller",
    "regime_asr_paid",
    "regime_asr_controller",
    "regime_covid",
    "weather_temperature_c",
    "weather_temperature_mean_24h_c",
    "weather_precipitation_24h_mm",
    "weather_snowfall_24h_cm",
    "oxcgrt_stringency",
    "public_holiday_dated",
    "nine_eighty_friday",
    "school_holiday",
    "academic_break",
)


class HazardTunedInputsModel(HazardTunedModel):
    name = "hazard_tuned_inputs"
    extra_inputs: tuple[str, ...] = EXTRA_INPUTS
    look_ahead = False

    def __init__(
        self,
        *,
        rho: float,
        kappas: Sequence[float],
        tuned: Setting,
        context: model_inputs.Context,
    ) -> None:
        super().__init__(rho=rho, kappas=kappas, tuned=tuned)
        self.context_sources = context.sources
        self.context_files = dict(context.files)

    def _extra(self) -> model_inputs.ExtraInputs:
        return model_inputs.ExtraInputs(self.extra_inputs, self.context_sources, self.look_ahead)

    def _fit_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._extra().add(super()._fit_table(frame), frame.reset_index(drop=True))

    def _fit_scaler(self, table: pd.DataFrame):
        return model_inputs.InputsScaler.fit(table, self.settings, self.extra_inputs)

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        read = {*KEY_COLUMNS, *INPUT_COLUMNS, *self.extra_inputs}
        return with_context(frame[[c for c in frame.columns if c in read]])

    def _inputs_of(self, context: pd.DataFrame):
        missing = [c for c in self.extra_inputs if c not in context.columns]
        if missing:
            raise KeyError(f"the frame lacks the extra input columns {missing}")
        return super()._inputs_of(context)


def make_model(matrix) -> HazardTunedInputsModel:
    tuned = load_tuned()
    return HazardTunedInputsModel(
        rho=matrix.rho,
        kappas=matrix_kappas(matrix),
        tuned=tuned,
        context=model_inputs.load_context(),
    )

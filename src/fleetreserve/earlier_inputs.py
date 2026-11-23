"""The numeric inputs of the earlier analysis's learned per-car model, from this project's feature
table (plan sections 7, 12; `models.earlier_gbm`).

The earlier analysis's package (`ev-fleet-offer`, commit 05a6cf9, module `features`) gives each
car present at the decision time 23 numeric inputs and a station. `numeric(frame)` builds the 19 of
them that a `learn_features.build_table` table (or its `feature_frame`) can give, in the earlier
order (`INPUTS`), reading only feature columns, each known at τ:

- copied from a column (`SOURCES`): `plugin_hour` from `plug_in_clock_h` (hours from the local
  midnight of the day to plug-in, negative for an earlier day), `elapsed_h` from
  `hours_since_plug_in`, `weekday`, `month`, `cars_present` from `fleet_at_tau` (cars plugged in at
  τ), `linked` from `user_known`, the user's earlier sessions' stays (`stay_q10`, `stay_q50` from
  `history_q10_stay_h`, `history_median_stay_h`), energies (`kwh_q10`, `kwh_q50` from
  `history_q10_energy_kwh`, `history_median_energy_kwh`) and `days_since_last` from
  `history_days_since_last_plug_in`, and `typed_remaining_h` from `declared_departure_h` (hours
  from τ to the declared departure, stated by τ);
- derived (`DERIVED`): `in_term`, `summer`, `holiday` from the row's date by `earlier_calendar`;
  `paid_regime`, 1 on and after `PAID_FROM` (1 November 2018, from which charging was paid for
  sessions linked to a driver: Lee et al. (2019)) and 0 before, by the row's date at every site,
  as the earlier inputs have it (reading E6 of `models.earlier_gbm`); `paid` = `linked` x
  `paid_regime`;
  `n_past`, the count of earlier sessions (`history_sessions`) for a car with a user id and missing
  for one without, as the earlier inputs have it; `typed`, 1 where a declared departure stated by
  τ is held and 0 elsewhere.

Not built (`LEFT_OUT`): the typed plan's minutes, energy and the flags for typed stays of exactly
60 and 480 minutes. The truth tables carry the declared departure alone (`truth`: the first app
entry's departure, written only when stated by τ), not the typed minutes or energy (reading E8 of
`models.earlier_gbm`).

The station, the earlier model's one categorical input, is not built here: the feature table does
not carry it, and `models.earlier_gbm` reads it from the cars tables it is fitted on (reading E7).
"""

import numpy as np
import pandas as pd

from fleetreserve import earlier_calendar

PAID_FROM = "2018-11-01"
SOURCES = {
    "plugin_hour": "plug_in_clock_h",
    "elapsed_h": "hours_since_plug_in",
    "weekday": "weekday",
    "month": "month",
    "cars_present": "fleet_at_tau",
    "linked": "user_known",
    "stay_q10": "history_q10_stay_h",
    "stay_q50": "history_median_stay_h",
    "kwh_q10": "history_q10_energy_kwh",
    "kwh_q50": "history_median_energy_kwh",
    "days_since_last": "history_days_since_last_plug_in",
    "typed_remaining_h": "declared_departure_h",
}
DERIVED = ("in_term", "summer", "holiday", "paid_regime", "paid", "n_past", "typed")
# The earlier order of its numeric inputs, without `LEFT_OUT`.
INPUTS = (
    "plugin_hour",
    "elapsed_h",
    "weekday",
    "month",
    "in_term",
    "summer",
    "holiday",
    "cars_present",
    "paid_regime",
    "linked",
    "paid",
    "n_past",
    "stay_q10",
    "stay_q50",
    "kwh_q10",
    "kwh_q50",
    "days_since_last",
    "typed",
    "typed_remaining_h",
)
LEFT_OUT = ("typed_minutes", "typed_kwh", "typed_60", "typed_480")


def numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """`INPUTS` as floats, one row per row of `frame`, index 0 to n - 1 (module text)."""
    frame = frame.reset_index(drop=True)
    out = pd.DataFrame({name: frame[source].astype(float) for name, source in SOURCES.items()})
    flags = earlier_calendar.day_flags(frame["date"])
    for name in earlier_calendar.FLAGS:
        out[name] = flags[name].to_numpy(float)
    dates = pd.to_datetime(frame["date"].astype(str), format="%Y-%m-%d")
    out["paid_regime"] = (dates >= pd.Timestamp(PAID_FROM)).astype(float).to_numpy()
    out["paid"] = out["linked"] * out["paid_regime"]
    known = frame["user_known"].astype(bool).to_numpy()
    out["n_past"] = np.where(known, frame["history_sessions"].astype(float), np.nan)
    out["typed"] = frame["declared_departure_h"].astype(float).notna().astype(float).to_numpy()
    return out[list(INPUTS)]

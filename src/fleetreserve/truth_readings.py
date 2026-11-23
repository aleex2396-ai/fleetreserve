"""Counts of how other readings of the plan's truth rules would change a unit's truth table.

The main rules are in `truth`. For the report only, each count compares, car by car on
counting days, the main rule with one alternative the plan's words also allow:

- `rate_history_ended_before_tau`: the rate history holds only the user's sessions that ended
  (plugged out) at or before the day's τ, the history known at the decision; a car with none
  gets the nominal cap.
- `rate_average_over_plug_in_to_plug_out`: every session's average is taken over plug-in to
  plug-out, ignoring the recorded end of charging (changes ACN and NREL only).
- `energy_before_charge_bound_switched`: the charge-field rule on energy drawn before τ is
  switched: applied where the main rule does not apply it (ACN, with `doneChargingTime` as the
  charging end), removed where it does (NREL). The rule has two parts: the nominal-cap hours
  count only inside the recorded charging time; and a car whose recorded charging ended at or
  before τ drew its whole session energy before τ (room 0).
- `energy_before_at_rate_not_nominal_cap`: energy before τ is drawn at the car's rate instead of
  the nominal cap.

`READINGS` holds these descriptions for the build report. Nothing here changes a truth table.

In the held setting (`truth.HELD`: no charging before τ) the two readings about energy before τ
have no meaning, because no energy is drawn before τ under any reading. They are reported as
`None` (not applicable), never as a count; the rate readings are unchanged.
"""

import numpy as np
import pandas as pd

from fleetreserve.truth import HELD, MAIN, energy_before, session_averages, user_rates
from fleetreserve.windows import utc_values

TOLERANCE = 1e-9
NOT_APPLICABLE_HELD = (
    "energy_before_charge_bound_switched",
    "energy_before_at_rate_not_nominal_cap",
)
READINGS = {
    "rate_history_ended_before_tau": (
        "rate from the user's sessions plugged out at or before τ only; none: nominal cap"
    ),
    "rate_average_over_plug_in_to_plug_out": (
        "every session's average over plug-in to plug-out, ignoring the recorded charging end"
    ),
    "energy_before_charge_bound_switched": (
        "charge-field rule switched (applied to ACN with doneChargingTime, removed from NREL); "
        "the rule: nominal-cap hours from plug-in to τ count only inside the recorded charging "
        "time, and recorded charging ended at or before τ means the whole session energy was "
        "drawn before τ (room 0)"
    ),
    "energy_before_at_rate_not_nominal_cap": "energy before τ drawn at the car's rate, not the cap",
}


def _key(frame: pd.DataFrame) -> pd.Series:
    return frame["site"].astype(str) + "\x1f" + frame["user_id"].astype(str)


def rate_history_before(cars: pd.DataFrame, sessions: pd.DataFrame, average: np.ndarray):
    """Each car's rate if only the user's sessions ended at or before τ count."""
    history = pd.DataFrame(
        {
            "key": _key(sessions),
            "when": utc_values(sessions["plug_out"]).astype("int64"),
            "average": average,
        }
    )[sessions["user_id"].notna().to_numpy()]
    history = history.sort_values(["key", "when"], kind="stable")
    history["best"] = history.groupby("key")["average"].cummax()
    history = history.sort_values("when", kind="stable")
    probe = pd.DataFrame(
        {
            "key": _key(cars),
            "when": utc_values(cars["tau"]).astype("int64"),
            "position": np.arange(len(cars)),
        }
    ).sort_values("when", kind="stable")
    joined = pd.merge_asof(probe, history[["key", "when", "best"]], on="when", by="key")
    joined = joined.sort_values("position")
    best = joined["best"].to_numpy(dtype=float)
    return np.where(np.isnan(best), cars["nominal_kw"].to_numpy(float), best)


def count_changes(
    cars: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    limit_kw: float,
    bound_by_charge_fields: bool,
    setting: str = MAIN,
) -> dict:
    """Cars on counting days whose rate or energy before τ would change under each alternative.

    `cars` is a unit's cars table (from `truth.build`); `sessions` is the session table its rate
    history was built from.
    """
    counting = cars[cars["day_counts"]].reset_index(drop=True)
    sessions = sessions.reset_index(drop=True)
    rate = counting["rate_kw"].to_numpy(float)
    with_user = counting["user_id"].notna().to_numpy()

    full = sessions.set_index(["site", "session_id"])
    keys = pd.MultiIndex.from_arrays([counting["site"], counting["session_id"]])
    rows = full.reindex(keys).reset_index()
    rows["tau"] = counting["tau"].array

    average, _ = session_averages(sessions, limit_kw)
    before = rate_history_before(rows, sessions, average)

    plain_average, plain_kind = session_averages(sessions, limit_kw, use_charging_time=False)
    plain = user_rates(sessions, plain_average, plain_kind)
    plain.index = pd.MultiIndex.from_arrays([sessions["site"], sessions["session_id"]])
    plain_rate = plain["rate_kw"].reindex(keys).to_numpy(float)

    counts = {
        "cars_on_counting_days": len(counting),
        "rate_history_ended_before_tau": int(
            np.sum(with_user & (np.abs(before - rate) > TOLERANCE))
        ),
        "rate_average_over_plug_in_to_plug_out": int(np.sum(np.abs(plain_rate - rate) > TOLERANCE)),
    }
    if setting == HELD:
        return {**counts, **dict.fromkeys(NOT_APPLICABLE_HELD)}
    drawn = counting["energy_before_kwh"].to_numpy(float)
    switched, _ = energy_before(rows, rows["tau"], not bound_by_charge_fields)
    at_rate_rows = rows.assign(nominal_kw=rate)
    at_rate, _ = energy_before(at_rate_rows, at_rate_rows["tau"], bound_by_charge_fields)
    return {
        **counts,
        "energy_before_charge_bound_switched": int(np.sum(np.abs(switched - drawn) > TOLERANCE)),
        "energy_before_at_rate_not_nominal_cap": int(np.sum(np.abs(at_rate - drawn) > TOLERANCE)),
    }

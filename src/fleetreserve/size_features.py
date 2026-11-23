"""The context row of one offering day, for sizing the day (plan section 7, day-size net; T7).

`context_row(prepared)` reads a day prepared by `gate_day.prepare_day` that offers and returns the
values of `COLUMNS`, in that order. It reads the day's date, its number of cars at τ
(`n_cars`), its steered and unsteered counts, the certificate's policy (`m` is not read; `w` is)
and value, and the forecasts of the steered cars (their departure and the departures of their
driver-risk scenarios). It reads no field of the true cars (`Day.cars`). The forecasts' rate,
cap and need are not read, because under known energy bounds they are the true rate and room.
The certificate itself is solved on those bounds, so its value and widths depend on them.

- `weekday` (Monday 0), `month`, `day_of_year`: parts of the day's date;
- `fleet_at_tau`: the cars of the fleet at τ; `steered_cars`, `unsteered_cars`;
- `service_hours`: the number of service hours;
- `box_value_kwh`: the certified value at scale 1; `width_min_kw`, `width_mean_kw`,
  `width_max_kw`: the smallest, mean and largest half-width `w` over the service hours;
- `departure_min_h`, `departure_mean_h`, `departure_max_h`: the smallest, mean and largest
  forecast departure (hours from τ) over the steered cars; `departure_in_window_share`: the share
  of steered cars whose forecast departure is at most the number of service hours;
- `lowest_level_departure_min_h`, `lowest_level_departure_mean_h`: the same smallest and mean of
  each steered car's first driver-risk scenario departure (its departure at the lowest level), or
  of its forecast departure when it has no scenario.

The calendar columns of `learn_features.CONTEXT_COLUMNS` (`public_holiday`, `site_type`) are added
by the command, which runs where the `holidays` package is installed.
"""

import datetime

import numpy as np

from fleetreserve.gate_day import PreparedDay

COLUMNS = (
    "weekday",
    "month",
    "day_of_year",
    "fleet_at_tau",
    "steered_cars",
    "unsteered_cars",
    "service_hours",
    "box_value_kwh",
    "width_min_kw",
    "width_mean_kw",
    "width_max_kw",
    "departure_min_h",
    "departure_mean_h",
    "departure_max_h",
    "departure_in_window_share",
    "lowest_level_departure_min_h",
    "lowest_level_departure_mean_h",
)


def context_row(prepared: PreparedDay) -> dict[str, float]:
    """The values of `COLUMNS` for an offering day (module text)."""
    if prepared.abstains:
        raise ValueError(f"the day {prepared.date} does not offer, so it has no context row")
    date = datetime.date.fromisoformat(prepared.date)
    widths = np.asarray(prepared.certificate.policy.w, dtype=float)
    n = int(widths.size)
    forecasts = prepared.day.forecasts
    departures = np.array([f.departure for f in forecasts], dtype=float)
    lowest = np.array(
        [f.scenarios[0][0] if f.scenarios else f.departure for f in forecasts], dtype=float
    )
    values = (
        date.weekday(),
        date.month,
        date.timetuple().tm_yday,
        prepared.n_cars,
        len(prepared.steered),
        len(prepared.unsteered),
        n,
        prepared.value,
        widths.min(),
        widths.mean(),
        widths.max(),
        departures.min(),
        departures.mean(),
        departures.max(),
        float(np.mean(departures <= n)),
        lowest.min(),
        lowest.mean(),
    )
    return {name: float(v) for name, v in zip(COLUMNS, values, strict=True)}

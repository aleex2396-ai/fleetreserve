"""The empirical forecaster of WP6 (`forecast_empirical.EmpiricalForecaster`) as a departure model
(`forecast_model.DepartureModel`), the first model of WP8.

Nothing is fitted (`fit_units` is empty): each unit's forecasts come from the reference cars of
its own other days, with the matrix's readings `reference` and `min_cell`, exactly as the gate
run's forecaster makes them. It therefore declares `reads_other_day_outcomes`, and `predict` is
given the units' whole cars tables, outcomes included (`forecast_model`); the forecaster leaves
out the day itself and every session plugged in at its τ. Per car of a counting day's fleet and
level kappa:

- `d_hat`, `r_hat`, `e_cap`, `e_need`: what `EmpiricalForecaster.forecast_car` returns in its
  "empirical" mode (`e_need` at the matrix's `rho`); all NaN when the car has no reference sample;
- `survival_h1` to `survival_h8`: `EmpiricalForecaster.survival_bound` (NaN where it gives None);
- `d_median`: the median (`lower_quantile` at 0.5) of the reference departures.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from fleetreserve.days_from_truth import FleetCar
from fleetreserve.forecast_empirical import EmpiricalForecaster, lower_quantile, upper_quantile
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    MEDIAN,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    empty_table,
)


@dataclass(frozen=True)
class _Cars:
    cars: pd.DataFrame  # what `EmpiricalForecaster` reads of a `UnitDays`


@dataclass(frozen=True)
class _Record:
    date: str


@dataclass(frozen=True)
class _Day:
    """What `EmpiricalForecaster.reference_for` and `survival_bound` read of a `TruthDay`."""

    fleet: tuple[FleetCar, ...]
    record: _Record


class EmpiricalModel:
    name = "empirical"
    fit_units: tuple[str, ...] = ()
    reads_other_day_outcomes = True

    def __init__(self, *, rho: float, reference: str, min_cell: int) -> None:
        self.rho = rho
        self.reference = reference
        self.min_cell = min_cell

    def fit(self, frame: pd.DataFrame) -> None:
        """Nothing to fit (module text)."""

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        rows = []
        for _, part in frame.groupby("unit", sort=False):
            part = part.reset_index(drop=True)
            forecaster = EmpiricalForecaster(
                _Cars(part), reference=self.reference, min_cell=self.min_cell
            )
            targets = part[part["counting"].astype(bool) & part["in_fleet"].astype(bool)]
            for _, cars in targets.groupby(["site", "date"], sort=False):
                rows += self._day(forecaster, cars, kappas)
        if not rows:
            return empty_table()
        return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))

    def _day(self, forecaster, cars: pd.DataFrame, kappas) -> list[dict]:
        fleet = tuple(
            FleetCar(
                row=int(index),
                session_id=str(item.session_id),
                site=str(item.site),
                arrival_hour=int(item.arrival_hour),
                arrived_earlier_day=bool(item.arrived_earlier_day),
                departure_exact=float(item.departure_exact),
            )
            for index, item in zip(cars.index, cars.itertuples(index=False), strict=True)
        )
        day = _Day(fleet, _Record(str(cars["date"].iloc[0])))
        units = cars["unit"].astype(str).tolist()
        rows = []
        for i, car in enumerate(fleet):
            found = forecaster.reference_for(day, i)
            for kappa in kappas:
                key = (units[i], car.site, day.record.date, car.session_id)
                row = dict(zip(KEY_COLUMNS, key, strict=True))
                row.update(kappa=float(kappa), rho=float(self.rho))
                if found is None:
                    row.update(d_hat=math.nan, r_hat=math.nan, e_cap=math.nan, e_need=math.nan)
                    row["d_median"] = math.nan
                else:
                    row.update(
                        d_hat=lower_quantile(found.departures, kappa),
                        r_hat=lower_quantile(found.rates, kappa),
                        e_cap=lower_quantile(found.rooms, kappa),
                        e_need=upper_quantile((1 - self.rho) * found.rooms, kappa),
                        d_median=lower_quantile(found.departures, MEDIAN),
                    )
                for hour, column in enumerate(SURVIVAL_COLUMNS, start=1):
                    bound = forecaster.survival_bound(day, i, hour, kappa)
                    row[column] = math.nan if bound is None else bound
                rows.append(row)
        return rows


def make_model(matrix) -> EmpiricalModel:
    return EmpiricalModel(rho=matrix.rho, reference=matrix.reference, min_cell=matrix.min_cell)

"""Empirical forecast bounds at level kappa, by site and arrival hour (plan sections 1, 2, 5, 6, 7).

For each car of a counting day (`days_from_truth.TruthDay`), the forecast `day.ForecastCar`
`(d_hat, r_hat, e_cap, e_need)` is taken from reference cars: the cars of the same unit plugged in
at τ on other days, of the same site, in the same cell.

Departure, `d_hat`: the kappa-quantile (`lower_quantile`) of the reference cars' departures, in
fractional hours from τ to the exact plug-out time.

Survival-tightened bound at hour t (plan section 1): the same quantile among the reference cars
still plugged in at the start of hour t (exact departure above t), so it is always later than the
start of hour t (`EmpiricalForecaster.survival_bound`).

Rate and energy, three modes; the plan's section 7 defines them for the later learned model and
its WP6 line names only departure quantiles, so the mode is a required argument with no default:
- `"empirical"`: `r_hat` and `e_cap` the kappa-quantiles of the reference cars' rates and rooms,
  `e_need` the (1 - kappa)-quantile (`upper_quantile`) of their protected energy
  `(1 - rho) * room`;
- `"known"`: `r_hat = r`, `e_cap = room`, `e_need = (1 - rho) * room` (only departure uncertain);
- `"user-rate"`: `r_hat` the car's own truth rate (the user's highest observed average, or the
  nominal cap), `e_cap` and `e_need` as in `"empirical"`.

Oracle departure (`departure="oracle"`): `d_hat` is the car's true departure hour, rounded down
as in the true set. The exact fractional plug-out time is not used: with it the forecast set
would let the car draw a share of the hour in which it leaves, which the true set forbids, so the
forecast set would not lie inside the true set and plan T2's event E'_i (`d_hat <= d`) would fail
for every car whose plug-out is not on the hour.

Quantile convention: for `n` reference values sorted `x_(1) <= ... <= x_(n)` and
`k = floor(kappa * n) + 1` (kappa taken as the exact decimal it is written as), the lower
kappa-quantile is `x_(k)` and the upper (1 - kappa)-quantile is `x_(n + 1 - k)`. At most
`floor(kappa * n)` reference values lie strictly below the lower one (strictly above the upper
one), so the bound holds for at least a share `1 - kappa` of the reference sample; `x_(k)` is the
largest order statistic with that property.

Readings (the plan's text does not decide them):

1. Reference days (`reference=`). `"other-days"` (the default): every other day of the unit, so a
   day's own outcomes never enter its forecast (plan T6: "any model fitted on other days", and
   exchangeable days). Left out with the day itself: every row of a session plugged in at the
   day's τ, because a session spanning several days carries the same plug-out on each of them.
   `"earlier-days"`: only days before the day (and, again, no session of the day). A third
   alternative, training sites only, is not built here.
2. The reference rows are those of the whole cars table of the unit, on counting and
   non-counting days, as "presence at τ" is the conditioning event. For departure every car
   plugged in at τ counts, also one leaving within the decision hour (its departure is an
   outcome). For rate and energy only cars kept in a fleet by the rules of `days_from_truth`
   count, with the room after those rules, the same quantity the true cars carry.
3. Arrival hour: the clock hour of plug-in; cars plugged in on an earlier calendar day than the
   day's date form their own cell (`EARLIER_DAY_CELL`).
4. Small cells: a cell with fewer than `min_cell` reference cars (default `MIN_CELL` = 20, fixed
   before any coverage was computed) falls back to the site's reference cars, any arrival hour.
   The choice is made on the cars plugged in at τ; rate and energy use the fleet cars of the
   chosen reference, or of the whole site when it holds none. With no reference car at all
   (possible only under `"earlier-days"`), the car has no forecast and its day is skipped.
5. Survival bound: at hour t the same rule is applied to the reference cars still plugged in at
   the start of hour t (the cell if it holds at least `min_cell` of them, else the site). With
   none left in the site the bound is None: the caller keeps its earlier bound.
6. Day length: the recovery hours are extended so that every forecast departure lies inside the
   day (`Day` requires it). Extra recovery hours change no true set (reading 2 of `day`), and the
   forecast sets then do not depend on the true departures through the day's length.
"""

import math
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from fleetreserve.day import Day, ForecastCar
from fleetreserve.days_from_truth import TruthDay, UnitDays

KAPPAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)  # plan section 2: the one home of the grid of kappa
# The levels the tuned models were scored over (`learn_tune`, `hazard_tune`): the grid's levels up
# to 0.3, the grid before 0.4 and 0.5 were added. Derived from the grid, not a second copy of it.
TUNING_KAPPAS = tuple(k for k in KAPPAS if k <= 0.3)
# Levels off plan section 2's grid, read only by the what-if of `dial_choice --what-if-kappa`.
WHAT_IF_KAPPAS = (0.6, 0.7)
MODES = ("empirical", "known", "user-rate")
DEPARTURES = ("empirical", "oracle")
REFERENCES = ("other-days", "earlier-days")
MIN_CELL = 20
EARLIER_DAY_CELL = -1


def order_index(kappa: float, n: int) -> int:
    """`k = floor(kappa * n) + 1`, the 1-based order statistic of the lower kappa-quantile."""
    if not 0 < kappa < 1:
        raise ValueError(f"kappa must be in (0, 1), got {kappa}")
    if n < 1:
        raise ValueError("a quantile needs at least one reference value")
    return math.floor(Fraction(str(kappa)) * n) + 1


def lower_quantile(values, kappa: float) -> float:
    """The lower kappa-quantile `x_(k)` of the module text."""
    x = np.sort(np.asarray(values, dtype=float))
    return float(x[order_index(kappa, x.size) - 1])


def upper_quantile(values, kappa: float) -> float:
    """The upper (1 - kappa)-quantile `x_(n + 1 - k)` of the module text."""
    x = np.sort(np.asarray(values, dtype=float))
    return float(x[x.size - order_index(kappa, x.size)])


@dataclass(frozen=True)
class Reference:
    """The reference sample of one car of one day."""

    departures: np.ndarray  # cars plugged in at τ of the chosen cell (or the site)
    rates: np.ndarray  # fleet cars of the same choice (or of the site)
    rooms: np.ndarray
    fallback: bool  # True when the cell was too small and the site was used


@dataclass(frozen=True)
class ForecastDay:
    """A day with forecasts at level kappa, and for each car whether the site fallback was used."""

    truth: TruthDay
    day: Day
    fallback: tuple[bool, ...]


class EmpiricalForecaster:
    """Forecasts for the counting days of one unit (see the module text)."""

    def __init__(
        self, unit: UnitDays, *, reference: str = "other-days", min_cell: int = MIN_CELL
    ) -> None:
        if reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")
        if not isinstance(min_cell, int) or min_cell < 1:
            raise ValueError(f"min_cell must be a positive whole number, got {min_cell!r}")
        self.unit = unit
        self.reference = reference
        self.min_cell = min_cell
        self._sites = {}
        table = unit.cars
        cell = np.where(
            table["arrived_earlier_day"].to_numpy(bool),
            EARLIER_DAY_CELL,
            table["arrival_hour"].to_numpy(int),
        )
        for site in table["site"].unique():
            part = (table["site"] == site).to_numpy()
            self._sites[site] = {
                "date": table["date"].to_numpy(str)[part],
                "session": table["session_id"].to_numpy(str)[part],
                "cell": cell[part],
                "departure": table["departure_exact"].to_numpy(float)[part],
                "rate": table["rate"].to_numpy(float)[part],
                "room": table["room"].to_numpy(float)[part],
                "fleet": table["in_fleet"].to_numpy(bool)[part],
            }
        self._sessions = {
            key: set(frame["session_id"]) for key, frame in table.groupby(["site", "date"])
        }
        self._usable_cache: dict[tuple[str, str], np.ndarray] = {}
        self._reference_cache: dict[tuple[str, str, int], Reference | None] = {}

    def _usable(self, site: str, day_date: str) -> np.ndarray:
        """Reference rows of the site usable for the day (reading 1)."""
        key = (site, day_date)
        if key not in self._usable_cache:
            rows = self._sites[site]
            own = np.isin(rows["session"], list(self._sessions.get(key, ())))
            usable = (rows["date"] != day_date) & ~own
            if self.reference == "earlier-days":
                usable &= rows["date"] < day_date
            self._usable_cache[key] = usable
        return self._usable_cache[key]

    def reference_for(self, truth: TruthDay, i: int) -> Reference | None:
        """The reference sample of car `i` of the day, or None when there is none."""
        car = truth.fleet[i]
        key = (car.site, truth.record.date, i)
        if key in self._reference_cache:
            return self._reference_cache[key]
        rows = self._sites[car.site]
        usable = self._usable(car.site, truth.record.date)
        cell_value = EARLIER_DAY_CELL if car.arrived_earlier_day else car.arrival_hour
        in_cell = usable & (rows["cell"] == cell_value)
        fallback = int(in_cell.sum()) < self.min_cell
        chosen = usable if fallback else in_cell
        if not chosen.any():
            found = None
        else:
            fleet = chosen & rows["fleet"]
            if not fleet.any():
                fleet = usable & rows["fleet"]
            if not fleet.any():
                found = None
            else:
                found = Reference(
                    departures=np.sort(rows["departure"][chosen]),
                    rates=rows["rate"][fleet],
                    rooms=rows["room"][fleet],
                    fallback=fallback,
                )
        self._reference_cache[key] = found
        return found

    def forecast_car(
        self, truth: TruthDay, i: int, kappa: float, mode: str, departure: str = "empirical"
    ) -> ForecastCar | None:
        """Car `i`'s forecast at level `kappa`, or None when it has no reference sample."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if departure not in DEPARTURES:
            raise ValueError(f"departure must be one of {DEPARTURES}, got {departure!r}")
        true = truth.day.cars[i]
        rho = truth.day.rho
        found = self.reference_for(truth, i)
        if found is None and not (mode == "known" and departure == "oracle"):
            return None
        if departure == "oracle":
            d_hat = float(true.departure)
        else:
            d_hat = lower_quantile(found.departures, kappa)
        if mode == "known":
            return ForecastCar(d_hat, true.rate, true.room, true.need(rho))
        rate = true.rate if mode == "user-rate" else lower_quantile(found.rates, kappa)
        cap = lower_quantile(found.rooms, kappa)
        need = upper_quantile((1 - rho) * found.rooms, kappa)
        return ForecastCar(d_hat, rate, cap, need)

    def forecast_day(
        self, truth: TruthDay, kappa: float, mode: str, departure: str = "empirical"
    ) -> ForecastDay | None:
        """The day with every car's forecast, or None when some car has no reference sample."""
        forecasts = []
        for i in range(len(truth.day.cars)):
            forecast = self.forecast_car(truth, i, kappa, mode, departure)
            if forecast is None:
                return None
            forecasts.append(forecast)
        day = truth.day
        latest = max([0.0, *(f.departure for f in forecasts)])
        n_recovery = max(day.n_recovery, math.ceil(latest) - 1 - day.n_service)
        with_forecasts = replace(
            day, n_recovery=n_recovery, forecasts=tuple(forecasts), kappa=kappa
        )
        fallback = []
        for i in range(len(day.cars)):
            found = self.reference_for(truth, i)
            fallback.append(bool(found.fallback) if found is not None else False)
        return ForecastDay(truth, with_forecasts, tuple(fallback))

    def survival_bound(self, truth: TruthDay, i: int, hour: int, kappa: float) -> float | None:
        """The kappa-quantile of departure among reference cars still plugged in at the start of
        hour `hour` (hours from τ), or None when no reference car of the site is (reading 5)."""
        if isinstance(hour, bool) or not isinstance(hour, int) or hour < 0:
            raise ValueError(f"hour must be a whole number from 0, got {hour!r}")
        car = truth.fleet[i]
        rows = self._sites[car.site]
        present = self._usable(car.site, truth.record.date) & (rows["departure"] > hour)
        cell_value = EARLIER_DAY_CELL if car.arrived_earlier_day else car.arrival_hour
        in_cell = present & (rows["cell"] == cell_value)
        chosen = present if int(in_cell.sum()) < self.min_cell else in_cell
        if not chosen.any():
            return None
        return lower_quantile(rows["departure"][chosen], kappa)

"""Made-up cars with two kinds of stay, for the tests of the conditional-fit survival bound.

`two_stay_cars(n_days, first_day)`: `PER_DAY` cars a day at one site, plugged in between 7:00 and
9:00 (τ is 10:00), one per user. Each car, by an independent draw, leaves early with probability
`EARLY_SHARE`, uniformly between `EARLY[0]` and `EARLY[1]` hours after τ, and otherwise late,
uniformly between `LATE[0]` and `LATE[1]` hours after τ. So no car leaves between `EARLY[1]` and
`LATE[0]`: a car still plugged in at the start of service hours 4 and 5 leaves after `LATE[0]`,
while the lower quantiles of departure given presence at τ lie below `EARLY[1]`. No car is left at
the start of hour 8.

`three_stay_cars(n_days, first_day)`: the same with three groups (`THREE_STAYS`): a fifth of the
cars leave 0.3 to 3.8 hours after τ, a fifth 5.2 to 5.9 hours after τ and the rest 8 to 10 hours
after τ. With `THREE_SERVICE_HOURS` (4) service hours the 0.3-quantile of departure given presence
at τ lies in the middle group, after the service hours, so a day has something to offer; given
presence at the start of hour 4 the first group is gone and the same quantile lies in the last
group.

`cars_table(fakes, users)`: the cars table (`UnitDays.cars`) of the fakes with those user ids.
`write_unit(folder, unit, fakes, users, service_hours)`: the unit's truth tables with those user
ids and that many service hours, and its days as `days_from_truth.load_unit_days` reads them.

`older_tables(train, fresh, kappas)`: the forecast tables of `lgbm`, `cqr` and `lgbm_tail`, fitted
on `train` and written for `fresh` as `forecast_model.write_table` writes them, and
`sha256_of(tables)` their SHA-256. This file imports nothing of the conditional-fit rule, so it
runs unchanged at a commit that does not have the rule.
"""

import hashlib

import numpy as np
import pandas as pd
from truth_tables_fake import FakeCar, date_of, tables

from fleetreserve.days_from_truth import apply_car_rules, load_unit_days
from fleetreserve.forecast_model import write_table
from fleetreserve.models import cqr, lgbm, lgbm_tail

PER_DAY = 10
EARLY_SHARE = 0.4
EARLY = (0.2, 3.8)
LATE = (5.0, 7.5)
THREE_STAYS = ((0.2, 0.3, 3.8), (0.4, 5.2, 5.9), (1.0, 8.0, 10.0))
THREE_SERVICE_HOURS = 4
OLDER = {"lgbm": lgbm.LgbmModel, "cqr": cqr.CqrModel, "lgbm_tail": lgbm_tail.LgbmTailModel}


def stay_cars(n_days: int, first_day: int, groups, seed: int = 11, site: str = "S"):
    """(fakes, users) of `n_days` days from day `first_day`: each car falls in one of `groups`
    (cumulative share, earliest stay, latest stay; the last share is 1), by an independent draw,
    and stays a uniform time in it."""
    rng = np.random.default_rng(seed + first_day)
    fakes, users = [], []
    for k in range(first_day, first_day + n_days):
        for j in range(PER_DAY):
            draw = rng.random()
            low, high = next(g[1:] for g in groups if draw < g[0])
            stay = low + (high - low) * rng.random()
            fakes.append(
                FakeCar(
                    site,
                    date_of(k),
                    7.0 + 0.2 * j,
                    10.0 + stay,
                    rate=3.0 + rng.random(),
                    room=1.0 + 2.0 * rng.random(),
                    session=f"{site}-{k}-{j}",
                    energy=4.0,
                )
            )
            users.append(f"user{j}")
    return fakes, users


def two_stay_cars(n_days: int, first_day: int = 0):
    """(fakes, users) of the two-stay cars (module text)."""
    return stay_cars(n_days, first_day, ((EARLY_SHARE, *EARLY), (1.0, *LATE)))


def three_stay_cars(n_days: int, first_day: int = 0):
    """(fakes, users) of the three-stay cars (module text)."""
    return stay_cars(n_days, first_day, THREE_STAYS)


def _with_users(car_table: pd.DataFrame, fakes, users) -> pd.DataFrame:
    by_session = dict(zip([f.session for f in fakes], users, strict=True))
    car_table["user_id"] = [by_session[s] for s in car_table["session_id"]]
    return car_table


def cars_table(fakes, users, rho: float = 0.0) -> pd.DataFrame:
    car_table, _ = tables(fakes)
    return apply_car_rules(_with_users(car_table, fakes, users), rho, "cap")


def write_unit(folder, unit: str, fakes, users, service_hours: int, rho: float = 0.0):
    car_table, day_table = tables(fakes, unit)
    day_table["service_hours"] = str(service_hours)
    _with_users(car_table, fakes, users).to_csv(
        folder / f"{unit}-cars.csv", index=False, lineterminator="\n"
    )
    day_table.to_csv(folder / f"{unit}-days.csv", index=False, lineterminator="\n")
    return load_unit_days(folder, unit, rho=rho, no_fit="cap", small_fleet="keep")


def older_tables(train: pd.DataFrame, fresh: pd.DataFrame, kappas) -> dict[str, pd.DataFrame]:
    out = {}
    for name, make in OLDER.items():
        model = make(rho=0.0, kappas=kappas)
        model.fit(train)
        out[name] = model.predict(fresh, kappas)
    return out


def sha256_of(found: dict[str, pd.DataFrame], folder) -> dict[str, str]:
    out = {}
    for name, table in found.items():
        path = folder / f"{name}.csv"
        write_table(table, path)
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out

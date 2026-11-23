"""The departure model's plan inputs that `learn_features.build_table` does not give, as new
columns beside the old ones (plan section 7; WP8).

`with_inputs(table, cars, names, sources)` returns `table` (from `build_table(cars, ...)`, one row
per row of `cars`, in the same order) with the columns `names` appended, each of `NEW_COLUMNS`.
With no names it returns `table` unchanged, so every existing model, which builds its frames
without this function, reads the same frames as before. No model that existed before them reads
a new column; `models.hazard_tuned_inputs` takes them through `model_inputs`. Every column is a
function of facts known at the row's decision time τ:

- `energy_drawn_kwh` (plan section 7, "energy drawn"), reading E3: the source's nominal power
  times the hours from plug-in to τ, uncapped (`energy_drawn`), and 0 under the held setting
  (the truth's rule `truth.BEFORE_RULE_HELD`: no charging before τ). The nominal power is the
  cars table's `nominal_kw` where it is one constant per source, fixed by its loader before any
  data (ACN 6.66 kW, Norway 7.4 kW, a.s.r. 11 kW, Korea 7 kW; a site whose cars carry more
  than one value raises `ValueError`), and `NREL_NOMINAL_KW` (6.6 kW, a stand-in of a common
  on-board charger's power, not read from the data) at the NREL sites, whose `nominal_kw` is
  `max_charge_power`, the session's highest measured power, which can come after τ. What the
  column is: a function of the plug-in time, τ, the source and the setting, so a fixed multiple
  of `hours_since_plug_in` per source and setting. What it is not: the energy a meter at τ reads.
  The cars table holds no measurement of it: the truth's `energy_before_kwh`, min(session
  energy, nominal power x hours before τ), is capped by the whole session's energy, so where
  the car charged more slowly than the nominal power the cap binds on charging done after τ and
  the value gives the session total, and so the departure, away (`learn_features` lists it as
  not known at τ; a review of 15 Nov 2026 found the earlier reading, which copied it, leaking).
  Every new column passes the test that cars equal up to τ and different after it get equal
  values (`tests/learn/test_energy_drawn.py`). [Alternative E4: the session energy where the
  source records that charging ended at or before τ (NREL's `termin_charge`, possibly ACN's
  `doneChargingTime`), E3 otherwise. Not taken: the cars table carries no charging time (only
  NREL's `energy_before_rule` says "ended"), no training unit's source gives it there (ACN's
  `doneChargingTime` is read by the truth for rates only), so the column would mean the energy
  of a finished session at NREL, a development site, and a multiple of the hours at every unit
  the model is fitted on.]
- regime: `regime_inputs.COLUMNS` (dated rules of plan section 3);
- weather: `weather_inputs.COLUMNS` (the archived ECMWF IFS HRES series, values at least 8 hours
  before τ; see that module for why no forecast of the service hours exists in the archive);
- OxCGRT: `oxcgrt_inputs.COLUMNS` (the stringency index 14 days before the row's date);
- calendars: `calendar_inputs.COLUMNS` (public holidays dated by enactment; JPL's 9/80 Fridays);
- school terms: `school_inputs.COLUMNS` (official published term dates, transcribed into
  `learn/school-terms.csv`, each read only from a source published before the row's date).

The plan's inputs and where they are (`learn_features`, `hazard_data`, `hazard_net` for the ones
the model had before this module):

| plan input | before this module | here |
|---|---|---|
| time since plug-in | `hours_since_plug_in` | - |
| energy drawn | no | `energy_drawn_kwh` (nominal x hours since plug-in: no meter reading) |
| site | site vector (`hazard_net`, codes `hazard_data.site_codes`); `site_type` | - |
| user's past sessions | `HISTORY_COLUMNS`, `TAIL_HISTORY_COLUMNS`, `user_known` | - |
| regime | no | `regime_*` |
| event rows (A2) | no (the language-model step is cut, plan section 13 cut 1) | - |
| python-holidays | `public_holiday` (calendars as they stand) | - |
| laws by enactment, one-off holidays by announcement | no | `public_holiday_dated` |
| school terms | no | `school_holiday`, `academic_break` (Caltech) |
| OxCGRT | no | `oxcgrt_stringency` |
| JPL 9/80 Fridays | no | `nine_eighty_friday` (its phase must be given) |
| archived ECMWF IFS HRES forecasts | no | `weather_*` (past values only, as above) |

School terms (the owner's decision of 15 Nov 2026, plan section 7): `school_inputs`, with its
sources, readings and the places and years not covered. Korea has none: its school calendars
are held by the national education information service's open API, which needs a key.

`sources` (`ContextSources`) holds what the columns read: the weather of every place of the
sites in `cars`, the OxCGRT index, the 9/80 parity and the school-term rows.
`load_sources(root, sites, parity)` reads the files under `<root>/data/raw/` that
`tools/fetch_data.py --context` downloads, and the committed `learn/school-terms.csv`. A column
whose source is missing raises `ValueError`; a name that is not in `NEW_COLUMNS`, one already in the
table, or a `table` whose keys differ from `cars`' raises `ValueError`.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fleetreserve import (
    calendar_inputs,
    oxcgrt_inputs,
    regime_inputs,
    school_inputs,
    truth,
    weather_inputs,
)
from fleetreserve.context_places import SITE_PLACES
from fleetreserve.forecast_model import KEY_COLUMNS

ENERGY_COLUMNS = ("energy_drawn_kwh",)
NREL_NOMINAL_KW = 6.6
NEW_COLUMNS = (
    *ENERGY_COLUMNS,
    *regime_inputs.COLUMNS,
    *weather_inputs.COLUMNS,
    *oxcgrt_inputs.COLUMNS,
    *calendar_inputs.COLUMNS,
    *school_inputs.COLUMNS,
)
OXCGRT_FILE = Path("data") / "raw" / "oxcgrt" / "OxCGRT_timeseries_StringencyIndex_v1.csv"


@dataclass(frozen=True)
class ContextSources:
    weather: Mapping[str, pd.DataFrame] | None = None
    oxcgrt: Mapping[str, pd.Series] | None = None
    nine_eighty_parity: int | None = None
    school: tuple | None = None


def load_sources(root: Path, sites: Sequence[str], parity: int | None = None) -> ContextSources:
    """The context files under `root` for the given sites (module text)."""
    places = sorted({place for site in sites for place in SITE_PLACES[site]})
    weather = weather_inputs.load_weather(root, places)
    oxcgrt = oxcgrt_inputs.read_oxcgrt((Path(root) / OXCGRT_FILE).read_bytes())
    return ContextSources(weather, oxcgrt, parity, school_inputs.load_terms())


def energy_drawn(cars: pd.DataFrame) -> pd.Series:
    """`energy_drawn_kwh` of the rows of `cars` (module text, reading E3), read from `site`,
    `plug_in`, `tau`, `nominal_kw` and `energy_before_rule` only."""
    sites = cars["site"].astype(str)
    nominal = cars["nominal_kw"].astype(float)
    nrel = sites.isin(regime_inputs.NREL_SITES)
    varying = sorted(s for s, n in nominal[~nrel].groupby(sites[~nrel]).nunique().items() if n > 1)
    if varying:
        raise ValueError(f"the nominal power is not one constant at the sites {varying}")
    nominal = nominal.where(~nrel, NREL_NOMINAL_KW)
    tau = pd.to_datetime(cars["tau"].astype(str), utc=True, format="ISO8601")
    plug_in = pd.to_datetime(cars["plug_in"].astype(str), utc=True, format="ISO8601")
    hours = ((tau - plug_in) / pd.Timedelta(hours=1)).clip(lower=0.0)
    held = cars["energy_before_rule"].astype(str) == truth.BEFORE_RULE_HELD
    return (nominal * hours).where(~held, 0.0).astype(float)


def aligned(table: pd.DataFrame, cars: pd.DataFrame) -> pd.DataFrame:
    """`cars` with the index of `table`, after checking that their keys agree row by row."""
    cars = cars.reset_index(drop=True)
    if len(cars) != len(table):
        raise ValueError("the table and the cars table have different numbers of rows")
    keys = cars[list(KEY_COLUMNS)].astype(str).reset_index(drop=True)
    if not keys.equals(table[list(KEY_COLUMNS)].astype(str).reset_index(drop=True)):
        raise ValueError("the table's keys differ from the cars table's, row by row")
    return cars.set_axis(table.index)


def with_inputs(
    table: pd.DataFrame,
    cars: pd.DataFrame,
    names: Sequence[str] = (),
    sources: ContextSources | None = None,
) -> pd.DataFrame:
    """`table` with the columns `names` of `NEW_COLUMNS` appended (module text)."""
    sources = ContextSources() if sources is None else sources
    names = tuple(names)
    unknown = [n for n in names if n not in NEW_COLUMNS]
    if unknown:
        raise ValueError(f"not new input columns: {unknown}")
    if len(set(names)) != len(names) or set(names) & set(table.columns):
        raise ValueError("a new column is named twice or is already in the table")
    if not names:
        return table
    cars = aligned(table, cars)
    found: dict[str, pd.Series] = {}
    wanted = set(names)
    if wanted & set(ENERGY_COLUMNS):
        found[ENERGY_COLUMNS[0]] = energy_drawn(cars)
    if wanted & set(regime_inputs.COLUMNS):
        found.update(regime_inputs.regime_columns(cars).items())
    if wanted & set(weather_inputs.COLUMNS):
        if sources.weather is None:
            raise ValueError("the weather columns need the weather files")
        found.update(weather_inputs.weather_columns(cars, sources.weather).items())
    if wanted & set(oxcgrt_inputs.COLUMNS):
        if sources.oxcgrt is None:
            raise ValueError("the OxCGRT column needs the OxCGRT file")
        found.update(oxcgrt_inputs.oxcgrt_columns(cars, sources.oxcgrt).items())
    if "public_holiday_dated" in wanted:
        found.update(calendar_inputs.dated_holiday_flags(cars).items())
    if "nine_eighty_friday" in wanted:
        found.update(calendar_inputs.nine_eighty_flags(cars, sources.nine_eighty_parity).items())
    if wanted & set(school_inputs.COLUMNS):
        if sources.school is None:
            raise ValueError("the school columns need the school-term rows")
        found.update(school_inputs.school_columns(cars, sources.school).items())
    return pd.concat([table, pd.DataFrame({n: found[n] for n in names}, index=table.index)], axis=1)

"""Scores of a departure model on the development units (plan sections 7, 13 WP8).

WP8's pass mark: "development pinball loss, concordance index and offer (WP6 calibration) logged
per model". Per unit and level kappa, over every car of every counting day's fleet:

- pinball loss (`pinball_loss`), the mean over the cars with a forecast of
  `max(level * (y - q), (level - 1) * (y - q))`: departure `d_hat` at level kappa against the exact
  departure (hours from τ to the plug-out time, the quantity the empirical model's quantiles are
  taken of); `r_hat` and `e_cap` at kappa against the true rate (kW) and room (kWh); `e_need` at
  1 - kappa against the true protected energy `(1 - rho) * room` (kWh). These rate and energy
  losses are logged only: the offer below takes rate and energy as known;
- Harrell's concordance index (`concordance_index`) of the forecast median departure `d_median`
  against the exact departure, with no censoring (every plug-out is observed): over the pairs of
  cars with different departures, the share whose medians are in the same order, a tie in the
  medians counting one half. Cars without a median are left out; with no such pair it is None;
- the calibrated offer: `offers` reads, from the gate-format results of the model's forecast table
  run through `gate_run.run_gate` (`gate_day.prepare_day`, then `gate_calibrate.run_job`), the
  test-day scores of the fixed and one-way rungs of each empirical-departure configuration and of
  its oracle partner, under tier A (driver-protected) and, beside it and never combined with it,
  tier B (deadline contract) with its stay-to-bound check (plan section 6; `gate_calibrate`).

Every car of a counting day's fleet must have a row in the table at each level scored; a missing
row raises `ValueError` (`forecast_model.TableForecaster` refuses it too).
"""

import math

import numpy as np

from fleetreserve.forecast_model import KEY_COLUMNS, check_table

OFFER_RUNGS = ("fixed", "one-way")
CONCORDANCE_CELLS = 4_000_000  # pairs compared at once in `concordance_index`


def pinball_loss(truth, forecast, level: float) -> float | None:
    """The mean pinball loss at `level` over the pairs where the forecast is not NaN; None when
    there is none."""
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    y = np.asarray(truth, dtype=float)
    q = np.asarray(forecast, dtype=float)
    if y.shape != q.shape:
        raise ValueError("truth and forecast differ in length")
    keep = ~np.isnan(q)
    if not keep.any():
        return None
    u = y[keep] - q[keep]
    return float(np.mean(np.maximum(level * u, (level - 1) * u)))


def concordance_index(times, scores) -> float | None:
    """Harrell's concordance index without censoring (module text): a later median is read as a
    later departure."""
    t = np.asarray(times, dtype=float)
    s = np.asarray(scores, dtype=float)
    if t.shape != s.shape:
        raise ValueError("times and scores differ in length")
    keep = ~np.isnan(s)
    t, s = t[keep], s[keep]
    agree = ties = pairs = 0
    step = max(1, CONCORDANCE_CELLS // max(1, t.size))
    for start in range(0, t.size, step):
        ti = t[start : start + step, None]
        si = s[start : start + step, None]
        comparable = t[None, :] > ti  # each pair once: the later car in the column
        pairs += int(comparable.sum())
        agree += int((comparable & (s[None, :] > si)).sum())
        ties += int((comparable & (s[None, :] == si)).sum())
    if pairs == 0:
        return None
    return (agree + 0.5 * ties) / pairs


def _lookup(table) -> dict:
    checked = check_table(table)
    keys = zip(*(checked[c] for c in KEY_COLUMNS), checked["kappa"], strict=True)
    columns = ("d_hat", "r_hat", "e_cap", "e_need", "d_median")
    values = checked[list(columns)].to_numpy(float)
    return {key: dict(zip(columns, values[k], strict=True)) for k, key in enumerate(keys)}


def unit_scores(unit_days, table, kappas) -> dict:
    """Per level (written as text), the losses and the concordance index of one unit (module
    text)."""
    rows = _lookup(table)
    truth = {"departure": [], "rate": [], "room": [], "need": []}
    keys = []
    for day in unit_days.days:
        for car, true in zip(day.fleet, day.day.cars, strict=True):
            truth["departure"].append(car.departure_exact)
            truth["rate"].append(true.rate)
            truth["room"].append(true.room)
            truth["need"].append(true.need(day.day.rho))
            keys.append((day.day.unit, car.site, day.record.date, car.session_id))
    out = {}
    for kappa in kappas:
        found = []
        for key in keys:
            row = rows.get((*key, float(kappa)))
            if row is None:
                raise ValueError(f"the forecast table has no row for {key} at kappa {kappa}")
            found.append(row)

        def column(name, found=found):
            return np.array([r[name] for r in found], dtype=float)

        d_hat = column("d_hat")
        out[repr(float(kappa))] = {
            "cars": len(keys),
            "cars_with_departure_forecast": int((~np.isnan(d_hat)).sum()),
            "pinball_loss": {
                "departure_hours": _loss(truth["departure"], d_hat, kappa),
                "rate_kw": _loss(truth["rate"], column("r_hat"), kappa),
                "cap_kwh": _loss(truth["room"], column("e_cap"), kappa),
                "need_kwh": _loss(truth["need"], column("e_need"), 1 - kappa),
            },
            "concordance_index": _concordance(truth["departure"], column("d_median")),
        }
    return out


def _loss(truth, forecast, level) -> dict:
    return {
        "level": level,
        "cars_scored": int((~np.isnan(forecast)).sum()),
        "mean": pinball_loss(truth, forecast, level),
    }


def _concordance(times, medians) -> dict:
    return {
        "cars_scored": int((~np.isnan(medians)).sum()),
        "value": concordance_index(times, medians),
    }


def _rungs(entry: dict) -> dict:
    """The offer rungs' results of one configuration and unit: tier A's scale, reason and test-day
    scores as the gate wrote them, and under `tier_b` the same for tier B (plan section 6) with its
    stay-to-bound check, when the gate scored it. The tiers are never combined."""
    if "rungs" not in entry:
        return {"reason": entry.get("failed") or entry.get("reason", "no result")}
    out = {}
    for rung in OFFER_RUNGS:
        found = entry["rungs"][rung]
        out[rung] = {
            "beta_hat": found["beta_hat"],
            "reason": found["reason"],
            "test": found["test"],
        }
        if "tier_b" in found:
            tier_b = found["tier_b"]
            out[rung]["tier_b"] = {
                "beta_hat": tier_b["beta_hat"],
                "reason": tier_b["reason"],
                "test": tier_b["test"],
                "stay_to_bound": tier_b["stay_to_bound"],
            }
    return out


def offers(document: dict, matrix, unit: str) -> dict:
    """Per empirical-departure configuration of `matrix`, the calibrated offers of `unit` and of
    the configuration's oracle partner, from a gate-format results document."""
    out = {}
    for c in matrix.configurations:
        if c.departure == "oracle" or c.name not in document["configurations"]:
            continue
        own = document["configurations"][c.name]["units"][unit]
        partner = document["configurations"].get(c.oracle_partner)
        out[c.name] = {
            "kappa": c.kappa,
            "family": c.family,
            "departure_seen": c.departure_seen,
            "driver_risk": c.driver_risk,
            "rungs": _rungs(own),
            "oracle_partner": {
                "name": c.oracle_partner,
                "rungs": {"reason": "not run"}
                if partner is None
                else _rungs(partner["units"][unit]),
            },
        }
    return out


def finite(value):
    """`value` with every NaN replaced by None, for a JSON file."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

"""The neural hazard model, the plan's main forecasting model, as a departure model (plan sections
2, 7, 13 WP8; `forecast_model.DepartureModel`).

Plan section 7: "discrete-time neural hazard over plugged-in cars (Set Transformer encoder, site
embeddings, shared day factor by amortised inference, fractional last hour, energy head)". The
network is `hazard_net`, the inputs and sets `hazard_data`, the training `hazard_fit`, the curve
rules `hazard_curves`, the settings and their defaults `hazard_settings`. This module fits it on
the training units, reads a forecast table from it and meets `learn_run`'s interface:
`python -m fleetreserve.learn_run --model hazard`, `dial_choice --model hazard`.

Fit: `fit_units` is `truth_build.TRAIN_UNITS`, their rows pooled. The table is
`learn_features.build_table` plus `learn_features.with_context`. A set is the cars of one site-day
of a unit (`hazard_data.make_sets`); every car plugged in at τ is in its set (on counting and
non-counting days). The departure is fitted on every row with a departure label. The energy head is
fitted on the rows where rate, room and protected energy are all numbers (`hazard_data.fit_data`,
`has_energy`); `models.lgbm` fits each quantity on the rows where that quantity and the room are
numbers (`models.lgbm.rows_for`). The two choices are the same set of rows when every row with a
room has a rate: protected energy is `(1 - rho)` times the room (`learn_features.build_table`), so
it is a number where the room is; the room is a number only on rows the rules keep in a fleet
(`days_from_truth.apply_car_rules`, `room.where(in_fleet)`); and the rate is a positive number on
every row of a truth table in which `truth_verify.check_tables` (run by `verify_role`) finds no
problem (it reports a rate that is not positive). On a cars table that was not checked so and has
a room but no rate on a row, the energy head is fitted on fewer rows than `models.lgbm` fits the
room on. A test pins the equality on made-up cars (`tests/learn/test_hazard_data.py`).

Predict: the frame is `learn_features.feature_frame` of the units to forecast; the context columns
are set here from `site` and `date`, as `models.lgbm_ctx` does. A site with no entry in the
committed mappings fails the model (`ValueError`). The forecast of a car is a function of the
fitted model and of the cars of its own site-day only: their inputs (with the context columns as
set here), their site and their order in the set, which their session ids give
(`hazard_data.make_sets` orders the cars of a set by session id as text). It is not a function
of the order of the rows of the frame (when the session ids of a set differ), nor of other
site-days, nor of any label or outcome column, nor of `declared_departure_h`. The order in the
set enters because `curves` gives the network the cars of a set as one float32 array in that
order, and a float32 result can depend on the order of the cars: for example a sum over the
cars, as in the softmax and the weighted sum of `hazard_net.Attention`, adds the same terms in
another order. `d_hat`, `d_median` and `survival_h<t>` are read from the curve by
`hazard_curves.lower_quantiles`: the first grid point at which a comparison of the curve with the
level holds, or NaN where none does. A `survival_h<t>` cell is also NaN where the car's
probability of being present at the start of hour t is below `min_presence_probability`
(`hazard_curves.departure_table`). So a change of the curve, however small, can move a bound to
another grid point, or make a cell NaN or a number, where the curve is that close to the level or
to the floor. On made-up cars at four levels a test observes that reversing the order of the
cars of every set changes no number in the columns `d_hat`, `r_hat`, `e_cap`, `e_need`,
`d_median` and `survival_h<t>` by 1e-5 or more and leaves every NaN in place
(`tests/learn/test_hazard_likelihood.py`, point 6). A site not seen in fitting gets the
unknown-site vector, so its name changes nothing (its site type and holiday flags, which come
from the committed mappings, do).

Table, per car and level kappa, from the car's survival curve (with the day factor on, the
prior-weighted average over its Gauss-Hermite nodes, so no random draw: `hazard_net.prior_nodes`):

- `d_hat`: the lower kappa-quantile of departure given presence at τ, in hours from τ, on the
  grid's points (multiples of a quarter hour up to 12 h with the default grid;
  `hazard_curves`); `d_median` the median;
- `survival_h1` to `survival_h8`: the kappa-quantile of departure given presence at the start of
  service hour t, read off the same curve (`hazard_curves`); at least t; it may equal t. It
  is a bound given presence at that hour, not the bound made at τ, and it does not fall as t
  rises (tested). With the day factor on the curve is a mixture over the nodes, so conditioning
  on presence gives more weight to the nodes in which cars stay longer. It is NaN
  (the gate then keeps the bound of the hour before) where it would be an extrapolation: for
  every car of a set when fewer than `min_presence_support` fit rows (default 100) of the set's
  own site type have a departure at or above t (cars of that type present at the start of hour
  t; the count is over every fit row with a departure label, not only fleet rows; `site_type` is
  1 for a workplace and 0 otherwise, `learn_features.CONTEXT_COLUMNS`), and for a car whose own
  modelled probability of being present at t is below `min_presence_probability` (default
  0.001). So long stays at one site type do not vouch for an hour that the other type's cars
  never reach. A bin that starts after every fit departure enters the likelihood of no fit car,
  and a number read from it would be an extrapolation (`hazard_curves`). A set has one site type;
  `site_types_of` raises if the rows of a set disagree;
- `r_hat`, `e_cap`: kappa-quantiles of rate (kW) and room (kWh); `e_need` the (1 - kappa)-quantile
  of protected energy `(1 - rho) * room` (kWh), from the energy head, each raised to at least 0
  after undoing the standardisation (levels are not forced into order).

Levels: the model is fitted at the kappas of the run matrix (`make_model`) and can be asked for
those only: the energy head has one output per quantity and level. Departure bounds could be read
at any level.

Inputs named in the plan that are not among `hazard_data.INPUT_COLUMNS`: energy drawn before τ,
regime, event rows (`learn_features.build_table` can join event columns when it is given an
events table; this model does not read them), school terms and calendars other than the
public-holiday flag, weather forecasts (`hazard_data`). `models.hazard_tuned_inputs` is the tuned
model with those of them that `hazard_inputs` builds as extra input columns (`model_inputs`).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import torch

from fleetreserve import hazard_curves, hazard_fit
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    empty_table,
)
from fleetreserve.hazard_data import (
    INPUT_COLUMNS,
    Scaler,
    SetIndex,
    fit_data,
    make_sets,
    site_codes,
)
from fleetreserve.hazard_net import HazardNet, prior_nodes
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.learn_features import OTHER_SITE, WORKPLACE, build_table, with_context
from fleetreserve.models.lgbm import levels_of, matrix_kappas
from fleetreserve.truth_build import TRAIN_UNITS

QUANTITIES = ("rate", "room", "need")
# The columns of a frame the model looks at: the keys it sets up with, and its inputs.
_READ = frozenset((*KEY_COLUMNS, *INPUT_COLUMNS))


def energy_outputs(levels: dict[str, tuple[float, ...]]) -> list[tuple[str, float]]:
    """The energy head's outputs: (quantity, level) for rate, room and need, in this order."""
    return [(q, level) for q in QUANTITIES for level in levels[q]]


def site_types_of(context: pd.DataFrame, sets: SetIndex) -> np.ndarray:
    """Per set, the `site_type` of its rows; `ValueError` if the rows of a set disagree."""
    kinds = context["site_type"].to_numpy(int)
    out = np.empty(sets.n_sets, dtype=int)
    for k in range(sets.n_sets):
        found = np.unique(kinds[sets.members(k)])
        if len(found) != 1:
            raise ValueError(f"hazard: the rows of the set {sets.keys[k]} have site types {found}")
        out[k] = found[0]
    return out


def presence_support_of(table: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per site type, the number of fit rows of that type whose departure label is at or above
    each service hour 1 to 8 (module text, `survival_h1` to `survival_h8`)."""
    departures = table["label_departure_h"].to_numpy(float)
    kinds = table["site_type"].to_numpy(int)
    return {
        kind: np.array(
            [
                int(((departures >= t) & (kinds == kind)).sum())
                for t in range(1, len(SURVIVAL_COLUMNS) + 1)
            ]
        )
        for kind in (OTHER_SITE, WORKPLACE)
    }


@dataclass
class SetCurves:
    """`HazardModel.set_curves`: the number of rows of the frame, then per set its rows, its site
    type and the survival curves (cars, bins + 1) of its cars; the standardised energy outputs
    (rows, outputs)."""

    n: int
    rows: list[np.ndarray]
    kinds: np.ndarray
    survival: list[np.ndarray]
    energy: np.ndarray


class HazardModel:
    name = "hazard"
    fit_units: tuple[str, ...] = TRAIN_UNITS

    def __init__(
        self,
        *,
        rho: float,
        kappas: Sequence[float],
        settings: HazardSettings | None = None,
    ) -> None:
        self.rho = rho
        self.levels = levels_of(kappas)
        self.settings = HazardSettings() if settings is None else settings
        self.outputs = energy_outputs(self.levels)
        self.net: HazardNet | None = None
        self.scaler: Scaler | None = None
        self.fit_sites: tuple[str, ...] = ()
        self.energy_mean: np.ndarray | None = None
        self.energy_std: np.ndarray | None = None
        self.presence_support: dict[int, np.ndarray] | None = None
        self.history: list[float] = []

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit on the pooled cars tables of `fit_units` (module text)."""
        if len(frame) == 0:
            raise ValueError("hazard: there are no training cars to fit on")
        table = self._fit_table(frame)
        self.fit_sites = tuple(sorted(set(table["site"].astype(str))))
        self.presence_support = presence_support_of(table)
        self.scaler = self._fit_scaler(table)
        data = fit_data(table, self.scaler, self.fit_sites, self.settings)
        self.energy_mean, self.energy_std = data.energy_mean, data.energy_std
        position = {q: k for k, q in enumerate(QUANTITIES)}
        fitted = hazard_fit.fit(
            data,
            self.settings,
            len(self.fit_sites),
            np.array([level for _, level in self.outputs]),
            np.array([position[q] for q, _ in self.outputs]),
        )
        self.net, self.history = fitted.net, fitted.history

    def _fit_table(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The fit table of the cars tables `frame`: `build_table` with the context columns."""
        return with_context(build_table(frame, rho=self.rho))

    def _fit_scaler(self, table: pd.DataFrame):
        """The encoding of the inputs, fitted on the fit table (`hazard_data.Scaler`)."""
        return Scaler.fit(table, self.settings)

    def inputs(self, frame: pd.DataFrame):
        """The standardised inputs and the sets of the frame to forecast (module text)."""
        return self._inputs_of(self._context(frame))

    def _context(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The columns of `frame` the model looks at, with the context columns set anew."""
        return with_context(frame[[c for c in frame.columns if c in _READ]])

    def _inputs_of(self, context: pd.DataFrame):
        missing = [c for c in INPUT_COLUMNS if c not in context.columns]
        if missing:
            raise KeyError(f"the frame lacks the input columns {missing}")
        return self.scaler.transform(context), make_sets(context)

    def curves(self, x: np.ndarray, site: int):
        """Survival curves (cars, bins + 1) and standardised energy outputs of one set."""
        s = self.settings
        nodes, weights = prior_nodes(s)
        device = torch.device(s.device)
        with hazard_fit.torch_state(s), torch.inference_mode():
            tensor = torch.as_tensor(x[None]).to(device)
            mask = torch.ones(tensor.shape[:2], dtype=torch.bool, device=device)
            encoded, fleet = self.net.encode(tensor, mask)
            energy = self.net.energy_outputs(encoded)[0].cpu().numpy().astype(np.float64)
            site_code = torch.as_tensor([site]).to(device)
            if s.day_factor:
                mean, log_scale = self.net.prior_parameters(fleet)
                node = torch.as_tensor(nodes, dtype=torch.float32).to(device)
                factor = mean + torch.exp(log_scale) * node  # (points, dim)
                points = factor.shape[0]
                repeated = encoded.expand(points, -1, -1)
            else:
                factor, points, weights = None, 1, np.ones(1)
                repeated = encoded
            site_vector = self.net.site_vector(site_code.expand(points), encoded.shape[1])
            logits = self.net.hazard_logits(repeated, factor, site_vector)
            hazards = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        survival = hazard_curves.survival_curves(hazards)  # (points, cars, bins + 1)
        return np.tensordot(weights, survival, axes=1), energy

    def predict(self, frame: pd.DataFrame, kappas: Sequence[float]) -> pd.DataFrame:
        asked = self.check_levels(kappas)
        if len(frame) == 0:
            return empty_table()
        found = self.set_curves(frame)
        departure = self.departure_columns(found, asked)
        return forecast_table(frame, kappas, self.rho, departure, self.energy_columns(found, asked))

    def check_levels(self, kappas: Sequence[float]) -> set[float]:
        """The levels asked for; `ValueError` before fit or for a level not fitted."""
        if self.net is None:
            raise ValueError("predict before fit")
        asked = {float(k) for k in kappas}
        for kappa in asked:
            for quantity, level in energy_outputs(levels_of([kappa])):
                if (quantity, level) not in self.outputs:
                    raise ValueError(f"no {quantity} output was fitted at level {level}")
        return asked

    def set_curves(self, frame: pd.DataFrame) -> SetCurves:
        """Per set of a non-empty `frame`, its rows, its site type, the survival curves of its
        cars (`curves`) and their standardised energy outputs. `predict` computes the forecast
        columns of its table (`d_hat`, `d_median`, `survival_h1` to `survival_h8`, `r_hat`,
        `e_cap`, `e_need`) from these, with the model's fit-row counts, settings and energy
        scaling; the key columns come from the frame and `kappa` and `rho` from the levels asked
        and the model. `models.hazard_ens` reads them for each member."""
        context = self._context(frame)
        x, sets = self._inputs_of(context)
        codes = site_codes(sets, self.fit_sites)
        kinds = site_types_of(context, sets)
        energy = np.full((len(frame), len(self.outputs)), math.nan)
        rows, survival = [], []
        for k in range(sets.n_sets):
            members = sets.members(k)
            curve, standardised = self.curves(x[members], int(codes[k]))
            rows.append(members)
            survival.append(curve)
            energy[members] = standardised
        return SetCurves(len(frame), rows, kinds, survival, energy)

    def departure_columns(self, found: SetCurves, asked: set[float]) -> dict:
        """Per level, `d_hat`, `d_median` and `survival_h1` to `survival_h8` read from the curves
        of `found` by `hazard_curves.departure_table`, with the presence rules of the module
        text (this model's fit-row counts and settings)."""
        times = hazard_curves.bin_starts(self.settings)
        supported = {
            kind: support >= self.settings.min_presence_support
            for kind, support in self.presence_support.items()
        }
        n = found.n
        departure = {
            k: {c: np.full(n, math.nan) for c in ("d_hat", "d_median", *SURVIVAL_COLUMNS)}
            for k in asked
        }
        for rows, kind, survival in zip(found.rows, found.kinds, found.survival, strict=True):
            columns = hazard_curves.departure_table(
                times,
                survival,
                sorted(asked),
                supported[int(kind)],
                self.settings.min_presence_probability,
            )
            for kappa, part in columns.items():
                for name, values in part.items():
                    departure[kappa][name][rows] = values
        return departure

    def energy_columns(self, found: SetCurves, asked: set[float]) -> dict:
        """Per level, `r_hat`, `e_cap` and `e_need` from the standardised outputs of `found`."""
        return {
            k: {
                "r_hat": self._energy(found.energy, "rate", k),
                "e_cap": self._energy(found.energy, "room", k),
                "e_need": self._energy(found.energy, "need", round(1 - k, 12)),
            }
            for k in asked
        }

    def _energy(self, energy: np.ndarray, quantity: str, level: float) -> np.ndarray:
        column = self.outputs.index((quantity, level))
        k = QUANTITIES.index(quantity)
        values = self.energy_mean[k] + self.energy_std[k] * energy[:, column]
        return np.maximum(values, 0.0)


def forecast_table(frame, kappas, rho: float, departure: dict, energy: dict) -> pd.DataFrame:
    """The forecast table of `frame` at `kappas` from per-level departure and energy columns."""
    parts = []
    for kappa in kappas:
        k = float(kappa)
        part = pd.DataFrame({c: frame[c].astype(str).to_numpy() for c in KEY_COLUMNS})
        part["kappa"] = k
        part["rho"] = float(rho)
        part["d_hat"] = departure[k]["d_hat"]
        part["r_hat"] = energy[k]["r_hat"]
        part["e_cap"] = energy[k]["e_cap"]
        part["e_need"] = energy[k]["e_need"]
        part["d_median"] = departure[k]["d_median"]
        for name in SURVIVAL_COLUMNS:
            part[name] = departure[k][name]
        parts.append(part[list(TABLE_COLUMNS)])
    return pd.concat(parts, ignore_index=True)


def make_model(matrix) -> HazardModel:
    return HazardModel(rho=matrix.rho, kappas=matrix_kappas(matrix))

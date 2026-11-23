"""The day-size quantile net (plan section 7: "Day-size quantile net: pinball loss on fleet
encoding and context; label: T1(b)'s exact scale of the certified shape on true training days
(quantile set in WP8); one calibrated scalar on top"), as a predictor of a day's size beside the
boosters of `size_gbm`.

Input per labelled day: the cars plugged in at τ, each as the departure model's input vector
(`size_net_inputs`). Label: the boosters' `size_gbm.LABEL`, the exact safe scale of the day's
certified shape on its true cars (`size_label`), clipped to [0, 1]. Output: one prediction per
level of `LEVELS`, the boosters' levels. The one calibrated scalar on top is not made here: the
predictions are raw sizes of `size_whatif.predicted_sizings`, and `size_resolve.calibrate` (or
`size_scale.calibrate`) calibrates them as it calibrates the boosters'.

Network (`SizeNet`), a deep-sets network over the cars of a day:

- car encoding: a small network (`hazard_net.mlp`: one hidden layer of `width` with a ReLU, an
  output of `width`) on each car's input vector;
- fleet encoding: the mean of the car encodings over the day's cars (`hazard_net.masked_mean`),
  the same for any order of the cars (up to rounding) and of any number of cars;
- context: the mean over the day's cars of their input vectors, which for an input that is a
  function of the site-day (date parts, the number of cars at τ, holidays, site type, regime,
  weather, OxCGRT, 9/80 Fridays, school terms) is its value;
- head: a small network (one hidden layer of `hidden` with a ReLU) on the fleet encoding and the
  context together, one output per level fitted. Its output layer's weights start at 0 and its
  bias at the best constant of each level on the training labels (`size_gbm.constant`), so a
  network that has learned nothing is exactly that constant.

Fit (`fit`): the fit days are split under `seed` into training days and a share
`validation_share` of validation days (`split_days`); the mean pinball loss over the days of a
batch and the levels (`loss`), AdamW with `NetSettings`' learning rate and weight decay
(PyTorch's default betas and epsilon), the gradient norm clipped to `grad_clip`, at most `epochs`
passes over the training days in batches of `batch_sets` days in an order drawn per epoch from
`seed`; after each pass the validation loss, and the network kept is the one of the epoch with
the smallest validation loss, epoch 0 (the starting constant) among them; initial weights drawn
under `seed` with the global random state kept; on the CPU at `threads` threads with
deterministic algorithms (`hazard_fit.torch_state`, which leaves the thread count set). Two fits
on the same rows give the same bytes. Prediction (`predict`) takes one day at a time, so a day's
output reads its own cars only and no padding.

Training units only (`fit_days`, `leave_one_unit_out`): every unit of the labelled days a network
is fitted on, and every unit of the table a leave-one-unit-out run reads, must be one of
`truth_build.TRAIN_UNITS` (`learn_tune.check_units`); any other raises `learn_tune.NotTraining`
before anything is fitted.

`leave_one_unit_out(rows, cars)`: as `size_gbm.leave_one_unit_out`, per unit u and level a
network fitted on the labelled days of the other units (the same rows as the boosters'), its
encoder fitted on their cars only, and read at u's days; a unit whose other units hold no day gets
NaN. Returns, per level, {"net": one prediction per row}. The other units' labels can depend on
u's outcomes, as the boosters' can (`size_whatif`, step 3).

Readings (the plan's sentence leaves these open; each with the alternative not taken; N1 to N3
are in `size_net_inputs`):

N4. Context enters as the mean of the cars' inputs (module text), not as a separate vector of the
    site-day columns. [Alternative: a separate context vector of those columns only.]
N5. One network per level of `LEVELS`, as the boosters are one model per level, each fitted and
    kept on its own level's pinball loss; the plan's "quantile set in WP8" is read as the level
    being chosen from these, as the boosters' are reported at every level. [Alternative: one
    network with one output per level, fitted on the mean pinball loss over the levels; on
    made-up days its epoch kept suited the middle levels and not the lowest.]
N6. The settings are fixed here, not tuned; the only choice made on data is the epoch kept, on a
    seeded fifth of the fit days held out of training, the same seed for every fit.
    [Alternative: tuning the settings on folds of the training units, as `hazard_tune` does; or
    a fixed number of epochs with no validation days, which on made-up days overfitted.]
N7. Predictions are not clipped (as the boosters'); the arms of `size_scale.sizes` clip them.
N8. The output layer starts at weights 0 and each level's best constant on the training labels.
    [Alternative: PyTorch's default initialisation, with which the epoch-0 network is not the
    constant.]
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import torch
from torch import nn

from fleetreserve import learn_tune, size_gbm, size_net_inputs
from fleetreserve.hazard_data import SetIndex, batch_of
from fleetreserve.hazard_fit import pinball, torch_state
from fleetreserve.hazard_net import masked_mean, mlp

LEVELS = size_gbm.LEVELS
LABEL = size_gbm.LABEL
KIND = "net"


@dataclass(frozen=True)
class NetSettings:
    """The fixed settings of the network and its fit (module text, reading N6)."""

    width: int = 32
    hidden: int = 32
    epochs: int = 200
    batch_sets: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    validation_share: float = 0.2
    seed: int = 0
    threads: int = 1
    input_clip: float = size_net_inputs.INPUT_CLIP

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_share < 1.0:
            raise ValueError("the validation share must lie strictly between 0 and 1")
        if self.epochs < 0 or self.batch_sets < 1:
            raise ValueError("epochs must be at least 0 and batch_sets at least 1")


class SizeNet(nn.Module):
    """Car encoding, fleet encoding, context and head (module text)."""

    def __init__(self, n_inputs: int, n_levels: int, settings: NetSettings) -> None:
        super().__init__()
        self.car = mlp(n_inputs, settings.width, settings.width)
        self.head = mlp(settings.width + n_inputs, settings.hidden, n_levels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fleet = masked_mean(self.car(x), mask)
        context = masked_mean(x, mask)
        return self.head(torch.cat([fleet, context], dim=-1))


def build_net(n_inputs: int, start, settings: NetSettings) -> SizeNet:
    """A network with initial weights drawn under the settings' seed, the global random state
    kept, its output layer's weights 0 and its bias `start` (one value per level), so that it
    starts as the constant `start` on every day."""
    with torch.random.fork_rng():
        torch.manual_seed(settings.seed)
        net = SizeNet(n_inputs, len(start), settings)
    with torch.no_grad():
        net.head[2].weight.zero_()
        net.head[2].bias.copy_(torch.as_tensor(np.asarray(start, dtype=np.float32)))
    return net


def loss(prediction: torch.Tensor, target: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """The mean pinball loss over days and levels; `prediction` (days, levels), `target` (days)."""
    return pinball(prediction, target[:, None], levels).mean()


@dataclass
class Fitted:
    """A fitted network and its record: per epoch the mean training loss over its batches
    (weighted by days) and the validation loss, the epoch kept (0: the starting constant) and
    the numbers of training and validation days."""

    net: SizeNet
    history: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)
    best_epoch: int = 0
    training_days: int = 0
    validation_days: int = 0

    def record(self) -> dict:
        return {
            "training_days": self.training_days,
            "validation_days": self.validation_days,
            "epochs_run": len(self.history),
            "epoch_kept": self.best_epoch,
            "validation_loss_at_start": self.validation[0],
            "validation_loss_kept": self.validation[self.best_epoch],
        }


def split_days(n_sets: int, settings: NetSettings) -> tuple[np.ndarray, np.ndarray]:
    """(training sets, validation sets): a share `validation_share` of the sets, at least one,
    drawn under the seed (a generator of its own), the rest for training, each in order."""
    if n_sets < 2:
        raise ValueError("a fit needs at least two labelled days (one for validation)")
    held = min(n_sets - 1, max(1, round(settings.validation_share * n_sets)))
    order = np.random.default_rng([settings.seed, 1]).permutation(n_sets)
    return np.sort(order[held:]), np.sort(order[:held])


def _validation_loss(net, x, sets, chosen, target, at, settings) -> float:
    """The mean pinball loss over the days `chosen` and the levels, the network in evaluation."""
    site = np.zeros(sets.n_sets, dtype=int)
    total = 0.0
    net.eval()
    with torch.no_grad():
        for first in range(0, len(chosen), settings.batch_sets):
            part = chosen[first : first + settings.batch_sets]
            batch = batch_of(x, sets, site, part)
            found = net(torch.as_tensor(batch.x), torch.as_tensor(batch.mask))
            total += float(pinball(found, target[torch.as_tensor(part)][:, None], at).sum())
    net.train()
    return total / (len(chosen) * len(at))


def fit(x: np.ndarray, sets: SetIndex, y, levels, settings: NetSettings) -> Fitted:
    """A network fitted on the sets `sets` of the car inputs `x` with labels `y` (module text):
    trained on the training days of `split_days`, kept at the epoch (0 to `epochs`) with the
    smallest validation loss, the first such epoch on a tie."""
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (sets.n_sets,) or not np.isfinite(y).all():
        raise ValueError("one finite label per set is needed")
    train, validation = split_days(sets.n_sets, settings)
    start = [size_gbm.constant(y[train], float(level)) for level in levels]
    site = np.zeros(sets.n_sets, dtype=int)
    with torch_state(settings):
        net = build_net(x.shape[1], start, settings)
        optimiser = torch.optim.AdamW(
            net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        order = np.random.default_rng(settings.seed)
        at = torch.as_tensor(np.asarray(levels, dtype=np.float32))
        target = torch.as_tensor(y)
        fitted = Fitted(net, training_days=len(train), validation_days=len(validation))
        net.train()
        fitted.validation.append(_validation_loss(net, x, sets, validation, target, at, settings))
        kept = {k: v.detach().clone() for k, v in net.state_dict().items()}
        for epoch in range(1, settings.epochs + 1):
            total = 0.0
            permutation = train[order.permutation(len(train))]
            for first in range(0, len(train), settings.batch_sets):
                chosen = permutation[first : first + settings.batch_sets]
                batch = batch_of(x, sets, site, chosen)
                found = net(torch.as_tensor(batch.x), torch.as_tensor(batch.mask))
                step = loss(found, target[torch.as_tensor(chosen)], at)
                optimiser.zero_grad()
                step.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
                optimiser.step()
                total += float(step.detach()) * len(chosen)
            fitted.history.append(total / len(train))
            fitted.validation.append(
                _validation_loss(net, x, sets, validation, target, at, settings)
            )
            if fitted.validation[-1] < fitted.validation[fitted.best_epoch]:
                fitted.best_epoch = epoch
                kept = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(kept)
        net.eval()
        return fitted


def predict(net: SizeNet, x: np.ndarray, sets: SetIndex, settings: NetSettings) -> np.ndarray:
    """The outputs of every set, one day at a time (module text), shape (sets, levels)."""
    out = []
    with torch_state(settings), torch.no_grad():
        net.eval()
        for k in range(sets.n_sets):
            cars = torch.as_tensor(np.ascontiguousarray(x[sets.members(k)]))[None]
            mask = torch.ones(cars.shape[:2], dtype=torch.bool)
            out.append(net(cars, mask)[0].numpy().astype(float))
    width = len(net.head[2].bias)
    return np.array(out, dtype=float).reshape(sets.n_sets, width)


def check_fit_units(units) -> tuple[str, ...]:
    """The units, each named once, when each is a training unit; `learn_tune.NotTraining`
    otherwise (module text)."""
    return learn_tune.check_units(tuple(dict.fromkeys(str(u) for u in units)))


@dataclass(frozen=True)
class FittedDays:
    """A network fitted on labelled days, with the encoder of their cars and its fit's record
    (`Fitted.record`)."""

    encoder: size_net_inputs.Encoder
    net: SizeNet
    record: dict


def fit_days(
    rows: pd.DataFrame,
    cars: pd.DataFrame,
    levels: Sequence[float] = LEVELS,
    settings: NetSettings | None = None,
) -> FittedDays:
    """A network fitted on the labelled days `rows` (`unit`, `site`, `date`, `LABEL`) with the
    car frame `cars` (`size_net_inputs.car_frame`); training units only (module text)."""
    settings = NetSettings() if settings is None else settings
    check_fit_units(rows["unit"])
    sets = size_net_inputs.fleets(cars, rows)
    fit_cars = cars.iloc[sets.rows].reset_index(drop=True)
    compact = SetIndex(np.arange(len(fit_cars), dtype=int), sets.starts, sets.keys)
    encoder = size_net_inputs.Encoder.fit(fit_cars, settings.input_clip)
    x = encoder.transform(fit_cars)
    found = fit(x, compact, rows[LABEL].to_numpy(float), levels, settings)
    return FittedDays(encoder, found.net, found.record())


def predict_days(
    fitted: FittedDays, rows: pd.DataFrame, cars: pd.DataFrame, settings: NetSettings | None = None
) -> np.ndarray:
    """The outputs of a fitted network at the labelled days `rows`, shape (rows, levels)."""
    settings = NetSettings() if settings is None else settings
    sets = size_net_inputs.fleets(cars, rows)
    day_cars = cars.iloc[sets.rows].reset_index(drop=True)
    compact = SetIndex(np.arange(len(day_cars), dtype=int), sets.starts, sets.keys)
    return predict(fitted.net, fitted.encoder.transform(day_cars), compact, settings)


def leave_one_unit_out(
    rows: pd.DataFrame,
    cars: pd.DataFrame,
    levels: Sequence[float] = LEVELS,
    settings: NetSettings | None = None,
    records: dict | None = None,
) -> dict[float, dict[str, np.ndarray]]:
    """Per level, {"net": predictions}, one per row of `rows` (module text). `records`, when
    given, gets per scored unit and level the record of the fit that predicts it
    (`Fitted.record`)."""
    settings = NetSettings() if settings is None else settings
    out = {float(level): {KIND: np.full(len(rows), np.nan)} for level in levels}
    if rows.empty:
        return out
    check_fit_units(rows["unit"])
    rows = rows.reset_index(drop=True)
    for unit in dict.fromkeys(rows["unit"]):
        scored = (rows["unit"] == unit).to_numpy(bool)
        if scored.all():
            continue
        for level in levels:
            fitted = fit_days(rows[~scored], cars, (float(level),), settings)
            found = predict_days(fitted, rows[scored], cars, settings)
            out[float(level)][KIND][scored] = found[:, 0]
            if records is not None:
                records.setdefault(str(unit), {})[f"{float(level):g}"] = fitted.record
    return out

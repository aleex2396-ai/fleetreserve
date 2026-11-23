"""Conformal training of the day-size model, the sizings `ct-<arm>` of `size_resolve_whatif` (plan
section 13, WP9: "conformal training kept only if it raises the development offer"; section 11,
O11; Yeh et al. (2025b)).

The day-size net (`size_net`) is fitted to the label by the pinball loss, and its prediction is
then calibrated. Here the same network on the same inputs is trained further for the quantity the
what-if reports: the offer after calibration. Each training step calibrates the scalar of T6 on
part of a batch of days, sized by the network, with a smoothed loss (`conformal_scalar`), applies
it to the rest of the batch and takes minus that offer as the loss, so the gradient reaches the
network both through the sizes of the days offered and through the calibration. At evaluation
nothing is smoothed: the network's outputs are raw sizes that go through `size_scale.sizes` and
the exact `size_resolve.calibrate` as every other sizing's, so T6's bound covers them as it covers
the net's (sizes fixed before the scored unit's calibration days are seen; the caveat of
`size_whatif`, step 3, applies unchanged).

Per labelled day (the rows of `size_net`, with `LABEL` and the day's certified value at scale 1
`VALUE`, which the boosters' context row holds): the cars plugged in at τ, each as the departure
model's input vector (`size_net_inputs`, unchanged). The value enters the training loss only,
never the network.

Start (`fit_start`): the day-size net at the level `alpha` (`size_net.fit_days` with the levels
`(alpha,)` and the settings `CtSettings.start`), so the same network, encoder and split of the fit
days into training and validation days (`size_net.split_days`) as the what-if's sizing
`net-<alpha>-<arm>`. Its one output `z` per day is the day's raw size.

One network per arm of `size_scale.ARMS`, each trained under its arm's rule (`size_scale.sizes`),
the arm's own clip being the link from `z` to the size `q` (`sized`):

- arm "scale": `q = min(max(z, FLOOR), 1)`, every day offers (`a = 1`);
- arm "abstain": a day offers when `z >= FLOOR`, with `q = min(z, 1)`. In training its offering
  weight is `a = sigmoid((z - FLOOR) / gate_tau)` and `q = min(max(z, Q_MIN), 1)` (`gate`).

Fit (`fit`), from the start's weights, on its training days. An epoch cuts each unit's training
days, in an order drawn from `seed`, into `round(m / batch_days)` batches (at least one) of near
equal size, and a unit with fewer than `MIN_BATCH` training days into none; the batches of all
units run in an order drawn from `seed`. A batch is split `splits_per_batch` times into a
calibration half and a prediction part: first in the batch's own order (its first half
calibrates), then in further orders drawn from `seed`. The step (`batch_offer`):

1. the network's sizes `q` and weights `a` of every day of the batch, in one pass;
2. per split, the smoothed scalar `g` on the calibration half (`conformal_scalar.smooth_scalar`,
   at `alpha` and the ramp width `tau`), capped at `1 / max q` over the batch (no day's scale
   above 1, as the what-if's grid stops at 1), and the offer share: the sum over the prediction
   part of `a g q v` over the sum of its values `v`; a split whose smoothed scalar is 0 (too
   little offering weight, `conformal_scalar`) offers 0;
3. the loss is minus the mean of the offer shares over the splits, and nothing else; AdamW with
   `CtSettings`' learning rate and weight decay (PyTorch's default betas and epsilon), the
   gradient norm clipped to `grad_clip`. A batch whose every split offers 0 has no gradient: no
   step is taken, and it is counted.

Before the first epoch and after each, the validation offer (`exact_offer`), with nothing
smoothed: per unit with validation days, the exact scalar (`conformal_scalar.exact_scalar`) on its
validation days that offer under the arm, capped at `1 / max q` over them, and the offer share of
the same days; the mean over the units weighted by their validation days. The network kept is the
one of the epoch with the largest validation offer, epoch 0 (the start: the day-size net) among
them, the first such epoch on a tie; a network kept at epoch 0 gives the net's outputs. On the CPU
at `threads` threads with deterministic algorithms (`hazard_fit.torch_state`); no draw of the
fit is random but those from `seed`, so two fits on the same rows give the same bytes. Prediction
(`size_net.predict`) takes one day at a time.

Training units only (`fit_start`, `leave_one_unit_out`): as `size_net`, every unit of the rows a
network is fitted on, and every unit a leave-one-unit-out run reads, must be one of
`truth_build.TRAIN_UNITS`; any other raises `learn_tune.NotTraining` before anything is fitted.

`leave_one_unit_out(rows, cars)`: per unit u, the start fitted on the labelled days of the other
units (its encoder on their cars only), then per arm a network trained from it on the same days,
read at u's days; a unit whose other units hold no day gets NaN. Returns, per arm, one raw size
per row.

Readings (the plan's words "conformal training" leave these open; each with the alternative not
taken):

C1. What is trained: the day-size model (plan section 7's day-size net, on its inputs), for the
    calibrated offer. The day's score is its label, fixed by its certificate and its true cars,
    which the trained model does not change, so T8's soft scale is not used. [Alternative: the
    bound route, where a learned model gives each car its bounds and the day's size is the soft
    exact scale (T8, `soft_scale`) of the certified shape on them, differentiated into that
    model; it is a second departure model, and it fixes a shape that T2's programme itself takes
    from the bounds.]
C2. The day's loss in training is the whole-day step at its label (the strict variant's score,
    plan section 6): it loses once its scale is above the label. [Alternatives: the one-way
    re-solve's own losses at the grid scales of the other units' days, which the what-if has only
    for units it scores and only after every unit is evaluated; the fixed plan's hour
    thresholds, which the labelled-days table does not hold.]
C3. One level: `alpha` is the plan's slot alpha 0.10, and the same network serves both tiers,
    each with its own scalar on top. [Alternative: one network per alpha, as the net has one per
    quantile level.]
C4. The smoothing is `conformal_scalar`'s ramp (its readings C4a, C4b), width `tau` 0.05, the
    smallest scale of the grid, fixed here. [Alternative: the exact scalar with the gradient of
    its one critical day (tau 0), which reaches one calibration day per batch.]
C5. A batch holds the days of one unit, because each unit is calibrated on its own days; batches
    of about 200 days, so calibration halves of about 100 days, the plan's 100 calibration days;
    `MIN_BATCH` 20 days, so a calibration half holds at least 10 days and the bound can hold
    (`crc.min_days`: 9 at alpha 0.10). [Alternative: batches mixing units, which would teach
    differences between units that each unit's own scalar then removes.]
C6. The loss is minus the prediction part's offer as a share of its value at scale 1, so that a
    batch of a large site weighs as much as one of a small site, with the scalar capped as the
    what-if caps it. [Alternative: minus the offer in kWh.]
C7. One network per arm, each trained under its arm's rule, `ct-scale` and `ct-abstain` each read
    through its own arm; the abstain arm's weight is a soft gate of width `gate_tau` 0.01 about
    `FLOOR`. [Alternative: one network read through both arms, as the net's are.]
C8. The epoch is kept on the exact offer of each unit's validation days calibrated on themselves
    (module text): how well the sizes order the days, read at alpha 0.10 on the start's
    validation days. [Alternative: the offer of random halves calibrated on the other half; with
    15 to 65 calibration days alpha 0.10 allows 0 to 5 losing days, so the criterion is mostly
    the one worst day.]
C9. The start is the day-size net at the level alpha, so conformal training is a fine-tuning of
    the net by the offer (Yeh et al. (2025b) describe training or fine-tuning), on the net's own
    training days, and its epoch 0 is the net. [Alternative: the same network from the constant
    size 0.5 on every day, trained by the offer alone. On made-up days of four units in T7's case
    (`tests/learn/ct_fake.py`) it offered about as much as the net with 400 days per unit (4.40
    against 4.38 kWh per counting test day, two held-out units) and less with 250 (3.8, 4.0 and
    4.3 against the net's 4.5, 4.3 and 4.5, four held-out units, three builds): the offer's
    gradient reaches only the calibration days within `tau` of their label, and the network
    fitted the other days' noise first.]
C10. The link is the arm's own clip, so the network's output is the raw size as the net's is.
    [Alternative: a sigmoid link of the output, which keeps a gradient on every day; trained
    from the constant on made-up design builds it offered less than the clip.]
C11. Each batch is split `splits_per_batch` 8 times per step, the objective being the mean offer
    over the splits. [Alternative: one split per batch (Yeh et al. (2025b), Algorithm 2), whose
    step reads the risk signal of one calibration half only.]
C12. The settings are fixed here, the network's and the start's being the net's (`NetSettings`;
    learning rate 0.003, weight decay 1e-4, clip 5 and 200 epochs as the net's); the only choices
    made on data are the epochs kept, the net's on its validation loss and this one on the offer,
    both on the same validation days.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import torch

from fleetreserve import conformal_scalar, crc, size_gbm, size_net, size_net_inputs, size_scale
from fleetreserve.hazard_data import SetIndex, batch_of
from fleetreserve.hazard_fit import torch_state

LABEL = size_gbm.LABEL
VALUE = "box_value_kwh"
ARMS = size_scale.ARMS
FLOOR = size_scale.FLOOR
KIND = "ct"
Q_MIN = 1e-6  # the smallest size in training (sizes must be above 0, `conformal_scalar`)
MIN_BATCH = 2 * (crc.min_days(crc.ALPHA_SLOT) + 1)


@dataclass(frozen=True)
class CtSettings:
    """The fixed settings of the fit and of its start (module text, reading C12)."""

    epochs: int = 200
    batch_days: int = 200
    splits_per_batch: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    alpha: float = crc.ALPHA_SLOT
    tau: float = 0.05
    gate_tau: float = 0.01
    seed: int = 0
    threads: int = 1
    start: size_net.NetSettings = size_net.NetSettings()

    def __post_init__(self) -> None:
        if self.epochs < 0 or self.batch_days < MIN_BATCH or self.splits_per_batch < 1:
            raise ValueError(
                f"epochs must be at least 0, batch_days at least {MIN_BATCH} and "
                "splits_per_batch at least 1"
            )
        if not (self.tau > 0 and self.gate_tau > 0):
            raise ValueError("tau and gate_tau must be positive")


def _check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return arm


def sized(z: torch.Tensor, arm: str) -> torch.Tensor:
    """The size of a raw size `z` in training (module text)."""
    if _check_arm(arm) == "scale":
        return torch.clamp(z, FLOOR, 1.0)
    return torch.clamp(z, Q_MIN, 1.0)


def gate(z: torch.Tensor, arm: str, settings: CtSettings) -> torch.Tensor:
    """The offering weight of a raw size in training (module text): 1 in the arm "scale"."""
    if _check_arm(arm) == "scale":
        return torch.ones_like(z)
    return torch.sigmoid((z - FLOOR) / settings.gate_tau)


def unit_order(units) -> list[str]:
    return list(dict.fromkeys(str(u) for u in units))


def epoch_batches(train, units, rng, settings: CtSettings) -> list[np.ndarray]:
    """The batches of one epoch (module text), each an array of day positions in the order whose
    first half is the first split's calibration half."""
    units = np.asarray([str(u) for u in units])
    batches = []
    for unit in unit_order(units[train]):
        days = train[units[train] == unit]
        if len(days) < MIN_BATCH:
            continue
        days = days[rng.permutation(len(days))]
        count = max(1, round(len(days) / settings.batch_days))
        batches += list(np.array_split(days, count))
    return [batches[i] for i in rng.permutation(len(batches))]


def batch_offer(net, x, sets, chosen, labels, values, arm, settings: CtSettings, rng):
    """Steps 1 and 2 of the module text on the batch `chosen`: the mean offer share over the
    splits as a tensor with its gradient, or None when every split offers 0."""
    chosen = np.asarray(chosen, dtype=int)
    half = len(chosen) // 2
    batch = batch_of(x, sets, np.zeros(sets.n_sets, dtype=int), chosen)
    z = net(torch.as_tensor(batch.x), torch.as_tensor(batch.mask))[:, 0].double()
    q, a = sized(z, arm), gate(z, arm, settings)
    s = torch.as_tensor(labels[chosen], dtype=torch.float64)
    v = torch.as_tensor(values[chosen], dtype=torch.float64)
    cap = 1.0 / q.max()
    orders = [np.arange(len(chosen))]
    orders += [rng.permutation(len(chosen)) for _ in range(settings.splits_per_batch - 1)]
    shares, any_offer = [], False
    for order in orders:
        first, rest = torch.as_tensor(order[:half]), torch.as_tensor(order[half:])
        g = conformal_scalar.smooth_scalar(
            q[first], s[first], a[first], settings.alpha, settings.tau
        )
        if not float(g.detach()) > 0:
            shares.append(torch.zeros((), dtype=torch.float64))
            continue
        any_offer = True
        used = torch.minimum(g, cap)
        shares.append((a[rest] * used * q[rest] * v[rest]).sum() / v[rest].sum())
    return torch.stack(shares).mean() if any_offer else None


def _outputs(net, x, sets, chosen, settings: CtSettings) -> np.ndarray:
    """The network's raw sizes at the days `chosen`, in batches, the network in evaluation."""
    out = []
    net.eval()
    with torch.no_grad():
        for first in range(0, len(chosen), settings.batch_days):
            part = np.asarray(chosen[first : first + settings.batch_days], dtype=int)
            batch = batch_of(x, sets, np.zeros(sets.n_sets, dtype=int), part)
            found = net(torch.as_tensor(batch.x), torch.as_tensor(batch.mask))[:, 0]
            out.append(found.double().numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros(0)


def exact_offer(raw, labels, values, groups, arm: str, alpha: float) -> float:
    """The validation offer of the module text for the raw sizes `raw` (one per day position);
    `groups`: one array of day positions per unit."""
    raw = np.asarray(raw, dtype=float)
    total = weight = 0.0
    for days in groups:
        offers = ~np.isnan(size_scale.sizes(raw[days], np.ones(len(days), dtype=bool), arm))
        q = np.minimum(np.maximum(raw[days], FLOOR), 1.0)
        share = 0.0
        if offers.any():
            g = conformal_scalar.exact_scalar(q, labels[days], offers, alpha)
            g = min(g, 1.0 / float(q[offers].max()))
            got = float((offers * g * q * values[days]).sum())
            share = got / float(values[days].sum())
        total += len(days) * share
        weight += len(days)
    return total / weight if weight else 0.0


@dataclass
class Fitted:
    """A fitted network and its record: per epoch the mean offer share over its steps and the
    validation offer, the epoch kept (0: the start), the numbers of training and validation days,
    of steps taken and of batches without a gradient."""

    net: size_net.SizeNet
    history: list[float] = field(default_factory=list)
    validation: list[float] = field(default_factory=list)
    best_epoch: int = 0
    training_days: int = 0
    validation_days: int = 0
    steps: int = 0
    batches_without_gradient: int = 0

    def record(self) -> dict:
        return {
            "training_days": self.training_days,
            "validation_days": self.validation_days,
            "epochs_run": len(self.history),
            "epoch_kept": self.best_epoch,
            "validation_offer_at_start": self.validation[0],
            "validation_offer_kept": self.validation[self.best_epoch],
            "steps": self.steps,
            "batches_without_gradient": self.batches_without_gradient,
        }


def check_rows(labels, values, n_sets: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=float)
    values = np.asarray(values, dtype=float)
    if labels.shape != (n_sets,) or values.shape != (n_sets,):
        raise ValueError("one label and one value per set is needed")
    if not (np.isfinite(labels).all() and ((labels >= 0) & (labels <= 1)).all()):
        raise ValueError("labels must be finite and in [0, 1]")
    if not (np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("values must be finite and above 0 (offering days)")
    return labels, values


def fit(x, sets: SetIndex, labels, values, units, arm: str, start, settings) -> Fitted:
    """A network trained from the start `start` (a `size_net.SizeNet` with one output, left
    unchanged) on the sets `sets` of the car inputs `x`, with one label, value and unit per set,
    under the arm `arm`, on the start's split of the sets (module text)."""
    _check_arm(arm)
    labels, values = check_rows(labels, values, sets.n_sets)
    if len(units) != sets.n_sets:
        raise ValueError("one unit per set is needed")
    units = np.asarray([str(u) for u in units])
    train, validation = size_net.split_days(sets.n_sets, settings.start)
    groups = [validation[units[validation] == u] for u in unit_order(units)]
    groups = [g for g in groups if len(g)]
    with torch_state(settings):
        net = size_net.build_net(x.shape[1], [0.0], settings.start)
        net.load_state_dict(start.state_dict())
        optimiser = torch.optim.AdamW(
            net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        order = np.random.default_rng(settings.seed)
        fitted = Fitted(net, training_days=len(train), validation_days=len(validation))
        net.train()

        def measured() -> float:
            raw = np.zeros(sets.n_sets)
            raw[validation] = _outputs(net, x, sets, validation, settings)
            return exact_offer(raw, labels, values, groups, arm, settings.alpha)

        fitted.validation.append(measured())
        kept = {k: v.detach().clone() for k, v in net.state_dict().items()}
        for epoch in range(1, settings.epochs + 1):
            shares = []
            for chosen in epoch_batches(train, units, order, settings):
                offer = batch_offer(net, x, sets, chosen, labels, values, arm, settings, order)
                if offer is None:
                    fitted.batches_without_gradient += 1
                    continue
                optimiser.zero_grad()
                (-offer).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
                optimiser.step()
                fitted.steps += 1
                shares.append(float(offer.detach()))
            fitted.history.append(float(np.mean(shares)) if shares else 0.0)
            fitted.validation.append(measured())
            if fitted.validation[-1] > fitted.validation[fitted.best_epoch]:
                fitted.best_epoch = epoch
                kept = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(kept)
        net.eval()
        return fitted


def predict(net, x: np.ndarray, sets: SetIndex, settings: CtSettings) -> np.ndarray:
    """The raw sizes of every set, one day at a time (`size_net.predict`)."""
    return size_net.predict(net, x, sets, settings.start)[:, 0]


@dataclass(frozen=True)
class FittedDays:
    """A network trained on labelled days under one arm, with the encoder of their cars and its
    fit's record (`Fitted.record`)."""

    arm: str
    encoder: size_net_inputs.Encoder
    net: size_net.SizeNet
    record: dict


def _compact(cars: pd.DataFrame, rows: pd.DataFrame) -> tuple[pd.DataFrame, SetIndex]:
    sets = size_net_inputs.fleets(cars, rows)
    day_cars = cars.iloc[sets.rows].reset_index(drop=True)
    return day_cars, SetIndex(np.arange(len(day_cars), dtype=int), sets.starts, sets.keys)


def fit_start(
    rows: pd.DataFrame, cars: pd.DataFrame, settings: CtSettings | None = None
) -> size_net.FittedDays:
    """The start of the module text on the labelled days `rows`: `size_net.fit_days` at the level
    `alpha`; training units only."""
    settings = CtSettings() if settings is None else settings
    columns = ["unit", "site", "date", LABEL]
    return size_net.fit_days(rows[columns], cars, (settings.alpha,), settings.start)


def fit_days(
    rows: pd.DataFrame,
    cars: pd.DataFrame,
    arm: str,
    settings: CtSettings | None = None,
    start: size_net.FittedDays | None = None,
) -> FittedDays:
    """A network trained on the labelled days `rows` (`unit`, `site`, `date`, `LABEL`, `VALUE`)
    with the car frame `cars` (`size_net_inputs.car_frame`) from `start` (`fit_start` on the same
    rows, fitted here when None); training units only."""
    settings = CtSettings() if settings is None else settings
    size_net.check_fit_units(rows["unit"])
    if start is None:
        start = fit_start(rows, cars, settings)
    fit_cars, compact = _compact(cars, rows)
    found = fit(
        start.encoder.transform(fit_cars),
        compact,
        rows[LABEL].to_numpy(float),
        rows[VALUE].to_numpy(float),
        rows["unit"].astype(str).to_numpy(),
        arm,
        start.net,
        settings,
    )
    return FittedDays(arm, start.encoder, found.net, found.record())


def predict_days(
    fitted: FittedDays, rows: pd.DataFrame, cars: pd.DataFrame, settings: CtSettings | None = None
) -> np.ndarray:
    """The raw sizes of a trained network at the labelled days `rows`, one per row."""
    settings = CtSettings() if settings is None else settings
    day_cars, compact = _compact(cars, rows)
    return predict(fitted.net, fitted.encoder.transform(day_cars), compact, settings)


def leave_one_unit_out(
    rows: pd.DataFrame,
    cars: pd.DataFrame,
    arms: Sequence[str] = ARMS,
    settings: CtSettings | None = None,
    records: dict | None = None,
) -> dict[str, np.ndarray]:
    """Per arm, one raw size per row of `rows` (module text). `records`, when given, gets per
    scored unit and arm the record of the fit that predicts it (`Fitted.record`)."""
    settings = CtSettings() if settings is None else settings
    out = {_check_arm(arm): np.full(len(rows), np.nan) for arm in arms}
    if rows.empty:
        return out
    size_net.check_fit_units(rows["unit"])
    rows = rows.reset_index(drop=True)
    for unit in dict.fromkeys(rows["unit"]):
        scored = (rows["unit"] == unit).to_numpy(bool)
        if scored.all():
            continue
        start = fit_start(rows[~scored], cars, settings)
        for arm in arms:
            fitted = fit_days(rows[~scored], cars, arm, settings, start)
            out[arm][scored] = predict_days(fitted, rows[scored], cars, settings)
            if records is not None:
                records.setdefault(str(unit), {})[arm] = fitted.record
    return out

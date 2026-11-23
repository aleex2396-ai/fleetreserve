"""Training of the neural hazard model (`models.hazard`; plan sections 7, 13 WP8).

Loss per step, the sum of the terms below:

- departure: minus the log-likelihood of the discrete-time hazard (`log_likelihood`): the sum of
  `log(1 - h_j)` over the bins before the car's departure bin plus `log h_b` for its bin `b`; a
  car whose departure is in the open bin (beyond the grid) contributes the survival part only;
  rows without a departure label contribute nothing. The sum is divided by the number of cars
  with a departure label in the batch;
- divergence (day factor on): `beta` times the divergence of the posterior of the set's day
  factor from its prior, summed over the sets of the batch and divided by the same number of
  cars. The likelihood is taken under a draw from the posterior, so with `beta` 1 the two terms
  are the negative evidence lower bound divided by that number. `beta` is `kl_weight` times the
  share of the warm-up done (0 at the first step, linear in the step number over
  `kl_warmup_epochs` epochs, then constant; constant `kl_weight` from the start when that is 0);
- energy (when `energy_weight` is above 0): `energy_weight` times the mean pinball loss of the
  energy head over the rows with all three energy labels and over the outputs, the quantities
  rate, room and protected energy at their levels, in the standardised units of
  `hazard_data.FitData`.

Repeatable fits: seeds fixed (`HazardSettings.seed`: the initial weights, the order of the sets
in each epoch, the draws), the thread count set to `threads`,
`torch.use_deterministic_algorithms(True)` while fitting and (in `models.hazard`) while
predicting (the previous value is restored afterwards). The thread count is not restored: it
stays at `threads` for the rest of the process, because PyTorch at several threads and LightGBM
in one process crash or hang (`hazard_net`, module text), so the default is one thread and
raising it again afterwards would bring that back. Two fits in one process on the same rows give
the same forecast table, byte for byte (a test in `tests/learn/test_hazard_table.py`). The
optimiser is AdamW with the settings' learning rate and weight decay and PyTorch's default betas
and epsilon, and the gradient norm is clipped to `grad_clip`.

Checkpoints. Nothing in the loop reads the total number of epochs: the learning rate is constant,
the divergence warm-up is counted in epochs of its own (`kl_warmup_epochs`), and the order of the
sets and the draws come from generators that are advanced epoch after epoch. So a fit of `n`
epochs passes through the state of a fit of `m < n` epochs with every other setting equal (a test
in `tests/learn/test_hazard_tune_parts.py` compares forecast tables, weights and loss histories
byte for byte). The argument `checkpoints` of `fit` is a mapping from an epoch number (1 to
`epochs`) to a function; at the end of that epoch the function is called with the network in
evaluation mode, and the network is put back in training mode afterwards. A function must not
change the network. With no checkpoints (the default) the loop is as it was.
"""

import contextlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import torch
import torch.nn.functional as functional

from fleetreserve import hazard_curves
from fleetreserve.hazard_data import N_INPUTS, UNKNOWN_SITE, FitData, batch_of
from fleetreserve.hazard_net import HazardNet, initial_bias
from fleetreserve.hazard_settings import HazardSettings


@contextlib.contextmanager
def torch_state(settings: HazardSettings):
    """Deterministic algorithms while the block runs (restored on exit) and the thread count of
    the settings, which stays set (module text)."""
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != settings.threads:
        torch.set_num_threads(settings.threads)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


def log_likelihood(logits: torch.Tensor, departure_bin: torch.Tensor) -> torch.Tensor:
    """Log-likelihood of each car's departure bin under hazard `sigmoid(logits)` (module text):
    `logits` is (..., bins with a hazard), `departure_bin` (...) holds 0 to the number of bins,
    the last value being the open bin."""
    bins = logits.shape[-1]
    log_stay = functional.logsigmoid(-logits)
    log_leave = functional.logsigmoid(logits)
    before = torch.cat([torch.zeros_like(log_stay[..., :1]), torch.cumsum(log_stay, dim=-1)], -1)
    stayed = before.gather(-1, departure_bin[..., None])[..., 0]
    own = departure_bin.clamp(max=bins - 1)
    left = log_leave.gather(-1, own[..., None])[..., 0]
    return stayed + torch.where(departure_bin < bins, left, torch.zeros_like(left))


def kl_weight_at(settings: HazardSettings, step: int, steps_per_epoch: int) -> float:
    """The divergence weight at training step `step`, counting from 0 (module text)."""
    warmup_steps = settings.kl_warmup_epochs * steps_per_epoch
    if warmup_steps == 0:
        return settings.kl_weight
    return settings.kl_weight * min(1.0, step / warmup_steps)


def hide_sites(site: torch.Tensor, probability: float, generator: torch.Generator) -> torch.Tensor:
    """The site codes with each replaced by the unknown-site code with `probability`."""
    hidden = torch.rand(len(site), generator=generator) < probability
    return torch.where(hidden.to(site.device), torch.full_like(site, UNKNOWN_SITE), site)


def pinball(prediction: torch.Tensor, target: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Pinball loss at `levels` (broadcast over the last axis), elementwise."""
    error = target - prediction
    return torch.maximum(levels * error, (levels - 1.0) * error)


@dataclass
class Fitted:
    """A fitted network and what its fit recorded: the loss of every epoch, the mean over its steps
    weighted by the number of cars with a departure label in each."""

    net: HazardNet
    history: list[float] = field(default_factory=list)


def _tensors(batch, device) -> dict:
    out = {
        "x": batch.x,
        "mask": batch.mask,
        "label": batch.label,
        "has_departure": batch.has_departure,
        "departure_bin": batch.departure_bin,
        "energy": batch.energy,
        "has_energy": batch.has_energy,
    }
    return {k: torch.as_tensor(v).to(device) for k, v in out.items()}


def build_net(
    settings: HazardSettings,
    n_sites: int,
    n_energy: int,
    bin_widths: np.ndarray,
    n_inputs: int = N_INPUTS,
) -> HazardNet:
    """A network with initial weights drawn under the settings' seed, global random state kept;
    `n_inputs` inputs per car."""
    with torch.random.fork_rng():
        torch.manual_seed(settings.seed)
        return HazardNet(
            settings,
            n_sites,
            len(bin_widths),
            n_energy,
            initial_bias(bin_widths, settings),
            n_inputs,
        )


def fit(
    data: FitData,
    settings: HazardSettings,
    n_sites: int,
    energy_levels: np.ndarray,
    energy_quantity: np.ndarray,
    checkpoints: Mapping[int, Callable[[HazardNet], None]] | None = None,
) -> Fitted:
    """Fit a network on `data`. `energy_levels` and `energy_quantity` give, for each energy
    output, its level and the column of `hazard_data.ENERGY_LABELS` it is a quantile of.
    `checkpoints` is described in the module text."""
    if checkpoints and not all(1 <= epoch <= settings.epochs for epoch in checkpoints):
        raise ValueError(f"checkpoint epochs must lie in 1 to {settings.epochs}")
    device = torch.device(settings.device)
    with torch_state(settings):
        widths = hazard_curves.bin_widths(settings)
        net = build_net(settings, n_sites, len(energy_levels), widths, data.x.shape[1]).to(device)
        optimiser = torch.optim.AdamW(
            net.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        order = np.random.default_rng(settings.seed)
        generator = torch.Generator().manual_seed(settings.seed)
        levels = torch.as_tensor(energy_levels, dtype=torch.float32).to(device)
        quantity = torch.as_tensor(energy_quantity, dtype=torch.long).to(device)
        n_sets = data.sets.n_sets
        steps_per_epoch = -(-n_sets // settings.batch_sets)
        fitted = Fitted(net)
        step = 0
        net.train()
        for epoch in range(1, settings.epochs + 1):
            total, cars = 0.0, 0
            permutation = order.permutation(n_sets)
            for first in range(0, n_sets, settings.batch_sets):
                chosen = permutation[first : first + settings.batch_sets]
                batch = batch_of(data.x, data.sets, data.site, chosen, data)
                site = torch.as_tensor(batch.site).to(device)
                if settings.site_embedding:
                    site = hide_sites(site, settings.unknown_site_probability, generator)
                t = _tensors(batch, device)
                beta = kl_weight_at(settings, step, steps_per_epoch)
                logits, energy, divergence = net.forward_fit(t, site, generator)
                weight = t["has_departure"].to(logits.dtype)
                n_cars = weight.sum().clamp(min=1.0)
                loss = -(log_likelihood(logits, t["departure_bin"]) * weight).sum() / n_cars
                if divergence is not None:
                    loss = loss + beta * divergence.sum() / n_cars
                have = t["has_energy"].to(logits.dtype)
                if settings.energy_weight > 0:
                    target = t["energy"][..., quantity]
                    loss_energy = pinball(energy, target, levels) * have[..., None]
                    loss = loss + settings.energy_weight * (
                        loss_energy.sum() / (have.sum().clamp(min=1.0) * len(energy_levels))
                    )
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
                optimiser.step()
                step += 1
                total += float(loss.detach()) * float(n_cars)
                cars += int(n_cars)
            fitted.history.append(total / max(cars, 1))
            if checkpoints and epoch in checkpoints:
                net.eval()
                checkpoints[epoch](net)
                net.train()
        net.eval()
        return fitted

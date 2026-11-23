"""The network of the neural hazard model (`models.hazard`; plan sections 7, 13 WP8).

The parts, per site-day (a set of cars, `hazard_data`):

1. Car tokens: a small network on each car's inputs (`n_inputs` of them, by default
   `hazard_data.N_INPUTS`; a model with extra input columns, `model_inputs`, passes its count).
2. Set encoder (switch `set_encoder`): `blocks` attention blocks over the cars of the set, with a
   mask that hides padding, then pooling by attention to one fleet vector (a learned query). A
   car's encoding is its token after the blocks, so it knows the rest of the fleet; the fleet
   vector summarises the day. Without the switch the encoding is the token and the fleet vector
   is the mean of the tokens of the set's cars, so a car's encoding does not depend on the
   others. There is no
   position input and the attention weights depend on the cars' values and the padding mask only,
   so each car's encoding is the same for any order of the cars (up to rounding;
   `hazard_data.make_sets` fixes the order).
3. Site vector (switch `site_embedding`): a learned vector per site seen in fitting and one for
   a site not seen (code 0); in fitting a set's site code is replaced by 0 at random
   (`hazard_fit.hide_sites`). Without the switch the hazard head gets no site input.
4. Shared day factor (switch `day_factor`): a Gaussian latent vector of `day_factor_dim`
   dimensions per set. Prior network: mean and log scale from the fleet vector, which is made
   from the inputs of the cars only (`hazard_data.INPUT_COLUMNS`) and no label. Posterior network
   (fitting only): from the fleet vector and a pooled summary of the set's cars with their
   departure labels. At prediction there is no draw: `models.hazard` averages the curves over
   Gauss-Hermite nodes of the prior (`prior_nodes`).
5. Hazard head: per car and bin, a logit from the car's encoding, the day factor and the site
   vector (`hazard_logits`); the hazard is its sigmoid. The output layer's bias starts at the
   logit of a flat guess (`initial_bias`).
6. Energy head: from the car's encoding alone, one output for each energy quantity and level the
   caller asks for (`models.hazard`: rate, room and protected energy), trained by pinball loss in
   the standardised units of `hazard_data.FitData`.

Fixed in the code, not settings: every small network (`mlp`) has one hidden layer with a ReLU;
its hidden size is `width` in the car-token, prior, label-token and posterior networks,
`2 * width` in the feed-forward layer of each attention block and of the pooling, and
`head_hidden` (a setting) in the hazard and energy heads; the log scale of the day factor, prior
and posterior, is clamped to `LOG_SCALE_RANGE` (-4 to 2); there is one pooling query, which
starts as normal draws times 0.1; the normalisation layers are PyTorch's `LayerNorm` at its
defaults.

PyTorch next to LightGBM (the pooled LightGBM models and this model run in one test process).
In the learn environment on macOS, each of two things breaks LightGBM or PyTorch (ran, short
scripts; two OpenMP runtimes is a guess, not checked):

- Import order: a process that imports PyTorch before LightGBM ends with a segmentation fault the
  first time LightGBM trains, with no PyTorch call made. Every module here that imports torch
  imports LightGBM first (`import lightgbm` with a `noqa`; the sorted imports keep it before
  torch), and so does every test file that imports torch at its top (checked as text by
  `tests/learn/test_hazard_imports.py`).
- Threads: with LightGBM imported first, a softmax on a small tensor at PyTorch's default four
  threads and then LightGBM training hangs or crashes; at one thread both work. The model sets
  the thread count itself (`hazard_fit.torch_state`); code that calls this module's classes
  directly has to set it first (`torch.set_num_threads(1)`), as `tests/learn/test_hazard_net.py`
  and `tests/learn/test_hazard_likelihood.py` do.

The network's parameters and tensors are float32 (`prior_nodes` returns float64 arrays, which the
caller converts). The module draws no random number except at construction (initial weights,
from PyTorch's global generator, which `hazard_fit.build_net` seeds) and through the generator
the caller passes to `HazardNet.forward_fit`.
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import torch
from torch import nn

from fleetreserve.hazard_data import N_INPUTS
from fleetreserve.hazard_settings import HazardSettings

LOG_SCALE_RANGE = (-4.0, 2.0)


def initial_bias(widths: np.ndarray, settings: HazardSettings) -> torch.Tensor:
    """The logit of the flat guess for each bin: the hazard bin width over `initial_hazard_scale_h`,
    at most one half."""
    start = np.minimum(widths / settings.initial_hazard_scale_h, 0.5)
    return torch.tensor(np.log(start / (1.0 - start)), dtype=torch.float32)


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs))


class Attention(nn.Module):
    """Multi-head attention of `query` over `keys`; a key with a false mask is never attended."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, m, width = query.shape[0], keys.shape[1], query.shape[2]
        depth = width // self.heads
        q = self.q(query).view(n, -1, self.heads, depth).transpose(1, 2)
        k = self.k(keys).view(n, m, self.heads, depth).transpose(1, 2)
        v = self.v(keys).view(n, m, self.heads, depth).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(depth)
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        mixed = torch.softmax(scores, dim=-1) @ v
        return self.out(mixed.transpose(1, 2).reshape(n, -1, width))


class Block(nn.Module):
    """One attention block: attention and a feed-forward layer, each with a residual and a norm."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.attention = Attention(width, heads)
        self.norm_a = nn.LayerNorm(width)
        self.forward_net = mlp(width, 2 * width, width)
        self.norm_b = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm_a(x + self.attention(x, x, mask))
        return self.norm_b(x + self.forward_net(x))


class Pooling(nn.Module):
    """Pooling by attention: one learned query attends to the cars of the set."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, width) * 0.1)
        self.attention = Attention(width, heads)
        self.norm_a = nn.LayerNorm(width)
        self.forward_net = mlp(width, 2 * width, width)
        self.norm_b = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seed = self.seed.expand(x.shape[0], 1, -1)
        pooled = self.norm_a(seed + self.attention(seed, x, mask))
        return self.norm_b(pooled + self.forward_net(pooled))[:, 0, :]


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(x.dtype)[..., None]
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)


def kl_gaussians(mean_q, log_q, mean_p, log_p) -> torch.Tensor:
    """KL(q || p) of diagonal Gaussians given means and log scales, summed over dimensions."""
    var_q, var_p = torch.exp(2 * log_q), torch.exp(2 * log_p)
    return (log_p - log_q + (var_q + (mean_q - mean_p) ** 2) / (2 * var_p) - 0.5).sum(dim=-1)


class HazardNet(nn.Module):
    def __init__(
        self,
        settings: HazardSettings,
        n_sites: int,
        n_bins: int,
        n_energy: int,
        bias: torch.Tensor,
        n_inputs: int = N_INPUTS,
    ) -> None:
        super().__init__()
        w, dim = settings.width, settings.day_factor_dim
        self.settings = settings
        self.token = mlp(n_inputs, w, w)
        if settings.set_encoder:
            self.blocks = nn.ModuleList(Block(w, settings.heads) for _ in range(settings.blocks))
            self.pool = Pooling(w, settings.heads)
        if settings.site_embedding:
            self.site = nn.Embedding(n_sites + 1, settings.site_dim)
        if settings.day_factor:
            self.prior = mlp(w, w, 2 * dim)
            self.label_token = mlp(w + 1, w, w)
            self.posterior = mlp(2 * w, w, 2 * dim)
        head_inputs = (
            w
            + (dim if settings.day_factor else 0)
            + (settings.site_dim if settings.site_embedding else 0)
        )
        self.hazard = mlp(head_inputs, settings.head_hidden, n_bins)
        with torch.no_grad():
            self.hazard[-1].bias.copy_(bias)
        self.energy = mlp(w, settings.head_hidden, n_energy)

    def encode(self, x: torch.Tensor, mask: torch.Tensor):
        """Car encodings (sets, cars, width) and fleet vectors (sets, width) (module text)."""
        tokens = self.token(x)
        if not self.settings.set_encoder:
            return tokens, masked_mean(tokens, mask)
        encoded = tokens
        for block in self.blocks:
            encoded = block(encoded, mask)
        return encoded, self.pool(encoded, mask)

    def site_vector(self, site: torch.Tensor, cars: int):
        """The site vector of every set, repeated over `cars` (None without the switch)."""
        if not self.settings.site_embedding:
            return None
        return self.site(site)[:, None, :].expand(-1, cars, -1)

    def prior_parameters(self, fleet: torch.Tensor):
        mean, log_scale = self.prior(fleet).chunk(2, dim=-1)
        return mean, log_scale.clamp(*LOG_SCALE_RANGE)

    def posterior_parameters(self, fleet, encoded, label, label_mask):
        """Mean and log scale of the posterior of the day factor given the departure labels."""
        tokens = self.label_token(torch.cat([encoded, label[..., None]], dim=-1))
        summary = masked_mean(tokens, label_mask)
        mean, log_scale = self.posterior(torch.cat([fleet, summary], dim=-1)).chunk(2, dim=-1)
        return mean, log_scale.clamp(*LOG_SCALE_RANGE)

    def hazard_logits(self, encoded, factor, site_vector) -> torch.Tensor:
        """Hazard logits (sets, cars, bins); `factor` is (sets, dim) or None."""
        parts = [encoded]
        if factor is not None:
            parts.append(factor[:, None, :].expand(-1, encoded.shape[1], -1))
        if site_vector is not None:
            parts.append(site_vector)
        return self.hazard(torch.cat(parts, dim=-1))

    def energy_outputs(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.energy(encoded)

    def forward_fit(self, batch, site, generator: torch.Generator):
        """One training pass: (hazard logits, energy outputs, divergence per set or None).
        `batch` holds tensors `x`, `mask`, `label`, `has_departure`."""
        encoded, fleet = self.encode(batch["x"], batch["mask"])
        factor, divergence = None, None
        if self.settings.day_factor:
            mean_p, log_p = self.prior_parameters(fleet)
            mean_q, log_q = self.posterior_parameters(
                fleet, encoded, batch["label"], batch["has_departure"]
            )
            noise = torch.randn(mean_q.shape, generator=generator).to(mean_q.device)
            factor = mean_q + torch.exp(log_q) * noise
            divergence = kl_gaussians(mean_q, log_q, mean_p, log_p)
        logits = self.hazard_logits(encoded, factor, self.site_vector(site, encoded.shape[1]))
        return logits, self.energy_outputs(encoded), divergence


def prior_nodes(settings: HazardSettings) -> tuple[np.ndarray, np.ndarray]:
    """Standard-normal quadrature: nodes (points, dim) and weights (points,) summing to 1, a
    product grid over the dimensions of the day factor (Gauss-Hermite for the normal weight)."""
    points, weights = np.polynomial.hermite_e.hermegauss(settings.gauss_hermite_nodes)
    weights = weights / weights.sum()
    dim = settings.day_factor_dim
    grid = np.stack(np.meshgrid(*([points] * dim), indexing="ij"), axis=-1).reshape(-1, dim)
    weight = np.stack(np.meshgrid(*([weights] * dim), indexing="ij"), axis=-1).reshape(-1, dim)
    return grid, weight.prod(axis=1)

"""The settings of the neural hazard model (`models.hazard`; plan sections 7, 13 WP8).

The sizes of the network, the time grid and the training values listed below are fields of
`HazardSettings`, so a test or an ablation changes them in one place. Not fields, fixed in the
code and listed in the module text of the file that holds them: in `hazard_net` the clamp on the
log scale of the day factor, the hidden sizes of the networks other than the hazard and energy
heads (the multiples of `width` named there) and the start of the pooling query; in `hazard_fit`
the optimiser and the form of the loss; in `hazard_data` the periods of the cyclic columns, the
count columns and the floor under a standard deviation; in `hazard_curves` the shift
`PRESENCE_SHIFT`. How the defaults were chosen is the author's statement, not checkable from the
code: from the plan's wording and from an assumed size of the problem (about 100,000 training
cars in about 15,000 site-days, an estimate that this code does not measure), with no scored
result used.

Switches (three of the plan's four ablations of the hazard model, section 7; the fourth,
calendar-only inputs, is not a switch here), each on by default:

- `set_encoder`: attention over the cars of a site-day; off, a car's encoding depends on its own
  inputs only and the fleet is summarised by the mean of the car encodings;
- `site_embedding`: one learned vector per site seen in fitting, and one vector for a site not
  seen; off, the hazard head is given no site input;
- `day_factor`: the shared day factor; off, there is no latent variable and no divergence term.

Time grid (hours from the decision time τ): steps of `fine_step_h` up to `fine_until_h`, steps of
`coarse_step_h` up to `grid_end_h`, then one last open bin that has no hazard of its own (a car
that is still plugged in at `grid_end_h` leaves in it). The default is quarter-hour bins to 12 h,
hourly bins to 36 h, so 72 bins with a hazard and one open bin. `check_grid` refuses a grid
on which any service hour t, from 1 to the number of `survival_h<t>` columns (8), is not a grid
point: t must not exceed `grid_end_h`, and must be at most `fine_until_h` (every whole hour up
to `fine_until_h` is a point, because the fine step divides an hour) or a whole number of coarse
steps after it. So on every accepted grid "present at the start of service hour t" is read off
the curve at a point of the grid (`hazard_curves`) for each service hour t. Hours after the last
service hour need not be grid points. Tests pin the refusal of grids that fail the rule and the
acceptance of grids that meet it (`test_hazard_likelihood`).

Defaults and the reason for each:

- `width` 32, `heads` 2, `blocks` 2, `head_hidden` 64: a small network, chosen on the assumption
  that the problem's size does not support a large one and that the fit should run in minutes on
  a laptop CPU;
- `site_dim` 8: plan section 3 lists 8 training sites, so a short vector;
- `day_factor_dim` 1, `gauss_hermite_nodes` 5: one shared factor, meant for an "early or late
  day" (the author's reading of the plan's amortised day factor); five nodes are the Gauss-Hermite
  quadrature of a one-dimensional standard normal (`hazard_net.prior_nodes`). With a larger
  dimension the nodes form a product grid (at most 3 dimensions);
- `epochs` 30, `batch_sets` 64, `learning_rate` 0.003, `weight_decay` 0.0001, `grad_clip` 5:
  training values chosen as ordinary for a small network (the author's judgement). The
  learning rate and the weight decay are the AdamW optimiser's; `epochs` is the number of passes
  over the sets and `batch_sets` the most sets in one batch; `grad_clip` is the clip on the
  gradient norm, meant to guard the first steps against a large gradient;
- `kl_weight` 1 (the evidence lower bound as written), `kl_warmup_epochs` 5: the divergence
  weight rises linearly from 0 over the first `kl_warmup_epochs` epochs (constant `kl_weight`
  when that is 0), meant so that the day factor is used before it is pulled to the prior;
- `energy_weight` 1: the pinball loss of the energy head counts as much as the departure
  likelihood; the departure term is the mean over the cars with a departure label and the energy
  term the mean over the cars with energy labels and over the outputs;
- `unknown_site_probability` 0.5 (`UNKNOWN_SITE_PROBABILITY`): in training the site code of each
  site-day is replaced by the unknown-site code with this probability, so that vector is trained.
  It is the vector a site not seen in fitting gets (`hazard_data.site_codes`);
- `initial_hazard_scale_h` 8: the output bias of the hazard head starts at the logit of the
  bin's width over 8 hours (capped at one half), a flat hazard of an eight-hour mean stay
  (a judgement), meant so that training starts from a plausible curve. The other weights are
  random, so the logit of a starting hazard is that value plus the part of the head's output
  that comes from them, and the hazard is the sigmoid of that logit
  (`hazard_net.initial_bias`, `hazard_net.HazardNet.hazard_logits`; a test checks the bias and
  that the starting hazards lie in (0, 1));
- `input_clip` 10: a standardised input is clipped to plus or minus 10, meant so that one
  extreme history value cannot dominate a site-day;
- `min_presence_support` 100, `min_presence_probability` 0.001: a bound given presence at the
  start of a service hour (`survival_h<t>` of the forecast table) is given only where the fit
  rows of the set's own site type and the car itself support it (`models.hazard`, module text);
  a round number of cars and a round small probability, chosen without a scored result (the
  author's statement of how they were chosen);
- `seed` 0, `threads` 1, `device` "cpu": see Device below;
- `inputs` None: the network is given every input column. A non-empty tuple of names from
  `hazard_data.INPUT_COLUMNS` keeps only those columns (with their missing-value flags, or both
  phases of a cyclic column) and sets every other input to 0 after the encoding
  (`hazard_data.input_mask`, `Scaler`); the tuning and `models.hazard_tuned_calendar` use it for
  the calendar-only part (`hazard_parts.CALENDAR_INPUTS`, also importable from `hazard_tune`). With
  None the inputs are as they were before the field existed.

Device: the plan names MPS. The default is the CPU because the forecasts must be repeatable (two
fits on the same rows give the same table), the data are assumed small, and the CPU path with one
thread and deterministic algorithms was found repeatable by a test (two fits in one process give
byte-identical tables). "mps" is selectable and untested here.
"""

from dataclasses import dataclass

from fleetreserve.forecast_model import SURVIVAL_COLUMNS

UNKNOWN_SITE_PROBABILITY = 0.5
MAX_DAY_FACTOR_DIM = 3
DEVICES = ("cpu", "mps")


@dataclass(frozen=True)
class HazardSettings:
    set_encoder: bool = True
    site_embedding: bool = True
    day_factor: bool = True
    width: int = 32
    heads: int = 2
    blocks: int = 2
    head_hidden: int = 64
    site_dim: int = 8
    day_factor_dim: int = 1
    gauss_hermite_nodes: int = 5
    fine_step_h: float = 0.25
    fine_until_h: float = 12.0
    coarse_step_h: float = 1.0
    grid_end_h: float = 36.0
    epochs: int = 30
    batch_sets: int = 64
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    grad_clip: float = 5.0
    kl_weight: float = 1.0
    kl_warmup_epochs: int = 5
    energy_weight: float = 1.0
    unknown_site_probability: float = UNKNOWN_SITE_PROBABILITY
    initial_hazard_scale_h: float = 8.0
    input_clip: float = 10.0
    seed: int = 0
    threads: int = 1
    device: str = "cpu"
    min_presence_support: int = 100
    min_presence_probability: float = 1e-3
    inputs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("width", "heads", "blocks", "head_hidden", "site_dim", "epochs", "batch_sets"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.threads < 1 or self.gauss_hermite_nodes < 1 or self.kl_warmup_epochs < 0:
            raise ValueError("threads and gauss_hermite_nodes are at least 1, warm-up at least 0")
        if self.width % self.heads:
            raise ValueError(f"width {self.width} must be a multiple of heads {self.heads}")
        if not 1 <= self.day_factor_dim <= MAX_DAY_FACTOR_DIM:
            raise ValueError(f"day_factor_dim must be 1 to {MAX_DAY_FACTOR_DIM}")
        if not 0.0 <= self.unknown_site_probability <= 1.0:
            raise ValueError("unknown_site_probability must be in [0, 1]")
        positive = ("learning_rate", "grad_clip", "initial_hazard_scale_h", "input_clip")
        if any(getattr(self, name) <= 0 for name in positive):
            raise ValueError(f"{positive} must be positive")
        if self.kl_weight < 0 or self.energy_weight < 0 or self.weight_decay < 0:
            raise ValueError("kl_weight, energy_weight and weight_decay must not be negative")
        if self.min_presence_support < 0 or not 0.0 <= self.min_presence_probability < 1.0:
            raise ValueError(
                "min_presence_support is at least 0, min_presence_probability in [0, 1)"
            )
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.inputs is not None and (
            not isinstance(self.inputs, tuple)
            or not self.inputs
            or not all(isinstance(name, str) for name in self.inputs)
        ):
            raise ValueError("inputs is None or a non-empty tuple of input column names")
        self.check_grid()

    def check_grid(self) -> None:
        """The steps fit their ranges, and each service hour is a grid point (module text): the
        hours 1 to the number of `survival_h<t>` columns, each within `grid_end_h` and either in
        the fine range or a whole number of coarse steps after `fine_until_h`."""
        fine, coarse = self.fine_step_h, self.coarse_step_h
        if not (0 < fine <= 1 and coarse > 0 and 0 < self.fine_until_h <= self.grid_end_h):
            raise ValueError("the time grid needs 0 < fine step <= 1, 0 < coarse step, 0 < end")
        if not _whole(1.0 / fine) or not _whole(self.fine_until_h):
            raise ValueError("the fine step must divide an hour and the fine range be whole hours")
        if (
            not _whole(coarse)
            or not _whole(self.grid_end_h - self.fine_until_h)
            or not _whole((self.grid_end_h - self.fine_until_h) / coarse)
        ):
            raise ValueError("the coarse step must be whole hours and fit the coarse range")
        for hour in range(1, len(SURVIVAL_COLUMNS) + 1):
            in_range = hour <= self.grid_end_h
            on_fine_grid = hour <= self.fine_until_h
            on_coarse_grid = _whole((hour - self.fine_until_h) / coarse)
            if not (in_range and (on_fine_grid or on_coarse_grid)):
                raise ValueError(f"service hour {hour} must be a point of the time grid")


def _whole(value: float) -> bool:
    return abs(value - round(value)) < 1e-12

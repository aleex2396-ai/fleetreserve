"""The forecast of the neural hazard family averaged over the networks of several fit seeds, for
the what-if `dial_choice --what-if-fit-seeds` only (WHAT-IF, NOT A MODEL OF THE PLAN).

`average_of(members)` takes two or more models of the neural hazard family (models whose
`settings` are `HazardSettings`), each built and given its seed exactly as `dial_choice
--what-if-fit-seed N` builds it (`dial_choice.with_fit_seed`), with distinct seeds. It returns
one model that `dial_choice` fits and reads like any other:

- fit: every member is fitted, one after another, on the same cars tables, as its own run with
  `--what-if-fit-seed N` fits it (`HazardModel.fit`; two fits in one process on the same rows give
  the same table, `hazard_fit`); no member is left out and no network is fitted twice;
- forecast: the rule of the seed ensemble `models.hazard_ens`, whose code is used here unchanged
  (reading A1 below): per car the equal-weight mean of the members' survival curves over the time
  grid, `d_hat`, `d_median` and `survival_h1` to `survival_h8` read from that mean curve by the
  single model's code with its presence rules, and `r_hat`, `e_cap`, `e_need` the plain mean of
  the members' values.

The members are put in increasing order of seed, so the order in which the seeds are listed
changes no number. The name, fit units and extra input columns are the first member's (every
member is the same model at another seed), so `learn_run.check_model`, `model_inputs.extra_of`
and `kappa_choice`'s agreement of model names read it as that model; the seeds are in `seeds`
and each member's own settings.

Reading A1 (what is averaged). The survival curve: the mean of the members' curves is the
mixture, with equal weights, of the members' departure distributions, so it is a proper
distribution (it starts at 1, does not rise, stays in [0, 1]) and its value at an hour is the
mean of the members' probabilities that the car is still present then. The bounds the gate
reads (`d_hat` and the `survival_h<t>` bounds) and the presence-probability floor are then read
from one distribution with the single model's rules. [Alternatives: (a) the mean of the predicted
hazard per hour, which also gives a proper curve but not the mixture: the product of mean stay
probabilities is not the mean of the members' products, so the result is the distribution of no
average of the members; (b) the mean of the members' departure quantiles (`d_hat`, the
`survival_h<t>` bounds), which keeps each column ordered in the level but gives no curve, so the
presence-probability floor would have to be read from another average, and a NaN cell of one
member would make the mean NaN.] The energy head gives quantiles at fixed levels only, not a
distribution, so its columns are averaged level by level (the mean of quantile functions, still
a quantile function in the level).
"""

from collections.abc import Sequence

from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.models.hazard_ens import HazardEnsembleModel

# The attributes `learn_run` and `model_inputs` read from a model, taken from the first member.
DELEGATED = (
    "name",
    "fit_units",
    "extra_inputs",
    "context_sources",
    "context_files",
    "look_ahead",
    "reads_other_day_outcomes",
    "levels",
)


class SeedAverage(HazardEnsembleModel):
    """Two or more members of the neural hazard family at distinct seeds, averaged (module text).
    Fit and forecast are `HazardEnsembleModel.fit` and `.predict` on these members."""

    def __init__(self, members: Sequence) -> None:
        members = sorted(members, key=lambda m: m.settings.seed)
        self.members = list(members)
        self.seeds = tuple(int(m.settings.seed) for m in self.members)
        self.rho = self.members[0].rho
        self.fitted = False
        for attribute in DELEGATED:
            if hasattr(self.members[0], attribute):
                setattr(self, attribute, getattr(self.members[0], attribute))


def average_of(members: Sequence) -> SeedAverage:
    """The averaged model of `members` (module text); `ValueError` for fewer than two members, a
    member outside the neural hazard family, members of different models, two members at one
    seed or one member given twice."""
    if len(members) < 2:
        raise ValueError(f"an average over seeds needs two or more members, got {len(members)}")
    if len({id(m) for m in members}) != len(members):
        raise ValueError("an average over seeds was given one member twice")
    for member in members:
        if not isinstance(getattr(member, "settings", None), HazardSettings):
            raise ValueError("an average over seeds takes models of the neural hazard family only")
    if len({type(m) for m in members}) != 1:
        raise ValueError("the members of an average over seeds are one model")
    seeds = [m.settings.seed for m in members]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"the members of an average over seeds have distinct seeds, got {seeds}")
    return SeedAverage(members)

"""The pieces of the neural hazard model that compute: likelihood, divergence, loss rules, time
grid, survival curves and quantiles (`hazard_fit`, `hazard_net`, `hazard_curves`,
`hazard_settings`), and the fit against a known hazard.

1. The log-likelihood of a departure bin equals a hand computation on a two-car, three-bin
   example (two such examples), and a departure in the open bin contributes survival only.
2. On made-up cars drawn from a known mass function (`hazard_fake`), the fitted model's departure
   bounds at the plan's four levels are within `BOUND_TOLERANCE` hours of the true quantiles of
   each group (mean over the group's cars), and the energy head's quantiles cover their levels
   within `COVERAGE_TOLERANCE`.
3. The vectorised quantile equals `forecast_model.survival_quantile` curve by curve, with and
   without `after`.
4. The divergence is the closed-form divergence of diagonal Gaussians; the pinball loss, the
   divergence warm-up, the random hiding of sites and the quadrature nodes are as written.
5. The time grid: 72 bins with a hazard and one open bin, every whole hour a grid point; a
   departure goes in the bin that holds it; bad grids are refused, among them every grid on which
   a service hour from 1 to 8 (one per `survival_h<t>` column) is not a grid point.
6. The order of the cars within a set, which the session ids give, on the made-up cars of
   `fitted_known` at the levels `KAPPAS`, in the columns `FORECAST_NUMBERS`: new session ids that
   keep the order give the same numbers and NaNs, and new ids that reverse the order of the cars
   of every set change no number by 1e-5 or more and leave every NaN where it was. The 1e-5 and
   the NaNs in place are observed on these cars, not guaranteed by the code: a bound is the first
   grid point at which a comparison of the curve with the level holds, and a `survival_h<t>` cell
   is also NaN where the car's probability of being present is below a floor, so a rounding change
   of a curve that close to the level or the floor can move the bound to another grid point or
   make the cell NaN or a number (`models.hazard`).
"""

import math

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pytest
import torch
from hazard_fake import KAPPAS, fitted_known, true_bound

from fleetreserve import hazard_curves, hazard_fit
from fleetreserve.forecast_model import SURVIVAL_COLUMNS, survival_quantile
from fleetreserve.hazard_net import kl_gaussians, prior_nodes
from fleetreserve.hazard_settings import HazardSettings

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net
BOUND_TOLERANCE = 0.5
COVERAGE_TOLERANCE = 0.07


def _logit(p):
    p = np.asarray(p, dtype=float)
    return torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)


# --- 1. the likelihood ---------------------------------------------------------------------------


def test_the_likelihood_of_a_two_car_three_bin_example_is_the_hand_computed_one():
    hazards = [[0.2, 0.5, 0.4], [0.1, 0.3, 0.6]]
    # car 1 stays through bin 0 and leaves in bin 1; car 2 leaves beyond the grid (the open bin,
    # number 3) and so stays through all three bins
    found = hazard_fit.log_likelihood(_logit(hazards), torch.tensor([1, 3]))
    by_hand = [math.log(0.8) + math.log(0.5), math.log(0.9) + math.log(0.7) + math.log(0.4)]
    assert found.tolist() == pytest.approx(by_hand, abs=1e-6)


def test_the_likelihood_of_a_departure_in_the_first_and_in_the_last_bin_with_a_hazard():
    hazards = [[0.2, 0.5, 0.4], [0.1, 0.3, 0.6]]
    found = hazard_fit.log_likelihood(_logit(hazards), torch.tensor([0, 2]))
    by_hand = [math.log(0.2), math.log(0.9) + math.log(0.7) + math.log(0.6)]
    assert found.tolist() == pytest.approx(by_hand, abs=1e-6)


def test_the_probabilities_of_all_departure_bins_add_to_one():
    logits = _logit([[0.2, 0.5, 0.4]])
    total = sum(
        math.exp(hazard_fit.log_likelihood(logits, torch.tensor([b])).item()) for b in range(4)
    )
    assert total == pytest.approx(1.0, abs=1e-6)


# --- 2. the fit against a known hazard -----------------------------------------------------------


def test_the_fitted_bounds_are_the_true_quantiles_of_the_known_distribution():
    _, fresh, _, table = fitted_known()
    group = np.where(fresh["arrival_hour"].to_numpy() == 7, "A", "B")
    for kappa in KAPPAS:
        part = table[table["kappa"] == kappa]
        for name in "AB":
            found = part["d_hat"].to_numpy()[group == name]
            assert abs(found.mean() - true_bound(name, kappa)) <= BOUND_TOLERANCE, (kappa, name)
            assert (found >= 0).all()
    median = table[table["kappa"] == 0.3]["d_median"].to_numpy()
    for name in "AB":
        assert abs(median[group == name].mean() - true_bound(name, 0.5)) <= BOUND_TOLERANCE


def test_the_energy_head_quantiles_cover_their_levels():
    _, _, later, table = fitted_known()
    fleet = later["label_room_kwh"].notna().to_numpy()
    assert fleet.sum() > 100
    for kappa in KAPPAS:
        part = table[table["kappa"] == kappa].reset_index(drop=True)
        below = {
            "r_hat": later["label_rate_kw"].to_numpy() < part["r_hat"].to_numpy(),
            "e_cap": later["label_room_kwh"].to_numpy() < part["e_cap"].to_numpy(),
            "e_need": later["label_need_kwh"].to_numpy() > part["e_need"].to_numpy(),
        }
        for column, miss in below.items():
            assert abs(miss[fleet].mean() - kappa) <= COVERAGE_TOLERANCE, (kappa, column)


# --- 3. the vectorised quantile ------------------------------------------------------------------


def _random_curves(rng, rows: int, bins: int):
    times = np.cumsum(rng.random(bins) + 0.1)
    steps = rng.random((rows, bins)) * (rng.random((rows, bins)) < 0.6)  # some plateaus
    steps /= np.maximum(steps.sum(axis=1, keepdims=True), 1e-9) * rng.uniform(1.0, 1.4)
    return times, np.clip(1.0 - np.cumsum(steps, axis=1), 0.0, 1.0)


@pytest.mark.parametrize("after", [None, -1.0, 0.0, 1.3, 2.0, 5.0, 100.0])
def test_the_vectorised_quantile_equals_survival_quantile_curve_by_curve(after):
    rng = np.random.default_rng(3)
    times, survival = _random_curves(rng, rows=40, bins=12)
    for level in (0.05, 0.1, 0.2, 0.3, 0.5):
        found = hazard_curves.lower_quantiles(times, survival, level, after)
        wanted = [survival_quantile(times, s, level, after) for s in survival]
        assert np.array_equal(found, wanted, equal_nan=True), (level, after)


def test_the_vectorised_quantile_equals_survival_quantile_on_the_grid_at_every_hour():
    rng = np.random.default_rng(4)
    settings = HazardSettings()
    times = hazard_curves.bin_starts(settings)
    hazards = rng.random((30, len(times) - 1)) * 0.2
    survival = hazard_curves.survival_curves(hazards)
    for hour in range(0, 9):
        after = hour - hazard_curves.PRESENCE_SHIFT
        for level in (0.05, 0.3):
            found = hazard_curves.lower_quantiles(times, survival, level, after)
            wanted = [survival_quantile(times, s, level, after) for s in survival]
            assert np.array_equal(found, wanted, equal_nan=True)


def test_a_curve_made_from_a_sample_gives_that_samples_quantile_despite_rounding():
    times = np.arange(1.0, 11.0)  # ten equally likely departures, at 1 to 10 hours
    survival = (1.0 - np.arange(1, 11) / 10)[None]
    for level, expected in ((0.1, 2.0), (0.2, 3.0), (0.3, 4.0), (0.5, 6.0)):
        assert survival_quantile(times, survival[0], level) == expected
        assert hazard_curves.lower_quantiles(times, survival, level)[0] == expected


def test_survival_curves_multiply_the_stay_probabilities_and_end_at_zero():
    curves = hazard_curves.survival_curves(np.array([[0.5, 0.5, 0.5]]))
    assert curves.tolist() == [[0.5, 0.25, 0.125, 0.0]]


def test_presence_at_an_hour_keeps_the_bin_that_starts_at_that_hour():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    survival = np.array(
        [[0.9, 0.6, 0.3, 0.0]]
    )  # share left after each bin; mass 0.1, 0.3, 0.3, 0.3
    assert survival_quantile(times, survival[0], 0.3) == 1.0
    shifted = 1.0 - hazard_curves.PRESENCE_SHIFT
    # present at hour 1: 0.9 are left and 0.3 of them (a third) leave in the bin at 1
    assert hazard_curves.lower_quantiles(times, survival, 0.3, shifted)[0] == 1.0
    assert hazard_curves.lower_quantiles(times, survival, 0.34, shifted)[0] == 2.0
    # `after` = 1 itself would count only the cars still there after the bin at 1
    assert hazard_curves.lower_quantiles(times, survival, 0.3, 1.0)[0] == 2.0


def test_presence_probability_at_a_grid_hour_is_the_share_left_after_the_bin_before_it():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    survival = np.array([[0.9, 0.6, 0.3, 0.0], [1.0, 0.5, 0.25, 0.0]])
    # the share still plugged in at the start of hour h is S at the last time before h
    assert hazard_curves.presence_probability(times, survival, 1).tolist() == [0.9, 1.0]
    assert hazard_curves.presence_probability(times, survival, 2).tolist() == [0.6, 0.5]
    assert hazard_curves.presence_probability(times, survival, 3).tolist() == [0.3, 0.25]
    assert hazard_curves.presence_probability(times, survival, 0).tolist() == [1.0, 1.0]
    # on the default grid, at hour 8: S at the bin that starts at 7.75
    default = hazard_curves.bin_starts(HazardSettings())
    curve = np.linspace(1.0, 0.0, len(default))[None]
    place = int(np.flatnonzero(default == 7.75)[0])
    assert hazard_curves.presence_probability(default, curve, 8)[0] == curve[0, place]


# --- 4. the loss rules ---------------------------------------------------------------------------


def test_the_divergence_is_the_closed_form_of_diagonal_gaussians():
    zero = torch.zeros(1, 1)
    assert kl_gaussians(zero, zero, zero, zero).item() == pytest.approx(0.0, abs=1e-7)
    assert kl_gaussians(zero + 1, zero, zero, zero).item() == pytest.approx(0.5, abs=1e-6)
    log_q = zero + math.log(2.0)  # scale 2 against scale 1
    assert kl_gaussians(zero, log_q, zero, zero).item() == pytest.approx(
        -math.log(2.0) + 2.0 - 0.5, abs=1e-6
    )
    two = torch.zeros(1, 2)
    assert kl_gaussians(two + 1, two, two, two).item() == pytest.approx(1.0, abs=1e-6)


def test_the_pinball_loss_is_asymmetric_by_the_level():
    levels = torch.tensor([0.1, 0.9])
    loss = hazard_fit.pinball(torch.tensor([[1.0, 1.0]]), torch.tensor([[3.0, 3.0]]), levels)
    assert loss.tolist()[0] == pytest.approx([0.2, 1.8])  # target above: 0.1*2, 0.9*2
    loss = hazard_fit.pinball(torch.tensor([[3.0, 3.0]]), torch.tensor([[1.0, 1.0]]), levels)
    assert loss.tolist()[0] == pytest.approx([1.8, 0.2])


def test_the_divergence_weight_rises_linearly_from_zero_over_the_warm_up():
    settings = HazardSettings(kl_weight=0.8, kl_warmup_epochs=4)
    weights = [hazard_fit.kl_weight_at(settings, step, 5) for step in range(0, 30, 5)]
    assert weights[0] == 0.0
    assert weights[2] == pytest.approx(0.4)
    assert weights[4:] == [0.8, 0.8]
    assert hazard_fit.kl_weight_at(HazardSettings(kl_warmup_epochs=0), 0, 5) == 1.0


def test_sites_are_hidden_with_the_given_probability():
    site = torch.arange(1, 2001)
    generator = torch.Generator().manual_seed(0)
    assert hazard_fit.hide_sites(site, 0.0, generator).tolist() == site.tolist()
    assert (hazard_fit.hide_sites(site, 1.0, generator) == 0).all()
    share = (hazard_fit.hide_sites(site, 0.5, generator) == 0).float().mean().item()
    assert abs(share - 0.5) < 0.05
    assert HazardSettings().unknown_site_probability == 0.5


def test_the_quadrature_nodes_integrate_a_standard_normal():
    for dim in (1, 2):
        nodes, weights = prior_nodes(HazardSettings(day_factor_dim=dim))
        assert nodes.shape == (5**dim, dim)
        assert weights.sum() == pytest.approx(1.0)
        assert (weights @ nodes) == pytest.approx(np.zeros(dim), abs=1e-12)
        assert (weights @ nodes**2) == pytest.approx(np.ones(dim))
        assert (weights @ nodes**4) == pytest.approx(3 * np.ones(dim))


# --- 5. the time grid ----------------------------------------------------------------------------


def test_the_default_grid_has_72_bins_with_a_hazard_and_an_open_bin_and_whole_hours_are_points():
    settings = HazardSettings()
    starts = hazard_curves.bin_starts(settings)
    assert len(starts) == 73 and hazard_curves.n_hazard_bins(settings) == 72
    assert starts[0] == 0.0 and starts[-1] == 36.0
    assert set(np.diff(starts[:49])) == {0.25} and set(np.diff(starts[48:])) == {1.0}
    assert set(range(37)) <= set(starts.tolist())


def test_a_departure_is_in_the_bin_that_holds_it():
    departures = np.array([-1.0, 0.0, 0.2499, 0.25, 11.99, 12.0, 35.99, 36.0, 99.0])
    found = hazard_curves.bin_of(departures, HazardSettings())
    assert found.tolist() == [0, 0, 0, 1, 47, 48, 71, 72, 72]


@pytest.mark.parametrize(
    "changes",
    [
        {"fine_step_h": 0.3},
        {"coarse_step_h": 0.5},
        {"grid_end_h": 12.5},
        {"width": 15},
        {"day_factor_dim": 4},
        {"device": "cuda"},
        {"unknown_site_probability": 1.5},
    ],
)
def test_settings_that_cannot_work_are_refused(changes):
    with pytest.raises(ValueError):
        HazardSettings(**changes)


SERVICE_HOURS = range(1, 9)


def _points(**grid):
    """The points of a grid as `hazard_curves.bin_starts` builds them, for settings that the
    constructor may refuse (so they are made without it). Only the fields given are set on the
    object; a field not given is read from the class, where it is the field's default
    (`HazardSettings` is a dataclass without slots)."""
    settings = object.__new__(HazardSettings)
    for name, value in grid.items():
        object.__setattr__(settings, name, value)
    return set(hazard_curves.bin_starts(settings).tolist())


def test_the_number_of_service_hours_the_grid_must_hold_is_the_number_of_survival_columns():
    assert [f"survival_h{t}" for t in SERVICE_HOURS] == list(SURVIVAL_COLUMNS)


@pytest.mark.parametrize(
    "grid",
    [
        {"fine_until_h": 4.0, "coarse_step_h": 2.0},  # hours 5 and 7 fall inside a coarse step
        {"fine_until_h": 5.0, "coarse_step_h": 2.0, "grid_end_h": 37.0},  # 6, 8 inside a step
        {"fine_until_h": 4.0, "coarse_step_h": 3.0, "grid_end_h": 37.0},  # hours 5, 6, 8
        {"fine_until_h": 4.0, "grid_end_h": 6.0},  # the grid ends before hours 7 and 8
        {"fine_until_h": 4.0, "grid_end_h": 7.0},  # the grid ends before hour 8
        {"fine_until_h": 1.0, "grid_end_h": 1.0},  # the fine range is the whole grid
    ],
)
def test_a_grid_that_leaves_a_service_hour_off_the_grid_is_refused(grid):
    assert not all(float(t) in _points(**grid) for t in SERVICE_HOURS)  # the premise
    with pytest.raises(ValueError, match="service hour"):
        HazardSettings(**grid)


@pytest.mark.parametrize(
    "grid",
    [
        {},
        {"fine_until_h": 4.0, "coarse_step_h": 1.0},
        {"fine_until_h": 4.0, "coarse_step_h": 1.0, "grid_end_h": 8.0},  # ends at hour 8
        {"fine_until_h": 8.0, "coarse_step_h": 2.0, "grid_end_h": 20.0},  # coarse after hour 8
        {"fine_step_h": 1.0, "fine_until_h": 2.0, "coarse_step_h": 1.0, "grid_end_h": 10.0},
        {"fine_step_h": 0.5, "fine_until_h": 3.0, "coarse_step_h": 1.0, "grid_end_h": 9.0},
    ],
)
def test_a_grid_with_every_service_hour_as_a_point_is_accepted(grid):
    HazardSettings(**grid)
    points = _points(**grid)
    assert [t for t in SERVICE_HOURS if float(t) not in points] == []


def test_the_constructor_accepts_a_grid_exactly_when_every_service_hour_is_a_point():
    """Over grids on which the other rules of the grid hold (whole steps that fit the ranges)."""
    counts = {True: 0, False: 0}
    for fine_step in (0.25, 0.5, 1.0):
        for fine_until in (1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 12.0):
            for coarse in (1.0, 2.0, 3.0):
                for n_coarse in range(6):
                    grid = {
                        "fine_step_h": fine_step,
                        "fine_until_h": fine_until,
                        "coarse_step_h": coarse,
                        "grid_end_h": fine_until + n_coarse * coarse,
                    }
                    expected = all(float(t) in _points(**grid) for t in SERVICE_HOURS)
                    try:
                        HazardSettings(**grid)
                        accepted = True
                    except ValueError:
                        accepted = False
                    assert accepted == expected, grid
                    counts[accepted] += 1
    assert counts[True] >= 20 and counts[False] >= 20, counts


# --- 6. the order of the cars within a set -------------------------------------------------------

FORECAST_NUMBERS = ["d_hat", "r_hat", "e_cap", "e_need", "d_median", *SURVIVAL_COLUMNS]


def _forecast_after_renaming(model, fresh, table, reverse):
    """`table` against the forecast of the same cars under new session ids that keep, or with
    `reverse` reverse, the order of the cars within every site-day (ordered by id as text)."""
    old = fresh["session_id"].astype(str).to_numpy()
    key = fresh[["unit", "site", "date"]].astype(str).agg("|".join, axis=1).to_numpy()
    new = np.empty(len(old), dtype=object)
    for site_day in dict.fromkeys(key):
        places = np.flatnonzero(key == site_day)
        for rank, place in enumerate(places[np.argsort(old[places], kind="stable")]):
            new[place] = f"x{len(places) - 1 - rank if reverse else rank:05d}"
    after = model.predict(fresh.assign(session_id=new), KAPPAS)
    before = table.assign(
        session_id=table["session_id"].astype(str).map(dict(zip(old, new, strict=True)))
    )
    keys = ["site", "date", "session_id", "kappa"]
    before = before.sort_values(keys).reset_index(drop=True)
    after = after.sort_values(keys).reset_index(drop=True)
    assert (before[keys].astype(str).to_numpy() == after[keys].astype(str).to_numpy()).all()
    return before[FORECAST_NUMBERS].to_numpy(float), after[FORECAST_NUMBERS].to_numpy(float)


def test_on_these_cars_keeping_the_order_changes_no_number_and_reversing_it_none_by_1e_5():
    model, fresh, _, table = fitted_known()
    same, same_after = _forecast_after_renaming(model, fresh, table, reverse=False)
    assert np.array_equal(same, same_after, equal_nan=True)
    before, after = _forecast_after_renaming(model, fresh, table, reverse=True)
    assert (np.isnan(before) == np.isnan(after)).all()
    assert np.nanmax(np.abs(before - after)) < 1e-5

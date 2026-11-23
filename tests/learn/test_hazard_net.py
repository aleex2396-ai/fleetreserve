"""The network of the neural hazard model (`hazard_net`), untrained and with random weights.

1. Padding is invisible: the encodings and the fleet vector of a set are the same (to rounding)
   alone and padded with extra cars that the mask hides, and a set in a batch is not affected by
   the other sets of the batch.
2. The set encoder is symmetric: permuting the cars permutes the encodings and leaves the fleet
   vector unchanged (to rounding). Without the set encoder a car's encoding does not depend on
   the other cars at all.
3. The hazard head: one logit per bin, so hazards in (0, 1); the starting hazards are the flat
   guess of the settings; the day factor and the site vector change the logits; without their
   switches the head has no input for them.
"""

import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import torch

from fleetreserve import hazard_curves, hazard_fit
from fleetreserve.hazard_data import N_INPUTS
from fleetreserve.hazard_settings import HazardSettings

torch.set_num_threads(1)  # PyTorch and LightGBM share the lane's processes: see hazard_net
SETTINGS = HazardSettings(width=16, heads=2, blocks=2, head_hidden=24)
TOLERANCE = 1e-5


def _net(settings: HazardSettings = SETTINGS):
    widths = hazard_curves.bin_widths(settings)
    return hazard_fit.build_net(settings, n_sites=3, n_energy=6, bin_widths=widths).eval()


def _cars(n: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, N_INPUTS, generator=generator)


def _true(shape) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.bool)


def test_padding_and_the_other_sets_of_a_batch_do_not_change_a_set():
    net = _net()
    alone = _cars(5)
    with torch.no_grad():
        encoded, fleet = net.encode(alone, _true(alone.shape[:2]))
        padded = torch.cat([alone, 7.0 * _cars(4, seed=1)], dim=1)
        mask = torch.cat([_true((1, 5)), torch.zeros(1, 4, dtype=torch.bool)], dim=1)
        encoded_p, fleet_p = net.encode(padded, mask)
        batch = torch.cat([padded, _cars(9, seed=2)], dim=0)
        mask_b = torch.cat([mask, _true((1, 9))], dim=0)
        encoded_b, fleet_b = net.encode(batch, mask_b)
    assert torch.allclose(encoded, encoded_p[:, :5], atol=TOLERANCE)
    assert torch.allclose(fleet, fleet_p, atol=TOLERANCE)
    assert torch.allclose(encoded, encoded_b[:1, :5], atol=TOLERANCE)
    assert torch.allclose(fleet, fleet_b[:1], atol=TOLERANCE)


def test_permuting_the_cars_permutes_the_encodings_and_keeps_the_fleet_vector():
    net = _net()
    cars = _cars(6)
    order = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        encoded, fleet = net.encode(cars, _true((1, 6)))
        encoded_o, fleet_o = net.encode(cars[:, order], _true((1, 6)))
    assert torch.allclose(encoded[:, order], encoded_o, atol=TOLERANCE)
    assert torch.allclose(fleet, fleet_o, atol=TOLERANCE)


def test_with_the_set_encoder_off_a_cars_encoding_ignores_the_other_cars():
    net = _net(HazardSettings(**{**SETTINGS.__dict__, "set_encoder": False}))
    cars = _cars(5)
    changed = cars.clone()
    changed[:, 1:] += 3.0
    with torch.no_grad():
        encoded, fleet = net.encode(cars, _true((1, 5)))
        encoded_c, fleet_c = net.encode(changed, _true((1, 5)))
    assert torch.equal(encoded[:, 0], encoded_c[:, 0])
    assert not torch.allclose(fleet, fleet_c)  # the fleet vector is the mean of the encodings


def test_with_the_set_encoder_on_a_cars_encoding_depends_on_the_other_cars():
    net = _net()
    cars = _cars(5)
    changed = cars.clone()
    changed[:, 1:] += 3.0
    with torch.no_grad():
        encoded, _ = net.encode(cars, _true((1, 5)))
        encoded_c, _ = net.encode(changed, _true((1, 5)))
    assert not torch.allclose(encoded[:, 0], encoded_c[:, 0], atol=1e-3)


def test_the_starting_hazards_are_the_flat_guess_and_lie_strictly_between_0_and_1():
    net = _net()
    widths = hazard_curves.bin_widths(SETTINGS)
    expected = np.minimum(widths / SETTINGS.initial_hazard_scale_h, 0.5)
    bias = net.hazard[-1].bias.detach().numpy()
    assert np.allclose(1 / (1 + np.exp(-bias)), expected, atol=1e-6)
    cars = _cars(4)
    with torch.no_grad():
        encoded, _ = net.encode(cars, _true((1, 4)))
        logits = net.hazard_logits(
            encoded, torch.zeros(1, 1), net.site_vector(torch.tensor([0]), 4)
        )
    hazards = torch.sigmoid(logits)
    assert logits.shape == (1, 4, len(widths))
    assert ((hazards > 0) & (hazards < 1)).all()


def test_the_day_factor_and_the_site_vector_change_the_hazards():
    net = _net()
    cars = _cars(4)
    with torch.no_grad():
        encoded, _ = net.encode(cars, _true((1, 4)))

        def logits(factor: float, site: int) -> torch.Tensor:
            vector = net.site_vector(torch.tensor([site]), 4)
            return net.hazard_logits(encoded, torch.full((1, 1), factor), vector)

        assert torch.equal(logits(0.5, 1), logits(0.5, 1))
        assert not torch.allclose(logits(0.0, 1), logits(2.0, 1))
        assert not torch.allclose(logits(0.0, 1), logits(0.0, 2))


def test_without_the_site_embedding_there_is_no_site_vector_and_without_the_factor_no_input():
    cars = _cars(4)
    no_site = _net(HazardSettings(**{**SETTINGS.__dict__, "site_embedding": False}))
    assert no_site.site_vector(torch.tensor([1]), 4) is None
    no_factor = _net(HazardSettings(**{**SETTINGS.__dict__, "day_factor": False}))
    with torch.no_grad():
        encoded, _ = no_factor.encode(cars, _true((1, 4)))
        site = no_factor.site_vector(torch.tensor([1]), 4)
        logits = no_factor.hazard_logits(encoded, None, site)
        other = no_factor.hazard_logits(encoded, None, no_factor.site_vector(torch.tensor([2]), 4))
    assert logits.shape == (1, 4, len(hazard_curves.bin_widths(SETTINGS)))
    assert not torch.allclose(logits, other)  # the site vector still reaches the head


def test_torch_state_sets_the_thread_count_of_the_settings_and_restores_the_flag():
    was = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(2)  # no torch operation runs while it is 2; see the module text
    try:
        with hazard_fit.torch_state(HazardSettings(threads=1)):
            assert torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.set_num_threads(1)
    assert torch.are_deterministic_algorithms_enabled() == was

"""The two changes the tuning makes to the hazard model's code: checkpoints of a fit
(`hazard_fit.fit`, argument `checkpoints`) and the restriction of the inputs
(`HazardSettings.inputs`), and the parts (the plan's ablations) built on them.

1. A checkpoint of a longer fit is the state of the shorter fit, byte for byte: the forecast table,
   every weight and the loss history, inside and after the divergence warm-up, with and without
   the day factor and the set encoder. Fitting with checkpoints does not change the fit. The
   network is in evaluation mode in a checkpoint and back in training mode after it. A checkpoint
   outside the fit's epochs is refused.
2. The default model is unchanged by the new settings field: the same inputs given as the full
   list of columns give the same table byte for byte as no restriction, and `Scaler` without the
   field's value makes no mask and transforms as its mean and spread say.
3. The restriction: the mask keeps the named columns, their missing-value flags and both phases of
   a cyclic column; a name that is not an input is refused; a calendar-only model's forecast does
   not change when any other input changes, a full model's does.
4. Each of the five parts is a network or input rule that differs from the full model: no encoder
   blocks, no site vectors, no day factor, neither of the two, a mask; each has a result on the
   made-up folds.
5. The fold's frame as the model reads it (`TuneModel.attach`) is served for that frame object
   and for no other: a frame of another size, a frame of the same size with other content and an
   equal copy each get the context the model makes from that frame.
"""

from dataclasses import replace

import hazard_fake as hf
import lightgbm  # noqa: F401  (before torch: see hazard_net)
import numpy as np
import pandas as pd
import pytest
import torch
from hazard_tune_fake import frames

from fleetreserve import hazard_data, hazard_fit, hazard_tune, learn_tune
from fleetreserve.forecast_empirical import TUNING_KAPPAS
from fleetreserve.hazard_settings import HazardSettings
from fleetreserve.hazard_tune_grid import PRESENT, Setting
from fleetreserve.models.hazard import HazardModel


def bytes_of(table) -> bytes:
    return b"".join(table[c].to_numpy().tobytes() for c in table if table[c].dtype.kind in "fi")


@pytest.fixture(scope="module")
def fold(tmp_path_factory):
    table = learn_tune.tune_table(frames(tmp_path_factory.mktemp("units")), rho=0.0)
    return hazard_tune.prepare_fold(table, "jpl")


def fitted(fold, settings, checkpoints=None):
    prepared = hazard_tune.prepare(fold, settings)
    model = hazard_tune.TuneModel(kappas=TUNING_KAPPAS, settings=settings)
    model.attach(fold)
    model.fit_prepared(prepared, checkpoints)
    return model


def weights(model) -> list:
    return [(name, tensor.clone()) for name, tensor in model.net.state_dict().items()]


def same_weights(a, b) -> bool:
    return len(a) == len(b) and all(
        n == m and torch.equal(x, y) for (n, x), (m, y) in zip(a, b, strict=True)
    )


# --- 1. checkpoints ------------------------------------------------------------------------------

VARIANTS = {
    "full": {},
    "no day factor": {"day_factor": False},
    "no set encoder": {"set_encoder": False},
    "no warm-up": {"kl_warmup_epochs": 0},
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_checkpoint_of_a_longer_fit_is_the_shorter_fit_byte_for_byte(fold, variant):
    base = HazardSettings(width=16, **VARIANTS[variant])
    seen = {}

    def keep(epoch):
        def read(net):
            model.net = net
            seen[epoch] = (model.predict(fold.frame, TUNING_KAPPAS), weights(model))

        return read

    longer = replace(base, epochs=9)
    prepared = hazard_tune.prepare(fold, longer)
    model = hazard_tune.TuneModel(kappas=TUNING_KAPPAS, settings=longer)
    model.attach(fold)
    model.fit_prepared(prepared, {3: keep(3), 7: keep(7)})  # 3 is inside the warm-up of 5 epochs
    final = (model.predict(fold.frame, TUNING_KAPPAS), weights(model))
    for epochs in (3, 7):
        short = fitted(fold, replace(base, epochs=epochs))
        table = short.predict(fold.frame, TUNING_KAPPAS)
        assert seen[epochs][0].equals(table) and bytes_of(seen[epochs][0]) == bytes_of(table)
        assert same_weights(seen[epochs][1], weights(short))
        assert short.history == model.history[:epochs]
    plain = fitted(fold, longer)  # checkpoints do not change the fit they are read from
    assert bytes_of(final[0]) == bytes_of(plain.predict(fold.frame, TUNING_KAPPAS))
    assert same_weights(final[1], weights(plain)) and plain.history == model.history
    assert bytes_of(seen[3][0]) != bytes_of(final[0])  # the epochs did change the forecast


def test_the_network_is_in_evaluation_mode_in_a_checkpoint_and_in_training_mode_after_it(fold):
    settings = HazardSettings(width=16, epochs=3)
    prepared = hazard_tune.prepare(fold, settings)
    in_checkpoint, in_training_steps = [], []

    def look(net):
        in_checkpoint.append(net.training)
        if len(in_checkpoint) == 1:  # from now on note the mode of each training step
            original = net.forward_fit

            def spy(*args, **kwargs):
                in_training_steps.append(net.training)
                return original(*args, **kwargs)

            net.forward_fit = spy

    fitted_net = hazard_fit.fit(
        prepared.data,
        settings,
        len(prepared.fit_sites),
        np.array([0.5]),
        np.array([0]),
        {1: look, 2: look},
    )
    assert in_checkpoint == [False, False]
    assert len(in_training_steps) > 0 and all(in_training_steps)  # epochs 2 and 3 train
    assert fitted_net.net.training is False  # as after any fit
    assert len(fitted_net.history) == 3


@pytest.mark.parametrize("epoch", [0, 4, -1])
def test_a_checkpoint_outside_the_epochs_of_the_fit_is_refused(fold, epoch):
    settings = HazardSettings(width=16, epochs=3)
    prepared = hazard_tune.prepare(fold, settings)
    with pytest.raises(ValueError, match="checkpoint epochs"):
        hazard_fit.fit(
            prepared.data,
            settings,
            len(prepared.fit_sites),
            np.array([0.5]),
            np.array([0]),
            {epoch: lambda net: None},
        )


# --- 2. the default model ------------------------------------------------------------------------


def test_all_the_input_columns_named_give_the_default_model_byte_for_byte():
    """`inputs=None` (no mask) and every input column named (a mask of ones) give the same
    forecast table, survival columns included."""
    train, fresh, _ = hf.known_data(60, 10)
    tables = {}
    for label, inputs in (("default", None), ("all named", hazard_data.INPUT_COLUMNS)):
        settings = HazardSettings(width=16, epochs=4, inputs=inputs)
        model = HazardModel(rho=0.0, kappas=hf.KAPPAS, settings=settings)
        model.fit(train)
        assert (model.scaler.keep is None) is (inputs is None)
        tables[label] = model.predict(fresh, hf.KAPPAS)
    assert tables["default"].equals(tables["all named"])
    assert bytes_of(tables["default"]) == bytes_of(tables["all named"])
    assert HazardSettings().inputs is None


def test_the_default_scaler_makes_no_mask_and_transforms_as_the_mean_and_spread_say(fold):
    hazard_tune.prepare(fold, HazardSettings())
    scaler = hazard_data.Scaler.fit(fold.fit_table, HazardSettings())
    assert scaler.keep is None
    out = scaler.transform(fold.context)
    plain = hazard_data._plain_values(fold.context)
    scaled = np.clip((plain - scaler.mean) / scaler.std, -scaler.clip, scaler.clip)
    assert (
        out[:, : plain.shape[1]].tobytes()
        == np.where(np.isnan(plain), 0.0, scaled).astype(np.float32).tobytes()
    )


# --- 3. the restriction --------------------------------------------------------------------------


def test_the_mask_keeps_the_named_columns_their_flags_and_both_phases_of_a_cyclic_column():
    mask = hazard_data.input_mask(hazard_tune.CALENDAR_INPUTS)
    assert mask.shape == (hazard_data.N_INPUTS,) and mask.dtype == np.float32
    plain, flags = len(hazard_data.PLAIN_COLUMNS), len(hazard_data.MISSING_COLUMNS)
    kept = hazard_data.PLAIN_COLUMNS.index("public_holiday")
    assert mask[:plain].tolist() == [1.0 if k == kept else 0.0 for k in range(plain)]
    assert not mask[plain : plain + flags].any()
    cyclic = mask[plain + flags :].reshape(len(hazard_data.CYCLIC_PERIODS), 2)
    wanted = {"weekday": 1.0, "month": 1.0, "day_of_year": 1.0, "arrival_hour": 0.0}
    assert dict(zip(hazard_data.CYCLIC_PERIODS, cyclic[:, 0], strict=True)) == wanted
    assert (cyclic[:, 0] == cyclic[:, 1]).all() and mask.sum() == 7
    flagged = hazard_data.input_mask(("history_mean_stay_h",))
    assert flagged.sum() == 2  # the column and its missing-value flag
    assert hazard_data.input_mask(hazard_data.INPUT_COLUMNS).sum() == hazard_data.N_INPUTS


def test_a_name_that_is_not_an_input_and_a_bad_inputs_value_are_refused(fold):
    with pytest.raises(ValueError, match="not input columns"):
        hazard_data.input_mask(("weekday", "label_departure_h"))
    with pytest.raises(ValueError, match="not input columns"):
        hazard_data.Scaler.fit(fold.fit_table, HazardSettings(inputs=("declared_departure_h",)))
    for bad in ((), ["weekday"], (1,)):
        with pytest.raises(ValueError, match="inputs is None"):
            HazardSettings(inputs=bad)


def test_the_scaler_puts_the_other_inputs_to_zero_and_keeps_the_named_ones(fold):
    hazard_tune.prepare(fold, HazardSettings())  # sets the frame as the model reads it
    full = hazard_data.Scaler.fit(fold.fit_table, HazardSettings())
    restricted = hazard_data.Scaler.fit(
        fold.fit_table, HazardSettings(inputs=hazard_tune.CALENDAR_INPUTS)
    )
    x_full, x_calendar = full.transform(fold.context), restricted.transform(fold.context)
    kept = hazard_data.input_mask(hazard_tune.CALENDAR_INPUTS) > 0
    assert full.keep is None and restricted.keep is not None
    assert not x_calendar[:, ~kept].any() and x_full[:, ~kept].any()
    assert x_calendar[:, kept].tobytes() == x_full[:, kept].tobytes()


@pytest.fixture
def altered_frame(fold):
    """The forecast frame with every input but the calendar ones changed."""
    frame = fold.frame.copy()
    frame["arrival_hour"] = (frame["arrival_hour"] + 5) % 24
    frame["hours_since_plug_in"] = frame["hours_since_plug_in"] * 2.0 + 1.0
    frame["fleet_at_tau"] = frame["fleet_at_tau"] + 3
    frame["user_known"] = ~frame["user_known"].astype(bool)
    frame["history_mean_stay_h"] = frame["history_mean_stay_h"].fillna(1.0) + 2.0
    frame["site_type"] = 1 - frame["site_type"]
    return frame


def test_a_calendar_only_forecast_ignores_every_other_input_and_a_full_forecast_does_not(
    fold, altered_frame
):
    for name, changes, ignores in (
        ("calendar only", {"inputs": hazard_tune.CALENDAR_INPUTS}, True),
        ("full", {}, False),
    ):
        settings = HazardSettings(width=16, epochs=4, **changes)
        model = fitted(fold, settings)
        original = model.predict(fold.frame, TUNING_KAPPAS)
        model.shared = None
        moved = model.predict(altered_frame, TUNING_KAPPAS)
        assert (bytes_of(original) == bytes_of(moved)) is ignores, name


# --- 4. the parts --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(hazard_tune.PARTS))
def test_each_part_is_a_different_network_or_input_rule_and_has_a_result(fold, name):
    changes = hazard_tune.PARTS[name]
    full = fitted(fold, PRESENT.hazard(0, epochs=2))
    part = fitted(fold, PRESENT.hazard(0, epochs=2, **changes))
    assert hasattr(full.net, "blocks") and hasattr(full.net, "site") and hasattr(full.net, "prior")
    assert full.scaler.keep is None
    assert hasattr(part.net, "blocks") is changes.get("set_encoder", True)
    assert hasattr(part.net, "site") is changes.get("site_embedding", True)
    assert hasattr(part.net, "prior") is changes.get("day_factor", True)
    assert (part.scaler.keep is not None) is ("inputs" in changes)
    found = hazard_tune.forecast_bounds(fold, PRESENT.hazard(0, epochs=2, **changes), (1, 2))
    assert set(found) == {1, 2}
    result = hazard_tune.score_bounds(fold, found[2])
    assert result["score"] > 0 and result["scored_rows"] == fold.scored_rows
    # the forecast table (the energy head's columns too) differs from the full model's
    assert bytes_of(part.predict(fold.frame, TUNING_KAPPAS)) != bytes_of(
        full.predict(fold.frame, TUNING_KAPPAS)
    )


def test_the_five_parts_are_the_plans_ablations():
    assert list(hazard_tune.PARTS) == [
        "no_set_encoder",
        "no_site_embedding",
        "no_day_factor",
        "no_set_encoder_no_day_factor",
        "calendar_only_inputs",
    ]
    assert hazard_tune.PARTS["no_set_encoder_no_day_factor"] == {
        "set_encoder": False,
        "day_factor": False,
    }
    assert hazard_tune.CALENDAR_INPUTS == ("weekday", "month", "day_of_year", "public_holiday")
    assert Setting(32, 0.003, 1.0, 30) == PRESENT


# --- 5. the fold's frame as the model reads it ---------------------------------------------------


def test_the_context_kept_by_attach_is_served_for_the_folds_own_frame_only(fold):
    model = hazard_tune.TuneModel(kappas=TUNING_KAPPAS, settings=HazardSettings(width=16, epochs=1))
    model.attach(fold)
    assert fold.context is not None and model._context(fold.frame) is fold.context
    smaller = fold.frame.iloc[:-3].reset_index(drop=True)
    moved = fold.frame.copy()
    moved["site"] = "OSL_S"  # a site of the other type: the context columns change
    copy = fold.frame.copy()
    assert moved.shape == fold.frame.shape == copy.shape and smaller.shape != fold.frame.shape
    assert not HazardModel._context(model, moved).equals(fold.context)  # the content matters
    for other in (smaller, moved, copy):
        made = model._context(other)
        assert made is not fold.context
        pd.testing.assert_frame_equal(made, HazardModel._context(model, other))
    assert len(model._context(smaller)) == len(smaller)
    assert model._context(copy).equals(fold.context)  # equal content, but made anew

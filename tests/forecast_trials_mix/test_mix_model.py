"""`models.mix_tail_inputs.MixModel` with made-up members. Written, not run. Each test says which
edit turns it red.

1. `fit` fits both members, each on the frame it is given. Red if either `fit` call is removed.
2. `predict` is the equal-weight average of the two members' tables, asked for the same levels:
   members giving 2 and 4 give 3; members giving the same table give that table. Red if the
   neural member's `predict` is replaced by the tree member's (the mean would be 2), or if the
   levels are not passed on as given to both.
3. Members fitted on different units, or on units that are not the training units, are refused.
   Red if either check in `__init__` is removed.
4. The mix declares what the neural member declares (`extra_inputs`, `context_sources`), so
   `model_inputs.extra_of` builds the neural member's extra columns for the frame; it is not a
   look-ahead model. Red if `extra_inputs` is not taken from the neural member.
5. The module imports no learned-model package at its top (nothing of the `learn` group): a
   command refused before it builds a model never imports PyTorch or LightGBM. Red if an import
   of `lightgbm`, `torch` or a member module is added at the top.
6. The members are the two the plan names, and both exist as modules. Red if `MEMBERS` changes.
"""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fleetreserve import model_inputs
from fleetreserve.forecast_model import (
    KEY_COLUMNS,
    NUMBER_COLUMNS,
    SURVIVAL_COLUMNS,
    TABLE_COLUMNS,
    check_table,
)
from fleetreserve.models import mix_tail_inputs
from fleetreserve.models.mix_tail_inputs import MEMBERS, MixModel
from fleetreserve.truth_build import TRAIN_UNITS

MODULE = Path(mix_tail_inputs.__file__)


def cars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ["U", "U", "U"],
            "site": ["A", "A", "B"],
            "date": ["2026-01-05", "2026-01-05", "2026-01-06"],
            "session_id": ["s1", "s2", "s3"],
        }
    )


class Member:
    """A member that gives the departure value `value` at every car and level, and records its
    calls. `survival` is the value of the survival columns (NaN: none, as for the tree model)."""

    def __init__(self, value, survival=np.nan, extra=(), fit_units=TRAIN_UNITS, sources=None):
        self.value, self.survival = value, survival
        self.extra_inputs = extra
        self.fit_units = fit_units
        self.context_sources = sources
        self.context_files = {}
        self.fitted_on = []
        self.asked = []

    def fit(self, frame):
        self.fitted_on.append(frame)

    def predict(self, frame, kappas):
        self.asked.append(list(kappas))
        rows = []
        for level in kappas:
            for _, car in frame.iterrows():
                row = {c: car[c] for c in KEY_COLUMNS}
                row.update({c: self.value for c in NUMBER_COLUMNS})
                row.update({c: self.survival for c in SURVIVAL_COLUMNS})
                row.update(kappa=level, rho=0.0)
                rows.append(row)
        return check_table(pd.DataFrame(rows, columns=list(TABLE_COLUMNS)))


def test_fit_fits_both_members_on_the_frame_given():
    tail, neural = Member(2.0), Member(4.0)
    frame = cars()
    MixModel(tail=tail, neural=neural).fit(frame)
    assert len(tail.fitted_on) == 1 and tail.fitted_on[0] is frame
    assert len(neural.fitted_on) == 1 and neural.fitted_on[0] is frame


def test_predict_is_the_equal_weight_average_of_the_two_tables_at_the_same_levels():
    tail, neural = Member(2.0), Member(4.0, survival=6.0)
    model = MixModel(tail=tail, neural=neural)
    table = model.predict(cars(), (0.1, 0.5))
    assert tail.asked == [[0.1, 0.5]] and neural.asked == [[0.1, 0.5]]
    assert len(table) == 6
    assert (table["d_hat"] == 3.0).all() and (table["e_cap"] == 3.0).all()
    assert table[list(SURVIVAL_COLUMNS)].isna().all().all()  # the tree member gives none
    same = MixModel(tail=Member(2.5, 1.0), neural=Member(2.5, 1.0)).predict(cars(), [0.3])
    assert (same["d_hat"] == 2.5).all() and (same["survival_h8"] == 1.0).all()
    assert model.name == "mix_tail_inputs"


def test_members_fitted_on_other_units_are_refused():
    with pytest.raises(ValueError, match="same units"):
        MixModel(tail=Member(1.0), neural=Member(1.0, fit_units=("caltech",)))
    with pytest.raises(ValueError, match="training units"):
        MixModel(tail=Member(1.0, fit_units=("OSL_1",)), neural=Member(1.0, fit_units=("OSL_1",)))
    assert MixModel(tail=Member(1.0), neural=Member(1.0)).fit_units == TRAIN_UNITS


def test_the_mix_declares_what_the_neural_member_declares():
    extra = tuple(model_inputs.COLUMNS[:2])
    sources = object()
    model = MixModel(tail=Member(1.0), neural=Member(1.0, extra=extra, sources=sources))
    assert model.extra_inputs == extra and model.context_sources is sources
    assert model_inputs.declared(model) == extra
    assert not model_inputs.reads_look_ahead(model) and model.look_ahead is False
    plain = MixModel(tail=Member(1.0), neural=Member(1.0))
    assert plain.extra_inputs == () and model_inputs.declared(plain) == ()


def test_the_module_imports_no_learned_model_package_at_its_top():
    names = []
    for node in ast.parse(MODULE.read_text()).body:
        if isinstance(node, ast.ImportFrom):
            names += [node.module or "", *(a.name for a in node.names)]
        elif isinstance(node, ast.Import):
            names += [a.name for a in node.names]
    assert names
    for name in names:
        assert not any(w in name for w in ("torch", "lightgbm", "lgbm", "hazard")), name


def test_the_members_are_the_two_the_plan_names_and_both_exist():
    assert MEMBERS == ("lgbm_tail", "hazard_tuned_inputs")
    for name in MEMBERS:
        assert importlib.util.find_spec(f"fleetreserve.models.{name}") is not None

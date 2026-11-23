"""The learn lane works: the packages of the `learn` dependency group load and compute.

These tests run only in the learn lane (`uv run --frozen --group learn python tools/run_tests.py
--lane learn`, which `check.sh` runs). They do not download model weights.
"""

import numpy as np


def test_lifelines_fits_a_kaplan_meier_curve():
    from lifelines import KaplanMeierFitter

    fitter = KaplanMeierFitter().fit([1, 2, 3, 4], event_observed=[1, 1, 1, 1])
    assert fitter.survival_function_at_times([0, 2, 4]).tolist() == [1.0, 0.5, 0.0]


def test_lightgbm_fits_a_small_regression():
    import lightgbm

    x = np.arange(40, dtype=float).reshape(-1, 1)
    model = lightgbm.LGBMRegressor(n_estimators=20, min_child_samples=2, verbose=-1)
    model.fit(x, 2.0 * x[:, 0])
    assert abs(model.predict(np.array([[20.0]]))[0] - 40.0) < 4.0


def test_sklearn_fits_a_line():
    from sklearn.linear_model import LinearRegression

    x = np.arange(10, dtype=float).reshape(-1, 1)
    model = LinearRegression().fit(x, 3.0 * x[:, 0] + 1.0)
    assert abs(model.coef_[0] - 3.0) < 1e-9 and abs(model.intercept_ - 1.0) < 1e-9


def test_holidays_knows_christmas_day():
    import holidays

    assert "2026-12-25" in holidays.country_holidays("NO", years=2026)


def test_torch_takes_a_gradient_step(capsys):
    import torch

    weight = torch.tensor(1.0, requires_grad=True)
    ((weight * 3.0 - 6.0) ** 2).backward()
    assert weight.grad.item() == -18.0
    with capsys.disabled():
        print(f"\ntorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")

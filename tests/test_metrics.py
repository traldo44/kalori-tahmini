import numpy as np
import pytest

from src.metrics import (bootstrap_ci, calorie_bins, classification_metrics, mae,
                         paired_bootstrap, regression_metrics, to_bins)


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200, 300], [110, 180, 300])
    assert m["MAE"] == pytest.approx(10)
    assert m["PMAE"] == pytest.approx(100 * 10 / 200)            # MAE / ortalama gerçek
    assert m["MAPE"] == pytest.approx(100 * (0.1 + 0.1 + 0) / 3)  # örnek başına yüzde hatanın ortalaması
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 3))
    assert m["within_100kcal"] == 100


def test_pmae_and_mape_differ_when_small_dishes_have_errors():
    m = regression_metrics([10, 500], [30, 500])
    assert m["PMAE"] < m["MAPE"]


def test_zero_calorie_dish_does_not_divide_by_zero():
    assert np.isfinite(regression_metrics([0, 100], [5, 100])["MAPE"])


def test_classification_metrics_with_top5():
    y = [0, 1, 2, 3, 4, 5]
    scores = np.eye(6)
    m = classification_metrics(y, y, scores)
    assert m["accuracy"] == m["f1"] == m["top5"] == 1


def test_calorie_bins_use_train_tertiles():
    edges = calorie_bins(np.arange(1, 301))
    assert list(to_bins([10, 150, 290], edges)) == [0, 1, 2]


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 800, 500)
    p = y + rng.normal(0, 80, 500)
    lo, hi = bootstrap_ci(y, p, mae)
    assert lo < mae(y, p) < hi
    m = regression_metrics(y, p, ci=True)
    assert (m["MAE_ci_low"], m["MAE_ci_high"]) == (lo, hi)


def test_paired_bootstrap_detects_real_difference_and_ignores_noise():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 800, 500)
    good = y + rng.normal(0, 60, 500)
    bad = y + rng.normal(0, 120, 500)
    assert paired_bootstrap(y, good, bad)["significant"]
    assert paired_bootstrap(y, good, bad)["diff"] < 0
    almost_same = good + rng.normal(0, 1, 500)
    assert not paired_bootstrap(y, good, almost_same)["significant"]

"""Tests for provider-neutral cheap fractal reference estimators."""

from __future__ import annotations

import math

import pytest

from liq.fractal import estimate_dfa, estimate_ghe, estimate_hurst_rs


@pytest.mark.parametrize("estimator", [estimate_hurst_rs, estimate_dfa, estimate_ghe])
def test_reference_estimators_return_finite_scaling_values(estimator) -> None:
    analytical_path = tuple(
        math.sin(index / 5.0) + 0.2 * math.cos(index / 11.0) for index in range(240)
    )

    value = estimator(analytical_path)

    assert math.isfinite(value)


@pytest.mark.parametrize("estimator", [estimate_hurst_rs, estimate_dfa, estimate_ghe])
def test_reference_estimators_reject_non_finite_and_short_inputs(estimator) -> None:
    with pytest.raises(ValueError, match="at least"):
        estimator((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="finite"):
        estimator(tuple(float(index) for index in range(31)) + (float("nan"),))


def test_ghe_validates_order_and_lag_range() -> None:
    values = tuple(math.sin(index / 7.0) for index in range(64))

    with pytest.raises(ValueError, match="q"):
        estimate_ghe(values, q=0.0)
    with pytest.raises(ValueError, match="max_lag"):
        estimate_ghe(values, max_lag=32)


def test_dfa_supports_the_frozen_thirty_observation_window() -> None:
    values = [float(index % 7) / 10.0 for index in range(30)]

    estimate = estimate_dfa(values)

    assert math.isfinite(estimate)

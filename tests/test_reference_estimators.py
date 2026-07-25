"""Tests for provider-neutral cheap fractal reference estimators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from liq.fractal import (
    estimate_dfa,
    estimate_ghe,
    estimate_hurst_rs,
    estimate_mfdfa_width,
    estimate_mfdfa_width_surrogate_adjusted,
    estimate_wavelet_energy_ratio,
    estimate_wavelet_entropy,
)


def _binomial_cascade(levels: int, p: float) -> list[float]:
    """Deterministic binomial multiplicative measure — canonically multifractal."""
    weights = np.array([1.0])
    for _ in range(levels):
        weights = np.concatenate([weights * p, weights * (1.0 - p)])
    return weights.tolist()


def _monofractal_path(n: int) -> list[float]:
    return [math.sin(i / 5.0) + 0.2 * math.cos(i / 11.0) for i in range(n)]


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


def test_mfdfa_width_is_finite_and_non_negative() -> None:
    width = estimate_mfdfa_width(_monofractal_path(256))
    assert math.isfinite(width)
    assert width >= 0.0


def test_mfdfa_width_larger_for_multifractal_than_monofractal() -> None:
    mono = estimate_mfdfa_width(_monofractal_path(512))
    multi = estimate_mfdfa_width(_binomial_cascade(9, 0.3))
    assert multi > mono


def test_mfdfa_rejects_short_and_too_few_q() -> None:
    with pytest.raises(ValueError, match="at least"):
        estimate_mfdfa_width((1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="three q values"):
        estimate_mfdfa_width(_monofractal_path(256), q_values=(1.0, 2.0))


def test_mfdfa_surrogate_adjusted_is_finite_and_deterministic() -> None:
    series = _binomial_cascade(9, 0.3)
    first = estimate_mfdfa_width_surrogate_adjusted(series, n_surrogates=8, seed=7)
    second = estimate_mfdfa_width_surrogate_adjusted(series, n_surrogates=8, seed=7)
    assert math.isfinite(first)
    assert first == second  # deterministic in seed


def test_mfdfa_surrogate_adjustment_changes_the_raw_width() -> None:
    series = _binomial_cascade(9, 0.3)
    raw = estimate_mfdfa_width(series)
    adjusted = estimate_mfdfa_width_surrogate_adjusted(series, n_surrogates=8, seed=1)
    assert adjusted != raw


def test_wavelet_entropy_is_normalized_between_zero_and_one() -> None:
    entropy = estimate_wavelet_entropy(_monofractal_path(256))
    assert 0.0 <= entropy <= 1.0


def test_wavelet_entropy_higher_for_broadband_than_single_scale() -> None:
    rng = np.random.default_rng(3)
    broadband = rng.standard_normal(512).tolist()  # energy spread across scales
    single_scale = [math.sin(i / 2.0) for i in range(512)]  # energy at one scale
    assert estimate_wavelet_entropy(broadband) > estimate_wavelet_entropy(single_scale)


def test_wavelet_energy_ratio_in_unit_interval_and_validates() -> None:
    ratio = estimate_wavelet_energy_ratio(_monofractal_path(256), fine_levels=2)
    assert 0.0 <= ratio <= 1.0
    with pytest.raises(ValueError, match="fine_levels"):
        estimate_wavelet_energy_ratio(_monofractal_path(256), fine_levels=0)


def test_wavelet_energy_ratio_higher_for_choppy_series() -> None:
    choppy = [(-1.0) ** i for i in range(256)]  # all energy at the finest scale
    smooth = [float(i) for i in range(256)]  # energy at long scales (trend)
    assert estimate_wavelet_energy_ratio(choppy) > estimate_wavelet_energy_ratio(smooth)


@pytest.mark.parametrize(
    "estimator", [estimate_wavelet_entropy, estimate_wavelet_energy_ratio]
)
def test_wavelet_estimators_reject_short_inputs(estimator) -> None:
    with pytest.raises(ValueError, match="at least"):
        estimator((1.0, 2.0, 3.0))


@pytest.mark.parametrize(
    "estimator", [estimate_wavelet_entropy, estimate_wavelet_energy_ratio]
)
def test_wavelet_estimators_reject_zero_energy_series(estimator) -> None:
    flat = [5.0] * 256  # constant -> zero wavelet detail energy
    with pytest.raises(ValueError, match="positive"):
        estimator(flat)

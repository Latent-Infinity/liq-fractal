"""Small, provider-neutral reference estimators for cheap descriptors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

_MIN_OBSERVATIONS = 16


def estimate_hurst_rs(values: Sequence[float]) -> float:
    """Estimate a scaling exponent from rescaled ranges across window sizes."""

    series = _validated_series(values)
    scales = _dyadic_scales(series.size)
    statistics: list[float] = []
    used_scales: list[int] = []
    for scale in scales:
        segment_statistics: list[float] = []
        for segment in _segments(series, scale):
            centered = segment - np.mean(segment)
            standard_deviation = float(np.std(centered, ddof=1))
            if standard_deviation <= 0.0:
                continue
            cumulative = np.cumsum(centered)
            segment_statistics.append(
                float((np.max(cumulative) - np.min(cumulative)) / standard_deviation)
            )
        if segment_statistics:
            used_scales.append(scale)
            statistics.append(float(np.mean(segment_statistics)))
    return _log_slope(used_scales, statistics, estimator="rescaled-range")


def estimate_dfa(values: Sequence[float]) -> float:
    """Estimate the linear detrended-fluctuation scaling exponent."""

    series = _validated_series(values)
    integrated = np.cumsum(series - np.mean(series))
    scales = _dyadic_scales(series.size)
    fluctuations: list[float] = []
    used_scales: list[int] = []
    for scale in scales:
        residual_squares: list[float] = []
        x = np.arange(scale, dtype=float)
        for segment in _segments(integrated, scale):
            coefficients = np.polyfit(x, segment, 1)
            residual_squares.extend(np.square(segment - np.polyval(coefficients, x)))
        fluctuation = float(np.sqrt(np.mean(residual_squares)))
        if fluctuation > 0.0:
            used_scales.append(scale)
            fluctuations.append(fluctuation)
    return _log_slope(used_scales, fluctuations, estimator="DFA")


def estimate_ghe(
    values: Sequence[float], *, q: float = 1.0, max_lag: int | None = None
) -> float:
    """Estimate a generalized Hurst exponent from q-order increment moments."""

    series = _validated_series(values)
    if q <= 0.0 or not np.isfinite(q):
        raise ValueError("q must be finite and positive")
    largest_allowed_lag = series.size // 4
    resolved_max_lag = min(20, largest_allowed_lag) if max_lag is None else max_lag
    if resolved_max_lag < 2 or resolved_max_lag > largest_allowed_lag:
        raise ValueError(f"max_lag must be between 2 and {largest_allowed_lag}")
    lags = np.arange(1, resolved_max_lag + 1, dtype=float)
    moments = np.asarray(
        [
            np.mean(np.abs(series[lag:] - series[:-lag]) ** q)
            for lag in range(1, resolved_max_lag + 1)
        ]
    )
    positive = moments > 0.0
    if np.count_nonzero(positive) < 2:
        raise ValueError("GHE requires at least two positive increment moments")
    slope = float(np.polyfit(np.log(lags[positive]), np.log(moments[positive]), 1)[0])
    return slope / q


def _validated_series(values: Sequence[float]) -> npt.NDArray[np.float64]:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size < _MIN_OBSERVATIONS:
        raise ValueError(
            f"descriptor estimators require at least {_MIN_OBSERVATIONS} values"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("descriptor estimator inputs must be finite")
    return series


def _dyadic_scales(observation_count: int) -> tuple[int, ...]:
    maximum = observation_count // 4
    scales: list[int] = []
    scale = 4
    while scale <= maximum:
        scales.append(scale)
        scale *= 2
    if len(scales) < 2:
        raise ValueError("descriptor estimator requires at least two usable scales")
    return tuple(scales)


def _segments(
    values: npt.NDArray[np.float64], scale: int
) -> tuple[npt.NDArray[np.float64], ...]:
    segment_count = values.size // scale
    return tuple(
        values[index * scale : (index + 1) * scale] for index in range(segment_count)
    )


def _log_slope(
    scales: Sequence[int], statistics: Sequence[float], *, estimator: str
) -> float:
    if len(scales) < 2:
        raise ValueError(f"{estimator} requires at least two usable scales")
    return float(np.polyfit(np.log(scales), np.log(statistics), 1)[0])

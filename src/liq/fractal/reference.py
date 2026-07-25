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
    scales = _dfa_scales(series.size)
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


_MFDFA_Q_VALUES: tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)


def estimate_mfdfa_width(
    values: Sequence[float], *, q_values: Sequence[float] = _MFDFA_Q_VALUES
) -> float:
    """Estimate the multifractal spectrum width ``Delta alpha`` via MF-DFA.

    Generalizes DFA to a range of moment orders ``q``: the q-order fluctuation
    ``F_q(s) ~ s^h(q)`` yields a generalized Hurst spectrum ``h(q)``, hence
    ``tau(q) = q*h(q) - 1``, ``alpha = d tau / d q`` and the singularity
    spectrum width ``Delta alpha = max(alpha) - min(alpha)``. A monofractal
    series has ``h(q)`` constant and ``Delta alpha ~ 0``; multifractality
    widens it. Reduces to :func:`estimate_dfa` at ``q = 2``.

    Raw ``Delta alpha`` is inflated by heavy tails / breaks — prefer
    :func:`estimate_mfdfa_width_surrogate_adjusted` for alpha-facing use.
    """

    series = _validated_series(values)
    qs = np.asarray(sorted(q_values), dtype=float)
    if qs.size < 3:
        raise ValueError("MF-DFA requires at least three q values")
    integrated = np.cumsum(series - np.mean(series))
    scales = _dfa_scales(series.size)

    # Detrended segment variances per scale (forward + backward coverage).
    variances_by_scale: list[npt.NDArray[np.float64]] = []
    for scale in scales:
        x = np.arange(scale, dtype=float)
        segment_variances: list[float] = []
        for source in (integrated, integrated[::-1]):
            for segment in _segments(source, scale):
                coefficients = np.polyfit(x, segment, 1)
                residual = segment - np.polyval(coefficients, x)
                segment_variances.append(float(np.mean(np.square(residual))))
        variances_by_scale.append(np.asarray(segment_variances, dtype=float))

    hq: list[float] = []
    for q in qs:
        log_scales: list[float] = []
        log_fluctuations: list[float] = []
        for scale, variances in zip(scales, variances_by_scale, strict=True):
            positive = variances[variances > 0.0]
            if positive.size == 0:
                continue
            if abs(q) < 1e-6:
                fluctuation = float(np.exp(0.5 * np.mean(np.log(positive))))
            else:
                fluctuation = float(np.mean(positive ** (q / 2.0)) ** (1.0 / q))
            if fluctuation > 0.0:
                log_scales.append(np.log(scale))
                log_fluctuations.append(np.log(fluctuation))
        if len(log_scales) < 2:
            raise ValueError("MF-DFA requires at least two usable scales per q")
        hq.append(float(np.polyfit(log_scales, log_fluctuations, 1)[0]))

    tau = qs * np.asarray(hq) - 1.0
    alpha = np.gradient(tau, qs)
    return float(np.max(alpha) - np.min(alpha))


def estimate_mfdfa_width_surrogate_adjusted(
    values: Sequence[float],
    *,
    n_surrogates: int = 20,
    seed: int = 0,
    q_values: Sequence[float] = _MFDFA_Q_VALUES,
) -> float:
    """Surrogate-adjusted MF-DFA width: raw ``Delta alpha`` minus the mean width
    of shuffled surrogates.

    Random permutation destroys temporal (multifractal) correlations while
    preserving the value distribution, so the surrogate width isolates the
    apparent multifractality contributed by heavy tails alone. The adjusted
    value estimates the *correlation-borne* multifractality (may be negative).
    Deterministic in ``seed``.
    """

    raw = estimate_mfdfa_width(values, q_values=q_values)
    series = _validated_series(values)
    rng = np.random.default_rng(seed)
    surrogate_widths = [
        estimate_mfdfa_width(rng.permutation(series).tolist(), q_values=q_values)
        for _ in range(n_surrogates)
    ]
    return raw - float(np.mean(surrogate_widths))


def _haar_detail_energies(series: npt.NDArray[np.float64]) -> list[float]:
    """Haar-DWT detail energies per level, finest -> coarsest (pure numpy)."""

    approx = series.astype(float)
    root_two = float(np.sqrt(2.0))
    energies: list[float] = []
    while approx.size >= 2:
        if approx.size % 2:
            approx = approx[:-1]
        even = approx[0::2]
        odd = approx[1::2]
        detail = (even - odd) / root_two
        energies.append(float(np.sum(np.square(detail))))
        approx = (even + odd) / root_two
    return energies


def estimate_wavelet_entropy(values: Sequence[float]) -> float:
    """Normalized Shannon entropy of the Haar wavelet energy across scales.

    Relative detail energies ``p_j = E_j / sum E_j`` give a scale-energy
    distribution; the normalized Shannon entropy ``-sum p_j ln p_j / ln J``
    is ~1 when energy is spread evenly across scales (noise-like) and ~0 when
    concentrated at a single scale (a dominant oscillation).
    """

    series = _validated_series(values)
    energies = np.asarray(_haar_detail_energies(series), dtype=float)
    total = float(np.sum(energies))
    if total <= 0.0 or energies.size < 2:
        raise ValueError("wavelet entropy requires positive multi-scale energy")
    proportions = energies[energies > 0.0] / total
    entropy = float(-np.sum(proportions * np.log(proportions)))
    return entropy / float(np.log(energies.size))


def estimate_wavelet_energy_ratio(
    values: Sequence[float], *, fine_levels: int = 2
) -> float:
    """Fraction of Haar wavelet energy in the ``fine_levels`` finest scales.

    High values mean short-scale (rough/choppy) energy dominates; low values
    mean the series' energy sits at longer scales (smoother/trending).
    """

    if fine_levels < 1:
        raise ValueError("fine_levels must be >= 1")
    series = _validated_series(values)
    energies = np.asarray(_haar_detail_energies(series), dtype=float)
    total = float(np.sum(energies))
    if total <= 0.0:
        raise ValueError("wavelet energy ratio requires positive energy")
    return float(np.sum(energies[:fine_levels]) / total)


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


def _dfa_scales(observation_count: int) -> tuple[int, ...]:
    """Return stable DFA scales, including a second scale for a 30-point window."""

    maximum = observation_count // 4
    if maximum >= 8:
        return _dyadic_scales(observation_count)
    scales = (4, maximum)
    if maximum <= 4:
        raise ValueError("DFA requires at least two usable scales")
    return scales


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

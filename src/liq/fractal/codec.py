"""Reference 1D compression codec and well-posedness diagnostics.

A slow, correct NumPy reference for the fractal compression codebook: each rolling
window of a return / volatility-proxy series is split into range segments that are
reconstructed from longer, downsampled domain segments via a robust contractive
affine map, and summarised into codebook features. This reference is the
differential-test oracle for any future optimised kernel; it is deliberately
unoptimised. It touches no market-data source and makes no economic claim.

Design defaults where the R&D plan is silent (all configurable via ``CodecParams``):
within-window z-score normalisation; domain downsampling by block-averaging; Huber
robust affine fit; boundary threshold ``tau=0.95``; turnover lag 1; Hill top-fraction
0.1. The search grid (window, range, domain factor, stride, top-K) is a pre-registered
choice made by the experiment layer, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from liq.fractal.contracts import _ALPHA_SAFE_INPUTS, InputDomain
from liq.fractal.reference import estimate_hurst_rs

_FloatArray = npt.NDArray[np.float64]
_MIN_HURST_SAMPLES = 16


@dataclass(frozen=True, slots=True)
class CodecParams:
    """Reproducibility-bearing configuration for the reference codec."""

    window: int
    range_length: int
    domain_factor: int = 2
    stride_domain: int = 1
    top_k: int = 3
    boundary_tau: float = 0.95
    turnover_lag: int = 1
    hill_fraction: float = 0.1
    input_domain: InputDomain = InputDomain.RETURNS
    huber_delta: float = 1.345
    max_iter: int = 25

    def __post_init__(self) -> None:
        if self.input_domain not in _ALPHA_SAFE_INPUTS:
            raise ValueError(
                "codec accepts returns/vol proxies only, never raw-price inputs"
            )
        if self.window <= 0 or self.range_length <= 0:
            raise ValueError("window and range_length must be positive")
        if self.window % self.range_length != 0:
            raise ValueError("window must be an integer multiple of range_length")
        if self.domain_factor not in (2, 4):
            raise ValueError(
                "domain_factor must be 2 or 4 (domain length d = 2r or 4r)"
            )
        if self.domain_length > self.window:
            raise ValueError("domain length must not exceed the window")
        if self.stride_domain < 1:
            raise ValueError("stride_domain must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 < self.boundary_tau <= 2.0:
            raise ValueError("boundary_tau must be in (0, 2]")
        if self.turnover_lag < 1:
            raise ValueError("turnover_lag must be at least 1")
        if not 0.0 < self.hill_fraction < 1.0:
            raise ValueError("hill_fraction must be in (0, 1)")

    @property
    def domain_length(self) -> int:
        """Domain segment length ``d = domain_factor * range_length``."""

        return self.domain_factor * self.range_length


@dataclass(frozen=True, slots=True)
class RangeCode:
    """One range segment's winning affine codeword."""

    domain_index: int
    scale: float  # stored, contractive |scale| < 1
    scale_hat: float  # unconstrained robust slope, for boundary pressure
    offset: float
    match_error: float
    topk_errors: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WindowCodebook:
    """The codebook and per-sample residuals for one encoded window."""

    codes: tuple[RangeCode, ...]
    residuals: tuple[float, ...]
    effective_domain_count: int
    params: CodecParams


@dataclass(frozen=True, slots=True)
class HillResult:
    """Hill tail-index estimate plus its always-available fallback descriptors."""

    value: float
    stable: bool
    fallback: dict[str, float] = field(default_factory=dict)


def encode_window(values: Sequence[float], params: CodecParams) -> WindowCodebook:
    """Encode one window into a codebook via robust contractive affine matches."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or series.size != params.window:
        raise ValueError(
            f"encode_window expects exactly {params.window} values (window length)"
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("codec inputs must be finite")

    normalized = _normalize_within_window(series)
    ranges = normalized.reshape(
        params.window // params.range_length, params.range_length
    )
    domains = _domain_pool(normalized, params)

    codes: list[RangeCode] = []
    residual_samples: list[float] = []
    for target in ranges:
        code, residual = _best_code(target, domains, params)
        codes.append(code)
        residual_samples.extend(float(value) for value in residual)
    return WindowCodebook(
        codes=tuple(codes),
        residuals=tuple(residual_samples),
        effective_domain_count=domains.shape[0],
        params=params,
    )


def normalized_entropy(
    domain_indices: Sequence[int], effective_domain_count: int
) -> float:
    """Shannon entropy of domain-index selections, normalised by effective domains.

    Refuses to return an un-normalised, scale-dependent entropy: the effective
    domain count must be at least 2 and cover every selected index (R&D 16.4).
    """

    indices = np.asarray(domain_indices, dtype=int)
    if effective_domain_count < 2:
        raise ValueError(
            "effective domain count must be at least 2 to normalise entropy"
        )
    if indices.size and int(indices.max(initial=0)) >= effective_domain_count:
        raise ValueError(
            "effective domain count must cover every selected domain index"
        )
    counts = np.bincount(indices, minlength=effective_domain_count).astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(effective_domain_count))


def codeword_identities(
    codebook: WindowCodebook, *, scale_levels: int = 3
) -> frozenset[Any]:
    """Discrete codeword identities: ``(domain_index, scale_bucket)`` tokens.

    Discretisation happens before any Jaccard turnover so set overlap is defined
    on discrete tokens, not continuous codewords (R&D 16.4).
    """

    if scale_levels < 1:
        raise ValueError("scale_levels must be at least 1")
    tokens: set[tuple[int, int]] = set()
    for code in codebook.codes:
        bucket = int(
            np.clip(
                np.floor((code.scale + 1.0) / 2.0 * scale_levels), 0, scale_levels - 1
            )
        )
        tokens.add((code.domain_index, bucket))
    return frozenset(tokens)


def jaccard_turnover(current: frozenset[Any], previous: frozenset[Any]) -> float:
    """Jaccard distance between two discrete codeword-identity sets."""

    for token in (*current, *previous):
        if not isinstance(token, tuple):
            raise ValueError(
                "jaccard turnover requires discrete codeword identities; call "
                "codeword_identities first"
            )
    union = current | previous
    if not union:
        return 0.0
    return 1.0 - len(current & previous) / len(union)


def hill_tail_index(residuals: Sequence[float], *, fraction: float) -> HillResult:
    """Hill tail index on residual magnitudes, with a residual-quantile fallback."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    magnitudes = np.sort(np.abs(np.asarray(residuals, dtype=float)))[::-1]
    positive = magnitudes[magnitudes > 0.0]
    fallback = _residual_tail_fallback(magnitudes)
    order = int(np.floor(fraction * positive.size))
    if order < 2:
        return HillResult(value=float("nan"), stable=False, fallback=fallback)
    top = positive[:order]
    reference = positive[order]
    value = float(np.mean(np.log(top) - np.log(reference)))
    stable = bool(np.isfinite(value) and value > 0.0)
    return HillResult(value=value, stable=stable, fallback=fallback)


def boundary_pressure(scale_hats: Sequence[float] | _FloatArray, tau: float) -> float:
    """Share of unconstrained scales with magnitude at or above ``tau``."""

    if not 0.0 < tau <= 2.0:
        raise ValueError("tau must be in (0, 2]")
    values = np.abs(np.asarray(scale_hats, dtype=float))
    if values.size == 0:
        return 0.0
    return float(np.mean(values >= tau))


def codebook_features(
    codebook: WindowCodebook, *, previous: WindowCodebook | None = None
) -> dict[str, Any]:
    """Compute the Block-D codebook feature panel for one encoded window."""

    scales = np.asarray([code.scale for code in codebook.codes], dtype=float)
    scale_hats = np.asarray([code.scale_hat for code in codebook.codes], dtype=float)
    offsets = np.asarray([code.offset for code in codebook.codes], dtype=float)
    indices = [code.domain_index for code in codebook.codes]
    residuals = np.asarray(codebook.residuals, dtype=float)

    hill = hill_tail_index(codebook.residuals, fraction=codebook.params.hill_fraction)
    turnover: float | None = None
    if previous is not None:
        turnover = jaccard_turnover(
            codeword_identities(codebook), codeword_identities(previous)
        )

    features: dict[str, Any] = {
        "scale_dispersion": float(np.var(scales)),
        "boundary_pressure": boundary_pressure(
            scale_hats, codebook.params.boundary_tau
        ),
        "offset_mean": float(np.mean(offsets)),
        "offset_dispersion": float(np.std(offsets)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residuals))))
        if residuals.size
        else 0.0,
        "residual_tail_index": hill.value if hill.stable else None,
        "residual_tail_fallback": hill.fallback,
        "codebook_entropy": normalized_entropy(
            indices, codebook.effective_domain_count
        ),
        "domain_concentration": _herfindahl(indices, codebook.effective_domain_count),
        "codebook_turnover": turnover,
        "topk_match_margin": _mean_topk_margin(codebook.codes),
        "residual_hurst": (
            estimate_hurst_rs(codebook.residuals)
            if residuals.size >= _MIN_HURST_SAMPLES
            else None
        ),
        "window": codebook.params.window,
        "range_length": codebook.params.range_length,
        "domain_factor": codebook.params.domain_factor,
        "top_k": codebook.params.top_k,
        "input_domain": codebook.params.input_domain.value,
    }
    return features


def _normalize_within_window(series: _FloatArray) -> _FloatArray:
    centered = series - np.mean(series)
    std = float(np.std(centered))
    if std <= 0.0:
        return np.zeros_like(series)
    return centered / std


def _domain_pool(normalized: _FloatArray, params: CodecParams) -> _FloatArray:
    length = params.domain_length
    factor = params.domain_factor
    starts = range(0, params.window - length + 1, params.stride_domain)
    pool = [
        normalized[start : start + length]
        .reshape(params.range_length, factor)
        .mean(axis=1)
        for start in starts
    ]
    if len(pool) < 2:
        raise ValueError(
            "codec requires at least two domain candidates; widen the window"
        )
    return np.asarray(pool, dtype=float)


def _best_code(
    target: _FloatArray, domains: _FloatArray, params: CodecParams
) -> tuple[RangeCode, _FloatArray]:
    errors: list[float] = []
    fits: list[tuple[float, float, float]] = []
    for domain in domains:
        scale_hat, offset = _huber_affine(domain, target, params)
        scale = float(np.clip(scale_hat, -0.999, 0.999))
        residual = target - (scale * domain + offset)
        rms = float(np.sqrt(np.mean(np.square(residual))))
        errors.append(rms)
        fits.append((scale, scale_hat, offset))
    order = np.argsort(errors)
    best = int(order[0])
    scale, scale_hat, offset = fits[best]
    top_indices = order[: params.top_k]
    topk_errors = tuple(float(errors[int(i)]) for i in top_indices)
    if len(topk_errors) < params.top_k:
        topk_errors = topk_errors + (topk_errors[-1],) * (
            params.top_k - len(topk_errors)
        )
    residual = target - (scale * domains[best] + offset)
    return (
        RangeCode(
            domain_index=best,
            scale=scale,
            scale_hat=scale_hat,
            offset=offset,
            match_error=errors[best],
            topk_errors=topk_errors,
        ),
        residual,
    )


def _huber_affine(
    x: _FloatArray, y: _FloatArray, params: CodecParams
) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    for _ in range(params.max_iter):
        residual = y - design @ beta
        scale = _robust_scale(residual)
        if scale <= 0.0:
            break
        weights = _huber_weights(residual / scale, params.huber_delta)
        root = np.sqrt(weights)
        weighted_design = design * root[:, None]
        new_beta, *_ = np.linalg.lstsq(weighted_design, y * root, rcond=None)
        if np.allclose(new_beta, beta, atol=1e-10):
            beta = new_beta
            break
        beta = new_beta
    return float(beta[0]), float(beta[1])


def _robust_scale(residual: _FloatArray) -> float:
    mad = float(np.median(np.abs(residual - np.median(residual))))
    if mad > 0.0:
        return 1.4826 * mad
    return float(np.std(residual))


def _huber_weights(standardized: _FloatArray, delta: float) -> _FloatArray:
    magnitude = np.abs(standardized)
    weights = np.ones_like(magnitude)
    outliers = magnitude > delta
    weights[outliers] = delta / magnitude[outliers]
    return weights


def _residual_tail_fallback(sorted_magnitudes: _FloatArray) -> dict[str, float]:
    if sorted_magnitudes.size == 0:
        return {"p95": 0.0, "p99": 0.0, "exceedance_rate": 0.0}
    p95 = float(np.percentile(sorted_magnitudes, 95))
    p99 = float(np.percentile(sorted_magnitudes, 99))
    exceedance = float(np.mean(sorted_magnitudes > p95))
    return {"p95": p95, "p99": p99, "exceedance_rate": exceedance}


def _herfindahl(indices: Sequence[int], effective_domain_count: int) -> float:
    counts = np.bincount(
        np.asarray(indices, dtype=int), minlength=effective_domain_count
    )
    total = counts.sum()
    if total <= 0:
        return 0.0
    shares = counts.astype(float) / float(total)
    return float(np.sum(np.square(shares)))


def _mean_topk_margin(codes: Sequence[RangeCode]) -> float:
    margins = [code.topk_errors[-1] - code.topk_errors[0] for code in codes]
    return float(np.mean(margins)) if margins else 0.0


__all__ = [
    "CodecParams",
    "HillResult",
    "RangeCode",
    "WindowCodebook",
    "boundary_pressure",
    "codebook_features",
    "codeword_identities",
    "encode_window",
    "hill_tail_index",
    "jaccard_turnover",
    "normalized_entropy",
]

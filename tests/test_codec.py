"""Reference 1D compression codec and its well-posedness diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from liq.fractal import InputDomain
from liq.fractal.codec import (
    CodecParams,
    WindowCodebook,
    boundary_pressure,
    codebook_features,
    codeword_identities,
    encode_window,
    hill_tail_index,
    jaccard_turnover,
    normalized_entropy,
)


def _return_series(count: int) -> list[float]:
    # Deterministic analytic return-like series (bounded, zero-mean-ish), no raw prices.
    return [0.01 * math.sin(i / 5.0) + 0.004 * math.cos(i / 11.0) for i in range(count)]


def _params(
    *,
    window: int = 64,
    range_length: int = 8,
    domain_factor: int = 2,
    stride_domain: int = 4,
    top_k: int = 3,
    input_domain: InputDomain = InputDomain.RETURNS,
) -> CodecParams:
    return CodecParams(
        window=window,
        range_length=range_length,
        domain_factor=domain_factor,
        stride_domain=stride_domain,
        top_k=top_k,
        input_domain=input_domain,
    )


# --- input-domain guard: returns/vol proxies only, never raw prices --------------


def test_params_reject_raw_price_input_domain() -> None:
    with pytest.raises(ValueError, match="raw.price|alpha"):
        _params(input_domain=InputDomain.RAW_PRICE)


@pytest.mark.parametrize(
    "domain",
    [InputDomain.RETURNS, InputDomain.LOG_RETURNS, InputDomain.VOLATILITY_PROXY],
)
def test_params_accept_alpha_safe_domains(domain: InputDomain) -> None:
    assert _params(input_domain=domain).input_domain is domain


def test_params_reject_incoherent_geometry() -> None:
    with pytest.raises(ValueError):
        _params(window=64, range_length=7)  # window not divisible by range
    with pytest.raises(ValueError):
        _params(domain_factor=3)  # d must be 2r or 4r


# --- encode ---------------------------------------------------------------------


def test_encode_window_produces_a_codebook_with_metadata() -> None:
    params = _params()
    codebook = encode_window(_return_series(params.window), params)

    assert isinstance(codebook, WindowCodebook)
    assert len(codebook.codes) == params.window // params.range_length
    assert codebook.effective_domain_count >= 2
    # Contractivity enforced on the stored scale; the unconstrained s_hat is retained.
    for code in codebook.codes:
        assert abs(code.scale) < 1.0
        assert math.isfinite(code.scale_hat)
        assert 0 <= code.domain_index < codebook.effective_domain_count
        assert len(code.topk_errors) == params.top_k
    # Reproducibility metadata is attached.
    assert codebook.params == params


def test_encode_is_deterministic() -> None:
    params = _params()
    series = _return_series(params.window)
    a = encode_window(series, params)
    b = encode_window(series, params)
    assert [c.domain_index for c in a.codes] == [c.domain_index for c in b.codes]
    assert [c.scale for c in a.codes] == [c.scale for c in b.codes]


def test_encode_rejects_wrong_length_window() -> None:
    params = _params()
    with pytest.raises(ValueError, match="length"):
        encode_window(_return_series(params.window - 1), params)


def test_encode_rejects_non_finite_inputs() -> None:
    params = _params()
    series = _return_series(params.window)
    series[3] = math.inf
    with pytest.raises(ValueError, match="finite"):
        encode_window(series, params)


# --- well-posedness diagnostic 1: entropy normalized by effective domain count ---


def test_normalized_entropy_is_in_unit_interval() -> None:
    indices = [0, 1, 2, 3, 0, 1, 2, 3]
    value = normalized_entropy(indices, effective_domain_count=4)
    assert 0.0 <= value <= 1.0
    # Uniform over the effective domains → maximal normalized entropy.
    assert value == pytest.approx(1.0, abs=1e-9)


def test_entropy_rejects_absent_effective_domain_normalization() -> None:
    # Refusing to normalize (effective domain count < 2) is the guard against a
    # scale-dependent raw entropy leaking in as a feature (R&D 16.4).
    with pytest.raises(ValueError, match="effective domain"):
        normalized_entropy([0, 0, 0], effective_domain_count=1)


def test_entropy_rejects_more_selections_than_available_domains() -> None:
    with pytest.raises(ValueError, match="effective domain"):
        normalized_entropy([0, 1, 2, 3], effective_domain_count=2)


# --- well-posedness diagnostic 2: discrete codeword identity before Jaccard ------


def test_codeword_identities_are_discrete_tokens() -> None:
    codebook = encode_window(_return_series(64), _params())
    identities = codeword_identities(codebook)
    assert isinstance(identities, frozenset)
    assert all(isinstance(token, tuple) for token in identities)


def test_jaccard_turnover_requires_discrete_identities() -> None:
    # Passing continuous codewords (floats) instead of discrete identities is refused.
    with pytest.raises(ValueError, match="discrete codeword"):
        jaccard_turnover(frozenset({0.5, 0.9}), frozenset({0.5}))  # type: ignore[arg-type]


def test_jaccard_turnover_on_discrete_sets() -> None:
    a = frozenset({(0, 1), (2, 0)})
    b = frozenset({(0, 1)})
    # |A ∩ B| = 1, |A ∪ B| = 2 → turnover = 1 - 1/2 = 0.5
    assert jaccard_turnover(a, b) == pytest.approx(0.5)
    assert jaccard_turnover(a, a) == pytest.approx(0.0)


# --- well-posedness diagnostic 4: Hill stabilization / fallback ------------------


def test_hill_tail_index_reports_stability_and_fallback() -> None:
    residuals = [0.1 * (i % 7) + 0.001 * i for i in range(120)]
    result = hill_tail_index(residuals, fraction=0.1)
    assert math.isfinite(result.value) or result.fallback is not None
    assert isinstance(result.stable, bool)
    # The fallback tail descriptors are always available.
    assert set(result.fallback) >= {"p95", "p99", "exceedance_rate"}


def test_hill_falls_back_when_too_few_tail_samples() -> None:
    result = hill_tail_index([0.0, 0.0, 0.0, 0.0], fraction=0.1)
    assert result.stable is False


# --- well-posedness diagnostic 5: boundary pressure from unconstrained s_hat -----


def test_boundary_pressure_uses_unconstrained_scale() -> None:
    # Two of four scales sit at/above the boundary threshold.
    pressure = boundary_pressure([0.2, 0.98, 1.4, -1.2], tau=0.95)
    assert pressure == pytest.approx(0.75)


def test_boundary_pressure_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        boundary_pressure([0.2, 0.9], tau=0.0)


# --- Block-D feature panel ------------------------------------------------------


def test_codebook_features_cover_the_block_d_panel() -> None:
    params = _params()
    series = _return_series(params.window)
    codebook = encode_window(series, params)
    previous = encode_window([v * 1.3 for v in series], params)

    features = codebook_features(codebook, previous=previous)

    expected = {
        "scale_dispersion",
        "boundary_pressure",
        "offset_mean",
        "offset_dispersion",
        "residual_rms",
        "residual_tail_index",
        "codebook_entropy",
        "domain_concentration",
        "codebook_turnover",
        "topk_match_margin",
        "residual_hurst",
    }
    assert expected.issubset(features)
    assert 0.0 <= features["codebook_entropy"] <= 1.0
    assert 0.0 <= features["boundary_pressure"] <= 1.0
    assert 0.0 <= features["codebook_turnover"] <= 1.0
    assert features["residual_rms"] >= 0.0
    # Metadata for reproducibility travels with the panel.
    assert features["window"] == params.window
    assert features["range_length"] == params.range_length


def test_codebook_features_without_previous_have_no_turnover() -> None:
    codebook = encode_window(_return_series(64), _params())
    features = codebook_features(codebook, previous=None)
    assert features["codebook_turnover"] is None


# --- parameter validation branches ---------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: CodecParams(window=0, range_length=8),
        lambda: CodecParams(window=8, range_length=8, domain_factor=2),  # d=16 > window
        lambda: CodecParams(window=64, range_length=8, stride_domain=0),
        lambda: CodecParams(window=64, range_length=8, top_k=0),
        lambda: CodecParams(window=64, range_length=8, boundary_tau=3.0),
        lambda: CodecParams(window=64, range_length=8, turnover_lag=0),
        lambda: CodecParams(window=64, range_length=8, hill_fraction=0.0),
        lambda: CodecParams(window=64, range_length=8, hill_fraction=1.0),
    ],
)
def test_params_reject_invalid_values(make: Callable[[], CodecParams]) -> None:
    with pytest.raises(ValueError):
        make()


def test_domain_length_matches_factor() -> None:
    assert _params(domain_factor=4, range_length=8, window=64).domain_length == 32


# --- degenerate / edge encode paths --------------------------------------------


def test_encode_constant_series_yields_zero_residuals() -> None:
    params = _params()
    codebook = encode_window([0.0] * params.window, params)
    assert all(abs(value) < 1e-9 for value in codebook.residuals)


def test_encode_rejects_too_few_domain_candidates() -> None:
    # window 16, range 8, d=16, stride 8 → a single domain candidate.
    params = CodecParams(window=16, range_length=8, domain_factor=2, stride_domain=8)
    with pytest.raises(ValueError, match="two domain candidates"):
        encode_window(_return_series(16), params)


def test_topk_errors_pad_when_domains_are_scarce() -> None:
    # 2 domains but top_k=3 → the margin list is padded to length top_k.
    params = CodecParams(
        window=24, range_length=8, domain_factor=2, stride_domain=8, top_k=3
    )
    codebook = encode_window(_return_series(24), params)
    assert all(len(code.topk_errors) == 3 for code in codebook.codes)


def test_normalized_entropy_of_empty_selection_is_zero() -> None:
    assert normalized_entropy([], effective_domain_count=4) == 0.0

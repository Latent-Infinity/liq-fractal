from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

fractal_api = importlib.import_module("liq.fractal")
FractalDescriptorSpec = fractal_api.FractalDescriptorSpec
FractalEstimate = fractal_api.FractalEstimate
InputDomain = fractal_api.InputDomain
KernelBackend = fractal_api.KernelBackend


def test_public_api_exports_contract_types() -> None:
    assert InputDomain.RETURNS.value == "returns"
    assert KernelBackend.PYTHON_REFERENCE.value == "python_reference"
    assert FractalDescriptorSpec.__name__ == "FractalDescriptorSpec"
    assert FractalEstimate.__name__ == "FractalEstimate"


def test_descriptor_spec_exposes_read_only_metadata() -> None:
    spec = FractalDescriptorSpec(
        estimator="dfa",
        input_domain=InputDomain.RETURNS,
        window=120,
        sampling="1m",
        detrending="linear",
    )

    metadata = spec.metadata()

    assert metadata["estimator"] == "dfa"
    assert metadata["input_domain"] == "returns"
    assert metadata["window"] == 120
    assert metadata["alpha_safe"] is True
    with pytest.raises(AttributeError):
        object.__getattribute__(metadata, "pop")


def test_raw_price_inputs_are_not_alpha_safe() -> None:
    spec = FractalDescriptorSpec(
        estimator="hurst",
        input_domain=InputDomain.RAW_PRICE,
        window=60,
        sampling="5m",
    )

    assert spec.is_alpha_safe is False


def test_descriptor_spec_rejects_invalid_identity_fields() -> None:
    with pytest.raises(ValueError, match="estimator"):
        FractalDescriptorSpec("", InputDomain.RETURNS, 60, "1m")
    with pytest.raises(ValueError, match="window"):
        FractalDescriptorSpec("dfa", InputDomain.RETURNS, 0, "1m")
    with pytest.raises(ValueError, match="sampling"):
        FractalDescriptorSpec("dfa", InputDomain.RETURNS, 60, "")


def test_estimate_requires_alpha_safe_input_and_causal_availability() -> None:
    safe_spec = FractalDescriptorSpec(
        estimator="dfa",
        input_domain=InputDomain.VOLATILITY_PROXY,
        window=120,
        sampling="1m",
    )
    estimate = FractalEstimate(
        spec=safe_spec,
        value=0.42,
        feature_timestamp_ns=1_700_000_000_000_000_000,
        feature_available_at_ns=1_700_000_060_000_000_000,
    )

    assert estimate.metadata()["feature_available_at_ns"] == 1_700_000_060_000_000_000

    raw_price_spec = FractalDescriptorSpec(
        estimator="hurst",
        input_domain=InputDomain.RAW_PRICE,
        window=120,
        sampling="1m",
    )
    with pytest.raises(ValueError, match="raw-price"):
        FractalEstimate(raw_price_spec, 0.5, 10, 11)
    with pytest.raises(ValueError, match="feature_available_at_ns"):
        FractalEstimate(safe_spec, 0.5, 11, 10)

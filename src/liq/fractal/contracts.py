"""Provider-neutral contracts for fractal descriptors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

MetadataValue = str | int | bool | float | None


class InputDomain(StrEnum):
    """Supported input domains for alpha-facing fractal estimators."""

    RETURNS = "returns"
    LOG_RETURNS = "log_returns"
    VOLATILITY_PROXY = "volatility_proxy"
    RAW_PRICE = "raw_price"


class KernelBackend(StrEnum):
    """Execution backends for a fractal estimator implementation."""

    PYTHON_REFERENCE = "python_reference"
    RUST_OPTIONAL = "rust_optional"


_ALPHA_SAFE_INPUTS = frozenset(
    {InputDomain.RETURNS, InputDomain.LOG_RETURNS, InputDomain.VOLATILITY_PROXY}
)


@dataclass(frozen=True, slots=True)
class FractalDescriptorSpec:
    """Immutable descriptor metadata shared by features and validation."""

    estimator: str
    input_domain: InputDomain
    window: int
    sampling: str
    detrending: str | None = None
    backend: KernelBackend = KernelBackend.PYTHON_REFERENCE

    def __post_init__(self) -> None:
        if not self.estimator:
            raise ValueError("estimator must be non-empty")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if not self.sampling:
            raise ValueError("sampling must be non-empty")

    @property
    def is_alpha_safe(self) -> bool:
        """Whether this descriptor avoids raw-price alpha-facing inputs."""

        return self.input_domain in _ALPHA_SAFE_INPUTS

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return a read-only metadata mapping for downstream artifacts."""

        return MappingProxyType(
            {
                "estimator": self.estimator,
                "input_domain": self.input_domain.value,
                "window": self.window,
                "sampling": self.sampling,
                "detrending": self.detrending,
                "backend": self.backend.value,
                "alpha_safe": self.is_alpha_safe,
            }
        )


@dataclass(frozen=True, slots=True)
class FractalEstimate:
    """Single timestamped fractal descriptor value plus metadata."""

    spec: FractalDescriptorSpec
    value: float
    feature_timestamp_ns: int
    feature_available_at_ns: int

    def __post_init__(self) -> None:
        if not self.spec.is_alpha_safe:
            raise ValueError(
                "alpha-facing fractal estimates cannot use raw-price inputs"
            )
        if self.feature_available_at_ns < self.feature_timestamp_ns:
            raise ValueError("feature_available_at_ns must be >= feature_timestamp_ns")

    def metadata(self) -> Mapping[str, MetadataValue]:
        """Return estimate metadata for feature artifacts."""

        metadata: dict[str, MetadataValue] = dict(self.spec.metadata())
        metadata["value"] = self.value
        metadata["feature_timestamp_ns"] = self.feature_timestamp_ns
        metadata["feature_available_at_ns"] = self.feature_available_at_ns
        return MappingProxyType(metadata)

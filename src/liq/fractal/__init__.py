"""Fractal-analysis contracts and reference helpers for the LIQ Stack."""

from liq.fractal.contracts import (
    FractalDescriptorSpec,
    FractalEstimate,
    InputDomain,
    KernelBackend,
)
from liq.fractal.reference import estimate_dfa, estimate_ghe, estimate_hurst_rs

__all__ = [
    "FractalDescriptorSpec",
    "FractalEstimate",
    "InputDomain",
    "KernelBackend",
    "estimate_dfa",
    "estimate_ghe",
    "estimate_hurst_rs",
]

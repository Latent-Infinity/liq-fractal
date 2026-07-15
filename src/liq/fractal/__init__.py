"""Fractal-analysis contracts and reference helpers for the LIQ Stack."""

from liq.fractal.codec import (
    CodecParams,
    HillResult,
    RangeCode,
    WindowCodebook,
    boundary_pressure,
    codebook_features,
    codeword_identities,
    encode_window,
    hill_tail_index,
    jaccard_turnover,
    normalized_entropy,
)
from liq.fractal.contracts import (
    FractalDescriptorSpec,
    FractalEstimate,
    InputDomain,
    KernelBackend,
)
from liq.fractal.reference import estimate_dfa, estimate_ghe, estimate_hurst_rs

__all__ = [
    "CodecParams",
    "FractalDescriptorSpec",
    "FractalEstimate",
    "HillResult",
    "InputDomain",
    "KernelBackend",
    "RangeCode",
    "WindowCodebook",
    "boundary_pressure",
    "codebook_features",
    "codeword_identities",
    "encode_window",
    "estimate_dfa",
    "estimate_ghe",
    "estimate_hurst_rs",
    "hill_tail_index",
    "jaccard_turnover",
    "normalized_entropy",
]

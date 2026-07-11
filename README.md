# liq-fractal

Fractal-analysis contracts and reference implementations for the Latent Infinity Quant (LIQ) Stack.

`liq-fractal` owns fractal-domain logic for LIQ. It is Python-first for research velocity and API stability, while keeping the public surface narrow enough to add Rust-backed kernels later without changing downstream callers.

## Scope

`liq-fractal` provides:

- provider-neutral fractal descriptor contracts;
- estimator metadata and availability policies for alpha-facing features;
- reference implementations used by `liq-features` and `liq-validation`;
- a stable place for future optimized kernels, including optional Rust/Python bridges once bottlenecks are proven.

It does **not** fetch market data, run experiments, size positions, simulate fills, or emit trading decisions.

## Installation

```bash
uv pip install liq-fractal
```

For local development:

```bash
uv sync --group dev
uv run pytest
```

## Quick start

```python
from liq.fractal import FractalDescriptorSpec, InputDomain

spec = FractalDescriptorSpec(
    estimator="dfa",
    input_domain=InputDomain.RETURNS,
    window=120,
    sampling="1m",
    detrending="linear",
)

assert spec.is_alpha_safe
```

## Design notes

The package starts with pure Python contracts and reference code because the current FCE plan needs correctness, traceability, and integration before performance optimization. When a descriptor or codec kernel becomes stable and hot, it can move behind the same API as an optional Rust implementation, similar to `liq-ta`.

## Development

```bash
uv run pytest --cov=liq.fractal --cov-report=term-missing
uv run ruff check src tests
uv run ruff format src tests
uv run ty check src tests
```

## License

MIT

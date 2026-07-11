# Contributing to liq-fractal

Thanks for contributing to the LIQ Stack. `liq-fractal` is an open-source library, but it participates in research workflows with strict reproducibility and data-integrity requirements.

## Development setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run ty check src tests
```

Use `uv`, `ruff`, `ty`, and `pytest`. Do not introduce pip-managed virtual environments, flake8, black, isort, or mypy.

## Scope rules

- Keep fractal estimator logic in `liq-fractal`.
- Keep market-data loading in `liq-data`.
- Keep feature pipeline orchestration in `liq-features`.
- Keep validation harnesses and statistical controls in `liq-validation`.
- Keep experiment manifests and runs in `liq-experiments`.

## Data and research integrity

- Do not add fabricated market, OHLCV, order-book, trade, quote, signal, or regime data.
- Do not read lockbox-period data from this library.
- Do not emit trading decisions, PnL, promotion verdicts, or PASS/FAIL research language from this package.
- Keep timestamps timezone-aware and normalized to UTC at storage/interface boundaries.

## Code standards

- Prefer small, typed, deterministic functions.
- Add tests before implementations.
- Maintain at least 90% coverage on touched code.
- Keep public APIs provider-neutral and array/Polars-friendly so Rust kernels can be added later behind the same interface.

## Pull requests

Please include:

1. What changed and why.
2. Tests and verification commands run.
3. Any compatibility or downstream dependency impact.
4. Any research-integrity considerations.

"""Architecture-fitness guard: fractal-domain modules stay provider-agnostic.

`liq-fractal` holds pure fractal-domain contracts and reference kernels. Provider
SDKs, broker clients, and provider data/execution boundaries may only be imported
inside designated adapter modules (none exist yet). This guard fails if any
domain module imports one, keeping the dependency direction pointing inward:
`liq-features`/`liq-validation` consume `liq-fractal`, never the reverse, and
`liq-fractal` never reaches out to data sourcing, storage, or brokers.

The consuming FCE surfaces in `liq-features` and `liq-validation` get their own
boundary guards in their own repos as those surfaces land.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "liq" / "fractal"

# External provider SDKs / broker clients that fractal-domain code must not import.
FORBIDDEN_EXTERNAL_ROOTS = frozenset(
    {
        "tradestation",
        "oanda",
        "coinbase",
        "binance",
        "databento",
        "ib_insync",
        "ibapi",
        "alpaca",
        "polygon",
        "ccxt",
    }
)

# LIQ subpackages that are data-sourcing / storage / broker-execution boundaries.
# Fractal-domain code depends inward only (self, and shared domain in liq.core).
FORBIDDEN_LIQ_SUBPACKAGES = frozenset(
    {"data", "live", "store", "sim", "runner", "experiments", "strategies", "scan"}
)


def _is_adapter_module(path: Path) -> bool:
    """Adapter modules are the only place provider imports are ever allowed."""
    parts = {part.lower() for part in path.parts}
    return "adapters" in parts or path.stem.endswith("_adapter")


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def _is_forbidden(module: str) -> bool:
    parts = module.split(".")
    root = parts[0]
    if root in FORBIDDEN_EXTERNAL_ROOTS:
        return True
    if root == "liq":
        sub = parts[1] if len(parts) > 1 else ""
        return sub in FORBIDDEN_LIQ_SUBPACKAGES
    return False


def test_fractal_domain_modules_do_not_import_providers() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if _is_adapter_module(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden = sorted(m for m in _imported_modules(tree) if _is_forbidden(m))
        if forbidden:
            offenders.append(f"{path.relative_to(SRC)}: {forbidden}")
    assert not offenders, "provider imports in fractal-domain modules: " + "; ".join(
        offenders
    )


def test_boundary_detector_is_not_vacuous() -> None:
    """The guard must actually catch forbidden imports, external and cross-library."""
    sample = "import oanda\nfrom liq.data.loader import load\nfrom liq.fractal.contracts import x\n"
    imported = _imported_modules(ast.parse(sample))
    forbidden = {m for m in imported if _is_forbidden(m)}

    assert forbidden == {"oanda", "liq.data.loader"}
    assert not _is_forbidden("liq.fractal.contracts")
    assert not _is_forbidden("numpy")

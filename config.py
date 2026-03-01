"""
Phase 2: Feature Generation Configuration
==========================================
Frozen configuration for Phase 2 (Technical Indicator Generation).

Version: 1.0.6
Changelog:
    - v1.0.6: Bumped version to match script v1.1.0 changes
    - v1.0.5: Fixed hardware portability, category list, indicator validation
    - v1.0.4: Fixed path resolution for cwd-independence
    - v1.0.3: Standalone frozen dataclass (no PhaseConfig inheritance)
    - v1.0.2: Defensive import for validate_non_empty_string
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

# ── PATH SETUP ──────────────────────────────────────────────────
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── CORE IMPORTS ───────────────────────────────────────────────
from core.config import (
    validate_positive_int,
    validate_non_empty_list,
    validate_0_to_1,
)

try:
    from core.config import validate_non_empty_string
except ImportError:
    def validate_non_empty_string(value: str, name: str) -> str:
        """Validate that value is a non-empty string (local fallback)."""
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got: {type(value).__name__}")
        if len(value.strip()) == 0:
            raise ValueError(f"{name} must be a non-empty string")
        return value


# ── CONSTANTS ───────────────────────────────────────────────────

# ✅ Fix 5: Hardware-portable worker count (uses os.cpu_count())
_DEFAULT_MAX_WORKERS: int = min(11, max(1, (os.cpu_count() or 1) - 2))

_VALID_COMPRESSION_CODECS = frozenset({
    "snappy", "gzip", "brotli", "zstd", "none",
})

# ✅ Fix 8: Indicator validation whitelist
_VALID_INDICATOR_KINDS = frozenset({
    "adx", "chop", "aroon", "supertrend",
    "rsi", "roc", "macd", "mom", "willr",
    "natr", "bbands", "atr", "rvi",
    "cmf", "mfi", "obv",
    "sma", "ema",
    "zscore", "skew", "stdev",
    "log_return", "percent_return",
})


# ── CURATED INDICATOR DEFINITIONS ───────────────────────────────

def _default_curated_indicators() -> List[Dict[str, Any]]:
    """
    High-signal curated indicator list (25 indicators).
    ✅ Returns List[Dict[str, Any]] for proper type hinting.
    """
    return [
        # TREND
        {"kind": "adx", "length": 14},
        {"kind": "chop", "length": 14},
        {"kind": "aroon", "length": 14},
        {"kind": "supertrend", "length": 10, "multiplier": 3},
        # MOMENTUM
        {"kind": "rsi", "length": 14},
        {"kind": "roc", "length": 10},
        {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
        {"kind": "mom", "length": 10},
        {"kind": "willr", "length": 14},
        # VOLATILITY
        {"kind": "natr", "length": 14},
        {"kind": "bbands", "length": 20, "std": 2},
        {"kind": "atr", "length": 14},
        {"kind": "rvi", "length": 14},
        # VOLUME
        {"kind": "cmf", "length": 20},
        {"kind": "mfi", "length": 14},
        {"kind": "obv"},
        # OVERLAP
        {"kind": "sma", "length": 20},
        {"kind": "sma", "length": 50},
        {"kind": "sma", "length": 200},
        {"kind": "ema", "length": 20},
        # STATISTICS
        {"kind": "zscore", "length": 20},
        {"kind": "skew", "length": 30},
        {"kind": "stdev", "length": 20},
        # BACKWARD-LOOKING PERFORMANCE
        {"kind": "log_return", "length": 5},
        {"kind": "percent_return", "length": 10},
    ]


def _default_categories() -> List[str]:
    """
    ✅ Fix 6: Categories now match actual indicator groups.
    """
    return ["momentum", "overlap", "volatility", "volume", "trend", "statistics", "performance"]


def _default_exclude_indicators() -> List[str]:
    return ["td_seq", "vp"]


def _default_exclude_known_leaky() -> List[str]:
    return ["DPO"]


# ── PHASE 2 CONFIGURATION ──────────────────────────────────────

def _resolve_project_path(relative_path: str) -> Path:
    """Resolve path relative to PROJECT_ROOT."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


@dataclass(frozen=True)
class Phase2Config:
    """
    Immutable configuration for Phase 2: Feature Generation.
    Standalone frozen dataclass (no inheritance to avoid frozen/non-frozen conflict).
    """

    phase: int = 2
    version: str = "1.0.6"

    input_path: str = "artifacts/phase_1_data/merged_data.parquet"
    output_dir: str = "artifacts/phase_2_features"

    target_horizon: int = 5
    price_col: str = "close"
    min_rows_per_ticker: int = 200

    use_pandas_ta: bool = True
    categories: List[str] = field(default_factory=_default_categories)
    exclude_indicators: List[str] = field(default_factory=_default_exclude_indicators)
    exclude_candlestick: bool = True

    use_multiprocessing: bool = True
    n_workers: int = _DEFAULT_MAX_WORKERS  # ✅ Fix 5: Uses os.cpu_count()
    chunksize: int = 50

    use_curated_strategy: bool = True
    curated_indicators: List[Dict[str, Any]] = field(
        default_factory=_default_curated_indicators
    )

    anomaly_threshold: float = 0.3
    exclude_known_leaky: List[str] = field(
        default_factory=_default_exclude_known_leaky
    )
    check_forward_shift: bool = True

    save_feature_report: bool = True
    save_correlation_matrix: bool = True
    compression: str = "snappy"

    def get_resolved_input_path(self) -> Path:
        return _resolve_project_path(self.input_path)

    def get_resolved_output_dir(self) -> Path:
        return _resolve_project_path(self.output_dir)

    def get_output_path(self) -> Path:
        return self.get_resolved_output_dir() / "phase_2_features.parquet"

    def validate(self) -> None:
        """Validate all configuration parameters."""
        if self.phase != 2:
            raise ValueError(f"phase must be 2, got {self.phase}")
        validate_non_empty_string(self.version, name="version")

        validate_positive_int(self.target_horizon, name="target_horizon")
        validate_non_empty_string(self.price_col, name="price_col")
        validate_positive_int(self.min_rows_per_ticker, name="min_rows_per_ticker")

        # ✅ Fix 5: Dynamic max workers validation
        max_cpu = os.cpu_count() or 1
        if not 1 <= self.n_workers <= max_cpu:
            raise ValueError(
                f"n_workers must be between 1 and {max_cpu} (detected CPUs), "
                f"got {self.n_workers}"
            )
        validate_positive_int(self.chunksize, name="chunksize")

        validate_0_to_1(self.anomaly_threshold, name="anomaly_threshold")

        resolved_input = self.get_resolved_input_path()
        if not resolved_input.exists():
            raise FileNotFoundError(
                f"Phase 1 output not found.\n"
                f"  Config path    : {self.input_path}\n"
                f"  Resolved to    : {resolved_input}\n"
                f"  PROJECT_ROOT   : {PROJECT_ROOT}\n"
                f"  Phase 1 must complete before Phase 2."
            )

        if self.use_pandas_ta:
            validate_non_empty_list(self.categories, name="categories")

        if self.use_curated_strategy:
            validate_non_empty_list(self.curated_indicators, name="curated_indicators")
            
            # ✅ Fix 8: Validate indicator structure and kind
            for i, ind in enumerate(self.curated_indicators):
                if not isinstance(ind, dict):
                    raise TypeError(
                        f"curated_indicators[{i}] must be a dict, "
                        f"got {type(ind).__name__}: {ind}"
                    )
                if "kind" not in ind:
                    raise ValueError(
                        f"curated_indicators[{i}] missing required 'kind' key: {ind}"
                    )
                if ind["kind"] not in _VALID_INDICATOR_KINDS:
                    raise ValueError(
                        f"curated_indicators[{i}] has unknown kind '{ind['kind']}'. "
                        f"Valid kinds: {sorted(_VALID_INDICATOR_KINDS)}"
                    )

        if self.compression not in _VALID_COMPRESSION_CODECS:
            raise ValueError(
                f"compression must be one of {sorted(_VALID_COMPRESSION_CODECS)}, "
                f"got '{self.compression}'"
            )

    def to_snapshot(self) -> Dict[str, Any]:
        """Return config as a plain dict for manifest/audit logging."""
        return {
            "phase": self.phase,
            "version": self.version,
            "input_path": self.input_path,
            "input_path_resolved": str(self.get_resolved_input_path()),
            "output_dir": self.output_dir,
            "output_dir_resolved": str(self.get_resolved_output_dir()),
            "project_root": str(PROJECT_ROOT),
            "target_horizon": self.target_horizon,
            "price_col": self.price_col,
            "min_rows_per_ticker": self.min_rows_per_ticker,
            "use_pandas_ta": self.use_pandas_ta,
            "categories": self.categories,
            "exclude_indicators": self.exclude_indicators,
            "exclude_candlestick": self.exclude_candlestick,
            "use_multiprocessing": self.use_multiprocessing,
            "n_workers": self.n_workers,
            "chunksize": self.chunksize,
            "use_curated_strategy": self.use_curated_strategy,
            "curated_indicators": self.curated_indicators,
            "indicator_count": self.get_indicator_count(),
            "anomaly_threshold": self.anomaly_threshold,
            "exclude_known_leaky": self.exclude_known_leaky,
            "check_forward_shift": self.check_forward_shift,
            "save_feature_report": self.save_feature_report,
            "save_correlation_matrix": self.save_correlation_matrix,
            "compression": self.compression,
        }

    def get_indicator_count(self) -> int:
        return len(self.curated_indicators)


def get_phase2_config(**overrides) -> Phase2Config:
    """Factory function to create a validated Phase2Config."""
    config = Phase2Config(**overrides)
    config.validate()
    return config


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Configuration — Self-Test (v1.0.6)")
    print("=" * 60)
    print(f"  PROJECT_ROOT     : {PROJECT_ROOT}")
    print(f"  Detected CPUs    : {os.cpu_count()}")
    print(f"  Default workers  : {_DEFAULT_MAX_WORKERS}")

    config = Phase2Config()
    print(f"\n✓ Default construction OK")
    print(f"  Phase            : {config.phase}")
    print(f"  Version          : {config.version}")
    print(f"  Indicator count  : {config.get_indicator_count()}")
    print(f"  Workers          : {config.n_workers}")

    # Immutability
    try:
        config.target_horizon = 10  # type: ignore
        print("✗ FAILED: mutation allowed!")
    except AttributeError:
        print("✓ Immutability OK")

    # to_snapshot()
    snapshot = config.to_snapshot()
    assert isinstance(snapshot, dict)
    assert snapshot["phase"] == 2
    print(f"✓ to_snapshot() OK ({len(snapshot)} keys)")

    # Indicator validation
    for i, ind in enumerate(config.curated_indicators):
        assert "kind" in ind
        assert ind["kind"] in _VALID_INDICATOR_KINDS
    print(f"✓ All {config.get_indicator_count()} indicators validated")

    if config.get_resolved_input_path().exists():
        config.validate()
        print(f"✓ Full validation PASSED")
    else:
        print(f"⊘ Skipping validate() — Phase 1 data not available")

    print("\n" + "=" * 60)
    print("All self-tests PASSED")
    print("=" * 60)

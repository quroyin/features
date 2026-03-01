"""
Phase 2 — Stage 0: Preselection Audit
======================================
Leakage detection and redundancy pre-filtering BEFORE feature selection.

Pipeline position:
    Phase 1 (Cleaned OHLCV) → [Stage 0: Preselection Audit] → Stage 1 (Individual Evaluation)

Version: 1.2.0
Changelog:
    - v1.2.0: Fixed MFI FutureWarning at the ROOT CAUSE — monkey-patches
              pandas_ta_classic.volume.mfi.mfi() to initialise "+mf"/"-mf"
              columns as float64 instead of int64, both in the main process
              and in each loky worker. No more warning suppression.
    - v1.1.0: Fixed MFI FutureWarning leak in loky workers (removed catch_warnings,
              set filter directly), fixed float equality in rep selection (use sorted),
              bumped config version to 1.0.6
    - v1.0.9: Clean terminal formatting (print utilities), silenced MFI FutureWarning
              in worker via catch_warnings context manager, verbose=0 for joblib
    - v1.0.8: Replaced sequential tqdm loop with joblib Parallel/delayed (loky backend),
              extracted filter_tickers_by_history() and _process_ticker_worker() to
              module level for pickling compatibility
    - v1.0.7: Fixed alphabetical bias in redundancy clustering (select rep by |ρ| vs target),
              removed dead exclude_candlestick block, fixed n_tickers to use nunique()
    - v1.0.6: Fixed pandas_ta FutureWarning — now suppressed BEFORE library execution
    - v1.0.5: Fixed version mismatch, removed unused imports, moved hash to module level
    - v1.0.4: Fixed pandas FutureWarning, magic numbers, redundant checks
    - v1.0.3: Fixed "daemonic processes" crash — sequential indicator generation
    - v1.0.2: cwd-independent path resolution
    - v1.0.1: Fixed cascading ImportError
    - v1.0.0: Initial implementation
"""

import hashlib
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# FIX PANDAS_TA UPSTREAM BUG (BEFORE IMPORT)
# ═══════════════════════════════════════════════════════════════
# pandas_ta_classic's MFI indicator initialises the "-mf" column as
# int64 and then assigns float64 money-flow values into it, which
# raises a FutureWarning in pandas 2.0+ (and will become an error in
# a future release).  We monkey-patch the function at import time so
# the column is initialised as float64 from the start — no suppression
# needed.

import pandas_ta_classic as ta  # noqa: E402

def _patched_mfi(high, low, close, volume, length=None, drift=None, offset=None, **kwargs):
    """
    Monkey-patched MFI that initialises money-flow columns as float64
    to avoid the pandas 2.0+ FutureWarning raised by the original.
    """
    import pandas_ta_classic.volume.mfi as _mfi_mod
    import numpy as np

    # Re-use the original helper utilities from the module
    _get = _mfi_mod.get
    _hlc3 = _mfi_mod.hlc3
    _non_zero_range = _mfi_mod.non_zero_range
    _unsigned_differences = _mfi_mod.unsigned_differences
    _verify_series = _mfi_mod.verify_series
    _signals = _mfi_mod.signals

    # --- begin: adapted from original mfi() with dtype fix applied ---
    length = int(length) if length and length > 0 else 14
    drift = _get(kwargs, "drift", 1)
    _drift = int(drift) if drift and drift > 0 else 1

    high = _verify_series(high, length)
    low = _verify_series(low, length)
    close = _verify_series(close, length)
    volume = _verify_series(volume, length)
    if high is None or low is None or close is None or volume is None:
        return

    offset = int(offset) if offset else 0

    m = close.size
    df = pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df["hlc3"] = _hlc3(df["high"], df["low"], df["close"])
    df["diff"] = df["hlc3"].diff(_drift).fillna(0)

    raw_money_flow = df["hlc3"] * df["volume"]

    # ✅ FIX: initialise as float64 so the masked assignment never triggers
    # the incompatible-dtype FutureWarning.
    df["+mf"] = np.zeros(m, dtype=np.float64)
    df["-mf"] = np.zeros(m, dtype=np.float64)

    df.loc[df["diff"] > 0, "+mf"] = raw_money_flow
    df.loc[df["diff"] == -1, "-mf"] = raw_money_flow  # original logic preserved

    psum = df["+mf"].rolling(length).sum()
    nsum = df["-mf"].rolling(length).sum()
    nsum_nonzero = nsum.copy()
    nsum_nonzero[nsum_nonzero == 0] = 1  # avoid division by zero

    mfr = psum / nsum_nonzero
    result = 100 * psum / (psum + nsum_nonzero)
    result.name = f"MFI_{length}"

    result = result.iloc[offset:] if offset > 0 else result
    result.name = f"MFI_{length}"
    return result


# Apply the patch
import pandas_ta_classic.volume.mfi as _mfi_module
_mfi_module.mfi = _patched_mfi
# Also patch it into the volume sub-package namespace and the top-level ta namespace
import pandas_ta_classic.volume as _ta_volume
_ta_volume.mfi = _patched_mfi
ta.mfi = _patched_mfi
from joblib import Parallel, delayed

# ═══════════════════════════════════════════════════════════════
# PATH SETUP — Must be BEFORE all project imports
# ═══════════════════════════════════════════════════════════════

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════
# PROJECT IMPORTS
# ═══════════════════════════════════════════════════════════════

from core.audit import Auditor
from core.io import read_parquet, write_parquet, write_json, write_csv
from phases.phase2.config import Phase2Config


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Columns excluded from feature set (identifiers, raw OHLCV, target)
_NON_FEATURE_COLS_LOWER: frozenset = frozenset({
    "ticker", "date", "open", "high", "low", "close", "volume",
    "adj_close", "target_log_return",
})

# Phase 2's input contract: minimum columns required from Phase 1
_PHASE2_REQUIRED_COLS: frozenset = frozenset({
    "ticker", "date", "open", "high", "low", "close", "volume",
})

# Audit manifest sampling (full dataframe may be 600K+ rows)
_MANIFEST_SAMPLE_ROWS: int = 1000

_DEFAULT_ANOMALY_THRESHOLD: float = 0.3
_DEFAULT_REDUNDANCY_THRESHOLD: float = 0.85
_PARALLEL_N_JOBS: int = 6


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (module-level)
# ═══════════════════════════════════════════════════════════════

def _print_box(title: str, width: int = 60) -> None:
    """Print a clean centered header box."""
    padding = (width - len(title) - 2) // 2
    print(f"\n{'=' * width}")
    print(f"{' ' * padding} {title}")
    print(f"{'=' * width}\n")


def _print_section(title: str) -> None:
    """Print a clean section divider."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")


def _print_kv(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair with consistent formatting."""
    print(f"{' ' * indent}{key:<25s}: {value}")


def compute_file_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file for reproducibility tracking.

    Reads in 64KB chunks for memory efficiency on large parquet files.

    Args:
        path: File path to hash.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def filter_tickers_by_history(
    df: pd.DataFrame,
    min_rows: int,
) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
    """
    Split tickers into those meeting the minimum history requirement and those
    that do not.

    Args:
        df: Multi-ticker OHLCV dataframe with a 'ticker' column.
        min_rows: Minimum number of rows required to process a ticker.

    Returns:
        Tuple of (ticker_groups, skipped_messages) where ticker_groups is a
        list of (ticker, ticker_df) pairs for eligible tickers and
        skipped_messages contains human-readable reasons for skipped ones.
    """
    ticker_groups: List[Tuple[str, pd.DataFrame]] = []
    skipped_tickers: List[str] = []

    for ticker in sorted(df["ticker"].unique()):
        ticker_df = df[df["ticker"] == ticker].copy()

        if len(ticker_df) < min_rows:
            skipped_tickers.append(
                f"{ticker} ({len(ticker_df)} rows < {min_rows} min)"
            )
            continue

        ticker_groups.append((ticker, ticker_df))

    return ticker_groups, skipped_tickers


def _process_ticker_worker(
    ticker: str,
    ticker_df: pd.DataFrame,
    curated_indicators: List[Dict[str, Any]],
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """
    Worker function for joblib parallel indicator generation.

    Runs in a separate loky process. Each worker:
        1. Builds its own ta.Strategy (avoids pickling issues)
        2. Sets pandas_ta internal cores=1 (outer parallelism handles concurrency)
        3. Suppresses upstream FutureWarning inside context manager
        4. Applies the strategy to a single ticker's dataframe

    Args:
        ticker: Ticker symbol (for error reporting).
        ticker_df: Single-ticker OHLCV dataframe (pre-filtered, pre-copied).
        curated_indicators: List of indicator dicts from config.

    Returns:
        Tuple of (ticker, result_df_or_None, error_message_or_None).
    """
    # NOTE: MFI FutureWarning is fixed at the source via monkey-patch in the
    # main process. loky workers re-import the module, so we re-apply the patch
    # here to guarantee the fix is active in child processes too.
    try:
        import pandas as pd  # noqa: F811 (shadowed in worker scope intentionally)
        import numpy as np  # noqa: F811
        import pandas_ta_classic as _ta_worker
        import pandas_ta_classic.volume.mfi as _mfi_worker
        import pandas_ta_classic.volume as _ta_vol_worker

        def _worker_patched_mfi(high, low, close, volume, length=None, drift=None, offset=None, **kwargs):
            _get = _mfi_worker.get
            _hlc3 = _mfi_worker.hlc3
            _non_zero_range = _mfi_worker.non_zero_range
            _unsigned_differences = _mfi_worker.unsigned_differences
            _verify_series = _mfi_worker.verify_series

            length = int(length) if length and length > 0 else 14
            drift = _get(kwargs, "drift", 1)
            _drift = int(drift) if drift and drift > 0 else 1

            high = _verify_series(high, length)
            low = _verify_series(low, length)
            close = _verify_series(close, length)
            volume = _verify_series(volume, length)
            if high is None or low is None or close is None or volume is None:
                return

            offset = int(offset) if offset else 0

            m = close.size
            df = pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
            df["hlc3"] = _hlc3(df["high"], df["low"], df["close"])
            df["diff"] = df["hlc3"].diff(_drift).fillna(0)
            raw_money_flow = df["hlc3"] * df["volume"]

            # ✅ FIX: float64 columns — no incompatible-dtype warning
            df["+mf"] = np.zeros(m, dtype=np.float64)
            df["-mf"] = np.zeros(m, dtype=np.float64)
            df.loc[df["diff"] > 0, "+mf"] = raw_money_flow
            df.loc[df["diff"] == -1, "-mf"] = raw_money_flow

            psum = df["+mf"].rolling(length).sum()
            nsum = df["-mf"].rolling(length).sum()
            nsum_nonzero = nsum.copy()
            nsum_nonzero[nsum_nonzero == 0] = 1

            result = 100 * psum / (psum + nsum_nonzero)
            result = result.iloc[offset:] if offset > 0 else result
            result.name = f"MFI_{length}"
            return result

        _mfi_worker.mfi = _worker_patched_mfi
        _ta_vol_worker.mfi = _worker_patched_mfi
        _ta_worker.mfi = _worker_patched_mfi
    except Exception:
        pass  # If patch fails for any reason, proceed — it won't crash, just may warn

    try:
        ticker_df.ta.cores = 1

        strategy = ta.Strategy(
            name="curated_phase2",
            ta=curated_indicators,
        )

        ticker_df.ta.strategy(strategy)

        return (ticker, ticker_df, None)

    except Exception as e:
        return (ticker, None, str(e))


def greedy_correlation_clustering(
    corr_matrix: pd.DataFrame,
    threshold: float = _DEFAULT_REDUNDANCY_THRESHOLD,
) -> List[List[str]]:
    """
    Greedy correlation clustering for redundancy detection.

    Algorithm:
        1. Sort features alphabetically (deterministic seed order)
        2. Pick first unassigned feature as cluster seed
        3. Add all features with |correlation| > threshold
        4. Repeat until all features assigned

    Determinism guarantee:
        - Features sorted alphabetically before clustering
        - Within clusters, members added in alphabetical order
        - Same input → same clusters, always

    Args:
        corr_matrix: Square correlation matrix (features × features).
        threshold: Absolute correlation threshold for grouping.

    Returns:
        List of clusters, where each cluster is a list of feature names.
        Single-member clusters represent non-redundant features.
    """
    features = sorted(corr_matrix.columns.tolist())
    unassigned: Set[str] = set(features)
    clusters: List[List[str]] = []

    for seed_candidate in features:
        if seed_candidate not in unassigned:
            continue

        unassigned.remove(seed_candidate)
        cluster = [seed_candidate]

        for feat in sorted(unassigned):
            if abs(corr_matrix.loc[seed_candidate, feat]) > threshold:
                cluster.append(feat)

        for feat in cluster[1:]:
            unassigned.discard(feat)

        clusters.append(cluster)

    return clusters


# ═══════════════════════════════════════════════════════════════
# MAIN CLASS: PreselectionAuditor
# ═══════════════════════════════════════════════════════════════

class PreselectionAuditor:
    """
    Phase 2 Stage 0: Preselection audit for leakage and redundancy.

    Generates technical indicators from Phase 1 cleaned OHLCV data,
    detects potential look-ahead bias via correlation analysis, and
    pre-filters redundant features via greedy clustering.

    All paths resolved via config.get_resolved_*() methods — works
    regardless of the current working directory.

    Output artifacts:
        - preselection_report.json:   Full audit results + config snapshot
        - correlation_matrix.parquet: Feature cross-correlation matrix
        - candidate_features.csv:    Vetted feature list for Stage 1
    """

    def __init__(self, config: Phase2Config) -> None:
        """
        Initialize the PreselectionAuditor.

        Args:
            config: Validated Phase2Config instance (v1.0.5+).
        """
        self.config = config
        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._leakage_flagged: List[str] = []
        self._redundancy_dropped: List[str] = []
        self._generation_errors: List[str] = []

    # ── INDICATOR GENERATION ────────────────────────────────────

    def generate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply curated pandas_ta strategy to each ticker in PARALLEL via joblib.

        Uses the loky backend with _PARALLEL_N_JOBS workers. Each worker runs
        _process_ticker_worker() which builds its own ta.Strategy to avoid
        pickling issues and suppresses the MFI FutureWarning via a
        catch_warnings context manager.

        Temporal integrity:
            - Each ticker processed independently (no cross-ticker leakage)
            - All indicators are backward-looking only
            - Tickers with < min_rows_per_ticker are skipped
        """
        ticker_groups, skipped_tickers = filter_tickers_by_history(
            df, self.config.min_rows_per_ticker
        )

        if skipped_tickers:
            print(f"[Stage 0] Skipped {len(skipped_tickers)} tickers "
                  f"(insufficient history):")
            for msg in skipped_tickers[:10]:
                print(f"  ⊘ {msg}")
            if len(skipped_tickers) > 10:
                print(f"  ... and {len(skipped_tickers) - 10} more")

        if not ticker_groups:
            raise ValueError(
                f"No tickers have >= {self.config.min_rows_per_ticker} rows. "
                f"Cannot generate indicators."
            )

        _print_kv("Tickers", f"{len(ticker_groups)} (n_jobs={_PARALLEL_N_JOBS}, loky)")

        results = Parallel(n_jobs=_PARALLEL_N_JOBS, backend="loky", verbose=0)(
            delayed(_process_ticker_worker)(
                ticker, ticker_df, self.config.curated_indicators
            )
            for ticker, ticker_df in ticker_groups
        )

        valid_results: List[pd.DataFrame] = []

        for ticker, result_df, error in results:
            if error:
                self._generation_errors.append(ticker)
                if len(self._generation_errors) <= 5:
                    warnings.warn(
                        f"[Stage 0] pandas_ta failed for '{ticker}': {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                valid_results.append(result_df)

        if not valid_results:
            raise RuntimeError(
                "All tickers failed indicator generation. "
                "Check pandas_ta_classic installation and input data quality."
            )

        if self._generation_errors:
            print(f"[Stage 0] ⚠ {len(self._generation_errors)} tickers "
                  f"failed: {self._generation_errors[:10]}")

        combined = pd.concat(valid_results, axis=0, ignore_index=False)

        # Sort for deterministic output ordering
        sort_cols = []
        if "ticker" in combined.columns:
            sort_cols.append("ticker")
        if "date" in combined.columns:
            sort_cols.append("date")
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)

        n_indicators = len(self._get_feature_columns(combined))
        n_success = len(valid_results)
        n_failed = len(self._generation_errors)
        _print_kv("Indicators generated", n_indicators)
        _print_kv("Tickers succeeded", n_success)
        _print_kv("Tickers failed", n_failed)

        return combined

    # ── TARGET COMPUTATION ──────────────────────────────────────

    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute forward log return target per-ticker.

        Formula: y_t = ln(P_{t+h} / P_t)
        Computed per-ticker to prevent cross-ticker contamination.
        Last h rows per ticker → NaN (never filled, dropped at analysis).
        """
        df = df.copy()
        horizon = self.config.target_horizon
        price_col = self.config.price_col

        if price_col not in df.columns:
            raise ValueError(
                f"Price column '{price_col}' not found. "
                f"Available: {sorted(df.columns.tolist())}"
            )

        def _compute_ticker_target(group: pd.DataFrame) -> pd.Series:
            prices = group[price_col]

            if (prices <= 0).any():
                non_pos = prices[prices <= 0]
                raise ValueError(
                    f"Non-positive prices in ticker "
                    f"'{group['ticker'].iloc[0]}': {non_pos.head().to_dict()}"
                )

            future_price = prices.shift(-horizon)
            log_ret = np.log(future_price / prices)

            # Temporal integrity assertion
            assert log_ret.iloc[-horizon:].isna().all(), (
                f"TEMPORAL INTEGRITY VIOLATION in ticker "
                f"'{group['ticker'].iloc[0]}': last {horizon} rows must be NaN"
            )
            return log_ret

        # Fixed: Added include_groups=False to suppress pandas FutureWarning
        df["target_log_return"] = df.groupby(
            "ticker", group_keys=False
        ).apply(_compute_ticker_target, include_groups=False)

        n_valid = df["target_log_return"].notna().sum()
        n_nan = df["target_log_return"].isna().sum()
        _print_kv("Valid targets", f"{n_valid:,}")
        _print_kv("NaN (boundary)", f"{n_nan:,}")

        return df

    # ── LEAKAGE DETECTION ───────────────────────────────────────

    def detect_leakage_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect features with suspiciously high Spearman correlation to target.

        |ρ| > anomaly_threshold flags a feature as a leakage suspect.
        Known leaky indicators (e.g. DPO) are pre-flagged regardless.
        """
        feature_cols = self._get_feature_columns(df)
        threshold = self.config.anomaly_threshold

        analysis_df = df.dropna(subset=["target_log_return"])

        if len(analysis_df) == 0:
            raise ValueError("No rows with valid target after dropping NaN.")

        target = analysis_df["target_log_return"]

        correlations: Dict[str, float] = {}
        flagged_features: List[str] = []
        computation_failures: List[str] = []

        for col in sorted(feature_cols):
            col_data = analysis_df[col]

            if col_data.isna().all() or col_data.nunique() < 2:
                computation_failures.append(col)
                continue

            valid_mask = col_data.notna() & target.notna()
            if valid_mask.sum() < 30:
                computation_failures.append(col)
                continue

            spearman_corr = col_data[valid_mask].corr(
                target[valid_mask], method="spearman"
            )

            if np.isnan(spearman_corr):
                computation_failures.append(col)
                continue

            correlations[col] = float(spearman_corr)

            if abs(spearman_corr) > threshold:
                flagged_features.append(col)

        # Add known leaky indicators
        known_leaky_found: List[str] = []
        for leaky_name in self.config.exclude_known_leaky:
            leaky_cols = [
                col for col in feature_cols
                if leaky_name.upper() in col.upper()
            ]
            for col in leaky_cols:
                if col not in flagged_features:
                    flagged_features.append(col)
                known_leaky_found.append(col)

        # Sort by absolute correlation (most suspicious first)
        flagged_features = sorted(
            flagged_features,
            key=lambda f: abs(correlations.get(f, 0.0)),
            reverse=True,
        )

        self._leakage_flagged = flagged_features

        _print_kv("Features analyzed", len(correlations))
        _print_kv("Computation failures", len(computation_failures))
        _print_kv("Anomaly threshold", f"|ρ| > {threshold}")
        _print_kv("Flagged (suspicious)", len(flagged_features))
        _print_kv("Known leaky found", len(known_leaky_found))

        if flagged_features:
            print(f"  Top flagged features:")
            for feat in flagged_features[:10]:
                corr_val = correlations.get(feat, float("nan"))
                source = " [KNOWN LEAKY]" if feat in known_leaky_found else ""
                print(f"    ⚠ {feat:<40s}  ρ = {corr_val:+.4f}{source}")

        return {
            "flagged_features": flagged_features,
            "correlations": correlations,
            "known_leaky_found": known_leaky_found,
            "computation_failures": computation_failures,
            "anomaly_threshold": threshold,
        }

    # ── REDUNDANCY CLUSTERING ───────────────────────────────────

    def cluster_redundant_features(
        self,
        df: pd.DataFrame,
        threshold: float = _DEFAULT_REDUNDANCY_THRESHOLD,
        target_correlations: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """Group redundant features via greedy correlation clustering.

        The representative of each cluster is chosen as the member with the
        highest absolute Spearman correlation to the target.  Ties are broken
        alphabetically (deterministic).  Features whose target correlation
        cannot be computed (all-NaN, etc.) default to 0.0 so they are least
        preferred.

        Args:
            df: DataFrame containing feature columns.
            threshold: Absolute correlation threshold for grouping.
            target_correlations: Dict mapping feature name → Spearman ρ vs
                target (pre-computed by detect_leakage_anomalies).  When
                omitted, all features default to 0.0 and the representative
                falls back to alphabetical order.
        """
        if target_correlations is None:
            target_correlations = {}

        feature_cols = sorted(self._get_feature_columns(df))

        analysis_df = df[feature_cols].dropna(how="all")

        if len(analysis_df) == 0:
            raise ValueError("No valid rows for correlation computation.")

        print(f"\n[Stage 0] Computing {len(feature_cols)}×{len(feature_cols)} "
              f"correlation matrix...")

        corr_matrix = analysis_df[feature_cols].corr(method="spearman")
        corr_matrix = corr_matrix.fillna(0.0)

        clusters = greedy_correlation_clustering(corr_matrix, threshold)

        representatives: List[str] = []
        dropped_as_redundant: List[str] = []

        for cluster in clusters:
            # Select member with highest |ρ| vs target.
            # Ties broken alphabetically (deterministic).
            # Avoids float equality — sort by (-|ρ|, name) and take first.
            rep = sorted(
                cluster,
                key=lambda f: (-abs(target_correlations.get(f, 0.0)), f),
            )[0]
            representatives.append(rep)
            for feat in cluster:
                if feat != rep:
                    dropped_as_redundant.append(feat)

        # Build (cluster, rep) pairs for reporting
        cluster_rep_pairs = list(zip(clusters, representatives))

        self._redundancy_dropped = dropped_as_redundant

        n_multi = sum(1 for c in clusters if len(c) > 1)
        _print_kv("Total features", len(feature_cols))
        _print_kv("Correlation threshold", f"|r| > {threshold}")
        _print_kv("Clusters formed", len(clusters))
        _print_kv("Multi-member clusters", n_multi)
        _print_kv("Representatives kept", len(representatives))
        _print_kv("Dropped as redundant", len(dropped_as_redundant))

        if n_multi > 0:
            print(f"  Largest clusters:")
            sorted_pairs = sorted(cluster_rep_pairs, key=lambda x: len(x[0]), reverse=True)
            for cluster, rep in sorted_pairs[:5]:
                if len(cluster) > 1:
                    rep_rho = target_correlations.get(rep, 0.0)
                    others = [f for f in cluster if f != rep]
                    print(f"    [{len(cluster)} members] "
                          f"rep='{rep}' (ρ={rep_rho:+.4f}) ← {others}")

        return {
            "clusters": clusters,
            "cluster_rep_pairs": cluster_rep_pairs,
            "representatives": representatives,
            "dropped_as_redundant": dropped_as_redundant,
            "correlation_matrix": corr_matrix,
            "threshold": threshold,
            "target_correlations": target_correlations,
        }

    # ── MAIN ENTRY POINT ────────────────────────────────────────

    def run_audit(self) -> Dict[str, Any]:
        """Execute the full preselection audit pipeline."""
        t_start = time.time()
        output_dir = self.config.get_resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.auditor.start()

        _print_box("PHASE 2 — STAGE 0: PRESELECTION AUDIT")
        _print_kv("Config version", self.config.version)
        _print_kv("Project root", PROJECT_ROOT)
        _print_kv("Input", self.config.get_resolved_input_path())
        _print_kv("Output dir", output_dir)
        _print_kv("Target horizon", f"{self.config.target_horizon} days")
        _print_kv("Indicators", f"{self.config.get_indicator_count()} curated")
        _print_kv("Parallel jobs", f"{_PARALLEL_N_JOBS} (loky)")

        # ── 1. Load and validate Phase 1 data ───────────────────
        print(f"\n  Loading Phase 1 data...")
        input_path = self.config.get_resolved_input_path()
        input_hash = compute_file_hash(input_path)

        raw_df = read_parquet(input_path)

        # Validate Phase 2's input contract
        missing = _PHASE2_REQUIRED_COLS - set(raw_df.columns)
        if missing:
            raise ValueError(
                f"Phase 1 output missing columns required by Phase 2: "
                f"{sorted(missing)}\n"
                f"  Available: {sorted(raw_df.columns.tolist())}\n"
                f"  File: {input_path}"
            )

        self.auditor.record_input(str(input_path), raw_df)

        n_tickers = raw_df["ticker"].nunique()
        _print_kv("Rows", f"{len(raw_df):,}")
        _print_kv("Tickers", n_tickers)
        _print_kv("SHA-256", f"{input_hash[:16]}...")

        # ── 2. Generate indicators ──────────────────────────────
        _print_section("Step 1/4 · Indicator Generation")

        indicator_df = self.generate_indicators(raw_df)

        # ── 3. Compute target ───────────────────────────────────
        _print_section("Step 2/4 · Target Computation")

        target_df = self.compute_target(indicator_df)

        # ── 4. Leakage detection ────────────────────────────────
        _print_section("Step 3/4 · Leakage Detection")

        leakage_report = self.detect_leakage_anomalies(target_df)

        # ── 5. Redundancy clustering ────────────────────────────
        _print_section("Step 4/4 · Redundancy Clustering")

        redundancy_report = self.cluster_redundant_features(
            target_df,
            target_correlations=leakage_report["correlations"],
        )
        corr_matrix = redundancy_report.pop("correlation_matrix")

        # ── 6. Compile candidate feature list ───────────────────
        representatives = set(redundancy_report["representatives"])
        flagged = set(leakage_report["flagged_features"])

        candidates = sorted(representatives - flagged)

        removed_by_leakage = sorted(representatives & flagged)
        removed_by_redundancy = sorted(
            set(self._redundancy_dropped) - flagged
        )

        all_features = sorted(self._get_feature_columns(target_df))
        _print_box("CANDIDATE FEATURE SUMMARY")
        _print_kv("Total generated", len(all_features))
        _print_kv("Removed (leakage)", len(leakage_report['flagged_features']))
        _print_kv("Removed (redundancy)", len(self._redundancy_dropped))
        _print_kv("Removed (overlap)", len(removed_by_leakage))
        print(f"  {'─' * 40}")
        _print_kv("Candidates → Stage 1", len(candidates))

        # ── 7. Save output artifacts ────────────────────────────
        _print_section("Saving Artifacts")

        candidate_path = output_dir / "candidate_features.csv"
        candidate_df = pd.DataFrame({
            "feature": candidates,
            "status": "candidate",
            "source": "preselection_audit_v" + self.config.version,
        })
        write_csv(candidate_df, candidate_path)
        print(f"  ✓ {candidate_path.name} ({len(candidates)} features)")

        corr_path = output_dir / "correlation_matrix.parquet"
        write_parquet(corr_matrix, corr_path, compression=self.config.compression)
        print(f"  ✓ {corr_path.name} ({corr_matrix.shape[0]}×{corr_matrix.shape[1]})")

        t_elapsed = time.time() - t_start

        report = {
            "stage": "0_preselection_audit",
            "version": self.config.version,
            "config_snapshot": self.config.to_snapshot(),
            "input": {
                "path": str(input_path),
                "sha256": input_hash,
                "rows": len(raw_df),
                "tickers": n_tickers,
            },
            "indicator_generation": {
                "total_indicators_generated": len(all_features),
                "tickers_processed": raw_df["ticker"].nunique() - len(self._generation_errors),
                "tickers_failed": self._generation_errors,
                "curated_strategy_used": self.config.use_curated_strategy,
                "indicator_definitions": self.config.get_indicator_count(),
            },
            "leakage_detection": {
                "anomaly_threshold": leakage_report["anomaly_threshold"],
                "features_analyzed": len(leakage_report["correlations"]),
                "features_flagged": leakage_report["flagged_features"],
                "known_leaky_found": leakage_report["known_leaky_found"],
                "computation_failures": leakage_report["computation_failures"],
                "top_correlations": dict(sorted(
                    leakage_report["correlations"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:20]),
            },
            "redundancy_clustering": {
                "threshold": redundancy_report["threshold"],
                "total_clusters": len(redundancy_report["clusters"]),
                "multi_member_clusters": sum(
                    1 for c in redundancy_report["clusters"] if len(c) > 1
                ),
                "representatives": redundancy_report["representatives"],
                "dropped_as_redundant": redundancy_report["dropped_as_redundant"],
                "cluster_details": [
                    {
                        "representative": rep,
                        "representative_target_rho": redundancy_report["target_correlations"].get(rep, 0.0),
                        "members": cluster,
                        "size": len(cluster),
                    }
                    for cluster, rep in redundancy_report["cluster_rep_pairs"]
                    if len(cluster) > 1
                ],
            },
            "output": {
                "candidates": candidates,
                "candidate_count": len(candidates),
                "removed_by_leakage": removed_by_leakage,
                "removed_by_redundancy": removed_by_redundancy,
                "total_removed": len(all_features) - len(candidates),
            },
            "runtime_seconds": round(t_elapsed, 2),
        }

        report_path = output_dir / "preselection_report.json"
        write_json(report, report_path)
        print(f"  ✓ {report_path.name}")

        # ── 8. Complete audit lifecycle ─────────────────────────
        # Sample for audit manifest — full dataset may be 600K+ rows
        output_df = target_df.head(_MANIFEST_SAMPLE_ROWS)
        self.auditor.record_output(output_df, self.config.to_snapshot())
        self.auditor.success()

        _print_box(f"COMPLETE — {len(candidates)} candidates → Stage 1  [{t_elapsed:.0f}s]")

        return report

    # ── PRIVATE HELPERS ───────────────────────────────���─────────

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Extract feature column names (excludes identifiers, OHLCV, target).
        
        Fixed: Removed redundant check — lowercase comparison suffices.
        """
        return sorted([
            col for col in df.columns
            if col.lower() not in _NON_FEATURE_COLS_LOWER
        ])


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from phases.phase2.config import get_phase2_config

    config = get_phase2_config()

    auditor = PreselectionAuditor(config)
    report = auditor.run_audit()

    # ── Clean candidate table ────────────────────────────
    n_candidates = report["output"]["candidate_count"]
    _print_section(f"Candidate Features ({n_candidates})")
    for i, feat in enumerate(report["output"]["candidates"], 1):
        print(f"  {i:>3d}. {feat}")

    # ── Leakage warnings ─────────────────────────────────
    flagged = report["leakage_detection"]["features_flagged"]
    if flagged:
        _print_section(f"Leakage-Flagged Features ({len(flagged)})")
        for feat in flagged[:10]:
            corr = report["leakage_detection"]["top_correlations"].get(feat, "N/A")
            print(f"    ✗ {feat:<35s}  ρ = {corr}")

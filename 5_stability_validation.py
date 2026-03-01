# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 4 — Walk-Forward Stability Validation                     ║
║                                                                            ║
║  Pipeline position:                                                        ║
║    Stage 0 (Preselection) → Stage 1 (Univariate) → Stage 2 (SFFS)         ║
║      → Stage 3 (Interactions) → [Stage 4: Stability Validation]            ║
║                                                                            ║
║  Purpose:                                                                  ║
║    Verify that the final feature model performs CONSISTENTLY over time,    ║
║    not just on average. A model with high average IC but wild variance      ║
║    across time periods is unreliable for live trading. Walk-forward         ║
║    analysis simulates how the model would have been retrained and           ║
║    deployed sequentially through history.                                   ║
║                                                                            ║
║  Method: Walk-Forward Validation (Rolling Window)                          ║
║    1. Load final features from interaction_report.json.                    ║
║    2. Regenerate base TA features in parallel, then create interactions.   ║
║    3. Flatten the panel (all tickers stacked, sorted by date then ticker). ║
║    4. Roll a sliding window through time:                                  ║
║       ┌────────────────────────┬───────────┐                               ║
║       │   Train (500 days)     │Test (125d) │                              ║
║       └────────────────────────┴───────────┘                               ║
║                                 ──step (125d)──►                           ║
║       ┌────────────────────────┬───────────┐                               ║
║       │   Train (500 days)     │Test (125d) │                              ║
║       └────────────────────────┴───────────┘                               ║
║    5. For each window: fit StandardScaler→Ridge on train, predict on test, ║
║       compute Spearman IC on test predictions.                             ║
║    6. Aggregate: Mean IC, Std IC, Hit Rate, Min/Max, Decay Check.          ║
║                                                                            ║
║  Walk-Forward vs TimeSeriesSplit:                                           ║
║    Stages 1-3 used TimeSeriesSplit (sklearn) — a single static split       ║
║    of the full dataset. Walk-forward is different: it simulates actual     ║
║    deployment by SLIDING through time, so each test window contains        ║
║    data the model has never seen AND that comes strictly after training.   ║
║    This is the gold standard for temporal validation in quant finance.     ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_2_features/interaction_report.json       (Stage 3)    ║
║    • artifacts/phase_1_data/merged_data.parquet                (Phase 1)   ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/stability_report.json                      ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)               ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — reads only disk artifacts from prior stages               ║
║    ✓ Deterministic — no random state; windows are date-ordered             ║
║    ✓ Atomic — writes to .tmp then renames                                  ║
║    ✓ Efficient — TA regenerated ONCE via joblib parallel;                  ║
║      walk-forward loop uses column slicing only                            ║
║    ✓ Traceable — per-window stats with exact date boundaries               ║
║    ✓ Temporal — strict train < test ordering; no future leakage            ║
║    ✓ Graceful — last window may be smaller; handled explicitly             ║
║    ✓ Unbiased — StandardScaler normalises all features before Ridge        ║
║                                                                            ║
║  TA Generation Note:                                                       ║
║    Parallel per-ticker via joblib (loky backend, n_jobs=6).                ║
║    loky uses non-daemonic workers → no "daemon child" crash.               ║
║    cores=1 per worker prevents OS thread thrashing.                        ║
║    Interaction columns are computed via element-wise product AFTER TA.     ║
║                                                                            ║
║  ⚠ PREREQUISITE: interaction_report.json must be produced by Stage 3      ║
║    v1.1.0+ (with StandardScaler). Re-run 4_interaction_discovery.py       ║
║    before this script if the report predates v1.1.0.                      ║
║                                                                            ║
║  Version: 1.1.0                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# SUPPRESS PANDAS_TA UPSTREAM FUTUREWARNING (BEFORE IMPORT)
# ═══════════════════════════════════════════════════════════════
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Setting an item of incompatible dtype is deprecated",
    module="pandas_ta_classic",
)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ─── Project path bootstrap ────────────────────────────────────────────────
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas_ta_classic as ta                         # noqa: E402
from core.audit import Auditor                         # noqa: E402
from core.io import write_json                         # noqa: E402
from phases.phase2.config import Phase2Config          # noqa: E402


# ─── Constants ──────────────────────────────────────────────────────────────
VERSION = "1.1.0"
STAGE_NAME = "4_stability_validation"

PHASE1_ARTIFACT = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase_2_features"
INTERACTION_REPORT_JSON = PHASE2_ARTIFACT_DIR / "interaction_report.json"
OUTPUT_JSON = PHASE2_ARTIFACT_DIR / "stability_report.json"

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

# Matched exactly to Stage 1/2/3 hyperparameters
RIDGE_ALPHA = 1.0
FORWARD_DAYS = 5

# Walk-forward window parameters (in UNIQUE trading days)
TRAIN_DAYS = 500     # ~2 years of trading days
TEST_DAYS = 125      # ~6 months of trading days
STEP_DAYS = 125      # ~6 months step (non-overlapping test windows)

# Parallelism parameters
N_PARALLEL_JOBS = 6     # Worker processes (leaves ~5 cores free for OS)
MIN_TICKER_ROWS = 200   # Minimum history required per ticker

# Interaction column prefix — must match Stage 3 naming convention
_IX_PREFIX = "IX_"
_IX_SEPARATOR = "_x_"


# ─── Indicator-to-column mapping ───────────────────────────────────────────
_KIND_TO_PREFIX: Dict[str, List[str]] = {
    "adx": ["ADX_", "DMP_", "DMN_"],
    "chop": ["CHOP_"],
    "aroon": ["AROOND_", "AROONU_", "AROONOSC_"],
    "supertrend": ["SUPERT_", "SUPERTd_", "SUPERTl_", "SUPERTs_"],
    "rsi": ["RSI_"],
    "roc": ["ROC_"],
    "macd": ["MACD_", "MACDh_", "MACDs_"],
    "mom": ["MOM_"],
    "willr": ["WILLR_"],
    "natr": ["NATR_"],
    "bbands": ["BBL_", "BBM_", "BBU_", "BBB_", "BBP_"],
    "atr": ["ATRr_"],
    "rvi": ["RVI_"],
    "cmf": ["CMF_"],
    "mfi": ["MFI_"],
    "obv": ["OBV"],
    "sma": ["SMA_"],
    "ema": ["EMA_"],
    "zscore": ["ZS_"],
    "skew": ["SKEW_"],
    "stdev": ["STDEV_"],
    "log_return": ["LOGRET_"],
    "percent_return": ["PCTRET_"],
}


# ─── Parallel Worker Function ───────────────────────────────────────────────
def _process_ticker_worker(
    ticker: str,
    ticker_df: pd.DataFrame,
    filtered_specs: List[Dict[str, Any]],
) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Worker function for parallel TA generation (one ticker per call).

    Uses the `loky` backend (non-daemonic workers) so that pandas_ta can
    spawn its own subprocesses without hitting the "daemon child" restriction.

    Sets ``ticker_df.ta.cores = 1`` to prevent each worker from spawning
    all available CPU threads internally — without this, 6 workers × 11
    threads = 66 competing threads on 11 physical cores (OS thrashing).

    Args:
        ticker: Ticker symbol string.
        ticker_df: OHLCV DataFrame slice for this ticker (already copied).
        filtered_specs: List of pandas_ta indicator spec dicts to compute.

    Returns:
        Tuple of (ticker, result_df_or_None, error_str_or_None).
    """
    import warnings as _warnings
    _warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="Setting an item of incompatible dtype",
    )

    try:
        strategy = ta.Strategy(
            name="Phase2_Stability_Base",
            ta=filtered_specs,
        )

        # CRITICAL: Limit internal threading to 1 per worker process
        ticker_df.ta.cores = 1
        ticker_df.ta.strategy(strategy, verbose=False)

        return (ticker, ticker_df, None)
    except Exception as exc:
        return (ticker, None, str(exc))


# ─── Main Class ─────────────────────────────────────────────────────────────
class StabilityValidator:
    """Walk-forward stability validation for the final feature set.

    Slides a train/test window through the date axis to verify that the
    final model performs consistently across all time periods, not
    just on average via a single CV split.
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._final_features: Optional[List[str]] = None
        self._base_features: Optional[List[str]] = None
        self._interaction_features: Optional[List[str]] = None
        self._stage3_score: Optional[float] = None
        self._filtered_specs: Optional[List[Dict[str, Any]]] = None

    # ── 1. Load Stage 3 (Interaction) results ───────────────────────────────
    def load_stage3_results(self) -> List[str]:
        """Load interaction_report.json and extract the final features.

        Separates features into base (TA indicators) and interactions
        (product columns, IX_ prefix) for correct regeneration.

        Warns if the report predates v1.1.0 (the unscaled/biased version),
        since running Stage 4 against stale Stage 3 output would validate
        the wrong model.

        Returns:
            List of all final feature names.

        Raises:
            FileNotFoundError: If Stage 3 output is missing.
            AssertionError: If report is malformed.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Loading Stage 3 (Interaction) Results...")
        print(f"{'='*72}")

        if not INTERACTION_REPORT_JSON.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 3 output not found: {INTERACTION_REPORT_JSON}\n"
                f"       Run 4_interaction_discovery.py before this script."
            )

        with open(INTERACTION_REPORT_JSON, "r", encoding="utf-8") as f:
            stage3_data = json.load(f)

        assert "final_feature_set" in stage3_data, (
            f"[SCHEMA VIOLATION] interaction_report.json missing "
            f"'final_feature_set' key.\n"
            f"  Found keys: {list(stage3_data.keys())}"
        )

        final_features = stage3_data["final_feature_set"]
        assert isinstance(final_features, list) and len(final_features) > 0, (
            f"[SCHEMA VIOLATION] final_feature_set must be non-empty list, "
            f"got: {final_features}"
        )

        # Warn if interaction_report.json predates the StandardScaler fix
        stage3_version = stage3_data.get("version", "unknown")
        if stage3_version < "1.1.0":
            print(f"  ⚠ WARNING: interaction_report.json was produced by "
                  f"Stage 3 version {stage3_version}.")
            print(f"    This is the BIASED (unscaled) version — you are "
                  f"validating the WRONG model.")
            print(f"    Re-run 4_interaction_discovery.py (v1.1.0+) first, "
                  f"then re-run this script.")

        # Separate base features from interactions
        base_features = [f for f in final_features if not f.startswith(_IX_PREFIX)]
        interaction_features = [f for f in final_features if f.startswith(_IX_PREFIX)]

        self._final_features = final_features
        self._base_features = base_features
        self._interaction_features = interaction_features
        self._stage3_score = stage3_data.get("final_score_mean_ic")

        print(f"  ✓ Stage 3 report version: {stage3_version}")
        print(f"  ✓ Final feature set ({len(final_features)} features):")
        print(f"    Base features ({len(base_features)}):")
        for i, feat in enumerate(base_features, 1):
            print(f"      {i}. {feat}")
        if interaction_features:
            print(f"    Interaction features ({len(interaction_features)}):")
            for i, ix in enumerate(interaction_features, 1):
                parts = ix.replace(_IX_PREFIX, "").split(_IX_SEPARATOR)
                desc = f"{parts[0]} × {parts[1]}" if len(parts) == 2 else ix
                print(f"      +{i}. {ix}  ({desc})")
        if self._stage3_score is not None:
            print(f"  ✓ Stage 3 reported Mean IC: {self._stage3_score:.6f}")

        return final_features

    # ── 2. Load Phase 1 data ────────────────────────────────────────────────
    def load_phase1_data(self) -> pd.DataFrame:
        """Load merged OHLCV and validate schema.

        Returns:
            pd.DataFrame: Raw OHLCV sorted by (ticker, date).
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Loading Phase 1 Data...")
        print(f"{'='*72}")

        if not PHASE1_ARTIFACT.exists():
            raise FileNotFoundError(
                f"[FATAL] Phase 1 artifact not found: {PHASE1_ARTIFACT}\n"
                f"       Run Phase 1 before Phase 2."
            )

        t0 = perf_counter()
        df = pd.read_parquet(PHASE1_ARTIFACT)
        t_load = perf_counter() - t0

        df.columns = df.columns.str.lower().str.strip()
        print(f"  ✓ Loaded merged_data.parquet: {df.shape[0]:,} rows × "
              f"{df.shape[1]} cols  ({t_load:.2f}s)")

        missing = REQUIRED_COLUMNS - set(df.columns)
        assert len(missing) == 0, (
            f"[SCHEMA VIOLATION] Missing required columns: {missing}\n"
            f"  Present columns: {sorted(df.columns.tolist())}"
        )
        print(f"  ✓ Schema validated: {sorted(REQUIRED_COLUMNS)} all present")

        self.auditor.start()
        self.auditor.record_input(str(PHASE1_ARTIFACT), df)

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    # ── 3. Compute target ───────────────────────────────────────────────────
    @staticmethod
    def compute_target(df: pd.DataFrame) -> pd.Series:
        """Compute 5-day forward log return per ticker.

        Identical formula to Stage 1/2/3:  target_t = ln(close_{t+5} / close_t)

        Args:
            df: DataFrame sorted by (ticker, date) with 'close' column.

        Returns:
            pd.Series: Named 'target_log_return', index-aligned to df.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Computing Target: {FORWARD_DAYS}-day forward log return")
        print(f"{'='*72}")

        t0 = perf_counter()

        future_close = df.groupby("ticker")["close"].shift(-FORWARD_DAYS)
        target = np.log(future_close / df["close"])
        target.name = "target_log_return"

        n_valid = target.notna().sum()
        n_nan = target.isna().sum()
        t_elapsed = perf_counter() - t0
        print(f"  ✓ Target computed: {n_valid:,} valid / {n_nan:,} NaN  "
              f"({t_elapsed:.3f}s)")

        return target

    # ── 4. Regenerate base features + create interactions ───────────────────
    def regenerate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regenerate base TA features in parallel, then create interaction cols.

        FIX v1.1.0 — Replaced sequential for-loop with joblib Parallel:
            Original: 750 tickers × ~11s each = ~81 min on 1 core.
            Fixed:    750 tickers / 6 workers  = ~2 min wall-clock time.

        Two-phase feature construction:
          Phase A: pandas_ta generates base TA indicators in parallel.
                   loky backend + cores=1 per worker prevents OS thrashing.
          Phase B: Interaction columns computed as element-wise products of
                   base features after TA generation (not TA indicators).

        Why loky backend?
            pandas_ta_classic may try to spawn child processes internally.
            Standard multiprocessing daemon workers cannot have children
            (RuntimeError). loky uses non-daemonic workers.

        Why cores=1 per worker?
            Each of the 6 workers would otherwise use all 11 available
            threads → 66 threads competing for 11 cores (OS thrashing).

        Args:
            df: Raw OHLCV DataFrame with target column.

        Returns:
            pd.DataFrame: Input df augmented with all final feature columns.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Regenerating Features "
              f"(parallel, n_jobs={N_PARALLEL_JOBS})")
        print(f"{'='*72}")

        # ═══════════════════════════════════════════════════════════════════
        # Phase A: Regenerate base TA features in parallel
        # ═══════════════════════════════════════════════════════════════════
        t0 = perf_counter()

        all_specs = self.config.curated_indicators
        base_set = set(self._base_features)

        filtered_specs: List[Dict[str, Any]] = []
        for spec in all_specs:
            kind = spec.get("kind", "")
            prefixes = _KIND_TO_PREFIX.get(kind, [])
            produces_needed = any(
                any(feat.startswith(prefix) for feat in base_set)
                for prefix in prefixes
            )
            if produces_needed:
                filtered_specs.append(spec)

        self._filtered_specs = filtered_specs
        print(f"  ℹ Phase A: Filtered {len(all_specs)} curated specs → "
              f"{len(filtered_specs)} needed for "
              f"{len(self._base_features)} base features")

        # ── Prepare per-ticker slices ───────────────────────────────────────
        tickers = sorted(df["ticker"].unique())  # Deterministic ordering
        ticker_groups: List[Tuple[str, pd.DataFrame]] = []
        skipped_short: List[str] = []

        for ticker in tickers:
            tdf = df[df["ticker"] == ticker].copy()
            if len(tdf) < MIN_TICKER_ROWS:
                skipped_short.append(ticker)
                continue
            ticker_groups.append((ticker, tdf))

        if skipped_short:
            print(f"  ⚠ Skipped {len(skipped_short)} tickers with < "
                  f"{MIN_TICKER_ROWS} rows: {skipped_short[:10]}"
                  f"{'...' if len(skipped_short) > 10 else ''}")

        print(f"  ℹ Processing {len(ticker_groups)} tickers with "
              f"{len(filtered_specs)} indicator specs "
              f"(parallel, n_jobs={N_PARALLEL_JOBS}, backend=loky)")

        # ── Parallel TA generation ──────────────────────────────────────────
        parallel_results = Parallel(
            n_jobs=N_PARALLEL_JOBS,
            backend="loky",
            verbose=0,
        )(
            delayed(_process_ticker_worker)(ticker, tdf, filtered_specs)
            for ticker, tdf in ticker_groups
        )

        # ── Collect results ─────────────────────────────────────────────────
        frames: List[pd.DataFrame] = []
        generation_errors: List[str] = []

        for ticker, result_df, error in parallel_results:
            if error is not None:
                generation_errors.append(ticker)
                if len(generation_errors) <= 5:
                    warnings.warn(
                        f"[Stage 4] pandas_ta failed for '{ticker}': {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                frames.append(result_df)

        if not frames:
            raise RuntimeError(
                "All tickers failed indicator generation. "
                "Check pandas_ta_classic installation and input data quality."
            )

        if generation_errors:
            print(f"  ⚠ {len(generation_errors)} tickers failed: "
                  f"{generation_errors[:10]}"
                  f"{'...' if len(generation_errors) > 10 else ''}")

        initial_cols = set(df.columns)
        result_df = pd.concat(frames, axis=0, ignore_index=True)
        new_cols = set(result_df.columns) - initial_cols
        print(f"  ✓ TA strategy generated {len(new_cols)} new columns "
              f"across {len(frames)} tickers")

        # ── Verify base columns are present ─────────────────────────────────
        generated_cols = set(result_df.columns)
        matched_base = [c for c in self._base_features if c in generated_cols]
        missing_base = [c for c in self._base_features if c not in generated_cols]

        # Case-insensitive fallback
        if missing_base:
            col_map = {c.upper(): c for c in generated_cols}
            for m in missing_base[:]:
                if m.upper() in col_map:
                    actual = col_map[m.upper()]
                    result_df.rename(columns={actual: m}, inplace=True)
                    matched_base.append(m)
                    missing_base.remove(m)

        if missing_base:
            raise RuntimeError(
                f"[FATAL] Cannot regenerate base features: {missing_base}\n"
                f"  Generated columns: {sorted(new_cols)}"
            )

        print(f"  ✓ Matched {len(matched_base)}/{len(self._base_features)} "
              f"base features")

        t_ta = perf_counter() - t0

        # ═══════════════════════════════════════════════════════════════════
        # Phase B: Create interaction columns from base features
        # ═══════════════════════════════════════════════════════════════════
        t_ix = perf_counter()

        for ix_name in self._interaction_features:
            # Parse IX_ADX_14_x_CMF_20 → ("ADX_14", "CMF_20")
            inner = ix_name[len(_IX_PREFIX):]          # "ADX_14_x_CMF_20"
            parts = inner.split(_IX_SEPARATOR)          # ["ADX_14", "CMF_20"]

            if len(parts) != 2:
                raise ValueError(
                    f"[FATAL] Cannot parse interaction column '{ix_name}'.\n"
                    f"  Expected format: IX_<featureA>_x_<featureB>\n"
                    f"  Got parts: {parts}"
                )

            feat_a, feat_b = parts

            if feat_a not in result_df.columns:
                raise RuntimeError(
                    f"[FATAL] Interaction '{ix_name}' requires base feature "
                    f"'{feat_a}' which is not present."
                )
            if feat_b not in result_df.columns:
                raise RuntimeError(
                    f"[FATAL] Interaction '{ix_name}' requires base feature "
                    f"'{feat_b}' which is not present."
                )

            result_df[ix_name] = result_df[feat_a] * result_df[feat_b]

        t_ix_elapsed = perf_counter() - t_ix

        if self._interaction_features:
            print(f"  ✓ Phase B: Created {len(self._interaction_features)} "
                  f"interaction columns ({t_ix_elapsed:.3f}s)")
        else:
            print(f"  ℹ Phase B: No interaction columns to create")

        # Final verification: all features present
        all_missing = [
            f for f in self._final_features
            if f not in result_df.columns
        ]

        if all_missing:
            raise RuntimeError(
                f"[FATAL] Final feature set incomplete. Missing: {all_missing}"
            )

        print(f"  ✓ All {len(self._final_features)} features verified present")
        print(f"  ✓ Feature regeneration complete ({perf_counter() - t0:.1f}s)")

        return result_df

    # ── 5. Walk-Forward Validation ──────────────────────────────────────────
    def run_walk_forward(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute walk-forward validation with rolling train/test windows.

        FIX v1.1.0 — StandardScaler added inside the validation loop:
            Interaction terms are products of base features, creating extreme
            scale differences (e.g. SMA_50 × SMA_50 ≈ 400,000,000 vs
            LOGRET_5 × CMF_20 ≈ 0.002). Without scaling, Ridge's L2 penalty
            suppresses large-magnitude interaction terms regardless of their
            predictive signal, invalidating the stability test entirely.
            StandardScaler is fit ONLY on X_train in each window — no leakage.

        Window mechanics (using UNIQUE trading dates across all tickers):
          - Extract the sorted unique date array from the dataset.
          - Slide a window: train on dates[i:i+500], test on dates[i+500:i+625].
          - Step forward by 125 dates.
          - For each window, select ALL rows (all tickers) within those dates.

        This is a PANEL walk-forward: at each step, we train on ALL tickers
        in the train period and test on ALL tickers in the test period.

        Temporal integrity:
          - Train dates are STRICTLY before test dates (no overlap).
          - The target (forward log return) may have NaN at ticker
            boundaries — these rows are dropped via .dropna().

        Graceful end-of-data:
          - The last window's test period may be shorter than TEST_DAYS.
          - Windows with < 50 test rows are skipped (not enough for IC).

        Args:
            df: DataFrame with all final features, target, ticker, date.
                Must be sorted by (ticker, date).

        Returns:
            Dict with per-window results and aggregate stability metrics.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Running Walk-Forward Validation")
        print(f"{'='*72}")
        print(f"  ℹ Features        : {len(self._final_features)}")
        print(f"  ℹ Model           : StandardScaler → Ridge(alpha={RIDGE_ALPHA})")
        print(f"  ℹ Train window    : {TRAIN_DAYS} trading days (~2 years)")
        print(f"  ℹ Test window     : {TEST_DAYS} trading days (~6 months)")
        print(f"  ℹ Step size       : {STEP_DAYS} trading days (~6 months)")

        t0 = perf_counter()

        target_col = "target_log_return"
        feature_cols = list(self._final_features)

        assert target_col in df.columns, (
            f"[FATAL] Target column '{target_col}' not in DataFrame."
        )

        # ── Build date index ────────────────────────────────────────────────
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        unique_dates = np.sort(df["date"].unique())
        n_dates = len(unique_dates)
        print(f"  ℹ Unique trading dates: {n_dates}")
        print(f"  ℹ Date range: {unique_dates[0]} to {unique_dates[-1]}")

        min_window = TRAIN_DAYS + 50  # Need at least 50 test rows
        if n_dates < min_window:
            raise ValueError(
                f"[FATAL] Insufficient data for walk-forward validation.\n"
                f"  Need at least {min_window} unique dates, have {n_dates}.\n"
                f"  Train={TRAIN_DAYS} + min_test=50 = {min_window}"
            )

        # ── Enumerate windows ───────────────────────────────────────────────
        windows: List[Dict[str, Any]] = []

        window_idx = 0
        train_start_idx = 0

        while train_start_idx + TRAIN_DAYS < n_dates:
            window_idx += 1

            # Define date boundaries via index into unique_dates
            train_end_idx = train_start_idx + TRAIN_DAYS    # exclusive
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + TEST_DAYS, n_dates)

            # Actual dates
            train_date_start = unique_dates[train_start_idx]
            train_date_end = unique_dates[train_end_idx - 1]  # inclusive
            test_date_start = unique_dates[test_start_idx]
            test_date_end = unique_dates[test_end_idx - 1]    # inclusive

            actual_test_days = test_end_idx - test_start_idx

            # ── Extract train and test DataFrames ───────────────────────────
            train_dates_set = set(unique_dates[train_start_idx:train_end_idx])
            test_dates_set = set(unique_dates[test_start_idx:test_end_idx])

            train_mask = df["date"].isin(train_dates_set)
            test_mask = df["date"].isin(test_dates_set)

            train_df = df.loc[train_mask, feature_cols + [target_col]].dropna()
            test_df = df.loc[test_mask, feature_cols + [target_col]].dropna()

            n_train = len(train_df)
            n_test = len(test_df)

            # Skip windows with insufficient test data
            if n_test < 50:
                print(f"  ⊘ Window {window_idx}: Skipped "
                      f"(only {n_test} test rows < 50 minimum)")
                windows.append({
                    "window": window_idx,
                    "train_start": str(train_date_start)[:10],
                    "train_end": str(train_date_end)[:10],
                    "test_start": str(test_date_start)[:10],
                    "test_end": str(test_date_end)[:10],
                    "train_days": TRAIN_DAYS,
                    "test_days": actual_test_days,
                    "train_rows": n_train,
                    "test_rows": n_test,
                    "ic": None,
                    "status": "SKIPPED_INSUFFICIENT_TEST_DATA",
                })
                train_start_idx += STEP_DAYS
                continue

            # ── Fit Pipeline on train, predict on test ──────────────────────
            X_train = train_df[feature_cols].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values

            # FIX v1.1.0: StandardScaler → Ridge pipeline removes magnitude
            # bias from interaction terms. Scaler is fit only on X_train
            # (no data leakage into the test window).
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
            ])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ── Compute Spearman IC on test window ──────────────────────────
            if np.std(y_pred) < 1e-15 or np.std(y_test) < 1e-15:
                ic = 0.0
                p_value = 1.0
            else:
                ic, p_value = spearmanr(y_pred, y_test)
                if np.isnan(ic):
                    ic = 0.0
                p_value = float(p_value) if not np.isnan(p_value) else 1.0

            ic = round(float(ic), 6)

            window_result = {
                "window": window_idx,
                "train_start": str(train_date_start)[:10],
                "train_end": str(train_date_end)[:10],
                "test_start": str(test_date_start)[:10],
                "test_end": str(test_date_end)[:10],
                "train_days": TRAIN_DAYS,
                "test_days": actual_test_days,
                "train_rows": n_train,
                "test_rows": n_test,
                "ic": ic,
                "status": "OK",
            }
            windows.append(window_result)

            ic_str = f"{ic:+.6f}"
            indicator = "✓" if ic > 0 else "✗"
            print(f"  {indicator} Window {window_idx}: "
                  f"train [{str(train_date_start)[:10]} → "
                  f"{str(train_date_end)[:10]}] "
                  f"test [{str(test_date_start)[:10]} → "
                  f"{str(test_date_end)[:10]}] "
                  f"IC = {ic_str}  "
                  f"(train={n_train:,} test={n_test:,})")

            # Step forward
            train_start_idx += STEP_DAYS

        total_time = perf_counter() - t0

        # ── Aggregate stability metrics ─────────────────────────────────────
        scored_windows = [w for w in windows if w["ic"] is not None]
        ics = [w["ic"] for w in scored_windows]

        if not ics:
            raise RuntimeError(
                "[FATAL] No valid walk-forward windows produced. "
                "Check data coverage."
            )

        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        median_ic = float(np.median(ics))
        min_ic = float(np.min(ics))
        max_ic = float(np.max(ics))
        hit_rate = sum(1 for ic in ics if ic > 0) / len(ics)

        # Information Ratio across windows
        if std_ic > 1e-10:
            ir = mean_ic / std_ic
        else:
            ir = float("inf") if mean_ic > 0 else 0.0

        # Decay check: compare first half vs second half mean IC
        n_windows = len(ics)
        half = n_windows // 2
        if half > 0:
            first_half_ic = float(np.mean(ics[:half]))
            second_half_ic = float(np.mean(ics[half:]))
            decay_delta = second_half_ic - first_half_ic
        else:
            first_half_ic = mean_ic
            second_half_ic = mean_ic
            decay_delta = 0.0

        # Worst window
        worst_window_idx = int(np.argmin(ics))
        worst_window = scored_windows[worst_window_idx]

        stability_metrics = {
            "n_windows_total": len(windows),
            "n_windows_scored": len(scored_windows),
            "n_windows_skipped": len(windows) - len(scored_windows),
            "mean_ic": round(mean_ic, 6),
            "std_ic": round(std_ic, 6),
            "median_ic": round(median_ic, 6),
            "min_ic": round(min_ic, 6),
            "max_ic": round(max_ic, 6),
            "ir": round(ir, 4) if np.isfinite(ir) else ir,
            "hit_rate": round(hit_rate, 4),
            "hit_rate_pct": round(hit_rate * 100, 1),
            "first_window_ic": round(float(ics[0]), 6),
            "last_window_ic": round(float(ics[-1]), 6),
            "first_half_mean_ic": round(first_half_ic, 6),
            "second_half_mean_ic": round(second_half_ic, 6),
            "decay_delta": round(decay_delta, 6),
            "decay_assessment": (
                "STABLE" if abs(decay_delta) < 0.005
                else ("IMPROVING" if decay_delta > 0 else "DECAYING")
            ),
            "worst_window": {
                "window": worst_window["window"],
                "test_period": (
                    f"{worst_window['test_start']} → "
                    f"{worst_window['test_end']}"
                ),
                "ic": worst_window["ic"],
            },
        }

        # ── Assemble output ─────────────────────────────────────────────────
        output = {
            "stage": STAGE_NAME,
            "version": VERSION,
            "algorithm": "Walk-Forward Validation (Rolling Window)",
            "config_snapshot": {
                "ridge_alpha": RIDGE_ALPHA,
                "forward_days": FORWARD_DAYS,
                "train_days": TRAIN_DAYS,
                "test_days": TEST_DAYS,
                "step_days": STEP_DAYS,
                "features": list(self._final_features),
                "feature_count": len(self._final_features),
                "base_feature_count": len(self._base_features),
                "interaction_count": len(self._interaction_features),
                "curated_specs_used": len(self._filtered_specs)
                    if self._filtered_specs else 0,
                "scaler": "StandardScaler",
                "n_parallel_jobs": N_PARALLEL_JOBS,
                "parallel_backend": "loky",
            },
            "stage3_reference": {
                "reported_mean_ic": self._stage3_score,
                "note": (
                    "Stage 3 IC was computed via TimeSeriesSplit (static). "
                    "Walk-forward IC below may differ due to rolling windows."
                ),
            },
            "windows": windows,
            "stability_metrics": stability_metrics,
            "runtime_seconds": round(total_time, 2),
        }

        # ── Print summary ───────────────────────────────────────────────────
        print(f"\n  {'═'*60}")
        print(f"  WALK-FORWARD STABILITY REPORT:")
        print(f"  {'═'*60}")
        print(f"  Windows scored    : {len(scored_windows)}")
        print(f"  Mean IC           : {mean_ic:.6f}")
        print(f"  Std IC            : {std_ic:.6f}")
        print(f"  Median IC         : {median_ic:.6f}")
        if np.isfinite(ir):
            print(f"  IR (Mean/Std)     : {ir:.4f}")
        else:
            print(f"  IR (Mean/Std)     : ∞")
        print(f"  Hit Rate          : {hit_rate*100:.1f}% "
              f"({sum(1 for ic in ics if ic > 0)}/{len(ics)} windows IC > 0)")
        print(f"  Min IC            : {min_ic:.6f}")
        print(f"  Max IC            : {max_ic:.6f}")
        print(f"  ──────────────────────────────────────────────")
        print(f"  1st half Mean IC  : {first_half_ic:.6f}")
        print(f"  2nd half Mean IC  : {second_half_ic:.6f}")
        print(f"  Decay Δ           : {decay_delta:+.6f}")
        print(f"  Assessment        : {stability_metrics['decay_assessment']}")
        print(f"  ──────────────────────────────────────────────")
        print(f"  Worst window      : #{worst_window['window']} "
              f"({worst_window['test_start']} → {worst_window['test_end']}) "
              f"IC = {worst_window['ic']:.6f}")
        if self._stage3_score is not None:
            print(f"  Stage 3 ref IC    : {self._stage3_score:.6f}")
        print(f"  Runtime           : {total_time:.2f}s")
        print(f"  {'═'*60}")

        return output

    # ── 6. Save results atomically ──────────────────────────────────────────
    def save_results(self, results: Dict[str, Any]) -> None:
        """Write stability_report.json atomically (.tmp → rename).

        Also completes the Auditor lifecycle.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 4] Saving Results")
        print(f"{'='*72}")

        PHASE2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        # ── Atomic write ────────────────────────────────────────────────────
        tmp_path = OUTPUT_JSON.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    results, f, indent=2, ensure_ascii=False, default=str
                )
            tmp_path.replace(OUTPUT_JSON)
            print(f"  ✓ {OUTPUT_JSON.name} "
                  f"({OUTPUT_JSON.stat().st_size:,} bytes)")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"[FATAL] Failed to write results: {e}"
            ) from e

        # ── Auditor lifecycle completion ────────────────────────────────────
        try:
            metrics = results.get("stability_metrics", {})
            summary_df = pd.DataFrame([{
                "mean_ic": metrics.get("mean_ic"),
                "std_ic": metrics.get("std_ic"),
                "hit_rate": metrics.get("hit_rate"),
                "n_windows": metrics.get("n_windows_scored"),
                "decay_assessment": metrics.get("decay_assessment"),
            }])
            self.auditor.record_output(summary_df, self.config.to_snapshot())
            self.auditor.success()
            print(f"  ✓ manifest.json (via Auditor)")
        except Exception as e:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {e}")

    # ── Orchestrator ────────────────────────────────────────────────────────
    def run(self) -> None:
        """Full pipeline: Stage3 → Data → Target → Features → Walk-Forward → Save."""
        pipeline_t0 = perf_counter()

        # Step 1: Load Stage 3 results → extract final features
        self.load_stage3_results()

        # Step 2: Load Phase 1 raw OHLCV
        df = self.load_phase1_data()

        # Step 3: Compute target
        target = self.compute_target(df)
        df["target_log_return"] = target

        # Step 4: Regenerate base TA features (parallel) + interaction columns
        df = self.regenerate_features(df)

        # Step 5: Walk-forward validation (StandardScaler→Ridge per window)
        results = self.run_walk_forward(df)

        # Step 6: Save atomically
        self.save_results(results)

        total_time = perf_counter() - pipeline_t0
        print(f"\n{'='*72}")
        print(f"[Stage 4] COMPLETE — Total runtime: {total_time:.1f}s")
        print(f"{'='*72}\n")


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 4 — Walk-Forward Stability Validation     ║")
    print(f"║  Version: {VERSION:<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        config = Phase2Config()

        validator = StabilityValidator(config=config)
        validator.run()

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Walk-forward validation aborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)

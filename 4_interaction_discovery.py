# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 3 — Interaction Term Discovery                            ║
║                                                                            ║
║  Pipeline position:                                                        ║
║    Stage 0 (Preselection) → Stage 1 (Univariate) → Stage 2 (SFFS)         ║
║      → [Stage 3: Interaction Terms]                                        ║
║                                                                            ║
║  Purpose:                                                                  ║
║    Discover synergistic feature interactions that improve predictive power  ║
║    beyond what the SFFS base features achieve alone. For example,           ║
║    LOGRET_5 * CMF_20 captures "volume-confirmed momentum" — a signal       ║
║    that neither feature encodes individually.                               ║
║                                                                            ║
║  Method:                                                                   ║
║    1. Load SFFS final subset from sffs_report.json.                        ║
║    2. Regenerate those features from raw OHLCV (one-time TA cost).         ║
║    3. Create C(N,2) interaction columns (element-wise product).            ║
║    4. Run Forward Selection over the interaction candidates:               ║
║       - Start with the SFFS base features (frozen — never removed).        ║
║       - Greedily add the interaction that maximises Mean IC.               ║
║       - Stop when improvement < threshold OR max_interactions reached.     ║
║    5. Output the final feature set (base + selected interactions).         ║
║                                                                            ║
║  Key constraint: Base features are FROZEN. We only ADD interactions.       ║
║  This prevents the interaction search from destabilising the SFFS core.    ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_2_features/sffs_report.json            (Stage 2)      ║
║    • artifacts/phase_1_data/merged_data.parquet              (Phase 1)     ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/interaction_report.json                    ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)               ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — reads only disk artifacts from prior stages               ║
║    ✓ Deterministic — Ridge has no random state; alphabetical tie-breaking  ║
║    ✓ Atomic — writes to .tmp then renames                                  ║
║    ✓ Efficient — TA regenerated ONCE via joblib parallel;                  ║
║      forward selection loop uses column slicing only                       ║
║    ✓ Traceable — full iteration log with scores at every step              ║
║    ✓ Temporal — TimeSeriesSplit preserves chronological order              ║
║    ✓ Conservative — max 3 interactions to guard against overfitting        ║
║    ✓ Unbiased — StandardScaler normalises all features before Ridge        ║
║                                                                            ║
║  TA Generation Note:                                                       ║
║    Parallel per-ticker via joblib (loky backend, n_jobs=6).                ║
║    loky uses non-daemonic workers → no "daemon child" crash.               ║
║    cores=1 per worker prevents OS thread thrashing.                        ║
║                                                                            ║
║  Version: 1.1.0                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
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
from sklearn.model_selection import TimeSeriesSplit
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
STAGE_NAME = "3_interaction_terms"

PHASE1_ARTIFACT = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase_2_features"
SFFS_REPORT_JSON = PHASE2_ARTIFACT_DIR / "sffs_report.json"
OUTPUT_JSON = PHASE2_ARTIFACT_DIR / "interaction_report.json"

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

# Matched exactly to Stage 1/2 hyperparameters
RIDGE_ALPHA = 1.0
CV_FOLDS = 5
FORWARD_DAYS = 5

# Interaction-specific parameters
MAX_INTERACTIONS = 3               # Conservative cap to prevent overfitting
IC_IMPROVEMENT_THRESHOLD = 0.0005  # Same threshold as SFFS (Stage 2)

# Parallelism parameters
N_PARALLEL_JOBS = 6     # Worker processes (leaves ~5 cores free for OS)
MIN_TICKER_ROWS = 200   # Minimum history required per ticker


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
            name="Phase2_Interaction_Base",
            ta=filtered_specs,
        )

        # CRITICAL: Limit internal threading to 1 per worker process
        ticker_df.ta.cores = 1
        ticker_df.ta.strategy(strategy, verbose=False)

        return (ticker, ticker_df, None)
    except Exception as exc:
        return (ticker, None, str(exc))


# ─── CV Scoring Function ───────────────────────────────────────────────────
def _evaluate_subset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    tscv: TimeSeriesSplit,
    ridge_alpha: float = RIDGE_ALPHA,
) -> Tuple[float, List[float]]:
    """Evaluate a feature subset via StandardScaler→Ridge + TSCV + Spearman IC.

    FIX v1.1.0 — StandardScaler added before Ridge:
        Interaction terms are products of base features, creating extreme
        scale differences. For example:
          SMA_50 × SMA_50 ≈ 400,000,000
          LOGRET_5 × CMF_20 ≈ 0.002
        Without scaling, Ridge's L2 penalty completely distorts coefficients,
        suppressing large interaction terms regardless of predictive signal.
        StandardScaler (mean=0, std=1) is fit only on training data inside
        each CV fold — no data leakage.

    Args:
        df: Pre-computed DataFrame with all features and target.
        feature_cols: Column names forming the candidate subset.
        target_col: Name of the target column.
        tscv: Pre-configured TimeSeriesSplit instance.
        ridge_alpha: Regularisation strength for Ridge.

    Returns:
        Tuple of (mean_ic, fold_ics_list).
        Returns (float('-inf'), []) if evaluation fails.
    """
    cols_needed = feature_cols + [target_col]
    subset = df[cols_needed].dropna()

    if len(subset) < CV_FOLDS * 2:
        return float("-inf"), []

    X = subset[feature_cols].values
    y = subset[target_col].values

    fold_ics: List[float] = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # FIX: StandardScaler → Ridge pipeline removes magnitude bias.
        # Scaler is fit ONLY on train fold (no data leakage).
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha, fit_intercept=True)),
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Spearman IC — identical to Stage 1/2
        if np.std(y_pred) < 1e-15 or np.std(y_test) < 1e-15:
            ic = 0.0
        else:
            ic, _ = spearmanr(y_pred, y_test)
            if np.isnan(ic):
                ic = 0.0

        fold_ics.append(round(float(ic), 6))

    mean_ic = float(np.mean(fold_ics))
    return mean_ic, fold_ics


# ─── Main Class ─────────────────────────────────────────────────────────────
class InteractionTermSelector:
    """Discover interaction terms that improve the SFFS base model.

    Creates C(N,2) pairwise product features from the SFFS subset,
    then greedily adds the best interactions via forward selection with
    the base features frozen.
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._base_features: Optional[List[str]] = None
        self._base_score: Optional[float] = None
        self._interaction_names: Optional[List[str]] = None
        self._filtered_specs: Optional[List[Dict[str, Any]]] = None
        self._iteration_log: List[Dict[str, Any]] = []

    # ── 1. Load Stage 2 (SFFS) results ──────────────────────────────────────
    def load_sffs_results(self) -> Tuple[List[str], float]:
        """Load sffs_report.json and extract the final subset + baseline score.

        Returns:
            Tuple of (base_features_list, baseline_mean_ic).

        Raises:
            FileNotFoundError: If SFFS report is missing.
            AssertionError: If report is malformed.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Loading Stage 2 (SFFS) Results...")
        print(f"{'='*72}")

        if not SFFS_REPORT_JSON.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 2 output not found: {SFFS_REPORT_JSON}\n"
                f"       Run 3_sffs_selection.py before this script."
            )

        with open(SFFS_REPORT_JSON, "r", encoding="utf-8") as f:
            sffs_data = json.load(f)

        assert "final_subset" in sffs_data, (
            f"[SCHEMA VIOLATION] sffs_report.json missing 'final_subset' key.\n"
            f"  Found keys: {list(sffs_data.keys())}"
        )
        assert "final_score_mean_ic" in sffs_data, (
            f"[SCHEMA VIOLATION] sffs_report.json missing 'final_score_mean_ic'."
        )

        base_features = sffs_data["final_subset"]
        base_score = sffs_data["final_score_mean_ic"]

        assert isinstance(base_features, list) and len(base_features) > 0, (
            f"[SCHEMA VIOLATION] final_subset must be a non-empty list, "
            f"got: {base_features}"
        )

        # Warn if the SFFS report appears to be from the unscaled (biased) run
        sffs_version = sffs_data.get("version", "unknown")
        if sffs_version < "1.1.0":
            print(f"  ⚠ WARNING: sffs_report.json was produced by Stage 2 "
                  f"version {sffs_version}.")
            print(f"    This may be the biased (unscaled) version. Re-run "
                  f"Stage 2 (v1.1.0+) for correct results.")

        print(f"  ✓ SFFS report version: {sffs_version}")
        print(f"  ✓ SFFS final subset ({len(base_features)} features):")
        for i, feat in enumerate(base_features, 1):
            print(f"    {i}. {feat}")
        print(f"  ✓ SFFS baseline Mean IC: {base_score:.6f}")

        self._base_features = base_features
        self._base_score = base_score
        return base_features, base_score

    # ── 2. Load Phase 1 data ────────────────────────────────────────────────
    def load_phase1_data(self) -> pd.DataFrame:
        """Load merged OHLCV and validate schema.

        Returns:
            pd.DataFrame: Raw OHLCV sorted by (ticker, date).
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Loading Phase 1 Data...")
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

        Identical formula to Stage 1/2:  target_t = ln(close_{t+5} / close_t)

        Args:
            df: DataFrame sorted by (ticker, date) with 'close' column.

        Returns:
            pd.Series: Named 'target_log_return', index-aligned to df.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Computing Target: {FORWARD_DAYS}-day forward log return")
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

    # ── 4. Regenerate base features ─────────────────────────────────────────
    def regenerate_features(
        self, df: pd.DataFrame, feature_list: List[str]
    ) -> pd.DataFrame:
        """Regenerate ONLY the SFFS base features from raw OHLCV in parallel.

        FIX v1.1.0 — Replaced sequential for-loop with joblib Parallel:
            Original: 750 tickers × ~11s each = ~81 min on 1 core.
            Fixed:    750 tickers / 6 workers  = ~2 min wall-clock time.

        Why loky backend?
            pandas_ta_classic may try to spawn child processes internally.
            Standard multiprocessing uses daemon workers which cannot have
            children (raises RuntimeError). loky uses non-daemonic workers,
            which permits pandas_ta's internal subprocesses.

        Why cores=1 per worker?
            Without this, each of the 6 workers would use all 11 available
            threads → 66 threads competing for 11 cores (OS thrashing).
            Setting ticker_df.ta.cores = 1 inside the worker confines each
            worker to a single thread, keeping total threads ≤ 6.

        Args:
            df: Raw OHLCV DataFrame with (ticker, date, open, high, low,
                close, volume).
            feature_list: SFFS base feature column names to regenerate.

        Returns:
            pd.DataFrame: Input df augmented with regenerated columns.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Regenerating {len(feature_list)} Base Features "
              f"via pandas_ta_classic (parallel, n_jobs={N_PARALLEL_JOBS})")
        print(f"{'='*72}")

        t0 = perf_counter()

        # ── Filter curated_indicators to only needed specs ──────────────────
        all_specs = self.config.curated_indicators
        feature_set = set(feature_list)

        filtered_specs: List[Dict[str, Any]] = []
        for spec in all_specs:
            kind = spec.get("kind", "")
            prefixes = _KIND_TO_PREFIX.get(kind, [])
            produces_needed = any(
                any(feat.startswith(prefix) for feat in feature_set)
                for prefix in prefixes
            )
            if produces_needed:
                filtered_specs.append(spec)

        self._filtered_specs = filtered_specs
        print(f"  ℹ Filtered {len(all_specs)} curated specs → "
              f"{len(filtered_specs)} needed for {len(feature_list)} features")

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
                        f"[Stage 3] pandas_ta failed for '{ticker}': {error}",
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
        matched = [c for c in feature_list if c in generated_cols]
        missing_cols = [c for c in feature_list if c not in generated_cols]

        # Case-insensitive fallback
        if missing_cols:
            col_map = {c.upper(): c for c in generated_cols}
            for m in missing_cols[:]:
                if m.upper() in col_map:
                    actual = col_map[m.upper()]
                    result_df.rename(columns={actual: m}, inplace=True)
                    matched.append(m)
                    missing_cols.remove(m)

        if missing_cols:
            raise RuntimeError(
                f"[FATAL] Cannot regenerate SFFS base features: {missing_cols}\n"
                f"  These are required for interaction term computation.\n"
                f"  Generated columns: {sorted(new_cols)}"
            )

        print(f"  ✓ Matched {len(matched)}/{len(feature_list)} base features")

        t_elapsed = perf_counter() - t0
        print(f"  ✓ Feature regeneration complete ({t_elapsed:.1f}s)")

        return result_df

    # ── 5. Create interaction columns ───────────────────────────────────────
    def create_interaction_columns(
        self, df: pd.DataFrame, base_features: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Create C(N,2) pairwise interaction columns via element-wise product.

        Interaction formula:  IX_{A_B} = A * B  (element-wise)

        This captures non-linear relationships that Ridge cannot model
        from the individual features alone. For example:
          - LOGRET_5 * CMF_20  → "volume-confirmed momentum"
          - ATRr_14 * ADX_14   → "volatility-adjusted trend strength"

        StandardScaler in _evaluate_subset handles the extreme scale
        differences that arise from multiplying already-large features
        (e.g. SMA_50 × SMA_50 ≈ 400,000,000).

        Args:
            df: DataFrame with base feature columns.
            base_features: Ordered list of SFFS base feature names.

        Returns:
            Tuple of (augmented_df, interaction_column_names).
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Creating Interaction Terms")
        print(f"{'='*72}")

        t0 = perf_counter()

        # Generate all C(N,2) pairs in deterministic order
        pairs = list(combinations(sorted(base_features), 2))

        interaction_names: List[str] = []
        pair_descriptions: List[Dict[str, str]] = []

        for feat_a, feat_b in pairs:
            # Naming convention: IX_{featureA}_x_{featureB}
            ix_name = f"IX_{feat_a}_x_{feat_b}"
            df[ix_name] = df[feat_a] * df[feat_b]
            interaction_names.append(ix_name)
            pair_descriptions.append({
                "name": ix_name,
                "feature_a": feat_a,
                "feature_b": feat_b,
            })

        t_elapsed = perf_counter() - t0

        print(f"  ✓ Created {len(interaction_names)} interaction columns "
              f"from C({len(base_features)},2) pairs  ({t_elapsed:.3f}s)")
        print(f"  Interactions:")
        for desc in pair_descriptions:
            print(f"    • {desc['name']}")

        self._interaction_names = interaction_names
        return df, interaction_names

    # ── 6. Forward selection over interactions ──────────────────────────────
    def run_interaction_selection(
        self, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Forward selection: greedily add interactions to the frozen base.

        The SFFS base features are FROZEN — they are always included.
        We only search over which interaction columns to ADD.

        Algorithm:
            1. Compute baseline score: StandardScaler→Ridge(base_features).
            2. For each remaining interaction:
                - Evaluate Pipeline(base_features + selected + [ix]).
                - Track best ix by Mean IC.
            3. If best ix improves score by ≥ threshold, add it.
            4. Stop when: no improvement, or max_interactions reached.

        Determinism:
            - Remaining interactions iterated in sorted order.
            - Ties broken alphabetically.

        Args:
            df: DataFrame with base features, interactions, and target.
                Must be sorted by (ticker, date).

        Returns:
            Dict with selected interactions, scores, and full iteration log.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Running Interaction Forward Selection")
        print(f"{'='*72}")
        print(f"  ℹ Base features (frozen): {self._base_features}")
        print(f"  ℹ Interaction candidates : {len(self._interaction_names)}")
        print(f"  ℹ Max interactions       : {MAX_INTERACTIONS}")
        print(f"  ℹ Model                  : StandardScaler → Ridge(alpha={RIDGE_ALPHA})")
        print(f"  ℹ CV                     : TimeSeriesSplit(n_splits={CV_FOLDS})")
        print(f"  ℹ Stop threshold         : Δ IC < {IC_IMPROVEMENT_THRESHOLD}")

        t0 = perf_counter()

        target_col = "target_log_return"
        assert target_col in df.columns, (
            f"[FATAL] Target column '{target_col}' not in DataFrame."
        )

        # Sort for temporal integrity
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        base_features = list(self._base_features)

        # ── Step 0: Verify baseline score ───────────────────────────────────
        # Re-evaluate base features locally to get a consistent baseline
        # (uses the same StandardScaler→Ridge pipeline as the search)
        baseline_ic, baseline_folds = _evaluate_subset(
            df, base_features, target_col, tscv, RIDGE_ALPHA
        )
        print(f"\n  ✓ Verified baseline Mean IC: {baseline_ic:.6f}")
        print(f"    (Stage 2 reported: {self._base_score:.6f})")

        if abs(baseline_ic - self._base_score) > 0.005:
            print(f"  ⚠ WARNING: Baseline mismatch > 0.005. "
                  f"Proceeding with locally computed value.")

        self._iteration_log.append({
            "iteration": 0,
            "step": "BASELINE",
            "features": list(base_features),
            "score_mean_ic": round(baseline_ic, 6),
            "fold_ics": baseline_folds,
            "sffs_reported_ic": self._base_score,
        })

        # ── Forward selection loop ──────────────────────────────────────────
        selected_interactions: List[str] = []
        remaining_interactions: List[str] = sorted(self._interaction_names)
        best_score: float = baseline_ic

        for iteration in range(1, MAX_INTERACTIONS + 1):
            if not remaining_interactions:
                self._iteration_log.append({
                    "iteration": iteration,
                    "step": "STOP",
                    "reason": "No remaining interaction candidates.",
                })
                print(f"\n  ▮ Iteration {iteration}: STOP — "
                      f"no remaining candidates")
                break

            # Try each remaining interaction
            candidates: List[Tuple[str, float, List[float]]] = []

            current_features = base_features + selected_interactions

            for ix_name in sorted(remaining_interactions):
                test_features = current_features + [ix_name]
                mean_ic, fold_ics = _evaluate_subset(
                    df, test_features, target_col, tscv, RIDGE_ALPHA
                )
                candidates.append((ix_name, mean_ic, fold_ics))

            # Sort: highest Mean IC first, alphabetical tie-breaking
            candidates.sort(key=lambda x: (-x[1], x[0]))
            best_ix, best_ix_score, best_ix_folds = candidates[0]

            improvement = best_ix_score - best_score

            # Check stopping criterion
            if improvement < IC_IMPROVEMENT_THRESHOLD:
                self._iteration_log.append({
                    "iteration": iteration,
                    "step": "STOP",
                    "reason": (
                        f"Improvement ({improvement:.6f}) < "
                        f"threshold ({IC_IMPROVEMENT_THRESHOLD})"
                    ),
                    "best_candidate": best_ix,
                    "candidate_score": round(best_ix_score, 6),
                    "current_best_score": round(best_score, 6),
                    "improvement": round(improvement, 6),
                    "all_candidate_scores": {
                        c[0]: round(c[1], 6) for c in candidates
                    },
                })
                print(f"\n  ▮ Iteration {iteration}: STOP")
                print(f"    Best candidate: {best_ix} → "
                      f"{best_ix_score:.6f}  (Δ = {improvement:+.6f})")
                print(f"    Improvement below threshold.")
                break

            # Accept the interaction
            selected_interactions.append(best_ix)
            remaining_interactions.remove(best_ix)
            best_score = best_ix_score

            self._iteration_log.append({
                "iteration": iteration,
                "step": "FORWARD",
                "added": best_ix,
                "score_after_add": round(best_ix_score, 6),
                "fold_ics": best_ix_folds,
                "improvement": round(improvement, 6),
                "selected_interactions": list(selected_interactions),
                "total_features": len(base_features) + len(selected_interactions),
                "remaining_candidates": len(remaining_interactions),
                "all_candidate_scores": {
                    c[0]: round(c[1], 6) for c in candidates
                },
            })

            # Decode the interaction for human readability
            parts = best_ix.replace("IX_", "").split("_x_")
            signal_desc = f"{parts[0]} × {parts[1]}" if len(parts) == 2 else best_ix

            print(f"\n  ▶ Iteration {iteration} — FORWARD")
            print(f"    Added: {best_ix}")
            print(f"    Signal: {signal_desc}")
            print(f"    Score: {best_ix_score:.6f}  (Δ = +{improvement:.6f})")
            print(f"    Total features: "
                  f"{len(base_features)} base + "
                  f"{len(selected_interactions)} interactions = "
                  f"{len(base_features) + len(selected_interactions)}")

        total_time = perf_counter() - t0

        # ── Final feature set ───────────────────────────────────────────────
        final_features = base_features + selected_interactions
        final_ic = best_score

        # Compute IC improvement over locally-verified baseline
        ic_gain_over_sffs = final_ic - baseline_ic
        ic_gain_pct = (
            (ic_gain_over_sffs / baseline_ic * 100)
            if baseline_ic > 0 else 0.0
        )

        # ── Assemble output ─────────────────────────────────────────────────
        output = {
            "stage": STAGE_NAME,
            "version": VERSION,
            "algorithm": "Forward Selection over Pairwise Interaction Terms",
            "config_snapshot": {
                "ridge_alpha": RIDGE_ALPHA,
                "cv_folds": CV_FOLDS,
                "forward_days": FORWARD_DAYS,
                "max_interactions": MAX_INTERACTIONS,
                "ic_improvement_threshold": IC_IMPROVEMENT_THRESHOLD,
                "base_features_frozen": list(base_features),
                "interaction_candidates_total": len(self._interaction_names),
                "curated_specs_used": len(self._filtered_specs)
                    if self._filtered_specs else 0,
                "scaler": "StandardScaler",
                "n_parallel_jobs": N_PARALLEL_JOBS,
                "parallel_backend": "loky",
            },
            "sffs_baseline": {
                "features": list(base_features),
                "feature_count": len(base_features),
                "mean_ic_reported": self._base_score,
                "mean_ic_verified": round(baseline_ic, 6),
            },
            "interaction_candidates": [
                {
                    "name": ix,
                    "feature_a": ix.replace("IX_", "").split("_x_")[0],
                    "feature_b": ix.replace("IX_", "").split("_x_")[1]
                        if "_x_" in ix else "?",
                }
                for ix in sorted(self._interaction_names)
            ],
            "selected_interactions": list(selected_interactions),
            "selected_interaction_count": len(selected_interactions),
            "final_feature_set": list(final_features),
            "final_feature_count": len(final_features),
            "final_score_mean_ic": round(final_ic, 6),
            "improvement_over_sffs": {
                "absolute_ic_gain": round(ic_gain_over_sffs, 6),
                "relative_gain_pct": round(ic_gain_pct, 2),
            },
            "iterations": self._iteration_log,
            "runtime_seconds": round(total_time, 2),
        }

        # ── Print summary ───────────────────────────────────────────────────
        print(f"\n  {'═'*60}")
        print(f"  INTERACTION SELECTION RESULT:")
        print(f"  {'═'*60}")
        print(f"  Base features (frozen, from SFFS):")
        for i, feat in enumerate(base_features, 1):
            print(f"    {i}. {feat}")
        if selected_interactions:
            print(f"  Selected interactions:")
            for i, ix in enumerate(selected_interactions, 1):
                parts = ix.replace("IX_", "").split("_x_")
                desc = f"{parts[0]} × {parts[1]}" if len(parts) == 2 else ix
                print(f"    +{i}. {ix}  ({desc})")
        else:
            print(f"  No interactions improved the model.")
        print(f"  ──────────────────────────────────────────────")
        print(f"  SFFS baseline IC  : {baseline_ic:.6f}")
        print(f"  Final IC          : {final_ic:.6f}")
        print(f"  IC gain           : {ic_gain_over_sffs:+.6f} "
              f"({ic_gain_pct:+.2f}%)")
        print(f"  Total features    : {len(final_features)}")
        print(f"  Runtime           : {total_time:.2f}s")
        print(f"  {'═'*60}")

        return output

    # ── 7. Save results atomically ──────────────────────────────────────────
    def save_results(self, results: Dict[str, Any]) -> None:
        """Write interaction_report.json atomically (.tmp → rename).

        Also completes the Auditor lifecycle.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 3] Saving Results")
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
            summary_df = pd.DataFrame([{
                "final_feature_count": results["final_feature_count"],
                "final_score_mean_ic": results["final_score_mean_ic"],
                "selected_interaction_count": results["selected_interaction_count"],
                "runtime_seconds": results["runtime_seconds"],
            }])
            self.auditor.record_output(summary_df, self.config.to_snapshot())
            self.auditor.success()
            print(f"  ✓ manifest.json (via Auditor)")
        except Exception as e:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {e}")

    # ── Orchestrator ────────────────────────────────────────────────────────
    def run(self) -> None:
        """Full pipeline: SFFS → Data → Target → Features → Interactions → Select → Save."""
        pipeline_t0 = perf_counter()

        # Step 1: Load SFFS results → extract base features + baseline IC
        base_features, base_score = self.load_sffs_results()

        # Step 2: Load Phase 1 raw OHLCV
        df = self.load_phase1_data()

        # Step 3: Compute target
        target = self.compute_target(df)
        df["target_log_return"] = target

        # Step 4: Regenerate ONLY the base features (parallel, one-time TA cost)
        df = self.regenerate_features(df, base_features)

        # Step 5: Create C(N,2) interaction columns (instant — column math)
        df, interaction_names = self.create_interaction_columns(df, base_features)

        # Step 6: Forward selection over interactions (StandardScaler→Ridge)
        results = self.run_interaction_selection(df)

        # Step 7: Save atomically
        self.save_results(results)

        total_time = perf_counter() - pipeline_t0
        print(f"\n{'='*72}")
        print(f"[Stage 3] COMPLETE — Total runtime: {total_time:.1f}s")
        print(f"{'='*72}\n")


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 3 — Interaction Term Discovery            ║")
    print(f"║  Version: {VERSION:<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        config = Phase2Config()

        selector = InteractionTermSelector(config=config)
        selector.run()

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Interaction selection aborted.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)

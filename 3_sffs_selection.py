# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 2 — SFFS (Sequential Forward Floating Selection)          ║
║                                                                            ║
║  Pipeline position:                                                        ║
║    Phase 1 (Clean OHLCV) → Stage 0 (Preselection) → Stage 1 (Univariate)  ║
║      → [Stage 2: SFFS Multivariate Selection]                              ║
║                                                                            ║
║  Purpose:                                                                  ║
║    Transition from univariate analysis (features evaluated independently)  ║
║    to multivariate selection (finding the best COMBINATION of features).   ║
║    SFFS allows features added in earlier iterations to be removed if a     ║
║    later addition makes them redundant — unlike plain greedy forward.      ║
║                                                                            ║
║  Algorithm: SFFS (Pudil et al., 1994)                                      ║
║    1. Start with empty set S = ∅.                                          ║
║    2. FORWARD: Try adding each remaining feature. Keep f* that maximises   ║
║       Mean IC of StandardScaler→Ridge(S ∪ {f*}) under 5-fold TSCV.        ║
║    3. If improvement < threshold (0.0005), STOP.                           ║
║    4. BACKWARD (floating): If |S| ≥ 2, try removing each feature in S     ║
║       except f*. If removing s_worst improves or maintains score with      ║
║       fewer features, remove s_worst. Repeat backward until no removal     ║
║       helps.                                                               ║
║    5. Go to step 2.                                                        ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_2_features/individual_feature_scores.json  (Stage 1)  ║
║    • artifacts/phase_1_data/merged_data.parquet                 (Phase 1)  ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/sffs_report.json                           ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)               ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — reads only disk artifacts from prior stages               ║
║    ✓ Deterministic — Ridge has no random state; alphabetical tie-breaking  ║
║    ✓ Atomic — writes to .tmp then renames                                  ║
║    ✓ Efficient — TA indicators regenerated ONCE via joblib parallel;       ║
║      SFFS loop uses column slicing only (no re-computation inside loop)    ║
║    ✓ Traceable — full iteration log with scores at every step              ║
║    ✓ Temporal — TimeSeriesSplit preserves chronological order              ║
║    ✓ Unbiased — StandardScaler normalises features before Ridge fitting    ║
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
STAGE_NAME = "2_sffs_selection"

PHASE1_ARTIFACT = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase_2_features"
STAGE1_SCORES_JSON = PHASE2_ARTIFACT_DIR / "individual_feature_scores.json"
OUTPUT_JSON = PHASE2_ARTIFACT_DIR / "sffs_report.json"

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

# Matched exactly to Stage 1 (2_individual_evaluation.py lines 83-85)
RIDGE_ALPHA = 1.0
CV_FOLDS = 5
FORWARD_DAYS = 5

# SFFS parameters
TOP_N_FEATURES = 10                  # Number of top features to feed into SFFS
IC_IMPROVEMENT_THRESHOLD = 0.0005   # Minimum Mean IC gain for forward step

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
            name="Phase2_SFFS_Top10",
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
        Raw feature magnitudes vary wildly (e.g. SMA_50 ≈ 20,000 vs
        LOGRET_5 ≈ 0.05). Without scaling, Ridge's L2 penalty suppresses
        large-magnitude features regardless of their predictive value.
        StandardScaler (mean=0, std=1) removes this artefact so that Ridge
        penalises all features equally by their *information content*, not
        their units.

    This is the innermost scoring function called O(N²) times by SFFS.
    It must be fast: no copies, no TA generation — just column slicing.

    Args:
        df: Pre-computed DataFrame with all features and target (no NaN rows).
        feature_cols: List of column names forming the candidate subset.
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

        # Spearman IC — identical logic to Stage 1
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
class SFFSSelector:
    """Sequential Forward Floating Selection for multivariate feature subset.

    Finds the optimal combination of the Top 10 features (from Stage 1)
    by iteratively adding features that maximise multivariate Mean IC,
    with a backward step that removes features made redundant by later
    additions.
    """

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._top_n_features: Optional[List[str]] = None
        self._filtered_specs: Optional[List[Dict[str, Any]]] = None
        self._iteration_log: List[Dict[str, Any]] = []

    # ── 1. Load Stage 1 scores and extract Top N ────────────────────────────
    def load_stage1_scores(self) -> List[str]:
        """Load individual_feature_scores.json and extract Top N by IR.

        Determinism: Ties broken alphabetically (sorted feature name).

        Returns:
            List of Top N feature names, ordered by IR descending.

        Raises:
            FileNotFoundError: If Stage 1 output is missing.
            AssertionError: If Stage 1 output is malformed.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 2] Loading Stage 1 Scores...")
        print(f"{'='*72}")

        if not STAGE1_SCORES_JSON.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 1 output not found: {STAGE1_SCORES_JSON}\n"
                f"       Run 2_individual_evaluation.py before this script."
            )

        with open(STAGE1_SCORES_JSON, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)

        assert "features" in stage1_data, (
            f"[SCHEMA VIOLATION] individual_feature_scores.json missing 'features' key.\n"
            f"  Found keys: {list(stage1_data.keys())}"
        )

        features = stage1_data["features"]
        print(f"  ✓ Loaded {len(features)} feature scores from Stage 1")

        # Filter to scorable features (status == "OK", non-null IR)
        scorable: List[Tuple[str, float, float]] = []
        for fname, fdata in features.items():
            if fdata.get("status") != "OK":
                continue
            ir = fdata.get("ir")
            mean_ic = fdata.get("mean_ic")
            if ir is not None and mean_ic is not None and np.isfinite(ir):
                scorable.append((fname, ir, mean_ic))

        # Sort by IR descending, then mean_ic descending, then name ascending
        scorable.sort(key=lambda x: (-x[1], -x[2], x[0]))

        top_n = scorable[:TOP_N_FEATURES]

        print(f"  ✓ Top {len(top_n)} features (by IR):")
        print(f"  {'Rank':<6}{'Feature':<25}{'IR':>10}{'Mean IC':>10}")
        print(f"  {'─'*51}")
        for rank, (fname, ir, mean_ic) in enumerate(top_n, 1):
            print(f"  {rank:<6}{fname:<25}{ir:>10.4f}{mean_ic:>10.6f}")

        self._top_n_features = [t[0] for t in top_n]
        return self._top_n_features

    # ── 2. Load Phase 1 data ────────────────────────────────────────────────
    def load_phase1_data(self) -> pd.DataFrame:
        """Load merged OHLCV from Phase 1 and validate schema.

        Returns:
            pd.DataFrame: Raw OHLCV data sorted by (ticker, date).
        """
        print(f"\n{'='*72}")
        print(f"[Stage 2] Loading Phase 1 Data...")
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

        Identical formula to Stage 1:  target_t = ln(close_{t+5} / close_t)
        Computed per-ticker to prevent cross-contamination at boundaries.

        Args:
            df: DataFrame sorted by (ticker, date) with 'close' column.

        Returns:
            pd.Series: Named 'target_log_return', index-aligned to df.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 2] Computing Target: {FORWARD_DAYS}-day forward log return")
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

    # ── 4. Regenerate Top N features ────────────────────────────────────────
    def regenerate_features(
        self, df: pd.DataFrame, feature_list: List[str]
    ) -> pd.DataFrame:
        """Regenerate ONLY the Top N features from raw OHLCV in parallel.

        FIX v1.1.0 — Replaced sequential for-loop with joblib Parallel:
            Original: 750 tickers × ~11s each = ~82 min on 1 core.
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
            feature_list: Top N feature column names to regenerate.

        Returns:
            pd.DataFrame: Input df augmented with regenerated feature columns.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 2] Regenerating {len(feature_list)} Features "
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
        # verbose=0: suppress joblib's internal progress output; tqdm below
        # handles user-facing progress.
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
                        f"[Stage 2] pandas_ta failed for '{ticker}': {error}",
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

        # ── Verify Top N columns are present ────────────────────────────────
        generated_cols = set(result_df.columns)
        matched = [c for c in feature_list if c in generated_cols]
        missing = [c for c in feature_list if c not in generated_cols]

        # Case-insensitive fallback
        if missing:
            col_map = {c.upper(): c for c in generated_cols}
            for m in missing[:]:
                if m.upper() in col_map:
                    actual = col_map[m.upper()]
                    result_df.rename(columns={actual: m}, inplace=True)
                    matched.append(m)
                    missing.remove(m)

        if missing:
            print(f"  ⚠ WARNING: {len(missing)} features could not be "
                  f"regenerated: {missing}")
            print(f"    These will be excluded from SFFS.")

        print(f"  ✓ Matched {len(matched)}/{len(feature_list)} features")

        t_elapsed = perf_counter() - t0
        print(f"  ✓ Feature regeneration complete ({t_elapsed:.1f}s)")

        # Update feature list to only matched
        self._top_n_features = matched

        return result_df

    # ── 5. SFFS Core Algorithm ──────────────────────────────────────────────
    def run_sffs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute Sequential Forward Floating Selection.

        CRITICAL PERFORMANCE NOTE:
            TA indicators were regenerated ONCE in step 4. This loop
            performs ONLY column slicing (df[subset]) and Pipeline fitting.
            No indicator recomputation occurs inside the loop.

        Determinism guarantees:
            - Ridge(alpha=1.0) is deterministic (no random_state needed)
            - TimeSeriesSplit is deterministic (no shuffle)
            - StandardScaler is deterministic
            - Alphabetical tie-breaking when features have equal Mean IC
            - Remaining features iterated in sorted order

        Args:
            df: DataFrame with all Top N features + target column.
                Must be sorted by (ticker, date).

        Returns:
            Dict with final_subset, final_score, and full iteration log.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 2] Running SFFS (Sequential Forward Floating Selection)")
        print(f"{'='*72}")
        print(f"  ℹ Candidates     : {len(self._top_n_features)} features")
        print(f"  ℹ Model          : StandardScaler → Ridge(alpha={RIDGE_ALPHA})")
        print(f"  ℹ CV             : TimeSeriesSplit(n_splits={CV_FOLDS})")
        print(f"  ℹ Metric         : Mean Spearman IC")
        print(f"  ℹ Stop threshold : Δ IC < {IC_IMPROVEMENT_THRESHOLD}")

        t0 = perf_counter()

        target_col = "target_log_return"
        assert target_col in df.columns, (
            f"[FATAL] Target column '{target_col}' not in DataFrame."
        )

        # Sort for temporal integrity
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Pre-configure TimeSeriesSplit (reused for every _evaluate_subset call)
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        selected: List[str] = []
        remaining: List[str] = sorted(self._top_n_features)  # Alphabetical
        best_score: float = float("-inf")

        iteration = 0

        while remaining:
            iteration += 1
            iter_t0 = perf_counter()

            # ════════════════════════════════════════════════════════════════
            # FORWARD STEP: Find the best feature to add
            # ════════════════════════════════════════════════════════════════
            forward_results: List[Tuple[str, float, List[float]]] = []

            for f in sorted(remaining):  # Sorted for deterministic iteration
                candidate_set = selected + [f]
                mean_ic, fold_ics = _evaluate_subset(
                    df, candidate_set, target_col, tscv, RIDGE_ALPHA
                )
                forward_results.append((f, mean_ic, fold_ics))

            # Select f_best: highest Mean IC, alphabetical tie-breaking
            forward_results.sort(key=lambda x: (-x[1], x[0]))
            f_best, f_best_score, f_best_folds = forward_results[0]

            # Check stopping criterion
            improvement = f_best_score - best_score
            if improvement < IC_IMPROVEMENT_THRESHOLD:
                stop_reason = (
                    f"Forward step improvement ({improvement:.6f}) < "
                    f"threshold ({IC_IMPROVEMENT_THRESHOLD})"
                )
                self._iteration_log.append({
                    "iteration": iteration,
                    "step": "STOP",
                    "reason": stop_reason,
                    "best_candidate": f_best,
                    "candidate_score": round(f_best_score, 6),
                    "current_best_score": round(best_score, 6),
                    "improvement": round(improvement, 6),
                    "selected_features": list(selected),
                    "selected_count": len(selected),
                })
                print(f"\n  ▮ Iteration {iteration}: STOP")
                print(f"    {stop_reason}")
                break

            # Accept f_best
            selected.append(f_best)
            remaining.remove(f_best)
            best_score = f_best_score

            forward_log = {
                "iteration": iteration,
                "step": "FORWARD",
                "added": f_best,
                "score_after_add": round(f_best_score, 6),
                "fold_ics": f_best_folds,
                "improvement": round(improvement, 6),
                "selected_features": list(selected),
                "selected_count": len(selected),
                "remaining_count": len(remaining),
            }
            self._iteration_log.append(forward_log)

            print(f"\n  ▶ Iteration {iteration} — FORWARD")
            print(f"    Added: {f_best}")
            print(f"    Score: {f_best_score:.6f}  "
                  f"(Δ = +{improvement:.6f})")
            print(f"    Selected ({len(selected)}): {selected}")

            # ════════════════════════════════════════════════════════════════
            # BACKWARD STEP (Floating): Try removing features made redundant
            # ════════════════════════════════════════════════════════════════
            while len(selected) >= 2:
                backward_results: List[Tuple[str, float, List[float]]] = []

                for s in sorted(selected):
                    if s == f_best:
                        continue  # Never remove the feature just added
                    test_set = [feat for feat in selected if feat != s]
                    mean_ic, fold_ics = _evaluate_subset(
                        df, test_set, target_col, tscv, RIDGE_ALPHA
                    )
                    backward_results.append((s, mean_ic, fold_ics))

                if not backward_results:
                    break

                # Sort: highest score when removed = worst feature to keep
                backward_results.sort(key=lambda x: (-x[1], x[0]))
                s_worst, score_without, folds_without = backward_results[0]

                # Remove if score improves or is maintained (Occam's razor)
                score_maintained = (score_without >= best_score - 1e-7)

                if score_without > best_score or score_maintained:
                    selected.remove(s_worst)
                    remaining.append(s_worst)
                    remaining.sort()  # Keep remaining sorted

                    if score_without > best_score:
                        best_score = score_without

                    backward_log = {
                        "iteration": iteration,
                        "step": "BACKWARD",
                        "removed": s_worst,
                        "score_after_remove": round(score_without, 6),
                        "fold_ics": folds_without,
                        "score_change": round(
                            score_without - f_best_score, 6
                        ),
                        "selected_features": list(selected),
                        "selected_count": len(selected),
                        "remaining_count": len(remaining),
                    }
                    self._iteration_log.append(backward_log)

                    print(f"  ◀ Iteration {iteration} — BACKWARD")
                    print(f"    Removed: {s_worst}")
                    print(f"    Score: {score_without:.6f}")
                    print(f"    Selected ({len(selected)}): {selected}")

                    # Update f_best_score for next backward check
                    f_best_score = score_without
                else:
                    # No beneficial removal found — exit backward loop
                    break

            iter_elapsed = perf_counter() - iter_t0
            print(f"    Iteration time: {iter_elapsed:.2f}s")

            # Safety: stop if remaining is empty
            if not remaining:
                self._iteration_log.append({
                    "iteration": iteration + 1,
                    "step": "STOP",
                    "reason": "All features consumed (remaining_features empty)",
                    "selected_features": list(selected),
                    "selected_count": len(selected),
                })
                print(f"\n  ▮ STOP: All features consumed")
                break

        total_time = perf_counter() - t0

        # ── Assemble output ─────────────────────────────────────────────────
        output = {
            "stage": STAGE_NAME,
            "version": VERSION,
            "algorithm": "SFFS (Sequential Forward Floating Selection)",
            "config_snapshot": {
                "ridge_alpha": RIDGE_ALPHA,
                "cv_folds": CV_FOLDS,
                "forward_days": FORWARD_DAYS,
                "top_n_input": TOP_N_FEATURES,
                "ic_improvement_threshold": IC_IMPROVEMENT_THRESHOLD,
                "input_features": sorted(self._top_n_features),
                "curated_specs_used": len(self._filtered_specs)
                    if self._filtered_specs else 0,
                "scaler": "StandardScaler",
                "n_parallel_jobs": N_PARALLEL_JOBS,
                "parallel_backend": "loky",
            },
            "final_subset": list(selected),
            "final_subset_count": len(selected),
            "final_score_mean_ic": round(best_score, 6),
            "total_iterations": iteration,
            "iterations": self._iteration_log,
            "runtime_seconds": round(total_time, 2),
        }

        # ── Print final summary ─────────────────────────────────────────────
        print(f"\n  {'═'*60}")
        print(f"  SFFS RESULT:")
        print(f"  {'═'*60}")
        print(f"  Final subset ({len(selected)} features):")
        for i, feat in enumerate(selected, 1):
            print(f"    {i}. {feat}")
        print(f"  Final Mean IC   : {best_score:.6f}")
        print(f"  Total iterations: {iteration}")
        print(f"  Runtime         : {total_time:.2f}s")
        print(f"  {'═'*60}")

        return output

    # ── 6. Save results atomically ──────────────────────────────────────────
    def save_results(self, results: Dict[str, Any]) -> None:
        """Write sffs_report.json atomically (.tmp → rename).

        Also completes the Auditor lifecycle.
        """
        print(f"\n{'='*72}")
        print(f"[Stage 2] Saving Results")
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
                "final_subset_count": results["final_subset_count"],
                "final_score_mean_ic": results["final_score_mean_ic"],
                "total_iterations": results["total_iterations"],
                "runtime_seconds": results["runtime_seconds"],
            }])
            self.auditor.record_output(summary_df, self.config.to_snapshot())
            self.auditor.success()
            print(f"  ✓ manifest.json (via Auditor)")
        except Exception as e:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {e}")

    # ── Orchestrator ────────────────────────────────────────────────────────
    def run(self) -> None:
        """Full pipeline: Scores → Data → Target → Features → SFFS → Save."""
        pipeline_t0 = perf_counter()

        # Step 1: Load Stage 1 scores → extract Top 10
        top_features = self.load_stage1_scores()

        # Step 2: Load Phase 1 raw OHLCV
        df = self.load_phase1_data()

        # Step 3: Compute target
        target = self.compute_target(df)
        df["target_log_return"] = target

        # Step 4: Regenerate ONLY Top N features (parallel, one-time TA cost)
        df = self.regenerate_features(df, top_features)

        # Step 5: Run SFFS (column slicing + StandardScaler→Ridge pipeline)
        results = self.run_sffs(df)

        # Step 6: Save atomically
        self.save_results(results)

        total_time = perf_counter() - pipeline_t0
        print(f"\n{'='*72}")
        print(f"[Stage 2] COMPLETE — Total runtime: {total_time:.1f}s")
        print(f"{'='*72}\n")


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 2 — SFFS Multivariate Feature Selection   ║")
    print(f"║  Version: {VERSION:<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    try:
        config = Phase2Config()

        selector = SFFSSelector(config=config)
        selector.run()

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] SFFS aborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)

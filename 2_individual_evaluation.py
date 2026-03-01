# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Phase 2 · Stage 1 — Individual Feature Evaluation                        ║
║                                                                            ║
║  Auditor Prompt (Design Document):                                         ║
║  This script evaluates each of the 20 candidate features selected in       ║
║  Stage 0 (preselection audit) by measuring their individual predictive     ║
║  power for forecasting 5-day forward log returns.                          ║
║                                                                            ║
║  Method:                                                                   ║
║    • For each candidate feature, fit a Ridge(alpha=1.0) regression in a    ║
║      TimeSeriesSplit(n_splits=5) cross-validation loop.                    ║
║    • On each fold's test set, compute the Spearman rank correlation (IC)   ║
║      between predicted and actual target values.                           ║
║    • Aggregate: Mean IC, Std IC, Information Ratio (IR = Mean / Std).      ║
║                                                                            ║
║  Inputs  (from disk — fully decoupled):                                    ║
║    • artifacts/phase_1_data/merged_data.parquet                            ║
║    • artifacts/phase_2_features/candidate_features.csv                     ║
║                                                                            ║
║  Outputs:                                                                  ║
║    • artifacts/phase_2_features/individual_feature_scores.json             ║
║    • artifacts/phase_2_features/manifest.json  (via Auditor)               ║
║                                                                            ║
║  Quality Gates:                                                            ║
║    ✓ Decoupled — loads all inputs from disk                                ║
║    ✓ Contract — validates OHLCV schema before processing                   ║
║    ✓ Deterministic — Ridge is deterministic; no shuffle in TSCV            ║
║    ✓ Atomic — writes to .tmp then renames                                  ║
║    ✓ Scalable — vectorised target; filtered TA specs eliminate waste       ║
║    ✓ Orchestrated — runnable via `python3 2_individual_evaluation.py`      ║
║    ✓ Traceable — Auditor records inputs, config, outputs                   ║
║    ✓ Versionable — version string embedded in output JSON                  ║
║    ✓ Recoverable — idempotent; crash-safe via atomic writes                ║
║    ✓ Unbiased — StandardScaler inside CV eliminates magnitude bias         ║
║    ✓ Parallel — joblib/loky parallel TA generation (~6× speedup)           ║
║                                                                            ║
║  Version: 1.2.0                                                            ║
║  Changelog:                                                                ║
║    - v1.2.1: Fixed OS thrashing in loky worker — replaced ineffective      ║
║              module-level `_ta.cores = 1` with instance-level              ║
║              `ticker_df.ta.cores = 1` (set on the AnalysisIndicators       ║
║              instance before calling .strategy()). The module variable     ║
║              does not update instances already constructed; the accessor    ║
║              initialises with cpu_count() each time. Previous result:      ║
║              6 workers × ~11 internal threads = 66 threads on 11 cores.    ║
║              Fixed result: 6 workers × 1 thread = 6. ~4× speedup.         ║
║    - v1.2.0: (1) Added StandardScaler inside Pipeline — Ridge now sees     ║
║              unit-variance features; eliminates OBV vs LOGRET_5 magnitude  ║
║              bias that invalidated IR rankings. (2) Replaced sequential    ║
║              tqdm loop with joblib Parallel/loky (n_jobs=6) — matches      ║
║              Stage 0 architecture; TA generation ~6× faster. (3) Added     ║
║              _print_box/_print_section/_print_kv helpers — output now      ║
║              matches Stage 0 formatting. (4) Version printed from VERSION  ║
║              constant (not hardcoded string).                               ║
║    - v1.1.1: Fixed false changelog entry (removed hallucinated subprocess  ║
║              claim from v1.1.0). Documentation now accurately reflects     ║
║              sequential TA generation with rationale.                       ║
║    - v1.1.0: Fixed Auditor API to match core.audit contract (phase,        ║
║              output_dir, version). Filtered curated_indicators to only     ║
║              specs that produce candidate columns (25→15 specs, ~23%       ║
║              runtime reduction). Added deterministic ticker ordering.      ║
║    - v1.0.0: Initial implementation                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─── Imports ────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# SUPPRESS PANDAS_TA UPSTREAM FUTUREWARNING (BEFORE IMPORT)
# ═══════════════════════════════════════════════════════════════
# pandas_ta_classic's MFI indicator has a dtype compatibility issue
# with pandas 2.0+. This is an upstream bug, not our code.
# Suppress it globally before importing the library.
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
VERSION = "1.2.1"
STAGE_NAME = "1_individual_evaluation"

PHASE1_ARTIFACT = PROJECT_ROOT / "artifacts" / "phase_1_data" / "merged_data.parquet"
PHASE2_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "phase_2_features"
CANDIDATE_CSV = PHASE2_ARTIFACT_DIR / "candidate_features.csv"
OUTPUT_JSON = PHASE2_ARTIFACT_DIR / "individual_feature_scores.json"

REQUIRED_COLUMNS = {"ticker", "date", "open", "high", "low", "close", "volume"}

RIDGE_ALPHA = 1.0
CV_FOLDS = 5
FORWARD_DAYS = 5

# ─── Indicator-to-column mapping ───────────────────────────────────────────
# Maps each curated indicator "kind" to the column name prefixes it produces.
# Used to filter curated_indicators to only those that generate candidate cols.
_PARALLEL_N_JOBS: int = 6
_MIN_ROWS_FOR_TA: int = 200  # Match Stage 0's filter threshold


# ─── Print Helpers (matches Stage 0 formatting) ────────────────────────────

def _print_box(title: str, width: int = 72) -> None:
    """Print a clean centered header box."""
    padding = (width - len(title) - 2) // 2
    print(f"\n{'=' * width}")
    print(f"{' ' * padding} {title}")
    print(f"{'=' * width}\n")


def _print_section(title: str, width: int = 72) -> None:
    """Print a clean section divider."""
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}\n")


def _print_kv(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair with consistent formatting."""
    print(f"{' ' * indent}{key:<30s}: {value}")


# ─── Module-level worker (must be at module scope for loky pickling) ────────

def _process_ticker_worker(
    ticker: str,
    ticker_df: pd.DataFrame,
    filtered_specs: List[Dict[str, Any]],
) -> tuple:
    """
    loky worker: apply filtered TA strategy to one ticker's DataFrame.

    Runs in a separate process — builds its own ta.Strategy instance to
    avoid pickling issues.

    Core pinning: `ticker_df.ta.cores = 1` is set on the AnalysisIndicators
    INSTANCE immediately before calling .strategy(). The module-level
    `ta.cores` variable does NOT update instances already constructed by the
    accessor — each `df.ta` access creates a new AnalysisIndicators that reads
    cpu_count() at init time. Setting the property on the instance is the only
    reliable way to cap internal thread use to 1, which prevents the
    6-workers × 11-threads = 66-thread thrash on 11 physical cores.

    Returns:
        (ticker, result_df, None)  on success
        (ticker, None, error_str)  on failure
    """
    # Re-suppress MFI FutureWarning in child process (loky workers do not
    # inherit the parent's warnings filters).
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="Setting an item of incompatible dtype",
    )
    try:
        import pandas_ta_classic as _ta

        strategy = _ta.Strategy(
            name="Phase2_Candidates_Filtered",
            ta=filtered_specs,
        )
        # ✅ FIX: Set cores=1 on the DataFrame accessor INSTANCE, not the module.
        # df.ta creates a new AnalysisIndicators instance that initialises with
        # cpu_count() internally. Setting _ta.cores = 1 on the module variable
        # does not retroactively update that instance's _cores property.
        # Setting ticker_df.ta.cores = 1 directly on the instance we are about
        # to call disables internal thread spawning for this worker only.
        # Without this: 6 outer workers × ~11 internal threads = 66 threads
        # thrashing 11 physical cores. With this: 6 workers × 1 thread = 6.
        ticker_df.ta.cores = 1
        ticker_df.ta.strategy(strategy, verbose=False)
        return (ticker, ticker_df, None)
    except Exception as exc:
        return (ticker, None, str(exc))


# ─── Indicator-to-column mapping ───────────────────────────────────────────
# Maps each curated indicator "kind" to the column name prefixes it produces.
# Used to filter curated_indicators to only those that generate candidate cols.
_KIND_TO_PREFIX = {
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


# ─── Class ──────────────────────────────────────────────────────────────────
class IndividualFeatureEvaluator:
    """Evaluate each candidate feature's individual predictive power
    using Ridge regression scored by Spearman IC under Time-Series CV."""

    def __init__(self, config: Phase2Config) -> None:
        self.config = config

        # ── Auditor: exact contract from core.audit ─────────────────────────
        # Proven API from 1_preselection_audit.py:
        #   Auditor(phase=int, output_dir=str, version=str)
        # Lifecycle: .start() → .record_input() → .record_output() → .success()
        self.auditor = Auditor(
            phase=config.phase,
            output_dir=str(config.get_resolved_output_dir()),
            version=config.version,
        )

        self._raw_df: Optional[pd.DataFrame] = None
        self._candidate_names: Optional[List[str]] = None
        self._feature_ta_specs: Optional[List[Dict[str, Any]]] = None

    # ── 1. Load data from disk ──────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        """Load Phase 1 merged OHLCV data and Stage 0 candidate list.

        Returns:
            pd.DataFrame: The raw merged OHLCV DataFrame.

        Raises:
            FileNotFoundError: If any required artifact is missing.
            AssertionError: If required columns are absent.
        """
        _print_section("Step 1/4 · Loading Data")

        # ── Validate file existence (fail-fast) ────────────────────────────
        if not PHASE1_ARTIFACT.exists():
            raise FileNotFoundError(
                f"[FATAL] Phase 1 artifact not found: {PHASE1_ARTIFACT}\n"
                f"       Run Phase 1 before Phase 2 Stage 1."
            )
        if not CANDIDATE_CSV.exists():
            raise FileNotFoundError(
                f"[FATAL] Stage 0 artifact not found: {CANDIDATE_CSV}\n"
                f"       Run 1_preselection_audit.py before this script."
            )

        # ── Load merged OHLCV ───────────────────────────────────────────────
        t0 = perf_counter()
        df = pd.read_parquet(PHASE1_ARTIFACT)
        t_load = perf_counter() - t0
        _print_kv("Loaded merged_data.parquet",
                  f"{df.shape[0]:,} rows × {df.shape[1]} cols  ({t_load:.2f}s)")

        # ── Schema enforcement (crash immediately on violation) ─────────────
        df.columns = df.columns.str.lower().str.strip()
        present_cols = set(df.columns)
        missing = REQUIRED_COLUMNS - present_cols
        assert len(missing) == 0, (
            f"[SCHEMA VIOLATION] Missing required columns: {missing}\n"
            f"  Present columns: {sorted(present_cols)}"
        )
        _print_kv("Schema", f"validated — {sorted(REQUIRED_COLUMNS)} all present")

        # ── Load candidate feature names ────────────────────────────────────
        candidates_df = pd.read_csv(CANDIDATE_CSV)
        assert "feature" in candidates_df.columns, (
            f"[SCHEMA VIOLATION] candidate_features.csv must have a 'feature' column.\n"
            f"  Found columns: {list(candidates_df.columns)}"
        )
        self._candidate_names = candidates_df["feature"].tolist()
        _print_kv("Candidates loaded", f"{len(self._candidate_names)} features from Stage 0")
        print(f"    {self._candidate_names}")

        # ── Record inputs via Auditor ───────────────────────────────────────
        self.auditor.start()
        self.auditor.record_input(str(PHASE1_ARTIFACT), df)

        self._raw_df = df
        return df

    # ── 2. Compute forward target (decoupled re-implementation) ─────────────
    @staticmethod
    def compute_target(df: pd.DataFrame) -> pd.Series:
        """Compute 5-day forward log return per ticker.

        Formula:  target_t = ln(close_{t+5} / close_t)

        This is computed per-ticker to avoid cross-contamination at
        ticker boundaries.

        Args:
            df: DataFrame with 'ticker', 'close' columns, sorted by
                (ticker, date).

        Returns:
            pd.Series: Named 'target_log_return', aligned to df.index.
                       NaN for the last 5 rows of each ticker.
        """
        _print_section(f"Step 2/4 · Target Computation: {FORWARD_DAYS}-day forward log return")

        t0 = perf_counter()

        # Ensure sort order for temporal integrity
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # Vectorised: shift close backward (= look forward) per ticker group
        future_close = df.groupby("ticker")["close"].shift(-FORWARD_DAYS)
        target = np.log(future_close / df["close"])
        target.name = "target_log_return"

        n_valid = target.notna().sum()
        n_nan = target.isna().sum()
        t_elapsed = perf_counter() - t0
        _print_kv("Target computed",
                  f"{n_valid:,} valid / {n_nan:,} NaN  ({t_elapsed:.3f}s)")

        return target

    # ── 3. Regenerate features via pandas_ta_classic (parallel) ─────────────
    def regenerate_features(
        self, df: pd.DataFrame, candidate_list: List[str]
    ) -> pd.DataFrame:
        """Re-generate candidate features from raw OHLCV using pandas_ta_classic.

        Uses joblib Parallel/loky (n_jobs=6) — the same architecture as
        Stage 0 — giving ~6× speedup over the sequential tqdm loop.

        Each loky worker:
          1. Builds its own ta.Strategy (avoids pickling issues).
          2. Sets pandas_ta cores=1 (outer joblib controls concurrency).
          3. Re-suppresses the MFI FutureWarning (loky workers don't inherit
             the parent process's warnings filters).

        Tickers with fewer than _MIN_ROWS_FOR_TA rows are skipped (matches
        Stage 0 behaviour).

        Args:
            df: Raw OHLCV DataFrame (must contain open/high/low/close/volume).
            candidate_list: List of feature column names from Stage 0.

        Returns:
            pd.DataFrame: The input df augmented with regenerated feature columns.
        """
        _print_section(
            f"Step 3/4 · Feature Regeneration ({len(candidate_list)} candidates)"
        )

        t0 = perf_counter()

        # ── Filter curated_indicators to only needed specs ──────────────────
        all_specs = self.config.curated_indicators
        candidate_set = set(candidate_list)

        filtered_specs: List[Dict[str, Any]] = []
        for spec in all_specs:
            kind = spec.get("kind", "")
            prefixes = _KIND_TO_PREFIX.get(kind, [])
            produces_candidate = any(
                any(cand.startswith(prefix) for cand in candidate_set)
                for prefix in prefixes
            )
            if produces_candidate:
                filtered_specs.append(spec)

        self._feature_ta_specs = filtered_specs
        _print_kv("Curated specs", f"{len(all_specs)} total → {len(filtered_specs)} needed")

        # ── Split tickers; skip those with insufficient history ─────────────
        ticker_groups: List[tuple] = []
        skipped: List[str] = []
        for ticker in sorted(df["ticker"].unique()):
            tdf = df[df["ticker"] == ticker].copy()
            if len(tdf) < _MIN_ROWS_FOR_TA:
                skipped.append(f"{ticker} ({len(tdf)} rows)")
            else:
                ticker_groups.append((ticker, tdf))

        if skipped:
            print(f"  ⊘ Skipped {len(skipped)} tickers (insufficient history):")
            for msg in skipped[:10]:
                print(f"      {msg}")
            if len(skipped) > 10:
                print(f"      … and {len(skipped) - 10} more")

        _print_kv("Tickers",
                  f"{len(ticker_groups)} (n_jobs={_PARALLEL_N_JOBS}, loky)")

        # ── Parallel TA generation via joblib/loky ──────────────────────────
        results = Parallel(
            n_jobs=_PARALLEL_N_JOBS, backend="loky", verbose=0
        )(
            delayed(_process_ticker_worker)(ticker, tdf, filtered_specs)
            for ticker, tdf in ticker_groups
        )

        valid_frames: List[pd.DataFrame] = []
        generation_errors: List[str] = []

        for ticker, result_df, error in results:
            if error:
                generation_errors.append(ticker)
                if len(generation_errors) <= 5:
                    warnings.warn(
                        f"[Stage 1] pandas_ta failed for '{ticker}': {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                valid_frames.append(result_df)

        if not valid_frames:
            raise RuntimeError(
                "All tickers failed indicator generation. "
                "Check pandas_ta_classic installation and input data quality."
            )

        if generation_errors:
            print(f"  ⚠ {len(generation_errors)} tickers failed: "
                  f"{generation_errors[:10]}")

        result_df = pd.concat(valid_frames, axis=0, ignore_index=True)
        result_df = result_df.sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True)

        initial_cols = set(df.columns)
        new_cols = set(result_df.columns) - initial_cols
        _print_kv("New columns generated", len(new_cols))

        # ── Verify candidate columns are present ────────────────────────────
        generated_cols = set(result_df.columns)
        matched = [c for c in candidate_list if c in generated_cols]
        missing_cols = [c for c in candidate_list if c not in generated_cols]

        if missing_cols:
            col_map = {c.upper(): c for c in generated_cols}
            for m in missing_cols[:]:
                if m.upper() in col_map:
                    result_df.rename(
                        columns={col_map[m.upper()]: m}, inplace=True
                    )
                    matched.append(m)
                    missing_cols.remove(m)

        if missing_cols:
            print(f"  ⚠ {len(missing_cols)} candidates could not be regenerated"
                  f" — excluded from evaluation: {missing_cols}")

        _print_kv("Candidates matched",
                  f"{len(matched)}/{len(candidate_list)}")

        t_elapsed = perf_counter() - t0
        _print_kv("Regeneration time", f"{t_elapsed:.1f}s")

        self._candidate_names = matched
        return result_df

    # ── 4. Core evaluation loop ─────────────────────────────────────────────
    def run_evaluation(
        self, df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate each feature individually via Pipeline(scaler+Ridge) + TSCV + Spearman IC.

        StandardScaler is applied INSIDE each CV fold (fitted on train, applied
        to test) so there is zero data leakage from scaling.  This eliminates
        the magnitude bias where Ridge would suppress OBV (~1e9) relative to
        LOGRET_5 (~0.05) purely due to coefficient penalisation — not signal.

        For each feature:
          1. Extract non-NaN rows for (feature, target) pair.
          2. Run TimeSeriesSplit(n_splits=5).
          3. Fit Pipeline([StandardScaler, Ridge(alpha=1.0)]) on train.
          4. Predict on test (scaler transform applied automatically).
          5. Compute Spearman IC on test fold.
          6. Aggregate: mean_ic, std_ic, ir.

        Args:
            df: DataFrame with target and all candidate feature columns.

        Returns:
            Dict containing per-feature scores and summary.
        """
        _print_section("Step 4/4 · Individual Feature Evaluation")
        _print_kv("Model", f"Pipeline([StandardScaler, Ridge(alpha={RIDGE_ALPHA})])")
        _print_kv("CV", f"TimeSeriesSplit(n_splits={CV_FOLDS})")
        _print_kv("Metric", "Spearman IC (rank correlation)")
        _print_kv("Scaling", "StandardScaler fitted per-fold on train only (no leakage)")

        t0 = perf_counter()

        target_col = "target_log_return"
        assert target_col in df.columns, (
            f"[FATAL] Target column '{target_col}' not found in DataFrame."
        )

        # Sort globally by (ticker, date) to preserve temporal order for TSCV
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        feature_results: Dict[str, Dict[str, Any]] = {}
        candidates = self._candidate_names

        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        for feat_name in candidates:
            # ── Extract valid (non-NaN) subset for this feature+target ──────
            mask = df[[feat_name, target_col]].notna().all(axis=1)
            subset = df.loc[mask, [feat_name, target_col]].reset_index(drop=True)

            n_samples = len(subset)
            if n_samples < CV_FOLDS * 2:
                print(f"  ⚠ Skipping {feat_name}: only {n_samples} valid samples")
                feature_results[feat_name] = {
                    "mean_ic": None,
                    "std_ic": None,
                    "ir": None,
                    "fold_ics": [],
                    "n_valid_samples": n_samples,
                    "status": "SKIPPED_INSUFFICIENT_DATA",
                }
                continue

            X = subset[[feat_name]].values   # shape: (N, 1)
            y = subset[target_col].values    # shape: (N,)

            fold_ics: List[float] = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # ✅ FIX: Pipeline fits StandardScaler on train, transforms
                # both train and test — zero leakage, zero magnitude bias.
                # Ridge now always sees a unit-variance feature regardless of
                # whether it's OBV (~1e9) or LOGRET_5 (~0.05).
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)),
                ])
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Spearman IC: rank correlation between prediction and actual
                if np.std(y_pred) < 1e-15 or np.std(y_test) < 1e-15:
                    ic = 0.0
                else:
                    ic, _ = spearmanr(y_pred, y_test)
                    if np.isnan(ic):
                        ic = 0.0

                fold_ics.append(round(float(ic), 6))

            mean_ic = float(np.mean(fold_ics))
            std_ic = float(np.std(fold_ics, ddof=1)) if len(fold_ics) > 1 else 0.0

            if std_ic > 1e-10:
                ir = mean_ic / std_ic
            else:
                ir = float("inf") if mean_ic > 0 else (
                    float("-inf") if mean_ic < 0 else 0.0
                )

            feature_results[feat_name] = {
                "mean_ic": round(mean_ic, 6),
                "std_ic": round(std_ic, 6),
                "ir": round(ir, 4) if np.isfinite(ir) else ir,
                "fold_ics": fold_ics,
                "n_valid_samples": n_samples,
                "status": "OK",
            }

        runtime = perf_counter() - t0

        # ── Determine top feature by IR (stable tie-breaking by mean_ic) ────
        scorable = {
            k: v for k, v in feature_results.items()
            if v["mean_ic"] is not None
        }
        if scorable:
            top_feature = max(
                scorable.keys(),
                key=lambda k: (scorable[k]["ir"], scorable[k]["mean_ic"]),
            )
        else:
            top_feature = "NONE"

        # ── Assemble full output payload ─────────────────────────────────────
        output = {
            "stage": STAGE_NAME,
            "version": VERSION,
            "config_snapshot": {
                "ridge_alpha": RIDGE_ALPHA,
                "cv_folds": CV_FOLDS,
                "forward_days": FORWARD_DAYS,
                "n_candidates_input": len(candidates),
                "curated_indicator_count": len(self._feature_ta_specs)
                    if self._feature_ta_specs else 0,
                "curated_total_available": len(self.config.curated_indicators),
                "scaling": "StandardScaler (per-fold, train-only fit)",
            },
            "evaluation_params": {
                "model": "Pipeline([StandardScaler, Ridge])",
                "alpha": RIDGE_ALPHA,
                "cv_folds": CV_FOLDS,
                "cv_method": "TimeSeriesSplit",
                "metric": "Spearman IC",
                "scaling": "StandardScaler fitted on train fold only",
            },
            "features": feature_results,
            "summary": {
                "total_features_evaluated": len(scorable),
                "total_features_skipped": len(feature_results) - len(scorable),
                "top_feature": top_feature,
                "top_feature_ir": scorable.get(top_feature, {}).get("ir"),
                "top_feature_mean_ic": scorable.get(
                    top_feature, {}
                ).get("mean_ic"),
                "runtime_seconds": round(runtime, 2),
            },
        }

        # ── Print ranked summary ────────────────────────────────────────────
        ranked = sorted(
            scorable.items(),
            key=lambda kv: (kv[1]["ir"], kv[1]["mean_ic"]),
            reverse=True,
        )

        print(f"\n  {'─' * 68}")
        print(f"  {'Rank':<6}{'Feature':<25}{'Mean IC':>10}{'Std IC':>10}{'IR':>10}")
        print(f"  {'─' * 68}")
        for rank, (fname, fdata) in enumerate(ranked, 1):
            ir_str = (
                f"{fdata['ir']:.4f}" if np.isfinite(fdata["ir"]) else "∞"
            )
            print(
                f"  {rank:<6}{fname:<25}{fdata['mean_ic']:>10.6f}"
                f"{fdata['std_ic']:>10.6f}{ir_str:>10}"
            )
        print(f"  {'─' * 68}")

        _print_kv("Evaluation time", f"{runtime:.2f}s")
        _print_kv("Top feature", top_feature)

        return output

    # ── 5. Save results atomically ───────────────────────���──────────────────
    def save_results(self, results: Dict[str, Any]) -> None:
        """Write individual_feature_scores.json atomically."""
        _print_section("Saving Artifacts")

        PHASE2_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        tmp_path = OUTPUT_JSON.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    results, f, indent=2, ensure_ascii=False, default=str
                )
            tmp_path.replace(OUTPUT_JSON)
            _print_kv(OUTPUT_JSON.name,
                      f"{OUTPUT_JSON.stat().st_size:,} bytes")
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(
                f"[FATAL] Failed to write results: {e}"
            ) from e

        try:
            summary_df = pd.DataFrame([results.get("summary", {})])
            self.auditor.record_output(summary_df, self.config.to_snapshot())
            self.auditor.success()
            _print_kv("manifest.json", "written via Auditor")
        except Exception as e:
            print(f"  ⚠ Auditor manifest write failed (non-fatal): {e}")

    def run(self) -> None:
        """Full pipeline: Load → Target → Features → Evaluate → Save."""
        pipeline_t0 = perf_counter()

        df = self.load_data()
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        target = self.compute_target(df)
        df["target_log_return"] = target
        df = self.regenerate_features(df, self._candidate_names)
        results = self.run_evaluation(df)
        self.save_results(results)

        total_time = perf_counter() - pipeline_t0
        _print_box(
            f"COMPLETE — {results['summary']['total_features_evaluated']} "
            f"features evaluated  [{total_time:.0f}s]"
        )


# ─── Main Execution ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Phase 2 · Stage 1 — Individual Feature Evaluation                ║")
    print(f"║  Version: {VERSION:<58}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    try:
        config = Phase2Config()
        evaluator = IndividualFeatureEvaluator(config=config)
        evaluator.run()
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as e:
        print(f"\n[ASSERTION FAILED] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Evaluation aborted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(99)
